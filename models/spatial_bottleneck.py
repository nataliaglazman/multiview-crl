"""Continuous map -> small vector -> map, with no spatial-feature bypass."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialBottleneck(nn.Module):
    """Shared readout/decoder for registered, continuous content maps.

    Flattening retains the spatial ordering; no global pooling or striding occurs.
    A fixed training-set template is removed before encoding and restored after
    decoding. This template contains no subject-dependent information.
    """

    def __init__(self, map_shape, latent_dim=16, width=8, hidden=128):
        super().__init__()
        shape = tuple(map_shape)
        if len(shape) != 4 or any(not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError("map_shape must contain four positive integers: C,D,H,W")
        if any(not isinstance(n, int) or n < 1 for n in (latent_dim, width, hidden)):
            raise ValueError("latent_dim, width and hidden must be positive integers")
        if latent_dim >= math.prod(shape):
            raise ValueError("latent_dim must be smaller than the complete map")
        self.config = dict(map_shape=shape, latent_dim=latent_dim, width=width, hidden=hidden)
        self.map_shape = shape
        self.latent_dim = latent_dim
        channels, *grid = shape
        flat_width = width * math.prod(grid)
        self.readout = nn.Sequential(
            nn.Conv3d(channels, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(1),
            nn.Linear(flat_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_width),
            nn.SiLU(),
            nn.Unflatten(1, (width, *grid)),
            nn.Conv3d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(width, channels, 1),
        )
        self.register_buffer("template", torch.zeros(shape))
        self.register_buffer("scale", torch.ones(channels, 1, 1, 1))

    @torch.no_grad()
    def fit_normalization(self, training_maps):
        """Fit on training subjects only; both views use the same transformation."""
        self._check_pairs(training_maps)
        template = training_maps.mean((0, 1))
        # Accumulate per subject to avoid another full map-bank allocation.
        variance = torch.zeros_like(self.scale, device=training_maps.device)
        for pair in training_maps:
            variance += (pair - template).square().mean((0, 2, 3, 4))[:, None, None, None]
        scale = (variance / len(training_maps)).sqrt().clamp_min(1e-5)
        self.template.copy_(template)
        self.scale.copy_(scale)

    def _check_pairs(self, maps):
        if maps.ndim != 6 or maps.shape[1] != 2 or tuple(maps.shape[2:]) != self.map_shape:
            raise ValueError(f"Expected paired maps (B,2,{self.map_shape}), got {tuple(maps.shape)}")
        if len(maps) < 2 or not torch.isfinite(maps).all():
            raise ValueError("Need at least two subjects with finite maps")

    def encode(self, maps):
        if maps.ndim != 5 or tuple(maps.shape[1:]) != self.map_shape:
            raise ValueError(f"Expected maps (B,{self.map_shape}), got {tuple(maps.shape)}")
        return self.readout((maps - self.template) / self.scale)

    def decode(self, code):
        if code.ndim != 2 or code.shape[1] != self.latent_dim:
            raise ValueError(f"Decoder accepts only (B,{self.latent_dim}) vectors")
        return self.decoder(code) * self.scale + self.template

    def forward(self, maps):
        code = self.encode(maps)
        return code, self.decode(code)


@torch.no_grad()
def position_weights(model, training_maps, floor=0.25):
    """Loss weights that put each map entry's error in units of its own spread across subjects.

    The spread is each (channel, position) entry's RMS deviation from the template over the
    training subjects and both views, floored at ``floor`` times its channel's scale so that
    near-constant entries cannot dominate. Weights are (channel scale / spread)^2, renormalised to
    mean 1 so the reconstruction term keeps its size against the alignment and variance terms.
    Returns the weights and the fraction of entries held at the floor.
    """
    model._check_pairs(training_maps)
    if not 0 < floor <= 1:
        raise ValueError("floor must lie in (0, 1]")
    template, scale = model.template.cpu(), model.scale.cpu()
    variance = torch.zeros(model.map_shape, dtype=torch.float64)
    for pair in training_maps:
        variance += (pair.cpu() - template).double().square().mean(0)
    spread = (variance / len(training_maps)).sqrt().float()
    floored = spread < floor * scale
    weights = (scale / spread.maximum(floor * scale)).square()
    return weights / weights.mean(), floored.float().mean().item()


@torch.no_grad()
def deviation_weights(model, maps, strength=1.0):
    """Per-position loss weights that grow with how far each map departs from the template.

    The departure is the RMS over channels of the standardised deviation, so all channels at a
    position share the weight 1 + strength * departure, renormalised to mean 1 within each map.
    It is computed from the targets: it re-weights errors and is not itself optimised.
    """
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    departure = ((maps - model.template) / model.scale).square().mean(-4, keepdim=True).sqrt()
    weights = 1 + strength * departure
    return weights / weights.mean((-3, -2, -1), keepdim=True)


def reconstruction_terms(
    error,
    recon_weights=None,
    foreground=None,
    hard_patch_weight=0.0,
    hard_patch_fraction=0.1,
    hard_patch_size=(4, 4, 4),
):
    """Mean error + weighted top-fraction patch error, selected per subject and view.

    ``error`` is (B,2,C,D,H,W) squared error in channel-normalised units. Foreground
    is (B,2,1,D,H,W) occupancy in [0,1], not a feature-value threshold. Empty
    patches never enter top-k. Nonoverlapping patches cover partial edge cells.
    Optional reconstruction weights are renormalised over foreground per map;
    patch means then divide by foreground occupancy, not patch volume.
    """
    if error.ndim != 6 or error.shape[1] != 2:
        raise ValueError("Expected error shaped (B,2,C,D,H,W)")
    if not math.isfinite(hard_patch_weight) or hard_patch_weight < 0:
        raise ValueError("hard_patch_weight must be finite and nonnegative")
    if not 0 < hard_patch_fraction <= 1:
        raise ValueError("hard_patch_fraction must lie in (0, 1]")
    if len(hard_patch_size) != 3 or any(not isinstance(s, int) or s < 1 for s in hard_patch_size):
        raise ValueError("hard_patch_size must contain three positive integers")
    weights = None
    if recon_weights is not None:
        weights = torch.broadcast_to(recon_weights.detach().to(error), error.shape)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Reconstruction weights must be finite and nonnegative")
    occupancy = None
    if foreground is not None:
        if tuple(foreground.shape) != (*error.shape[:2], 1, *error.shape[-3:]):
            raise ValueError("Foreground must have shape (B,2,1,D,H,W) matching the maps")
        occupancy = foreground.detach().to(error)
        if not torch.isfinite(occupancy).all() or (occupancy < 0).any() or (occupancy > 1).any():
            raise ValueError("Foreground occupancy must be finite and lie in [0, 1]")
        mass = occupancy.sum((2, 3, 4, 5))
        if (mass <= 0).any():
            raise ValueError("Every subject and view needs nonempty foreground")
        if weights is not None:
            weighted_mass = (weights * occupancy).sum((2, 3, 4, 5))
            if (weighted_mass <= 0).any():
                raise ValueError("Reconstruction weights must have positive mass in foreground")
            weights = weights * (mass * error.shape[2] / weighted_mass)[:, :, None, None, None, None]
    weighted_error = error if weights is None else error * weights
    if occupancy is None:
        mean = weighted_error.mean()
    else:
        mean = ((weighted_error * occupancy).sum((2, 3, 4, 5)) / (mass * error.shape[2])).mean()
    hard = error.new_zeros(())
    if hard_patch_weight > 0:
        if occupancy is None:
            occupancy = torch.ones_like(error[:, :, :1])
        numerator = (weighted_error.mean(2, keepdim=True) * occupancy).flatten(0, 1)
        denominator = occupancy.flatten(0, 1)
        padding = tuple(
            p for n, size in zip(reversed(error.shape[-3:]), reversed(hard_patch_size)) for p in (0, -n % size)
        )
        numerator = F.avg_pool3d(F.pad(numerator, padding), hard_patch_size, stride=hard_patch_size).flatten(1)
        denominator = F.avg_pool3d(F.pad(denominator, padding), hard_patch_size, stride=hard_patch_size).flatten(1)
        selected = []
        for sums, counts in zip(numerator, denominator):
            valid = counts > 0
            patch_means = sums[valid] / counts[valid]
            k = max(1, math.ceil(hard_patch_fraction * patch_means.numel()))
            selected.append(patch_means.topk(k).values.mean())
        hard = torch.stack(selected).mean()
    return dict(reconstruction=mean + hard_patch_weight * hard, reconstruction_mean=mean, reconstruction_hard=hard)


def paired_objective(
    model,
    maps,
    reconstruction_weight=1.0,
    alignment_weight=1.0,
    variance_weight=0.1,
    recon_weights=None,
    foreground=None,
    hard_patch_weight=0.0,
    hard_patch_fraction=0.1,
    hard_patch_size=(4, 4, 4),
):
    """No labels or interventions; variance is across subjects within each view.

    ``recon_weights`` (broadcastable to the paired maps, mean 1) re-weight the reconstruction error,
    as ``position_weights`` or ``deviation_weights`` produce; None is the plain mean.
    """
    model._check_pairs(maps)
    weights = (reconstruction_weight, alignment_weight, variance_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or reconstruction_weight <= 0:
        raise ValueError("Weights must be finite/nonnegative and reconstruction must be positive")
    targets = maps.detach()
    code, reconstruction = model(targets.flatten(0, 1))
    code = code.reshape(len(maps), 2, -1)
    reconstruction = reconstruction.reshape_as(targets)
    error = ((reconstruction - targets) / model.scale).square()
    terms = reconstruction_terms(
        error, recon_weights, foreground, hard_patch_weight, hard_patch_fraction, hard_patch_size
    )
    terms["alignment"] = F.mse_loss(code[:, 0], code[:, 1])
    terms["variance"] = F.relu(1 - (code.var(0, unbiased=False) + 1e-4).sqrt()).mean()
    loss = sum(w * terms[name] for w, name in zip(weights, ("reconstruction", "alignment", "variance")))
    return loss, terms
