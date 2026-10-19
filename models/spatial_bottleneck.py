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


def paired_objective(
    model, maps, reconstruction_weight=1.0, alignment_weight=1.0, variance_weight=0.1, recon_weights=None
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
    terms = {
        "reconstruction": error.mean() if recon_weights is None else (error * recon_weights).mean(),
        "alignment": F.mse_loss(code[:, 0], code[:, 1]),
        "variance": F.relu(1 - (code.var(0, unbiased=False) + 1e-4).sqrt()).mean(),
    }
    loss = sum(w * terms[name] for w, name in zip(weights, terms))
    return loss, terms
