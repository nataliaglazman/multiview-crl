"""Frozen-feature local readouts and a decoder with no spatial-feature bypass."""

import torch
import torch.nn.functional as F
from torch import nn

from models.keypoint_pool import KeypointPool3d, brain_frame
from models.scalar_readout import ScalarReadout, coordinates, pool_grid


def frame(support, grid):
    centre, spread = brain_frame(support, grid)
    return centre, spread.clamp_min(1e-4)


class LocalReadout(nn.Module):
    """One shared readout for both modalities; K geometric heads and one signed scalar."""

    def __init__(self, channels, grid, resolution, heads=4, free=False):
        super().__init__()
        self.heads, self.free = (1 if free else heads), free
        self.register_buffer("grid", coordinates(grid, resolution))
        if free:
            # Reuse the earlier free-oracle architecture; optimize only the four
            # local semantic outputs here, matching the hybrid oracle's targets.
            self.readout = ScalarReadout(channels, grid, 16, resolution, geometric=False)
        else:
            self.keypoints = KeypointPool3d(channels, heads, frame="brain")
            self.amplitude = nn.Sequential(
                nn.Conv3d(channels, 4, 1),
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(4 * grid**3, 1),
            )
        width = 3 * self.heads + 1
        # Do not expand a linear projector beyond its input rank: Barlow Twins
        # could never make a larger cross-correlation matrix full-rank identity.
        self.projector = nn.Linear(width, min(16, width))

    def forward(self, features, support):
        # Leading dimensions may include subject, endpoint, and modality.
        leading = features.shape[:-4]
        h, s = features.reshape(-1, *features.shape[-4:]), support.reshape(-1, *support.shape[-4:])
        centre, spread = frame(s, self.grid)
        if self.free:
            code = self.readout(h)[:, [2, 3, 4, 8]]
            physical, amplitude = code[:, None, :3], code[:, 3:]
            relative = (physical - centre[:, None]) / spread[:, None]
        else:
            relative = self.keypoints(h, s).reshape(-1, self.heads, 3)
            physical = relative * spread[:, None] + centre[:, None]
            amplitude = self.amplitude(h)
        code = torch.cat((relative.flatten(1), amplitude), -1)
        return {
            "code": code.reshape(*leading, -1),
            "physical": physical.reshape(*leading, self.heads, 3),
            "relative": relative.reshape(*leading, self.heads, 3),
            "amplitude": amplitude.reshape(*leading, 1),
        }

    def project(self, code):
        # Standardize within each view; no modality-offset contribution to moments.
        sd = code.std(0, unbiased=False, keepdim=True).clamp_min(1e-4)
        return self.projector((code - code.mean(0, keepdim=True)) / sd)


class LocalDecoder(nn.Module):
    """Fixed-width localized blobs plus one signed spatial template per modality.

    No image features, global code, subject-dependent blob amplitude, or decoder
    bias are accepted. All subject dependence passes through the local scalars.
    """

    def __init__(self, channels, grid, resolution, heads=4, views=2, width_vox=3.0):
        super().__init__()
        self.grid_size = grid
        self.register_buffer("grid", coordinates(grid, resolution))
        self.sigma = 2 * width_vox / (resolution - 1)
        self.blob_values = nn.Parameter(torch.randn(views, heads, channels) * 0.05)
        self.template = nn.Parameter(torch.randn(views, channels, grid, grid, grid) * 0.05)

    def forward(self, physical, amplitude):
        distance = (physical[..., None, :] - self.grid).square().sum(-1)
        bumps = torch.exp(-distance / (2 * self.sigma**2))
        blobs = torch.einsum("bvkp,vkc->bvcp", bumps, self.blob_values)
        template = self.template / self.template.square().mean((1, 2, 3, 4), keepdim=True).add(1e-6).sqrt()
        return blobs.reshape(-1, *template.shape) + amplitude[..., None, None, None] * template[None]


def bands(x):
    leading, shape = x.shape[:-4], x.shape[-4:]
    flat = x.reshape(-1, *shape)
    smooth = F.avg_pool3d(F.pad(flat, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, stride=1)
    coarse = pool_grid(flat, shape[-1] // 2)
    return {
        "fine": x,
        "highpass": (flat - smooth).reshape(*leading, *shape),
        "coarse": coarse.reshape(*leading, *coarse.shape[-4:]),
    }


def within_view_decorrelation(code, global_code):
    def standardize(z):
        z = z - z.mean(0, keepdim=True)
        return z / z.square().mean(0, keepdim=True).add(1e-8).sqrt()

    a, b = standardize(code), standardize(global_code.detach())
    correlation = torch.einsum("bvi,bvj->vij", a, b) / len(a)
    return correlation.square().mean()


def infonce(head, original, photo, temperature=0.1):
    paired = original.shape[1] == 2
    a = head.project(original)[:, 0]
    b = head.project(original)[:, 1] if paired else head.project(photo)[:, 0]
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
    labels = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)


def reconstruction_loss(prediction, residual, support, scales, return_maps=False):
    terms, maps = {}, {}
    predicted_bands = bands(prediction)
    for name, actual in bands(residual).items():
        estimate = predicted_bands[name]
        size = actual.shape[-1]
        mask = pool_grid(support.flatten(0, 1), size).reshape(*support.shape[:3], size, size, size)
        error = ((estimate - actual) / scales[name]).square()
        terms[name] = (error * mask).sum() / (mask.sum() * actual.shape[2]).clamp_min(1)
        if return_maps:
            maps[name] = error.mean(2)
    return (sum(terms.values()) / len(terms), terms, maps)


def barlow_twins(head, original, photo, off_diagonal_weight=0.005):
    projected = head.project(original)
    a = projected[:, 0]
    b = projected[:, 1] if original.shape[1] == 2 else head.project(photo)[:, 0]

    def standardize(z):
        z = z - z.mean(0)
        return z / z.square().mean(0).add(1e-5).sqrt()

    correlation = standardize(a).T @ standardize(b) / len(a)
    diagonal = correlation.diagonal()
    mask = ~torch.eye(len(diagonal), device=a.device, dtype=torch.bool)
    return (diagonal - 1).square().sum() + off_diagonal_weight * correlation[mask].square().sum()
