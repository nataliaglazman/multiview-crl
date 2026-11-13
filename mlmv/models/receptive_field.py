"""Receptive-field radius rho, theoretical (from the architecture) and effective (from input gradients).

Effective radius at a feature position u: the squared input gradient of u's whole feature vector, estimated by
Hutchinson draws (``sum_c (dF_c(u)/dx)^2 = E_v (d v.F(u) / dx)^2`` for Rademacher v), summed over the batch;
the radius is the smallest Chebyshev distance (input voxels, from the centre of u's footprint) holding
``mass`` of that energy, divided by the stride to give feature voxels. The run uses the median over positions
and the larger of the two modalities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def interior_positions(shape, margin: int, n: int, gen: torch.Generator, mask=None) -> np.ndarray:
    """``n`` random feature positions at least ``margin`` from every face (and inside ``mask`` if given)."""
    grid = np.zeros(shape, dtype=bool)
    sl = tuple(slice(margin, s - margin) for s in shape)
    grid[sl] = True
    if mask is not None:
        grid &= np.asarray(mask, dtype=bool)
    cand = np.argwhere(grid)
    if len(cand) == 0:
        raise ValueError(f"no interior position at margin {margin} in grid {shape}")
    pick = torch.randperm(len(cand), generator=gen)[: min(n, len(cand))].numpy()
    return cand[pick]


def effective_radius(
    encoder, x, modality: str, positions, mass: float = 0.95, n_draws: int = 4, gen: torch.Generator | None = None
) -> list:
    """Per-position effective radius in feature voxels. ``x [B, 1, X, Y, Z]``, ``positions [n, 3]``."""
    was_training = encoder.training
    encoder.eval()
    s, low, high = encoder.theoretical_footprint()
    centre_offset = (low + high) / 2
    x = x.detach().clone().requires_grad_(True)
    feats = encoder.features(x, modality)
    spatial = x.shape[2:]
    axes = [torch.arange(n, device=x.device, dtype=torch.float32) for n in spatial]
    radii = []
    for pos in positions:
        energy = torch.zeros(spatial, device=x.device)
        for _ in range(n_draws):
            v = torch.randint(0, 2, feats.shape[:2], generator=gen).to(x.device, feats.dtype) * 2 - 1
            out = (feats[(slice(None), slice(None)) + tuple(int(p) for p in pos)] * v).sum()
            (grad,) = torch.autograd.grad(out, x, retain_graph=True)
            energy += grad.float().pow(2).sum((0, 1))
        centre = [s * int(p) + centre_offset for p in pos]
        dist = torch.stack(torch.meshgrid(*[(a - c).abs() for a, c in zip(axes, centre)], indexing="ij")).amax(0)
        d, e = dist.flatten(), energy.flatten()
        order = torch.argsort(d)
        cum = torch.cumsum(e[order], 0)
        idx = int(torch.searchsorted(cum, mass * cum[-1]).clamp(max=len(cum) - 1))
        radii.append(float(d[order][idx]) / s)
    encoder.train(was_training)
    return radii


def measure_rho(encoder, batch: dict, positions, mass: float, gen: torch.Generator) -> dict:
    """Effective rho per modality (median over ``positions``) and the larger of the two."""
    out = {}
    for modality in ("t1", "flair"):
        radii = effective_radius(encoder, batch[modality], modality, positions, mass=mass, gen=gen)
        out[modality] = dict(median=float(np.median(radii)), p90=float(np.percentile(radii, 90)), n=len(radii))
    out["effective"] = max(out["t1"]["median"], out["flair"]["median"])
    return out


def write_rho(path, theoretical: float, effective, used: float, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(theoretical=theoretical, effective=effective, used=used, **(extra or {})), indent=2)
    )
