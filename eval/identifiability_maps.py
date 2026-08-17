#!/usr/bin/env python3
"""identifiability_maps.py -- per-patch, decoder-intrinsic identifiability maps.

Two model-intrinsic maps, both derived from one accumulated statistic:

    G(P) = E_n [ J(P,n)^T J(P,n) ]          J = [ dx/dz_content | dx/dz_style ]

restricted to voxels in patch P.  Writing G in blocks,

    G = [ G_cc  G_cs ]
        [ G_sc  G_ss ]

  * cond(G_cc)          local conditioning of the content factors.  Large means
                        some direction in content space produces (almost) no
                        image change inside P, so that direction is not locally
                        identified.  This is the map.

  * canon. corr. from   principal angles between the content and style Jacobian
    G_cc, G_cs, G_ss    column spaces.  A correlation near 1 means content and
                        style act along the same image directions in P, i.e.
                        local leakage between the blocks.

Accumulating the Gram (rather than per-sample angles) is what makes the second
quantity well defined across samples: it is exactly a canonical correlation
analysis between the two sets of Jacobian columns.

A third, empirical map (multi-seed agreement) and a synthetic-only validation
against ground-truth footprints are provided separately.

Where this sits in the project
------------------------------
This is the *decoder-side* companion to the encoder-side maps:

  * ``eval/recovery_maps.py``    -- needs >=2 solutions, quotients an explicit gauge
                                   group, and asks whether two runs recovered the same
                                   content.  Voxel-wise, encoder features.
  * ``probe/jacobian_spread.py`` -- the same decoder Jacobian, reduced radially to answer
                                   "how far does ONE latent cell reach" (rho).
  * this module                  -- the same decoder Jacobian, reduced per patch to answer
                                   "is the content block locally identified HERE, and does
                                   style act along the same image directions".

One decoder is enough: these are model-intrinsic maps, so they run as a kill-switch
before anything that needs a second training run.

What z_content and z_style are here
-----------------------------------
The abstract contract is ``decode(z_content, z_style) -> (*volume_shape)`` on 1-D
latents.  This repo's decoder takes spatial tensors, so ``decoder_closure`` binds a
real operating point ``(base_z, base_style)`` captured from a forward pass and lets
``z_content`` / ``z_style`` be **per-channel offsets broadcast over the latent grid**:

    decode(z_c, z_s) = decoder(base_z + z_c[None, :, None, None, None],
                               style=base_style + z_s[None, :, None, None, None])

so ``d_content`` is the channel count of the level-0 decoder input (the concatenated,
upscaled content codes of every level) and ``d_style`` is the injected style tensor's
channel count.  This is a genuine restriction: the maps describe the ``d_content``
*global channel* directions, not all ``C x D x H x W`` latent directions.  It is the
right restriction for this project — the recovered factors measured so far are global
scalars — but it is a restriction, and a direction that is invisible here may still be
carried by a spatially structured latent perturbation.

Two arms, because a spatially reduced normalization breaks locality
-------------------------------------------------------------------
Every claim above is *local to P*.  A GroupNorm inside the decoder computes statistics
over the whole volume, so ``d mu / d z`` and ``d sigma / d z`` couple every output voxel
to every latent channel, and the patch-local Jacobian picks up a global path that has
nothing to do with the convolutions.  ``probe/jacobian_spread.py`` established this for
the radial reduction; it applies here unchanged.  So both arms are reported:

    full         the decoder as trained.  This is what the model does.
    frozen_norm  every normalization statistic pinned at the operating point (output
                 identical, Jacobian without the global path).  This is what the
                 convolutions do.

A large gap between them means the maps are reading the normalization, not local
structure.  ``probe.jacobian_spread.freeze_norm_statistics`` does the pinning; it is
reused rather than reimplemented.

Usage
    # model-intrinsic maps, both arms
    python -m eval.identifiability_maps --run-dir results/synthetic/<run>

    # + the synthetic ground-truth validation (renderer footprints)
    python -m eval.identifiability_maps --run-dir results/synthetic/<run> --synthetic

    # generator-only degeneracy check, no checkpoint needed
    python -m eval.identifiability_maps --run-dir results/synthetic/<run> --footprints-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from torch.func import jacfwd

import probe.jacobian_spread as jspread
import probe.receptive_field as rfmod
from eval.dci import CONTENT_FACTOR_NAMES

logger = logging.getLogger(__name__)

Decoder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# A patch whose accumulated Jacobian energy is at or below this fraction of the largest
# patch energy carries no measurement.  Padding voxels and out-of-mask patches are
# exactly zero; the threshold exists so they are reported as NaN rather than as the
# perfectly-conditioned patches an unguarded eigenvalue clamp would make them.
DEAD_PATCH_FRACTION = 1e-12


# ----------------------------------------------------------------------------
# patching
# ----------------------------------------------------------------------------


def patch_grid_shape(vol_shape: Sequence[int], patch: int) -> tuple[int, int, int]:
    """Patch-grid shape of a volume that is already a multiple of ``patch``."""
    return tuple(s // patch for s in vol_shape)


def padded_shape(vol_shape: Sequence[int], patch: int) -> tuple[int, int, int]:
    """Smallest multiple-of-``patch`` shape containing ``vol_shape``.

    ADNI volumes are (91, 109, 91), which no useful patch size divides.  Padding at the
    far edge (never centred) keeps patch ``p`` covering original voxels
    ``[p*patch, (p+1)*patch)``, so map indices stay aligned with the volume origin.
    Padded voxels contribute exactly zero to the Gram, and any patch made entirely of
    them falls out through ``DEAD_PATCH_FRACTION``.
    """
    return tuple(int(np.ceil(s / patch)) * patch for s in vol_shape)


def _pad_flat(x: torch.Tensor, vol_shape: Sequence[int], target: Sequence[int]) -> torch.Tensor:
    """(V, k) laid out as ``vol_shape`` -> (prod(target), k), zero-filled at the far edge."""
    if tuple(vol_shape) == tuple(target):
        return x
    k = x.shape[-1]
    out = x.new_zeros(*target, k)
    out[: vol_shape[0], : vol_shape[1], : vol_shape[2]] = x.reshape(*vol_shape, k)
    return out.reshape(-1, k)


def _patch_view(jac: torch.Tensor, vol_shape: Sequence[int], patch: int) -> torch.Tensor:
    """(V, k) -> (n_patches, patch**3, k), non-overlapping cubic patches."""
    d, h, w = vol_shape
    if d % patch or h % patch or w % patch:
        raise ValueError(f"patch={patch} must divide volume shape {tuple(vol_shape)}")
    k = jac.shape[-1]
    x = jac.reshape(d, h, w, k)
    x = x.reshape(d // patch, patch, h // patch, patch, w // patch, patch, k)
    x = x.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, patch**3, k)
    return x


# ----------------------------------------------------------------------------
# accumulation
# ----------------------------------------------------------------------------


@dataclass
class GramAccumulator:
    """Running sum of patch-local Jacobian Grams over a dataset."""

    d_content: int
    d_style: int
    vol_shape: Sequence[int]
    patch: int = 4
    mask: torch.Tensor | None = None  # (*vol_shape) bool, foreground
    pad: bool = True  # zero-pad to a multiple of `patch` instead of refusing the shape
    _diag: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        k = self.d_content + self.d_style
        self.vol_shape = tuple(int(s) for s in self.vol_shape)
        self.grid_shape = padded_shape(self.vol_shape, self.patch) if self.pad else self.vol_shape
        if any(s % self.patch for s in self.grid_shape):
            raise ValueError(
                f"patch={self.patch} does not divide volume shape {self.vol_shape}; "
                f"pass pad=True to zero-pad to {padded_shape(self.vol_shape, self.patch)}"
            )
        n_patch = int(np.prod(patch_grid_shape(self.grid_shape, self.patch)))
        self.gram = torch.zeros(n_patch, k, k, dtype=torch.float64)
        self.count = 0
        if self.mask is not None:
            m = _pad_flat(self.mask.reshape(-1, 1).float(), self.vol_shape, self.grid_shape)
            self.patch_mask = _patch_view(m, self.grid_shape, self.patch)  # (P,p3,1)
        else:
            self.patch_mask = None

    @property
    def rank_budget(self) -> int:
        """Rows available per patch to span ``k`` columns: ``count * patch**3``.

        ``J(P, n)`` has ``patch**3`` rows and ``k`` columns, so a single sample cannot
        span the block when ``patch**3 < k``: ``G_cc`` is then singular by construction,
        ``cond`` saturates at the eigenvalue clamp and the canonical correlations are
        identically 1 — on pure noise as readily as on a trained decoder.  Averaging over
        ``count`` samples adds independent rows, which is why the budget is a product.
        """
        return self.count * self.patch**3

    def check_rank(self, strict: bool = True) -> dict:
        """Refuse (or flag) a configuration whose maps would be rank artefacts."""
        k = self.d_content + self.d_style
        out = {
            "k": k,
            "patch_voxels": self.patch**3,
            "rank_budget": self.rank_budget,
            "single_sample_rank_deficient": self.patch**3 < k,
            "ok": self.rank_budget >= k,
        }
        if not out["ok"]:
            msg = (
                f"rank budget {self.rank_budget} (= {self.count} samples x {self.patch**3} voxels) "
                f"is below k = {k}: G is singular by construction, leakage would read 1.0 and "
                f"cond would saturate regardless of the model. Increase --patch (>= "
                f"{int(np.ceil(k ** (1 / 3)))} for a single sample) or --n-samples."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        elif out["single_sample_rank_deficient"]:
            logger.warning(
                "patch**3 = %d < k = %d: no single sample spans the block, so the maps rest on "
                "averaging over %d samples rather than on a per-sample local subspace.",
                self.patch**3,
                k,
                self.count,
            )
        return out

    @torch.no_grad()
    def _add_jac(self, jac: torch.Tensor) -> None:
        """jac: (V, k) float32."""
        jp = _pad_flat(jac, self.vol_shape, self.grid_shape)
        jp = _patch_view(jp, self.grid_shape, self.patch).double()
        if self.patch_mask is not None:
            jp = jp * self.patch_mask.double()
        self.gram += torch.einsum("pvk,pvl->pkl", jp, jp).cpu()
        self.count += 1

    def add_sample(self, decode: Decoder, z_c: torch.Tensor, z_s: torch.Tensor) -> None:
        """One forward-mode Jacobian for one latent draw.

        Cost is (d_content + d_style) decoder evaluations, vectorised by jacfwd.
        """
        jc = jacfwd(decode, argnums=0)(z_c, z_s)  # (*vol, d_content)
        cols = [jc.reshape(-1, self.d_content)]
        if self.d_style:
            js = jacfwd(decode, argnums=1)(z_c, z_s)  # (*vol, d_style)
            cols.append(js.reshape(-1, self.d_style))
        jac = torch.cat(cols, dim=-1) if len(cols) > 1 else cols[0]
        self._add_jac(jac.detach().float())

    def run(
        self,
        decode: Decoder,
        latents: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> "GramAccumulator":
        for z_c, z_s in latents:
            self.add_sample(decode, z_c, z_s)
        return self

    @property
    def mean_gram(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("no samples accumulated")
        return self.gram / self.count


# ----------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------


def _blocks(gram: torch.Tensor, d_c: int) -> tuple[torch.Tensor, ...]:
    return gram[:, :d_c, :d_c], gram[:, :d_c, d_c:], gram[:, d_c:, d_c:]


def patch_energy(acc: GramAccumulator) -> torch.Tensor:
    """Total Jacobian energy per patch, ``sum_v sum_k J[v,k]**2`` = tr G(P)."""
    return torch.diagonal(acc.mean_gram, dim1=-2, dim2=-1).sum(-1)


def live_patches(acc: GramAccumulator, dead_fraction: float = DEAD_PATCH_FRACTION) -> torch.Tensor:
    """Boolean mask of patches carrying a measurement.

    Padding, out-of-mask and structurally invisible patches have (near-)zero energy.
    Left unmasked they would read as the *best* patches in the volume: the eigenvalue
    clamp in ``conditioning_map`` turns an all-zero ``G_cc`` into cond = 1, i.e. log10
    cond = 0, which is exactly the value a perfectly conditioned patch produces.
    """
    e = patch_energy(acc)
    peak = float(e.max())
    if not np.isfinite(peak) or peak <= 0:
        return torch.zeros_like(e, dtype=torch.bool)
    return e > dead_fraction * peak


def _reshape(vals: torch.Tensor, acc: GramAccumulator, live: torch.Tensor) -> np.ndarray:
    out = torch.where(live, vals, torch.full_like(vals, float("nan")))
    return out.reshape(patch_grid_shape(acc.grid_shape, acc.patch)).numpy()


def conditioning_map(
    acc: GramAccumulator,
    eps: float = 1e-10,
    log10: bool = True,
    dead_fraction: float = DEAD_PATCH_FRACTION,
) -> np.ndarray:
    """Local conditioning of the content Jacobian, per patch.

    Returns an array of shape patch_grid_shape.  Values are sigma_1 / sigma_d
    of the content Jacobian (log10 by default).  High values mark patches where
    at least one content direction is locally invisible.  Patches with no Jacobian
    energy at all (padding, outside the mask) are NaN, not 0.
    """
    g_cc, _, _ = _blocks(acc.mean_gram, acc.d_content)
    ev = torch.linalg.eigvalsh(g_cc).clamp_min(eps)  # ascending
    cond = torch.sqrt(ev[:, -1] / ev[:, 0])
    out = torch.log10(cond) if log10 else cond
    return _reshape(out, acc, live_patches(acc, dead_fraction))


def clamped_fraction(acc: GramAccumulator, eps: float = 1e-10) -> float:
    """Fraction of live patches whose smallest content eigenvalue hit the clamp.

    Where it did, ``conditioning_map`` reports a lower bound on cond, not cond.
    """
    g_cc, _, _ = _blocks(acc.mean_gram, acc.d_content)
    ev = torch.linalg.eigvalsh(g_cc)
    live = live_patches(acc)
    if not bool(live.any()):
        return float("nan")
    return float((ev[live, 0] <= eps).double().mean())


def per_channel_sensitivity_map(acc: GramAccumulator) -> np.ndarray:
    """RMS image sensitivity to each content channel, per patch.

    Shape (d_content, *patch_grid).  This is the estimated analogue of the
    ground-truth footprints phi_j, and is what `footprint_agreement` scores.
    """
    g_cc, _, _ = _blocks(acc.mean_gram, acc.d_content)
    diag = torch.diagonal(g_cc, dim1=-2, dim2=-1).sqrt()  # (P, d_c)
    grid = patch_grid_shape(acc.grid_shape, acc.patch)
    return diag.T.reshape(acc.d_content, *grid).numpy()


def leakage_map(
    acc: GramAccumulator,
    ridge: float = 1e-6,
    reduce: str = "max",
    dead_fraction: float = DEAD_PATCH_FRACTION,
) -> np.ndarray:
    """Canonical correlations between content and style Jacobian subspaces.

    cos^2(theta) are the eigenvalues of  G_cc^-1 G_cs G_ss^-1 G_cs^T.
    Returns values in [0, 1]; 1 means a content direction and a style direction
    are locally indistinguishable in their effect on the image.

    An all-NaN map is returned when the model has no style block: with d_style = 0
    the expression degenerates to 0 everywhere, which would read as a measured
    "no leakage" rather than as "not applicable".

    Dead patches are dropped *before* the solve, not masked after it: the ridge is
    relative to the block scale, so on an all-zero patch it is itself zero and
    ``linalg.solve`` raises instead of returning something maskable.
    """
    if reduce not in ("max", "mean"):
        raise ValueError(reduce)
    grid = patch_grid_shape(acc.grid_shape, acc.patch)
    if acc.d_style == 0:
        return np.full(grid, np.nan)

    live = live_patches(acc, dead_fraction)
    val = torch.full((live.numel(),), float("nan"), dtype=torch.float64)
    if not bool(live.any()):
        return val.reshape(grid).numpy()

    g_cc, g_cs, g_ss = _blocks(acc.mean_gram[live], acc.d_content)
    scale_c = torch.diagonal(g_cc, dim1=-2, dim2=-1).mean(-1)[:, None, None]
    scale_s = torch.diagonal(g_ss, dim1=-2, dim2=-1).mean(-1)[:, None, None]
    eye_c = torch.eye(g_cc.shape[-1], dtype=g_cc.dtype)
    eye_s = torch.eye(g_ss.shape[-1], dtype=g_ss.dtype)
    a = g_cc + ridge * scale_c * eye_c
    b = g_ss + ridge * scale_s * eye_s

    m = torch.linalg.solve(a, g_cs)  # G_cc^-1 G_cs
    n = torch.linalg.solve(b, g_cs.transpose(-1, -2))  # G_ss^-1 G_cs^T
    ev = torch.linalg.eigvals(m @ n).real.clamp(0.0, 1.0)
    cos2 = ev.sort(dim=-1, descending=True).values

    if reduce == "max":
        val[live] = cos2[:, 0].sqrt()
    else:
        k = min(acc.d_content, acc.d_style)
        val[live] = cos2[:, :k].sqrt().mean(-1)
    return val.reshape(grid).numpy()


def map_diagnostics(acc: GramAccumulator, dead_fraction: float = DEAD_PATCH_FRACTION) -> dict:
    """Everything needed to decide whether the maps above mean anything."""
    live = live_patches(acc, dead_fraction)
    e = patch_energy(acc)
    return {
        "n_patches": int(live.numel()),
        "n_live_patches": int(live.sum()),
        "live_fraction": float(live.double().mean()),
        "energy_peak": float(e.max()),
        "energy_median_live": float(e[live].median()) if bool(live.any()) else float("nan"),
        "eigenvalue_clamped_fraction": clamped_fraction(acc),
        **acc.check_rank(strict=False),
    }


# ----------------------------------------------------------------------------
# empirical: multi-seed agreement
# ----------------------------------------------------------------------------


def seed_agreement(codes_a: np.ndarray, codes_b: np.ndarray, ridge: float = 1e-3) -> dict[str, np.ndarray]:
    """Compare two independently trained encoders on the same subjects.

    codes_*: (n_samples, d) pooled latent codes from seed A and seed B.

    Returns
      mcc  : Hungarian-matched |Spearman| per channel  -- permutation only
      r2   : per channel of B predicted from ALL of A  -- any linear map
      gap  : r2 - mcc

    A large gap is the signature of rotation-only stability: the information is
    present but the basis is not fixed.  This is the quantity the specialization
    penalties are meant to close, and is more informative than MCC alone.
    """
    ra = rankdata(codes_a, axis=0)
    rb = rankdata(codes_b, axis=0)
    ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-12)
    rb = (rb - rb.mean(0)) / (rb.std(0) + 1e-12)
    corr = np.abs(ra.T @ rb) / len(ra)
    row, col = linear_sum_assignment(-corr)
    mcc = corr[row, col]

    xa = (codes_a - codes_a.mean(0)) / (codes_a.std(0) + 1e-12)
    xb = (codes_b - codes_b.mean(0)) / (codes_b.std(0) + 1e-12)
    g = xa.T @ xa + ridge * len(xa) * np.eye(xa.shape[1])
    w = np.linalg.solve(g, xa.T @ xb)
    resid = ((xb - xa @ w) ** 2).mean(0)
    r2 = 1.0 - resid
    r2 = r2[col]

    return {"mcc": mcc, "r2": r2, "gap": r2 - mcc}


# ----------------------------------------------------------------------------
# synthetic-only: validation against ground-truth footprints
# ----------------------------------------------------------------------------


def ground_truth_footprints(
    render: Callable[[np.ndarray], np.ndarray],
    base_gammas: np.ndarray,
    delta: float = 0.05,
) -> np.ndarray:
    """Central-difference footprints phi_j(u) = d c(u) / d gamma_j.

    render: gamma (d,) -> tissue map or continuous volume (*vol_shape)
    base_gammas: (n_base, d) points at which to evaluate; averaged, since the
    renderer is nonlinear and a single base point is not representative.

    Cost is 2 * d * n_base renders.  Returns (d, *vol_shape).
    """
    n_base, d = base_gammas.shape
    out = None
    for g in base_gammas:
        for j in range(d):
            gp, gm = g.copy(), g.copy()
            gp[j] += delta
            gm[j] -= delta
            diff = render(gp).astype(np.float64) - render(gm).astype(np.float64)
            diff /= 2 * delta
            if out is None:
                out = np.zeros((d, *diff.shape), dtype=np.float64)
            out[j] += np.abs(diff)
    return out / n_base


def pool_footprints(truth: np.ndarray, patch: int) -> np.ndarray:
    """(d, *vol) -> (d, *patch_grid) by mean pooling, zero-padded like the accumulator."""
    d = truth.shape[0]
    vol = truth.shape[1:]
    tgt = padded_shape(vol, patch)
    if tuple(vol) != tgt:
        padded = np.zeros((d, *tgt), dtype=truth.dtype)
        padded[:, : vol[0], : vol[1], : vol[2]] = truth
        truth = padded
    grid = patch_grid_shape(tgt, patch)
    return truth.reshape(d, grid[0], patch, grid[1], patch, grid[2], patch).mean(axis=(2, 4, 6))


def match_footprints(estimated: np.ndarray, truth: np.ndarray, patch: int) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match estimated channels to true factors on spatial correlation.

    ``footprint_agreement`` assumes the two stacks are already aligned, but a model
    channel index has no reason to equal a factor index and ``d_content`` is usually far
    larger than the factor count.  Returns ``(estimated[matched], assignment)`` where
    ``assignment[j]`` is the estimated channel scoring factor ``j``.
    """
    pooled = pool_footprints(truth, patch)
    d_est = estimated.shape[0]
    d_true = pooled.shape[0]
    corr = np.zeros((d_est, d_true))
    for i in range(d_est):
        a = estimated[i].ravel()
        a = np.nan_to_num(a - np.nanmean(a))
        for j in range(d_true):
            b = pooled[j].ravel()
            b = b - b.mean()
            corr[i, j] = abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    row, col = linear_sum_assignment(-corr)
    assignment = np.zeros(d_true, dtype=int)
    assignment[col] = row
    return estimated[assignment], assignment


def footprint_agreement(estimated: np.ndarray, truth: np.ndarray, patch: int) -> np.ndarray:
    """Spatial correlation between estimated and true footprints, per channel.

    estimated: (d, *patch_grid) from `per_channel_sensitivity_map`
    truth:     (d, *vol_shape)   from `ground_truth_footprints`

    Channels must already be matched (use `match_footprints`, the Hungarian assignment
    from `seed_agreement`, or a global MCC against ground-truth latents).
    """
    pooled = pool_footprints(truth, patch)
    d = pooled.shape[0]
    if estimated.shape[0] != d:
        raise ValueError(
            f"estimated has {estimated.shape[0]} channels but truth has {d} factors; " f"run match_footprints() first"
        )
    if estimated.shape[1:] != pooled.shape[1:]:
        raise ValueError(f"patch grids differ: estimated {estimated.shape[1:]} vs pooled truth {pooled.shape[1:]}")

    out = np.zeros(d)
    for j in range(d):
        a = np.nan_to_num(estimated[j].ravel())
        b = pooled[j].ravel()
        a = a - a.mean()
        b = b - b.mean()
        out[j] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return out


def footprint_gram(truth: np.ndarray) -> np.ndarray:
    """Normalised Gram of the ground-truth footprints.

    Run this BEFORE training anything.  Near-degenerate off-diagonal entries
    identify pairs of factors that are not separable by the renderer itself,
    and therefore cannot be separated by any model.  Expect gamma_0 (brain
    size) and gamma_5 (cortical thickness) to be the worst pair, since both
    move the GM/background boundary and are distinguished only by the WM/GM
    boundary.

    This is the footprint-space view of what ``eval/generator_defects.py`` measures in
    factor space; the two should agree on which pairs are hopeless.
    """
    d = truth.shape[0]
    f = truth.reshape(d, -1)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    return f @ f.T


# ----------------------------------------------------------------------------
# repo adapters
# ----------------------------------------------------------------------------


def decoder_closure(dec: torch.nn.Module, base_z: torch.Tensor, base_style: torch.Tensor | None) -> Decoder:
    """Bind an operating point and expose the module's 1-D ``decode(z_c, z_s)`` contract.

    ``z_c`` / ``z_s`` are per-channel offsets broadcast over the latent grid (see the
    module docstring).  ``base_z`` and ``base_style`` must be single-sample tensors.
    """

    def decode(z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        z = base_z + z_c.reshape(1, -1, 1, 1, 1)
        if base_style is None:
            return dec(z)[0, 0]
        return dec(z, style=base_style + z_s.reshape(1, -1, 1, 1, 1))[0, 0]

    return decode


def renderer_closure(inner, lat: dict, clean: bool) -> Callable[[np.ndarray], np.ndarray]:
    """``gamma (9,) -> tissue map``, holding this sample's nuisance fields fixed.

    The nuisance grids (``z_deformation``, ``z_fissure``) are kept at the sample's own
    values rather than resampled, so the central difference isolates ``d c / d gamma_j``
    instead of mixing in the nuisance field the footprint is meant to be free of.
    """
    z_def, z_fis = lat["z_deformation"], lat["z_fissure"]
    z_les = lat.get("z_lesion")

    def render(gamma: np.ndarray) -> np.ndarray:
        zc = torch.as_tensor(np.asarray(gamma), dtype=torch.float32)
        with torch.no_grad():
            tissue, _ = inner.renderer.render_structure(
                zc, z_def, z_fis, device=torch.device("cpu"), clean=clean, z_lesion=z_les
            )
        return tissue.detach().cpu().numpy()

    return render


def latent_draws(base_z: torch.Tensor, base_style: torch.Tensor | None, n: int, scale: float, seed: int):
    """``n`` offset draws around the operating point.

    ``scale`` is relative to the per-channel spread of the captured latent, so the
    linearisation point moves by a comparable amount whatever the code scale.  ``scale=0``
    evaluates the Jacobian exactly at the operating point.
    """
    gen = torch.Generator().manual_seed(seed)
    d_c = base_z.shape[1]
    d_s = 0 if base_style is None else base_style.shape[1]
    sd_c = base_z.reshape(d_c, -1).std(dim=1) if scale else torch.zeros(d_c)
    sd_s = base_style.reshape(d_s, -1).std(dim=1) if (scale and d_s) else torch.zeros(d_s)
    for _ in range(n):
        z_c = torch.randn(d_c, generator=gen) * sd_c * scale
        z_s = torch.randn(d_s, generator=gen) * sd_s * scale
        yield z_c, z_s


def content_style_dims(base_z: torch.Tensor, base_style: torch.Tensor | None) -> tuple[int, int]:
    return int(base_z.shape[1]), 0 if base_style is None else int(base_style.shape[1])


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(str(type(o)))


def _nan_summary(a: np.ndarray) -> dict:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {k: float("nan") for k in ("min", "p25", "median", "p75", "max", "mean")}
    return {
        "min": float(finite.min()),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def group_mask(subjects) -> torch.Tensor:
    """Voxels in-brain for EVERY subject, matching ``recovery_maps.Fields.mask``.

    One Gram is accumulated across subjects, so the mask has to be the same for all of
    them; taking subject 0's would weight each patch by a different voxel count per
    subject and read that as structure.
    """
    m = subjects[0]["subject_mask"].clone()
    for subj in subjects[1:]:
        m &= subj["subject_mask"]
    return m


def run_arm(decoder, arm_name, norm_mode, subjects, cli, vol_shape, mask=None):
    """Accumulate the Gram for one arm across subjects, then reduce to maps."""
    d_c, d_s = content_style_dims(subjects[0]["z"], subjects[0]["style"])
    acc = GramAccumulator(
        d_content=d_c,
        d_style=d_s,
        vol_shape=vol_shape,
        patch=cli.patch,
        mask=mask,
        pad=True,
    )
    t0 = time.time()
    for si, subj in enumerate(subjects):
        dec = jspread.build_arm(decoder, None, norm_mode, subj["z"], subj["style"])
        decode = decoder_closure(dec, subj["z"], subj["style"])
        acc.run(decode, latent_draws(subj["z"], subj["style"], cli.draws, cli.jitter, cli.seed * 31 + si))
        del dec
    acc.check_rank(strict=not cli.allow_rank_deficient)
    logger.info("arm %-12s %d samples in %.0fs", arm_name, acc.count, time.time() - t0)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0, help="Decoder level to probe (0 = finest).")
    ap.add_argument("--patch", type=int, default=4, help="Cubic patch side, in output voxels.")
    ap.add_argument("--n-subjects", type=int, default=8)
    ap.add_argument("--draws", type=int, default=2, help="Latent draws per subject.")
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.25,
        help="Latent offset scale, relative to the per-channel std of the captured code. 0 = evaluate at "
        "the operating point exactly.",
    )
    ap.add_argument("--mask", action="store_true", help="Restrict each Gram to in-brain voxels.")
    ap.add_argument("--reduce", choices=["max", "mean"], default="max", help="Leakage reduction over angles.")
    ap.add_argument("--skip-frozen", action="store_true", help="Run the 'full' arm only.")
    ap.add_argument("--synthetic", action="store_true", help="Also validate against renderer footprints.")
    ap.add_argument("--footprints-only", action="store_true", help="Renderer footprints only; no checkpoint needed.")
    ap.add_argument("--footprint-bases", type=int, default=8, help="Base points for the central-difference footprints.")
    ap.add_argument("--footprint-delta", type=float, default=0.05)
    ap.add_argument("--allow-rank-deficient", action="store_true", help="Warn instead of refusing a singular G.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--threads", type=int, default=8)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    torch.set_num_threads(cli.threads)
    device = torch.device("cpu")  # Jacobian magnitudes are the measurement: float32, no AMP

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    os.makedirs(cli.out_dir, exist_ok=True)
    pre = cli.prefix
    report: dict = {
        "run_dir": os.path.abspath(cli.run_dir),
        "repo_git_sha": jspread.git_sha(os.path.dirname(os.path.abspath(__file__)) + "/.."),
        "level": cli.level,
        "patch": cli.patch,
        "n_subjects": cli.n_subjects,
        "draws_per_subject": cli.draws,
        "jitter": cli.jitter,
        "masked": bool(cli.mask),
        "seed": cli.seed,
    }

    # The footprints are a property of the renderer alone, so --footprints-only reads the
    # run's settings for the generator config and never touches a checkpoint.
    model = None
    if cli.footprints_only:
        args = load_run_args(cli.run_dir)
    else:
        ckpt = cli.checkpoint or os.path.join(cli.run_dir, "vqvae_best.pt")
        model, args, device = load_model_from_run_dir(cli.run_dir, ckpt, device)
        model.eval()
        report["checkpoint"] = os.path.abspath(ckpt)
        report["checkpoint_sha256"] = jspread.file_sha256(ckpt)

    ds = build_synthetic_test_set(args, max(cli.n_subjects, 8), cache=False, causal=True)
    inner = getattr(ds, "_inner", None)

    # ---------------- footprints (no checkpoint needed) ------------------------------
    if cli.synthetic or cli.footprints_only:
        if inner is None or not hasattr(inner, "renderer"):
            raise SystemExit("--synthetic needs a synthetic run: this run's dataset has no renderer.")
        gen = np.random.RandomState(cli.seed)
        lat0 = ds[0]["gt_latents"]
        n_factors = int(inner.renderer.N_CONTENT_COMPONENTS)
        bases = np.stack(
            [
                np.asarray(ds[int(i)]["gt_latents"]["z_content"], dtype=np.float64).ravel()[:n_factors]
                for i in gen.choice(len(ds), size=min(cli.footprint_bases, len(ds)), replace=False)
            ]
        )
        render = renderer_closure(inner, lat0, clean=getattr(inner, "clean_content", False))
        t0 = time.time()
        truth = ground_truth_footprints(render, bases, delta=cli.footprint_delta)
        logger.info("ground-truth footprints: %s in %.0fs", truth.shape, time.time() - t0)
        gram = footprint_gram(truth)
        names = CONTENT_FACTOR_NAMES[:n_factors]
        # A zero diagonal is not a degenerate pair but a dead factor: the renderer moved no
        # voxel when it moved, so no encoder can recover it and every off-diagonal entry in
        # its row is a meaningless 0 rather than "orthogonal to everything".
        inert = [names[j] for j in range(n_factors) if gram[j, j] < 0.5]
        live_idx = [j for j in range(n_factors) if gram[j, j] >= 0.5]
        off = np.abs(np.triu(gram - np.eye(n_factors), 1))
        off[[j for j in range(n_factors) if j not in live_idx], :] = 0.0
        off[:, [j for j in range(n_factors) if j not in live_idx]] = 0.0
        worst = np.unravel_index(np.argmax(off), off.shape)
        report["footprints"] = {
            "factor_names": names,
            "n_base_points": int(bases.shape[0]),
            "delta": cli.footprint_delta,
            "gram": gram,
            "inert_factors": inert,
            "worst_pair": [names[worst[0]], names[worst[1]]],
            "worst_pair_cosine": float(gram[worst]),
        }
        np.save(os.path.join(cli.out_dir, f"{pre}gt_footprints.npy"), truth.astype(np.float32))
        print("\n" + "=" * 88)
        print("GENERATOR FOOTPRINTS  normalised Gram (checkpoint-free)")
        print("=" * 88)
        print("  " + " ".join(f"{n[:9]:>9s}" for n in names))
        for i, n in enumerate(names):
            print(f"  {' '.join(f'{gram[i, j]:9.3f}' for j in range(n_factors))}   {n}")
        if inert:
            print(
                f"\n  INERT (zero footprint — the renderer never moves for these, so no model can "
                f"recover them): {', '.join(inert)}"
            )
        print(f"  worst separable LIVE pair: {names[worst[0]]} / {names[worst[1]]}  cos = {gram[worst]:.3f}")
        if cli.footprints_only:
            with open(os.path.join(cli.out_dir, f"{pre}identifiability_maps.json"), "w") as f:
                json.dump(report, f, indent=2, default=_json_default)
            print(f"\n  wrote {os.path.join(cli.out_dir, pre + 'identifiability_maps.json')}")
            return 0

    # ---------------- guard scan -----------------------------------------------------
    decoder = model.decoders[cli.level]
    jspread.disable_inplace(decoder)
    scan = rfmod.scan_for_global_ops(decoder)
    report["guard"] = scan
    print("\n" + "=" * 88)
    print("GUARD SCAN  spatially global operations in the decoder")
    print("=" * 88)
    if scan["ok"]:
        print("  PASS — no global or non-local operation found; patch-local claims are well posed.")
    else:
        print(f"  FAIL — {len(scan['offenders'])} spatially global operation(s):")
        for o in scan["offenders"]:
            print(f"    {o['module']:32s} {o['reason']}")
        print("\n  Every output voxel depends on every latent channel through this path, so the")
        print("  'full' maps below are not purely local.  Read the frozen_norm arm alongside them.")

    # ---------------- operating points ------------------------------------------------
    subjects = []
    for i in range(cli.n_subjects):
        item = ds[i]
        x = torch.stack([item["image"][0], item["image"][1]], 0).to(device)
        z, style = jspread.capture_decoder_inputs(model, x, cli.level)
        subjects.append(
            {
                "z": z[:1].contiguous(),
                "style": None if style is None else style[:1].contiguous(),
                "mask": item["mask"][0][0],
            }
        )

    with torch.no_grad():
        probe_out = decoder(subjects[0]["z"], style=subjects[0]["style"])
    vol_shape = tuple(probe_out.shape[2:])
    d_c, d_s = content_style_dims(subjects[0]["z"], subjects[0]["style"])
    for subj in subjects:
        subj["subject_mask"] = subj["mask"].bool() if tuple(subj["mask"].shape) == vol_shape else None
    gmask = None
    if cli.mask:
        if subjects[0]["subject_mask"] is None:
            raise SystemExit(
                f"--mask needs a brain mask at the decoder output shape {vol_shape}; got {subjects[0]['mask'].shape}"
            )
        gmask = group_mask(subjects)
        report["group_mask_voxel_fraction"] = float(gmask.double().mean())

    report.update(
        {
            "volume_shape": list(vol_shape),
            "padded_shape": list(padded_shape(vol_shape, cli.patch)),
            "patch_grid": list(patch_grid_shape(padded_shape(vol_shape, cli.patch), cli.patch)),
            "d_content": d_c,
            "d_style": d_s,
            "latent_grid": list(subjects[0]["z"].shape[2:]),
        }
    )
    print(f"\n  decoder {cli.level}: latent {tuple(subjects[0]['z'].shape[1:])} -> volume {vol_shape}")
    print(f"  d_content = {d_c}, d_style = {d_s}, patch = {cli.patch} ({cli.patch ** 3} voxels)")

    # ---------------- arms -------------------------------------------------------------
    arms = [("full", "live")] + ([] if cli.skip_frozen else [("frozen_norm", "frozen")])
    accs, maps = {}, {}
    for arm_name, norm_mode in arms:
        acc = run_arm(decoder, arm_name, norm_mode, subjects, cli, vol_shape, mask=gmask)
        accs[arm_name] = acc
        cond = conditioning_map(acc)
        leak = leakage_map(acc, reduce=cli.reduce)
        sens = per_channel_sensitivity_map(acc)
        maps[arm_name] = {"cond": cond, "leak": leak, "sens": sens}
        np.save(os.path.join(cli.out_dir, f"{pre}cond_{arm_name}.npy"), cond.astype(np.float32))
        np.save(os.path.join(cli.out_dir, f"{pre}leak_{arm_name}.npy"), leak.astype(np.float32))
        np.save(os.path.join(cli.out_dir, f"{pre}sensitivity_{arm_name}.npy"), sens.astype(np.float32))
        report.setdefault("arms", {})[arm_name] = {
            "log10_cond": _nan_summary(cond),
            "leakage": _nan_summary(leak),
            "diagnostics": map_diagnostics(acc),
        }

    print("\n" + "=" * 88)
    print("MAPS  per-patch content conditioning and content/style leakage")
    print("=" * 88)
    print(f"  {'arm':14s} {'log10 cond':>28s} | {'leakage (canon. corr.)':>28s} | {'live':>6s} {'clamped':>8s}")
    print(f"  {'':14s} {'p25':>9s}{'median':>9s}{'p75':>9s} | {'p25':>9s}{'median':>9s}{'p75':>9s} |")
    for arm_name, _ in arms:
        s = report["arms"][arm_name]
        c, lk, d = s["log10_cond"], s["leakage"], s["diagnostics"]
        print(
            f"  {arm_name:14s} {c['p25']:9.3f}{c['median']:9.3f}{c['p75']:9.3f} | "
            f"{lk['p25']:9.3f}{lk['median']:9.3f}{lk['p75']:9.3f} | "
            f"{d['live_fraction']:6.3f} {d['eigenvalue_clamped_fraction']:8.3f}"
        )
    print("  'live' = patches with non-zero Jacobian energy (the rest are NaN, not 0).")
    print("  'clamped' = live patches whose smallest content eigenvalue hit the floor; there")
    print("  log10 cond is a LOWER BOUND — the direction is invisible past the measurable range.")

    if len(arms) > 1:
        f_c = report["arms"]["full"]["log10_cond"]["median"]
        z_c_ = report["arms"]["frozen_norm"]["log10_cond"]["median"]
        f_l = report["arms"]["full"]["leakage"]["median"]
        z_l = report["arms"]["frozen_norm"]["leakage"]["median"]
        report["normalization_effect"] = {
            "log10_cond_median_full": f_c,
            "log10_cond_median_frozen": z_c_,
            "log10_cond_delta": f_c - z_c_,
            "leakage_median_full": f_l,
            "leakage_median_frozen": z_l,
            "leakage_delta": f_l - z_l,
        }
        print(
            f"\n  normalization path: log10 cond {f_c:.3f} (full) vs {z_c_:.3f} (frozen), "
            f"delta {f_c - z_c_:+.3f}; leakage {f_l:.3f} vs {z_l:.3f}, delta {f_l - z_l:+.3f}."
        )
        print("  A large delta means these maps are reading the normalization, not local structure.")

    # ---------------- synthetic validation ---------------------------------------------
    if cli.synthetic:
        truth = np.load(os.path.join(cli.out_dir, f"{pre}gt_footprints.npy"))
        names = report["footprints"]["factor_names"]
        for arm_name, _ in arms:
            matched, assignment = match_footprints(maps[arm_name]["sens"], truth, cli.patch)
            agree = footprint_agreement(matched, truth, cli.patch)
            report["arms"][arm_name]["footprint_agreement"] = {
                "per_factor": {n: float(a) for n, a in zip(names, agree)},
                "mean": float(np.mean(agree)),
                "matched_channel": {n: int(c) for n, c in zip(names, assignment)},
            }
        print("\n" + "=" * 88)
        print("SYNTHETIC VALIDATION  estimated vs renderer footprints (Hungarian-matched)")
        print("=" * 88)
        print(f"  {'factor':22s} " + " ".join(f"{a:>14s}" for a, _ in arms))
        for j, n in enumerate(names):
            vals = " ".join(f"{report['arms'][a]['footprint_agreement']['per_factor'][n]:14.3f}" for a, _ in arms)
            print(f"  {n:22s} {vals}")
        means = " ".join(f"{report['arms'][a]['footprint_agreement']['mean']:14.3f}" for a, _ in arms)
        print(f"  {'MEAN':22s} {means}")

    json_path = os.path.join(cli.out_dir, f"{pre}identifiability_maps.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\n  wrote {json_path}")
    print(f"  wrote {os.path.join(cli.out_dir, pre + 'cond_<arm>.npy')}, leak_<arm>.npy, sensitivity_<arm>.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
