"""Pre-``7ac56a3`` ``render_structure`` — for scoring checkpoints trained before the fix.

Commit ``7ac56a3`` (2026-07-05) rewrote three content factors in
``eval.synthetic_dataset.PseudoMRIRenderer.render_structure``:

* **lesion** (``z[2:5]``) — the centre was an absolute coordinate, so the lesion landed
  outside the WM in ~95% of draws (lesion-present fraction ~5%, vs ~100% now).
* **temporal atrophy** (``z[6]``) — a product of two half-space sigmoids shrank roughly
  half the brain rather than a compact temporal region.
* **sulcal widening** (``z[8]``) — rode on the *nuisance* deformation field, so it
  vanished entirely under ``--synthetic-clean-content``.

A checkpoint trained before that commit never saw the current rendering.  Scoring it on
the current generator asks the encoder to recover factors that did not exist during its
training, which makes those factors' numbers meaningless rather than merely low.  This
module restores the old function verbatim (the only edit is ``self`` -> ``r``) so the
eval generator's vintage can be matched to a training SHA without a git worktree.

The fix touched ``render_structure`` and nothing else in that file — ``_upsample_field``,
``coords`` and ``CLEAN_NUISANCE_SCALE`` (0.0, unchanged) are shared — so swapping this one
method is a faithful restoration rather than an approximation.

Numbers produced with the legacy renderer are NOT comparable with current-generator
numbers for ``lesion_*``, ``temporal_atrophy`` or ``sulcal_widening``.  ``brain_size``,
``ventricle_size``, ``cortical_thickness`` and ``lr_asymmetry`` are untouched by the fix
and stay comparable across both.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from eval.synthetic_dataset import CLEAN_NUISANCE_SCALE

logger = logging.getLogger(__name__)

# Factors the fix changed — their scores are only meaningful when the generator vintage
# matches the checkpoint. Used by callers to caveat their output.
FIXED_FACTORS = ("lesion_x", "lesion_y", "lesion_z", "temporal_atrophy", "sulcal_widening")
UNAFFECTED_FACTORS = ("brain_size", "ventricle_size", "cortical_thickness", "lr_asymmetry")
FIX_COMMIT = "7ac56a3"


def render_structure_legacy(r, z_content, z_deformation, z_fissure, device, clean=False):
    """``PseudoMRIRenderer.render_structure`` as of ``7ac56a3^`` (verbatim; ``self`` -> ``r``)."""
    # Right-pad z_content with zeros when the caller supplies fewer dims
    # than the renderer consumes (back-compat with n_content=5 runs).
    if z_content.numel() < r.N_CONTENT_COMPONENTS:
        pad = torch.zeros(r.N_CONTENT_COMPONENTS - z_content.numel())
        z_content = torch.cat([z_content.flatten(), pad])

    # clean-content mode: tanh-squash (monotone → fully recoverable) instead of the
    # hard clamp (which flattens the ~1/3 of N(0,1) values that overflow ±1), and
    # zero out the unlabeled deformation/fissure nuisance so the named factors dominate.
    nuisance = CLEAN_NUISANCE_SCALE if clean else 1.0

    def _sq(z, a=1.0):
        return a * torch.tanh(z) if clean else z.clamp(-a, a)

    radii_wm = 0.5 + _sq(z_content[0]) * 0.1 * r.content_scale
    cortical_thickness = 0.15 + _sq(z_content[5]) * 0.06 * r.content_scale
    radii_gm = radii_wm + cortical_thickness
    ventricle_size = 0.15 + _sq(z_content[1]) * 0.05 * r.content_scale

    dist = torch.norm(r.coords, dim=-1)

    # Sulcal widening: amplifies the deformation field at boundary regions,
    # making gyri/sulci more or less pronounced.
    sulcal_factor = 1.0 + _sq(z_content[8]) * 0.6 * r.content_scale
    deformation = r._upsample_field(z_deformation, device) * 0.1 * sulcal_factor * nuisance
    deformed_dist = dist + deformation

    # Temporal-lobe atrophy: shrink the brain radius in the inferior–
    # posterior octant (y < 0, z < 0) to simulate hippocampal / medial-
    # temporal volume loss. Smooth falloff via a sigmoid so the boundary
    # isn't artificially sharp.
    y_coords = r.coords[..., 1]
    z_coords = r.coords[..., 2]
    temporal_weight = torch.sigmoid(-8 * y_coords) * torch.sigmoid(-8 * z_coords)
    temporal_shrink = _sq(z_content[6]) * 0.12 * r.content_scale
    deformed_dist = deformed_dist + temporal_weight * temporal_shrink

    # Left–right asymmetry: differential atrophy across hemispheres.
    # Positive z_content[7] → left hemisphere (x < 0) more atrophied.
    x_coords = r.coords[..., 0]
    lr_weight = torch.tanh(-3 * x_coords)  # smooth L/R gradient, ∈ (−1, 1)
    lr_shift = _sq(z_content[7]) * 0.08 * r.content_scale
    deformed_dist = deformed_dist + lr_weight * lr_shift

    mask_gm = deformed_dist < radii_gm
    mask_wm = deformed_dist < radii_wm

    ventricle_split = torch.abs(x_coords) > 0.05
    mask_csf = (deformed_dist < ventricle_size) & ventricle_split

    fissure_noise = r._upsample_field(z_fissure, device) * 0.05 * nuisance
    fissure_mask = (torch.abs(x_coords + fissure_noise) < 0.03) & mask_gm

    tissue_map = torch.zeros_like(dist, dtype=torch.long)
    tissue_map[mask_gm] = 3
    tissue_map[mask_wm] = 2
    tissue_map[mask_csf] = 1
    tissue_map[fissure_mask] = 1

    lesion_xyz = _sq(z_content[2:5], 0.6).to(device)
    lesion_mask = (torch.norm(r.coords - lesion_xyz, dim=-1) < 0.1) & mask_wm

    return tissue_map, lesion_mask


def _find_renderer(dataset):
    """Reach the PseudoMRIRenderer through Subset / SyntheticBrainDataset wrappers."""
    d = dataset
    for _ in range(4):  # Subset -> SyntheticBrainDataset -> Synthetic3DDisentanglementDataset
        if hasattr(d, "renderer"):
            return d, d.renderer
        d = getattr(d, "dataset", None) or getattr(d, "_inner", None)
        if d is None:
            break
    raise AttributeError(
        "no `.renderer` found — the legacy renderer only applies to synthetic_mode='pseudo_mri' datasets"
    )


def _clear_caches(dataset):
    """Drop rendered-image and fixed-reference caches so the swap actually takes effect.

    Without this the dataset serves volumes rendered by the *other* generator, which is
    silent and produces a result that looks plausible and is wrong.
    """
    d = dataset
    for _ in range(4):
        if getattr(d, "_cache", None) is not None:
            d._cache = [None] * len(d._cache)
        if hasattr(d, "_fixed_mean"):  # recompute fixed_reference norm on the active renderer
            d._fixed_mean = d._fixed_scale = None
        nxt = getattr(d, "dataset", None) or getattr(d, "_inner", None)
        if nxt is None:
            break
        d = nxt


def lesion_present_fraction(renderer, n=200, seed=1, k_deformation=4, k_fissure=8):
    """Fraction of random draws whose lesion mask is non-empty.

    The sharpest available signature of which generator is live: ~5% under the legacy
    renderer (the lesion usually landed outside the WM), ~100% under the current one.
    The nuisance fields are passed as zeros, so the grid sizes only need to be valid
    shapes — they do not affect the lesion.
    """
    g = torch.Generator().manual_seed(seed)
    dev = renderer.coords.device
    nc = renderer.N_CONTENT_COMPONENTS
    hits = []
    for _ in range(n):
        zc = torch.randn(nc, generator=g).to(dev)
        _, lesion = renderer.render_structure(
            zc,
            torch.zeros(k_deformation, k_deformation, k_deformation, device=dev),
            torch.zeros(k_fissure, k_fissure, k_fissure, device=dev),
            dev,
        )
        hits.append(int(lesion.sum()) > 0)
    return float(np.mean(hits))


def use_legacy_renderer(dataset, verify=True):
    """Swap ``render_structure`` on *dataset*'s renderer to the pre-``7ac56a3`` version.

    Returns a zero-argument callable that restores the current renderer.  Caches are
    cleared on both the swap and the restore.  With ``verify``, logs the lesion-present
    fraction before and after — the check that the swap actually took effect.
    """
    holder, renderer = _find_renderer(dataset)
    if hasattr(holder, "_legacy_orig_render_structure"):
        logger.warning("legacy renderer already active — not re-applying")
        return lambda: None

    before = lesion_present_fraction(renderer) if verify else float("nan")
    holder._legacy_orig_render_structure = renderer.render_structure
    renderer.render_structure = render_structure_legacy.__get__(renderer, type(renderer))
    _clear_caches(dataset)
    after = lesion_present_fraction(renderer) if verify else float("nan")

    if verify:
        logger.info(
            "legacy renderer (pre-%s) ACTIVE — lesion-present fraction %.0f%% -> %.0f%% (expect ~100%% -> ~5%%)",
            FIX_COMMIT,
            before * 100,
            after * 100,
        )
        if after > 0.5:
            logger.warning(
                "lesion-present fraction is still %.0f%% — the swap may not have taken effect; "
                "treat %s scores as unreliable",
                after * 100,
                ", ".join(FIXED_FACTORS),
            )

    def restore():
        renderer.render_structure = holder._legacy_orig_render_structure
        del holder._legacy_orig_render_structure
        _clear_caches(dataset)
        logger.info("current renderer restored")

    return restore
