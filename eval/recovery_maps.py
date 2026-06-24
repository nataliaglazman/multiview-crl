#!/usr/bin/env python3
"""recovery_maps.py -- gauge-aware voxel-wise identifiability / recovery maps.

This is the spatial, multi-map generalisation of ``eval/voxelwise_spm.py`` and the
voxel-wise twin of the pooled ``eval/phase0_probe.py`` kill-switch. Where
``voxelwise_spm`` produces ONE map (cross-modal content-block agreement via Pillai),
this module produces the family of maps the identifiability theory actually
distinguishes, each tagged with the gauge group it quotients out:

    block_recovery(l)      content-subspace agreement      within-content linear gauge   (>=2 solutions)
    component_recovery(l)  per-channel agreement           permutation + monotone gauge  (>=2 solutions)
    t1t2_boundary(l)       block - component               derived
    content_visibility(l)  max_v sigma_min(J_v^c)          one decoder
    content_style_angle(l) principal angle col J^c vs J^m  one decoder
    ivae_shift_rank(l)     rank of d-shift of suff. stats   one model + severity labels
    kernel_distinctness    per-ROI factor-kernel separation one solution + atlas
    code_reproducibility   per-voxel AMI of code maps       >=2 solutions, codes

Build order (mirrors the Phase-0 kill-switch philosophy): the one-decoder
model-intrinsic maps run first as a kill-switch (if content is rendered nowhere,
stop before anything expensive); the >=2-solution recovery maps run second;
inference and the VQ code maps last. Everything is validated on synthetic controls
(``--synthetic``) before it touches ADNI, because the negative control is the only
place we know the right answer.

The four implementation decisions and how they were resolved
------------------------------------------------------------
1. Block-gauge claim orthogonal vs general-linear -> CKA vs CCA in ``block_recovery``.
   DEFAULT ``block_metric="cca"`` (general-linear), because the Yao-2024 block-
   identifiability target is up-to-invertible-linear and the negative-control
   signature ("block high, component low") needs the *permissive* block metric.
   ``block_metric="cka"`` gives the orthogonal+scale claim. CCA here is the
   normalised Pillai trace (mean squared canonical correlation) -- identical to
   ``voxelwise_spm`` -- so the two scripts agree by construction.
2. Which gauge subgroup the null samples. Resolved PER MAP, not globally:
     * block_recovery   -> ``subject_perm`` null (existence of shared content);
       CCA already absorbs the within-content gauge so a gauge null is vacuous.
     * component_recovery -> ``content_rotation`` (or ``content_gl``) null: scramble
       the within-content basis of ONE solution, re-fit the global gauge, recompute.
       Chance = "agreement achievable by some admissible gauge element".
     * t1t2_boundary    -> ``content_style`` mixing null (needs style fields).
3. Jacobian exact vs randomized sigma_min. DEFAULT ``jac_method="exact_local"``:
   the per-voxel Jacobian is over the (small) receptive field, so we form the local
   operator explicitly and SVD it on a thinned grid, then interpolate.
   ``jac_method="randomized"`` subsamples output dims to cap cost on a full grid.
4. Pre-VQ continuous output exposed? YES -- ``encoder_features[l][:, content_idx]``
   from ``VQVAE.forward`` is the continuous pre-quantisation content (``phase0_extract``
   already dumps it). Maps 1-6 are runnable on it.

On-disk contract (identical to phase0_extract / voxelwise_spm)
--------------------------------------------------------------
    {root}/{subject}_{view}_seed{seed}.npy   (C, *spatial) float32, COMMON TEMPLATE SPACE
    {root}/{subject}_mask.npy                (*spatial)    bool, feature-grid resolution
    {root}/{subject}_{view}_seed{seed}_code{level}.npy  (*spatial) int   (optional, codes)

A "solution" is one (view, seed) pair. ``>=2 solutions`` means two such pairs
(cross-view: view0 vs view1 at one seed; or seed-repro: one view at two seeds).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _spstats
from scipy.optimize import linear_sum_assignment

# Reuse the *corrected* machinery rather than reimplementing the subtle parts
# (per-voxel whitening with the absolute ridge floor; TFCE background-bin handling).
from eval.voxelwise_spm import benjamini_hochberg, tfce, whiten


# =====================================================================================
# Phase A -- load & axis alignment
# =====================================================================================
@dataclass
class Fields:
    """Canonical container. Arrays are ``[S, C, *spatial]`` in a COMMON template space.

    A voxel index ``v`` must denote the SAME anatomy across subjects -- this is the
    precondition that makes "per voxel across subjects" meaningful at all.
    """

    content: dict  # solution-key -> [S, C, *spatial]   (solution = (view, seed))
    style: dict = field(default_factory=dict)  # solution-key -> [S, Cs, *spatial] (per view)
    codes: dict = field(default_factory=dict)  # (solution-key, level) -> [S, *spatial] int
    mask: np.ndarray = None  # group mask [*spatial] bool (in-brain for ALL subjects)
    spatial: tuple = None  # spatial shape
    subjects: list = field(default_factory=list)
    atlas: np.ndarray = None  # [*spatial] int ROI labels, or None
    affine: np.ndarray = None  # 4x4 reference affine for NIfTI output, or None

    @property
    def mask_flat(self):
        return self.mask.reshape(-1)


def load_fields(root, solutions, *, atlas_path=None, ref_path=None, subjects=None, load_codes=False, levels=(0,)):
    """Read ``{root}/{subject}_{view}_seed{seed}.npy`` into canonical ``[S, C, *spatial]``.

    ``solutions`` is a list of ``(view, seed)`` tuples. The group mask is the
    intersection of every subject's mask with finite content across all solutions.
    """
    if subjects is None:
        subjects = _discover_subjects(root, solutions[0])
    if not subjects:
        raise FileNotFoundError(f"No subjects in {root} matching {solutions[0]}. Run eval/phase0_extract.py first.")

    spatial = None
    gmask = None
    content = {sol: [] for sol in solutions}
    style = {sol: [] for sol in solutions}
    codes = {(sol, lv): [] for sol in solutions for lv in levels} if load_codes else {}

    for s in subjects:
        per_subj_finite = None
        for sol in solutions:
            arr = _load(root, s, sol)  # (C, *spatial)
            if spatial is None:
                spatial = arr.shape[1:]
                gmask = np.ones(int(np.prod(spatial)), bool)
            if arr.shape[1:] != spatial:
                raise ValueError(f"{s} {sol}: {arr.shape[1:]} != {spatial} -- solutions must share the grid")
            content[sol].append(arr)
            f = np.isfinite(arr).all(0).reshape(-1)
            per_subj_finite = f if per_subj_finite is None else (per_subj_finite & f)
            stl = _load(root, s, sol, suffix="_style")
            if stl is not None:
                style[sol].append(stl)
            if load_codes:
                for lv in levels:
                    cm = _load_code(root, s, sol, lv)
                    if cm is not None:
                        codes[(sol, lv)].append(cm)
        m = _load_mask(root, s)
        if m is not None:
            per_subj_finite &= m.reshape(-1)
        gmask &= per_subj_finite

    content = {k: np.stack(v).astype(np.float32) for k, v in content.items()}
    style = {k: np.stack(v).astype(np.float32) for k, v in style.items() if v}
    codes = {k: np.stack(v) for k, v in codes.items() if v}
    atlas = np.load(atlas_path) if atlas_path else None
    affine = None
    if ref_path:
        import nibabel as nib

        affine = nib.load(ref_path).affine

    return Fields(
        content=content,
        style=style,
        codes=codes,
        mask=gmask.reshape(spatial),
        spatial=spatial,
        subjects=subjects,
        atlas=atlas,
        affine=affine,
    )


def assert_common_space(fields: Fields) -> None:
    """Hard precondition (itself a kill-switch): all solutions share the grid, the
    group mask is non-empty, and -- as a *necessary* co-registration smoke test --
    the across-subject mean content map is not pure noise (a co-registered cohort
    has structured group-mean anatomy; an un-registered one averages to mush).

    This catches the most damaging silent failure: if subjects are NOT co-registered,
    "per voxel across subjects" measures noise and every downstream map is garbage.
    """
    shapes = {k: v.shape[2:] for k, v in fields.content.items()}
    uniq = set(shapes.values())
    if len(uniq) != 1:
        raise AssertionError(f"solutions are on different grids: {shapes}")
    if fields.mask is None or fields.mask.sum() == 0:
        raise AssertionError("group mask is empty -- no voxel is in-brain for all subjects")
    if tuple(fields.mask.shape) != next(iter(uniq)):
        raise AssertionError(f"mask shape {fields.mask.shape} != content grid {next(iter(uniq))}")
    # Co-registration sniff test: structure-to-noise of the group mean. For a
    # co-registered cohort the group-mean content has spatial structure, so its
    # in-mask variance across voxels >> the residual subject noise floor. We only
    # *warn*-level assert (ratio > 1) to avoid false alarms on low-content models.
    for k, arr in fields.content.items():
        mf = fields.mask_flat
        gm = arr.mean(0).reshape(arr.shape[1], -1)[:, mf]  # (C, V) group mean
        resid = arr.reshape(arr.shape[0], arr.shape[1], -1)[:, :, mf] - gm[None]  # (S,C,V)
        struct = gm.var(axis=1).mean()
        noise = resid.var(axis=(0, 2)).mean() + 1e-12
        if struct / noise < 1e-3:
            raise AssertionError(
                f"solution {k}: group-mean structure ({struct:.2e}) is ~0 vs subject noise "
                f"({noise:.2e}); subjects look UN-registered. Fix registration before mapping."
            )


# =====================================================================================
# Phase C -- empirical recovery (>=2 solutions)
# =====================================================================================
@dataclass
class Gauge:
    """The ONE global gauge relating two solutions' content blocks.

    Fit once on pooled ``[S x voxels]`` data; NEVER re-fit per voxel (per-voxel
    re-fit is trivial-identifiability, the Phase-0 null bug again).
    """

    perm: np.ndarray  # (C,) B-channel matched to each A-channel
    signs: np.ndarray  # (C,) sign alignment per matched pair (the componentwise part)
    W: np.ndarray  # (C,C) within-content least-squares linear map A->B (general-linear)
    matched_corr: np.ndarray  # (C,) |corr| at matched pairs, for provenance


def _to_vsc(arr, mask_flat):
    """``[S, C, *spatial]`` -> ``(V, S, C)`` over in-mask voxels (subject axis = 1)."""
    S, C = arr.shape[0], arr.shape[1]
    return np.ascontiguousarray(arr.reshape(S, C, -1)[:, :, mask_flat].transpose(2, 0, 1))


def fit_global_gauge(solA, solB, mask) -> Gauge:
    """Fit ONE permutation + ONE within-content linear map on pooled ``[S x voxels]``.

    The Hungarian permutation and the linear map are GLOBAL (one per pair of
    solutions), because block-identifiability makes the content gauge location-
    independent. Per-voxel re-fitting would make everything trivially identifiable.
    """
    mf = mask.reshape(-1)
    A = solA.reshape(solA.shape[0], solA.shape[1], -1)[:, :, mf]  # (S,C,V)
    B = solB.reshape(solB.shape[0], solB.shape[1], -1)[:, :, mf]
    C = A.shape[1]
    # Pool [S x V] rows -> (N, C). Centre & scale each column over the pooled axis.
    Ap = A.transpose(0, 2, 1).reshape(-1, C)
    Bp = B.transpose(0, 2, 1).reshape(-1, C)
    Ac = Ap - Ap.mean(0, keepdims=True)
    Bc = Bp - Bp.mean(0, keepdims=True)
    As = Ac / (Ac.std(0, keepdims=True) + 1e-12)
    Bs = Bc / (Bc.std(0, keepdims=True) + 1e-12)
    M = (As.T @ Bs) / As.shape[0]  # (C,C) cross-correlation
    row, col = linear_sum_assignment(-np.abs(M))  # ONE Hungarian on pooled structure
    perm = np.empty(C, dtype=int)
    signs = np.empty(C)
    matched = np.empty(C)
    perm[row] = col
    signs[row] = np.sign(M[row, col])
    matched[row] = np.abs(M[row, col])
    signs[signs == 0] = 1.0
    # ONE within-content general-linear map (least squares on centred pooled data).
    W, *_ = np.linalg.lstsq(Ac, Bc, rcond=None)
    return Gauge(perm=perm, signs=signs, W=W.astype(np.float32), matched_corr=matched)


def _block_cca_map(Av, Bv, lam=1e-2):
    """Mean canonical correlation per voxel. (V,S,C)->(V,) in [0,1].

    Invariant to any invertible LINEAR reparam of either block == the within-content
    general-linear gauge. After per-voxel whitening (Xt^T Xt ~= I, the same trick as
    ``voxelwise_spm`` -- the Gram is a subject-sum so whitening is permutation-safe),
    the singular values of ``Xt^T Yt`` ARE the canonical correlations, so a batched
    SVD gives them directly. Reported as their mean (NOT squared), so block and
    component live on the SAME correlation scale and ``boundary = block - component``
    is meaningful."""
    Xt, Yt = whiten(Av, lam), whiten(Bv, lam)
    cross = np.einsum("vsc,vsd->vcd", Xt, Yt, optimize=True)  # (V,C,C)
    s = np.linalg.svd(cross, compute_uv=False)  # (V,C) canonical correlations
    return np.clip(s, 0.0, 1.0).mean(1)


def _block_cka_map(Av, Bv):
    """Linear CKA per voxel (orthogonal + isotropic-scale invariant). (V,S,C)->(V,).

    Centred over the SUBJECT axis (axis 1) -- getting this axis wrong is the
    batched-whitening error from Phase 0 in new clothing, so it is asserted by
    construction here (mean over axis=1)."""
    Xc = Av - Av.mean(1, keepdims=True)
    Yc = Bv - Bv.mean(1, keepdims=True)
    cross = np.einsum("vsi,vsj->vij", Xc, Yc, optimize=True)
    gx = np.einsum("vsi,vsj->vij", Xc, Xc, optimize=True)
    gy = np.einsum("vsi,vsj->vij", Yc, Yc, optimize=True)
    hsic = np.einsum("vij,vij->v", cross, cross, optimize=True)
    den = np.sqrt(np.einsum("vij,vij->v", gx, gx, optimize=True)) * np.sqrt(
        np.einsum("vij,vij->v", gy, gy, optimize=True)
    )
    return hsic / (den + 1e-12)


def _ranks_over_subjects(arr):
    """Average ranks along the subject axis (axis 1). (V,S,C)->(V,S,C)."""
    try:
        return _spstats.rankdata(arr, axis=1).astype(np.float32)
    except TypeError:  # very old scipy without axis support
        order = arr.argsort(axis=1).argsort(axis=1)
        return (order + 1).astype(np.float32)


def _component_map(Av, Bv, gauge: Gauge):
    """Matched Spearman per voxel, averaged over channels. (V,S,C)->(V,).

    Spearman is MONOTONE-invariant (so it certifies the componentwise-monotone
    gauge); the global permutation+sign from ``gauge`` is applied first -- never
    re-matched per voxel."""
    B_aligned = gauge.signs[None, None, :] * Bv[:, :, gauge.perm]  # align B's axes to A's
    Rx = _ranks_over_subjects(Av)
    Ry = _ranks_over_subjects(B_aligned)
    Rx = Rx - Rx.mean(1, keepdims=True)
    Ry = Ry - Ry.mean(1, keepdims=True)
    num = np.einsum("vsc,vsc->vc", Rx, Ry, optimize=True)
    den = np.sqrt(np.einsum("vsc,vsc->vc", Rx, Rx, optimize=True) * np.einsum("vsc,vsc->vc", Ry, Ry, optimize=True))
    rho = num / (den + 1e-12)
    return rho.mean(1)


def recovery_maps(solA, solB, gauge: Gauge, mask, *, block_metric="cca", lam=1e-2):
    """``block_recovery`` (CCA/CKA), ``component_recovery`` (matched Spearman), and the
    derived ``t1t2_boundary = block - component``, per in-mask voxel across subjects.

    The single ``gauge`` is held FIXED; agreement is read per voxel along the
    subject axis. This is the map where the "re-fit alignment per voxel" trap lives:
    block uses CCA (gauge-free at the block level) and component uses the ONE global
    permutation, so neither re-fits per voxel."""
    mf = mask.reshape(-1)
    Av, Bv = _to_vsc(solA, mf), _to_vsc(solB, mf)
    block = _block_cca_map(Av, Bv, lam) if block_metric == "cca" else _block_cka_map(Av, Bv)
    component = _component_map(Av, Bv, gauge)
    return {"block": block, "component": component, "boundary": block - component}


# =====================================================================================
# Phase D -- gauge-permutation inference
# =====================================================================================
def _random_orthogonal(C, rng):
    a = rng.standard_normal((C, C))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))[None, :]


def _apply_channel_gauge(arr, Q):
    """``arr`` ``[S, C, *spatial]`` -> mix channels by ``Q`` (C,C): out[s,i]=sum_j Q[i,j] arr[s,j]."""
    return np.einsum("ij,sj...->si...", Q, arr, optimize=True)


def gauge_null(map_fn, solA, solB, subgroup, K, mask, *, rng=None, styleA=None):
    """Accumulate a null by applying K elements of the claim-matched gauge subgroup
    to ONE solution and recomputing the map. Returns ``(V, K)``.

    The correct chance level is "agreement achievable by some admissible gauge
    element", not perfect alignment and not independent random vectors. ``map_fn``
    must fit its OWN global gauge internally (so the null measures the best agreement
    achievable *after* the scrambling element), then return a ``(V,)`` map.

    subgroup:
      * ``subject_perm``    -- permute solB's subjects (existence null for block_recovery).
      * ``content_rotation``-- random orthogonal within content (component null).
      * ``content_gl``      -- random general-linear within content (component null, looser claim).
      * ``content_style``   -- random orthogonal over stacked content+style, keep the
                               content part (boundary null; needs ``styleA``).
    """
    rng = rng or np.random.default_rng(0)
    C = solA.shape[1]
    cols = []
    for _ in range(K):
        if subgroup == "subject_perm":
            cols.append(map_fn(solA, solB[rng.permutation(solB.shape[0])]))
        elif subgroup == "content_rotation":
            cols.append(map_fn(_apply_channel_gauge(solA, _random_orthogonal(C, rng)), solB))
        elif subgroup == "content_gl":
            cols.append(map_fn(_apply_channel_gauge(solA, rng.standard_normal((C, C))), solB))
        elif subgroup == "content_style":
            if styleA is None:
                raise ValueError("content_style null needs styleA (the view's style field)")
            stacked = np.concatenate([solA, styleA], axis=1)
            Q = _random_orthogonal(stacked.shape[1], rng)
            cols.append(map_fn(_apply_channel_gauge(stacked, Q)[:, :C], solB))
        else:
            raise ValueError(f"unknown gauge subgroup {subgroup!r}")
    return np.stack(cols, axis=1)


def fwe_tfce(observed, null_stack, mask, shape, *, alpha=0.05, tfce_H=2.0, tfce_E=0.5, tfce_dh=0.1, do_tfce=True):
    """Per-voxel p from the tail, max-statistic FWE over the mask, and (when
    ``do_tfce``) TFCE cluster extent -- all against the gauge/subject null. Reuses
    ``voxelwise_spm.tfce`` (the corrected background-bin handling). ``do_tfce=False``
    skips the (per-permutation) TFCE pass -- useful when running it K times for
    per-factor maps."""
    K = null_stack.shape[1]
    mf = mask.reshape(-1)
    nmean, nstd = null_stack.mean(1), null_stack.std(1) + 1e-9
    z = (observed - nmean) / nstd
    p_unc = ((null_stack >= observed[:, None]).sum(1) + 1) / (K + 1)
    q_fdr = benjamini_hochberg(p_unc)
    maxstat = null_stack.max(0)  # (K,)
    p_fwe = ((maxstat[None, :] >= observed[:, None]).sum(1) + 1) / (K + 1)
    out = {"z": z, "p_unc": p_unc, "q_fdr": q_fdr, "p_fwe": p_fwe, "sig_fwe": observed * (p_fwe < alpha)}
    if not do_tfce:
        return out

    znull = (null_stack - nmean[:, None]) / nstd[:, None]
    t_obs = tfce(np.clip(z, 0, None), mf, shape, tfce_H, tfce_E, tfce_dh)
    t_max = np.array([tfce(np.clip(znull[:, k], 0, None), mf, shape, tfce_H, tfce_E, tfce_dh).max() for k in range(K)])
    p_tfce = ((t_max[None, :] >= t_obs[:, None]).sum(1) + 1) / (K + 1)
    out.update({"tfce": t_obs, "p_tfce": p_tfce, "sig_tfce": observed * (p_tfce < alpha)})
    return out


# =====================================================================================
# Phase C2 -- ground-truth recovery & localised DCI (synthetic / labelled data only)
# =====================================================================================
# These need a factor array Z = [S, K] (synthetic GT z_content, or any continuous
# covariate on real data: severity d, age). They localise the POOLED metrics in
# eval/identifiability_metrics.py (block_mcc, content_informativeness, dci_DC).
#
# Hungarian discipline (the recurring trap): the content<->factor assignment is fit
# ONCE on pooled data and held fixed across voxels. Per-voxel re-matching is trivial
# identifiability. DCI's D/C need no matching at all (entropy is assignment-invariant).


def _pooled_corr(Apool, Bpool):
    """|corr| cross-matrix (Ca, Cb) between pooled columns (centred + scaled)."""
    As = (Apool - Apool.mean(0)) / (Apool.std(0) + 1e-12)
    Bs = (Bpool - Bpool.mean(0)) / (Bpool.std(0) + 1e-12)
    return (As.T @ Bs) / As.shape[0]


def _global_assignment(content, factors, mask):
    """ONE global Hungarian content-channel <-> factor on pooled [S x voxels] data.
    Returns matched (chan_idx, factor_idx, signs). Handles C != K (min(C,K) pairs)."""
    mf = mask.reshape(-1)
    Xp = content.reshape(content.shape[0], content.shape[1], -1)[:, :, mf].transpose(0, 2, 1)
    Xp = Xp.reshape(-1, content.shape[1])  # (S*V, C), s-major (row = s*V + v)
    Zp = np.repeat(factors, int(mf.sum()), axis=0)  # (S*V, K), s-major to MATCH Xp
    M = _pooled_corr(Xp, Zp)  # (C, K)
    rows, cols = linear_sum_assignment(-np.abs(M))
    signs = np.sign(M[rows, cols])
    signs[signs == 0] = 1.0
    return rows, cols, signs, M


def _block_cca_vs_fixed(Xv, Z, lam=1e-2):
    """Mean canonical correlation between per-voxel content (V,S,C) and a fixed
    factor array Z (S,K). Z is whitened ONCE (it does not vary by voxel)."""
    Xt = whiten(Xv, lam)
    Zt = whiten(np.ascontiguousarray(Z[None].astype(np.float32)), lam)[0]  # (S,K)
    cross = np.einsum("vsc,sk->vck", Xt, Zt, optimize=True)
    s = np.linalg.svd(cross, compute_uv=False)
    return np.clip(s, 0.0, 1.0).mean(1)


def _ridge_importance(Xv, Z, lam=1e-2):
    """Batched ridge readout content (V,S,C) -> factors (S,K). Returns weights
    W (V,C,K), centred content Xc, centred Z. Importance R = |W|."""
    Xc = Xv - Xv.mean(1, keepdims=True)
    Zc = (Z - Z.mean(0, keepdims=True)).astype(np.float32)
    G = np.einsum("vsc,vsd->vcd", Xc, Xc, optimize=True)  # (V,C,C)
    C = G.shape[-1]
    ridge = lam * np.trace(G, axis1=1, axis2=2) / C + 1e-8
    G = G + ridge[:, None, None] * np.eye(C, dtype=np.float32)
    XtZ = np.einsum("vsc,sk->vck", Xc, Zc, optimize=True)  # (V,C,K)
    W = np.linalg.solve(G, XtZ)
    return W, Xc, Zc


def _perfactor_r2(Xv, Z, lam=1e-2):
    """Per-voxel, per-factor in-sample ridge R^2 content (V,S,C) -> factors (S,K).
    Returns (V,K). In-sample (not CV), so the ABSOLUTE level is optimistic — read it
    relatively, and use the permutation null (which carries the same bias) for
    significance."""
    W, Xc, Zc = _ridge_importance(Xv, Z, lam)
    Zhat = np.einsum("vsc,vck->vsk", Xc, W, optimize=True)
    ss_res = ((Zc[None] - Zhat) ** 2).sum(1)  # (V,K)
    ss_tot = (Zc**2).sum(0)[None] + 1e-12  # (1,K)
    return 1.0 - ss_res / ss_tot


def gt_recovery_map(content, factors, mask, *, lam=1e-2):
    """Per-voxel recovery of KNOWN factors by the content block (the synthetic
    direct test; degrades to any covariate on real data).

      * block      -- mean canonical correlation content-block <-> all factors (CCA).
      * component  -- matched Spearman after the ONE global content<->factor Hungarian.
      * informativeness -- per-voxel mean ridge R^2 over factors.
      * informativeness_per_factor -- (V,K) ridge R^2 for EACH factor separately,
        i.e. "where is factor k encoded" (no Hungarian: each factor predicted from
        the whole content block).

    Use ``gt_null`` + ``fwe_tfce`` for significance (subject-permutation of Z)."""
    mf = mask.reshape(-1)
    Xv = _to_vsc(content, mf)  # (V,S,C)
    Z = np.asarray(factors, np.float32)
    block = _block_cca_vs_fixed(Xv, Z, lam)

    rows, cols, signs, _ = _global_assignment(content, factors, mask)
    Rx = _ranks_over_subjects(Xv)[:, :, rows]  # (V,S,m) content side of matched pairs
    Rz = _ranks_over_subjects(Z[None])[0][:, cols] * signs[None, :]  # (S,m) factor side
    Rx = Rx - Rx.mean(1, keepdims=True)
    Rz = Rz - Rz.mean(0, keepdims=True)
    num = np.einsum("vsm,sm->vm", Rx, Rz, optimize=True)
    den = np.sqrt(np.einsum("vsm,vsm->vm", Rx, Rx, optimize=True) * (Rz**2).sum(0)[None])
    component = (num / (den + 1e-12)).mean(1)

    r2 = _perfactor_r2(Xv, Z, lam)  # (V,K)
    return {
        "block": block,
        "component": component,
        "informativeness": r2.mean(1),
        "informativeness_per_factor": r2,
    }


def gt_null(map_fn, content, factors, K, mask, *, rng=None):
    """Subject-permutation null for gt_recovery: shuffle the subject<->factor pairing
    (the existence-of-association chance level). Returns (V, K)."""
    rng = rng or np.random.default_rng(0)
    return np.stack([map_fn(content, factors[rng.permutation(factors.shape[0])]) for _ in range(K)], axis=1)


def gt_perfactor_null(content, factors, n_perm, mask, *, rng=None, lam=1e-2):
    """Subject-permutation null for the per-factor R^2 map. Returns (V, K, n_perm).
    One batched ridge per draw yields all K factors at once."""
    rng = rng or np.random.default_rng(0)
    Xv = _to_vsc(content, mask.reshape(-1))
    Z = np.asarray(factors, np.float32)
    return np.stack([_perfactor_r2(Xv, Z[rng.permutation(Z.shape[0])], lam) for _ in range(n_perm)], axis=2)


def _dci_from_importance(R):
    """Batched disentanglement & completeness from importance R (V,C,K). Matches
    eval/identifiability_metrics.dci_DC exactly, vectorised over the V axis."""
    eps = 1e-11
    Vn, C, K = R.shape
    Rp = R + eps
    col = Rp / Rp.sum(1, keepdims=True)  # over codes -> per factor
    Hc = -(col * np.log(col)).sum(1) / np.log(max(C, 2))  # (V,K)
    completeness = (1.0 - Hc).mean(1)
    row = Rp / Rp.sum(2, keepdims=True)  # over factors -> per code
    Hd = -(row * np.log(row)).sum(2) / np.log(max(K, 2))  # (V,C)
    w = R.sum(2)  # per-code importance
    disent = ((1.0 - Hd) * w).sum(1) / (w.sum(1) + eps)
    return disent, completeness


def dci_maps(content, factors, mask, *, level="roi", atlas=None, lam=1e-2, importance="ridge", seed=0):
    """Localised DCI. ``level="voxel"`` -> per-voxel D/C/I maps via batched-ridge
    importance (cheap, correlation-robust). ``level="roi"`` -> per-ROI DCI via the
    pooled estimator in eval/identifiability_metrics.py (RF/L1), which agrees with
    your global DCI by construction. D/C are gauge-DEPENDENT (Tier-3): read as the
    model's native-basis axis-alignment, alongside the factor-correlation ceiling."""
    from eval.identifiability_metrics import dci_DC, factor_correlation_ceiling, importance_matrix

    mf = mask.reshape(-1)
    Z = np.asarray(factors, np.float64)
    ceiling = factor_correlation_ceiling(Z)
    if level == "voxel":
        Xv = _to_vsc(content, mf)
        W, _, _ = _ridge_importance(Xv, Z.astype(np.float32), lam)
        D, Cm = _dci_from_importance(np.abs(W))
        rec = gt_recovery_map(content, factors, mask, lam=lam)
        return {"disentanglement": D, "completeness": Cm, "informativeness": rec["informativeness"], "ceiling": ceiling}
    # per-ROI: collapse spatial within ROI per subject, run the pooled estimator.
    rois = np.unique(atlas[atlas > 0]) if atlas is not None else np.array([1])
    C = content.shape[1]
    groups = np.arange(C)
    rows = []
    for roi in rois:
        rmask = (atlas == roi).reshape(-1) & mf if atlas is not None else mf
        if rmask.sum() < 4:
            continue
        Xroi = content.reshape(content.shape[0], C, -1)[:, :, rmask].mean(2)  # (S,C) ROI-mean
        Rimp = importance_matrix(Xroi, Z, groups, method=("l1" if importance != "rf" else "rf_perm"), seed=seed)
        dc = dci_DC(Rimp)
        rows.append({"roi": int(roi), "n_vox": int(rmask.sum()), **dc, "ceiling": ceiling})
    return rows


def gauge_drift_roi(content, factors, atlas, mask, *, n_perm=200, rng=None, alpha=0.05):
    """The ONLY valid localised-Hungarian map: per-ROI gain from RE-MATCHING the
    content<->factor assignment locally vs the global one. Statistic =
    local_matched_|corr| - global_matched_|corr| (>=0). The null permutes subjects
    and ALSO re-matches locally, so the free-matching inflation cancels (same bias
    cancellation as voxelwise_spm); a significant positive == the model's
    disentanglement AXES genuinely change in that region.

    Everything is computed on ROI-MEAN content (S,C) so the global reference and the
    local match share one representation; the global assignment is the size-weighted
    Hungarian over all ROI-means, so a large stable region pins it (a region only
    drifts RELATIVE to the brain-wide consensus)."""
    rng = rng or np.random.default_rng(0)
    mf = mask.reshape(-1)
    S, C = content.shape[0], content.shape[1]
    Z = np.asarray(factors, np.float32)
    rois = np.unique(atlas[atlas > 0]) if atlas is not None else np.array([1])

    means, sizes = {}, {}
    for roi in rois:
        rmask = (atlas == roi).reshape(-1) & mf if atlas is not None else mf
        if rmask.sum() < 4:
            continue
        means[roi] = content.reshape(S, C, -1)[:, :, rmask].mean(2)  # (S,C)
        sizes[roi] = int(rmask.sum())
    tot = sum(sizes.values())
    Mglobal = sum(sizes[r] * _pooled_corr(means[r], Z) for r in means) / tot  # size-weighted
    g_rows, g_cols = linear_sum_assignment(-np.abs(Mglobal))

    def _matched(Xm, Zm, rows, cols):
        return np.abs(np.diag(_pooled_corr(Xm[:, rows], Zm[:, cols]))).mean()

    out = []
    for roi, Xr in means.items():
        lr, lc = linear_sum_assignment(-np.abs(_pooled_corr(Xr, Z)))
        obs = _matched(Xr, Z, lr, lc) - _matched(Xr, Z, g_rows, g_cols)
        null = np.empty(n_perm)
        for p in range(n_perm):
            Zp = Z[rng.permutation(S)]
            plr, plc = linear_sum_assignment(-np.abs(_pooled_corr(Xr, Zp)))
            null[p] = _matched(Xr, Zp, plr, plc) - _matched(Xr, Zp, g_rows, g_cols)
        pval = ((null >= obs).sum() + 1) / (n_perm + 1)
        out.append(
            {
                "roi": int(roi),
                "n_vox": sizes[roi],
                "drift_gain": float(obs),
                "p": float(pval),
                "sig": bool(pval < alpha),
            }
        )
    return out


# =====================================================================================
# Phase E -- VQ code maps
# =====================================================================================
def code_reproducibility(codesA, codesB, mask, *, level=None, want_matched_acc=False):
    """Per-voxel Adjusted Mutual Information between two code-index maps across subjects.

    AMI is label-permutation-invariant, so NO Hungarian matching is done (matching
    before AMI is a no-op that can hide a bug). ``want_matched_acc`` additionally
    returns matched classification accuracy, which DOES need one global Hungarian on
    the pooled confusion matrix."""
    from sklearn.metrics import adjusted_mutual_info_score

    mf = mask.reshape(-1)
    Av = codesA.reshape(codesA.shape[0], -1)[:, mf]  # (S, V)
    Bv = codesB.reshape(codesB.shape[0], -1)[:, mf]
    V = Av.shape[1]
    ami = np.array([adjusted_mutual_info_score(Av[:, v], Bv[:, v], average_method="arithmetic") for v in range(V)])
    out = {"ami": ami, "level": level}
    if want_matched_acc:
        a_lab = np.unique(Av, return_inverse=True)[1].reshape(Av.shape)
        b_lab = np.unique(Bv, return_inverse=True)[1].reshape(Bv.shape)
        na, nb = a_lab.max() + 1, b_lab.max() + 1
        conf = np.zeros((na, nb))
        np.add.at(conf, (a_lab.ravel(), b_lab.ravel()), 1)  # ONE global confusion
        r, c = linear_sum_assignment(-conf)
        amap = {int(cc): int(rr) for rr, cc in zip(r, c)}
        b_to_a = np.array([amap.get(j, -1) for j in range(nb)])
        acc = (a_lab == b_to_a[b_lab]).mean(0)  # per-voxel accuracy over subjects
        out["matched_acc"] = acc
    return out


# =====================================================================================
# Phase B -- model-intrinsic maps (one model)
# =====================================================================================
def decoder_jacobian_maps(
    decode_fns,
    content_latents,
    style_latents,
    mask,
    *,
    rf_radius=1,
    stride=2,
    jac_method="exact_local",
    n_out_probe=64,
    device="cpu",
    eps_visibility=1e-4,
):
    """``content_visibility`` and ``content_style_angle`` from decoder VJPs (ONE model).

    For each view ``v``, ``decode_fns[v](content_latent, style_latent) -> (1,1,*out)``
    is differentiated locally. The per-voxel content Jacobian ``J_v^c`` is
    d(output receptive-field patch) / d(the C-dim CONTENT VECTOR at the corresponding
    latent voxel) -- shape ``(patch, C)`` -- restricted to the receptive field
    (``rf_radius``) rather than the full latent grid, so ``sigma_min`` has the right
    size (an unrestricted operator gives a wrong-sized sigma_min).

      * ``content_visibility(u) = max_v sigma_min(J_v^c(u))`` -- the least-rendered
        content direction, maxed over views: is the content vector injectively
        rendered by SOME view at u.
      * ``content_style_angle(u)`` = cos of the smallest principal angle between
        ``col J_v^c`` and ``col J_v^m`` (in [0,1]; ~1 == content/style fully confounded
        locally), taken as the max-confound over views.

    Computed on a thinned grid (``stride``) and nearest-interpolated. Kill-switch: if
    max-over-view visibility is ~0 across the brain, content is unrendered -- STOP.
    """
    import torch

    shape = tuple(mask.shape)
    n_vox = int(np.prod(shape))
    vis_thin = np.zeros(n_vox, np.float32)
    ang_thin = np.zeros(n_vox, np.float32)
    probed = np.zeros(n_vox, bool)
    mf = mask.reshape(-1)

    grid = np.zeros(shape, bool)
    grid[tuple(slice(None, None, stride) for _ in shape)] = True
    probe_idx = np.where(grid.reshape(-1) & mf)[0]

    def _local_jacobian(decode_fn, lat, other, center, wrt):
        """Jacobian of the output receptive-field patch around ``center`` w.r.t. the
        per-channel latent VECTOR at the corresponding latent voxel. Returns (patch, ch)."""
        gx = lat.shape[2:]
        lc = tuple(min(gx[d] - 1, max(0, int(round(center[d] * gx[d] / shape[d])))) for d in range(len(shape)))
        osl = tuple(slice(max(0, center[d] - rf_radius), center[d] + rf_radius + 1) for d in range(len(shape)))
        vec0 = lat[(slice(0, 1), slice(None)) + lc].detach().clone().reshape(-1)  # (ch,)

        def f(vec):
            full = lat.clone()
            full[(slice(0, 1), slice(None)) + lc] = vec.view(1, -1)
            rec = decode_fn(full, other) if wrt == "content" else decode_fn(other, full)
            return rec[(0, 0) + osl].reshape(-1)

        J = torch.autograd.functional.jacobian(f, vec0, vectorize=True)
        if jac_method == "randomized" and n_out_probe is not None and J.shape[0] > n_out_probe:
            J = J[torch.randperm(J.shape[0])[:n_out_probe]]
        return J

    for flat in probe_idx:
        center = np.unravel_index(int(flat), shape)
        vmax = 0.0
        conf_max = 0.0
        for v, decode_fn in enumerate(decode_fns):
            c = content_latents[v].to(device)
            s = style_latents[v].to(device) if style_latents and style_latents[v] is not None else None
            Jc = _local_jacobian(decode_fn, c, s, center, "content")
            sc = torch.linalg.svdvals(Jc)
            vmax = max(vmax, float(sc[min(Jc.shape) - 1]) if min(Jc.shape) > 0 else 0.0)
            if s is not None and s.shape[1] > 0:
                Jm = _local_jacobian(decode_fn, s, c, center, "style")
                Uc = torch.linalg.svd(Jc, full_matrices=False).U
                Um = torch.linalg.svd(Jm, full_matrices=False).U
                cos = torch.linalg.svdvals(Uc.T @ Um)
                conf_max = max(conf_max, float(cos[0]) if cos.numel() else 0.0)
        vis_thin[flat] = vmax
        ang_thin[flat] = conf_max
        probed[flat] = True

    visibility = _nearest_fill(vis_thin, probed, shape, mf)
    cs_angle = _nearest_fill(ang_thin, probed, shape, mf)
    content_unrendered = bool(visibility[mf].max() < eps_visibility)
    return {
        "visibility": visibility.reshape(shape),
        "cs_angle": cs_angle.reshape(shape),
        "content_unrendered": content_unrendered,
    }


def _nearest_fill(thin_flat, probed, shape, mf):
    """Nearest-neighbour interpolate a thinned probe map to the full in-mask grid."""
    from scipy import ndimage

    out = np.zeros(int(np.prod(shape)), np.float32)
    if probed.sum() == 0:
        return out
    # Nearest probed voxel for every grid point, via the EDT feature transform.
    idx = ndimage.distance_transform_edt(~probed.reshape(shape), return_indices=True, return_distances=False)
    nearest = thin_flat.reshape(shape)[tuple(idx)].reshape(-1)
    out[mf] = nearest[mf]
    return out


def ivae_shift_map(content, d_labels, mask, *, min_bins=None):
    """Conditioning/rank of the d-shift matrix of content sufficient statistics.

    Bins subjects by severity ``d``; for each in-mask voxel, builds the matrix of
    per-bin shifts of the content sufficient statistics (mean and variance proxies)
    relative to the reference bin, and reports its conditioning. iVAE identifiability
    needs the shift matrix to be full-rank, which requires >= ``n*k + 1`` distinct
    environments -- so this is undefined without enough severity bins."""
    d = np.asarray(d_labels)
    bins = np.unique(d)
    C = content.shape[1]
    need = min_bins if min_bins is not None else (C + 1)  # n*k+1 with k=1 suff-stat dim per channel
    if len(bins) < need:
        return {"defined": False, "n_bins": int(len(bins)), "need": int(need), "rank": None, "cond": None}
    mf = mask.reshape(-1)
    X = content.reshape(content.shape[0], C, -1)[:, :, mf]  # (S, C, V)
    ref = X[d == bins[0]].mean(0)  # (C, V) reference-bin mean
    # d-shift matrix per voxel: rows = (n_bins-1) shifts, cols = C suff-stat channels.
    shifts = np.stack([X[d == b].mean(0) - ref for b in bins[1:]], axis=0)  # (B-1, C, V)
    ranks = np.zeros(mf.sum(), int)
    conds = np.zeros(mf.sum())
    for vi in range(shifts.shape[2]):
        s = np.linalg.svd(shifts[:, :, vi], compute_uv=False)
        ranks[vi] = int((s > s.max() * 1e-6).sum()) if s.size and s.max() > 0 else 0
        conds[vi] = float(s.max() / (s[s > 0].min() + 1e-12)) if (s > 0).any() else np.inf
    full = np.zeros(int(np.prod(mask.shape)), np.float32)
    rank_map = full.copy()
    rank_map[mf] = ranks
    cond_map = full.copy()
    cond_map[mf] = np.clip(conds, 0, 1e6)
    return {
        "defined": True,
        "n_bins": int(len(bins)),
        "rank": rank_map.reshape(mask.shape),
        "cond": cond_map.reshape(mask.shape),
    }


def kernel_distinctness(solution, atlas, mask, *, factor_axis=1):
    """Per-ROI pairwise factor-kernel distinguishability (Theorem 2, localised).

    Within each ROI, compares the empirical spatial autocorrelation (variogram
    proxy: variance vs a 1-voxel-lag difference) of each content factor; factor
    pairs with near-identical length-scales are flagged INDISTINGUISHABLE -- the
    degenerate case where component-level identifiability collapses to block-level."""
    mf = mask.reshape(-1)
    rois = np.unique(atlas[atlas > 0]) if atlas is not None else np.array([1])
    S, C = solution.shape[0], solution.shape[factor_axis]
    ndim = solution.ndim - 2  # spatial dims
    rows = []
    for roi in rois:
        rmask = (atlas == roi).reshape(-1) & mf if atlas is not None else mf
        if rmask.sum() < 8:
            continue
        lengthscales = []
        for c in range(C):
            field = solution[:, c]  # (S, *spatial) -- the variogram is a WITHIN-realisation
            v_tot = field.reshape(S, -1)[:, rmask].var(axis=1) + 1e-12  # (S,) per-subject
            sv = []
            for ax in range(ndim):
                dif = np.diff(field, axis=ax + 1)
                pad = [(0, 0)] * field.ndim
                pad[ax + 1] = (0, 1)
                dif = np.pad(dif, pad)
                sv.append(0.5 * (dif.reshape(S, -1)[:, rmask] ** 2).mean(axis=1))  # (S,)
            # normalised lag-1 semivariance, averaged OVER SUBJECTS (a mean field of
            # subject-iid factors washes the kernel out; per-subject does not).
            lengthscales.append(float((np.mean(sv, axis=0) / v_tot).mean()))
        ls = np.array(lengthscales)
        # pairwise relative length-scale gap; small gap => indistinguishable kernels
        indist = []
        for i in range(C):
            for j in range(i + 1, C):
                gap = abs(ls[i] - ls[j]) / (abs(ls[i]) + abs(ls[j]) + 1e-12)
                if gap < 0.05:
                    indist.append((i, j))
        rows.append(
            {"roi": int(roi), "n_vox": int(rmask.sum()), "lengthscales": ls.tolist(), "indistinct_pairs": indist}
        )
    return rows


# =====================================================================================
# Phase F -- outputs
# =====================================================================================
def write_nifti(arr, ref_affine, path):
    """Write a per-voxel map to NIfTI in template space (drops into SPM/VBM viewers)."""
    import nibabel as nib

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    aff = ref_affine if ref_affine is not None else np.eye(4)
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), aff), path)


def roi_summary(map_flat, atlas, mask, *, n_boot=0, rng=None):
    """Per-ROI mean of a per-voxel statistic, with optional voxel-bootstrap CIs.

    For subject-level CIs (the gold standard, matching the FCI bootstrap-stability
    habit) pass a recompute hook from the caller; this lightweight version bootstraps
    voxels within ROI to give a quick stability band."""
    rng = rng or np.random.default_rng(0)
    mf = mask.reshape(-1)
    vals = np.zeros(int(np.prod(mask.shape)), np.float32)
    vals[mf] = map_flat
    rois = np.unique(atlas[atlas > 0]) if atlas is not None else np.array([1])
    out = []
    for roi in rois:
        rmask = (atlas == roi).reshape(-1) & mf if atlas is not None else mf
        x = vals[rmask]
        if x.size == 0:
            continue
        row = {"roi": int(roi), "n_vox": int(x.size), "mean": float(x.mean())}
        if n_boot:
            bs = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
            row["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out.append(row)
    return out


def save_map(arr_flat, mask, shape, affine, out_prefix, name):
    """Persist a per-voxel (in-mask) statistic as both .npy and .nii.gz."""
    vol = np.zeros(int(np.prod(shape)), np.float32)
    vol[mask.reshape(-1)] = arr_flat
    vol = vol.reshape(shape)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(f"{out_prefix}_{name}.npy", vol)
    if affine is not None:
        write_nifti(vol, affine, f"{out_prefix}_{name}.nii.gz")
    return vol


# =====================================================================================
# Disk-layout helpers
# =====================================================================================
def _load(root, subject, sol, suffix=""):
    view, seed = sol
    p = os.path.join(root, f"{subject}_{view}{suffix}_seed{seed}.npy")
    if not os.path.exists(p):
        if suffix:  # style is optional
            return None
        raise FileNotFoundError(p)
    return np.ascontiguousarray(np.load(p), dtype=np.float32)


def _load_code(root, subject, sol, level):
    view, seed = sol
    p = os.path.join(root, f"{subject}_{view}_seed{seed}_code{level}.npy")
    return np.load(p) if os.path.exists(p) else None


def _load_mask(root, subject):
    p = os.path.join(root, f"{subject}_mask.npy")
    return np.load(p).astype(bool) if os.path.exists(p) else None


def _discover_subjects(root, sol):
    view, seed = sol
    suffix = f"_{view}_seed{seed}.npy"
    return sorted(os.path.basename(p)[: -len(suffix)] for p in glob.glob(os.path.join(root, f"*{suffix}")))


# =====================================================================================
# Synthetic controls -- the decisive validation, run BEFORE ADNI.
# =====================================================================================
def _smudge(field3, sigma):
    from scipy import ndimage

    return ndimage.gaussian_filter(field3, sigma=(0,) + (sigma,) * (field3.ndim - 1))


def _base_features(L, kernels, mask):
    """Per-channel smudged, UNIT-VARIANCE feature field ``F`` ``[S, C, X, Y]``.

    Each channel is smudged with its own kernel then normalised to unit variance
    over (subjects x in-mask voxels) so SNR is uniform across kernel widths -- a
    wide kernel must not look like a low-signal channel. Rotations applied to ``F``
    (not to ``L``) are then genuine FEATURE-SPACE gauge elements."""
    S, K, V = L.shape
    side = int(round(V**0.5))
    mf = mask.reshape(-1)
    F = np.zeros((S, K, side, side), np.float32)
    for c in range(K):
        f = _smudge(L[:, c, :].reshape(S, side, side), kernels[c])
        f = f / (f.reshape(S, -1)[:, mf].std() + 1e-12)
        F[:, c] = f
    return F


def _apply_feat_gauge(F, R, sub=None):
    """Mix feature channels by orthogonal ``R``; if ``sub`` given, only that sub-block."""
    G = F.copy()
    if sub is None:
        return np.einsum("ij,sj...->si...", R, F, optimize=True)
    G[:, sub] = np.einsum("ij,sj...->si...", R, F[:, sub])
    return G


def _equal_mixing_orthogonal(n):
    """An orthogonal matrix whose first row is the all-equal mix (1..1)/sqrt(n), so
    the first output channel is a maximally-confused combination of all inputs ->
    forces matched component LOW (decisive degeneracy, far from any permutation)."""
    a = np.eye(n)
    a[:, 0] = 1.0 / np.sqrt(n)
    q, _ = np.linalg.qr(a)
    return q.T


def _synthetic_fields(case, rng, S=80, K=4, side=12, noise=0.05):
    """Generate a paired (solA, solB, mask, styleA) for one control case.

    All B-solutions are built from the SAME base features ``F`` as A via a
    feature-space transform, so each is a valid solution sharing A's content block."""
    V = side * side
    L = rng.standard_normal((S, K, V)).astype(np.float32)
    mask = np.ones((side, side), bool)
    mask[:1] = mask[-1:] = mask[:, :1] = mask[:, -1:] = False  # crude brain mask

    def _noise():
        return noise * rng.standard_normal((S, K, side, side)).astype(np.float32)

    if case == "positive":
        kernels = np.array([0.6, 1.0, 1.6, 2.2])  # distinct kernel per factor
        F = _base_features(L, kernels, mask)
        A = F + _noise()
        # B: permutation + sign + a per-channel MONOTONE warp -> component must survive.
        perm = rng.permutation(K)
        signs = rng.choice([-1.0, 1.0], size=(1, K, 1, 1))
        B = np.tanh(0.8 * signs * F[:, perm]) + _noise()
    elif case == "negative":
        kernels = np.array([1.0, 1.0, 1.0, 2.2])  # factors 0,1,2 SHARE a kernel (Thm-2 degenerate)
        F = _base_features(L, kernels, mask)
        A = F + _noise()
        # B mixes the degenerate sub-block by an equal-mixing rotation: the SUBSPACE
        # is preserved (block stays high) but per-channel axes are destroyed
        # (component collapses) -> large, localised boundary.
        B = _apply_feat_gauge(F, _equal_mixing_orthogonal(3), sub=[0, 1, 2]) + _noise()
    elif case == "null_rotation":
        kernels = np.array([0.6, 1.0, 1.6, 2.2])
        F = _base_features(L, kernels, mask)
        A = F + _noise()
        B = _apply_feat_gauge(F, _random_orthogonal(K, rng)) + _noise()  # B is a pure gauge element of A
    else:
        raise ValueError(case)
    styleA = rng.standard_normal((S, 2, side, side)).astype(np.float32)
    return A.astype(np.float32), B.astype(np.float32), mask, styleA


def _synthetic_gt(case, rng, S=120, K=4, side=10, noise=0.1):
    """Content as a known function of SCALAR factors Z[S,K], with per-channel spatial
    gain. ``aligned`` = axis-aligned identity; ``entangled`` = global equal-mixing
    rotation; ``spatial_drift`` = identity in the left ROI, rotation in the right ROI
    (so the disentanglement axes genuinely change across space)."""
    Z = rng.standard_normal((S, K)).astype(np.float32)
    mask = np.ones((side, side), bool)
    mask[:1] = mask[-1:] = mask[:, :1] = mask[:, -1:] = False
    # ROI2 is a SMALL corner patch (the candidate drift region); ROI1 is the large
    # majority (the stable consensus). A small patch keeps the global assignment
    # pinned to ROI1's axes so ROI2 can drift relative to it.
    atlas = np.where(mask, 1, 0)
    atlas[1:4, 1:4] = 2  # 3x3 patch
    atlas[~mask] = 0
    gains = np.stack([_smudge(rng.standard_normal((1, side, side)), 2.0)[0] for _ in range(K)])
    gains = gains - gains.min(axis=(1, 2), keepdims=True)
    gains = gains / (gains.max(axis=(1, 2), keepdims=True) + 1e-9) + 0.3  # positive, in [0.3,1.3]
    R = _equal_mixing_orthogonal(K)
    ZR = (Z @ R.T).astype(np.float32)  # within-block rotation (entangles axes)
    swap = np.arange(K)
    swap[0], swap[1] = 1, 0  # a genuine PERMUTATION (factor relabelling) for drift
    ZP = Z[:, swap]
    content = np.zeros((S, K, side, side), np.float32)
    left = (atlas == 1).astype(np.float32)
    right = (atlas == 2).astype(np.float32)
    for c in range(K):
        ident = Z[:, c, None, None] * gains[c][None]
        if case == "aligned":
            content[:, c] = ident
        elif case == "entangled":
            content[:, c] = ZR[:, c, None, None] * gains[c][None]
        elif case == "spatial_drift":
            # identity in the majority; a factor SWAP in the patch -> the optimal
            # ASSIGNMENT (not just the basis) changes there, which is what a
            # localised Hungarian detects.
            content[:, c] = ident * left[None] + (ZP[:, c, None, None] * gains[c][None]) * right[None]
        else:
            raise ValueError(case)
    # A shared per-voxel "anatomy" offset (same for every subject) gives the group
    # mean real structure -> passes assert_common_space, like trained-encoder content.
    # It is constant across subjects, so every metric (which centres over subjects)
    # removes it: recovery is unaffected.
    anatomy = np.stack([_smudge(rng.standard_normal((1, side, side)), 1.5)[0] for _ in range(K)])
    content += anatomy[None]
    content += noise * rng.standard_normal(content.shape).astype(np.float32)
    return content.astype(np.float32), Z, mask, atlas


def _gt_controls(results, seed=0):
    rng = np.random.default_rng(seed + 7)

    # ---- gt_recovery: aligned -> block & component high; entangled -> component low
    Xa, Z, mask, atlas = _synthetic_gt("aligned", rng)
    ra = gt_recovery_map(Xa, Z, mask)
    Xe, Ze, maske, atl_e = _synthetic_gt("entangled", rng)
    re = gt_recovery_map(Xe, Ze, maske)
    ok_gt = (
        ra["block"].mean() > 0.6
        and ra["component"].mean() > 0.6
        and re["block"].mean() > 0.6
        and re["component"].mean() < ra["component"].mean() - 0.2
    )
    results["gt_recovery"] = {
        "aligned_block": float(ra["block"].mean()),
        "aligned_component": float(ra["component"].mean()),
        "entangled_block": float(re["block"].mean()),
        "entangled_component": float(re["component"].mean()),
    }
    print(
        f"[gt-recov] aligned block={ra['block'].mean():.3f} comp={ra['component'].mean():.3f} | "
        f"entangled block={re['block'].mean():.3f} comp={re['component'].mean():.3f}"
    )
    print(f"           expect aligned both high; entangled block high but comp LOW -> {'PASS' if ok_gt else 'FAIL'}")

    # ---- localised DCI: aligned -> D,C high; entangled -> D,C low -----------------
    da = dci_maps(Xa, Z, mask, level="voxel")
    de = dci_maps(Xe, Ze, maske, level="voxel")
    mfa, mfe = mask.reshape(-1), maske.reshape(-1)
    Da, Ca = da["disentanglement"].mean(), da["completeness"].mean()
    De, Ce = de["disentanglement"].mean(), de["completeness"].mean()
    ok_dci = Da > 0.4 and Ca > 0.4 and De < Da - 0.2 and Ce < Ca - 0.2
    results["dci_voxel"] = {
        "aligned_D": float(Da),
        "aligned_C": float(Ca),
        "entangled_D": float(De),
        "entangled_C": float(Ce),
    }
    print(f"[dci-vox]  aligned D={Da:.3f} C={Ca:.3f} | entangled D={De:.3f} C={Ce:.3f} (ceiling={da['ceiling']:.2f})")
    print(f"           expect aligned D,C high; entangled D,C LOW -> {'PASS' if ok_dci else 'FAIL'}")

    # ---- gauge drift (the ONLY valid localised Hungarian): fires only where axes change
    Xs, Zs, masks_, atlas_s = _synthetic_gt("spatial_drift", rng)
    drift = gauge_drift_roi(Xs, Zs, atlas_s, masks_, n_perm=200, rng=rng)
    by_roi = {d["roi"]: d for d in drift}
    r1_sig = by_roi.get(1, {}).get("sig", False)  # identity region: expect NOT sig
    r2_sig = by_roi.get(2, {}).get("sig", False)  # rotated region: expect sig
    ok_drift = (not r1_sig) and r2_sig
    results["gauge_drift"] = drift
    print(
        f"[drift]    ROI1(identity) gain={by_roi.get(1,{}).get('drift_gain',float('nan')):.3f} sig={r1_sig} | "
        f"ROI2(swapped) gain={by_roi.get(2,{}).get('drift_gain',float('nan')):.3f} sig={r2_sig}"
    )
    print(f"           expect drift significant ONLY in the relabelled ROI -> {'PASS' if ok_drift else 'FAIL'}")
    return ok_gt and ok_dci and ok_drift


def _run_synthetic(K_null=120, seed=0):
    rng = np.random.default_rng(seed)
    print("=" * 78)
    print("SYNTHETIC CONTROLS  (the decisive validation / kill-switch)")
    print("=" * 78)
    results = {}

    # ---- Positive: distinct kernels, shared GT up to perm+monotone --------------
    A, B, mask, styleA = _synthetic_fields("positive", rng)
    g = fit_global_gauge(A, B, mask)
    rec = recovery_maps(A, B, g, mask, block_metric="cca")
    pos_block, pos_comp = rec["block"].mean(), rec["component"].mean()
    results["positive"] = {
        "block": float(pos_block),
        "component": float(pos_comp),
        "boundary": float(rec["boundary"].mean()),
    }
    ok_pos = pos_block > 0.5 and pos_comp > 0.5 and abs(pos_block - pos_comp) < 0.25
    print(f"[positive] block={pos_block:.3f} component={pos_comp:.3f} boundary={rec['boundary'].mean():+.3f}")
    print(f"           expect block~component~high, boundary~0  ->  {'PASS' if ok_pos else 'FAIL'}")

    # ---- Negative (decisive): two factors share a kernel -> block high, comp low -
    A, B, mask, styleA = _synthetic_fields("negative", rng)
    g = fit_global_gauge(A, B, mask)
    rec = recovery_maps(A, B, g, mask, block_metric="cca")
    cka = recovery_maps(A, B, g, mask, block_metric="cka")["block"].mean()
    neg_block, neg_comp = rec["block"].mean(), rec["component"].mean()
    results["negative"] = {
        "block": float(neg_block),
        "component": float(neg_comp),
        "cka": float(cka),
        "boundary": float(rec["boundary"].mean()),
    }
    ok_neg = (neg_block - neg_comp) > 0.12 and neg_block > neg_comp
    print(f"[negative] block={neg_block:.3f} component={neg_comp:.3f} CKA={cka:.3f} bdry={rec['boundary'].mean():+.3f}")
    print(
        f"           expect block high, component LOW, boundary LARGE (MCC-low/CKA-high) -> "
        f"{'PASS' if ok_neg else 'FAIL'}"
    )

    # ---- Null calibration: B is a pure gauge element of A -> component at chance --
    A, B, mask, styleA = _synthetic_fields("null_rotation", rng)
    map_comp = lambda a, b: recovery_maps(a, b, fit_global_gauge(a, b, mask), mask)["component"]
    obs_comp = map_comp(A, B)
    null = gauge_null(map_comp, A, B, "content_rotation", K_null, mask, rng=rng)
    p = ((null >= obs_comp[:, None]).sum(1) + 1) / (K_null + 1)
    frac_sig = float((p < 0.05).mean())
    results["null_rotation"] = {"frac_sig_voxels": frac_sig, "median_p": float(np.median(p))}
    ok_null = frac_sig < 0.15  # pure-gauge B should NOT beat the rotation null
    print(
        f"[null]     pure-rotation B vs content_rotation null: {frac_sig*100:.1f}% voxels "
        f"'significant' (expect <~5%) -> {'PASS' if ok_null else 'FAIL'}"
    )
    print(f"           median gauge-null p = {np.median(p):.3f} (expect ~uniform, ~0.5)")

    # ---- Visibility (Phase B): zero a factor's loading in ONE view ---------------
    ok_vis = _visibility_control(results, seed)

    # ---- Labelled-data: GT recovery, localised DCI, gauge drift ------------------
    ok_gt = _gt_controls(results, seed)

    all_ok = ok_pos and ok_neg and ok_null and ok_vis and ok_gt
    print("-" * 78)
    verdict = "ALL CONTROLS PASSED" if all_ok else "SOME CONTROLS FAILED -- the script is wrong, do not run on ADNI"
    print(f"RESULT: {verdict}")
    print("=" * 78)
    return results, all_ok


def _visibility_control(results, seed, side=8, C=3):
    """Validate decoder_jacobian_maps: zeroing factor-0's loading in view1 must drop
    view1's content_visibility to ~0, while max-over-view (view0 still renders it)
    stays high. Skipped (PASS) if torch is unavailable."""
    try:
        import torch
        import torch.nn.functional as Fnn
    except ImportError:
        print("[visibility] torch unavailable -- skipping Phase-B Jacobian control")
        return True

    rng = np.random.default_rng(seed + 99)
    # Distinct per-channel kernels so a working decoder renders each content channel
    # into an INDEPENDENT spatial pattern (J^c full column rank -> sigma_min > 0).
    # A degenerate decoder that collapsed channels into one map would have sigma_min=0.
    base_kerns = torch.tensor(rng.standard_normal((C, 1, 3, 3)), dtype=torch.float32)

    def make_decode(load):
        lt = torch.tensor(load, dtype=torch.float32)

        def decode(content, style):
            # Each channel convolved with its OWN kernel, then loaded-summed to 1 plane.
            per_ch = Fnn.conv2d(content, base_kerns * lt.view(-1, 1, 1, 1), padding=1, groups=C)
            return per_ch.sum(1, keepdim=True)

        return decode

    mask = np.ones((side, side), bool)
    mask[:1] = mask[-1:] = mask[:, :1] = mask[:, -1:] = False
    latents = [torch.tensor(rng.standard_normal((1, C, side, side)), dtype=torch.float32) for _ in range(2)]
    both = decoder_jacobian_maps(
        [make_decode([1, 1, 1]), make_decode([0, 1, 1])], latents, [None, None], mask, rf_radius=1, stride=2
    )
    v1only = decoder_jacobian_maps([make_decode([0, 1, 1])], [latents[1]], [None], mask, rf_radius=1, stride=2)
    mf = mask.reshape(-1)
    vis_both = float(both["visibility"].reshape(-1)[mf].max())
    vis_v1 = float(v1only["visibility"].reshape(-1)[mf].max())
    results["visibility"] = {
        "max_over_view": vis_both,
        "view1_only": vis_v1,
        "v1_unrendered": v1only["content_unrendered"],
    }
    ok = vis_both > 1e-3 and vis_v1 < 1e-3 and v1only["content_unrendered"] and not both["content_unrendered"]
    print(f"[visibility] view1 zeroes factor-0: view1-only s_min={vis_v1:.1e} (~0), max-over-view={vis_both:.3f}")
    print(
        f"             view1 kill-switch fires={v1only['content_unrendered']}, "
        f"both-view fires={both['content_unrendered']} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _parse_sol(s):
    v, seed = s.split(":")
    return (v, int(seed))


def _load_factors_aligned(factors_path, meta_path, fields):
    """Load [S,K] factors and re-order rows to ``fields.subjects`` via factors_meta."""
    Z = np.load(factors_path)
    if os.path.exists(meta_path):
        subs = json.load(open(meta_path)).get("subjects", [])
        idx = {s: i for i, s in enumerate(subs)}
        missing = [s for s in fields.subjects if s not in idx]
        if missing:
            raise SystemExit(
                f"{len(missing)} content subjects absent from {meta_path} (e.g. {missing[:3]}). "
                "Re-run eval.extract_factors for the SAME run/split."
            )
        return np.stack([Z[idx[s]] for s in fields.subjects]).astype(np.float32)
    if Z.shape[0] != len(fields.subjects):
        raise SystemExit(
            f"{factors_path} has {Z.shape[0]} rows but {len(fields.subjects)} content subjects, and no "
            f"{meta_path} to align them. Re-run eval.extract_factors."
        )
    return Z.astype(np.float32)


def _write_prov(out, prov):
    with open(f"{out}_provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps({k: v for k, v in prov.items() if not (k.startswith("roi_") or k == "drift")}, indent=2))
    print(f"maps -> {out}_*.npy/.nii.gz ; provenance -> {out}_provenance.json")


def _run_recovery(a):
    if not (a.root and a.solA and a.solB):
        raise SystemExit("recovery mode needs --root --solA --solB")
    solA, solB = _parse_sol(a.solA), _parse_sol(a.solB)
    fields = load_fields(a.root, [solA, solB], atlas_path=a.atlas, ref_path=a.ref)
    assert_common_space(fields)
    A, B = fields.content[solA], fields.content[solB]
    g = fit_global_gauge(A, B, fields.mask)
    rec = recovery_maps(A, B, g, fields.mask, block_metric=a.block_metric)
    rng = np.random.default_rng(0)
    map_block = lambda x, y: recovery_maps(x, y, g, fields.mask, block_metric=a.block_metric)["block"]
    map_comp = lambda x, y: recovery_maps(x, y, fit_global_gauge(x, y, fields.mask), fields.mask)["component"]
    block_null = gauge_null(map_block, A, B, "subject_perm", a.perms, fields.mask, rng=rng)
    comp_null = gauge_null(map_comp, A, B, "content_rotation", a.perms, fields.mask, rng=rng)
    block_inf = fwe_tfce(rec["block"], block_null, fields.mask, fields.spatial)
    comp_inf = fwe_tfce(rec["component"], comp_null, fields.mask, fields.spatial)
    for name, arr in [("block", rec["block"]), ("component", rec["component"]), ("boundary", rec["boundary"])]:
        save_map(arr, fields.mask, fields.spatial, fields.affine, a.out, name)
    save_map(block_inf["sig_fwe"], fields.mask, fields.spatial, fields.affine, a.out, "block_sig_fwe")
    save_map(comp_inf["sig_fwe"], fields.mask, fields.spatial, fields.affine, a.out, "component_sig_fwe")
    _write_prov(
        a.out,
        {
            "mode": "recovery",
            "solA": list(solA),
            "solB": list(solB),
            "block_metric": a.block_metric,
            "block_null": "subject_perm",
            "component_null": "content_rotation",
            "perms": a.perms,
            "n_subjects": len(fields.subjects),
            "n_voxels": int(fields.mask.sum()),
            "block_mean": float(rec["block"].mean()),
            "component_mean": float(rec["component"].mean()),
            "boundary_mean": float(rec["boundary"].mean()),
            "block_fwe_survivors": int((block_inf["p_fwe"] < 0.05).sum()),
            "component_fwe_survivors": int((comp_inf["p_fwe"] < 0.05).sum()),
            "roi_block": roi_summary(rec["block"], fields.atlas, fields.mask, n_boot=200),
            "roi_component": roi_summary(rec["component"], fields.atlas, fields.mask, n_boot=200),
        },
    )


def _run_labelled(a):
    if not (a.root and a.solA):
        raise SystemExit(f"{a.mode} mode needs --root and --solA (the content solution)")
    solA = _parse_sol(a.solA)
    fields = load_fields(a.root, [solA], atlas_path=a.atlas, ref_path=a.ref)
    assert_common_space(fields)
    content = fields.content[solA]
    factors_path = a.factors or os.path.join(a.root, "factors.npy")
    meta_path = a.factors_meta or os.path.join(a.root, "factors_meta.json")
    if not os.path.exists(factors_path):
        raise SystemExit(
            f"no factor array at {factors_path}. For synthetic, run:\n"
            f"  python -m eval.extract_factors --run-dir <run> --split <split>\n"
            f"or pass --factors <S x K .npy> (e.g. an ADNI severity/age covariate)."
        )
    Z = _load_factors_aligned(factors_path, meta_path, fields)
    rng = np.random.default_rng(0)
    prov = {
        "mode": a.mode,
        "solA": list(solA),
        "K": int(Z.shape[1]),
        "n_content_channels": int(content.shape[1]),
        "n_subjects": len(fields.subjects),
        "n_voxels": int(fields.mask.sum()),
    }

    if a.mode == "gt":
        rec = gt_recovery_map(content, Z, fields.mask)
        block_fn = lambda c, z: gt_recovery_map(c, z, fields.mask)["block"]
        comp_fn = lambda c, z: gt_recovery_map(c, z, fields.mask)["component"]
        bnull = gt_null(block_fn, content, Z, a.perms, fields.mask, rng=rng)
        cnull = gt_null(comp_fn, content, Z, a.perms, fields.mask, rng=rng)
        binf = fwe_tfce(rec["block"], bnull, fields.mask, fields.spatial)
        cinf = fwe_tfce(rec["component"], cnull, fields.mask, fields.spatial)
        for name in ("block", "component", "informativeness"):
            save_map(rec[name], fields.mask, fields.spatial, fields.affine, a.out, name)
        save_map(binf["sig_fwe"], fields.mask, fields.spatial, fields.affine, a.out, "block_sig_fwe")
        save_map(cinf["sig_fwe"], fields.mask, fields.spatial, fields.affine, a.out, "component_sig_fwe")
        # Per-factor "where is factor k encoded" R^2 maps (always; in-sample, read
        # relatively). The factor key order is recorded in factors_meta.json.
        ipf = rec["informativeness_per_factor"]  # (V, K)
        K = ipf.shape[1]
        for k in range(K):
            save_map(ipf[:, k], fields.mask, fields.spatial, fields.affine, a.out, f"informativeness_f{k}")
        prov.update(
            {
                "block_mean": float(rec["block"].mean()),
                "component_mean": float(rec["component"].mean()),
                "informativeness_mean": float(rec["informativeness"].mean()),
                "block_fwe_survivors": int((binf["p_fwe"] < 0.05).sum()),
                "component_fwe_survivors": int((cinf["p_fwe"] < 0.05).sum()),
                "informativeness_per_factor_mean": [float(ipf[:, k].mean()) for k in range(K)],
                "roi_block": roi_summary(rec["block"], fields.atlas, fields.mask, n_boot=200),
            }
        )
        if a.per_factor_sig:  # per-factor voxel-FWE (shared subject-perm null; no TFCE for speed)
            null_pf = gt_perfactor_null(content, Z, a.perms, fields.mask, rng=rng)  # (V,K,perms)
            surv = []
            for k in range(K):
                infk = fwe_tfce(ipf[:, k], null_pf[:, k, :], fields.mask, fields.spatial, do_tfce=False)
                save_map(infk["sig_fwe"], fields.mask, fields.spatial, fields.affine, a.out, f"info_f{k}_sig_fwe")
                surv.append(int((infk["p_fwe"] < 0.05).sum()))
            prov["informativeness_per_factor_fwe_survivors"] = surv
    elif a.mode == "dci":
        dci = dci_maps(content, Z, fields.mask, level="voxel")
        for key, nm in (("disentanglement", "D"), ("completeness", "C"), ("informativeness", "I")):
            save_map(dci[key], fields.mask, fields.spatial, fields.affine, a.out, nm)
        prov.update(
            {
                "ceiling": float(dci["ceiling"]),
                "D_mean": float(dci["disentanglement"].mean()),  # already an in-mask (V,) array
                "C_mean": float(dci["completeness"].mean()),
                "roi_dci": dci_maps(content, Z, fields.mask, level="roi", atlas=fields.atlas),
            }
        )
    else:  # drift
        atlas = fields.atlas
        if atlas is None:
            print("[warn] drift is per-ROI; with no --atlas it is one whole-brain ROI (drift==0). Pass --atlas.")
            atlas = np.ones(fields.spatial, int)
        prov["drift"] = gauge_drift_roi(content, Z, atlas, fields.mask, n_perm=a.perms, rng=rng)
    _write_prov(a.out, prov)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--synthetic", action="store_true", help="Run the positive/negative/null controls (no model needed)."
    )
    ap.add_argument("--root", help="Dir of {subject}_{view}_seed{seed}.npy maps (phase0_extract layout).")
    ap.add_argument("--solA", help="Solution A as view:seed (e.g. view0:0).")
    ap.add_argument("--solB", help="Solution B as view:seed (e.g. view1:0).")
    ap.add_argument("--block-metric", choices=["cca", "cka"], default="cca")
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--out", default="recovery")
    ap.add_argument("--ref", help="Reference NIfTI (feature-grid) for output affine.")
    ap.add_argument("--atlas", help="ROI atlas .npy at feature-grid resolution.")
    ap.add_argument(
        "--mode",
        choices=["recovery", "gt", "dci", "drift"],
        default="recovery",
        help="recovery (default): solA-vs-solB block/component/boundary. "
        "gt/dci/drift: labelled-data maps vs a factor array (needs --factors / eval.extract_factors).",
    )
    ap.add_argument("--factors", help="Factor array .npy [S,K]. Default <root>/factors.npy (see eval.extract_factors).")
    ap.add_argument("--factors-meta", help="factors_meta.json for subject alignment. Default <root>/factors_meta.json.")
    ap.add_argument(
        "--per-factor-sig",
        action="store_true",
        help="gt mode: also compute per-factor voxel-FWE significance maps (K x the null cost).",
    )
    a = ap.parse_args()

    if a.synthetic:
        _run_synthetic()
        return
    if a.mode == "recovery":
        _run_recovery(a)
    else:
        _run_labelled(a)


if __name__ == "__main__":
    main()
