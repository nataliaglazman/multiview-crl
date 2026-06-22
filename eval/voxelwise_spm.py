#!/usr/bin/env python3
"""
voxelwise_spm.py  --  Voxel-wise cross-modal content-block identifiability map (VBM-style).

This is the spatial counterpart to phase0_probe.py. Instead of one pooled number, it
produces a STATISTICAL PARAMETRIC MAP over the feature grid: at each voxel, how strongly
is the C-dim T1 content vector associated with the C-dim FLAIR content vector ACROSS
SUBJECTS. High, significant voxels = where the shared content block is identified.

STATISTIC (per voxel)
    Pillai's trace  V = sum_i rho_i^2  = || Xtilde^T Ytilde ||_F^2
    where rho_i are the canonical correlations between the two modalities' content vectors
    over subjects, and Xtilde/Ytilde are the per-voxel whitened data. Pillai is invariant
    to any invertible LINEAR reparameterisation of either block -- i.e. to the content
    gauge (use phase0 to confirm the gauge is ~linear; if it is strongly nonlinear, swap
    in a distance-correlation statistic, noted at the bottom -- much costlier).
    Reported normalised as Pillai / min(C, S-1) in [0,1] (mean squared canonical corr).

INFERENCE (the reason this is valid despite the statistic being biased upward when C is
non-trivial relative to S): permutation. Under H0 of no cross-modal association at a voxel,
the subject pairing is exchangeable, so we permute subject labels of one modality. The
SAME permutations are applied at every voxel, giving a valid max-statistic distribution for
voxel-level FWE control. The statistic's upward bias cancels because the null is computed
with the identical (biased) statistic.

WHY IT'S CHEAP: the whitening matrices depend only on Xc^T Xc and Yc^T Yc, which are
INVARIANT to permuting subjects (a sum over subjects). So we whiten ONCE; each permutation
is then Ytilde[:, perm, :] and a batched matmul. No re-fitting per permutation.

PREREQUISITES that differ from phase0:
  * Content maps must be in a COMMON TEMPLATE SPACE at feature-grid resolution -- here a
    voxel is held fixed while subjects vary, so voxel u must be the same anatomy in everyone.
  * A GROUP mask is used (voxels in-brain for ALL subjects), so every voxel has all S subjects.

OUTPUT (saved as .npy, and .nii.gz if a reference affine is given):
  stat (normalised Pillai), z (vs permutation null), p_unc, q_fdr (BH), p_fwe (voxel-level),
  plus a thresholded map. Run with --tfce for TFCE + TFCE-FWE (slower; see notes).

USAGE
  Smoke test (plants a signal blob; null elsewhere):
      python -m eval.voxelwise_spm --synthetic
  On a trained synthetic VQ-VAE-2: first dump content maps with eval/phase0_extract.py
  (writes the {subject}_{view}_seed{seed}.npy layout these loaders consume), then point
  --root at the dump. Subjects are auto-discovered from --root, so no code editing:
      python -m eval.phase0_extract --run-dir results/<run> --seed 0
      python -m eval.voxelwise_spm --root results/<run>/phase0_content_maps \
            --src view0:0 --tgt view1:0 --perms 1000 --lam 1e-2 --out cm_spm
  Real run from any .npy dump in the same layout:
      python -m eval.voxelwise_spm --root /maps --src T1:0 --tgt FLAIR:0 \
            --perms 1000 --lam 1e-2 --ref /path/template_featuregrid.nii.gz --out cm_spm
  Seed-reproducibility ceiling map instead:  --src view0:0 --tgt view0:1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
@dataclass
class Config:
    root: str = "./content_maps"
    subjects: list = field(default_factory=list)
    src: tuple = ("T1", 0)  # (modality, seed)
    tgt: tuple = ("FLAIR", 0)
    perms: int = 1000
    lam: float = 1e-2  # whitening ridge (shrinkage); raise if C close to S
    covariates: str = ""  # optional .npy of shape (S, k) nuisance regressors
    tfce: bool = False
    tfce_H: float = 2.0
    tfce_E: float = 0.5
    tfce_dh: float = 0.1
    alpha: float = 0.05
    rng_seed: int = 0
    backend: str = "auto"  # "numpy" | "torch" | "auto"
    ref: str = ""  # reference NIfTI (feature-grid) to copy affine for output
    out: str = "voxelwise_spm"


# ===== DATA ADAPTER (identical contract to phase0_probe.py) ============================
def load_content_map(subject, modality, seed, cfg):
    """(C, *spatial) float32, in COMMON TEMPLATE SPACE at feature-grid resolution."""
    arr = np.load(os.path.join(cfg.root, f"{subject}_{modality}_seed{seed}.npy"))
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_mask(subject, cfg):
    """Brain mask (*spatial) at feature-grid resolution, or None."""
    p = os.path.join(cfg.root, f"{subject}_mask.npy")
    return np.load(p).astype(bool) if os.path.exists(p) else None


def discover_subjects(cfg):
    """Enumerate subjects in ``cfg.root`` from the ``{subject}_{src}_seed{s}.npy``
    files (using the source modality/seed). Lets you point ``--root`` at an
    eval/phase0_extract dump without ``--synthetic`` or hand-editing cfg.subjects."""
    suffix = f"_{cfg.src[0]}_seed{cfg.src[1]}.npy"
    matches = glob.glob(os.path.join(cfg.root, f"*{suffix}"))
    return sorted(os.path.basename(p)[: -len(suffix)] for p in matches)


# ======================================================================================


def build_arrays(cfg):
    """Returns Xv,Yv of shape (V, S, C), the group mask (flat bool), and spatial shape."""
    subs = cfg.subjects
    shape = None
    gmask = None
    # first pass: shapes + group mask (intersection of per-subject masks & finite content)
    for s in subs:
        cx = load_content_map(s, cfg.src[0], cfg.src[1], cfg)
        cy = load_content_map(s, cfg.tgt[0], cfg.tgt[1], cfg)
        if cx.shape != cy.shape:
            raise ValueError(f"{s}: {cx.shape} vs {cy.shape} -- src/tgt must share the grid")
        if shape is None:
            shape = cx.shape[1:]
            gmask = np.ones(int(np.prod(shape)), bool)
        finite = np.isfinite(cx).all(0).reshape(-1) & np.isfinite(cy).all(0).reshape(-1)
        m = load_mask(s, cfg)
        if m is not None:
            finite &= m.reshape(-1)
        gmask &= finite
    V = int(gmask.sum())
    S = len(subs)
    C = cx.shape[0]
    print(f"group mask: {V} voxels in-brain for all {S} subjects | C={C} | grid={shape}")
    mem = 2 * V * S * C * 4 / 1e9
    print(f"approx whitened-array memory: {mem:.1f} GB (x2 for both modalities held in RAM)")
    Xv = np.empty((V, S, C), np.float32)
    Yv = np.empty((V, S, C), np.float32)
    # second pass: fill
    for j, s in enumerate(subs):
        cx = load_content_map(s, cfg.src[0], cfg.src[1], cfg).reshape(C, -1).T[gmask]  # (V,C)
        cy = load_content_map(s, cfg.tgt[0], cfg.tgt[1], cfg).reshape(C, -1).T[gmask]
        Xv[:, j, :] = cx
        Yv[:, j, :] = cy
    return Xv, Yv, gmask, shape


def residualise(arr, M):
    """Project out nuisances on the subject axis. arr (V,S,C), M (S,S) = I - Z Z^+."""
    return np.einsum("st,vtc->vsc", M, arr, optimize=True).astype(np.float32)


def whiten(arr, lam):
    """Per-voxel whitening so Xtilde^T Xtilde ~ I. arr (V,S,C) -> (V,S,C).
    Permutation-invariant: depends only on the (subject-summed) Gram Xc^T Xc."""
    Xc = arr - arr.mean(1, keepdims=True)
    G = np.matmul(Xc.transpose(0, 2, 1), Xc)  # (V,C,C) = Xc^T Xc
    C = G.shape[-1]
    # Scaled ridge plus a tiny ABSOLUTE floor: a voxel with ~no across-subject
    # variance (e.g. a degenerate in-mask voxel) has trace(G)->0, so the scaled
    # ridge alone vanishes, leaving a ~0 eigenvalue and w^-1/2 -> inf/nan that
    # would poison the max-statistic FWE null. The absolute term keeps G PD; such
    # a voxel then whitens to ~0 (zero signal), not nan.
    ridge = lam * np.trace(G, axis1=1, axis2=2) / C + 1e-8  # (V,)
    G = G + ridge[:, None, None] * np.eye(C)
    w, Q = np.linalg.eigh(G)  # G = Q diag(w) Q^T, w>0
    w = np.clip(w, 1e-12, None)  # guard eigh round-off (tiny neg)
    inv_sqrt = Q @ (Q.transpose(0, 2, 1) * (w[:, :, None] ** -0.5))  # Q diag(w^-1/2) Q^T
    return np.matmul(Xc, inv_sqrt).astype(np.float32)  # Xtilde


def pillai_map(Xt, Yt_perm):
    """|| Xt^T Yt ||_F^2 per voxel. Xt (V,S,C), Yt_perm (V,S,C) -> (V,)."""
    A = np.matmul(Xt.transpose(0, 2, 1), Yt_perm)  # (V,C,C) batched GEMM
    return np.einsum("vij,vij->v", A, A, optimize=True)


def run_permutations(Xt, Yt, perms, rng, backend):
    """Returns observed (V,) and null (V,P) Pillai statistics."""
    V, S, C = Xt.shape
    obs = pillai_map(Xt, Yt)
    null = np.empty((V, perms), np.float32)
    if backend == "torch":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Xtt = torch.tensor(Xt, device=dev)
        Ytt = torch.tensor(Yt, device=dev)
        XtT = Xtt.transpose(1, 2)
        for p in range(perms):
            idx = torch.tensor(rng.permutation(S), device=dev)
            A = torch.matmul(XtT, Ytt[:, idx, :])
            null[:, p] = (A * A).sum((1, 2)).cpu().numpy()
    else:
        for p in range(perms):
            null[:, p] = pillai_map(Xt, Yt[:, rng.permutation(S), :])
    return obs, null


def benjamini_hochberg(p):
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def tfce(stat_flat, gmask, shape, H, E, dh):
    """3D TFCE on a positive statistic. Returns enhanced values on in-mask voxels."""
    from scipy import ndimage

    img = np.zeros(int(np.prod(shape)), np.float32)
    img[gmask] = stat_flat
    img = img.reshape(shape)
    out = np.zeros_like(img)
    smax = img.max()
    if smax <= 0:
        return out.reshape(-1)[gmask]
    struct = ndimage.generate_binary_structure(img.ndim, img.ndim)  # full connectivity
    for h in np.arange(dh, smax + dh, dh):
        lab, n = ndimage.label(img >= h, structure=struct)
        if n == 0:
            continue
        sizes = np.bincount(lab.reshape(-1))
        sizes[0] = 0  # label 0 = below threshold
        ext = sizes[lab]  # cluster extent per voxel (0 if below)
        out += (ext**E) * (h**H) * dh
    return out.reshape(-1)[gmask]


def save_maps(maps, gmask, shape, cfg):
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    full = {}
    for k, v in maps.items():
        vol = np.zeros(int(np.prod(shape)), np.float32)
        vol[gmask] = v
        full[k] = vol.reshape(shape)
        np.save(f"{cfg.out}_{k}.npy", full[k])
    if cfg.ref:
        try:
            import nibabel as nib

            aff = nib.load(cfg.ref).affine
            for k, vol in full.items():
                nib.save(nib.Nifti1Image(vol, aff), f"{cfg.out}_{k}.nii.gz")
            print(f"wrote NIfTI maps with affine from {cfg.ref}")
        except Exception as e:
            print(f"[warn] NIfTI save skipped ({e}); .npy maps written")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--src")
    ap.add_argument("--tgt")
    ap.add_argument("--perms", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--covariates")
    ap.add_argument("--tfce", action="store_true")
    ap.add_argument("--backend", choices=["numpy", "torch", "auto"])
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--rng-seed", type=int)
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit subject ids. Default: auto-discover from --root (no --synthetic needed).",
    )
    a = ap.parse_args()
    cfg = Config()
    for k in ("root", "perms", "lam", "covariates", "backend", "ref", "out", "alpha"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.rng_seed is not None:
        cfg.rng_seed = a.rng_seed
    if a.tfce:
        cfg.tfce = True
    if a.src:
        cfg.src = tuple(a.src.split(":")[:1]) + (int(a.src.split(":")[1]),)
    if a.tgt:
        cfg.tgt = tuple(a.tgt.split(":")[:1]) + (int(a.tgt.split(":")[1]),)

    if a.synthetic and a.root and glob.glob(os.path.join(cfg.root, "*.npy")):
        sys.exit(
            f"--synthetic would overwrite .npy files already in {cfg.root}. It is a self-test "
            "that writes toy data; run it without --root (uses ./content_maps), or drop "
            "--synthetic to map a real extracted dump."
        )
    if a.synthetic:
        signal = make_synthetic(cfg)
    elif a.subjects:
        cfg.subjects = a.subjects
    elif not cfg.subjects:
        cfg.subjects = discover_subjects(cfg)
    if not cfg.subjects:
        sys.exit(
            f"No subjects found in {cfg.root} matching *_{cfg.src[0]}_seed{cfg.src[1]}.npy. "
            "Run eval/phase0_extract.py first, or pass --subjects / --synthetic."
        )

    backend = cfg.backend
    if backend == "auto":
        try:
            import torch  # noqa

            backend = "torch"
        except ImportError:
            backend = "numpy"
    rng = np.random.default_rng(cfg.rng_seed)
    print(f"src={cfg.src} tgt={cfg.tgt} | perms={cfg.perms} | lam={cfg.lam} | backend={backend}\n")

    Xv, Yv, gmask, shape = build_arrays(cfg)
    S = Xv.shape[1]
    if cfg.covariates:
        Z = np.load(cfg.covariates).astype(np.float64)
        Z = np.column_stack([np.ones(S), Z])  # add intercept
        M = (np.eye(S) - Z @ np.linalg.pinv(Z)).astype(np.float32)
        Xv, Yv = residualise(Xv, M), residualise(Yv, M)
        print(
            f"residualised on {Z.shape[1]-1} covariate(s) " "(approx Freedman-Lane; use PALM for publication inference)"
        )

    Xt, Yt = whiten(Xv, cfg.lam), whiten(Yv, cfg.lam)
    del Xv, Yv
    obs, null = run_permutations(Xt, Yt, cfg.perms, rng, backend)

    denom = min(Xt.shape[2], S - 1)
    stat = obs / denom  # normalised Pillai in ~[0,1]
    nmean, nstd = null.mean(1), null.std(1) + 1e-9
    z = (obs - nmean) / nstd
    p_unc = (null >= obs[:, None]).mean(1)
    p_unc = (p_unc * cfg.perms + 1) / (cfg.perms + 1)  # add-one (no zero p-values)
    q_fdr = benjamini_hochberg(p_unc)
    maxstat = null.max(0)  # (P,) max over voxels per perm
    p_fwe = ((maxstat[None, :] >= obs[:, None]).sum(1) + 1) / (cfg.perms + 1)

    maps = {"stat": stat, "z": z, "p_unc": p_unc, "q_fdr": q_fdr, "p_fwe": p_fwe, "sig_fwe": stat * (p_fwe < cfg.alpha)}

    if cfg.tfce:
        print("computing TFCE (full map per permutation -- slower)...")
        znull = (null - nmean[:, None]) / nstd[:, None]
        t_obs = tfce(np.clip(z, 0, None), gmask, shape, cfg.tfce_H, cfg.tfce_E, cfg.tfce_dh)
        t_max = np.empty(cfg.perms)
        for p in range(cfg.perms):
            t_max[p] = tfce(np.clip(znull[:, p], 0, None), gmask, shape, cfg.tfce_H, cfg.tfce_E, cfg.tfce_dh).max()
        p_tfce = ((t_max[None, :] >= t_obs[:, None]).sum(1) + 1) / (cfg.perms + 1)
        maps["tfce"] = t_obs
        maps["p_tfce"] = p_tfce
        maps["sig_tfce"] = stat * (p_tfce < cfg.alpha)

    full = save_maps(maps, gmask, shape, cfg)

    # ---- report ----
    n_fwe = int((p_fwe < cfg.alpha).sum())
    print("\n" + "=" * 70)
    print(f"voxels: {gmask.sum()}   normalised Pillai: " f"median {np.median(stat):.3f}  max {stat.max():.3f}")
    print(f"survivors @ FWE<{cfg.alpha} (voxel-level): {n_fwe}")
    print(f"survivors @ FDR<{cfg.alpha}: {int((q_fdr < cfg.alpha).sum())}")
    if cfg.tfce:
        print(f"survivors @ TFCE-FWE<{cfg.alpha}: {int((maps['p_tfce'] < cfg.alpha).sum())}")
    if a.synthetic:
        sig = signal.reshape(-1)[gmask].astype(bool)
        surv = p_fwe < cfg.alpha
        tp = int((surv & sig).sum())
        fp = int((surv & ~sig).sum())
        print(
            f"[synthetic check] planted-signal voxels recovered (FWE): {tp}/{int(sig.sum())}"
            f"   false positives in null region: {fp}/{int((~sig).sum())}"
        )
    print("=" * 70)
    json.dump(
        {
            "config": {**asdict(cfg), "src": list(cfg.src), "tgt": list(cfg.tgt)},
            "n_fwe": n_fwe,
            "median_stat": float(np.median(stat)),
            "max_stat": float(stat.max()),
        },
        open(f"{cfg.out}_summary.json", "w"),
        indent=2,
    )
    print(f"maps -> {cfg.out}_*.npy ; summary -> {cfg.out}_summary.json")


# --------------------------------------------------------------------------------------
# SYNTHETIC: a central signal blob carries shared content across modalities (per subject),
# the rest is independent per-modality noise. The map should localise to the blob and the
# null region should not survive FWE.
# --------------------------------------------------------------------------------------
def make_synthetic(cfg, S=60, K=4, C=16, side=20, noise=0.4, smooth=1.0):
    os.makedirs(cfg.root, exist_ok=True)
    from scipy import ndimage

    rng = np.random.default_rng(7)
    cfg.subjects = [f"sub-{i:03d}" for i in range(S)]
    cfg.src, cfg.tgt = ("T1", 0), ("FLAIR", 0)
    yy, xx = np.mgrid[0:side, 0:side]
    signal = ((xx - side / 2) ** 2 + (yy - side / 2) ** 2) < (side / 4) ** 2  # blob
    sig_flat = signal.reshape(-1)
    W1 = rng.normal(0, 1, (K, C))
    W2 = rng.normal(0, 1, (K, C))  # diff gauge per modality

    def smudge(a):  # (C, side, side) spatial smoothing, like a receptive field
        return ndimage.gaussian_filter(a, sigma=(0, smooth, smooth))

    for s in cfg.subjects:
        L = rng.normal(0, 1, (side * side, K))  # shared latent (only used in blob)
        t1 = np.where(sig_flat[:, None], np.tanh(L @ W1), rng.normal(0, 1, (side * side, C)))
        fl = np.where(sig_flat[:, None], np.tanh(L @ W2), rng.normal(0, 1, (side * side, C)))
        t1 = t1 + noise * rng.normal(0, 1, t1.shape)
        fl = fl + noise * rng.normal(0, 1, fl.shape)
        t1 = smudge(t1.T.reshape(C, side, side)).astype(np.float32)
        fl = smudge(fl.T.reshape(C, side, side)).astype(np.float32)
        np.save(os.path.join(cfg.root, f"{s}_T1_seed0.npy"), t1)
        np.save(os.path.join(cfg.root, f"{s}_FLAIR_seed0.npy"), fl)
    return signal


if __name__ == "__main__":
    main()

# ======================================================================================
# NONLINEAR GAUGE?  Pillai is a LINEAR-gauge statistic. If phase0 showed MLP >> ridge,
# the linear map understates consistency. Replace pillai_map with a per-voxel distance
# correlation (captures any dependence): precompute per-voxel doubly-centred distance
# matrices for X once; each permutation is dCor with the row/col-permuted Y distance
# matrix. Cost is O(V * S^2) memory/time -- only feasible for small grids or subsampled
# voxels, which is why the nonlinear check is better done pooled (phase0) or on extracted
# low-d factors than across the whole map.
# DISCRETE CODE MAPS: Pillai/CCA need continuous inputs -- run on pre-quantisation features.
# ======================================================================================
