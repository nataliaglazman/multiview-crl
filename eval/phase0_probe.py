#!/usr/bin/env python3
"""
phase0_probe.py  --  Phase 0 "kill-switch" probe for cross-modal content-block
identifiability in a multimodal (T1/FLAIR) VQ-VAE-2 with a contrastive content head.

WHAT IT TESTS (both probe-based, hence GAUGE-INVARIANT: invariant to any invertible
reparameterisation of the content block, which is all block-identifiability gives you):

  (1) CROSS-MODAL PREDICTABILITY
      Can the T1 content block predict the FLAIR content block, and vice versa?
      High BIDIRECTIONAL R^2  =>  a shared content block is identified across modalities.
      (One-directional high R^2 is not enough: predictability up to a non-invertible
       map is possible, so we report both directions and take the conservative min.)

  (2) SEED REPRODUCIBILITY
      Is the content block recovered consistently across training seeds?
      Same machinery, SAME modality, two seeds. This R^2 is the CEILING for (1):
      you cannot recover *shared* content more consistently than you recover the
      content at all. Report cross-modal R^2 as a fraction of this ceiling.

Also reports linear proxies (linear CKA, SVCCA mean canonical correlation) and a
shuffled-pairing null (chance-level R^2, should sit at ~0).

KEY DESIGN CHOICES
  * Unit of observation = one brain voxel of one subject's content map (a C-dim vector).
    Justified because the encoder applies the same map everywhere, so the gauge relating
    T1- and FLAIR-content is GLOBAL (location-independent) and voxels are exchangeable.
  * Modalities/seeds MUST be on the same feature grid and intra-subject spatially aligned
    (standard if your T1/FLAIR are co-registered before encoding).
  * Train/test split is BY SUBJECT (GroupKFold). A random voxel split would leak, because
    voxels within a subject are heavily correlated.
  * Background is excluded via a feature-grid brain mask. Background voxels are degenerate
    and will distort R^2 in either direction.
  * Content map is assumed CONTINUOUS (pre-quantisation). For DISCRETE code-index maps,
    R^2/predictability is the wrong operator -- see the note at the bottom of this file.

USAGE
  Smoke test (generates synthetic data with a known shared latent, then runs):
      python -m eval.phase0_probe --synthetic
  On a trained synthetic VQ-VAE-2 (recommended: dump content maps first with
  eval/phase0_extract.py, which writes the {subject}_{modality}_seed{seed}.npy
  layout these loaders expect):
      python -m eval.phase0_extract --run-dir results/<run> --seed 0
      python -m eval.phase0_probe --root results/<run>/phase0_content_maps \
                                  --t1 view0 --flair view1
  Real run from your own .npy dump (any encoder):
      python -m eval.phase0_probe --root /path/to/content_maps --t1 T1 --flair FLAIR \
                                  --seeds 0 1 --voxels-per-subject 1500 --folds 5

  See eval/phase0_extract.py for the two-seed (reproducibility ceiling) workflow.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("This script needs scikit-learn:  pip install scikit-learn")


# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    root: str = "./content_maps"
    subjects: list = field(default_factory=list)  # filled by --synthetic or your loader
    t1: str = "T1"
    flair: str = "FLAIR"
    seeds: list = field(default_factory=lambda: [0, 1])  # need >= 2 for reproducibility
    voxels_per_subject: int = 1500  # subsample to keep N tractable
    folds: int = 5
    standardize: bool = True
    nonlinear_probe: str = "auto"  # "torch" | "sklearn" | "auto" | "none"
    mlp_hidden: tuple = (256, 256)
    mlp_epochs: int = 60
    mlp_lr: float = 1e-3
    mlp_weight_decay: float = 1e-4
    mlp_batch: int = 4096
    ridge_alpha: float = 1.0
    svcca_var: float = 0.99
    rng_seed: int = 0
    device: str = "cuda"  # falls back to cpu automatically
    out_json: str = "phase0_results.json"


# ======================================================================================
# DATA ADAPTER  --  EDIT THESE TWO FUNCTIONS to match your storage layout.
# Each content map must be a CONTINUOUS array of shape (C, *spatial), float.
# ======================================================================================
def load_content_map(subject, modality, seed, cfg: Config) -> np.ndarray:
    """Return the content map for (subject, modality, seed) as (C, *spatial) float32.

    Default layout:  {root}/{subject}_{modality}_seed{seed}.npy
    Swap in nibabel / torch.load / an in-memory cache as needed. If your encoder
    outputs at a downsampled grid, save the maps already at that grid.
    """
    path = os.path.join(cfg.root, f"{subject}_{modality}_seed{seed}.npy")
    arr = np.load(path)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_mask(subject, cfg: Config):
    """Brain mask at FEATURE-GRID resolution, shape == spatial (NO channel axis).

    Return None to use all voxels (NOT recommended -- background distorts R^2).
    To build it: take your subject brain mask and nearest-neighbour downsample it
    to the encoder's feature-grid resolution.
    """
    path = os.path.join(cfg.root, f"{subject}_mask.npy")
    if os.path.exists(path):
        return np.load(path).astype(bool)
    return None


def discover_subjects(cfg: Config) -> list:
    """Enumerate subjects in ``cfg.root`` from the ``{subject}_{t1}_seed{s}.npy``
    files (using the first seed). Lets you point ``--root`` at an extracted dump
    without ``--synthetic`` or hand-editing ``cfg.subjects``."""
    seed = cfg.seeds[0]
    suffix = f"_{cfg.t1}_seed{seed}.npy"
    matches = glob.glob(os.path.join(cfg.root, f"*{suffix}"))
    subjects = sorted(os.path.basename(p)[: -len(suffix)] for p in matches)
    return subjects


# --------------------------------------------------------------------------------------
# ASSEMBLY
# --------------------------------------------------------------------------------------
def _flatten(cmap: np.ndarray, mask) -> np.ndarray:
    """(C, *spatial) -> (n_vox, C), masked to brain voxels."""
    C = cmap.shape[0]
    flat = cmap.reshape(C, -1).T  # (n_vox_all, C)
    if mask is not None:
        flat = flat[mask.reshape(-1).astype(bool)]
    return flat


def assemble_xy(subjects, src, tgt, cfg: Config, rng):
    """src, tgt are (modality, seed). Returns X (N,C), Y (N,C), groups (N,)."""
    Xs, Ys, G = [], [], []
    for s in subjects:
        cx = load_content_map(s, src[0], src[1], cfg)
        cy = load_content_map(s, tgt[0], tgt[1], cfg)
        if cx.shape != cy.shape:
            raise ValueError(
                f"{s}: shape mismatch {cx.shape} vs {cy.shape} " f"(modalities/seeds must share the feature grid)"
            )
        mask = load_mask(s, cfg)
        fx, fy = _flatten(cx, mask), _flatten(cy, mask)
        n = fx.shape[0]
        if cfg.voxels_per_subject and n > cfg.voxels_per_subject:
            idx = rng.choice(n, cfg.voxels_per_subject, replace=False)
            fx, fy = fx[idx], fy[idx]
        Xs.append(fx)
        Ys.append(fy)
        G.append(np.full(fx.shape[0], str(s)))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(G)


# --------------------------------------------------------------------------------------
# PROBES
# --------------------------------------------------------------------------------------
def ridge_r2(Xtr, Ytr, Xte, Yte, alpha):
    m = Ridge(alpha=alpha).fit(Xtr, Ytr)
    return float(r2_score(Yte, m.predict(Xte), multioutput="uniform_average"))


def mlp_r2_sklearn(Xtr, Ytr, Xte, Yte, cfg):
    from sklearn.neural_network import MLPRegressor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore non-convergence chatter
        m = MLPRegressor(
            hidden_layer_sizes=cfg.mlp_hidden,
            max_iter=cfg.mlp_epochs,
            alpha=cfg.mlp_weight_decay,
            learning_rate_init=cfg.mlp_lr,
            random_state=cfg.rng_seed,
        ).fit(Xtr, Ytr)
    return float(r2_score(Yte, m.predict(Xte), multioutput="uniform_average"))


def mlp_r2_torch(Xtr, Ytr, Xte, Yte, cfg):
    import torch
    import torch.nn as nn

    dev = cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    layers, d = [], Xtr.shape[1]
    for h in cfg.mlp_hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, Ytr.shape[1])]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.mlp_lr, weight_decay=cfg.mlp_weight_decay)
    lossf = nn.MSELoss()
    n = Xtr_t.shape[0]
    for _ in range(cfg.mlp_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, cfg.mlp_batch):
            b = perm[i : i + cfg.mlp_batch]
            opt.zero_grad()
            lossf(net(Xtr_t[b]), Ytr_t[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).cpu().numpy()
    return float(r2_score(Yte, pred, multioutput="uniform_average"))


def pick_nonlinear(cfg):
    if cfg.nonlinear_probe == "none":
        return None
    if cfg.nonlinear_probe in ("torch", "auto"):
        try:
            import torch  # noqa

            return mlp_r2_torch
        except ImportError:
            if cfg.nonlinear_probe == "torch":
                raise
    return mlp_r2_sklearn


# --------------------------------------------------------------------------------------
# LINEAR REPRESENTATION-SIMILARITY PROXIES  (no training; gauge classes they respect:
#   linear CKA -> orthogonal + isotropic scaling;  CCA/SVCCA -> any invertible linear)
# --------------------------------------------------------------------------------------
def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (denom + 1e-12))


def _cca_corrs(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def svcca_mean_corr(X, Y, var=0.99, max_comp=256):
    def reduce(M):
        Mc = M - M.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Mc, full_matrices=False)
        cum = np.cumsum(S**2) / np.sum(S**2)
        k = int(np.searchsorted(cum, var) + 1)
        k = min(k, max_comp)
        return U[:, :k] * S[:k]

    return float(np.mean(_cca_corrs(reduce(X), reduce(Y))))


# --------------------------------------------------------------------------------------
# CV DRIVER
# --------------------------------------------------------------------------------------
def run_direction(Xsrc, Ytgt, groups, cfg, nonlinear_fn):
    """Predict Ytgt from Xsrc with subject-grouped CV. Returns dict of metric lists."""
    gkf = GroupKFold(n_splits=cfg.folds)
    out = {"ridge": [], "mlp": [], "null": []}
    rng = np.random.default_rng(cfg.rng_seed)
    for tr, te in gkf.split(Xsrc, Ytgt, groups):
        Xtr, Xte, Ytr, Yte = Xsrc[tr], Xsrc[te], Ytgt[tr], Ytgt[te]
        if cfg.standardize:
            sx = StandardScaler().fit(Xtr)
            Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)
            sy = StandardScaler().fit(Ytr)
            Ytr, Yte = sy.transform(Ytr), sy.transform(Yte)
        out["ridge"].append(ridge_r2(Xtr, Ytr, Xte, Yte, cfg.ridge_alpha))
        # chance floor: destroy the X<->Y relationship in TRAINING, then test on real
        # pairs. A model that learned nothing predicts ~the mean -> R^2 ~ 0.
        out["null"].append(ridge_r2(Xtr, Ytr[rng.permutation(len(Ytr))], Xte, Yte, cfg.ridge_alpha))
        if nonlinear_fn is not None:
            out["mlp"].append(nonlinear_fn(Xtr, Ytr, Xte, Yte, cfg))
    return out


def _ms(vals):
    a = np.asarray(vals, float)
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), 0.0)


def compare(name, src, tgt, cfg, nonlinear_fn):
    rng = np.random.default_rng(cfg.rng_seed)
    X, Y, groups = assemble_xy(cfg.subjects, src, tgt, cfg, rng)
    fwd = run_direction(X, Y, groups, cfg, nonlinear_fn)  # predict tgt from src
    rev = run_direction(Y, X, groups, cfg, nonlinear_fn)  # predict src from tgt
    # similarity proxies on standardised pooled data (no training -> no leakage)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    res = {
        "name": name,
        "src": list(src),
        "tgt": list(tgt),
        "n_samples": int(X.shape[0]),
        "C": int(X.shape[1]),
        "ridge_fwd": _ms(fwd["ridge"]),
        "ridge_rev": _ms(rev["ridge"]),
        "mlp_fwd": _ms(fwd["mlp"]),
        "mlp_rev": _ms(rev["mlp"]),
        "null_fwd": _ms(fwd["null"]),
        "null_rev": _ms(rev["null"]),
        "linear_cka": linear_cka(Xs, Ys),
        "svcca_mean_corr": svcca_mean_corr(Xs, Ys, cfg.svcca_var),
    }
    # headline bidirectional numbers (conservative = min over directions)
    probe = "mlp" if fwd["mlp"] else "ridge"
    res["bidir_R2"] = min(res[f"{probe}_fwd"][0], res[f"{probe}_rev"][0])
    res["probe_used"] = probe
    return res


def fmt(t):
    return f"{t[0]:+.3f} ± {t[1]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--t1", default=None)
    ap.add_argument("--flair", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--voxels-per-subject", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--nonlinear", default=None, choices=["torch", "sklearn", "auto", "none"])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit subject ids. Default: auto-discover from --root (no --synthetic needed).",
    )
    a = ap.parse_args()

    cfg = Config()
    if a.root:
        cfg.root = a.root
    if a.t1:
        cfg.t1 = a.t1
    if a.flair:
        cfg.flair = a.flair
    if a.seeds:
        cfg.seeds = a.seeds
    if a.voxels_per_subject:
        cfg.voxels_per_subject = a.voxels_per_subject
    if a.folds:
        cfg.folds = a.folds
    if a.nonlinear:
        cfg.nonlinear_probe = a.nonlinear
    if a.out:
        cfg.out_json = a.out

    if a.synthetic and a.root and glob.glob(os.path.join(cfg.root, "*.npy")):
        sys.exit(
            f"--synthetic would overwrite .npy files already in {cfg.root}. It is a self-test "
            "that writes toy data; run it without --root (uses ./content_maps), or drop "
            "--synthetic to probe a real extracted dump."
        )
    if a.synthetic:
        cfg.subjects = make_synthetic(cfg)
    elif a.subjects:
        cfg.subjects = a.subjects
    elif not cfg.subjects:
        cfg.subjects = discover_subjects(cfg)
    if not cfg.subjects:
        sys.exit(
            f"No subjects found in {cfg.root} matching *_{cfg.t1}_seed{cfg.seeds[0]}.npy. "
            "Run eval/phase0_extract.py first, or pass --subjects / --synthetic."
        )
    if len(cfg.seeds) < 2:
        print("[warn] seed reproducibility needs >=2 seeds; skipping that comparison.")

    nonlinear_fn = pick_nonlinear(cfg)
    print(
        f"Subjects: {len(cfg.subjects)} | seeds: {cfg.seeds} | "
        f"voxels/subj: {cfg.voxels_per_subject} | folds: {cfg.folds} | "
        f"nonlinear probe: {nonlinear_fn.__name__ if nonlinear_fn else 'none'}\n"
    )

    results = []
    s0 = cfg.seeds[0]
    results.append(compare("cross_modal_T1_to_FLAIR", (cfg.t1, s0), (cfg.flair, s0), cfg, nonlinear_fn))
    if len(cfg.seeds) >= 2:
        results.append(compare("seed_repro_T1", (cfg.t1, cfg.seeds[0]), (cfg.t1, cfg.seeds[1]), cfg, nonlinear_fn))

    # ---- report ----
    print("=" * 84)
    for r in results:
        print(f"[{r['name']}]   N={r['n_samples']:,}  C={r['C']}")
        print(f"   predict tgt<-src :  ridge {fmt(r['ridge_fwd'])}   mlp {fmt(r['mlp_fwd'])}")
        print(f"   predict src<-tgt :  ridge {fmt(r['ridge_rev'])}   mlp {fmt(r['mlp_rev'])}")
        print(f"   shuffled null    :  ridge {fmt(r['null_fwd'])}  (should be ~0)")
        print(f"   linear CKA {r['linear_cka']:.3f}   SVCCA mean corr {r['svcca_mean_corr']:.3f}")
        print(f"   --> bidirectional R^2 ({r['probe_used']}, conservative min) = {r['bidir_R2']:+.3f}")
        print("-" * 84)

    cm = next(r for r in results if r["name"].startswith("cross_modal"))
    sr = next((r for r in results if r["name"].startswith("seed_repro")), None)
    print("INTERPRETATION")
    if sr is not None:
        ceil = sr["bidir_R2"]
        frac = cm["bidir_R2"] / ceil if ceil > 1e-6 else float("nan")
        print(f"  seed-repro bidir R^2 (ceiling)   = {ceil:+.3f}")
        print(f"  cross-modal bidir R^2            = {cm['bidir_R2']:+.3f}")
        print(f"  cross-modal as fraction of ceil  = {frac:.2f}")
        print("  Read: ceiling near 0 => block not even reproducible (fix the model, not the eval).")
        print("        cross-modal << ceiling      => content not shared across modalities (style leak).")
        print("        cross-modal ~ ceiling, both high => shared content block is identified. Proceed.")
    else:
        print(f"  cross-modal bidir R^2 = {cm['bidir_R2']:+.3f}  (add a 2nd seed to get the ceiling).")

    with open(cfg.out_json, "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2)
    print(f"\nSaved -> {cfg.out_json}")


# --------------------------------------------------------------------------------------
# SYNTHETIC DATA  --  shared latent L, different nonlinear gauges per modality/seed.
# Validates: cross-modal & seed-repro R^2 high; null ~0; nonlinear probe >= linear.
# --------------------------------------------------------------------------------------
def make_synthetic(cfg: Config, n_subjects=40, K=6, C=32, hw=16, noise=0.15):
    os.makedirs(cfg.root, exist_ok=True)
    rng = np.random.default_rng(123)
    cfg.subjects = [f"sub-{i:04d}" for i in range(n_subjects)]
    cfg.seeds = [0, 1]

    def gauge():  # random mildly-nonlinear injective-ish map R^K -> R^C
        W1 = rng.normal(0, 1, (K, C))
        W2 = rng.normal(0, 1, (C, C))
        return lambda L: np.tanh(L @ W1) @ W2

    g = {("T1", 0): gauge(), ("T1", 1): gauge(), ("FLAIR", 0): gauge()}  # diff seed -> diff gauge
    for s in cfg.subjects:
        L = rng.normal(0, 1, (hw * hw, K))  # shared content field (per voxel)
        mask = np.ones((hw, hw), bool)
        mask[:2] = False
        mask[-2:] = False  # crude brain mask
        for (mod, seed), gm in g.items():
            X = gm(L) + noise * rng.normal(0, 1, (hw * hw, C))
            np.save(os.path.join(cfg.root, f"{s}_{mod}_seed{seed}.npy"), X.T.reshape(C, hw, hw).astype(np.float32))
        np.save(os.path.join(cfg.root, f"{s}_mask.npy"), mask)
    return cfg.subjects


if __name__ == "__main__":
    main()

# ======================================================================================
# DISCRETE CODE-INDEX MAPS?  Do NOT use this script as-is. R^2/ridge/MLP assume a
# continuous target. For VQ code maps:
#   * cross-modal "predictability" -> classification accuracy / adjusted mutual
#     information between code-index maps (after Hungarian codebook alignment), or
#   * compare per-code spatial occupancy maps.
# Run THIS probe on the continuous PRE-quantisation content features instead, which is
# also where the BYOL-style contrastive objective actually operates.
# ======================================================================================
