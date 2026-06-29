#!/usr/bin/env python
"""Content-block geometry diagnostic: effective rank + PCA-truncation R² + ceiling + per-factor.

Answers four questions that ``run_dci_compare``'s single GAP ``content_rank`` cannot,
for a set of trained runs scored on one shared synthetic test set:

1. **Is a low GAP effective rank a *collapse* or just *spatial storage*?**
   Effective rank is reported under every pooling (``gap`` / ``stats`` / ``patch``).
   A reconstruction baseline stores content in the spatial layout, so its GAP rank
   can be tiny (~4) while its stats/patch rank is large — that is spatial storage,
   not collapse.  A genuinely collapsed block is low-rank under *every* pooling.

2. **Is a high effective rank real content dimensionality or surplus
   redundancy/nuisance?**  For each block we PCA the features and fit a ridge probe
   from the first ``k`` PCs to the (shared) content factors, sweeping ``k``.  If the
   R² curve saturates at ``k`` far below the effective rank (``n_pcs@95%`` ≪ rank),
   the variance past that point is *not* about the content factors.

3. **Is weak *linear* recovery a model failure or just non-linear encoding /
   a data ceiling?**  The CEILING table puts the linear (ridge) full-block R² next
   to a non-linear (MLP) one, plus a raw-downsampled-pixel row.

4. **Which content factors are actually recovered?**  The PER-FACTOR table splits the
   variance-weighted aggregate into one linear R² per factor, each read at the pooling
   that can expose it (global→gap, localized→patch), so "0.35 overall" is resolved into
   e.g. "global anatomy ~0.7, lesions ~0".

The R² is the variance-weighted CV test-R² of predicting the content factors via
``cv_probe_r2`` (the same probe family as ``run_dci_compare``), averaged over seeds.

Usage
-----
    python -m eval.content_rank_pca \
        --run-dirs runs/no_contrastive runs/infonce \
        --names baseline contrastive \
        --num-samples 2000 --level 0 --out content_rank_out

PCA is fit on all samples (unsupervised, transductive but label-free — the same
choice ``run_dci_compare._reduce_reprs`` makes).  The MLP ceiling is the slow part;
pass ``--no-mlp`` to skip it, ``--raw-grid 0`` to skip the raw-pixel row.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.decomposition import PCA

from eval.dci import _extract_synthetic_representations
from eval.identifiability_metrics import block_mcc, cv_probe_r2
from eval.run_dci_compare import _CONTENT, FACTOR_POOLING, _effective_rank, parse_poolings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _pc_grid(ncomp):
    """Coarse-but-dense grid of cumulative-PC counts up to ``ncomp`` (always incl. ncomp)."""
    base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
    return sorted({k for k in base if k <= ncomp} | {ncomp}) if ncomp >= 1 else []


def _pca_reduce(X, n_comp):
    """PCA-reduce ``X`` to ``n_comp`` components (unsupervised, fit on all samples — the
    same transductive choice ``pca_truncation_curve`` makes). No-op if ``n_comp<=0`` or
    ``>=`` the current width."""
    if n_comp is None or n_comp <= 0:
        return X
    n, d = X.shape
    k = int(min(n_comp, d, n - 1))
    if k >= d:
        return X
    return PCA(n_components=k, random_state=0).fit_transform(X)


def pca_truncation_curve(X, factors, max_pcs, seeds):
    """R² of predicting ``factors`` from the first k content PCs, swept over k.

    Returns ``(grid, r2_by_k, r2_full, n_pcs_95)`` where ``r2_full`` is the
    full-block (no-PCA) ceiling and ``n_pcs_95`` is the smallest k reaching 95% of it.
    """
    n, d = X.shape
    if d == 0 or n < 20:
        return [], {}, float("nan"), None

    r2_full = cv_probe_r2(X, factors, seeds=seeds)["mean"]

    ncomp = int(min(max_pcs, d, n - 1))
    pca = PCA(n_components=ncomp, random_state=0).fit(X)
    scores = pca.transform(X)

    grid = _pc_grid(ncomp)
    r2_by_k = {k: cv_probe_r2(scores[:, :k], factors, seeds=seeds)["mean"] for k in grid}

    n_pcs_95 = None
    if np.isfinite(r2_full) and r2_full > 0:
        thresh = 0.95 * r2_full
        for k in grid:
            if np.isfinite(r2_by_k[k]) and r2_by_k[k] >= thresh:
                n_pcs_95 = k
                break
    return grid, r2_by_k, r2_full, n_pcs_95


def analyse_model(
    name,
    run_dir,
    dataset,
    poolings,
    level,
    checkpoint_name,
    max_pcs,
    seeds,
    batch_size,
    num_workers,
    compute_mlp=True,
    compute_per_factor_mlp=False,
    probe_dim=0,
):
    """Load one run, extract its content block under every pooling, and score geometry."""
    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir

    logger.info("=== %s (%s) ===", name, run_dir)
    ckpt = os.path.join(run_dir, checkpoint_name)
    if not os.path.exists(ckpt):
        ckpt = None  # loader falls back to vqvae_model.pt
    model, _args, device = load_model_from_run_dir(run_dir, ckpt, None)

    rows = []
    gt_content = None
    for key, value in poolings:
        level_data, gc, _gsv1, _gsv2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value
        )
        if gt_content is None:
            gt_content = gc
        if level not in level_data:
            logger.warning("  level %d absent under pooling %s — skipping", level, key)
            continue
        content = level_data[level][_CONTENT]
        info = level_data[level][4]
        if content is None or content.shape[1] == 0:
            continue

        rank = _effective_rank(content)
        # Probe-capacity control: PCA-reduce the block to ``probe_dim`` comps before the
        # probes only (eff_rank/feat_dim stay on the FULL block). Kills the p>>n overfit on
        # the huge patch block and the capacity confound when models/baselines differ in
        # channel count. probe_dim<=0 = no reduction.
        content_p = _pca_reduce(content, probe_dim)
        grid, r2_by_k, r2_full, n95 = pca_truncation_curve(content_p, gt_content, max_pcs, seeds)
        # Non-linear ceiling on the full block: mlp >> ridge => factors are there but
        # encoded non-linearly (identifiable up to a smooth map, not linearly).
        r2_full_mlp = (
            cv_probe_r2(content_p, gt_content, seeds=seeds, kind="mlp")["mean"] if compute_mlp else float("nan")
        )
        # Per-factor linear R²: splits the variance-weighted aggregate into one number
        # per content factor, so "global recovered / localized not" is visible.
        names = info["content_names"]
        per_factor = {f: cv_probe_r2(content_p, gt_content[:, j], seeds=seeds)["mean"] for j, f in enumerate(names)}
        # Per-factor MLP R², at gap/patch only (expensive). Lets the format-breakdown
        # table tell reformatting (a factor weak at lin@gap but recoverable by mlp@gap
        # or lin@patch) from genuinely-absent content (weak under every readout).
        per_factor_mlp = {}
        if compute_per_factor_mlp and key in ("gap", "patch"):
            per_factor_mlp = {
                f: cv_probe_r2(content_p, gt_content[:, j], seeds=seeds, kind="mlp")["mean"]
                for j, f in enumerate(names)
            }
        # Block-MCC (run_dci_compare's headline metric) on THIS pooling — so it can be read
        # across the gap->stats->patch locality ladder, not only the stats pooling.
        mcc_full = block_mcc(content_p, gt_content, seeds=seeds)["mean"]
        rows.append(
            {
                "model": name,
                "pooling": key,
                "feat_dim": content.shape[1],
                "nominal_channels": info.get("n_content_channels", content.shape[1]),
                "eff_rank": rank,
                "r2_full": r2_full,
                "block_mcc": mcc_full,
                "r2_full_mlp": r2_full_mlp,
                "n_pcs_95": n95,
                "grid": grid,
                "r2_by_k": r2_by_k,
                "per_factor": per_factor,
                "per_factor_mlp": per_factor_mlp,
                "content_names": list(names),
            }
        )

    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def extract_raw_pixels(dataset, grid, batch_size=32, num_workers=0):
    """Downsampled raw-image features (view 1) + content factors, aligned by sample order."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, gtc = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"]
            if isinstance(img, (list, tuple)):
                img = img[0]  # primary view
            img = img.float()
            if img.dim() == 5:
                pooled = F.adaptive_avg_pool3d(img, (grid, grid, grid))
            elif img.dim() == 4:
                pooled = F.adaptive_avg_pool2d(img, (grid, grid))
            else:
                pooled = img
            feats.append(pooled.reshape(pooled.shape[0], -1).cpu().numpy())
            gtc.append(batch["gt_latents"]["z_content"].numpy())
    return np.concatenate(feats), np.concatenate(gtc)


def raw_pixel_ceiling(dataset, grid, seeds, batch_size, num_workers, compute_mlp=True, probe_dim=0):
    """The data ceiling: how well the content factors are recoverable from raw pixels."""
    X, Z = extract_raw_pixels(dataset, grid, batch_size, num_workers)
    Xp = _pca_reduce(X, probe_dim)
    return {
        "feat_dim": X.shape[1],
        "grid": grid,
        "r2_lin": cv_probe_r2(Xp, Z, seeds=seeds)["mean"],
        "r2_mlp": cv_probe_r2(Xp, Z, seeds=seeds, kind="mlp")["mean"] if compute_mlp else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _f(x, nd=2):
    return f"{x:.{nd}f}" if (x is not None and isinstance(x, (int, float)) and np.isfinite(x)) else "  -  "


def _rnd(x):
    return round(float(x), 4) if isinstance(x, (int, float)) and np.isfinite(x) else ""


def print_rank_table(rows):
    print("\n" + "=" * 72)
    print("  CONTENT EFFECTIVE RANK BY POOLING")
    print("  low under gap but high under stats/patch => content is SPATIAL, not collapsed")
    print("  low under EVERY pooling => genuine dimensional collapse")
    print("=" * 72)
    print(f"  {'model':<16s} {'pooling':<8s} {'feat_dim':>8s} {'eff_rank':>9s} {'rank/dim':>9s}")
    print("  " + "-" * 60)
    for r in rows:
        ratio = r["eff_rank"] / r["feat_dim"] if r["feat_dim"] else float("nan")
        print(f"  {r['model']:<16s} {r['pooling']:<8s} {r['feat_dim']:>8d} {_f(r['eff_rank'], 1):>9s} {_f(ratio):>9s}")


def print_mcc_table(rows):
    print("\n" + "=" * 72)
    print("  BLOCK-MCC BY POOLING  (affine identifiability: content block -> gt_content)")
    print("  read across gap->stats->patch: at which locality the content is recoverable")
    print("=" * 72)
    print(f"  {'model':<16s} {'pooling':<8s} {'block_mcc':>10s}")
    print("  " + "-" * 38)
    for r in rows:
        print(f"  {r['model']:<16s} {r['pooling']:<8s} {_f(r['block_mcc'], 3):>10s}")


def print_pca_table(rows):
    print("\n" + "=" * 72)
    print("  PCA-TRUNCATION R²  (content PCs -> all content factors, variance-weighted)")
    print("  n_pcs@95% ≪ eff_rank  => surplus rank is redundancy/nuisance, not factors")
    print("=" * 72)
    checkpts = [1, 2, 4, 8, 12, 16, 24]
    hdr = "  ".join(f"{f'R²@{k}':>7s}" for k in checkpts)
    print(f"  {'model':<16s} {'pooling':<8s} {hdr}  {'R²full':>7s} {'n@95%':>6s}")
    print("  " + "-" * 96)
    for r in rows:
        cells = []
        for k in checkpts:
            kk = next((g for g in r["grid"] if g >= k), None)
            cells.append(_f(r["r2_by_k"].get(kk), 3) if kk is not None else "  -  ")
        n95 = str(r["n_pcs_95"]) if r["n_pcs_95"] is not None else "  -  "
        print(
            f"  {r['model']:<16s} {r['pooling']:<8s} "
            + "  ".join(f"{c:>7s}" for c in cells)
            + f"  {_f(r['r2_full'], 3):>7s} {n95:>6s}"
        )


def print_ceiling_table(rows, raw=None):
    print("\n" + "=" * 72)
    print("  LINEAR vs NON-LINEAR CEILING  (R² -> all content factors, full block)")
    print("  mlp >> linear  => factors encoded NON-LINEARLY (identifiable up to a smooth map)")
    print("  model R² ≈ raw-pixel R²  => recovery is data-limited, not a model failure")
    print("=" * 72)
    print(f"  {'scope':<16s} {'pooling':<8s} {'linear':>8s} {'mlp':>8s} {'mlp-lin':>8s}")
    print("  " + "-" * 56)
    for r in rows:
        lin, mlp = r.get("r2_full", float("nan")), r.get("r2_full_mlp", float("nan"))
        gain = mlp - lin if (np.isfinite(lin) and np.isfinite(mlp)) else float("nan")
        print(f"  {r['model']:<16s} {r['pooling']:<8s} {_f(lin, 3):>8s} {_f(mlp, 3):>8s} {_f(gain, 3):>8s}")
    if raw is not None:
        gain = (
            raw["r2_mlp"] - raw["r2_lin"]
            if (np.isfinite(raw["r2_lin"]) and np.isfinite(raw["r2_mlp"]))
            else float("nan")
        )
        glabel = f"{raw['grid']}x{raw['grid']}x{raw['grid']}"
        print("  " + "-" * 56)
        print(
            f"  {'RAW PIXELS':<16s} {glabel:<8s} {_f(raw['r2_lin'], 3):>8s} {_f(raw['r2_mlp'], 3):>8s} {_f(gain, 3):>8s}"
        )
        print("  (raw = downsampled input image — the best any readout could extract)")


def print_per_factor_table(rows):
    """One linear R² per content factor, read at the pooling that can expose it."""
    by, models, names = {}, [], None
    for r in rows:
        by[(r["model"], r["pooling"])] = r.get("per_factor", {})
        if r["model"] not in models:
            models.append(r["model"])
        if names is None and r.get("content_names"):
            names = r["content_names"]
    if not names:
        return
    avail = sorted({r["pooling"] for r in rows})

    def assigned(f):
        a = FACTOR_POOLING.get(f, "gap")
        return a if a in avail else (avail[0] if avail else "gap")

    print("\n" + "=" * 72)
    print("  PER-FACTOR LINEAR R²  (each factor at its assigned pooling)")
    print("  splits the aggregate: which content factors are actually recovered")
    print("=" * 72)
    fw = max([len(n) for n in names] + [16])
    width = fw + 2 + 6 + 12 * len(models)
    print(f"  {'factor':<{fw}s} {'pool':<6s}" + "".join(f"{m:>12s}" for m in models))
    print("  " + "-" * width)
    sums = {m: [] for m in models}
    for f in names:
        pool = assigned(f)
        cells = []
        for m in models:
            v = by.get((m, pool), {}).get(f, float("nan"))
            cells.append(v)
            if np.isfinite(v):
                sums[m].append(v)
        print(f"  {f:<{fw}s} {pool:<6s}" + "".join(f"{_f(v, 3):>12s}" for v in cells))
    print("  " + "-" * width)
    means = [float(np.mean(sums[m])) if sums[m] else float("nan") for m in models]
    print(f"  {'(mean of factors)':<{fw}s} {'-':<6s}" + "".join(f"{_f(v, 3):>12s}" for v in means))


def _format_cells(lin, mlp, model, factor):
    """The four {linear, mlp} x {gap, patch} readouts for one (model, factor), + best."""
    cells = {
        "lin_gap": lin.get((model, "gap"), {}).get(factor, float("nan")),
        "mlp_gap": mlp.get((model, "gap"), {}).get(factor, float("nan")),
        "lin_patch": lin.get((model, "patch"), {}).get(factor, float("nan")),
        "mlp_patch": mlp.get((model, "patch"), {}).get(factor, float("nan")),
    }
    finite = [v for v in cells.values() if np.isfinite(v)]
    best = max(finite) if finite else float("nan")
    return cells, best


def _format_index(rows):
    """``(lin, mlp, models, names)`` indexed by ``(model, pooling)`` for the format table."""
    lin, mlp, models, names = {}, {}, [], None
    for r in rows:
        lin[(r["model"], r["pooling"])] = r.get("per_factor", {})
        mlp[(r["model"], r["pooling"])] = r.get("per_factor_mlp", {})
        if r["model"] not in models:
            models.append(r["model"])
        if names is None and r.get("content_names"):
            names = r["content_names"]
    return lin, mlp, models, names


def print_per_factor_format_table(rows):
    """Per-factor linear-vs-MLP at gap-vs-patch — separates reformatting from new content.

    For a model, ``lin_gap`` far below ``best`` means the factor's content is stored
    non-globally (recoverable at ``lin_patch``) or non-linearly (recoverable at
    ``mlp_gap``), not absent.  If one model's ``lin_gap`` ≈ its ``best`` while another's
    ``lin_gap`` ≪ ``best`` for the same factor, the gap between them is a global-linear
    *reformat* (same content, made more accessible), not extra identifiable content.
    """
    lin, mlp, models, names = _format_index(rows)
    if not names or not any(mlp.values()):
        return

    aliases = {m: f"m{i}" for i, m in enumerate(models)}
    print("\n" + "=" * 72)
    print("  PER-FACTOR FORMAT BREAKDOWN  (linear vs MLP × gap vs patch)")
    print("  lin_gap ≪ best => content stored non-globally/non-linearly, not absent;")
    print("  other model's lin_gap ≈ best => global-linear REFORMAT, not new content")
    print("=" * 72)
    for m in models:
        print(f"  {aliases[m]} = {m}")
    fw = max([len(n) for n in names] + [16])
    print(
        f"  {'factor':<{fw}s} {'m':<3s} {'lin_gap':>8s} {'mlp_gap':>8s} "
        f"{'lin_patch':>9s} {'mlp_patch':>9s} {'best':>7s}"
    )
    print("  " + "-" * (fw + 3 + 8 + 8 + 9 + 9 + 7 + 6))
    for f in names:
        for m in models:
            cells, best = _format_cells(lin, mlp, m, f)
            print(
                f"  {f:<{fw}s} {aliases[m]:<3s} {_f(cells['lin_gap'], 3):>8s} "
                f"{_f(cells['mlp_gap'], 3):>8s} {_f(cells['lin_patch'], 3):>9s} "
                f"{_f(cells['mlp_patch'], 3):>9s} {_f(best, 3):>7s}"
            )


def save_plot(rows, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib unavailable (%s) — skipping plot.", e)
        return

    pools = sorted({r["pooling"] for r in rows})
    fig, axes = plt.subplots(1, len(pools), figsize=(5 * len(pools), 4), squeeze=False)
    for ax, pool in zip(axes[0], pools):
        for r in [r for r in rows if r["pooling"] == pool]:
            ks = r["grid"]
            ys = [r["r2_by_k"][k] for k in ks]
            ax.plot(ks, ys, marker="o", ms=3, label=f"{r['model']} (rank {r['eff_rank']:.0f})")
            if np.isfinite(r["r2_full"]):
                ax.axhline(r["r2_full"], ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"pooling = {pool}")
        ax.set_xlabel("# content PCs")
        ax.set_ylabel("CV R² -> content factors")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "content_rank_pca.png")
    fig.savefig(path, dpi=130)
    logger.info("Wrote %s", path)


def write_csv(rows, out_dir):
    path = os.path.join(out_dir, "content_rank_pca.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "pooling",
                "feat_dim",
                "nominal_channels",
                "eff_rank",
                "r2_full",
                "block_mcc",
                "n_pcs_95",
                "k",
                "r2_at_k",
            ]
        )
        for r in rows:
            for k in r["grid"]:
                w.writerow(
                    [
                        r["model"],
                        r["pooling"],
                        r["feat_dim"],
                        r["nominal_channels"],
                        _rnd(r["eff_rank"]),
                        _rnd(r["r2_full"]),
                        _rnd(r["block_mcc"]),
                        r["n_pcs_95"] if r["n_pcs_95"] is not None else "",
                        k,
                        _rnd(r["r2_by_k"][k]),
                    ]
                )
    logger.info("Wrote %s", path)


def write_ceiling_csv(rows, raw, out_dir):
    path = os.path.join(out_dir, "content_rank_ceiling.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "pooling", "feat_dim", "r2_full_linear", "r2_full_mlp"])
        for r in rows:
            w.writerow([r["model"], r["pooling"], r["feat_dim"], _rnd(r.get("r2_full")), _rnd(r.get("r2_full_mlp"))])
        if raw is not None:
            w.writerow(
                [
                    "RAW_PIXELS",
                    f"{raw['grid']}x{raw['grid']}x{raw['grid']}",
                    raw["feat_dim"],
                    _rnd(raw["r2_lin"]),
                    _rnd(raw["r2_mlp"]),
                ]
            )
    logger.info("Wrote %s", path)


def write_per_factor_csv(rows, out_dir):
    path = os.path.join(out_dir, "content_rank_per_factor.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pooling", "factor", "r2_linear"])
        for r in rows:
            for fname, v in r.get("per_factor", {}).items():
                w.writerow([r["model"], r["pooling"], fname, _rnd(v)])
    logger.info("Wrote %s", path)


def write_per_factor_format_csv(rows, out_dir):
    lin, mlp, models, names = _format_index(rows)
    if not names or not any(mlp.values()):
        return
    path = os.path.join(out_dir, "content_rank_per_factor_format.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "factor", "lin_gap", "mlp_gap", "lin_patch", "mlp_patch", "best"])
        for fac in names:
            for m in models:
                cells, best = _format_cells(lin, mlp, m, fac)
                w.writerow(
                    [m, fac] + [_rnd(cells[k]) for k in ("lin_gap", "mlp_gap", "lin_patch", "mlp_patch")] + [_rnd(best)]
                )
    logger.info("Wrote %s", path)


def main():
    p = argparse.ArgumentParser(description="Content effective rank + PCA-truncation R² + ceiling + per-factor.")
    p.add_argument("--run-dirs", nargs="+", required=True, help="Run directories (settings.json each).")
    p.add_argument("--names", nargs="*", default=None, help="Labels (default: basename of each run-dir).")
    p.add_argument("--checkpoint-name", default="vqvae_best.pt", help="Checkpoint file (falls back to vqvae_model.pt).")
    p.add_argument("--num-samples", type=int, default=2000, help="Frozen test-set size, shared across runs.")
    p.add_argument("--poolings", default="gap,stats,2x2x2", help="Comma list: gap, stats, and/or DxHxW.")
    p.add_argument("--level", type=int, default=0, help="Encoder level to analyse.")
    p.add_argument("--max-pcs", type=int, default=48, help="Cap on PCs in the truncation sweep.")
    p.add_argument("--seeds", default="0,1", help="CV seeds for the probe (fewer = faster).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-mlp", action="store_true", help="Skip the (slow) non-linear MLP ceiling.")
    p.add_argument(
        "--per-factor-mlp",
        action="store_true",
        help="Also compute per-factor MLP R² at gap/patch (slow) for the format-breakdown table.",
    )
    p.add_argument("--raw-grid", type=int, default=16, help="Raw-pixel ceiling grid GxGxG (0 = skip the raw row).")
    p.add_argument(
        "--probe-dim",
        type=int,
        default=0,
        help="PCA-reduce every block to this many components before the probes (ceiling, per-factor, "
        "block-MCC, raw-pixel). Removes the p>>n overfit on the huge patch block and the capacity "
        "confound when models/baselines differ in channel count. eff_rank/feat_dim stay on the full "
        "block. 0 = no reduction.",
    )
    p.add_argument("--out", default="content_rank_out", help="Output directory.")
    cli = p.parse_args()

    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in cli.run_dirs]
    if len(names) != len(cli.run_dirs):
        p.error("--names must have one entry per --run-dirs")
    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))
    compute_mlp = not cli.no_mlp
    if cli.probe_dim and cli.probe_dim > 0:
        logger.info("Probes PCA-reduced to %d components (eff_rank/feat_dim stay on the full block).", cli.probe_dim)

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    ref_args = load_run_args(cli.run_dirs[0])
    dataset = build_synthetic_test_set(ref_args, cli.num_samples)
    logger.info("Frozen test set: %d samples, shared across %d run(s).", cli.num_samples, len(cli.run_dirs))

    all_rows = []
    for name, run_dir in zip(names, cli.run_dirs):
        try:
            all_rows.extend(
                analyse_model(
                    name,
                    run_dir,
                    dataset,
                    poolings,
                    cli.level,
                    cli.checkpoint_name,
                    cli.max_pcs,
                    seeds,
                    cli.batch_size,
                    cli.num_workers,
                    compute_mlp=compute_mlp,
                    compute_per_factor_mlp=cli.per_factor_mlp,
                    probe_dim=cli.probe_dim,
                )
            )
        except Exception as e:
            logger.error("Skipping %s (%s): %s", name, run_dir, e)

    if not all_rows:
        logger.error("Nothing analysed successfully.")
        return

    raw = None
    if cli.raw_grid and cli.raw_grid > 0:
        try:
            logger.info("Computing raw-pixel ceiling (grid %d^3)...", cli.raw_grid)
            raw = raw_pixel_ceiling(
                dataset, cli.raw_grid, seeds, cli.batch_size, cli.num_workers, compute_mlp, cli.probe_dim
            )
        except Exception as e:
            logger.error("Raw-pixel ceiling failed: %s", e)

    print_rank_table(all_rows)
    print_mcc_table(all_rows)
    print_pca_table(all_rows)
    print_ceiling_table(all_rows, raw)
    print_per_factor_table(all_rows)
    if cli.per_factor_mlp:
        print_per_factor_format_table(all_rows)
    os.makedirs(cli.out, exist_ok=True)
    write_csv(all_rows, cli.out)
    write_ceiling_csv(all_rows, raw, cli.out)
    write_per_factor_csv(all_rows, cli.out)
    if cli.per_factor_mlp:
        write_per_factor_format_csv(all_rows, cli.out)
    save_plot(all_rows, cli.out)


if __name__ == "__main__":
    main()
