#!/usr/bin/env python
"""Content-block geometry diagnostic: effective rank per pooling + PCA-truncation R² curve.

Answers two questions that ``run_dci_compare``'s single GAP ``content_rank`` cannot,
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
   the variance past that point is *not* about the content factors — redundancy, or
   leaked nuisance/style worth checking against ``content→style``.

The R² is the variance-weighted CV test-R² of predicting *all* content factors
jointly (``cv_probe_r2`` with a multioutput target), averaged over seeds — the same
probe family as ``run_dci_compare``.

Usage
-----
    python -m eval.content_rank_pca \
        --run-dirs runs/no_contrastive runs/infonce \
        --names baseline contrastive \
        --num-samples 2000 --level 0 --out content_rank_out

PCA is fit on all samples (unsupervised, transductive but label-free — the same
choice ``run_dci_compare._reduce_reprs`` makes), so the curve is an honest
"how many directions carry the factors" readout.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.decomposition import PCA

from eval.dci import _extract_synthetic_representations
from eval.identifiability_metrics import cv_probe_r2
from eval.run_dci_compare import _CONTENT, _effective_rank, parse_poolings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _pc_grid(ncomp):
    """Coarse-but-dense grid of cumulative-PC counts up to ``ncomp`` (always incl. ncomp)."""
    base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
    return sorted({k for k in base if k <= ncomp} | {ncomp}) if ncomp >= 1 else []


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


def analyse_model(name, run_dir, dataset, poolings, level, checkpoint_name, max_pcs, seeds, batch_size, num_workers):
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
        grid, r2_by_k, r2_full, n95 = pca_truncation_curve(content, gt_content, max_pcs, seeds)
        rows.append(
            {
                "model": name,
                "pooling": key,
                "feat_dim": content.shape[1],
                "nominal_channels": info.get("n_content_channels", content.shape[1]),
                "eff_rank": rank,
                "r2_full": r2_full,
                "n_pcs_95": n95,
                "grid": grid,
                "r2_by_k": r2_by_k,
            }
        )

    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _f(x, nd=2):
    return f"{x:.{nd}f}" if (x is not None and isinstance(x, (int, float)) and np.isfinite(x)) else "  -  "


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
        print(
            f"  {r['model']:<16s} {r['pooling']:<8s} {r['feat_dim']:>8d} " f"{_f(r['eff_rank'], 1):>9s} {_f(ratio):>9s}"
        )


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
            ["model", "pooling", "feat_dim", "nominal_channels", "eff_rank", "r2_full", "n_pcs_95", "k", "r2_at_k"]
        )
        for r in rows:
            for k in r["grid"]:
                w.writerow(
                    [
                        r["model"],
                        r["pooling"],
                        r["feat_dim"],
                        r["nominal_channels"],
                        round(r["eff_rank"], 4) if np.isfinite(r["eff_rank"]) else "",
                        round(r["r2_full"], 4) if np.isfinite(r["r2_full"]) else "",
                        r["n_pcs_95"] if r["n_pcs_95"] is not None else "",
                        k,
                        round(r["r2_by_k"][k], 4) if np.isfinite(r["r2_by_k"][k]) else "",
                    ]
                )
    logger.info("Wrote %s", path)


def main():
    p = argparse.ArgumentParser(description="Content effective rank (per pooling) + PCA-truncation R² curve.")
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
    p.add_argument("--out", default="content_rank_out", help="Output directory.")
    cli = p.parse_args()

    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in cli.run_dirs]
    if len(names) != len(cli.run_dirs):
        p.error("--names must have one entry per --run-dirs")
    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))

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
                )
            )
        except Exception as e:
            logger.error("Skipping %s (%s): %s", name, run_dir, e)

    if not all_rows:
        logger.error("Nothing analysed successfully.")
        return

    print_rank_table(all_rows)
    print_pca_table(all_rows)
    os.makedirs(cli.out, exist_ok=True)
    write_csv(all_rows, cli.out)
    save_plot(all_rows, cli.out)


if __name__ == "__main__":
    main()
