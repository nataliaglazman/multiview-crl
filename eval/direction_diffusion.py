#!/usr/bin/env python
"""Why is the contrastive model's content DIFFUSE — does alignment sustain the surplus
directions, or does the redundancy term create them?

Matched comparison established the problem: the contrastive block reaches only 40% of its
own full R² from its top 8 principal components, where the recon baseline reaches 76% — and
it does this at effective rank 37.9 against the baseline's 8.0.  More directions, less
information in each.  Its patch probes lose even at a matched 24 components (0.623 vs
0.729), so this is not a capacity difference.

Two mechanisms could put information into many weak directions, and they need opposite fixes:

    A  ALIGNMENT sustains them.  BT's on_diag rewards every channel for cross-view
       correlation, and a redundant, spread-out code is easy to align.  Fix is on the
       alignment side.
    B  The REDUNDANCY term creates them.  off_diag requires d mutually decorrelated
       directions to exist; with only ~9 factors, the surplus must be filled with
       decorrelated non-factor variance.  Fix is bt_lambda / bt_gap_lambda.

B has the stronger prior for a structural reason — Roy-Vetterli rank measures exactly the
flatness of the singular spectrum, which is exactly what a decorrelation penalty controls,
so "high rank, diffuse information" is its predicted signature.  This measures rather than
assumes it.

The discriminator
-----------------
Score every direction on THREE quantities instead of one:

    variance share    how much of the representation the direction occupies
    information       univariate CV R^2 from that direction alone to the GT factors
    alignment         corr(view1, view2) projected onto the SAME direction

Then look at the **surplus** directions — high variance, low information — which is where
the diffusion lives, and read their alignment:

    surplus WELL aligned   -> A. The alignment term is what keeps them alive.
    surplus POORLY aligned -> B. Nothing rewards their alignment; they exist because
                                 decorrelation requires d directions and only ~9 carry
                                 factors.

Two controls come free
----------------------
* The **recon-only baseline** has no contrastive term at all and sits at rank 8.0, so
  reconstruction demonstrably does not produce high rank.  That rules out recon as the
  cause before the analysis starts, and its profile is the reference for "dense".
* **Spearman(variance rank, information rank)** is the diffusion itself as one number:
  high = information lives where the variance is; low/negative = it does not.

Both views are projected onto a COMMON basis (fit on view 1, applied to both).  Fitting a
separate basis per view rotates them differently and manufactures a difference out of
nothing — the exact bug planted data caught in ``eval/alignment_scale_test.py``.

Validated on planted features (2000 samples, 44 dims, 9 factors):

    planted representation              spearman   surplus align   productive align
    concentrated + aligned surplus         +0.781           0.980             0.980
    concentrated + unaligned surplus       +0.779           0.023             0.728
    diffuse (info in the tail)             -0.694           0.980               nan

Spearman separates concentrated from diffuse (+0.78 vs -0.69); surplus alignment separates
the two mechanisms (0.980 vs 0.023) independently of it, which is what makes them a usable
pair rather than one number.  ``productive_align`` is NaN in the diffuse row BY CONSTRUCTION
— the informative directions are all low-variance there, so the (hi-var AND hi-info) set is
empty.  Read ``surplus_align``; ``productive_align`` is context, not the discriminator.

Usage
-----
    python -m eval.direction_diffusion \
        --run-dirs results/synthetic/BASELINE results/synthetic/CONTRASTIVE \
        --names baseline contrastive --poolings gap --causal match
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from eval.identifiability_metrics import cv_probe_r2

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def common_basis(c1, c2, n_comp):
    """Project both views onto ONE basis fit on view 1. Returns (p1, p2, variance share)."""
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    n_comp = int(min(n_comp, c1.shape[1], c1.shape[0] - 1))
    mu = c1.mean(0, keepdims=True)
    pca = PCA(n_components=n_comp, random_state=0).fit(c1 - mu)
    return pca.transform(c1 - mu), pca.transform(c2 - mu), pca.explained_variance_ratio_


def direction_stats(p1, p2, gt, seeds):
    """Per-direction cross-view alignment and univariate factor information."""
    n = p1.shape[1]
    align = np.empty(n)
    info = np.empty(n)
    for k in range(n):
        a, b = p1[:, k], p2[:, k]
        align[k] = 0.0 if a.std() < 1e-12 or b.std() < 1e-12 else abs(np.corrcoef(a, b)[0, 1])
        info[k] = cv_probe_r2(p1[:, [k]], gt, seeds=seeds)["mean"]
    return align, np.clip(np.nan_to_num(info, nan=0.0), 0.0, None)


def summarise(var_share, info, align):
    """Diffusion (Spearman) plus the surplus-vs-productive alignment split."""
    rho = spearmanr(var_share, info).statistic if len(var_share) > 2 else float("nan")
    # RANK-based halves, not value-based. Most directions carry ~zero information, so the
    # MEDIAN of `info` is ~0 and `info >= median` marks almost everything as informative —
    # which empties the surplus set entirely and drags `productive_align` toward the noise
    # directions. Caught on planted data, where it returned NaN surplus in all three regimes.
    n = len(var_share)
    var_rank = np.argsort(np.argsort(-var_share))
    info_rank = np.argsort(np.argsort(-info))
    hi_var = var_rank < n // 2
    hi_info = info_rank < n // 2
    surplus = hi_var & ~hi_info  # occupies space, carries little
    productive = hi_var & hi_info
    return {
        "spearman_var_info": float(rho),
        "n_surplus": int(surplus.sum()),
        "surplus_align": float(align[surplus].mean()) if surplus.any() else float("nan"),
        "surplus_var": float(var_share[surplus].sum()) if surplus.any() else float("nan"),
        "productive_align": float(align[productive].mean()) if productive.any() else float("nan"),
        "productive_var": float(var_share[productive].sum()) if productive.any() else float("nan"),
        "mean_align": float(align.mean()),
    }


def main():
    p = argparse.ArgumentParser(description="Is the content diffuse because of alignment or decorrelation?")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--poolings", default="gap", help="gap first: 44 dims, and where the GAP-BT term acts.")
    p.add_argument("--n-comp", type=int, default=44, help="Directions to score (capped by block width).")
    p.add_argument("--top", type=int, default=12, help="Rows of the per-direction table to print.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="direction_diffusion_out")
    cli = p.parse_args()

    seeds = tuple(int(s) for s in cli.seeds.split(","))
    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in cli.run_dirs]
    if len(names) != len(cli.run_dirs):
        p.error("--names must have one entry per --run-dirs")

    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2, parse_poolings
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref_args = load_run_args(cli.run_dirs[0])
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")

    rows = []
    for name, run_dir in zip(names, cli.run_dirs):
        ckpt = os.path.join(run_dir, cli.checkpoint_name)
        model, _a, device = load_model_from_run_dir(run_dir, ckpt if os.path.exists(ckpt) else None, None)
        for key, value in parse_poolings(cli.poolings):
            ld, gt, _s1, _s2 = _extract_synthetic_representations(
                model, dataset, device, cli.batch_size, cli.num_workers, pooling=value
            )
            if cli.level not in ld:
                continue
            c1, c2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
            if c1 is None or c2 is None or c1.shape[1] == 0:
                continue
            p1, p2, var_share = common_basis(c1, c2, cli.n_comp)
            align, info = direction_stats(p1, p2, gt, seeds)
            s = summarise(var_share, info, align)

            print("\n" + "=" * 78)
            print(f"  DIRECTION PROFILE — {name}, pooling '{key}', {p1.shape[1]} directions scored")
            print("=" * 78)
            print(f"  {'PC':>4}{'var share':>12}{'cum var':>10}{'info R²':>10}{'alignment':>12}")
            print("  " + "-" * 46)
            cum = np.cumsum(var_share)
            for k in range(min(cli.top, len(var_share))):
                print(f"  {k:>4}{var_share[k]:>12.4f}{cum[k]:>10.3f}{info[k]:>10.3f}{align[k]:>12.3f}")
            if len(var_share) > cli.top:
                tail = slice(cli.top, None)
                print(
                    f"  {'tail':>4}{var_share[tail].sum():>12.4f}{cum[-1]:>10.3f}"
                    f"{info[tail].mean():>10.3f}{align[tail].mean():>12.3f}   (mean over the rest)"
                )
            print(
                f"\n  spearman(variance, information) = {s['spearman_var_info']:+.3f}"
                "   (high = information lives where the variance is)"
            )
            print(
                f"  surplus directions (hi var, lo info): n={s['n_surplus']}, "
                f"{100 * s['surplus_var']:.0f}% of variance, mean alignment {s['surplus_align']:.3f}"
            )
            print(
                f"  productive (hi var, hi info):         n={int((var_share >= np.median(var_share)).sum()) - s['n_surplus']}, "
                f"{100 * s['productive_var']:.0f}% of variance, mean alignment {s['productive_align']:.3f}"
            )
            rows.append({"model": name, "pooling": key, **s})
        del model

    if rows:
        print("\n" + "=" * 78)
        print("  SUMMARY")
        print("=" * 78)
        hdr = f"  {'model':<16}{'pool':<7}{'spearman':>10}{'surplus align':>15}{'productive align':>18}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            print(
                f"  {r['model']:<16}{r['pooling']:<7}{r['spearman_var_info']:>+10.3f}"
                f"{r['surplus_align']:>15.3f}{r['productive_align']:>18.3f}"
            )
        con = [r for r in rows if "base" not in r["model"].lower()]
        if con:
            c = con[0]
            print()
            if not np.isfinite(c["surplus_align"]):
                print("  => no surplus directions isolated; read the per-direction table directly.")
            elif c["surplus_align"] >= 0.8 * c["productive_align"]:
                print("  => (A) ALIGNMENT SUSTAINS THE SURPLUS. The high-variance/low-information")
                print("     directions are as well aligned as the productive ones, so on_diag is")
                print("     rewarding them. The lever is the alignment term, not bt_lambda.")
            elif c["surplus_align"] < 0.5 * c["productive_align"]:
                print("  => (B) THE REDUNDANCY TERM CREATED THEM. The surplus directions are poorly")
                print("     aligned — nothing in the alignment term wants them. They exist because")
                print("     decorrelation requires d directions and only ~9 carry factors.")
                print("     The lever is bt_lambda / bt_gap_lambda, or a narrower content block.")
            else:
                print("  => MIXED. Surplus directions are partially aligned; both terms contribute.")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "direction_diffusion.csv")
    cols = [
        "model",
        "pooling",
        "spearman_var_info",
        "n_surplus",
        "surplus_align",
        "surplus_var",
        "productive_align",
        "productive_var",
        "mean_align",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
