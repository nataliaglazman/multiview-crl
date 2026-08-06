#!/usr/bin/env python
"""Is the content block's view leak a few contaminated channels, or the whole block?

``content_view`` is 1.000 at stats pooling and 0.839 at GAP — the content block identifies
the modality about as well as the *style* block does (0.990 / 0.842).  Those numbers say the
leak is large; they do not say whether it is *localised*.

    a few channels   ->  a targeted penalty (or simply excluding them) could fix it, and the
                         rest of the block is genuinely view-invariant
    all channels     ->  view-invariance is not being learned at all, and no channel-level
                         intervention will help

The distinction decides whether the next intervention is surgical or structural, so it is
worth 15 lines before spending a training run.

Protocol
--------
1. Rank every content channel by how well it ALONE predicts view identity (AUC, folded to
   ``max(auc, 1-auc)`` so the direction is irrelevant).
2. Drop the top-``k`` leakers and re-measure view-accuracy on what remains, with the same
   grouped-CV MLP protocol as ``eval/view_offset_probe.py``.
3. At each ``k``, also measure what the drop COSTS: mean factor R² from the surviving
   channels.  A fix that removes the leak by deleting the content is not a fix.

Read the two curves together:

    view-acc collapses at small k, factor R² holds   ->  localised and separable. Act on it.
    view-acc holds as k grows                        ->  distributed. The block is not
                                                         view-invariant anywhere.
    both fall together                               ->  view info and content share the
                                                         same channels; they cannot be
                                                         separated by channel selection.

Validated on planted features (1200 samples, 44 dims, grouped CV, MLP probe):

    planted leak                     k=0     k=1     k=2     k=4     k=8
    3 contaminated channels        1.000   1.000   0.995   0.508   0.501   <- collapses at k=4
    spread over all 44 channels    1.000   1.000   1.000   1.000   1.000   <- never collapses

The two regimes separate cleanly, so a real result can be read off the same curve.

Usage
-----
    python -m eval.view_leak_channels --run-dir results/synthetic/RUN --poolings gap,stats
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from eval.identifiability_metrics import cv_probe_r2
from eval.view_offset_probe import view_acc

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DROP_KS = (0, 1, 2, 4, 8, 16, 24)


def per_channel_view_auc(a, b):
    """AUC of each single channel predicting view identity, folded so 0.5 = uninformative."""
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    out = np.empty(a.shape[1])
    for c in range(a.shape[1]):
        s = np.r_[a[:, c], b[:, c]]
        if np.allclose(s, s[0]):
            out[c] = 0.5
            continue
        auc = roc_auc_score(y, s)
        out[c] = max(auc, 1.0 - auc)
    return out


def main():
    p = argparse.ArgumentParser(description="Is the view leak localised to a few content channels?")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--poolings", default="gap", help="gap is where the objective acts; stats is where content_view is.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="view_leak_out")
    cli = p.parse_args()

    seeds = tuple(int(s) for s in cli.seeds.split(","))

    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2, parse_poolings
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    rows = []
    for key, value in parse_poolings(cli.poolings):
        ld, gt_content, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.batch_size, cli.num_workers, pooling=value
        )
        if cli.level not in ld:
            continue
        b1, b2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
        if b1 is None or b2 is None or b1.shape[1] == 0:
            continue

        auc = per_channel_view_auc(b1, b2)
        order = np.argsort(-auc)  # worst leakers first
        d = b1.shape[1]

        print("\n" + "=" * 84)
        print(f"  PER-CHANNEL VIEW LEAK — pooling '{key}', {d} content features")
        print("=" * 84)
        print(f"  channels with AUC > 0.9 : {int((auc > 0.9).sum()):d} / {d}")
        print(f"  channels with AUC > 0.7 : {int((auc > 0.7).sum()):d} / {d}")
        print(f"  channels with AUC < 0.6 : {int((auc < 0.6).sum()):d} / {d}")
        print(f"  median AUC              : {np.median(auc):.3f}      max {auc.max():.3f}")
        top = ", ".join(f"{auc[i]:.2f}" for i in order[:10])
        print(f"  top-10 channel AUCs     : {top}")

        print(f"\n  {'drop top-k':>11}{'kept':>7}{'view acc (MLP)':>17}{'mean factor R²':>17}")
        print("  " + "-" * 52)
        for k in DROP_KS:
            if k >= d:
                continue
            keep = order[k:]
            va = view_acc(b1[:, keep], b2[:, keep], "mlp")
            r2 = cv_probe_r2(b1[:, keep], gt_content, seeds=seeds)["mean"]
            rows.append({"pooling": key, "k": k, "kept": len(keep), "view_acc": va, "factor_r2": r2})
            print(f"  {k:>11}{len(keep):>7}{va:>17.3f}{r2:>17.3f}")

        sub = [r for r in rows if r["pooling"] == key]
        first, last = sub[0], sub[-1]
        print()
        if last["view_acc"] > 0.8:
            print("  => DISTRIBUTED. Dropping the worst leakers does not restore view-invariance;")
            print("     every channel carries modality. No channel-level intervention will help —")
            print("     the objective is simply not producing a view-invariant block.")
        elif last["factor_r2"] > 0.8 * first["factor_r2"]:
            print("  => LOCALISED AND SEPARABLE. The leak sits in a few channels that carry little")
            print("     content: excluding them buys view-invariance almost for free.")
        else:
            print("  => ENTANGLED. View info and content live in the same channels — the leak falls")
            print("     only as fast as the content does, so channel selection cannot separate them.")

    del model
    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "view_leak_channels.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pooling", "k", "kept", "view_acc", "factor_r2"])
        for r in rows:
            w.writerow([r["pooling"], r["k"], r["kept"], r["view_acc"], r["factor_r2"]])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
