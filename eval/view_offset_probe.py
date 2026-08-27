#!/usr/bin/env python
"""Is the content block's view leak a per-view CONSTANT offset — the one thing BT cannot see?

``content_view`` reads 1.0000 on this project: a probe identifies which modality a content
vector came from, perfectly, at every checkpoint and in both arms.  That is the most basic
thing a "content" block is supposed not to do.

This asks whether the leak is the specific transform Barlow Twins is **structurally blind
    logger.info("Wrote %s", path)
    run_view_offset_probe(cli)
    ...
    from eval.view_leak_shared import offset_size, per_view_center, per_view_standardise, run_view_offset_probe, view_acc
    ...
    p = argparse.ArgumentParser(description="Is the content view-leak a per-view constant offset?")
    ...
    if __name__ == "__main__":
        main()
per-view *constant* mean/scale shift costs it exactly nothing — verified numerically:

    view-2 features                    BT on_diag       MSE   view-probe acc
    baseline                               0.3007     0.179            0.503
    + per-view CONSTANT gain/bias          0.3007     18.821           1.000   <- invisible
    + per-SAMPLE gain                      1.2931     0.428            0.501   <- BT sees this

A constant offset is free under BT and makes view identity perfectly decodable, while an MSE
term penalises it 105x.  Note row three: BT is *not* blind to per-sample gain, which is what
``z_style[0]`` actually is.  The blind spot is narrower than "affine" — it is specifically
the component that is constant across samples within a view.

The test
--------
Measure view-predictability, changing only what is removed BEFORE the probe:

    raw            as the model emits them        (reproduces run_dci_compare's content_view)
    centered       minus each view's channel MEANS
    standardised   minus its means, over its own channel STDs — with BOTH a linear and a
                   nonlinear probe, because a linear probe is blind here by construction

    MLP drops to ~0.5  =>  no detectable non-affine difference remains. The leak was the
                           constant component, which BT cannot express at any bt_lambda, and
                           which an MSE term targets directly.
    MLP stays high     =>  the views differ in shape, not just location/scale. BT's blind
                           spot is not the explanation and an MSE term will not fix it.

Two measurement traps this had to fix, both found by planted data
-----------------------------------------------------------------
1. **Paired rows must stay in the same CV fold.**  Each sample contributes a row under
   *both* labels, so an ungrouped split lets the probe recognise the sample and infer the
   other label — which drives accuracy BELOW chance when there is no real signal (0.430 vs
   0.494 grouped, measured).  ``eval.identifiability_metrics.view_invariance`` splits
   ungrouped, so ``content_view``'s effective chance level is not exactly 0.5.  It does not
   change the 1.0000 reading on real data, but it matters for any near-chance number.
2. **A linear probe cannot be the test.**  Per-view standardisation removes the first two
   moments exactly, and a linear probe reads only those — so it returns ~chance for ANY
   difference, affine or not.  The standardised condition must use a nonlinear probe.

Validated on planted features (1500 samples, 44 dims, grouped CV):

    planted view-2 difference          raw lin   std lin   std MLP
    none                                 0.494     0.477     0.498
    AFFINE constant offset + gain        1.000     0.477     0.497   <- removed, as intended
    cubed (heavy tail)                   0.560     0.469     0.787   <- survives: control works
    abs()                                0.999     0.435     0.484   <- NOT detected
    bimodal  base + 2*sign(base)         0.543     0.470     0.504   <- NOT detected

**Scope limit, from those last two rows:** the probe reads per-sample *marginals*, so it
detects shape differences like heavy tails but misses sign flips and symmetric folds, which
barely move the marginal.  So an MLP at chance means "no *detectable* non-affine difference",
not "provably affine".  Read a drop as strong support, not proof.

Usage
-----
    python -m eval.view_offset_probe --run-dir results/synthetic/RUN --checkpoint-name vqvae_model.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def per_view_center(a, b):
    return a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)


def per_view_standardise(a, b):
    a, b = per_view_center(a, b)
    return a / (a.std(0, keepdims=True) + 1e-8), b / (b.std(0, keepdims=True) + 1e-8)


def view_acc(a, b, kind="lin", n_splits=3, seed=0):
    """Accuracy of a probe predicting which view a row came from.

    Grouped by SAMPLE: a sample's view-1 and view-2 rows always land in the same fold, so
    the probe cannot recognise the sample and infer the label by elimination.
    """
    X = np.vstack([a, b]).astype(np.float64)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    groups = np.r_[np.arange(len(a)), np.arange(len(b))]
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        clf = (
            LogisticRegression(max_iter=2000)
            if kind == "lin"
            else MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)
        )
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
    return float(np.mean(accs))


def offset_size(a, b):
    """How big is the per-view constant shift, in units of the pooled spread?"""
    pooled = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-12
    dmean = np.abs(a.mean(0) - b.mean(0)) / pooled
    ratio = (a.std(0) + 1e-12) / (b.std(0) + 1e-12)
    return {
        "dmean_mean": float(dmean.mean()),
        "dmean_max": float(dmean.max()),
        "std_ratio_med": float(np.median(ratio)),
    }


def main():
    p = argparse.ArgumentParser(description="Is the content view-leak a per-view constant offset?")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--poolings", default="stats,gap", help="stats matches run_dci_compare's content_view.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="view_offset_out")
    cli = p.parse_args()

    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2, _STYLE, _STYLE_V2, parse_poolings
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    rows = []
    for key, value in parse_poolings(cli.poolings):
        ld, _gc, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.batch_size, cli.num_workers, pooling=value
        )
        if cli.level not in ld:
            continue
        for bname, idx in (("content", (_CONTENT, _CONTENT_V2)), ("style", (_STYLE, _STYLE_V2))):
            b1, b2 = ld[cli.level][idx[0]], ld[cli.level][idx[1]]
            if b1 is None or b2 is None or b1.shape[1] == 0:
                continue
            s1, s2 = per_view_standardise(b1, b2)
            c1, c2 = per_view_center(b1, b2)
            rows.append(
                {
                    "pooling": key,
                    "block": bname,
                    "raw_lin": view_acc(b1, b2, "lin"),
                    "cent_lin": view_acc(c1, c2, "lin"),
                    "std_lin": view_acc(s1, s2, "lin"),
                    "std_mlp": view_acc(s1, s2, "mlp"),
                    **offset_size(b1, b2),
                }
            )
    del model

    print("\n" + "=" * 96)
    print("  CAN A PROBE TELL WHICH VIEW A FEATURE VECTOR CAME FROM?   0.5 = view-invariant")
    print("  grouped CV (a sample's two rows share a fold); 'raw_lin' at stats ~ content_view")
    print("=" * 96)
    hdr = (
        f"  {'pooling':<9}{'block':<9}{'raw_lin':>9}{'cent_lin':>10}{'std_lin':>9}"
        f"{'std_MLP':>9}{'|dmean|/sd':>12}{'max':>8}{'sd ratio':>10}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['pooling']:<9}{r['block']:<9}{r['raw_lin']:>9.3f}{r['cent_lin']:>10.3f}"
            f"{r['std_lin']:>9.3f}{r['std_mlp']:>9.3f}{r['dmean_mean']:>12.3f}"
            f"{r['dmean_max']:>8.2f}{r['std_ratio_med']:>10.2f}"
        )
    print("\n  std_lin is uninformative BY CONSTRUCTION (standardising removes exactly what a")
    print("  linear probe reads). std_MLP is the number that decides it.")

    con = [r for r in rows if r["block"] == "content"]
    if con:
        c = con[0]
        print("\n" + "=" * 96)
        print(f"  VERDICT   content  raw {c['raw_lin']:.3f} -> standardised+MLP {c['std_mlp']:.3f}")
        if c["raw_lin"] - c["std_mlp"] > 0.25 and c["std_mlp"] < 0.65:
            print("  => NO DETECTABLE NON-AFFINE DIFFERENCE REMAINS. The leak is the per-view constant")
            print("     component — invisible to BT's alignment term at any bt_lambda, and penalised")
            print("     directly by an MSE term. Prediction: --bt-sim-coeff drives content_view -> 0.5.")
            print("     (Caveat: the probe reads marginals, so sign flips / symmetric folds escape it.)")
        elif c["std_mlp"] > 0.8:
            print("  => NOT AFFINE. The views stay separable after removing per-view means and scales,")
            print("     so BT's blind spot does not explain this leak and an MSE term will not fix it.")
        else:
            print("  => PARTIAL. Some of the leak is affine and some is not; an MSE term should help")
            print("     but will not close it.")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "view_offset.csv")
    cols = ["pooling", "block", "raw_lin", "cent_lin", "std_lin", "std_mlp", "dmean_mean", "dmean_max", "std_ratio_med"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
