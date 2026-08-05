#!/usr/bin/env python
"""Does the objective distort the *correlation structure* of the recovered factors?

Barlow Twins' redundancy-reduction term drives the off-diagonal of the representation's
cross-correlation toward zero.  That is the right prior when the generative factors are
independent.  Under ``--synthetic-causal`` they are **not**: the SCM makes them correlated
by construction, so a faithful encoding *must* carry a non-zero off-diagonal, and the
objective penalises exactly that.  This measures whether the penalty is winning.

Protocol
--------
For each (run, checkpoint): extract the GAP-pooled content block, fit the repo's ridge
probe to every content factor, and keep the **out-of-fold** predictions ``f_hat``.  Then
compare two correlation matrices over the same samples:

    R_true = corr(f_true)      what the SCM actually generated
    R_hat  = corr(f_hat)       what a reader of the representation would infer

and regress the off-diagonal of ``R_hat`` on the off-diagonal of ``R_true`` through the
origin.  The slope is the headline:

    slope < 1   correlations SHRUNK  -> the decorrelation term is winning against the data
    slope ~ 1   correlation structure faithfully represented
    slope > 1   correlations INFLATED -> factors forced through a low-rank bottleneck

Those two failure directions are opposite, so one number separates the two competing
explanations for a falling GAP recovery: an over-strong decorrelation penalty, versus
dimensional collapse.

The attenuation correction is what makes it decisive
----------------------------------------------------
An imperfect probe shrinks correlations on its own.  With independent probe errors,
``corr(f_hat_k, f_hat_l) ~= corr(f_k, f_l) * sqrt(rho_k * rho_l)`` where the reliability
``rho_k = corr(f_hat_k, f_k)^2``, so a raw slope below 1 is expected whenever the probe is
imperfect — and since accuracy itself falls over training, the raw slope would fall even if
nothing were being distorted.  Dividing each entry by ``sqrt(rho_k * rho_l)`` removes that,
and the residual is the part the probe cannot explain.

Note ``rho`` is the squared prediction-truth *correlation*, not R².  They coincide for a
calibrated prediction, but only ``corr^2`` is scale-invariant — and on a planted
faithful-but-noisy probe, using R² over-corrects the slope to ~1.75 instead of ~1.0.
Validated on planted data: faithful probes recover 0.99/1.02/1.00 at noise 0.8x/1.5x and a
3x rescale, a whitened (decorrelated) estimate gives -3.8, a rank-2 bottleneck gives +3.0.

The correction also sharpens the diagnosis, because its own assumption fails informatively:
probe errors are independent only if the representation is rich.  Under a low-rank
bottleneck the errors share structure, which pushes the corrected slope *above* 1.  So

    corrected slope < 1   decorrelation pressure beyond probe noise
    corrected slope > 1   shared error structure, i.e. a rank bottleneck

Usage
-----
    python -m eval.factor_correlation_fidelity \
        --run-dirs results/synthetic/RUN results/synthetic/RUN \
        --checkpoint-names vqvae_model.pt vqvae_model_late.pt \
        --names 17k 44k --causal match

``--causal match`` is the default and is required for the question to mean anything: under
i.i.d. factors ``R_true`` is the identity and there is no structure to preserve or destroy.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from eval.dci import _extract_synthetic_representations
from eval.run_dci_compare import _CONTENT, parse_poolings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Matches eval.identifiability_metrics._make_multi_regressor so these numbers sit next to
# the R² the other scripts report rather than describing a different probe.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def oof_predictions(X, Y, n_splits=5, seed=0):
    """Out-of-fold ridge predictions of every column of ``Y`` from ``X``.

    Out-of-fold rather than in-sample: an in-sample fit inherits the training targets'
    correlation structure directly, which is the very thing being measured.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    pred = np.empty_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        reg = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)
        reg.fit(sc.transform(X[tr]), Y[tr])
        pred[te] = np.asarray(reg.predict(sc.transform(X[te]))).reshape(len(te), Y.shape[1])
    return pred


def _reliability(Y, P):
    """Per-factor reliability: the squared correlation between prediction and truth.

    This, not R², is the quantity that attenuates a correlation — for ``f_hat = f + noise``
    the shrinkage factor is ``sqrt(rho_k * rho_l)`` with ``rho_k = corr(f_hat_k, f_k)^2``.
    The two coincide for a calibrated prediction (so this matches the per-factor R² the
    other scripts report), but ``corr^2`` is also scale-invariant, which is what we want
    here: a factor recovered up to an arbitrary scale is fully recovered as far as the
    correlation structure is concerned.  Verified against a planted faithful-but-noisy
    probe, where using R² instead over-corrects the slope to ~1.75 rather than ~1.0.
    """
    rho = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        sy, sp = Y[:, k].std(), P[:, k].std()
        rho[k] = 0.0 if sy < 1e-12 or sp < 1e-12 else np.corrcoef(Y[:, k], P[:, k])[0, 1] ** 2
    return np.clip(np.nan_to_num(rho, nan=0.0), 0.0, 1.0)


def _offdiag(M):
    """(correlation matrix, its upper-triangle off-diagonal as a flat vector)."""
    R = np.corrcoef(np.asarray(M, dtype=np.float64), rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    iu = np.triu_indices(R.shape[0], k=1)
    return R, R[iu]


def _slope(r_hat, r_true):
    """Least-squares slope through the origin of r_hat on r_true."""
    denom = float((r_true**2).sum())
    return float((r_hat * r_true).sum() / denom) if denom > 1e-12 else float("nan")


def fidelity_stats(content, gt_content, seeds, n_splits=5):
    """Correlation-fidelity statistics for one (run, checkpoint), averaged over seeds."""
    R_true, r_true = _offdiag(gt_content)
    iu = np.triu_indices(R_true.shape[0], k=1)
    per_seed = []
    r2_acc = []
    for seed in seeds:
        pred = oof_predictions(content, gt_content, n_splits=n_splits, seed=seed)
        r2 = _reliability(gt_content, pred)
        _, r_hat = _offdiag(pred)
        # Disattenuation: divide out the shrinkage a probe of this accuracy causes on its
        # own.  Pairs whose factors are essentially unrecovered carry no information about
        # the correlation structure, so they are dropped rather than divided by ~0.
        rel = np.sqrt(np.outer(r2, r2))[iu]
        keep = rel > 0.05
        r_corr = np.where(keep, r_hat / np.where(keep, rel, 1.0), 0.0)
        per_seed.append(
            {
                "slope_raw": _slope(r_hat, r_true),
                "slope_corrected": _slope(r_corr[keep], r_true[keep]),
                "rms_hat": float(np.sqrt((r_hat**2).mean())),
                "structure_r": float(np.corrcoef(r_hat, r_true)[0, 1]) if len(r_true) > 2 else float("nan"),
                "n_pairs_used": int(keep.sum()),
            }
        )
        r2_acc.append(r2)
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}
    out["n_pairs_used"] = int(round(out["n_pairs_used"]))
    out["slope_corrected_std"] = float(np.std([s["slope_corrected"] for s in per_seed]))
    out["rms_true"] = float(np.sqrt((r_true**2).mean()))
    out["r2_per_factor"] = np.mean(r2_acc, axis=0)
    return out


def channel_offdiag(c_v1, c_v2, batch_size, draws=32, seed=0):
    """RMS off-diagonal of the BT cross-correlation, at the TRAINING batch size.

    Reported against the sampling floor ``d(d-1)/B``: BT estimates a d x d matrix from B
    rows, so an encoder with genuinely independent channels still shows an off-diagonal of
    that size.  A measured value at the floor means the term is fitting noise; a value
    above it means there is real cross-channel structure left for it to remove.
    """
    if c_v1 is None or c_v2 is None:
        return None
    rng = np.random.RandomState(seed)
    n, d = c_v1.shape
    b = min(batch_size, n)
    vals = []
    for _ in range(draws):
        idx = rng.choice(n, size=b, replace=False)
        zi, zj = c_v1[idx].astype(np.float64), c_v2[idx].astype(np.float64)
        zi = (zi - zi.mean(0)) / (zi.std(0) + 1e-6)
        zj = (zj - zj.mean(0)) / (zj.std(0) + 1e-6)
        c = zi.T @ zj / b
        off = (c**2).sum() - (np.diag(c) ** 2).sum()
        vals.append(off)
    off_mean = float(np.mean(vals))
    return {
        "off_diag": off_mean,
        "floor": float(d * (d - 1) / b),
        "rms_offdiag": float(np.sqrt(off_mean / max(d * d - d, 1))),
        "diag_mean": float(np.mean(np.diag((zi.T @ zj) / b))),
        "d": d,
        "b": b,
    }


def analyse_one(name, run_dir, checkpoint_name, dataset, level, seeds, batch_size, num_workers, train_batch):
    """Load one checkpoint, extract the GAP content block, and score correlation fidelity."""
    from eval.run_dci_synthetic import load_model_from_run_dir

    logger.info("=== %s (%s / %s) ===", name, run_dir, checkpoint_name)
    ckpt = os.path.join(run_dir, checkpoint_name)
    if not os.path.exists(ckpt):
        logger.warning("  %s not found — falling back to the loader's default", ckpt)
        ckpt = None
    model, _args, device = load_model_from_run_dir(run_dir, ckpt, None)

    _key, value = parse_poolings("gap")[0]
    level_data, gt_content, _a, _b = _extract_synthetic_representations(
        model, dataset, device, batch_size, num_workers, pooling=value
    )
    del model
    if level not in level_data:
        logger.warning("  level %d absent — skipping", level)
        return None
    content = level_data[level][_CONTENT]
    content_v2 = level_data[level][2]
    info = level_data[level][4]
    if content is None or content.shape[1] == 0:
        logger.warning("  empty content block — skipping")
        return None

    row = fidelity_stats(content, gt_content, seeds)
    row["name"] = name
    row["names"] = list(info["content_names"])
    row["channels"] = channel_offdiag(content, content_v2, train_batch)
    return row


def main():
    p = argparse.ArgumentParser(
        description="Is the recovered factor correlation structure faithful, shrunk, or inflated?"
    )
    p.add_argument("--run-dirs", nargs="+", required=True, help="Run directories (repeat one to compare checkpoints).")
    p.add_argument(
        "--checkpoint-names",
        nargs="*",
        default=None,
        help="One per --run-dirs, or a single name reused for all. Default: vqvae_best.pt.",
    )
    p.add_argument("--names", nargs="*", default=None, help="Labels (default: run-dir basename + checkpoint).")
    p.add_argument("--num-samples", type=int, default=2000, help="Frozen test-set size, shared across all entries.")
    p.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="match",
        help="Factor distribution of the test set. Default 'match' — under 'iid' the true correlation "
        "matrix is the identity and this diagnostic has nothing to measure.",
    )
    p.add_argument("--level", type=int, default=0, help="Encoder level to analyse.")
    p.add_argument("--seeds", default="0,1,2", help="Probe CV seeds.")
    p.add_argument(
        "--train-batch", type=int, default=0, help="Training batch size for the BT floor (0 = read settings)."
    )
    p.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="factor_corr_out", help="Output directory.")
    cli = p.parse_args()

    ckpts = cli.checkpoint_names or ["vqvae_best.pt"]
    if len(ckpts) == 1:
        ckpts = ckpts * len(cli.run_dirs)
    if len(ckpts) != len(cli.run_dirs):
        p.error("--checkpoint-names must have one entry per --run-dirs (or exactly one)")
    names = cli.names or [f"{os.path.basename(os.path.normpath(d))}:{c}" for d, c in zip(cli.run_dirs, ckpts)]
    if len(names) != len(cli.run_dirs):
        p.error("--names must have one entry per --run-dirs")
    seeds = tuple(int(s) for s in cli.seeds.split(","))

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    ref_args = load_run_args(cli.run_dirs[0])
    if cli.causal == "match" and not getattr(ref_args, "synthetic_causal", False):
        logger.warning(
            "This run was NOT trained with --synthetic-causal, so the true factors are independent "
            "and the true correlation matrix is ~identity. The slope below has no signal to measure."
        )
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    train_batch = cli.train_batch or int(getattr(ref_args, "batch_size", 128) or 128)

    rows = []
    for name, run_dir, ckpt in zip(names, cli.run_dirs, ckpts):
        r = analyse_one(name, run_dir, ckpt, dataset, cli.level, seeds, cli.batch_size, cli.num_workers, train_batch)
        if r is not None:
            rows.append(r)
    if not rows:
        logger.error("Nothing to report.")
        return

    print("\n" + "=" * 78)
    print("  CORRELATION FIDELITY OF THE RECOVERED FACTORS  (GAP-pooled content block)")
    print("  slope <1 = correlations shrunk (decorrelation winning)")
    print("  slope >1 = correlations inflated (rank bottleneck)")
    print("=" * 78)
    print(f"  true RMS off-diagonal correlation of the factors: {rows[0]['rms_true']:.3f}")
    print(f"\n  {'run':<22}{'slope_raw':>11}{'slope_corr':>12}{'±':>7}{'rms_hat':>9}{'struct_r':>10}{'pairs':>7}")
    print("  " + "-" * 76)
    for r in rows:
        print(
            f"  {r['name']:<22}{r['slope_raw']:>11.3f}{r['slope_corrected']:>12.3f}"
            f"{r['slope_corrected_std']:>7.3f}{r['rms_hat']:>9.3f}{r['structure_r']:>10.3f}{r['n_pairs_used']:>7d}"
        )
    print("\n  slope_raw   uncorrected — falls with probe accuracy even when nothing is distorted")
    print("  slope_corr  divided by sqrt(rho_k*rho_l), rho = corr(pred, truth)^2; THIS is the one to read")
    print("  struct_r    correlation between the two off-diagonal patterns (shape, ignoring scale)")

    if rows[0]["channels"]:
        print("\n" + "=" * 78)
        print("  WHAT THE OBJECTIVE SEES  (BT cross-correlation at the training batch size)")
        print("=" * 78)
        print(f"  {'run':<22}{'d':>4}{'B':>6}{'off_diag':>11}{'floor':>9}{'% floor':>9}{'diag':>8}")
        print("  " + "-" * 69)
        for r in rows:
            c = r["channels"]
            print(
                f"  {r['name']:<22}{c['d']:>4}{c['b']:>6}{c['off_diag']:>11.2f}{c['floor']:>9.2f}"
                f"{100 * c['floor'] / max(c['off_diag'], 1e-9):>8.0f}%{c['diag_mean']:>8.3f}"
            )
        print("\n  off_diag at ~100% of floor => the term is fitting sampling noise, not structure.")

    print("\n" + "=" * 78)
    print("  PER-FACTOR RELIABILITY  rho = corr(out-of-fold prediction, truth)^2")
    print("  (equals the per-factor R² the other scripts report when the probe is calibrated)")
    print("=" * 78)
    print(f"  {'factor':<22}" + "".join(f"{r['name'][:14]:>16}" for r in rows))
    print("  " + "-" * (22 + 16 * len(rows)))
    for j, f in enumerate(rows[0]["names"]):
        print(f"  {f:<22}" + "".join(f"{r['r2_per_factor'][j]:>16.3f}" for r in rows))

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        d = b["slope_corrected"] - a["slope_corrected"]
        band = a["slope_corrected_std"] + b["slope_corrected_std"]
        print("\n" + "=" * 78)
        print(
            f"  VERDICT   corrected slope {a['name']} {a['slope_corrected']:.3f} -> "
            f"{b['name']} {b['slope_corrected']:.3f}  (delta {d:+.3f}, seed band ±{band:.3f})"
        )
        if abs(d) <= band:
            print("  => NO SIGNIFICANT CHANGE. The correlation structure is as faithfully represented")
            print("     at the end as at the start; look elsewhere for the falling GAP recovery.")
        elif d < 0 and b["slope_corrected"] < 1.0:
            print("  => CORRELATIONS ARE BEING SHRUNK beyond what probe noise explains. The")
            print("     decorrelation term is fighting the SCM's dependence structure. Levers:")
            print("     lower bt_gap_lambda, or drop the off-diagonal term at GAP entirely.")
        elif d > 0 and b["slope_corrected"] > 1.0:
            print("  => CORRELATIONS ARE BEING INFLATED: probe errors share structure, the signature")
            print("     of a rank bottleneck rather than decorrelation pressure.")
        else:
            print("  => Moved, but not into either failure regime. Read the per-factor R² and the")
            print("     channel table before concluding.")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "factor_correlation_fidelity.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "run",
                "slope_raw",
                "slope_corrected",
                "slope_corrected_std",
                "rms_hat",
                "rms_true",
                "structure_r",
                "n_pairs_used",
                "off_diag",
                "floor",
            ]
        )
        for r in rows:
            c = r["channels"] or {}
            w.writerow(
                [
                    r["name"],
                    r["slope_raw"],
                    r["slope_corrected"],
                    r["slope_corrected_std"],
                    r["rms_hat"],
                    r["rms_true"],
                    r["structure_r"],
                    r["n_pairs_used"],
                    c.get("off_diag"),
                    c.get("floor"),
                ]
            )
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
