#!/usr/bin/env python
"""Do the selection metrics reward DISCARDING content?  Tested by destroying it on purpose.

``content_purity`` and ``separation`` measure what is *absent* from the content block;
``mcc_cc`` measures what is *present*.  A contrastive objective improves the first family
by construction, so a run where purity and separation keep climbing while ``mcc_cc`` turns
over is either (a) genuinely trading completeness for a cleaner split, or (b) emptying the
block into metrics that cannot tell emptiness from cleanliness.

Reading that off a training curve cannot separate the two: everything moves at once and the
objective is a confound.  So this script removes the objective from the question entirely.
It takes ONE checkpoint, degrades its content block by a controlled amount, and rescores.
No training, no style dynamics, no view-invariance — just information destruction.

    metric rises as content is destroyed   ->  it is rewarding discarding, and its climb in
                                               your run is not evidence of anything good
    metric falls with the content          ->  it is measuring what it claims to

The suspicion is concrete and comes from the formulas (``eval/run_dci_compare.py``):

    separation     = mcc_cc - mcc_cs                 a DIFFERENCE: an empty block sends both
                                                     terms to their null floors, so this one
                                                     should be emptiness-protected
    purity_info    = clip01(1 - leak_c2s / info_c2c) a RATIO: climbs whenever the leak shrinks
                                                     faster than the signal, however small the
                                                     signal gets
    purity_view    = 1 when content predicts view at chance
                                                     a constant block predicts nothing, view
                                                     included, and scores a perfect 1

Validated on planted data (12 content channels faithfully encoding 3 factors, plus a
deliberate style leak in the low-variance directions; PCA mode, fraction of rank kept):

    kept    mcc_cc   separation   content_purity   overall
    1.00     0.999        0.010            0.500     0.750
    0.25     0.991        0.801            0.979     0.867     <- rank gutted, mcc intact
    0.08     0.465        0.331            0.964     0.722     <- half the content GONE

Two things that table settles.  ``content_purity`` and ``separation`` both **peak at a
degraded representation**, and purity is still near its maximum after half the content is
destroyed — so a climbing purity curve is not by itself evidence of anything good.  And the
guess that ``separation`` would be emptiness-protected is **wrong in the useful range**: it
does eventually turn over, but only after peaking, and at 92% rank destruction it is still
33x its undegraded value.

The mechanism the planted case exposes: degradation removes the *weakest* directions first,
so whenever the contamination is weaker than the signal, exclusion metrics improve while
completeness holds.  Whether a given checkpoint has that structure is exactly what running
this on it answers.

Degradation modes
-----------------
``pca``   Project the content block onto its top ``k`` principal components and reconstruct
          in the original space.  Rank is reduced; the feature count is NOT, so probe
          capacity is held fixed and rank is isolated.  This is the mode that mirrors an
          observed effective-rank decline.
``noise`` Add isotropic per-channel noise at a multiple of each channel's own std.  Generic
          information loss, not rank-specific — a useful control: a metric that only
          misbehaves under ``pca`` is responding to rank, one that misbehaves under both is
          responding to information.

Only the CONTENT block is touched (tuple positions 0 and 2).  Style is left exactly as the
encoder produced it, so ``style_modality`` and ``style_purity`` are constant by construction
and any movement in ``overall`` comes from the content side alone.

Usage
-----
    python -m eval.metric_degeneracy_sweep \
        --run-dir results/synthetic/RUN --checkpoint-name vqvae_model.pt --causal match

Validity check built in: the first row is the UNDEGRADED block, so its ``mcc_cc`` /
``separation`` / ``content_purity`` must match what ``run_dci_compare`` reports for the same
checkpoint.  If they do not, the extraction path here has diverged and nothing below is
trustworthy.  Seeds and null count default lower than ``run_dci_compare`` because the
deliverable is the TREND across rows, not the absolute values in any one of them.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.decomposition import PCA

from eval.run_dci_compare import (
    _CONTENT,
    _CONTENT_V2,
    _leak_rejection,
    _view_invariance_score,
    derive_scores,
    parse_poolings,
    score_reprs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PCA_LEVELS = (1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08)
NOISE_LEVELS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def pca_projector(X, level, seed=0):
    """Fit the rank-k projector for one block.  ``None`` when the level is a no-op.

    Fit ONCE and shared across both views by the caller.  Fitting a separate basis per view
    would rotate them differently and manufacture a view-distinguishing signal out of thin
    air — caught on planted data, where per-view bases sent ``purity_view`` 1.000 -> 0.393
    under degradation that should have left it alone.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        return None
    max_k = min(X.shape[1], X.shape[0] - 1)
    k = max(1, int(round(level * max_k)))
    if k >= max_k:
        return None
    mu = X.mean(0, keepdims=True)
    return PCA(n_components=k, random_state=seed).fit(X - mu), mu


def apply_pca(proj, X):
    """Rank-k reconstruction IN THE ORIGINAL basis: D features preserved, only rank moves."""
    p, mu = proj
    X = np.asarray(X, dtype=np.float64)
    return p.inverse_transform(p.transform(X - mu)) + mu


def add_noise(X, level, seed=0):
    """Isotropic per-channel noise at ``level`` x each channel's own std."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0 or level <= 0:
        return X
    rng = np.random.RandomState(seed)
    return X + rng.randn(*X.shape) * (X.std(0, keepdims=True) * level)


def degrade_reprs(reprs, mode, level, seed=0):
    """Copy ``reprs`` with the content block degraded at every pooling; style untouched."""
    out = {}
    for key, level_data in reprs.items():
        new_ld = {}
        for lvl, ld in level_data.items():
            ld = list(ld)
            if mode == "pca":
                # One basis, fit on view 1, applied to both — see pca_projector.
                proj = pca_projector(ld[_CONTENT], level, seed) if ld[_CONTENT] is not None else None
                if proj is not None:
                    for ci in (_CONTENT, _CONTENT_V2):
                        if ld[ci] is not None:
                            ld[ci] = apply_pca(proj, ld[ci])
            else:
                # Independent draws per view: that is ordinary measurement noise and, unlike a
                # per-view rotation, adds no systematic difference between the two.
                for ci in (_CONTENT, _CONTENT_V2):
                    if ld[ci] is not None:
                        ld[ci] = add_noise(ld[ci], level, seed + ci)
            new_ld[lvl] = tuple(ld)
        out[key] = new_ld
    return out


def score_one(reprs, gt_content, gt_style, gt_style_v2, info, level_idx, n_null, seeds, n_jobs, probe_kind):
    """One degradation level -> the row plus its derived scores and purity components."""
    row = score_reprs(
        reprs,
        gt_content,
        gt_style,
        info,
        level_idx,
        n_null=n_null,
        seeds=seeds,
        n_jobs=n_jobs,
        gt_style_v2=gt_style_v2,
        probe_kind=probe_kind,
    )
    row.update(derive_scores(row, ""))
    # derive_scores only returns the combined content_purity; recompute the two halves with
    # the module's own helpers so the split is exact rather than re-derived here.
    row["purity_info"] = _leak_rejection(row.get("leak_c2s"), row.get("info_c2c"))
    row["purity_view"] = _view_invariance_score(row.get("content_view"))
    return row


def _f(v, nd=3):
    return "  -  " if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _trend(rows, key):
    """(clean value, value at the most-degraded level, index of the peak)."""
    vals = [r.get(key, float("nan")) for r in rows]
    finite = [v if np.isfinite(v) else -np.inf for v in vals]
    return vals[0], vals[-1], int(np.argmax(finite))


def main():
    p = argparse.ArgumentParser(description="Do the selection metrics reward discarding content?")
    p.add_argument("--run-dir", required=True, help="Run directory (settings.json inside).")
    p.add_argument("--checkpoint-name", default="vqvae_model.pt", help="Checkpoint filename.")
    p.add_argument("--mode", choices=("pca", "noise", "both"), default="both", help="Degradation mode.")
    p.add_argument("--levels", default=None, help="Comma list overriding the default sweep for --mode.")
    p.add_argument("--num-samples", type=int, default=2000, help="Frozen test-set size.")
    p.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="match",
        help="Factor distribution of the test set, for runs trained with --synthetic-causal.",
    )
    p.add_argument("--poolings", default="gap,stats,2x2x2", help="Comma list: gap, stats, and/or DxHxW.")
    p.add_argument("--level", type=int, default=0, help="Encoder level.")
    p.add_argument("--seeds", default="0,1", help="Probe CV seeds (lowered by default: trends, not absolutes).")
    p.add_argument("--n-null", type=int, default=2, help="Permutations for the null floor.")
    p.add_argument("--probe-kind", default="ridge", choices=("ridge", "kernel", "mlp"))
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="Seed for the degradation itself.")
    p.add_argument("--out", default="metric_degeneracy_out", help="Output directory.")
    cli = p.parse_args()

    seeds = tuple(int(s) for s in cli.seeds.split(","))
    poolings = parse_poolings(cli.poolings)

    import torch

    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref_args = load_run_args(cli.run_dir)
    if cli.causal == "match" and not getattr(ref_args, "synthetic_causal", False):
        logger.info("Run was not trained with --synthetic-causal; 'match' and 'iid' coincide here.")
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")

    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    if not os.path.exists(ckpt):
        logger.warning("%s not found — falling back to the loader's default", ckpt)
        ckpt = None
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt, None)

    reprs, gt_content, gt_style, gt_style_v2, info = {}, None, None, None, None
    for key, value in poolings:
        level_data, gc, gsv1, gsv2 = _extract_synthetic_representations(
            model, dataset, device, cli.batch_size, cli.num_workers, pooling=value
        )
        reprs[key] = level_data
        if gt_content is None:
            gt_content, gt_style, gt_style_v2 = gc, gsv1, gsv2
        if info is None and cli.level in level_data:
            info = level_data[cli.level][4]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if info is None:
        raise RuntimeError(f"level {cli.level} not present in encoder outputs")

    modes = ("pca", "noise") if cli.mode == "both" else (cli.mode,)
    all_rows = []
    for mode in modes:
        if cli.levels:
            levels = tuple(float(x) for x in cli.levels.split(","))
        else:
            levels = PCA_LEVELS if mode == "pca" else NOISE_LEVELS

        rows = []
        for lv in levels:
            logger.info("scoring %s level %.3f ...", mode, lv)
            d = degrade_reprs(reprs, mode, lv, cli.seed) if lv != levels[0] else reprs
            r = score_one(
                d,
                gt_content,
                gt_style,
                gt_style_v2,
                info,
                cli.level,
                cli.n_null,
                seeds,
                cli.n_jobs,
                cli.probe_kind,
            )
            r["mode"], r["degradation"] = mode, lv
            rows.append(r)
        all_rows.extend(rows)

        label = "PCA rank kept (fraction)" if mode == "pca" else "noise added (x channel std)"
        print("\n" + "=" * 104)
        print(f"  DEGRADING THE CONTENT BLOCK — {label}")
        print("  first row is UNDEGRADED: cross-check it against run_dci_compare for this checkpoint")
        print("=" * 104)
        hdr = (
            f"  {'level':>7}{'rank':>7}{'mcc_cc':>9}{'mcc_cs':>9}{'separat':>9}{'leak_c2s':>10}"
            f"{'info_c2c':>10}{'pur_info':>10}{'pur_view':>10}{'purity':>9}{'anatomy':>9}{'overall':>9}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            print(
                f"  {r['degradation']:>7.2f}{_f(r.get('content_rank'), 1):>7}{_f(r.get('mcc_cc')):>9}"
                f"{_f(r.get('mcc_cs')):>9}{_f(r.get('separation')):>9}{_f(r.get('leak_c2s')):>10}"
                f"{_f(r.get('info_c2c')):>10}{_f(r.get('purity_info')):>10}{_f(r.get('purity_view')):>10}"
                f"{_f(r.get('content_purity')):>9}{_f(r.get('content_anatomy')):>9}"
                f"{_f(r.get('disentanglement')):>9}"
            )

        print(f"\n  {'metric':<18}{'clean':>9}{'worst':>9}{'peaks at':>11}   verdict")
        print("  " + "-" * 74)
        for key, nice in (
            ("mcc_cc", "mcc_cc"),
            ("separation", "separation"),
            ("content_purity", "content_purity"),
            ("purity_info", "  purity_info"),
            ("purity_view", "  purity_view"),
            ("content_anatomy", "content_anatomy"),
            ("disentanglement", "overall_score"),
        ):
            clean, worst, peak_i = _trend(rows, key)
            peak_lv = rows[peak_i]["degradation"]
            if not np.isfinite(clean) or not np.isfinite(worst):
                verdict = "not available"
            elif peak_i > 0:
                verdict = f"REWARDS DISCARDING (peaks after {rows[peak_i - 1]['degradation']:.2f})"
            elif worst < clean:
                verdict = "falls with the content — measures what it claims"
            else:
                verdict = "flat"
            print(f"  {nice:<18}{_f(clean):>9}{_f(worst):>9}{peak_lv:>11.2f}   {verdict}")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "metric_degeneracy.csv")
    cols = [
        "mode",
        "degradation",
        "content_rank",
        "mcc_cc",
        "mcc_cs",
        "separation",
        "leak_c2s",
        "info_c2c",
        "content_view",
        "purity_info",
        "purity_view",
        "content_purity",
        "content_anatomy",
        "style_modality",
        "style_purity",
        "disentanglement",
        "info_all",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in all_rows:
            w.writerow([r.get(c) for c in cols])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
