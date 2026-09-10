#!/usr/bin/env python
"""One readable table: R², MCC and DCI for a run, each against its untrained floor.

``run_dci_compare`` is the full protocol — six sections, every scope, CSV for the paper.
This is the thing you actually read when the question is "did this checkpoint learn
anything identifiable, and where does it live".  Same numbers, one page.

Everything here is a thin layer over ``eval.run_dci_compare`` and
``eval.identifiability_metrics``: pooling parsing, FACTOR_POOLING routing, the
permutation nulls, the probe widths and the untrained floor all come from there by
import, never re-implemented.  That is deliberate — the metric rule living in two places
is what produced the cross-axis mistakes in this project's changelog.

What it prints
--------------
1. A per-factor table.  For every ground-truth factor: the R² gap at the pooling
   ``FACTOR_POOLING`` assigns it, the matched |corr| from block-MCC, and each one's
   distance from the untrained twin.  The FLOOR-SUBTRACTED column is the answer; the
   raw column is there so a saturated floor is visible rather than silent.  Under
   ``--causal match`` it also carries a PARTIAL column: the same score on each factor
   residualised on its SCM parents.  Without it, a factor with a well-recovered parent
   scores well without being encoded at all (measured: ``ventricle_size`` reads 0.695
   while ``brain_size`` is recovered at 0.92 and correlates ~0.8 with it), which is the
   confound a linear residual only partially addresses. The legacy fit uses all samples;
   use ``--parent-adjustment nonlinear`` for fold-local nonlinear adjustment instead.
1b. The same factors at EVERY rung — gap / stats / patch side by side, each with its own
   floor.  Table 1 answers "was this factor learned"; this one answers "where does it
   live", which is the axis the gap-vs-patch findings in the changelog turn on, and which
   previously took one run per ``--factor-pooling`` to see.  It is a locality profile,
   not a menu: the reportable number stays table 1's, at the assigned rung.
1c. Optional nonlinear parent-residual decoding from content and style, with full-target
   references using the same fold-local PCA/scaling. Nonlinear nuisance fits are trained
   only on outer training samples; training residuals use inner cross-fitting.
2. The MCC pooling ladder, gap → stats → patch, with floors.  A single rung is not
   interpretable: stats cannot express position (it is permutation-invariant over
   voxels) and patch sits on a floor of ~0.86, so only the SHAPE across rungs carries
   the "where does the information live" signal.
3. The LEAKAGE matrix: the other three cells of block×factor (content→style, style→style,
   style→content) plus a label-free view probe, each against its own floor.  Table 1 is
   the content→content cell alone, and a content score that rises because the content
   block absorbed style looks exactly like one that rises because content improved — this
   is the section that tells them apart.  Skip with ``--no-leakage``.
3b. Per-factor decoding from CONTENT and STYLE side by side, at every extracted pooling.
   Includes both content targets (e.g. ventricle_size) and style targets (e.g. bias).
   Raw cross-validated R² answers how well each factor is decoded; null-adjusted and
   untrained-floor-subtracted scores give its baselines. Reuses the already fitted probes.
4. DCI per scope, when ``--with-dci`` is passed, with the code count beside each row so
   two scopes are never silently compared at different normalisations.
5. A verdict block that refuses to report a metric whose learned part is inside its own
   noise, and says so instead of printing a number that looks like a result.

Why the floor is not optional here
----------------------------------
Measured on this generator (untrained encoder, 8³ patch, PCA-64, iid, N=200) six of nine
content factors read above R² 0.8 — brain_size 0.986, lr_asymmetry 0.974,
cortical_thickness 0.928, ventricle_size 0.915, sulcal_widening 0.866, temporal_atrophy
0.815 — and block-MCC at patch reads ~0.86.  An absolute number at patch pooling is
therefore almost entirely a statement about the architecture, not about training.  So
``--floor`` defaults ON in this script, unlike in ``run_dci_compare`` where it stays off
for backward-comparable CSVs.

Usage
-----
    python -m eval.identifiability_report --run-dir results/synthetic/RUN
    python -m eval.identifiability_report --run-dir RUN --with-dci --poolings gap,stats,4x4x4
    python -m eval.identifiability_report --run-dir RUN --no-floor       # not reportable
    python -m eval.identifiability_report --from-json report.json       # no model/probe rerun
    python -m eval.identifiability_report --run-dir RUN --parent-adjustment nonlinear
    python -m eval.identifiability_report --run-dir RUN --causal shuffled --shuffle-seed 0

``--causal shuffled`` re-renders independently permuted columns from matched content
draws, preserving each factor's empirical marginal. It changes their joint distribution
and is a complementary stress test, not an in-distribution replacement. See
``eval/CAUSAL_EVALUATION.md`` for the protocol and interpretation.

The scoring layer is torch-free and unit-testable: ``--self-test`` runs it on planted
numpy data with no checkpoint and no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os

import numpy as np
from joblib import Parallel, delayed

from eval.identifiability_metrics import (
    block_mcc,
    cv_probe_r2,
    n_parents_per_factor,
    parents_from_adjacency,
    residualise_on_parents,
    view_invariance,
)
from eval.run_dci_compare import (
    _CONTENT,
    _CONTENT_V2,
    _STYLE,
    _STYLE_V2,
    FACTOR_POOLING,
    PROBE_DIM_AUTO,
    _auto_probe_dim,
    _block_array,
    _has_v2,
    _resolve_key,
    mean_std_structs,
    parse_poolings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# A learned delta smaller than this is not distinguishable from probe noise at the sample
# sizes this script runs at, so the verdict block reports it as "not resolved" rather than
# as a number.  Calibrated from `patch_mcc_decay --calibrate`, where the smallest
# perturbation that reliably moved block-MCC was +0.0447 and the full training-time
# excursion on a real run was 0.04.
NOISE_FLOOR = 0.05


# --------------------------------------------------------------------------- #
# Scoring (pure numpy — no torch, no checkpoint; see --self-test)
# --------------------------------------------------------------------------- #


def per_factor_scores(
    reprs, level, gt, names, seeds, n_null, rng, kind="ridge", n_jobs=1, factor_pooling="assigned", block=_CONTENT
):
    """R² gap per factor at EVERY scored pooling, plus which one is the reportable rung.

    ``block`` selects which array of the level tuple is probed — ``_CONTENT`` for table 1,
    ``_STYLE`` for the leakage cells.  It is a parameter rather than a second function so
    that every cell of the block×factor matrix is scored by one implementation: the nulls,
    the rung ladder and the ``FACTOR_POOLING`` routing cannot drift between the diagonal
    and the off-diagonal, and a leak number is therefore on table 1's scale.

    Returns ``{name: {"r2", "r2_raw", "pooling", "by_pooling": {key: {...}}}}``.  The
    top-level ``r2``/``r2_raw`` are the assigned rung's — one factor, one pooling, fixed in
    advance by ``FACTOR_POOLING``, which is what keeps the headline from being a max over
    noisy pooling estimates.  ``by_pooling`` carries the same factor at every other rung
    too, so "where is this factor readable" is answerable without re-running the script
    once per ``--factor-pooling``; it is a locality profile, not a menu to pick from.

    The permutations are drawn up-front, in factor order, and SHARED across rungs: the
    rungs are then differenced against one null draw rather than three, and the ``rng`` is
    consumed exactly as it was when only the assigned rung was scored, so a floor stays
    reproducible against runs made before the ladder existed.  Nothing here depends on
    which worker finishes first, so ``n_jobs`` cannot move a number.
    """
    avail = set(reprs.keys())
    # Coarse -> fine, and only the rungs that were actually extracted.
    ladder = [k for k in ("gap", "stats", "patch") if k in avail]
    blocks = {}
    for key in ladder:
        X = _block_array(reprs, key, level, block)
        if X is not None and X.shape[1]:
            blocks[key] = X

    tasks, meta = [], []
    for j, name in enumerate(names):
        # ``factor_pooling`` overrides FACTOR_POOLING's per-factor routing and reports every
        # factor on one axis.  The routing is the honest default (each factor read where it
        # can physically appear, fixed in advance so the headline is not a max over
        # poolings); the override exists to ask the different question "how does this
        # factor read at THIS pooling", e.g. to compare all factors on one rung.
        want = FACTOR_POOLING.get(name, "stats") if factor_pooling == "assigned" else factor_pooling
        pkey = _resolve_key(want, avail)
        if pkey not in blocks:
            continue
        perms = [rng.permutation(gt.shape[0]) for _ in range(n_null)]
        for key, X in blocks.items():
            tasks.append((X, gt[:, j], perms))
            meta.append((name, pkey, key))

    def _one(X, y, perms):
        real = cv_probe_r2(X, y, seeds=seeds, kind=kind)["mean"]
        nulls = [cv_probe_r2(X, y[p], seeds=seeds, kind=kind)["mean"] for p in perms]
        return real, (float(np.mean(nulls)) if nulls else float("nan"))

    results = Parallel(n_jobs=n_jobs)(delayed(_one)(X, y, perms) for X, y, perms in tasks)

    out = {}
    for (name, pkey, key), (real, null) in zip(meta, results):
        d = out.setdefault(name, {"r2": float("nan"), "r2_raw": float("nan"), "pooling": pkey, "by_pooling": {}})
        d["by_pooling"][key] = {"r2": real - null, "r2_raw": real, "r2_null": null}
    for d in out.values():
        head = d["by_pooling"][d["pooling"]]
        d["r2"], d["r2_raw"] = head["r2"], head["r2_raw"]
    return out


def nonlinear_parent_scores(
    reprs,
    level,
    gt,
    names,
    adjacency,
    seeds,
    n_null,
    probe_dim=PROBE_DIM_AUTO,
    kind="ridge",
    n_jobs=1,
    factor_pooling="assigned",
    block=_CONTENT,
    parent_cache=None,
):
    """Decode nonlinear parent residuals, including fold-local feature preprocessing.

    Nuisance fits are shared across blocks, poolings and floor twins for the same
    labels/graph/splits. These scores have their own full-target reference because
    the legacy headline fits PCA before CV and is a different evaluation pipeline.
    """
    from eval.parent_adjusted import prepare_parent_folds, prepare_probe_folds, score_parent_folds

    cache = parent_cache if parent_cache is not None else {}
    key = (
        hashlib.sha256(np.asarray(gt, dtype=np.float64).tobytes()).hexdigest(),
        hashlib.sha256(np.asarray(adjacency, dtype=bool).tobytes()).hexdigest(),
        tuple(gt.shape),
        tuple(seeds),
    )
    if key not in cache:
        parents = parents_from_adjacency(adjacency)
        cache[key] = Parallel(n_jobs=n_jobs)(
            delayed(prepare_parent_folds)(
                gt[:, j], gt[:, [p for p in parents.get(j, []) if p < gt.shape[1]]], seeds=seeds
            )
            for j in range(len(names))
        )
    plans = cache[key]
    blocks = {}
    for pool in ("gap", "stats", "patch"):
        X = _block_array(reprs, pool, level, block)
        if X is not None and X.shape[1]:
            blocks[pool] = prepare_probe_folds(X, plans[0], probe_dim)

    tasks, meta = [], []
    if not blocks:
        return {}
    for j, name in enumerate(names):
        want = FACTOR_POOLING.get(name, "stats") if factor_pooling == "assigned" else factor_pooling
        assigned = _resolve_key(want, set(blocks))
        if assigned not in blocks:
            continue
        for pool, features in blocks.items():
            tasks.append(delayed(score_parent_folds)(features, plans[j], kind, n_null, null_seed=j))
            meta.append((name, assigned, pool))
    out = {}
    for (name, assigned, pool), score in zip(meta, Parallel(n_jobs=n_jobs)(tasks)):
        entry = out.setdefault(name, {"pooling": assigned, "by_pooling": {}})
        entry["by_pooling"][pool] = score
    for entry in out.values():
        entry.update(entry["by_pooling"][entry["pooling"]])
    return out


def mcc_ladder(reprs, level, gt_content, seeds, kind="ridge", names=()):
    """Block-MCC at every pooling + the per-factor matched |corr| at each.

    Kept per-pooling rather than collapsed to one headline because the rungs disagree in
    a way that matters: stats is permutation-invariant over voxels, so lesion positions
    are structurally unreadable there, while patch can express them but starts from a far
    higher floor.  The ladder's SHAPE is the readable object, not any single rung.
    """
    out = {}
    for key in sorted(reprs):
        X = _block_array(reprs, key, level, _CONTENT)
        if X is None or not X.shape[1]:
            continue
        res = block_mcc(X, gt_content, seeds=seeds, kind=kind)
        pf = res.get("per_factor")
        out[key] = {
            "mean": res["mean"],
            "std": res.get("std", float("nan")),
            "assignment_identity": res.get("assignment_identity", float("nan")),
            "per_factor": (
                {(names[j] if j < len(names) else f"factor{j}"): float(pf[j]) for j in range(len(pf))}
                if pf is not None
                else {}
            ),
        }
    return out


def _view_blocks(reprs, key, level):
    """(content_v1, content_v2, style_v1, style_v2) at one rung, with None widened to (N, 0).

    ``view_invariance`` stacks the two views of a block, so a missing style block has to
    arrive as a zero-width array rather than as None: ``cv_probe_acc`` already returns nan
    for a zero-width X, and that is the reading we want for a run with no style channels.
    """
    c1 = _block_array(reprs, key, level, _CONTENT)
    if c1 is None or not c1.shape[1]:
        return None
    c2 = _block_array(reprs, key, level, _CONTENT_V2)
    if c2 is None or c2.shape != c1.shape:
        return None
    n = c1.shape[0]

    def _widen(arr):
        return np.zeros((n, 0)) if arr is None else arr

    s1, s2 = _widen(_block_array(reprs, key, level, _STYLE)), _widen(_block_array(reprs, key, level, _STYLE_V2))
    if s1.shape[1] != s2.shape[1]:  # a half-present style block would stack into nonsense
        s1 = s2 = np.zeros((n, 0))
    return c1, c2, s1, s2


def leakage_scores(
    reprs,
    level,
    gt_content,
    gt_style,
    content_names,
    style_names,
    seeds,
    n_null,
    rng,
    kind="ridge",
    n_jobs=1,
    factor_pooling="assigned",
):
    """The off-diagonal cells of the block×factor matrix, plus the view probe.

    Table 1 is the content→content cell, and on its own it cannot distinguish the two
    things a rising content score can mean.  This architecture is specifically exposed to
    the second: reconstruction needs the view-specific appearance from somewhere, and when
    the style block cannot supply it the encoder routes it through content instead — which
    reads as MORE content information, not less.  Three more cells settle which happened,
    all scored by the same ``per_factor_scores`` as table 1 (same nulls, same rung ladder,
    same routing), so the four numbers sit on one scale:

      content→style   style factors read from the content block.  THE leak.  ~0 is clean.
      style→style     style doing its own job.  ~0 means the style block is DEAD — a
                      different diagnosis from a greedy content block, with a different
                      fix, so read this row before reading the one above it.
      style→content   anatomy in the style block.  Under an SCM the content factors are
                      correlated and style sees the same image, so this sits above zero
                      even when nothing is wrong.  Only its floor-subtracted column is
                      worth anything.

    ``view`` is the sharpest of them: a logistic probe for which view a feature came from,
    per rung, on content / style / both.  Chance is 0.5.  It needs no factor labels and no
    SCM assumption, and it is the probe that caught this leak on this project before — see
    ``training/losses.py``, where every content channel sat at view-AUC 1.000 under Barlow
    Twins while VICReg held 0/44 above 0.7.  Its floor matters as much as anywhere else
    here: an untrained encoder already separates two views that differ in intensity
    statistics, so a raw accuracy near 1.0 is not by itself evidence of anything.

    Comparing two runs whose content/style split differs: the probe width IS the channel
    count, so a 40/8 model hands the content probe 40·P features and a 30/18 model 30·P,
    and ridge in the p≫n regime is not indifferent to that.  Pass an explicit
    ``--probe-dim`` (not ``auto``, which leaves a well-conditioned block at full width) to
    put every block of both models at one width; otherwise the width confound rides along
    inside every number in this section and in table 1.
    """
    avail = set(reprs.keys())
    style_ok = gt_style is not None and len(style_names) and np.asarray(gt_style).shape[1] > 0

    # Order matters: the shared ``rng`` is consumed in call order, and table 1 draws its
    # permutations before this function is reached.  Appending cells here therefore cannot
    # move table 1's floor, while inserting one above it silently would.
    cells = {}
    if style_ok:
        cells["content→style"] = per_factor_scores(
            reprs, level, gt_style, style_names, seeds, n_null, rng, kind, n_jobs, factor_pooling, block=_CONTENT
        )
        cells["style→style"] = per_factor_scores(
            reprs, level, gt_style, style_names, seeds, n_null, rng, kind, n_jobs, factor_pooling, block=_STYLE
        )
    cells["style→content"] = per_factor_scores(
        reprs, level, gt_content, content_names, seeds, n_null, rng, kind, n_jobs, factor_pooling, block=_STYLE
    )
    cells = {k: v for k, v in cells.items() if v}

    view = {}
    if _has_v2(reprs, level):
        for key in ("gap", "stats", "patch"):
            if key not in avail:
                continue
            blocks = _view_blocks(reprs, key, level)
            if blocks is None:
                continue
            c1, c2, s1, s2 = blocks
            view[key] = view_invariance(
                c1, c2, s1, s2, all_v1=np.hstack([c1, s1]), all_v2=np.hstack([c2, s2]), seeds=seeds
            )
    return {"cells": cells, "view": view}


def _causal_adjacency(dataset):
    """The eval set's SCM adjacency, or None when it has no SCM.

    None is the correct answer in two cases and they are not failures: a run trained
    without ``--synthetic-causal``, and any run scored with ``--causal iid`` (which builds
    the test set with ``synthetic_causal=False``, so ``Synthetic3DDisentanglementDataset``
    leaves ``scm`` unset).  The second is the useful one to keep in mind — the partial-R²
    column exists precisely so that per-factor claims do not require leaving the training
    distribution, so it is only defined on the mode where the confound it removes exists.
    """
    for obj in (getattr(dataset, "_inner", None), dataset):
        scm = getattr(obj, "scm", None)
        if isinstance(scm, dict) and scm.get("adj") is not None:
            return np.asarray(scm["adj"])
    return None


def _cell_mean(cell):
    """Mean floor-subtractable R² gap over a cell's factors (nan when it has none)."""
    vals = [_f(d.get("r2")) for d in (cell or {}).values()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _delta(a, b):
    a, b = _f(a), _f(b)
    return a - b if np.isfinite(a) and np.isfinite(b) else float("nan")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def per_factor_decoding_rows(res, floor=None):
    """Expose all four block×target cells without fitting probes again.

    Feature blocks and style targets are from view 1, matching score_run. Each floor
    subtraction stays within the same block, target factor and pooling. Missing scores
    remain NaN, including missing style blocks and reports made with --no-leakage.
    """

    def cells(report):
        leak = ((report or {}).get("leakage") or {}).get("cells") or {}
        return {
            "content": ((report or {}).get("per_factor") or {}, leak.get("style→content") or {}),
            "style": (leak.get("content→style") or {}, leak.get("style→style") or {}),
        }

    def at_pool(score, pooling):
        by_pool = score.get("by_pooling")
        if by_pool is not None:
            return by_pool.get(pooling) or {}
        # Older JSON reports only stored the assigned rung. Never reuse it at another rung.
        return score if score.get("pooling") == pooling else {}

    current, baseline = cells(res), cells(floor)
    rows = []
    for target, (content, style) in current.items():
        for factor in dict.fromkeys([*content, *style]):
            scores = [content.get(factor) or {}, style.get(factor) or {}]
            assigned = next((s["pooling"] for s in scores if s.get("pooling")), None)
            for pooling in ("gap", "stats", "patch"):
                entries = [at_pool(s, pooling) for s in scores]
                if not any(entries):
                    continue
                row = {"target": target, "factor": factor, "pooling": pooling, "assigned_pooling": assigned}
                for block, cur, floor_cell in zip(("content", "style"), entries, baseline[target]):
                    base = at_pool(floor_cell.get(factor) or {}, pooling)
                    row[f"{block}_r2"] = _f(cur.get("r2_raw"))
                    row[f"{block}_r2_gap"] = _f(cur.get("r2"))
                    row[f"{block}_learned"] = _delta(cur.get("r2"), base.get("r2"))
                rows.append(row)
    return rows


def write_report_json(path, res, floor=None, floor_std=None):
    """Keep existing nested scores and add convenient per-factor/block/pooling rows."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "run": res,
                "floor": floor,
                "floor_std": floor_std,
                "per_factor_decoding": per_factor_decoding_rows(res, floor),
            },
            fh,
            indent=2,
            default=float,
        )
    logger.info("Wrote %s", path)


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _n(x, nd=3):
    v = _f(x)
    return f"{v:+.{nd}f}" if np.isfinite(v) else "   -  "


def _acc(x, nd=3):
    """Unsigned formatter for accuracies. A probability printed as '+0.410' reads as a
    delta, and the view table sits next to columns that really are deltas."""
    v = _f(x)
    return f"{v:.{nd}f}" if np.isfinite(v) else "   -  "


def _bar(x, width=8):
    """Signed mini-bar, so a column of numbers has a shape you can scan."""
    v = _f(x)
    if not np.isfinite(v):
        return " " * width
    n = int(round(min(abs(v), 1.0) * width))
    return ("█" * n).ljust(width) if v >= 0 else ("▁" * n).ljust(width)


def _rule(title, w=92):
    head = f"── {title} "
    return head + "─" * max(4, w - len(head))


def print_per_factor_decoding(res, floor=None):
    rows = per_factor_decoding_rows(res, floor)
    if not rows:
        return
    print()
    print(_rule("3b. PER-FACTOR DECODING FROM CONTENT AND STYLE"))
    print("   Both representation blocks are from view 1; style targets use that same view.")
    print("   R2 raw = held-out probe score; R2 gap = raw minus the label-permutation null.")
    print("   learned = R2 gap minus the SAME block/factor/pooling in the untrained twin.")
    print("   * marks the assigned pooling. Other rungs describe where the factor is readable.")
    print("   '-' means not measured. Correlated factors can be predictable through other factors;")
    print("   these rows measure decodability, not each factor's unique causal information.")
    fw = max(14, max(len(row["factor"]) for row in rows))
    for target in ("content", "style"):
        group = [row for row in rows if row["target"] == target]
        if not group:
            continue
        print(f"\n   {target.upper()} FACTORS (prediction targets)")
        print(f"   {'':<{fw}s} {'':<7s}{'from content':^27s}{'from style':^27s}")
        print(f"   {'factor':<{fw}s} {'pool':<7s}" + f"{'R2 raw':>9s}{'R2 gap':>9s}{'learned':>9s}" * 2)
        for row in group:
            pooling = row["pooling"] + ("*" if row["pooling"] == row["assigned_pooling"] else "")
            line = f"   {row['factor']:<{fw}s} {pooling:<7s}"
            for block in ("content", "style"):
                for metric in ("r2", "r2_gap", "learned"):
                    line += f"{_n(row[f'{block}_{metric}']):>9s}"
            print(line)


def print_report(res, floor=None, with_dci=False, floor_std=None):
    """The whole report.

    ``floor`` is the same structure scored on an untrained twin; ``floor_std`` is its
    across-seed spread when several draws were averaged.  A 'learned' value inside that
    spread is not a finding — it is which floor you happened to draw.
    """
    has_floor = floor is not None
    fstd = floor_std or {}
    print()
    print("=" * 92)
    print(f"  IDENTIFIABILITY REPORT — {res['name']}")
    print(f"  N={res['n_samples']}  level={res['level']}  poolings={res['poolings']}  probe-dim={res['probe_dim']}")
    _cz = res.get("causal")
    if _cz:
        _note = {
            "match": "training-matched factors; correlated parents can inflate individual decoding scores",
            "iid": "legacy causal-off sampler; factor marginals may differ from training",
            "shuffled": "independently shuffled content columns, matched empirical marginals, images re-rendered",
        }.get(_cz, _cz)
        print(f"  causal={_cz}  ({_note})")
    distribution = res.get("evaluation_distribution") or {}
    if distribution.get("mode") == "shuffled":
        print(
            f"  shuffle-seed={distribution['shuffle_seed']}  mean |content correlation|: "
            f"{distribution['mean_abs_correlation_before']:.3f} -> {distribution['mean_abs_correlation_after']:.3f}"
        )
        print("  Marginals are exact; finite-sample dependencies need not be zero. The joint distribution changes.")
    print("=" * 92)
    if has_floor:
        print("  LEARNED = this checkpoint minus the same architecture UNTRAINED. Read that column.")
        print("  raw alone is not a result: at patch pooling an untrained encoder already scores")
        print("  R2 >0.8 on most factors and block-MCC ~0.86.")
    else:
        print("  !! NO FLOOR (--no-floor). Nothing below is reportable: an untrained encoder scores")
        print("  !! R2 >0.8 on most factors at patch pooling. Re-run without --no-floor.")

    # 1. per factor -----------------------------------------------------------
    pf, fpf = res["per_factor"], (floor or {}).get("per_factor", {})
    mladder, fladder = res["mcc"], (floor or {}).get("mcc", {})
    mpool = res["mcc_per_factor_pooling"]
    mpf = (mladder.get(mpool) or {}).get("per_factor", {})
    fmpf = ((fladder.get(mpool) or {}).get("per_factor", {})) if has_floor else {}

    part, fpart = res.get("partial") or {}, (floor or {}).get("partial") or {}
    npa = res.get("n_parents") or {}
    pcol = f"{'partial':>8s} {'pa':>3s}" if part else ""

    fw = max([len(k) for k in pf] + [14])
    print()
    print(_rule("1. PER FACTOR"))
    print(
        f"   {'factor':<{fw}s} {'pool':<6s} {'R2 gap':>7s} {'learned':>8s}          "
        f"{pcol}  {'MCC raw':>8s} {'learned':>8s}"
    )
    parent_carried = []
    for name, d in pf.items():
        r2d = _delta(d["r2"], (fpf.get(name) or {}).get("r2")) if has_floor else float("nan")
        m_raw = mpf.get(name, float("nan"))
        mccd = _delta(m_raw, fmpf.get(name)) if has_floor else float("nan")
        sd = _f(((fstd.get("per_factor") or {}).get(name) or {}).get("r2")) if has_floor else float("nan")
        sds = f" +-{sd:.3f}" if np.isfinite(sd) else ""
        pstr = ""
        if part:
            praw = (part.get(name) or {}).get("r2")
            # Floor-subtracted where a floor exists, so the column is read against table 1's
            # `learned` and not against its raw `R2 gap`. The residual has its own floor:
            # an untrained projection recovers a residualised factor differently from the
            # factor itself, so reusing the raw factor's floor here would be a cross-axis
            # subtraction of exactly the kind this script exists to avoid.
            pval = _delta(praw, (fpart.get(name) or {}).get("r2")) if has_floor else praw
            pstr = f"{_n(pval)} {npa.get(name, 0):>3d}"
            if npa.get(name, 0) and np.isfinite(_f(pval)) and np.isfinite(r2d) and (r2d - _f(pval)) > NOISE_FLOOR:
                parent_carried.append(name)
        print(
            f"   {name:<{fw}s} {d['pooling']:<6s} {_n(d['r2'])} {_n(r2d)}{sds} {_bar(r2d)}  "
            f"{pstr}  {_n(m_raw)} {_n(mccd)} {_bar(mccd)}"
        )
    fp = res.get("factor_pooling", "assigned")
    if fp == "assigned":
        print("   R2 gap = null-subtracted, at each factor's own assigned pooling (FACTOR_POOLING).")
    else:
        print(f"   R2 gap = null-subtracted, ALL factors forced to '{fp}' (--factor-pooling), not their")
        print("   assigned pooling. Same axis for every factor; not comparable to an 'assigned' run.")
    print(f"   MCC raw = matched |corr| at {mpool} pooling; block-MCC has no permutation null here,")
    print(f"   which is exactly why its 'learned' column is the only one worth reading.")
    if part:
        print(
            f"   partial = the same score on each factor RESIDUALISED on its SCM parents"
            f"{', floor-subtracted' if has_floor else ' (raw gap — no floor)'}; pa = how many parents it has."
        )
        print("   Legacy adjustment is linear and fitted before CV; nonlinear parent dependence can remain.")
        print("   Use --parent-adjustment nonlinear for cross-fitted nonlinear adjustment (table 1c).")
        if parent_carried:
            print(f"   possible parent contribution (learned - partial > {NOISE_FLOOR}): {', '.join(parent_carried)}")
    elif res.get("parent_adjustment") == "nonlinear":
        print("   Parent adjustment: " + res.get("parent_adjustment_status", "see nonlinear scores in table 1c."))
    elif res.get("causal") == "match":
        print("   partial: no observed-parent SCM on the evaluation set; adjustment is unavailable.")
    elif res.get("causal") in ("iid", "shuffled"):
        print("   Parent adjustment is not scored under this distribution. Use --causal match for it.")

    # 1b. r2 ladder -----------------------------------------------------------
    # Table 1 shows each factor once, at the rung it is reportable at.  That answers "did
    # it learn this factor" but not "where does this factor live", which is the question
    # the whole gap-vs-patch axis of this project turns on — and answering it used to mean
    # running the script once per --factor-pooling and diffing two outputs by eye.
    rungs = [k for k in ("gap", "stats", "patch") if any(k in (d.get("by_pooling") or {}) for d in pf.values())]
    if len(rungs) > 1:
        cw = 9
        print()
        print(_rule("1b. R2 POOLING LADDER (every factor at every rung)"))
        print("   the SHAPE across rungs is the signal: a factor that only reads at patch lives in the")
        print("   spatial layout, one that reads at gap lives in channel identity. R2 columns are")
        print("   null-subtracted as in table 1 ('gap' the rung is not 'gap' the real-minus-null).")
        print("   NOT a menu: the reportable number is table 1's, at the rung named in `assigned`.")
        print("   Taking the best rung here is selection over noisy estimates, and these deltas are")
        print("   not spread-gated — only table 1's +-spread and the verdict are.")
        head, sub = "   " + " " * fw, f"   {'factor':<{fw}s}"
        for k in rungs:
            head += f"{k:^{2 * cw}s}"
            sub += f"{'R2':>{cw}s}{'learned':>{cw}s}"
        print(head + f"{'assigned':>10s}")
        print(sub + f"{'':>10s}")
        for name, d in pf.items():
            row = f"   {name:<{fw}s}"
            for k in rungs:
                cur = (d.get("by_pooling") or {}).get(k) or {}
                base = ((fpf.get(name) or {}).get("by_pooling") or {}).get(k) or {}
                learned = _delta(cur.get("r2"), base.get("r2")) if has_floor else float("nan")
                row += f"{_n(cur.get('r2')):>{cw}s}{_n(learned):>{cw}s}"
            print(row + f"{d['pooling']:>10s}")

    # 1c. nonlinear parent-adjusted decoding ----------------------------------
    adjusted = res.get("parent_adjusted") or {}
    if adjusted:
        print()
        print(_rule("1c. NONLINEAR PARENT-ADJUSTED DECODING"))
        print("   Parent predictors: outer-train only; training residuals: inner cross-fitting.")
        print("   PCA/scaling: outer-train only. Full and residual raw R2 use these same folds/features.")
        print("   Residual gap = residual R2 minus within-split permutation null; learned = gap minus twin.")
        print("   Parent R2 measures held-out nuisance fit quality. Residuals can retain unmodelled dependence;")
        print("   this is a decoding diagnostic, not proof of causal identifiability or independence.")
        print(
            f"   {'block':<8} {'factor':<{fw}} {'pool':<6} {'pa':>3} {'parent':>7} "
            f"{'full':>7} {'resid':>7} {'gap':>7} {'learned':>8} {'floor sd':>8}"
        )
        for block, factors in adjusted.items():
            for factor, entry in factors.items():
                for pool, score in entry["by_pooling"].items():
                    base = (
                        (floor or {})
                        .get("parent_adjusted", {})
                        .get(block, {})
                        .get(factor, {})
                        .get("by_pooling", {})
                        .get(pool, {})
                    )
                    spread = (
                        fstd.get("parent_adjusted", {})
                        .get(block, {})
                        .get(factor, {})
                        .get("by_pooling", {})
                        .get(pool, {})
                        .get("r2")
                    )
                    learned = _delta(score["r2"], base.get("r2")) if has_floor else float("nan")
                    marker = "*" if pool == entry["pooling"] else " "
                    print(
                        f"   {block:<8} {factor:<{fw}} {pool + marker:<6} {npa.get(factor, 0):>3d} "
                        f"{_n(score['parent_r2'])} {_n(score['full_r2_raw'])} {_n(score['r2_raw'])} "
                        f"{_n(score['r2'])} {_n(learned)} {_n(spread)}"
                    )
        print("   * = assigned pooling. Full and residual R2 have different target variances; their difference")
        print("   is not a fraction of causally mediated information. The verdict below uses the legacy headline.")

    # 2. mcc ladder -----------------------------------------------------------
    print()
    print(_rule("2. MCC POOLING LADDER"))
    print("   where the content lives: a rung that rises as pooling gets more spatial means the")
    print("   information is in the layout, not in channel identity.")
    print(f"   {'pooling':<8s} {'raw':>7s} {'+-std':>7s} {'learned':>8s}          assign")
    for key in ("gap", "stats", "patch"):
        if key not in mladder:
            continue
        m = mladder[key]
        d = _delta(m["mean"], (fladder.get(key) or {}).get("mean")) if has_floor else float("nan")
        ident = _f(m.get("assignment_identity"))
        istr = f"{ident:.2f}" if np.isfinite(ident) else "  - "
        warn = "  <- Hungarian permuted" if np.isfinite(ident) and ident < 1.0 else ""
        sd = _f(((fstd.get("mcc") or {}).get(key) or {}).get("mean")) if has_floor else float("nan")
        sds = f" +-{sd:.3f}" if np.isfinite(sd) else ""
        print(f"   {key:<8s} {_n(m['mean'])} {_n(m.get('std'))} {_n(d)}{sds} {_bar(d)}  {istr}{warn}")

    # 3. leakage --------------------------------------------------------------
    leak = res.get("leakage") or {}
    fleak = (floor or {}).get("leakage") or {}
    if not (leak.get("cells") or leak.get("view")):
        print()
        print(_rule("3. LEAKAGE (which block carries which factors)"))
        if "leakage" in res:
            print("   no style block and no second view — nothing to leak into or out of. This is the")
            print("   expected reading for an all-content model (--content-size == hidden channels).")
        else:
            print("   skipped — --no-leakage. Table 1 alone cannot tell a better content block from")
            print("   one that absorbed style; re-run without it before reporting a content result.")
    else:
        print()
        print(_rule("3. LEAKAGE (which block carries which factors)"))
        print("   table 1 is this matrix's content→content cell; these are the others, scored by the")
        print("   same probe against the same nulls and the same floor, so they sit on table 1's")
        print("   scale. A rising content→content is only good news if content→style stays flat")
        print("   beside it: reconstruction has to get the view-specific appearance from somewhere,")
        print("   and content absorbing it also reads as more content information.")
        fcells = fleak.get("cells") or {}
        scells = ((fstd.get("leakage") or {}).get("cells")) or {}
        rows = [("content→content", pf, fpf, (fstd.get("per_factor") or {}))]
        for cname in ("content→style", "style→style", "style→content"):
            cell = (leak.get("cells") or {}).get(cname)
            if cell:
                rows.append((cname, cell, fcells.get(cname) or {}, scells.get(cname) or {}))
        print(f"   {'cell':<16s} {'fac':>3s} {'R2 gap':>7s} {'learned':>8s}")
        for cname, cell, fcell, _sd in rows:
            learned = _delta(_cell_mean(cell), _cell_mean(fcell)) if has_floor else float("nan")
            mark = ""
            if cname == "content→style" and np.isfinite(learned) and learned > NOISE_FLOOR:
                mark = "  <- LEAK"
            elif cname == "style→style" and has_floor and np.isfinite(learned) and learned <= NOISE_FLOOR:
                mark = "  <- style block is dead"
            print(f"   {cname:<16s} {len(cell):>3d} {_n(_cell_mean(cell))} {_n(learned)} {_bar(learned)}{mark}")
        print("   content→style is THE leak. style→style ~0 means the style block is DEAD instead —")
        print("   a different diagnosis with a different fix, so read it first. style→content sits")
        print("   above 0 even when nothing is wrong (correlated SCM factors, same image), so only")
        print("   its learned column carries anything.")
        if leak.get("view"):
            print()
            print("   view probe — which view did this feature come from (logistic, chance 0.500).")
            print("   No factor labels and no SCM assumption, so it is the one number here that does")
            print("   not inherit the generator's correlations. content ≈ chance ⇒ view-invariant.")
            fview = fleak.get("view") or {}
            head = f"   {'pooling':<8s}"
            for lbl in ("content", "style", "all"):
                head += f"{lbl:>8s}{'learned':>9s}"
            print(head)
            for key in ("gap", "stats", "patch"):
                v = (leak["view"] or {}).get(key)
                if not v:
                    continue
                fv = fview.get(key) or {}
                row = f"   {key:<8s}"
                for blk in ("content_acc", "style_acc", "all_acc"):
                    learned = _delta(v.get(blk), fv.get(blk)) if has_floor else float("nan")
                    row += f"{_acc(v.get(blk)):>8s}{_n(learned):>9s}"
                print(row)
            print("   raw content accuracy near 1.0 is NOT by itself a finding — an untrained encoder")
            print("   separates two views that differ in intensity statistics. Read learned. Style")
            print("   ABOVE content is the healthy ordering: the view-specific signal sits in style.")

    if "leakage" in res:
        print_per_factor_decoding(res, floor)

    # 4. dci ------------------------------------------------------------------
    if not (with_dci and res.get("dci")):
        print()
        print(_rule("4. DCI"))
        print("   skipped — pass --with-dci (GBT importance, noticeably slower).")
    else:
        print()
        print(_rule("4. DCI"))
        print("   D = is each code dedicated to few factors.  C = is each factor carried by few codes.")
        print("   Both are basis-dependent, so these are scored on the encoder's OWN channels, never")
        print("   on principal components. Compare a row only with the same row in another model,")
        print("   and only at equal n_codes: C is normalised by log(n_codes).")
        # "gap" is this codebase's word for real-minus-null everywhere (`*_gap`), but it is
        # also a pooling name, and `content@gap` sits in the very next column. Say which.
        print("   'gap' COLUMN = real - permutation null. Not the same thing as the gap POOLING")
        print("   in the scope names below. Read the null too: raw D/C sit high from shape alone")
        print("   (D normalised by log(n_factors), C by log(n_codes)), so a null near 1 - or at")
        print("   0 - leaves a gap that means nothing.")
        print(f"   {'scope':<14s} {'':>2s} {'real':>7s} {'null':>7s} {'gap':>7s} {'learned':>8s}   {'n_codes':>8s}")
        fdci = (floor or {}).get("dci", {})
        for scope, d in res["dci"].items():
            fd = fdci.get(scope, {})
            nc = _f(d.get("n_codes"))
            ncs = f"{int(nc):>8d}" if np.isfinite(nc) else "       ?"
            for m in ("d", "c"):
                learned = _delta(d.get(f"{m}_gap"), fd.get(f"{m}_gap")) if has_floor else float("nan")
                print(
                    f"   {scope:<14s} {m.upper():>2s} {_n(d.get(m))} {_n(d.get(f'{m}_null'))} "
                    f"{_n(d.get(f'{m}_gap'))} {_n(learned)}   {ncs}"
                )
                null = _f(d.get(f"{m}_null"))
                if np.isfinite(null) and null > 0.9:
                    print(f"   {'':<17s} ^ null saturated (>0.9) — this gap cannot go positive; not reportable")
        ncs = {_f(d.get("n_codes")) for d in res["dci"].values() if np.isfinite(_f(d.get("n_codes")))}
        if len(ncs) > 1:
            print("   ! rows differ in n_codes — they are on different scales. Do not compare them.")

    # 5. verdict --------------------------------------------------------------
    print()
    print(_rule("5. VERDICT"))
    if not has_floor:
        print("   No floor measured — no verdict. Re-run without --no-floor.")
        print()
        return
    r2_learned = [_delta(d["r2"], (fpf.get(name) or {}).get("r2")) for name, d in pf.items()]
    r2_learned = [v for v in r2_learned if np.isfinite(v)]
    mean_r2 = float(np.mean(r2_learned)) if r2_learned else float("nan")

    # A factor counts as resolved only if its learned delta clears BOTH bars: the fixed
    # probe-noise floor, and 2x this factor's own across-seed floor spread.  The second is
    # the one that catches a delta that is really just which untrained draw came up — and
    # it is per factor, because the spread is not uniform across them.
    def _bar_for(name):
        sd = _f(((fstd.get("per_factor") or {}).get(name) or {}).get("r2"))
        return max(NOISE_FLOOR, 2.0 * sd) if np.isfinite(sd) else NOISE_FLOOR

    resolved, borderline = [], []
    for name, d in pf.items():
        delta = abs(_delta(d["r2"], (fpf.get(name) or {}).get("r2")))
        if not np.isfinite(delta):
            continue
        (resolved if delta > _bar_for(name) else borderline).append(name)
    nseeds = _f((floor or {}).get("floor_seeds"))
    print(f"   mean learned R2 over {len(r2_learned)} factors: {_n(mean_r2)}")
    if np.isfinite(nseeds) and nseeds > 1:
        print(f"   floor averaged over {int(nseeds)} untrained draws; a factor is RESOLVED only if its")
        print(f"   learned delta clears max({NOISE_FLOOR}, 2x its own across-seed floor spread).")
    elif np.isfinite(nseeds):
        print(f"   floor is ONE untrained draw — no spread to test against. Raise --floor-seeds;")
        print(f"   only the fixed {NOISE_FLOOR} probe-noise floor is applied below.")
    if np.isfinite(mean_r2) and abs(mean_r2) <= NOISE_FLOOR:
        print(f"   INSIDE NOISE (|delta| <= {NOISE_FLOOR}). This checkpoint is not distinguishable")
        print("   from its own untrained architecture on aggregate factor recovery.")
    print(f"   RESOLVED: {', '.join(resolved) if resolved else 'NONE'}")
    print(f"   not resolved: {', '.join(borderline) if borderline else '-'}")

    # Parent-carried factors are RESOLVED and yet not a per-factor result, so they get
    # their own line rather than a demotion: the content block did learn something real
    # here, it is just not this factor's own variation, and collapsing that into the
    # resolved/unresolved split would lose which of the two a reader is looking at.
    if part and parent_carried:
        pm = float(np.mean([_f(v.get("r2")) for v in part.values() if np.isfinite(_f(v.get("r2")))]))
        print(f"   PARENT-CARRIED: {', '.join(parent_carried)} — resolved, but the score drops by more")
        print(f"   than {NOISE_FLOOR} once the SCM parents are residualised out, so it is not evidence that")
        print(f"   these factors are encoded in their own right. (mean partial R2 gap {_n(pm)})")

    # Leak verdict. Deliberately separate from the RESOLVED list: a leak is not a factor
    # failing to be learned, it is the content block having learned the wrong thing, and
    # collapsing the two into one pass/fail is what would let a leaking run read as a win.
    if leak.get("cells") or leak.get("view"):
        fcells = fleak.get("cells") or {}
        c2s = _delta(
            _cell_mean((leak.get("cells") or {}).get("content→style")), _cell_mean(fcells.get("content→style"))
        )
        s2s = _delta(_cell_mean((leak.get("cells") or {}).get("style→style")), _cell_mean(fcells.get("style→style")))
        if np.isfinite(s2s) and s2s <= NOISE_FLOOR:
            print(f"   STYLE BLOCK DEAD: style→style learned {_n(s2s)} — style carries none of its own")
            print("   factors, so reconstruction has nowhere but content to put the view-specific")
            print("   appearance. Fix that before reading content→style as a property of the content block.")
        if np.isfinite(c2s) and c2s > NOISE_FLOOR:
            print(f"   LEAK: style factors read from content at learned {_n(c2s)} (> {NOISE_FLOOR}). The")
            print("   content block is not view-invariant; a higher content→content is not a clean win.")
        elif np.isfinite(c2s):
            print(f"   no style leak into content (content→style learned {_n(c2s)}, inside {NOISE_FLOOR}).")
        for key in ("gap", "stats", "patch"):
            v = (leak.get("view") or {}).get(key)
            fv = ((fleak.get("view") or {}).get(key)) or {}
            if not v:
                continue
            d = _delta(v.get("content_acc"), fv.get("content_acc"))
            if np.isfinite(d) and d > NOISE_FLOOR:
                print(f"   content→view@{key}: learned {_n(d)} — view is decodable from content above its")
                print("   untrained floor. This is the label-free confirmation of the row above.")
    for key in ("gap", "stats", "patch"):
        if key not in mladder:
            continue
        d = _delta(mladder[key]["mean"], (fladder.get(key) or {}).get("mean"))
        if np.isfinite(d) and abs(d) <= NOISE_FLOOR:
            print(f"   MCC@{key}: learned {_n(d)} — inside noise, not reportable as a difference.")
    print()


# --------------------------------------------------------------------------- #
# Model path (torch, lazily imported)
# --------------------------------------------------------------------------- #


def score_run(
    run_dir,
    dataset,
    poolings,
    level,
    seeds,
    n_null,
    batch_size,
    num_workers,
    device,
    checkpoint=None,
    probe_dim=PROBE_DIM_AUTO,
    with_dci=False,
    with_leakage=True,
    dci_max_codes=4096,
    random_init=False,
    probe_kind="ridge",
    n_jobs=1,
    factor_pooling="assigned",
    init_seed=None,
    causal=None,
    name=None,
    parent_adjustment="legacy",
    parent_cache=None,
    freeze_content_mask=False,
):
    """Extract this run's representations under every pooling and score them.

    ``random_init=True`` builds the same architecture UNTRAINED — the floor twin.  It must
    go through this identical function, not a second script, so the floor and the
    checkpoint share pooling, probe width, null count and factor routing; differencing two
    scripts' numbers is the cross-axis mistake this project has already paid for twice.
    """
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _reduce_reprs, _resolve_checkpoint, _score_dci
    from eval.run_dci_synthetic import load_model_from_run_dir

    if parent_adjustment not in ("legacy", "nonlinear"):
        raise ValueError("parent_adjustment must be legacy or nonlinear")
    if parent_adjustment == "nonlinear" and causal in ("iid", "shuffled"):
        raise ValueError("Nonlinear parent adjustment requires the matched evaluation distribution.")
    model, _args, device = load_model_from_run_dir(
        run_dir,
        None if random_init else _resolve_checkpoint(run_dir, checkpoint),
        device,
        random_init=random_init,
        seed=init_seed,
    )
    reprs, gt_content, gt_style, info = {}, None, None, None
    for key, value in poolings:
        level_data, gc, gs1, _gs2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value, freeze_content_mask=freeze_content_mask
        )
        reprs[key] = level_data
        if gt_content is None:
            gt_content = gc
        # View-1 style factors, paired with the _STYLE / _CONTENT (view-1) blocks. The
        # loader runs unshuffled, so every pooling returns the same rows in the same order.
        if gt_style is None:
            gt_style = gs1
        if info is None and level in level_data:
            info = level_data[level][4]
    del model
    if info is None:
        raise RuntimeError(f"level {level} not found in encoder outputs for {run_dir}")

    names = info["content_names"]
    style_names = info.get("style_names") or []
    rng = np.random.RandomState(0)

    # DCI reads the unreduced blocks; the probes read the reduced ones.  Same split as
    # run_dci_compare.score_reprs, for the same reason.
    dci_reprs = reprs
    probed = _reduce_reprs(reprs, level, probe_dim) if probe_dim else reprs

    res = {
        "name": name or os.path.basename(os.path.normpath(run_dir)),
        "level": level,
        "n_samples": int(gt_content.shape[0]),
        # Render the VALUE, not the bucket key: "patch" alone hides whether the grid was
        # 2x2x2 or 8x8x8, and that is a 64x difference in code count.
        "poolings": ",".join("x".join(str(d) for d in v) if isinstance(v, tuple) else str(v) for _k, v in poolings),
        "probe_dim": probe_dim,
        "factor_pooling": factor_pooling,
        "causal": causal,
        "freeze_content_mask": freeze_content_mask,
        "per_factor": per_factor_scores(
            probed, level, gt_content, names, seeds, n_null, rng, probe_kind, n_jobs, factor_pooling
        ),
        "mcc": mcc_ladder(probed, level, gt_content, seeds, probe_kind, names),
        "mcc_per_factor_pooling": "patch" if "patch" in probed else _resolve_key("stats", set(probed)),
    }
    if with_leakage:
        # After ``per_factor`` in this dict literal, and that ordering is load-bearing: the
        # two share ``rng``, so scoring the leakage cells first would redraw table 1's
        # permutations and move a floor that was reproducible before this section existed.
        res["leakage"] = leakage_scores(
            probed,
            level,
            gt_content,
            gt_style,
            names,
            style_names,
            seeds,
            n_null,
            rng,
            probe_kind,
            n_jobs,
            factor_pooling,
        )

    if hasattr(dataset, "evaluation_distribution"):
        res["evaluation_distribution"] = dataset.evaluation_distribution

    # Legacy adjustment remains the default. Nonlinear adjustment has a separate
    # table with fold-local preprocessing and its own full-target reference scores.
    adjacency = _causal_adjacency(dataset)
    if parent_adjustment == "nonlinear":
        res["parent_adjustment"] = "nonlinear"
        if adjacency is None:
            res["parent_adjustment_status"] = "unavailable: no observed-parent SCM on the evaluation set"
            logger.warning(res["parent_adjustment_status"])
        else:
            cache = parent_cache if parent_cache is not None else {}
            res["parent_adjustment_config"] = dict(
                regressor="HistGradientBoostingRegressor",
                outer_splits=5,
                inner_splits=3,
                seeds=list(seeds),
                n_null=n_null,
                probe_kind=probe_kind,
                preprocessing="outer_train_only",
            )
            res["parent_adjusted"] = {}
            for block_name, block_idx in (("content", _CONTENT), ("style", _STYLE)):
                if block_name == "style" and not with_leakage:
                    continue
                res["parent_adjusted"][block_name] = nonlinear_parent_scores(
                    reprs,
                    level,
                    gt_content,
                    names,
                    adjacency,
                    seeds,
                    n_null,
                    probe_dim,
                    probe_kind,
                    n_jobs,
                    factor_pooling,
                    block_idx,
                    cache,
                )
            res["n_parents"] = n_parents_per_factor(adjacency, names)
            # Keep downstream DCI permutation draws identical to legacy mode, even
            # though the old partial probes themselves are skipped.
            for factor in names:
                want = FACTOR_POOLING.get(factor, "stats") if factor_pooling == "assigned" else factor_pooling
                X = _block_array(probed, _resolve_key(want, set(probed)), level, _CONTENT)
                if X is not None and X.shape[1]:
                    for _ in range(n_null):
                        rng.permutation(len(gt_content))
    elif adjacency is not None:
        res["partial"] = per_factor_scores(
            probed,
            level,
            residualise_on_parents(gt_content, adjacency),
            names,
            seeds,
            n_null,
            rng,
            probe_kind,
            n_jobs,
            factor_pooling,
        )
        res["n_parents"] = n_parents_per_factor(adjacency, names)
    if with_dci:
        avail = set(dci_reprs.keys())
        dci = {}
        # Content block x content factors at EVERY requested pooling — the whole report is
        # content-side, so mixing the style block in here would make the DCI rows answer a
        # different question from every other row.  `dci_reprs`, not `probed`: D/C are
        # basis-dependent and only mean something on the encoder's own channels.
        #
        # gap is included and listed first because it is the only rung where a "code" IS a
        # latent dimension, which is the object DCI is defined over.  stats splits each
        # channel into 4 codes (mean/std/max/min) and patch into one per cell, so a single
        # clean channel arrives as many near-duplicate codes: that dilutes C and inflates D
        # by construction, and no null subtraction undoes it.  Omitting gap left the report
        # showing only the two rungs where D/C are hardest to interpret.
        for pooling_key in ("gap", "stats", "patch"):
            if pooling_key not in avail:
                continue
            scope = f"content@{pooling_key}"
            d = _score_dci(
                dci_reprs,
                level,
                _CONTENT,
                gt_content,
                None,
                avail,
                n_null,
                rng,
                key_prefix="dci",
                pooling_key=pooling_key,
                max_codes=dci_max_codes,
                n_jobs=n_jobs,
            )
            # Keep real and null, not just the gap: a gap alone is not interpretable.
            # D is normalised by log(n_factors) and C by log(n_codes), so both sit high
            # from shape alone, and a null saturating near 1 (or collapsing to 0) makes
            # the gap meaningless in a way that is only visible next to the null.
            dci[scope] = {
                "d": d.get("dci_d"),
                "d_null": d.get("dci_d_null"),
                "d_gap": d.get("dci_d_gap"),
                "c": d.get("dci_c"),
                "c_null": d.get("dci_c_null"),
                "c_gap": d.get("dci_c_gap"),
                "n_codes": d.get("dci_n_codes"),
            }
        res["dci"] = dci
    return res


def _self_test():
    """Score planted numpy data — no torch, no checkpoint, no GPU.

    Two arms: features that genuinely encode the factors, and pure noise of identical
    shape.  The report must separate them; if it does not, the plumbing is wrong in a way
    no checkpoint run would make obvious.

    The signal arm also plants a deliberate LEAK — the style factors are written into the
    content block as well as the style block — so section 3 is exercised on data whose
    answer is known.  A leak detector that only ever runs on checkpoints is a detector
    nobody can tell is working: this is the arm that would catch the block indices being
    swapped, or ``gt_style`` arriving paired with the wrong view.
    """
    rng = np.random.RandomState(0)
    n, n_fac, n_ch, n_sty_ch = 400, 4, 12, 6
    names = ["brain_size", "ventricle_size", "lesion_x", "gain"][:n_fac]
    style_names = ["bias", "noise_sigma"]
    gt = rng.randn(n, n_fac)
    # ventricle_size is a CHILD of brain_size and is never encoded on its own (A's row for
    # it is zeroed).  A probe should still read it well through the parent, and the partial
    # column should be the thing that says so — this is the confound the column exists for,
    # planted at a known strength so the detector is testable without a checkpoint.
    gt[:, 1] = 0.9 * gt[:, 0] + 0.44 * rng.randn(n)
    adj = np.zeros((n_fac, n_fac), dtype=bool)
    adj[0, 1] = True  # adjacency[i, j] = i is a parent of j
    gt_s = rng.randn(n, len(style_names))
    A = rng.randn(n_fac, n_ch)
    A[1, :] = 0.0
    LEAK = 0.8  # style→content coupling in the signal arm; big enough to clear NOISE_FLOOR
    signal = gt @ A + LEAK * (gt_s @ rng.randn(len(style_names), n_ch)) + 0.1 * rng.randn(n, n_ch)
    noise = rng.randn(n, n_ch)
    style_sig = gt_s @ rng.randn(len(style_names), n_sty_ch) + 0.1 * rng.randn(n, n_sty_ch)
    style_noise = rng.randn(n, n_sty_ch)

    def _mk(X, S):
        # (content, style, content_v2, style_v2, info) at level 0, one entry per pooling.
        # View 2 differs from view 1 only in the style block, which is the arrangement the
        # view probe is supposed to read as "content invariant, style view-specific".
        info = {
            "content_names": names,
            "style_names": style_names,
            "n_content_channels": X.shape[1],
            "n_style_channels": S.shape[1],
            "has_split": True,
            "pooling": "gap",
            "level": 0,
        }
        v2c = X + 0.05 * rng.randn(*X.shape)
        v2s = S + 2.0 * rng.randn(*S.shape)
        return {p: {0: (X, S, v2c, v2s, info)} for p in ("gap", "stats", "patch")}

    out = {}
    for label, X, S in (("signal", signal, style_sig), ("noise", noise, style_noise)):
        r = np.random.RandomState(0)
        reprs = _mk(X, S)
        res = {
            "name": label,
            "level": 0,
            "n_samples": n,
            "poolings": "gap,stats,patch",
            "probe_dim": 0,
            "per_factor": per_factor_scores(reprs, 0, gt, names, (0, 1), 2, r),
            "mcc": mcc_ladder(reprs, 0, gt, (0, 1), names=names),
            "mcc_per_factor_pooling": "patch",
        }
        res["leakage"] = leakage_scores(reprs, 0, gt, gt_s, names, style_names, (0, 1), 2, r)
        res["partial"] = per_factor_scores(reprs, 0, residualise_on_parents(gt, adj), names, (0, 1), 2, r)
        res["n_parents"] = n_parents_per_factor(adj, names)
        out[label] = res
    print_report(out["signal"], floor=out["noise"])
    s = float(np.mean([d["r2"] for d in out["signal"]["per_factor"].values()]))
    z = float(np.mean([d["r2"] for d in out["noise"]["per_factor"].values()]))
    m_s, m_z = out["signal"]["mcc"]["gap"]["mean"], out["noise"]["mcc"]["gap"]["mean"]
    print(f"  self-test: mean R2 gap  signal {s:+.3f}  noise {z:+.3f}")
    print(f"  self-test: block-MCC    signal {m_s:+.3f}  noise {m_z:+.3f}")
    assert s > 0.5, f"planted signal should be recovered, got {s}"
    assert abs(z) < 0.15, f"pure noise should sit at its null, got {z}"
    assert m_s > m_z + 0.2, f"MCC should separate signal from noise, got {m_s} vs {m_z}"
    assert _auto_probe_dim(400, 10) == 0 and _auto_probe_dim(400, 5000) == 64

    def _cell(arm, key):
        return _cell_mean((out[arm]["leakage"]["cells"] or {}).get(key))

    leak_s, leak_z = _cell("signal", "content→style"), _cell("noise", "content→style")
    ss_s, ss_z = _cell("signal", "style→style"), _cell("noise", "style→style")
    view_s = out["signal"]["leakage"]["view"]["gap"]
    view_z = out["noise"]["leakage"]["view"]["gap"]
    print(f"  self-test: content→style  signal {leak_s:+.3f}  noise {leak_z:+.3f}")
    print(f"  self-test: style→style    signal {ss_s:+.3f}  noise {ss_z:+.3f}")
    print(f"  self-test: style→view     signal {view_s['style_acc']:.3f}  content {view_s['content_acc']:.3f}")
    assert leak_s - leak_z > NOISE_FLOOR, f"planted leak should clear the floor, got {leak_s} vs {leak_z}"
    assert ss_s - ss_z > NOISE_FLOOR, f"style block should recover its own factors, got {ss_s} vs {ss_z}"
    assert abs(leak_z) < 0.15, f"the floor arm should have no leak, got {leak_z}"
    # The planted arrangement is content-invariant / style-view-specific across views, and
    # the probe must read it in that direction — a swap of the two would still "work" on a
    # checkpoint and silently invert every leak conclusion drawn from it.
    assert (
        view_s["style_acc"] > view_s["content_acc"]
    ), f"style should be more view-separable than content, got {view_s} / {view_z}"

    def _learned(key, factor):
        return out["signal"][key][factor]["r2"] - out["noise"][key][factor]["r2"]

    v_full, v_part = _learned("per_factor", "ventricle_size"), _learned("partial", "ventricle_size")
    b_full, b_part = _learned("per_factor", "brain_size"), _learned("partial", "brain_size")
    print(f"  self-test: ventricle (child, never encoded)  full {v_full:+.3f}  partial {v_part:+.3f}")
    print(f"  self-test: brain_size (parent, encoded)      full {b_full:+.3f}  partial {b_part:+.3f}")
    assert v_full > 0.5, f"the child should still read well THROUGH its parent, got {v_full}"
    assert v_full - v_part > NOISE_FLOOR, f"partial must expose the parent-carried child, got {v_full} vs {v_part}"
    # A parentless factor must come back untouched. Asserted on the residualiser itself and
    # not on the two probe columns: those draw different permutation nulls from the shared
    # rng, so they agree only to null noise (~0.01 here) even when the targets are identical.
    _resid = residualise_on_parents(gt, adj)
    for j in (0, 2, 3):
        assert np.array_equal(_resid[:, j], gt[:, j]), f"parentless factor {names[j]} was residualised"
    assert not np.allclose(_resid[:, 1], gt[:, 1]), "the child factor was NOT residualised"
    # ...and must therefore never be flagged parent-carried by the table-1 rule.
    assert b_full - b_part < NOISE_FLOOR, f"a parentless factor read as parent-carried, {b_full} vs {b_part}"
    assert n_parents_per_factor(adj, names) == {"brain_size": 0, "ventricle_size": 1, "lesion_x": 0, "gain": 0}
    _assert_dci_basis()
    print("  self-test PASSED")


def _assert_dci_basis():
    """Regression guard: DCI must never be handed a PCA-projected block.

    This is the defect this report exists to avoid reproducing — `score_reprs` used to
    reduce before scoring, so `dci_*` described the principal components rather than the
    encoder, and under `--probe-dim auto` different scopes landed in different bases.  It
    is silent when it regresses (the numbers stay plausible), so it is asserted here
    rather than left to a reviewer.
    """
    import eval.run_dci_compare as R
    from eval.run_dci_compare import _blank_dci, _reduce_reprs
    from eval.run_dci_compare import _score_dci as real_score_dci
    from eval.run_dci_compare import _select_codes, score_reprs

    rng = np.random.RandomState(0)
    n, wide = 300, 900  # 900 > n/4 = 75, so `auto` reduces the probe blocks to 64
    content, style = rng.randn(n, wide), rng.randn(n, 4)
    info = {
        "content_names": ["brain_size", "ventricle_size", "lesion_x"],
        "style_names": ["gain", "bias"],
        "n_content_channels": wide,
        "n_style_channels": 4,
        "has_split": True,
    }
    reprs = {p: {0: (content, style, None, None, info)} for p in ("stats", "patch")}

    seen = {}

    def _spy(reprs_arg, level, block_idx, gc, gs, avail, n_null, rng_, key_prefix="dci", **kw):
        X = R._block_array(reprs_arg, kw.get("pooling_key") or "stats", level, block_idx)
        seen[key_prefix] = X.shape[1]
        return _blank_dci(key_prefix)

    R._score_dci = _spy
    try:
        score_reprs(
            reprs,
            rng.randn(n, 3),
            rng.randn(n, 2),
            info,
            0,
            n_null=0,
            seeds=(0,),
            with_dci=True,
            probe_dim=PROBE_DIM_AUTO,
        )
    finally:
        R._score_dci = real_score_dci

    assert seen["dci"] == wide + 4, f"DCI saw {seen['dci']} codes, expected unreduced {wide + 4}"
    assert seen["dci_content"] == wide, f"DCI content saw {seen['dci_content']}, expected {wide}"
    assert seen["dci_patch"] == wide + 4, f"DCI patch saw {seen['dci_patch']}, expected {wide + 4}"
    # ...while the probes ARE still reduced, i.e. the fix did not just disable probe_dim.
    assert _reduce_reprs(reprs, 0, PROBE_DIM_AUTO)["stats"][0][0].shape[1] == 64

    # And the cap selects real features rather than rotating them.
    X = rng.randn(200, 50) * np.concatenate([np.full(40, 0.1), np.full(10, 20.0)])
    Xs, k = _select_codes(X, 10)
    assert k == 10 and np.allclose(Xs, X[:, 40:]), "cap must SELECT top-variance codes, in order"
    assert _select_codes(X, 0)[1] == 50, "max_codes=0 must disable the cap"

    # n_jobs must not change any number: permutations are drawn up-front in factor order,
    # and each GBT fit owns its own importance column.
    _rng_a, _rng_b = np.random.RandomState(7), np.random.RandomState(7)
    _gt = np.random.RandomState(1).randn(200, 3)
    _A = np.random.RandomState(2).randn(3, 8)
    _rp = {p: {0: (_gt @ _A, None, None, None, {})} for p in ("gap", "stats", "patch")}
    _nm = ["brain_size", "lesion_x", "gain"]
    _s1 = per_factor_scores(_rp, 0, _gt, _nm, (0,), 2, _rng_a, n_jobs=1)
    _s4 = per_factor_scores(_rp, 0, _gt, _nm, (0,), 2, _rng_b, n_jobs=4)
    for _k in _s1:
        assert abs(_s1[_k]["r2"] - _s4[_k]["r2"]) < 1e-9, f"n_jobs changed {_k}: {_s1[_k]} vs {_s4[_k]}"
        for _p, _v in _s1[_k]["by_pooling"].items():
            assert abs(_v["r2"] - _s4[_k]["by_pooling"][_p]["r2"]) < 1e-9, f"n_jobs changed {_k}@{_p}"
    # Every factor is scored at every extracted rung, and the headline is the assigned
    # rung's own entry — not a max, a mean, or a different probe. Asserted because the
    # ladder is the only place two rungs of the same factor sit next to each other, which
    # is exactly where a silent mix-up would look like a locality finding.
    for _k, _d in _s1.items():
        assert set(_d["by_pooling"]) == {"gap", "stats", "patch"}, f"{_k} missing rungs: {set(_d['by_pooling'])}"
        _h = _d["by_pooling"][_d["pooling"]]
        assert _d["r2"] == _h["r2"] and _d["r2_raw"] == _h["r2_raw"], f"{_k} headline != its assigned rung"
    # --factor-pooling forces every factor onto one rung instead of its assigned one.
    # Asserted against FACTOR_POOLING itself rather than against hardcoded pooling names:
    # the table is a research parameter that gets retuned, and a test that pins its values
    # fails on every retune while testing nothing about the routing mechanism.
    _forced = per_factor_scores(_rp, 0, _gt, _nm, (0,), 0, np.random.RandomState(7), factor_pooling="patch")
    assert {d["pooling"] for d in _forced.values()} == {"patch"}, _forced
    for _k, _d in _s1.items():
        assert _d["pooling"] == FACTOR_POOLING[_k], f"{_k} routed to {_d['pooling']}, table says {FACTOR_POOLING[_k]}"
    # Forcing a rung must only move which entry is quoted, never rescore it: the ladder
    # would otherwise disagree with a --factor-pooling run of the same checkpoint.
    _f2 = per_factor_scores(_rp, 0, _gt, _nm, (0,), 2, np.random.RandomState(7), factor_pooling="patch")
    _f1 = per_factor_scores(_rp, 0, _gt, _nm, (0,), 2, np.random.RandomState(7))
    for _k in _f1:
        assert abs(_f2[_k]["r2"] - _f1[_k]["by_pooling"]["patch"]["r2"]) < 1e-9, f"forced rung rescored {_k}"

    # --checkpoint defaulting to None used to reach os.path.join and raise a TypeError that
    # said nothing about checkpoints. Both the guard and this script's default are asserted.
    from eval.run_dci_compare import _resolve_checkpoint

    assert _resolve_checkpoint("/nonexistent/run").endswith("vqvae_model.pt")
    assert _resolve_checkpoint("/nonexistent/run", None).endswith("vqvae_model.pt")
    assert _resolve_checkpoint("/nonexistent/run", "vqvae_best.pt").endswith("vqvae_best.pt")

    # The header must record the grid, not just the bucket: 2x2x2 and 8x8x8 are a 64x
    # difference in code count and would otherwise both print as "patch".
    assert (
        ",".join(
            "x".join(str(d) for d in v) if isinstance(v, tuple) else str(v)
            for _k, v in parse_poolings("gap,stats,8x8x8")
        )
        == "gap,stats,8x8x8"
    )
    # Floor averaging: numeric leaves averaged, metadata passed through, std reported.
    _a = {"per_factor": {"brain_size": {"r2": 0.90, "pooling": "gap"}}}
    _b = {"per_factor": {"brain_size": {"r2": 0.94, "pooling": "gap"}}}
    _c = {"per_factor": {"brain_size": {"r2": 0.86, "pooling": "gap"}}}
    _m, _sd = mean_std_structs([_a, _b, _c])
    assert abs(_m["per_factor"]["brain_size"]["r2"] - 0.90) < 1e-9, _m
    assert abs(_sd["per_factor"]["brain_size"]["r2"] - float(np.std([0.90, 0.94, 0.86]))) < 1e-9, _sd
    assert _m["per_factor"]["brain_size"]["pooling"] == "gap", "metadata must not be averaged"
    assert mean_std_structs([_a])[1]["per_factor"]["brain_size"]["r2"] == 0.0, "one draw => zero spread"
    assert mean_std_structs([])[0] == {}, "empty input must not raise"
    assert abs(mean_std_structs([{"x": float("nan")}, {"x": 0.5}])[0]["x"] - 0.5) < 1e-9, "nan leaves dropped"
    print("  self-test: floor averaging — means, spreads, metadata passthrough OK")
    print("  self-test: DCI reads the unreduced block (probes still reduced to 64) — basis OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--run-dir", help="Run directory to score.")
    source.add_argument("--from-json", help="Redisplay a saved --out JSON with current tables; no model or probe fits.")
    p.add_argument("--name", default=None, help="Label for the report (default: basename of --run-dir).")
    p.add_argument(
        "--checkpoint",
        default="vqvae_model.pt",
        help="Checkpoint filename inside the run dir (same default as run_dci_compare "
        "--checkpoint-name). Use vqvae_best.pt for the best-by-loss copy; whichever you pick, use "
        "the same one for every run you intend to compare.",
    )
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument(
        "--poolings",
        default="gap,stats,8x8x8",
        help="Comma list: gap, stats, DxHxW. Every factor is probed at every rung listed here "
        "(table 1b), so this also sets the per-factor scoring cost; the MCC ladder needs all "
        "three rungs to be readable, so trim it only if the probes are the bottleneck.",
    )
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-null", type=int, default=3, help="Permutations for the label-permutation null.")
    p.add_argument("--probe-dim", default=PROBE_DIM_AUTO, help="'auto', or an integer width, or 0.")
    p.add_argument("--probe-kind", default="ridge", choices=("ridge", "kernel", "mlp"))
    p.add_argument("--with-dci", action="store_true", help="Also score DCI (GBT — slow).")
    p.add_argument(
        "--no-leakage",
        action="store_true",
        help="Skip sections 3/3b (leakage, per-factor content/style decoding, and the view probe). "
        "It roughly doubles the probe cost, since the off-diagonal cells are scored at every "
        "rung in --poolings exactly as table 1 is, and it runs on the floor twin too. Skip it "
        "only when the question is genuinely content-side; a content→content number read "
        "without it cannot tell a better content block from one that absorbed style.",
    )
    p.add_argument(
        "--dci-max-codes",
        type=int,
        default=4096,
        help="Cap DCI width by selecting highest-variance codes (0 = no cap). Never a rotation.",
    )
    p.add_argument(
        "--floor-seeds",
        type=int,
        default=3,
        help="How many untrained draws to average the floor over (>=1). The untrained weights ARE "
        "the measurement, and nothing in the eval path seeded them: the floor was one random "
        "projection, redrawn every invocation, and its noise went silently into every 'learned' "
        "number. Seeds are 0..N-1, so the floor is reproducible, and the across-seed std is printed "
        "beside each delta so you can see which ones clear it.",
    )
    p.add_argument(
        "--no-floor", action="store_true", help="Skip the untrained twin. The report then refuses to give a verdict."
    )
    p.add_argument(
        "--causal",
        default="match",
        choices=("match", "iid", "shuffled"),
        help="'match': training-matched sampler (default). 'iid': legacy causal-off sampler. "
        "'shuffled': independently permute matched content columns and re-render, preserving "
        "empirical marginals (pseudo_mri only); this changes the joint distribution.",
    )
    p.add_argument(
        "--parent-adjustment",
        default="legacy",
        choices=("legacy", "nonlinear"),
        help="'nonlinear': replace legacy partial scores with cross-fitted nonlinear parent-residual "
        "decoding and train-fold-only PCA/scaling (table 1c). Requires --causal match. "
        "Also scores residuals from style unless --no-leakage is set.",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Content-column permutation seed for --causal shuffled; use the same seed across models.",
    )
    p.add_argument(
        "--freeze-content-mask",
        action="store_true",
        help="Pin the Gumbel mask's channel selection to the first batch, so every row of a "
        "feature column describes the same physical channel. Only affects mask_mode=onthefly "
        "checkpoints, which redraw the mask on every forward; the extractor warns when the "
        "selection actually moves. Off by default because turning it on moves numbers for exactly "
        "those checkpoints, and the reports already published were produced without it. Match it "
        "to eval.export_vq_bundle (which freezes by default) when comparing the two.",
    )
    p.add_argument(
        "--factor-pooling",
        default="assigned",
        choices=("assigned", "gap", "stats", "patch"),
        help="Which pooling each factor's REPORTABLE R2 is read at — the number in table 1 that "
        "the verdict is computed from. 'assigned' (default) uses FACTOR_POOLING: each factor read "
        "where it can physically appear, fixed in advance so the headline is never a max over "
        "poolings. Naming one pooling quotes EVERY factor from that rung instead, putting them all "
        "on one axis. This no longer changes what is MEASURED — table 1b already shows every "
        "factor at every rung in --poolings, so there is no need to re-run the script per pooling "
        "just to see them. The pooling must be in --poolings or it falls back. Patch has a high "
        "floor, so read the learned column.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for the probes and the DCI factor fits (-1 = all cores, 1 = "
        "sequential). sklearn's GBT is single-threaded, so this is the main lever on "
        "--with-dci wall time. Results are identical at any setting: permutations are drawn "
        "up-front in factor order and each DCI fit owns its own importance column.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="Optional JSON path for the scored numbers.")
    p.add_argument("--self-test", action="store_true", help="Run the numpy self-test and exit.")
    cli = p.parse_args()

    if cli.self_test:
        _self_test()
        return
    if cli.from_json:
        with open(cli.from_json) as fh:
            saved = json.load(fh)
        if not isinstance(saved, dict) or not isinstance(saved.get("run"), dict):
            p.error("--from-json expects a report JSON written by --out, containing a 'run' object.")
        res, floor, floor_std = saved["run"], saved.get("floor"), saved.get("floor_std")
        print_report(res, floor=floor, floor_std=floor_std, with_dci=cli.with_dci or bool(res.get("dci")))
        if cli.out:
            write_report_json(cli.out, res, floor, floor_std)
        return
    if not cli.run_dir:
        p.error("--run-dir is required (or pass --from-json / --self-test)")
    if cli.parent_adjustment == "nonlinear":
        if cli.causal != "match":
            p.error(
                "--parent-adjustment nonlinear requires --causal match; the original parent graph "
                "does not describe the independent evaluation distribution."
            )
        if cli.num_samples < 20:
            p.error("--parent-adjustment nonlinear requires at least 20 samples.")
    if cli.causal == "shuffled" and (cli.num_samples < 2 or not 0 <= cli.shuffle_seed < 2**32):
        p.error("--causal shuffled requires at least two samples and a shuffle seed in [0, 2**32).")
    if cli.n_null < 0:
        p.error("--n-null must be nonnegative.")

    if cli.probe_dim != PROBE_DIM_AUTO:
        try:
            cli.probe_dim = int(cli.probe_dim)
        except ValueError:
            p.error(f"--probe-dim must be an integer or '{PROBE_DIM_AUTO}' (got {cli.probe_dim!r})")

    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    if cli.causal == "shuffled":
        from eval.marginal_independence import MarginalShuffledDataset

        source_dataset = build_synthetic_test_set(
            load_run_args(cli.run_dir),
            cli.num_samples,
            causal=True,
            cache=False,
        )
        dataset = MarginalShuffledDataset(source_dataset, seed=cli.shuffle_seed)
    else:
        dataset = build_synthetic_test_set(load_run_args(cli.run_dir), cli.num_samples, causal=cli.causal == "match")
    common = dict(
        dataset=dataset,
        poolings=poolings,
        level=cli.level,
        seeds=seeds,
        n_null=cli.n_null,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        device=cli.device,
        checkpoint=cli.checkpoint,
        probe_dim=cli.probe_dim,
        with_dci=cli.with_dci,
        with_leakage=not cli.no_leakage,
        dci_max_codes=cli.dci_max_codes,
        probe_kind=cli.probe_kind,
        n_jobs=cli.n_jobs,
        factor_pooling=cli.factor_pooling,
        causal=cli.causal,
        parent_adjustment=cli.parent_adjustment,
        parent_cache={} if cli.parent_adjustment == "nonlinear" else None,
        freeze_content_mask=cli.freeze_content_mask,
    )
    logger.info("Scoring checkpoint ...")
    # init_seed=0 for the checkpoint too: strict=False leaves any unmatched parameter at
    # its random init, so an unseeded load is not quite deterministic either.
    res = score_run(cli.run_dir, name=cli.name, random_init=False, init_seed=0, **common)
    floor = floor_std = None
    if not cli.no_floor:
        # Several untrained draws, each seeded, averaged into one floor. A single draw is a
        # single random projection whose noise lands in every 'learned' number unbounded.
        draws = []
        for seed in range(max(1, cli.floor_seeds)):
            logger.info("Scoring untrained twin (floor draw %d/%d) ...", seed + 1, max(1, cli.floor_seeds))
            draws.append(
                score_run(
                    cli.run_dir,
                    name=f"{cli.name or 'run'}-floor-s{seed}",
                    random_init=True,
                    init_seed=seed,
                    **common,
                )
            )
        floor, floor_std = mean_std_structs(draws)
        floor["name"] = (cli.name or "run") + "-floor"
        floor["floor_seeds"] = len(draws)

    print_report(res, floor=floor, floor_std=floor_std, with_dci=cli.with_dci)
    if cli.out:
        write_report_json(cli.out, res, floor, floor_std)


if __name__ == "__main__":
    main()
