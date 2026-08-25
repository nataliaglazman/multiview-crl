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
   raw column is there so a saturated floor is visible rather than silent.
2. The MCC pooling ladder, gap → stats → patch, with floors.  A single rung is not
   interpretable: stats cannot express position (it is permutation-invariant over
   voxels) and patch sits on a floor of ~0.86, so only the SHAPE across rungs carries
   the "where does the information live" signal.
3. DCI per scope, when ``--with-dci`` is passed, with the code count beside each row so
   two scopes are never silently compared at different normalisations.
4. A verdict block that refuses to report a metric whose learned part is inside its own
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

The scoring layer is torch-free and unit-testable: ``--self-test`` runs it on planted
numpy data with no checkpoint and no GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
from joblib import Parallel, delayed

from eval.identifiability_metrics import block_mcc, cv_probe_r2
from eval.run_dci_compare import (
    _CONTENT,
    FACTOR_POOLING,
    PROBE_DIM_AUTO,
    _auto_probe_dim,
    _block_array,
    _resolve_key,
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


def per_factor_scores(reprs, level, gt, names, seeds, n_null, rng, kind="ridge", n_jobs=1, factor_pooling="assigned"):
    """R² gap per factor at its assigned pooling, plus that pooling's key.

    One factor is scored under exactly one pooling — the one ``FACTOR_POOLING`` says can
    physically expose it — so this is never a max over noisy pooling estimates.  Returns
    ``{name: {"r2": gap, "pooling": key}}``.

    The permutations are drawn up-front, in factor order, so the shared ``rng`` is
    consumed deterministically no matter which worker finishes first — the null floor is
    identical whatever ``n_jobs`` is set to.
    """
    avail = set(reprs.keys())
    tasks = []
    for j, name in enumerate(names):
        # ``factor_pooling`` overrides FACTOR_POOLING's per-factor routing and scores every
        # factor on one axis.  The routing is the honest default (each factor read where it
        # can physically appear, fixed in advance so the headline is not a max over
        # poolings); the override exists to ask the different question "how does this
        # factor read at THIS pooling", e.g. to compare all factors on one rung.
        want = FACTOR_POOLING.get(name, "stats") if factor_pooling == "assigned" else factor_pooling
        pkey = _resolve_key(want, avail)
        X = _block_array(reprs, pkey, level, _CONTENT)
        if X is None or not X.shape[1]:
            continue
        perms = [rng.permutation(gt.shape[0]) for _ in range(n_null)]
        tasks.append((name, pkey, X, gt[:, j], perms))

    def _one(X, y, perms):
        real = cv_probe_r2(X, y, seeds=seeds, kind=kind)["mean"]
        nulls = [cv_probe_r2(X, y[p], seeds=seeds, kind=kind)["mean"] for p in perms]
        return real, (float(np.mean(nulls)) if nulls else float("nan"))

    results = Parallel(n_jobs=n_jobs)(delayed(_one)(X, y, perms) for _n, _p, X, y, perms in tasks)
    return {
        name: {"r2": real - null, "r2_raw": real, "pooling": pkey}
        for (name, pkey, _X, _y, _perms), (real, null) in zip(tasks, results)
    }


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


def _delta(a, b):
    a, b = _f(a), _f(b)
    return a - b if np.isfinite(a) and np.isfinite(b) else float("nan")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _n(x, nd=3):
    v = _f(x)
    return f"{v:+.{nd}f}" if np.isfinite(v) else "   -  "


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


def print_report(res, floor=None, with_dci=False):
    """The whole report.  ``floor`` is the same structure scored on an untrained twin."""
    has_floor = floor is not None
    print()
    print("=" * 92)
    print(f"  IDENTIFIABILITY REPORT — {res['name']}")
    print(f"  N={res['n_samples']}  level={res['level']}  poolings={res['poolings']}  probe-dim={res['probe_dim']}")
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

    fw = max([len(k) for k in pf] + [14])
    print()
    print(_rule("1. PER FACTOR"))
    print(
        f"   {'factor':<{fw}s} {'pool':<6s} {'R2 gap':>7s} {'learned':>8s}          " f"{'MCC raw':>8s} {'learned':>8s}"
    )
    for name, d in pf.items():
        r2d = _delta(d["r2"], (fpf.get(name) or {}).get("r2")) if has_floor else float("nan")
        m_raw = mpf.get(name, float("nan"))
        mccd = _delta(m_raw, fmpf.get(name)) if has_floor else float("nan")
        print(
            f"   {name:<{fw}s} {d['pooling']:<6s} {_n(d['r2'])} {_n(r2d)} {_bar(r2d)}  "
            f"{_n(m_raw)} {_n(mccd)} {_bar(mccd)}"
        )
    fp = res.get("factor_pooling", "assigned")
    if fp == "assigned":
        print("   R2 gap = null-subtracted, at each factor's own assigned pooling (FACTOR_POOLING).")
    else:
        print(f"   R2 gap = null-subtracted, ALL factors forced to '{fp}' (--factor-pooling), not their")
        print("   assigned pooling. Same axis for every factor; not comparable to an 'assigned' run.")
    print(f"   MCC raw = matched |corr| at {mpool} pooling; block-MCC has no permutation null here,")
    print(f"   which is exactly why its 'learned' column is the only one worth reading.")

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
        print(f"   {key:<8s} {_n(m['mean'])} {_n(m.get('std'))} {_n(d)} {_bar(d)}  {istr}{warn}")

    # 3. dci ------------------------------------------------------------------
    if not (with_dci and res.get("dci")):
        print()
        print(_rule("3. DCI"))
        print("   skipped — pass --with-dci (GBT importance, noticeably slower).")
    else:
        print()
        print(_rule("3. DCI"))
        print("   D = is each code dedicated to few factors.  C = is each factor carried by few codes.")
        print("   Both are basis-dependent, so these are scored on the encoder's OWN channels, never")
        print("   on principal components. Compare a row only with the same row in another model,")
        print("   and only at equal n_codes: C is normalised by log(n_codes).")
        print(f"   {'scope':<14s} {'D raw':>7s} {'D learn':>8s}   {'C raw':>7s} {'C learn':>8s}   {'n_codes':>8s}")
        fdci = (floor or {}).get("dci", {})
        for scope, d in res["dci"].items():
            fd = fdci.get(scope, {})
            dd = _delta(d.get("d_gap"), fd.get("d_gap")) if has_floor else float("nan")
            cd = _delta(d.get("c_gap"), fd.get("c_gap")) if has_floor else float("nan")
            nc = _f(d.get("n_codes"))
            ncs = f"{int(nc):>8d}" if np.isfinite(nc) else "       ?"
            print(f"   {scope:<14s} {_n(d.get('d_gap'))} {_n(dd)}   {_n(d.get('c_gap'))} {_n(cd)}   {ncs}")
        ncs = {_f(d.get("n_codes")) for d in res["dci"].values() if np.isfinite(_f(d.get("n_codes")))}
        if len(ncs) > 1:
            print("   ! rows differ in n_codes — they are on different scales. Do not compare them.")

    # 4. verdict --------------------------------------------------------------
    print()
    print(_rule("4. VERDICT"))
    if not has_floor:
        print("   No floor measured — no verdict. Re-run without --no-floor.")
        print()
        return
    r2_learned = [_delta(d["r2"], (fpf.get(name) or {}).get("r2")) for name, d in pf.items()]
    r2_learned = [v for v in r2_learned if np.isfinite(v)]
    mean_r2 = float(np.mean(r2_learned)) if r2_learned else float("nan")
    resolved = [name for name, d in pf.items() if abs(_delta(d["r2"], (fpf.get(name) or {}).get("r2"))) > NOISE_FLOOR]
    print(f"   mean learned R2 over {len(r2_learned)} factors: {_n(mean_r2)}")
    if np.isfinite(mean_r2) and abs(mean_r2) <= NOISE_FLOOR:
        print(f"   INSIDE NOISE (|delta| <= {NOISE_FLOOR}). This checkpoint is not distinguishable")
        print("   from its own untrained architecture on aggregate factor recovery.")
    print(f"   factors resolved above the {NOISE_FLOOR} noise floor: " f"{', '.join(resolved) if resolved else 'NONE'}")
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
    dci_max_codes=4096,
    random_init=False,
    probe_kind="ridge",
    n_jobs=1,
    factor_pooling="assigned",
    name=None,
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

    model, _args, device = load_model_from_run_dir(
        run_dir,
        None if random_init else _resolve_checkpoint(run_dir, checkpoint),
        device,
        random_init=random_init,
    )
    reprs, gt_content, info = {}, None, None
    for key, value in poolings:
        level_data, gc, _gs1, _gs2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value
        )
        reprs[key] = level_data
        if gt_content is None:
            gt_content = gc
        if info is None and level in level_data:
            info = level_data[level][4]
    del model
    if info is None:
        raise RuntimeError(f"level {level} not found in encoder outputs for {run_dir}")

    names = info["content_names"]
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
        "per_factor": per_factor_scores(
            probed, level, gt_content, names, seeds, n_null, rng, probe_kind, n_jobs, factor_pooling
        ),
        "mcc": mcc_ladder(probed, level, gt_content, seeds, probe_kind, names),
        "mcc_per_factor_pooling": "patch" if "patch" in probed else _resolve_key("stats", set(probed)),
    }
    if with_dci:
        avail = set(dci_reprs.keys())
        dci = {}
        # Content block x content factors at each pooling — the whole report is
        # content-side, so mixing the style block in here would make the DCI rows answer a
        # different question from every other row.  `dci_reprs`, not `probed`: D/C are
        # basis-dependent and only mean something on the encoder's own channels.
        for scope, pooling_key in (("content@stats", "stats"), ("content@patch", "patch")):
            if pooling_key not in avail:
                continue
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
            dci[scope] = {
                "d_gap": d.get("dci_d_gap"),
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
    """
    rng = np.random.RandomState(0)
    n, n_fac, n_ch = 400, 4, 12
    names = ["brain_size", "ventricle_size", "lesion_x", "gain"][:n_fac]
    gt = rng.randn(n, n_fac)
    A = rng.randn(n_fac, n_ch)
    signal = gt @ A + 0.1 * rng.randn(n, n_ch)
    noise = rng.randn(n, n_ch)

    def _mk(X):
        # (content, style, content_v2, style_v2, info) at level 0, one entry per pooling.
        info = {
            "content_names": names,
            "style_names": [],
            "n_content_channels": X.shape[1],
            "n_style_channels": 0,
            "has_split": False,
            "pooling": "gap",
            "level": 0,
        }
        return {p: {0: (X, None, None, None, info)} for p in ("gap", "stats", "patch")}

    out = {}
    for label, X in (("signal", signal), ("noise", noise)):
        r = np.random.RandomState(0)
        res = {
            "name": label,
            "level": 0,
            "n_samples": n,
            "poolings": "gap,stats,patch",
            "probe_dim": 0,
            "per_factor": per_factor_scores(_mk(X), 0, gt, names, (0, 1), 2, r),
            "mcc": mcc_ladder(_mk(X), 0, gt, (0, 1), names=names),
            "mcc_per_factor_pooling": "patch",
        }
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
    # --factor-pooling forces every factor onto one rung instead of its assigned one.
    # Asserted against FACTOR_POOLING itself rather than against hardcoded pooling names:
    # the table is a research parameter that gets retuned, and a test that pins its values
    # fails on every retune while testing nothing about the routing mechanism.
    _forced = per_factor_scores(_rp, 0, _gt, _nm, (0,), 0, np.random.RandomState(7), factor_pooling="patch")
    assert {d["pooling"] for d in _forced.values()} == {"patch"}, _forced
    for _k, _d in _s1.items():
        assert _d["pooling"] == FACTOR_POOLING[_k], f"{_k} routed to {_d['pooling']}, table says {FACTOR_POOLING[_k]}"

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
    print("  self-test: DCI reads the unreduced block (probes still reduced to 64) — basis OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Run directory to score.")
    p.add_argument("--name", default=None, help="Label for the report (default: basename of --run-dir).")
    p.add_argument(
        "--checkpoint",
        default="vqvae_model.pt",
        help="Checkpoint filename inside the run dir (same default as run_dci_compare "
        "--checkpoint-name). Use vqvae_best.pt for the best-by-loss copy; whichever you pick, use "
        "the same one for every run you intend to compare.",
    )
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--poolings", default="gap,stats,8x8x8", help="Comma list: gap, stats, DxHxW.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-null", type=int, default=3, help="Permutations for the label-permutation null.")
    p.add_argument("--probe-dim", default=PROBE_DIM_AUTO, help="'auto', or an integer width, or 0.")
    p.add_argument("--probe-kind", default="ridge", choices=("ridge", "kernel", "mlp"))
    p.add_argument("--with-dci", action="store_true", help="Also score DCI (GBT — slow).")
    p.add_argument(
        "--dci-max-codes",
        type=int,
        default=4096,
        help="Cap DCI width by selecting highest-variance codes (0 = no cap). Never a rotation.",
    )
    p.add_argument(
        "--no-floor", action="store_true", help="Skip the untrained twin. The report then refuses to give a verdict."
    )
    p.add_argument(
        "--causal",
        default="match",
        choices=("match", "iid"),
        help="'match' for aggregate ranking, 'iid' for per-factor attribution.",
    )
    p.add_argument(
        "--factor-pooling",
        default="assigned",
        choices=("assigned", "gap", "stats", "patch"),
        help="Which pooling each factor's R2 is read at. 'assigned' (default) uses "
        "FACTOR_POOLING — each factor read where it can physically appear, fixed in advance so "
        "the headline is never a max over poolings. Naming one pooling scores EVERY factor "
        "there, which answers a different question ('how does this factor read at this rung') "
        "and is the honest way to put all factors on one axis. The pooling must be in "
        "--poolings or it falls back. Patch has a high floor, so read the learned column.",
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
    if not cli.run_dir:
        p.error("--run-dir is required (or pass --self-test)")

    if cli.probe_dim != PROBE_DIM_AUTO:
        try:
            cli.probe_dim = int(cli.probe_dim)
        except ValueError:
            p.error(f"--probe-dim must be an integer or '{PROBE_DIM_AUTO}' (got {cli.probe_dim!r})")

    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

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
        dci_max_codes=cli.dci_max_codes,
        probe_kind=cli.probe_kind,
        n_jobs=cli.n_jobs,
        factor_pooling=cli.factor_pooling,
    )
    logger.info("Scoring checkpoint ...")
    res = score_run(cli.run_dir, name=cli.name, random_init=False, **common)
    floor = None
    if not cli.no_floor:
        logger.info("Scoring untrained twin (the floor) ...")
        floor = score_run(cli.run_dir, name=(cli.name or "run") + "-floor", random_init=True, **common)

    print_report(res, floor=floor, with_dci=cli.with_dci)
    if cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
        with open(cli.out, "w") as fh:
            json.dump({"run": res, "floor": floor}, fh, indent=2, default=float)
        logger.info("Wrote %s", cli.out)


if __name__ == "__main__":
    main()
