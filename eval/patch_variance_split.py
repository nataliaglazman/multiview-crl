#!/usr/bin/env python
"""What --patch-center-mode keeps and what it deletes, per content channel.

    python -m eval.patch_variance_split --run-dir results/synthetic/RUN --causal iid

A patch feature z[s, p] for subject s at position p splits into three orthogonal parts:

    z[s, p] = mean + a[p] + b[s] + e[s, p]

    a[p]     what is typical AT THAT POSITION across subjects -- shared anatomy
    b[s]     what is typical FOR THAT SUBJECT across positions -- a spatially flat code
    e[s, p]  this subject, specifically here -- the subject x position interaction

The three centering modes each keep a different suffix of that sum:

    none       mean + a + b + e
    position   b + e          (removes the across-subject mean at each position)
    double     e              (also removes each subject's mean over positions)

So switching to ``double`` deletes ``b`` from the objective. That is the documented
objection to it -- global factors live in ``b`` -- and this script measures the cost
channel by channel instead of guessing it: a channel whose variance is nearly all ``b``
contributes almost nothing after ``double``, and one that is nearly all ``e`` is unchanged.

It also reports each component's CROSS-VIEW correlation, which is what the Barlow Twins
on_diag/sim terms actually see. The interaction's correlation is the pos_sim to expect
under ``double``, so the drop is predicted here rather than discovered after a training run.

Extraction reuses ``eval.bt_term_balance`` -- same grid, foreground mask, content channels
and pooling the run trains with -- so these numbers refer to the real objective. The
decomposition is pure numpy and exact (the three sums of squares add to the total, asserted
at runtime); ``--self-test`` runs it on planted components with no torch and no checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

# A channel keeping less than this share of its variance as interaction is reported as
# effectively deleted by `double`. Not a threshold the analysis depends on -- the shares
# are printed -- only the wording of the count in the verdict.
ANNIHILATED_BELOW = 0.05


def components(z):
    """Return (a[p], b[s], e[s,p], grand) for one channel/view, z of shape (S, P)."""
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    grand = z.mean()
    a = z.mean(axis=0) - grand
    b = z.mean(axis=1) - grand
    e = z - grand - a[None, :] - b[:, None]
    return a, b, e, grand


def split_variance(z):
    """Sums of squares for the three parts. They add to the total exactly."""
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    subjects, positions = z.shape
    a, b, e, grand = components(z)
    ss = {
        "position": float(subjects * np.square(a).sum()),
        "subject": float(positions * np.square(b).sum()),
        "interaction": float(np.square(e).sum()),
    }
    total = float(np.square(z - grand).sum())
    if total > 1e-12 and abs(sum(ss.values()) - total) / total > 1e-6:
        raise AssertionError(f"Variance split does not close: {sum(ss.values())} vs {total}")
    ss["total"] = total
    return ss


def correlate(x, y):
    """Pearson correlation over all entries, or None when either side is constant."""
    import numpy as np

    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    xs, ys = x - x.mean(), y - y.mean()
    den = float(np.sqrt(np.square(xs).sum() * np.square(ys).sum()))
    return float((xs * ys).sum() / den) if den > 1e-20 else None


def analyse(hz):
    """Per-channel variance split and per-component cross-view correlation.

    hz is (2, S, C, P): two views, subjects, content channels, patch positions.
    """
    import numpy as np

    hz = np.asarray(hz, dtype=np.float64)
    if hz.ndim != 4 or hz.shape[0] != 2:
        raise ValueError(f"Expected (2, S, C, P), got {hz.shape}")
    if hz.shape[3] < 2:
        raise ValueError("Need at least two patch positions; with P=1 there is no interaction")
    if not np.isfinite(hz).all():
        raise ValueError("Non-finite patch features")
    channels = []
    for channel in range(hz.shape[2]):
        per_view = [split_variance(hz[view, :, channel, :]) for view in range(2)]
        total = sum(v["total"] for v in per_view)
        shares = {
            part: (sum(v[part] for v in per_view) / total if total > 1e-20 else None)
            for part in ("position", "subject", "interaction")
        }
        parts = [components(hz[view, :, channel, :]) for view in range(2)]
        channels.append(
            {
                "channel": channel,
                "shares": shares,
                "variance": total,
                "cross_view": {
                    name: correlate(parts[0][i], parts[1][i])
                    for i, name in enumerate(("position", "subject", "interaction"))
                },
            }
        )
    # Pooled over channels, weighted by variance -- the objective sees all channels at once,
    # so an unweighted mean would let a near-dead channel count like a live one.
    weights = np.array([c["variance"] for c in channels], dtype=float)
    pooled_shares = {}
    for part in ("position", "subject", "interaction"):
        values = np.array([c["shares"][part] or 0.0 for c in channels], dtype=float)
        pooled_shares[part] = float((values * weights).sum() / weights.sum()) if weights.sum() > 0 else None
    pooled_cross = {
        name: correlate(
            np.concatenate([components(hz[0, :, c, :])[i].ravel() for c in range(hz.shape[2])]),
            np.concatenate([components(hz[1, :, c, :])[i].ravel() for c in range(hz.shape[2])]),
        )
        for i, name in enumerate(("position", "subject", "interaction"))
    }
    return {
        "n_subjects": int(hz.shape[1]),
        "n_channels": int(hz.shape[2]),
        "n_positions": int(hz.shape[3]),
        "channels": channels,
        "pooled_shares": pooled_shares,
        "pooled_cross_view": pooled_cross,
    }


def split_by_lesion(hz, lesion_cells):
    """Cross-view correlation of e[s,p] inside vs outside lesion-containing patches.

    ``lesion_cells`` is a (S, P) boolean: does this subject's lesion touch this patch. The
    interaction is the component ``double`` keeps, and it aligns across views at ~0.9 in
    aggregate. This asks whether the lesion's own cells are the exception, which removes the
    aggregate-vs-lesion caveat on that comparison: everything here is e[s,p], the same
    component, differing only in where it is measured.
    """
    import numpy as np

    hz = np.asarray(hz, dtype=np.float64)
    cells = np.asarray(lesion_cells, dtype=bool)
    if cells.shape != (hz.shape[1], hz.shape[3]):
        raise ValueError(f"lesion mask {cells.shape} does not match (S, P) = {(hz.shape[1], hz.shape[3])}")
    if not cells.any():
        return None
    per_channel, stacked = [], {"lesion": [[], []], "other": [[], []]}
    for channel in range(hz.shape[2]):
        parts = [components(hz[view, :, channel, :])[2] for view in range(2)]
        row = {"channel": channel}
        for name, selector in (("lesion", cells), ("other", ~cells)):
            a, b = parts[0][selector], parts[1][selector]
            row[name] = correlate(a, b)
            row[f"n_{name}"] = int(selector.sum())
            # Residual ENERGY per cell, so a near-zero response is visible as such rather
            # than showing up as a confident correlation between two tiny vectors.
            row[f"rms_{name}"] = float(np.sqrt(np.mean(np.square(np.concatenate([a, b])))))
            stacked[name][0].append(a)
            stacked[name][1].append(b)
        per_channel.append(row)
    pooled = {
        name: {
            "cross_view": correlate(np.concatenate(v[0]), np.concatenate(v[1])),
            "rms": float(np.sqrt(np.mean(np.square(np.concatenate(v[0] + v[1]))))),
        }
        for name, v in stacked.items()
    }
    pooled["n_lesion_cells"] = int(cells.sum())
    pooled["n_other_cells"] = int((~cells).sum())
    pooled["lesion_cell_fraction"] = float(cells.mean())
    return {"channels": per_channel, "pooled": pooled}


def _sign_p(diffs):
    """Two-sided exact sign test that the paired differences are centred on zero."""
    import math

    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def within_position_split(hz, lesion_cells, min_each=3, draws=20, seed=0):
    """Lesion vs no lesion AT THE SAME PATCH POSITION.

    Comparing lesion cells against every other cell confounds the lesion with where it is
    put: lesions land in white matter by construction, and a homogeneous region may align
    across views differently from a boundary one whether or not a lesion is present. The
    lesion moves between subjects, so each position is a lesion cell for some subjects and
    not others, and that gives a matched control for free -- same position, same anatomy,
    lesion present or absent. Anatomy cannot explain a difference measured this way.

    Each channel's residual is divided by its own global RMS first, so pooling channels
    within a position does not let the widest-swinging channel decide the correlation.

    The control is SIZE-MATCHED, and it has to be. Using every lesion-free subject at a
    position gives the control side far more cells than the lesion side, so its correlation
    is less attenuated toward zero, and the difference carries a systematic negative bias
    (~-0.05 measured on planted data with no lesion effect at all). Systematic, not random:
    it shifts every position the same way, so a sign test against zero rejects under the
    null and cannot arbitrate. Instead the control is resampled to exactly the lesion
    group's size, averaged over ``draws`` draws, which removes the bias by construction and
    leaves the sign test valid.
    """
    import numpy as np

    hz = np.asarray(hz, dtype=np.float64)
    cells = np.asarray(lesion_cells, dtype=bool)
    if cells.shape != (hz.shape[1], hz.shape[3]):
        raise ValueError(f"lesion mask {cells.shape} does not match (S, P) = {(hz.shape[1], hz.shape[3])}")
    residual = np.stack(
        [np.stack([components(hz[view, :, c, :])[2] for c in range(hz.shape[2])], axis=1) for view in range(2)]
    )  # (2, S, C, P)
    scale = np.sqrt(np.mean(np.square(residual), axis=(0, 1, 3)))  # per channel
    residual = residual / np.where(scale > 1e-20, scale, 1.0)[None, None, :, None]

    rng = np.random.default_rng(seed)
    rows = []
    for position in range(hz.shape[3]):
        has = cells[:, position]
        pool = np.flatnonzero(~has)
        n_lesion = int(has.sum())
        if n_lesion < min_each or len(pool) < min_each:
            continue
        entry = {"position": position, "n_lesion": n_lesion, "n_control_available": int(len(pool))}
        entry["lesion"] = correlate(residual[0][has, :, position], residual[1][has, :, position])
        # Size-matched: draw exactly as many lesion-free subjects as there are lesion ones,
        # so both correlations are attenuated by the same sample size.
        size = min(n_lesion, len(pool))
        drawn = []
        for _ in range(draws):
            pick = rng.choice(pool, size=size, replace=False)
            value = correlate(residual[0][pick, :, position], residual[1][pick, :, position])
            if value is not None:
                drawn.append(value)
        entry["control"] = float(np.mean(drawn)) if drawn else None
        entry["n_control_matched"] = size
        # Kept for reference only: the unmatched control is what carries the bias.
        entry["control_all"] = correlate(residual[0][~has, :, position], residual[1][~has, :, position])
        if entry["lesion"] is not None and entry["control"] is not None:
            entry["difference"] = entry["lesion"] - entry["control"]
            rows.append(entry)
    if not rows:
        return None
    weights = np.array([r["n_lesion"] for r in rows], dtype=float)
    diffs = [r["difference"] for r in rows]
    return {
        "positions": rows,
        "n_positions_used": len(rows),
        "n_positions_total": int(hz.shape[3]),
        "min_each": min_each,
        "lesion": float((np.array([r["lesion"] for r in rows]) * weights).sum() / weights.sum()),
        "control": float((np.array([r["control"] for r in rows]) * weights).sum() / weights.sum()),
        "difference_weighted_mean": float((np.array(diffs) * weights).sum() / weights.sum()),
        "difference_median": float(np.median(diffs)),
        "sign_p": _sign_p(diffs),
    }


def fmt_pct(v):
    return "  n/a" if v is None else f"{100 * v:5.1f}%"


def fmt_r(v):
    return " n/a " if v is None else f"{v:+.3f}"


def print_report(result, center_mode=None):
    print("\n" + "=" * 92)
    print("PATCH VARIANCE SPLIT   z[s,p] = mean + a[p] + b[s] + e[s,p]")
    print(
        f"  {result['n_subjects']} subjects x {result['n_channels']} content channels"
        f" x {result['n_positions']} patch positions"
        + (f"   (run trains with center_mode={center_mode})" if center_mode else "")
    )
    print("=" * 92)
    print("\n  a[p] = anatomy (removed by 'position')   b[s] = flat per-subject code (also removed")
    print("  by 'double')   e[s,p] = subject x position interaction (all that 'double' leaves)")

    print("\n  SHARE OF VARIANCE, per channel                    CROSS-VIEW CORRELATION")
    print("  chan      a[p]     b[s]   e[s,p]   var share      a[p]     b[s]   e[s,p]   under double")
    order = sorted(result["channels"], key=lambda c: -c["variance"])
    total_var = sum(c["variance"] for c in result["channels"]) or 1.0
    for c in order:
        interaction = c["shares"]["interaction"]
        note = "DELETED" if interaction is not None and interaction < ANNIHILATED_BELOW else ""
        print(
            f"  {c['channel']:4d}  {fmt_pct(c['shares']['position'])}  {fmt_pct(c['shares']['subject'])}"
            f"  {fmt_pct(c['shares']['interaction'])}     {fmt_pct(c['variance'] / total_var)}"
            f"     {fmt_r(c['cross_view']['position'])}   {fmt_r(c['cross_view']['subject'])}"
            f"   {fmt_r(c['cross_view']['interaction'])}   {note}"
        )
    pooled = result["pooled_shares"]
    cross = result["pooled_cross_view"]
    print(
        f"  all   {fmt_pct(pooled['position'])}  {fmt_pct(pooled['subject'])}"
        f"  {fmt_pct(pooled['interaction'])}     100.0%"
        f"     {fmt_r(cross['position'])}   {fmt_r(cross['subject'])}   {fmt_r(cross['interaction'])}"
    )

    dead = [c["channel"] for c in result["channels"] if (c["shares"]["interaction"] or 0) < ANNIHILATED_BELOW]
    print(
        f"\n  Under 'double' the objective keeps {fmt_pct(pooled['interaction']).strip()} of the current patch"
        f" variance, and {len(dead)}/{result['n_channels']} channels"
    )
    print(f"  fall below {100 * ANNIHILATED_BELOW:.0f}% of their own{f': {dead}' if dead else ''}.")
    if cross["interaction"] is not None:
        print(
            f"\n  Expect pos_sim to fall to about {cross['interaction']:+.3f} under 'double': that is the"
            " cross-view correlation"
        )
        print("  of the part that survives, which is what BT would then be scored on.")
        if cross["subject"] is not None:
            print(f"  It currently sits near the b[s] correlation of {cross['subject']:+.3f}, which 'double' deletes.")
    print("\n  Read it as a trade: a large b[s] share means 'double' costs you real signal (global")
    print("  factors live there), and a large e[s,p] share means it costs little. Neither is free.")

    lesion = result.get("lesion_split")
    if lesion:
        p = lesion["pooled"]
        print("\n" + "-" * 92)
        print("e[s,p] CROSS-VIEW CORRELATION, LESION PATCHES vs THE REST")
        print("-" * 92)
        print(
            f"  lesion-containing cells   r {fmt_r(p['lesion']['cross_view'])}   rms {p['lesion']['rms']:.4g}"
            f"   n {p['n_lesion_cells']}  ({100 * p['lesion_cell_fraction']:.2f}% of cells)"
        )
        print(
            f"  every other cell          r {fmt_r(p['other']['cross_view'])}   rms {p['other']['rms']:.4g}"
            f"   n {p['n_other_cells']}"
        )
        both = p["lesion"]["cross_view"], p["other"]["cross_view"]
        if None not in both:
            print(f"\n  gap: {both[0] - both[1]:+.3f}")
            if both[0] < both[1] - 0.15:
                print("  The lesion's own cells align WORSE than the rest of the same component. Localized")
                print("  structure is not the problem; something specific to the lesion is.")
            elif both[0] > both[1] - 0.05:
                print("  The lesion's cells align as well as the rest. Whatever limits lesion recovery is")
                print("  NOT a cross-view disagreement at the lesion site; look at magnitude and probes.")
        print("  Check rms before reading r: a near-zero residual makes the correlation meaningless.")

    matched = result.get("within_position")
    if matched:
        print("\n  MATCHED WITHIN PATCH POSITION   same location, lesion present vs absent")
        print(f"    positions used            {matched['n_positions_used']}/{matched['n_positions_total']}")
        print(f"    lesion present            r {matched['lesion']:+.3f}")
        print(f"    lesion absent, same pos   r {matched['control']:+.3f}")
        print(
            f"    paired difference         {matched['difference_weighted_mean']:+.3f}"
            f"   (median {matched['difference_median']:+.3f}, sign p {matched['sign_p']:.1e})"
        )
        naive = None
        if lesion and None not in (lesion["pooled"]["lesion"]["cross_view"], lesion["pooled"]["other"]["cross_view"]):
            naive = lesion["pooled"]["lesion"]["cross_view"] - lesion["pooled"]["other"]["cross_view"]
        matched_gap = matched["difference_weighted_mean"]
        if naive is not None:
            print(f"\n    unmatched gap {naive:+.3f}  ->  matched gap {matched_gap:+.3f}")
            shrunk = abs(matched_gap) < 0.5 * abs(naive)
            if shrunk:
                print("    Most of the unmatched gap was WHERE lesions go, not the lesions. Lesions sit in")
                print("    white matter by construction, and those positions align differently anyway.")
        if matched["sign_p"] > 0.05 or abs(matched_gap) < 0.02:
            print("    Matched, the lesion does not measurably change alignment at its own location.")
        elif matched_gap < 0:
            print("    Matched, the lesion still degrades alignment at its own location: this is the")
            print("    lesion, not its anatomy.")
        print("    The control is resampled to the lesion group's size, so both sides are attenuated")
        print("    equally and the sign test is against a null that really is centred on zero.")


# main()'s imports are deferred so --self-test needs no torch, which also means a rename in
# one of those modules would not surface until a run had already loaded a checkpoint on the
# cluster. This checks the names exist by parsing, no import required.
REQUIRED_IMPORTS = {
    "eval/bt_term_balance.py": ("_as_views", "foreground_keep"),
    "eval/dci.py": ("_extract_synthetic_representations",),
    "eval/run_dci_compare.py": ("_CONTENT", "_CONTENT_V2"),
    "eval/run_dci_synthetic.py": ("build_synthetic_test_set", "load_model_from_run_dir", "load_run_args"),
}


def module_bindings(path):
    """Every name a module binds at top level: defs, classes, assignments, imports."""
    import ast

    names = set()
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            stack = list(targets)
            while stack:
                target = stack.pop()
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    # _CONTENT, _STYLE = 0, 1 binds both, and missing this is what let a
                    # wrong import module reach the cluster.
                    stack.extend(target.elts)
    return names


def _check_imports(root=None):
    root = Path(root or Path(__file__).resolve().parent.parent)
    results = []
    for path, names in REQUIRED_IMPORTS.items():
        full = root / path
        if not full.exists():
            results.extend((f"{path}::{n}", False) for n in names)
            continue
        bound = module_bindings(full)
        results.extend((f"{path}::{n}", n in bound) for n in names)
    return results


def lesion_patch_mask(dataset, grid, n_subjects, keep=None):
    """(S, P) boolean: does subject s's rendered lesion touch patch p?

    Subject order is the dataset's own index order, which is what
    ``_extract_synthetic_representations`` iterates with a sequential sampler, so row s here
    is the same subject as row s of the features. Returns None if the dataset cannot render
    a lesion (field-lesion modes have no sphere to pool).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    inner = getattr(dataset, "_inner", None)
    renderer = getattr(inner, "renderer", None)
    if renderer is None or not hasattr(renderer, "render_structure"):
        return None
    rows = []
    with torch.inference_mode():
        for idx in range(n_subjects):
            _v1, _v2, lat = inner[idx]
            _tissue, lesion = renderer.render_structure(
                lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=inner.clean_content
            )
            volume = lesion.reshape(1, 1, *lesion.shape[-3:]).float()
            # Mean pooling gives each patch's lesion volume fraction; any nonzero fraction
            # means the lesion reaches into that patch.
            rows.append((F.adaptive_avg_pool3d(volume, tuple(grid)).flatten() > 0).cpu().numpy())
    cells = np.stack(rows)
    return cells[:, keep.cpu().numpy()] if keep is not None else cells


def _raises(fn, *args):
    try:
        fn(*args)
    except (ValueError, AssertionError):
        return True
    return False


def _self_test():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"self-test needs numpy ({exc})")

    import_checks = _check_imports()
    print("self-test (deferred imports resolve)")
    for label, ok in import_checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print()

    rng = np.random.default_rng(0)
    subjects, positions = 60, 24

    # A raw (S, P) Gaussian is NOT a pure interaction -- it carries its own position and
    # subject marginals -- so each planted component is projected onto its own subspace
    # first and scaled to contribute exactly 1 to the sum of squares. Then a mixture's
    # shares are its squared coefficients, because the three subspaces are orthogonal.
    def plant_position():
        a = rng.normal(size=positions)
        a -= a.mean()
        return a / np.sqrt(subjects * np.square(a).sum())

    def plant_subject():
        b = rng.normal(size=subjects)
        b -= b.mean()
        return b / np.sqrt(positions * np.square(b).sum())

    def plant_interaction():
        e = rng.normal(size=(subjects, positions))
        e = e - e.mean(axis=0)[None, :] - e.mean(axis=1)[:, None] + e.mean()
        return e / np.sqrt(np.square(e).sum())

    a, b, e, other = plant_position(), plant_subject(), plant_interaction(), plant_interaction()
    # Four channels: pure anatomy, pure flat-subject code, pure interaction, 50/30/20 mixture.
    hz = np.zeros((2, subjects, 4, positions))
    for view in range(2):
        hz[view, :, 0, :] = a[None, :] * 10.0
        hz[view, :, 1, :] = b[:, None] * 10.0
        hz[view, :, 2, :] = e * 10.0
        hz[view, :, 3, :] = (a[None, :] * np.sqrt(0.50) + b[:, None] * np.sqrt(0.30) + e * np.sqrt(0.20)) * 30.0
    # Decorrelate view 1's interaction only, so the components are read independently.
    # Equal-weight mix of two orthonormal interactions correlates at 1/sqrt(2) ~ 0.707.
    hz[1, :, 2, :] = (e + other) / np.sqrt(2.0) * 10.0
    got = analyse(hz)
    ch = {c["channel"]: c for c in got["channels"]}

    def share(channel, part):
        return ch[channel]["shares"][part]

    checks = [
        ("pure a[p] reads 100% position", abs(share(0, "position") - 1.0) < 1e-6),
        ("pure b[s] reads 100% subject", abs(share(1, "subject") - 1.0) < 1e-6),
        ("pure e reads 100% interaction", abs(share(2, "interaction") - 1.0) < 1e-6),
        ("pure a[p] would be DELETED by double", share(0, "interaction") < ANNIHILATED_BELOW),
        ("pure b[s] would be DELETED by double", share(1, "interaction") < ANNIHILATED_BELOW),
        ("pure e survives double", share(2, "interaction") > 0.99),
        ("mixture recovers ~50% position", abs(share(3, "position") - 0.5) < 0.02),
        ("mixture recovers ~30% subject", abs(share(3, "subject") - 0.3) < 0.02),
        ("mixture recovers ~20% interaction", abs(share(3, "interaction") - 0.2) < 0.02),
        ("shares sum to 1 per channel", all(abs(sum(c["shares"].values()) - 1.0) < 1e-9 for c in got["channels"])),
        ("identical views correlate at +1 on b[s]", abs(ch[1]["cross_view"]["subject"] - 1.0) < 1e-6),
        ("decorrelated interaction reads ~0.7", 0.6 < ch[2]["cross_view"]["interaction"] < 0.8),
        ("pooled shares sum to 1", abs(sum(got["pooled_shares"].values()) - 1.0) < 1e-9),
    ]

    # Lesion split: plant an interaction that agrees across views everywhere EXCEPT the
    # cells a "lesion" occupies, where view 1 sees it with the opposite sign. That is the
    # sign-reversal hypothesis in its purest form, so the split has to separate the two.
    subjects_l, positions_l = 80, 40
    cells = np.zeros((subjects_l, positions_l), dtype=bool)
    for s in range(subjects_l):
        cells[s, (s * 7) % positions_l] = True  # one lesion patch each, moving between subjects
    base = rng.normal(size=(subjects_l, positions_l))
    flipped = base.copy()
    flipped[cells] *= -1.0
    hz_l = np.stack([base[:, None, :], flipped[:, None, :]], axis=0)
    split = split_by_lesion(hz_l, cells)
    got["lesion_split"] = split
    lp = split["pooled"]
    checks += [
        ("lesion cells read anti-aligned", lp["lesion"]["cross_view"] < -0.5),
        ("other cells read aligned", lp["other"]["cross_view"] > 0.9),
        ("lesion cell count is right", lp["n_lesion_cells"] == subjects_l),
        ("cells partition exactly", lp["n_lesion_cells"] + lp["n_other_cells"] == subjects_l * positions_l),
        ("residual rms is reported non-zero", lp["lesion"]["rms"] > 0 and lp["other"]["rms"] > 0),
        ("a mask of the wrong shape is rejected", _raises(split_by_lesion, hz_l, cells[:, :-1])),
        ("an empty mask returns None", split_by_lesion(hz_l, np.zeros_like(cells)) is None),
    ]

    # The confound, planted deliberately: positions differ in how well the two views agree,
    # and lesions are placed ONLY at the poorly-agreeing half. Nothing about the lesion
    # itself changes alignment. The unmatched split must therefore report a large spurious
    # gap and the within-position control must report ~0.
    def build(subjects, positions, rho_by_position, lesion_cells, lesion_penalty=0.0, channels=12):
        # 12 channels, like the real content block: with one channel the per-position
        # correlations are too noisy for a bias of this size to be visible at all.
        x = rng.normal(size=(subjects, channels, positions))
        noise = rng.normal(size=(subjects, channels, positions))
        rho = np.asarray(rho_by_position, dtype=float)[None, None, :]
        if lesion_penalty:
            rho = np.repeat(np.repeat(rho, subjects, axis=0), channels, axis=1).copy()
            rho[np.broadcast_to(lesion_cells[:, None, :], rho.shape)] -= lesion_penalty
        view1 = rho * x + np.sqrt(np.clip(1 - rho**2, 0, 1)) * noise
        return np.stack([x, view1]) * 10.0

    subjects_c, positions_c = 160, 40
    good = np.full(positions_c, 0.95)
    good[: positions_c // 2] = 0.30  # first half agrees poorly
    confounded = np.zeros((subjects_c, positions_c), dtype=bool)
    for s in range(subjects_c):
        confounded[s, s % (positions_c // 2)] = True  # every lesion in the poor half
    spurious = build(subjects_c, positions_c, good, confounded)
    naive_spurious = split_by_lesion(spurious, confounded)["pooled"]
    matched_spurious = within_position_split(spurious, confounded)

    # And the real effect: lesions everywhere, each one genuinely lowering agreement.
    everywhere = np.zeros((subjects_c, positions_c), dtype=bool)
    for s in range(subjects_c):
        everywhere[s, (s * 3) % positions_c] = True
    real = build(subjects_c, positions_c, np.full(positions_c, 0.95), everywhere, lesion_penalty=0.45)
    matched_real = within_position_split(real, everywhere)

    got["within_position"] = matched_spurious
    checks += [
        (
            "confound: unmatched split reports a big spurious gap",
            naive_spurious["lesion"]["cross_view"] - naive_spurious["other"]["cross_view"] < -0.2,
        ),
        ("confound: matched control removes it", abs(matched_spurious["difference_weighted_mean"]) < 0.10),
        ("confound: matched control is not significant", matched_spurious["sign_p"] > 0.05),
        # Structural, not statistical: the size matching is what removes the attenuation
        # bias, and its magnitude is data-dependent (measured between -0.02 and -0.05
        # across planted scenarios), so asserting a threshold on it would be tuning to
        # one scenario. Assert the property that makes the null centred instead.
        (
            "control is drawn at exactly the lesion group's size",
            all(r["n_control_matched"] == r["n_lesion"] for r in matched_spurious["positions"]),
        ),
        ("real effect: matched control still detects it", matched_real["difference_weighted_mean"] < -0.2),
        ("real effect: and calls it significant", matched_real["sign_p"] < 0.01),
        ("positions with too few of either side are skipped", matched_spurious["n_positions_used"] <= positions_c),
        ("min_each is honoured", within_position_split(spurious, confounded, min_each=10_000) is None),
        ("wrong-shaped mask is rejected", _raises(within_position_split, spurious, confounded[:, :-1])),
    ]
    print("self-test (variance split)")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print_report(got, center_mode="position")
    if not all(ok for _, ok in checks + import_checks):
        raise SystemExit("self-test failed")
    print("\nall checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir")
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--causal", choices=("match", "iid"), default="iid")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default=None, help="Also write the per-channel numbers as JSON + CSV")
    p.add_argument(
        "--no-lesion-split",
        action="store_true",
        help="Skip the lesion-patch vs rest comparison; it re-renders every subject's lesion mask",
    )
    p.add_argument("--self-test", action="store_true")
    cli = p.parse_args()
    if cli.self_test:
        _self_test()
        return
    if not cli.run_dir:
        p.error("Need --run-dir, or --self-test")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import os

    import numpy as np

    # Same sources bt_term_balance imports these from, so the extraction is byte-for-byte
    # the path that script measures the BT terms on.
    from eval.bt_term_balance import _as_views, foreground_keep
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    args_ = load_run_args(cli.run_dir)
    grid = getattr(args_, "patch_grid", None)
    if not grid:
        raise SystemExit("This run has no --patch-grid; there are no patch positions to split")
    center_mode = getattr(args_, "patch_center_mode", "none") or "none"
    dataset = build_synthetic_test_set(args_, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)
    levels, _gt, _s1, _s2 = _extract_synthetic_representations(
        model, dataset, device, cli.encode_batch, cli.num_workers, pooling=tuple(grid)
    )
    if cli.level not in levels:
        raise SystemExit(f"Level {cli.level} not in extracted representations: {sorted(levels)}")
    c1, c2 = levels[cli.level][_CONTENT], levels[cli.level][_CONTENT_V2]
    if c1 is None or c2 is None or c1.shape[1] == 0:
        raise SystemExit("No content channels at this level")
    hz = _as_views(c1, c2, int(np.prod(grid)))
    # Stays None unless positions were actually dropped, so the lesion mask below is
    # subset by exactly the same selection that was applied to hz -- and never by a
    # selection that was computed but not used.
    keep = None
    if bool(getattr(args_, "patch_foreground_mask", False)):
        thresh = float(getattr(args_, "patch_foreground_thresh", 0.05))
        candidate = foreground_keep(dataset, tuple(grid), thresh, cli.encode_batch, hz.device)
        if candidate is not None and bool(candidate.any()):
            logger.info("Foreground patches: keeping %d/%d positions.", int(candidate.sum()), int(candidate.numel()))
            hz = hz[..., candidate.to(hz.device)]
            keep = candidate
        else:
            logger.warning("Foreground mask kept nothing; measuring over all positions, unlike training.")
    result = analyse(hz.cpu().numpy())
    result["center_mode"] = center_mode
    result["patch_grid"] = list(grid)
    if not cli.no_lesion_split:
        cells = lesion_patch_mask(dataset, tuple(grid), int(hz.shape[1]), keep)
        if cells is None:
            logger.warning("Could not render lesions for this dataset; skipping the lesion-patch split.")
        else:
            features = hz.cpu().numpy()
            result["lesion_split"] = split_by_lesion(features, cells)
            result["within_position"] = within_position_split(features, cells)
    print_report(result, center_mode)
    if cli.out:
        out = Path(cli.out)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with out.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "channel",
                    "share_position",
                    "share_subject",
                    "share_interaction",
                    "variance_share",
                    "cross_view_position",
                    "cross_view_subject",
                    "cross_view_interaction",
                ]
            )
            total = sum(c["variance"] for c in result["channels"]) or 1.0
            for c in result["channels"]:
                writer.writerow(
                    [
                        c["channel"],
                        c["shares"]["position"],
                        c["shares"]["subject"],
                        c["shares"]["interaction"],
                        c["variance"] / total,
                        c["cross_view"]["position"],
                        c["cross_view"]["subject"],
                        c["cross_view"]["interaction"],
                    ]
                )
        print(f"\nWrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
