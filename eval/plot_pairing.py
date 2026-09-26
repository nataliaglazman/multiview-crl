#!/usr/bin/env python
"""Figures for the pairing question: does the cross-modal pair beat augmenting one view?

Reads ``compare.json`` from ``eval.compare_bundles`` exactly as written, so plotting never
re-scores and a figure can never disagree with its table.

    FRESH=1 ARMS="cross=results/dino_cross within=results/dino_within" \
        bash scripts/compare_dino_objectives.sh
    python -m eval.plot_pairing --json results/dino_objective_comparison/content/compare.json \
        --cross cross --within within --out figures/

``plot_compare_bundles`` already draws the generic N-model table. This script exists for
the two-arm case, where the quantity a reader actually wants is the DIFFERENCE between the
arms, and reading it off two bars in separate bands is work the figure should have done.

What it draws
-------------
``pairing_recovery.png``
    One row per factor: a dot for each arm on a shared axis, joined by a connector whose
    length IS the difference. Each arm's untrained floor is an open tick in the same hue,
    so a pair of dots sitting on top of their own floors reads as "neither arm learned
    this factor" rather than as a tie between two good models. Sorted by the difference,
    so the factors that separate the arms are at the top.
``pairing_advantage.png``
    The difference alone, anchored at zero: positive means the cross-modal pair won that
    factor. Bars carry the two arms' seed spreads added in quadrature; a bar whose spread
    crosses zero is hatched, because "cross-modal wins here" is not a claim those numbers
    support. Coloured by which arm won, in the same hues as the dots above.

Both ship a ``.csv`` twin, and a ``_style`` twin when the run scored style factors.

Choosing the metric
-------------------
``--metric gap`` (default) is R² minus the permutation null: the headline number.
``--metric delta_floor`` subtracts each arm's OWN untrained floor instead, which removes
the readability the architecture has before any training.

They can disagree in SIGN on a factor, because the two floors are separate extractions and
their difference is noise that the subtraction keeps. The script checks for that and warns,
naming the factors, since those are exactly the ones where a conclusion would be fragile.

Colour
------
Two hues only — one per arm, slots 0 and 1 of the shared ``plot_identifiability.THEME``, so
an arm keeps its colour across every figure in this repo. The difference chart uses the
same two hues as its diverging pair with a neutral zero rule, which keeps "blue = the
cross-modal arm" true in both figures rather than introducing a third colour language.
Validated with the dataviz palette checker on both surfaces (light #2a78d6/#eb6834: CVD ΔE
24.7 protan, 32.7 tritan, normal 33.6; dark #3987e5/#d95926: 26.8 / 32.4 / 31.8 — all
checks PASS).
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from eval.plot_identifiability import THEME, _f, _finite, _save, _style, _write_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

METRICS = {
    "gap": ("R² − permutation null", "gap"),
    "delta_floor": ("R² gained over this arm's own untrained floor", "delta_floor"),
}
#: Rows per figure past which the bands get too thin to label; advisory, not a limit on
#: the analysis.
CROWDED = 24


def load(path):
    with open(path) as fh:
        return json.load(fh)


def pick_arms(report, cross, within):
    """Resolve the two arm labels, defaulting to the order the comparison scored them."""
    labels = list(report.get("results") or {})
    if cross is None and within is None:
        if len(labels) != 2:
            raise SystemExit(
                f"This figure is for two arms; {path_labels(labels)} were scored. "
                "Name them with --cross/--within, or use eval.plot_compare_bundles."
            )
        cross, within = labels
    missing = [n for n in (cross, within) if n not in labels]
    if missing:
        raise SystemExit(f"No such arm in the report: {', '.join(missing)}. Scored: {path_labels(labels)}")
    if cross == within:
        raise SystemExit("--cross and --within name the same arm, so there is no difference to draw")
    return cross, within


def path_labels(labels):
    return ", ".join(labels) if labels else "none"


def rows_for(report, arm, block):
    return {k: v for k, v in (report["results"][arm].get(block) or {}).items() if k != "_block"}


def paired(report, cross, within, block, field):
    """[(factor, cross value, within value, cross floor, within floor, combined std)] sorted by difference."""
    a, b = rows_for(report, cross, block), rows_for(report, within, block)
    out = []
    for name in a:
        if name not in b:
            continue
        av, bv = _finite(a[name].get(field)), _finite(b[name].get(field))
        if av is None or bv is None:
            continue
        # Seed spreads add in quadrature: the two arms were scored on the same rows with
        # the same folds, but their seed-to-seed wobble is independent.
        sa, sb = _f(a[name].get("std")), _f(b[name].get("std"))
        spread = float(np.sqrt(np.nansum([sa**2, sb**2]))) if np.isfinite([sa, sb]).any() else float("nan")
        out.append((name, av, bv, _finite(a[name].get("floor_gap")), _finite(b[name].get("floor_gap")), spread))
    return sorted(out, key=lambda r: r[1] - r[2], reverse=True)


def sign_disagreements(report, cross, within, block):
    """Factors where gap and delta_floor pick different winners.

    Both are legitimate readings of the same scoring run, so a factor that flips between
    them is one where the floors -- two separate extractions of the same untrained
    architecture -- differ by more than the arms do. That is noise deciding the answer.
    """
    by_metric = {
        m: {r[0]: r[1] - r[2] for r in paired(report, cross, within, block, f)} for m, (_, f) in METRICS.items()
    }
    shared = set(by_metric["gap"]) & set(by_metric["delta_floor"])
    return sorted(
        n
        for n in shared
        if by_metric["gap"][n] * by_metric["delta_floor"][n] < 0
        and abs(by_metric["gap"][n]) > 1e-9
        and abs(by_metric["delta_floor"][n]) > 1e-9
    )


def _geometry(n_rows):
    return max(2.8, 0.40 * n_rows + 2.0)


def _headroom(ax, values, left=0.06, right=0.16):
    """Widen the value axis so the end-of-row labels sit inside the figure.

    ``bbox_inches="tight"`` grows the canvas for text that overflows the AXES but not for
    text that overflows the DATA limits -- those labels were being clipped, and the
    negative ones ran into the factor names on the left.
    """
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return 0.0
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or (abs(hi) or 1.0)
    ax.set_xlim(lo - span * left, hi + span * right)
    return span


# --------------------------------------------------------------------------- #
# Figure 1 — the two arms as a dumbbell, difference as the connector
# --------------------------------------------------------------------------- #


def fig_recovery(report, cross, within, block, metric, t, path):
    label, field = METRICS[metric]
    data = paired(report, cross, within, block, field)
    if not data:
        return False
    if len(data) > CROWDED:
        logger.warning("%d factors is a tall figure; consider splitting %s", len(data), os.path.basename(path))
    c_hue, w_hue = t["series"][0], t["series"][1]
    fig, ax = plt.subplots(figsize=(8.6, _geometry(len(data))))

    any_floor = False
    for i, (name, av, bv, af, bf, _spread) in enumerate(data):
        y = len(data) - 1 - i
        # The connector is the difference, drawn first so the dots sit on top of it.
        ax.plot([bv, av], [y, y], color=t["axis"], linewidth=2.0, solid_capstyle="round", zorder=2)
        for value, floor, hue in ((av, af, c_hue), (bv, bf, w_hue)):
            if floor is not None:
                any_floor = True
                ax.plot(
                    [floor],
                    [y],
                    marker="|",
                    markersize=9,
                    markeredgewidth=1.6,
                    color=hue,
                    alpha=0.55,
                    zorder=3,
                )
            ax.plot(
                [value],
                [y],
                marker="o",
                markersize=8,
                color=hue,
                zorder=4,
                markeredgecolor=t["surface"],
                markeredgewidth=1.2,  # 2px-equivalent surface ring
            )
        diff = av - bv
        # Clear of the dot's own radius: at 0.012 the leading +/- sat under the marker and
        # every negative difference read as positive.
        ax.text(
            max(av, bv) + 0.055 * _span(data),
            y,
            f"{diff:+.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=t["ink2"],
        )

    _headroom(ax, [v for r in data for v in (r[1], r[2], r[3], r[4])])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([r[0] for r in reversed(data)])
    ax.set_ylim(-0.7, len(data) - 0.3)
    sub = f"{cross} vs {within} · number at each row is {cross} − {within}"
    _style(ax, t, xlabel=label, title=f"Per-factor recovery, {block} block", subtitle=sub)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=c_hue, markersize=8, label=cross),
        Line2D([], [], marker="o", linestyle="none", color=w_hue, markersize=8, label=within),
    ]
    if any_floor:
        handles.append(
            Line2D(
                [], [], marker="|", linestyle="none", color=t["muted"], markersize=9, label="that arm's untrained floor"
            )
        )
    _legend(fig, handles, t)
    _save(fig, path, t)
    _write_csv(
        path.replace(".png", ".csv"),
        ["factor", f"{cross}_{field}", f"{within}_{field}", f"{cross}_floor_gap", f"{within}_floor_gap", "difference"],
        [[n, av, bv, af, bf, av - bv] for n, av, bv, af, bf, _ in data],
    )
    return True


def _span(data):
    values = [v for r in data for v in (r[1], r[2]) if v is not None]
    return (max(values) - min(values)) or 1.0


# --------------------------------------------------------------------------- #
# Figure 2 — the difference alone, anchored at zero
# --------------------------------------------------------------------------- #


def fig_advantage(report, cross, within, block, metric, t, path):
    label, field = METRICS[metric]
    data = paired(report, cross, within, block, field)
    if not data:
        return False
    c_hue, w_hue = t["series"][0], t["series"][1]
    fig, ax = plt.subplots(figsize=(8.0, _geometry(len(data))))

    any_uncertain = False
    rows = []
    diffs = [r[1] - r[2] for r in data]
    spreads = [r[5] if np.isfinite(r[5]) else 0.0 for r in data]
    pad = 0.03 * (max(abs(d) + s for d, s in zip(diffs, spreads)) or 1.0)
    for i, (name, av, bv, _af, _bf, spread) in enumerate(data):
        y = len(data) - 1 - i
        diff = av - bv
        hue = c_hue if diff >= 0 else w_hue
        # Hatched when the arms' combined seed spread reaches across zero: the sign of the
        # bar is then not something these numbers establish.
        uncertain = np.isfinite(spread) and spread > 0 and abs(diff) < spread
        any_uncertain |= bool(uncertain)
        ax.barh(
            y,
            diff,
            height=0.36,
            color=hue if not uncertain else "none",
            zorder=3,
            linewidth=0,
            hatch="////" if uncertain else None,
            edgecolor=hue if uncertain else None,
        )
        reach = abs(diff)
        if np.isfinite(spread) and spread > 0:
            ax.plot([diff - spread, diff + spread], [y, y], color=t["ink2"], linewidth=1.0, alpha=0.7, zorder=4)
            reach = max(reach, abs(diff) + spread)  # past the whisker, not on top of it
        ax.text(
            (reach + pad) * (1 if diff >= 0 else -1),
            y,
            f"{diff:+.3f}",
            va="center",
            ha="left" if diff >= 0 else "right",
            fontsize=7.5,
            color=t["ink2"],
        )
        # No winner is recorded for a row the spread cannot separate: the CSV is the table
        # view of this figure, and a name in that column is what ends up quoted.
        rows.append(
            [name, diff, spread, "no" if uncertain else "yes", "" if uncertain else (cross if diff >= 0 else within)]
        )

    ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
    # Both directions: a left-pointing bar's label was running into the factor names.
    ends = [d + s for d, s in zip(diffs, spreads)] + [d - s for d, s in zip(diffs, spreads)] + [0.0]
    _headroom(ax, ends, left=0.22, right=0.22)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([r[0] for r in reversed(data)])
    ax.set_ylim(-0.7, len(data) - 0.3)
    _style(
        ax,
        t,
        xlabel=f"{cross} − {within}   ({label})",
        title=f"What the cross-modal pair bought, {block} block",
        subtitle=f"right of zero: {cross} recovered the factor better · bars are ±combined seed spread",
    )
    handles = [Patch(facecolor=c_hue, label=f"{cross} ahead"), Patch(facecolor=w_hue, label=f"{within} ahead")]
    if any_uncertain:
        handles.append(Patch(facecolor="none", edgecolor=t["muted"], hatch="////", label="spread crosses zero"))
    _legend(fig, handles, t)
    _save(fig, path, t)
    _write_csv(
        path.replace(".png", ".csv"),
        ["factor", "difference", "combined_std", "separated", "winner"],
        rows,
    )
    return True


def _legend(fig, handles, t):
    """Below the figure: these rows are dense and there is no reliably empty corner."""
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        labelcolor=t["ink2"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 4),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", help="compare.json written by eval.compare_bundles --out")
    p.add_argument("--out", help="Directory for the PNGs and their CSV twins")
    p.add_argument("--cross", help="Label of the cross-modal arm (default: the first of two)")
    p.add_argument("--within", help="Label of the within-modality arm (default: the second of two)")
    p.add_argument("--metric", default="gap", choices=sorted(METRICS), help="Quantity compared (default: gap)")
    p.add_argument("--dark", action="store_true", help="Re-step every colour to the dark surface")
    p.add_argument("--self-test", action="store_true", help="Draw from a planted report and exit")
    cli = p.parse_args(argv)

    if cli.self_test:
        return _self_test(cli)
    if not cli.json or not cli.out:
        p.error("--json and --out are required (or pass --self-test)")

    report = load(cli.json)
    cross, within = pick_arms(report, cli.cross, cli.within)
    t = THEME["dark" if cli.dark else "light"]
    os.makedirs(cli.out, exist_ok=True)

    drawn = 0
    for block in ("content", "style"):
        suffix = "" if block == "content" else "_style"
        flipped = sign_disagreements(report, cross, within, block)
        if flipped:
            logger.warning(
                "%s block: gap and delta_floor disagree on which arm wins %s. The two floors are "
                "separate extractions of the same untrained architecture, so on those factors the "
                "answer is being set by floor noise -- do not quote them either way.",
                block,
                ", ".join(flipped),
            )
        drawn += fig_recovery(
            report, cross, within, block, cli.metric, t, os.path.join(cli.out, f"pairing_recovery{suffix}.png")
        )
        drawn += fig_advantage(
            report, cross, within, block, cli.metric, t, os.path.join(cli.out, f"pairing_advantage{suffix}.png")
        )
    if not drawn:
        raise SystemExit(f"{cli.json} has no per-factor rows for either arm; nothing to draw")
    logger.info("PC skeleton recovery for these arms is in plot_compare_bundles' causal_discovery.png")
    return 0


def _self_test(cli):
    """Draw both figures from a planted two-arm report -- no scoring, no GPU."""
    import tempfile

    rng = np.random.RandomState(0)
    names = [f"f{i}" for i in range(6)]

    def arm(scale, floor):
        return {
            n: dict(
                gap=float(scale[i]),
                std=float(0.01 + 0.01 * i),
                floor_gap=float(floor[i]),
                delta_floor=float(scale[i] - floor[i]),
            )
            for i, n in enumerate(names)
        } | {"_block": {"probe_features": 24}}

    report = {
        "results": {
            "cross": {"content": arm(rng.uniform(0.4, 0.9, 6), rng.uniform(0.1, 0.3, 6)), "style": {}},
            "within": {"content": arm(rng.uniform(0.3, 0.8, 6), rng.uniform(0.1, 0.3, 6)), "style": {}},
        }
    }
    out = cli.out or tempfile.mkdtemp()
    os.makedirs(out, exist_ok=True)
    t = THEME["dark" if cli.dark else "light"]
    assert fig_recovery(report, "cross", "within", "content", cli.metric, t, os.path.join(out, "pairing_recovery.png"))
    assert fig_advantage(
        report, "cross", "within", "content", cli.metric, t, os.path.join(out, "pairing_advantage.png")
    )
    assert not fig_recovery(report, "cross", "within", "style", cli.metric, t, os.path.join(out, "x.png"))
    print(f"self-test PASSED -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
