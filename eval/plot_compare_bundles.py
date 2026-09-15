#!/usr/bin/env python
"""Figures from ``compare_bundles``' JSON: the matched comparison, drawn.

Reads ``compare.json`` as written, so plotting never re-scores and a figure can never
disagree with the table it came from.

    python -m eval.compare_bundles --bundles vq=... dino=... --floors ... \
        --with-graph --graph-repeats 20 --out results/matched/compare.json
    python -m eval.plot_compare_bundles --json results/matched/compare.json --out figures/

What it draws
-------------
``factor_recovery_content.png``
    The headline. Per factor, one bar per model, length = ``gap`` (R² minus the
    permutation null). Each bar carries a tick at that model's OWN untrained floor, so
    "how much of this is training" is the distance from tick to bar end and a bar that
    stops SHORT of its tick reads immediately as below floor. A tick is one mark rather
    than a second stacked fill because ``delta_floor`` goes negative and a stacked
    segment cannot draw that. ``_style`` twin when the bundles carry style factors.
``delta_floor.png``
    The same quantity as the tick gap, given its own axis and anchored at zero, because
    "what did training buy" is the question a collaborator asks first and reading it off
    a distance between two marks is harder than reading it off a baseline. Only drawn
    when at least one bundle has a floor.
``causal_discovery.png``
    PC skeleton recovery against the TRUE adjacency: F1 and skeleton SHD as two panels,
    never one axis with two scales. Error bars are the ``--graph-repeats`` resampling
    band when the run has one; without it the bars are single point estimates and the
    subtitle says so. The ground-truth ceiling is a reference RULE, not a bar, because it
    is a benchmark rather than a model and a reader should not rank against it by length.
``partial_r2.png``
    Per factor, partial R² (the factor's OWN variation, after its SCM parents are
    regressed out) as the solid bar, with the rest of raw R² as a recessive extension —
    the part a probe only gets through the parents. A long pale tail on a short bar is a
    factor the representation is reading second-hand, which is what usually explains a
    graph difference.

Every figure ships a ``.csv`` twin: a colour-encoded figure is never the only way to read
a value, and the light-mode palette's relief rule requires it.

Colour
------
One hue per BUNDLE, taken from ``plot_identifiability.THEME`` in the reference palette's
fixed order and never cycled, so a colour means the same model here as in the other two
plot scripts. A bundle's floor is drawn in its own hue with a hatch, so floor-vs-model is
a texture difference and the hue keeps meaning "which model". ``--dark`` re-steps every
colour to the dark surface rather than inverting the light one.
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
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from eval.plot_identifiability import THEME, _f, _finite, _save, _style, _write_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CEILING_LABEL = "truth (ceiling)"
FLOOR_SUFFIX = " · floor"
# Past this many models the grouped bands get too thin to read; the palette has 8 slots
# and stops there outright. Both are advisory limits on the FIGURE, not on the analysis.
CROWDED = 4
MAX_SERIES = 8


def load(path):
    with open(path) as fh:
        return json.load(fh)


def bundle_labels(report):
    """Model labels in the order the comparison scored them."""
    return list(report.get("results") or {})


def series_colours(labels, t):
    if len(labels) > MAX_SERIES:
        raise SystemExit(
            f"{len(labels)} bundles is past the {MAX_SERIES} validated categorical slots. "
            "Plot them in two runs, or facet, rather than cycling a hue onto a second model."
        )
    if len(labels) > CROWDED:
        logger.warning("%d models per factor band is dense; consider splitting the figure", len(labels))
    return {label: t["series"][i] for i, label in enumerate(labels)}


def factor_order(report, block):
    """Factors every bundle scored, in the first bundle's order."""
    per_label = [(report["results"][label].get(block) or {}) for label in bundle_labels(report)]
    if not per_label or not per_label[0]:
        return []
    names = [n for n in per_label[0] if n != "_block"]
    return [n for n in names if all(n in rows for rows in per_label)]


def _grouped_geometry(n_factors, n_series):
    """(bar height, per-series offsets, figure height) for a horizontal grouped bar band."""
    height = min(0.8 / max(n_series, 1), 0.34)
    span = height * n_series
    offsets = [(-span / 2) + height * (i + 0.5) for i in range(n_series)]
    fig_h = max(3.0, 0.34 * n_factors * n_series + 1.8)
    return height, offsets, fig_h


def _label_bar(ax, x, y, value, t, pad):
    ax.text(x + pad, y, f"{value:+.2f}", va="center", ha="left", color=t["ink2"], fontsize=7.5)


def _legend(fig, handles, t):
    """Below the whole figure, horizontal.

    Inside the axes it collided with the last factor's bars on every one of these plots --
    they are dense by construction (one band per factor x one bar per model) and there is
    no reliably empty corner. ``bbox_inches="tight"`` at save time keeps it in frame.
    """
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        labelcolor=t["ink2"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 5),
    )


# --------------------------------------------------------------------------- #
# Figure 1 — per-factor recovery, with each model's own floor as a tick
# --------------------------------------------------------------------------- #


def fig_factor_recovery(report, block, t, path):
    labels = bundle_labels(report)
    factors = factor_order(report, block)
    if not factors:
        return False
    colours = series_colours(labels, t)
    height, offsets, fig_h = _grouped_geometry(len(factors), len(labels))
    fig, ax = plt.subplots(figsize=(8.4, fig_h))

    rows, any_floor = [], False
    span = max(
        [
            abs(_f(report["results"][l][block][f].get("gap")))
            for l in labels
            for f in factors
            if _finite(report["results"][l][block][f].get("gap")) is not None
        ]
        or [1.0]
    )
    pad = span * 0.02
    for fi, factor in enumerate(factors):
        for si, label in enumerate(labels):
            row = report["results"][label][block][factor]
            gap, floor = _finite(row.get("gap")), _finite(row.get("floor_gap"))
            y = fi + offsets[si]
            if gap is None:
                continue
            ax.barh(y, gap, height=height * 0.86, color=colours[label], zorder=3, linewidth=0)
            _label_bar(ax, max(gap, 0), y, gap, t, pad)
            if floor is not None:
                any_floor = True
                # A tick, not a stacked segment: delta_floor goes negative and a segment
                # cannot draw a bar that stops short of its own baseline.
                ax.plot(
                    [floor, floor],
                    [y - height * 0.46, y + height * 0.46],
                    color=t["ink"],
                    linewidth=1.6,
                    solid_capstyle="butt",
                    zorder=4,
                )
            rows.append(
                [
                    block,
                    factor,
                    label,
                    row.get("gap"),
                    row.get("real"),
                    row.get("null"),
                    row.get("floor_gap"),
                    row.get("delta_floor"),
                    row.get("mcc"),
                ]
            )

    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors)
    ax.invert_yaxis()
    ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
    _style(
        ax,
        t,
        xlabel="R² gap  (cross-validated R² − permutation null)",
        title=f"{block.capitalize()} factor recovery",
        subtitle=(
            "one bar per model; tick = that model's own untrained floor"
            if any_floor
            else "one bar per model; no untrained floor was supplied"
        ),
    )
    handles = [Patch(facecolor=colours[l], label=l) for l in labels]
    if any_floor:
        handles.append(Line2D([0], [0], color=t["ink"], linewidth=1.6, label="untrained floor"))
    _legend(fig, handles, t)
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        ["block", "factor", "model", "gap", "real", "null", "floor_gap", "delta_floor", "mcc"],
        rows,
    )
    return True


# --------------------------------------------------------------------------- #
# Figure 2 — what training bought, on its own baseline
# --------------------------------------------------------------------------- #


def fig_delta_floor(report, block, t, path):
    labels = [l for l in bundle_labels(report) if report["results"][l].get("floor_path")]
    factors = factor_order(report, block)
    if not labels or not factors:
        return False
    colours = series_colours(bundle_labels(report), t)
    height, offsets, fig_h = _grouped_geometry(len(factors), len(labels))
    fig, ax = plt.subplots(figsize=(8.4, fig_h))

    rows = []
    values = [_finite(report["results"][l][block][f].get("delta_floor")) for l in labels for f in factors]
    span = max([abs(v) for v in values if v is not None] or [1.0])
    for fi, factor in enumerate(factors):
        for si, label in enumerate(labels):
            row = report["results"][label][block][factor]
            delta = _finite(row.get("delta_floor"))
            if delta is None:
                continue
            y = fi + offsets[si]
            ax.barh(y, delta, height=height * 0.86, color=colours[label], zorder=3, linewidth=0)
            ax.text(
                delta + (span * 0.02 if delta >= 0 else -span * 0.02),
                y,
                f"{delta:+.2f}",
                va="center",
                ha="left" if delta >= 0 else "right",
                color=t["ink2"],
                fontsize=7.5,
            )
            rows.append([block, factor, label, row.get("delta_floor"), row.get("gap"), row.get("floor_gap")])

    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors)
    ax.invert_yaxis()
    ax.axvline(0, color=t["ink2"], linewidth=1.2, zorder=4)
    _style(
        ax,
        t,
        xlabel="R² gap above the model's OWN untrained floor",
        title="What training bought",
        subtitle="at or below zero: the untrained architecture already reads this factor",
    )
    _legend(fig, [Patch(facecolor=colours[l], label=l) for l in labels], t)
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv", ["block", "factor", "model", "delta_floor", "gap", "floor_gap"], rows
    )
    return True


# --------------------------------------------------------------------------- #
# Figure 3 — causal discovery against the true adjacency
# --------------------------------------------------------------------------- #


def graph_rows(report):
    """``[(label, panel, base_model)]`` — each model then its floor, ceiling handled apart."""
    out = []
    for label in bundle_labels(report):
        panels = report["results"][label].get("graph") or {}
        if panels.get("embeddings"):
            out.append((label, panels["embeddings"], label))
        if panels.get("floor"):
            out.append((label + FLOOR_SUFFIX, panels["floor"], label))
    return out


def _panel_row(panel, alpha):
    for row in panel.get("alpha_sweep") or []:
        if _finite(row.get("alpha")) is not None and abs(row["alpha"] - alpha) < 1e-12 and "f1" in row:
            return row
    return None


def fig_causal_discovery(report, t, path):
    rows = graph_rows(report)
    if not rows:
        return False
    alpha = _f((report.get("options") or {}).get("diagnostic_alpha")) or 0.05
    stability = report.get("graph_stability") or {}
    colours = series_colours(bundle_labels(report), t)
    truth = report.get("truth_panel")
    truth_row = _panel_row(truth, alpha) if truth else None

    fig, axes = plt.subplots(1, 2, figsize=(10.4, max(2.6, 0.52 * len(rows) + 1.9)))
    csv_rows = []
    for ax, (metric, nice, fmt) in zip(axes, (("f1", "F1", "{:.3f}"), ("skeleton_shd", "Skeleton SHD", "{:.0f}"))):
        scored = [_panel_row(p, alpha) for _l, p, _b in rows]
        reach = [
            _f(r[metric]) + (_finite((stability.get(l) or {}).get(f"{metric}_std")) or 0.0)
            for r, (l, _p, _b) in zip(scored, rows)
            if r is not None
        ]
        pad = (max(reach + [_f((truth_row or {}).get(metric, 0))] or [1.0])) * 0.03
        for i, (label, panel, base) in enumerate(rows):
            row = _panel_row(panel, alpha)
            if row is None:
                continue
            value = _f(row[metric])
            band = stability.get(label) or {}
            err = _finite(band.get(f"{metric}_std"))
            ax.barh(
                i,
                value,
                height=0.46,
                color=colours[base],
                zorder=3,
                linewidth=0,
                hatch="///" if label.endswith(FLOOR_SUFFIX) else None,
                edgecolor=t["surface"],
            )
            if err is not None:
                ax.errorbar(value, i, xerr=err, fmt="none", ecolor=t["ink2"], elinewidth=1.2, capsize=3, zorder=5)
            # Past the error-bar cap, not centred above the bar: centred it landed on the
            # ceiling rule whenever a source sat near the ceiling, which is exactly the
            # case a reader is looking at.
            label_text = ax.text(
                value + (err or 0.0) + pad,
                i,
                fmt.format(value),
                va="center",
                ha="left",
                color=t["ink2"],
                fontsize=7.5,
                zorder=7,
            )
            # A source near the ceiling puts its own value label on the ceiling rule. The
            # surface ring keeps the digits readable instead of moving the label somewhere
            # that no longer reads as belonging to this bar.
            label_text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground=t["surface"])])
            if metric == "f1":
                csv_rows.append(
                    [
                        label,
                        alpha,
                        row.get("f1"),
                        row.get("precision"),
                        row.get("recall"),
                        row.get("skeleton_shd"),
                        row.get("tp"),
                        row.get("fp"),
                        row.get("fn"),
                        band.get("f1_mean"),
                        band.get("f1_std"),
                        band.get("skeleton_shd_mean"),
                        band.get("skeleton_shd_std"),
                        band.get("repeats"),
                    ]
                )
        if truth_row is not None:
            ax.axvline(_f(truth_row[metric]), color=t["ink"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=6)
            ax.text(_f(truth_row[metric]), -0.95, " ceiling", va="bottom", ha="left", color=t["ink"], fontsize=7.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([l for l, _p, _b in rows] if ax is axes[0] else [""] * len(rows))
        ax.invert_yaxis()
        ax.set_ylim(len(rows) - 0.4, -1.1)
        _style(ax, t, xlabel=nice)

    band_note = next((v for v in stability.values() if v), None)
    subtitle = (
        f"error bars: ±1 SD over {band_note['repeats']} row subsamples of {band_note['subsample']}"
        if band_note
        else "no resampling band — single point estimates; pass --graph-repeats"
    )
    axes[0].set_title("PC skeleton vs the true SCM adjacency", color=t["ink"], fontsize=11, loc="left", pad=22)
    axes[0].text(
        0.0,
        1.0,
        f"{subtitle}  ·  alpha={alpha:g}",
        transform=axes[0].transAxes,
        color=t["muted"],
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    handles = [Patch(facecolor=colours[l], label=l) for l in bundle_labels(report)]
    if any(l.endswith(FLOOR_SUFFIX) for l, _p, _b in rows):
        handles.append(Patch(facecolor=t["muted"], hatch="///", edgecolor=t["surface"], label="untrained floor"))
    if truth_row is not None:
        handles.append(
            Line2D([0], [0], color=t["ink"], linewidth=1.2, linestyle=(0, (4, 3)), label="ground-truth ceiling")
        )
    _legend(fig, handles, t)
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        [
            "source",
            "alpha",
            "f1",
            "precision",
            "recall",
            "skeleton_shd",
            "tp",
            "fp",
            "fn",
            "f1_mean",
            "f1_std",
            "shd_mean",
            "shd_std",
            "repeats",
        ],
        csv_rows,
    )
    return True


# --------------------------------------------------------------------------- #
# Figure 4 — the factor's own variation vs what comes through its parents
# --------------------------------------------------------------------------- #


def fig_partial_r2(report, t, path):
    rows = graph_rows(report)
    if not rows:
        return False
    colours = series_colours(bundle_labels(report), t)
    names = [f["name"] for f in (rows[0][1].get("factors") or [])]
    if not names:
        return False
    height, offsets, fig_h = _grouped_geometry(len(names), len(rows))
    fig, ax = plt.subplots(figsize=(8.8, fig_h))

    csv_rows = []
    for fi, name in enumerate(names):
        for si, (label, panel, base) in enumerate(rows):
            entry = next((f for f in (panel.get("factors") or []) if f["name"] == name), None)
            if entry is None:
                continue
            partial, raw = _finite(entry.get("partial_r2")), _finite(entry.get("raw_r2"))
            y = fi + offsets[si]
            if raw is not None and partial is not None and raw > partial:
                # The recessive extension: what the probe only reaches through the parents.
                ax.barh(
                    y,
                    raw - partial,
                    left=partial,
                    height=height * 0.86,
                    color=colours[base],
                    alpha=0.28,
                    zorder=2,
                    linewidth=0,
                )
            if partial is not None:
                ax.barh(
                    y,
                    partial,
                    height=height * 0.86,
                    color=colours[base],
                    zorder=3,
                    linewidth=0,
                    hatch="///" if label.endswith(FLOOR_SUFFIX) else None,
                    edgecolor=t["surface"],
                )
            csv_rows.append(
                [name, label, entry.get("parents"), entry.get("partial_r2"), entry.get("raw_r2"), entry.get("gap")]
            )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=1)
    _style(
        ax,
        t,
        xlabel="R²  (solid = partial, after SCM parents removed)",
        title="Own variation vs variation borrowed from parents",
        subtitle="pale tail = reached only through the parents; a root factor has no tail by construction",
    )
    handles = [Patch(facecolor=colours[l], label=l) for l in bundle_labels(report)]
    if any(l.endswith(FLOOR_SUFFIX) for l, _p, _b in rows):
        handles.append(Patch(facecolor=t["muted"], hatch="///", edgecolor=t["surface"], label="untrained floor"))
    handles.append(Patch(facecolor=t["muted"], alpha=0.28, label="via parents"))
    _legend(fig, handles, t)
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv", ["factor", "source", "parents", "partial_r2", "raw_r2", "gap"], csv_rows
    )
    return True


# --------------------------------------------------------------------------- #


def draw_all(report, out_dir, t):
    os.makedirs(out_dir, exist_ok=True)
    drawn = []
    for block in ("content", "style"):
        path = os.path.join(out_dir, f"factor_recovery_{block}.png")
        if fig_factor_recovery(report, block, t, path):
            drawn.append(path)
        path = os.path.join(out_dir, f"delta_floor_{block}.png")
        if fig_delta_floor(report, block, t, path):
            drawn.append(path)
    for fn, name in ((fig_causal_discovery, "causal_discovery.png"), (fig_partial_r2, "partial_r2.png")):
        path = os.path.join(out_dir, name)
        if fn(report, t, path):
            drawn.append(path)
    return drawn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", required=True, help="compare.json written by eval.compare_bundles --out")
    parser.add_argument("--out", default="figures", help="Directory for the PNGs and their .csv twins")
    parser.add_argument("--dark", action="store_true", help="Re-step every colour to the dark surface")
    cli = parser.parse_args(argv)

    report = load(cli.json)
    t = THEME["dark" if cli.dark else "light"]
    drawn = draw_all(report, cli.out, t)
    if not drawn:
        logger.warning("nothing to draw from %s — it has no scored bundles", cli.json)
        return 1
    print(f"\n{len(drawn)} figure(s) in {os.path.abspath(cli.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
