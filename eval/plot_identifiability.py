#!/usr/bin/env python
"""Figures from ``identifiability_report --out`` JSON: one model, or two compared.

Reads the JSON that ``eval.identifiability_report`` already writes, so plotting never
re-scores anything and a figure can never disagree with the table it came from.  Run the
report once per model with ``--out``, then point this at the files.

    python -m eval.identifiability_report --run-dir runs/contrastive --out fig/contrastive.json
    python -m eval.identifiability_report --run-dir runs/baseline    --out fig/baseline.json
    python -m eval.plot_identifiability --json fig/contrastive.json fig/baseline.json \
        --labels contrastive baseline --out fig/

What it draws
-------------
``learned_per_factor.png``   The headline. Learned R2 per factor (trained minus untrained),
                             one bar per model, error bars at 2x the floor's across-seed
                             spread -- the same bar the report's verdict uses, so a bar that
                             does not clear its whisker is not a finding.
``floor_decomposition.png``  Why the raw numbers are not the story: each factor's raw score
                             split into the part an UNTRAINED encoder already gets and the
                             part training added. Usually most of the bar is floor.
``mcc_ladder.png``           Learned block-MCC across gap -> stats -> patch. The shape says
                             where the information lives: rising with spatial resolution
                             means it is in the layout, flat-at-gap means channel identity.
``dci.png``                  Only when the report was run with --with-dci.

Every figure carries a ``.csv`` twin with the same numbers, since a colour-encoded figure
should never be the only way to read a value.

Colours are a validated categorical pair (blue / orange, slots 1-2), assigned by the order
you list the models so a colour always means the same run. ``--dark`` re-steps both to the
dark surface rather than inverting the light ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Validated categorical slots 1-2 plus chart chrome, light and dark. Both columns pass the
# lightness band, chroma floor, CVD separation, normal-vision floor and 3:1 contrast on
# their own surface (validate_palette.js). The dark column is separately stepped for the
# dark surface, not a flip of the light one.
THEME = {
    "light": {
        "series": ["#2a78d6", "#eb6834"],
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "floor_fill": "#e1e0d9",
    },
    "dark": {
        "series": ["#3987e5", "#d95926"],
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "floor_fill": "#383835",
    },
}

BAR_H = 0.36  # thin marks: leaves air in each category band rather than filling it
GAP = 0.02  # 2px-equivalent surface gap between touching fills


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _finite(x):
    v = _f(x)
    return v if v == v and abs(v) != float("inf") else None


def _style(ax, t, xlabel=None, title=None, subtitle=None):
    """Recessive chrome: hairline solid grid on the value axis only, no box."""
    ax.set_facecolor(t["surface"])
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_color(t["ink2"])
    if xlabel:
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=8.5)
    # Subtitle sits just above the axes; the title's pad is set to clear it, in points, so
    # the two never overlap however tall the axes is (an axes-fraction offset does not
    # scale that way and collided on the taller figures).
    if subtitle:
        ax.text(0.0, 1.0, subtitle, transform=ax.transAxes, color=t["muted"], fontsize=8.5, va="bottom", ha="left")
    if title:
        ax.set_title(title, color=t["ink"], fontsize=11, loc="left", pad=22 if subtitle else 8)


def _save(fig, path, t):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    logger.info("wrote %s", path)


def _write_csv(path, header, rows):
    """The table view. A value must be readable without decoding a colour."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    logger.info("wrote %s", path)


# --------------------------------------------------------------------------- #
# Data access — one place that knows the report's JSON shape
# --------------------------------------------------------------------------- #


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    run, floor, fstd = d.get("run") or {}, d.get("floor") or {}, d.get("floor_std") or {}
    if not run:
        raise ValueError(f"{path}: no 'run' key — is this an identifiability_report --out file?")
    return run, floor, fstd


def learned_r2(run, floor, fstd):
    """``{factor: (learned, 2x floor spread, pooling)}`` — the report's own verdict bar."""
    out = {}
    for name, d in (run.get("per_factor") or {}).items():
        real = _finite(d.get("r2"))
        base = _finite(((floor.get("per_factor") or {}).get(name) or {}).get("r2"))
        if real is None or base is None:
            continue
        sd = _finite(((fstd.get("per_factor") or {}).get(name) or {}).get("r2")) or 0.0
        out[name] = (real - base, 2.0 * sd, d.get("pooling") or "?")
    return out


def raw_and_floor(run, floor):
    """``{factor: (floor_gap, learned)}`` — the two parts the raw number is made of."""
    out = {}
    for name, d in (run.get("per_factor") or {}).items():
        real = _finite(d.get("r2"))
        base = _finite(((floor.get("per_factor") or {}).get(name) or {}).get("r2"))
        if real is None or base is None:
            continue
        out[name] = (base, real - base)
    return out


def learned_mcc_ladder(run, floor, fstd):
    """``{pooling: (learned, 2x spread)}`` over the rungs present, in coarse->fine order."""
    out = {}
    for key in ("gap", "stats", "patch"):
        m = (run.get("mcc") or {}).get(key)
        fm = (floor.get("mcc") or {}).get(key)
        if not m or not fm:
            continue
        real, base = _finite(m.get("mean")), _finite(fm.get("mean"))
        if real is None or base is None:
            continue
        sd = _finite(((fstd.get("mcc") or {}).get(key) or {}).get("mean")) or 0.0
        out[key] = (real - base, 2.0 * sd)
    return out


def learned_dci(run, floor):
    """``{(scope, 'D'|'C'): learned}`` from the gap columns."""
    out = {}
    for scope, d in (run.get("dci") or {}).items():
        fd = (floor.get("dci") or {}).get(scope) or {}
        for m in ("d", "c"):
            real, base = _finite(d.get(f"{m}_gap")), _finite(fd.get(f"{m}_gap"))
            if real is not None and base is not None:
                out[(scope, m.upper())] = real - base
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_learned_per_factor(models, t, path):
    """Grouped horizontal bars: learned R2 per factor, one bar per model.

    Horizontal because factor names are long and would otherwise be rotated.  Ordered by
    the first model's value so the eye has a ranking to follow; the order is stated in the
    caption so it is not mistaken for a property of the data.
    """
    per = [(label, learned_r2(run, fl, sd)) for label, run, fl, sd in models]
    factors = list(per[0][1].keys())
    if not factors:
        logger.warning("no per-factor overlap with the floor — skipping %s", path)
        return
    factors.sort(key=lambda f: per[0][1].get(f, (0,))[0])

    n = len(per)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(factors) + 1.9), facecolor=t["surface"])
    step = BAR_H + GAP
    offs = [(i - (n - 1) / 2) * step for i in range(n)]
    for i, (label, vals) in enumerate(per):
        ys = [j + offs[i] for j in range(len(factors))]
        xs = [vals.get(f, (float("nan"),))[0] for f in factors]
        es = [vals.get(f, (0, 0))[1] for f in factors]
        ax.barh(ys, xs, height=BAR_H, color=t["series"][i], label=label, zorder=3)
        ax.errorbar(xs, ys, xerr=es, fmt="none", ecolor=t["ink2"], elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4)
    ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([f"{f}  ({per[0][1].get(f, (0, 0, '?'))[2]})" for f in factors])
    _style(
        ax,
        t,
        xlabel="learned R²  (trained − untrained, null-subtracted)",
        title="What training added, per factor",
        subtitle="whiskers = 2× the floor's across-seed spread — a bar inside its whisker is not a finding",
    )
    if len(per) > 1:
        leg = ax.legend(frameon=False, loc="lower right", fontsize=8.5)
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        ["factor", "pooling"] + [f"{lab}_learned_r2" for lab, _ in per] + [f"{lab}_2x_floor_spread" for lab, _ in per],
        [
            [f, per[0][1].get(f, (0, 0, "?"))[2]]
            + [f"{v.get(f, (float('nan'),))[0]:.4f}" for _, v in per]
            + [f"{v.get(f, (0, 0))[1]:.4f}" for _, v in per]
            for f in reversed(factors)
        ],
    )


def fig_floor_decomposition(models, t, path):
    """Stacked horizontal bars, one panel per model: how much of the raw score is floor.

    Part-to-whole within one entity, so the floor segment is recessive grey and only the
    learned segment carries the model's colour — the colour still means the model, and the
    eye goes to the part that is about training.
    """
    parts = [(label, raw_and_floor(run, fl)) for label, run, fl, _ in models]
    factors = list(parts[0][1].keys())
    if not factors:
        logger.warning("no per-factor overlap with the floor — skipping %s", path)
        return
    factors.sort(key=lambda f: sum(parts[0][1].get(f, (0, 0))))

    fig, axes = plt.subplots(
        1,
        len(parts),
        figsize=(4.6 * len(parts) + 0.6, 0.42 * len(factors) + 2.0),
        sharex=True,
        sharey=True,
        facecolor=t["surface"],
    )
    axes = [axes] if len(parts) == 1 else list(axes)
    for i, ((label, vals), ax) in enumerate(zip(parts, axes)):
        ys = range(len(factors))
        base = [vals.get(f, (0, 0))[0] for f in factors]
        learned = [vals.get(f, (0, 0))[1] for f in factors]
        ax.barh(ys, base, height=BAR_H, color=t["floor_fill"], zorder=3, label="untrained floor")
        ax.barh(
            ys,
            learned,
            left=[b + GAP for b in base],
            height=BAR_H,
            color=t["series"][i],
            zorder=3,
            label="added by training",
        )
        ax.set_yticks(list(ys))
        ax.set_yticklabels(factors)
        _style(ax, t, xlabel="R² (null-subtracted)", title=label)
    axes[0].text(
        0.0,
        1.10,
        "Most of each raw score is what an untrained encoder already gets",
        transform=axes[0].transAxes,
        color=t["muted"],
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    leg = axes[-1].legend(frameon=False, loc="lower right", fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        ["factor"] + [c for lab, _ in parts for c in (f"{lab}_floor", f"{lab}_learned", f"{lab}_raw")],
        [
            [f]
            + [
                x
                for _, v in parts
                for x in (
                    f"{v.get(f, (0, 0))[0]:.4f}",
                    f"{v.get(f, (0, 0))[1]:.4f}",
                    f"{sum(v.get(f, (0, 0))):.4f}",
                )
            ]
            for f in reversed(factors)
        ],
    )


def fig_mcc_ladder(models, t, path):
    """Learned block-MCC across the pooling rungs. One ordered axis, one line per model."""
    series = [(label, learned_mcc_ladder(run, fl, sd)) for label, run, fl, sd in models]
    rungs = [k for k in ("gap", "stats", "patch") if any(k in s for _, s in series)]
    if len(rungs) < 2:
        logger.warning("fewer than two poolings scored — skipping %s", path)
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=t["surface"])
    xs = range(len(rungs))
    for i, (label, vals) in enumerate(series):
        ys = [vals.get(k, (float("nan"),))[0] for k in rungs]
        es = [vals.get(k, (0, 0))[1] for k in rungs]
        ax.errorbar(
            list(xs),
            ys,
            yerr=es,
            color=t["series"][i],
            linewidth=2.0,
            marker="o",
            markersize=7,
            markeredgecolor=t["surface"],
            markeredgewidth=2.0,
            ecolor=t["ink2"],
            elinewidth=1.0,
            capsize=2.5,
            label=label,
            zorder=3,
            solid_capstyle="round",
        )
        # Direct-label the endpoint only — the axis carries the rest.
        if ys and ys[-1] == ys[-1]:
            ax.annotate(
                f"{ys[-1]:+.3f}",
                (len(rungs) - 1, ys[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                color=t["ink2"],
                fontsize=8.5,
                va="center",
            )
    ax.axhline(0, color=t["axis"], linewidth=1.0, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(rungs)
    ax.set_xlim(-0.3, len(rungs) - 0.55)
    _style(
        ax,
        t,
        xlabel="pooling  (coarse → fine)",
        title="Where the content lives",
        subtitle="rising = information is in the spatial layout, not channel identity",
    )
    # After _style: it sets the grid for the horizontal-bar case (value on x), and here the
    # value axis is y. Flipping it back afterwards is the whole difference.
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.6, zorder=0)
    ax.set_ylabel("learned block-MCC", color=t["ink2"], fontsize=8.5)
    if len(series) > 1:
        # Proxy handles: the errorbar artist drags its caps into the legend box.
        handles = [
            Line2D(
                [],
                [],
                color=t["series"][i],
                linewidth=2.0,
                marker="o",
                markersize=7,
                markeredgecolor=t["surface"],
                markeredgewidth=2.0,
                label=label,
            )
            for i, (label, _) in enumerate(series)
        ]
        leg = ax.legend(handles=handles, frameon=False, loc="best", fontsize=8.5)
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        ["pooling"] + [f"{lab}_learned_mcc" for lab, _ in series],
        [[k] + [f"{v.get(k, (float('nan'),))[0]:.4f}" for _, v in series] for k in rungs],
    )


def fig_dci(models, t, path):
    """Learned D and C per scope. Skipped unless the report was run with --with-dci."""
    series = [(label, learned_dci(run, fl)) for label, run, fl, _ in models]
    keys = sorted({k for _, v in series for k in v}, key=lambda kv: (kv[1], kv[0]))
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(keys) + 1.9), facecolor=t["surface"])
    n = len(series)
    step = BAR_H + GAP
    offs = [(i - (n - 1) / 2) * step for i in range(n)]
    for i, (label, vals) in enumerate(series):
        ys = [j + offs[i] for j in range(len(keys))]
        ax.barh(
            ys, [vals.get(k, float("nan")) for k in keys], height=BAR_H, color=t["series"][i], label=label, zorder=3
        )
    ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{scope}  {m}" for scope, m in keys])
    _style(
        ax,
        t,
        xlabel="learned (trained − untrained)",
        title="Disentanglement / Completeness",
        subtitle="compare a row only with the same row in the other model — code counts differ",
    )
    if n > 1:
        leg = ax.legend(frameon=False, loc="lower right", fontsize=8.5)
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        ["scope", "metric"] + [f"{lab}_learned" for lab, _ in series],
        [[s, m] + [f"{v.get((s, m), float('nan')):.4f}" for _, v in series] for s, m in keys],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", nargs="+", required=True, help="identifiability_report --out files, one per model.")
    p.add_argument("--labels", nargs="*", default=None, help="Series labels (default: file basenames).")
    p.add_argument("--out", default="figures", help="Output directory.")
    p.add_argument("--dark", action="store_true", help="Re-step to the dark surface.")
    cli = p.parse_args()

    if len(cli.json) > len(THEME["light"]["series"]):
        p.error(
            f"at most {len(THEME['light']['series'])} models — categorical hues are assigned in fixed "
            "order and never generated. Plot them in pairs."
        )
    labels = cli.labels or [os.path.splitext(os.path.basename(j))[0] for j in cli.json]
    if len(labels) != len(cli.json):
        p.error("--labels must have one entry per --json")

    t = THEME["dark" if cli.dark else "light"]
    models = []
    for label, path in zip(labels, cli.json):
        run, floor, fstd = load(path)
        if not floor:
            logger.warning("%s has no floor — every 'learned' value will be missing. Re-run without --no-floor.", path)
        models.append((label, run, floor, fstd))

    os.makedirs(cli.out, exist_ok=True)
    fig_learned_per_factor(models, t, os.path.join(cli.out, "learned_per_factor.png"))
    fig_floor_decomposition(models, t, os.path.join(cli.out, "floor_decomposition.png"))
    fig_mcc_ladder(models, t, os.path.join(cli.out, "mcc_ladder.png"))
    fig_dci(models, t, os.path.join(cli.out, "dci.png"))


if __name__ == "__main__":
    main()
