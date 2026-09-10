#!/usr/bin/env python
"""Figures from ``run_causal_recovery``'s JSON: the causal panel, drawn.

Reads ``causal_recovery.json`` as written, so plotting never re-runs PC and a figure can
never disagree with the table it came from.  One file already holds every run it scored,
so usually there is one ``--json``; pass several to overlay separate output directories.

    python -m eval.run_causal_recovery --run-dirs RUN --num-samples 2000 --pooling gap \
        --alphas 0.05 --orientation
    python -m eval.plot_causal_recovery --json results/causal_recovery/causal_recovery.json \
        --out figures/

What it draws
-------------
``edges.png``        The one a scalar cannot give you: WHICH edges were recovered. A cell
                     per factor pair — blue filled + check for a true edge PC found, blue
                     outline + minus for one it missed, orange + plus for an edge it
                     invented. Reading down a row says whether one factor is carrying the
                     errors, which ``skeleton_shd`` averages away.
``alpha_sweep.png``  F1 / precision / recall / SHD against alpha, one line per run, the
                     selected alpha ringed. This is the figure that exposes an alpha
                     confound: two runs whose sweeps cross are not comparable at their
                     own best alphas, and the headline F1 difference is then selection,
                     not recovery. Four panels rather than two y-axes, deliberately.
``factor_r2.png``    Per factor, partial R² (the factor's OWN variation) as the solid bar
                     and the rest of raw R² as a recessive extension — the part a probe
                     gets through the factor's SCM parents rather than from the factor.
                     A long pale tail on a short bar is a factor that is not encoded.
``orientation.png``  Only when the panel was run with ``--orientation``. Edge directions
                     against the true CPDAG, ordered matched -> unoriented -> wrong, with
                     CPDAG SHD beside each bar.

Every figure carries a ``.csv`` twin, since a colour-encoded figure should never be the
only way to read a value; the orientation CSV keeps all six raw categories, not the three
the bar collapses them into.

Colour
------
Run identity uses the same validated categorical slots as ``plot_identifiability`` (blue,
orange), imported rather than restated, so a colour means the same run across both
scripts.  ``edges.png`` uses those two hues for "in the true graph" vs "invented" with
fill-vs-outline and a glyph as secondary encoding — NOT green/red, which fails
colourblind separation outright (deutan ΔE 4.1, measured with the palette validator).
``orientation.png``'s three classes are ordered, so they take a single-hue ordinal ramp.
``--dark`` re-steps every one of those to the dark surface rather than inverting them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from eval.plot_identifiability import THEME, _f, _finite, _save, _style, _write_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Ordered classes take a single-hue ordinal ramp, validated with the palette script's
# --ordinal mode in both surfaces (light 250/450/650, dark 200/400/600).
ORDINAL = {"light": ["#86b6ef", "#2a78d6", "#104281"], "dark": ["#9ec5f4", "#3987e5", "#184f95"]}
# matched / unoriented / wrong, best to worst. Six raw categories collapse to three
# because past ~7 colour classes adjacent bins blur; the CSV keeps all six.
ORIENTATION_CLASSES = (
    ("matched", ("correct_directed", "both_undirected")),
    ("unoriented", ("undirected_in_estimate", "directed_in_estimate")),
    ("wrong", ("reversed", "bidirected_in_estimate")),
)


# --------------------------------------------------------------------------- #
# Data access — one place that knows run_causal_recovery's JSON shape
# --------------------------------------------------------------------------- #


def load(path):
    """``[(label, run), ...]`` for every scored run in one report file."""
    with open(path) as fh:
        payload = json.load(fh)
    if "runs" not in payload:
        raise ValueError(f"{path}: no 'runs' key — is this a run_causal_recovery output file?")
    out = []
    for run in payload["runs"]:
        if not run.get("factors"):
            logger.warning(
                "skipping %s: %s", run.get("run_dir", "?"), run.get("reason", run.get("status", "no scores"))
            )
            continue
        out.append((os.path.basename(str(run.get("run_dir", "run")).rstrip("/")), run))
    return out


def factor_names(run):
    return [factor.get("name", f"d{factor['dim']}") for factor in run["factors"]]


def edge_classes(run):
    """``(names, {(i, j): 'tp'|'fp'|'fn'})`` for the selected alpha's skeleton."""
    best = run.get("best")
    if not best or "adjacency" not in best:
        return factor_names(run), {}
    truth, estimate = run["true_skeleton"], best["adjacency"]
    classes = {}
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            in_truth, in_estimate = bool(truth[i][j]), bool(estimate[i][j])
            if in_truth or in_estimate:
                classes[(i, j)] = "tp" if in_truth and in_estimate else ("fn" if in_truth else "fp")
    return factor_names(run), classes


def sweep_rows(run):
    """``[(alpha, f1, precision, recall, shd), ...]`` in alpha order, errors dropped."""
    rows = [r for r in run.get("alpha_sweep", []) if "f1" in r]
    rows.sort(key=lambda r: r["alpha"])
    return [(_f(r["alpha"]), _f(r["f1"]), _f(r["precision"]), _f(r["recall"]), _f(r["skeleton_shd"])) for r in rows]


def orientation_counts(run):
    """``(three ordered class totals, cpdag_shd)`` at the selected alpha, or None."""
    best = run.get("best") or {}
    found = best.get("orientation")
    if not found:
        return None
    totals = [sum(int(found.get(key, 0)) for key in keys) for _, keys in ORIENTATION_CLASSES]
    return totals, int(found.get("cpdag_shd", 0)), found


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _panel_grid(n, width, height, t):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False, facecolor=t["surface"])
    return fig, list(axes[0])


def fig_edges(runs, t, series, path):
    """Which pairs PC got, per run. Fill/outline + glyph, so colour is never the only cue.

    Colour encodes the OUTCOME here, not the run — slot 1 for "in the true graph", slot 2
    for "invented" — and stays fixed across panels; the run is named in the panel title.
    Colouring by run instead would repaint the classes panel to panel and contradict the
    legend, which is the recolour-on-filter mistake in another costume.
    """
    truth_colour, spurious_colour = series[0], series[1]
    fig, axes = _panel_grid(len(runs), 5.0, 4.6, t)
    fig.subplots_adjust(wspace=0.55)
    rows = []
    for panel, (ax, (label, run)) in enumerate(zip(axes, runs)):
        names, classes = edge_classes(run)
        n = len(names)
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        for i in range(n):
            for j in range(i + 1, n):
                kind = classes.get((i, j))
                # 2px-equivalent surface gap between touching cells.
                x, y, w = j - 0.5 + 0.03, i - 0.5 + 0.03, 1 - 0.06
                if kind is None:
                    ax.add_patch(Rectangle((x, y), w, w, facecolor=t["grid"], edgecolor="none", alpha=0.35))
                    continue
                fill = {"tp": truth_colour, "fn": t["surface"], "fp": spurious_colour}[kind]
                edge = truth_colour if kind in ("tp", "fn") else spurious_colour
                ax.add_patch(Rectangle((x, y), w, w, facecolor=fill, edgecolor=edge, linewidth=1.6))
                glyph, ink = {"tp": ("✓", t["surface"]), "fn": ("−", truth_colour), "fp": ("+", t["surface"])}[kind]
                ax.text(j, i, glyph, ha="center", va="center", fontsize=9, color=ink)
                rows.append([label, names[i], names[j], {"tp": "recovered", "fn": "missed", "fp": "spurious"}[kind]])
        ax.set_xlim(0.5, n - 0.5)
        ax.set_ylim(n - 1.5, -0.5)
        ax.set_xticks(range(1, n))
        ax.set_yticks(range(n - 1))
        ax.set_xticklabels(names[1:], rotation=90, fontsize=7.5, color=t["ink2"])
        # Only the leftmost panel carries row names: repeating them puts the next panel's
        # labels on top of this one's cells.
        ax.set_yticklabels(names[:-1] if panel == 0 else [], fontsize=7.5, color=t["ink2"])
        ax.tick_params(colors=t["muted"], length=0)
        best = run.get("best") or {}
        ax.set_title(label, color=t["ink"], fontsize=10, loc="left", pad=20)
        ax.text(
            0.0,
            1.0,
            f"alpha {best.get('alpha', '—')} · SHD {best.get('skeleton_shd', '—')}"
            f" · {sum(1 for v in classes.values() if v == 'tp')}/{sum(1 for v in classes.values() if v != 'fp')} found",
            transform=ax.transAxes,
            color=t["muted"],
            fontsize=8.5,
            va="bottom",
            ha="left",
        )
    handles = [
        Patch(facecolor=truth_colour, edgecolor=truth_colour, label="✓ recovered — true edge, found"),
        Patch(facecolor=t["surface"], edgecolor=truth_colour, linewidth=1.6, label="−  missed — true edge, not found"),
        Patch(facecolor=spurious_colour, edgecolor=spurious_colour, label="+  spurious — found, not a true edge"),
    ]
    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.28),
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink2"],
        handlelength=1.2,
    )
    _save(fig, path, t)
    _write_csv(path.replace(".png", ".csv"), ["run", "factor_a", "factor_b", "outcome"], rows)


def fig_alpha_sweep(runs, t, series, path):
    """Four panels, not two y-axes: F1/precision/recall share 0-1, SHD is a count."""
    metrics = [("F1", 1), ("precision", 2), ("recall", 3), ("skeleton SHD", 4)]
    fig, axes = _panel_grid(len(metrics), 3.0, 2.9, t)
    rows = []
    for ax, (title, idx) in zip(axes, metrics):
        for (label, run), colour in zip(runs, series):
            data = sweep_rows(run)
            if not data:
                continue
            alphas = [d[0] for d in data]
            values = [d[idx] for d in data]
            ax.plot(alphas, values, color=colour, linewidth=2.0, marker="o", markersize=5, label=label, zorder=3)
            chosen = _finite((run.get("best") or {}).get("alpha"))
            if chosen is not None and chosen in alphas:
                pick = alphas.index(chosen)
                # A 2px surface ring marks the alpha the panel actually reported.
                ax.plot(
                    chosen,
                    values[pick],
                    marker="o",
                    markersize=10,
                    markerfacecolor="none",
                    markeredgecolor=colour,
                    markeredgewidth=2.0,
                    zorder=4,
                )
        _style(ax, t, xlabel="alpha", title=title)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.6, zorder=0)
        ax.set_xscale("log")
        if idx != 4:
            ax.set_ylim(-0.02, 1.02)
    for label, run in runs:
        for alpha, f1, precision, recall, shd in sweep_rows(run):
            rows.append([label, alpha, f1, precision, recall, shd])
    if len(runs) > 1:
        axes[0].legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="best")
    fig.text(
        0.0,
        -0.12,
        "The ringed point is the alpha the report selected, against the truth. Runs whose curves cross "
        "are not comparable at their own best alphas.",
        color=t["muted"],
        fontsize=8.5,
        ha="left",
    )
    _save(fig, path, t)
    _write_csv(path.replace(".png", ".csv"), ["run", "alpha", "f1", "precision", "recall", "skeleton_shd"], rows)


def fig_factor_r2(runs, t, series, path):
    """Partial R² solid, the parent-mediated remainder recessive behind it."""
    fig, axes = _panel_grid(len(runs), 4.6, 4.2, t)
    fig.subplots_adjust(wspace=0.35)
    rows = []
    for panel, (ax, (label, run), colour) in enumerate(zip(axes, runs, series)):
        factors = run["factors"]
        names = [f"{f.get('name', f['dim'])}  ({len(f['parents'])}p)" for f in factors]
        y = range(len(factors))
        for pos, factor in zip(y, factors):
            raw, partial = _f(factor["raw_r2"]), _f(factor["partial_r2"])
            lo = min(partial, raw)
            if raw > partial:  # the parent-mediated part, behind and recessive
                ax.barh(pos, raw - lo - 0.004, left=lo + 0.004, height=0.36, color=t["floor_fill"], zorder=2)
            ax.barh(pos, partial, height=0.36, color=colour, zorder=3)
            rows.append([label, factor.get("name", factor["dim"]), len(factor["parents"]), raw, partial])
        ax.set_yticks(list(y))
        # Factor names are long; repeating them lands the next panel's labels on this
        # panel's bars. One column of names serves every panel — the rows are the same.
        ax.set_yticklabels(names if panel == 0 else [], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=t["axis"], linewidth=0.8, zorder=1)
        _style(
            ax,
            t,
            xlabel="R²",
            title=label,
            subtitle=f"mean partial R² {_f(run.get('partial_r2_mean')):+.3f}",
        )
    axes[0].legend(
        handles=[Patch(facecolor=t["floor_fill"], label="remainder of raw R² — read through SCM parents")],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink2"],
        handlelength=1.2,
    )
    _save(fig, path, t)
    _write_csv(path.replace(".png", ".csv"), ["run", "factor", "n_parents", "raw_r2", "partial_r2"], rows)


def fig_orientation(runs, t, ramp, path):
    """Ordered classes -> a single-hue ordinal ramp, with every count direct-labelled."""
    scored = [(label, run, orientation_counts(run)) for label, run in runs]
    scored = [(label, run, found) for label, run, found in scored if found]
    if not scored:
        logger.info("no orientation data in these runs; skipping %s (re-run with --orientation)", path)
        return
    fig, ax = plt.subplots(figsize=(7.4, 0.9 + 0.62 * len(scored)), facecolor=t["surface"])
    rows = []
    for pos, (label, run, (totals, cpdag_shd, raw)) in enumerate(scored):
        left = 0
        for (name, _keys), value, colour in zip(ORIENTATION_CLASSES, totals, ramp):
            if value:
                ax.barh(pos, value - 0.04, left=left + 0.02, height=0.36, color=colour, zorder=3)
                ax.text(
                    left + value / 2,
                    pos,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=t["surface"] if colour == ramp[-1] else t["ink"],
                )
            left += value
        ax.text(left + 0.25, pos, f"CPDAG SHD {cpdag_shd}", va="center", fontsize=8.5, color=t["ink2"])
        rows.append([label, *totals, cpdag_shd, *[raw.get(k, 0) for _, keys in ORIENTATION_CLASSES for k in keys]])
    ax.set_yticks(range(len(scored)))
    ax.set_yticklabels([label for label, _, _ in scored], fontsize=8.5)
    ax.invert_yaxis()
    _style(
        ax,
        t,
        xlabel="edges adjacent in both the estimate and the true CPDAG",
        title="Edge directions vs the true CPDAG",
        subtitle="PC identifies an equivalence class, so 'matched' includes correctly leaving an edge undirected",
    )
    ax.legend(
        handles=[Patch(facecolor=c, label=n) for (n, _), c in zip(ORIENTATION_CLASSES, ramp)],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22 - 0.02 * len(scored)),
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink2"],
        ncol=3,
        handlelength=1.2,
    )
    _save(fig, path, t)
    header = ["run", *[n for n, _ in ORIENTATION_CLASSES], "cpdag_shd"]
    header += [k for _, keys in ORIENTATION_CLASSES for k in keys]
    _write_csv(path.replace(".png", ".csv"), header, rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", nargs="+", required=True, help="run_causal_recovery causal_recovery.json file(s).")
    p.add_argument("--labels", nargs="*", default=None, help="Run labels (default: each run's directory name).")
    p.add_argument("--out", default="figures", help="Output directory.")
    p.add_argument("--dark", action="store_true", help="Re-step to the dark surface.")
    cli = p.parse_args(argv)

    runs = [entry for path in cli.json for entry in load(path)]
    if not runs:
        p.error("no scored runs in those files — every run was skipped or errored")
    t = THEME["dark" if cli.dark else "light"]
    if len(runs) > len(t["series"]):
        p.error(
            f"{len(runs)} runs but only {len(t['series'])} categorical hues — they are assigned in fixed "
            "order and never generated. Plot them in pairs, or split the report."
        )
    labels = cli.labels or [label for label, _ in runs]
    if len(labels) != len(runs):
        p.error(f"--labels needs one entry per scored run ({len(runs)} here)")
    runs = [(label, run) for label, (_, run) in zip(labels, runs)]

    os.makedirs(cli.out, exist_ok=True)
    series = t["series"]
    fig_edges(runs, t, series, os.path.join(cli.out, "edges.png"))
    fig_alpha_sweep(runs, t, series, os.path.join(cli.out, "alpha_sweep.png"))
    fig_factor_r2(runs, t, series, os.path.join(cli.out, "factor_r2.png"))
    fig_orientation(runs, t, ORDINAL["dark" if cli.dark else "light"], os.path.join(cli.out, "orientation.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
