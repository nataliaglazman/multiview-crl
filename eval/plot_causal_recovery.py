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
``truth_graph.png``  The SCM itself, drawn once: every true edge as an arc over the factors
                     in causal order, arrow on the child. Each arc is shaded by how many of
                     the plotted runs recovered it, so the SPANS answer what a per-pair
                     matrix cannot — whether what PC misses is the long-range structure or
                     the local structure.
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


def load(path, roles=("trained",)):
    """``[(label, run), ...]`` for the scored runs of the requested roles in one report file.

    ``--floor``/``--ceiling`` put reference rows in the same file as the runs they reference,
    so a report of two arms with three floor seeds holds nine scored rows against two
    categorical hues.  Selecting on role is what keeps a figure to the runs it is about.  A
    row from before those flags existed carries no role and counts as ``trained``.
    """
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
        if run.get("role", "trained") not in roles:
            continue
        out.append((os.path.basename(str(run.get("run_dir", "run")).rstrip("/")), run))
    return out


def factor_names(run):
    return [factor.get("name", f"d{factor['dim']}") for factor in run["factors"]]


def truth_edges(run):
    """``(edges, directed)`` from the report's stored true graph.

    Direction comes from ``true_dag``. ``true_skeleton`` is its symmetrisation and is the
    fallback for a report written before the DAG was stored -- but then the pairs are
    UNORDERED, so ``directed`` is False and callers must not draw an arrowhead on them.
    Ordering a skeleton's pairs by index would silently assert a direction the report does
    not contain, and on this generator it would even look right, since it only ever draws
    parents with a lower index.
    """
    dag = run.get("true_dag")
    if dag:
        return [(i, j) for i in range(len(dag)) for j in range(len(dag)) if dag[i][j]], True
    skeleton = run.get("true_skeleton") or []
    pairs = [(i, j) for i in range(len(skeleton)) for j in range(i + 1, len(skeleton)) if skeleton[i][j]]
    return pairs, False


def truth_recovery(runs):
    """``{(i, j): count}`` -- how many of these runs recovered each true pair, i < j.

    Keyed on the unordered pair because the skeleton metrics are: PC is scored on
    adjacency, and an edge it orients backwards still counts as recovered here.
    """
    found = {}
    for _label, run in runs:
        _names, classes = edge_classes(run)
        for pair, kind in classes.items():
            if kind == "tp":
                found[pair] = found.get(pair, 0) + 1
    return found


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
    # The truth panel first, so every run panel is read against the graph it was scored on
    # rather than against the reader's memory of it. It is a panel, not a colour: the
    # outcome classes below already own both hues, and repainting them per panel is the
    # recolour-on-filter mistake this docstring warns about.
    fig, axes = _panel_grid(len(runs) + 1, 5.0, 4.6, t)
    fig.subplots_adjust(wspace=0.55)
    rows = []
    panels = [("ground truth", runs[0][1], True)] + [(label, run, False) for label, run in runs]
    for panel, (ax, (label, run, is_truth)) in enumerate(zip(axes, panels)):
        names, classes = edge_classes(run)
        if is_truth:
            # Direction is kept here and nowhere else in this figure: the skeleton metrics
            # score adjacency only, so a run panel has no direction to show.
            edges, has_direction = truth_edges(run)
            directed = {(min(p, c), max(p, c)): ("→" if p < c else "←") if has_direction else "·" for p, c in edges}
            classes = dict.fromkeys(directed, "truth")
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
                fill = {"tp": truth_colour, "fn": t["surface"], "fp": spurious_colour, "truth": truth_colour}[kind]
                edge = spurious_colour if kind == "fp" else truth_colour
                ax.add_patch(Rectangle((x, y), w, w, facecolor=fill, edgecolor=edge, linewidth=1.6))
                glyph, ink = {
                    "tp": ("✓", t["surface"]),
                    "fn": ("−", truth_colour),
                    "fp": ("+", t["surface"]),
                    "truth": (directed.get((i, j), "·"), t["surface"]),
                }[kind]
                ax.text(j, i, glyph, ha="center", va="center", fontsize=9, color=ink)
                outcome = {"tp": "recovered", "fn": "missed", "fp": "spurious", "truth": "true edge"}[kind]
                rows.append([label, names[i], names[j], outcome])
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
        subtitle = (
            f"{len(classes)} edges · " + ("arrow points parent → child" if has_direction else "skeleton only")
            if is_truth
            else f"alpha {best.get('alpha', '—')} · SHD {best.get('skeleton_shd', '—')}"
            f" · {sum(1 for v in classes.values() if v == 'tp')}/{sum(1 for v in classes.values() if v != 'fp')} found"
        )
        ax.text(0.0, 1.0, subtitle, transform=ax.transAxes, color=t["muted"], fontsize=8.5, va="bottom", ha="left")
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


def fig_truth_graph(runs, t, ramp, path):
    """The SCM itself: every true edge as an arc over the factors in causal order.

    ``edges.png`` shows the truth only per run and only as a cell's colour, which answers
    "was this pair an edge" but never "what does the graph look like". This draws it once,
    with direction, and shades each arc by how many of the plotted runs recovered it.

    An arc diagram rather than a spring layout, because the factors have a real order --
    the generator only ever draws parents with a lower index -- and reading the SPANS is
    the question a matrix cannot answer: whether what PC misses is the long-range structure
    or the local structure. Recovery is an ordered class, so it takes the ordinal ramp, with
    a dotted line as the second cue for the edges nothing recovered.
    """
    import numpy as np

    _label, first = runs[0]
    names = factor_names(first)
    edges, has_direction = truth_edges(first)
    if not edges:
        logger.info("no true graph stored in these runs; skipping %s", path)
        return
    disagreeing = [lbl for lbl, run in runs if truth_edges(run)[0] != edges]
    if disagreeing:
        logger.warning(
            "these runs were scored against DIFFERENT true graphs (%s); drawing the first run's",
            ", ".join(disagreeing),
        )
    found = truth_recovery(runs)
    n_runs = len(runs)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * n, 4.4), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    rows = []
    for parent, child in edges:
        lo, hi = min(parent, child), max(parent, child)
        count = found.get((lo, hi), 0)
        tier = 0 if count == 0 else (1 if count < n_runs else 2)
        colour, dashes = ramp[tier], (1.4, 1.8) if tier == 0 else ()
        centre, radius = (parent + child) / 2.0, abs(child - parent) / 2.0
        theta = np.linspace(np.pi, 0.0, 80) if parent < child else np.linspace(0.0, np.pi, 80)
        x, y = centre + radius * np.cos(theta), radius * np.sin(theta)
        line = ax.plot(x, y, color=colour, linewidth=1.9, solid_capstyle="round", zorder=3)[0]
        if dashes:
            line.set_dashes(dashes)
        # The head sits ~90% along rather than at the endpoint: the node markers are drawn
        # last and would paint over a head placed on the node itself, silently losing the
        # one thing this figure adds over the skeleton matrix.
        if has_direction:
            ax.annotate(
                "",
                xy=(x[-8], y[-8]),
                xytext=(x[-14], y[-14]),
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45", color=colour, shrinkA=0, shrinkB=0),
                zorder=6,
            )
        rows.append([names[parent], names[child], hi - lo, count, n_runs])
    ax.scatter(range(n), [0] * n, s=64, color=t["surface"], edgecolors=t["axis"], linewidths=1.4, zorder=5)
    for i, name in enumerate(names):
        ax.text(i, -0.22, name, rotation=45, ha="right", va="top", fontsize=8, color=t["ink2"])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.35, max(abs(c - p) for p, c in edges) / 2.0 * 1.12 + 0.15)
    ax.set_title("Ground-truth causal graph", color=t["ink"], fontsize=10, loc="left", pad=18)
    missed = sum(1 for p, c in edges if found.get((min(p, c), max(p, c)), 0) == 0)
    ax.text(
        0.0,
        1.0,
        f"{len(edges)} edges over {n} factors · {missed} recovered by no run · "
        + ("factors left to right in causal order" if has_direction else "skeleton only — no direction stored"),
        transform=ax.transAxes,
        color=t["muted"],
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    labels = ["recovered by no run", f"recovered by some ({n_runs} plotted)", "recovered by every run"]
    handles = [
        plt.Line2D([], [], color=ramp[k], linewidth=1.9, linestyle=":" if k == 0 else "-", label=labels[k])
        for k in (2, 1, 0)
        if n_runs > 1 or k != 1
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.26),
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink2"],
        handlelength=1.8,
    )
    _save(fig, path, t)
    _write_csv(path.replace(".png", ".csv"), ["parent", "child", "span", "runs_recovering", "runs_plotted"], rows)


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
    p.add_argument(
        "--roles",
        nargs="+",
        default=["trained"],
        choices=["trained", "floor", "ceiling"],
        help="Which rows to draw (default: trained). A report written with --floor/--ceiling holds "
        "their reference rows too, and there are only two categorical hues, so pick a pair: "
        "'--roles trained ceiling' for one arm against what the protocol can reach, or "
        "'--roles trained floor' for one arm against one untrained seed.",
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="SUBSTRING",
        help="Keep only rows whose name contains one of these (case-insensitive). Needed to pair "
        "ONE of several arms with its reference row: two arms plus a ceiling is three rows against "
        "two hues, so '--roles trained ceiling --only ident-vent' picks the arm and its ceiling.",
    )
    cli = p.parse_args(argv)

    runs = [entry for path in cli.json for entry in load(path, tuple(cli.roles))]
    if cli.only:
        wanted = [s.lower() for s in cli.only]
        runs = [(name, run) for name, run in runs if any(s in name.lower() for s in wanted)]
    if not runs:
        p.error(
            f"no scored runs with role(s) {', '.join(cli.roles)}"
            + (f" matching {', '.join(cli.only)}" if cli.only else "")
            + " in those files"
        )
    t = THEME["dark" if cli.dark else "light"]
    if len(runs) > len(t["series"]):
        p.error(
            f"{len(runs)} runs but only {len(t['series'])} categorical hues — they are assigned in fixed "
            "order and never generated. Narrow with --roles/--only, or split the report.\n  got: "
            + ", ".join(name for name, _ in runs)
        )
    labels = cli.labels or [label for label, _ in runs]
    if len(labels) != len(runs):
        p.error(f"--labels needs one entry per scored run ({len(runs)} here)")
    runs = [(label, run) for label, (_, run) in zip(labels, runs)]

    os.makedirs(cli.out, exist_ok=True)
    series = t["series"]
    fig_truth_graph(runs, t, ORDINAL["dark" if cli.dark else "light"], os.path.join(cli.out, "truth_graph.png"))
    fig_edges(runs, t, series, os.path.join(cli.out, "edges.png"))
    fig_alpha_sweep(runs, t, series, os.path.join(cli.out, "alpha_sweep.png"))
    fig_factor_r2(runs, t, series, os.path.join(cli.out, "factor_r2.png"))
    fig_orientation(runs, t, ORDINAL["dark" if cli.dark else "light"], os.path.join(cli.out, "orientation.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
