#!/usr/bin/env python
"""Figures from ``score_checkpoint``'s JSON: one checkpoint's recovery, drawn.

Reads the JSON ``eval.score_checkpoint`` already writes, so plotting never re-scores and a
figure can never disagree with the table it came from.

    python -m eval.score_checkpoint --run-dir results/dummy_infonce
    python -m eval.plot_score_checkpoint --json results/dummy_infonce/score_report.json \
        --out figures/

What it draws
-------------
``per_factor_recovery.png``  The headline, and the one the block means hide: ridge R² per
                             ground-truth factor beside the untrained floor's, with the
                             delta called out. A model can average respectably while
                             carrying two coarse factors and leaving seven at chance, and
                             only this figure says which.
``summary.png``              The three block-level numbers, trained against floor. Two
                             panels, not one axis: block-MCC and R² are higher-is-better on
                             [0,1], while content->view accuracy targets 0.5 (chance =
                             view-invariant content), so putting them on a shared axis
                             would read a good leakage score as a bad one.
``graph_recovery.png``       Skeleton F1/precision/recall for the decoded embeddings against
                             the two reference panels the score run computes — the untrained
                             floor and PC on the ground-truth factors. SHD sits in its own
                             panel because it is a count, not a rate. Drawn only when the
                             run had an SCM to score against.

Every figure carries a ``.csv`` twin with the same numbers, since a colour-encoded figure
should never be the only way to read a value.

It also writes ``causal_panels.json``, the graph panels reshaped into the ``{"runs": [...]}``
form ``plot_causal_recovery`` reads, each tagged with that script's ``role`` (embeddings =
``trained``, floor = ``floor``, ground-truth = ``ceiling``), so the edge map, alpha sweep and
per-factor partial-R² figures come from that script rather than being drawn twice here:

    python -m eval.plot_causal_recovery --json figures/causal_panels.json --out figures/
    python -m eval.plot_causal_recovery --json figures/causal_panels.json --roles trained ceiling

Colour
------
The validated categorical slots are imported from ``plot_identifiability`` rather than
restated, so a colour means the same thing across every figure in the repo. The floor is
deliberately NOT a third hue: it is chart chrome (recessive grey), because it is a
reference the model is measured against rather than a peer series.
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

from eval.plot_identifiability import BAR_H, THEME, _f, _save, _style, _write_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# The order score_checkpoint emits them in; also the order they are drawn.
GRAPH_PANELS = (("graph", "embeddings"), ("graph_floor", "untrained floor"), ("graph_truth", "ground-truth factors"))


def _group(n, band=0.78, gap=0.18):
    """``(bar_height, [offset per series])`` leaving a visible gap between adjacent fills.

    matplotlib will happily draw grouped bars flush against each other; the spacer is what
    keeps two fills from reading as one.
    """
    slot = band / n
    return slot * (1.0 - gap), [(i - (n - 1) / 2) * slot for i in range(n)]


def _legend_below(ax, t, ncol):
    """Legends go under the axes. Inside a horizontal bar chart every corner is occupied by
    a bar at some data value, so an in-axes legend collides as soon as the data changes."""
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink2"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
        handlelength=1.4,
        columnspacing=1.6,
    )


def load(path):
    with open(path) as fh:
        payload = json.load(fh)
    if "trained" not in payload:
        raise ValueError(f"{path}: no 'trained' key — is this a score_checkpoint output file?")
    return payload


def factor_rows(report):
    """``[(name, trained_r2, floor_r2_or_None, block_mcc, block_mcc_std, channel_mcc), ...]``."""
    trained, floor = report["trained"]["per_factor"], (report.get("floor") or {}).get("per_factor")
    rows = []
    for name, v in trained.items():
        f = None if floor is None else _f(floor[name]["ridge_r2"])
        rows.append((name, _f(v["ridge_r2"]), f, _f(v["mcc"]), _f(v["mcc_std"]), _f(v.get("channel_mcc"))))
    return rows


def fig_per_factor(report, t, path):
    """Grouped bars per factor. Grouped rather than stacked because the delta is signed:
    a factor the model made WORSE than random init is a real outcome and a stacked
    increment cannot draw it."""
    rows = factor_rows(report)
    has_floor = any(r[2] is not None for r in rows)
    names = [r[0] for r in rows]
    ys = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 0.46 * len(rows) + 2.0), facecolor=t["surface"])
    h, offs = _group(2 if has_floor else 1)
    ax.barh(ys + offs[0], [r[1] for r in rows], height=h, color=t["series"][0], zorder=3, label="trained")
    if has_floor:
        ax.barh(ys + offs[1], [r[2] for r in rows], height=h, color=t["floor_fill"], zorder=3, label="untrained floor")
    ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    if has_floor:
        # Selective direct labels: the delta only, at the right margin. It is the quantity
        # the figure exists to communicate, and it cannot be read off two bar lengths.
        xmax = max([r[1] for r in rows] + [r[2] for r in rows] + [0.0])
        for y, r in zip(ys, rows):
            ax.text(
                xmax * 1.06,
                y,
                f"{r[1] - r[2]:+.3f}",
                color=t["ink2"] if r[1] - r[2] > 0 else t["muted"],
                fontsize=8.5,
                va="center",
                ha="left",
            )
        ax.text(xmax * 1.06, ys[0] + 0.85, "Δ", color=t["ink"], fontsize=9, va="center", ha="left")

    _style(
        ax,
        t,
        xlabel="cross-validated ridge R²",
        title="Per-factor recovery",
        subtitle=f"{report.get('pooling', '?')} pooling, {report.get('num_samples', '?')} samples"
        + ("  ·  Δ = trained − untrained floor" if has_floor else "  ·  NO FLOOR: nothing here is reportable"),
    )
    if has_floor:
        _legend_below(ax, t, 2)
    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        [
            "factor",
            "trained_ridge_r2",
            "floor_ridge_r2",
            "delta",
            "trained_block_mcc",
            "block_mcc_std",
            "trained_channel_mcc",
        ],
        [[r[0], r[1], r[2], (None if r[2] is None else r[1] - r[2]), r[3], r[4], r[5]] for r in rows],
    )


def fig_summary(report, t, path):
    """Two panels: the higher-is-better pair, and view leakage whose target is 0.5."""
    trained, floor = report["trained"], report.get("floor")
    # Every row here is higher-is-better on [0,1], which is what lets them share one axis.
    # DCI is flattened in from its sub-dict; rows absent from an older report are dropped
    # rather than plotted as zero.
    flat_t = dict(trained, **{f"dci_{k}": v for k, v in (trained.get("dci") or {}).items()})
    flat_f = None if not floor else dict(floor, **{f"dci_{k}": v for k, v in (floor.get("dci") or {}).items()})
    left = [
        ("block MCC", "block_mcc"),
        ("channel MCC", "channel_mcc"),
        ("ridge R\u00b2 (mean)", "ridge_r2_mean"),
        ("DCI disentangle.", "dci_disentanglement"),
        ("DCI completeness", "dci_completeness"),
        ("DCI informativeness", "dci_informativeness_test"),
    ]
    left = [(lbl, k) for lbl, k in left if flat_t.get(k) is not None]
    h, offs = _group(2 if floor else 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), facecolor=t["surface"], gridspec_kw={"width_ratios": [2, 1]})

    ys = np.arange(len(left))[::-1]
    axes[0].barh(
        ys + offs[0], [_f(flat_t[k]) for _, k in left], height=h, color=t["series"][0], zorder=3, label="trained"
    )
    if floor:
        axes[0].barh(
            ys + offs[1],
            [_f(flat_f[k]) for _, k in left],
            height=h,
            color=t["floor_fill"],
            zorder=3,
            label="untrained floor",
        )
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lbl for lbl, _ in left])
    axes[0].set_ylim(-0.7, len(left) - 0.3)
    axes[0].axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
    _style(axes[0], t, xlabel="score (higher is better)", title="Block-level recovery")
    if floor:
        _legend_below(axes[0], t, 2)

    y2 = np.array([0.0])
    axes[1].barh(y2 + offs[0], [_f(trained["content_to_view_acc"])], height=h, color=t["series"][0], zorder=3)
    if floor:
        axes[1].barh(y2 + offs[1], [_f(floor["content_to_view_acc"])], height=h, color=t["floor_fill"], zorder=3)
    axes[1].axvline(0.5, color=t["series"][1], linewidth=1.5, zorder=4, linestyle=(0, (4, 2)))
    # Blended transform: x in data (pinned to the line), y in axes fraction (stays inside
    # the panel no matter how few bars it holds).
    axes[1].text(
        0.5,
        0.97,
        " chance",
        transform=axes[1].get_xaxis_transform(),
        color=t["series"][1],
        fontsize=8.5,
        va="top",
        ha="left",
    )
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(["content\u2192view acc"])
    axes[1].set_ylim(-0.7, 0.7)
    # Zero baseline, not 0.4: a bar truncated at its own floor overstates every difference.
    axes[1].set_xlim(0, 1.02)
    _style(axes[1], t, xlabel="accuracy (0.5 is the target)", title="View leakage")

    _save(fig, path, t)
    rows = [[lbl, _f(flat_t[k]), (None if not floor else _f(flat_f[k]))] for lbl, k in left]
    rows.append(
        [
            "content_to_view_acc",
            _f(trained["content_to_view_acc"]),
            None if not floor else _f(floor["content_to_view_acc"]),
        ]
    )
    _write_csv(os.path.splitext(path)[0] + ".csv", ["metric", "trained", "floor"], rows)


def graph_panels(report):
    """``[(label, panel), ...]`` for whichever graph panels the score run produced."""
    return [(label, report[key]) for key, label in GRAPH_PANELS if isinstance(report.get(key), dict)]


def fig_graph(report, t, path):
    """Rates and SHD in separate panels: a count and a ratio do not share an axis."""
    panels = graph_panels(report)
    if not panels:
        return False
    rates = [("F1", "f1"), ("precision", "precision"), ("recall", "recall")]
    # The model is the subject; the floor is chrome and the truth panel is the upper bound.
    colours = {"embeddings": t["series"][0], "untrained floor": t["floor_fill"], "ground-truth factors": t["series"][1]}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2), facecolor=t["surface"], gridspec_kw={"width_ratios": [2.2, 1]})
    n = len(panels)
    h = min(BAR_H, 0.8 / n)
    ys = np.arange(len(rates))[::-1]
    for i, (label, panel) in enumerate(panels):
        best = panel.get("best") or {}
        offs = (i - (n - 1) / 2) * (h + 0.015)
        axes[0].barh(
            ys - offs,
            [_f(best.get(k, float("nan"))) for _, k in rates],
            height=h,
            color=colours[label],
            zorder=3,
            label=label,
        )
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lbl for lbl, _ in rates])
    axes[0].set_xlim(0, 1.02)
    _style(
        axes[0],
        t,
        xlabel="skeleton score at each panel's best alpha",
        title="Causal graph recovery",
        subtitle="PC skeleton vs the generator's SCM",
    )
    # Only the left panel carries the legend: the right panel tick-labels each bar, so a
    # second legend there would repeat the same three strings beside themselves.
    _legend_below(axes[0], t, n)

    ys2 = np.arange(n)[::-1]
    axes[1].barh(
        ys2,
        [_f((p.get("best") or {}).get("skeleton_shd", float("nan"))) for _, p in panels],
        height=_group(1)[0],
        color=[colours[lbl] for lbl, _ in panels],
        zorder=3,
    )
    shds = [_f((p.get("best") or {}).get("skeleton_shd", float("nan"))) for _, p in panels]
    for y, v in zip(ys2, shds):
        # SHD 0 is a perfect score and draws no bar; without the label it reads as missing.
        axes[1].text(v + max(shds + [1]) * 0.02, y, f"{v:.0f}", color=t["ink2"], fontsize=8.5, va="center", ha="left")
    axes[1].set_yticks(ys2)
    axes[1].set_yticklabels([lbl for lbl, _ in panels])
    axes[1].set_ylim(-0.7, n - 0.3)
    axes[1].set_xlim(0, max(shds + [1]) * 1.12)
    _style(axes[1], t, xlabel="skeleton SHD (lower is better)", title="Structural distance")

    _save(fig, path, t)
    _write_csv(
        os.path.splitext(path)[0] + ".csv",
        ["panel", "alpha", "f1", "precision", "recall", "skeleton_shd", "tp", "fp", "fn"],
        [
            [lbl]
            + [
                (p.get("best") or {}).get(k)
                for k in ("alpha", "f1", "precision", "recall", "skeleton_shd", "tp", "fp", "fn")
            ]
            for lbl, p in panels
        ],
    )
    return True


# plot_causal_recovery selects rows by ``role`` and holds two categorical hues, so the
# panels are tagged with the roles it understands rather than filtered down here. Its
# default (--roles trained) then draws the embeddings alone, and a pair is one flag away:
# --roles trained floor, or --roles trained ceiling.
PANEL_ROLES = {"embeddings": "trained", "untrained floor": "floor", "ground-truth factors": "ceiling"}


def write_causal_panels(report, path):
    """Reshape the graph panels into plot_causal_recovery's ``{"runs": [...]}`` input.

    That script already draws the edge map, the alpha sweep and per-factor partial R²; the
    panels come out of ``evaluate_arrays`` with exactly the keys it reads, so they are
    handed over rather than redrawn here.
    """
    runs = [dict(panel, run_dir=label, role=PANEL_ROLES[label]) for label, panel in graph_panels(report)]
    if not runs:
        return None
    with open(path, "w") as fh:
        json.dump({"runs": runs}, fh, indent=2, default=float)
    logger.info("wrote %s", path)
    return path


def render(report, out_dir, dark=False):
    """Draw every figure the report supports. Returns the paths written."""
    t = THEME["dark" if dark else "light"]
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, fn in (("per_factor_recovery.png", fig_per_factor), ("summary.png", fig_summary)):
        p = os.path.join(out_dir, name)
        fn(report, t, p)
        written.append(p)
    gp = os.path.join(out_dir, "graph_recovery.png")
    if fig_graph(report, t, gp):
        written.append(gp)
        cp = write_causal_panels(report, os.path.join(out_dir, "causal_panels.json"))
        if cp:
            written.append(cp)
            logger.info(
                "edge map / alpha sweep: python -m eval.plot_causal_recovery --json %s --out %s "
                "[--roles trained ceiling]",
                cp,
                out_dir,
            )
    else:
        logger.info("no graph panels in the report (run scored without an SCM) — skipping graph figures")
    return written


def _self_test():
    """Render every figure from a synthetic report, including the no-floor path."""
    import tempfile

    names = ["brain_size", "ventricle_size", "lesion_x"]
    mk = lambda scale: {  # noqa: E731
        "block_mcc": 0.5 * scale,
        "ridge_r2_mean": 0.3 * scale,
        "content_to_view_acc": 0.6,
        "assignment_identity": 1.0,
        "channel_mcc": 0.4 * scale,
        # DCI on BOTH trained and floor. Rows are dropped when absent, so a report without
        # them exercises none of the flattening — which is how a floor-side KeyError on
        # exactly these keys got past this test once already.
        "dci": {
            "disentanglement": 0.3 * scale,
            "completeness": 0.25 * scale,
            "informativeness_test": 0.5 * scale,
            "informativeness_train": 0.9 * scale,
        },
        "per_factor": {
            n: {"ridge_r2": 0.4 * scale - 0.1 * i, "mcc": 0.6 * scale, "mcc_std": 0.02, "channel_mcc": 0.5 * scale}
            for i, n in enumerate(names)
        },
    }
    panel = {
        "best": {
            "alpha": 0.05,
            "f1": 0.7,
            "precision": 0.8,
            "recall": 0.6,
            "skeleton_shd": 3,
            "tp": 3,
            "fp": 1,
            "fn": 2,
            "adjacency": [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        },
        "alpha_sweep": [{"alpha": 0.05, "f1": 0.7, "precision": 0.8, "recall": 0.6, "skeleton_shd": 3}],
        "factors": [{"dim": i, "name": n, "raw_r2": 0.5, "partial_r2": 0.3} for i, n in enumerate(names)],
        "true_skeleton": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    }
    report = {
        "pooling": "gap",
        "num_samples": 400,
        "trained": mk(1.0),
        "floor": mk(0.5),
        "graph": panel,
        "graph_floor": panel,
        "graph_truth": panel,
    }
    with tempfile.TemporaryDirectory() as d:
        got = render(report, d)
        assert len([p for p in got if p.endswith(".png")]) == 3, got
        for p in got:
            assert os.path.getsize(p) > 0, p
        assert os.path.exists(os.path.join(d, "per_factor_recovery.csv"))
        assert json.load(open(os.path.join(d, "causal_panels.json")))["runs"][0]["run_dir"] == "embeddings"
        # No floor, no graph: must still render without a KeyError.
        got2 = render({"pooling": "gap", "num_samples": 10, "trained": mk(1.0)}, d)
        assert len(got2) == 2, got2
    print("self-test OK (3 figures + csv twins + causal_panels.json; no-floor/no-graph path renders)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", help="score_checkpoint --out JSON file.")
    p.add_argument("--out", default="figures", help="Output directory.")
    p.add_argument("--dark", action="store_true", help="Re-step to the dark surface.")
    p.add_argument("--self-test", action="store_true")
    cli = p.parse_args()
    if cli.self_test:
        _self_test()
        return
    if not cli.json:
        raise SystemExit("--json is required (or pass --self-test)")
    render(load(cli.json), cli.out, dark=cli.dark)


if __name__ == "__main__":
    main()
