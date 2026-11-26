"""Stage 82 / 82b figure: identify wide, chart narrow. Reads chart_check.json (stage 82) and b_*.json (82b) beside this
file and never re-scores; cells still running are marked pending. Writes figures/fig_chart_narrow.{png,csv}.

    python make_chart_figure.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
CHART, LINEAR, WIDE = "#257de0", "#f47230", "#00b894"
GREY, REF, INK, MUTE, RULE = "#a3a9b8", "#3d3d45", "#191919", "#666666", "#dddddd"
FACTORS = ["brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal"]
NAMES = {"lr": "left–right"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def box(ax, x, y, w, h, text, colour, fill=None, size=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", lw=1.6,
                                edgecolor=colour, facecolor=fill or "white"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.15)


def arrow(ax, a, b, colour=MUTE, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=colour, lw=1.5, ls=ls))


def schematic(ax):
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.set_title("A   Two ways to an 8-unit code of the shared anatomy", loc="left", fontsize=15, fontweight="bold")
    # trained directly
    y = 3.0
    ax.text(0.0, y + 1.05, "Trained directly at 8 units", fontsize=12.5, fontweight="bold", color=INK)
    box(ax, 0.0, y, 1.6, 0.8, "T1 / FLAIR\nscans", REF)
    box(ax, 2.1, y, 1.5, 0.8, "encoder", REF)
    box(ax, 4.1, y, 1.6, 0.8, "8-unit\ncode", GREY, "#f3f4f7")
    box(ax, 6.2, y, 2.6, 0.8, "InfoNCE\nalignment + entropy", REF)
    for a_, b_ in ((1.6, 2.1), (3.6, 4.1), (5.7, 6.2)):
        arrow(ax, (a_, y + 0.4), (b_, y + 0.4))
    ax.text(9.1, y + 0.4, "no spare room: entropy bends strong factors,\nand a weak one is lost in about half of runs",
            fontsize=11.5, color=MUTE, va="center")
    # identify wide, chart narrow
    y = 0.6
    ax.text(0.0, y + 1.55, "Identify wide, then chart narrow", fontsize=12.5, fontweight="bold", color=CHART)
    box(ax, 0.0, y, 1.6, 0.8, "T1 / FLAIR\nscans", REF)
    box(ax, 2.1, y, 1.5, 0.8, "encoder", REF)
    box(ax, 4.1, y, 1.6, 0.8, "10-unit\ncode", WIDE, "#e6f8f3")
    box(ax, 4.1, y + 1.0, 1.6, 0.42, "InfoNCE", REF, size=10.5)
    box(ax, 6.2, y, 1.5, 0.8, "chart\n(MLP)", CHART)
    box(ax, 8.2, y, 1.4, 0.8, "8-unit\ncode", CHART, "#e8f1fc")
    box(ax, 10.1, y, 1.5, 0.8, "decoder\n(MLP)", CHART)
    for a_, b_ in ((1.6, 2.1), (3.6, 4.1), (5.7, 6.2), (7.7, 8.2), (9.6, 10.1)):
        arrow(ax, (a_, y + 0.4), (b_, y + 0.4))
    arrow(ax, (4.9, y + 0.8), (4.9, y + 1.0), REF, style="-")
    arrow(ax, (11.6, y + 0.4), (12.1, y + 0.4), CHART)
    ax.text(12.15, y + 0.4, "rebuilds\nthe 10", fontsize=10.5, color=CHART, va="center")
    arrow(ax, (12.4, y + 0.85), (5.0, y + 0.85), CHART, ls="--", style="-|>")
    ax.text(8.7, y + 0.95, "target: the fixed, whitened shared part of the 10-unit code", fontsize=10.5, color=CHART,
            ha="center", va="bottom")
    ax.text(0.0, y - 0.45, "Stage 1 identifies the content (spare units become curves); stage 2 keeps it at 8 units "
            "with no contrastive loss: the decoder rebuilds the curves, and every shared direction counts alike.",
            fontsize=11, color=MUTE)


def load():
    stage82 = json.loads((HERE / "chart_check.json").read_text())
    b = {}
    for part in HERE.glob("b_*.json"):
        b.update(json.loads(part.read_text()))
    return stage82, b


def cells(b):
    out = {}
    for label, row in b.items():
        out.setdefault(label.rsplit("_s", 1)[0], []).append(row)
    return out


def rates(ax, stage82, b, rows_csv):
    c = cells(b)
    entries = [("8 units, trained directly", 8, 15, REF, "stages 64 and 76")]
    s82 = list(stage82.values())
    holders82 = [r for r in s82 if r["parent"]["holds"]]
    entries.append(("10-unit parents", len(holders82), len(s82), WIDE, "stage 82"))
    entries.append(("10 → 8, linear chart", sum(r["linear"]["holds"] for r in holders82), len(holders82), LINEAR,
                    "of the parents that hold"))
    entries.append(("10 → 8, nonlinear chart", sum(r["chart"]["holds"] for r in holders82), len(holders82), CHART,
                    "of the parents that hold"))
    for cell, title in (("c10to7", "10 → 7"), ("c9e500", "9 → 8 (every 500)"), ("c9e100", "9 → 8 (every 100)"),
                        ("c12", "12 → 8"), ("c16", "16 → 8")):
        rows = c.get(cell)
        if not rows:
            entries.append((f"{title}, nonlinear chart", None, None, CHART, "running"))
            continue
        holders = [r for r in rows if r["parent"]["holds"]]
        entries.append((f"{title}, nonlinear chart", sum(r["chart"]["holds"] for r in holders), len(holders), CHART,
                        f"{len(holders)} of {len(rows)} parents hold"))
    y = np.arange(len(entries))[::-1]
    for yy, (label, k, n, colour, note) in zip(y, entries):
        if k is None:
            ax.text(0.01, yy, "pending", va="center", fontsize=11, color=MUTE, style="italic")
        else:
            share = k / n if n else 0.0
            ax.barh(yy, share, height=0.6, color=colour, alpha=0.9 if colour != REF else 0.75)
            ax.text(share + 0.015, yy, f"{k}/{n}", va="center", fontsize=12, color=INK)
        rows_csv.append([label, "" if k is None else f"{k}/{n}", "" if k is None else f"{k / n:.3f}", note])
    ax.set_yticks(y, [e[0] for e in entries])
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("share that hold all seven own parts")
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("B   Hold rates (charts: share of holding parents kept)", loc="left", fontsize=15, fontweight="bold")


def own_parts(ax, stage82, rows_csv):
    holders = [r for r in stage82.values() if r["parent"]["holds"]]

    def mean(key, f):
        return float(np.mean([np.mean([r[key]["own_r2"][v][f] for v in ("T1", "FLAIR")]) for r in holders]))

    x = np.arange(len(FACTORS))
    for key, colour, marker, label, dx in (("parent", WIDE, "o", "10-unit parent", -0.22),
                                           ("chart", CHART, "o", "8-unit nonlinear chart", 0.0),
                                           ("linear", LINEAR, "X", "8-unit linear chart", 0.22)):
        vals = [mean(key, f) for f in FACTORS]
        ax.scatter(x + dx, vals, s=75, color=colour, marker=marker, edgecolor="white", lw=1.2, zorder=3, label=label)
        for f, v in zip(FACTORS, vals):
            rows_csv.append([f"own part: {label}", f, f"{v:.3f}", "mean of the 4 holding parents, T1 and FLAIR"])
    ax.set_xticks(x, [NAMES.get(f, f) for f in FACTORS], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.set_ylabel("own-part R² (ridge, test subjects)")
    ax.grid(axis="y", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    ax.set_title("C   What the 8 units keep (the 4 parents that hold)", loc="left", fontsize=15, fontweight="bold")


def main():
    stage82, b = load()
    fig = plt.figure(figsize=(16, 10.5))
    schematic(fig.add_axes([0.0, 0.57, 1.0, 0.40]))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.2, right=0.98, top=0.47, bottom=0.08, wspace=0.42)
    rows_csv = []
    rates(fig.add_subplot(grid[0, 0]), stage82, b, rows_csv)
    own_parts(fig.add_subplot(grid[0, 1]), stage82, rows_csv)
    fig.text(0.01, 0.005, "Fixed-position synthetic bank, 8 = the cosine's matched width for 7 factors. Holds: the "
             "own-parts spectrum's last value and the ventricle–burden direction each ≥ .40 in both views (1,000 test "
             "subjects). Charts fitted label-free on training subjects only.", fontsize=10.5, color=MUTE)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig_chart_narrow.png")
    with open(OUT / "fig_chart_narrow.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "item", "value", "note"])
        w.writerows(rows_csv)
    print("wrote", OUT / "fig_chart_narrow.png")


if __name__ == "__main__":
    main()
