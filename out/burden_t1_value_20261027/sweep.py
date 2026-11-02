"""How findable is ONE burden lesion in T1, as a function of --synthetic-lesion-t1-value?

Same generator, detector and population as render.py: identifiable ventricle, sulcal atrophy, clean_content,
wm_interior, burden target, 1 lesion of max radius 0.1, styled lesion, fixed_reference, res 64, seed 42, each subject
at its own burden and style. Only the T1 lesion value changes between settings. FLAIR doesn't depend on it and is
scored once. The detector is the scale-normalised LoG (sigma 1, 1.5, 2, 3 voxels), signed to the lesion's polarity:
dark blobs for values below white matter (0.8), bright blobs above it.
  strongest blob   share of subjects whose lesion holds the strongest response in the brain eroded by 6 voxels
  R2 no tissue     5-fold CV ridge R2 of the burden latent from that region's strongest responses (top 1/10/50 per
                   scale), as in burden_blob_t1.py
  R2 WM mask       the same inside the true white-matter mask (a tissue prior no encoder gets)
  contrast         median over subjects of the lesion core minus a 2-voxel white-matter shell (normalised image)

    PYTHONPATH=. python out/burden_t1_value_20261027/sweep.py out/burden_t1_value_20261027 [N] [--replot]
"""

import csv
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_laplace
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.datasets import SyntheticBrainDataset

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 300
REPLOT = "--replot" in sys.argv
RES, ERODE, WM_VALUE = 64, 6, 0.8
SIGMAS = (1.0, 1.5, 2.0, 3.0)
VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6)


def build(t1, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SyntheticBrainDataset(
            mode="val",
            spatial_size=(RES,) * 3,
            synthetic_num_samples=n,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_identifiable_ventricle=True,
            synthetic_sulcal_mode="atrophy",
            synthetic_lesion_placement="wm_interior",
            synthetic_lesion_target="burden",
            synthetic_lesion_count=1,
            synthetic_lesion_radius=0.1,
            synthetic_lesion_intensity="styled",
            synthetic_lesion_t1_value=t1,
            synthetic_seed=42,
        )


def top_responses(stack, region):
    out = []
    for response in stack:
        v = np.sort(response[region])[::-1]
        out += [float(v[0]), float(v[:10].mean()), float(v[:50].mean())]
    return out


def run(t1):
    """Per subject: burden latent and detector results for T1 at this lesion value, or for FLAIR when t1 is None."""
    torch.set_num_threads(1)
    ds = build(0.4 if t1 is None else t1, N)
    inner = ds._inner
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N):
            x1, x2, m1, _, lat = ds._render(i)
            d = inner._draw_pseudo_mri(inner.sample_seed_for(i))
            with torch.no_grad():
                tissue, load = inner.renderer.render_structure(
                    d["z_content"], d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
                )
            tissue, load = tissue.numpy(), load.numpy()
            core, wm = load > 0.5, tissue == 2
            brain = binary_erosion(m1.squeeze(0).numpy() > 0, iterations=ERODE)
            lesion = binary_dilation(load > 0, iterations=1)
            shell = binary_dilation(core, iterations=2) & ~binary_dilation(core, iterations=1) & wm
            x, sign = (x2, -1.0) if t1 is None else (x1, 1.0 if t1 <= WM_VALUE else -1.0)
            x = x.squeeze(0).numpy().astype(np.float64)
            stack = np.stack([sign * s**2 * gaussian_laplace(x, s) for s in SIGMAS])
            response = stack.max(0)
            inside, outside = response[brain & lesion], response[brain & ~lesion]
            rows.append(
                {
                    "z": float(lat["z_content"][2]),
                    "win": bool(inside.size and inside.max() > outside.max()),
                    "brain": top_responses(stack, brain),
                    "wm": top_responses(stack, wm),
                    "contrast": float(x[core].mean() - x[shell].mean())
                    if core.sum() >= 8 and shell.sum() >= 8
                    else None,
                }
            )
    return rows


def ridge_r2(X, y):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 17)))
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def summarise(rows):
    z = np.array([r["z"] for r in rows])
    contrast = [r["contrast"] for r in rows if r["contrast"] is not None]
    return {
        "strongest": float(np.mean([r["win"] for r in rows])),
        "r2_brain": ridge_r2(np.array([r["brain"] for r in rows]), z),
        "r2_wm": ridge_r2(np.array([r["wm"] for r in rows]), z),
        "contrast": float(np.median(contrast)),
    }, z


def compute():
    settings = [None, *VALUES]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = dict(zip(settings, pool.map(run, settings)))
    table, z_ref = {}, None
    for setting, rows in results.items():
        table["FLAIR" if setting is None else f"{setting:.1f}"], z = summarise(rows)
        z_ref = z if z_ref is None else z_ref
        assert np.allclose(z, z_ref), "every setting must score the same subjects"
    return table


def plot(table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # dataviz reference palette, as in render.py: blue = T1 0.4, orange = T1 0.0, green = FLAIR
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddcd7"
    MARK = {0.4: ("your run, 0.4", "#2a78d6"), 0.0: ("k1 recipe, 0.0", "#eb6834")}
    FLAIR = "#1baf7a"
    TISSUE = (("CSF", 0.1), ("fissure", 0.3), ("GM", 0.5), ("WM", 0.8))
    x = np.array(VALUES)

    def series(key, scale=1):
        return np.array([table[f"{v:.1f}"][key] for v in VALUES]) * scale

    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "font.size": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), facecolor=SURFACE, gridspec_kw={"wspace": 0.2})
    panels = (
        (axes[0], 100, ("strongest",), "Lesion holds the strongest blob response in the brain\n(% of subjects)"),
        (axes[1], 1, ("r2_brain", "r2_wm"), "Burden R² from the detector's strongest responses\n(5-fold ridge)"),
    )
    for ax, scale, keys, title in panels:
        ax.set_facecolor(SURFACE)
        ax.axvspan(-0.06, 0.1, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.text(
            0.02,
            0.04,
            "darker than CSF",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            fontsize=8,
            color=INK_2,
            ha="center",
            va="bottom",
        )
        for name, value in TISSUE:
            ax.axvline(value, color=INK_2, lw=0.8, ls=":", zorder=1)
            ax.text(
                value,
                1.015,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK_2,
            )
        for key, ls in zip(keys, ("-", "--")):
            ax.plot(x, series(key, scale), color=INK_2, lw=2, ls=ls, marker="o", ms=4.5, zorder=3)
        ax.axhline(table["FLAIR"][keys[0]] * scale, color=FLAIR, lw=1.6, ls="--", zorder=2)
        for value, (_, color) in MARK.items():
            ax.plot(
                value,
                series(keys[0], scale)[VALUES.index(value)],
                "o",
                ms=10,
                color=color,
                mec=SURFACE,
                mew=1.5,
                zorder=5,
            )
        ax.set_xlim(-0.06, 1.66)
        ax.set_xticks(np.arange(0, 1.7, 0.2), [f"{t:.1f}" for t in np.arange(0, 1.7, 0.2)])
        ax.set_xlabel("T1 lesion value  (FLAIR lesion = 1.0 on FLAIR's own table)")
        ax.set_title(title, fontsize=10.5, pad=20)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
    axes[0].set_ylim(-4, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axes[0].text(
        0.62,
        table["FLAIR"]["strongest"] * 100 + 2,
        f"FLAIR {table['FLAIR']['strongest']:.0%}",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    axes[1].set_ylim(-0.1, 1.0)
    axes[1].text(
        0.62,
        table["FLAIR"]["r2_brain"] + 0.02,
        f"FLAIR {table['FLAIR']['r2_brain']:.2f}",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    axes[1].text(0.53, 0.58, "told where white\nmatter is", fontsize=8.5, color=INK, va="bottom")
    axes[1].text(
        0.47,
        0.13,
        "no tissue information",
        fontsize=8.5,
        color=INK,
        va="bottom",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        zorder=4,
    )
    handles = [
        Line2D([], [], color=INK_2, lw=2, marker="o", ms=4.5, label="T1, no tissue information"),
        Line2D([], [], color=INK_2, lw=2, ls="--", marker="o", ms=4.5, label="T1, told where white matter is"),
        Line2D([], [], color=FLAIR, lw=1.6, ls="--", label="FLAIR, no tissue information"),
        *[Line2D([], [], ls="", marker="o", ms=9, color=c, mec=SURFACE, label=n) for n, c in MARK.values()],
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=3.4,
    )
    fig.suptitle(
        f"How findable one burden lesion is in T1, by T1 lesion value ({N} subjects per value, own burden and style; "
        "identifiable ventricle + sulcal atrophy)",
        fontsize=10.5,
        y=1.06,
    )
    fig.text(
        0.5,
        -0.075,
        "Dotted lines: T1 tissue values. Same detector as figure 1 (no tissue information unless stated), signed to the "
        "lesion's polarity: dark below white matter, bright above.\nFLAIR doesn't depend on the T1 value. "
        "Lesion-minus-surrounding-white-matter contrast is in the CSV.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    fig.savefig(f"{OUT}/2_t1_value_sweep.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    path = f"{OUT}/2_t1_value_sweep.json"
    if REPLOT:
        table = json.load(open(path))
    else:
        table = compute()
        json.dump(table, open(path, "w"), indent=1)
        with open(f"{OUT}/2_t1_value_sweep.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["setting", "strongest_blob", "r2_no_tissue", "r2_wm_mask", "contrast_vs_wm"])
            for key, row in table.items():
                writer.writerow([key, row["strongest"], row["r2_brain"], row["r2_wm"], row["contrast"]])
    print(f"{'setting':>8}{'strongest blob':>16}{'R2 no tissue':>14}{'R2 WM mask':>12}{'contrast':>10}")
    for key, row in table.items():
        print(
            f"{key:>8}{row['strongest']:>16.0%}{row['r2_brain']:>14.2f}{row['r2_wm']:>12.2f}{row['contrast']:>+10.3f}"
        )
    plot(table)
