"""T1 lesion value 0.4 (the parser default) vs 0.0 (the k1 recipe) for ONE burden lesion, under the default generator.

Generator: identifiable ventricle, sulcal atrophy, clean_content, wm_interior, burden target, 1 lesion of max radius
0.1, styled lesion, res 64. Images are raw intensities at gain 1 / bias 0, the lesion at the median burden (latent 0,
half the full-size volume). The two T1 rows differ only in --synthetic-lesion-t1-value; anatomy, lesion position,
burden and FLAIR are identical (asserted below).

Detector column: a blob detector with no tissue information. Scale-normalised LoG at sigma 1, 1.5, 2, 3 voxels,
signed so the lesion's polarity is positive (dark blobs in T1, bright in FLAIR), max over scales, inside the brain
eroded by 6 voxels; shown as an axial maximum projection. The population panel runs the same detector on N_POP
dataset subjects (each with its own burden and style): how often the lesion holds the strongest response.

Without a subject argument the example is the first subject whose median-burden render shows the most common
outcome: the detector misses the lesion at T1 0.4 and finds it at T1 0.0 and in FLAIR.

    PYTHONPATH=. python out/burden_t1_value_20261027/render.py out/burden_t1_value_20261027 [subject]
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.ndimage import binary_dilation, binary_erosion, center_of_mass, gaussian_laplace

from data.datasets import SyntheticBrainDataset

OUT = sys.argv[1]
SUBJECT = int(sys.argv[2]) if len(sys.argv) > 2 else None
RES = 64
N_POP = 300
STYLE = torch.tensor([0.0, 0.0, 0.3])
SIGMAS = (1.0, 1.5, 2.0, 3.0)
ERODE = 6
CROP = 9
PROFILE = 12
T1_VALUES = (0.4, 0.0)
# dataviz reference palette: surface, text inks, categorical slots 1-3, blue sequential ramp
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddcd7"
COLOR = {"T1 0.4": "#2a78d6", "T1 0.0": "#eb6834", "FLAIR": "#1baf7a"}
SEQ = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
T1_TISSUE = (("white matter", 0.8), ("grey matter", 0.5), ("fissure", 0.3), ("CSF", 0.1))
R2 = {"T1 0.4": 0.09, "T1 0.0": 0.74, "FLAIR": 0.73}  # burden_blob_t1.py, brain eroded by 6, 300 subjects


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


def structure(inner, d, zc):
    with torch.no_grad():
        return inner.renderer.render_structure(
            zc, d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
        )


def detector(x, sign):
    """Voxelwise max over scales of the scale-normalised LoG, signed so the lesion's polarity is positive."""
    x = np.asarray(x, dtype=np.float64)
    return np.max([sign * s**2 * gaussian_laplace(x, s) for s in SIGMAS], axis=0)


def lesion_vs_rest(response, region, lesion):
    """(lesion peak / strongest response elsewhere in region, voxels elsewhere above the lesion peak)."""
    inside, outside = response[region & lesion], response[region & ~lesion]
    if inside.size == 0:
        return 0.0, int(outside.size)
    peak = inside.max()
    return float(peak / outside.max()), int((outside > peak).sum())


def population(ds):
    """Per view, per subject: does the lesion hold the strongest detector response in the eroded brain?"""
    inner = ds._inner
    wins = {"T1": [], "FLAIR": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_POP):
            x1, x2, m1, _, _ = ds._render(i)
            d = inner._draw_pseudo_mri(inner.sample_seed_for(i))
            load = structure(inner, d, d["z_content"])[1].numpy()
            region = binary_erosion(m1.squeeze(0).numpy() > 0, iterations=ERODE)
            lesion = binary_dilation(load > 0, iterations=1)
            for view, x, sign in (("T1", x1, +1.0), ("FLAIR", x2, -1.0)):
                wins[view].append(lesion_vs_rest(detector(x.squeeze(0).numpy(), sign), region, lesion)[0] > 1)
    return {view: np.array(w) for view, w in wins.items()}


def example(ds, subject):
    """A subject at median burden: (T1, FLAIR, brain mask, lesion load, full-size lesion)."""
    inner = ds._inner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seed = inner.sample_seed_for(subject)
    d = inner._draw_pseudo_mri(seed)
    zc = d["z_content"].clone()
    zc[2] = 0.0
    x1, x2, mask = inner.render_pseudo_mri(
        zc, d["z_deformation"], d["z_fissure"], STYLE, STYLE, seed, z_lesion=d["z_lesion"]
    )
    load = structure(inner, d, zc)[1].numpy()
    zc[2] = 10.0
    full = structure(inner, d, zc)[1].numpy()
    return x1.squeeze(0).numpy(), x2.squeeze(0).numpy(), np.asarray(mask).squeeze() > 0, load, full


def ratios(renders):
    """{row: lesion peak / strongest elsewhere} for T1 0.4, T1 0.0 and FLAIR of one subject's renders."""
    (t1_a, flair, brain, load, _), (t1_b, *_rest) = renders[0.4], renders[0.0]
    region, lesion = binary_erosion(brain, iterations=ERODE), binary_dilation(load > 0, iterations=1)
    return {
        "T1 0.4": lesion_vs_rest(detector(t1_a, +1.0), region, lesion)[0],
        "T1 0.0": lesion_vs_rest(detector(t1_b, +1.0), region, lesion)[0],
        "FLAIR": lesion_vs_rest(detector(flair, -1.0), region, lesion)[0],
    }


def crop(img2d, cx, cy):
    p = np.pad(img2d, CROP)
    return np.rot90(p[cx : cx + 2 * CROP + 1, cy : cy + 2 * CROP + 1])


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)


if __name__ == "__main__":
    datasets = {t1: build(t1, N_POP) for t1 in T1_VALUES}
    pop = {t1: population(ds) for t1, ds in datasets.items()}
    typical = ~pop[0.4]["T1"] & pop[0.0]["T1"] & pop[0.4]["FLAIR"]
    print(
        f"lesion is the strongest blob: T1 0.4 {pop[0.4]['T1'].mean():.1%}, T1 0.0 {pop[0.0]['T1'].mean():.1%}, "
        f"FLAIR {pop[0.4]['FLAIR'].mean():.1%} / {pop[0.0]['FLAIR'].mean():.1%}; missed at 0.4 but found at 0.0 "
        f"and in FLAIR: {typical.mean():.1%}",
        flush=True,
    )
    candidates = [SUBJECT] if SUBJECT is not None else range(N_POP)
    for subject in candidates:
        renders = {t1: example(ds, subject) for t1, ds in datasets.items()}
        r = ratios(renders)
        if SUBJECT is not None or (r["T1 0.4"] < 1 < r["T1 0.0"] and r["FLAIR"] > 1):
            break
    print(f"subject {subject}: lesion peak / strongest elsewhere " + ", ".join(f"{k} {v:.2f}" for k, v in r.items()))

    (t1_a, flair, brain, load, full), (t1_b, flair_b, _, _, _) = renders[0.4], renders[0.0]
    assert np.allclose(flair, flair_b), "FLAIR must not depend on the T1 lesion value"
    away = ~binary_dilation(load > 0, iterations=2)
    print(
        f"T1 max |0.4 - 0.0| away from the lesion: {np.abs(t1_a - t1_b)[away].max():.2e}; lesion voxels {load.sum():.1f}"
    )

    region = binary_erosion(brain, iterations=ERODE)
    lesion = binary_dilation(load > 0, iterations=1)
    cx, cy, cz = np.round(center_of_mass(full)).astype(int)
    spacing = 2.0 / (RES - 1)
    r_full = 0.1 / spacing
    r_now = r_full * 0.5 ** (1 / 3)
    rows = [
        ("T1 0.4", "T1, lesion value 0.4\n(your run)", t1_a, detector(t1_a, +1.0)),
        ("T1 0.0", "T1, lesion value 0.0\n(k1 recipe)", t1_b, detector(t1_b, +1.0)),
        ("FLAIR", "FLAIR\n(identical in both)", flair, detector(flair, -1.0)),
    ]
    mips = {key: np.where(region, np.clip(resp, 0, None), 0).max(axis=2) for key, _, _, resp in rows}
    vmax = {"T1 0.4": max(mips["T1 0.4"].max(), mips["T1 0.0"].max()), "FLAIR": mips["FLAIR"].max()}
    vmax["T1 0.0"] = vmax["T1 0.4"]

    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "font.size": 9.5})
    fig = plt.figure(figsize=(10.4, 13.2), facecolor=SURFACE)
    gs = GridSpec(4, 3, height_ratios=[1, 1, 1, 1.05], hspace=0.34, wspace=0.08, figure=fig, top=0.925, bottom=0.05)
    col_titles = (
        "Axial slice through the lesion",
        "Zoom, 19 × 19 voxels",
        "Blob detector, no tissue information\n(max projection; both T1 rows share a scale)",
    )
    for i, (key, label, img, resp) in enumerate(rows):
        ratio, beaten = lesion_vs_rest(resp, region, lesion)
        axes = [fig.add_subplot(gs[i, j]) for j in range(3)]
        axes[0].imshow(np.rot90(img[:, :, cz]), cmap="gray", vmin=0, vmax=1.1)
        axes[0].add_patch(Circle((cx, RES - 1 - cy), r_full + 2.5, fill=False, ls="--", lw=0.9, ec=INK))
        axes[0].set_ylabel(label, fontsize=10.5, labelpad=8)
        axes[1].imshow(crop(img[:, :, cz], cx, cy), cmap="gray", vmin=0, vmax=1.1)
        axes[1].add_patch(Circle((CROP, CROP), r_full + 0.6, fill=False, ls="--", lw=0.9, ec=INK))
        axes[2].imshow(np.rot90(mips[key]), cmap=SEQ, vmin=0, vmax=vmax[key])
        axes[2].add_patch(Circle((cx, RES - 1 - cy), r_full + 2.5, fill=False, ls="--", lw=0.9, ec=INK))
        wx, wy, _ = np.unravel_index(np.argmax(np.where(region & ~lesion, resp, -np.inf)), resp.shape)
        axes[2].plot(wx, RES - 1 - wy, marker="x", ms=8, mew=2, color=INK)
        verdict = (
            "the lesion is the strongest blob" if ratio > 1 else f"{beaten} voxels elsewhere respond more strongly"
        )
        axes[2].set_xlabel(
            f"lesion peak = {ratio:.2f} × the strongest response elsewhere (×)\n{verdict}", fontsize=8.5, color=INK_2
        )
        for j, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, pad=8)

    bottom = gs[3, :].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.42)
    ax = fig.add_subplot(bottom[0])
    style_axes(ax)
    offsets = np.arange(-PROFILE, PROFILE + 1)
    ax.axvspan(-r_now, r_now, color=GRID, alpha=0.5, lw=0)
    ax.text(0, 1.0, "lesion", ha="center", va="center", fontsize=8.5, color=INK_2)
    for name, value in T1_TISSUE:
        ax.axhline(value, color=INK_2, lw=0.8, ls=":")
        ax.text(PROFILE + 0.9, value, name, va="center", fontsize=8.5, color=INK_2)
    for key, img in (("T1 0.4", t1_a), ("T1 0.0", t1_b)):
        profile = img[cx + offsets, cy, cz]
        ax.plot(offsets, profile, color=COLOR[key], lw=1.8, marker="o", ms=3.5, label=f"T1, lesion {key[3:]}")
        ax.text(r_now + 0.9, profile[PROFILE], f"T1, lesion {key[3:]}", color=INK, fontsize=9, va="center")
    ax.set_xlim(-PROFILE - 0.5, PROFILE + 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("voxels from the lesion centre (left–right)")
    ax.set_ylabel("T1 intensity (raw)")
    ax.set_title("T1 along a line through the lesion (dotted = T1 tissue values)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = fig.add_subplot(bottom[1])
    style_axes(ax)
    keys = ("T1 0.4", "T1 0.0", "FLAIR")
    shares = [100 * pop[0.4]["T1"].mean(), 100 * pop[0.0]["T1"].mean(), 100 * pop[0.4]["FLAIR"].mean()]
    bars = ax.bar(range(3), shares, width=0.62, color=[COLOR[k] for k in keys], edgecolor=SURFACE, linewidth=1.5)
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, share + 2, f"{share:.0f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(
        range(3), [f"T1, 0.4\nR² {R2['T1 0.4']:.2f}", f"T1, 0.0\nR² {R2['T1 0.0']:.2f}", f"FLAIR\nR² {R2['FLAIR']:.2f}"]
    )
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Lesion is the strongest blob in the brain\n({N_POP} subjects, own burden and style)", fontsize=10)

    fig.suptitle(
        f"One burden lesion, T1 lesion value 0.4 vs 0.0 (subject {subject} at median burden, {load.sum():.0f} voxels; "
        "identifiable ventricle + sulcal atrophy)\n"
        "Only the T1 lesion intensity differs between the top two rows. Dashed circle = lesion; × = the strongest "
        "detector response outside it.",
        fontsize=10,
        y=0.985,
    )
    fig.text(
        0.5,
        0.005,
        f"R² under the bars: burden predicted from the same detector's strongest responses ({N_POP} subjects, no tissue "
        "information).\nThe example is the first subject showing the most common outcome "
        f"({typical.mean():.0%} of subjects): missed at T1 0.4, found at T1 0.0 and in FLAIR.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    fig.savefig(f"{OUT}/1_t1_lesion_value.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)
