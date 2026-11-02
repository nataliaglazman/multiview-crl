"""The range of lesion burden with ONE lesion: one subject swept through the prior, plus the population histogram.

Recipe = experiments/encoder/conv_mlp_burden_oracle_k1.yaml with synthetic_lesion_count=1 (radius 0.1, burden
target, styled lesion, T1 lesion value 0.0, wm_interior, clean_content, res 64). Raw intensities, gain 1 / bias 0.
Only z_content[2] changes between columns; anatomy, lesion position and style are fixed.

    PYTHONPATH=. python out/burden_1lesion_range_20261027/render.py out/burden_1lesion_range_20261027 [subject]
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.ndimage import center_of_mass
from scipy.stats import norm

from data.datasets import SyntheticBrainDataset

OUT = sys.argv[1]
SUBJECT = int(sys.argv[2]) if len(sys.argv) > 2 else 15
RES = 64
N_POP = 500
STYLE = torch.tensor([0.0, 0.0, 0.3])
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
CROP = 9


def build(n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SyntheticBrainDataset(
            mode="train",
            spatial_size=(RES,) * 3,
            synthetic_num_samples=n,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_lesion_placement="wm_interior",
            synthetic_lesion_target="burden",
            synthetic_lesion_count=1,
            synthetic_lesion_radius=0.1,
            synthetic_lesion_intensity="styled",
            synthetic_lesion_t1_value=0.0,
            synthetic_seed=42,
        )._inner


def structure(inner, d, zc):
    with torch.no_grad():
        return inner.renderer.render_structure(
            zc, d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
        )


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def crop(img2d, cx, cy):
    p = np.pad(img2d, CROP)
    return np.rot90(p[cx : cx + 2 * CROP + 1, cy : cy + 2 * CROP + 1])


if __name__ == "__main__":
    inner = build(N_POP)
    spacing = 2.0 / (RES - 1)
    r_full = inner.renderer.lesion_radius / spacing

    # population: every subject's own burden draw
    volumes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_POP):
            d = inner._draw_pseudo_mri(inner.sample_seed_for(i))
            volumes.append(float(structure(inner, d, d["z_content"])[1].sum()))
            if i == SUBJECT:
                brain = int((structure(inner, d, d["z_content"])[0] > 0).sum())
    volumes = np.array(volumes)

    # one subject swept through the prior
    seed = inner.sample_seed_for(SUBJECT)
    d = inner._draw_pseudo_mri(seed)
    levels = []
    for p in PERCENTILES:
        z = float(norm.ppf(p / 100))
        zc = d["z_content"].clone()
        zc[2] = z
        _, load = structure(inner, d, zc)
        x1, x2, _ = inner.render_pseudo_mri(
            zc, d["z_deformation"], d["z_fissure"], STYLE, STYLE, seed, z_lesion=d["z_lesion"]
        )
        frac = (np.tanh(z) + 1) / 2
        levels.append((p, z, frac, float(load.sum()), x1.squeeze(0).numpy(), x2.squeeze(0).numpy()))
    zc = d["z_content"].clone()
    zc[2] = 3.0
    full = structure(inner, d, zc)[1].numpy()
    cx, cy, cz = np.round(center_of_mass(full)).astype(int)

    k = len(levels)
    fig = plt.figure(figsize=(1.9 * k, 10.6))
    gs = GridSpec(5, k, height_ratios=[1, 1, 1, 1, 1.15], hspace=0.35, wspace=0.06, figure=fig)
    rows = (("T1", 4, False), ("FLAIR", 5, False), ("T1 zoom", 4, True), ("FLAIR zoom", 5, True))
    for j, (p, z, frac, vol, x1, x2) in enumerate(levels):
        radius_vox = r_full * frac ** (1 / 3)
        for i, (name, slot, zoom) in enumerate(rows):
            vol3d = x1 if slot == 4 else x2
            img = crop(vol3d[:, :, cz], cx, cy) if zoom else np.rot90(vol3d[:, :, cz])
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1.1)
            if zoom:
                ax.add_patch(Circle((CROP, CROP), r_full + 0.5, fill=False, ls="--", lw=0.7, ec="#e8590c"))
            off(ax)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
            if i == 0:
                ax.set_title(
                    f"{p}{'st' if p == 1 else 'th'} percentile\nz = {z:+.2f}\n{vol:.0f} lesion voxels\nradius {radius_vox:.1f} vox",
                    fontsize=8.5,
                )
    ax = fig.add_subplot(gs[4, :])
    ax.hist(volumes, bins=np.linspace(0, 140, 36), color="#adb5bd", edgecolor="white")
    for p, z, frac, vol, _, _ in levels:
        ax.axvline(vol, color="#e8590c", lw=1)
        ax.text(
            vol,
            ax.get_ylim()[1] * 0.97,
            f"{p}",
            color="#e8590c",
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )
    q = np.percentile(volumes, [5, 50, 95])
    ax.set_xlabel(
        f"lesion voxels per subject ({N_POP} subjects of this recipe): median {q[1]:.0f}, 5–95% {q[0]:.0f}–{q[2]:.0f}; "
        f"one full sphere = {volumes.max():.0f}. Orange lines = the columns above.",
        fontsize=9,
    )
    ax.set_ylabel("subjects")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Burden with ONE lesion: subject {SUBJECT} ({brain} brain voxels), only the burden latent changes.\n"
        f"--synthetic-lesion-radius 0.1 is the MAXIMUM radius ({r_full:.1f} voxels, dashed circle); each subject's actual "
        "radius is 0.1·((tanh z + 1)/2)^(1/3), given above each column.\n"
        "Axial slice through the lesion centre; zooms are 19×19 voxels. T1 lesion value 0.0, FLAIR lesion 1.0.",
        fontsize=9.5,
        y=0.995,
    )
    fig.savefig(f"{OUT}/1_burden_range_1lesion.png", dpi=130, bbox_inches="tight")
    print("levels:", [(p, round(vol, 1)) for p, _, _, vol, _, _ in levels], "population 5/50/95%:", np.round(q, 1))
    print(
        f"population: <10 vox {np.mean(volumes < 10):.1%}, <20 vox {np.mean(volumes < 20):.1%}, >108 vox (top of the 75th pct render) {np.mean(volumes > 108):.1%}"
    )
