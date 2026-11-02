"""Low vs high lesion burden for one subject: anatomy, lesion positions and style fixed, only z_content[2] changes.

Recipe = experiments/encoder/conv_mlp_burden_oracle_k1.yaml (burden target, styled lesion, T1 lesion value 0.0,
4 spheres, wm_interior, clean_content, res 64). Raw intensities (pre-normalisation), gain 1 / bias 0.

    PYTHONPATH=. python out/burden_preview_20261026/render.py out/burden_preview_20261026 [subject]
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from scipy.ndimage import center_of_mass, label

from data.datasets import SyntheticBrainDataset

OUT = sys.argv[1]
SUBJECT = (
    int(sys.argv[2]) if len(sys.argv) > 2 else 15
)  # a median-sized brain (35k voxels; subject 3 is the largest of 16)
RES = 64
STYLE = torch.tensor([0.0, 0.0, 0.3])
# 5th / 50th / 95th percentile of the N(0,1) prior on z_content[2]
LEVELS = [(-1.645, "low burden\n(5th percentile)"), (0.0, "median burden"), (1.645, "high burden\n(95th percentile)")]
CROP = 9


def build():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SyntheticBrainDataset(
            mode="train",
            spatial_size=(RES,) * 3,
            synthetic_num_samples=16,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_lesion_placement="wm_interior",
            synthetic_lesion_target="burden",
            synthetic_lesion_intensity="styled",
            synthetic_lesion_t1_value=0.0,
            synthetic_seed=42,
        )._inner


def render(inner, d, seed, z):
    zc = d["z_content"].clone()
    zc[2] = z
    with torch.no_grad():
        tissue, load = inner.renderer.render_structure(
            zc, d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
        )
    x1, x2, _ = inner.render_pseudo_mri(
        zc, d["z_deformation"], d["z_fissure"], STYLE, STYLE, seed, z_lesion=d["z_lesion"]
    )
    return x1.squeeze(0).numpy(), x2.squeeze(0).numpy(), tissue.numpy(), load.numpy()


def axial(v, k):
    return np.rot90(v[:, :, k])


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def crop(v, c):
    p = np.pad(v[:, :, c[2]], CROP)
    return np.rot90(p[c[0] : c[0] + 2 * CROP + 1, c[1] : c[1] + 2 * CROP + 1])


if __name__ == "__main__":
    inner = build()
    seed = inner.sample_seed_for(SUBJECT)
    d = inner._draw_pseudo_mri(seed)
    renders = [render(inner, d, seed, z) for z, _ in LEVELS]
    tissue = renders[0][2]
    brain = (tissue > 0).sum()
    full = renders[-1][3]
    comps, n = label(full > 0.5)
    centres = sorted(
        (np.round(center_of_mass(full, comps, i)).astype(int) for i in range(1, n + 1)), key=lambda c: -c[2]
    )
    r_full = inner.renderer.lesion_radius / (2.0 / (RES - 1))
    k_best = int(np.argmax(full.sum((0, 1))))

    titles = []
    for (z, name), (_, _, _, load) in zip(LEVELS, renders):
        frac = (np.tanh(z) + 1) / 2
        titles.append(
            f"{name}\nz = {z:+.2f}, (tanh z + 1)/2 = {frac:.2f}\n{load.sum():.0f} lesion voxels ({load.sum() / brain * 100:.2f}% of brain)"
        )
        print(f"z={z:+.3f} frac={frac:.3f} lesion voxels={load.sum():.1f} ({load.sum() / brain * 100:.2f}% of brain)")

    # ---- overview: where the lesions are, and one slice through the busiest level
    silhouette = np.rot90((tissue > 0).sum(2) > 0)
    proj_max = max(np.rot90(r[3].sum(2)).max() for r in renders)
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.4))
    for j, (x1, x2, _, load) in enumerate(renders):
        ax = axes[0, j]
        ax.imshow(silhouette, cmap="gray", vmin=0, vmax=3)
        proj = np.rot90(load.sum(2))
        ax.imshow(np.ma.masked_where(proj == 0, proj), cmap="autumn_r", vmin=0, vmax=proj_max)
        for i, c in enumerate(centres):
            ax.add_patch(Circle((c[0], RES - 1 - c[1]), r_full + 0.5, fill=False, ls="--", lw=0.7, ec="#4dabf7"))
            ax.text(c[0] + r_full + 1.5, RES - 1 - c[1], str(i + 1), color="#4dabf7", fontsize=8, va="center")
        ax.set_title(titles[j], fontsize=9)
        off(ax)
        for row, x, name in ((1, x1, "T1"), (2, x2, "FLAIR")):
            ax = axes[row, j]
            ax.imshow(axial(x, k_best), cmap="gray", vmin=0, vmax=1.1)
            off(ax)
            if j == 0:
                ax.set_ylabel(f"{name}, axial slice {k_best}", fontsize=10)
    axes[0, 0].set_ylabel("lesion load, summed top-down\n(dashed = full-size sphere)", fontsize=9)
    fig.suptitle(
        "--synthetic-lesion-target burden: same subject, lesion positions and style; only z_content[2] changes.\n"
        "All four spheres share one radius, r_max·((tanh z + 1)/2)^(1/3), so total volume is linear in (tanh z + 1)/2.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/1_burden_overview.png", dpi=130)

    # ---- every lesion, zoomed, T1 and FLAIR
    fig, axes = plt.subplots(3, 2 * len(centres), figsize=(1.55 * 2 * len(centres), 5.6))
    for j, (x1, x2, _, _) in enumerate(renders):
        for i, c in enumerate(centres):
            for m, (x, name) in enumerate(((x1, "T1"), (x2, "FLAIR"))):
                ax = axes[j, 2 * i + m]
                ax.imshow(crop(x, c), cmap="gray", vmin=0, vmax=1.1)
                ax.add_patch(Circle((CROP, CROP), r_full + 0.5, fill=False, ls="--", lw=0.7, ec="#e8590c"))
                off(ax)
                if j == 0:
                    ax.set_title(f"lesion {i + 1} · {name}", fontsize=8.5)
                if i == 0 and m == 0:
                    ax.set_ylabel(LEVELS[j][1].replace("\n", " ").replace(" (", "\n("), fontsize=8.5)
    fig.suptitle(
        "Each lesion zoomed (19×19 voxels, axial slice through its centre). Dashed circle = the full-size sphere "
        f"(radius {inner.renderer.lesion_radius:g} = {r_full:.1f} voxels).\n"
        "T1 lesion value 0.0 (darker than CSF), FLAIR 1.0 (brightest tissue); the renderer's 3×3×3 blur softens the smallest ones.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/2_burden_each_lesion.png", dpi=130)
