"""1 lesion vs 4 lesions at matched total volume: one subject, one burden value, four generator arms.

Recipe = experiments/encoder/conv_mlp_burden_oracle_k1.yaml (burden target, styled lesion, T1 lesion value 0.0,
wm_interior, clean_content, res 64). Raw intensities, gain 1 / bias 0. The subject is one that fits its lesions
in every arm on the first draw, so anatomy and style are identical across rows; only the lesions differ.

    PYTHONPATH=. python out/burden_matched_volume_20261027/render.py out/burden_matched_volume_20261027 [z]
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.datasets import SyntheticBrainDataset

OUT = sys.argv[1]
Z = float(sys.argv[2]) if len(sys.argv) > 2 else 1.645  # 95th percentile of the burden latent
RES = 64
STYLE = torch.tensor([0.0, 0.0, 0.3])
ARMS = [  # (label, count, radius), in matched-volume pairs
    ("4 lesions, r 0.100", 4, 0.1),
    ("1 lesion, r 0.159", 1, 0.1 * 4 ** (1 / 3)),
    ("4 lesions, r 0.063", 4, 0.1 * 4 ** (-1 / 3)),
    ("1 lesion, r 0.100", 1, 0.1),
]


def build(count, radius):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SyntheticBrainDataset(
            mode="train",
            spatial_size=(RES,) * 3,
            synthetic_num_samples=64,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_lesion_placement="wm_interior",
            synthetic_lesion_target="burden",
            synthetic_lesion_intensity="styled",
            synthetic_lesion_t1_value=0.0,
            synthetic_lesion_count=count,
            synthetic_lesion_radius=radius,
            synthetic_seed=42,
        )._inner


def render(inner, idx):
    seed = inner.sample_seed_for(idx)
    d = inner._draw_pseudo_mri(seed)
    zc = d["z_content"].clone()
    zc[2] = Z
    with torch.no_grad():
        tissue, load = inner.renderer.render_structure(
            zc, d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
        )
    x1, x2, _ = inner.render_pseudo_mri(
        zc, d["z_deformation"], d["z_fissure"], STYLE, STYLE, seed, z_lesion=d["z_lesion"]
    )
    return x1.squeeze(0).numpy(), x2.squeeze(0).numpy(), tissue.numpy(), load.numpy()


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def pick_subject(inners):
    """A subject every arm accepts on its first draw, with the brain size closest to the median."""
    sizes = {}
    for idx in range(40):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for inner in inners:
                inner.sample_seed_for(idx)
        if all(inner._accepted_attempt.get(idx, 0) == 0 for inner in inners):
            sizes[idx] = (render(inners[0], idx)[2] > 0).sum()
    median = np.median(list(sizes.values()))
    return min(sizes, key=lambda idx: abs(sizes[idx] - median))


if __name__ == "__main__":
    inners = [build(count, radius) for _, count, radius in ARMS]
    idx = pick_subject(inners)
    renders = [render(inner, idx) for inner in inners]
    brain = (renders[0][2] > 0).sum()
    silhouettes = [np.rot90((renders[0][2] > 0).sum(axis) > 0) for axis in (2, 1)]
    proj_max = max(r[3].sum(axis).max() for r in renders for axis in (2, 1))
    frac = (np.tanh(Z) + 1) / 2

    fig, axes = plt.subplots(len(ARMS), 5, figsize=(12.8, 11.6))
    for i, ((label, count, radius), (x1, x2, _, load)) in enumerate(zip(ARMS, renders)):
        kz = int(np.argmax(load.sum((0, 1))))
        ky = int(np.argmax(load.sum((0, 2))))
        for col, (axis, title) in enumerate(((2, "lesion load, top-down"), (1, "lesion load, front-on"))):
            ax = axes[i, col]
            ax.imshow(silhouettes[col], cmap="gray", vmin=0, vmax=3)
            proj = np.rot90(load.sum(axis))
            ax.imshow(np.ma.masked_where(proj == 0, proj), cmap="autumn_r", vmin=0, vmax=proj_max)
            ax.set_title(title, fontsize=8.5)
        axes[i, 0].set_ylabel(
            f"{label}\n{load.sum():.0f} lesion voxels\n({load.sum() / brain * 100:.2f}% of brain)", fontsize=9.5
        )
        panels = (
            (2, np.rot90(x1[:, :, kz]), f"T1, axial {kz}"),
            (3, np.rot90(x2[:, :, kz]), f"FLAIR, axial {kz}"),
            (4, np.rot90(x2[:, ky, :]), f"FLAIR, coronal {ky}"),
        )
        for col, img, title in panels:
            axes[i, col].imshow(img, cmap="gray", vmin=0, vmax=1.1)
            axes[i, col].set_title(title, fontsize=8.5)
        for ax in axes[i]:
            off(ax)
    fig.subplots_adjust(top=0.9, hspace=0.32)
    fig.suptitle(
        f"1 lesion vs 4 lesions at matched total volume (subject {idx}, same anatomy and style in every row; "
        f"burden z = {Z:+.2f}, (tanh z + 1)/2 = {frac:.2f}).\n"
        "Rows 1–2: the 4-lesion recipe's volume, as four spheres or one. Rows 3–4: the 1-lesion setting's volume, "
        "as one sphere or four.\nProjections sum the lesion load, so two lesions can stack in one view; slices are each "
        "row's busiest level. T1 lesion value 0.0, FLAIR lesion 1.0.",
        fontsize=9,
    )
    y = (axes[1, 0].get_position().y0 + axes[2, 0].get_position().y1) / 2
    fig.add_artist(plt.Line2D([0.03, 0.97], [y, y], color="#868e96", lw=1, ls="--"))
    fig.savefig(f"{OUT}/1_matched_volume.png", dpi=130, bbox_inches="tight")
    print(f"subject {idx}, brain {brain} voxels; lesion voxels per row:", [f"{r[3].sum():.1f}" for r in renders])
