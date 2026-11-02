"""What the run patch8_sulcal_ventricle_strat_burden actually feeds the encoder.

flags.txt is the run's command line, unedited. It goes through the trainer's own parser and make_dataset, so the
images below are the tensors in batch["image"]: fixed_reference normalised and masked, background exactly 0.

    1_subjects.png       training subjects 0..N-1, T1 and FLAIR, three planes through each lesion centre
    2_burden_range.png   one subject swept through the burden prior, real style pinned, plus the population histogram

    PYTHONPATH=. python out/patch8_preview_20261102/render.py out/patch8_preview_20261102 [n_subjects] [sweep_subject]
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

import eval.metrics.dci as dci
import training.encoder_args as encoder_args
from training.main_conv_synthetic import make_dataset

OUT = sys.argv[1]
N_SUBJECTS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SWEEP_SUBJECT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_POP = 300
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
CROP = 9
LESION = "#e8590c"


def structure(inner, d, zc):
    with torch.no_grad():
        return inner.renderer.render_structure(
            zc, d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
        )


def network_input(ds, d, zc, seed):
    x1, x2, mask = ds._inner.render_pseudo_mri(
        zc, d["z_deformation"], d["z_fissure"], d["z_style_v1"], d["z_style_v2"], seed, z_lesion=d["z_lesion"]
    )
    x1, x2 = ds.normalize_views(x1, x2, mask, mask.clone())
    return x1.squeeze(0).numpy(), x2.squeeze(0).numpy()


def planes(vol, c):
    cx, cy, cz = c
    return [np.rot90(vol[:, :, cz]), np.rot90(vol[:, cy, :]), np.rot90(vol[cx, :, :])]


def plane_centres(c, res):
    """Where (cx, cy, cz) lands in the rot90'd axial, coronal and sagittal panels."""
    cx, cy, cz = c
    return [(cx, res - 1 - cy), (cx, res - 1 - cz), (cy, res - 1 - cz)]


def crop(img2d, c):
    p = np.pad(img2d, CROP)
    return np.rot90(p[c[0] : c[0] + 2 * CROP + 1, c[1] : c[1] + 2 * CROP + 1])


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def subjects_figure(ds, args, window, full_radius):
    res = args.res
    fig = plt.figure(figsize=(1.75 * 6 + 0.3, 1.82 * N_SUBJECTS + 0.9))
    gs = GridSpec(N_SUBJECTS, 7, width_ratios=[1, 1, 1, 0.08, 1, 1, 1], wspace=0.03, hspace=0.05, top=0.93, figure=fig)
    names = ("axial", "coronal", "sagittal")
    rows = []
    for i in range(N_SUBJECTS):
        item = ds[i]
        x1, x2 = item["image"][0][0].numpy(), item["image"][1][0].numpy()
        lat = item["gt_latents"]
        d = ds._inner._draw_pseudo_mri(ds._inner.sample_seed_for(i))
        assert torch.equal(d["z_content"], lat["z_content"])
        tissue, load = structure(ds._inner, d, d["z_content"])
        full = lat["z_content"].clone()
        full[2] = 3.0
        centre = tuple(int(v) for v in np.round(center_of_mass(structure(ds._inner, d, full)[1].numpy())))
        volume = float(load.sum())
        radius = full_radius * ((np.tanh(float(lat["z_content"][2])) + 1) / 2) ** (1 / 3)
        rows.append(
            (i, volume, int((tissue > 0).sum()), float(x1.min()), float(x1.max()), float(x2.min()), float(x2.max()))
        )
        for col, (vol, title) in zip((0, 4), ((x1, "T1"), (x2, "FLAIR"))):
            for k, (img, (px, py)) in enumerate(zip(planes(vol, centre), plane_centres(centre, res))):
                ax = fig.add_subplot(gs[i, col + k])
                ax.imshow(img, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
                ax.add_patch(Circle((px, py), full_radius + 3.0, fill=False, ls="--", lw=0.7, ec=LESION))
                off(ax)
                if i == 0:
                    ax.set_title(f"{title} {names[k]}", fontsize=9.5)
        ax = fig.axes[-6]
        s1, s2 = lat["z_style_v1"], lat["z_style_v2"]
        ax.set_ylabel(
            f"subject {i}\nburden {volume:.0f} vox, r {radius:.1f}\ngain z {s1[0]:+.1f} / {s2[0]:+.1f}\nbias z {s1[1]:+.1f} / {s2[1]:+.1f}",
            fontsize=8,
        )
    fig.suptitle(
        "Encoder inputs for patch8_sulcal_ventricle_strat_burden: training subjects 0–"
        f"{N_SUBJECTS - 1}, each cut through its own lesion centre (dashed ring around the lesion site).\n"
        f"One grey window for all panels, [{window[0]:.2f}, {window[1]:.2f}]. Background is exactly 0, so it reads as mid-grey: "
        "values below 0 are darker than the background.",
        fontsize=10,
        y=0.995,
    )
    fig.savefig(f"{OUT}/1_subjects.png", dpi=140, bbox_inches="tight")
    return rows


def burden_figure(ds, args, window, full_radius):
    inner = ds._inner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volumes = []
        for i in range(N_POP):
            d = inner._draw_pseudo_mri(inner.sample_seed_for(i))
            volumes.append(float(structure(inner, d, d["z_content"])[1].sum()))
    volumes = np.array(volumes)

    seed = inner.sample_seed_for(SWEEP_SUBJECT)
    d = inner._draw_pseudo_mri(seed)
    item = ds[SWEEP_SUBJECT]["image"]
    for rendered, stored in zip(network_input(ds, d, d["z_content"], seed), item):
        assert np.array_equal(rendered, stored[0].numpy())
    print("sweep render path reproduces the dataset's tensors exactly")
    levels = []
    for p in PERCENTILES:
        z = float(norm.ppf(p / 100))
        zc = d["z_content"].clone()
        zc[2] = z
        load = structure(inner, d, zc)[1]
        x1, x2 = network_input(ds, d, zc, seed)
        levels.append((p, z, float(load.sum()), x1, x2))
    zc = d["z_content"].clone()
    zc[2] = 3.0
    full = structure(inner, d, zc)[1].numpy()
    cx, cy, cz = (int(v) for v in np.round(center_of_mass(full)))

    k = len(levels)
    fig = plt.figure(figsize=(1.9 * k, 10.6))
    gs = GridSpec(5, k, height_ratios=[1, 1, 1, 1, 1.15], hspace=0.35, wspace=0.06, figure=fig)
    rows = (("T1", 3, False), ("FLAIR", 4, False), ("T1 zoom", 3, True), ("FLAIR zoom", 4, True))
    for j, level in enumerate(levels):
        p, z, vol = level[:3]
        frac = (np.tanh(z) + 1) / 2
        for i, (name, slot, zoom) in enumerate(rows):
            axial = level[slot][:, :, cz]
            img = crop(axial, (cx, cy)) if zoom else np.rot90(axial)
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
            if zoom:
                ax.add_patch(Circle((CROP, CROP), full_radius + 0.5, fill=False, ls="--", lw=0.7, ec=LESION))
            off(ax)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
            if i == 0:
                ax.set_title(
                    f"{p}{'st' if p == 1 else 'th'} percentile\nz = {z:+.2f}\n{vol:.0f} lesion voxels\n"
                    f"radius {full_radius * frac ** (1 / 3):.1f} vox",
                    fontsize=8.5,
                )
    ax = fig.add_subplot(gs[4, :])
    ax.hist(volumes, bins=np.linspace(0, volumes.max() * 1.02, 36), color="#adb5bd", edgecolor="white")
    for p, _, vol, _, _ in levels:
        ax.axvline(vol, color=LESION, lw=1)
        ax.text(
            vol,
            ax.get_ylim()[1] * 0.97,
            f"{p}",
            color=LESION,
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )
    q = np.percentile(volumes, [5, 50, 95])
    ax.set_xlabel(
        f"lesion voxels per subject ({N_POP} training subjects): median {q[1]:.0f}, 5–95% {q[0]:.0f}–{q[2]:.0f}. "
        "Orange lines = the columns above.",
        fontsize=9,
    )
    ax.set_ylabel("subjects")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Burden with one lesion: subject {SWEEP_SUBJECT}, only z_content[2] changes. Anatomy, lesion site, style and noise are that subject's own.\n"
        f"Axial slice through the lesion centre; zooms are 19×19 voxels; dashed circle = the maximum radius ({full_radius:.1f} voxels).\n"
        f"T1 lesion value {args.synthetic_lesion_t1_value}, styled. Same grey window as figure 1.",
        fontsize=9.5,
        y=0.995,
    )
    fig.savefig(f"{OUT}/2_burden_range.png", dpi=130, bbox_inches="tight")
    return q, volumes, levels


if __name__ == "__main__":
    args = encoder_args.parse_args(open(f"{OUT}/flags.txt").read().split())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = make_dataset(args, "train", args.num_train_samples)
        first = ds[0]
    print("factors:", dci.content_factor_names(args.n_content, args.synthetic_lesion_target))
    print("fixed_reference mean/scale:", round(ds._fixed_mean, 4), round(ds._fixed_scale, 4))
    print("shape:", tuple(first["image"][0].shape), "synthetic_causal:", args.synthetic_causal)
    spacing = 2.0 / (args.res - 1)
    full_radius = ds._inner.renderer.lesion_radius / spacing
    sample = torch.cat([ds[i]["image"][v].flatten() for i in range(12) for v in (0, 1)])
    sample = sample[sample != 0]
    lo, hi = (
        float(v) for v in torch.quantile(sample[:: max(1, sample.numel() // 2_000_000)], torch.tensor([0.002, 0.998]))
    )
    window = (lo, hi)
    print("display window:", window)
    rows = subjects_figure(ds, args, window, full_radius)
    for r in rows:
        print("subject %d: %.0f lesion vox, %d brain vox, T1 [%.2f, %.2f], FLAIR [%.2f, %.2f]" % r)
    q, volumes, levels = burden_figure(ds, args, window, full_radius)
    print("burden 5/50/95%:", np.round(q, 1), "max", round(float(volumes.max()), 1))
    print(
        f"population: <5 vox {np.mean(volumes < 5):.1%}, <10 vox {np.mean(volumes < 10):.1%}, <20 vox {np.mean(volumes < 20):.1%}"
    )
    print("sweep:", [(p, round(v, 1)) for p, _, v, _, _ in levels])
