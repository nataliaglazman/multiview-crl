"""Before/after renders for three generator flags, same subject and seeds, one flag changed at a time.

Recipe = experiments/encoder/synthetic_defaults.yaml (clean_content, wm_interior, non-causal).
Raw intensities (pre-normalisation) so the acquisition gain/bias is visible on one fixed scale.
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
RES = 64
SUBJECT = 3


def build(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SyntheticBrainDataset(
            mode="train",
            spatial_size=(RES,) * 3,
            synthetic_num_samples=16,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_lesion_placement="wm_interior",
            synthetic_seed=42,
            **overrides,
        )._inner


def latents(inner, idx):
    lat = inner[idx][2]
    return lat, inner.sample_seed_for(idx)


def render(inner, lat, seed, zc=None, s1=None, s2=None):
    zc = lat["z_content"] if zc is None else zc
    s1 = lat["z_style_v1"] if s1 is None else s1
    s2 = lat["z_style_v2"] if s2 is None else s2
    with torch.no_grad():
        tissue, lesion = inner.renderer.render_structure(
            zc, lat["z_deformation"], lat["z_fissure"], device=torch.device("cpu"), clean=True
        )
    x1, x2, _ = inner.render_pseudo_mri(zc, lat["z_deformation"], lat["z_fissure"], s1, s2, seed)
    return x1.squeeze(0).numpy(), x2.squeeze(0).numpy(), tissue.numpy(), lesion.numpy()


def ax_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def axial(v, k):
    return np.rot90(v[:, :, k])


def coronal(v, k):
    return np.rot90(v[:, k, :])


# ---------------------------------------------------------------- 1. lesion intensity
def fig_lesion():
    fixed, styled = build(synthetic_lesion_intensity="fixed"), build(synthetic_lesion_intensity="styled")
    lat, seed = latents(fixed, SUBJECT)
    _, _, tissue, lesion = render(fixed, lat, seed)
    c = np.round(np.argwhere(lesion > 0.5).mean(0)).astype(int)
    kz = c[2]
    les = lesion > 0.5
    ring = np.zeros_like(les)
    lo, hi = np.maximum(c - 6, 0), c + 7
    ring[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    wm_ring = ring & (tissue == 2) & ~les

    styles = [("low gain, −bias", -1.0, -1.0), ("mid", 0.0, 0.0), ("high gain, +bias", 1.0, 1.0)]
    rows = [
        ("fixed · T1", fixed, 0),
        ("styled · T1", styled, 0),
        ("fixed · FLAIR", fixed, 1),
        ("styled · FLAIR", styled, 1),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(8.4, 11.4))
    zoom = slice(max(c[0] - 16, 0), c[0] + 16), slice(max(c[1] - 16, 0), c[1] + 16)
    for j, (name, g, b) in enumerate(styles):
        s = torch.tensor([g, b, 0.3])
        for i, (rname, inner, view) in enumerate(rows):
            x = render(inner, lat, seed, s1=s, s2=s)[view]
            contrast = x[les].mean() - x[wm_ring].mean()
            ax = axes[i, j]
            ax.imshow(np.rot90(x[zoom[0], zoom[1], kz]), cmap="gray", vmin=0, vmax=1.3)
            ax.contour(np.rot90(les[zoom[0], zoom[1], kz]), levels=[0.5], colors="#e8590c", linewidths=0.8)
            ax.text(
                0.03,
                0.04,
                f"lesion − WM = {contrast:+.2f}",
                transform=ax.transAxes,
                color="w",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.6, pad=2, lw=0),
            )
            ax_off(ax)
            if i == 0:
                ax.set_title(f"{name}\n(z_style[0]={g:+.0f}, z_style[1]={b:+.0f})", fontsize=9)
            if j == 0:
                ax.set_ylabel(rname, fontsize=10)
    fig.suptitle(
        "--synthetic-lesion-intensity fixed vs styled\n"
        "Same subject and lesion; axial slice through the lesion centre (orange outline), zoomed 32×32.\n"
        "fixed: T1 lesion stays at 0.40 while WM moves with gain/bias, so its contrast shrinks at low gain.\n"
        "styled: the lesion goes through the same gain/bias as tissue — contrast 0.4·gain (T1), 0.6·gain (FLAIR).",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/1_lesion_intensity.png", dpi=130)


# ---------------------------------------------------------------- shared factor sweep
def fig_sweep(fname, dim, label, off_kw, on_kw, off_name, on_name, note, slicer, k):
    off, on = build(**off_kw), build(**on_kw)
    lat, seed = latents(off, SUBJECT)
    s = torch.tensor([0.0, 0.0, 0.3])
    vals = [-2.0, 0.0, 2.0]
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.4))
    for i, (inner, rname) in enumerate([(off, off_name), (on, on_name)]):
        imgs = []
        for j, v in enumerate(vals):
            zc = lat["z_content"].clone()
            zc[dim] = v
            x = render(inner, lat, seed, zc=zc, s1=s, s2=s)[0]
            imgs.append(x)
            ax = axes[i, j]
            ax.imshow(slicer(x, k), cmap="gray", vmin=0, vmax=1.1)
            ax_off(ax)
            if i == 0:
                ax.set_title(f"{label} z = {v:+.0f}", fontsize=10)
            if j == 0:
                ax.set_ylabel(rname, fontsize=10)
        d = imgs[2] - imgs[0]
        ax = axes[i, 3]
        ax.imshow(slicer(d, k), cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        ax_off(ax)
        frac = (np.abs(d) > 0.05).mean() * 100
        ax.text(
            0.03,
            0.04,
            f"|Δ|>0.05 in {frac:.1f}% of voxels",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, pad=2, lw=0),
        )
        if i == 0:
            ax.set_title("difference (z=+2) − (z=−2)\nred brighter, blue darker", fontsize=10)
    fig.suptitle(note, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=130)


if __name__ == "__main__":
    fig_lesion()
    fig_sweep(
        "2_identifiable_ventricle.png",
        1,
        "ventricle_size",
        dict(synthetic_identifiable_ventricle=False),
        dict(synthetic_identifiable_ventricle=True),
        "identifiable_ventricle OFF",
        "identifiable_ventricle ON",
        "--synthetic-identifiable-ventricle (T1, coronal mid-slice, gain/bias fixed at 0)\n"
        "ON: radius 0.12–0.28 instead of 0.10–0.20, read off the undeformed radius, and the midline fissure gets its "
        "own label/intensity (0.3) instead of CSF (0.1).",
        coronal,
        RES // 2,
    )
    fig_sweep(
        "3_sulcal_mode.png",
        8,
        "sulcal_widening",
        dict(synthetic_sulcal_mode="corrugation"),
        dict(synthetic_sulcal_mode="atrophy"),
        "sulcal_mode corrugation",
        "sulcal_mode atrophy",
        "--synthetic-sulcal-mode corrugation vs atrophy (T1, axial slice above the ventricles, gain/bias fixed at 0)\n"
        "corrugation: signed depth of a zero-mean ripple on the pial surface (cancels under GAP). "
        "atrophy: fixed sulcal clefts widen, GM→CSF only; outline/WM/ventricles unchanged.",
        axial,
        RES // 2 + 10,
    )
