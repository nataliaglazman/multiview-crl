"""Matched T1/FLAIR images for the temporal-atrophy width control.

python -m eval.synthetic.preview_temporal_atrophy --out-dir out/temporal_width
python -m eval.synthetic.preview_temporal_atrophy --out-dir out/temporal_width_centered \
    --center-local-deformations

Uses the real generator, including blur, bias field and Rician noise. All other
latents and acquisition seeds are held fixed. Saves coronal/axial PNGs, the raw
volumes and tissue labels in NPZ, and settings/changed-voxel counts in JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.synthetic.synthetic_dataset import Synthetic3DDisentanglementDataset


def render_comparison(res=64, sigmas=(0.18, 0.12, 0.10), atrophy_value=1.0, seed=42, centered=False, index=0):
    """Return matched raw images and labels; the first entry has z6=0."""
    inner = Synthetic3DDisentanglementDataset(
        num_samples=index + 1,
        res=res,
        seed=seed,
        mode="pseudo_mri",
        n_content=9,
        n_style=3,
        clean_content=True,
        center_local_deformations=centered,
    )
    lat = inner[index][2]
    sample_seed = inner.sample_seed_for(index)
    images, tissues = [], []
    for sigma, value in [(0.18, 0.0)] + [(s, atrophy_value) for s in sigmas]:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Temporal sigmas must be finite and positive")
        inner.renderer.temporal_sigma = float(sigma)
        z = lat["z_content"].clone()
        z[6] = value
        pair = inner.render_pseudo_mri(
            z, lat["z_deformation"], lat["z_fissure"], lat["z_style_v1"], lat["z_style_v2"], sample_seed
        )
        tissue, _ = inner.renderer.render_structure(z, lat["z_deformation"], lat["z_fissure"], "cpu", clean=True)
        images.append(torch.cat(pair[:2]).numpy())
        tissues.append(tissue.numpy())
    images, tissues = np.stack(images), np.stack(tissues)
    changed = (tissues[1:] != tissues[0]).sum(axis=(1, 2, 3)).tolist()
    metadata = {
        "res": res,
        "sigmas": list(sigmas),
        "atrophy_value": atrophy_value,
        "seed": seed,
        "subject_index": index,
        "sample_seed": sample_seed,
        "center_local_deformations": centered,
        "clean_content": True,
        "content_squash": "tanh",
        "normalization": "raw generator intensity; common display window per modality",
        "z_content_sampled": lat["z_content"].tolist(),
        "z_style_v1": lat["z_style_v1"].tolist(),
        "z_style_v2": lat["z_style_v2"].tolist(),
        "changed_tissue_voxels_vs_z6_zero": changed,
        "scope": "One matched synthetic subject; counts include all tissue-label changes, not a clinical volume measure.",
    }
    return images, tissues, metadata


def plot_comparison(images, metadata, plane, plt):
    """Slices pass through the atrophy centres; difference maps share one scale."""
    axis, coordinate = {"coronal": (1, 0.05), "axial": (2, -0.35)}[plane]
    grid = np.linspace(-1, 1, metadata["res"])
    slice_index = int(np.abs(grid - coordinate).argmin())
    # x is horizontal in both planes; superior (coronal) / anterior (axial) is up.
    slices = np.take(images, slice_index, axis=axis + 2).swapaxes(-2, -1)
    differences = slices - slices[0]
    bound = max(float(np.abs(differences).max()), 1e-6)
    ncols = len(images)
    fig, axes = plt.subplots(4, ncols, figsize=(3.1 * ncols, 10.8), layout="constrained")
    titles = ["Reference\nz6 = 0"] + [f"σ = {s:.2f}\nz6 = {metadata['atrophy_value']:g}" for s in metadata["sigmas"]]
    labels = ["T1", "T1 − reference", "FLAIR", "FLAIR − reference"]
    for row in range(4):
        modality, is_difference = row // 2, row % 2
        vmax = float(np.quantile(images[:, modality], 0.999))
        for col in range(ncols):
            ax = axes[row, col]
            values = differences[col, modality] if is_difference else slices[col, modality]
            im = ax.imshow(
                values,
                origin="lower",
                extent=(-1, 1, -1, 1),
                interpolation="nearest",
                cmap="RdBu_r" if is_difference else "gray",
                vmin=-bound if is_difference else 0,
                vmax=bound if is_difference else vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=11)
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
        if is_difference:
            bar = fig.colorbar(im, ax=list(axes[row]), fraction=0.015, pad=0.02)
            bar.set_label("Intensity change", fontsize=9)
    centering = "ON (includes compensation elsewhere)" if metadata["center_local_deformations"] else "OFF"
    fig.suptitle(
        f"Temporal atrophy: narrower spatial extent | {plane} slice\n"
        f"{metadata['res']}³ voxels · same anatomy, style and noise · centring {centering}\n"
        f"{'y' if plane == 'coronal' else 'z'} = {grid[slice_index]:.3f}; "
        "red = brighter, blue = darker than reference",
        fontsize=13,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.18, 0.12, 0.10])
    parser.add_argument("--atrophy-value", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject-index", type=int, default=0)
    parser.add_argument("--center-local-deformations", action="store_true")
    args = parser.parse_args()
    if args.res < 8 or args.subject_index < 0:
        parser.error("--res must be >= 8 and --subject-index must be nonnegative")
    if not all(np.isfinite(s) and s > 0 for s in args.sigmas):
        parser.error("--sigmas must be finite and positive")
    if not np.isfinite(args.atrophy_value):
        parser.error("--atrophy-value must be finite")
    torch.set_num_threads(1)
    images, tissues, metadata = render_comparison(
        args.res, args.sigmas, args.atrophy_value, args.seed, args.center_local_deformations, args.subject_index
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "volumes.npz", images=images, tissues=tissues)
    (args.out_dir / "comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for plane in ("coronal", "axial"):
        fig = plot_comparison(images, metadata, plane, plt)
        path = args.out_dir / f"{plane}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")
    print(json.dumps(dict(zip(args.sigmas, metadata["changed_tissue_voxels_vs_z6_zero"])), indent=2))


if __name__ == "__main__":
    main()
