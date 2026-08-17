"""Render the synthetic generator to PNG so a config change can be SEEN, not just measured.

`eval/generator_defects.py` says what a config does to identifiability. This says what it
does to the pictures — which is the check that catches "the fix worked but the anatomy is
now nonsense".

Three panels:

  subjects    the same subjects under two configs, both views. Row-matched by seed, so
              differences are the config and nothing else.
  sweep       one factor swept from -1 to +1 with everything else pinned, as a difference
              map against the z=0 render. This is the panel that shows what
              `--synthetic-cortex-parameterization patterned` and
              `--synthetic-center-local-deformations` actually changed: a factor that used
              to shift the whole boundary now shows a signed front/back (or lobe-vs-rest)
              pattern.
  contrast    the tissue LUT after normalization, which is why the WM/GM edge dominates
              the GM/background edge under fixed_reference.

Usage
-----
    python -m eval.preview_generator --out /tmp/gen.png
    python -m eval.preview_generator --out /tmp/sweep.png --panel sweep --sweep-dim 5
"""

import argparse
import warnings

import numpy as np

from data.datasets import SyntheticBrainDataset

CONTENT_NAMES = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]

# The two configs this project actually compares: the flagship, and the flagship with every
# generator-side defect closed (experiments/synthetic_identifiable.yaml).
OLD = dict(
    synthetic_content_prior="normal",
    synthetic_content_squash="auto",
    synthetic_cortex_parameterization="additive",
    synthetic_center_local_deformations=False,
    synthetic_identifiable_ventricle=False,
    synthetic_lesion_radius=0.1,
)
NEW = dict(
    synthetic_content_prior="uniform",
    synthetic_content_squash="none",
    synthetic_cortex_parameterization="patterned",
    synthetic_center_local_deformations=True,
    synthetic_identifiable_ventricle=True,
    synthetic_lesion_radius=0.14,
)


def build(res, overrides, n=64):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SyntheticBrainDataset(
            mode="train",
            spatial_size=(res,) * 3,
            synthetic_num_samples=n,
            synthetic_n_content=9,
            synthetic_n_style=3,
            synthetic_normalize="fixed_reference",
            synthetic_causal=True,
            synthetic_causal_graph="random",
            synthetic_causal_edge_prob=0.5,
            synthetic_clean_content=True,
            synthetic_seed=42,
            **overrides,
        )


def render(ds, idx, z_content=None):
    inner = ds._inner
    lat = inner[idx][2]
    zc = lat["z_content"] if z_content is None else z_content
    x1, x2, mask = inner.render_pseudo_mri(
        zc,
        lat["z_deformation"],
        lat["z_fissure"],
        lat["z_style_v1"],
        lat["z_style_v2"],
        inner.sample_seed_for(idx),
    )
    x1, x2 = ds.normalize_views(x1, x2, mask, mask.clone())
    return x1.squeeze(0).numpy(), x2.squeeze(0).numpy()


def _slices(vol):
    """Mid axial / coronal / sagittal, stacked side by side."""
    r = vol.shape[0] // 2
    return [np.rot90(vol[r]), np.rot90(vol[:, r]), np.rot90(vol[:, :, r])]


def panel_subjects(args, plt):
    ds_old, ds_new = build(args.res, OLD), build(args.res, NEW)
    n = args.n_subjects
    fig, axes = plt.subplots(4, n * 3, figsize=(1.5 * n * 3, 6.6))
    for col, idx in enumerate(range(n)):
        for row, (ds, view, label) in enumerate(
            [(ds_old, 0, "OLD T1"), (ds_old, 1, "OLD FLAIR"), (ds_new, 0, "NEW T1"), (ds_new, 1, "NEW FLAIR")]
        ):
            vols = render(ds, idx)
            for k, sl in enumerate(_slices(vols[view])):
                ax = axes[row, col * 3 + k]
                ax.imshow(sl, cmap="gray", vmin=-1.2, vmax=1.2)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0 and k == 0:
                    ax.set_ylabel(label, fontsize=9)
                if row == 0 and k == 1:
                    ax.set_title(f"subject {idx}", fontsize=9)
    fig.suptitle(
        "Same subjects, same seeds — flagship generator (OLD) vs every defect closed (NEW). "
        "Anatomy should look equally plausible; only identifiability changed.",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def panel_sweep(args, plt):
    """One factor swept, shown as a DIFFERENCE map against its own z=0 render.

    The difference is the point: a factor whose Jacobian is a uniform radial shift makes a
    symmetric ring, and one that has left the radial subspace makes a signed pattern. That
    visual difference IS the degeneracy fix.
    """
    d = args.sweep_dim
    vals = [-1.0, -0.5, 0.5, 1.0]
    fig, axes = plt.subplots(4, len(vals) + 1, figsize=(2.0 * (len(vals) + 1), 8.2))
    for row, (ds, tag) in enumerate([(build(args.res, OLD), "OLD"), (build(args.res, NEW), "NEW")]):
        base_z = ds._inner[0][2]["z_content"].clone()
        base_z[:] = 0.0
        ref = render(ds, 0, base_z)[0]
        for j, v in enumerate(vals):
            z = base_z.clone()
            z[d] = v
            vol = render(ds, 0, z)[0]
            # The plane MUST contain the axis the factor varies along, or the pattern is
            # constant across the slice by construction: the l=2 thickness harmonic runs
            # along y, so a slice at fixed y shows a plain ring and hides the whole effect.
            for k, (sl, r_) in enumerate(zip(_slices(vol), _slices(ref))):
                if k != args.plane:
                    continue
                axes[2 * row, j].imshow(sl, cmap="gray", vmin=-1.2, vmax=1.2)
                dm = sl - r_
                lim = max(np.abs(dm).max(), 1e-6)
                axes[2 * row + 1, j].imshow(dm, cmap="bwr", vmin=-lim, vmax=lim)
            axes[2 * row, j].set_title(f"z[{d}] = {v:+.1f}", fontsize=9)
            for rr in (2 * row, 2 * row + 1):
                axes[rr, j].set_xticks([])
                axes[rr, j].set_yticks([])
        axes[2 * row, 0].set_ylabel(f"{tag} render", fontsize=9)
        axes[2 * row + 1, 0].set_ylabel(f"{tag} diff vs z=0", fontsize=9)
        axes[2 * row, len(vals)].imshow(_slices(ref)[args.plane], cmap="gray", vmin=-1.2, vmax=1.2)
        axes[2 * row, len(vals)].set_title("z = 0 (ref)", fontsize=9)
        axes[2 * row + 1, len(vals)].axis("off")
        for rr in (2 * row, 2 * row + 1):
            axes[rr, len(vals)].set_xticks([])
            axes[rr, len(vals)].set_yticks([])
    fig.suptitle(
        f"z_content[{d}] = {CONTENT_NAMES[d]} swept, plane {args.plane}. Red/blue = brighter/darker than the z=0 render.\n"
        "A uniform radial shift makes a symmetric ring; a signed front/back or lobe pattern means the "
        "factor has left the radial subspace and can no longer be mimicked by brain_size.",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def panel_contrast(args, plt):
    ds = build(args.res, NEW)
    ds._compute_fixed_reference()
    mu, sc = ds._fixed_mean, ds._fixed_scale
    names = ["background", "CSF", "WM", "GM", "fissure"]
    raw = [0.0, 0.1, 0.8, 0.5, 0.3]
    norm = [(v - mu) / sc for v in raw]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(names, raw, color="0.6")
    ax[0].axhline(mu, color="crimson", ls="--", label=f"fixed_reference mean = {mu:.3f}")
    ax[0].set_title("raw T1 LUT")
    ax[0].legend(fontsize=8)
    ax[1].bar(names, norm, color="0.6")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("after fixed_reference normalization")
    ax[1].annotate(
        f"GM lands at {norm[3]:+.3f}, i.e. almost the masked background (0):\n"
        f"outer GM/background edge {abs(norm[3]):.3f}  vs  inner WM/GM edge {abs(norm[2] - norm[3]):.3f}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--panel", default="subjects", choices=["subjects", "sweep", "contrast"])
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--n-subjects", type=int, default=4)
    p.add_argument("--sweep-dim", type=int, default=5)
    p.add_argument(
        "--plane",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Slice plane. 2 = x-y (default) and 0 = y-z both contain the y axis the l=2 "
        "thickness harmonic runs along; 1 = x-z fixes y and hides it entirely. Prefer 2: "
        "plane 0 is the mid-sagittal slice, where the fissure sheet (|x|<0.03) fills the "
        "image and the ventricle split (|x|>0.05) excludes the ventricle.",
    )
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = {"subjects": panel_subjects, "sweep": panel_sweep, "contrast": panel_contrast}[args.panel](args, plt)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
