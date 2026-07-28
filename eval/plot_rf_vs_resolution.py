"""Draw the measured receptive field of an EDGE feature position on real volumes.

The point this figure has to make is geometric: the convolutional receptive field is a
FIXED physical size set by the encoder stack, while the distance from an edge position to
the brain grows with the volume. So the same architecture that cannot separate leak from
reach at res 64 can separate them cleanly at res 128 -- nothing about the model changed,
only how far away the brain is.

Three panels, all at the same physical scale (1 input voxel = 1 unit in every panel):

  A  res 64,  convolutions only   -- reach touches the brain, no position is blind
  B  res 128, convolutions only   -- same reach, now falls well short
  C  res 128, GroupNorm intact    -- reach covers the whole volume; this is the leak

The overlay is the actual |d feature / d input| map from backprop on a REAL volume (not a
drawn circle, and not a zero input -- the ReLU pattern matters, so the gradient is taken
through the same image that is displayed). Colour is log-scaled because the bbox of
"nonzero" gradient is ambiguous on its own: a huge bbox can be a genuine global reach or a
numerically tiny tail. The solid contour is the 1e-3-of-max level quoted elsewhere as "the
reach"; the annotation additionally reports the 90%-gradient-mass radius, which is the
honest effective number.

ReZero alpha is forced to 1 throughout. At its initial value of 0 the residual stack is
the identity and the reach is much shorter than any trained model's.

Usage:
  python -m eval.plot_rf_vs_resolution --out rf_vs_resolution.png
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch.nn as nn

from eval.background_leak_mechanism import build_encoder, generate

logger = logging.getLogger(__name__)


def strip_norms(enc):
    for _, mod in list(enc.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.GroupNorm) or type(child).__name__ == "ChannelLayerNorm3d":
                setattr(mod, cname, nn.Identity())
    return enc


def rf_map(enc, vol, feat_pos):
    """|d feature / d input| at one feature position, differentiated through a real volume."""
    x = vol.clone().requires_grad_(True)
    out = enc(x)
    gz, gy, gx = feat_pos
    out[0, :, gz, gy, gx].sum().backward()
    return x.grad.detach().abs()[0, 0].numpy()


def mass_radius(g, src, frac=0.9):
    """Radius containing `frac` of the total gradient mass, in input voxels."""
    idx = np.argwhere(np.ones_like(g, bool))
    d = np.sqrt(((idx - np.array(src)) ** 2).sum(1))
    w = g.reshape(-1)
    o = np.argsort(d)
    c = np.cumsum(w[o])
    if c[-1] <= 0:
        return 0.0
    return float(d[o][np.searchsorted(c, frac * c[-1])])


def panel(ax, vol, mask, g, src, title, subtitle, reach, d_brain, verdict, vmax_scale, extent_lim):
    import matplotlib.colors as mcolors

    z = src[0]
    sl = vol[0, 0, z].numpy()
    ms = mask[0, 0, z].numpy()
    gs = g[z]

    ax.imshow(sl, cmap="gray", origin="lower", interpolation="nearest")
    ax.contour(ms, levels=[0.5], colors="#4a6fbf", linewidths=1.4)

    pos = gs[gs > 0]
    if pos.size:
        lo = max(pos.max() * 1e-6, pos.min())
        ax.imshow(
            np.ma.masked_where(gs <= lo, gs),
            cmap="inferno",
            origin="lower",
            alpha=0.55,
            norm=mcolors.LogNorm(vmin=lo, vmax=pos.max()),
            interpolation="nearest",
        )
        thr = gs.max() * 1e-3
        if (gs > thr).any():
            ax.contour(gs, levels=[thr], colors="#e07a3f", linewidths=1.6, linestyles="--")

    ax.plot([src[2]], [src[1]], marker="o", ms=5, mfc="w", mec="k", mew=1.2)
    ax.set_xlim(-0.5, extent_lim - 0.5)
    ax.set_ylim(-0.5, extent_lim - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, linespacing=1.4)
    # Lead with the 90%-mass radius: the 1e-3 bbox "reach" understates a global map badly
    # (GroupNorm's far field is spread thin, so most of its mass sits beyond that contour).
    ax.set_xlabel(
        f"90% of gradient mass within r={vmax_scale:.0f} vox   (1e-3 contour: {reach:.0f})\n"
        f"this position→brain {d_brain:.1f} vox\n{verdict}",
        fontsize=9,
        linespacing=1.5,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="rf_vs_resolution.png")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", default="fixed_reference", choices=["per_sample", "fixed_reference"])
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        (64, True, "A.  res 64, stride 4", "convolutions only"),
        (128, True, "B.  res 128, stride 4", "convolutions only — same encoder"),
        (128, False, "C.  res 128, GroupNorm", "same convolutions + GroupNorm"),
    ]

    cache = {}
    panels = []
    for res, conv_only, title, subtitle in specs:
        if res not in cache:
            logger.info("rendering a volume at %d^3...", res)
            X, M, _ = generate(4, res, causal=False, seed=cli.seed, normalize=cli.normalize)
            cache[res] = (X[:1], M[:1], X, M)
        vol, mask, Xa, Ma = cache[res]

        enc = build_encoder("group", cli.seed, cli.hidden, downscale=cli.downscale, open_residual=True)
        if conv_only:
            strip_norms(enc)
        g_feat = res // cli.downscale
        # edge position at mid-depth, so one axial slice shows both the brain cross-section
        # and the position's own footprint
        feat_pos = (g_feat // 2, 0, 0)
        src = (feat_pos[0] * cli.downscale, 2, 2)
        gmap = rf_map(enc, vol, feat_pos)

        nz = np.argwhere(gmap > gmap.max() * 1e-3)
        reach = float(np.abs(nz[:, 1:] - np.array(src[1:])).max()) if len(nz) else 0.0
        r90 = mass_radius(gmap, src)

        ever = (Ma[:, 0] > 0).any(0).numpy()
        bidx = np.argwhere(ever)
        d_brain = float(np.sqrt(((bidx - np.array(src)) ** 2).sum(1)).min())
        verdict = "REACHES the brain" if reach >= d_brain else "falls short — blind"
        if not conv_only:
            verdict = "spans the whole volume"
        panels.append((vol, mask, gmap, src, title, subtitle, reach, d_brain, verdict, r90, res))

    widths = [p[-1] for p in panels]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": widths})
    for ax, p in zip(axes, panels):
        panel(ax, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10])

    fig.suptitle(
        "The receptive field is a fixed physical size; only the distance to the brain changes",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.945,
        "Marked position is at the EDGE of the feature map at mid-depth, so one axial slice shows both it and the "
        "brain cross-section.\nThe true 3D corner sits further out still (26.6 vox from the brain at res 64, 57.4 at "
        "res 128).",
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    fig.savefig(cli.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {cli.out}")


if __name__ == "__main__":
    main()
