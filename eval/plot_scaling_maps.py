#!/usr/bin/env python
"""What a scaling-rate-4 encoder sees versus a scaling-rate-2 one, drawn at true scale.

Checkpoint-free. Everything here is a property of the ARCHITECTURE and the GENERATOR,
so it says what the two configurations can possibly resolve before any weight is trained.
Point it at a run dir only to pick up that run's generator settings (and, with
--measure-rf, its actual gradient footprint).

The three figures answer three separate questions.

1. `scaling_feature_grids.png` -- what one feature position covers.
   `--vqvae-scaling-rates [4]` at --synthetic-res 64 gives a 16^3 feature map; `[2]` gives
   32^3. That is the part everyone pictures, and it is the SMALLER half of the difference.
   The larger half: every conv after the downsampling path runs at the downsampled
   resolution, so its kernel spans `scaling_rate` input voxels per tap. Doubling the
   scaling rate therefore doubles the RECEPTIVE FIELD as well as halving the grid --
   measured by walking the real `models.vqvae.Encoder`, ~50 input voxels wide at rate 4
   against ~24 at rate 2 (nb_res_layers=2). The brain is ~42 voxels across at res 64, so
   at rate 4 a single feature position already integrates the whole head and locality is
   gone; at rate 2 it covers a bit over half of it. Row 2 box-averages the input onto each
   grid, which is the most a stride-s output can localise.

2. `patch_overlay.png` -- what one PATCH token covers.
   `--patch-grid 8 8 8` pools the feature map to 8^3 regardless of scaling rate, so a patch
   is 8 input voxels on a side under BOTH settings -- the patch grid does not change the
   physical patch size, only how many feature cells get averaged into it (2^3 at rate 4,
   4^3 at rate 2). What does change is the halo: each patch's real footprint is its 8-voxel
   cell plus the receptive field spilling out either side, so adjacent patches overlap by
   2r/(cell + 2r) -- ~86% at rate 4 against ~74% at rate 2. Column 3 counts, per voxel, how
   many of the 8^3 patches actually see it. Patches this correlated are close to copies of
   each other, and a patch-level contrastive term over copies is close to a global one.

3. `ventricle_lesion.png` -- the two structures the probes fail on, at that scale.
   The ventricle is ~1.2k voxels and the lesion ~0.4k, against a ~46k-voxel brain. Rows 1-2
   show each structure in both views, the render delta from resampling its own factor
   (rendering seed held fixed, so noise cancels and the delta is signal), and that delta
   box-averaged at each rate. Row 3 puts three candidate causes side by side. What they
   say, on the flagship-identifiable generator:

     * The LESION is explained twice over. It carries the least signal of any factor
       (||delta|| 0.17 of brain_size's) and its contrast genuinely inverts between views
       -- sign agreement -0.86, identical on the raw render and after normalization, so it
       is the modality LUT and nothing else (lesion 0.40 under a 0.80 WM surround in T1,
       1.00 over 0.40 in FLAIR).
     * The VENTRICLE is explained by NEITHER. Sign agreement is +1.000 -- CSF is 0.10 in
       both LUTs, so it is the single most view-consistent structure in the generator --
       and it survives the stride better than average (88% at rate 2, 57% at rate 4,
       against cortical_thickness's 46/16). It is also the second-loudest factor. Nothing
       geometric predicts its collapse to R^2 -0.012, so the scaling rate is not where to
       look for it.
     * The other five factors read as cross-view INCONSISTENT only after normalization,
       and that is an artefact worth knowing about. `fixed_reference` centres the
       foreground on ~0.553 and then multiplies by the sample's OWN brain mask. The affine
       part cancels in any difference (verified: centred-without-mask reproduces the raw
       number exactly); the mask does not. It pins background to 0 while leaving T1's GM
       (0.50) just BELOW zero and FLAIR's GM (0.80) well above, so the brain's outer edge
       is a dark rim in one view and a bright rim in the other. Every outer-boundary factor
       inherits the flip: brain_size 0.76 raw -> -0.51 as trained, lr_asymmetry 0.82 ->
       -0.25, sulcal_widening 0.84 -> 0.14. brain_size still recovers at R^2 0.92, so this
       is not a bar to learning -- but it does mean a cross-view statistic computed on
       normalized renders cannot be read as a property of the modality LUT.

This is why row 3 plots sign agreement raw AND as-trained. `eval.view_consistency` reports
the cosine on normalized renders only, and its pre-registered positive control ("the other
five cos ~ +1") fails there for the reason above rather than for any reason to do with the
hypothesis it is testing.

Usage:
  python -m eval.plot_scaling_maps --preset flagship --identifiable-ventricle \
      --content-prior uniform --content-squash none --lesion-radius 0.14 \
      --cortex-parameterization patterned --center-local-deformations
  python -m eval.plot_scaling_maps --run-dir results/synthetic/<run> --measure-rf
  python -m eval.plot_scaling_maps --scalings 2 4 8 --figures grids
"""

from __future__ import annotations

import argparse
import logging
import os
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from eval.generator_defects import CONTENT_NAMES, build_dataset, render

logger = logging.getLogger(__name__)

# Tissue labels as written by PseudoMRIRenderer.render_structure.
LBL_CSF, LBL_WM, LBL_GM, LBL_FISSURE = 1, 2, 3, 4

# Per-view intensity LUTs, copied from render_modality. Used only for the contrast
# annotation in figure 3 -- every image on the page is a real render.
LUT = {"T1": [0.0, 0.1, 0.8, 0.5, 0.3], "FLAIR": [0.0, 0.1, 0.4, 0.8, 0.3]}
LESION_INT = {"T1": 0.4, "FLAIR": 1.0}

SCALE_COLORS = {2: "#3b7dd8", 4: "#e07a3f", 8: "#8b5fbf", 16: "#4aa564"}


# ───────────────────────────── architecture geometry ─────────────────────────────


def encoder_receptive_field(scaling_rate, nb_res_layers, hidden_channels=48, res_channels=32, norm_type="group"):
    """(trained RF, init RF, stride) in input voxels, walked off the real Encoder.

    Walked rather than derived so it cannot drift from `models.vqvae.Encoder`. Composing
    every Conv3d in forward order gives the LONGEST path through the stack, which is the
    correct max receptive field: each ReZero block's identity branch is strictly shorter
    than its two-conv branch.

    Two numbers, because ReZero's alpha starts at 0. At init the residual stack is the
    identity and only the downsampling path counts ("init"); once alpha moves off zero the
    residual convs are live ("trained"). A trained model is at the trained number, and it
    is the one every annotation in these figures uses.

    The walk reduces to a closed form, verified equal for s in {2,4,8} x L in {0..3}:

        RF = s * (4L + 5) - 2          s = scaling_rate, L = nb_res_layers

    log2(s) stride-2 kernel-4 convs contribute 3s - 2, the stride-1 3x3 that follows adds
    2s, and each ReZero block adds two more 3x3s at 2s each. Every term after the
    downsampling path is LINEAR IN s, which is why the scaling rate moves the receptive
    field at all -- doubling it doubles the physical span of every later kernel tap.
    """
    import models.vqvae as vqvae

    enc = vqvae.Encoder(
        1,
        hidden_channels,
        res_channels,
        nb_res_layers,
        scaling_rate,
        use_checkpoint=False,
        norm_type=norm_type,
    )
    residual_convs = {id(m) for blk in enc.modules() if type(blk).__name__ == "ReZero" for m in blk.modules()}

    rf, rf_init, jump = 1, 1, 1
    for m in enc.modules():
        if not isinstance(m, nn.Conv3d):
            continue
        k, s = m.kernel_size[0], m.stride[0]
        rf += (k - 1) * jump
        if id(m) not in residual_convs:
            rf_init = rf
        jump *= s
    return rf, rf_init, jump


def measured_rf_radius(model, res, level, scaling_rate, device, frac=0.9):
    """Radius holding `frac` of |d feature / d input| for a CENTRE feature position.

    The analytic number above is the support of the convolution stack -- a hard bound that
    a trained model rarely fills. This is the effective one. Same estimator as
    `eval.plot_receptive_field`, reduced to the single number these figures annotate.
    """
    g = res // scaling_rate
    grid = (g, g, g)
    flat = (g // 2) * g * g + (g // 2) * g + (g // 2)
    x = torch.zeros(1, 1, res, res, res, device=device, requires_grad=True)
    out = model(x, pool_only=True, n_views=1, patch_grid=grid)
    feats = out[2] if isinstance(out, tuple) else out
    feats[level][0, :, flat].sum().backward()
    gr = x.grad.detach().abs()[0, 0].cpu().numpy()

    src = np.array([(g // 2 + 0.5) * scaling_rate] * 3)
    zz, yy, xx = np.meshgrid(*[np.arange(res)] * 3, indexing="ij")
    d = np.sqrt((zz - src[0]) ** 2 + (yy - src[1]) ** 2 + (xx - src[2]) ** 2).ravel()
    w = gr.ravel()
    o = np.argsort(d)
    c = np.cumsum(w[o])
    if c[-1] <= 0:
        return 0.0
    return float(d[o][np.searchsorted(c, frac * c[-1])])


def patch_overlap_fraction(rf, patch_voxels):
    """Fraction of a patch's footprint shared with its neighbour, as quoted in config.py.

    2r / (cell + 2r) with r the receptive-field radius: the halo is what two adjacent
    patches have in common, and it does not shrink when the patch grid gets finer.
    """
    r = (rf - 1) / 2.0
    return 2 * r / (patch_voxels + 2 * r)


def print_overlap_sweep(res, brain_diameter, scalings, res_layers, patch_ns, hidden, norm_type):
    """The three levers on patch overlap, priced against each other.

    There are only two ways to move 2r/(cell + 2r): shrink the receptive field, or grow
    the cell. `--patch-grid` is the cell lever and it runs BACKWARDS from intuition -- a
    FINER grid makes overlap worse, because the halo is fixed and the cell it is compared
    against gets smaller. The receptive field lever is `--vqvae-nb-res-layers` and
    `--vqvae-scaling-rates`, both of which change the architecture.

    The last column is the honest capacity number: how many non-overlapping receptive
    fields fit across the brain, (brain / RF)^3. Below 1 the receptive field is wider than
    the head and no configuration of the patch grid buys back any locality at all.
    """
    print(f"\npatch overlap = (RF-1) / (cell + RF-1),  cell = {res}/n_patch input voxels")
    print(f"RF = scaling * (4*nb_res_layers + 5) - 2   |   brain {brain_diameter:.0f} voxels across\n")
    head = f"{'scaling':>7} {'res_lyr':>7} {'RF':>5} |"
    for n in patch_ns:
        head += f"{'n=' + str(n) + ' (cell ' + str(res // n) + ')':>18}"
    print(head + f"{'indep. RFs':>12}")
    for s in scalings:
        for lay in res_layers:
            rf, _, _ = encoder_receptive_field(s, lay, hidden, norm_type=norm_type)
            row = f"{s:>7} {lay:>7} {rf:>5} |"
            for n in patch_ns:
                row += f"{100 * patch_overlap_fraction(rf, res / n):>17.0f}%"
            print(row + f"{(brain_diameter / rf) ** 3:>12.2f}")
    print(
        "\n  A finer patch grid does NOT reduce overlap -- it raises it. Only a smaller\n"
        "  receptive field or a coarser grid does, and they trade different things:\n"
        "  fewer res layers keeps the token count and makes each token local, a coarser\n"
        "  grid keeps the architecture and buys independence with spatial resolution.\n"
    )


def patch_rf_counts(res, scaling_rate, rf, n_patch):
    """Per-axis count of how many of `n_patch` patches have voxel i inside their footprint.

    Separable, so the 3-D count at (i, j, k) is the product of the three 1-D counts. The
    patch grid is applied to the feature map, so patch p pools feature cells
    [p*g/n_patch, (p+1)*g/n_patch) and its footprint is that span dilated by the halo.
    """
    g = res // scaling_rate
    edges = np.round(np.linspace(0, g, n_patch + 1)).astype(int)
    halo = (rf - scaling_rate) / 2.0
    counts = np.zeros(res)
    spans = []
    for p in range(n_patch):
        a, b = edges[p] * scaling_rate, edges[p + 1] * scaling_rate
        lo, hi = a - halo, b + halo
        spans.append((a, b, lo, hi))
        counts[max(int(np.floor(lo)), 0) : min(int(np.ceil(hi)), res)] += 1
    return counts, spans


# ───────────────────────────────── generator side ─────────────────────────────────


def structure_masks(inner, lat):
    """(tissue_map, lesion_load) for one sample, as numpy."""
    tissue, lesion = inner.renderer.render_structure(
        lat["z_content"],
        lat["z_deformation"],
        lat["z_fissure"],
        device=torch.device("cpu"),
        clean=inner.clean_content,
        z_lesion=lat.get("z_lesion"),
    )
    return tissue.numpy(), lesion.numpy()


def pick_slice(tissue, lesion, mode="auto"):
    """Index along the last axis of the plane to draw.

    "auto" maximises ventricle + 3*lesion voxels in the plane, so one slice shows both
    structures where the sample allows it. The ventricle is split by |x| > 0.05, and the
    plane spans x, so both of its lobes stay visible.
    """
    vent = (tissue == LBL_CSF).sum(axis=(0, 1))
    les = (lesion > 0.5).sum(axis=(0, 1))
    if mode == "ventricle":
        return int(np.argmax(vent))
    if mode == "lesion":
        return int(np.argmax(les)) if les.max() > 0 else int(np.argmax(vent))
    return int(np.argmax(vent + 3.0 * les))


def factor_delta(ds, inner, lat, seed, dim, delta):
    """Render delta in both views from a +/-delta finite difference on one content dim.

    The rendering seed is held fixed across the two renders, so the Rician noise and the
    bias field cancel and what is left is signal. Same trick as
    `generator_defects._sensitivity` and `eval.view_consistency`.
    """
    zp, zm = lat["z_content"].clone(), lat["z_content"].clone()
    zp[dim] += delta
    zm[dim] -= delta
    out = []
    for norm in (True, False):
        p1, p2, _ = render(ds, lat, seed, normalize=norm, z_content=zp)
        m1, m2, _ = render(ds, lat, seed, normalize=norm, z_content=zm)
        out.append(((p1 - m1)[0].numpy(), (p2 - m2)[0].numpy()))
    return out[0], out[1]


def sign_agreement(d1, d2):
    """Energy-weighted share of locations where the two views agree in SIGN.

    sum_v d1_v d2_v / sum_v |d1_v d2_v|. Same numerator as the cosine, but normalised by
    the L1 of the elementwise product rather than by the two L2 norms, which makes it
    read only the thing that matters for a view-shared code: +1 iff the two views push the
    same direction wherever they both write, -1 iff they push opposite everywhere.

    Preferred over the cosine here because the cosine conflates a genuine inversion with
    an amplitude mismatch and with the two views simply writing in DIFFERENT places -- and
    under `fixed_reference` the views do write in different places (see the note in
    `build_context`), so the cosine reads several perfectly learnable factors as
    "unrelated". This statistic does not.
    """
    p = d1.ravel() * d2.ravel()
    return float(p.sum() / max(np.abs(p).sum(), 1e-12))


def pool_survival(delta, scaling_rate):
    """Share of `delta`'s energy that is still resolvable on a stride-s grid.

    Box-averaging onto the output grid is (up to a constant) the orthogonal projection
    onto block-constant fields, so ||P d||^2 / ||d||^2 is exactly the fraction of the
    signature that survives it: s^3 * sum_b mean_b^2 over sum_v d_v^2.

    This is a statement about spatial frequency against the output stride, NOT a bound on
    the encoder -- the convolutions ahead of the stride can build a local nonlinearity
    (|contrast|, an edge response) that box-averaging has no way to represent. Read it as
    "how much of this factor is written above the grid's Nyquist", which is where a signed
    boundary shell and a bulk displacement part company.
    """
    t = torch.from_numpy(delta)[None, None].float()
    pooled = F.avg_pool3d(t, scaling_rate)
    num = (scaling_rate**3) * float((pooled**2).sum())
    den = float((t**2).sum())
    return num / max(den, 1e-12)


def box_average(vol, s):
    """Box-average onto the stride-s grid and put it back at input resolution."""
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None].float()
    return F.interpolate(F.avg_pool3d(t, s), scale_factor=s, mode="nearest")[0, 0].numpy()


# ─────────────────────────────────── drawing ───────────────────────────────────


def _grid_lines(ax, res, step, color, lw=0.35, alpha=0.75, offset=(0, 0)):
    """Cell boundaries every `step` voxels, in the axes' own coordinates.

    `offset` is the crop's origin in volume voxels, so the lines keep the phase of the
    real feature grid rather than restarting at the corner of a zoomed panel.
    """
    for axis, off in zip(("h", "v"), (offset[1], offset[0])):
        for t in range(-(off % step), res + 1, step):
            (ax.axhline if axis == "h" else ax.axvline)(t - 0.5, color=color, lw=lw, alpha=alpha)


def _headroom(fig, main, sub, fontsize=13):
    """Suptitle + explanatory subtitle at fixed INCH offsets from the top.

    Figure-fraction offsets do not work here: these figures grow a row per scaling rate,
    so a fraction that clears the suptitle at two rows overlaps it at four.
    Returns the rect top to hand to tight_layout.
    """
    h = fig.get_size_inches()[1]
    fig.suptitle(main, fontsize=fontsize, y=1 - 0.30 / h)
    fig.text(1 - 0.5, 1 - 0.62 / h, sub, ha="center", va="top", fontsize=9, linespacing=1.5)
    return 1 - (0.62 + 0.20 * (sub.count("\n") + 1)) / h


def _outline(ax, mask2d, color, lw=1.1):
    if mask2d.any():
        ax.contour(mask2d.T.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _show(ax, img, cmap="gray", **kw):
    """imshow with the volume's (x, y) order kept: x up the page, y across."""
    return ax.imshow(img.T, cmap=cmap, origin="lower", interpolation="nearest", **kw)


def fig_grids(cli, ctx, out_path):
    """Feature grid, receptive field and box-average, one row per scaling rate."""
    res, k = ctx["res"], ctx["slice"]
    x1 = ctx["x1"]
    vent2d, les2d, brain2d = ctx["vent"][:, :, k], ctx["les"][:, :, k], ctx["brain"][:, :, k]
    brain_diam = ctx["brain_diameter"]

    scalings = cli.scalings
    fig, axes = plt.subplots(len(scalings), 3, figsize=(13.2, 4.35 * len(scalings)), squeeze=False)

    for row, s in enumerate(scalings):
        rf, rf_init, _ = ctx["rf"][s]
        g = res // s
        color = SCALE_COLORS.get(s, "#c04040")

        # A: the grid itself, at true scale, with one cell picked out.
        ax = axes[row][0]
        _show(ax, x1[:, :, k], vmin=ctx["vlo"], vmax=ctx["vhi"])
        _grid_lines(ax, res, s, color)
        _outline(ax, vent2d, "#35d0e0")
        _outline(ax, les2d, "#ffd166")
        cx, cy = (res // 2 // s) * s, (res // 2 // s) * s
        # White, not `color`: the grid lines are already `color`, and at scaling 2 a
        # same-coloured highlight disappears into them.
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), s, s, fill=False, ec="w", lw=2.2))
        ax.set_title(f"scaling {s}  ->  feature map {g}$^3$\ncell = {s}$^3$ = {s**3} input voxels", fontsize=10)
        ax.set_ylabel(f"scaling rate {s}", fontsize=11, color=color, labelpad=8)

        # B: what that grid can localise, i.e. the input box-averaged onto it.
        ax = axes[row][1]
        _show(ax, box_average(x1, s)[:, :, k], vmin=ctx["vlo"], vmax=ctx["vhi"])
        _outline(ax, vent2d, "#35d0e0")
        _outline(ax, les2d, "#ffd166")
        n_vent = ctx["vent_cells"][s]
        n_les = ctx["les_cells"][s]
        ax.set_title(
            f"input box-averaged onto the {g}$^3$ grid\n" f"ventricle touches {n_vent:.0f} cells, lesion {n_les:.0f}",
            fontsize=10,
        )

        # C: the same cell's receptive field, drawn at true radius over the brain.
        ax = axes[row][2]
        _show(ax, x1[:, :, k], vmin=ctx["vlo"], vmax=ctx["vhi"])
        halo = (rf - s) / 2.0
        ax.add_patch(
            Rectangle(
                (cx - halo - 0.5, cy - halo - 0.5),
                s + 2 * halo,
                s + 2 * halo,
                fill=True,
                fc=color,
                alpha=0.22,
                ec=color,
                lw=1.8,
            )
        )
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), s, s, fill=False, ec="w", lw=1.6))
        _outline(ax, brain2d, "#35d0e0", lw=1.0)
        cover = min(rf / brain_diam, 99.0)
        verdict = "covers the whole brain" if rf >= brain_diam else f"covers {100*cover:.0f}% of it"
        ax.set_title(
            f"receptive field of that one cell: {rf} voxels wide\n" f"brain is {brain_diam:.0f} across -- {verdict}",
            fontsize=10,
        )
        ax.set_xlabel(f"(at init, alpha=0, the stack is only {rf_init} wide)", fontsize=8.5, color="#666666")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, res - 0.5)
        ax.set_ylim(-0.5, res - 0.5)

    top = _headroom(
        fig,
        "Doubling the scaling rate halves the grid AND doubles the receptive field",
        "Every conv after the downsampling path runs at the downsampled resolution, so each of its taps spans "
        "`scaling_rate` input voxels.\n"
        "cyan = ventricle,  yellow = lesion,  white square = one feature cell,  shaded = what that cell integrates.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_patches(cli, ctx, out_path):
    """The patch grid, one patch's real footprint, and how many patches see each voxel."""
    res, k, n_patch = ctx["res"], ctx["slice"], ctx["patch_grid"][0]
    x1 = ctx["x1"]
    patch_vox = res / n_patch
    keep2d = ctx["patch_keep"]

    scalings = cli.scalings
    fig, axes = plt.subplots(len(scalings), 3, figsize=(13.2, 4.35 * len(scalings)), squeeze=False)

    # One colour scale across every row of column 3. Per-row autoscaling would paint 64
    # patches and 252 patches the same brightness, hiding the entire comparison.
    counts_by_s = {s: patch_rf_counts(res, s, ctx["rf"][s][0], n_patch)[0] for s in scalings}
    cmax = max(float(c.max() ** 3) for c in counts_by_s.values())

    for row, s in enumerate(scalings):
        rf, _, _ = ctx["rf"][s]
        g = res // s
        color = SCALE_COLORS.get(s, "#c04040")
        cells_per_patch = g / n_patch

        # A: the patch grid, with the foreground-kept patches shaded.
        ax = axes[row][0]
        _show(ax, x1[:, :, k], vmin=ctx["vlo"], vmax=ctx["vhi"])
        edges = np.round(np.linspace(0, res, n_patch + 1)).astype(int)
        for i in range(n_patch):
            for j in range(n_patch):
                if keep2d[i, j]:
                    ax.add_patch(
                        Rectangle(
                            (edges[i] - 0.5, edges[j] - 0.5),
                            edges[i + 1] - edges[i],
                            edges[j + 1] - edges[j],
                            fill=True,
                            fc="#35d0e0",
                            alpha=0.13,
                            ec="none",
                        )
                    )
        _grid_lines(ax, res, int(patch_vox), "#dddddd", lw=0.6, alpha=0.9)
        _outline(ax, ctx["vent"][:, :, k], "#35d0e0")
        _outline(ax, ctx["les"][:, :, k], "#ffd166")
        ax.set_title(
            f"patch grid {n_patch}$^3$ -- patch = {patch_vox:.0f}$^3$ input voxels\n"
            f"= {cells_per_patch:.0f}$^3$ feature cells at scaling {s};  "
            f"{int(ctx['patch_keep_n'])}/{n_patch**3} kept as foreground",
            fontsize=10,
        )
        ax.set_ylabel(f"scaling rate {s}", fontsize=11, color=color, labelpad=8)

        # B: one patch and the halo its receptive field adds.
        ax = axes[row][1]
        _show(ax, x1[:, :, k], vmin=ctx["vlo"], vmax=ctx["vhi"])
        pi = n_patch // 2
        a, b = edges[pi], edges[pi + 1]
        halo = (rf - s) / 2.0
        ax.add_patch(
            Rectangle(
                (a - halo - 0.5, a - halo - 0.5),
                (b - a) + 2 * halo,
                (b - a) + 2 * halo,
                fill=True,
                fc=color,
                alpha=0.22,
                ec=color,
                lw=1.8,
            )
        )
        ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec="w", lw=2.0))
        ov = patch_overlap_fraction(rf, patch_vox)
        ax.set_title(
            f"one patch: {patch_vox:.0f} voxels of cell, "
            f"{2*halo:.0f} of halo\nneighbouring patches overlap by {100*ov:.0f}%",
            fontsize=10,
        )

        # C: per-voxel count of patches whose footprint contains it.
        ax = axes[row][2]
        counts1d = counts_by_s[s]
        cmap2d = np.outer(counts1d, counts1d) * counts1d[k]
        im = _show(ax, cmap2d, cmap="magma", vmin=0, vmax=cmax)
        _outline(ax, ctx["brain"][:, :, k], "#35d0e0", lw=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, label="patches seeing this voxel")
        inside = cmap2d[ctx["brain"][:, :, k]]
        med = float(np.median(inside)) if inside.size else 0.0
        ax.set_title(
            f"how many of the {n_patch**3} patches see each voxel\n"
            f"median inside the brain: {med:.0f}  ({100*med/n_patch**3:.0f}% of them)",
            fontsize=10,
        )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, res - 0.5)
        ax.set_ylim(-0.5, res - 0.5)

    top = _headroom(
        fig,
        "The patch grid fixes the patch SIZE; the scaling rate fixes how much patches overlap",
        f"--patch-grid {n_patch} {n_patch} {n_patch} pools the feature map to {n_patch}$^3$ whatever the scaling "
        f"rate, so a patch is {patch_vox:.0f} input voxels on a side in both rows.\n"
        "Only the halo differs -- and the halo is what makes two adjacent patch tokens near-copies of each other.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _zoom(img, centre, half):
    """Crop a 2*half window around `centre`, sliding it to stay inside the array.

    Slides rather than truncates, so a structure near the edge still gets a full-size
    panel and every crop in the figure is at the same magnification.
    """
    (a, b), n = centre, img.shape
    a0 = int(np.clip(int(a) - half, 0, max(n[0] - 2 * half, 0)))
    b0 = int(np.clip(int(b) - half, 0, max(n[1] - 2 * half, 0)))
    return img[a0 : a0 + 2 * half, b0 : b0 + 2 * half], (a0, b0)


def fig_structures(cli, ctx, out_path):
    """The ventricle and the lesion at feature-cell scale, plus the two failure numbers."""
    res = ctx["res"]
    scalings = cli.scalings
    half = cli.zoom_half

    # Rows 1-2 hold equal-aspect images, so their height is set by the column width and
    # cannot be stretched; the figure height and ratios are tuned to that, not the reverse.
    fig = plt.figure(figsize=(17.0, 11.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.12], hspace=0.28, wspace=0.42, top=0.87, bottom=0.09)

    for row, (name, dim, cmask, ccolor) in enumerate(
        [("ventricle", 1, ctx["vent"], "#35d0e0"), ("lesion", 3, ctx["les"], "#ffd166")]
    ):
        cen3 = ctx[f"{name}_centre"]
        kk = int(round(cen3[2]))
        cen = (cen3[0], cen3[1])
        d1 = ctx["deltas"][dim][0]
        m2 = cmask[:, :, kk]

        # A/B: the structure as each view renders it.
        for col, (view, vol) in enumerate([("T1", ctx["x1"]), ("FLAIR", ctx["x2"])]):
            ax = fig.add_subplot(gs[row, col])
            crop, off = _zoom(vol[:, :, kk], cen, half)
            _show(ax, crop, vmin=ctx["vlo"], vmax=ctx["vhi"])
            mc, _ = _zoom(m2, cen, half)
            _outline(ax, mc, ccolor)
            # Only the coarsest grid is drawn here: at this magnification the finer one
            # is a mesh, and the point of the panel is the structure against ONE cell.
            _grid_lines(
                ax, 2 * half, max(scalings), SCALE_COLORS.get(max(scalings), "#c04040"), lw=0.7, alpha=0.7, offset=off
            )
            ax.set_title(f"{name} in {view}", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"{name}\n{int(cmask.sum())} voxels  ({100*cmask.sum()/max(ctx['brain'].sum(),1):.1f}% of brain)",
                    fontsize=10,
                    color=ccolor,
                )

        # C: the signal -- the render delta from moving this factor.
        ax = fig.add_subplot(gs[row, 2])
        crop, off = _zoom(d1[:, :, kk], cen, half)
        lim = float(np.abs(crop).max()) or 1.0
        _show(ax, crop, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _grid_lines(ax, 2 * half, max(scalings), "#999999", lw=0.6, alpha=0.6, offset=off)
        ax.set_title(f"T1 delta from resampling\n{CONTENT_NAMES[dim]}  (+/-{cli.delta})", fontsize=10)

        # D: the same delta after the coarsest grid has averaged it.
        s_hi = max(scalings)
        ax = fig.add_subplot(gs[row, 3])
        crop, _ = _zoom(box_average(d1, s_hi)[:, :, kk], cen, half)
        _show(ax, crop, cmap="RdBu_r", vmin=-lim, vmax=lim)
        txt = "  ".join(f"s={s}: {100*ctx['survival'][dim][s]:.0f}%" for s in scalings)
        ax.set_title(f"box-averaged at s={s_hi}\nenergy surviving -- {txt}", fontsize=10)

    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 2 * half - 0.5)
        ax.set_ylim(-0.5, 2 * half - 0.5)

    # E: intensity profile through the lesion in both views -- the sign flip, drawn.
    ax = fig.add_subplot(gs[2, 0])
    lc = ctx["lesion_centre"]
    kk, jj = int(round(lc[2])), int(round(lc[1]))
    xs = np.arange(res)
    ax.plot(xs, ctx["x1"][:, jj, kk], color="#3b7dd8", lw=1.7, label="T1")
    ax.plot(xs, ctx["x2"][:, jj, kk], color="#e07a3f", lw=1.7, label="FLAIR")
    lo, hi = ctx["lesion_span_x"]
    ax.axvspan(lo, hi, color="#ffd166", alpha=0.35, lw=0)
    ax.annotate("lesion", (0.5 * (lo + hi), ax.get_ylim()[1]), ha="center", va="top", fontsize=8.5, color="#8a6d1f")
    ax.set_xlabel("x (input voxels)", fontsize=9)
    ax.set_ylabel("normalized intensity", fontsize=9)
    ax.set_title("the lesion is a DIP in T1 and a PEAK in FLAIR", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # F: where each structure's contrast sits, raw and as the encoder receives it.
    ax = fig.add_subplot(gs[2, 1])
    ref = ctx["ref"]
    rows = [
        ("lesion", LESION_INT["T1"], LESION_INT["FLAIR"], LUT["T1"][LBL_WM], LUT["FLAIR"][LBL_WM]),
        ("ventricle (CSF)", LUT["T1"][LBL_CSF], LUT["FLAIR"][LBL_CSF], LUT["T1"][LBL_WM], LUT["FLAIR"][LBL_WM]),
        ("brain edge (GM)", LUT["T1"][LBL_GM], LUT["FLAIR"][LBL_GM], 0.0, 0.0),
    ]
    if ref:
        # The brain edge is measured against the MASKED background, which normalization
        # pins to 0 rather than carrying through the affine map. That asymmetry is the
        # whole point of the panel, so the surround is transformed accordingly.
        m, sc = ref
        rows = [
            (n, (a - m) / sc, (b - m) / sc, ((c - m) / sc if c else 0.0), ((e - m) / sc if e else 0.0))
            for n, a, b, c, e in rows
        ]
    y = np.arange(len(rows))
    ax.barh(y - 0.19, [r[1] - r[3] for r in rows], 0.36, color="#3b7dd8", label="T1")
    ax.barh(y + 0.19, [r[2] - r[4] for r in rows], 0.36, color="#e07a3f", label="FLAIR")
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("intensity minus surround", fontsize=9)
    tail = f"\nafter {ctx['norm_mode']} (mean {ref[0]:.2f})" if ref else ""
    ax.set_title(f"contrast against the surround{tail}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    # G: how much signal each factor writes that the stride can still resolve.
    # Relative survival alone does not rank the factors -- the lesion survives the stride
    # as well as brain_size and still carries 6x less signal to survive.
    ax = fig.add_subplot(gs[2, 2])
    dims = list(range(len(CONTENT_NAMES)))
    y = np.arange(len(dims))
    w = 0.8 / len(scalings)
    for i, s in enumerate(scalings):
        ax.barh(
            y + (i - (len(scalings) - 1) / 2) * w,
            [ctx["resolvable"][d][s] for d in dims],
            w,
            color=SCALE_COLORS.get(s, "#c04040"),
            label=f"scaling {s}",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([CONTENT_NAMES[d] for d in dims], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\|\Delta\| \cdot \sqrt{survival}$, rel. to loudest", fontsize=9)
    ax.set_title("resolvable signal:\nsize x what the stride keeps", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    # H: sign agreement, raw against as-trained. The gap between the two bars is the
    # normalization's doing, not the modality LUT's -- only a factor inverted in BOTH is
    # inverted by the generator.
    ax = fig.add_subplot(gs[2, 3])
    ax.barh(y - 0.19, [ctx["sign_raw"][d] for d in dims], 0.36, color="#9bb7d4", label="raw render")
    ax.barh(
        y + 0.19,
        [ctx["sign_norm"][d] for d in dims],
        0.36,
        color=[
            ("#c0392b" if ctx["sign_norm"][d] < -0.3 else "#d9a441" if ctx["sign_norm"][d] < 0.2 else "#4aa564")
            for d in dims
        ],
        label=f"as trained ({ctx['norm_mode']})",
    )
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([CONTENT_NAMES[d] for d in dims], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(r"sign agreement ($\Delta$T1, $\Delta$FLAIR)", fontsize=9)
    ax.set_title("does the evidence agree across views?\nlesion: no, in BOTH rows", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.grid(axis="x", alpha=0.25)

    _headroom(
        fig,
        "The lesion is explained twice over; the ventricle is explained by neither",
        f"Rows 1-2 are real renders; the grid is the scaling-{max(scalings)} feature grid. "
        "The lesion is the smallest signal AND genuinely inverted between views.\n"
        "The ventricle is neither -- it agrees perfectly across views and survives the stride "
        "better than most, so its failure is not geometric.",
    )
    fig.savefig(out_path, dpi=cli.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────── driver ───────────────────────────────────


def generator_args(cli):
    """The generator settings, from --run-dir's settings.json if given, else the CLI."""
    if cli.run_dir:
        from eval.run_dci_synthetic import load_run_args

        a = load_run_args(cli.run_dir)
        return (
            types.SimpleNamespace(
                res=getattr(a, "synthetic_res", 64),
                n_content=getattr(a, "synthetic_n_content", 9),
                n_samples=cli.n_samples,
                seed=getattr(a, "synthetic_seed", 42),
                normalize=getattr(a, "synthetic_normalize", "per_sample"),
                causal=bool(getattr(a, "synthetic_causal", False)),
                clean_content=bool(getattr(a, "synthetic_clean_content", False)),
                lesion_mode=getattr(a, "synthetic_lesion_mode", "sphere"),
                identifiable_ventricle=bool(getattr(a, "synthetic_identifiable_ventricle", False)),
                content_scale=getattr(a, "synthetic_content_scale", 1.0),
                style_scale=getattr(a, "synthetic_style_scale", 1.0),
                content_prior=getattr(a, "synthetic_content_prior", "normal"),
                content_squash=getattr(a, "synthetic_content_squash", "auto"),
                content_amp_scale=getattr(a, "synthetic_content_amp_scale", None),
                lesion_radius=getattr(a, "synthetic_lesion_radius", 0.1),
                cortex_parameterization=getattr(a, "synthetic_cortex_parameterization", "additive"),
                center_local_deformations=bool(getattr(a, "synthetic_center_local_deformations", False)),
            ),
            a,
        )
    if cli.preset == "flagship":
        cli.causal, cli.clean_content, cli.normalize = True, True, "fixed_reference"
    return (
        types.SimpleNamespace(
            res=cli.res,
            n_content=cli.n_content,
            n_samples=cli.n_samples,
            seed=cli.seed,
            normalize=cli.normalize,
            causal=cli.causal,
            clean_content=cli.clean_content,
            lesion_mode=cli.lesion_mode,
            identifiable_ventricle=cli.identifiable_ventricle,
            content_scale=cli.content_scale,
            style_scale=cli.style_scale,
            content_prior=cli.content_prior,
            content_squash=cli.content_squash,
            content_amp_scale=cli.content_amp_scale,
            lesion_radius=cli.lesion_radius,
            cortex_parameterization=cli.cortex_parameterization,
            center_local_deformations=cli.center_local_deformations,
        ),
        None,
    )


def arch_params(cli, run_args):
    """(nb_res_layers, hidden_channels, norm_type) -- the run's if there is one, else the CLI's."""
    if run_args is None:
        return cli.nb_res_layers, cli.hidden_channels, "group"
    return (
        getattr(run_args, "vqvae_nb_res_layers", 2),
        getattr(run_args, "vqvae_hidden_channels", 48),
        getattr(run_args, "norm_type", "group"),
    )


def build_context(cli, gargs, run_args):
    """Everything the three figures share: one rendered sample plus the measured tables."""
    ds = build_dataset(gargs)
    inner = ds._inner
    res = gargs.res

    lat = dict(inner[cli.sample][2])
    if not cli.independent_style:
        # Matching the two views' style isolates the modality LUT: with independent draws
        # a low cross-view cosine could be the style difference rather than the LUT.
        lat["z_style_v2"] = lat["z_style_v1"].clone()
    seed = inner.sample_seed_for(cli.sample)
    x1, x2, _ = render(ds, lat, seed)
    x1, x2 = x1[0].numpy(), x2[0].numpy()

    tissue, lesion = structure_masks(inner, lat)
    vent = tissue == LBL_CSF
    les = lesion > 0.5
    brain = tissue > 0
    if not les.any():
        logger.warning("sample %d renders no lesion; try another --sample", cli.sample)

    k = pick_slice(tissue, lesion, cli.slice_mode)
    idx = np.argwhere(brain)
    brain_diameter = float(np.ptp(idx, axis=0).max() + 1)

    ctx = {
        "res": res,
        "slice": k,
        "x1": x1,
        "x2": x2,
        "vent": vent,
        "les": les,
        "brain": brain,
        "brain_diameter": brain_diameter,
        "vlo": float(np.percentile(x1[brain], 1)),
        "vhi": float(np.percentile(x1[brain], 99)),
        "patch_grid": tuple(cli.patch_grid),
        "ventricle_centre": np.argwhere(vent).mean(0) if vent.any() else np.array([res / 2] * 3),
        "lesion_centre": np.argwhere(les).mean(0) if les.any() else np.array([res / 2] * 3),
    }
    lx = np.argwhere(les)[:, 0] if les.any() else np.array([res // 2])
    ctx["lesion_span_x"] = (float(lx.min()), float(lx.max()) + 1)

    # Architecture geometry, per scaling rate.
    nb_res, hidden, norm = arch_params(cli, run_args)
    ctx["rf"] = {s: encoder_receptive_field(s, nb_res, hidden, norm_type=norm) for s in cli.scalings}
    ctx["vent_cells"] = {s: box_average(vent.astype(np.float32), s).astype(bool).sum() / s**3 for s in cli.scalings}
    ctx["les_cells"] = {s: box_average(les.astype(np.float32), s).astype(bool).sum() / s**3 for s in cli.scalings}

    # Foreground patch keep-mask, computed exactly as the training loop does.
    frac = F.adaptive_avg_pool3d(torch.from_numpy(brain.astype(np.float32))[None, None], ctx["patch_grid"])[0, 0]
    keep = (frac >= cli.patch_foreground_thresh).numpy()
    ctx["patch_keep_n"] = int(keep.sum())
    kk = int(round(k / res * ctx["patch_grid"][2]))
    ctx["patch_keep"] = keep[:, :, min(kk, ctx["patch_grid"][2] - 1)]

    # The LUT as the encoder actually receives it. `fixed_reference` maps every voxel
    # through (x - mean)/scale and then multiplies by the sample's OWN brain mask, so the
    # background is pinned to 0 while the tissue is centred on `mean`. The affine part
    # cancels in any difference; the mask does not, and it is what puts the brain's outer
    # boundary at polarity sign(tissue - mean). With mean ~0.553 that lands T1's GM (0.50)
    # just BELOW zero and FLAIR's GM (0.80) well above -- so the outer edge is a dark rim
    # in one view and a bright rim in the other, with no help from the modality LUT. Every
    # outer-boundary factor inherits that flip. Measured (sign agreement, this generator):
    # brain_size 0.76 raw -> -0.51 as trained, lr_asymmetry 0.82 -> -0.25, sulcal_widening
    # 0.84 -> 0.14. The two factors that never touch the outer boundary are unmoved:
    # ventricle_size 1.00 in both, lesion_* ~-0.7 in both.
    ctx["norm_mode"] = gargs.normalize
    ctx["ref"] = None
    if gargs.normalize == "fixed_reference":
        if ds._fixed_mean is None:
            ds._compute_fixed_reference()
        ctx["ref"] = (float(ds._fixed_mean), float(ds._fixed_scale))

    # Per-factor tables. Averaged over --n-samples base draws so one unlucky latent
    # cannot set the ranking.
    logger.info("measuring per-factor energy, survival and view agreement over %d draws...", cli.n_samples)
    n_c = gargs.n_content
    surv = {d: {s: [] for s in cli.scalings} for d in range(n_c)}
    energy, cosv, sgn_n, sgn_r = ({d: [] for d in range(n_c)} for _ in range(4))
    ctx["deltas"] = {}
    draws = sorted(set(range(cli.n_samples)) | {cli.sample})
    for i in draws:
        li = dict(inner[i][2])
        if not cli.independent_style:
            li["z_style_v2"] = li["z_style_v1"].clone()
        si = inner.sample_seed_for(i)
        for d in range(n_c):
            (d1, d2), (r1, r2) = factor_delta(ds, inner, li, si, d, cli.delta)
            if i == cli.sample:
                ctx["deltas"][d] = (d1, d2)
            if i >= cli.n_samples:
                continue
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 <= 1e-9:
                continue
            energy[d].append(float(n1))
            for s in cli.scalings:
                surv[d][s].append(pool_survival(d1, s))
            if n2 > 1e-9:
                cosv[d].append(float((d1 * d2).sum() / (n1 * n2)))
                sgn_n[d].append(sign_agreement(d1, d2))
                sgn_r[d].append(sign_agreement(r1, r2))

    _m = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    ctx["survival"] = {d: {s: (_m(v) if v else 0.0) for s, v in ss.items()} for d, ss in surv.items()}
    ctx["cos_view"] = {d: _m(v) for d, v in cosv.items()}
    ctx["sign_norm"] = {d: _m(v) for d, v in sgn_n.items()}
    ctx["sign_raw"] = {d: _m(v) for d, v in sgn_r.items()}

    # Energy relative to the loudest factor, and the part of it the stride can resolve:
    # ||delta|| * sqrt(survival). Relative survival alone does not rank the factors --
    # the lesion survives the stride as well as brain_size does and still carries 6x less
    # signal to survive.
    e = {d: _m(v) for d, v in energy.items()}
    emax = max([v for v in e.values() if v == v] or [1.0])
    ctx["energy"] = {d: (e[d] / emax if e[d] == e[d] else 0.0) for d in range(n_c)}
    ctx["resolvable"] = {
        d: {s: ctx["energy"][d] * np.sqrt(ctx["survival"][d][s]) for s in cli.scalings} for d in range(n_c)
    }
    return ctx


def print_tables(cli, ctx, gargs):
    res, n_patch = ctx["res"], ctx["patch_grid"][0]
    print(
        f"\nsample {cli.sample} at {res}^3   brain {int(ctx['brain'].sum())} voxels, {ctx['brain_diameter']:.0f} across"
    )
    print(f"ventricle {int(ctx['vent'].sum())} voxels   lesion {int(ctx['les'].sum())} voxels\n")

    print(
        f"{'scaling':>8} {'grid':>8} {'cell':>7} {'RF (trained)':>13} {'RF/brain':>9} "
        f"{'patch vox':>10} {'overlap':>8} {'RF 90% mass':>12}"
    )
    for s in cli.scalings:
        rf, rf_init, _ = ctx["rf"][s]
        pv = res / n_patch
        meas = ctx.get("rf_measured", {}).get(s)
        meas_s = f"{2*meas:.0f}" if meas else "--"
        print(
            f"{s:>8} {res//s:>6}^3 {s:>5}^3 {rf:>10} vox {rf/ctx['brain_diameter']:>8.2f} "
            f"{pv:>8.0f}^3 {100*patch_overlap_fraction(rf, pv):>7.0f}% {meas_s:>12}"
        )
    print("\n  RF (trained) walks every conv in models.vqvae.Encoder with the ReZero branches live.")
    print("  'RF 90% mass' is the measured effective diameter, only present with --measure-rf.\n")

    hdr = (
        f"{'dim':>3} {'factor':<20} {'energy':>7} "
        + " ".join(f"{'surv s='+str(s):>10}" for s in cli.scalings)
        + " "
        + " ".join(f"{'resolv s='+str(s):>12}" for s in cli.scalings)
        + f" {'sign raw':>9} {'sign trained':>13}"
    )
    print(hdr)
    for d in range(gargs.n_content):
        name = CONTENT_NAMES[d] if d < len(CONTENT_NAMES) else "?"
        surv = " ".join(f"{100*ctx['survival'][d][s]:>9.1f}%" for s in cli.scalings)
        res_ = " ".join(f"{ctx['resolvable'][d][s]:>12.2f}" for s in cli.scalings)
        # Only an inversion present in the RAW render is the generator's; one that appears
        # only after normalization is the masked background, and it is not a bar to
        # learning -- brain_size reads -0.51 as trained and still recovers at R^2 0.92.
        # -0.3 / +0.2 are `eval.view_consistency`'s own inverts / unrelated cut-offs.
        flag = ""
        if ctx["sign_raw"][d] < -0.3:
            flag = "  <- INVERTS between views in the generator itself"
        elif ctx["sign_raw"][d] < 0.2:
            flag = "  <- views carry unrelated evidence, before any normalization"
        elif ctx["sign_norm"][d] < -0.3:
            flag = "  <- inverted only by the masked normalization; the LUT agrees"
        elif ctx["sign_norm"][d] < 0.2:
            flag = "  <- decorrelated only by the masked normalization; the LUT agrees"
        elif ctx["resolvable"][d][max(cli.scalings)] < 0.2:
            flag = "  <- little resolvable signal at the coarsest grid"
        print(
            f"{d:>3} {name:<20} {ctx['energy'][d]:>7.2f} {surv} {res_} "
            f"{ctx['sign_raw'][d]:>9.2f} {ctx['sign_norm'][d]:>13.2f}{flag}"
        )
    print(
        "\n  energy    = ||render delta|| relative to the loudest factor.\n"
        "  surv      = share of that delta still resolvable after box-averaging at stride s.\n"
        "  resolv    = energy * sqrt(surv): absolute signal the grid can carry.\n"
        "  sign      = energy-weighted agreement of the two views' deltas; 'raw' skips normalize_views.\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="scaling_maps")
    ap.add_argument("--figures", default="grids,patches,structures", help="comma-separated subset")
    ap.add_argument("--scalings", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--patch-grid", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--patch-foreground-thresh", type=float, default=0.05)
    ap.add_argument("--sample", type=int, default=0, help="which generated sample to draw")
    ap.add_argument("--slice-mode", default="auto", choices=["auto", "ventricle", "lesion"])
    ap.add_argument("--zoom-half", type=int, default=16, help="half-width of the row 1-2 crops, in voxels")
    ap.add_argument("--delta", type=float, default=0.5, help="finite-difference step on each content dim")
    ap.add_argument("--n-samples", type=int, default=4, help="latent draws averaged into the per-factor tables")
    ap.add_argument("--independent-style", action="store_true", help="do not match the two views' style vectors")
    ap.add_argument("--dpi", type=int, default=150)

    ap.add_argument(
        "--overlap-sweep",
        action="store_true",
        help="Print patch overlap across scaling rate x --vqvae-nb-res-layers x patch grid, then exit "
        "without rendering anything. Architecture-only, so it runs in a second.",
    )
    ap.add_argument("--sweep-res-layers", type=int, nargs="+", default=[2, 1, 0])
    ap.add_argument("--sweep-patch-grids", type=int, nargs="+", default=[8, 4, 2], help="patches per axis")
    ap.add_argument(
        "--brain-diameter",
        type=float,
        default=0.0,
        help="Brain extent in voxels for the sweep's capacity column. 0 measures it from a render; "
        "pass a number to skip the render entirely.",
    )

    ap.add_argument("--run-dir", default=None, help="take generator + architecture settings from this run")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--measure-rf", action="store_true", help="also measure the gradient RF (needs --run-dir)")
    ap.add_argument("--device", default=None)

    ap.add_argument("--nb-res-layers", type=int, default=2)
    ap.add_argument("--hidden-channels", type=int, default=48)

    ap.add_argument("--preset", choices=["defaults", "flagship"], default="defaults")
    ap.add_argument(
        "--res",
        type=int,
        default=64,
        help="Training resolution is 64; 32 is ~8x faster but the "
        "structures are half as many voxels across, so the survival numbers are not comparable.",
    )
    ap.add_argument("--n-content", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", default="per_sample", choices=["per_sample", "shared", "fixed_reference"])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--clean-content", action="store_true")
    ap.add_argument("--lesion-mode", default="sphere", choices=["sphere", "field"])
    ap.add_argument("--identifiable-ventricle", action="store_true")
    ap.add_argument("--content-scale", type=float, default=1.0)
    ap.add_argument("--style-scale", type=float, default=1.0)
    ap.add_argument("--content-prior", default="normal", choices=["normal", "uniform"])
    ap.add_argument("--content-squash", default="auto", choices=["auto", "clamp", "tanh", "none"])
    ap.add_argument("--content-amp-scale", type=float, nargs="+", default=None)
    ap.add_argument("--lesion-radius", type=float, default=0.1)
    ap.add_argument(
        "--cortex-parameterization", default="additive", choices=["additive", "nested", "midsurface", "patterned"]
    )
    ap.add_argument("--center-local-deformations", action="store_true")
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gargs, run_args = generator_args(cli)
    if run_args is not None:
        cli.scalings = sorted(set(cli.scalings) | set(getattr(run_args, "vqvae_scaling_rates", [])))
        pg = getattr(run_args, "patch_grid", None)
        if pg:
            cli.patch_grid = list(pg)
        cli.patch_foreground_thresh = getattr(run_args, "patch_foreground_thresh", cli.patch_foreground_thresh)
    logger.info(
        "generator: res=%d causal=%s clean=%s normalize=%s identifiable_ventricle=%s lesion_radius=%s",
        gargs.res,
        gargs.causal,
        gargs.clean_content,
        gargs.normalize,
        gargs.identifiable_ventricle,
        gargs.lesion_radius,
    )

    nb_res, hidden, norm = arch_params(cli, run_args)

    if cli.overlap_sweep and cli.brain_diameter > 0:
        # Architecture-only path: nothing here needs a render, so skip the generator.
        print_overlap_sweep(
            gargs.res, cli.brain_diameter, cli.scalings, cli.sweep_res_layers, cli.sweep_patch_grids, hidden, norm
        )
        return

    ctx = build_context(cli, gargs, run_args)
    if cli.overlap_sweep:
        print_overlap_sweep(
            gargs.res,
            ctx["brain_diameter"],
            cli.scalings,
            cli.sweep_res_layers,
            cli.sweep_patch_grids,
            hidden,
            norm,
        )

    if cli.measure_rf:
        if not cli.run_dir:
            logger.warning("--measure-rf needs --run-dir; skipping the measured receptive field")
        else:
            from eval.run_dci_synthetic import load_model_from_run_dir

            device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model, _, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
            model.eval()
            trained = list(getattr(run_args, "vqvae_scaling_rates", []))
            ctx["rf_measured"] = {}
            for s in cli.scalings:
                if s not in trained:
                    continue
                ctx["rf_measured"][s] = measured_rf_radius(model, gargs.res, trained.index(s), s, device)

    print_tables(cli, ctx, gargs)

    os.makedirs(cli.out_dir, exist_ok=True)
    wanted = [f.strip() for f in cli.figures.split(",") if f.strip()]
    builders = {
        "grids": (fig_grids, "scaling_feature_grids.png"),
        "patches": (fig_patches, "patch_overlay.png"),
        "structures": (fig_structures, "ventricle_lesion.png"),
    }
    for name in wanted:
        if name not in builders:
            logger.warning("unknown figure %r; choose from %s", name, ", ".join(builders))
            continue
        fn, fname = builders[name]
        print(f"saved -> {fn(cli, ctx, os.path.join(cli.out_dir, fname))}")


if __name__ == "__main__":
    main()
