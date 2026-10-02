#!/usr/bin/env python
"""Look at T1 and FLAIR reconstructions next to their originals -- is the ventricle there?

Picks samples spanning the ventricle_size range and shows, per view, the original, the
reconstruction, and an intensity profile straight through the ventricle.  The profile is
the decisive panel: the ventricle is a dark cavity, so it reads as a DIP, and the question
is whether the reconstruction's dip tracks the ground-truth ventricle size or stays flat.

Factors are drawn i.i.d. (`causal=False`).  That matters here for a concrete reason: under
the run's SCM ventricle_size correlates ~0.8 with brain_size, so the "large ventricle"
sample would also be the large-brain sample and you could not tell which one the
reconstruction is responding to.

Images share one grayscale window per view across original and reconstruction -- per-panel
autoscaling would hide exactly the difference being looked for.

`--swap` adds a third panel per view: the OTHER view's content decoded with THIS view's
style, from `VQVAE.forward(cross_recon=True)` -- content codes held at their own values,
only the final decoder re-run.  It should look like this view's original.  If it looks like
the other view instead, the decoder is ignoring style and the modality rides in content.
`style gain` puts a number on it: 1 = swapping the style alone turns the content view's
reconstruction into this view's, 0 = the style changes nothing.

That swap cannot say whether style carries the per-scan gain and bias: most of what
separates T1 from FLAIR is the renderer's fixed per-modality intensity table, which is not a
latent, and the shared decoder can only learn which modality to draw from the style code.
`--style-gt` holds the modality fixed instead.  Each view is re-rendered with another scan's
gain and bias and nothing else changed -- by default the subject's other view, so the T1
ground truth is T1 anatomy and contrast with FLAIR's gain and bias -- and the original's
content is decoded with the style of that re-render, encoded by the SAME view's encoder.
`--style-gt other-subject` takes gain and bias from another subject instead, whose anatomy
then also shows whether anatomy rides in style.  Written to <out>_style_gt.png + .csv.

Usage:
  python -m eval.plots.plot_reconstruction --run-dir results/synthetic/<run>
  python -m eval.plots.plot_reconstruction --run-dir ... --factor-index 1 --n-samples 4
  python -m eval.plots.plot_reconstruction --run-dir ... --swap      # + cross-style decode per view
  python -m eval.plots.plot_reconstruction --run-dir ... --style-gt  # + gain/bias swap vs rendered truth
  python -m eval.plots.plot_reconstruction --self-test        # torch-free, checks the layout
"""

from __future__ import annotations

import argparse
import csv
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

VIEW_LABEL = ("T1", "FLAIR")
# Sequential ramp: the series IS a magnitude (small -> large ventricle), so one hue
# stepped by lightness, never categorical hues. Original vs recon is linestyle, not colour.
RAMP = ("#9dc3f0", "#2a78d6", "#0b3d7a")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
# Signed change maps: blue <-> red around a neutral gray, so "no change" reads as nothing.
DIVERGING = LinearSegmentedColormap.from_list(
    "change", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3aba5", "#d0393a", "#7c1716"]
)


def profile(vol):
    """Intensity along x through the volume centre -- crosses both ventricle lobes."""
    a = np.asarray(vol).squeeze()
    return a[:, a.shape[1] // 2, a.shape[2] // 2]


def mid_slice(vol):
    a = np.asarray(vol).squeeze()
    return a[:, :, a.shape[2] // 2].T


def dip_area(prof, centre_frac=0.30, ring=(0.35, 0.60)):
    """Integrated intensity deficit below the surrounding tissue, across the central window.

    NOT the dip's depth: depth is fixed by the CSF-vs-WM contrast and is the same whatever
    the cavity's size, so it cannot track ventricle_size at all.  Size changes the dip's
    WIDTH, so the area under the surround level is the quantity that moves.  Integrating
    also handles the septum split, which puts two lobes either side of a thin WM bridge.
    """
    p = np.asarray(prof, dtype=float)
    n = len(p)
    # Normalised distance from the profile's centre: 0 at the middle, 1 at either end.
    # Fractions rather than voxel counts so this is resolution-independent -- and the ring
    # must stay INSIDE the brain (edge at ~0.65 of the half-width), or the surround level
    # is read off background and every area collapses to zero.
    pos = np.abs(np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
    core = pos <= centre_frac
    surround = (pos >= ring[0]) & (pos <= ring[1])
    if not core.any() or not surround.any():
        return float("nan")
    base = float(np.median(p[surround]))
    return float(np.clip(base - p[core], 0, None).sum())


def swaps_by_target(cross, n):
    """Index forward(cross_recon=True)'s extra output by the view each decode should render.

    forward stacks [decode(content_v0, style_v1); decode(content_v1, style_v0)], so its FIRST
    half carries view 1's style and targets view 1.  Exchanging the halves gives
    swap[v] = decode(content_{1-v}, style_v), which is shown and scored against view v.
    """
    a = np.asarray(cross)
    return a.reshape(2, n, *a.shape[1:])[::-1]


def masked_mae(a, b, mask=None):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return float(d[mask].mean() if mask is not None else d.mean())


def projection_gain(effect, target, mask=None):
    """<effect, target> / |target|^2 within the mask: 1 = effect reproduces target, 0 = none of it.

    NaN when the target does not change.
    """
    arrays = [np.asarray(a, dtype=float) for a in (effect, target)]
    if mask is not None:
        arrays = [a[mask] for a in arrays]
    effect, target = (a.ravel() for a in arrays)
    energy = float(target @ target)
    return float(effect @ target / energy) if energy > 1e-12 else float("nan")


def style_gain(swap, recon_target, recon_source, mask=None):
    """How much of the difference between two reconstructions swapping in the style alone reproduces.

    Projects decode(c_source, s_target) - decode(c_source, s_source), the change from swapping
    only the style, onto decode(c_target, s_target) - decode(c_source, s_source), the whole
    difference between the two reconstructions.  1 = style alone carries the difference;
    0 = the decoder ignores style.  Scored on decoder outputs, so reconstruction error does not
    enter it, and within the brain mask, since the recon loss leaves the background
    unconstrained.  NaN when the two reconstructions do not differ.
    """
    swap, recon_target, recon_source = (np.asarray(a, dtype=float) for a in (swap, recon_target, recon_source))
    return projection_gain(swap - recon_source, recon_target - recon_source, mask)


def build_figure(samples, out_png, out_csv, factor_name="ventricle_size"):
    """samples: list of dicts with keys value, orig (2,D,H,W), recon (2,D,H,W).

    Optional, for --swap: swap (2,D,H,W), where swap[v] is the decode that should render view v
    (see swaps_by_target), and mask (D,H,W), the brain voxels its metrics are scored on.
    """
    n = len(samples)
    colours = [RAMP[min(int(i * len(RAMP) / max(n, 1)), len(RAMP) - 1)] for i in range(n)]
    keys = ("orig", "recon", "swap") if "swap" in samples[0] else ("orig", "recon")
    ncol = len(keys)
    titles = {"orig": "{0} original", "recon": "{0} reconstruction", "swap": "{1} content + {0} style"}

    swap_stats = {}
    if "swap" in keys:
        for v in range(2):
            for i, s in enumerate(samples):
                mask = s.get("mask")
                swap_stats[v, i] = {
                    "mae_recon": masked_mae(s["recon"][v], s["orig"][v], mask),
                    "mae_swap": masked_mae(s["swap"][v], s["orig"][v], mask),
                    # What the swap scores if style changes nothing: the content view's own recon.
                    "mae_if_style_ignored": masked_mae(s["recon"][1 - v], s["orig"][v], mask),
                    "style_gain": style_gain(s["swap"][v], s["recon"][v], s["recon"][1 - v], mask),
                }

    fig, axes = plt.subplots(n + 1, 2 * ncol, figsize=(6.5 * ncol, 2.9 * n + 3.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    rows = []
    for v in range(2):
        # The swap is left out of the window's range: it is judged against this view's
        # original, so it is drawn in that original's window rather than allowed to move it.
        allvals = np.concatenate([[s["orig"][v].ravel(), s["recon"][v].ravel()] for s in samples], axis=None)
        lo, hi = float(np.percentile(allvals, 1)), float(np.percentile(allvals, 99.5))
        for i, s in enumerate(samples):
            for k, key in enumerate(keys):
                ax = axes[i][ncol * v + k]
                ax.imshow(mid_slice(s[key][v]), cmap="gray", vmin=lo, vmax=hi, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d9d8d2")
                if i == 0:
                    ax.set_title(
                        titles[key].format(VIEW_LABEL[v], VIEW_LABEL[1 - v]),
                        fontsize=11.5,
                        fontweight="bold",
                        color=INK,
                        pad=6,
                    )
                if key == "swap":
                    ax.set_xlabel(f"style gain {swap_stats[v, i]['style_gain']:+.2f}", fontsize=10, color=INK2)
                if v == 0 and k == 0:
                    ax.text(
                        -0.07,
                        0.5,
                        f"{factor_name}\n{s['value']:+.2f}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=10,
                        color=INK,
                    )

    # Bottom row: the profiles, one panel per view, spanning all of that view's columns.
    for v in range(2):
        for k in range(ncol):
            axes[n][ncol * v + k].remove()
        ax = fig.add_subplot(n + 1, 2, (n * 2) + v + 1)
        xs = np.arange(len(profile(samples[0]["orig"][v])))
        for i, s in enumerate(samples):
            po, pr = profile(s["orig"][v]), profile(s["recon"][v])
            # Only the originals carry a legend entry: linestyle already encodes
            # original-vs-recon and the x-label says so, so labelling both doubles the
            # legend for no information and it then covers the curves.
            ax.plot(xs, po, color=colours[i], lw=2.0, label=f"{s['value']:+.2f}")
            ax.plot(xs, pr, color=colours[i], lw=2.0, ls="--")
            row = {
                "view": VIEW_LABEL[v],
                factor_name: round(float(s["value"]), 4),
                "dip_area_original": round(dip_area(po), 4),
                "dip_area_recon": round(dip_area(pr), 4),
            }
            if "swap" in keys:
                ps = profile(s["swap"][v])
                ax.plot(xs, ps, color=colours[i], lw=2.0, ls=":")
                row["dip_area_swap"] = round(dip_area(ps), 4)
                row.update({name: round(value, 4) for name, value in swap_stats[v, i].items()})
            rows.append(row)
        linestyles = "solid = original, dashed = reconstruction" + (", dotted = style swap" if "swap" in keys else "")
        ax.set_title(f"{VIEW_LABEL[v]}: profile through the ventricle", fontsize=11.5, fontweight="bold", color=INK)
        ax.set_xlabel(f"x (voxels)  — {linestyles}", fontsize=10, color=INK2)
        ax.set_ylabel("intensity", fontsize=10, color=INK2)
        ax.grid(True, color="#e8e7e1", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_edgecolor("#d9d8d2")
        ax.tick_params(colors=MUTED, labelsize=9)
        curves = [profile(s["orig"][v]) for s in samples]
        if "swap" in keys:
            # A swap that kept the other view's contrast can sit outside this view's range.
            curves += [profile(s["swap"][v]) for s in samples]
        top = max(float(np.nanmax(c)) for c in curves)
        bot = min(float(np.nanmin(c)) for c in curves)
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.45 * (top - bot))  # headroom for the legend
        leg = ax.legend(
            frameon=False,
            fontsize=9,
            ncol=min(len(samples), 4),
            labelcolor=INK2,
            loc="upper center",
            columnspacing=1.4,
            handlelength=1.6,
        )
        leg.set_title(factor_name, prop={"size": 9})
        leg.get_title().set_color(MUTED)

    fig.suptitle(
        f"{'Reconstruction and cross-style swap' if 'swap' in keys else 'Reconstruction'} "
        f"vs original across the {factor_name} range",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    if "swap" in keys:
        note = (
            "Each view shares one grayscale window across its panels. The style swap should match that view's "
            "original (style gain 1); gain 0 means the decoder ignores style and the modality rides in content."
        )
    else:
        note = (
            "Each view shares one grayscale window across original and reconstruction. "
            "If the dashed curve's cavity does not widen with the factor, the reconstruction is not carrying it."
        )
    fig.text(0.5, 0.004, note, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def render_style_gt(ds, subject, donor=None):
    """Original, gain/bias ground truth and style donor for one subject, normalized for the encoder.

    The ground truth re-renders `subject` with another scan's gain and bias (z_style[0:2]) and
    nothing else changed: anatomy, noise level, bias field and noise draw all stay.  With no
    `donor` the values come from the subject's other view -- T1 anatomy and contrast with
    FLAIR's gain and bias -- and the ground truth is its own style donor.  With a `donor`
    subject they come from that subject's same view, and the style donor is the donor's
    anatomy rendered with exactly the ground truth's acquisition, so the two differ only in
    anatomy.  Each item is ([view0, view1], brain_mask), through the dataset's normalize_views.
    """
    import torch

    inner = ds._inner
    x1, x2, lat = inner[subject]
    seed = inner.sample_seed_for(subject)
    source = lat if donor is None else inner[donor][2]
    styles = []
    for v in range(2):
        z = lat[f"z_style_v{v + 1}"].clone()
        z[:2] = source[f"z_style_v{2 - v}" if donor is None else f"z_style_v{v + 1}"][:2]
        styles.append(z)

    def render(anatomy, z1, z2):
        r1, r2, mask = inner.render_pseudo_mri(
            anatomy["z_content"],
            anatomy["z_deformation"],
            anatomy["z_fissure"],
            z1,
            z2,
            seed,
            z_lesion=anatomy.get("z_lesion"),
        )
        return list(ds.normalize_views(r1, r2, mask, mask)), mask

    original = list(ds.normalize_views(x1, x2, lat["brain_mask"], lat["brain_mask"])), lat["brain_mask"]
    # The subject's own latents must replay it exactly, or the ground truth would differ from
    # the original by more than gain and bias.
    replayed = render(lat, lat["z_style_v1"], lat["z_style_v2"])[0]
    if not all(torch.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(replayed, original[0])):
        raise ValueError(f"Re-rendering subject {subject} from its latents does not reproduce it")
    gt = render(lat, *styles)
    return original, gt, gt if donor is None else render(source, *styles)


def decode_style_gt(model, originals, gts, donors, device):
    """Decode each original's content with its style donor's style, on exact decoder-input tensors.

    Uses eval.diagnostics.style_path_audit's taps, so its restrictions hold: one VQ level, content/style at
    level 0, stable channel masks.  Each original's own replay must match its forward output
    before a swap is trusted.  Arrays are view-major like the forward: row v * n + i is
    subject i, view v.  Code-change fractions are per row, NaN when nothing is quantized.
    swap_gt decodes with the ground truth's own style, which differs from the donor's only
    through the donor's anatomy; it is the swap itself when the ground truth is the donor.
    """
    from eval.diagnostics.style_path_audit import check_endpoint, encode, replay, validate_model

    validate_model(model)

    def run(items):
        path = encode(model, [x for x, _ in items], [m for _, m in items], device)
        path["content_ids"] = [t.clone() for t in model._last_id_outputs if t is not None]
        return path

    o, g = run(originals), run(gts)
    d = g if donors is gts else run(donors)
    spatial = o["output"].shape[2:]
    own = replay(model, o["content"], o["injected"], spatial)
    for j in range(len(own)):  # per row: a batch mean must not hide one bad replay
        check_endpoint(o["output"][j].cpu(), own[j].cpu())
    swap = replay(model, o["content"], d["injected"], spatial)
    swap_gt = swap if d is g else replay(model, o["content"], g["injected"], spatial)

    def changed(before, after):
        if not before:
            return np.full(len(swap), np.nan)
        return np.mean([(a != b).flatten(1).float().mean(1).cpu().numpy() for a, b in zip(before, after)], axis=0)

    return {
        "recon_orig": o["output"][:, 0].cpu().numpy(),
        "recon_gt": g["output"][:, 0].cpu().numpy(),
        "recon_donor": d["output"][:, 0].cpu().numpy(),
        "swap": swap[:, 0].cpu().numpy(),
        "swap_gt": swap_gt[:, 0].cpu().numpy(),
        "style_codes_changed": changed(
            [o["ids"][k] for k in sorted(o["ids"])], [d["ids"][k] for k in sorted(d["ids"])]
        ),
        "content_codes_changed": changed(o["content_ids"], g["content_ids"]),
    }


def build_style_gt_figure(samples, out_png, out_csv, factor_name="ventricle_size"):
    """samples: dicts with value, orig/gt/swap/recon_orig/recon_gt (2,D,H,W), mask (D,H,W) and
    style_codes_changed/content_codes_changed (2,).  A donor (2,D,H,W) marks the other-subject
    layout, which adds the style donor as a column and needs donor_value, donor_mask, swap_gt
    and recon_donor: there the swap panel reports how much of the donor's anatomy came through
    style, and the gain/bias style gain is scored on swap_gt instead."""
    n = len(samples)
    cross = "donor" in samples[0]
    keys = ["orig", "gt"] + (["donor"] if cross else []) + ["swap", "d_gt", "d_swap"]
    titles = {
        "orig": "original",
        "gt": "ground truth",
        "donor": "style donor",
        "swap": "model swap",
        "d_gt": "Δ ground truth",
        "d_swap": "Δ model swap",
    }

    rows, stats = [], {}
    for v in range(2):
        for i, s in enumerate(samples):
            m, change = s["mask"], s["gt"][v] - s["orig"][v]
            stats[v, i] = {
                "gt_change_rms": float(np.sqrt(np.mean(change[m] ** 2))),
                # How much of the rendered change the model reproduces when it encodes the truth itself.
                "recon_fidelity": projection_gain(s["recon_gt"][v] - s["recon_orig"][v], change, m),
                "style_gain": style_gain(s.get("swap_gt", s["swap"])[v], s["recon_gt"][v], s["recon_orig"][v], m),
                "mae_swap": masked_mae(s["swap"][v], s["gt"][v], m),
                "mae_recon_gt": masked_mae(s["recon_gt"][v], s["gt"][v], m),
                # What the swap scores if style changes nothing: the original's own recon.
                "mae_if_style_ignored": masked_mae(s["recon_orig"][v], s["gt"][v], m),
                "style_codes_changed": float(s["style_codes_changed"][v]),
                "content_codes_changed": float(s["content_codes_changed"][v]),
            }
            if cross:
                # Donor and ground truth share every acquisition setting, so whatever the swap takes
                # from the donor beyond the ground truth's own style is the donor's anatomy.
                stats[v, i]["anatomy_from_donor"] = projection_gain(
                    s["swap"][v] - s["swap_gt"][v], s["recon_donor"][v] - s["recon_gt"][v], m | s["donor_mask"]
                )
                stats[v, i]["mae_donor_recon"] = masked_mae(s["recon_donor"][v], s["gt"][v], m)
            row = {"view": VIEW_LABEL[v], factor_name: round(float(s["value"]), 4)}
            if cross:
                row[f"donor_{factor_name}"] = round(float(s["donor_value"]), 4)
            row.update({name: round(value, 4) for name, value in stats[v, i].items()})
            rows.append(row)

    fig, axes = plt.subplots(2 * n, len(keys), figsize=(2.6 * len(keys) + 0.6, 5.2 * n + 1.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for v in range(2):
        # One grayscale window per view, from the rendered images the swap is judged against.
        allvals = np.concatenate([[s["orig"][v].ravel(), s["gt"][v].ravel()] for s in samples], axis=None)
        lo, hi = float(np.percentile(allvals, 1)), float(np.percentile(allvals, 99.5))
        for i, s in enumerate(samples):
            m, st = s["mask"], stats[v, i]
            change = s["gt"][v] - s["orig"][v]
            # One change scale per row, set by the ground truth and shared by both change panels.
            lim = max(float(np.percentile(np.abs(change[m]), 99.5)), 1e-6)
            panels = {
                "orig": s["orig"][v],
                "gt": s["gt"][v],
                "swap": s["swap"][v],
                "d_gt": change,
                "d_swap": np.where(m, s["swap"][v] - s["recon_orig"][v], 0.0),
            }
            if cross:
                panels["donor"] = s["donor"][v]
            if cross:
                leak = st["anatomy_from_donor"]
                verdict = f"anatomy from donor {leak:.0%}" if np.isfinite(leak) else "donor anatomy too similar"
            elif not st["gt_change_rms"] > 1e-6:
                verdict = "ground truth did not change"
            elif not abs(st["recon_fidelity"]) >= 0.1:
                # Style gain divides by the model's own response to the change; near zero it means nothing.
                verdict = "recon ignores this change"
            else:
                verdict = f"style gain {st['style_gain']:+.2f}"
            codes = st["style_codes_changed"]
            labels = {
                "gt": "donor's gain and bias" if cross else f"{VIEW_LABEL[1 - v]}'s gain and bias",
                "donor": f"{factor_name} {s['donor_value']:+.2f}" if cross else "",
                "swap": verdict,
                "d_gt": f"rms {st['gt_change_rms']:.3f}",
                "d_swap": f"style codes changed {codes:.0%}" if np.isfinite(codes) else "",
            }
            for k, key in enumerate(keys):
                ax = axes[v * n + i][k]
                signed = key.startswith("d_")
                ax.imshow(
                    mid_slice(panels[key]),
                    cmap=DIVERGING if signed else "gray",
                    vmin=-lim if signed else lo,
                    vmax=lim if signed else hi,
                    origin="lower",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d9d8d2")
                if v == 0 and i == 0:
                    ax.set_title(titles[key], fontsize=11.5, fontweight="bold", color=INK, pad=6)
                if labels.get(key):
                    ax.set_xlabel(labels[key], fontsize=9.5, color=INK2)
                if k == 0:
                    ax.text(
                        -0.09,
                        0.5,
                        f"{VIEW_LABEL[v]}\n{factor_name} {s['value']:+.2f}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=10,
                        color=INK,
                    )

    source = "another subject's gain and bias" if cross else "the other view's gain and bias"
    fig.suptitle(
        f"Style swap within modality vs rendered ground truth: each view re-rendered with {source}",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    note = (
        "Ground truth: the original with only gain and bias changed. Model swap: the original's content "
        "with the style donor's style, both encoded by that view's encoder.\nChange panels share one scale "
        "per row (red brighter, blue darker): if they match, style carries gain and bias (style gain 1); "
        "if the swap's stays blank, content does (0)."
    )
    if cross:
        note += (
            "\nThe style donor is another subject's anatomy with the ground truth's acquisition, "
            "so any of its anatomy in the swap came through style."
        )
    fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0.01, 0.045 if cross else 0.035, 1, 0.98])
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def style_gt_report(model, ds, picks, vals, loader_images, device, mode, factor_name, out_png, out_csv):
    """Render, encode, decode and plot --style-gt for the picked subjects; returns the CSV rows."""
    import torch

    if getattr(ds._inner, "mode", None) != "pseudo_mri":
        raise SystemExit("--style-gt needs pseudo_mri synthetic data: it re-renders from the latents.")
    if getattr(ds._inner, "n_style", 3) < 2:
        raise SystemExit("--style-gt needs gain and bias style latents (synthetic_n_style >= 2).")
    if ds.synthetic_normalize != "fixed_reference":
        logger.warning(
            "synthetic_normalize=%s divides out gain and bias, so the ground truth barely differs from the "
            "original. Use a fixed_reference run.",
            ds.synthetic_normalize,
        )
    n = len(picks)
    if mode == "other-subject" and n < 2:
        raise SystemExit("--style-gt other-subject needs --n-samples >= 2.")
    # Donor = the next picked subject: picks span the factor, so neighbours differ in anatomy.
    donors = [picks[(i + 1) % n] for i in range(n)] if mode == "other-subject" else [None] * n
    triples = [render_style_gt(ds, p, d) for p, d in zip(picks, donors)]
    for p, ((original, _), _, _) in zip(picks, triples):
        if not all(torch.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(original, loader_images[p])):
            raise ValueError(f"Subject {p}: re-rendered original differs from the training-path image")
    originals, gts, styled = (list(t) for t in zip(*triples))
    out = decode_style_gt(model, originals, gts, gts if mode == "same-subject" else styled, device)

    def views(a):
        return np.asarray(a).reshape(2, n, *np.shape(a)[1:])

    samples = []
    for i, ((orig, mask), (gt, _), (donor, donor_mask)) in enumerate(triples):
        s = {"value": vals[picks[i]], "mask": mask.numpy()[0] > 0}
        s["orig"], s["gt"] = (np.stack([x.numpy()[0] for x in pair]) for pair in (orig, gt))
        for key in ("recon_orig", "recon_gt", "swap", "style_codes_changed", "content_codes_changed"):
            s[key] = views(out[key])[:, i]
        if mode == "other-subject":
            s["donor"], s["donor_value"] = np.stack([x.numpy()[0] for x in donor]), vals[donors[i]]
            s["donor_mask"] = donor_mask.numpy()[0] > 0
            s["swap_gt"], s["recon_donor"] = views(out["swap_gt"])[:, i], views(out["recon_donor"])[:, i]
        samples.append(s)
    return build_style_gt_figure(samples, out_png, out_csv, factor_name)


def _self_test():
    """Planted volumes: the original's cavity tracks the factor, the recon's does not."""
    res = 32
    g = np.linspace(-1, 1, res)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    d = np.sqrt(X**2 + Y**2 + Z**2)

    def vol(vent):
        v = np.where(d < 0.65, 0.8, 0.0)
        return np.where(d < vent, 0.1, v)

    samples = []
    for value in (-0.8, 0.0, 0.8):
        vent = 0.20 + value * 0.08
        orig = np.stack([vol(vent), vol(vent)])
        recon = np.stack([vol(0.20), vol(0.20)])  # frozen: ignores the factor
        samples.append({"value": value, "orig": orig, "recon": recon})

    out = "/tmp/_plot_recon_selftest"
    rows = build_figure(samples, f"{out}.png", f"{out}.csv")
    orig_dips = [r["dip_area_original"] for r in rows if r["view"] == "T1"]
    recon_dips = [r["dip_area_recon"] for r in rows if r["view"] == "T1"]
    assert max(orig_dips) - min(orig_dips) > 1e-6, f"planted variation not detected: {orig_dips}"
    assert len(set(recon_dips)) == 1, f"frozen recon should give a constant area, got {recon_dips}"
    print(f"original cavity areas vary {orig_dips}; frozen-recon areas constant {recon_dips}")

    # forward's cross output, labelled 10 * content view + style view, in forward's row order.
    cross = np.concatenate([np.full((2, 4), 10 * 0 + 1.0), np.full((2, 4), 10 * 1 + 0.0)])
    swap = swaps_by_target(cross, 2)
    assert (swap[0] == 10).all() and (swap[1] == 1).all(), "swap[v] must pair view v's style with the other content"

    t1, flair = vol(0.20), 0.5 * vol(0.20)
    for planted, want in ((t1, 1.0), (flair, 0.0), ((t1 + flair) / 2, 0.5)):
        got = style_gain(planted, t1, flair)
        assert abs(got - want) < 1e-9, f"style gain {got} for a planted {want}"

    # Style ignored: each swap comes back as its content view's own reconstruction.
    brain = d < 0.65
    swap_samples = []
    for value in (-0.8, 0.0, 0.8):
        both = np.stack([vol(0.20 + value * 0.08), 0.5 * vol(0.20 + value * 0.08)])
        swap_samples.append({"value": value, "orig": both, "recon": both, "swap": both[::-1], "mask": brain})
    rows = build_figure(swap_samples, f"{out}_swap.png", f"{out}_swap.csv")
    assert all(abs(r["style_gain"]) < 1e-9 for r in rows), [r["style_gain"] for r in rows]
    assert all(r["mae_swap"] == r["mae_if_style_ignored"] > r["mae_recon"] == 0 for r in rows), rows
    print("swap order, style gain 1/0/0.5 and the style-ignored panel all recovered")

    # Rendered gain/bias ground truth with a perfect reconstruction: the swap either lands on the
    # ground truth (style carries gain and bias) or stays on the original (content does).
    gt_samples = []
    for value in (-0.8, 0.8):
        orig = np.stack([vol(0.20 + value * 0.08), 0.5 * vol(0.20 + value * 0.08)])
        gt = np.where(brain, 1.2 * orig + 0.05, 0.0)
        codes = {"style_codes_changed": np.array([0.4, 0.6]), "content_codes_changed": np.zeros(2)}
        gt_samples.append({"value": value, "orig": orig, "gt": gt, "recon_orig": orig, "recon_gt": gt, "mask": brain})
        gt_samples[-1].update(codes)
    rows = build_style_gt_figure([dict(s, swap=s["gt"]) for s in gt_samples], f"{out}_gt.png", f"{out}_gt.csv")
    assert all(r["style_gain"] == r["recon_fidelity"] == 1 and r["mae_swap"] == 0 for r in rows), rows
    # Other-subject layout: the donor's anatomy is the views swapped. Style ignored altogether, then a
    # style that carries gain and bias AND all of the donor's anatomy.
    donor = {"donor_mask": brain, "donor_value": 0.0}
    ignored = [
        dict(s, swap=s["orig"], swap_gt=s["orig"], donor=s["orig"][::-1], recon_donor=s["gt"][::-1], **donor)
        for s in gt_samples
    ]
    rows = build_style_gt_figure(ignored, f"{out}_gt_other.png", f"{out}_gt_other.csv")
    assert all(r["style_gain"] == 0 == r["anatomy_from_donor"] for r in rows), rows
    assert all(r["mae_swap"] == r["mae_if_style_ignored"] > 0 for r in rows), rows
    leaky = [
        dict(s, swap=s["gt"][::-1], swap_gt=s["gt"], donor=s["orig"][::-1], recon_donor=s["gt"][::-1], **donor)
        for s in gt_samples
    ]
    rows = build_style_gt_figure(leaky, f"{out}_gt_other.png", f"{out}_gt_other.csv")
    assert all(r["style_gain"] == 1 == r["anatomy_from_donor"] for r in rows), rows
    print("gain/bias ground truth: style gain 1/0 and anatomy from donor 0/1 recovered in both layouts")
    print(f"self-test OK: wrote {out}.png, {out}_swap.png, {out}_gt.png and {out}_gt_other.png")


def main():
    ap = argparse.ArgumentParser(description="Plot T1/FLAIR reconstructions against originals.")
    ap.add_argument("--run-dir", help="Training run dir with settings.json.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--factor-index", type=int, default=1, help="GT z_content index to span (1 = ventricle_size).")
    ap.add_argument("--factor-name", default="ventricle_size")
    ap.add_argument("--n-samples", type=int, default=3, help="Samples shown, spanning the factor's range.")
    ap.add_argument("--scan", type=int, default=48, help="Candidates to draw before picking the spread.")
    ap.add_argument("--out", default="reconstruction.png")
    ap.add_argument(
        "--swap",
        action="store_true",
        help="Add a panel per view: the other view's content decoded with this view's style "
        "(VQVAE.forward cross_recon=True). Needs style injected into the decoder at level 0.",
    )
    ap.add_argument(
        "--style-gt",
        nargs="?",
        const="same-subject",
        choices=("same-subject", "other-subject"),
        help="Also write <out>_style_gt.png: each view re-rendered with another scan's gain and bias (the "
        "subject's other view, or another subject's same view) against the model's within-modality style "
        "swap. Needs pseudo_mri data, one VQ level and a fixed or learned channel mask.",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        ap.error("--run-dir is required (or pass --self-test)")

    import torch
    from torch.utils.data import DataLoader

    from eval.protocol.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    model, run_args, device = load_model_from_run_dir(args.run_dir, args.checkpoint, None)
    model.eval()
    inner = model.module if hasattr(model, "module") else model

    ds = build_synthetic_test_set(run_args, args.scan, causal=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    imgs, masks, vals = [], [], []
    for batch in loader:
        v1, v2 = batch["image"]
        for b in range(v1.shape[0]):
            imgs.append((v1[b], v2[b]))
            m = batch.get("mask")
            masks.append(m[0][b] if m is not None else None)
            vals.append(float(batch["gt_latents"]["z_content"][b, args.factor_index]))
    order = np.argsort(vals)
    picks = [int(order[i]) for i in np.linspace(0, len(order) - 1, args.n_samples).round().astype(int)]
    logger.info("picked %s at %s = %s", picks, args.factor_name, [round(vals[p], 3) for p in picks])

    # View-major batching: separate encoders and per-view codebooks require all of view 0
    # then all of view 1, matching eval/lesion/lesion_reconstruction.reconstruct.
    x = torch.cat([torch.stack([imgs[p][v] for p in picks]) for v in range(2)]).to(device)
    fwd_mask = None
    if getattr(inner, "latent_mask", False) and masks[picks[0]] is not None:
        fwd_mask = torch.cat([torch.stack([masks[p] for p in picks])] * 2).to(device)
    with torch.no_grad():
        out = model(
            x, return_recon=True, pool_only=True, n_views=2, subsets=[(0, 1)], mask=fwd_mask, cross_recon=args.swap
        )
    # cross_recon appends the swapped decode after the usual eight-tuple.
    out, cross = out if args.swap else (out, None)
    y = out[0]
    if y is None or y.shape != x.shape:
        raise SystemExit("No reconstruction returned — was the run trained with a decoder?")
    recon = y.detach().cpu().numpy().reshape(2, len(picks), *y.shape[1:])[:, :, 0]
    orig = x.detach().cpu().numpy().reshape(2, len(picks), *x.shape[1:])[:, :, 0]

    samples = [
        {"value": vals[p], "orig": np.stack([orig[0][i], orig[1][i]]), "recon": np.stack([recon[0][i], recon[1][i]])}
        for i, p in enumerate(picks)
    ]
    if cross is not None:
        swap = swaps_by_target(cross.detach().cpu().numpy()[:, 0], len(picks))
        if masks[picks[0]] is None:
            logger.warning("No brain mask in the batch: swap metrics include the unconstrained background.")
        for i, (s, p) in enumerate(zip(samples, picks)):
            s["swap"] = swap[:, i]
            if masks[p] is not None:
                s["mask"] = masks[p].numpy().reshape(orig.shape[2:]) > 0
    out_csv = args.out.rsplit(".", 1)[0] + ".csv"
    rows = build_figure(samples, args.out, out_csv, factor_name=args.factor_name)
    print(f"\nwrote {args.out} and {out_csv}\n")
    print(f"  {'view':<8}{args.factor_name:>16}{'cavity area':>15}{'recon area':>12}")
    print("  " + "-" * 51)
    for r in rows:
        print(
            f"  {r['view']:<8}{r[args.factor_name]:>16.3f}{r['dip_area_original']:>15.3f}{r['dip_area_recon']:>12.3f}"
        )
    print("\n  A recon area that stays flat while the original's grows = the ventricle is not encoded.")
    if cross is not None:
        print(
            "\n  Style swap: the other view's content decoded with this view's style; MAEs against this view's original."
        )
        print(
            f"  {'view':<8}{args.factor_name:>16}{'recon MAE':>12}{'swap MAE':>11}{'style ignored':>15}{'style gain':>12}"
        )
        print("  " + "-" * 74)
        for r in rows:
            print(
                f"  {r['view']:<8}{r[args.factor_name]:>16.3f}{r['mae_recon']:>12.4f}{r['mae_swap']:>11.4f}"
                f"{r['mae_if_style_ignored']:>15.4f}{r['style_gain']:>12.2f}"
            )
        print(
            "\n  'style ignored' = the MAE the swap would score if style changed nothing (the content view's own recon)."
        )
        print("  Swap MAE near recon MAE, gain ~1: style carries the modality.")
        print("  Swap MAE near 'style ignored', gain ~0: the decoder ignores style and the modality rides in content.")
    if not args.style_gt:
        return

    stem = args.out.rsplit(".", 1)[0]
    gt_png, gt_csv = f"{stem}_style_gt.png", f"{stem}_style_gt.csv"
    rows = style_gt_report(inner, ds, picks, vals, imgs, device, args.style_gt, args.factor_name, gt_png, gt_csv)
    print(f"\nwrote {gt_png} and {gt_csv}\n")
    print(
        f"  Within-modality style swap vs rendered gain/bias ground truth ({args.style_gt}); MAEs vs the ground truth."
    )
    print(
        f"  {'view':<7}{args.factor_name:>15}{'change':>8}{'fidelity':>10}{'style gain':>12}{'swap MAE':>10}"
        f"{'recon(GT)':>11}{'ignored':>9}{'style codes':>13}{'content codes':>15}"
    )
    print("  " + "-" * 110)
    for r in rows:
        print(
            f"  {r['view']:<7}{r[args.factor_name]:>15.3f}{r['gt_change_rms']:>8.3f}{r['recon_fidelity']:>10.2f}"
            f"{r['style_gain']:>12.2f}{r['mae_swap']:>10.4f}{r['mae_recon_gt']:>11.4f}{r['mae_if_style_ignored']:>9.4f}"
            f"{r['style_codes_changed']:>13.0%}{r['content_codes_changed']:>15.0%}"
        )
    print("\n  change    RMS of (ground truth - original) in the brain: how much gain and bias moved the image.")
    print("  fidelity  share of that change the model reproduces when it encodes the ground truth itself.")
    print("  style gain  1 = swapping in the ground truth's style reproduces it, 0 = style ignores gain and bias.")
    print("  swap MAE vs recon(GT) MAE (the best the model does) vs 'ignored' (the original's own recon).")
    print("  codes changed: style should move between original and donor; content should not move with gain/bias.")
    if args.style_gt != "other-subject":
        return
    print("\n  Anatomy: the donor shares the ground truth's acquisition, so what the swap takes from it is anatomy.")
    donor_col = f"donor {args.factor_name}"
    print(
        f"  {'view':<7}{args.factor_name:>15}{donor_col:>21}{'anatomy from donor':>20}{'swap MAE':>10}{'donor recon':>13}"
    )
    print("  " + "-" * 86)
    for r in rows:
        print(
            f"  {r['view']:<7}{r[args.factor_name]:>15.3f}{r[f'donor_{args.factor_name}']:>21.3f}"
            f"{r['anatomy_from_donor']:>20.0%}{r['mae_swap']:>10.4f}{r['mae_donor_recon']:>13.4f}"
        )
    print("\n  anatomy from donor  0 = anatomy stays with content, 1 = the swap takes all of the donor's anatomy.")
    print(
        "  swap MAE near recon(GT) = clean; near 'donor recon' (the donor's own recon vs this truth) = style is anatomy."
    )


if __name__ == "__main__":
    main()
