"""The renderer's multiplicative field, on and off: the same subjects, lesions, styles and noise draws.

render_modality multiplies each view by 1 + 0.15 x noise drawn on a 16^3 grid, so the field varies every ~4 voxels,
about a lesion's size. bank.py --field off removes it while drawing the same random numbers, so nothing else changes.
This renders the local bank's recipe (T1 lesion 0.4) both ways and fits each condition's normative residual
(residual_bank.py: the scan z-scored over its brain, minus the training mean, minus 16 training PCs) on the same
training subjects. It then plots one test subject's axial slice through its lesion: each scan with and without the
field, the field itself, and the residual both ways, in units of the with-field residual's SD so they share a scale.

    PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:. \
        python out/field_preview_20261111/render.py out/burden_feasibility_20261102/bank out/field_preview_20261111
"""

import argparse
import textwrap
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import bank as B  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import residual  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

import training.main_conv_synthetic as T  # noqa: E402

FIELDS = ("lesion", "off")
VIEWS = ("t1", "flair")
SIZES = {"train": 2000, "test": 1000}
N_TRAIN, N_TEST, K = 300, 40, 16
CROP = 11  # zoom half-width in voxels
AWAY = 6.0  # voxels from the lesion centre beyond which the interior counts as background
INK, MUTED, SURFACE = "#222220", "#6b6b66", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8a1f1e"])
DIVERGING.set_bad(SURFACE)
GREY = plt.get_cmap("gray").copy()
GREY.set_bad("black")


def reference(field):
    """fixed_reference's mean and scale, fitted as bank.py fits them: on training renders under this field."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train = T.make_dataset(B.run_args(0.4), "train", SIZES["train"])
        B.set_field(train._inner.renderer, field)
        train[0]
    return train._fixed_mean, train._fixed_scale


def render(task):
    split, start, stop, field, ref, mask = task
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    ds = T.make_dataset(B.run_args(0.4), split, SIZES[split])
    renderer = ds._inner.renderer
    B.set_field(renderer, field)
    ds._fixed_mean, ds._fixed_scale = ref
    kept = {"fields": []}
    real_structure, real_noise = renderer.render_structure, renderer._seeded_noise

    def structure(*a, **k):
        kept["tissue"], kept["load"] = real_structure(*a, **k)
        return kept["tissue"], kept["load"]

    def noise(scale, gen, device):
        out = real_noise(scale, gen, device)
        kept["fields"].append((1.0 + 0.15 * renderer.style_scale * out).numpy())
        return out

    renderer.render_structure, renderer._seeded_noise = structure, noise
    rows = {"images": [], "load": [], "field": [], "tissue": []}
    for i in range(start, stop):
        kept["fields"] = []
        item = ds[i]
        images = np.stack([item["image"][v][0].numpy() for v in range(2)]).astype(np.float32)
        if split == "train":
            rows["images"].append(np.stack([residual.standard(x, mask) for x in images]))
            continue
        rows["images"].append(images)
        rows["load"].append(kept["load"].numpy().astype(np.float32))
        rows["field"].append(np.stack(kept["fields"][-2:]).astype(np.float32))
        rows["tissue"].append(kept["tissue"].numpy() > 0)
    return split, field, start, {k: np.stack(v) for k, v in rows.items() if v}


def fit_basis(rows):
    mean = rows.mean(0)
    _, _, v = torch.svd_lowrank(torch.from_numpy(rows - mean), q=K + 8, niter=4)
    return mean, v[:, :K].numpy()


def remove_anatomy(row, mean, basis):
    delta = row - mean
    return delta - basis @ (basis.T @ delta)


def residual_volume(image, mask, mean, basis, scale):
    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[mask] = remove_anatomy(residual.standard(image, mask), mean, basis) / scale
    return volume


def lesion_stats(res, load, interior):
    """|median residual| over the lesion core (load >= 0.5) and the residual's SD over the interior away from it."""
    grid = np.stack(np.indices(load.shape), -1)
    centre = (grid * load[..., None]).sum((0, 1, 2)) / load.sum()
    away = interior & (np.linalg.norm(grid - centre, axis=-1) > AWAY)
    return float(abs(np.median(res[load >= 0.5]))), float(res[away].std())


def panel(ax, image, cmap, vmin, vmax, box=None, outline=None, lesion=None, crop=None):
    if crop is not None:
        image = image[crop]
        outline = None if outline is None else outline[crop]
        lesion = None if lesion is None else lesion[crop]
    shown = ax.imshow(image.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    if outline is not None:
        ax.contour(outline.T.astype(float), levels=[0.5], colors=MUTED, linewidths=0.6)
    if lesion is not None:
        ax.contour(lesion.T, levels=[0.5], colors=INK, linewidths=0.9)
    if box is not None:
        (x0, x1), (y0, y1) = box
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=box_colour(cmap), lw=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    return shown


def box_colour(cmap):
    return "white" if cmap is GREY else INK


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank")
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    occupancy = (np.asarray(np.load(f"{a.bank}/train_flair.npy", mmap_mode="r")[:200]) != 0).mean(0)
    mask = occupancy >= 0.5
    interior = binary_erosion(occupancy >= 0.9, iterations=2)
    refs = {field: reference(field) for field in FIELDS}
    print("fixed_reference mean, scale:", {k: tuple(round(float(x), 4) for x in v) for k, v in refs.items()})
    tasks = []
    for field in FIELDS:
        tasks += [("train", s, min(s + 38, N_TRAIN), field, refs[field], mask) for s in range(0, N_TRAIN, 38)]
        tasks += [("test", s, min(s + 5, N_TEST), field, refs[field], mask) for s in range(0, N_TEST, 5)]
    with ProcessPoolExecutor(8) as pool:
        done = sorted(pool.map(render, tasks), key=lambda r: (r[0], r[1], r[2]))
    data = {}
    for split in SIZES:
        for field in FIELDS:
            parts = [r[3] for r in done if (r[0], r[1]) == (split, field)]
            data[split, field] = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    stored = [np.load(f"{a.bank}/test_{key}.npy", mmap_mode="r")[:N_TEST] for key in ("t1_04", "flair")]
    on = data[("test", "lesion")]
    worst = max(float(np.abs(on["images"][:, v].astype(np.float16) - stored[v]).max()) for v in range(2))
    print(f"with-field re-render vs stored bank, max abs difference {worst:.4g}")

    bases, scales = {}, {}
    for field in FIELDS:
        rows = data[("train", field)]["images"]
        for v, view in enumerate(VIEWS):
            bases[field, view] = fit_basis(rows[:, v])
            res = np.stack([remove_anatomy(r, *bases[field, view]) for r in rows[:200, v]])
            scales[field, view] = float(res.std())
    print(
        "residual SD on training subjects, with field / off:",
        {v: (scales["lesion", v], scales["off", v]) for v in VIEWS},
    )

    residuals, stats = {}, {}
    for field in FIELDS:
        test = data[("test", field)]
        for v, view in enumerate(VIEWS):
            scale = scales["lesion", view]  # one scale for both conditions, so the field's share is visible
            residuals[field, view] = np.stack(
                [residual_volume(x, mask, *bases[field, view], scale) for x in test["images"][:, v]]
            )
            stats[field, view] = np.array(
                [lesion_stats(r, load, interior) for r, load in zip(residuals[field, view], test["load"])]
            )
    for view in VIEWS:
        for field in FIELDS:
            lesion, spread = np.median(stats[field, view], 0)
            ratio = np.median(stats[field, view][:, 0] / stats[field, view][:, 1])
            print(f"{view:5s} field {field:6s}: lesion {lesion:.2f}, spread {spread:.2f}, lesion / spread {ratio:.1f}")

    vox = on["load"].sum((1, 2, 3))
    subject = int(np.argmin(np.abs(vox - np.median(vox))))
    load = on["load"][subject]
    grid = np.stack(np.indices(load.shape), -1)
    cx, cy, cz = (int(round(c)) for c in (grid * load[..., None]).sum((0, 1, 2)) / load.sum())
    brain = on["tissue"][subject][:, :, cz]
    crop = (slice(cx - CROP, cx + CROP + 1), slice(cy - CROP, cy + CROP + 1))
    box = ((cx - CROP, cx + CROP + 1), (cy - CROP, cy + CROP + 1))
    lesion_slice = load[:, :, cz]
    print(f"shown: test subject {subject}, lesion {vox[subject]:.0f} voxels, centre ({cx}, {cy}, {cz})")

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "font.family": "sans-serif"})
    fig = plt.figure(figsize=(14.5, 13.4), facecolor=SURFACE)
    layout = fig.add_gridspec(
        5,
        5,
        height_ratios=(1, 1, 1, 1, 0.045),
        left=0.045,
        right=0.99,
        top=0.93,
        bottom=0.115,
        wspace=0.05,
        hspace=0.07,
    )
    axes = np.array([[fig.add_subplot(layout[r, c]) for c in range(5)] for r in range(4)])
    bars = {"field": fig.add_subplot(layout[4, 2]), "res": fig.add_subplot(layout[4, 3:5])}
    titles = (
        "Scan, with field",
        "Scan, field off",
        "The field (multiplier)",
        "Residual, with field",
        "Residual, field off",
    )
    labels = ("T1", "FLAIR", "T1, zoomed", "FLAIR, zoomed")
    images = {}
    for v, view in enumerate(VIEWS):
        scans = [data[("test", f)]["images"][subject, v][:, :, cz] for f in FIELDS]
        inside = np.concatenate([s[brain] for s in scans])
        lo, hi = np.percentile(inside, [0.5, 99.5])
        scans = [np.where(brain, s, np.nan) for s in scans]
        field = np.where(brain, on["field"][subject, v][:, :, cz], np.nan)
        res = [np.where(brain, residuals[f, view][subject][:, :, cz], np.nan) for f in FIELDS]
        for zoomed in (False, True):
            row = axes[v + 2 * zoomed]
            kw = {"crop": crop} if zoomed else {"box": box}
            panel(row[0], scans[0], GREY, lo, hi, **kw)
            panel(row[1], scans[1], GREY, lo, hi, **kw)
            images["field"] = panel(row[2], field, DIVERGING, 0.75, 1.25, outline=brain, lesion=lesion_slice, **kw)
            images["res"] = panel(row[3], res[0], DIVERGING, -5, 5, outline=brain, **kw)
            panel(row[4], res[1], DIVERGING, -5, 5, outline=brain, **kw)
            if zoomed:
                for column, f in ((3, "lesion"), (4, "off")):
                    lesion, spread = stats[f, view][subject]
                    row[column].text(
                        0.03,
                        0.03,
                        f"lesion {lesion:.1f}, spread {spread:.2f}",
                        transform=row[column].transAxes,
                        fontsize=8.5,
                        color=INK,
                        bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": "none", "alpha": 0.85},
                    )
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=11, color=INK, pad=6)
    for ax, label in zip(axes[:, 0], labels):
        ax.set_ylabel(label, fontsize=11, color=INK, labelpad=6)
    for key, label in (("field", "multiplier (1 = no change)"), ("res", "residual, in SDs of the with-field residual")):
        bar = fig.colorbar(images[key], cax=bars[key], orientation="horizontal")
        bar.set_label(label, color=INK, fontsize=9.5)
        bar.ax.tick_params(colors=MUTED, labelsize=8.5)
        bar.outline.set_visible(False)
    fig.suptitle(
        f"The renderer's multiplicative field, on and off: test subject {subject}, lesion {vox[subject]:.0f} voxels"
        f" (median of {N_TEST}), axial slice z = {cz}",
        fontsize=13,
        color=INK,
        y=0.975,
    )
    notes = (
        "Same subject, lesion, style and noise draws; only the field differs. The grey scale is shared within each row."
        " Black circle on the field: the lesion's edge. Boxes: the zoomed window. Residual: the scan z-scored over its"
        " brain, minus the training mean image and its 16 largest principal components (fitted for each condition on"
        f" the same {N_TRAIN} training subjects), in SDs of the with-field residual. Lesion: |median residual| in the"
        f" lesion core. Spread: residual SD over the interior brain more than {AWAY:g} voxels from the lesion."
    )
    fig.text(0.045, 0.062, textwrap.fill(notes, 180), fontsize=9, color=MUTED, va="top", linespacing=1.4)
    out = a.out / "1_field_on_off.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
