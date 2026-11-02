"""Each training patch grid drawn over one validation subject's T1 and FLAIR, with a CSV twin of its numbers.

One column per grid the pilot tried (2^3, 3^3, 4^3, 8^3, 16^3). The squares are the real pooling bins: the backbone map
is 16^3 (one unit = 4^3 voxels) and adaptive_avg_pool3d cuts it into the grid, so a 3^3 cell is 24 voxels wide and
overlaps its neighbour by 4. Only cells the foreground rule keeps are drawn (>= 5% brain in some val subject; a batch
of 32 keeps a subset of these). Orange is the cell holding the most lesion.

The subject is the val subject nearest the median on both lesion volume and best-cell fill at 16^3, not the clearest
one. The slice runs through its lesion's centre, and every column shows the same pixels at the same window.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/patch_grids.py BANK_DIR
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regional as R  # noqa: E402

import lesion as L  # noqa: E402

pilot, T = L.pilot, R.T
GRIDS = (2, 3, 4, 8, 16)
MAP, STRIDE, SIZE = 16, 4, 64
THRESH = 0.05
AXIS = 2  # the slice is taken across this array axis
ZOOM = 32  # voxels shown in the zoomed row
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, BEST, EDGE = "#3987e5", "#d95926", "#199e70"


def bins(g):
    """Each pooling bin's voxel range along one axis, by adaptive_avg_pool3d's rule on the 16^3 map."""
    return [(int(np.floor(i * MAP / g)) * STRIDE, int(np.ceil((i + 1) * MAP / g)) * STRIDE) for i in range(g)]


def ranks(values):
    return values.argsort().argsort() / (len(values) - 1)


def pick_subject(cells):
    volume, fill = cells.sum((1, 2, 3)), cells.max((1, 2, 3))
    return int((np.abs(ranks(volume) - 0.5) + np.abs(ranks(fill) - 0.5)).argmin())


def voxel_lesion(subject, n):
    inner = T.make_dataset(pilot.user_args(), "val", n)._inner
    d = inner._draw_pseudo_mri(inner.sample_seed_for(subject))
    with torch.no_grad():
        _, load = inner.renderer.render_structure(
            d["z_content"], d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
        )
    return load.float().numpy()


def best_cell(load, g):
    edges = bins(g)
    held = np.array([[[load[a:b, c:d, e:f].sum() for e, f in edges] for c, d in edges] for a, b in edges])
    return np.unravel_index(held.argmax(), held.shape), held.max()


def population(cells, g):
    """Median over subjects: the share of the best cell that is lesion, of the lesion in that cell, and cells touched."""
    pooled = F.adaptive_avg_pool3d(torch.from_numpy(cells)[:, None], g)[:, 0].numpy()
    voxels = (np.array([b - a for a, b in bins(g)]) ** 3)[0]
    best = pooled.reshape(len(cells), -1).max(1)
    return (
        np.median(best),
        np.median(best * voxels / (cells.sum((1, 2, 3)) * STRIDE**3)),
        np.median((pooled > 0).sum((1, 2, 3))),
    )


def grid_lines(keep, s, u, v):
    """Edges of the kept cells that the slice at s passes through."""
    edges = bins(keep.shape[0])
    segments = set()
    for k, (start, stop) in enumerate(edges):
        if start <= s < stop:
            for i, j in np.argwhere(np.take(keep, k, axis=AXIS)):
                x0, x1, y0, y1 = *edges[i], *edges[j]
                segments |= {((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))}
    return sorted(segments)


def panel(ax, image, window, keep, best, g, s, lesion_slice, u, v, crop=None):
    ax.imshow(
        image.T,
        origin="lower",
        extent=(0, SIZE, 0, SIZE),
        cmap="gray",
        interpolation="nearest",
        vmin=window[0],
        vmax=window[1],
    )
    ax.add_collection(LineCollection(grid_lines(keep, s, u, v), colors=GRID, linewidths=0.55))
    edges = bins(g)
    x0, x1, y0, y1 = *edges[best[u]], *edges[best[v]]
    assert edges[best[AXIS]][0] <= s < edges[best[AXIS]][1], f"best {g}^3 cell misses the slice"
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=BEST, linewidth=1.8))
    if crop is not None:
        grid = np.arange(SIZE) + 0.5
        ax.contour(grid, grid, lesion_slice.T, levels=[0.5], colors=EDGE, linewidths=1.3)
        ax.set_xlim(crop[0], crop[0] + ZOOM)
        ax.set_ylim(crop[1], crop[1] + ZOOM)
    else:
        ax.set_xlim(0, SIZE)
        ax.set_ylim(0, SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    bank = sys.argv[1]
    t1, flair = np.load(f"{bank}/val_t1.npy", mmap_mode="r"), np.load(f"{bank}/val_flair.npy", mmap_mode="r")
    masks = torch.from_numpy(np.load(f"{bank}/val_mask.npy"))[:, None]
    cells = np.load(f"{HERE}/val_lesion_cells.npy")
    subject = pick_subject(cells)
    load = voxel_lesion(subject, len(cells))
    pooled = F.avg_pool3d(torch.from_numpy(load)[None, None], STRIDE)[0, 0].numpy()
    assert np.abs(pooled - cells[subject]).max() < 1e-4, "voxel lesion disagrees with the saved labels"

    axes = [0, 1, 2]
    centre = [
        float((load.sum(tuple(a for a in axes if a != axis)) * np.arange(SIZE)).sum() / load.sum()) for axis in axes
    ]
    s = int(round(centre[AXIS] - 0.5))
    u, v = (a for a in axes if a != AXIS)
    crop = [int(np.clip(round(centre[a]) - ZOOM // 2, 0, SIZE - ZOOM)) for a in (u, v)]
    images = {name: np.asarray(view[subject]).astype(np.float32) for name, view in (("T1", t1), ("FLAIR", flair))}
    brain = masks[subject, 0].numpy() > 0
    windows = {name: np.percentile(image[brain], [1, 99.5]) for name, image in images.items()}
    lesion_slice = np.take(load, s, axis=AXIS)

    rows = []
    keeps, bests = {}, {}
    for g in GRIDS:
        keep = T.foreground_positions(masks, (g, g, g), THRESH).reshape(g, g, g).numpy()
        best, held = best_cell(load, g)
        width = bins(g)[0][1] - bins(g)[0][0]
        fill, share, touched = population(cells, g)
        keeps[g], bests[g] = keep, best
        rows.append(
            dict(
                grid=g,
                cell_width_voxels=width,
                overlap_voxels=bins(g)[0][1] - bins(g)[1][0],
                cells_kept=int(keep.sum()),
                cells_total=g**3,
                subject_best_cell_fill=float(held / width**3),
                subject_lesion_share_in_best_cell=float(held / load.sum()),
                val_median_best_cell_fill=float(fill),
                val_median_lesion_share_in_best_cell=float(share),
                val_median_cells_touched=float(touched),
            )
        )
    with open(f"{HERE}/patch_grids.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"subject {subject}, lesion {load.sum():.1f} voxels, slice {s} of axis {AXIS}, centre {np.round(centre, 1)}, peak in slice {lesion_slice.max():.2f}"
    )
    for row in rows:
        print(row)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    fig, grid = plt.subplots(3, len(GRIDS), figsize=(13.2, 9.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.095, wspace=0.03, hspace=0.05)
    for c, (g, row) in enumerate(zip(GRIDS, rows)):
        for r, name in enumerate(("T1", "FLAIR", "FLAIR")):
            zoomed = crop if r == 2 else None
            image = np.take(images[name], s, axis=AXIS)
            panel(grid[r, c], image, windows[name], keeps[g], bests[g], g, s, lesion_slice, u, v, zoomed)
        overlap = f", overlap {row['overlap_voxels']}" if row["overlap_voxels"] else ""
        grid[0, c].set_title(
            f"{g}³ grid\n{row['cell_width_voxels']} voxels wide{overlap}\n{row['cells_kept']:,} of {row['cells_total']:,} kept",
            fontsize=10.5,
            loc="left",
            pad=6,
        )
        grid[2, c].set_xlabel(
            f"Fills {row['subject_best_cell_fill']:.1%} of orange cell\n"
            f"Median of 400 subjects: {row['val_median_best_cell_fill']:.1%}",
            fontsize=10,
            color=MUTED,
            loc="left",
            labelpad=6,
        )
    for r, label in enumerate(("T1", "FLAIR", f"FLAIR, {ZOOM}-voxel zoom")):
        grid[r, 0].set_ylabel(label, fontsize=11)
    fig.text(0.05, 0.965, TITLE, fontsize=15, fontweight="bold", va="top")
    fig.text(0.05, 0.93, SUBTITLE.format(volume=load.sum(), s=s), fontsize=10.5, color=MUTED, va="top")
    handles = [
        Line2D(
            [], [], color=GRID, lw=1.2, label="Cell kept for the loss (no line: under 5% brain in all 400 subjects)"
        ),
        Line2D([], [], color=BEST, lw=2, label="Cell holding the most lesion"),
        Line2D([], [], color=EDGE, lw=1.5, label="Lesion edge, from the generator's map"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.045, 0.012), ncol=3, frameon=False, fontsize=10)
    fig.savefig(f"{HERE}/patch_grids.png", dpi=170, facecolor=SURFACE)


TITLE = "Only the 16³ cells are about the size of the lesion"
SUBTITLE = (
    "One validation subject, lesion {volume:.0f} voxels, slice {s} through its centre. Each square is a slice through "
    "a 3D cell. Every column shows the same image."
)


if __name__ == "__main__":
    main()
