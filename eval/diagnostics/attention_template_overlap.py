"""Do a trained encoder-only run's global attention heads sit on the temporal-atrophy site?

Synthetic temporal atrophy (``z_content[6]``) is a fixed-site factor. The generator shrinks the
WM/GM boundary under one bilateral Gaussian bump (``render_structure`` in
``eval/synthetic/synthetic_dataset.py``),

    w(x, y, z) = exp(-((|x| - 0.30)^2 + (y - 0.05)^2 + (z + 0.35)^2) / (2 sigma^2))

on its [-1, 1] grid, with sigma = the run's ``--synthetic-temporal-sigma`` (default 0.18). At
sigma 0.09 GAP of an untrained backbone reads the factor at held-out R² ~0, while pooling the
same features under w reads ~0.55, so a ``--global-pool attention`` run gains on this factor by
moving a head onto that site. The trainer logs each head's effective fraction, which says a
head sharpened but not where it went: onto the temporal site, or onto brain-size and shape cues
such as the WM/GM boundary the site lies on.

This script renders the first ``--num-samples`` subjects of the run's own validation split (the
images the trainer evaluated), takes each head's attention map on the backbone grid per view
(``model.attention_maps``), averages it over subjects, and compares it with w, average-pooled onto
the same grid and normalised to sum 1:

overlap             sum_p min(a(p), t(p)) of the two normalised maps: 1 = identical, 0 = disjoint
                    (one minus their total-variation distance). Reported for the subject-mean
                    map. The per-subject overlaps give the spread.
null, shell null    the same overlap against same-shape bilateral templates centred farther than
                    ``--null-min-distance`` sigma (default 3) from the true centre: at random
                    in-brain voxels (``null``), and at random in-brain points on the sphere through
                    the true centre (``shell_null``). The site sits on the WM/GM boundary, so a
                    head that attends to that boundary everywhere (a brain-size cue) can beat the
                    in-brain null but not the shell null. Median and 90th percentile over
                    ``--null-templates`` draws (default 20) of each.
effective fraction  exp(entropy) / N of a head's map, averaged over subjects, as the trainer's
                    ``attention_spread`` computes it: 1 = uniform (GAP, as at step 0), 1/N = one
                    cell. Also given for the subject-mean map and for the template itself.
centre of mass      (x, y, z) of the mean map in the generator's [-1, 1] coordinates, and its
                    folded form (|x|, y, z). The template is bilateral, so its x is ~0 and its
                    |x| ~0.30; a head on one hemisphere's lobe has x ~ +-|x|. ``com_distance`` is
                    the folded distance to the template's own pooled centre of mass, also in
                    units of sigma.
mean radius         the map's weighted distance from the grid centre: ~0.46 for the template, which
                    lies on the WM/GM boundary; a boundary-shell head has the same radius but a
                    folded centre of mass near (0.23, 0, 0).
brain mass          the share of the head's weight on brain voxels, each cell weighted by its
                    brain-voxel fraction.

``reading`` condenses these into a label: ``uniform`` when the head barely left GAP (effective
fraction >= 0.9), ``prefers site`` when the mean map beats both nulls' 90th percentile, and
``no site preference`` otherwise. The numbers are the result; the label only sorts them. The
``uniform`` row is the untrained floor: both pool parameters start at zero, so every head is
exactly uniform at step 0. The ``template`` row is a head that sits exactly on the site.

``pool`` reports each head's query norm (the sharpness of its softmax) and, under
``--attention-pool-width full``, the norm of the C x C mix that carries its pooled vector into the
code (``mix_norm_init`` at step 0): a head the mix has shrunk away is not read, whatever its map.

Coordinates: cell i of the stride-4 conv backbone is centred on voxels 4i..4i+3 (two kernel-4,
stride-2, padding-1 convolutions), so a cell's coordinate is the generator grid averaged over its
block, the same pooling the template gets.

Only encoder-only runs with ``--global-pool attention`` on the synthetic generator have heads on a
known site; GAP and real-data runs are refused. Writes the JSON report, the subject-mean and
per-subject SD maps beside it (``*_maps.npz``), and with ``--png`` axial and coronal slices
through the site of every head's mean map, with the template's 0.1 contour and a ``.csv`` twin.

Example:
    python -m eval.diagnostics.attention_template_overlap \
        --run-dir results/encoder/runs/conv_mlp_temporal009_attnfull_s42 --png figures/attention_site.png
    python -m eval.diagnostics.attention_template_overlap --self-test
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

# Folded (|x|, y, z) centre of the generator's temporal-atrophy bump (render_structure).
TEMPORAL_CENTRE = (0.30, 0.05, -0.35)
# View 1 is T1 and view 2 FLAIR, named as the trainer's attention_spread names them.
VIEWS = ("t1", "flair")
# A head that keeps this share of the grid's effective size has barely left GAP.
UNIFORM_FRACTION = 0.9
CSV_COLUMNS = (
    "view",
    "head",
    "reading",
    "overlap",
    "overlap_subject_mean",
    "overlap_subject_sd",
    "null_median",
    "null_p90",
    "shell_null_median",
    "shell_null_p90",
    "effective_fraction",
    "effective_fraction_sd",
    "effective_fraction_mean_map",
    "com_x",
    "com_y",
    "com_z",
    "com_abs_x",
    "com_distance",
    "com_distance_sigma",
    "mean_radius",
    "brain_mass",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Encoder-only run directory holding settings.json and model.pt")
    p.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename inside --run-dir")
    p.add_argument(
        "--num-samples", type=int, default=200, help="Read the first N subjects of the run's own validation split"
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="mps needs PYTORCH_ENABLE_MPS_FALLBACK=1 exported before Python starts",
    )
    p.add_argument("--null-templates", type=int, default=20, help="Same-shape templates drawn for each null")
    p.add_argument(
        "--null-min-distance",
        type=float,
        default=3.0,
        help="Null centres lie farther than this many sigma from the true centre",
    )
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Report JSON (default: <run-dir>/attention_template_overlap.json)")
    p.add_argument("--png", default=None, help="Also draw each head's mean map through the site; writes a .csv twin")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.num_samples < 1 or args.batch_size < 1 or args.null_templates < 1:
        p.error("--num-samples, --batch-size and --null-templates must be positive")
    if not 0 <= args.null_min_distance < float("inf"):
        p.error("--null-min-distance must be finite and nonnegative")
    return args


def generator_axis(res):
    """The generator's coordinates along one axis: torch.linspace(-1, 1, res)."""
    return np.linspace(-1.0, 1.0, res)


def temporal_template(res, sigma, centre=TEMPORAL_CENTRE):
    """The temporal-atrophy weight on the generator's (res, res, res) grid, indexed (x, y, z).

    ``render_structure``'s formula: one Gaussian in (|x|, y, z), so it is bilateral. Another
    ``centre`` gives the same shape elsewhere, for the nulls. Centring local deformations (off in
    every encoder-only run) would subtract a constant inside the brain, which does not move the site.
    """
    g = generator_axis(res)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    cx, cy, cz = centre
    return np.exp(-((np.abs(x) - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma**2))


def pooling_matrix(size, bins):
    """(bins, size) rows averaging each bin, with the bin edges of ``F.adaptive_avg_pool3d``."""
    rows = np.zeros((bins, size))
    for i in range(bins):
        start, end = (i * size) // bins, ((i + 1) * size + bins - 1) // bins
        rows[i, start:end] = 1.0 / (end - start)
    return rows


def pool_to_grid(volume, grid):
    """Average-pool a (D, H, W) volume onto ``grid`` as ``F.adaptive_avg_pool3d`` does."""
    px, py, pz = (pooling_matrix(size, bins) for size, bins in zip(volume.shape, grid))
    return np.einsum("ai,bj,ck,ijk->abc", px, py, pz, volume, optimize=True)


def grid_template(res, sigma, grid, centre=TEMPORAL_CENTRE):
    """``temporal_template`` average-pooled onto the backbone grid and normalised to sum 1."""
    pooled = pool_to_grid(temporal_template(res, sigma, centre), grid)
    return pooled / pooled.sum()


def cell_coordinates(res, grid):
    """(3, *grid) generator (x, y, z) of each backbone cell: the mean over the voxels it pools."""
    g = generator_axis(res)
    return np.stack(np.meshgrid(*((pooling_matrix(res, bins) * g).sum(1) for bins in grid), indexing="ij"))


def overlap(maps, template):
    """sum_p min(a(p), t(p)) over the last three axes: 1 for identical maps, 0 for disjoint ones."""
    return np.minimum(maps, template).sum(axis=(-3, -2, -1))


def effective_fraction(maps):
    """exp(entropy) / N over the last three axes, as the trainer's ``attention_spread`` computes it."""
    flat = maps.reshape(*maps.shape[:-3], -1)
    entropy = -(flat * np.log(np.where(flat > 0, flat, 1.0))).sum(-1)
    return np.exp(entropy) / flat.shape[-1]


def centre_of_mass(maps, coords):
    """(..., 3) weighted mean (x, y, z) of each map, and the folded (|x|, y, z) one."""
    folded = coords.copy()
    folded[0] = np.abs(folded[0])
    mass = maps.sum(axis=(-3, -2, -1))[..., None]
    return np.einsum("...ijk,cijk->...c", maps, coords) / mass, np.einsum("...ijk,cijk->...c", maps, folded) / mass


def mean_radius(maps, coords):
    """Weighted mean distance of each map from the grid centre."""
    return (maps * np.linalg.norm(coords, axis=0)).sum(axis=(-3, -2, -1)) / maps.sum(axis=(-3, -2, -1))


def in_brain_centres(voxel_brain, sigma, count, min_distance, seed):
    """Folded centres (x >= 0) at random brain voxels, farther than ``min_distance`` sigma from the site."""
    g = generator_axis(voxel_brain.shape[0])
    points = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)[voxel_brain >= 0.5]
    far = np.linalg.norm(points - np.asarray(TEMPORAL_CENTRE), axis=1) > min_distance * sigma
    points = points[(points[:, 0] >= 0) & far]
    pick = np.random.default_rng(seed).choice(len(points), size=min(count, len(points)), replace=False)
    return points[pick]


def shell_centres(voxel_brain, sigma, count, min_distance, seed):
    """Folded centres on the sphere through the site, in the brain, farther than ``min_distance`` sigma from it."""
    res = voxel_brain.shape[0]
    centre = np.asarray(TEMPORAL_CENTRE)
    points = np.random.default_rng(seed).standard_normal((max(4096, 200 * count), 3))
    points *= np.linalg.norm(centre) / np.linalg.norm(points, axis=1, keepdims=True)
    points[:, 0] = np.abs(points[:, 0])
    index = np.clip(np.rint((points + 1) * (res - 1) / 2).astype(int), 0, res - 1)
    inside = voxel_brain[index[:, 0], index[:, 1], index[:, 2]] >= 0.5
    far = np.linalg.norm(points - centre, axis=1) > min_distance * sigma
    return points[inside & far][:count]


def describe(mean_map, template, coords, nulls, template_com, sigma):
    """One map (a head's subject mean, or a reference) against the template and both nulls."""
    com, folded = centre_of_mass(mean_map, coords)
    distance = float(np.linalg.norm(folded - template_com))
    row = {"overlap": float(overlap(mean_map, template))}
    for name, stack in nulls.items():
        scores = overlap(mean_map, stack)
        row[f"{name}_median"] = float(np.median(scores))
        row[f"{name}_p90"] = float(np.percentile(scores, 90))
    row.update(
        effective_fraction_mean_map=float(effective_fraction(mean_map)),
        com=[float(v) for v in com],
        com_folded=[float(v) for v in folded],
        com_distance=distance,
        com_distance_sigma=distance / sigma,
        mean_radius=float(mean_radius(mean_map, coords)),
    )
    return row


def spread(subject_maps, brain, template, coords):
    """One head's per-subject numbers, as means and SDs over subjects."""
    per_subject = overlap(subject_maps, template)
    fraction = effective_fraction(subject_maps)
    _, folded = centre_of_mass(subject_maps, coords)
    return {
        "overlap_subject_mean": float(per_subject.mean()),
        "overlap_subject_sd": float(per_subject.std()),
        "overlap_subject_p10": float(np.percentile(per_subject, 10)),
        "overlap_subject_p90": float(np.percentile(per_subject, 90)),
        "effective_fraction": float(fraction.mean()),
        "effective_fraction_sd": float(fraction.std()),
        "com_folded_subject_sd": [float(v) for v in folded.std(0)],
        "brain_mass": float((subject_maps * brain).sum(axis=(-3, -2, -1)).mean()),
    }


def reading(row):
    if row["effective_fraction"] >= UNIFORM_FRACTION:
        return "uniform"
    if row["overlap"] > row["null_p90"] and row["overlap"] > row["shell_null_p90"]:
        return "prefers site"
    return "no site preference"


def analyse(collected, res, sigma, null_count=20, min_distance=3.0, seed=0):
    """Score every head and view against the template; returns the report and the maps to draw.

    ``collected`` holds ``maps`` and ``brain`` as (view 1, view 2) arrays of shape
    (n, heads, D, H, W) and (n, D, H, W), and ``voxel_brain``, the mean brain mask at full resolution.
    """
    maps = collected["maps"]
    grid = tuple(maps[0].shape[2:])
    coords = cell_coordinates(res, grid)
    template = grid_template(res, sigma, grid)
    template_com = centre_of_mass(template, coords)[1]
    centres = {
        "null": in_brain_centres(collected["voxel_brain"], sigma, null_count, min_distance, (seed, 0)),
        "shell_null": shell_centres(collected["voxel_brain"], sigma, null_count, min_distance, (seed, 1)),
    }
    for name, points in centres.items():
        if len(points) == 0:
            raise ValueError(f"No {name} centre lies in the brain farther than {min_distance:g} sigma from the site")
    nulls = {
        name: np.stack([grid_template(res, sigma, grid, tuple(point)) for point in points])
        for name, points in centres.items()
    }
    cell_brain = np.mean([brain.mean(0) for brain in collected["brain"]], axis=0)
    references = {}
    for name, ref in (("template", template), ("uniform", np.full(grid, 1.0 / np.prod(grid)))):
        row = describe(ref, template, coords, nulls, template_com, sigma)
        row.update(effective_fraction=row["effective_fraction_mean_map"], brain_mass=float((ref * cell_brain).sum()))
        references[name] = row
    rows = []
    for v, view in enumerate(VIEWS):
        mean = maps[v].mean(0)
        for head in range(mean.shape[0]):
            row = {"view": view, "head": head}
            row.update(describe(mean[head], template, coords, nulls, template_com, sigma))
            row.update(spread(maps[v][:, head], collected["brain"][v], template, coords))
            row["reading"] = reading(row)
            rows.append(row)
    report = {
        "sigma": float(sigma),
        "res": int(res),
        "grid": list(grid),
        "num_subjects": int(maps[0].shape[0]),
        "template": {"centre_folded": list(TEMPORAL_CENTRE), **references["template"]},
        "uniform": references["uniform"],
        "nulls": {
            name: {
                "requested": int(null_count),
                "count": int(len(points)),
                "min_distance_sigma": float(min_distance),
                "seed": int(seed),
                "centres_folded": points.tolist(),
            }
            for name, points in centres.items()
        },
        "heads": rows,
    }
    arrays = {
        "mean_maps": np.stack([m.mean(0) for m in maps]),
        "sd_maps": np.stack([m.std(0) for m in maps]),
        "template": template,
        "coords": coords,
        "voxel_brain": collected["voxel_brain"],
    }
    return report, arrays


def collect_maps(model, dataset, device, batch_size):
    """Each subject's attention maps and per-cell brain fraction on the backbone grid, per view.

    Views are routed as in training (``n_views=2``). ``voxel_brain`` is view 1's brain mask
    averaged over subjects at full resolution; the in-brain null centres are drawn from it.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    maps, brain = ([], []), ([], [])
    voxel_brain, n = 0.0, 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            weights = model.attention_maps(torch.cat(batch["image"], dim=0).to(device), n_views=2).cpu().double()
            b = weights.shape[0] // 2
            for v in range(2):
                maps[v].append(weights[v * b : (v + 1) * b].numpy())
                mask = batch["mask"][v].double()
                brain[v].append(F.adaptive_avg_pool3d(mask, tuple(weights.shape[2:]))[:, 0].numpy())
            voxel_brain = voxel_brain + batch["mask"][0].double()[:, 0].sum(0).numpy()
            n += b
    return {
        "maps": tuple(np.concatenate(parts) for parts in maps),
        "brain": tuple(np.concatenate(parts) for parts in brain),
        "voxel_brain": voxel_brain / n,
    }


def pool_parameters(pool):
    """The attention pool's settings and per-head parameter norms."""
    out = {
        "width": pool.width,
        "heads": pool.num_heads,
        "frequencies": pool.num_frequencies,
        "channels": pool.channels,
        "query_norm": pool.query.detach().cpu().norm(dim=1).tolist(),
        "position_norm": None if pool.position is None else float(pool.position.detach().cpu().norm()),
        "mix_norm": None,
        "mix_norm_init": None,
    }
    if pool.mix is not None:
        out["mix_norm"] = pool.mix.detach().cpu().flatten(1).norm(dim=1).tolist()
        out["mix_norm_init"] = math.sqrt(pool.channels // pool.num_heads)
    return out


def refusal(cfg):
    """Why this run has no attention heads on a known temporal-atrophy site, or None."""
    if cfg.get("dataset_name", "synthetic") != "synthetic":
        return (
            f"This run trained on real data ({cfg['dataset_name']}): there is no generator, so there is no "
            "temporal-atrophy site to compare its heads with"
        )
    if cfg.get("global_pool", "gap") != "attention":
        return "This run pools globally with GAP: every position weighs 1/N, so there are no attention heads to place"
    if cfg.get("synthetic_mode", "pseudo_mri") != "pseudo_mri":
        return f"synthetic_mode={cfg['synthetic_mode']!r} renders no anatomy, so it has no temporal-atrophy site"
    if cfg.get("n_content", 0) < 7:
        return (
            f"n_content={cfg.get('n_content')}: temporal atrophy is z_content[6], which the renderer pads with "
            "zero here, so these subjects have none"
        )
    return None


def table_rows(report):
    """CSV rows: one per view and head, then the template and uniform references."""
    out = []
    labelled = [(row["view"], row["head"], row) for row in report["heads"]]
    labelled += [(name, "", report[name]) for name in ("template", "uniform")]
    for view, head, row in labelled:
        flat = dict(row, view=view, head=head, reading=row.get("reading", ""))
        flat.update(zip(("com_x", "com_y", "com_z"), row["com"]))
        flat["com_abs_x"] = row["com_folded"][0]
        out.append([flat.get(key, "") for key in CSV_COLUMNS])
    return out


def print_report(report):
    grid = "x".join(str(g) for g in report["grid"])
    print(
        f"\n=== global attention vs the temporal-atrophy template (sigma {report['sigma']:g}, {grid} grid, "
        f"{report['num_subjects']} val subjects) ===",
        flush=True,
    )
    for name, null in report["nulls"].items():
        if null["count"] < null["requested"]:
            print(f"  WARNING: only {null['count']} of {null['requested']} {name} centres could be drawn", flush=True)
    print(
        f"  {'view':<9s}{'head':>4s}{'overlap':>9s}{'subjects':>13s}{'null p50/p90':>15s}{'shell p50/p90':>15s}"
        f"{'eff.':>7s}{'CoM |x|, y, z':>22s}{'dist/σ':>8s}{'radius':>8s}{'brain':>7s}  reading",
        flush=True,
    )

    def line(label, head, row):
        subjects = (
            f"{row['overlap_subject_mean']:.3f}±{row['overlap_subject_sd']:.3f}"
            if "overlap_subject_mean" in row
            else ""
        )
        com = "(" + ", ".join(f"{v:+.2f}" for v in row["com_folded"]) + ")"
        return (
            f"  {label:<9s}{head:>4s}{row['overlap']:>9.3f}{subjects:>13s}"
            f"{row['null_median']:>9.3f}/{row['null_p90']:.3f}{row['shell_null_median']:>9.3f}/{row['shell_null_p90']:.3f}"
            f"{row['effective_fraction']:>7.3f}{com:>22s}{row['com_distance_sigma']:>8.2f}{row['mean_radius']:>8.2f}"
            f"{row['brain_mass']:>7.2f}  {row.get('reading', '')}"
        )

    for row in report["heads"]:
        print(line(row["view"], str(row["head"]), row), flush=True)
    for name in ("template", "uniform"):
        print(line(name, "", report[name]), flush=True)
    pool = report.get("pool")
    if pool is not None:
        text = "  |q| per head: " + " ".join(f"{v:.2f}" for v in pool["query_norm"])
        if pool["mix_norm"] is not None:
            text += "   |mix| per head: " + " ".join(f"{v:.2f}" for v in pool["mix_norm"])
            text += f" (step 0: {pool['mix_norm_init']:.2f})"
        print(text, flush=True)
    print(
        "  overlap = sum of min(map, template), 1 = identical; 'subjects' is its per-subject mean ± SD. A head\n"
        "  prefers the site when it beats both nulls' p90; the shell null lies on the site's sphere (the WM/GM\n"
        "  boundary), which a brain-size or boundary head matches as well as the site. 'uniform' is the step-0 floor.",
        flush=True,
    )


def write_outputs(report, arrays, out):
    """The JSON report, and the subject-mean and SD maps beside it."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump(report, fp, indent=2, default=float)
    maps_path = os.path.splitext(out)[0] + "_maps.npz"
    np.savez_compressed(
        maps_path,
        views=np.array(VIEWS),
        mean_maps=arrays["mean_maps"].astype(np.float32),
        sd_maps=arrays["sd_maps"].astype(np.float32),
        template=arrays["template"].astype(np.float32),
        cell_coordinates=arrays["coords"].astype(np.float32),
        voxel_brain=np.asarray(arrays["voxel_brain"], dtype=np.float32),
    )
    print(f"\nwrote {out}\nwrote {maps_path}", flush=True)
    return out, maps_path


def render_figure(report, arrays, path):
    """Axial and coronal slices through the site of each head's mean map, per view, plus a .csv twin.

    Each map is drawn as weight relative to uniform (1 = GAP), on one colour scale per head. The
    orange line is the template's 0.1-of-peak contour on that plane, the grey one the mean brain mask.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    from eval.plots.plot_identifiability import THEME, _save, _write_csv

    t = THEME["light"]
    ramp = LinearSegmentedColormap.from_list(
        "weight", [t["surface"], "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    )
    sigma, coords, voxel_brain = report["sigma"], arrays["coords"], np.asarray(arrays["voxel_brain"])
    relative = arrays["mean_maps"] * np.prod(report["grid"])
    n_heads = relative.shape[1]
    axis_values = (coords[0][:, 0, 0], coords[1][0, :, 0], coords[2][0, 0, :])
    # (name, fixed axis, its index, vertical axis): axial fixes z and shows (x, y); coronal fixes y and shows (x, z).
    planes = (
        ("axial", 2, int(np.argmin(np.abs(axis_values[2] - TEMPORAL_CENTRE[2]))), 1),
        ("coronal", 1, int(np.argmin(np.abs(axis_values[1] - TEMPORAL_CENTRE[1]))), 2),
    )
    rows = {(row["view"], row["head"]): row for row in report["heads"]}
    fine = np.linspace(-1.0, 1.0, 241)
    voxels = generator_axis(voxel_brain.shape[0])

    def extent(axis):
        values = axis_values[axis]
        half = (values[1] - values[0]) / 2 if len(values) > 1 else 1.0
        return [values[0] - half, values[-1] + half]

    def plane_grid(fixed, value, vertical, along):
        u, w = np.meshgrid(along, along, indexing="ij")
        position = [u, None, None]
        position[vertical], position[fixed] = w, np.full_like(u, value)
        return u, w, position

    fig, axes = plt.subplots(
        n_heads, 4, figsize=(9.6, 2.35 * n_heads + 0.9), squeeze=False, constrained_layout=True, facecolor=t["surface"]
    )
    for head in range(n_heads):
        panels = [(v, plane) for v in range(2) for plane in planes]
        vmax = max(2.0, max(float(np.take(relative[v, head], p[2], axis=p[1]).max()) for v, p in panels))
        for col, (v, (name, fixed, index, vertical)) in enumerate(panels):
            ax = axes[head, col]
            value = axis_values[fixed][index]
            image = np.take(relative[v, head], index, axis=fixed)
            im = ax.imshow(
                image.T,
                origin="lower",
                extent=extent(0) + extent(vertical),
                cmap=ramp,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            u, w, position = plane_grid(fixed, value, vertical, fine)
            folded = (np.abs(position[0]), position[1], position[2])
            weight = np.exp(-sum((p - c) ** 2 for p, c in zip(folded, TEMPORAL_CENTRE)) / (2 * sigma**2))
            if weight.max() > 0.1:
                ax.contour(u, w, weight, levels=[0.1], colors=[t["series"][1]], linewidths=1.2)
            mask = np.take(voxel_brain, int(np.argmin(np.abs(voxels - value))), axis=fixed)
            if mask.min() < 0.5 < mask.max():
                mu, mw = np.meshgrid(voxels, voxels, indexing="ij")
                ax.contour(mu, mw, mask, levels=[0.5], colors=[t["muted"]], linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(t["grid"])
            row = rows[(VIEWS[v], head)]
            if name == "axial":
                note = f"overlap {row['overlap']:.2f} | p90 {row['null_p90']:.2f}, shell {row['shell_null_p90']:.2f}"
            else:
                note = f"{row['reading']} | eff. {row['effective_fraction']:.2f}"
            plane = f"{VIEWS[v].upper()} {name}, {'z' if fixed == 2 else 'y'} = {value:+.2f}"
            ax.set_title(f"{plane}\n{note}", color=t["ink2"], fontsize=7, pad=3, loc="left")
            if head == n_heads - 1:
                ax.set_xlabel("x (L-R)", color=t["muted"], fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"head {head}\ny", color=t["ink2"], fontsize=8)
            elif col == 2:
                ax.set_ylabel("y", color=t["muted"], fontsize=7.5)
            else:
                ax.set_ylabel("z (I-S)", color=t["muted"], fontsize=7.5)
        bar = fig.colorbar(im, ax=axes[head, :].tolist(), fraction=0.025, pad=0.01)
        bar.set_label("× uniform", color=t["muted"], fontsize=7.5)
        bar.ax.tick_params(colors=t["muted"], labelsize=7, length=0)
        bar.outline.set_visible(False)
    fig.suptitle(
        f"Attention heads vs the temporal-atrophy site (sigma {sigma:g}, {report['num_subjects']} val subjects)",
        color=t["ink"],
        fontsize=10,
    )
    handles = [
        Line2D([], [], color=t["series"][1], linewidth=1.2, label="template, 0.1 of peak"),
        Line2D([], [], color=t["muted"], linewidth=0.6, label="brain (mean mask)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.5, labelcolor=t["ink2"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _save(fig, path, t)
    csv_path = (path[: -len(".png")] if path.endswith(".png") else path) + ".csv"
    _write_csv(csv_path, CSV_COLUMNS, table_rows(report))
    return [path, csv_path]


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        raise SystemExit("--run-dir is required (or pass --self-test)")

    import torch
    from torch.utils.data import Subset

    from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset
    from utils.encoder_runtime import configure_encoder_runtime, select_encoder_device

    cfg = load_settings(args.run_dir)
    reason = refusal(cfg)
    if reason is not None:
        raise SystemExit(reason)
    configure_encoder_runtime(cfg)
    device = select_encoder_device(args.device, False)
    sigma = float(cfg.get("synthetic_temporal_sigma", 0.18))
    print(f"run     : {args.run_dir}", flush=True)
    print(
        f"device  : {device}   pool: attention, width {cfg.get('attention_pool_width', 'slice')}, "
        f"{cfg.get('attention_pool_heads', 4)} heads",
        flush=True,
    )
    print(f"template: temporal atrophy, sigma {sigma:g}, res {cfg['res']}, {args.num_samples} val subjects", flush=True)

    state = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location="cpu")
    model = build_model(cfg, device, state)
    del state
    # fixed_reference standardises every volume with constants fitted to the first 64 subjects of the
    # split being built, so a smaller split would feed these subjects to the model differently from
    # training. Build the run's own validation split and read its first --num-samples subjects.
    split = make_dataset(cfg, max(args.num_samples, cfg.get("num_val_samples", 400)), mode="val")
    collected = collect_maps(model, Subset(split, range(args.num_samples)), device, args.batch_size)
    report, arrays = analyse(collected, cfg["res"], sigma, args.null_templates, args.null_min_distance, args.null_seed)
    report = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "split": "val",
        **report,
        "pool": pool_parameters(model.attention_pool),
        "settings": cfg,
    }
    print_report(report)
    write_outputs(report, arrays, args.out or os.path.join(args.run_dir, "attention_template_overlap.json"))
    if args.png:
        for path in render_figure(report, arrays, args.png):
            print(f"wrote {path}", flush=True)


def _self_test():
    """Synthetic maps first (numpy only), then the generator, the pool and the CLI (torch)."""
    res, sigma, grid = 64, 0.09, (16, 16, 16)
    cells = int(np.prod(grid))
    coords = cell_coordinates(res, grid)
    template = grid_template(res, sigma, grid)
    assert np.isclose(template.sum(), 1.0) and np.allclose(template, template[::-1])
    com, folded = centre_of_mass(template, coords)
    assert abs(com[0]) < 1e-9 and np.allclose(folded, TEMPORAL_CENTRE, atol=0.03), (com, folded)
    assert np.isclose(overlap(template, template), 1.0)
    uniform = np.full(grid, 1.0 / cells)
    point = np.zeros(grid)
    point[3, 7, 11] = 1.0
    assert np.isclose(effective_fraction(uniform), 1.0) and np.isclose(effective_fraction(point), 1.0 / cells)
    assert np.allclose(centre_of_mass(point, coords)[0], coords[:, 3, 7, 11])
    assert np.isclose(overlap(point, uniform), 1.0 / cells)
    assert np.allclose(pooling_matrix(10, 4).sum(1), 1.0) and (pooling_matrix(10, 4) > 0).sum() == 12

    g = generator_axis(res)
    voxel_brain = (np.linalg.norm(np.stack(np.meshgrid(g, g, g, indexing="ij")), axis=0) < 0.65).astype(float)
    for draw in (in_brain_centres, shell_centres):
        points = draw(voxel_brain, sigma, 20, 3.0, (0, 0))
        assert len(points) == 20 and (points[:, 0] >= 0).all()
        assert (np.linalg.norm(points - np.asarray(TEMPORAL_CENTRE), axis=1) > 3 * sigma).all()
    shell_points = shell_centres(voxel_brain, sigma, 20, 3.0, (0, 1))
    assert np.allclose(np.linalg.norm(shell_points, axis=1), np.linalg.norm(TEMPORAL_CENTRE))

    # Four heads with known answers: the template, uniform (step 0), one lobe, and the site's sphere.
    one_lobe = template * (coords[0] > 0)
    shell = np.exp(-((np.linalg.norm(coords, axis=0) - np.linalg.norm(TEMPORAL_CENTRE)) ** 2) / (2 * sigma**2))
    heads = np.stack([template, uniform, one_lobe / one_lobe.sum(), shell / shell.sum()])
    cell_brain = np.repeat(pool_to_grid(voxel_brain, grid)[None], 3, axis=0)
    collected = {
        "maps": (np.repeat(heads[None], 3, axis=0),) * 2,
        "brain": (cell_brain,) * 2,
        "voxel_brain": voxel_brain,
    }
    report, _ = analyse(collected, res, sigma)
    on, flat, lobe, sphere = (row for row in report["heads"] if row["view"] == "t1")
    assert on["reading"] == "prefers site" and np.isclose(on["overlap"], 1.0) and on["com_distance_sigma"] < 1e-6
    assert on["overlap_subject_sd"] < 1e-12 and np.isclose(on["brain_mass"], report["template"]["brain_mass"])
    assert flat["reading"] == "uniform" and abs(flat["overlap"] - flat["null_median"]) < 0.005, flat
    assert np.isclose(flat["overlap"], report["uniform"]["overlap"])
    assert lobe["reading"] == "prefers site" and abs(lobe["overlap"] - 0.5) < 0.01, lobe
    assert np.isclose(lobe["com"][0], lobe["com_folded"][0]) and abs(lobe["com"][0] - TEMPORAL_CENTRE[0]) < 0.03
    assert sphere["reading"] == "no site preference" and sphere["overlap"] > sphere["null_median"], sphere
    assert abs(sphere["mean_radius"] - report["template"]["mean_radius"]) < 0.05 and sphere["com_distance_sigma"] > 2
    assert np.isclose(report["template"]["overlap"], 1.0) and report["template"]["null_p90"] < 0.2
    assert len(table_rows(report)) == 2 * 4 + 2
    print("self-test: synthetic-map checks OK (numpy only)", flush=True)

    if importlib.util.find_spec("torch") is None:
        print("self-test: torch is not installed; the generator, pool and CLI checks were skipped", flush=True)
        return
    _self_test_torch(res, sigma, grid, template)
    print(
        "self-test OK (template on the generator's z6 change, pooling matches torch, a positional head "
        "fitted to the template reads as on the site through the CLI, untrained heads read as uniform)",
        flush=True,
    )


def _self_test_torch(res, sigma, grid, template):
    import tempfile

    import torch
    import torch.nn.functional as F

    from eval.protocol.score_checkpoint import build_model, make_dataset
    from models.attention_pool import fourier_positions

    # pool_to_grid is adaptive_avg_pool3d, on divisible and uneven grids.
    volume = np.random.default_rng(0).random((20, 17, 9))
    for bins in ((5, 4, 3), (7, 5, 4)):
        expected = F.adaptive_avg_pool3d(torch.from_numpy(volume)[None, None], bins)[0, 0].numpy()
        assert np.allclose(pool_to_grid(volume, bins), expected)

    # The generator's z6 change lies under this template, in this axis order.
    data = {
        "res": res,
        "n_content": 9,
        "n_style": 3,
        "synthetic_mode": "pseudo_mri",
        "synthetic_normalize": "fixed_reference",
        "synthetic_clean_content": True,
        "synthetic_temporal_sigma": sigma,
        "data_seed": 0,
    }
    dataset = make_dataset(data, 1, mode="val")
    inner, latents = dataset._inner, dataset[0]["gt_latents"]
    views = []
    for value in (-2.0, 2.0):
        z = latents["z_content"].clone()
        z[6] = value
        views.append(
            inner.render_pseudo_mri(
                z,
                latents["z_deformation"],
                latents["z_fissure"],
                latents["z_style_v1"],
                latents["z_style_v2"],
                inner.sample_seed_for(0),
                z_lesion=latents.get("z_lesion"),
            )
        )
    weight = temporal_template(res, sigma)
    for v in range(2):
        energy = (views[1][v] - views[0][v]).squeeze().double().numpy() ** 2
        inside = energy[weight > 0.05].sum() / energy.sum()
        swapped = max(
            energy[np.transpose(weight, p) > 0.05].sum() / energy.sum() for p in ((1, 0, 2), (2, 1, 0), (0, 2, 1))
        )
        assert inside > 0.9 and swapped < 0.75, (v, inside, swapped)

    # An untrained full-width pool is exact GAP; then fit head 0's positional logits to the template.
    cfg = dict(
        data,
        hidden_channels=64,
        res_channels=8,
        nb_res_layers=1,
        downscale_factor=4,
        latent_dim=12,
        content_channels=9,
        seed=0,
        conv_readout="mlp",
        global_pool="attention",
        attention_pool_width="full",
        attention_pool_heads=4,
        attention_pool_frequencies=4,
        num_val_samples=4,
    )
    model = build_model(cfg, "cpu")
    pool = model.attention_pool
    dataset = make_dataset(cfg, 4, mode="val")
    collected = collect_maps(model, dataset, "cpu", 4)
    assert collected["maps"][0].shape == (4, 4, *grid) and np.allclose(collected["maps"][0], 1.0 / np.prod(grid))
    assert 0.1 < collected["voxel_brain"].mean() < 0.9
    report, _ = analyse(collected, res, sigma)
    assert all(row["reading"] == "uniform" for row in report["heads"])

    phi = fourier_positions(grid, pool.num_frequencies, dtype=torch.float64)
    target = torch.from_numpy(template.reshape(-1))
    logits = torch.zeros(phi.shape[1], dtype=torch.float64, requires_grad=True)
    fit = torch.optim.LBFGS([logits], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-12)

    def closure():
        fit.zero_grad()
        loss = -(target * torch.log_softmax(phi @ logits, 0)).sum()
        loss.backward()
        return loss

    fit.step(closure)
    fitted = torch.softmax(phi @ logits, 0).detach().numpy().reshape(grid)
    assert overlap(fitted, template) > 0.99
    # A token of +-s on each Fourier feature has a constant norm (sin^2 + cos^2 = 1 per feature
    # pair), so the key LayerNorm is a fixed rescaling and the head's logits are exactly
    # logits . phi(p); at s = 1e5 the backbone features shift them by ~1e-3.
    k, c = phi.shape[1], pool.channels
    with torch.no_grad():
        pool.position.zero_()
        pool.position[:k] = 1e5 * torch.eye(k)
        pool.position[k : 2 * k] = -1e5 * torch.eye(k)
        half = (logits.detach() * math.sqrt(k / c) / 2).float()
        pool.query[0, :k], pool.query[0, k : 2 * k] = half, -half

    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        with open(os.path.join(run, "settings.json"), "w") as fp:
            json.dump(cfg, fp)
        torch.save(model.state_dict(), os.path.join(run, "model.pt"))
        png = os.path.join(tmp, "figures", "attention_site.png")
        main(["--run-dir", run, "--num-samples", "4", "--batch-size", "2", "--device", "cpu", "--png", png])
        with open(os.path.join(run, "attention_template_overlap.json")) as fp:
            saved = json.load(fp)
        assert saved["pool"]["width"] == "full" and saved["settings"]["attention_pool_width"] == "full"
        for row in saved["heads"]:
            if row["head"] == 0:
                assert row["reading"] == "prefers site" and row["overlap"] > 0.98, row
                assert row["com_distance_sigma"] < 0.5 and row["effective_fraction"] < 0.05, row
            else:
                assert row["reading"] == "uniform" and row["effective_fraction"] > 0.999, row
        maps = np.load(os.path.join(run, "attention_template_overlap_maps.npz"))
        assert maps["mean_maps"].shape == (2, 4, *grid) and np.allclose(
            maps["mean_maps"].sum((2, 3, 4)), 1.0, atol=1e-4
        )
        assert os.path.isfile(png) and os.path.isfile(png[: -len(".png")] + ".csv")
        # The same subjects and effective fractions the trainer logs for this checkpoint.
        from training.main_conv_synthetic import attention_spread

        logged = attention_spread(model, dataset, "cpu", 4)
        assert all(abs(row["effective_fraction"] - logged[row["view"]][row["head"]]) < 1e-6 for row in saved["heads"])

        for change, message in (
            ({"global_pool": "gap"}, "GAP"),
            ({"dataset_name": "ADNI_stripped_masks"}, "real data"),
            ({"n_content": 6}, "z_content[6]"),
        ):
            with open(os.path.join(run, "settings.json"), "w") as fp:
                json.dump(dict(cfg, **change), fp)
            try:
                main(["--run-dir", run])
            except SystemExit as exc:
                assert message in str(exc), exc
            else:
                raise AssertionError(f"{change} was not refused")


if __name__ == "__main__":
    main()
