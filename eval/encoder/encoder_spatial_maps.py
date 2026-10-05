"""Final-stage spatial maps of a frozen encoder-only checkpoint.

python -m eval.encoder.encoder_spatial_maps --run-dir RUN --out-dir NEW_OUTPUT

Stages: ``backbone`` is the encoder's last feature map (16³ x 64 for the Conv recipe),
``projected`` the active spatial content head applied at every native cell, and
``global`` the content vector the model outputs (GAP, then the global head).

1. Gallery: what the maps look like for a few test subjects.
2. Factor responses: change one content factor by -/+delta with anatomy, acquisition
   draws and the normalization affine fixed; map where the final features change and
   how much of that change survives spatial averaging.
3. Decodability: per-cell ridge probes at the training grid, fit on validation subjects
   and scored on held-out test subjects, beside the same probe on GAP features.

Evaluation only: weights are frozen and checked; no training and no checkpoint selection.
"""

import argparse
import hashlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from eval.encoder import encoder_spatial_target_audit as spatial  # noqa: E402
from eval.encoder.encoder_lesion_intervention import close_arrays  # noqa: E402
from eval.encoder.encoder_target_protocol import (  # noqa: E402
    TARGETS,
    VIEWS,
    dataset,
    digest,
    provenance,
    save_csv,
    save_report,
)
from eval.lesion.checkpoint_lesion_analysis import batch_features, state_digest  # noqa: E402
from eval.lesion.lesion_routing import render_pair  # noqa: E402
from eval.metrics.dci import CONTENT_FACTOR_NAMES  # noqa: E402
from eval.protocol.score_checkpoint import build_model, load_settings  # noqa: E402
from eval.ventricle.ventricle_routing import normalization_affine  # noqa: E402
from utils.encoder_runtime import select_encoder_device  # noqa: E402

LESION_DIMS = {2: "x", 3: "y", 4: "z"}
SPATIAL_STAGES = ("backbone", "projected")
SHOWN_TARGETS = (
    "brain_size",
    "ventricle_size",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_amplitude",
    "centroid_x",
    "centroid_y",
    "centroid_z",
)
ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


@torch.inference_mode()
def render_factor_pair(ds, idx, k, delta):
    """Two endpoints differing only in z_content[k] by -/+delta, as (T1, FLAIR) tensor lists.

    Acquisition draws and the subject's original normalization affine are fixed. Lesion
    coordinates use render_pair (only the lesion moves). Other factors keep the subject's
    own lesion load where it is, so the difference isolates that factor's tissue change.
    """
    if k in LESION_DIMS:
        sample = render_pair(ds, idx, LESION_DIMS[k], delta)
        return sample["a"], sample["b"]
    inner, renderer = ds._inner, ds._inner.renderer
    raw0, raw1, lat = inner[idx]
    mask = lat["brain_mask"]
    normalized = ds.normalize_views(raw0, raw1, mask, mask.clone())
    affines = [normalization_affine(raw, norm, mask) for raw, norm in zip((raw0, raw1), normalized)]
    fields = (lat["z_deformation"], lat["z_fissure"], "cpu")
    _, lesion = renderer.render_structure(
        lat["z_content"], *fields, clean=inner.clean_content, z_lesion=lat.get("z_lesion")
    )
    seed = inner.sample_seed_for(idx)
    endpoints = []
    for sign in (-1, 1):
        content = lat["z_content"].clone()
        content[k] += sign * delta
        tissue, _ = renderer.render_structure(content, *fields, clean=inner.clean_content, z_lesion=lat.get("z_lesion"))
        foreground = (tissue > 0).unsqueeze(0).float()
        views = []
        for v, modality in enumerate(("T1", "FLAIR")):
            raw = renderer.render_modality(tissue, lesion, lat[f"z_style_v{v + 1}"], modality, seed * 2 + v, "cpu")
            gain, bias = affines[v]
            views.append((raw * gain + bias) * foreground)
        endpoints.append(views)
    return endpoints[0], endpoints[1]


def survival(delta):
    """||mean over cells|| / mean ||cell||: 1 when every cell moves alike, ~0 when changes cancel.

    delta has shape (cells, channels); pass a scalar image as (voxels, 1).
    """
    local = np.linalg.norm(delta, axis=1).mean()
    return float(np.linalg.norm(delta.mean(0)) / local) if local > 1e-12 else np.nan


def cellwise_r2(fit, test, y_fit, y_test, alphas=ALPHAS, seed=1729):
    """Ridge on each cell's own features; alpha chosen on a 75/25 split of the fit subjects.

    fit: (n, cells, d); test: (m, cells, d); y_*: (subjects, targets). Returns (cells, targets)
    R² on the test subjects, which never take part in fitting or alpha selection.
    """
    fit, test = np.asarray(fit, np.float64), np.asarray(test, np.float64)
    y_fit, y_test = np.asarray(y_fit, np.float64), np.asarray(y_test, np.float64)
    order = np.random.default_rng(seed).permutation(len(fit))
    cut = int(0.75 * len(fit))
    if cut < 2 or len(fit) - cut < 2:
        raise ValueError("Need at least eight fit subjects for the alpha split")

    def standardize(train, *others):
        mean, scale = train.mean(0), train.std(0)
        scale[scale < 1e-8] = 1.0
        return [((a - mean) / scale).transpose(1, 0, 2) for a in (train, *others)]  # (cells, n, d)

    def predictions(train, y, other):
        offset = y.mean(0)
        gram = train.transpose(0, 2, 1) @ train
        rhs = train.transpose(0, 2, 1) @ (y - offset)
        eye = np.eye(train.shape[2])
        for alpha in alphas:
            yield other @ np.linalg.solve(gram + alpha * eye, rhs) + offset  # (cells, m, targets)

    def r2(y, predicted):
        total = ((y - y.mean(0)) ** 2).sum(0)
        return 1 - ((y - predicted) ** 2).sum(1) / np.where(total > 1e-12, total, np.nan)

    inner, tune = standardize(fit[order[:cut]], fit[order[cut:]])
    tuned = np.stack([r2(y_fit[order[cut:]], p) for p in predictions(inner, y_fit[order[:cut]], tune)])
    best = np.nan_to_num(tuned, nan=-np.inf).argmax(0)  # (cells, targets)
    full, held_out = standardize(fit, test)
    scores = np.stack([r2(y_test, p) for p in predictions(full, y_fit, held_out)])
    return np.take_along_axis(scores, best[None], 0)[0]


def cells(features, grid):
    """(n, C*g^3) channel-major feature banks -> (n, g^3, C)."""
    return np.asarray(features, np.float32).reshape(len(features), -1, grid**3).transpose(0, 2, 1)


def decodability(model, cfg, args, device, scratch, arm):
    options = SimpleNamespace(
        batch_size=args.batch_size, grids=[1, args.grid], include_native=False, stages=SPATIAL_STAGES
    )
    banks = {}
    try:
        for split, count in (("val", cfg["num_val_samples"]), ("test", args.test_samples)):
            banks[split] = spatial.extract(model, dataset(cfg, count, split), options, device, scratch / arm / split)
        (fit_arrays, y_fit, fit_meta), (test_arrays, y_test, test_meta) = banks["val"], banks["test"]
        rows, maps = [], {}
        with threadpool_limits(limits=args.threads):
            for view in VIEWS:
                for stage in SPATIAL_STAGES:
                    per_cell = cellwise_r2(
                        cells(fit_arrays[view, args.grid, stage], args.grid),
                        cells(test_arrays[view, args.grid, stage], args.grid),
                        y_fit,
                        y_test,
                        seed=args.seed,
                    )
                    gap = cellwise_r2(
                        cells(fit_arrays[view, 1, stage], 1),
                        cells(test_arrays[view, 1, stage], 1),
                        y_fit,
                        y_test,
                        seed=args.seed,
                    )[0]
                    for t, target in enumerate(TARGETS):
                        maps[f"{arm}/{view}/{stage}/{target}"] = (
                            per_cell[:, t].reshape((args.grid,) * 3).astype(np.float32)
                        )
                        best = int(np.nanargmax(per_cell[:, t]))
                        rows.append(
                            dict(
                                arm=arm,
                                view=view,
                                stage=stage,
                                target=target,
                                grid=args.grid,
                                max_cell_r2=float(per_cell[best, t]),
                                best_cell=[int(i) for i in np.unravel_index(best, (args.grid,) * 3)],
                                median_cell_r2=float(np.nanmedian(per_cell[:, t])),
                                gap_r2=float(gap[t]),
                            )
                        )
        # Unit scale for global responses: each content unit's spread over validation subjects.
        global_sd = {view: np.asarray(fit_arrays[view, 1, "projected"], np.float64).std(0) + 1e-8 for view in VIEWS}
        return rows, maps, global_sd, {"val": fit_meta, "test": test_meta}
    finally:
        for bank in banks.values():
            close_arrays(bank[0])


def native_maps(model, x, device, native):
    """(images, native³ cells, channels) backbone/projected maps and the global content code."""
    features, shape, channels = batch_features(model, x.to(device), [1, native])
    if shape != (native,) * 3:
        raise ValueError(f"Expected a cubic {native}³ backbone map, got {shape}")
    b = x.shape[0]
    return dict(
        backbone=features[native, "backbone"].reshape(b, channels, native**3).transpose(0, 2, 1),
        projected=features[native, "projected"].reshape(b, -1, native**3).transpose(0, 2, 1),
        global_code=features[1, "projected"],
    )


def gallery(model, ds, args, device, arm, out):
    """Unperturbed maps of the first test subjects; returns per-view channel scales."""
    n = args.native
    items = [ds[i] for i in range(args.gallery_subjects)]
    x = torch.cat([torch.stack([item["image"][v] for item in items]) for v in range(2)])
    maps = native_maps(model, x, device, n)
    g, k, stride = len(items), n // 2, ds.res // n
    columns = [
        "input",
        "|backbone|",
        "PC1",
        "PC2",
        "PC3",
        *[f"content {c + 1}" for c in range(maps["projected"].shape[2])],
    ]
    content_range = np.percentile(np.abs(maps["projected"]), 99, axis=(0, 1)) + 1e-8
    fig, axes = plt.subplots(2 * g, len(columns), figsize=(1.45 * len(columns), 1.6 * 2 * g), squeeze=False)
    scales = {}
    for v, view in enumerate(VIEWS):
        backbone, projected = maps["backbone"][v * g : (v + 1) * g], maps["projected"][v * g : (v + 1) * g]
        scales[view] = {
            stage: block.reshape(-1, block.shape[2]).std(0) + 1e-8
            for stage, block in (("backbone", backbone), ("projected", projected))
        }
        flat = backbone.reshape(-1, backbone.shape[2])
        centered = flat - flat.mean(0)
        _, _, components = np.linalg.svd(centered[:: max(1, len(centered) // 20000)], full_matrices=False)
        components = components[:3] * np.sign(components[np.arange(3), np.abs(components[:3]).argmax(1)])[:, None]
        scores = (centered @ components.T).reshape(g, n**3, 3)
        norm = np.linalg.norm(backbone, axis=2)
        for s in range(g):
            row = v * g + s
            panels = [
                (items[s]["image"][v].numpy()[0][:, :, int((k + 0.5) * stride)], dict(cmap="gray")),
                (norm[s].reshape(n, n, n)[:, :, k], dict(cmap="magma", vmin=0, vmax=norm.max())),
            ]
            for p in range(3):
                lim = np.abs(scores[..., p]).max() + 1e-8
                panels.append((scores[s, :, p].reshape(n, n, n)[:, :, k], dict(cmap="RdBu_r", vmin=-lim, vmax=lim)))
            for c in range(projected.shape[2]):
                lim = content_range[c]
                panels.append((projected[s, :, c].reshape(n, n, n)[:, :, k], dict(cmap="RdBu_r", vmin=-lim, vmax=lim)))
            for col, (panel, kw) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(panel.T, origin="lower", interpolation="nearest", **kw)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(columns[col], fontsize=7)
            axes[row, 0].set_ylabel(f"{view} s{items[s]['index']}", fontsize=7)
    fig.suptitle(
        f"{arm}: final-stage maps, axial cell slice {k} of {n}³ (input slice {int((k + 0.5) * stride)})", fontsize=9
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"gallery_{arm}.png", dpi=110)
    plt.close(fig)
    return scales


def responses(model, ds, args, device, arm, scales, global_sd):
    """Per factor and view: response summaries and subject-mean response maps."""
    rows, maps, skipped_reasons = [], {}, []
    for k, factor in enumerate(CONTENT_FACTOR_NAMES):
        values = {(view, stage): [] for view in VIEWS for stage in ("input", *SPATIAL_STAGES, "global")}
        fields = {(view, stage): [] for view in VIEWS for stage in ("input", *SPATIAL_STAGES)}
        skipped = 0
        for idx in range(args.intervention_subjects):
            try:
                a, b = render_factor_pair(ds, idx, k, args.delta)
            except ValueError as error:  # includes LesionPlacementError; recorded, never redrawn
                skipped += 1
                skipped_reasons.append(f"{factor} subject {idx}: {error}")
                continue
            encoded = native_maps(model, torch.cat([torch.stack([a[v], b[v]]) for v in range(2)]), device, args.native)
            for v, view in enumerate(VIEWS):
                difference = (b[v] - a[v]).double().numpy()[0]
                fields[view, "input"].append(np.abs(difference))
                values[view, "input"].append(
                    (np.abs(difference).mean(), abs(difference.mean()), survival(difference.reshape(-1, 1)), np.nan)
                )
                for stage in SPATIAL_STAGES:
                    delta = (encoded[stage][2 * v + 1] - encoded[stage][2 * v]) / scales[view][stage]
                    magnitude = np.linalg.norm(delta, axis=1) / np.sqrt(delta.shape[1])
                    fields[view, stage].append(magnitude.reshape((args.native,) * 3))
                    coherent = np.linalg.norm(delta.mean(0)) / np.sqrt(delta.shape[1])
                    values[view, stage].append((magnitude.mean(), coherent, survival(delta), np.nan))
                change = (encoded["global_code"][2 * v + 1] - encoded["global_code"][2 * v]) / global_sd[view]
                values[view, "global"].append((np.nan, np.nan, np.nan, np.linalg.norm(change) / np.sqrt(change.size)))
        for (view, stage), entries in values.items():
            entries = np.asarray(entries, dtype=float).reshape(-1, 4)
            summary = [np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in entries.T]
            rows.append(
                dict(
                    arm=arm,
                    view=view,
                    factor=factor,
                    stage=stage,
                    n_subjects=len(entries),
                    n_skipped=skipped,
                    local_response=float(summary[0]),
                    coherent_response=float(summary[1]),
                    gap_survival=float(summary[2]),
                    global_response=float(summary[3]),
                )
            )
        for (view, stage), entries in fields.items():
            if entries:
                maps[arm, view, stage, factor] = np.mean(entries, 0)
    return rows, maps, skipped_reasons


def response_figure(maps, rows, arms, view, out):
    lookup = {(r["arm"], r["view"], r["factor"], r["stage"]): r for r in rows}
    columns = [("input", arms[0])] + [(stage, arm) for stage in SPATIAL_STAGES for arm in arms]
    fig, axes = plt.subplots(
        len(CONTENT_FACTOR_NAMES), len(columns), figsize=(2.0 * len(columns), 1.9 * 9), squeeze=False
    )
    for r, factor in enumerate(CONTENT_FACTOR_NAMES):
        for c, (stage, arm) in enumerate(columns):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if (arm, view, stage, factor) not in maps:
                ax.set_title("no valid pairs", fontsize=7)
                continue
            # Trained and initial share one colour scale per stage, so their panels compare directly.
            peers = [maps[a, view, stage, factor].max() for a in arms if (a, view, stage, factor) in maps]
            ax.imshow(
                maps[arm, view, stage, factor].max(2).T,
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=max(peers) or 1.0,
                interpolation="nearest",
            )
            label = "input |Δx|" if stage == "input" else f"{arm} {stage}"
            ax.set_title(f"{label}\nsurvival {lookup[arm, view, factor, stage]['gap_survival']:.2f}", fontsize=7)
        shift = ", ".join(f"{a} {lookup[a, view, factor, 'global']['global_response']:.2f}" for a in arms)
        axes[r, 0].set_ylabel(f"{factor}\nglobal Δ/SD: {shift}", fontsize=7)
    fig.suptitle(
        f"{view}: where each factor changes the final maps (subject mean, max over z). "
        "survival = |mean Δ| / mean |Δ|: 1 survives GAP, 0 cancels",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / f"responses_{view}.png", dpi=110)
    plt.close(fig)


def decodability_figure(maps, rows, arm, grid, out):
    lookup = {(r["view"], r["stage"], r["target"]): r for r in rows if r["arm"] == arm}
    columns = [(view, stage) for view in VIEWS for stage in SPATIAL_STAGES]
    fig, axes = plt.subplots(len(SHOWN_TARGETS), len(columns), figsize=(2.1 * len(columns), 1.95 * len(SHOWN_TARGETS)))
    for r, target in enumerate(SHOWN_TARGETS):
        for c, (view, stage) in enumerate(columns):
            ax = axes[r, c]
            grid_map = np.clip(maps[f"{arm}/{view}/{stage}/{target}"], 0, 1)
            ax.imshow(grid_map.max(2).T, origin="lower", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            row = lookup[view, stage, target]
            ax.set_title(f"{view} {stage}\nbest cell {row['max_cell_r2']:.2f} | GAP {row['gap_r2']:.2f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(target, fontsize=8)
    fig.suptitle(
        f"{arm}: held-out R² from each {grid}³ cell alone (max over z); GAP = same probe on pooled features", fontsize=8
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / f"decodability_{arm}.png", dpi=110)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory")
    parser.add_argument("--grid", type=int, help="Decodability grid; default: the run's training patch grid, else 8")
    parser.add_argument("--test-samples", type=int, default=400, help="Held-out subjects for decodability")
    parser.add_argument("--gallery-subjects", type=int, default=3)
    parser.add_argument("--intervention-subjects", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.5, help="Change each raw content control by -/+delta")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="BLAS threads for the per-cell probes")
    parser.add_argument("--skip-initial", action="store_true", help="Omit the saved model_init.pt control")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    if min(args.gallery_subjects, args.intervention_subjects, args.batch_size, args.threads) < 1:
        parser.error("Subject counts, batch size and threads must be positive")
    if args.test_samples < 10 or max(args.gallery_subjects, args.intervention_subjects) > args.test_samples:
        parser.error("Need >=10 test subjects, covering the gallery and intervention subjects")
    if not np.isfinite(args.delta) or args.delta <= 0:
        parser.error("--delta must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_settings(args.run_dir)
    if cfg["num_val_samples"] < 20:
        raise ValueError("Need at least 20 original validation subjects for the per-cell probes")
    if args.grid is None:
        trained_grid = cfg.get("train_patch_grid") if cfg.get("patch_loss_weight", 0) > 0 else None
        args.grid = int(trained_grid[0]) if trained_grid else 8
    checkpoints = {"trained": args.run_dir / args.checkpoint}
    if not args.skip_initial:
        checkpoints["initial"] = args.run_dir / "model_init.pt"
    hashes = {arm: digest(path) for arm, path in checkpoints.items()}
    device = select_encoder_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, device)
    report.update(
        checkpoint_sha256=hashes,
        evaluation_only=True,
        cohorts={},
        notes=[
            "backbone = the encoder's last feature map; projected = the active spatial content head at every native "
            "cell; global = the model's content vector (GAP, then the global head).",
            "Interventions change one raw content control by -/+delta. Everything else in the anatomy, the acquisition "
            "draws and the subject's normalization affine is fixed. Non-lesion factors keep the subject's own lesion in "
            "place; lesion coordinates move only the lesion. Causal descendants are not propagated.",
            "Spatial responses are in units of each channel's SD over the gallery subjects' unperturbed cells (per arm, "
            "view and stage). global_response uses each content unit's SD over validation subjects. Compare factors "
            "within a stage; magnitudes are not comparable across architectures.",
            "gap_survival = ||mean over cells of the change|| / mean over cells of ||change||: 1 when every cell moves "
            "alike, so the change survives global average pooling; ~0 when it cancels. The input row uses the image.",
            "Per-cell probes: ridge on one cell's features, alpha chosen on a 75/25 split of validation subjects, refit "
            "on all of them, R² on held-out test subjects. gap_r2 applies the same probe to grid-1 features.",
            "Finite-probe readouts on a fixed checkpoint; not an identifiability guarantee.",
        ],
    )
    save_report(args.out_dir, report)
    try:
        maps, response_rows, decode_rows, response_maps, skipped = {}, [], [], {}, []
        with tempfile.TemporaryDirectory(prefix=".features-", dir=args.out_dir) as tmp:
            for arm, path in checkpoints.items():
                blob = path.read_bytes()
                if hashlib.sha256(blob).hexdigest() != hashes[arm]:
                    raise ValueError("Checkpoint changed before loading")
                model = build_model(cfg, device, torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
                del blob
                before = state_digest(model)
                test = dataset(cfg, args.test_samples, "test")
                probe = torch.cat([test[0]["image"][0][None], test[0]["image"][1][None]])
                _, shape, _ = batch_features(model, probe.to(device), [1])
                if len(set(shape)) != 1 or args.grid > shape[0]:
                    raise ValueError(f"Need a cubic backbone map at least {args.grid}³; got {shape}")
                args.native = shape[0]
                print(f"[{arm}] per-cell decodability at {args.grid}³ (native map {shape[0]}³)", flush=True)
                rows, arm_maps, global_sd, meta = decodability(model, cfg, args, device, Path(tmp), arm)
                if report["cohorts"] and any(
                    meta[s]["input_sha256"] != report["cohorts"]["trained"][s]["input_sha256"] for s in meta
                ):
                    raise ValueError("Initial/trained arms received different images")
                report["cohorts"][arm] = meta
                decode_rows.extend(rows)
                maps.update(arm_maps)
                print(f"[{arm}] gallery and factor interventions", flush=True)
                scales = gallery(model, test, args, device, arm, args.out_dir)
                rows, arm_maps, reasons = responses(model, test, args, device, arm, scales, global_sd)
                response_rows.extend(rows)
                response_maps.update(arm_maps)
                skipped.extend(f"{arm}: {reason}" for reason in reasons)
                if state_digest(model) != before:
                    raise RuntimeError("Analysis changed encoder parameters or buffers")
                decodability_figure(maps, decode_rows, arm, args.grid, args.out_dir)
                del model
        arms = list(checkpoints)
        for view in VIEWS:
            response_figure(response_maps, response_rows, arms, view, args.out_dir)
        np.savez_compressed(args.out_dir / "decodability_maps.npz", **maps)
        save_csv(args.out_dir / "responses.csv", response_rows)
        save_csv(args.out_dir / "decodability.csv", decode_rows)
        if {arm: digest(path) for arm, path in checkpoints.items()} != hashes:
            raise ValueError("Source checkpoint files changed during analysis")
        report.update(
            status="complete",
            source_checkpoints_unchanged=True,
            encoder_unchanged=True,
            native_grid=args.native,
            skipped_interventions=skipped,
            responses=response_rows,
            decodability=decode_rows,
        )
        save_report(args.out_dir, report)
        lookup = {(r["arm"], r["view"], r["factor"], r["stage"]): r for r in response_rows}
        print("\nT1: GAP survival of each factor's change (1 = survives averaging, 0 = cancels) and global shift")
        header = " ".join(f"{a[:4]}-{s[:4]:>4s}" for a in arms for s in SPATIAL_STAGES)
        print(f"{'factor':20s} {'input':>6s} {header} " + " ".join(f"{a[:4]}-glob" for a in arms))
        for factor in CONTENT_FACTOR_NAMES:
            spatial_values = [lookup[a, "t1", factor, s]["gap_survival"] for a in arms for s in SPATIAL_STAGES]
            shifts = [lookup[a, "t1", factor, "global"]["global_response"] for a in arms]
            print(
                f"{factor:20s} {lookup[arms[0], 't1', factor, 'input']['gap_survival']:6.2f} "
                + " ".join(f"{value:9.2f}" for value in spatial_values)
                + " "
                + " ".join(f"{value:9.2f}" for value in shifts)
            )
        print(f"Saved final-stage spatial map analysis: {args.out_dir}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise


if __name__ == "__main__":
    main()
