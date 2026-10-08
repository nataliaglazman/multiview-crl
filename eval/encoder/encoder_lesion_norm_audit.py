"""Trace frozen lesion movements through early GN/channel-LN layers.

See ENCODER_LESION_NORM_AUDIT.md. Probes fit on observational validation subjects;
paired test images change only lesion position. Saved initialization is included.
"""

import argparse
import gc
import hashlib
import io
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from eval.diagnostics.pooling_probe import fit_readouts
from eval.encoder import encoder_normalization_audit as norm_audit
from eval.encoder.encoder_lesion_intervention import CENTROIDS, close_arrays, feature_response, movement_metrics
from eval.encoder.encoder_target_protocol import (
    VIEWS,
    dataset,
    dataset_metadata,
    digest,
    provenance,
    save_csv,
    save_report,
)
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.lesion.lesion_routing import render_pair
from eval.protocol.score_checkpoint import build_model, load_settings
from models.vqvae import ChannelLayerNorm3d
from utils.encoder_runtime import select_encoder_device

STAGES = ("norm_pre_spatial", "norm_post_spatial", "norm_pre_gap", "norm_post_gap", "norm_mean", "norm_scale")


def resolve_layers(model, requested):
    layers = []
    for name in requested:
        if name == "early":
            layers.extend(
                f"layers.{i}.1"
                for i, layer in enumerate(model.encoder.layers)
                if isinstance(layer, torch.nn.Sequential)
                and len(layer) > 1
                and isinstance(layer[1], (torch.nn.GroupNorm, ChannelLayerNorm3d))
            )
        else:
            layers.append(norm_audit.select_norm(model.encoder, name)[0])
    layers = list(dict.fromkeys(layers))
    if not layers:
        raise ValueError("No requested normalization layers found")
    for name in layers:
        norm_audit.select_norm(model.encoder, name)
        if model.encoder_v1 is not None:
            norm_audit.select_norm(model.encoder_v1, name)
    return layers


def layer_response(module, inputs, outputs):
    """Native finite-pair response, plus a local counterfactual with A's stats fixed.

    No replacement tensors are fed back into the network. The shift/scale
    decomposition concerns input-change directions, not information fractions.
    """
    x, y = inputs.detach().cpu().double(), outputs.detach().cpu().double()
    if x.ndim != 5 or len(x) % 2:
        raise ValueError("Need adjacent A/B pairs of spatial tensors")
    a, b = x[::2], x[1::2]
    delta = b - a
    if isinstance(module, torch.nn.GroupNorm):

        def domains(t):
            return t.reshape(len(t), module.num_groups, -1)

        eps, weight = module.eps, module.weight
        variance = domains(a).var(-1, unbiased=False, keepdim=True)
        fixed_delta = (domains(delta) / (variance + eps).sqrt()).reshape_as(delta)
    elif isinstance(module, ChannelLayerNorm3d):

        def domains(t):
            return t.permute(0, 2, 3, 4, 1).reshape(len(t), -1, t.shape[1])

        eps, weight = module.norm.eps, module.norm.weight
        variance = a.var(1, unbiased=False, keepdim=True)
        fixed_delta = delta / (variance + eps).sqrt()
    else:
        raise ValueError("Expected GN or channel-only LN")
    if weight is not None:
        fixed_delta *= weight.detach().cpu().double()[None, :, None, None, None]
    actual_delta = y[1::2] - y[::2]
    d, base = domains(delta), domains(a)
    centered = base - base.mean(-1, keepdim=True)
    shifted = d.mean(-1, keepdim=True)
    denom = centered.square().sum(-1)
    dot = ((d - shifted) * centered).sum(-1)
    radial = torch.where(denom > 1e-24, dot.square() / denom.clamp_min(1e-24), 0).sum(-1)
    shift = shifted.square().sum((1, 2)) * d.shape[-1]
    total = d.square().sum((1, 2))
    actual_energy = actual_delta.flatten(1).square().sum(1)
    fixed_energy = fixed_delta.flatten(1).square().sum(1)
    records = []
    for i in range(len(a)):
        records.append(
            {
                "pre_native_delta_rms": float(delta[i].square().mean().sqrt()),
                "post_native_delta_rms": float(actual_delta[i].square().mean().sqrt()),
                "fixed_stats_delta_rms": float(fixed_delta[i].square().mean().sqrt()),
                "adaptive_to_fixed_response": (
                    float((actual_energy[i] / fixed_energy[i]).sqrt()) if fixed_energy[i] > 1e-24 else np.nan
                ),
                "shift_direction_fraction": float(shift[i] / total[i]) if total[i] > 1e-24 else np.nan,
                "scale_direction_fraction": float(radial[i] / total[i]) if total[i] > 1e-24 else np.nan,
                "orthogonal_direction_fraction": (
                    float(((total[i] - shift[i] - radial[i]) / total[i]).clamp(0, 1)) if total[i] > 1e-24 else np.nan
                ),
            }
        )
    return records


@torch.inference_mode()
def capture(model, images, layer, grid, with_response=False):
    if any(module.training for module in model.modules()):
        raise ValueError("Frozen extraction requires every module in eval mode")
    captured, native, descriptions = {}, {}, []
    encoders = [model.encoder] + ([model.encoder_v1] if model.encoder_v1 is not None else [])

    def hook(index):
        def receive(module, args, output):
            # Copy immediately: a subsequent in-place ReLU may alter output.
            captured[index, "pre"] = args[0].detach().clone()
            captured[index, "post"] = output.detach().clone()
            mean, scale = norm_audit.norm_statistics(module, args[0])
            captured[index, "mean"], captured[index, "scale"] = mean, scale
            if with_response:
                native[index] = layer_response(module, args[0], output)

        return receive

    with ExitStack() as stack:
        for i, encoder in enumerate(encoders):
            _, module = norm_audit.select_norm(encoder, layer)
            descriptions.append({"layer": layer, "type": type(module).__name__, "repr": repr(module)})
            stack.callback(module.register_forward_hook(hook(i)).remove)
        model(images, pool_only=True, n_views=2)
    values = {
        key: torch.cat([captured[i, key] for i in range(len(encoders))]) for key in ("pre", "post", "mean", "scale")
    }

    def flatten(value):
        if value.ndim == 5 and grid:
            if any(grid > size for size in value.shape[2:]):
                raise ValueError(f"Grid {grid} exceeds tapped native map {tuple(value.shape[2:])}")
            value = F.adaptive_avg_pool3d(value, (grid,) * 3)
        return value.flatten(1).cpu().numpy().copy()

    features = {}
    for stage in ("pre", "post"):
        features[f"norm_{stage}_spatial"] = flatten(values[stage])
        features[f"norm_{stage}_gap"] = flatten(values[stage].mean((2, 3, 4)))
    features.update(norm_mean=flatten(values["mean"]), norm_scale=flatten(values["scale"]))
    metadata = {
        "norm_shape": list(values["pre"].shape[1:]),
        "normalizers": descriptions,
        "probe_grid": grid or "native",
    }
    if with_response:
        records = [row for i in range(len(encoders)) for row in native[i]]
        return features, metadata, records
    return features, metadata


def extract_pairs(model, ds, args, device, directory, layer):
    directory.mkdir(parents=True)
    requests = [
        (i, axis) for i in range(args.subject_offset, args.subject_offset + args.num_samples) for axis in args.axes
    ]
    arrays, replay, rows, diagnostics = {}, {}, [], []
    image_hash, before = hashlib.sha256(), state_digest(model)
    try:
        for start in range(0, len(requests), args.batch_size):
            samples = [render_pair(ds, i, axis, args.eps) for i, axis in requests[start : start + args.batch_size]]
            endpoints = [sample[state] for sample in samples for state in ("a", "b")]
            images = torch.cat([torch.stack([pair[v] for pair in endpoints]) for v in range(2)])
            for endpoint in endpoints:
                image_hash.update(torch.stack(endpoint).contiguous().numpy().tobytes())
            features, metadata, native = capture(model, images.to(device), layer, args.spatial_grid, with_response=True)
            repeated, _ = capture(model, images.to(device), layer, args.spatial_grid)
            b, n = len(endpoints), len(samples)
            for stage, values in features.items():
                if not np.isfinite(values).all() or not np.isfinite(repeated[stage]).all():
                    raise ValueError("Non-finite intervention features")
                for v, view in enumerate(VIEWS):
                    key = (view, stage)
                    block = values[v * b : (v + 1) * b]
                    if key not in arrays:
                        arrays[key] = np.lib.format.open_memmap(
                            directory / f"{view}_{stage}.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(2 * len(requests), block.shape[1]),
                        )
                        replay[key] = np.zeros(len(requests))
                    arrays[key][2 * start : 2 * start + b] = block
                    difference = block.astype(np.float64) - repeated[stage][v * b : (v + 1) * b]
                    replay[key][start : start + n] = np.sqrt(np.mean(difference.reshape(n, -1) ** 2, axis=1))
            for j, sample in enumerate(samples):
                if not all(support.any() for support in sample["lesions"]):
                    raise ValueError("Empty lesion endpoint; refusing to redraw test subjects")
                centers = np.asarray(sample["centroids"])
                distance = float(np.linalg.norm(centers[1] - centers[0]))
                row = dict(
                    subject_id=sample["index"],
                    intervention_axis=sample["axis"],
                    eps=args.eps,
                    moved=distance > 1e-8,
                    true_displacement_vox=distance,
                    changed_lesion_voxels=int(sample["support"].sum()),
                )
                for endpoint, center in zip("ab", centers):
                    for k, axis in enumerate("xyz"):
                        row[f"centroid_{endpoint}_{axis}"] = float(2 * center[k] / (ds.res - 1) - 1)
                rows.append(row)
                for v, view in enumerate(VIEWS):
                    diagnostics.append({"view": view, **row, **native[v * n + j]})
            print(f"  {layer}: encoded {len(rows)}/{len(requests)} movement pairs", flush=True)
        if state_digest(model) != before:
            raise RuntimeError("Frozen extraction changed model state")
        for array in arrays.values():
            array.flush()
        truth = np.array([[r[f"centroid_{e}_{axis}"] for axis in "xyz"] for r in rows for e in "ab"])
        return (
            arrays,
            truth,
            rows,
            replay,
            diagnostics,
            {
                **dataset_metadata(ds, image_hash.hexdigest(), sorted({r["subject_id"] for r in rows})),
                "target_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
                "model_state_sha256": before,
                **metadata,
            },
        )
    except Exception:
        close_arrays(arrays)
        raise


def score(reference, reference_targets, pairs, args, info, directory):
    arrays, truth, pair_rows, replay, _, _ = pairs
    order = np.random.default_rng(args.seed).permutation(len(reference_targets))
    cut = int(0.75 * len(order))
    splits = (order[:cut], order[cut:], np.arange(len(order), len(order) + len(truth)))
    targets = np.concatenate((reference_targets, truth))
    subjects = np.array([r["subject_id"] for r in pair_rows])
    axes = np.array([r["intervention_axis"] for r in pair_rows])
    moved = np.array([r["moved"] for r in pair_rows])
    rows, parameters, sensitivity = [], [], []
    saved = dict(truth=truth.reshape(-1, 2, 3), subject_id=subjects, intervention_axis=axes)
    for view in VIEWS:
        grams = {}
        for stage in STAGES:
            feature = reference[view, stage]
            grams[stage] = (*norm_audit.bank_gram(feature, arrays[view, stage], splits[0]), feature.shape[1])
        for stage, parts in (
            ("norm_statistics", ("norm_mean", "norm_scale")),
            ("norm_post_plus_statistics", ("norm_post_spatial", "norm_mean", "norm_scale")),
        ):
            grams[stage] = tuple(sum(grams[p][i] for p in parts) for i in range(3))
        for stage, (gram, width, dimensions) in grams.items():
            print(f"  {info['run']} {info['checkpoint']} {info['layer']} {view} {stage}: fitting probes", flush=True)
            details = {**info, "view": view, "stage": stage, "dimensions": dimensions, "variable_dimensions": width}
            fitted, predicted = fit_readouts(gram, width, targets, splits, args.seed, CENTROIDS)
            if any(not np.isfinite(p).all() for p in predicted.values()):
                raise FloatingPointError("Non-finite probe predictions")
            parameters.extend({**details, **row} for row in fitted)
            rms, relative, null = (np.full(len(pair_rows), np.nan) for _ in range(3))
            if stage in STAGES:
                rms, relative, null = feature_response(
                    reference[view, stage], splits[0], arrays[view, stage], replay[view, stage]
                )
                sensitivity.extend(
                    {
                        **details,
                        "subject_id": pair["subject_id"],
                        "intervention_axis": pair["intervention_axis"],
                        "moved": pair["moved"],
                        "feature_delta_rms": float(rms[i]),
                        "response_to_subject_sd": float(relative[i]),
                        "replay_delta_rms": float(null[i]),
                    }
                    for i, pair in enumerate(pair_rows)
                )
            for kind, prediction in predicted.items():
                probe, condition = kind.split("_")
                prediction = prediction.reshape(-1, 2, 3)
                saved[f"{view}__{stage}__{kind}"] = prediction
                for axis in ("all", *args.axes):
                    selected = np.ones(len(pair_rows), dtype=bool) if axis == "all" else axes == axis
                    metrics = movement_metrics(
                        truth.reshape(-1, 2, 3)[selected],
                        prediction[selected],
                        subjects[selected],
                        (args.resolution - 1) / 2,
                        args.bootstrap,
                        args.seed,
                    )
                    active = selected & moved
                    rows.append(
                        {
                            **details,
                            "probe": probe,
                            "condition": condition,
                            "intervention_axis": axis,
                            **metrics,
                            "response_to_subject_sd": float(relative[active].mean()) if active.any() else np.nan,
                        }
                    )
    filename = f"{info['run']}__{info['checkpoint']}__{info['layer']}_predictions.npz"
    np.savez_compressed(directory / filename, **saved)
    return (
        rows,
        parameters,
        sensitivity,
        {"fit_validation_ids": splits[0].tolist(), "tune_validation_ids": splits[1].tolist()},
    )


def check_cohort(reference, candidate):
    for split in ("val", "moves"):
        for field in ("input_sha256", "target_sha256", "ids", "generator_split_seed"):
            if reference[split][field] != candidate[split][field]:
                raise ValueError(f"Unmatched {split} {field}: checkpoints/layers/runs must see identical subjects")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--skip-initial", action="store_true", help="Explicitly omit the original model_init.pt control")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--norm-layers",
        nargs="+",
        default=["early", "pre_residual"],
        help="early = all downsampling norms; explicit paths relative to encoder also accepted",
    )
    p.add_argument(
        "--spatial-grid",
        type=int,
        default=8,
        help="Probe maps pooled to this grid; 0 = native maps (large). Native response diagnostics are always unpooled.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Held-out intervention subjects; every subject gets all requested axes",
    )
    p.add_argument("--subject-offset", type=int, default=1000)
    p.add_argument("--axes", nargs="+", choices=("x", "y", "z"), default=["x", "y", "z"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1, help="Pairs/batch; four input images per pair")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = p.parse_args(argv)
    if (
        args.num_samples < 2
        or args.subject_offset < 0
        or args.batch_size < 1
        or args.spatial_grid < 0
        or args.bootstrap < 0
        or not np.isfinite(args.eps)
        or args.eps <= 0
    ):
        p.error("Need >=2 subjects, positive finite eps/batch size, and nonnegative offset/grid/bootstrap")
    if Path(args.checkpoint).is_absolute() or ".." in Path(args.checkpoint).parts:
        p.error("Checkpoint must be relative to each run directory")
    args.axes = list(dict.fromkeys(args.axes))
    return args


def main(argv=None):
    args = parse_args(argv)
    runs, hashes = {}, {}
    for specification in args.run:
        name, separator, path = specification.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path or name in runs:
            raise ValueError("Use unique NAME=DIR entries")
        path, cfg = Path(path), load_settings(path)
        if (
            cfg.get("encoder_architecture", "conv") != "conv"
            or cfg.get("synthetic_lesion_target", "position") != "position"
            or cfg.get("synthetic_lesion_placement", "legacy") != "wm_interior"
        ):
            raise ValueError("Need conv encoder, position targets and wm_interior lesion placement")
        if cfg["num_val_samples"] < 20:
            raise ValueError("Need >=20 original validation subjects")
        checkpoints = {"trained": path / args.checkpoint}
        if not args.skip_initial:
            checkpoints["initial"] = path / "model_init.pt"
        runs[name] = (path, cfg, checkpoints)
        for file in [path / "settings.json", *checkpoints.values()]:
            hashes[str(file)] = digest(file)
    device = select_encoder_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(next(iter(runs.values()))[1], args, device)
    report.update(
        target_names=list(CENTROIDS),
        evaluation_only=True,
        source_sha256_files=hashes,
        runs={name: {"run_dir": str(path.resolve()), "settings": cfg} for name, (path, cfg, _) in runs.items()},
        summary=[],
        probe_parameters=[],
        sensitivity=[],
        normalization_response=[],
        cohorts={},
        protocol="Original validation 75/25 probe fit/tune; separate fixed-anatomy lesion movements. Initial and trained checkpoints; native normalization-response counterfactual, not a model intervention.",
    )
    save_report(args.out_dir, report)
    reference_cohort = None
    try:
        for name, (run_path, cfg, checkpoints) in runs.items():
            for arm, path in checkpoints.items():
                blob = path.read_bytes()
                if hashlib.sha256(blob).hexdigest() != hashes[str(path)]:
                    raise ValueError("Checkpoint changed before loading")
                model = build_model(cfg, device, torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
                del blob
                before = state_digest(model)
                val_ds = dataset(cfg, cfg["num_val_samples"], "val")
                move_ds = dataset(cfg, max(64, args.subject_offset + args.num_samples), "test")
                for layer in resolve_layers(model, args.norm_layers):
                    _, module = norm_audit.select_norm(model.encoder, layer)
                    normalization = type(module).__name__
                    if model.encoder_v1 is not None and type(
                        norm_audit.select_norm(model.encoder_v1, layer)[1]
                    ) is not type(module):
                        raise ValueError("Different normalization types across views are not supported")
                    info = dict(
                        run=name,
                        checkpoint=arm,
                        normalization=normalization,
                        layer=layer,
                        probe_grid=args.spatial_grid or "native",
                    )
                    print(
                        f"\n{name}/{arm} {layer}: actual normalization = {normalization}; probe grid = {args.spatial_grid or 'native'}",
                        flush=True,
                    )
                    with tempfile.TemporaryDirectory(prefix=".features-", dir=args.out_dir) as temporary:
                        reference, pairs = {}, None
                        try:
                            extract_args = SimpleNamespace(
                                batch_size=2 * args.batch_size, norm_layer=layer, spatial_grid=args.spatial_grid
                            )
                            reference, targets, val_meta = norm_audit.extract(
                                model, val_ds, extract_args, device, Path(temporary) / "val", capture_fn=capture
                            )
                            val_meta["target_sha256"] = hashlib.sha256(targets[:, 9:12].tobytes()).hexdigest()
                            pairs = extract_pairs(model, move_ds, args, device, Path(temporary) / "moves", layer)
                            cohort = dict(val=val_meta, moves=pairs[5])
                            if reference_cohort is not None:
                                check_cohort(reference_cohort, cohort)
                            else:
                                reference_cohort = cohort
                                save_csv(args.out_dir / "pairs.csv", pairs[2])
                            report["cohorts"][f"{name}/{arm}/{layer}"] = cohort
                            with threadpool_limits(limits=1):
                                scoring_args = SimpleNamespace(**vars(args), resolution=cfg["res"])
                                rows, parameters, sensitivity, split = score(
                                    reference, targets[:, 9:12], pairs, scoring_args, info, args.out_dir
                                )
                            report["summary"].extend(rows)
                            report["probe_parameters"].extend(parameters)
                            report["sensitivity"].extend(sensitivity)
                            report["normalization_response"].extend({**info, **row} for row in pairs[4])
                            report["probe_split"] = split
                        finally:
                            close_arrays(reference)
                            if pairs is not None:
                                close_arrays(pairs[0])
                    save_report(args.out_dir, report)
                if state_digest(model) != before:
                    raise RuntimeError("Audit changed encoder state")
                del model, val_ds, move_ds, reference, pairs
                gc.collect()
        if any(digest(path) != value for path, value in hashes.items()):
            raise ValueError("Source checkpoint/settings files changed during the audit")
        report.update(status="complete", encoder_unchanged=True, source_checkpoints_unchanged=True)
        for table in ("summary", "probe_parameters", "sensitivity", "normalization_response"):
            save_csv(args.out_dir / f"{table}.csv", report[table])
        save_report(args.out_dir, report)
        print(
            "\nLesion movement skill: 1 perfect; 0 predicts no movement; negative worse. Ridge / RBF, observed labels."
        )
        print(
            "run       checkpoint actual_norm        layer       view stage                    skill    gain  error(vox)"
        )
        for row in report["summary"]:
            if row["condition"] == "observed" and row["intervention_axis"] == "all":
                print(
                    f"{row['run']:9s} {row['checkpoint']:10s} {row['normalization']:18s} {row['layer']:11s} {row['view']:5s} {row['stage']:24s} {row['probe']:5s} {row['movement_skill']:+.3f} {row['movement_gain']:+.3f} {row['movement_rmse_vox']:.3f}"
                )
        print(f"Saved early normalization lesion audit: {args.out_dir}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise


if __name__ == "__main__":
    main()
