"""Move only the lesion in a frozen encoder; measure sensitivity and position tracking.

python -m eval.encoder.encoder_lesion_intervention --run-dir RUN --out-dir NEW_OUTPUT

Probes fit/tune on original validation subjects. A separate subject cohort receives
paired -/+eps changes to one lesion control, with anatomy, acquisition random draws
and the original normalization affine fixed. No encoder training or checkpoint
selection. Feature banks are temporary; only small reports/predictions survive.
"""

import argparse
import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from eval.diagnostics.pooling_probe import fit_readouts, r2
from eval.encoder import encoder_spatial_target_audit as spatial
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
from eval.lesion.lesion_probe import block_gram
from eval.lesion.lesion_routing import render_pair
from eval.protocol.score_checkpoint import build_model, load_settings
from utils.encoder_runtime import select_encoder_device

CENTROIDS = ("centroid_x", "centroid_y", "centroid_z")


def close_arrays(arrays):
    for array in arrays.values():
        if isinstance(array, np.memmap):
            array._mmap.close()


@torch.inference_mode()
def extract_pairs(model, ds, args, device, directory):
    """A,B endpoints are adjacent rows; both modalities retain view-major batching."""
    directory.mkdir(parents=True)
    arrays, replay, rows = {}, {}, []
    before = state_digest(model)
    image_hash = hashlib.sha256()
    requests = [
        (idx, axis) for idx in range(args.subject_offset, args.subject_offset + args.num_samples) for axis in args.axes
    ]
    try:
        for start in range(0, len(requests), args.batch_size):
            samples = [render_pair(ds, idx, axis, args.eps) for idx, axis in requests[start : start + args.batch_size]]
            endpoints = [sample[state] for sample in samples for state in ("a", "b")]
            images = torch.cat([torch.stack([pair[v] for pair in endpoints]) for v in range(2)])
            for pair in endpoints:
                image_hash.update(torch.stack(pair).contiguous().numpy().tobytes())
            x = images.to(device)
            features, shape, channels = spatial.capture(model, x, args.grids, args.include_native)
            # Identical inputs, same batch layout: a numerical replay control, not
            # a second stochastic acquisition. No backward passes occur here.
            repeated, _, _ = spatial.capture(model, x, args.grids, args.include_native)
            b = len(endpoints)
            for (grid, stage), values in features.items():
                if stage not in ("backbone", "projected"):
                    continue
                if not np.isfinite(values).all() or not np.isfinite(repeated[grid, stage]).all():
                    raise ValueError("Non-finite intervention features")
                for v, view in enumerate(VIEWS):
                    key = (view, grid, stage)
                    block = values[v * b : (v + 1) * b]
                    if key not in arrays:
                        arrays[key] = np.lib.format.open_memmap(
                            directory / f"{view}_g{grid}_{stage}.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(2 * len(requests), block.shape[1]),
                        )
                        replay[key] = np.zeros(len(requests))
                    arrays[key][2 * start : 2 * start + b] = block
                    difference = block.astype(np.float64) - repeated[grid, stage][v * b : (v + 1) * b]
                    replay[key][start : start + len(samples)] = np.sqrt(
                        np.mean(difference.reshape(len(samples), -1) ** 2, axis=1)
                    )
            for sample in samples:
                if not all(lesion.any() for lesion in sample["lesions"]):
                    raise ValueError("Empty lesion endpoint; refusing to silently change the cohort")
                centroid = np.asarray(sample["centroids"])
                distance = float(np.linalg.norm(centroid[1] - centroid[0]))
                row = {
                    "subject_id": sample["index"],
                    "intervention_axis": sample["axis"],
                    "eps": args.eps,
                    "moved": distance > 1e-8,
                    "true_displacement_vox": distance,
                    "changed_lesion_voxels": int(sample["support"].sum()),
                }
                for endpoint, center, latent, lesion in zip("ab", centroid, sample["controls"], sample["lesions"]):
                    row[f"lesion_{endpoint}_voxels"] = int(lesion.sum())
                    for j, axis in enumerate("xyz"):
                        row[f"centroid_{endpoint}_{axis}"] = float(2 * center[j] / (ds.res - 1) - 1)
                        row[f"control_{endpoint}_{axis}"] = float(latent[j])
                for v, view in enumerate(VIEWS):
                    delta = (sample["b"][v] - sample["a"][v]).double()
                    row[f"{view}_input_delta_rms"] = float(delta.square().mean().sqrt())
                rows.append(row)
            print(f"  lesion moves: encoded {len(rows)}/{len(requests)} pairs", flush=True)
        if state_digest(model) != before:
            raise RuntimeError("Frozen intervention changed encoder parameters or buffers")
        for array in arrays.values():
            array.flush()
        truth = np.asarray(
            [[row[f"centroid_{endpoint}_{axis}"] for axis in "xyz"] for row in rows for endpoint in "ab"]
        )
        metadata = {
            **dataset_metadata(ds, image_hash.hexdigest(), sorted({r["subject_id"] for r in rows})),
            "native_shape": list(shape),
            "channels": channels,
            "model_state_sha256": before,
            "n_pairs": len(rows),
            "n_moved": sum(r["moved"] for r in rows),
        }
        return arrays, truth, rows, replay, metadata
    except Exception:
        close_arrays(arrays)
        raise


def feature_response(reference, fit_ids, endpoints, replay, chunk_size=2048):
    """RMS response and scale relative to variation across probe-fit subjects.

    A single RMS standard deviation calibrates units within a representation. It
    does not make response magnitude a measure of information across architectures.
    """
    n = len(endpoints) // 2
    energy, variance = np.zeros(n), 0.0
    for start in range(0, reference.shape[1], chunk_size):
        ref = np.asarray(reference[fit_ids, start : start + chunk_size], dtype=np.float64)
        values = np.asarray(endpoints[:, start : start + chunk_size], dtype=np.float64)
        energy += ((values[1::2] - values[::2]) ** 2).sum(1)
        variance += ref.var(0).sum()
    rms = np.sqrt(energy / reference.shape[1])
    relative = np.sqrt(energy / variance) if variance > 1e-12 else np.full(n, np.nan)
    return rms, relative, np.asarray(replay)


def movement_metrics(truth, prediction, subject_ids, voxel_scale, bootstrap=200, seed=1729):
    """Pair arrays have shape (pairs, 2, 3). Skill uses no-movement as its baseline."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[1:] != (2, 3):
        raise ValueError("Expected matching (pairs, 2, 3) centroid arrays")
    delta = truth[:, 1] - truth[:, 0]
    predicted = prediction[:, 1] - prediction[:, 0]
    valid = (delta**2).sum(1) > 1e-16
    result = dict(n_pairs=len(truth), n_moved=int(valid.sum()))
    names = (
        "movement_skill",
        "movement_gain",
        "movement_rmse_vox",
        "true_displacement_vox",
        "predicted_displacement_vox",
        "endpoint_mean_r2",
        "endpoint_error_vox",
        "movement_skill_ci95_low",
        "movement_skill_ci95_high",
    )
    result.update({name: np.nan for name in names})
    result["no_move_predicted_displacement_vox"] = (
        float(np.linalg.norm(predicted[~valid], axis=1).mean() * voxel_scale) if (~valid).any() else np.nan
    )
    if not valid.any():
        return result
    delta, predicted = delta[valid], predicted[valid]
    error = ((predicted - delta) ** 2).sum(1)
    energy = (delta**2).sum(1)
    scores = r2(truth[valid].reshape(-1, 3), prediction[valid].reshape(-1, 3))
    result.update(
        movement_skill=float(1 - error.sum() / energy.sum()),
        movement_gain=float((predicted * delta).sum() / energy.sum()),
        movement_rmse_vox=float(np.sqrt(error.mean()) * voxel_scale),
        true_displacement_vox=float(np.linalg.norm(delta, axis=1).mean() * voxel_scale),
        predicted_displacement_vox=float(np.linalg.norm(predicted, axis=1).mean() * voxel_scale),
        endpoint_mean_r2=float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan,
        endpoint_error_vox=float(np.linalg.norm(prediction[valid] - truth[valid], axis=2).mean() * voxel_scale),
    )
    # All x/y/z interventions from one anatomy stay together in a bootstrap draw.
    ids = np.asarray(subject_ids)[valid]
    unique = np.unique(ids)
    if bootstrap and len(unique) > 1:
        totals = np.array([(error[ids == idx].sum(), energy[ids == idx].sum()) for idx in unique])
        draws = np.random.default_rng(seed).integers(len(unique), size=(bootstrap, len(unique)))
        sampled = totals[draws].sum(1)
        low, high = np.quantile(1 - sampled[:, 0] / sampled[:, 1], (0.025, 0.975))
        result.update(movement_skill_ci95_low=float(low), movement_skill_ci95_high=float(high))
    return result


def score_features(reference, reference_targets, pairs, args, arm, directory, separate_readout=False, resolution=64):
    """Select probes only on observational fit/tune subjects, then score paired moves."""
    arrays, truth, pair_rows, replay, _ = pairs
    order = np.random.default_rng(args.seed).permutation(len(reference_targets))
    cut = int(0.75 * len(order))
    splits = (order[:cut], order[cut:], np.arange(len(order), len(order) + len(truth)))
    targets = np.concatenate((reference_targets, truth))
    rows, sensitivity, parameters = [], [], []
    predictions = {"truth": truth.reshape(-1, 2, 3)}
    subjects = np.array([row["subject_id"] for row in pair_rows])
    axes = np.array([row["intervention_axis"] for row in pair_rows])
    predictions.update(subject_id=subjects, intervention_axis=axes)
    for (view, grid, stage), feature in reference.items():
        key = (view, grid, stage)
        active_head = ("spatial" if separate_readout and grid != 1 else "global") if stage == "projected" else None
        info = dict(arm=arm, view=view, grid=grid, stage=stage, readout_head=active_head, dimensions=feature.shape[1])
        print(f"  {arm} {view} g{grid} {stage}: {feature.shape[1]} features", flush=True)
        rms, relative, null = feature_response(feature, splits[0], arrays[key], replay[key])
        for i, pair in enumerate(pair_rows):
            sensitivity.append(
                {
                    **info,
                    "subject_id": pair["subject_id"],
                    "intervention_axis": pair["intervention_axis"],
                    "moved": pair["moved"],
                    "feature_delta_rms": float(rms[i]),
                    "delta_to_subject_variation": float(relative[i]),
                    "replay_delta_rms": float(null[i]),
                }
            )
        x = np.concatenate((feature, arrays[key]))
        gram, width = block_gram(x, np.arange(x.shape[1]), splits[0])
        fitted, predicted = fit_readouts(gram, width, targets, splits, args.seed, target_names=CENTROIDS)
        parameters.extend({**info, **row} for row in fitted)
        for kind, endpoint_prediction in predicted.items():
            probe, condition = kind.split("_")
            prediction = endpoint_prediction.reshape(-1, 2, 3)
            predictions[f"{view}_g{grid}_{stage}_{kind}"] = prediction
            for axis in ("all", *args.axes):
                selected = np.ones(len(pair_rows), dtype=bool) if axis == "all" else axes == axis
                metrics = movement_metrics(
                    truth.reshape(-1, 2, 3)[selected],
                    prediction[selected],
                    subjects[selected],
                    (resolution - 1) / 2,
                    args.bootstrap,
                    args.seed,
                )
                moved = selected & np.array([pair["moved"] for pair in pair_rows])
                rows.append(
                    {
                        **info,
                        "probe": probe,
                        "condition": condition,
                        "intervention_axis": axis,
                        **metrics,
                        "feature_delta_rms": float(rms[moved].mean()) if moved.any() else np.nan,
                        "delta_to_subject_variation": float(relative[moved].mean()) if moved.any() else np.nan,
                        "replay_delta_rms": float(null[selected].mean()),
                    }
                )
        del x, gram
    np.savez_compressed(directory / f"{arm}_predictions.npz", **predictions)
    return (
        rows,
        sensitivity,
        parameters,
        {
            "fit_validation_ids": splits[0].tolist(),
            "tune_validation_ids": splits[1].tolist(),
        },
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; temporary features live here too")
    parser.add_argument(
        "--num-samples", type=int, default=64, help="Intervention subjects; each gets every requested axis"
    )
    parser.add_argument(
        "--subject-offset", type=int, default=1000, help="Test IDs start here; avoids the usual first 400"
    )
    parser.add_argument("--axes", nargs="+", choices=("x", "y", "z"), default=["x", "y", "z"])
    parser.add_argument(
        "--eps", type=float, default=0.5, help="Change one raw lesion control by -/+eps; report actual movement"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Pairs/batch (two endpoints per view per pair)")
    parser.add_argument("--grids", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--include-native", action="store_true", help="Also test native maps; uses more temporary disk/RAM"
    )
    parser.add_argument("--skip-initial", action="store_true", help="Omit the saved model_init.pt control")
    parser.add_argument(
        "--bootstrap", type=int, default=200, help="Subject bootstrap draws for movement skill; 0 disables"
    )
    parser.add_argument(
        "--seed", type=int, default=1729, help="Probe fit/tune split, shuffled-label controls and bootstrap"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    if args.num_samples < 2 or args.subject_offset < 0 or args.batch_size < 1 or args.bootstrap < 0:
        parser.error("Need >=2 subjects, nonnegative offset/bootstrap and positive batch size")
    if not np.isfinite(args.eps) or args.eps <= 0 or min(args.grids) < 1:
        parser.error("eps must be finite and positive, and grids must be positive")
    args.axes = list(dict.fromkeys(args.axes))
    args.grids = sorted(set([1, *args.grids]))
    args.stages = ("backbone", "projected")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_settings(args.run_dir)
    if cfg.get("synthetic_lesion_placement", "legacy") != "wm_interior":
        raise ValueError("This audit requires wm_interior placement so complete lesions remain inside white matter")
    if cfg["num_val_samples"] < 20:
        raise ValueError("Need at least 20 original validation subjects for probe fit/tune")
    checkpoints = {"trained": args.run_dir / args.checkpoint}
    if not args.skip_initial:
        checkpoints["initial"] = args.run_dir / "model_init.pt"
    hashes = {arm: digest(path) for arm, path in checkpoints.items()}
    settings_hash = digest(args.run_dir / "settings.json")
    device = select_encoder_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, device)
    report.update(
        target_names=CENTROIDS,
        checkpoint_sha256=hashes,
        settings_sha256=settings_hash,
        evaluation_only=True,
        feature_banks_retained=False,
        summary=[],
        sensitivity=[],
        probe_parameters=[],
        cohorts={},
        protocol="Observational validation 75/25 fit/tune; fixed-anatomy paired lesion moves on separate test subjects",
        notes=[
            "Only one raw lesion control changes by -/+eps; all other factors, acquisition draws and original normalization affine are fixed.",
            "Conditional WM placement may move multiple physical axes. Score the full actual centroid displacement, not the nominal control axis.",
            "Quantized or saturated controls can give identical lesion positions. Those pairs are counted and excluded from movement scores, never redrawn.",
            "Movement skill = 1 - sum ||predicted displacement - true displacement||² / sum ||true displacement||². Zero means predicting no movement; negative is worse.",
            "Movement gain projects predicted displacements onto true ones: 1 ideal, 0 no following, negative opposite direction. Read with skill/error.",
            "Response magnitude measures sensitivity, not decodability. It is calibrated by feature variation on probe-fit subjects, with identical-input replay as a numerical control.",
            "Bootstrap resamples subjects with their axes together; intervals condition on this checkpoint and fitted probes, not training-seed variability.",
            "Intervention labels are never used to fit/tune probes or encoders. Shuffled controls permute observational fit/tune target rows.",
            "Grid 1 projected uses the global head; larger grids use the active spatial head. No decoder is involved.",
            "These isolate an image factor; causal descendants are held fixed. Perturbations need not follow the observational joint distribution.",
        ],
    )
    save_report(args.out_dir, report)
    try:
        for arm, path in checkpoints.items():
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != hashes[arm]:
                raise ValueError("Checkpoint changed before loading")
            model = build_model(cfg, device, torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
            del blob
            before = state_digest(model)
            # Both datasets keep the original normalization convention. The test
            # reference uses the unchanged first 64 observations, even for quick runs.
            val_ds = dataset(cfg, cfg["num_val_samples"], "val")
            pair_ds = dataset(cfg, max(64, args.subject_offset + args.num_samples), "test")
            with tempfile.TemporaryDirectory(prefix=".features-", dir=args.out_dir) as tmp:
                reference, pairs = {}, None
                try:
                    reference, targets, val_meta = spatial.extract(model, val_ds, args, device, Path(tmp) / "val")
                    pairs = extract_pairs(model, pair_ds, args, device, Path(tmp) / "moves")
                    cohorts = {"val": val_meta, "moves": pairs[4]}
                    if report["cohorts"]:
                        for split in cohorts:
                            if cohorts[split]["input_sha256"] != report["cohorts"]["trained"][split]["input_sha256"]:
                                raise ValueError("Initial/trained arms received different images")
                    report["cohorts"][arm] = cohorts
                    if arm == "trained":
                        save_csv(args.out_dir / "pairs.csv", pairs[2])
                    with threadpool_limits(limits=1):
                        rows, sensitivity, parameters, split = score_features(
                            reference,
                            targets[:, 9:12],
                            pairs,
                            args,
                            arm,
                            args.out_dir,
                            model.separate_spatial_readout,
                            cfg["res"],
                        )
                    report["summary"].extend(rows)
                    report["sensitivity"].extend(sensitivity)
                    report["probe_parameters"].extend(parameters)
                    report["probe_split"] = split
                finally:
                    close_arrays(reference)
                    if pairs is not None:
                        close_arrays(pairs[0])
            if state_digest(model) != before:
                raise RuntimeError("Audit changed encoder parameters or buffers")
            save_report(args.out_dir, report)
            del model, reference, pairs, val_ds, pair_ds
        if {arm: digest(path) for arm, path in checkpoints.items()} != hashes or digest(
            args.run_dir / "settings.json"
        ) != settings_hash:
            raise ValueError("Source checkpoint/settings files changed during audit")
        report.update(status="complete", source_checkpoints_unchanged=True, encoder_unchanged=True)
        for name in ("summary", "sensitivity", "probe_parameters"):
            save_csv(args.out_dir / f"{name}.csv", report[name])
        save_report(args.out_dir, report)
        print("\nLesion movement: skill 1=perfect, 0=no movement, negative=worse; pooled over axes")
        print("arm     view  grid stage       probe  moved   response/SD    skill     gain   error(vox)")
        for row in report["summary"]:
            if row["condition"] == "observed" and row["intervention_axis"] == "all":
                print(
                    f"{row['arm']:7s} {row['view']:5s} {row['grid']:4d} {row['stage']:11s} {row['probe']:5s} "
                    f"{row['n_moved']:5d} {row['delta_to_subject_variation']:13.4g} "
                    f"{row['movement_skill']:+8.3f} {row['movement_gain']:+8.3f} {row['movement_rmse_vox']:12.3f}"
                )
        print(f"Saved frozen lesion intervention audit: {args.out_dir}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise


if __name__ == "__main__":
    main()
