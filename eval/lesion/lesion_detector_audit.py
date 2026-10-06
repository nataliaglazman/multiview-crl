"""Native lesion-head localisation and movement, with no fitted coordinate probe.

python -m eval.lesion.lesion_detector_audit --run-dir RUN --out-dir NEW_DIRECTORY

Validation labels select one head per view and checkpoint for a diagnostic only;
test labels and interventions never select heads or update the encoder. All heads
and fixed residual-peak controls are also reported. Positions use input voxel
indices, accounting for the actual pooling bins (not res-1 times cell coordinates).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from eval.encoder.encoder_lesion_intervention import movement_metrics
from eval.encoder.encoder_target_protocol import digest, provenance, save_csv, save_report
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.lesion.lesion_routing import render_pair
from eval.protocol.score_checkpoint import build_model
from eval.synthetic.factor_structure_audit import restore_dataset
from eval.synthetic.synthetic_dataset import LesionPlacementError

VIEWS = ("t1", "flair")


def native_positions(maps, input_shape):
    """Attention expectations in input voxel indices, (batch, heads, xyz).

    The feature grid is interpreted as adaptive-pooling bins. This is exact for
    the residual-input branch, including nondivisible input sizes. For backbone
    features it is a nominal cell location, not a receptive-field localisation.
    """
    axes = []
    for size, bins in zip(input_shape, maps.shape[2:]):
        idx = torch.arange(bins, device=maps.device, dtype=maps.dtype)
        start = torch.floor(idx * size / bins)
        stop = torch.ceil((idx + 1) * size / bins)
        axes.append((start + stop - 1) / 2)
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    return maps.flatten(2) @ grid


def true_centroid(dataset, idx):
    """Evaluation-only lesion-mask centroid, independent of a raw placement control."""
    inner = dataset._inner
    _, _, lat = inner[idx]
    _, lesion = inner.renderer.render_structure(
        lat["z_content"],
        lat["z_deformation"],
        lat["z_fissure"],
        "cpu",
        clean=inner.clean_content,
        z_lesion=lat.get("z_lesion"),
    )
    mask = lesion.numpy() > 0
    return np.argwhere(mask).mean(0) if mask.any() else np.full(3, np.nan)


def localization_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("Expected matching (subjects, 3) voxel positions")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predicted positions")
    valid = np.isfinite(truth).all(1)
    error = np.linalg.norm(prediction[valid] - truth[valid], axis=1)
    return dict(
        n_subjects=len(truth),
        n_nonempty=int(valid.sum()),
        n_empty=int((~valid).sum()),
        mean_error_vox=float(error.mean()) if len(error) else None,
        median_error_vox=float(np.median(error)) if len(error) else None,
        hit_within_3_vox=float(np.mean(error <= 3)) if len(error) else None,
    )


@torch.inference_mode()
def predict(model, images, device):
    """Images (B, 2, 1, D,H,W) -> named predictions (B, 2, 3). No targets accepted."""
    b = len(images)
    x = torch.cat([images[:, 0], images[:, 1]]).to(device)
    maps = model.lesion_maps(x, n_views=2)
    coords = native_positions(maps, x.shape[2:]).cpu().numpy()
    coords = np.stack([coords[:b], coords[b:]], axis=1)
    output = {f"head_{k}": coords[:, :, k] for k in range(coords.shape[2])}
    if model.normative is not None:
        for name in ("residual_peak_unsigned", "residual_peak_polarity_prior"):
            output[name] = np.empty((b, 2, 3))
        for v in range(2):
            view = x[v * b : (v + 1) * b]
            residual = model.normative(view, v)[:, 0].cpu().numpy()
            masks = view[:, 0].cpu().numpy() != 0
            for i, (z, mask) in enumerate(zip(residual, masks)):
                if not mask.any():
                    raise ValueError("Cannot localise in an empty brain")
                # Same fixed 1-input-voxel smoothing in every arm. The polarity
                # control explicitly uses prior knowledge: dark T1, bright FLAIR.
                scores = (np.abs(z), np.maximum((-1 if v == 0 else 1) * z, 0))
                for name, score in zip(("residual_peak_unsigned", "residual_peak_polarity_prior"), scores):
                    score = np.where(mask, gaussian_filter(score, sigma=1), -np.inf)
                    output[name][i, v] = np.unravel_index(np.argmax(score), score.shape)
    return output


def observations(model, cfg, split, count, batch_size, device):
    ds, _ = restore_dataset(cfg, max(64, count), split)
    predictions, truth = {}, []
    for start in range(0, count, batch_size):
        ids = range(start, min(start + batch_size, count))
        images = torch.stack([torch.stack(ds[i]["image"]) for i in ids])
        result = predict(model, images, device)
        for key, values in result.items():
            predictions.setdefault(key, []).append(values)
        truth.extend(true_centroid(ds, i) for i in ids)
    return np.asarray(truth), {k: np.concatenate(v) for k, v in predictions.items()}, ds


def select_heads(truth, predictions):
    """Select on validation only; never take the nearest head per test subject."""
    heads = [name for name in predictions if name.startswith("head_")]
    if not np.isfinite(truth).all():
        raise ValueError("Empty validation lesions; audit cohort is not silently changed")
    return [min(heads, key=lambda key: np.linalg.norm(predictions[key][:, v] - truth, axis=1).mean()) for v in range(2)]


def add_selected(predictions, selected):
    predictions["selected_validation_head"] = np.stack(
        [predictions[key][:, v] for v, key in enumerate(selected)], axis=1
    )


def interventions(model, cfg, args, device):
    # Disjoint from observational test subjects, even in a tiny smoke run.
    offset = args.num_samples + 1000
    ds, _ = restore_dataset(cfg, offset + args.movement_subjects, "test")
    truth, predictions, ids, failures = [], {}, [], []
    for idx in range(offset, offset + args.movement_subjects):
        for axis in "xyz":
            try:
                sample = render_pair(ds, idx, axis, args.eps)
            except LesionPlacementError as error:
                failures.append(dict(subject_id=idx, axis=axis, reason=str(error)))
                continue
            centers = np.asarray(sample["centroids"])
            if not np.isfinite(centers).all():
                failures.append(dict(subject_id=idx, axis=axis, reason="empty lesion endpoint"))
                continue
            images = torch.stack([torch.stack(sample[end]) for end in ("a", "b")])
            for name, value in predict(model, images, device).items():
                predictions.setdefault(name, []).append(value)
            truth.append(centers)
            ids.append(idx)
        if (idx - offset + 1) % 16 == 0:
            print(f"  movement subjects: {idx - offset + 1}/{args.movement_subjects}", flush=True)
    if not truth:
        raise ValueError("No valid movement pairs")
    return np.asarray(truth), {k: np.asarray(v) for k, v in predictions.items()}, np.asarray(ids), failures


def run_audit(args):
    cfg = json.loads((args.run_dir / "settings.json").read_text())
    if cfg.get("lesion_keypoints", 0) < 1 or cfg.get("synthetic_lesion_target", "position") != "position":
        raise ValueError("This audit requires a single-lesion position branch")
    if cfg.get("synthetic_lesion_placement", "legacy") != "wm_interior":
        raise ValueError("Movement audit requires wm_interior placement")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, device)
    report["protocol"] = {
        "positions": "input voxel indices, Euclidean distance; nominal adaptive-pool bin centres",
        "head_selection": "one head per view/checkpoint, selected by validation mean voxel error; test untouched",
        "baselines": "full-resolution PCA residual peaks, fixed sigma=1 voxel; polarity prior uses dark T1/bright FLAIR",
        "movement": "test subjects offset by num_samples+1000; each control +/-eps with fixed anatomy/acquisition",
        "labels": "evaluation/head-selection only, never encoder training",
    }
    save_report(args.out_dir, report)
    rows, moves, arrays = [], [], {}
    norm_reference = None
    selections = {}
    for arm, checkpoint in (("initial", "model_init.pt"), ("trained", args.checkpoint)):
        path = args.run_dir / checkpoint
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(cfg, device, state)
        if model.normative is not None:
            if not bool(model.normative.fitted):
                raise ValueError("Initial checkpoint must include the fitted normative model")
            normative = {k: v.cpu() for k, v in model.normative.state_dict().items()}
            if norm_reference is None:
                norm_reference = normative
            elif any(not torch.equal(v, norm_reference[k]) for k, v in normative.items()):
                raise ValueError("PCA buffers differ between initial and trained checkpoint")
        print(f"Auditing {arm}: {path}", flush=True)
        val_truth, val_pred, _ = observations(model, cfg, "val", args.validation_samples, args.batch_size, device)
        selected = select_heads(val_truth, val_pred)
        selections[arm] = dict(zip(VIEWS, selected))
        truth, pred, ds = observations(model, cfg, "test", args.num_samples, args.batch_size, device)
        if not np.isfinite(truth).all():
            raise ValueError("Empty test lesions; audit cohort is not silently changed")
        add_selected(pred, selected)
        arrays[f"{arm}_truth"] = truth
        arrays[f"{arm}_validation_truth"] = val_truth
        for name, values in val_pred.items():
            arrays[f"{arm}_validation_{name}"] = values
        for name, values in pred.items():
            arrays[f"{arm}_{name}"] = values
            for v, view in enumerate(VIEWS):
                rows.append(
                    dict(
                        arm=arm,
                        view=view,
                        method=name,
                        selected_head=selected[v] if name == "selected_validation_head" else "",
                        **localization_metrics(truth, values[:, v]),
                    )
                )
        move_truth, move_pred, ids, failures = interventions(model, cfg, args, device)
        # Pair predictions are (pairs, endpoints, views, xyz).
        move_pred["selected_validation_head"] = np.stack(
            [move_pred[key][:, :, v] for v, key in enumerate(selected)], axis=2
        )
        arrays[f"{arm}_movement_truth"] = move_truth
        arrays[f"{arm}_movement_subjects"] = ids
        for name, values in move_pred.items():
            arrays[f"{arm}_movement_{name}"] = values
            for v, view in enumerate(VIEWS):
                moves.append(
                    dict(
                        arm=arm,
                        view=view,
                        method=name,
                        **movement_metrics(
                            move_truth, values[:, :, v], ids, voxel_scale=1, bootstrap=args.bootstrap, seed=args.seed
                        ),
                    )
                )
        report[arm] = dict(
            checkpoint_sha256=digest(path),
            normative_sha256=state_digest(model.normative) if model.normative is not None else None,
            selected_heads=selections[arm],
            movement_failures=failures,
            requested_pairs=args.movement_subjects * 3,
            test_subject_seeds=[ds._inner.sample_seed_for(i) for i in range(args.num_samples)],
        )
        del model
    for table, names in (
        (rows, ("mean_error_vox", "hit_within_3_vox")),
        (moves, ("movement_skill", "movement_rmse_vox")),
    ):
        initial = {(r["view"], r["method"]): r for r in table if r["arm"] == "initial"}
        for row in table:
            for name in names:
                row[f"delta_{name}_vs_initial"] = row[name] - initial[row["view"], row["method"]][name]
    save_csv(args.out_dir / "localization.csv", rows)
    save_csv(args.out_dir / "movement.csv", moves)
    np.savez_compressed(args.out_dir / "predictions.npz", **arrays)
    report.update(status="complete", localization=rows, movement=moves)
    save_report(args.out_dir, report)
    print("Native held-out localisation (validation-selected head):", flush=True)
    for row in rows:
        if row["method"] == "selected_validation_head":
            print(
                f"  {row['arm']:8s} {row['view']:5s} {row['selected_head']}: "
                f"error {row['mean_error_vox']:.3f} vox; hit<=3 {row['hit_within_3_vox']:.3f}"
            )
    print(f"Saved native lesion audit: {args.out_dir}", flush=True)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--validation-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--movement-subjects", type=int, default=64)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args(argv)
    if min(args.num_samples, args.validation_samples, args.batch_size, args.movement_subjects) < 1:
        p.error("Sample counts and batch size must be positive")
    if args.bootstrap < 0 or not np.isfinite(args.eps) or args.eps <= 0:
        p.error("Need nonnegative bootstrap count and finite positive eps")
    return args


if __name__ == "__main__":
    run_audit(parse_args())
