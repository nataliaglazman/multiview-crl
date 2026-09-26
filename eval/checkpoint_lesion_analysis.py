"""Frozen backbone/readout probes used by eval.score_checkpoint --lesion-analysis.

Targets are diagnostic only. No gradients, optimizer, or checkpoint writes.
"""

import csv
import hashlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from eval.identifiability_metrics import cv_probe_r2_multi

VIEWS = ("t1", "flair")
TARGETS = tuple(f"{family}_{axis}" for family in ("latent", "centroid") for axis in "xyz")


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lesion_targets(inner, latents):
    """Generator controls and support centroid in renderer coordinates, before blur."""
    if inner.mode != "pseudo_mri" or inner.renderer.lesion_mode != "sphere":
        raise ValueError("Lesion position analysis requires pseudo_mri with lesion_mode='sphere'")
    _, support = inner.renderer.render_structure(
        latents["z_content"],
        latents["z_deformation"],
        latents["z_fissure"],
        device="cpu",
        clean=inner.clean_content,
    )
    mass = float(support.sum())
    centroid = np.full(3, np.nan)
    if mass > 0:
        centroid = ((support[..., None] * inner.renderer.coords).sum((0, 1, 2)) / mass).numpy()
    return np.concatenate((latents["z_content"][2:5].numpy(), centroid)), mass


@torch.inference_mode()
def batch_features(model, x, grids):
    """One real forward; capture backbone maps and the encoding's content and style units.

    ``projected`` is the content block and ``style`` the units after it; ``style`` is
    omitted when ``latent_dim == content_channels``.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Frozen lesion analysis requires model.eval(), including all BatchNorm modules")
    captured = {}

    def tap(name):
        def hook(module, inputs, output):
            captured[name] = output

        return hook

    with ExitStack() as stack:
        for name in ("encoder", "encoder_v1", "to_encoding"):
            module = getattr(model, name, None)
            if module is not None:
                handle = module.register_forward_hook(tap(name))
                stack.callback(handle.remove)
        global_code = model(x, pool_only=True, n_views=2)[2][0]
    h = captured["encoder"]
    if model.encoder_v1 is not None:
        h = torch.cat([h, captured["encoder_v1"]], dim=0)
    if h.ndim != 5 or h.shape[0] != x.shape[0]:
        raise ValueError("Expected spatial backbone maps in T1-then-FLAIR batch order")
    result = {}
    for grid in grids:
        if grid < 1 or any(grid > size for size in h.shape[2:]):
            raise ValueError(f"Lesion grid {grid} must fit the backbone map {tuple(h.shape[2:])}")
        pooled = F.adaptive_avg_pool3d(h, (grid,) * 3)
        if grid == 1:
            code = global_code
        elif model.encoder_architecture == "resnet18":
            # Nonlinear ResNet head is applied AFTER bin averaging, as in model.forward.
            code = model.to_encoding(pooled.flatten(2).transpose(1, 2)).transpose(1, 2)
        else:
            code = F.adaptive_avg_pool3d(captured["to_encoding"], (grid,) * 3)
        result[(grid, "backbone")] = pooled.flatten(1).cpu().numpy()
        result[(grid, "projected")] = code[:, : model.content_channels].flatten(1).cpu().numpy()
        if model.latent_dim > model.content_channels:
            result[(grid, "style")] = code[:, model.content_channels :].flatten(1).cpu().numpy()
    return result, tuple(h.shape[2:]), int(h.shape[1])


def extract(model, ds, device, batch_size, grids, with_targets=True):
    """Read the same deterministic subjects in order, retaining pooled features only."""
    before = state_digest(model)
    image_digest = hashlib.sha256()
    chunks, targets, masses, ids = {}, [], [], []
    spatial_shape, channels = None, None
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        x = torch.cat(batch["image"], dim=0)
        image_digest.update(x.contiguous().numpy().tobytes())
        features, spatial_shape, channels = batch_features(model, x.to(device), grids)
        b = len(batch["index"])
        for (grid, stage), values in features.items():
            for view, block in zip(VIEWS, (values[:b], values[b:])):
                chunks.setdefault((view, grid, stage), []).append(block)
        for j, idx in enumerate(batch["index"].tolist()):
            ids.append(int(idx))
            if with_targets:
                latents = {key: value[j] for key, value in batch["gt_latents"].items()}
                target, mass = lesion_targets(ds._inner, latents)
                targets.append(target)
                masses.append(mass)
    after = state_digest(model)
    if after != before:
        raise RuntimeError("Registered model state changed during frozen lesion extraction")
    return {
        "features": {key: np.concatenate(value) for key, value in chunks.items()},
        "targets": np.asarray(targets),
        "masses": np.asarray(masses),
        "ids": np.asarray(ids),
        "input_sha256": image_digest.hexdigest(),
        "model_sha256": before,
        "spatial_shape": spatial_shape,
        "backbone_channels": channels,
    }


def score_features(X, Y, permutations, seeds=(0, 1, 2), folds=5):
    """Batched ridge CV with per-target alpha selection and matched permutation nulls."""
    if len(Y) < folds * 4:
        raise ValueError(f"Need at least {folds * 4} subjects with nonempty lesions; got {len(Y)}")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Probe features and retained targets must be finite")
    augmented = np.concatenate([Y] + [Y[order] for order in permutations], axis=1)
    # sklearn's default R² makes a constant test target look perfect if predicted
    # exactly. Report it as undefined instead, including for a shuffled target.
    defined = np.ones(augmented.shape[1], dtype=bool)
    for seed in seeds:
        for _, test in KFold(folds, shuffle=True, random_state=seed).split(Y):
            defined &= np.ptp(augmented[test], axis=0) > 1e-12
    means = np.full(augmented.shape[1], np.nan)
    spreads = means.copy()
    if defined.any():
        with threadpool_limits(limits=1):
            scores = cv_probe_r2_multi(X, augmented[:, defined], n_splits=folds, seeds=seeds)
        means[defined], spreads[defined] = scores["mean"], scores["std"]
    per_target = {}
    for j, name in enumerate(TARGETS[: Y.shape[1]]):
        null = means[Y.shape[1] + j :: Y.shape[1]]
        per_target[name] = {
            "r2": float(means[j]),
            "seed_std": float(spreads[j]),
            "shuffled_r2": null.tolist(),
            "shuffled_mean": float(null.mean()),
            "above_shuffle": float(means[j] - null.mean()),
            "status": "ok" if defined[j] else "constant_in_test_fold",
        }
    return per_target


def run_analysis(model, floor_factory, ds, device, batch_size, grids=(1, 4), n_shuffles=3, seed=1729):
    """Score trained and optional untrained maps with identical subjects and folds."""
    grids = sorted(set((1, *grids)))  # Always include the actual training GAP readout.
    if n_shuffles < 1:
        raise ValueError("At least one shuffled-label control is required")
    if ds._inner.mode != "pseudo_mri" or ds._inner.renderer.lesion_mode != "sphere":
        raise ValueError("Lesion position analysis requires pseudo_mri with lesion_mode='sphere'")
    print("\nLesion analysis: extracting trained backbone, content and style features...", flush=True)
    trained = extract(model, ds, device, batch_size, grids)
    Y = trained.pop("targets")
    masses = trained.pop("masses")
    valid = (masses > 0) & np.isfinite(Y).all(axis=1)
    kept = np.flatnonzero(valid)
    if len(kept) < 20:
        raise ValueError(f"Need at least 20 subjects with nonempty lesions; got {len(kept)}/{len(Y)}")
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(kept)) for _ in range(n_shuffles)]
    report = {
        "probe": "ridge_cv",
        "folds": 5,
        "cv_seeds": [0, 1, 2],
        "shuffle_seed": seed,
        "n_shuffles": n_shuffles,
        "grids": grids,
        "target_names": list(TARGETS),
        "centroid_coordinates": "renderer xyz in [-1, 1], lesion support before intensity rendering and blur",
        "subjects": {
            "ids": trained["ids"].tolist(),
            "valid": valid.tolist(),
            "lesion_mass_voxels": masses.tolist(),
            "targets": Y.tolist(),
        },
        "retained_subject_ids": trained["ids"][valid].tolist(),
        "permuted_subject_ids": [trained["ids"][kept[order]].tolist() for order in permutations],
        "n_valid": int(valid.sum()),
        "n_total": len(Y),
        "rows": [],
        "state": {},
        "notes": [
            "All stages, views and controls use the same subjects and CV folds; scaling is fitted within each training fold.",
            "Regularization is selected separately per target with RidgeCV on each training fold; test folds are not used.",
            "seed_std is spread across three CV-seed means, not a confidence interval.",
            "Shuffles permute whole six-target rows; axes and latent/centroid relationships are kept together.",
            "Backbone and projected feature counts differ; probe scores measure accessibility, not total information.",
            "'projected' is the encoding's content block and 'style' the units after it (absent with no style units).",
            "wm_interior controls are anatomy-dependent quantiles, not Cartesian positions; voxelization can be many-to-one.",
            "ResNet patch readouts apply the nonlinear head after pooling each bin; their mean need not equal GAP.",
            "The untrained twin is a seeded initialization reference, not a saved pre-training checkpoint.",
        ],
    }

    def score_arm(arm, extracted):
        report["state"][arm] = {key: value for key, value in extracted.items() if key not in ("features", "ids")}
        report["state"][arm]["unchanged"] = True
        for (view, grid, stage), features in extracted["features"].items():
            print(f"  {arm}: {view} {stage} grid={grid}, {features.shape[1]} features", flush=True)
            scores = score_features(features[valid], Y[valid], permutations)
            for target, values in scores.items():
                report["rows"].append(
                    {
                        "arm": arm,
                        "view": view,
                        "grid": grid,
                        "stage": stage,
                        "n_features": features.shape[1],
                        "n_subjects": int(valid.sum()),
                        "target": target,
                        **values,
                    }
                )

    score_arm("trained", trained)
    trained.pop("features")  # Release large matrices before encoding the untrained twin.
    if floor_factory is not None:
        print("Lesion analysis: extracting the untrained reference...", flush=True)
        untrained_model = floor_factory()
        untrained = extract(untrained_model, ds, device, batch_size, grids, with_targets=False)
        untrained.pop("targets")
        untrained.pop("masses")
        if not np.array_equal(trained["ids"], untrained["ids"]) or trained["input_sha256"] != untrained["input_sha256"]:
            raise RuntimeError("Trained and untrained probes did not receive identical subjects/images")
        score_arm("untrained", untrained)
        del untrained, untrained_model
    by_key = {(r["arm"], r["view"], r["grid"], r["stage"], r["target"]): r for r in report["rows"]}
    for row in report["rows"]:
        reference = by_key.get(("untrained", row["view"], row["grid"], row["stage"], row["target"]))
        row["delta_untrained"] = row["r2"] - reference["r2"] if row["arm"] == "trained" and reference else None
        backbone = by_key[(row["arm"], row["view"], row["grid"], "backbone", row["target"])]
        gap = by_key[(row["arm"], row["view"], 1, row["stage"], row["target"])]
        row["delta_backbone"] = row["r2"] - backbone["r2"] if row["stage"] != "backbone" else None
        row["delta_gap"] = row["r2"] - gap["r2"] if row["grid"] != 1 else None
    report["centroid_to_latent_reference"] = {
        "description": "Ridge from true centroid to latent controls, on the same folds. Not an upper bound: it omits anatomy.",
        "scores": score_features(Y[valid, 3:], Y[valid, :3], permutations),
    }
    return report


def print_analysis(report):
    print(f"\n=== lesion location analysis: {report['n_valid']}/{report['n_total']} nonempty lesions ===")
    print("Mean xyz cross-validated ridge R²; per-axis scores and null repeats are saved.")
    print("view   grid stage       dims target      trained untrained shuffled    delta")
    rows = report["rows"]
    stages = [stage for stage in ("backbone", "projected", "style") if any(r["stage"] == stage for r in rows)]
    for view in VIEWS:
        for grid in report["grids"]:
            for stage in stages:
                for family in ("latent", "centroid"):
                    selected = [
                        r
                        for r in rows
                        if r["view"] == view
                        and r["grid"] == grid
                        and r["stage"] == stage
                        and r["target"].startswith(family)
                    ]
                    real = [r for r in selected if r["arm"] == "trained"]
                    floor = [r["r2"] for r in selected if r["arm"] == "untrained"]
                    scores = [
                        np.mean([r["r2"] for r in real]),
                        np.mean(floor) if floor else np.nan,
                        np.mean([r["shuffled_mean"] for r in real]),
                    ]
                    scores.append(scores[0] - scores[1])
                    values = " ".join(f"{s:+9.3f}" if np.isfinite(s) else f"{'n/a':>9}" for s in scores)
                    print(f"{view:<6} {grid:4} {stage:<10} {real[0]['n_features']:5} {family:<9} {values}")
    reference = report["centroid_to_latent_reference"]["scores"]
    print(
        "True-centroid -> latent ridge reference (not an upper bound): "
        + ", ".join(f"{axis}: {reference[f'latent_{axis}']['r2']:+.3f}" for axis in "xyz")
    )
    print("No registered model parameter or buffer changed. Negative R² values are retained.", flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_tables(report, output):
    """Save per-axis scores and targets alongside the enclosing score report JSON."""
    output = Path(output)
    score_path = output.with_name(output.stem + "_lesion_scores.csv")
    target_path = output.with_name(output.stem + "_lesion_targets.csv")
    fields = [key for key in report["rows"][0] if key != "shuffled_r2"]
    fields += [f"shuffled_r2_{j}" for j in range(report["n_shuffles"])]
    with score_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            flat = {key: value for key, value in row.items() if key != "shuffled_r2"}
            flat.update({f"shuffled_r2_{j}": value for j, value in enumerate(row["shuffled_r2"])})
            writer.writerow(json_safe(flat))
    with target_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "valid", "lesion_mass_voxels", *TARGETS])
        subjects = report["subjects"]
        for idx, valid, mass, target in zip(
            subjects["ids"], subjects["valid"], subjects["lesion_mass_voxels"], subjects["targets"]
        ):
            writer.writerow(json_safe([idx, valid, mass, *target]))
    return score_path, target_path
