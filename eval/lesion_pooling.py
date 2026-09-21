"""Frozen lesion-location readouts: global statistics, regional moments and spatial patches.

Run separately for baseline and contrastive checkpoints. No representation training.
See LESION_POOLING.md. Sphere-mode lesions have three location factors, not a
per-subject radius factor; both latent coordinates and rendered centroids are tested.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from eval import ventricle_pooling as vp
from eval.lesion_probe import block_gram, fit_probes
from eval.lesion_reconstruction import json_safe, make_dataset

logger = logging.getLogger(__name__)
POOLS = (*vp.POOLS, "patch_flat")


class LesionSubjects:
    """Add evaluation targets without changing images or using labels for pooling."""

    def __init__(self, dataset, order):
        self.dataset, self.order = dataset, np.asarray(order)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        import torch

        source = int(self.order[index])
        item = dict(self.dataset[source])
        lat = item["gt_latents"]
        inner = self.dataset._inner
        _, lesion = inner.renderer.render_structure(
            lat["z_content"],
            lat["z_deformation"],
            lat["z_fissure"],
            "cpu",
            clean=inner.clean_content,
        )
        positions = torch.nonzero(lesion > 0)
        if len(positions) == 0:
            raise ValueError(f"Subject {source} has no rendered lesion; its location is undefined.")
        # Tensor-axis voxel coordinates, matching renderer.coords; not scanner RAS.
        item["lesion_targets"] = torch.cat((positions.float().mean(0), lat["z_content"][2:5].float()))
        return item


def mean_r2(y, pred):
    denom = ((y - y.mean(0)) ** 2).sum(0)
    if np.any(denom <= 1e-12):
        raise ValueError("All three lesion coordinates must vary to compare localization R².")
    return float(np.mean(1 - ((y - pred) ** 2).sum(0) / denom))


def paired_comparison(y, pred, reference, seed, draws):
    """Resample subjects jointly across coordinates; conditional on fitted readouts."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ids = rng.integers(len(y), size=len(y))
        if np.any(np.var(y[ids], axis=0) <= 1e-12):
            continue
        values.append(mean_r2(y[ids], pred[ids]) - mean_r2(y[ids], reference[ids]))
    if not values:
        raise ValueError("No valid paired bootstrap draws.")
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "delta_mean_r2": mean_r2(y, pred) - mean_r2(y, reference),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def validate_cache(arrays, metadata):
    if metadata.get("format") != "lesion_pooling_v1":
        raise ValueError("Expected a lesion_pooling feature cache (ventricular caches have no lesion targets).")
    y = arrays["targets"]
    if y.ndim != 2 or y.shape[1] != 6 or not np.isfinite(y).all():
        raise ValueError("Expected finite (N,6) physical and latent lesion coordinates.")
    splits = tuple(arrays[k] for k in ("train", "validation", "test"))
    all_ids = np.concatenate(splits)
    if any(len(s) < 4 for s in splits) or not np.array_equal(np.sort(all_ids), np.arange(len(y))):
        raise ValueError("Train/validation/test must partition subjects exactly once, with >=4 subjects each.")
    if len(np.unique(arrays["subject_id"])) != len(y):
        raise ValueError("Duplicate source subjects in the cache.")
    for i, split in enumerate(splits):
        if np.any(np.var(y[split], axis=0) <= 1e-12):
            raise ValueError("Every lesion coordinate must vary in each subject partition.")
        for other in splits[i + 1 :]:
            if set(arrays["mask_group"][split]) & set(arrays["mask_group"][other]):
                raise ValueError("Foreground mask groups cross subject partitions.")
    for view in vp.VIEWS:
        for pool in POOLS:
            X = arrays[f"{view}/{pool}"]
            if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
                raise ValueError(f"Invalid descriptors for {view}/{pool}.")
    return splits


def evaluate(arrays, metadata, seed=0, draws=500):
    """Reuse the existing dual/kernel lesion probe, with float64 Gram matrices.

    Scaling fits on train only; validation selects kernel hyperparameters. Test
    coordinates never select a model. No PCA: even the flattened patch readout
    solves an N_train x N_train system instead of a feature-width system.
    """
    splits = validate_cache(arrays, metadata)
    results, predictions = [], {}
    for view in vp.VIEWS:
        for pool in POOLS:
            X = arrays[f"{view}/{pool}"]
            gram, active = block_gram(X, np.arange(X.shape[1]), splits[0])
            if active == 0:
                # A zero kernel predicts the TRAIN target mean and must be reported,
                # rather than silently removing a collapsed descriptor from the table.
                active_width = 1
            else:
                active_width = active
            rows, pred = fit_probes(gram, active_width, arrays["targets"], splits, seed)
            for row in rows:
                results.append(
                    {
                        "view": view,
                        "pooling": pool,
                        "width": X.shape[1],
                        "active_width": active,
                        **row,
                    }
                )
            for (probe, target), value in pred.items():
                predictions[f"{view}/{pool}/{probe}/{target}"] = value
            logger.info(
                "Fit %s/%s: %d descriptors, %d varying training columns",
                view,
                pool,
                X.shape[1],
                active,
            )
    table = []
    for row in results:
        scores = row["test"]
        axes = np.asarray(scores["r2_xyz"], dtype=float)
        entry = {
            k: row[k]
            for k in (
                "view",
                "pooling",
                "width",
                "active_width",
                "probe",
                "condition",
                "target",
            )
        }
        entry.update(
            r2_axis0=float(axes[0]),
            r2_axis1=float(axes[1]),
            r2_axis2=float(axes[2]),
            mean_r2=float(axes.mean()),
            median_error_vox=scores.get("median_error_vox"),
            alpha=row["alpha"],
            gamma=row["gamma"],
            validation_mse_standardized=row["validation_mse_standardized"],
        )
        for reference in ("gap", "mean_std", "patch_flat"):
            comparison = {"delta_mean_r2": None, "ci_low": None, "ci_high": None}
            if row["condition"] == "observed":
                prefix = f"{row['view']}/"
                suffix = f"/{row['probe']}/{row['target']}"
                sl = slice(0, 3) if row["target"] == "physical" else slice(3, 6)
                comparison = paired_comparison(
                    arrays["targets"][splits[2], sl],
                    predictions[prefix + row["pooling"] + suffix],
                    predictions[prefix + reference + suffix],
                    seed,
                    draws,
                )
            entry.update({f"vs_{reference}_{k}": v for k, v in comparison.items()})
        table.append(entry)
    predictions.update(targets=arrays["targets"][splits[2]], subject_id=arrays["subject_id"][splits[2]])
    return results, table, predictions


def run(cli):
    if cli.threads < 1 or cli.bootstrap < 1:
        raise ValueError("Threads and bootstrap draws must be positive.")
    root = Path(cli.run_dir) if cli.run_dir else Path(cli.features).resolve().parent
    output = Path(cli.out) if cli.out else root / datetime.now().strftime("lesion_pooling_%Y%m%d_%H%M%S_%f")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if cli.features:
        with np.load(cli.features, allow_pickle=False) as cache:
            metadata = json.loads(cache["metadata_json"].item())
            arrays = {k: cache[k] for k in cache.files if k != "metadata_json"}
    else:
        import torch

        from eval.reconstruction_attribution import frozen_checkpoint
        from eval.run_dci_synthetic import load_model_from_run_dir, load_run_args
        from eval.ventricle_routing import stable_replay_math

        if min(cli.train_samples, cli.val_samples, cli.test_samples) < 4 or cli.encode_batch < 1:
            raise ValueError("Use at least four subjects per partition and a positive encode batch.")
        torch.set_num_threads(cli.threads)
        args = load_run_args(cli.run_dir)
        if getattr(args, "synthetic_lesion_mode", "sphere") != "sphere":
            raise ValueError("This is a sphere-lesion location test; field-mode z_content[2:5] are inactive.")
        grids = getattr(args, "patch_grid_per_level", None)
        grid = grids[cli.level] if grids and 0 <= cli.level < len(grids) else getattr(args, "patch_grid", None)
        grid = cli.grid or grid
        vp.validate_settings(args, cli.level, grid, cli.regions)
        mask_batch = cli.mask_batch_size or int(getattr(args, "batch_size", 64))
        if mask_batch < 1:
            raise ValueError("Mask batch must be positive.")
        checkpoint = Path(cli.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path(cli.run_dir) / checkpoint
        checkpoint = checkpoint.resolve()
        model, args, device = load_model_from_run_dir(cli.run_dir, str(checkpoint), cli.device, seed=cli.seed)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = state.get("step")
        weights = state.get("encoders", state)
        model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
        del weights, state
        n = cli.train_samples + cli.val_samples + cli.test_samples
        ds = make_dataset(args, n, cli.causal, "test")
        # Randomize source IDs, then keep train/validation/test masks strictly separate.
        order = np.random.default_rng(cli.seed).permutation(n)
        subjects = LesionSubjects(ds, order)
        fit_end = cli.train_samples + cli.val_samples
        ends = [cli.train_samples, fit_end, n]
        with stable_replay_math(), frozen_checkpoint(model):
            arrays, details = vp.extract_features(
                model,
                subjects,
                args,
                device,
                grid,
                cli.regions,
                fit_end,
                mask_batch,
                cli.encode_batch,
                cli.level,
                target_key="lesion_targets",
                include_patch_flat=True,
                partition_ends=ends,
            )
        arrays.update(
            subject_id=order,
            train=np.arange(cli.train_samples),
            validation=np.arange(cli.train_samples, fit_end),
            test=np.arange(fit_end, n),
        )
        metadata = {
            "format": "lesion_pooling_v1",
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "settings": vars(args),
            "causal": cli.causal,
            "dataset_split": "test",
            "source_order_seed": cli.seed,
            "render_resolution": ds.res,
            "grid": list(grid),
            "regions": list(cli.regions),
            "level": cli.level,
            "mask_batch_size": mask_batch,
            "encode_batch": cli.encode_batch,
            "lesion_radius": ds._inner.renderer.lesion_radius,
            "feature_stage": "uncentered, unquantized content encoder patches from forward",
            "target_columns": [
                "centroid_axis0_vox",
                "centroid_axis1_vox",
                "centroid_axis2_vox",
                "z_content_2",
                "z_content_3",
                "z_content_4",
            ],
            **details,
        }
    validate_cache(arrays, metadata)
    output.mkdir(parents=True)
    vp.save_cache(output / "features.npz", arrays, metadata)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=cli.threads):
        details, table, predictions = evaluate(arrays, metadata, cli.seed, cli.bootstrap)
    report = {
        "metadata": metadata,
        "cli": vars(cli),
        "results": details,
        "comparisons": table,
        "limitations": [
            "Localization probes, not lesion-presence detection, segmentation or decoder routing.",
            "Sphere radius is a saved constant, not a per-subject target. Empty lesions are rejected.",
            "Labels train diagnostic readouts only. The representation checkpoint remains frozen.",
            "All summaries share masks and subjects. Global moments/extrema discard spatial order.",
            "Flattened patches preserve retained positions but not detail already averaged within patches.",
            "Dimensions and finite-sample readout difficulty differ. Probe failure does not prove absent information.",
            "Hyperparameters use validation, not test. Feature scaling uses train only. No final refit on validation.",
            "Intervals condition on fitted probes, split and masks; no training-seed uncertainty or multiplicity correction.",
            "IID evaluation removes SCM correlations but may be outside training distribution.",
        ],
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
    with (output / "pooling_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez_compressed(output / "predictions.npz", **predictions)
    print("\nHeld-out lesion LOCATION readouts (R² averaged over three coordinates):")
    nulls = {
        (r["view"], r["pooling"], r["probe"], r["target"]): r["mean_r2"] for r in table if r["condition"] == "shuffled"
    }
    for r in table:
        if r["condition"] != "observed":
            continue
        null = nulls[r["view"], r["pooling"], r["probe"], r["target"]]
        error = f" median_error={r['median_error_vox']:.2f}vox" if r["target"] == "physical" else ""
        print(
            f"{r['view']:5} {r['pooling']:18} {r['probe']:5} {r['target']:8} "
            f"d={r['width']:5} R²={r['mean_r2']:+.3f} shuffled={null:+.3f} "
            f"ΔGAP={r['vs_gap_delta_mean_r2']:+.3f} "
            f"[{r['vs_gap_ci_low']:+.3f}, {r['vs_gap_ci_high']:+.3f}]{error}"
        )
    print(f"Saved {output}\nNo representation training or checkpoint update.")
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--features", help="Reuse this diagnostic's cache; no rendering or encoding.")
    ap.add_argument("--checkpoint", default="vqvae_model.pt")
    ap.add_argument("--train-samples", type=int, default=256)
    ap.add_argument("--val-samples", type=int, default=128)
    ap.add_argument("--test-samples", type=int, default=128)
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument("--mask-batch-size", type=int)
    ap.add_argument("--grid", nargs=3, type=int)
    ap.add_argument("--regions", nargs=3, type=int, default=[2, 2, 2])
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--causal", choices=["iid", "match"], default="iid")
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out")
    return ap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run(parser().parse_args())
