"""Frozen spatial/GAP/readout probes for both views, with initial and shuffled controls.

Fit probes on 75% of the original validation cohort, tune on the other 25%,
and score on a separate test cohort. No encoder weights or buffers are updated.
"""

import argparse
import gc
import hashlib
import io
import shutil
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from eval.diagnostics.pooling_probe import fit_readouts
from eval.encoder.encoder_target_protocol import (
    TARGETS,
    VIEWS,
    dataset,
    dataset_metadata,
    digest,
    provenance,
    sample_targets,
    save_csv,
    save_report,
)
from eval.lesion.checkpoint_lesion_analysis import batch_features, state_digest
from eval.lesion.lesion_probe import block_gram
from eval.protocol.score_checkpoint import build_model, load_settings
from utils.encoder_runtime import select_encoder_device


@torch.inference_mode()
def capture(model, images, grids, include_native):
    hidden, handles = [], []

    def first_hidden(module, args, out):
        if not hidden:
            hidden.append(out.detach())

    if model.readout_type == "mlp":
        handles.append(model.to_encoding[1].register_forward_hook(first_hidden))
    try:
        # The helper performs one actual global forward and evaluates diagnostic
        # patch readouts afterward. The first hidden output is the true global one.
        values, shape, channels = batch_features(model, images, grids)
        if include_native and shape[0] not in grids:
            if len(set(shape)) != 1:
                raise ValueError("Native grid requires a cubic map")
            extra, _, _ = batch_features(model, images, [shape[0]])
            values.update(extra)
        if hidden:
            values[(1, "hidden")] = hidden[0].cpu().numpy()
        return values, shape, channels
    finally:
        for handle in handles:
            handle.remove()


def extract(model, ds, args, device, directory):
    directory.mkdir(parents=True)
    before = state_digest(model)
    arrays, targets, ids = {}, [], []
    h = hashlib.sha256()
    offset = 0
    for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0):
        images = torch.cat(batch["image"], dim=0)
        # Canonical per-subject order makes the digest independent of batch size.
        b = len(batch["index"])
        for i in range(b):
            h.update(torch.stack((images[i], images[b + i])).numpy().tobytes())
        features, shape, channels = capture(model, images.to(device), args.grids, args.include_native)
        for (grid, stage), values in features.items():
            if getattr(args, "stages", None) is not None and stage not in args.stages:
                continue
            for view, block in zip(VIEWS, (values[:b], values[b:])):
                key = (view, grid, stage)
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        directory / f"{view}_g{grid}_{stage}.npy",
                        mode="w+",
                        dtype="float32",
                        shape=(len(ds), block.shape[1]),
                    )
                if not np.isfinite(block).all():
                    raise ValueError("Non-finite encoder features")
                arrays[key][offset : offset + b] = block
        for i, idx in enumerate(batch["index"].tolist()):
            lat = {key: value[i] for key, value in batch["gt_latents"].items()}
            target, _ = sample_targets(ds._inner, lat)
            targets.append(target)
            ids.append(idx)
        offset += b
        if offset == len(ds) or offset % (args.batch_size * 20) == 0:
            print(f"  {directory.name}: encoded {offset}/{len(ds)}", flush=True)
    if state_digest(model) != before:
        raise RuntimeError("Frozen extraction changed model state")
    for array in arrays.values():
        array.flush()
    return (
        arrays,
        np.asarray(targets),
        {
            **dataset_metadata(ds, h.hexdigest(), ids),
            "native_shape": list(shape),
            "channels": channels,
            "model_state_sha256": before,
        },
    )


def score_banks(banks, args, arm, directory):
    val, test = banks["val"], banks["test"]
    n = len(val[1])
    order = np.random.default_rng(args.seed).permutation(n)
    cut = int(n * 0.75)
    splits = (order[:cut], order[cut:], np.arange(n, n + len(test[1])))
    targets = np.concatenate((val[1], test[1]))
    rows, predictions = [], {"truth": test[1]}
    for (view, grid, stage), feature in val[0].items():
        print(f"  {arm} {view} grid={grid} {stage}: {feature.shape[1]} dimensions", flush=True)
        x = np.concatenate((feature, test[0][view, grid, stage]))
        gram, width = block_gram(x, np.arange(x.shape[1]), splits[0])
        scores, predicted = fit_readouts(gram, width, targets, splits, args.seed, target_names=TARGETS)
        rows.extend(
            {"arm": arm, "view": view, "grid": grid, "stage": stage, "dimensions": x.shape[1], **row} for row in scores
        )
        predictions.update({f"{view}_g{grid}_{stage}_{key}": value for key, value in predicted.items()})
        del x, gram
    np.savez_compressed(directory / f"{arm}_predictions.npz", **predictions)
    split_info = {
        "fit_validation_ids": order[:cut].tolist(),
        "tune_validation_ids": order[cut:].tolist(),
        "test_ids": list(range(len(test[1]))),
    }
    return rows, split_info


def focused_summary(rows):
    """Keep useful final numbers in SLURM stdout as well as the full CSV."""
    groups = {}
    fields = ("arm", "view", "grid", "stage", "probe", "condition")
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, {})[row["target"]] = row["test_r2"]
    summary = []
    for key, values in groups.items():
        summary.append(
            {
                **dict(zip(fields, key)),
                "lesion_latent_mean_r2": float(np.mean([values[f"lesion_{axis}"] for axis in "xyz"])),
                "centroid_mean_r2": float(np.mean([values[f"centroid_{axis}"] for axis in "xyz"])),
                "sulcal_latent_r2": values["sulcal_widening"],
                "sulcal_amplitude_r2": values["sulcal_amplitude"],
                "sulcal_magnitude_r2": values["sulcal_magnitude"],
            }
        )
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--probe-samples", type=int, help="Default: original validation cohort size")
    p.add_argument("--test-samples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grids", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--include-native", action="store_true", help="Also flatten the full native map; may use several GB of disk/RAM"
    )
    p.add_argument(
        "--skip-initial", action="store_true", help="Omit model_init.pt; original checkpoint remains mandatory"
    )
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument(
        "--discard-features",
        action="store_true",
        help="Remove large feature banks after scoring; retain reports/predictions",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = p.parse_args(argv)
    cfg = load_settings(args.run_dir)
    args.probe_samples = args.probe_samples or cfg["num_val_samples"]
    if args.probe_samples != cfg["num_val_samples"]:
        p.error("Keep --probe-samples equal to the original validation size to preserve normalization")
    if args.probe_samples < 20 or args.test_samples < 10 or args.batch_size < 1 or min(args.grids) < 1:
        p.error("Need >=20 validation subjects, >=10 test subjects, and positive batch/grid sizes")
    args.grids = sorted(set([1, *args.grids]))
    checkpoints = {"trained": args.run_dir / args.checkpoint}
    if not args.skip_initial:
        checkpoints["initial"] = args.run_dir / "model_init.pt"
    hashes = {arm: digest(path) for arm, path in checkpoints.items()}
    device = select_encoder_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, device)
    report.update(
        checkpoint_sha256=hashes,
        probes=[],
        cohorts={},
        protocol="300/100 validation fit/tune at default n=400; independent test; standardization fitted only on probe-fit subjects",
    )
    save_report(args.out_dir, report)
    try:
        for arm, path in checkpoints.items():
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != hashes[arm]:
                raise ValueError("Checkpoint changed before loading")
            state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            model = build_model(cfg, device, state)
            del state, blob
            banks = {}
            for split, count in (("val", args.probe_samples), ("test", args.test_samples)):
                ds = dataset(cfg, count, split)
                banks[split] = extract(model, ds, args, device, args.out_dir / "features" / arm / split)
            if report["cohorts"]:
                for split in banks:
                    if banks[split][2]["input_sha256"] != report["cohorts"]["trained"][split]["input_sha256"]:
                        raise ValueError("Initial/trained extraction used different images")
            report["cohorts"][arm] = {split: bank[2] for split, bank in banks.items()}
            with threadpool_limits(limits=1):
                rows, report["probe_split"] = score_banks(banks, args, arm, args.out_dir)
            report["probes"].extend(rows)
            save_report(args.out_dir, report)
            del banks, model
            gc.collect()
        if {arm: digest(path) for arm, path in checkpoints.items()} != hashes:
            raise ValueError("Source checkpoint files changed during audit")
        report.update(status="complete", source_checkpoints_unchanged=True)
        report["summary"] = focused_summary(report["probes"])
        save_csv(args.out_dir / "probes.csv", report["probes"])
        save_csv(args.out_dir / "summary.csv", report["summary"])
        save_report(args.out_dir, report)
        print("\nHeld-out target recovery (observed labels; shuffled controls are in summary.csv):")
        for row in report["summary"]:
            if row["condition"] == "observed":
                print(
                    f"{row['arm']:7s} {row['view']:5s} g{row['grid']:<2} {row['stage']:9s} {row['probe']:5s} "
                    f"latent_xyz={row['lesion_latent_mean_r2']:+.3f} centroid_xyz={row['centroid_mean_r2']:+.3f} "
                    f"sulcal_z={row['sulcal_latent_r2']:+.3f} signed_amp={row['sulcal_amplitude_r2']:+.3f} "
                    f"magnitude={row['sulcal_magnitude_r2']:+.3f}"
                )
        print(f"Completed frozen spatial/pooled probes: {args.out_dir}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise
    finally:
        if args.discard_features:
            shutil.rmtree(args.out_dir / "features", ignore_errors=True)


if __name__ == "__main__":
    main()
