"""Frozen GN/channel-LN recovery audit; see ENCODER_NORMALIZATION_AUDIT.md.

Uses actual forward tensors, independent probe fit/tune/test subjects, and
matched shuffled-label controls. Never trains or writes an encoder checkpoint.
"""

import argparse
import gc
import hashlib
import io
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from eval.diagnostics.pooling_probe import fit_readouts, paired_delta, r2
from eval.encoder.encoder_target_protocol import (
    TARGETS,
    VIEWS,
    dataset,
    dataset_metadata,
    digest,
    provenance,
    save_csv,
    save_report,
)
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.metrics.dci import content_factor_names
from eval.protocol.score_checkpoint import build_model, load_settings
from models.vqvae import ChannelLayerNorm3d
from utils.encoder_runtime import select_encoder_device


def select_norm(encoder, name):
    """Default to the main-path norm before the residual stack, not a residual arm."""
    if name == "pre_residual":
        name = f"layers.{len(encoder.layers) - 2}"
    module = encoder.get_submodule(name)
    if not isinstance(module, (torch.nn.GroupNorm, ChannelLayerNorm3d)):
        raise ValueError(f"{name} is {type(module).__name__}, not GroupNorm or ChannelLayerNorm3d")
    return name, module


def norm_statistics(module, x):
    """The actual per-image reduction domains; scale includes the norm's epsilon."""
    if x.ndim != 5:
        raise ValueError("Expected a [batch, channels, depth, height, width] norm input")
    x = x.float()
    if isinstance(module, torch.nn.GroupNorm):
        grouped = x.reshape(x.shape[0], module.num_groups, -1)
        variance, mean = torch.var_mean(grouped, dim=2, unbiased=False)
        return mean, (variance + module.eps).sqrt()
    if isinstance(module, ChannelLayerNorm3d):
        variance, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        return mean, (variance + module.norm.eps).sqrt()
    raise ValueError("Only GroupNorm and channel-only LayerNorm are supported")


@torch.inference_mode()
def capture(model, images, norm_layer="pre_residual", spatial_grid=0):
    """One real forward, preserving view order and outputs preceding in-place ReLU."""
    if any(module.training for module in model.modules()):
        raise ValueError("Frozen extraction requires every module in eval mode")
    if images.shape[0] % 2:
        raise ValueError("Need equal T1 and FLAIR batches in view-major order")
    encoders = [model.encoder] + ([model.encoder_v1] if model.encoder_v1 is not None else [])
    captured, descriptions = {}, []

    def backbone_hook(index):
        def hook(module, inputs, output):
            captured[index, "backbone"] = output.detach().clone()

        return hook

    def norm_hook(index):
        def hook(module, inputs, output):
            # Both inputs and outputs may subsequently be mutated by in-place activations.
            captured[index, "pre"] = inputs[0].detach().clone()
            captured[index, "post"] = output.detach().clone()
            mean, scale = norm_statistics(module, inputs[0])
            captured[index, "mean"], captured[index, "scale"] = mean, scale

        return hook

    with ExitStack() as stack:
        for index, encoder in enumerate(encoders):
            name, module = select_norm(encoder, norm_layer)
            descriptions.append({"layer": name, "type": type(module).__name__, "repr": repr(module)})
            stack.callback(encoder.register_forward_hook(backbone_hook(index)).remove)
            stack.callback(module.register_forward_hook(norm_hook(index)).remove)
        code = model(images, pool_only=True, n_views=2)[2][0]

    values = {
        key: torch.cat([captured[i, key] for i in range(len(encoders))])
        for key in ("backbone", "pre", "post", "mean", "scale")
    }
    if values["backbone"].shape[0] != images.shape[0]:
        raise ValueError("Unexpected encoder routing or hook invocation count")

    def flatten(x):
        if spatial_grid and x.ndim == 5:
            if any(spatial_grid > size for size in x.shape[2:]):
                raise ValueError(f"Spatial grid {spatial_grid} exceeds native shape {tuple(x.shape[2:])}")
            x = F.adaptive_avg_pool3d(x, (spatial_grid,) * 3)
        return x.flatten(1).cpu().numpy().copy()

    features = {}
    for source, stage in (("backbone", "backbone"), ("pre", "norm_pre"), ("post", "norm_post")):
        features[f"{stage}_spatial"] = flatten(values[source])
        features[f"{stage}_gap"] = flatten(values[source].mean((2, 3, 4)))
    features["norm_mean"] = flatten(values["mean"])
    features["norm_scale"] = flatten(values["scale"])
    features["global_content"] = flatten(code[:, : model.content_channels])
    features["global_all"] = flatten(code[:, : model.latent_dim])
    if code.shape[1] > model.latent_dim:
        features["lesion_branch"] = flatten(code[:, model.latent_dim :])
    # Attention models still get a GAP diagnostic, but its output is not called
    # the actual readout input. The GN/LN experiment should keep pooling matched.
    if model.attention_pool is not None:
        features["actual_pool"] = flatten(model.attention_pool(values["backbone"]))
    return features, {
        "normalizers": descriptions,
        "backbone_shape": list(values["backbone"].shape[1:]),
        "norm_shape": list(values["pre"].shape[1:]),
        "spatial_grid": spatial_grid or "native",
        "global_pool": "gap" if model.attention_pool is None else "attention",
        "global_content_channels": model.content_channels,
        "global_all_channels": model.latent_dim,
        "lesion_branch_units": code.shape[1] - model.latent_dim,
    }


def targets_for_sample(inner, latents):
    """Position and burden recipes, including the burden recipe's placement draws."""
    renderer = inner.renderer
    z = latents["z_content"].detach().cpu()
    _, support = renderer.render_structure(
        z,
        latents["z_deformation"],
        latents["z_fissure"],
        "cpu",
        clean=inner.clean_content,
        z_lesion=latents.get("z_lesion"),
    )
    if not torch.isfinite(support).all() or support.sum() <= 0:
        raise ValueError("Empty/non-finite lesion; refusing to silently select a different cohort")
    centroid = (support[..., None] * renderer.coords).sum((0, 1, 2)) / support.sum()
    squash = renderer.content_squash
    if squash == "auto":
        squash = "tanh" if inner.clean_content else "clamp"
    if squash not in ("tanh", "clamp", "none"):
        raise ValueError(f"Unsupported content squash: {squash}")
    value = z[8].tanh() if squash == "tanh" else z[8].clamp(-1, 1) if squash == "clamp" else z[8]
    amp_scale = 1.0 if renderer.content_amp_scale is None else renderer.content_amp_scale[8]
    amplitude = value * (0.06 * renderer.content_scale * amp_scale)
    return torch.cat((z, centroid, amplitude.reshape(1), amplitude.abs().reshape(1))).numpy()


def extract(model, ds, args, device, directory):
    directory.mkdir(parents=True)
    arrays, targets, ids = {}, [], []
    image_hash = hashlib.sha256()
    offset, before = 0, state_digest(model)
    for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0):
        images = torch.cat(batch["image"], dim=0)
        b = len(batch["index"])
        for i in range(b):
            image_hash.update(torch.stack((images[i], images[b + i])).numpy().tobytes())
        features, metadata = capture(model, images.to(device), args.norm_layer, args.spatial_grid)
        for stage, values in features.items():
            for view, block in zip(VIEWS, (values[:b], values[b:])):
                key = (view, stage)
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        directory / f"{view}_{stage}.npy", mode="w+", dtype="float32", shape=(len(ds), block.shape[1])
                    )
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite features in {key}")
                arrays[key][offset : offset + b] = block
        for i, idx in enumerate(batch["index"].tolist()):
            targets.append(targets_for_sample(ds._inner, {k: v[i] for k, v in batch["gt_latents"].items()}))
            ids.append(idx)
        offset += b
        if offset == len(ds) or offset % (args.batch_size * 20) == 0:
            print(f"  {directory.name}: encoded {offset}/{len(ds)}", flush=True)
    if state_digest(model) != before:
        raise RuntimeError("Frozen extraction changed checkpoint state")
    for array in arrays.values():
        array.flush()
    targets = np.asarray(targets)
    if not np.isfinite(targets).all():
        raise ValueError("Non-finite targets")
    return (
        arrays,
        targets,
        {
            **dataset_metadata(ds, image_hash.hexdigest(), ids),
            "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
            "model_state_sha256": before,
            **metadata,
        },
    )


def matched_cohorts(reference, candidate):
    for split in ("val", "test"):
        for field in ("input_sha256", "target_sha256", "ids", "generator_split_seed"):
            if candidate[split][field] != reference[split][field]:
                raise ValueError(f"Runs have different {split} {field}; refusing an unmatched comparison")


def bank_gram(val, test, train, chunk_size=2048):
    """Concatenate column chunks, never whole native-map banks, in RAM."""
    if val.shape[1] != test.shape[1]:
        raise ValueError("Validation/test feature widths differ")
    n, width = len(val) + len(test), 0
    gram = np.zeros((n, n), dtype=np.float64)
    for start in range(0, val.shape[1], chunk_size):
        block = np.concatenate((val[:, start : start + chunk_size], test[:, start : start + chunk_size])).astype(
            np.float64
        )
        scaler = StandardScaler().fit(block[train])
        keep = scaler.var_ > 1e-12
        z = scaler.transform(block)[:, keep]
        gram += np.dot(z, z.T)
        width += int(keep.sum())
    return gram, width


def score_banks(banks, args, arm, target_names, directory):
    val, test = banks["val"], banks["test"]
    n = len(val[1])
    order = np.random.default_rng(args.seed).permutation(n)
    cut = int(0.75 * n)
    splits = (order[:cut], order[cut:], np.arange(n, n + len(test[1])))
    targets = np.concatenate((val[1], test[1]))
    rows, predictions = [], {}
    for view in VIEWS:
        grams = {}
        for v, stage in val[0]:
            if v != view:
                continue
            feature = val[0][v, stage]
            grams[stage] = (*bank_gram(feature, test[0][v, stage], splits[0]), feature.shape[1])
        for stage, parts in (
            ("norm_statistics", ("norm_mean", "norm_scale")),
            ("norm_post_plus_statistics", ("norm_post_spatial", "norm_mean", "norm_scale")),
        ):
            grams[stage] = tuple(sum(grams[part][i] for part in parts) for i in range(3))
        for stage, (gram, width, dimensions) in grams.items():
            print(f"  {arm} {view} {stage}: {dimensions} dimensions ({width} variable)", flush=True)
            scores, predicted = fit_readouts(gram, width, targets, splits, args.seed, target_names)
            if any(not np.isfinite(value).all() for value in predicted.values()):
                raise FloatingPointError(f"Non-finite probe predictions in {arm}/{view}/{stage}")
            rows.extend(
                {
                    "arm": arm,
                    "view": view,
                    "stage": stage,
                    "dimensions": dimensions,
                    "variable_dimensions": width,
                    **row,
                }
                for row in scores
            )
            predictions.update({(view, stage, key): value for key, value in predicted.items()})
    np.savez_compressed(
        directory / f"{arm}_predictions.npz",
        truth=test[1],
        target_names=np.array(target_names),
        **{"__".join(key): value for key, value in predictions.items()},
    )
    split_info = {
        "fit_validation_ids": splits[0].tolist(),
        "tune_validation_ids": splits[1].tolist(),
        "test_ids": list(range(len(test[1]))),
    }
    return rows, predictions, test[1], split_info


def contrast_rows(predictions, truth, names, args):
    """Paired test-subject intervals; fixed probes, not training-seed uncertainty."""
    rows = []

    def compare(candidate_arm, reference_arm, candidate_stage, reference_stage):
        for view in VIEWS:
            for probe in ("ridge", "rbf"):
                key = (view, candidate_stage, f"{probe}_observed")
                ref_key = (view, reference_stage, f"{probe}_observed")
                if key not in predictions[candidate_arm] or ref_key not in predictions[reference_arm]:
                    continue
                candidate, reference = predictions[candidate_arm][key], predictions[reference_arm][ref_key]
                delta = r2(truth, candidate) - r2(truth, reference)
                interval = paired_delta(truth, candidate, reference, args.seed, args.bootstrap_draws)
                for j, target in enumerate(names):
                    rows.append(
                        {
                            "candidate_arm": candidate_arm,
                            "reference_arm": reference_arm,
                            "candidate_stage": candidate_stage,
                            "reference_stage": reference_stage,
                            "view": view,
                            "probe": probe,
                            "target": target,
                            "delta_r2": float(delta[j]),
                            "ci_low": float(interval[0, j]),
                            "ci_high": float(interval[1, j]),
                        }
                    )

    for arm in predictions:
        for candidate, reference in (
            ("backbone_gap", "backbone_spatial"),
            ("global_content", "backbone_gap"),
            ("global_all", "global_content"),
            ("norm_post_spatial", "norm_pre_spatial"),
            ("norm_post_gap", "norm_pre_gap"),
            ("norm_post_plus_statistics", "norm_post_spatial"),
            ("actual_pool", "backbone_gap"),
            ("global_content", "actual_pool"),
        ):
            compare(arm, arm, candidate, reference)
    reference_arm = next(iter(predictions))
    for arm in list(predictions)[1:]:
        for stage in sorted({key[1] for key in predictions[arm]}):
            compare(arm, reference_arm, stage, stage)
    return rows


def summary_rows(rows):
    grouped = {}
    fields = ("arm", "view", "stage", "probe", "condition")
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in fields), {})[row["target"]] = row["test_r2"]
    return [{**dict(zip(fields, key)), **values} for key, values in grouped.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Repeat for matched checkpoints, e.g. gn=results/gn ln=results/ln",
    )
    parser.add_argument("--checkpoint", default="model.pt", help="Same relative checkpoint filename in each run")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--norm-layer", default="pre_residual", help="Path relative to encoder, or pre_residual (default)"
    )
    parser.add_argument(
        "--spatial-grid",
        type=int,
        default=0,
        help="0: full native maps; positive: pool spatial banks to this cubic grid",
    )
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument(
        "--keep-features", action="store_true", help="Keep potentially large disk feature banks; default deletes them"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.test_samples < 10 or args.spatial_grid < 0 or args.bootstrap_draws < 2:
        parser.error("Need positive batch size, >=10 test subjects, nonnegative grid, and >=2 bootstrap draws")
    if Path(args.checkpoint).is_absolute() or ".." in Path(args.checkpoint).parts:
        parser.error("--checkpoint must be a relative filename inside each run")
    runs = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if (
            not separator
            or not name
            or not path
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
            or name in runs
        ):
            parser.error("Use unique NAME=DIR entries with letters, numbers, underscores or hyphens in NAME")
        cfg = load_settings(path)
        if cfg.get("encoder_architecture", "conv") != "conv":
            parser.error("This audit targets the conv GN/channel-LN comparison, not ResNet/BatchNorm")
        if cfg["num_val_samples"] < 20:
            parser.error("Need >=20 original validation subjects")
        runs[name] = (Path(path), cfg)
    names = None
    for _, cfg in runs.values():
        current = (*content_factor_names(9, cfg.get("synthetic_lesion_target", "position")), *TARGETS[9:])
        if names is not None and current != names:
            parser.error("Runs use different target meanings (e.g. position vs burden)")
        names = current
    hashes = {arm: digest(path / args.checkpoint) for arm, (path, _) in runs.items()}
    device = select_encoder_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    first_cfg = next(iter(runs.values()))[1]
    report = provenance(first_cfg, args, device)
    report.update(
        target_names=list(names),
        checkpoint_sha256=hashes,
        cohorts={},
        probes=[],
        runs={arm: {"run_dir": str(path.resolve()), "settings": cfg} for arm, (path, cfg) in runs.items()},
        protocol="Original validation: 75% probe fit, 25% tuning; separate test. Train-only feature/target scaling. Labels only train probes.",
        interpretation="Scores test recoverability with finite probes, not identifiability or causal attribution to normalization. Final-code probes jointly use all listed units.",
    )
    save_report(args.out_dir, report)
    all_predictions, reference_cohorts = {}, None
    try:
        for arm, (path, cfg) in runs.items():
            print(f"\n{arm}: frozen {cfg.get('norm_type', 'group')} checkpoint {path / args.checkpoint}", flush=True)
            blob = (path / args.checkpoint).read_bytes()
            if hashlib.sha256(blob).hexdigest() != hashes[arm]:
                raise ValueError("Checkpoint changed before loading")
            model = build_model(cfg, device, torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
            del blob
            with ExitStack() as stack:
                if args.keep_features:
                    feature_root = args.out_dir / "features" / arm
                else:
                    feature_root = Path(
                        stack.enter_context(tempfile.TemporaryDirectory(prefix=f".features-{arm}-", dir=args.out_dir))
                    )
                banks = {}
                try:
                    for split, count in (("val", cfg["num_val_samples"]), ("test", args.test_samples)):
                        banks[split] = extract(model, dataset(cfg, count, split), args, device, feature_root / split)
                    cohorts = {split: bank[2] for split, bank in banks.items()}
                    if reference_cohorts is not None:
                        matched_cohorts(reference_cohorts, cohorts)
                    else:
                        reference_cohorts = cohorts
                    report["cohorts"][arm] = cohorts
                    with threadpool_limits(limits=1):
                        rows, predicted, truth, report["probe_split"] = score_banks(
                            banks, args, arm, names, args.out_dir
                        )
                    report["probes"].extend(rows)
                    all_predictions[arm] = predicted
                finally:
                    for bank in banks.values():
                        for array in bank[0].values():
                            array._mmap.close()
                    banks.clear()
            del model
            gc.collect()
            save_report(args.out_dir, report)
        if {arm: digest(path / args.checkpoint) for arm, (path, _) in runs.items()} != hashes:
            raise ValueError("Source checkpoints changed during the audit")
        report["contrasts"] = contrast_rows(all_predictions, truth, names, args)
        report["summary"] = summary_rows(report["probes"])
        report.update(status="complete", source_checkpoints_unchanged=True)
        for table in ("probes", "summary", "contrasts"):
            save_csv(args.out_dir / f"{table}.csv", report[table])
        save_report(args.out_dir, report)
        print("\nHeld-out ridge R² (joint feature probes; not single-unit identifiability):")
        print("arm          view  stage                       brain   ventricle centroid_xyz sulcal_z signed_amp")
        for row in report["summary"]:
            if row["probe"] == "ridge" and row["condition"] == "observed":
                centroid = np.mean([row[f"centroid_{axis}"] for axis in "xyz"])
                print(
                    f"{row['arm']:12s} {row['view']:5s} {row['stage']:27s} {row['brain_size']:+.3f}   {row['ventricle_size']:+.3f}     {centroid:+.3f}      {row['sulcal_widening']:+.3f}   {row['sulcal_amplitude']:+.3f}"
                )
        print(f"Saved normalization audit: {args.out_dir}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise


if __name__ == "__main__":
    main()
