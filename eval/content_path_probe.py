"""Frozen content/style stage probes; see CONTENT_PATH_PROBE.md.

One actual reconstruction forward per batch, independent readouts at each stage,
and the same held-out subjects everywhere. No training or checkpoint writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import tempfile
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from eval.lesion_probe import view_content_mask
from eval.lesion_reconstruction import json_safe, make_dataset
from eval.pooling_probe import TARGETS, VIEWS, block_gram, fit_readouts, paired_delta, r2, split_subjects
from eval.style_path_audit import capture_path, validate_model

LOG = logging.getLogger(__name__)


def state_digest(model):
    """Include every registered parameter and buffer, including codebook EMA state."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def stage_maps(model, x, mask):
    """Capture real tensors from a single-level reconstruction path, in view order.

    pre_norm matches pooling_probe; pre_quant is AFTER CodeLayer.conv_in.
    Decoder tensors are captured directly, never reconstructed from code IDs.
    """
    validate_model(model)
    if model.mask_mode not in ("fixed", "learned"):
        raise ValueError("Stage audit supports stable hard fixed/learned masks, not learned_split/onthefly")
    collected = {}

    def tap(name, tuple_output=False):
        def hook(module, args, output):
            value = output[0] if tuple_output else output
            collected.setdefault(name, []).append(value.detach().clone())

        return hook

    with ExitStack() as stack:

        def attach(module, name, tuple_output=False):
            handle = module.register_forward_hook(tap(name, tuple_output))
            stack.callback(handle.remove)

        attach(model.encoders[0], "encoder")
        if model.separate_encoders:
            attach(model.encoders_v1[0], "encoder")
        if "0" in model.content_norms:
            attach(model.content_norms["0"], "post_norm")
        codebooks = [model.codebooks[0]]
        if model.separate_content_codebooks:
            codebooks.append(model.codebooks_v1[0])
        for cb in codebooks:
            attach(cb.conv_in, "content_projection")
            attach(cb, "content_quantized", True)
        if model.quantize_style:
            codebooks = [model.style_codebooks["0"]]
            if model.separate_style_codebooks:
                codebooks.append(model.style_codebooks_v1["0"])
            for cb in codebooks:
                attach(cb.conv_in, "style_projection")
                attach(cb, "style_quantized", True)
        path = stack.enter_context(capture_path(model))
        with torch.inference_mode():
            output = model(
                x, n_views=2, subsets=[(0, 1)], pool_only=True, return_recon=True, mask=mask, patch_grid=None
            )

    def joined(name, expected):
        values = collected.get(name, [])
        if len(values) != expected:
            raise ValueError(f"Expected {expected} captures of {name}, got {len(values)}")
        value = torch.cat(values, dim=0)
        if value.ndim != 5 or value.shape[0] != x.shape[0] or not torch.isfinite(value).all():
            raise ValueError(f"Invalid spatial tensor at {name}")
        return value

    pre = joined("encoder", 2 if model.separate_encoders else 1)
    if model.latent_mask:
        valid = F.adaptive_avg_pool3d(mask.float(), pre.shape[2:]) > model.latent_mask_thresh
        pre = pre * valid.to(pre.dtype)
    if not torch.allclose(pre.mean((2, 3, 4)), output[2][0], atol=2e-5, rtol=2e-5):
        raise ValueError("Pre-normalization tap does not reproduce forward GAP")
    post = joined("post_norm", 2 if model.separate_encoders else 1) if "0" in model.content_norms else pre
    partitions = [view_content_mask(output[6], 0, v, pre.shape[1]) for v in range(2)]
    stages = {}
    b = x.shape[0] // 2
    for block in ("content", "style"):
        for name, value in (("pre_norm", pre), ("post_norm", post)):
            pieces = []
            for v in range(2):
                chosen = partitions[v] if block == "content" else ~partitions[v]
                if not chosen.any():
                    raise ValueError(f"Empty {block} block in view {v}")
                pieces.append(value[v * b : (v + 1) * b, chosen])
            stages[(block, name)] = torch.cat(pieces)
    if not torch.equal(stages[("style", "post_norm")], path["raw"]):
        raise ValueError("Post-normalization style does not match the actual bottleneck input")
    stages[("content", "pre_quant")] = joined("content_projection", 2 if model.separate_content_codebooks else 1)
    quantized = joined("content_quantized", 2 if model.separate_content_codebooks else 1)
    if not torch.equal(quantized, path["content"]):
        raise ValueError("Quantized content differs from actual decoder input")
    stages[("content", "decoder_input")] = path["content"]
    stages[("style", "bottleneck")] = path["pooled"]
    if model.quantize_style:
        stages[("style", "pre_quant")] = joined("style_projection", 2 if model.separate_style_codebooks else 1)
        quantized_style = joined("style_quantized", 2 if model.separate_style_codebooks else 1)
        if not torch.equal(quantized_style, path["injected"]):
            raise ValueError("Quantized style differs from actual decoder style argument")
    stages[("style", "decoder_input")] = path["injected"]
    if any(value.ndim != 5 or not torch.isfinite(value).all() for value in stages.values()):
        raise ValueError("Non-finite or non-spatial stage tensor")
    return stages, partitions


def mean_descriptor(features, grid):
    """Never invent spatial sites when style has already been bottlenecked."""
    effective = min(grid, *features.shape[2:])
    if grid < 1 or any(d % effective for d in features.shape[2:]):
        raise ValueError(f"Grid {grid} must divide spatial shape {tuple(features.shape[2:])}")
    return F.adaptive_avg_pool3d(features, (effective,) * 3).flatten(1), effective


def extract_stages(model, ds, device, directory, batch_size, grids):
    arrays, metadata, targets, previous_masks = {}, {}, [], None
    for start in range(0, len(ds), batch_size):
        images, masks = [], []
        for idx in range(start, min(start + batch_size, len(ds))):
            a, b, lat = ds._inner[idx]
            mask = lat["brain_mask"]
            images.append(ds.normalize_views(a, b, mask, mask))
            masks.append(mask)
            target = lat["z_content"].cpu().numpy()[: len(TARGETS)]
            if len(target) != len(TARGETS):
                raise ValueError("Need six content factors")
            targets.append(target)
        count = len(images)
        x = torch.cat([torch.stack([sample[v] for sample in images]) for v in range(2)]).to(device)
        mask = torch.cat([torch.stack(masks)] * 2).to(device)
        stages, partitions = stage_maps(model, x, mask)
        if previous_masks is not None and not np.array_equal(partitions, previous_masks):
            raise ValueError("Content selection changed between batches")
        previous_masks = partitions
        for (block, stage), maps in stages.items():
            for v, view in enumerate(VIEWS):
                features = maps[v * count : (v + 1) * count]
                for grid in grids:
                    descriptor, effective = mean_descriptor(features, grid)
                    key = (view, block, stage, grid)
                    name = f"{view}_{block}_{stage}_g{grid}"
                    info = {
                        "native_shape": list(features.shape[1:]),
                        "effective_grid": effective,
                        "requested_grid": grid,
                        "feature_count": descriptor.shape[1],
                    }
                    if name in metadata and info != metadata[name]:
                        raise ValueError(f"Stage dimensions changed at {name}")
                    metadata[name] = info
                    if key not in arrays:
                        arrays[key] = np.lib.format.open_memmap(
                            Path(directory) / f"{name}.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(len(ds), descriptor.shape[1]),
                        )
                    arrays[key][start : start + count] = descriptor.cpu().numpy()
        LOG.info("Captured all content/style stages: %d/%d subjects", start + count, len(ds))
    for array in arrays.values():
        array.flush()
    return arrays, np.asarray(targets), metadata, [np.flatnonzero(m).tolist() for m in previous_masks]


def evaluate_stages(arrays, targets, splits, grids, seed):
    rows, predictions = [], {"test_truth": targets[splits[2]], "test_indices": splits[2]}
    for view in VIEWS:
        for block in ("content", "style"):
            stages = [
                s
                for s in ("pre_norm", "post_norm", "bottleneck", "pre_quant", "decoder_input")
                if (view, block, s, grids[0]) in arrays
            ]
            for grid in grids:
                previous, previous_name = None, None
                for stage in stages:
                    x = arrays[(view, block, stage, grid)]
                    LOG.info("Probing %s %s %s grid=%d (%d columns)", view, block, stage, grid, x.shape[1])
                    gram, width = block_gram(x, np.arange(x.shape[1]), splits[0])
                    scores, fitted = fit_readouts(gram, width, targets, splits, seed)
                    if previous is None:
                        previous, previous_name = fitted, stage
                    intervals = {
                        kind: paired_delta(
                            targets[splits[2]], fitted[f"{kind}_observed"], previous[f"{kind}_observed"], seed
                        )
                        for kind in ("ridge", "rbf")
                    }
                    for row in scores:
                        if row["condition"] == "observed":
                            j = TARGETS.index(row["target"])
                            base = r2(targets[splits[2]], previous[f"{row['probe']}_observed"])[j]
                            row.update(
                                reference_stage=previous_name,
                                delta_vs_previous=row["test_r2"] - float(base),
                                delta_ci_low=float(intervals[row["probe"]][0, j]),
                                delta_ci_high=float(intervals[row["probe"]][1, j]),
                            )
                        rows.append(
                            {
                                "view": view,
                                "block": block,
                                "stage": stage,
                                "grid": grid,
                                "feature_count": x.shape[1],
                                "variable_feature_count": width,
                                **row,
                            }
                        )
                    for name, values in fitted.items():
                        predictions[f"{view}_{block}_{stage}_g{grid}_{name}"] = values
                    previous, previous_name = fitted, stage
    return rows, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grids", type=int, nargs="+", default=[8])
    parser.add_argument("--causal", choices=("iid", "match"), default="iid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args(argv)
    if cli.num_samples < 30 or min(cli.batch_size, cli.threads, *cli.grids) < 1:
        parser.error("Need >=30 samples and positive batch size, threads, and grids")
    cli.grids = list(dict.fromkeys(cli.grids))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.seed)
    from eval.run_dci_synthetic import load_model_from_run_dir

    model, args, device = load_model_from_run_dir(
        cli.run_dir, cli.checkpoint, torch.device(cli.device) if cli.device else None
    )
    checkpoint = Path(cli.checkpoint or "vqvae_model.pt")
    if checkpoint.parent == Path("."):
        checkpoint = Path(cli.run_dir) / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.get("encoders", state).items()}, strict=True)
    checkpoint_step = state.get("step")
    del state
    model.eval().requires_grad_(False)
    validate_model(model)
    before = state_digest(model)
    ds = make_dataset(args, cli.num_samples, cli.causal, "test")
    splits = split_subjects(cli.num_samples, cli.seed)
    output = Path(cli.output_dir or Path(cli.run_dir) / f"content_path_probe_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="content_path_probe_") as tmp, threadpool_limits(limits=cli.threads):
        arrays, targets, metadata, channels = extract_stages(model, ds, device, tmp, cli.batch_size, cli.grids)
        after = state_digest(model)
        if before != after:
            raise ValueError("Parameters or buffers changed during extraction; refusing to report frozen probes")
        rows, predictions = evaluate_stages(arrays, targets, splits, cli.grids, cli.seed)
        del arrays
    for row in rows:
        name = f"{row['view']}_{row['block']}_{row['stage']}_g{row['grid']}"
        row["effective_grid"] = metadata[name]["effective_grid"]
    np.savez_compressed(
        output / "predictions.npz",
        **predictions,
        train_indices=splits[0],
        validation_indices=splits[1],
        all_targets=targets,
    )
    report = {
        "config": vars(cli),
        "dataset_settings": vars(args),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "model_state_unchanged": before == after,
        "state_sha256": before,
        "target_names": TARGETS,
        "content_channels_by_view": channels,
        "stage_metadata": metadata,
        "scores": rows,
        "notes": [
            "Stages use independent fitted probes with shared subjects and train-only scaling.",
            "pre_quant follows the codebook projection; decoder_input is the actual quantized tensor.",
            "Unquantized style has no pre_quant stage; decoder_input is continuous style in that case.",
            "Smaller style maps use their actual available resolution; no upsampling for probes.",
            "No new foreground pooling mask. Lesion targets are latent coordinates, not size/presence.",
            "R2 changes measure probe accessibility, not information conservation or decoder use.",
            "Bootstrap intervals condition on fitted probes and do not adjust for multiple comparisons.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    print("\nHeld-out R²: ridge / RBF (validation-selected); mean pooling only")
    print("view   block    grid actual  stage             ventricle         lesion xyz mean")
    keys = list(dict.fromkeys((r["view"], r["block"], r["grid"], r["effective_grid"], r["stage"]) for r in rows))
    for view, block, grid, effective, stage in keys:
        numbers = []
        for kind in ("ridge", "rbf"):
            chosen = {
                r["target"]: r["test_r2"]
                for r in rows
                if (r["view"], r["block"], r["grid"], r["stage"], r["probe"], r["condition"])
                == (view, block, grid, stage, kind, "observed")
            }
            numbers.append((chosen["ventricle_size"], np.mean([chosen[f"lesion_{a}"] for a in "xyz"])))
        print(
            f"{view:6s} {block:8s} {grid:4d} {effective:6d}  {stage:17s} "
            f"{numbers[0][0]:+.3f} / {numbers[1][0]:+.3f}   {numbers[0][1]:+.3f} / {numbers[1][1]:+.3f}"
        )
    print(
        f"\nSaved {output}\nNo parameter or registered buffer changed; scores.csv includes shuffled controls and stage deltas."
    )


if __name__ == "__main__":
    main()
