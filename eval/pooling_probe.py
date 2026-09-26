"""Frozen native-map pooling comparison; see POOLING_PROBE.md.

No representation training or checkpoint writes. Ground-truth factors are used
only as probe targets, never to select channels, regions, or tail activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eval.lesion_probe import block_gram, split_subjects, view_content_mask
from eval.lesion_reconstruction import json_safe, make_dataset

LOG = logging.getLogger(__name__)
VIEWS = ("t1", "flair")
TARGETS = ("brain_size", "ventricle_size", "lesion_x", "lesion_y", "lesion_z", "cortical_thickness")
STATS = ("mean", "max", "upper", "lower")
METHODS = (*STATS, "mean_tails")


def regional_pool(x, grid, fraction, return_selection=False):
    """Non-overlapping cubic bins; average the ceil(fraction * sites) extremes.

    Reject non-divisible grids instead of silently introducing overlapping bins.
    Output [B,C,grid**3] retains region identity. Selection maps are label-free
    channel averages at native resolution, for inspecting where tails came from.
    """
    if x.ndim != 5 or grid < 1 or not 0 < fraction <= 1:
        raise ValueError("Need [B,C,D,H,W], positive grid, and 0 < tail fraction <= 1")
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite feature map")
    if any(d % grid or d < grid for d in x.shape[2:]):
        raise ValueError(f"Grid {grid} must divide each native spatial dimension {tuple(x.shape[2:])}")
    b, c, d, h, w = x.shape
    zd, zh, zw = d // grid, h // grid, w // grid
    bins = x.reshape(b, c, grid, zd, grid, zh, grid, zw)
    bins = bins.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(b, c, grid**3, -1)
    n = bins.shape[-1]
    k = max(1, math.ceil(fraction * n))
    upper = bins.topk(k, dim=-1)
    lower = bins.topk(k, dim=-1, largest=False)
    pooled = {
        "mean": bins.mean(-1),
        "max": bins.amax(-1),
        "upper": upper.values.mean(-1),
        "lower": lower.values.mean(-1),
    }
    selection = {}
    if return_selection:
        for name, indices in (("upper", upper.indices), ("lower", lower.indices)):
            selected = torch.zeros_like(bins).scatter_(-1, indices, 1)
            selected = selected.reshape(b, c, grid, grid, grid, zd, zh, zw)
            selection[name] = selected.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(b, c, d, h, w).mean(1)
    return pooled, {"sites_per_region": n, "k": k, "effective_fraction": k / n}, selection


def native_maps(model, x, mask, level):
    """Tap exactly the spatial maps used by forward's pre-content-norm pooling.

    Compare their GAP with the actual forward return to catch stage/view errors.
    pool_only avoids quantization and decoder work. Hooks are always removed.
    """
    if any(m.training for m in model.modules()):
        raise ValueError("All model modules must be in eval mode")
    if not 0 <= level < model.nb_levels:
        raise ValueError("Requested encoder level does not exist")
    if getattr(model, "mask_mode", "onthefly") not in ("fixed", "learned", "learned_split"):
        raise ValueError("Need stable content channels, not batch-dependent onthefly masks")
    captured = []

    def tap(module, args, result):
        captured.append(result.detach().clone())

    stacks = [model.encoders]
    if model.separate_encoders:
        stacks.append(model.encoders_v1)
    handles = [stack[level].register_forward_hook(tap) for stack in stacks]
    try:
        with torch.inference_mode():
            out = model(x, return_recon=False, pool_only=True, n_views=2, subsets=[(0, 1)], mask=mask, patch_grid=None)
        if len(captured) != len(stacks):
            raise ValueError("Unexpected encoder hook invocation count")
        features = torch.cat(captured, dim=0)
        if model.latent_mask:
            valid = F.adaptive_avg_pool3d(mask.float(), features.shape[2:]) > model.latent_mask_thresh
            features = features * valid.to(features.dtype)
        if not torch.allclose(features.mean((2, 3, 4)), out[2][level], atol=2e-5, rtol=2e-5):
            raise ValueError("Tapped map does not reproduce forward GAP; stage comparison would be invalid")
        return features, out[6]
    finally:
        for handle in handles:
            handle.remove()


def extract(model, ds, device, directory, batch_size, level, grids, fraction, block, examples):
    arrays, partitions, metadata, saved = {}, None, {}, {}
    targets = []
    for start in range(0, len(ds), batch_size):
        images, masks = [], []
        for idx in range(start, min(start + batch_size, len(ds))):
            a, b, lat = ds._inner[idx]
            mask = lat["brain_mask"]
            images.append(ds.normalize_views(a, b, mask, mask))
            masks.append(mask)
            z = lat["z_content"].cpu().numpy()
            if len(z) < len(TARGETS):
                raise ValueError("Need six content factors for ventricle/lesion and control probes")
            targets.append(z[: len(TARGETS)])
        count = len(images)
        x = torch.cat([torch.stack([sample[v] for sample in images]) for v in range(2)]).to(device)
        mask = torch.cat([torch.stack(masks)] * 2).to(device)
        features, forward_masks = native_maps(model, x, mask, level)
        current = [view_content_mask(forward_masks, level, v, features.shape[1]) for v in range(2)]
        if partitions is not None and not np.array_equal(current, partitions):
            raise ValueError("Content channel selection changed across batches")
        partitions = current
        for v, view in enumerate(VIEWS):
            chosen = current[v] if block == "content" else ~current[v]
            if not chosen.any():
                raise ValueError(f"Empty {block} block for {view}")
            fmap = features[v * count : (v + 1) * count, chosen]
            n_save = max(0, min(count, examples - start))
            for grid in grids:
                pooled, info, _ = regional_pool(fmap, grid, fraction)
                metadata[f"{view}_g{grid}"] = {
                    **info,
                    "native_shape": list(fmap.shape[1:]),
                    "encoder_channels": np.flatnonzero(chosen).tolist(),
                }
                for stat, values in pooled.items():
                    key = (view, grid, stat)
                    flat = values.flatten(1).cpu().numpy()
                    if key not in arrays:
                        arrays[key] = np.lib.format.open_memmap(
                            Path(directory) / f"{view}_g{grid}_{stat}.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(len(ds), flat.shape[1]),
                        )
                    arrays[key][start : start + count] = flat
                if n_save:
                    _, _, selected = regional_pool(fmap[:n_save], grid, fraction, True)
                    for name, values in selected.items():
                        saved.setdefault(f"{view}_g{grid}_{name}_selection", []).append(values.cpu().numpy())
            if n_save:
                saved.setdefault(f"{view}_image", []).append(x[v * count : v * count + n_save].cpu().numpy())
                saved.setdefault(f"{view}_native_features", []).append(fmap[:n_save].cpu().numpy())
        LOG.info("Encoded %d/%d subjects (all pooling variants)", start + count, len(ds))
    for array in arrays.values():
        array.flush()
    return arrays, np.asarray(targets), metadata, {k: np.concatenate(v) for k, v in saved.items()}


def r2(truth, prediction):
    ss = ((truth - truth.mean(0)) ** 2).sum(0)
    return 1 - np.divide(
        ((truth - prediction) ** 2).sum(0), ss, out=np.full_like(ss, np.nan, dtype=float), where=ss > 1e-12
    )


def fit_readouts(gram, width, y, splits, seed, target_names=TARGETS):
    """Per-target ridge/RBF selection on validation, with train-only scaling.

    Shuffled controls permute labels within train and within validation, separately;
    test labels never enter fitting or selection, even for the null control.
    """
    if y.ndim != 2 or y.shape[1] != len(target_names):
        raise ValueError("Need one target name for each target column")
    train, val, test = splits
    if width == 0:
        prediction = np.broadcast_to(y[train].mean(0), (len(test), y.shape[1])).copy()
        return [
            {
                "probe": kind,
                "condition": cond,
                "target": name,
                "alpha": None,
                "gamma": None,
                "validation_mse_standardized": None,
                "test_r2": float(r2(y[test], prediction)[j]),
            }
            for kind in ("ridge", "rbf")
            for cond in ("observed", "shuffled")
            for j, name in enumerate(target_names)
        ], {f"{kind}_{cond}": prediction.copy() for kind in ("ridge", "rbf") for cond in ("observed", "shuffled")}
    if not np.isfinite(gram).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite kernel or target")
    linear = gram / width
    distances = np.maximum(np.diag(linear)[:, None] + np.diag(linear)[None, :] - 2 * linear, 0)
    scaler = StandardScaler().fit(y[train])
    actual = scaler.transform(y)
    null = actual.copy()
    rng = np.random.default_rng(seed + 17)
    null[train] = actual[rng.permutation(train)]
    null[val] = actual[rng.permutation(val)]
    selected = {}
    for kind, gamma in (("ridge", None), ("rbf", 0.1), ("rbf", 1.0), ("rbf", 10.0)):
        kernel = linear if gamma is None else np.exp(-gamma * distances)
        values, vectors = eigh(kernel[np.ix_(train, train)])
        values = np.maximum(values, 0)
        cross_val = kernel[np.ix_(val, train)] @ vectors
        cross_test = kernel[np.ix_(test, train)] @ vectors
        for condition, z in (("observed", actual), ("shuffled", null)):
            projection = vectors.T @ z[train]
            for alpha in np.logspace(-6, 2, 9):
                weights = projection / (values[:, None] + alpha)
                loss = np.mean((cross_val @ weights - z[val]) ** 2, axis=0)
                for j, name in enumerate(target_names):
                    key = (kind, condition, j)
                    if key not in selected or loss[j] < selected[key]["validation_mse_standardized"]:
                        selected[key] = {
                            "probe": kind,
                            "condition": condition,
                            "target": name,
                            "alpha": float(alpha),
                            "gamma": gamma,
                            "validation_mse_standardized": float(loss[j]),
                            "prediction": (cross_test @ weights[:, j]) * scaler.scale_[j] + scaler.mean_[j],
                        }
    rows, predictions = [], {}
    for (kind, condition, j), info in selected.items():
        prediction = info.pop("prediction")
        predictions.setdefault(f"{kind}_{condition}", np.empty((len(test), len(target_names))))[:, j] = prediction
        rows.append({**info, "test_r2": float(r2(y[test, j : j + 1], prediction[:, None])[0])})
    return rows, predictions


def paired_delta(truth, candidate, reference, seed, draws=500):
    """Conditional paired bootstrap, fixed fitted probes; not training-seed uncertainty."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        idx = rng.integers(len(truth), size=len(truth))
        deltas.append(r2(truth[idx], candidate[idx]) - r2(truth[idx], reference[idx]))
    return np.nanquantile(deltas, (0.025, 0.975), axis=0)


def evaluate(arrays, targets, splits, grids, seed):
    rows, saved = [], {"test_truth": targets[splits[2]], "test_indices": splits[2]}
    for view in VIEWS:
        for grid in grids:
            grams = {}
            for stat in STATS:
                x = arrays[(view, grid, stat)]
                grams[stat] = block_gram(x, np.arange(x.shape[1]), splits[0])
            grams["mean_tails"] = (
                sum(grams[s][0] for s in ("mean", "upper", "lower")),
                sum(grams[s][1] for s in ("mean", "upper", "lower")),
            )
            baseline = None
            for method in METHODS:
                gram, width = grams[method]
                LOG.info("Probing %s grid=%d %s (%d variable features)", view, grid, method, width)
                result, predictions = fit_readouts(gram, width, targets, splits, seed)
                if method == "mean":
                    baseline = predictions
                intervals = {
                    kind: paired_delta(
                        targets[splits[2]], predictions[f"{kind}_observed"], baseline[f"{kind}_observed"], seed
                    )
                    for kind in ("ridge", "rbf")
                }
                total_width = arrays[(view, grid, "mean")].shape[1] * (3 if method == "mean_tails" else 1)
                for row in result:
                    j = TARGETS.index(row["target"])
                    if row["condition"] == "observed":
                        ref = r2(targets[splits[2]], baseline[f"{row['probe']}_observed"])[j]
                        row.update(
                            delta_vs_mean=row["test_r2"] - float(ref),
                            delta_ci_low=float(intervals[row["probe"]][0, j]),
                            delta_ci_high=float(intervals[row["probe"]][1, j]),
                        )
                    rows.append(
                        {
                            "view": view,
                            "grid": grid,
                            "method": method,
                            "feature_count": total_width,
                            "variable_feature_count": width,
                            **row,
                        }
                    )
                for name, prediction in predictions.items():
                    saved[f"{view}_g{grid}_{method}_{name}"] = prediction
    return rows, saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grids", type=int, nargs="+", default=[1, 8], help="Cubic region grids; 1 is global")
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--block", choices=("content", "style"), default="content")
    parser.add_argument("--causal", choices=("iid", "match"), default="iid")
    parser.add_argument("--seed", type=int, default=0, help="Subject split, shuffled null, and bootstrap seed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4, help="Save native feature and tail-selection examples")
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args(argv)
    if cli.num_samples < 30 or min(cli.batch_size, cli.threads, *cli.grids) < 1 or cli.examples < 0:
        parser.error("Need >=30 samples, positive batch/threads/grids and nonnegative examples")
    if not 0 < cli.tail_fraction <= 1:
        parser.error("--tail-fraction must be in (0,1]")
    cli.grids = list(dict.fromkeys(cli.grids))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.seed)
    from eval.run_dci_synthetic import load_model_from_run_dir

    model, args, device = load_model_from_run_dir(
        cli.run_dir, cli.checkpoint, torch.device(cli.device) if cli.device else None
    )
    # Existing shared loader only warns on missing parameters. Refuse such a run.
    checkpoint = Path(cli.checkpoint or "vqvae_model.pt")
    if checkpoint.parent == Path("."):
        checkpoint = Path(cli.run_dir) / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.get("encoders", state).items()}, strict=True)
    del state
    model.eval().requires_grad_(False)
    ds = make_dataset(args, cli.num_samples, cli.causal, "test")
    splits = split_subjects(cli.num_samples, cli.seed)
    output = Path(cli.output_dir or Path(cli.run_dir) / f"pooling_probe_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="pooling_probe_") as tmp, threadpool_limits(limits=cli.threads):
        arrays, targets, metadata, examples = extract(
            model, ds, device, tmp, cli.batch_size, cli.level, cli.grids, cli.tail_fraction, cli.block, cli.examples
        )
        for name, info in metadata.items():
            LOG.info(
                "%s: %d sites/region, tail k=%d (%.1f%%)",
                name,
                info["sites_per_region"],
                info["k"],
                100 * info["effective_fraction"],
            )
        rows, predictions = evaluate(arrays, targets, splits, cli.grids, cli.seed)
        del arrays
    np.savez_compressed(
        output / "predictions.npz",
        **predictions,
        train_indices=splits[0],
        validation_indices=splits[1],
        all_targets=targets,
    )
    if examples:
        np.savez_compressed(output / "examples.npz", **examples)
    summary = {
        "config": vars(cli),
        "checkpoint": str(checkpoint.resolve()),
        "dataset_settings": vars(args),
        "target_names": TARGETS,
        "stage": "native encoder output before content_norm, after configured latent mask",
        "pooling": metadata,
        "scores": rows,
        "notes": [
            "Frozen readout comparison, not evidence that retraining will reroute anatomy.",
            "IID separates factors but may differ from the training distribution; match can exploit SCM correlations.",
            "All spatial bins retained; no new foreground or anatomical mask for pooling.",
            "Lesion targets are latent coordinates, not lesion size/presence or rendered centroid.",
            "Combined descriptor has 3x the columns; inspect equal-width individual methods too.",
            "Confidence intervals condition on fitted probes; no correction for multiple comparisons.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\nHeld-out R²; each cell is ridge / RBF (selected on validation)")
    print("view   grid  method          ventricle       lesion xyz mean")
    for view in VIEWS:
        for grid in cli.grids:
            for method in METHODS:
                numbers = []
                for kind in ("ridge", "rbf"):
                    selected = {
                        r["target"]: r["test_r2"]
                        for r in rows
                        if (r["view"], r["grid"], r["method"], r["probe"], r["condition"])
                        == (view, grid, method, kind, "observed")
                    }
                    numbers.append((selected["ventricle_size"], np.mean([selected[f"lesion_{a}"] for a in "xyz"])))
                print(
                    f"{view:6s} {grid:4d}  {method:12s} {numbers[0][0]:+.3f} / {numbers[1][0]:+.3f}"
                    f"   {numbers[0][1]:+.3f} / {numbers[1][1]:+.3f}"
                )
    print(f"\nSaved {output}\nSee scores.csv for individual factors, shuffled controls and paired changes versus mean.")


if __name__ == "__main__":
    main()
