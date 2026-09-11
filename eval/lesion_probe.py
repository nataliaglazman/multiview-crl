"""Single-observation spatial probes for lesion coordinates, without PCA.

    python -m eval.lesion_probe --run-dir /path/to/run --causal iid

Compare content/style/joint at one encoder level, separately for each modality.
Joint means concatenated content and style from ONE view, not concatenated views.
Use native pre-quantization encoder maps by default; --grid 8 explicitly adds
spatial average pooling. Targets are rendered mask centroid (voxel coordinates)
and original z_content[2:5], fitted separately. No lesion-removal images are used.

Ridge and RBF kernel ridge share a 60/20/20 subject split. Feature/target scaling
is fitted on train only; validation selects regularization and RBF bandwidth;
test is used only for final scoring. A separately fitted shuffled-label control
uses the same protocol. Features are temporary disk-backed arrays; Gram matrices
avoid a parameter matrix proportional to the native spatial feature dimension.
Failure of these finite-sample probes is not proof of information loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eval.lesion_reconstruction import json_safe, location_metrics, make_dataset

logger = logging.getLogger(__name__)


def split_subjects(n, seed):
    if n < 30:
        raise ValueError("Use at least 30 subjects; 512–1024 is a more useful starting point")
    order = np.random.default_rng(seed).permutation(n)
    return (
        order[: int(0.6 * n)],
        order[int(0.6 * n) : int(0.8 * n)],
        order[int(0.8 * n) :],
    )


def view_content_mask(masks, level, view, channels):
    mask = masks.get(level)
    if mask is None:
        return np.ones(channels, dtype=bool)
    if isinstance(mask, tuple):
        mask = mask[view]
    result = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    if len(result) != channels:
        raise ValueError("Forward channel mask does not match the spatial feature map")
    return result


def extract_features(model, ds, device, directory, batch_size, level=0, grid=0):
    """Write native feature columns in channel-major order, with stable view masks."""
    import torch
    import torch.nn.functional as F

    arrays, partitions, shape = [], [], None
    targets, ids = [], []
    for start in range(0, len(ds), batch_size):
        images, masks = [], []
        for idx in range(start, min(start + batch_size, len(ds))):
            inner = ds._inner
            x1, x2, lat = inner[idx]
            mask = lat["brain_mask"]
            _, lesion = inner.renderer.render_structure(
                lat["z_content"],
                lat["z_deformation"],
                lat["z_fissure"],
                "cpu",
                clean=inner.clean_content,
            )
            support = np.argwhere(lesion.numpy() > 0)
            if not len(support):
                raise ValueError(f"Sample {idx} has no rendered lesion; its physical centroid is undefined")
            images.append(ds.normalize_views(x1, x2, mask, mask))
            masks.append(mask)
            targets.append(np.concatenate([support.mean(0), lat["z_content"].numpy()[2:5]]))
            ids.append(idx)
        x = torch.cat([torch.stack([s[v] for s in images]) for v in range(2)]).to(device)
        mask = torch.cat([torch.stack(masks)] * 2).to(device)
        with torch.inference_mode():
            out = model(
                x,
                return_recon=False,
                pool_only=False,
                n_views=2,
                subsets=[(0, 1)],
                mask=mask,
            )
        if level >= len(out[2]):
            raise ValueError(f"Requested level {level}, but model has {len(out[2])} levels")
        features = out[2][level]
        if features.ndim != 5 or not bool(torch.isfinite(features).all()):
            raise ValueError("Expected finite spatial encoder maps [2B,C,D,H,W]")
        if grid:
            if any(grid > d for d in features.shape[2:]):
                raise ValueError("--grid must not exceed the native feature resolution")
            features = F.adaptive_avg_pool3d(features, (grid,) * 3)
        b, channels = len(images), features.shape[1]
        if shape is None:
            shape = tuple(features.shape[1:])
            logger.info(
                "Temporary feature storage: %.2f GiB",
                2 * len(ds) * np.prod(shape) * 4 / 2**30,
            )
            for v in range(2):
                arrays.append(
                    np.lib.format.open_memmap(
                        Path(directory) / f"view{v}.npy",
                        mode="w+",
                        dtype="float32",
                        shape=(len(ds), int(np.prod(shape))),
                    )
                )
                partitions.append(view_content_mask(out[6], level, v, channels))
            logger.info(
                "Feature map C,D,H,W=%s; %d columns per view; no PCA",
                shape,
                np.prod(shape),
            )
        for v in range(2):
            if not np.array_equal(partitions[v], view_content_mask(out[6], level, v, channels)):
                raise ValueError(
                    "Content channel selection changes across batches; separate-block columns are not stable"
                )
            arrays[v][start : start + b] = features[v * b : (v + 1) * b].flatten(1).cpu().numpy()
        logger.info("Extracted %d/%d subjects", start + b, len(ds))
    for a in arrays:
        a.flush()
    return arrays, partitions, shape, np.asarray(targets), np.asarray(ids)


def block_gram(features, columns, train, chunk_size=2048):
    """Train-only standardization in bounded chunks; sum of feature outer products."""
    gram = np.zeros((len(features), len(features)), dtype=np.float64)
    active = 0
    for start in range(0, len(columns), chunk_size):
        x = np.asarray(features[:, columns[start : start + chunk_size]], dtype=np.float64)
        scaler = StandardScaler().fit(x[train])
        keep = scaler.var_ > 1e-12
        if not keep.any():
            continue
        z = scaler.transform(x)[:, keep]
        gram += np.dot(z, z.T)
        active += int(keep.sum())
    return gram, active


def score_predictions(truth, prediction, target):
    if target == "physical":
        return location_metrics(truth, prediction)
    ss = ((truth - truth.mean(0)) ** 2).sum(0)
    errors = ((truth - prediction) ** 2).sum(0)
    r2 = 1 - np.divide(errors, ss, out=np.full(3, np.nan), where=ss > 1e-12)
    return {
        "r2_xyz": r2.tolist(),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
    }


def fit_probes(gram, width, targets, splits, seed=0):
    """Fit all candidates on train; select on validation; never tune against test."""
    train, val, test = splits
    if width == 0:
        return [], {}
    if not np.isfinite(gram).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite feature kernel or target")
    linear = gram / width
    dist = np.maximum(np.diag(linear)[:, None] + np.diag(linear)[None, :] - 2 * linear, 0)
    conditions = {
        "observed": targets,
        "shuffled": targets[np.random.default_rng(seed + 17).permutation(len(targets))],
    }
    prepared = {}
    for condition, y in conditions.items():
        scaler = StandardScaler().fit(y[train])
        prepared[condition] = (y, scaler, scaler.transform(y))
    best = {}
    alphas = np.logspace(-6, 2, 9)
    for kind, gamma in [("ridge", None), ("rbf", 0.1), ("rbf", 1.0), ("rbf", 10.0)]:
        kernel = linear if gamma is None else np.exp(-gamma * dist)
        eigenvalues, vectors = eigh(kernel[np.ix_(train, train)])
        eigenvalues = np.maximum(eigenvalues, 0)
        cross_val = np.dot(kernel[np.ix_(val, train)], vectors)
        for condition, (y, scaler, standardized) in prepared.items():
            projection = np.dot(vectors.T, standardized[train])
            for alpha in alphas:
                weights = projection / (eigenvalues[:, None] + alpha)
                val_prediction = np.dot(cross_val, weights)
                if not np.isfinite(val_prediction).all():
                    raise FloatingPointError("Non-finite probe predictions; cannot select hyperparameters")
                for target, sl in (("physical", slice(0, 3)), ("latent", slice(3, 6))):
                    key = (kind, condition, target)
                    loss = float(np.mean((standardized[val, sl] - val_prediction[:, sl]) ** 2))
                    if key not in best or loss < best[key]["validation_mse_standardized"]:
                        best[key] = {
                            "validation_mse_standardized": loss,
                            "alpha": float(alpha),
                            "gamma": gamma,
                            "coefficients": np.dot(vectors, weights[:, sl]),
                        }
    rows, predictions = [], {}
    for (kind, condition, target), selected in best.items():
        y, scaler, standardized = prepared[condition]
        sl = slice(0, 3) if target == "physical" else slice(3, 6)
        kernel = linear if kind == "ridge" else np.exp(-selected["gamma"] * dist)
        weights = selected.pop("coefficients")
        row = {"probe": kind, "condition": condition, "target": target, **selected}
        for name, indices in (("train", train), ("validation", val), ("test", test)):
            prediction = np.dot(kernel[np.ix_(indices, train)], weights) * scaler.scale_[sl] + scaler.mean_[sl]
            row[name] = score_predictions(y[indices, sl], prediction, target)
            if name == "test" and condition == "observed":
                predictions[(kind, target)] = prediction
        rows.append(row)
    return rows, predictions


def run_probes(arrays, partitions, shape, targets, splits, ids, seed):
    results, saved_predictions = [], []
    sites = int(np.prod(shape[1:]))
    for view, features, content in zip(("t1", "flair"), arrays, partitions):
        grams = {}
        for block, channel_mask in (("content", content), ("style", ~content)):
            columns = np.flatnonzero(np.repeat(channel_mask, sites))
            grams[block] = block_gram(features, columns, splits[0])
        grams["joint"] = (
            grams["content"][0] + grams["style"][0],
            grams["content"][1] + grams["style"][1],
        )
        for block, (gram, width) in grams.items():
            logger.info("Fitting %s %s (%d variable spatial features)", view, block, width)
            if width == 0:
                results.append({"view": view, "block": block, "status": "no variable features"})
                continue
            rows, predictions = fit_probes(gram, width, targets, splits, seed)
            for row in rows:
                results.append({"view": view, "block": block, "feature_count": width, **row})
            for (kind, target), pred in predictions.items():
                offset = 0 if target == "physical" else 3
                for i, idx in enumerate(splits[2]):
                    saved_predictions.append(
                        {
                            "index": int(ids[idx]),
                            "view": view,
                            "block": block,
                            "probe": kind,
                            "target": target,
                            **{f"truth_{a}": targets[idx, offset + a] for a in range(3)},
                            **{f"pred_{a}": pred[i, a] for a in range(3)},
                        }
                    )
            for row in rows:
                if row["condition"] == "observed":
                    r2 = "/".join(f"{x:.2f}" for x in row["test"]["r2_xyz"])
                    logger.info(
                        "%s %s %s %s: held-out R²=%s",
                        view,
                        block,
                        row["probe"],
                        row["target"],
                        r2,
                    )
    return results, saved_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--causal", choices=["iid", "match"], default="iid")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--level", type=int, default=0)
    p.add_argument(
        "--grid",
        type=int,
        default=0,
        help="0=native spatial map; positive value=explicit pooled grid",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--temp-dir",
        default=None,
        help="Scratch parent for temporary spatial feature arrays",
    )
    cli = p.parse_args()
    if cli.num_samples < 30 or min(cli.batch_size, cli.cpu_threads) < 1 or min(cli.level, cli.grid) < 0:
        p.error("Need >=30 samples, positive batch size/threads, nonnegative level/grid")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir

    torch.set_num_threads(cli.cpu_threads)
    splits = split_subjects(cli.num_samples, cli.seed)
    with threadpool_limits(limits=cli.cpu_threads):
        model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device=cli.device, seed=cli.seed)
        ds = make_dataset(args, cli.num_samples, cli.causal, cli.split)
        with tempfile.TemporaryDirectory(prefix="lesion-probe-", dir=cli.temp_dir) as tmp:
            arrays, partitions, shape, targets, ids = extract_features(
                model, ds, device, tmp, cli.batch_size, cli.level, cli.grid
            )
            results, predictions = run_probes(arrays, partitions, shape, targets, splits, ids, cli.seed)
            del arrays
    directory = Path(cli.out_dir or Path(cli.run_dir) / f"lesion_probe_{cli.causal}_L{cli.level}_grid{cli.grid}")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": vars(cli),
        "run_settings": vars(args),
        "feature_shape": shape,
        "content_channels_by_view": [np.flatnonzero(mask).tolist() for mask in partitions],
        "subject_indices": {k: ids[v].tolist() for k, v in zip(("train", "validation", "test"), splits)},
        "feature_source": "Pre-quantization encoder spatial output at selected level; joint is content+style within one view",
        "preprocessing": "No PCA; train-only per-feature standardization; constant train features dropped",
        "protocol": "60/20/20 train/validation/test; validation selects alpha/gamma separately by target group; no refit on validation",
        "limitations": "A finite-sample probe, not an information-theoretic ceiling or a test of all decoder inputs/levels",
        "results": results,
    }
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    if predictions:
        with (directory / "predictions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    print("\nHeld-out test scores. Physical errors are in voxels; latent scores target z_content[2:5].")
    for row in results:
        if row.get("condition") != "observed":
            continue
        null = next(
            r
            for r in results
            if r.get("condition") == "shuffled" and all(r[k] == row[k] for k in ("view", "block", "probe", "target"))
        )
        r2 = "/".join(f"{x:.2f}" for x in row["test"]["r2_xyz"])
        error = row["test"].get("median_error_vox")
        suffix = f"  median={error:.2f} vox" if error is not None else ""
        print(
            f"  {row['view']:5s} {row['block']:7s} {row['probe']:5s} {row['target']:8s}"
            f"  R²={r2}  train_mean={np.mean(row['train']['r2_xyz']):.2f}"
            f"  null_mean={np.mean(null['test']['r2_xyz']):.2f}{suffix}"
        )
    print("\nCompare joint vs separate blocks, RBF vs ridge, and physical vs latent targets.")
    print("Low scores remain probe/sample-size limits unless stronger decoding tests agree.")
    print(f"Saved {directory}")


if __name__ == "__main__":
    main()
