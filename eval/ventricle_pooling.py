"""Compare frozen ventricular readouts of the patch tensors supplied to training.

No encoder updates. Anatomical targets are used only for diagnostic probes. Features
are extracted once, before patch centering, using the forward content mask and the
training foreground rule. See VENTRICLE_POOLING.md for interpretation and limitations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
VIEWS = ("t1", "flair")
POOLS = ("gap", "mean_std", "stats", "regional_mean_std")


def region_ids(grid, regions):
    """Fixed, disjoint, equal-sized cells in the ORIGINAL patch lattice; no ROI labels."""
    grid, regions = tuple(grid), tuple(regions)
    if len(grid) != 3 or len(regions) != 3 or any(g < 1 or r < 1 or g % r for g, r in zip(grid, regions)):
        raise ValueError("Each positive regional-grid dimension must divide the corresponding patch-grid dimension.")
    coordinates = np.indices(grid).reshape(3, -1)
    coarse = coordinates // (np.array(grid) // np.array(regions))[:, None]
    return np.ravel_multi_index(coarse, regions)


def pool_descriptors(hz, keep, grid, regions):
    """(2,B,C,P) -> four (2,B,width) descriptors, all using the same retained positions.

    Global statistics use unbiased std, matching bt-gap-pooling=stats. Regional
    blocks contain [all channel means, all channel stds] in each fixed cell. Empty
    cells are zero-filled; one-position cells have std=0. Counts are reported, not
    supplied to probes. Regions never reindex the foreground-filtered patch list.
    """
    import torch

    labels = region_ids(grid, regions)
    if hz.ndim != 4 or hz.shape[0] != 2 or hz.shape[-1] != len(labels):
        raise ValueError("Expected paired features (2,B,C,prod(grid)).")
    keep = torch.as_tensor(keep, device=hz.device, dtype=torch.bool)
    if keep.shape != (len(labels),) or int(keep.sum()) < 2:
        raise ValueError("Stats pooling requires at least two retained positions.")
    if not bool(torch.isfinite(hz).all()):
        raise ValueError("Non-finite encoder features.")
    z = hz.float()[..., keep]
    mean, std = z.mean(-1), z.std(-1, unbiased=True)
    mean_std = torch.cat((mean, std), dim=-1)
    labels = torch.as_tensor(labels, device=hz.device)
    blocks, counts = [], []
    for cell in range(int(np.prod(regions))):
        selected = keep & (labels == cell)
        count = int(selected.sum())
        counts.append(count)
        if count == 0:
            mu = sd = torch.zeros_like(mean)
        else:
            local = hz.float()[..., selected]
            mu = local.mean(-1)
            sd = local.std(-1, unbiased=True) if count > 1 else torch.zeros_like(mu)
        blocks.append(torch.cat((mu, sd), dim=-1))
    return {
        "gap": mean,
        "mean_std": mean_std,
        "stats": torch.cat((mean_std, z.amax(-1), z.amin(-1)), dim=-1),
        "regional_mean_std": torch.cat(blocks, dim=-1),
    }, counts


def select_content(features, forward_mask, model, level):
    """Use the actual fixed forward selection, including separate per-view masks."""
    import torch

    if forward_mask is None:
        count = getattr(model, "content_channels_per_level", {}).get(level)
        if count != features.shape[2]:
            raise ValueError("Missing content mask for a partially selected encoder level.")
        return features, [list(range(features.shape[2]))] * 2
    masks = forward_mask if isinstance(forward_mask, tuple) else (forward_mask, forward_mask)
    indices = [torch.where(m.reshape(-1).bool())[0] for m in masks]
    if len(indices[0]) == 0 or len(indices[0]) != len(indices[1]):
        raise ValueError("Views must select the same positive number of content channels.")
    return torch.stack([features[v][:, idx] for v, idx in enumerate(indices)]), [i.tolist() for i in indices]


def extract_features(
    model,
    dataset,
    args,
    device,
    grid,
    regions,
    fit_samples,
    mask_batch,
    encode_batch,
    level=0,
    target_key=None,
    include_patch_flat=False,
    partition_ends=None,
):
    """Encode microbatches once, then apply foreground ANY over each full logical batch.

    Fit and held-out subjects never share a mask batch. CPU patch maps are retained
    only until a logical batch is complete; full-resolution images are not cached.
    Optional target_key/partition_ends/patch_flat also support lesion localization.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    from eval.reconstruction_attribution import _inputs

    pools = (*POOLS, "patch_flat") if include_patch_flat else POOLS
    features = {f"{v}/{p}": [] for v in VIEWS for p in pools}
    targets, ids, groups, group_info = [], [], [], []
    canonical_indices = None
    grid_arg = getattr(args, "patch_grid_per_level", None)
    if grid_arg:
        grid_arg = [tuple(g) for g in grid_arg]
        grid_arg[level] = tuple(grid)
    else:
        grid_arg = tuple(grid)
    use_foreground = bool(getattr(args, "patch_foreground_mask", False))
    threshold = float(getattr(args, "patch_foreground_thresh", 0.05))
    ends = list(partition_ends) if partition_ends is not None else [fit_samples, len(dataset)]
    if not ends or ends[-1] != len(dataset) or any(a >= b for a, b in zip([0] + ends, ends)):
        raise ValueError("Partition ends must increase strictly and end at dataset length.")
    with torch.no_grad():
        for number, (begin, end) in enumerate(zip([0] + ends, ends)):
            partition = ("fit" if end <= fit_samples else "test") if partition_ends is None else f"partition_{number}"
            for start in range(begin, end, mask_batch):
                stop = min(start + mask_batch, end)
                loader = DataLoader(
                    Subset(dataset, range(start, stop)),
                    batch_size=encode_batch,
                    shuffle=False,
                )
                chunks, ys = [], []
                keep = torch.zeros(int(np.prod(grid)), dtype=torch.bool)
                for batch in loader:
                    x, masks = _inputs(batch, device)
                    out = model(
                        x,
                        return_recon=False,
                        pool_only=True,
                        n_views=2,
                        subsets=[(0, 1)],
                        patch_grid=grid_arg,
                        mask=masks,
                    )
                    raw = out[2][level]
                    if raw.ndim != 3 or len(raw) != len(x):
                        raise ValueError("Forward did not return view-major patch maps.")
                    hz = raw.reshape(2, len(x) // 2, *raw.shape[1:])
                    hz, indices = select_content(hz, out[6].get(level), model, level)
                    if canonical_indices is not None and indices != canonical_indices:
                        raise ValueError("Content selection changed between batches; fixed descriptors are required.")
                    canonical_indices = indices
                    chunks.append(hz.detach().float().cpu())
                    if use_foreground:
                        if masks is None:
                            raise ValueError("Training uses foreground filtering but the dataset has no masks.")
                        frac = F.adaptive_avg_pool3d(masks.float(), tuple(grid)).flatten(1)
                        keep |= (frac >= threshold).any(0).cpu()
                    else:
                        keep.fill_(True)
                    target = batch[target_key] if target_key else batch["gt_latents"]["z_content"][:, 1]
                    ys.append(target.detach().cpu().numpy())
                    del out, raw, hz, x, masks
                fallback = not bool(keep.any())
                if fallback:
                    keep.fill_(True)  # exact training fallback when EVERY position was dropped
                patches = torch.cat(chunks, dim=1)
                descriptors, counts = pool_descriptors(patches, keep, grid, regions)
                if include_patch_flat:
                    # Preserve ORIGINAL spatial coordinates despite changing keep sets.
                    # A removed position is zero-filled, not deleted/reindexed.
                    descriptors["patch_flat"] = (patches * keep).flatten(2)
                for view, name in enumerate(VIEWS):
                    for pool, tensor in descriptors.items():
                        features[f"{name}/{pool}"].append(tensor[view].numpy())
                group = len(group_info)
                targets.extend(np.concatenate(ys).tolist())
                ids.extend(range(start, stop))
                groups.extend([group] * (stop - start))
                group_info.append(
                    {
                        "partition": partition,
                        "start": start,
                        "stop": stop,
                        "kept_positions": torch.where(keep)[0].tolist(),
                        "region_counts": counts,
                        "all_background_fallback": fallback,
                    }
                )
                logger.info(
                    "Pooled %s subjects %d:%d; kept %d/%d patches",
                    partition,
                    start,
                    stop,
                    int(keep.sum()),
                    len(keep),
                )
    arrays = {k: np.concatenate(v) for k, v in features.items()}
    arrays.update(
        targets=np.asarray(targets),
        subject_id=np.asarray(ids),
        mask_group=np.asarray(groups),
        is_fit=np.arange(len(dataset)) < fit_samples,
    )
    return arrays, {
        "mask_groups": group_info,
        "content_indices_by_view": canonical_indices,
    }


def validate_settings(args, level, grid, regions):
    if getattr(args, "mask_mode", "onthefly") != "fixed":
        raise ValueError("This test requires mask_mode=fixed so content coordinates remain consistent.")
    if (
        int(getattr(args, "contrastive_proj_dim", 0) or 0) > 0
        or getattr(args, "contrastive_proj_mode", "head") != "head"
    ):
        raise ValueError(
            "Projected/bounded/entropy loss-facing transforms are not supported; this test reads raw encoder patches."
        )
    if getattr(args, "split_encoder_norm", False) or getattr(args, "use_moco", False):
        raise ValueError("split_encoder_norm/MoCo are not supported by this diagnostic's checkpoint loader.")
    if not 0 <= level < int(getattr(args, "vqvae_nb_levels", 1)):
        raise ValueError("Invalid encoder level.")
    if [tuple(s) for s in getattr(args, "subsets", [(0, 1)])] != [(0, 1)]:
        raise ValueError("Expected one paired subset (0, 1).")
    if grid is None:
        raise ValueError("No patch grid in settings; supply --grid, e.g. --grid 8 8 8 for a baseline run.")
    region_ids(grid, regions)
    if np.prod(grid) < 2:
        raise ValueError("Global stats require at least two patch positions.")


def fit_readout(X, y, probe, seed, folds):
    """All scaling and tuning fit within CV training folds; test data is never passed."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if probe == "ridge":
        estimator = Ridge()
        grid = {"regressor__ridge__alpha": [1e-4, 0.01, 1.0, 100.0, 1e4]}
    else:
        estimator = KernelRidge(kernel="rbf")
        grid = {
            "regressor__kernelridge__alpha": [0.001, 0.1, 10.0],
            "regressor__kernelridge__gamma": [g / X.shape[1] for g in (0.1, 1.0, 10.0)],
        }
    model = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), estimator),
        transformer=StandardScaler(),
    )
    search = GridSearchCV(
        model,
        grid,
        cv=KFold(folds, shuffle=True, random_state=seed),
        scoring="r2",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "params": search.best_params_,
        "fit_cv_r2": float(search.best_score_),
    }


def paired_score(y, prediction, reference, seed, draws):
    """Subject bootstrap conditional on fitted probes, extracted features and mask groups."""
    den = np.sum((y - y.mean()) ** 2)
    if den <= 1e-12:
        raise ValueError("Held-out ventricular target is constant.")
    error, ref_error = (y - prediction) ** 2, (y - reference) ** 2
    rng = np.random.default_rng(seed)
    # Loop keeps bootstrap memory bounded even for a large cached test set.
    scores, deltas = [], []
    for _ in range(draws):
        idx = rng.integers(len(y), size=len(y))
        d = np.sum((y[idx] - y[idx].mean()) ** 2)
        if d > 1e-12:
            scores.append(1 - error[idx].sum() / d)
            deltas.append((ref_error[idx] - error[idx]).sum() / d)
    if not scores:
        raise ValueError("No non-degenerate bootstrap samples.")
    lo, hi = np.quantile(scores, [0.025, 0.975])
    dl, dh = np.quantile(deltas, [0.025, 0.975])
    return {
        "r2": float(1 - error.sum() / den),
        "r2_ci_low": float(lo),
        "r2_ci_high": float(hi),
        "delta_r2": float((ref_error - error).sum() / den),
        "delta_ci_low": float(dl),
        "delta_ci_high": float(dh),
    }


def validate_arrays(arrays):
    y, fit = arrays["targets"], arrays["is_fit"]
    if y.ndim != 1 or fit.shape != y.shape or fit.dtype != np.bool_:
        raise ValueError("Cache must contain 1D targets and a boolean is_fit partition.")
    if min(int(fit.sum()), int((~fit).sum())) < 4 or not np.isfinite(y).all():
        raise ValueError("Need at least four finite targets in each partition.")
    if np.var(y[fit]) <= 1e-12 or np.var(y[~fit]) <= 1e-12:
        raise ValueError("Ventricular targets must vary in both partitions.")
    if len(np.unique(arrays["subject_id"])) != len(y):
        raise ValueError("Duplicate subject IDs in the feature cache.")
    if set(arrays["mask_group"][fit]) & set(arrays["mask_group"][~fit]):
        raise ValueError("Fit and held-out subjects must not share foreground mask groups.")
    for view in VIEWS:
        for pool in POOLS:
            X = arrays[f"{view}/{pool}"]
            if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] < 1 or not np.isfinite(X).all():
                raise ValueError(f"Invalid feature matrix for {view}/{pool}.")


def evaluate(arrays, probes, seed, folds, draws):
    validate_arrays(arrays)
    y, fit = arrays["targets"], arrays["is_fit"]
    if folds < 2 or fit.sum() // folds < 2:
        raise ValueError("CV needs at least two subjects in every validation fold.")
    predictions, selections = {}, {}
    for view in VIEWS:
        for pool in POOLS:
            X = arrays[f"{view}/{pool}"]
            for probe in probes:
                key = f"{view}/{pool}/{probe}"
                estimator, selections[key] = fit_readout(X[fit], y[fit], probe, seed, folds)
                predictions[key] = estimator.predict(X[~fit])
                logger.info(
                    "Fit %s (%d descriptors); fit CV R² %.3f",
                    key,
                    X.shape[1],
                    selections[key]["fit_cv_r2"],
                )
    rows = []
    for key, prediction in predictions.items():
        view, pool, probe = key.split("/")
        base = predictions[f"{view}/gap/{probe}"]
        row = {
            "view": view,
            "pooling": pool,
            "probe": probe,
            "width": arrays[f"{view}/{pool}"].shape[1],
            "n_fit": int(fit.sum()),
            "n_test": int((~fit).sum()),
        }
        row.update(paired_score(y[~fit], prediction, base, seed, draws))
        # This is the comparison that tests whether extrema add anything beyond moments.
        moments = paired_score(y[~fit], prediction, predictions[f"{view}/mean_std/{probe}"], seed, draws)
        row.update({f"vs_mean_std_{k}": v for k, v in moments.items() if k.startswith("delta")})
        rows.append(row)
    predictions.update(targets=y[~fit], subject_id=arrays["subject_id"][~fit])
    return rows, predictions, selections


def save_cache(path, arrays, metadata):
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)), **arrays)


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported feature cache format.")
    validate_arrays(arrays)
    return arrays, metadata


def run(cli):
    if cli.threads < 1 or cli.bootstrap < 1:
        raise ValueError("--threads and --bootstrap must be positive.")
    parent = Path(cli.run_dir) if cli.run_dir else Path(cli.features).resolve().parent
    directory = Path(cli.out) if cli.out else parent / datetime.now().strftime("ventricle_pooling_%Y%m%d_%H%M%S_%f")
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")
    if cli.features:
        arrays, metadata = load_cache(cli.features)
        logger.info("Reusing cached descriptors; no rendering or checkpoint loading.")
    else:
        import torch

        from eval.reconstruction_attribution import frozen_checkpoint
        from eval.run_dci_synthetic import load_model_from_run_dir, load_run_args
        from eval.ventricle_routing import make_dataset, stable_replay_math

        if min(cli.fit_samples, cli.test_samples) < 4 or cli.encode_batch < 1:
            raise ValueError("Need >=4 subjects per partition and a positive encode batch.")
        if cli.folds < 2 or cli.fit_samples // cli.folds < 2:
            raise ValueError("CV needs at least two subjects per validation fold.")
        torch.set_num_threads(cli.threads)
        args = load_run_args(cli.run_dir)
        grid = getattr(args, "patch_grid_per_level", None)
        grid = grid[cli.level] if grid and 0 <= cli.level < len(grid) else getattr(args, "patch_grid", None)
        grid = cli.grid or grid
        validate_settings(args, cli.level, grid, cli.regions)
        mask_batch = cli.mask_batch_size or int(getattr(args, "batch_size", 64))
        if mask_batch < 1:
            raise ValueError("Mask batch size must be positive.")
        checkpoint = Path(cli.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path(cli.run_dir) / checkpoint
        checkpoint = checkpoint.resolve()
        model, args, device = load_model_from_run_dir(cli.run_dir, str(checkpoint), cli.device, seed=cli.seed)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = state.get("step")
        weights = state.get("encoders", state)
        model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
        del state, weights
        ds = make_dataset(args, cli.fit_samples + cli.test_samples, cli.causal, "test")
        with stable_replay_math(), frozen_checkpoint(model):
            arrays, details = extract_features(
                model,
                ds,
                args,
                device,
                grid,
                cli.regions,
                cli.fit_samples,
                mask_batch,
                cli.encode_batch,
                cli.level,
            )
        with checkpoint.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
        metadata = {
            "format_version": 1,
            "run_dir": str(Path(cli.run_dir).resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "checkpoint_step": step,
            "settings": vars(args),
            "causal": cli.causal,
            "dataset_split": "test",
            "render_resolution": ds.res,
            "grid": list(grid),
            "regions": cli.regions,
            "mask_batch_size": mask_batch,
            "encode_batch": cli.encode_batch,
            "level": cli.level,
            "feature_stage": "unquantized, uncentered encoder patches returned by model forward",
            "target": "z_content[1] (ventricle_size)",
            **details,
        }
    validate_arrays(arrays)
    # Save BEFORE CPU probe fitting, so an interrupted fitting job does not waste encoding.
    directory.mkdir(parents=True)
    save_cache(directory / "features.npz", arrays, metadata)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=cli.threads):
        rows, predictions, selections = evaluate(arrays, cli.probes, cli.seed, cli.folds, cli.bootstrap)
    report = {
        "metadata": metadata,
        "cli": vars(cli),
        "results": rows,
        "probe_selection": selections,
        "limitations": [
            "Frozen readout comparison; does not demonstrate better training or decoder routing.",
            "IID evaluation removes SCM factor correlation but can be outside the training distribution; matched evaluation may exploit correlated factors.",
            "Bootstrap intervals condition on fitted probes, this subject split and foreground groups; they omit training-seed and probe-fit uncertainty.",
            "Pooling widths differ; regularized readouts are tuned separately on fit subjects only.",
            "Eval-mode forward uses training's patch and mask operations, not historical training batches, stochastic layers or autocast.",
            "Regional pooling uses fixed equal cells, zero-filled empty cells, and zero std for one-position cells. No anatomical ROI or label selects cells.",
        ],
    }
    with (directory / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with (directory / "pooling_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "predictions.npz", **predictions)
    print("\nFrozen ventricular R² on held-out subjects (Δ relative to the same view/probe's GAP):")
    for row in rows:
        print(
            f"{row['view']:5} {row['pooling']:18} {row['probe']:5} d={row['width']:4} "
            f"R²={row['r2']:+.3f}  ΔGAP={row['delta_r2']:+.3f} "
            f"[{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}]  "
            f"Δmean/std={row['vs_mean_std_delta_r2']:+.3f} "
            f"[{row['vs_mean_std_delta_ci_low']:+.3f}, {row['vs_mean_std_delta_ci_high']:+.3f}]"
        )
    print(f"Saved {directory}\nReadout evidence only; no model or optimizer update.")
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--features", help="Reuse features.npz; extraction options are ignored.")
    ap.add_argument("--checkpoint", default="vqvae_model.pt")
    ap.add_argument("--fit-samples", type=int, default=384)
    ap.add_argument("--test-samples", type=int, default=128)
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument(
        "--mask-batch-size",
        type=int,
        help="Foreground ANY group size; default is training batch size.",
    )
    ap.add_argument(
        "--grid",
        type=int,
        nargs=3,
        help="Override training patch grid (recorded in report).",
    )
    ap.add_argument("--regions", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--causal", choices=["iid", "match"], default="iid")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Probe CV seed; renderer/SCM seed comes from settings.json.",
    )
    ap.add_argument("--probes", choices=["ridge", "rbf"], nargs="+", default=["ridge", "rbf"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads for extraction and linear algebra.",
    )
    ap.add_argument("--out")
    return ap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run(parser().parse_args())
