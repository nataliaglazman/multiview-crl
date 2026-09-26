"""Train-only content-codebook refit with paired held-out evaluation.

See CODEBOOK_RECALIBRATION.md. Only content embed buffers in a disposable copy
change. No anatomical targets enter fitting and no model checkpoint is written.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from eval.content_path_probe import stage_maps, state_digest
from eval.lesion_reconstruction import json_safe, make_dataset
from eval.pooling_probe import VIEWS
from eval.style_path_audit import validate_model
from eval.ventricle_quantizer_audit import audit, codebook_for, summarize

LOG = logging.getLogger(__name__)


def natural_batch(ds, start, batch_size, device):
    images, masks = [], []
    for index in range(start, min(start + batch_size, len(ds))):
        a, b, metadata = ds._inner[index]
        mask = metadata["brain_mask"]
        images.append(ds.normalize_views(a, b, mask, mask))
        masks.append(mask)
    x = torch.cat([torch.stack([sample[v] for sample in images]) for v in range(2)]).to(device)
    mask = torch.cat([torch.stack(masks)] * 2).to(device)
    return x, mask, len(images)


def share_training_reference(train, test):
    """Fit the fixed affine on train only; per-sample/shared modes need no dataset fit."""
    if train.synthetic_normalize != test.synthetic_normalize:
        raise ValueError("Train and test normalizers must match")
    if train.synthetic_normalize != "fixed_reference":
        return {"mode": train.synthetic_normalize}
    train._compute_fixed_reference()
    test._fixed_mean, test._fixed_scale = train._fixed_mean, train._fixed_scale
    return {"mode": "fixed_reference", "source": "train", "mean": train._fixed_mean, "scale": train._fixed_scale}


def collect_points(model, ds, device, batch_size, sites_per_subject, seed):
    """Uniform sites from natural observations; no factor/ROI-based selection."""
    generator = torch.Generator().manual_seed(seed)
    banks, previous_masks = {}, None
    for start in range(0, len(ds), batch_size):
        x, mask, count = natural_batch(ds, start, batch_size, device)
        stages, masks = stage_maps(model, x, mask)
        if previous_masks is not None and not np.array_equal(masks, previous_masks):
            raise ValueError("Unstable content channels across calibration batches")
        previous_masks = masks
        maps = stages[("content", "pre_quant")]
        for view in range(2):
            name = VIEWS[view] if model.separate_content_codebooks else "shared"
            for subject in range(count):
                points = maps[view * count + subject].flatten(1).T
                ids = torch.randperm(len(points), generator=generator)[: min(sites_per_subject, len(points))]
                banks.setdefault(name, []).append(points[ids.to(points.device)].detach().float().cpu())
        LOG.info("Collected training features: %d/%d subjects", start + count, len(ds))
    return {name: torch.cat(parts) for name, parts in banks.items()}


@torch.no_grad()
def assignments(points, centers):
    distances = points.square().sum(1, keepdim=True) - 2 * points @ centers.T + centers.square().sum(1)[None]
    return distances.argmin(1)


@torch.no_grad()
def training_mse(points, centers, chunk_size):
    error = 0.0
    for start in range(0, len(points), chunk_size):
        x = points[start : start + chunk_size].to(centers.device)
        ids = assignments(x, centers)
        error += float((x.double() - centers[ids].double()).square().sum())
    return error / points.numel()


@torch.no_grad()
def refit_centers(points, initial, iterations=10, chunk_size=8192):
    """Lloyd updates from current code vectors; retain empty entries and indices.

    Select the best iterate using calibration distortion ONLY. No EMA mode,
    optimizer, reset/reseed, whitening, anatomical target or held-out selection.
    points are [N,D] on CPU; centers are [K,D] on the requested fit device.
    """
    if points.ndim != 2 or initial.ndim != 2 or points.shape[1] != initial.shape[1] or not len(points):
        raise ValueError("Need nonempty [N,D] training points and [K,D] centers")
    if iterations < 1 or chunk_size < 1 or not torch.isfinite(points).all() or not torch.isfinite(initial).all():
        raise ValueError("Invalid fitting settings or non-finite data")
    centers = initial.detach().float().clone()
    best = centers.clone()
    best_loss = training_mse(points, centers, chunk_size)
    trace = [{"iteration": 0, "training_mse": best_loss}]
    best_iteration = 0
    for iteration in range(1, iterations + 1):
        sums = torch.zeros_like(centers, dtype=torch.float64)
        counts = torch.zeros(len(centers), dtype=torch.float64, device=centers.device)
        for start in range(0, len(points), chunk_size):
            x = points[start : start + chunk_size].to(centers.device)
            ids = assignments(x, centers)
            sums.index_add_(0, ids, x.double())
            counts += torch.bincount(ids, minlength=len(centers))
        occupied = counts > 0
        centers[occupied] = (sums[occupied] / counts[occupied, None]).float()
        loss = training_mse(points, centers, chunk_size)
        trace.append(
            {
                "iteration": iteration,
                "training_mse": loss,
                "occupied_entries": int(occupied.sum()),
                "empty_entries_retained": int((~occupied).sum()),
            }
        )
        LOG.info("Refit iteration %d/%d: train MSE %.6g", iteration, iterations, loss)
        if loss < best_loss:
            best, best_loss, best_iteration = centers.clone(), loss, iteration
    return best, {
        "training_points": len(points),
        "embedding_dim": points.shape[1],
        "selected_iteration": best_iteration,
        "trace": trace,
        "center_movement_rms": float((best - initial).square().mean().sqrt()),
        "entries_moved": int(((best - initial).square().sum(1) > 0).sum()),
    }


def recalibrated_copy(model, banks, device, iterations, chunk_size):
    candidate = copy.deepcopy(model).eval().requires_grad_(False)
    fit, vectors, allowed = {}, {}, set()
    for name, points in banks.items():
        view = 1 if name == "flair" else 0
        cb = codebook_for(candidate, "content", view)
        initial = cb.embed.detach().T.to(device).contiguous()
        centers, info = refit_centers(points, initial, iterations, chunk_size)
        with torch.no_grad():
            cb.embed.copy_(centers.T.to(cb.embed.device))
        allowed.add("codebooks_v1.0.embed" if view else "codebooks.0.embed")
        fit[name] = info
        vectors[f"{name}_original"] = initial.cpu().numpy()
        vectors[f"{name}_recalibrated"] = centers.cpu().numpy()
    changed = []
    for name, value in candidate.state_dict().items():
        if not torch.equal(value, model.state_dict()[name]):
            if name not in allowed:
                raise ValueError(f"Non-content-embedding state changed: {name}")
            changed.append(name)
    return candidate, fit, vectors, changed


def paired_change(original, candidate, seed=0, draws=500):
    a, b = np.asarray(original, dtype=float), np.asarray(candidate, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if not len(a):
        return {"n": 0}
    delta = b - a
    rng = np.random.default_rng(seed)
    bootstrap = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(draws)]
    lo, hi = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "n": len(a),
        "original_mean": float(a.mean()),
        "recalibrated_mean": float(b.mean()),
        "original_median": float(np.median(a)),
        "recalibrated_median": float(np.median(b)),
        "mean_delta": float(delta.mean()),
        "mean_delta_ci_low": float(lo),
        "mean_delta_ci_high": float(hi),
    }


def reconstruction_comparison(original, candidate, ds, device, batch_size):
    """Natural held-out observations, raw predictions and original brain mask."""
    rows = []
    for start in range(0, len(ds), batch_size):
        x, mask, count = natural_batch(ds, start, batch_size, device)
        with torch.inference_mode():
            predictions = [
                model(x, n_views=2, subsets=[(0, 1)], pool_only=True, return_recon=True, mask=mask)[0]
                for model in (original, candidate)
            ]
        if any(y.shape != x.shape or not torch.isfinite(y).all() for y in predictions):
            raise ValueError("Invalid reconstruction")
        for v, view in enumerate(VIEWS):
            for subject in range(count):
                idx = v * count + subject
                foreground = mask[idx].bool()
                if not foreground.any():
                    raise ValueError("Empty held-out foreground")
                row = {"index": start + subject, "view": view}
                for condition, prediction in zip(("original", "recalibrated"), predictions):
                    error = (prediction[idx] - x[idx])[foreground].double()
                    row[f"{condition}_mae"] = float(error.abs().mean())
                    row[f"{condition}_rmse"] = float(error.square().mean().sqrt())
                row["prediction_change_mae"] = float(
                    (predictions[1][idx] - predictions[0][idx])[foreground].abs().mean()
                )
                rows.append(row)
        LOG.info("Paired held-out reconstruction: %d/%d subjects", start + count, len(ds))
    summary = []
    for view in VIEWS:
        subset = [r for r in rows if r["view"] == view]
        for metric in ("mae", "rmse"):
            summary.append(
                {
                    "view": view,
                    "metric": metric,
                    **paired_change(
                        [r[f"original_{metric}"] for r in subset], [r[f"recalibrated_{metric}"] for r in subset]
                    ),
                }
            )
    return rows, summary


def response_comparison(original, candidate):
    def key(row):
        return tuple(row[k] for k in ("index", "eps", "view", "block", "region"))

    a, b = {key(r): r for r in original}, {key(r): r for r in candidate}
    if a.keys() != b.keys() or len(a) != len(original) or len(b) != len(candidate):
        raise ValueError("Original/recalibrated intervention subjects do not match")
    for k, row in a.items():
        other = b[k]
        for metric in ("input_delta_rms", "pre_delta_rms", "projection_input_delta_rms"):
            if metric in row and not np.isclose(row[metric], other[metric], atol=2e-7, rtol=2e-5):
                raise ValueError(f"Frozen input/continuous features changed: {k}, {metric}")
    groups = list(dict.fromkeys(k[1:] for k in a))
    comparisons = []
    for group in groups:
        keys = [
            k
            for k in a
            if k[1:] == group
            and a[k]["input_measurable"]
            and b[k]["input_measurable"]
            and a[k]["sites"]
            and b[k]["sites"]
        ]
        for metric in (
            "endpoint_quantization_error_rms",
            "quant_response_relative_error",
            "quant_response_cosine",
            "quant_to_pre_rms_ratio",
            "changed_code_fraction",
        ):
            values_a = [a[k].get(metric) if a[k].get(metric) is not None else np.nan for k in keys]
            values_b = [b[k].get(metric) if b[k].get(metric) is not None else np.nan for k in keys]
            comparisons.append(
                {
                    "eps": group[0],
                    "view": group[1],
                    "block": group[2],
                    "region": group[3],
                    "metric": metric,
                    **paired_change(values_a, values_b),
                }
            )
    return comparisons


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fit-samples", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=64, help="Held-out reconstruction and intervention subjects")
    parser.add_argument(
        "--sites-per-subject", type=int, default=256, help="Uniform native sites per view and training subject"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.25])
    parser.add_argument("--halo", type=int, default=1)
    parser.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="match",
        help="Distribution of calibration AND held-out base subjects",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Training site sampling only; renderer seed comes from settings"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args(argv)
    if (
        min(
            cli.fit_samples,
            cli.num_samples,
            cli.sites_per_subject,
            cli.iterations,
            cli.chunk_size,
            cli.batch_size,
            cli.threads,
        )
        < 1
        or min(cli.halo, cli.examples) < 0
    ):
        parser.error("Counts must be positive, halo/examples nonnegative")
    if any(not np.isfinite(e) or e <= 0 for e in cli.eps):
        parser.error("eps must be finite and positive")
    cli.eps = list(dict.fromkeys(cli.eps))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.threads)
    from eval.run_dci_synthetic import load_model_from_run_dir

    original, args, device = load_model_from_run_dir(
        cli.run_dir, cli.checkpoint, torch.device(cli.device) if cli.device else None
    )
    checkpoint = Path(cli.checkpoint or "vqvae_model.pt")
    if checkpoint.parent == Path("."):
        checkpoint = Path(cli.run_dir) / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original.load_state_dict(
        {k.removeprefix("module."): v for k, v in state.get("encoders", state).items()}, strict=True
    )
    step = state.get("step")
    del state
    original.eval().requires_grad_(False)
    validate_model(original)
    before = state_digest(original)
    train = make_dataset(args, cli.fit_samples, cli.causal, "train")
    test = make_dataset(args, cli.num_samples, cli.causal, "test")
    train_seeds = {train._inner.sample_seed_for(i) for i in range(len(train))}
    test_seeds = {test._inner.sample_seed_for(i) for i in range(len(test))}
    if train_seeds & test_seeds:
        raise ValueError("Calibration/evaluation rendering seeds overlap")
    normalization = share_training_reference(train, test)
    output = Path(cli.output_dir or Path(cli.run_dir) / f"codebook_recalibration_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    banks = collect_points(original, train, device, cli.batch_size, cli.sites_per_subject, cli.seed)
    candidate, fit, vectors, changed = recalibrated_copy(original, banks, device, cli.iterations, cli.chunk_size)
    del banks
    calibrated_digest = state_digest(candidate)
    # Neither held-out reconstruction nor ventricular interventions were accessed during fitting.
    recon_rows, recon_summary = reconstruction_comparison(original, candidate, test, device, cli.batch_size)
    response_rows, summaries, usage = {}, {}, {}
    for condition, model in (("original", original), ("recalibrated", candidate)):
        LOG.info("Held-out ventricle audit: %s", condition)
        rows, counts, examples = audit(model, test, device, cli.eps, cli.batch_size, cli.halo, cli.examples)
        response_rows[condition], summaries[condition], usage[condition] = rows, summarize(rows), counts
        if examples:
            np.savez_compressed(output / f"{condition}_examples.npz", **examples)
    comparison = response_comparison(response_rows["original"], response_rows["recalibrated"])
    if state_digest(original) != before or state_digest(candidate) != calibrated_digest:
        raise ValueError("Model state changed unexpectedly during calibration/evaluation")
    report = {
        "config": vars(cli),
        "dataset_settings": vars(args),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": step,
        "normalization": normalization,
        "train_render_seeds": sorted(train_seeds),
        "test_render_seeds": sorted(test_seeds),
        "original_unchanged": True,
        "candidate_unchanged_during_evaluation": True,
        "changed_state_keys": changed,
        "fit": fit,
        "reconstruction": recon_summary,
        "response_comparison": comparison,
        "response_summaries": summaries,
        "code_usage": usage,
        "notes": [
            "Only content embed buffers change in a disposable model copy. EMA statistics are unchanged and candidate is evaluation-only.",
            "Lloyd iterations start from original vectors and retain empty entries; selected by training distortion only.",
            "Training sites are uniform across the whole native map, not chosen using anatomy or interventions.",
            "Fixed-reference normalization is estimated on training subjects only and shared by both held-out conditions.",
            "All deltas are recalibrated minus original; confidence intervals are paired subject bootstraps conditional on the fitted codebook.",
            "Better quantizer responses can coexist with worse reconstruction; no automatic adoption decision.",
            "No codebook improvement in this conservative refit does not rule out other codebook sizes or initializations.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    write_csv(output / "reconstruction.csv", recon_rows)
    write_csv(output / "response_comparison.csv", comparison)
    for condition, rows in response_rows.items():
        write_csv(output / f"{condition}_responses.csv", rows)
    np.savez_compressed(output / "diagnostic_centers.npz", **vectors)
    print("\nTraining-only refit")
    for name, info in fit.items():
        chosen = info["trace"][info["selected_iteration"]]
        print(
            f"{name}: MSE {info['trace'][0]['training_mse']:.6g} -> {chosen['training_mse']:.6g}; "
            f"{info['entries_moved']} entries moved; selected iteration {info['selected_iteration']}"
        )
    print("\nHeld-out reconstruction (lower is better; delta = recalibrated - original)")
    for row in recon_summary:
        print(
            f"{row['view']:6s} {row['metric']:4s} {row['original_mean']:.6g} -> {row['recalibrated_mean']:.6g}; "
            f"delta {row['mean_delta']:+.4g} [{row['mean_delta_ci_low']:+.4g}, {row['mean_delta_ci_high']:+.4g}]"
        )
    print("\nHeld-out affected-bin content responses (paired means)")
    for row in comparison:
        if row["block"] == "content" and row["region"] == "affected_bins" and row["n"]:
            print(
                f"eps={row['eps']:g} {row['view']:6s} {row['metric']:34s} "
                f"{row['original_mean']:.4g} -> {row['recalibrated_mean']:.4g} (n={row['n']})"
            )
    print(
        f"\nSaved {output}\nOriginal checkpoint untouched; recalibrated vectors are diagnostic artifacts, not a deployable checkpoint."
    )


if __name__ == "__main__":
    main()
