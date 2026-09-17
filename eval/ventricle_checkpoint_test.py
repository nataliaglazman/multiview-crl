"""Frozen-checkpoint ventricular probes and local GAP-loss interventions.

Training directions use images/masks only, never anatomical targets. Labels are used
only to fit diagnostic readouts and score held-out subjects. This is not retraining,
an AdamW update, or evidence about historical causation. See VENTRICLE_CHECKPOINT_TEST.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from eval.gradient_attribution import _encoder_params, _flat
from eval.reconstruction_attribution import (
    _content_features,
    _gradient,
    _inputs,
    _validate_run,
    frozen_checkpoint,
    sim_weights,
    similarity_loss,
    temporary_step,
)
from eval.ventricle_routing import decode_swaps, make_dataset, render_pair, score_swaps, stable_replay_math

logger = logging.getLogger(__name__)
VIEWS = ("t1", "flair")


def gap_correlation(hz):
    """Same biased standardization/epsilon as barlow_twins_loss, on (2,B,C)."""
    if hz.ndim != 3 or hz.shape[0] != 2 or hz.shape[1] < 2:
        raise ValueError("GAP correlation needs (2, B>=2, C) features.")
    a, b = hz.float()
    a = (a - a.mean(0)) / (a.std(0, unbiased=False) + 1e-6)
    b = (b - b.mean(0)) / (b.std(0, unbiased=False) + 1e-6)
    return a.T @ b / a.shape[0]


def gap_redundancy(hz, normalize, decay=0.0, reference=None):
    """Off-diagonal term with a stationary, detached EMA reference.

    The historical EMA is not in the checkpoint. reference is estimated at the frozen
    checkpoint, then FIXED across gradients and trials. m*reference+(1-m)*C reproduces
    the settled EMA's current-batch derivative; it does not replay historical training.
    """
    c = gap_correlation(hz)
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1).")
    if decay:
        if reference is None or reference.shape != c.shape:
            raise ValueError("EMA needs a matching frozen correlation reference.")
        c = decay * reference.detach().to(c) + (1 - decay) * c
    off = c.square().sum() - c.diagonal().square().sum()
    d = c.shape[0]
    return off / max(d * (d - 1), 1) if normalize else off


def gap_weights(args):
    scale = float(getattr(args, "scale_contrastive_loss", 1.0))
    levels = getattr(args, "contrastive_level_weights", None)
    if levels is not None:
        scale *= float(levels[0])
    lam = getattr(args, "bt_gap_lambda", None)
    lam = getattr(args, "bt_lambda", 0.005) if lam is None else lam
    return {
        "gap_redundancy": scale * float(getattr(args, "bt_gap_weight", 0)) * float(lam),
        "gap_mse": sim_weights(args, 0)["gap_sim"],
    }


def gap_losses(hz, args, reference):
    return {
        "gap_redundancy": gap_redundancy(
            hz,
            bool(getattr(args, "bt_normalize_terms", False)),
            float(getattr(args, "bt_corr_ema", 0)),
            reference,
        ),
        "gap_mse": similarity_loss(hz, bool(getattr(args, "bt_sim_normalize", False))),
    }


def collect_directions(model, loader, args, device, grid, params):
    """Two encoder passes: frozen reference, then weighted gradients per batch."""
    import torch

    correlations = []
    with torch.no_grad():
        for batch in loader:
            hz = _content_features(model, batch, args, device, grid, 0).mean(-1)
            correlations.append(gap_correlation(hz))
    reference = torch.stack(correlations).mean(0).detach()
    weights = gap_weights(args)
    gradients = {k: [] for k in weights}
    values = {k: [] for k in weights}
    for i, batch in enumerate(loader):
        hz = _content_features(model, batch, args, device, grid, 0).mean(-1)
        for key, loss in gap_losses(hz, args, reference).items():
            values[key].append(float(loss.detach()))
            gradients[key].append(_gradient(weights[key] * loss, params).cpu())
        logger.info("GAP gradients: batch %d/%d", i + 1, len(loader))
    stacks = {k: torch.stack(v) for k, v in gradients.items()}
    means = {k: v.mean(0).to(device) for k, v in stacks.items()}
    for key, g in means.items():
        if not bool(torch.isfinite(g).all()):
            raise ValueError(f"Non-finite gradient for {key}.")
    stats = {}
    for key, stack in stacks.items():
        mean = means[key].cpu()
        norms = stack.norm(dim=1)
        denom = norms * mean.norm()
        cos = (stack @ mean) / denom.clamp_min(1e-30)
        stats[key] = {
            "coefficient": weights[key],
            "unweighted_loss": float(np.mean(values[key])),
            "weighted_gradient_norm": float(mean.norm()),
            "batch_gradient_norms": norms.tolist(),
            "batch_cosines_to_mean": [float(c) if d > 0 else None for c, d in zip(cos, denom)],
        }
    return means, reference, stats


def measure_gap_losses(model, loader, args, device, grid, reference):
    import torch

    values = {k: [] for k in gap_weights(args)}
    with torch.no_grad():
        for batch in loader:
            hz = _content_features(model, batch, args, device, grid, 0).mean(-1)
            for key, loss in gap_losses(hz, args, reference).items():
                values[key].append(float(loss))
    return {k: float(np.mean(v)) for k, v in values.items()}


def decoder_features(model, loader, device, probe_grid):
    """Pool ACTUAL quantized decoder inputs; never re-embed IDs or slice embedding axes.

    The content embedding width can differ from the selected encoder channel count.
    All of the content embedding is probed. Style is the actual post-quantization
    decoder-bound tensor. This implementation explicitly supports one VQ level only.
    """
    import torch
    import torch.nn.functional as F

    features, targets, ids = {}, [], []
    for batch in loader:
        captured, handles = {}, []

        def hook(key):
            def capture(module, inputs, output):
                if key in captured:
                    raise ValueError("A content codebook ran twice; decoder input mapping is ambiguous.")
                captured[key] = output[0].detach().clone()

            return capture

        split = bool(getattr(model, "separate_content_codebooks", False))
        try:
            handles.append(model.codebooks[0].register_forward_hook(hook(0)))
            if split:
                handles.append(model.codebooks_v1[0].register_forward_hook(hook(1)))
            x, mask = _inputs(batch, device)
            with torch.no_grad():
                out = model(x, return_recon=True, pool_only=True, n_views=2, subsets=[(0, 1)], mask=mask)
        finally:
            for handle in handles:
                handle.remove()
        q = torch.cat([captured[0], captured[1]]) if split else captured[0]
        style = model._last_style_spatials.get(0)
        if style is None:
            raise ValueError("No decoder-bound style tensor at level 0.")
        for block, tensor in (("content", q), ("style", style)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Non-finite {block} decoder input.")
            for pool, size in (("gap", 1), ("spatial", probe_grid)):
                if size > min(tensor.shape[2:]):
                    # Global style bottlenecks must not be upsampled into duplicate features.
                    size = min(tensor.shape[2:])
                pooled = F.adaptive_avg_pool3d(tensor.float(), size).flatten(1)
                if len(pooled) != len(x):
                    raise ValueError("Decoder inputs must retain the view-major batch.")
                for view, name in enumerate(VIEWS):
                    key = f"{name}/{block}/{pool}"
                    features.setdefault(key, []).append(pooled.chunk(2)[view].detach().cpu().numpy())
        if len(out[5]) != 1 or out[5][0] is None:
            raise ValueError("Missing level-0 code IDs.")
        ids.append(out[5][0].detach().cpu().numpy().reshape(2, len(x) // 2, -1))
        y = batch["gt_latents"]["z_content"][:, 1].numpy()
        targets.append(y)
    return (
        {k: np.concatenate(v) for k, v in features.items()},
        np.concatenate(targets),
        np.concatenate(ids, axis=1),
    )


def fit_probes(features, y, seed=0):
    """Tune ridge/RBF readouts on fit subjects ONLY, including all preprocessing."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probes, choices = {}, {}
    cv = KFold(3, shuffle=True, random_state=seed)
    for key, X in features.items():
        if not np.isfinite(X).all() or not np.isfinite(y).all() or np.var(y) <= 1e-12:
            raise ValueError("Probes need finite features and a varying ventricular target.")
        for name, estimator, grid in (
            ("ridge", Ridge(), {"regressor__ridge__alpha": [0.01, 1.0, 100.0]}),
            (
                "rbf",
                KernelRidge(kernel="rbf"),
                {
                    "regressor__kernelridge__alpha": [0.01, 1.0],
                    "regressor__kernelridge__gamma": [v / X.shape[1] for v in (0.1, 1.0, 10.0)],
                },
            ),
        ):
            model = TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), estimator), transformer=StandardScaler()
            )
            search = GridSearchCV(model, grid, cv=cv, scoring="r2", n_jobs=1, error_score="raise")
            search.fit(X, y)
            pkey = f"{key}/{name}"
            probes[pkey] = search.best_estimator_
            choices[pkey] = {"params": search.best_params_, "fit_cv_r2": float(search.best_score_)}
        logger.info("Fit diagnostic ridge/RBF probes: %s", key)
    return probes, choices


def predict_probes(probes, features, fit_features=None, fit_y=None):
    """Optionally refit fixed baseline hyperparameters, never tune on test subjects."""
    from sklearn.base import clone

    predictions = {}
    for key, model in probes.items():
        feature_key = key.rsplit("/", 1)[0]
        if fit_features is not None:
            model = clone(model).fit(fit_features[feature_key], fit_y)
        predictions[key] = model.predict(features[feature_key])
    return predictions


def r2(y, prediction):
    den = float(np.sum((y - np.mean(y)) ** 2))
    return float(1 - np.sum((y - prediction) ** 2) / den) if den > 1e-12 else float("nan")


def paired_delta(y, baseline, prediction, seed=0, draws=500):
    """Paired subject bootstrap, conditional on the fitted probes and chosen direction."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), size=(draws, len(y)))
    ys = y[indices]
    den = ((ys - ys.mean(1, keepdims=True)) ** 2).sum(1)
    num = ((ys - baseline[indices]) ** 2 - (ys - prediction[indices]) ** 2).sum(1)
    valid = den > 1e-12
    bounds = np.quantile(num[valid] / den[valid], [0.025, 0.975]) if valid.any() else [np.nan, np.nan]
    return {
        "r2": r2(y, prediction),
        "delta_r2": r2(y, prediction) - r2(y, baseline),
        "delta_r2_ci_low": float(bounds[0]),
        "delta_r2_ci_high": float(bounds[1]),
    }


def routing_rows(model, pairs, device, batch_size):
    rows = []
    for start in range(0, len(pairs), batch_size):
        samples = pairs[start : start + batch_size]
        decoded, diagnostics = decode_swaps(model, samples, device)
        for i, sample in enumerate(samples):
            for view, modality in enumerate(VIEWS):
                row = {"index": sample["index"], "modality": modality}
                row.update(
                    score_swaps(
                        sample["a"][view].numpy()[0],
                        sample["b"][view].numpy()[0],
                        {k: value[i] for k, value in decoded[view].items()},
                        sample["support"],
                        sample["mask"].numpy()[0],
                    )
                )
                row.update({k: float(value[i]) for k, value in diagnostics[view].items()})
                row["valid_routing"] = row["valid_input"] and bool(row["endpoint_signal_resolved"])
                rows.append(row)
    return rows


def routing_delta(baseline, current):
    """Use the SAME resolved subjects before/after; never compare changing populations."""
    result = {}
    metrics = ("content_mean_gain", "style_mean_gain", "joint_gain", "joint_relative_error")
    base = {(r["index"], r["modality"]): r for r in baseline}
    for view in VIEWS:
        matched = [(base[r["index"], view], r) for r in current if r["modality"] == view]
        valid = [(a, b) for a, b in matched if a["valid_routing"] and b["valid_routing"]]
        result[view] = {"n_common_valid": len(valid)}
        for metric in metrics:
            before = [a[metric] for a, _ in valid]
            after = [b[metric] for _, b in valid]
            result[view][metric] = {
                "baseline_median": float(np.median(before)) if valid else None,
                "trial_median": float(np.median(after)) if valid else None,
                "median_paired_delta": float(np.median(np.subtract(after, before))) if valid else None,
            }
    return result


def validate(args):
    _validate_run(args, 0)
    if int(getattr(args, "vqvae_nb_levels", 1)) != 1:
        raise ValueError("This diagnostic supports one VQ level; multilevel decoder inputs can mix pathways.")
    if getattr(args, "mask_mode", None) != "fixed":
        raise ValueError("Use a fixed content/style split; changing masks would confound encoder attribution.")
    if not getattr(args, "inject_style_to_decoder", False):
        raise ValueError("An injected style pathway is required for the content/style comparison.")
    if getattr(args, "split_encoder_norm", False):
        raise ValueError("The shared eval loader does not restore split_encoder_norm; refusing a mismatched model.")
    weights = gap_weights(args)
    if not any(weights.values()) or any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Need finite, nonnegative GAP coefficients and at least one active GAP term.")


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_results(directory, report, records, predictions):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "summary.json").open("w") as handle:
        json.dump(clean_json(report), handle, indent=2, allow_nan=False)
    if records:
        with (directory / "probe_deltas.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(directory / "predictions.npz", **predictions)


def run(cli):
    import torch
    from torch.utils.data import DataLoader, Subset

    from eval.run_dci_synthetic import load_model_from_run_dir, load_run_args

    args = load_run_args(cli.run_dir)
    validate(args)
    checkpoint = Path(cli.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(cli.run_dir) / checkpoint
    model, args, device = load_model_from_run_dir(cli.run_dir, str(checkpoint.resolve()), cli.device, seed=cli.seed)
    # The shared loader warns on missing weights; this diagnostic must fail instead.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("encoders", state)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    del state
    grid = getattr(args, "patch_grid_per_level", None) or getattr(args, "patch_grid", None)
    if not grid:
        raise ValueError("The run has no patch grid.")
    batch_size = cli.grad_batch_size or int(args.batch_size)
    if batch_size < 2:
        raise ValueError("Gradient batch size must be >= 2.")
    if batch_size != int(args.batch_size):
        logger.warning(
            "Gradient batch %d differs from training %d; correlation estimates change.", batch_size, args.batch_size
        )
    n_grad = cli.grad_batches * batch_size
    grad_ds = make_dataset(args, n_grad, "match", "val")
    n_probe = cli.fit_samples + cli.test_samples
    probe_ds = make_dataset(args, n_probe + cli.routing_samples, cli.probe_causal, "test")
    if cli.cache_images:
        grad_ds._cache = [None] * len(grad_ds)
        probe_ds._cache = [None] * len(probe_ds)
    grad_loader = DataLoader(grad_ds, batch_size=batch_size, shuffle=False)
    # Keep fit/test batch boundaries separate so the null replay uses identical shapes,
    # even when fit_samples is not divisible by encode_batch (CUDA kernels can differ).
    probe_batches = [
        list(range(start, min(start + cli.encode_batch, end)))
        for begin, end in ((0, cli.fit_samples), (cli.fit_samples, n_probe))
        for start in range(begin, end, cli.encode_batch)
    ]
    probe_loader = DataLoader(probe_ds, batch_sampler=probe_batches)
    replay_loader = DataLoader(
        Subset(probe_ds, range(cli.fit_samples, n_probe)), batch_size=cli.encode_batch, shuffle=False
    )
    pairs = [render_pair(probe_ds, n_probe + i, cli.eps) for i in range(cli.routing_samples)]
    directory = (
        Path(cli.out) if cli.out else Path(cli.run_dir) / datetime.now().strftime("ventricle_checkpoint_%Y%m%d_%H%M%S")
    )
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "cli": vars(cli),
        "settings": vars(args),
        "render_resolution": probe_ds.res,
        "gradient_distribution": "matched SCM, val split",
        "probe_distribution": f"{cli.probe_causal}, test split",
        "partitions": {
            "gradient": n_grad,
            "probe_fit": cli.fit_samples,
            "probe_test": cli.test_samples,
            "routing": cli.routing_samples,
        },
        "step_convention": "theta' = theta - relative_step * ||theta|| * g / ||g||; encoder only; no AdamW",
        "ema_reference": "mean correlation of the frozen gradient batches; stationary surrogate, not historical state",
        "limitations": [
            "Local directions, not historical causation or retraining predictions.",
            "Probe failure is not proof of absent information. Pooled probes omit spatial detail.",
            "Refitted probes reuse baseline hyperparameters; CIs condition on fit data/direction.",
            "Zero quantized response can mean no code boundary was crossed.",
            "IID evaluation can be outside the training distribution.",
        ],
    }
    records, predictions = [], {}
    with stable_replay_math(), frozen_checkpoint(model) as restore:
        params = [p for _, p in _encoder_params(model)]
        theta_norm = float(_flat(params).norm())
        logger.info("Extracting baseline decoder-bound features on %d subjects", n_probe)
        baseline, y, baseline_ids = decoder_features(model, probe_loader, device, cli.probe_grid)
        n = cli.fit_samples
        fit = {k: X[:n] for k, X in baseline.items()}
        test = {k: X[n:] for k, X in baseline.items()}
        probes, report["probe_selection"] = fit_probes(fit, y[:n], cli.seed)
        base_predictions = predict_probes(probes, test)
        report["baseline_r2"] = {k: r2(y[n:], p) for k, p in base_predictions.items()}
        report["feature_widths"] = {k: X.shape[1] for k, X in baseline.items()}
        predictions.update({f"baseline/{k}": p for k, p in base_predictions.items()})
        predictions["test_targets"] = y[n:]
        baseline_routing = routing_rows(model, pairs, device, cli.routing_batch)
        report["baseline_routing_rows"] = baseline_routing
        report["baseline_routing"] = routing_delta(baseline_routing, baseline_routing)
        # A no-step replay detects numerical drift before attributing small changes.
        replay, replay_y, replay_ids = decoder_features(model, replay_loader, device, cli.probe_grid)
        np.testing.assert_array_equal(replay_y, y[n:])
        report["null_replay_max_abs"] = {k: float(np.max(np.abs(replay[k] - test[k]))) for k in test}
        replay_predictions = predict_probes(probes, replay)
        report["null_replay_delta_r2"] = {
            k: r2(y[n:], p) - report["baseline_r2"][k] for k, p in replay_predictions.items()
        }
        for key in test:
            np.testing.assert_allclose(replay[key], test[key], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(replay_ids, baseline_ids[:, n:])
        directions, reference, report["gradients"] = collect_directions(model, grad_loader, args, device, grid, params)
        report["trials"] = []
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "baseline_features.npz", targets=y, **baseline)
        write_results(directory, report, records, predictions)
        print("\nBaseline ventricular R² (held-out subjects; actual decoder-bound tensors):", flush=True)
        for key, score in report["baseline_r2"].items():
            print(f"  {key:<30} {score:+.4f}", flush=True)
        for name, gradient in directions.items():
            norm = float(gradient.norm())
            if norm == 0:
                logger.warning("%s has zero weighted gradient; skipping its steps.", name)
                continue
            for relative in cli.relative_steps:
                restore()
                eta = relative * theta_norm / norm
                temporary_step(params, gradient, eta)
                tag = f"{name}/{relative:g}"
                logger.info("Temporary %s step: relative norm %.3g", name, relative)
                losses = measure_gap_losses(model, grad_loader, args, device, grid, reference)
                trial, trial_y, trial_ids = decoder_features(model, probe_loader, device, cli.probe_grid)
                np.testing.assert_array_equal(trial_y, y)
                trial_test = {k: X[n:] for k, X in trial.items()}
                frozen_predictions = predict_probes(probes, trial_test)
                refit_predictions = predict_probes(probes, trial_test, {k: X[:n] for k, X in trial.items()}, y[:n])
                trial_report = {
                    "direction": name,
                    "relative_step": relative,
                    "equivalent_raw_gradient_eta": eta,
                    "unweighted_losses_after": losses,
                    "source_loss_delta": losses[name] - report["gradients"][name]["unweighted_loss"],
                    "source_loss_decreased": losses[name] < report["gradients"][name]["unweighted_loss"],
                    "code_change_fraction": {
                        v: float(np.mean(trial_ids[i, n:] != baseline_ids[i, n:])) for i, v in enumerate(VIEWS)
                    },
                }
                for mode, predicted in (("fixed", frozen_predictions), ("refit", refit_predictions)):
                    for key, p in predicted.items():
                        predictions[f"{tag}/{mode}/{key}"] = p
                        row = {
                            "direction": name,
                            "relative_step": relative,
                            "probe_mode": mode,
                            "probe": key,
                            "baseline_r2": report["baseline_r2"][key],
                        }
                        row.update(paired_delta(y[n:], base_predictions[key], p, cli.seed))
                        records.append(row)
                        if mode == "refit" and "/content/" in key:
                            print(
                                f"  {tag} {key}: ΔR² {row['delta_r2']:+.4f} "
                                f"[{row['delta_r2_ci_low']:+.4f}, {row['delta_r2_ci_high']:+.4f}]",
                                flush=True,
                            )
                rows = routing_rows(model, pairs, device, cli.routing_batch)
                trial_report["routing_rows"] = rows
                trial_report["routing"] = routing_delta(baseline_routing, rows)
                if trial_report["source_loss_delta"] >= 0:
                    logger.warning("%s did not reduce its source loss; this step is inconclusive.", tag)
                report["trials"].append(trial_report)
                write_results(directory, report, records, predictions)
        restore()
        restored, restored_y, restored_ids = decoder_features(model, replay_loader, device, cli.probe_grid)
        np.testing.assert_array_equal(restored_y, y[n:])
        np.testing.assert_array_equal(restored_ids, baseline_ids[:, n:])
        for key in test:
            np.testing.assert_allclose(restored[key], test[key], rtol=1e-6, atol=1e-6)
        report["restoration_verified"] = True
    write_results(directory, report, records, predictions)
    print(
        f"\nSaved {directory}\nPositive ΔR² = better recovery. Read joint routing fidelity before pathway gains."
        "\nParameters and buffers restored; no checkpoint or optimizer was written.",
        flush=True,
    )
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="vqvae_model.pt")
    ap.add_argument("--fit-samples", type=int, default=256)
    ap.add_argument("--test-samples", type=int, default=128)
    ap.add_argument(
        "--probe-grid", type=int, default=4, help="Readout grid, independent of the exact training loss grid."
    )
    ap.add_argument("--probe-causal", choices=("iid", "match"), default="iid")
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument("--grad-batches", type=int, default=4)
    ap.add_argument("--grad-batch-size", type=int, default=0, help="0 uses training batch size.")
    ap.add_argument("--relative-steps", type=float, nargs="+", default=[1e-5, 1e-4])
    ap.add_argument("--routing-samples", type=int, default=16, help="0 skips the optional decoder-routing checks.")
    ap.add_argument("--routing-batch", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-images", action="store_true", help="Cache rendered images in RAM; can require several GB.")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args(argv)
    if min(cli.fit_samples, cli.test_samples) < 12:
        ap.error("Need at least 12 fit and 12 held-out subjects.")
    if min(cli.encode_batch, cli.grad_batches, cli.probe_grid, cli.routing_batch) < 1 or cli.routing_samples < 0:
        ap.error("Batch/grid sizes must be positive and routing-samples nonnegative.")
    if cli.grad_batch_size < 0 or not np.isfinite(cli.eps) or cli.eps <= 0:
        ap.error("Invalid gradient batch size or intervention epsilon.")
    if any(not np.isfinite(s) or s <= 0 or s > 0.01 for s in cli.relative_steps):
        ap.error("Relative steps must be finite and in (0, 0.01].")
    # Fix small dense linear algebra thread counts to avoid oversubscribing ridge/RBF fits.
    from threadpoolctl import threadpool_limits

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with threadpool_limits(limits=2):
        run(cli)


if __name__ == "__main__":
    main()
