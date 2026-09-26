"""Frozen retrieval, stage probes, and train-only BatchNorm recalibration.

For encoder-only runs from training.main_conv_synthetic. See
ENCODER_GENERALIZATION_AUDIT.md for the split protocol and interpretation.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, Subset

from eval.checkpoint_lesion_analysis import json_safe, state_digest
from eval.dci import CONTENT_FACTOR_NAMES
from eval.lesion_probe import block_gram
from eval.pooling_probe import fit_readouts
from eval.score_checkpoint import build_model, load_settings, make_dataset

VIEWS = ("t1", "flair")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="model.pt", help="Filename within the run, or absolute path")
    p.add_argument("--num-samples", type=int, default=400, help="Subjects in EACH retrieval cohort: train and test")
    p.add_argument("--probe-samples", type=int, default=400, help="Validation subjects; 75%% probe fit, 25%% tuning")
    p.add_argument("--batch-size", type=int, default=4, help="Encoding subjects per view; independent of negative pool")
    p.add_argument("--retrieval-batch-size", type=int, help="Negative-pool size; default: training batch size")
    p.add_argument("--retrieval-draws", type=int, default=8)
    p.add_argument("--bn-samples", type=int, default=512, help="Maximum original training subjects for calibration")
    p.add_argument("--bn-batch-size", type=int, help="Calibration subjects per view; default: training batch size")
    p.add_argument("--seed", type=int, default=1729, help="Audit sampling, probe split and shuffled controls")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--out-dir", help="New output directory; default: a timestamped folder inside the run")
    args = p.parse_args(argv)
    if args.num_samples < 10 or args.probe_samples < 20:
        p.error("Need --num-samples >= 10 and --probe-samples >= 20 (400 each recommended)")
    for name in ("batch_size", "retrieval_draws", "bn_samples"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("retrieval_batch_size", "bn_batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 2:
            p.error(f"--{name.replace('_', '-')} must be at least 2")
    return args


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def capture_batch(model, x):
    """Capture actual global endpoints from one forward, in T1-then-FLAIR order."""
    if any(module.training for module in model.modules()):
        raise ValueError("Feature extraction requires every module in evaluation mode")
    captured = {}

    def tap(name):
        def hook(module, inputs, output):
            captured[name] = output

        return hook

    with ExitStack() as stack:
        modules = {"encoder": model.encoder, "encoder_v1": model.encoder_v1}
        if model.encoder_architecture == "resnet18":
            modules["hidden"] = model.to_encoding[1]  # Actual post-LeakyReLU readout.
        for name, module in modules.items():
            if module is not None:
                stack.callback(module.register_forward_hook(tap(name)).remove)
        content = model(x, pool_only=True, n_views=2)[2][0][:, : model.content_channels]
    backbone = captured["encoder"]
    if model.encoder_v1 is not None:
        backbone = torch.cat((backbone, captured["encoder_v1"]), dim=0)
    values = {"backbone": backbone.mean((2, 3, 4))}
    if "hidden" in captured:
        values["hidden"] = captured["hidden"]
    values.update(
        content=content,
        content_l2=F.normalize(content, dim=1, eps=1e-8),
        content_norm=content.norm(dim=1, keepdim=True),
        loss_space=model.project(content),
    )
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Non-finite encoder features")
    return {name: value.cpu().numpy() for name, value in values.items()}


def extract(model, dataset, device, batch_size, label):
    before = state_digest(model)
    chunks, ids, targets = {}, [], []
    image_digest = hashlib.sha256()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    interval = max(1, len(loader) // 10)
    for step, batch in enumerate(loader, 1):
        images = torch.cat(batch["image"], dim=0)
        image_digest.update(images.contiguous().numpy().tobytes())
        values = capture_batch(model, images.to(device))
        b = len(batch["index"])
        for stage, value in values.items():
            for view, block in zip(VIEWS, (value[:b], value[b:])):
                chunks.setdefault((view, stage), []).append(block)
        ids.extend(batch["index"].tolist())
        targets.append(batch["gt_latents"]["z_content"].numpy())
        if step % interval == 0 or step == len(loader):
            print(f"  {label}: encoded {len(ids)}/{len(dataset)} subjects", flush=True)
    if state_digest(model) != before:
        raise RuntimeError("Model state changed during evaluation-mode extraction")
    return {
        "features": {key: np.concatenate(value) for key, value in chunks.items()},
        "ids": np.asarray(ids),
        "targets": np.concatenate(targets),
        "input_sha256": image_digest.hexdigest(),
        "model_sha256": before,
    }


def effective_rank(features):
    x = np.asarray(features, dtype=np.float64)
    singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
    spectrum = singular**2
    return float(spectrum.sum() ** 2 / np.square(spectrum).sum()) if spectrum.sum() > 0 else 0.0


def retrieval_metrics(t1, flair, tau, cross_view_negs_only=True):
    """Cross-view retrieval and the exact two-view training loss reduction.

    Cross-only training sums directional mean CEs; the all-negatives branch
    averages them. Retrieval always ranks ONLY the other view's candidates.
    Ties receive their expected rank/accuracy under random tie-breaking.
    """
    t1, flair = (np.asarray(x, dtype=np.float64) for x in (t1, flair))
    if t1.ndim != 2 or t1.shape != flair.shape or len(t1) < 2:
        raise ValueError("Retrieval requires matching [subjects, features] arrays and at least two subjects")
    if not np.isfinite(tau) or tau <= 0 or not all(np.isfinite(x).all() for x in (t1, flair)):
        raise ValueError("Need finite features and a positive finite temperature")
    a, b = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8) for x in (t1, flair))
    sim = a @ b.T
    n = len(a)
    diagonal = np.diag(sim)
    result = {}
    ces = []
    for direction, scores, own in (("t1_to_flair", sim, a), ("flair_to_t1", sim.T, b)):
        logits = scores / tau
        if not cross_view_negs_only:
            within = own @ own.T / tau
            np.fill_diagonal(within, -np.inf)
            logits = np.concatenate((logits, within), axis=1)
        ce = float(np.mean(logsumexp(logits, axis=1) - diagonal / tau))
        ces.append(ce)
        greater = (scores > diagonal[:, None]).sum(1)
        equal = (scores == diagonal[:, None]).sum(1)
        harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, n + 1))]
        result[direction] = {
            "top1": float(np.clip((1 - greater) / equal, 0, 1).mean()),
            "top5": float(np.clip((min(5, n) - greater) / equal, 0, 1).mean()),
            "mrr": float(((harmonic[greater + equal] - harmonic[greater]) / equal).mean()),
            "mean_rank": float((greater + (equal + 1) / 2).mean()),
            "directional_ce": ce,
            "positive_cosine": float(diagonal.mean()),
            "negative_cosine": float(scores[~np.eye(n, dtype=bool)].mean()),
            "chance_top1": 1.0 / n,
            "retrieval_candidates": n,
            "loss_candidates": n if cross_view_negs_only else 2 * n - 1,
        }
    training_loss = sum(ces) if cross_view_negs_only else float(np.mean(ces))
    for row in result.values():
        row["training_loss"] = training_loss
    return result


def retrieval_rows(bank, arm, split, cfg, batch_size, draws, seed):
    a, b = (bank["features"][(view, "loss_space")] for view in VIEWS)
    rng = np.random.default_rng(seed)
    pools = [("full", -1, np.arange(len(a)))]
    pools += [("training_batch", draw, rng.choice(len(a), batch_size, replace=False)) for draw in range(draws)]
    rows = []
    for pool, draw, indices in pools:
        scores = retrieval_metrics(a[indices], b[indices], cfg.get("tau", 0.1), cfg.get("cross_view_negs_only", True))
        for direction, score in scores.items():
            rows.append({"arm": arm, "split": split, "pool": pool, "draw": draw, "direction": direction, **score})
    return rows


@torch.no_grad()
def recalibrate_batchnorm(model, training_dataset, device, batch_size, max_samples, seed):
    """Reset/re-estimate only BN buffers in a disposable copy; no labels or gradients."""
    if batch_size < 2 or max_samples < 1:
        raise ValueError("Need at least two subjects per calibration batch and a positive sample budget")
    original_digest = state_digest(model)
    candidate = copy.deepcopy(model).eval()
    layers = {
        name: module
        for name, module in candidate.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if not layers:
        return None, {"status": "not_applicable", "reason": "This architecture has no BatchNorm layers"}
    if any(not module.track_running_stats for module in layers.values()):
        raise ValueError("BatchNorm recalibration requires tracked running statistics")
    requested = min(max_samples, len(training_dataset))
    n = requested // batch_size * batch_size
    if n < batch_size:
        raise ValueError("Need at least one full calibration batch; reduce --bn-batch-size or increase --bn-samples")
    indices = np.random.default_rng(seed).permutation(len(training_dataset))[:n]
    parameters_before = tensor_digest(candidate.named_parameters())
    allowed_buffers = {
        f"{name}.{key}" for name in layers for key in ("running_mean", "running_var", "num_batches_tracked")
    }
    fixed_before = tensor_digest(
        (name, value) for name, value in candidate.named_buffers() if name not in allowed_buffers
    )
    old = {
        name: (module.running_mean.clone(), module.running_var.clone(), module.momentum)
        for name, module in layers.items()
    }
    for module in layers.values():
        module.reset_running_stats()
        module.momentum = None  # Equal-sized batches, cumulative average from a fresh start.
        module.train()
    loader = DataLoader(Subset(training_dataset, indices.tolist()), batch_size=batch_size, shuffle=False)
    for step, batch in enumerate(loader, 1):
        x = torch.cat(batch["image"], dim=0).to(device)
        candidate(x, pool_only=True, n_views=2)
        print(f"  BatchNorm calibration: {step * batch_size}/{n} training subjects", flush=True)
    changes = {}
    for name, module in layers.items():
        mean, var, momentum = old[name]
        if not torch.isfinite(module.running_mean).all() or not torch.isfinite(module.running_var).all():
            raise RuntimeError(f"Non-finite recalibrated BatchNorm buffers: {name}")
        changes[name] = {
            "running_mean_delta_rms": float((module.running_mean - mean).square().mean().sqrt()),
            "running_var_delta_rms": float((module.running_var - var).square().mean().sqrt()),
            "batches": int(module.num_batches_tracked),
        }
        module.momentum = momentum
    candidate.eval()
    if parameters_before != tensor_digest(candidate.named_parameters()):
        raise RuntimeError("Recalibration changed model parameters")
    if fixed_before != tensor_digest(
        (name, value) for name, value in candidate.named_buffers() if name not in allowed_buffers
    ):
        raise RuntimeError("Recalibration changed non-BatchNorm buffers")
    if original_digest != state_digest(model):
        raise RuntimeError("Recalibration changed the original model")
    return candidate, {
        "status": "ok",
        "subject_split": "train",
        "subject_ids": indices.tolist(),
        "requested_samples": max_samples,
        "used_samples": n,
        "batch_size_per_view": batch_size,
        "method": "reset running buffers, one pass, cumulative average of equal-sized batches",
        "parameters_unchanged": True,
        "non_bn_buffers_unchanged": True,
        "layers": changes,
    }


def probe_rows(banks, arm, names, seed):
    """Fit/tune on validation subjects; score ONLY on the separate test split."""
    val, test = banks["val"], banks["test"]
    order = np.random.default_rng(seed).permutation(len(val["ids"]))
    cut = int(0.75 * len(order))
    splits = (order[:cut], order[cut:], np.arange(len(val["ids"]), len(val["ids"]) + len(test["ids"])))
    y = np.concatenate((val["targets"], test["targets"]))
    rows, predictions = [], {"test_truth": test["targets"], "test_ids": test["ids"]}
    for (view, stage), feature in val["features"].items():
        if stage == "loss_space":
            continue
        print(f"  Probes {arm}: {view} {stage} ({feature.shape[1]} dimensions)", flush=True)
        x = np.concatenate((feature, test["features"][(view, stage)]))
        with threadpool_limits(limits=1):
            gram, width = block_gram(x, np.arange(x.shape[1]), splits[0])
            scores, predicted = fit_readouts(gram, width, y, splits, seed, target_names=names)
        rows.extend({"arm": arm, "view": view, "stage": stage, "dims": x.shape[1], **row} for row in scores)
        predictions.update({f"{view}__{stage}__{key}": value for key, value in predicted.items()})
    return (
        rows,
        predictions,
        {
            "fit_validation_ids": val["ids"][splits[0]].tolist(),
            "tune_validation_ids": val["ids"][splits[1]].tolist(),
            "test_ids": test["ids"].tolist(),
        },
    )


def save_banks(banks, path):
    arrays = {}
    for split, bank in banks.items():
        arrays[f"{split}__ids"] = bank["ids"]
        arrays[f"{split}__targets"] = bank["targets"]
        for (view, stage), value in bank["features"].items():
            arrays[f"{split}__{view}__{stage}"] = value
    np.savez_compressed(path, **arrays)


def write_report(report, directory):
    (directory / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    for name in ("retrieval", "probes", "ranks"):
        rows = report[name]
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(json_safe(rows))


def print_summary(report):
    print("\nRetrieval in evaluation mode: T1->FLAIR / FLAIR->T1 top-1; InfoNCE uses the training reduction")
    print("arm               split  full-pool top1     batch-pool top1    batch InfoNCE")
    for arm in report["arms"]:
        for split in ("train", "test"):
            rows = [r for r in report["retrieval"] if r["arm"] == arm and r["split"] == split]
            full = [r["top1"] for r in rows if r["pool"] == "full"]
            batch = [
                np.mean([r["top1"] for r in rows if r["pool"] == "training_batch" and r["direction"] == d])
                for d in ("t1_to_flair", "flair_to_t1")
            ]
            loss = np.mean([r["training_loss"] for r in rows if r["pool"] == "training_batch"])
            print(
                f"{arm:<18} {split:<5}  {full[0]:.3f} / {full[1]:.3f}     {batch[0]:.3f} / {batch[1]:.3f}      {loss:.4f}"
            )
    print("\nHeld-out factor recovery: ridge / RBF R² (validation-selected, refitted for each arm)")
    print("arm               view   stage          brain size       ventricle        sulcal           mean factors")
    for arm in report["arms"]:
        for view in VIEWS:
            for stage in report["probe_stages"]:
                rows = [
                    r
                    for r in report["probes"]
                    if (r["arm"], r["view"], r["stage"], r["condition"]) == (arm, view, stage, "observed")
                ]
                values = []
                for target in ("brain_size", "ventricle_size", "sulcal_widening", None):
                    score = []
                    for kind in ("ridge", "rbf"):
                        selected = [
                            r["test_r2"]
                            for r in rows
                            if r["probe"] == kind and (target is None or r["target"] == target)
                        ]
                        score.append(float(np.mean(selected)) if selected else float("nan"))
                    values.append(f"{score[0]:+.3f}/{score[1]:+.3f}")
                print(f"{arm:<18} {view:<6} {stage:<14} " + "   ".join(values))
    print("Per-factor scores, shuffled controls, stage changes and recalibration changes are in probes.csv.")
    print(f"BatchNorm: {report['batchnorm']['status']}. Original model parameters and buffers unchanged.", flush=True)


def run_audit(model, cfg, cli, device, directory):
    if cfg.get("contrastive_loss_type", "infonce") != "infonce":
        raise ValueError("This audit matches InfoNCE runs; the selected run used a different loss")
    original_state = state_digest(model)
    n_train = int(cfg.get("num_train_samples", 2000))
    if cli.num_samples > n_train:
        raise ValueError(f"--num-samples cannot exceed the {n_train} original training subjects")
    retrieval_batch = cli.retrieval_batch_size or int(cfg.get("batch_size", 64))
    bn_batch = cli.bn_batch_size or int(cfg.get("batch_size", 64))
    if not 2 <= retrieval_batch <= cli.num_samples:
        raise ValueError("Retrieval batch size must be between 2 and --num-samples; it is never silently reduced")
    tau = cfg.get("tau", 0.1)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Saved InfoNCE temperature must be positive and finite")
    bn_layers = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if bn_layers and (bn_batch < 2 or min(cli.bn_samples, n_train) < bn_batch):
        raise ValueError("Need at least one full BatchNorm batch; check --bn-samples and --bn-batch-size")
    if any(not module.track_running_stats for module in bn_layers):
        raise ValueError("BatchNorm recalibration requires tracked running statistics")
    training = make_dataset(cfg, n_train, mode="train")
    ids = np.random.default_rng(cli.seed).permutation(n_train)[: cli.num_samples]
    datasets = {
        "train": Subset(training, ids.tolist()),
        "val": make_dataset(cfg, cli.probe_samples, mode="val"),
        "test": make_dataset(cfg, cli.num_samples, mode="test"),
    }
    names = [
        CONTENT_FACTOR_NAMES[j] if j < len(CONTENT_FACTOR_NAMES) else f"content_{j}" for j in range(cfg["n_content"])
    ]
    report = {
        "status": "running",
        "settings": cfg,
        "options": vars(cli),
        "arms": [],
        "target_names": names,
        "retrieval": [],
        "probes": [],
        "ranks": [],
        "state": {},
        "retrieval_protocol": {
            "batch_size": retrieval_batch,
            "draws": cli.retrieval_draws,
            "seed": cli.seed,
            "space": "model.project(final content), matching the training objective",
        },
        "notes": [
            "Training retrieval uses only original training indices; test uses the generator's independent test seed.",
            "Readouts fit on 75% of validation subjects, tune on the other 25%, and score on separate test subjects.",
            "The run may have used validation subjects to select checkpoints; the final test cohort is separate.",
            "All encoder extraction and retrieval use eval mode, including the recalibrated arm.",
            "Only a disposable copy's BatchNorm running buffers are updated, using unlabeled training images.",
            "Hyperparameters, shuffles, subject subsets and negative pools are paired across arms and stages.",
            "Negative-pool loss may differ from training logs because of eval mode and a different subject draw.",
            "A causal graph can allow factor prediction from correlated anatomy; recovery is not localized sensitivity.",
            "Ridge/RBF failure is not proof of information absence. High effective rank is not factor identifiability.",
            "No encoder weights are trained, and no modified model checkpoint is written.",
        ],
    }
    baseline_banks = None

    def evaluate_arm(arm, candidate):
        nonlocal baseline_banks
        report["arms"].append(arm)
        banks = {
            split: extract(candidate, ds, device, cli.batch_size, f"{arm}/{split}") for split, ds in datasets.items()
        }
        if baseline_banks is not None:
            for split, bank in banks.items():
                reference = baseline_banks[split]
                if (
                    bank["input_sha256"] != reference["input_sha256"]
                    or not np.array_equal(bank["ids"], reference["ids"])
                    or not np.array_equal(bank["targets"], reference["targets"])
                ):
                    raise RuntimeError("Recalibrated and original arms received different subjects/images/targets")
        save_banks(banks, directory / f"{arm}_features.npz")
        report["state"][arm] = {
            split: {key: value for key, value in bank.items() if key not in ("features", "targets", "ids")}
            for split, bank in banks.items()
        }
        report["cohorts"] = {split: bank["ids"].tolist() for split, bank in banks.items()}
        for split in ("train", "test"):
            report["retrieval"].extend(
                retrieval_rows(banks[split], arm, split, cfg, retrieval_batch, cli.retrieval_draws, cli.seed)
            )
            for (view, stage), features in banks[split]["features"].items():
                report["ranks"].append(
                    {
                        "arm": arm,
                        "split": split,
                        "view": view,
                        "stage": stage,
                        "dims": features.shape[1],
                        "effective_rank": effective_rank(features),
                        "mean_feature_std": float(np.std(features.astype(np.float64), axis=0).mean()),
                    }
                )
        report["probe_stages"] = [
            stage for view, stage in banks["val"]["features"] if view == "t1" and stage != "loss_space"
        ]
        # Persist encoded features and retrieval even if a later probe fails.
        write_report(report, directory)
        scores, predicted, split_info = probe_rows(banks, arm, names, cli.seed)
        report["probes"].extend(scores)
        report["probe_split"] = split_info
        np.savez_compressed(directory / f"{arm}_predictions.npz", **predicted)
        write_report(report, directory)
        if baseline_banks is None:
            baseline_banks = {
                split: {key: value for key, value in bank.items() if key != "features"} for split, bank in banks.items()
            }

    evaluate_arm("original", model)
    candidate, report["batchnorm"] = recalibrate_batchnorm(
        model, training, device, bn_batch, cli.bn_samples, cli.seed + 1
    )
    if candidate is not None:
        evaluate_arm("bn_recalibrated", candidate)
        del candidate
    by_key = {(r["arm"], r["view"], r["stage"], r["probe"], r["condition"], r["target"]): r for r in report["probes"]}
    for row in report["probes"]:
        suffix = (row["probe"], row["condition"], row["target"])
        original = by_key[("original", row["view"], row["stage"], *suffix)]
        backbone = by_key[(row["arm"], row["view"], "backbone", *suffix)]
        row["delta_original"] = row["test_r2"] - original["test_r2"]
        row["delta_backbone"] = row["test_r2"] - backbone["test_r2"]
    if state_digest(model) != original_state:
        raise RuntimeError("Audit changed the original model state")
    report["original_state_unchanged"] = True
    report["status"] = "complete"
    return report


def main(argv=None):
    cli = parse_args(argv)
    cfg = load_settings(cli.run_dir)
    device = "cuda" if torch.cuda.is_available() and not cli.no_cuda else "cpu"
    checkpoint = Path(cli.run_dir) / cli.checkpoint
    # One immutable in-memory snapshot, even if training later overwrites model.pt.
    checkpoint_bytes = checkpoint.read_bytes()
    snapshot_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    del checkpoint_bytes
    model = build_model(cfg, device, state)
    del state
    directory = (
        Path(cli.out_dir) if cli.out_dir else Path(cli.run_dir) / f"encoder_audit_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Checkpoint: {checkpoint}\nDevice: {device}\nOutput: {directory}", flush=True)
    snapshot = {"checkpoint": str(checkpoint), "checkpoint_sha256": snapshot_digest, "settings": cfg}
    (directory / "checkpoint_source.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    with threadpool_limits(limits=1):
        report = run_audit(model, cfg, cli, device, directory)
    report.update({key: value for key, value in snapshot.items() if key != "settings"})
    write_report(report, directory)
    print_summary(report)
    print(f"Saved {directory}", flush=True)


if __name__ == "__main__":
    main()
