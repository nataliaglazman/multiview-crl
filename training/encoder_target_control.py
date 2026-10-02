"""Supervised observability control for lesions and signed sulcal corrugation.

Fresh per-view small CNN; no pretrained encoder checkpoint is changed or loaded.
Train only on the original training cohort. Use a fixed step budget, log validation,
then score once on independent test subjects. This is not unsupervised CRL.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eval.diagnostics.pooling_probe import r2
from eval.encoder.encoder_target_protocol import (
    TARGETS,
    VIEWS,
    dataset,
    dataset_metadata,
    provenance,
    sample_targets,
    save_csv,
    save_report,
)
from eval.protocol.score_checkpoint import load_settings
from utils.encoder_runtime import configure_encoder_runtime, select_encoder_device

REGRESSION_NAMES = ("lesion_x", "lesion_y", "lesion_z", "sulcal_widening", "sulcal_amplitude")
REGRESSION_INDICES = [TARGETS.index(name) for name in REGRESSION_NAMES]


class TargetControl(nn.Module):
    """Full-volume image -> stride-2 lesion map and spatial (non-GAP) factor readout."""

    def __init__(self, width=24, grid=8):
        super().__init__()
        if width < 4 or width % 4 or grid < 1:
            raise ValueError("Width must be a positive multiple of four (>=4), grid positive")
        layers = []
        for incoming, stride in ((1, 1), (width, 2), (width, 1), (width, 1)):
            layers.extend((nn.Conv3d(incoming, width, 3, stride=stride, padding=1), nn.GroupNorm(4, width), nn.SiLU()))
        self.features = nn.Sequential(*layers)
        self.lesion = nn.Conv3d(width, 1, 1)
        self.regression = nn.Sequential(
            nn.Conv3d(width, 4, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(grid),
            nn.Flatten(),
            nn.Linear(4 * grid**3, 128),
            nn.SiLU(),
            nn.Linear(128, len(REGRESSION_NAMES)),
        )
        self.grid = grid

    def forward(self, images):
        features = self.features(images)
        if min(features.shape[2:]) < self.grid:
            raise ValueError("Readout grid exceeds actual spatial map")
        return self.lesion(features)[:, 0], self.regression(features)


def heatmap_centroid(logits, resolution):
    """Softmax expectation in renderer coordinates; bins match avg_pool3d targets."""
    if len(set(logits.shape[1:])) != 1 or resolution % logits.shape[1]:
        raise ValueError("Need cubic heatmaps whose size divides input resolution")
    stride = resolution // logits.shape[1]
    axis = (torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype) + 0.5) * stride - 0.5
    axis = 2 * axis / (resolution - 1) - 1
    coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return logits.flatten(1).softmax(1) @ coordinates


def render_bank(cfg, split, count, view, directory):
    """Float32 cache in a NEW output directory; never reuse an unverified cache."""
    directory.mkdir(parents=True)
    ds = dataset(cfg, count, split)
    res = cfg["res"]
    images = np.lib.format.open_memmap(
        directory / "images.npy", mode="w+", dtype="float32", shape=(count, 1, res, res, res)
    )
    heatmaps = np.lib.format.open_memmap(
        directory / "heatmaps.npy", mode="w+", dtype="float32", shape=(count, res // 2, res // 2, res // 2)
    )
    targets, h, ids = [], hashlib.sha256(), []
    for idx in range(count):
        item = ds[idx]
        pair = torch.stack(item["image"])
        h.update(pair.contiguous().numpy().tobytes())
        images[idx] = item["image"][VIEWS.index(view)].numpy()
        target, support = sample_targets(ds._inner, item["gt_latents"])
        pooled = F.avg_pool3d(support[None, None], kernel_size=2)[0, 0]
        heatmaps[idx] = (pooled / pooled.sum()).numpy()
        targets.append(target)
        ids.append(int(item["index"]))
        if (idx + 1) % 100 == 0 or idx + 1 == count:
            print(f"  Rendered {split}: {idx + 1}/{count}", flush=True)
    images.flush()
    heatmaps.flush()
    targets = np.asarray(targets)
    np.save(directory / "targets.npy", targets)
    return {
        "images": images,
        "heatmaps": heatmaps,
        "targets": targets,
        "metadata": dataset_metadata(ds, h.hexdigest(), ids),
    }


def target_scaler(training_targets):
    values = training_targets[:, REGRESSION_INDICES]
    mean, std = values.mean(0), values.std(0)
    if not np.isfinite(values).all() or np.any(std < 1e-8):
        raise ValueError("Training targets must be finite and nonconstant")
    return mean.astype(np.float32), std.astype(np.float32)


@torch.inference_mode()
def predict(model, bank, device, batch_size, resolution, scaler):
    model.eval()
    centroids, regression = [], []
    for start in range(0, len(bank["images"]), batch_size):
        images = torch.from_numpy(np.array(bank["images"][start : start + batch_size])).to(device)
        logits, estimates = model(images)
        centroids.append(heatmap_centroid(logits, resolution).cpu().numpy())
        regression.append(estimates.cpu().numpy() * scaler[1] + scaler[0])
    return {"centroid": np.concatenate(centroids), "regression": np.concatenate(regression)}


def score_predictions(predictions, truth, training_targets, resolution, radius):
    rows = []
    estimates = predictions["regression"]
    targets = truth[:, REGRESSION_INDICES]
    baseline = np.broadcast_to(training_targets[:, REGRESSION_INDICES].mean(0), targets.shape)
    for j, name in enumerate(REGRESSION_NAMES):
        rows.append(
            {
                "target": name,
                "r2": float(r2(targets[:, j : j + 1], estimates[:, j : j + 1])[0]),
                "rmse": float(np.sqrt(np.mean((targets[:, j] - estimates[:, j]) ** 2))),
                "train_mean_baseline_r2": float(r2(targets[:, j : j + 1], baseline[:, j : j + 1])[0]),
            }
        )
    actual = truth[:, 9:12]
    pred = predictions["centroid"]
    base = np.broadcast_to(training_targets[:, 9:12].mean(0), actual.shape)
    for j, axis in enumerate("xyz"):
        rows.append(
            {
                "target": f"centroid_{axis}",
                "r2": float(r2(actual[:, j : j + 1], pred[:, j : j + 1])[0]),
                "rmse": float(np.sqrt(np.mean((actual[:, j] - pred[:, j]) ** 2))),
                "train_mean_baseline_r2": float(r2(actual[:, j : j + 1], base[:, j : j + 1])[0]),
            }
        )
    # No second magnitude head: test whether the signed prediction also recovers depth.
    mag = np.abs(estimates[:, -1:])
    mag_true = truth[:, -1:]
    rows.append(
        {
            "target": "sulcal_magnitude",
            "r2": float(r2(mag_true, mag)[0]),
            "rmse": float(np.sqrt(np.mean((mag_true - mag) ** 2))),
            "train_mean_baseline_r2": float(r2(mag_true, np.full_like(mag_true, training_targets[:, -1].mean()))[0]),
        }
    )
    error = np.linalg.norm(pred - actual, axis=1) * (resolution - 1) / 2
    sign_mask = np.abs(truth[:, -2]) > 1e-8
    return {
        "factors": rows,
        "median_centroid_error_vox": float(np.median(error)),
        "mean_centroid_error_vox": float(np.mean(error)),
        "within_one_lesion_radius": float(np.mean(error <= radius * (resolution - 1) / 2)),
        "sulcal_sign_accuracy": float(np.mean(np.sign(estimates[sign_mask, -1]) == np.sign(truth[sign_mask, -2]))),
    }


def train(model, banks, args, device, scaler, directory, report):
    # This function deliberately never reads banks['test'].
    training = banks["train"]
    mean, std = (torch.as_tensor(x, device=device) for x in scaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    rng = np.random.default_rng(args.seed + 1)
    permutation = (
        np.random.default_rng(args.seed + 2).permutation(len(training["images"]))
        if args.shuffle_targets
        else np.arange(len(training["images"]))
    )
    report["training_target_order"] = permutation.tolist()
    report["history"] = []
    t0 = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.choice(len(training["images"]), args.batch_size, replace=False)
        label_ids = permutation[ids]
        images = torch.from_numpy(np.array(training["images"][ids])).to(device)
        heat = torch.from_numpy(np.array(training["heatmaps"][label_ids])).to(device)
        labels = torch.from_numpy(training["targets"][label_ids][:, REGRESSION_INDICES].copy()).to(device)
        logits, estimates = model(images)
        location_loss = -(heat.flatten(1) * logits.flatten(1).log_softmax(1)).sum(1).mean()
        regression_loss = F.mse_loss(estimates, (labels - mean) / std)
        loss = args.lesion_weight * location_loss + args.regression_weight * regression_loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps:
            print(
                f"[{args.view}] step {step}/{args.steps}: heatmap={location_loss.item():.4f}, standardized regression={regression_loss.item():.4f}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            predictions = predict(model, banks["val"], device, args.batch_size, args.resolution, scaler)
            scores = score_predictions(predictions, banks["val"]["targets"], training["targets"], args.resolution, 0.1)
            report["history"].append({"step": step, "validation": scores, "loss": float(loss.item())})
            report["completed_steps"] = step
            save_report(directory, report)
            sulcal = next(row["r2"] for row in scores["factors"] if row["target"] == "sulcal_amplitude")
            print(
                f"  validation: lesion median={scores['median_centroid_error_vox']:.3f} vox; signed sulcal R2={sulcal:.3f}",
                flush=True,
            )
    report["training_seconds"] = time.monotonic() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Supplies original data settings only")
    p.add_argument("--out-dir", type=Path, required=True, help="New directory; existing directories are refused")
    p.add_argument("--view", choices=VIEWS, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--test-samples", type=int, default=400)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lesion-weight", type=float, default=1.0)
    p.add_argument("--regression-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--shuffle-targets", action="store_true", help="Optional negative control; permute TRAINING target rows only"
    )
    args = p.parse_args(argv)
    cfg = load_settings(args.run_dir)
    args.resolution = cfg["res"]
    if min(args.steps, args.batch_size, args.test_samples, args.eval_every, args.log_every, args.grid) < 1:
        p.error("Step, sample, batch, and grid counts must be positive")
    if cfg["res"] % 2 or args.grid > cfg["res"] // 2 or args.batch_size > cfg["num_train_samples"]:
        p.error("Need even resolution, grid <= res/2, and batch <= training cohort")
    if (
        args.width < 4
        or args.width % 4
        or not np.isfinite([args.lr, args.lesion_weight, args.regression_weight]).all()
        or min(args.lr, args.lesion_weight, args.regression_weight) <= 0
    ):
        p.error("Need width divisible by four and finite positive learning rate/loss weights")
    if min(cfg["num_train_samples"], cfg["num_val_samples"], args.test_samples) < 2:
        p.error("Each split needs at least two subjects")
    device = select_encoder_device(args.device)
    configure_encoder_runtime(cfg)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, device)
    report.update(
        protocol="Supervised from scratch, fixed final step, training-only target scaling, independent final test",
        selection="final_step",
        view=args.view,
        regression_targets=list(REGRESSION_NAMES),
    )
    save_report(args.out_dir, report)
    try:
        banks = {
            split: render_bank(cfg, split, count, args.view, args.out_dir / "data" / split)
            for split, count in (
                ("train", cfg["num_train_samples"]),
                ("val", cfg["num_val_samples"]),
                ("test", args.test_samples),
            )
        }
        report["cohorts"] = {split: bank["metadata"] for split, bank in banks.items()}
        scaler = target_scaler(banks["train"]["targets"])
        report["target_scaler"] = {"mean": scaler[0].tolist(), "std": scaler[1].tolist(), "fit_split": "train"}
        # Rendering can change global RNGs. Seed AFTER all data creation.
        torch.manual_seed(args.seed)
        model = TargetControl(args.width, args.grid).to(device)
        report["parameters"] = sum(p.numel() for p in model.parameters())
        train(model, banks, args, device, scaler, args.out_dir, report)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "arguments": report["arguments"],
                "target_scaler": report["target_scaler"],
            },
            args.out_dir / "model.pt",
        )
        predictions = predict(model, banks["test"], device, args.batch_size, cfg["res"], scaler)
        report["test"] = score_predictions(
            predictions, banks["test"]["targets"], banks["train"]["targets"], cfg["res"], 0.1
        )
        np.savez_compressed(args.out_dir / "test_predictions.npz", truth=banks["test"]["targets"], **predictions)
        save_csv(args.out_dir / "test_scores.csv", report["test"]["factors"])
        report["status"] = "complete"
        save_report(args.out_dir, report)
        print(f"Completed {args.view} supervised control: {args.out_dir}", flush=True)
        for row in report["test"]["factors"]:
            print(f"  {row['target']:20s} test R2={row['r2']:+.3f}")
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise


if __name__ == "__main__":
    main()
