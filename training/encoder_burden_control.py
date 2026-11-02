"""Supervised rendered-volume control through a reference encoder's content vector.

Uses a fresh instance of the reference architecture and two linear scalar heads,
one per modality. Only the exported content units reach the heads. There is no
contrastive, patch, reconstruction or spatially supervised loss. Success tests
capacity/observability; failure of a finite supervised run is not an information
theoretic ceiling. Reference checkpoints are never modified or fine-tuned.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.cluster_batches import bank_latents
from eval.encoder.encoder_target_protocol import dataset
from eval.encoder.regional_lesion_detection import lesion_fractions
from eval.encoder.spatial_recovery_monitor import preserve_training_state
from eval.protocol.score_checkpoint import build_model, load_settings
from training import encoder_args
from training.main_conv_synthetic import stratified_batch_sampler
from utils.encoder_runtime import select_encoder_device


class BurdenControl(nn.Module):
    """Original paired encoder -> exported content vector -> per-view linear scalar."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            self.heads = nn.ModuleList(nn.Linear(encoder.content_channels, 1) for _ in range(2))

    def forward(self, images):
        content = self.encoder(images, pool_only=True, n_views=2)[2][0][:, : self.encoder.content_channels]
        return torch.cat([head(rows) for head, rows in zip(self.heads, content.chunk(2))], dim=0)[:, 0]


def burden_table(subjects):
    """Actual soft lesion load in native voxel units, plus the original latent for rank diagnostics."""
    volume = lesion_fractions(subjects, (1, 1, 1))[:, 0] * subjects.res**3
    latent = bank_latents(subjects)[:, 2]
    values = np.column_stack((volume, latent)).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Rendered burden targets must be finite")
    return values


def paired_images(batch, device):
    return torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)


def metrics(prediction, table):
    truth = table[:, 0]
    variance = float(np.square(truth - truth.mean()).sum())
    return {
        "volume_r2": float(1 - np.square(prediction - truth).sum() / variance) if variance > 0 else None,
        "volume_rmse_voxels": float(np.sqrt(np.square(prediction - truth).mean())),
        "volume_spearman": (
            None
            if np.ptp(truth) == 0
            else float(spearmanr(prediction, truth).statistic)
            if np.ptp(prediction) > 0
            else 0.0
        ),
        "latent_spearman": (
            None
            if np.ptp(table[:, 1]) == 0
            else float(spearmanr(prediction, table[:, 1]).statistic)
            if np.ptp(prediction) > 0
            else 0.0
        ),
    }


@torch.no_grad()
def evaluate(model, subjects, table, scaler, device, batch_size):
    predicted = [[], []]
    with preserve_training_state(model, device):
        loader = DataLoader(subjects, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(0))
        for batch in loader:
            rows = model(paired_images(batch, device)).cpu().numpy() * scaler[1] + scaler[0]
            for view, part in enumerate(np.split(rows, 2)):
                predicted[view].append(part)
    return {view: metrics(np.concatenate(predicted[i]), table) for i, view in enumerate(("t1", "flair"))}


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", required=True, type=Path, help="Reference run with settings.json; no trained weights loaded"
    )
    p.add_argument("--out-dir", required=True, type=Path, help="New directory; existing outputs are refused")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--steps", type=int, default=None, help="Default: reference train_steps")
    p.add_argument("--eval-every", type=int, default=None, help="Default: reference eval_every")
    p.add_argument("--test-samples", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true", help="Validate settings and print plan without rendering/training")
    args = p.parse_args(argv)
    if args.test_samples < 2 or any(x is not None and x < 1 for x in (args.steps, args.eval_every)):
        p.error("Need positive steps/eval-every and at least two test subjects")
    return args


def run(options):
    cfg = load_settings(options.run_dir)
    if cfg.get("synthetic_lesion_target") != "burden" or cfg.get("dataset_name", "synthetic") != "synthetic":
        raise ValueError("Reference must be a synthetic lesion-burden run")
    if cfg.get("mask_mode", "fixed") != "fixed" or cfg.get("lesion_keypoints", 0) != 0:
        raise ValueError("This control requires fixed content units and no lesion-keypoint branch")
    if cfg.get("vector_prediction_weight", 0) != 0:
        raise ValueError("Use the original contrastive reference, without a vector-prediction decoder")
    if cfg.get("stratify_batches", "none") not in ("none", "oracle"):
        raise ValueError("Only uniform or oracle reference batching is supported; no external code sampler")
    destinations = {a.dest for a in encoder_args.build_parser()._actions}
    args = encoder_args.parse_args(encoder_args.cli_arguments({k: v for k, v in cfg.items() if k in destinations}))
    steps = options.steps or args.train_steps
    every = options.eval_every or args.eval_every
    plan = {
        "objective": "supervised rendered-volume MSE only, two linear heads on exported content units",
        "reference": str(options.run_dir.resolve()),
        "reference_settings_sha256": hashlib.sha256((options.run_dir / "settings.json").read_bytes()).hexdigest(),
        "minimum_radius_voxels": args.synthetic_lesion_min_radius_vox,
        "content_channels": args.content_channels,
        "batch_size": args.batch_size,
        "steps": steps,
        "eval_every": every,
        "test_samples": options.test_samples,
        "device": options.device,
        "initialization": "fresh reference architecture and seed; trained reference weights are not loaded",
        "normalization": "fixed_reference constants from training subjects reused in validation and test",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if options.dry_run:
        return plan
    device = select_encoder_device(options.device)
    options.out_dir.mkdir(parents=True, exist_ok=False)
    model = BurdenControl(build_model(cfg, device)).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(model.encoder.state_dict(), options.out_dir / "model_init.pt")
    train = dataset(cfg, args.num_train_samples, "train")
    validation = dataset(cfg, args.num_val_samples, "val")
    # Keep the training input normalization, and freeze it for all held-out images.
    train[0]
    reference_normalization = (train._fixed_mean, train._fixed_scale)
    if cfg.get("synthetic_normalize") == "fixed_reference":
        validation._fixed_mean, validation._fixed_scale = reference_normalization
    tables = {"train": burden_table(train), "val": burden_table(validation)}
    scaler = (float(tables["train"][:, 0].mean()), float(tables["train"][:, 0].std()))
    if scaler[1] < 1e-6:
        raise ValueError("Training subjects need varying rendered lesion volume")
    args.train_steps = steps
    sampler = (
        stratified_batch_sampler(args, train, device, options.out_dir) if args.stratify_batches != "none" else None
    )
    loader_options = dict(num_workers=0, generator=torch.Generator().manual_seed(args.loader_seed))
    loader = (
        DataLoader(train, batch_sampler=sampler, **loader_options)
        if sampler is not None
        else DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_options)
    )
    if not len(loader):
        raise ValueError("Training loader has no complete batches")
    settings = {
        **cfg,
        "model_id": options.out_dir.name,
        "out_dir": str(options.out_dir.parent),
        "device": device,
        "train_steps": steps,
        "supervised_burden_control": plan,
    }
    write_report(options.out_dir / "settings.json", settings)
    report = {**plan, "status": "running", "step": 0, "target_mean_sd": scaler, "history": []}
    report["validation_floor"] = evaluate(model, validation, tables["val"], scaler, device, args.batch_size)
    write_report(options.out_dir / "report.json", report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch_hash, input_hash = hashlib.sha256(), hashlib.sha256()
    step = 0
    while step < steps:
        model.train()
        for batch in loader:
            if step >= steps:
                break
            indices = batch["index"].numpy()
            batch_hash.update(indices.astype("<i8").tobytes())
            images = paired_images(batch, device)
            if args.hash_training_inputs:
                input_hash.update(torch.cat(batch["image"]).as_subclass(torch.Tensor).contiguous().numpy().tobytes())
            target = torch.as_tensor((tables["train"][indices, 0] - scaler[0]) / scaler[1], device=device).repeat(2)
            prediction = model(images)
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite supervised burden loss at step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % args.log_every == 0 or step == steps:
                print(f"step {step:6d} | supervised volume MSE {loss.item():.4f}", flush=True)
            if step % every == 0 or step == steps:
                scores = evaluate(model, validation, tables["val"], scaler, device, args.batch_size)
                report["history"].append({"step": step, "validation": scores})
                report.update(
                    step=step, batch_order_sha256=batch_hash.hexdigest(), training_input_sha256=input_hash.hexdigest()
                )
                write_report(options.out_dir / "report.json", report)
                torch.save(model.encoder.state_dict(), options.out_dir / "model.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "target_mean_sd": scaler,
                    },
                    options.out_dir / "control.pt",
                )
                print(
                    f"  validation volume R2: T1 {scores['t1']['volume_r2']:.3f}, FLAIR {scores['flair']['volume_r2']:.3f}",
                    flush=True,
                )
    test = dataset(cfg, options.test_samples, "test")
    if cfg.get("synthetic_normalize") == "fixed_reference":
        test._fixed_mean, test._fixed_scale = reference_normalization
    test_table = burden_table(test)
    report["test"] = evaluate(model, test, test_table, scaler, device, args.batch_size)
    model.load_state_dict(initial)
    report["test_floor"] = evaluate(model, test, test_table, scaler, device, args.batch_size)
    report["input_normalization_mean_scale"] = reference_normalization
    report["status"] = "complete"
    write_report(options.out_dir / "report.json", report)
    print("Final held-out test: " + json.dumps(report["test"]), flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
