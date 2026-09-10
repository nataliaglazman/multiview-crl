"""Fine-tune a shared DINO backbone on the two synthetic MRI views.

    python -m training.finetune_dino --output-dir results/dino_infonce --epochs 20

Each subject contributes one feature per modality after pooling its selected planes.
The corresponding subject in the other modality is the positive; the other B-1
subjects in that modality are negatives for InfoNCE. Barlow Twins instead aligns
the cross-correlation diagonal and penalizes off-diagonal correlations. Only
content channels (75% by default) enter either loss or the optional MLP projector.
Style channels are excluded from alignment; shared backbone updates can still
change them. This exclusion alone does not enforce style informativeness.

Uses the generator's TRAIN split; the existing embedding script defaults to TEST.
The output includes an HF-loadable encoder, projection/optimizer state, the generator
settings, fixed preprocessing, and JSONL training metrics. Checkpoints are replaced
after each complete epoch. Evaluation uses the backbone before the projection head.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from contextlib import nullcontext

from eval import dinov3_embed_synthetic as embed
from models.dino_partition import make_partition, split_embeddings
from utils.config import parse_dino_finetune_args

logger = logging.getLogger(__name__)


def symmetric_infonce(view1, view2, temperature=0.1):
    """Mean of cross-view InfoNCE in both directions, with cosine similarities."""
    import torch
    import torch.nn.functional as F

    if view1.ndim != 2 or view1.shape != view2.shape or view1.shape[0] < 2:
        raise ValueError("InfoNCE requires matching (B, D) tensors with at least two subjects")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Float32 keeps normalization and softmax stable under mixed precision.
    similarity = F.normalize(view1.float(), dim=-1) @ F.normalize(view2.float(), dim=-1).T
    logits = similarity / temperature
    labels = torch.arange(len(view1), device=view1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    with torch.no_grad():
        positive = similarity.diag()
        metrics = {
            "loss": loss.item(),
            "accuracy": (
                (logits.argmax(1) == labels).float().mean() + (logits.argmax(0) == labels).float().mean()
            ).item()
            / 2,
            "positive_similarity": positive.mean().item(),
            "negative_similarity": ((similarity.sum() - positive.sum()) / (len(view1) * (len(view1) - 1))).item(),
        }
    return loss, metrics


def barlow_twins(view1, view2, lambd=0.0051, eps=1e-5):
    """Batch-normalized cross-correlation objective, using population variance.

    Objective follows https://github.com/facebookresearch/barlowtwins .
    Correlation statistics use this process's actual batch, in float32.
    """
    import torch

    if view1.ndim != 2 or view1.shape != view2.shape or min(view1.shape) < 1 or len(view1) < 2:
        raise ValueError("Barlow Twins requires matching (B, D) tensors with B >= 2 and D >= 1")
    if not math.isfinite(lambd) or lambd < 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Barlow lambda must be nonnegative and eps positive, both finite")
    normalized = []
    for view in (view1, view2):
        centered = view.float() - view.float().mean(0)
        normalized.append(centered / (centered.square().mean(0) + eps).sqrt())
    corr = normalized[0].T @ normalized[1] / len(view1)
    diagonal = corr.diagonal()
    on_diag = (diagonal - 1).square().sum()
    off_diag = corr[~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)].square().sum()
    loss = on_diag + lambd * off_diag
    # Retrieval is only a diagnostic for BT; it does not contribute to its objective.
    with torch.no_grad():
        _, metrics = symmetric_infonce(view1, view2)
        metrics.update(
            loss=loss.item(),
            barlow_on_diag=on_diag.item(),
            barlow_off_diag=off_diag.item(),
            barlow_diag_mean=diagonal.mean().item(),
        )
    return loss, metrics


def alignment_loss(content1, content2, projector, cli):
    """The projector receives content only; style has no direct alignment gradient."""
    first, second = projector(content1), projector(content2)
    # Disable autocast for correlation / similarity matrix products.
    import torch

    with torch.autocast(device_type=first.device.type, enabled=False):
        if cli.loss == "barlow_twins":
            return barlow_twins(first, second, cli.barlow_lambda, cli.barlow_eps)
        return symmetric_infonce(first, second, cli.temperature)


def partition_diagnostics(parts):
    import torch
    import torch.nn.functional as F

    metrics = {}
    with torch.no_grad():
        for index, name in enumerate(("content", "style")):
            first, second = [p[index].float() for p in parts]
            if first.shape[1]:
                metrics[f"{name}_std"] = (
                    (first.std(0, unbiased=False).mean() + second.std(0, unbiased=False).mean()) / 2
                ).item()
                metrics[f"{name}_paired_cosine"] = F.cosine_similarity(first, second, dim=-1).mean().item()
    return metrics


def encode_volumes(volumes, encoder, config, cli, device, window, mean, std):
    """Differentiable counterpart of the evaluation script's extraction and aggregation."""
    if getattr(cli, "backbone", "dinov3") == "3dino":
        from models.three_dino import encode_volumes as encode_3dino

        return encode_3dino(volumes, encoder, config, cli, device, window)

    import torch

    patch = cli.patch_size or embed.patch_size_of(config)
    if cli.image_size % patch:
        raise ValueError(f"--image-size {cli.image_size} must be a multiple of patch size {patch}")
    grid = (cli.image_size // patch,) * 2
    planes = [
        plane
        for volume in volumes.detach().cpu().squeeze(1).numpy()
        for plane in embed.volume_planes(volume, cli.axes, cli.slices)
    ]
    features = []
    for start in range(0, len(planes), cli.plane_batch_size):
        pixels = embed.prepare_batch(planes[start : start + cli.plane_batch_size], window, cli, mean, std)
        hidden = encoder(pixel_values=pixels.to(device)).last_hidden_state
        prefix = embed.token_prefix(hidden.shape[1], grid[0] * grid[1], config)
        features.append(embed.pool_tokens(hidden.float(), prefix, grid, cli.token_pool, cli.grid_size))
    per_slot = torch.cat(features).reshape(len(volumes), len(cli.axes) * cli.slices, -1)
    return per_slot.mean(1) if cli.slice_agg == "mean" else per_slot.flatten(1)


def make_projector(feature_dim, cli):
    import torch.nn as nn

    if cli.projection_dim == 0:
        return nn.Identity()
    return nn.Sequential(
        nn.Linear(feature_dim, cli.projection_hidden_dim),
        nn.GELU(),
        nn.Linear(cli.projection_hidden_dim, cli.projection_dim),
    )


def train_epoch(loader, encoder, projector, optimizer, scaler, config, cli, device, window, mean, std, partition):
    import torch

    encoder.train()
    projector.train()
    totals, count = {}, 0
    parameters = list(encoder.parameters()) + list(projector.parameters())
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(cli.dtype)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        views = batch["image"]
        context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with context:
            parts = [
                split_embeddings(encode_volumes(view, encoder, config, cli, device, window, mean, std), partition)
                for view in views
            ]
            loss, metrics = alignment_loss(parts[0][0], parts[1][0], projector, cli)
        metrics.update(partition_diagnostics(parts))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {cli.loss} loss; no optimizer update was applied")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, cli.grad_clip or float("inf"), error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(views[0])
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    if count == 0:
        raise ValueError("No contrastive batches: need at least --batch-size training subjects")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(output_dir, encoder, projector, optimizer, scaler, epoch, cli, partition):
    import torch

    encoder.save_pretrained(output_dir / "encoder")
    (output_dir / "encoder" / "embedding_partition.json").write_text(json.dumps(partition, indent=2) + "\n")
    path = output_dir / "training_state.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "embedding_partition": partition,
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": {key: str(value) if key == "output_dir" else value for key, value in vars(cli).items()},
        },
        temporary,
    )
    temporary.replace(path)


def main(argv=None):
    cli = parse_dino_finetune_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if cli.output_dir.exists() and (not cli.output_dir.is_dir() or any(cli.output_dir.iterdir())):
        raise ValueError("--output-dir must be new or empty to avoid overwriting a previous run")

    import numpy as np
    import torch

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    # Keep trainable weights and AdamW state in float32; --dtype controls autocast.
    encoder_cli = argparse.Namespace(**vars(cli))
    encoder_cli.dtype = "float32"
    encoder, device, _, config = embed.load_encoder(encoder_cli)
    if cli.dtype != "float32" and device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use --dtype float32 on CPU/MPS")
    if cli.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; use float32 or float16")
    encoder.requires_grad_(True)
    if cli.gradient_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dataset, _, settings = embed.build_dataset(cli, split="train")
    mean, std, norm_source = embed.resolve_normalization(cli)
    window = embed.estimate_window(dataset, cli) if cli.window == "dataset" else (0.0, 0.0)
    if cli.window == "per_slice":
        logger.warning("Per-slice windowing removes affine intensity style before the encoder sees it")
    # Infer width through the same pooling path, without retaining an initialization graph.
    with torch.no_grad():
        example = dataset[0]["image"][0].unsqueeze(0)
        width = encode_volumes(example, encoder, config, cli, device, window, mean, std).shape[1]
    partition = make_partition(width, config.hidden_size, cli)
    projector = make_projector(partition["content_dim"], cli).to(device)
    groups = [{"params": encoder.parameters(), "lr": cli.lr}]
    if cli.projection_dim:
        groups.append({"params": projector.parameters(), "lr": cli.head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cli.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cli.dtype == "float16")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cli.num_workers,
        generator=torch.Generator().manual_seed(cli.seed),
        persistent_workers=cli.num_workers > 0,
    )
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        key: getattr(cli, key)
        for key in (
            "backbone",
            "volume_size",
            "slices",
            "slice_agg",
            "token_pool",
            "grid_size",
            "image_size",
            "window",
            "window_pct",
        )
    }
    preprocessing.update(
        embedding_partition=partition,
        axes=",".join(cli.axes),
        patch_size=cli.patch_size or embed.patch_size_of(config),
        image_mean=None if cli.backbone == "3dino" else mean,
        image_std=None if cli.backbone == "3dino" else std,
        window_bounds=window,
        normalization_source=norm_source,
        synthetic_normalize=dataset.synthetic_normalize,
        synthetic_fixed_reference=(
            {"mean": dataset._fixed_mean, "scale": dataset._fixed_scale}
            if dataset.synthetic_normalize == "fixed_reference"
            else None
        ),
    )
    (cli.output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n")
    (cli.output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    metadata = {key: str(value) if key == "output_dir" else value for key, value in vars(cli).items()}
    metadata.update(
        git_sha=embed.git_sha(),
        split="train",
        feature_dim=width,
        objective="content_barlow_twins" if cli.loss == "barlow_twins" else "content_symmetric_cross_view_infonce",
        embedding_partition=partition,
        style_objective="none",
        model_provenance=getattr(encoder, "provenance", None),
    )
    (cli.output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(
        "Aligning %d content dimensions; %d style dimensions excluded; loss=%s",
        partition["content_dim"],
        partition["style_dim"],
        cli.loss,
    )
    if cli.loss == "barlow_twins" and cli.batch_size < (cli.projection_dim or partition["content_dim"]):
        logger.warning(
            "BT correlation rank is limited to batch_size-1; small batches cannot attain identity "
            "at this projection width. Consider a larger batch or smaller projection."
        )
    logger.info(
        "Training %d subjects, %d batches/epoch; dropping %d leftover subjects per epoch",
        len(dataset),
        len(loader),
        len(dataset) % cli.batch_size,
    )
    for epoch in range(1, cli.epochs + 1):
        metrics = train_epoch(
            loader, encoder, projector, optimizer, scaler, config, cli, device, window, mean, std, partition
        )
        metrics["epoch"] = epoch
        with (cli.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        logger.info(
            "Epoch %d/%d: loss=%.4f accuracy=%.3f pos=%.3f neg=%.3f",
            epoch,
            cli.epochs,
            metrics["loss"],
            metrics["accuracy"],
            metrics["positive_similarity"],
            metrics["negative_similarity"],
        )
        save_checkpoint(cli.output_dir, encoder, projector, optimizer, scaler, epoch, cli, partition)
    logger.info("Saved fine-tuned encoder to %s", cli.output_dir / "encoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
