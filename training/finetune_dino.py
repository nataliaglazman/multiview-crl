"""Fine-tune a shared DINO backbone on the two synthetic MRI views.

    python -m training.finetune_dino --output-dir results/dino_infonce --epochs 20

Each subject contributes one feature per modality after pooling its selected planes.
The corresponding subject in the other modality is the positive. All backbone
parameters and an optional shared MLP projector receive gradients.

``--objective`` selects what that pairing is used for, which is the axis an ablation
varies:

* ``infonce`` (default) -- the other B-1 subjects in that modality are negatives, and
  both directions contribute equally.
* ``barlow`` / ``vicreg`` -- negative-free: the pairing is kept and the negatives are
  dropped, so the arm answers "are the negatives what matters?" rather than "does any
  training on this data help?". Both come from ``training.losses``, the same
  implementations the VQ-VAE arm of this project trains with.

Cross-view retrieval accuracy and positive/negative similarity are logged for every
objective and are never part of a negative-free loss, so the arms can be read against one
another on a quantity that means the same thing under each.

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
from utils.config import parse_dino_finetune_args

logger = logging.getLogger(__name__)


def _check_pair(view1, view2, objective):
    if view1.ndim != 2 or view1.shape != view2.shape or view1.shape[0] < 2:
        raise ValueError(f"{objective} requires matching (B, D) tensors with at least two subjects")


def pair_diagnostics(view1, view2, temperature=0.1):
    """Cross-view retrieval accuracy and mean positive/negative cosine similarity.

    Reported for EVERY objective, not just InfoNCE, and never part of the loss for the
    negative-free ones.  The point of running more than one objective is to compare the
    arms, and that needs one set of numbers whose meaning does not change between them --
    Barlow Twins' loss and InfoNCE's loss are not on a common scale, but "can you retrieve
    a subject's other modality" is the same question under both.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        similarity = F.normalize(view1.float(), dim=-1) @ F.normalize(view2.float(), dim=-1).T
        labels = torch.arange(len(view1), device=view1.device)
        positive = similarity.diag()
        n = len(view1)
        return {
            "accuracy": (
                (similarity.argmax(1) == labels).float().mean() + (similarity.argmax(0) == labels).float().mean()
            ).item()
            / 2,
            "positive_similarity": positive.mean().item(),
            "negative_similarity": ((similarity.sum() - positive.sum()) / (n * (n - 1))).item(),
        }


def _extra_diagnostics(loss):
    """The per-term breakdown ``training.losses`` hangs off its returned tensor.

    Only finite scalars are kept: the epoch aggregator averages every key it is handed and
    ``metrics.jsonl`` is written with ``allow_nan=False``, so one NaN diagnostic would end
    the run after the optimizer had already stepped.
    """
    out = {}
    for key, value in (getattr(loss, "_contrastive_diag", None) or {}).items():
        # Barlow Twins reports a hardcoded top1_acc of 0.0 as "not applicable". Next to the
        # real cross-view accuracy pair_diagnostics measures, that reads as a score of zero.
        if key == "top1_acc":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[key] = float(value)
    return out


def symmetric_infonce(view1, view2, temperature=0.1):
    """Mean of cross-view InfoNCE in both directions, with cosine similarities."""
    import torch
    import torch.nn.functional as F

    _check_pair(view1, view2, "InfoNCE")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Float32 keeps normalization and softmax stable under mixed precision.
    similarity = F.normalize(view1.float(), dim=-1) @ F.normalize(view2.float(), dim=-1).T
    logits = similarity / temperature
    labels = torch.arange(len(view1), device=view1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    metrics = {"loss": loss.item(), **pair_diagnostics(view1, view2)}
    return loss, metrics


def negative_free(view1, view2, cli):
    """Barlow Twins or VICReg over the paired views -- the same objective, minus negatives.

    Delegates to ``training.losses``, which already owns both, rather than restating them:
    the VQ-VAE arm of this project is trained with those exact implementations, so a DINO
    encoder fine-tuned here and a VQ-VAE trained there are optimising the same thing.

    Those losses take ``(n_views, B, C)`` and split content from style via
    ``estimated_content_indices``. A DINO embedding has no such split, and passing ``None``
    is how they are told so -- they then treat every view and every channel as content,
    which is the whole embedding, which is what we want.
    """
    import torch

    from training.losses import barlow_twins_loss, vicreg_loss

    _check_pair(view1, view2, cli.objective)
    # float32 deliberately: both losses standardise per channel and square the result, and
    # the upstream implementations note fp16 is not safe for that.
    hz = torch.stack([view1.float(), view2.float()])
    if cli.objective == "barlow":
        total = barlow_twins_loss(hz, lambd=cli.bt_lambda)
    else:
        total = vicreg_loss(
            hz, sim_coeff=cli.vicreg_sim_coeff, std_coeff=cli.vicreg_std_coeff, cov_coeff=cli.vicreg_cov_coeff
        )
    # Read the per-term breakdown BEFORE reshaping: it is a plain Python attribute hung off
    # the returned tensor, and squeeze() returns a new tensor that does not carry it.
    extra = _extra_diagnostics(total)
    # They accumulate into a (1,) tensor; squeeze so isfinite/backward/item behave as they
    # do for the InfoNCE scalar and the two arms stay interchangeable in the loop.
    loss = total.squeeze()
    metrics = {"loss": loss.item(), **pair_diagnostics(view1, view2), **extra}
    return loss, metrics


#: name -> (loss fn, what goes in training_config.json). The recorded string is the claim a
#: reader gets, so it names the paper objective rather than the flag.
OBJECTIVES = {
    "infonce": "symmetric_cross_view_infonce",
    "barlow": "cross_view_barlow_twins",
    "vicreg": "cross_view_vicreg",
}


def compute_objective(view1, view2, cli):
    """The selected objective's ``(loss, metrics)``."""
    if cli.objective == "infonce":
        return symmetric_infonce(view1, view2, temperature=cli.temperature)
    return negative_free(view1, view2, cli)


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


def train_epoch(loader, encoder, projector, optimizer, scaler, config, cli, device, window, mean, std):
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
            projected = [
                projector(encode_volumes(view, encoder, config, cli, device, window, mean, std)) for view in views
            ]
        # Keep the similarity matrix multiplication out of autocast too.
        loss, metrics = compute_objective(*projected, cli)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {cli.objective} loss; no optimizer update was applied")
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


def save_checkpoint(output_dir, encoder, projector, optimizer, scaler, epoch, cli):
    import torch

    encoder.save_pretrained(output_dir / "encoder")
    path = output_dir / "training_state.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
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
    projector = make_projector(width, cli).to(device)
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
        objective=OBJECTIVES[cli.objective],
        model_provenance=getattr(encoder, "provenance", None),
    )
    (cli.output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(
        "Training %d subjects, %d batches/epoch; dropping %d leftover subjects per epoch",
        len(dataset),
        len(loader),
        len(dataset) % cli.batch_size,
    )
    for epoch in range(1, cli.epochs + 1):
        metrics = train_epoch(loader, encoder, projector, optimizer, scaler, config, cli, device, window, mean, std)
        metrics["epoch"] = epoch
        with (cli.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        logger.info(
            "Epoch %d/%d [%s]: loss=%.4f accuracy=%.3f pos=%.3f neg=%.3f",
            epoch,
            cli.epochs,
            cli.objective,
            metrics["loss"],
            metrics["accuracy"],
            metrics["positive_similarity"],
            metrics["negative_similarity"],
        )
        save_checkpoint(cli.output_dir, encoder, projector, optimizer, scaler, epoch, cli)
    logger.info("Saved fine-tuned encoder to %s", cli.output_dir / "encoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
