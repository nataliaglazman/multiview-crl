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
from models.dino_partition import make_partition, split_embeddings
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


def barlow_twins(view1, view2, lambd=0.0051, eps=1e-5):
    """Batch-normalized cross-correlation objective, using population variance.

    Objective follows https://github.com/facebookresearch/barlowtwins .
    Correlation statistics use this process's actual batch, in float32.

    Kept as its own implementation rather than routed through
    ``training.losses.barlow_twins_loss``: that one is the VQ-VAE pipeline's, carrying
    centering modes, a correlation EMA and patch handling that mean nothing for a
    ``(B, D)`` embedding, and it has no variance stabiliser -- ``eps`` here is what keeps a
    constant channel finite, which ``tests/test_dino_partition.py`` pins.
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
    metrics = {
        "loss": loss.item(),
        **pair_diagnostics(view1, view2),
        "barlow_on_diag": on_diag.item(),
        "barlow_off_diag": off_diag.item(),
        "barlow_diag_mean": diagonal.mean().item(),
    }
    return loss, metrics


def cross_view_vicreg(view1, view2, cli):
    """VICReg over the paired views, from ``training.losses``.

    Unlike Barlow Twins there is no leaner local variant to prefer, and this is the same
    implementation the VQ-VAE arm of this project trains with. It takes ``(n_views, B, C)``
    and splits content from style via ``estimated_content_indices``; the split has already
    happened by the time we get here, so passing none tells it to treat every column of the
    content block as content.
    """
    import torch

    from training.losses import vicreg_loss

    _check_pair(view1, view2, "VICReg")
    # float32 deliberately: it standardises per channel and squares the result, which the
    # upstream implementation notes is not safe in fp16.
    hz = torch.stack([view1.float(), view2.float()])
    total = vicreg_loss(
        hz, sim_coeff=cli.vicreg_sim_coeff, std_coeff=cli.vicreg_std_coeff, cov_coeff=cli.vicreg_cov_coeff
    )
    # Read the per-term breakdown BEFORE reshaping: it is a plain Python attribute hung off
    # the returned tensor, and squeeze() returns a new tensor that does not carry it.
    extra = _extra_diagnostics(total)
    loss = total.squeeze()
    metrics = {"loss": loss.item(), **pair_diagnostics(view1, view2), **extra}
    return loss, metrics


#: flag -> what goes in training_config.json. Every arm aligns the CONTENT block only, so
#: the recorded name says so: a reader comparing two runs needs to know the loss AND what
#: it was applied to, and "barlow_twins" alone would not distinguish this from a run that
#: aligned the whole embedding.
OBJECTIVES = {
    "infonce": "content_symmetric_cross_view_infonce",
    "barlow": "content_barlow_twins",
    "vicreg": "content_vicreg",
}


#: The same three losses under --pairing within_modality. Spelled out rather than derived
#: from OBJECTIVES by string surgery: "cross_view" is baked into the InfoNCE name, and
#: patching a prefix onto it yields "within_modality_...cross_view...", which contradicts
#: itself. The cross-modal names are left exactly as they were, so runs trained before this
#: existed still report the string they recorded.
WITHIN_MODALITY_OBJECTIVES = {
    "infonce": "content_symmetric_within_modality_infonce",
    "barlow": "content_within_modality_barlow_twins",
    "vicreg": "content_within_modality_vicreg",
}


def recorded_objective(cli):
    """The string written to training_config.json: the loss AND what it was paired on.

    Both halves matter. Two runs can share a loss and differ in the pairing, or share the
    pairing and differ in the loss, and a reader comparing them needs the name to separate
    those -- "content_barlow_twins" alone would not say whether the two views were a
    subject's two modalities or one modality augmented twice.
    """
    table = OBJECTIVES if cli.pairing == "cross_modal" else WITHIN_MODALITY_OBJECTIVES
    return table[cli.objective]


def compute_objective(view1, view2, cli):
    """The selected objective's ``(loss, metrics)``, on whatever block it is handed."""
    if cli.objective == "infonce":
        return symmetric_infonce(view1, view2, temperature=cli.temperature)
    if cli.objective == "barlow":
        return barlow_twins(view1, view2, cli.barlow_lambda, cli.barlow_eps)
    return cross_view_vicreg(view1, view2, cli)


def alignment_loss(content1, content2, projector, cli):
    """The projector receives content only; style has no direct alignment gradient."""
    import torch

    first, second = projector(content1), projector(content2)
    # Disable autocast for correlation / similarity matrix products.
    with torch.autocast(device_type=first.device.type, enabled=False):
        return compute_objective(first, second, cli)


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


#: Per-sample intensity augmentation for the within-modality arm, scaled by --aug-strength.
#:
#: NO spatial transform. Rotation, scale or shear would move brain_size, lr_asymmetry and
#: the lesion coordinates -- the factors the evaluation then probes for -- so a spatially
#: augmented arm would be trained to discard its own measurement.
#:
#: ``gain``/``bias``/``noise`` deliberately stay mild: the generator renders a modality as
#: ``lut = base * gain + bias`` plus noise, so those three ARE its style model, and an arm
#: driven by them would re-derive the cross-modal relationship rather than stand as an
#: alternative to it. ``gamma`` and ``blur`` sit outside that family and are what make this
#: a different objective from the paired one.
AUGMENTATION = dict(gain=0.15, bias=0.10, gamma=0.30, noise=0.05, blur=1.0)


def _blur3d(volumes, sigma):
    """Separable 3-D Gaussian blur, applied one axis at a time."""
    import torch
    import torch.nn.functional as F

    if sigma <= 0.05:
        return volumes
    radius = max(1, int(round(3 * sigma)))
    grid = torch.arange(-radius, radius + 1, device=volumes.device, dtype=volumes.dtype)
    kernel = torch.exp(-grid.pow(2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = volumes.shape[1]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        weight = kernel.view(shape).expand(channels, 1, *shape[2:]).contiguous()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = radius
        volumes = F.conv3d(F.pad(volumes, pad, mode="replicate"), weight, groups=channels)
    return volumes


def augment_volumes(volumes, strength=1.0):
    """One draw of the within-modality augmentation. Two draws make a positive pair.

    Every parameter is sampled per SAMPLE, not per batch, so two subjects in one batch are
    not perturbed identically -- otherwise the shared perturbation is itself a signal the
    encoder can align on, and the batch's negatives become separable for the wrong reason.
    """
    import torch

    if strength <= 0:
        return volumes
    amount = {key: value * strength for key, value in AUGMENTATION.items()}
    x = volumes.float()
    batch = x.shape[0]
    per_sample = (batch,) + (1,) * (x.ndim - 1)

    flat = x.reshape(batch, -1)
    low = flat.min(dim=1).values.view(per_sample)
    span = (flat.max(dim=1).values.view(per_sample) - low).clamp_min(1e-6)

    def uniform(lo, hi):
        return torch.empty(batch, device=x.device, dtype=x.dtype).uniform_(lo, hi).view(per_sample)

    # Contrast first, on a per-sample [0, 1] rescale so the exponent is well defined.
    unit = ((x - low) / span).clamp(0, 1)
    x = unit.pow(uniform(1 - amount["gamma"], 1 + amount["gamma"])) * span + low
    x = _blur3d(x, float(torch.empty(1).uniform_(0, amount["blur"]).item()))
    x = x * uniform(1 - amount["gain"], 1 + amount["gain"]) + uniform(-amount["bias"], amount["bias"]) * span
    if amount["noise"] > 0:
        x = x + torch.randn_like(x) * amount["noise"] * span
    return x


def paired_views(batch, cli):
    """The two volumes whose CONTENT the objective is asked to align.

    ``cross_modal`` is the real acquisition pair: same subject, T1 and FLAIR, differing by
    the generator's style draw. ``within_modality`` never looks at the second modality --
    it augments ONE modality twice -- so the difference between the two arms is the
    pairing itself rather than the loss, the optimizer or the data budget.
    """
    views = batch["image"]
    if cli.pairing == "cross_modal":
        return views
    source = views[cli.aug_view - 1]
    return [augment_volumes(source, cli.aug_strength), augment_volumes(source, cli.aug_strength)]


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
        views = paired_views(batch, cli)
        context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with context:
            parts = [
                split_embeddings(encode_volumes(view, encoder, config, cli, device, window, mean, std), partition)
                for view in views
            ]
            # Content only. alignment_loss disables autocast around the matrix products.
            loss, metrics = alignment_loss(parts[0][0], parts[1][0], projector, cli)
        metrics.update(partition_diagnostics(parts))
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
        objective=recorded_objective(cli),
        embedding_partition=partition,
        style_objective="none",
        model_provenance=getattr(encoder, "provenance", None),
    )
    (cli.output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(
        "Aligning %d content dimensions; %d style dimensions excluded; loss=%s",
        partition["content_dim"],
        partition["style_dim"],
        cli.objective,
    )
    if cli.objective == "barlow" and cli.batch_size < (cli.projection_dim or partition["content_dim"]):
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
            "Epoch %d/%d [%s]: loss=%.4f accuracy=%.3f pos=%.3f neg=%.3f",
            epoch,
            cli.epochs,
            cli.objective,
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
