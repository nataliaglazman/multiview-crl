# isort: skip_file
# Multiview Contrastive Representation Learning — main training script.
#
# High-level structure
# --------------------
# config.py          parse_args / update_args / compute_gt_idx
# logging_setup.py   setup_logging
# checkpointing.py   save_checkpoint / load_checkpoint / save_emergency_checkpoint
# visualisation.py   save_decoded_images / save_vqvae_decoded_images
# evaluation.py      val_step / get_data / eval_step
# main_multimodal.py train_step + main (this file)

import collections
import csv
import json
import math
import os
import random
import traceback
import uuid
import warnings
from datetime import datetime

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models.vae as vae
import models.vqvae as vqvae
import utils.utils as utils
from data.infinite_iterator import InfiniteIterator
from eval.evaluation import eval_step, get_data
from losses import BaselineLoss, infonce_loss, moco_loss
from models.encoders import TextEncoder2D
from utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_emergency_checkpoint,
)
from utils.config import parse_args, update_args
from utils.logging_setup import setup_logging
from utils.visualisation import save_decoded_images, save_vqvae_decoded_images

device_ids = [0]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    data,
    encoders,
    decoders,
    loss_func,
    optimizer,
    params,
    args,
    scaler=None,
    recon_loss_fn=None,
    accumulation_step=0,
    total_accumulation_steps=1,
    moco_loss_func=None,
):
    """
    Perform a single forward + (optionally) backward pass.

    Args:
        data: Batch dictionary from the DataLoader.
        encoders: List of encoder models (or a single MoCoEncoder-wrapped VQVAE).
        decoders: List of decoder models (empty for VQ-VAE mode).
        loss_func: InfoNCE loss callable ``(hz, content_indices, subsets) -> loss``.
        optimizer: Optimizer (``None`` during validation).
        params: Iterable of parameters to clip gradients for.
        args: Parsed argument namespace.
        scaler: ``GradScaler`` for AMP (``None`` disables AMP).
        recon_loss_fn: Reconstruction loss instance (instantiated on first call if ``None``).
        accumulation_step: Index within the current accumulation window (0-based).
        total_accumulation_steps: Total number of mini-steps per optimizer update.
        moco_loss_func: MoCo loss callable (``None`` → standard in-batch InfoNCE).

    Returns:
        tuple: ``(total_loss, contrastive_loss, recon_loss, vq_loss, estimated_content_indices)``
    """
    if optimizer is not None and accumulation_step == 0:
        # set_to_none=True frees gradient tensors rather than zeroing them (~1× param memory saved)
        optimizer.zero_grad(set_to_none=True)

    if recon_loss_fn is None:
        recon_loss_fn = BaselineLoss().to(next(encoders[0].parameters()).device)

    use_amp = scaler is not None
    device = next(encoders[0].parameters()).device

    with autocast("cuda", enabled=use_amp):
        samples = data["image"]
        n_views = len(samples)
        images = torch.concat(samples, 0).to(device, non_blocking=True)  # (n_views * B, 1, D, H, W)
        input_shape = images.shape[2:]

        # ------------------------------------------------------------------
        # VQ-VAE-2 path
        # ------------------------------------------------------------------
        if args.encoder_type == "vqvae":
            vqvae_model = encoders[0]

            skip_recon_ratio = getattr(args, "skip_recon_ratio", 0.0)
            compute_recon = (skip_recon_ratio == 0.0) or (torch.rand(1).item() > skip_recon_ratio)

            (
                recon,
                diffs,
                encoder_outputs,
                estimated_content_indices,
                _,
                _,
            ) = vqvae_model(
                images,
                return_recon=compute_recon,
                pool_only=True,
                n_views=n_views,
                subsets=args.subsets,
            )

            # Compute momentum-encoder key embeddings BEFORE deleting images
            use_moco = getattr(args, "use_moco", False)
            if use_moco:
                from models.vqvae import MoCoEncoder

                assert isinstance(
                    vqvae_model, MoCoEncoder
                ), "MoCo requested but encoders[0] is not a MoCoEncoder instance."
                with torch.no_grad():
                    key_outputs = vqvae_model.encode_keys(images)

            if compute_recon and recon is not None:
                if recon.shape[2:] != input_shape:
                    recon = F.interpolate(recon, size=input_shape, mode="trilinear", align_corners=False)
                recon_loss = (
                    recon_loss_fn(
                        {"reconstruction": [recon], "quantization_losses": diffs},
                        images,
                    )
                    * args.scale_recon_loss
                )
                del recon, images
            else:
                recon_loss = torch.zeros(1, device=device)
                del images

            vq_loss = sum(diffs) * args.vq_commitment_weight

            total_contrastive_loss = torch.zeros(1, device=device)
            level_losses = []
            content_ratio = len(args.content_indices[0]) / (len(args.content_indices[0]) + len(args.style_indices))

            # Unwrap DataParallel / MoCoEncoder to reach the bare VQVAE so we
            # can read channel_logits (fix #1 / #4).
            _raw_vqvae = vqvae_model.online if hasattr(vqvae_model, "online") else vqvae_model
            _raw_vqvae = _raw_vqvae.module if hasattr(_raw_vqvae, "module") else _raw_vqvae

            for level_idx, enc_pooled in enumerate(encoder_outputs):
                hz_level = enc_pooled.reshape(n_views, -1, enc_pooled.shape[-1])
                n_channels = hz_level.shape[-1]
                content_size = max(1, int(content_ratio * n_channels))

                if _raw_vqvae.channel_logits is not None:
                    # Fix #1: use learnable channel_logits instead of batch-derived
                    # statistics — consistent across batches and co-trained with the
                    # model via reconstruction gradients through the soft Gumbel mask.
                    # Fix #4: eval mode gets deterministic top-k (no Gumbel noise)
                    # so content_idx is identical for every evaluation batch.
                    logits = _raw_vqvae.channel_logits.detach().unsqueeze(0)
                    if _raw_vqvae.training:
                        content_masks = utils.gumbel_softmax_mask(
                            avg_logits=logits,
                            content_sizes=[content_size],
                            subsets=args.subsets,
                        )
                    else:
                        hard_mask = torch.zeros_like(logits)
                        topk_idx = torch.topk(logits, content_size, dim=1).indices
                        hard_mask.scatter_(1, topk_idx, 1.0)
                        content_masks = [hard_mask] * len(args.subsets)
                else:
                    # Fallback: no channel_logits configured, use batch statistics.
                    avg_logits = hz_level.mean(dim=[0, 1], keepdim=False).unsqueeze(0)
                    if len(args.subsets) > 1 and content_size > 0:
                        content_masks = utils.smart_gumbel_softmax_mask(
                            avg_logits=avg_logits,
                            content_sizes=[content_size],
                            subsets=args.subsets,
                        )
                    else:
                        content_masks = utils.gumbel_softmax_mask(
                            avg_logits=avg_logits,
                            content_sizes=[content_size],
                            subsets=args.subsets,
                        )
                estimated_content_indices = [torch.where(m.bool())[-1].tolist() for m in content_masks]
                level_content_indices = estimated_content_indices

                if use_moco:
                    key_pooled = key_outputs[level_idx]
                    k_level = key_pooled.reshape(n_views, -1, key_pooled.shape[-1])
                    queue_snapshot = vqvae_model.queues[level_idx].clone().detach()
                    level_loss = moco_loss_func(
                        hz_level,
                        k_level,
                        queue_snapshot,
                        level_content_indices,
                        args.subsets,
                    )
                else:
                    level_loss = loss_func(hz_level, level_content_indices, args.subsets)

                level_losses.append(level_loss.item())
                total_contrastive_loss = total_contrastive_loss + level_loss * args.scale_contrastive_loss

            # Enqueue all levels in one call after the loss loop
            if use_moco:
                with torch.no_grad():
                    vqvae_model.enqueue([k.detach() for k in key_outputs])

            contrastive_loss = total_contrastive_loss
            total_loss = contrastive_loss + recon_loss + vq_loss
            recon_loss_value = recon_loss.item()
            vq_loss_value = vq_loss.item()
            contrastive_loss_value = contrastive_loss.item()
            # NOTE: estimated_content_indices was already set to the dynamically
            # computed indices from channel_logits inside the level loop above
            # (line: estimated_content_indices = [torch.where(m.bool())[-1].tolist()
            # for m in content_masks]).  Do NOT overwrite it here with
            # args.content_indices — that would replace the learned channel
            # selection with the static config-based indices and break evaluation
            # in get_data() for any run that uses channel_logits.

        # ------------------------------------------------------------------
        # VAE path
        # ------------------------------------------------------------------
        else:
            hz = []
            for m_midx, m in enumerate(args.modalities):
                samples = data[m]
                hz_m = encoders[m_midx](torch.concat(samples, 0))
                hz += [hz_m]
            hz = torch.concat(hz, 0)
            hz_flat = hz.view(hz.size(0), -1)

            decoded_images = decoders[0](hz_flat)
            ground_truth_images = torch.concat(data["image"], 0).to(decoded_images.device)
            recon_loss = (
                recon_loss_fn(
                    {"reconstruction": [decoded_images], "quantization_losses": []},
                    ground_truth_images,
                )
                * args.scale_recon_loss
            )

            avg_logits = hz_flat.mean(0)[None]
            if "content_indices" not in data:
                data["content_indices"] = args.content_indices
            content_size = [len(content) for content in data["content_indices"]]

            if args.selection in ["ground_truth", "concat"]:
                estimated_content_indices = args.content_indices
            else:
                if args.subsets[-1] == list(range(args.n_views)) and content_size[-1] > 0:
                    content_masks = utils.smart_gumbel_softmax_mask(
                        avg_logits=avg_logits,
                        content_sizes=content_size,
                        subsets=args.subsets,
                    )
                else:
                    content_masks = utils.gumbel_softmax_mask(
                        avg_logits=avg_logits,
                        content_sizes=content_size,
                        subsets=args.subsets,
                    )
                estimated_content_indices = [torch.where(c_mask)[-1].tolist() for c_mask in content_masks]

            contrastive_loss = loss_func(
                hz_flat.reshape(n_views, -1, hz_flat.shape[-1]),
                estimated_content_indices,
                args.subsets,
            )
            total_loss = contrastive_loss + recon_loss
            recon_loss_value = recon_loss.item()
            vq_loss_value = 0.0
            contrastive_loss_value = contrastive_loss.item()

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    if optimizer is not None:
        scaled_loss = total_loss / total_accumulation_steps
        if use_amp:
            scaler.scale(scaled_loss).backward()
            if accumulation_step == total_accumulation_steps - 1:
                scaler.unscale_(optimizer)
                clip_grad_norm_(params, max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
        else:
            scaled_loss.backward()
            if accumulation_step == total_accumulation_steps - 1:
                clip_grad_norm_(params, max_norm=2.0, norm_type=2)
                optimizer.step()

    return (
        total_loss.item(),
        contrastive_loss_value,
        recon_loss_value,
        vq_loss_value,
        estimated_content_indices,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    # CUDA memory settings — must be applied before any allocation
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    # Resolve paths
    if args.dataset_name != "mpi3d":
        args.datapath = os.path.join(args.dataroot, args.dataset_name)
    else:
        args.datapath = os.path.join(
            args.dataroot,
            f"{args.dataset_name}/real3d_complicated_shapes_ordered.npz",
        )
    args.model_dir = os.path.join(args.model_dir, args.dataset_name)
    if args.model_id is None:
        setattr(args, "model_id", uuid.uuid4())
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    os.makedirs(args.save_dir, exist_ok=True)

    # Logging
    logger = setup_logging(args.save_dir)
    logger.info("=" * 60)
    logger.info("MULTIVIEW CONTRASTIVE REPRESENTATION LEARNING")
    logger.info("=" * 60)
    logger.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'EVALUATION' if args.evaluate else 'TRAINING'}")
    logger.info("")
    logger.info("[PATHS]")
    logger.info(f"  Data root:  {args.dataroot}")
    logger.info(f"  Data path:  {args.datapath}")
    logger.info(f"  Save dir:   {args.save_dir}")
    logger.info(f"  Model ID:   {args.model_id}")

    # Optionally reload saved args (evaluation only)
    if args.evaluate and args.load_args:
        with open(os.path.join(args.save_dir, "settings.json"), "r") as fp:
            loaded_args = json.load(fp)
        for arg in ["encoding_size", "hidden_size"]:
            setattr(args, arg, loaded_args[arg])

    args = update_args(args)

    logger.info("")
    logger.info("[CONFIGURATION]")
    logger.info(f"  Dataset:       {args.dataset_name}")
    logger.info(f"  Modalities:    {args.modalities}")
    logger.info(f"  Num views:     {args.n_views}")
    logger.info(f"  Encoding size: {args.encoding_size}")
    logger.info(f"  Batch size:    {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Temperature:   {args.tau}")
    logger.info(f"  Train steps:   {args.train_steps}")
    logger.info(f"  Subsets:       {args.subsets}")

    # Print all args for backwards compatibility
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # Reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"  Seed: {args.seed}")

    # Persist settings
    if not args.evaluate:
        settings_dict = {k: v for k, v in args.__dict__.items() if k != "DATASETCLASS"}
        settings_path = os.path.join(args.save_dir, "settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings_dict, f, indent=4)
        logger.info(f"  Settings saved to: {settings_path}")

    # Device
    logger.info("")
    logger.info("[DEVICE]")
    if torch.cuda.is_available() and not args.no_cuda:
        device = f"cuda:{device_ids[0]}"
        logger.info(f"  GPU: {device} — {torch.cuda.get_device_name(device_ids[0])}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(device_ids[0]).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        warnings.warn("CUDA not available or --no-cuda set; running on CPU.")
        logger.warning("  Using CPU.")

    # Loss functions
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    def loss_func(z_rec_tuple, estimated_content_indices, subsets):
        return infonce_loss(
            z_rec_tuple,
            sim_metric=sim_metric,
            criterion=criterion,
            tau=args.tau,
            projector=(lambda x: x),
            estimated_content_indices=estimated_content_indices,
            subsets=subsets,
        )

    def moco_loss_func(q, k, queue, estimated_content_indices, subsets):
        return moco_loss(
            q,
            k,
            queue,
            sim_metric=sim_metric,
            tau=args.tau,
            estimated_content_indices=estimated_content_indices,
            subsets=subsets,
        )

    # Augmentations / transforms
    if HAS_FAISS:
        faiss.omp_set_num_threads(args.faiss_omp_threads)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                args.DATASETCLASS.mean_per_channel,
                args.DATASETCLASS.std_per_channel,
            ),
        ]
    )

    dataset_kwargs = {"transform": transform}
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }

    # Datasets
    logger.info("")
    logger.info("[DATASETS]")
    train_dataset = args.DATASETCLASS(
        data_dir=args.datapath,
        mode="train",
        change_lists=args.change_lists,
        spacing=getattr(args, "image_spacing", 2.0),
        crop_margin=getattr(args, "crop_margin", 0),
        cache=getattr(args, "cache_dataset", False),
        cache_dir=getattr(args, "cache_dir", None),
        **dataset_kwargs,
    )
    logger.info(f"  Train: {len(train_dataset)} samples from {args.datapath}")

    if args.dataset_name == "multimodal3di":
        dataset_kwargs["vocab_filepath"] = train_dataset.vocab_filepath
    if args.dataset_name == "mpi3d":
        dataset_kwargs["collate_random_pair"] = True
        train_dataset.collate_random_pair = True
        if args.collate_random_pair:
            dataloader_kwargs["collate_fn"] = train_dataset.__collate_fn__random_pair__

    if args.evaluate:
        val_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="val",
            change_lists=args.change_lists,
            spacing=getattr(args, "image_spacing", 2.0),
            crop_margin=getattr(args, "crop_margin", 0),
            **dataset_kwargs,
        )
        test_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="test",
            change_lists=args.change_lists,
            spacing=getattr(args, "image_spacing", 2.0),
            crop_margin=getattr(args, "crop_margin", 0),
            **dataset_kwargs,
        )
    else:
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        train_iterator = InfiniteIterator(train_loader)

    print(f"Train dataset size: {len(train_dataset)} samples.")

    # Model
    logger.info("")
    logger.info("[MODEL]")
    if args.encoder_type == "vqvae":
        use_checkpoint = getattr(args, "gradient_checkpointing", False)
        logger.info(
            f"  VQ-VAE-2 | levels={args.vqvae_nb_levels} "
            f"hidden={args.vqvae_hidden_channels} embed={args.vqvae_embed_dim} "
            f"entries={args.vqvae_nb_entries} grad_ckpt={use_checkpoint}"
        )
        vqvae_model = vqvae.VQVAE(
            in_channels=1,
            hidden_channels=args.vqvae_hidden_channels,
            res_channels=args.vqvae_res_channels,
            nb_res_layers=2,
            nb_levels=args.vqvae_nb_levels,
            embed_dim=args.vqvae_embed_dim,
            nb_entries=args.vqvae_nb_entries,
            scaling_rates=args.vqvae_scaling_rates,
            use_checkpoint=use_checkpoint,
            content_size=len(args.content_indices[0]),
            style_size=len(args.style_indices),
            inject_style_to_decoder=getattr(args, "inject_style_to_decoder", False),
        )
        if getattr(args, "compile_model", False):
            logger.info("  Compiling VQ-VAE-2 with torch.compile (this may take a minute)...")
            vqvae_model = torch.compile(vqvae_model)
        vqvae_model = torch.nn.DataParallel(vqvae_model, device_ids=device_ids)
        vqvae_model.to(device)
        logger.info(f"  Parameters: {sum(p.numel() for p in vqvae_model.parameters()):,}")

        encoders = [vqvae_model]
        decoders = []

        if getattr(args, "use_moco", False):
            from models.vqvae import MoCoEncoder

            moco_model = MoCoEncoder(
                vqvae_model,
                queue_size=args.moco_queue_size,
                momentum=args.moco_momentum,
                nb_levels=args.vqvae_nb_levels,
            )
            moco_model.to(device)
            encoders = [moco_model]
            logger.info(f"  MoCo: queue_size={args.moco_queue_size} momentum={args.moco_momentum}")

        total_params = sum(p.numel() for p in vqvae_model.parameters())

    else:
        logger.info("  VAE encoder + decoder")
        encoder_img = torch.nn.DataParallel(vae.Encoder(), device_ids=device_ids).to(device)
        encoders = [encoder_img]

        if "text" in args.modalities:
            encoder_txt = TextEncoder2D(
                input_size=train_dataset.vocab_size,
                output_size=args.encoding_size,
                sequence_length=train_dataset.max_sequence_length,
            )
            encoder_txt = torch.nn.DataParallel(encoder_txt, device_ids=device_ids).to(device)
            encoders += [encoder_txt]

        # Compute spatial size from spacing/crop settings to match the encoder's output
        spacing = getattr(args, "image_spacing", 2.0)
        crop_margin = getattr(args, "crop_margin", 0)
        if spacing == 1.0:
            spatial_size = (182, 218, 182)
        elif spacing == 2.0:
            spatial_size = (91, 109, 91)
        else:
            spatial_size = tuple(int(s / spacing) for s in (182, 218, 182))
        if crop_margin > 0:
            spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)
        decoder = torch.nn.DataParallel(
            vae.Decoder(latent_dim=512, spatial_size=spatial_size), device_ids=device_ids
        ).to(device)
        decoders = [decoder]
        total_params = sum(sum(p.numel() for p in m.parameters()) for m in encoders + decoders)

    logger.info(f"  Total trainable parameters: {total_params:,}")

    # Load pretrained weights (evaluation mode)
    if args.evaluate:
        logger.info("")
        logger.info("[LOADING PRETRAINED MODELS]")
        if args.encoder_type == "vqvae":
            path = os.path.join(args.save_dir, "vqvae_model.pt")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            encoders[0].load_state_dict(checkpoint["encoders"])
            if getattr(args, "use_moco", False) and "moco_queues" in checkpoint:
                from models.vqvae import MoCoEncoder

                if isinstance(encoders[0], MoCoEncoder):
                    for lvl, q_cpu in enumerate(checkpoint["moco_queues"]):
                        encoders[0]._get_queue(lvl).copy_(q_cpu.to(device))
                    encoders[0].queue_ptrs = list(checkpoint["moco_queue_ptrs"])
            logger.info(f"  Loaded VQ-VAE-2 from {path}")
        else:
            for m_idx, m in enumerate(args.modalities):
                path = os.path.join(args.save_dir, f"encoder_{m}.pt")
                encoders[m_idx].load_state_dict(torch.load(path, map_location=device, weights_only=False))
                logger.info(f"  Loaded encoder_{m} from {path}")

    # Optimizer
    params = []
    for f in encoders:
        params += list(f.parameters())
    for d in decoders:
        params += list(d.parameters())
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(params, lr=args.lr, fused=use_fused)
    logger.info("")
    logger.info(f"[OPTIMIZER] AdamW (fused={use_fused}) lr={args.lr} " f"params={sum(p.numel() for p in params):,}")

    # LR schedule: linear warmup then cosine annealing (or constant)
    warmup_steps = getattr(args, "warmup_steps", 0)
    lr_schedule = getattr(args, "lr_schedule", "constant")
    lr_min = getattr(args, "lr_min", 0.0)
    lr_min_ratio = lr_min / args.lr if args.lr > 0 else 0.0
    total_steps = args.train_steps

    def _lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        if lr_schedule == "constant":
            return 1.0
        # Cosine annealing
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    logger.info(
        f"[LR SCHEDULE] {lr_schedule} | warmup={warmup_steps} steps | " f"lr_min={lr_min} | total={total_steps} steps"
    )

    scaler = GradScaler("cuda") if args.use_amp else None
    if args.use_amp:
        logger.info("  Mixed precision: enabled (AMP)")

    recon_loss_fn = BaselineLoss().to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    file_name = os.path.join(args.save_dir, "Training.csv")

    if not args.evaluate:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(
            f"  Steps: {args.train_steps}  |  Log every: {args.log_steps}  "
            f"|  Checkpoint every: {args.checkpoint_steps}"
        )

        step = 1
        loss_values = collections.deque(maxlen=args.log_steps)
        contrastive_losses = collections.deque(maxlen=args.log_steps)
        recon_losses = collections.deque(maxlen=args.log_steps)
        vq_losses = collections.deque(maxlen=args.log_steps)

        loss_deques = {
            "loss": loss_values,
            "contrastive_loss": contrastive_losses,
            "recon_loss": recon_losses,
            "vq_loss": vq_losses,
        }
        step = load_checkpoint(args, encoders, decoders, optimizer, device, loss_deques, scheduler=scheduler)

        # Restore best loss from best checkpoint if resuming
        best_total_loss = float("inf")
        best_ckpt_path = os.path.join(
            args.save_dir, "vqvae_best.pt" if args.encoder_type == "vqvae" else "checkpoint_best.pt"
        )
        if getattr(args, "resume_training", False) and os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_total_loss = best_ckpt.get("loss", float("inf"))
            logger.info(f"  Restored best loss: {best_total_loss:.4f} from {best_ckpt_path}")
            del best_ckpt

        oom_count = 0
        MAX_OOM_RETRIES = 5

        try:
            while step <= args.train_steps:
                try:
                    accum_steps = getattr(args, "gradient_accumulation_steps", 1)
                    accum_total = accum_contrastive = accum_recon = accum_vq = 0.0

                    for accum_idx in range(accum_steps):
                        data = next(train_iterator)
                        (
                            total_loss,
                            contrastive_loss,
                            recon_loss,
                            vq_loss,
                            _,
                        ) = train_step(
                            data,
                            encoders,
                            decoders,
                            loss_func,
                            optimizer,
                            params,
                            args=args,
                            scaler=scaler,
                            recon_loss_fn=recon_loss_fn,
                            accumulation_step=accum_idx,
                            total_accumulation_steps=accum_steps,
                            moco_loss_func=moco_loss_func,
                        )
                        accum_total += total_loss / accum_steps
                        accum_contrastive += contrastive_loss / accum_steps
                        accum_recon += recon_loss / accum_steps
                        accum_vq += vq_loss / accum_steps

                    scheduler.step()
                    oom_count = 0
                    loss_values.append(accum_total)
                    contrastive_losses.append(accum_contrastive)
                    recon_losses.append(accum_recon)
                    vq_losses.append(accum_vq)

                    print(
                        f"Step {step}: Total={accum_total:.4f} | "
                        f"Contrastive={accum_contrastive:.4f} | "
                        f"Recon={accum_recon:.4f} | VQ={accum_vq:.4f}",
                        flush=True,
                    )

                    tb_writer.add_scalar("Loss/Total", accum_total, step)
                    tb_writer.add_scalar("Loss/Contrastive", accum_contrastive, step)
                    tb_writer.add_scalar("Loss/Recon", accum_recon, step)
                    tb_writer.add_scalar("Loss/VQ", accum_vq, step)
                    tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

                    if step % args.log_steps == 1 or step == args.train_steps:
                        with open(file_name, "a+") as f:
                            csv.writer(f).writerow(
                                [
                                    "Step",
                                    step,
                                    "Total",
                                    f"{np.mean(loss_values):.3f}",
                                    "Contrastive",
                                    f"{np.mean(contrastive_losses):.3f}",
                                    "Recon",
                                    f"{np.mean(recon_losses):.3f}",
                                    "VQ",
                                    f"{np.mean(vq_losses):.3f}",
                                ]
                            )
                        tb_writer.flush()

                    if (step % 200 == 0 or step == 1) and args.encoder_type != "vqvae":
                        save_decoded_images(encoders, decoders, data, args, step)
                    if (step % 200 == 0 or step == 1) and args.encoder_type == "vqvae":
                        save_vqvae_decoded_images(encoders[0], data, args, step)

                    if step % args.checkpoint_steps == 1 or step == args.train_steps or step == args.log_steps * 2:
                        # Check if rolling average loss is a new best
                        rolling_loss = np.mean(loss_values) if len(loss_values) == loss_values.maxlen else None
                        new_best = None
                        if rolling_loss is not None and rolling_loss < best_total_loss:
                            best_total_loss = rolling_loss
                            new_best = rolling_loss

                        save_checkpoint(
                            args,
                            step,
                            encoders,
                            decoders,
                            optimizer,
                            total_loss,
                            contrastive_loss,
                            recon_loss,
                            vq_loss,
                            scheduler=scheduler,
                            best_loss=new_best,
                        )

                    step += 1

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_count += 1
                        if torch.cuda.is_available():
                            alloc = torch.cuda.memory_allocated() / 1024**3
                            reserv = torch.cuda.memory_reserved() / 1024**3
                            peak = torch.cuda.max_memory_allocated() / 1024**3
                            logger.error(
                                f"[OOM] Step {step}: allocated={alloc:.2f}GB "
                                f"reserved={reserv:.2f}GB peak={peak:.2f}GB "
                                f"({oom_count}/{MAX_OOM_RETRIES})"
                            )
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()
                        if optimizer is not None:
                            optimizer.zero_grad(set_to_none=True)
                        if oom_count >= MAX_OOM_RETRIES:
                            logger.error(f"[OOM] {MAX_OOM_RETRIES} consecutive OOMs — aborting.")
                            save_emergency_checkpoint(
                                args,
                                step,
                                encoders,
                                decoders,
                                optimizer,
                                reason=f"oom_x{oom_count}",
                                scheduler=scheduler,
                            )
                            raise
                        logger.warning(f"[OOM] Skipping step {step}, continuing...")
                        step += 1
                    else:
                        logger.error(f"[ERROR] Step {step}: {e}\n{traceback.format_exc()}")
                        save_emergency_checkpoint(
                            args,
                            step,
                            encoders,
                            decoders,
                            optimizer,
                            reason=f"runtime_error_step{step}",
                            scheduler=scheduler,
                        )
                        raise

                except Exception as e:
                    logger.error(f"[ERROR] Step {step}: {type(e).__name__} — {e}\n{traceback.format_exc()}")
                    save_emergency_checkpoint(
                        args,
                        step,
                        encoders,
                        decoders,
                        optimizer,
                        reason=f"{type(e).__name__}_step{step}",
                        scheduler=scheduler,
                    )
                    raise

        except KeyboardInterrupt:
            logger.warning(f"\n[INTERRUPTED] Training stopped at step {step}")
            save_emergency_checkpoint(
                args,
                step,
                encoders,
                decoders,
                optimizer,
                reason=f"keyboard_interrupt_step{step}",
                scheduler=scheduler,
            )
            tb_writer.close()
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        if loss_values:
            logger.info(f"  Final total loss:       {loss_values[-1]:.4f}")
            logger.info(f"  Final contrastive loss: {contrastive_losses[-1]:.4f}")
            logger.info(f"  Final recon loss:       {recon_losses[-1]:.4f}")
            logger.info(f"  Final VQ loss:          {vq_losses[-1]:.4f}")
            logger.info(f"  Rolling avg total (last {args.log_steps}): {np.mean(loss_values):.4f}")
        logger.info(f"  Models saved to: {args.save_dir}")
        tb_writer.close()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if args.evaluate:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)

        logger.info("[EVALUATION] Collecting validation encodings...")
        val_dict = get_data(
            val_dataset,
            encoders,
            decoders,
            loss_func,
            dataloader_kwargs,
            args=args,
            num_samples=args.val_size,
            recon_loss_fn=recon_loss_fn,
            moco_loss_func=moco_loss_func,
        )
        logger.info("[EVALUATION] Collecting test encodings...")
        test_dict = get_data(
            test_dataset,
            encoders,
            decoders,
            loss_func,
            dataloader_kwargs,
            args=args,
            num_samples=args.test_size,
            recon_loss_fn=recon_loss_fn,
            moco_loss_func=moco_loss_func,
        )

        logger.info(f"  Val loss:  {np.mean(val_dict['loss_values']):.4f}")
        logger.info(f"  Test loss: {np.mean(test_dict['loss_values']):.4f}")
        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

        if args.encoding_size == 1:
            for split in (val_dict, test_dict):
                for m in args.modalities:
                    split[f"hz_{m}"] = split[f"hz_{m}"].reshape(-1, 1)

        for m in args.modalities:
            sc = StandardScaler()
            val_dict[f"hz_{m}"] = sc.fit_transform(val_dict[f"hz_{m}"])
            test_dict[f"hz_{m}"] = sc.transform(test_dict[f"hz_{m}"])
            for s in args.subsets:
                sc = StandardScaler()
                val_dict[f"hz_{m}_subsets"][s] = sc.fit_transform(val_dict[f"hz_{m}_subsets"][s])
                test_dict[f"hz_{m}_subsets"][s] = sc.transform(test_dict[f"hz_{m}_subsets"][s])

        results = []
        for m_idx, m in enumerate(args.modalities):
            factors_m = args.DATASETCLASS.FACTORS[m]
            discrete_factors_m = args.DATASETCLASS.DISCRETE_FACTORS[m]

            if args.eval_dci:
                import eval.dci as dci

                def repr_fn(samples):
                    with torch.no_grad():
                        if m == "image" and args.dataset_name == "mpi3d":
                            samples = torch.stack([transform(i) for i in samples], dim=0)
                        return encoders[m_idx](samples).cpu().numpy()

                dci_score = dci.compute_dci(
                    ground_truth_data=val_dataset,
                    representation_function=repr_fn,
                    num_train=10000,
                    num_test=5000,
                    random_state=np.random.RandomState(),
                    factor_types=["discrete" if ix in discrete_factors_m else "continuous" for ix in factors_m],
                )
                with open(os.path.join(args.save_dir, f"dci_{m}.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=dci_score.keys())
                    w.writeheader()
                    w.writerow(dci_score)
                continue

            for ix, factor_name in factors_m.items():
                for s in args.subsets:
                    data_eval = [
                        val_dict[f"hz_{m}_subsets"][s],
                        val_dict[f"labels_{m}"][factor_name],
                        test_dict[f"hz_{m}_subsets"][s],
                        test_dict[f"labels_{m}"][factor_name],
                    ]
                    results.append(eval_step(ix, s, m, factor_name, discrete_factors_m, data_eval, args))
                if args.eval_style and len(args.style_indices) > 0:
                    data_eval = [
                        val_dict[f"hz_{m}"][..., args.style_indices],
                        val_dict[f"labels_{m}"][factor_name],
                        test_dict[f"hz_{m}"][..., args.style_indices],
                        test_dict[f"labels_{m}"][factor_name],
                    ]
                    results.append(eval_step(ix, -1, m, factor_name, discrete_factors_m, data_eval, args))

        columns = [
            "subset",
            "ix",
            "modality",
            "factor_name",
            "factor_type",
            "r2_linreg",
            "r2_krreg",
            "acc_logreg",
            "acc_mlp",
        ]
        df_results = pd.DataFrame(results, columns=columns)
        results_path = os.path.join(args.save_dir, "results.csv")
        df_results.to_csv(results_path)
        logger.info(f"  Results saved to: {results_path}")
        print(df_results.to_string())


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(args)
