# flake8: noqa
"""Argument parsing and dataset-specific configuration for multiview-CRL."""

import argparse
import functools
import operator

import numpy as np

import data.datasets as datasets
import utils.utils as utils


def parse_args() -> argparse.ArgumentParser:
    """
    Build and return the argument parser.

    Returns:
        argparse.ArgumentParser: Parser (call ``.parse_args()`` to get the namespace).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="/data/natalia/")
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Path to the labels CSV file (e.g. labels_cleaned_3class.csv). " "Required for ADNI / custom datasets.",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Root directory containing per-subject brain masks "
        "(same <subject>/t1, <subject>/t2 layout as the image data). "
        "Masks are identified by a '_brain_mask.nii.gz' suffix. "
        "When set, reconstruction loss is computed only over brain voxels. "
        "Defaults to None, in which case masks are expected alongside the images.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ADNI_registered",
        choices=[
            "mpi3d",
            "independent3di",
            "causal3di",
            "multimodal3di",
            "adni",
            "ADNI_registered",
            "ADNI_stripped_masks",
            "custom",
            "synthetic",
        ],
    )
    parser.add_argument(
        "--synthetic-mode",
        type=str,
        default="pseudo_mri",
        choices=["pseudo_mri", "primitives", "random"],
        help="Renderer used by the synthetic dataset (only when --dataset_name=synthetic).",
    )
    parser.add_argument(
        "--synthetic-num-train",
        type=int,
        default=1000,
        help="Number of synthetic training samples per epoch.",
    )
    parser.add_argument(
        "--synthetic-num-val",
        type=int,
        default=100,
        help="Number of synthetic validation samples.",
    )
    parser.add_argument(
        "--synthetic-num-test",
        type=int,
        default=200,
        help="Number of synthetic test samples.",
    )
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--synthetic-n-content", type=int, default=5)
    parser.add_argument("--synthetic-n-style", type=int, default=3)
    parser.add_argument(
        "--synthetic-res",
        type=int,
        default=32,
        help="Cubic resolution for synthetic volumes. Used as the default "
        "--spatial-size when --dataset_name=synthetic and --spatial-size is unset. "
        "Should be divisible by 8 for the 3-level VQ-VAE.",
    )
    parser.add_argument("--model-dir", type=str, default="results")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (applied to all params except biases, norms, and ReZero alphas)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=300001)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--checkpoint-steps", type=int, default=200)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--val-size", default=25000, type=int)
    parser.add_argument("--test-size", default=25000, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers. For 3D MRI with pin_memory, each worker holds "
        "prefetch_factor batches in pinned memory (~330 MB each). Keep this low (4-8).",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision (fp16) to reduce memory",
    )
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume training from last checkpoint if available",
    )
    parser.add_argument("--load-args", action="store_true")
    parser.add_argument("--collate-random-pair", action="store_true")
    parser.add_argument("--modalities", default=["image"], choices=[["image"], ["image", "text"]])
    parser.add_argument(
        "--scale-recon-loss",
        type=float,
        default=1,
        help="Scale factor for the reconstruction loss",
    )
    parser.add_argument("--scale-contrastive-loss", type=float, default=1)
    parser.add_argument(
        "--scale-style-contrastive-loss",
        type=float,
        default=0.0,
        help="Scale factor for the within-modality style InfoNCE loss. 0 disables it.",
    )
    parser.add_argument(
        "--scale-content-modality-adv",
        type=float,
        default=0.0,
        help="Weight on gradient-reversal modality classifier from content. "
        "Enforces content invariance explicitly, decoupled from style bottleneck size. "
        "Watch ModAdv/acc_L0 → 0.5 means invariant.",
    )
    parser.add_argument(
        "--content-modality-adv-lambda",
        type=float,
        default=1.0,
        help="Gradient-reversal scale lambda for the content→modality adversarial loss.",
    )
    parser.add_argument(
        "--scale-style-modality-ce",
        type=float,
        default=0.0,
        help="Weight on CE modality classifier from style (sufficiency). "
        "Pushes style to carry modality info so demographic signal stays in style, "
        "not discarded. Watch ModSuf/acc_L0 → 1.0 means sufficient.",
    )
    parser.add_argument(
        "--separation-floor-diagnosis-info",
        type=float,
        default=0.1,
        help="Minimum content/diagnosis_info (chance-adjusted probe accuracy in [0,1]) "
        "at the finest level. Below this, separation_score_gated is linearly penalised "
        "so a collapsed-content encoder cannot win the sweep.",
    )
    parser.add_argument(
        "--select-by-gated-score",
        action="store_true",
        help="Use separation_score_gated (not separation_score) for best-checkpoint selection. "
        "Requires labels in the val loader; otherwise gate is 1.0 and behaviour is unchanged.",
    )

    # GAN discriminator (improves reconstruction sharpness)
    parser.add_argument(
        "--use-gan",
        action="store_true",
        help="Add a 3-D PatchGAN discriminator to sharpen reconstructions.",
    )
    parser.add_argument(
        "--scale-adv-loss",
        type=float,
        default=0.1,
        help="Weight of the generator adversarial loss term (default: 0.1).",
    )
    parser.add_argument(
        "--gan-start-step",
        type=int,
        default=0,
        help="Step at which the GAN loss activates. Setting this to e.g. 5000 "
        "lets reconstruction stabilise before the discriminator is introduced.",
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=4e-4,
        help="Learning rate for the discriminator optimizer (default: 4e-4).",
    )
    parser.add_argument(
        "--disc-base-channels",
        type=int,
        default=32,
        help="Base channel width of the PatchDiscriminator3D (default: 32).",
    )

    parser.add_argument(
        "--contrastive-loss-type",
        type=str,
        default="infonce",
        choices=["infonce", "barlow_twins", "vicreg"],
        help="Contrastive objective: 'infonce' (default, uses negatives — pair with "
        "--use-moco for small batches), 'barlow_twins' (negative-free, redundancy "
        "reduction — works well at any batch size), or 'vicreg' (negative-free, "
        "variance-invariance-covariance — more stable than Barlow Twins at very small "
        "batch sizes).",
    )

    parser.add_argument(
        "--recon-loss-fn",
        type=str,
        default="BaselineLoss",
        help="Reconstruction loss function: 'BaselineLoss' (default) or 'JukeboxPerceptualLoss'",
    )

    parser.add_argument(
        "--jukebox-pixel-loss-type",
        type=str,
        default="mse",
        choices=["mse", "l1"],
        help="Distance used on the pixel reconstruction term inside JukeboxPerceptualLoss. "
        "'mse' (default) matches the original formulation; 'l1' is more robust to outliers.",
    )

    parser.add_argument(
        "--bt-lambda",
        type=float,
        default=0.005,
        help="Barlow Twins off-diagonal weight (redundancy reduction). "
        "Only used when --contrastive-loss-type barlow_twins. Default: 0.005.",
    )
    parser.add_argument(
        "--vicreg-sim-coeff",
        type=float,
        default=25.0,
        help="VICReg invariance (MSE) coefficient. Default: 25.0.",
    )
    parser.add_argument(
        "--vicreg-std-coeff",
        type=float,
        default=25.0,
        help="VICReg variance (hinge) coefficient. Default: 25.0.",
    )
    parser.add_argument(
        "--vicreg-cov-coeff",
        type=float,
        default=1.0,
        help="VICReg covariance (decorrelation) coefficient. Default: 1.0.",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="vqvae",
        choices=["vae", "vqvae"],
        help="Encoder architecture: vae or vqvae",
    )
    # VQ-VAE-2 specific
    parser.add_argument("--vqvae-hidden-channels", type=int, default=64)
    parser.add_argument("--vqvae-res-channels", type=int, default=32)
    parser.add_argument("--vqvae-nb-levels", type=int, default=3)
    parser.add_argument("--vqvae-embed-dim", type=int, default=32)
    parser.add_argument(
        "--vqvae-nb-entries",
        type=int,
        nargs="+",
        default=[384],
        help="Codebook size(s) for the content codebooks. Pass a single int to broadcast "
        "to all levels, or one int per level (length must equal --vqvae-nb-levels). "
        "E.g. '--vqvae-nb-entries 512' or '--vqvae-nb-entries 256 384 512'.",
    )
    parser.add_argument("--vqvae-scaling-rates", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--vq-commitment-weight", type=float, default=0.25)
    parser.add_argument(
        "--content-style-levels",
        type=int,
        nargs="+",
        default=[0],
        help="Encoder levels at which to apply the learnable content/style Gumbel mask. "
        "Default: [0] (finest level only). Use e.g. '0 1 2' for all levels.",
    )
    parser.add_argument(
        "--content-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Per-level content ratio (fraction of hidden_channels that are content). "
        "One float per entry in --content-style-levels, same order. "
        "E.g. '--content-style-levels 0 1 2 --content-ratios 0.5 0.75 0.9' gives "
        "level 0 → 50%% content, level 1 → 75%%, level 2 → 90%%. "
        "If omitted, all levels use the global ratio from --content-dim / --total-dim.",
    )
    parser.add_argument(
        "--separate-encoders",
        action="store_true",
        default=False,
        help="Use separate encoder stacks per view (one VQVAE encoder per modality). "
        "Codebooks, decoders, and Gumbel content masks remain shared. "
        "Consistent with the view-specific encoder identifiability theory (Yao et al., 2024).",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="onthefly",
        choices=["learned", "onthefly", "fixed", "learned_split"],
        help="How the content/style Gumbel mask logits are determined. "
        "'learned': learnable nn.Parameter per level (and per view when --separate-encoders is set). "
        "'onthefly': mask logits computed on-the-fly from average encoder activations, "
        "shared across views (matches the original multiview-crl repo). "
        "'fixed': first K channels are content, rest are style — no learning, no Gumbel noise. "
        "Eliminates mask instability and MoCo queue inconsistency. "
        "'learned_split': per-channel sigmoid gates that learn which channels are content vs "
        "style AND how many. The content/style split size is not fixed — it emerges from "
        "training. Initialized near the ratio from --content-dim/--total-dim. "
        "Default: onthefly.",
    )
    parser.add_argument(
        "--quantize-style",
        action="store_true",
        default=False,
        help="Quantize style channels through independent per-level codebooks (Option A). "
        "Requires --inject-style-to-decoder. When active, style channels are vector-quantized "
        "before injection into the decoder, giving style its own discrete bottleneck.",
    )
    parser.add_argument(
        "--style-dropout-prob",
        type=float,
        default=0.0,
        help="Per-sample, per-level probability of zeroing the style tensor before it is "
        "injected into the decoder during training. Forces the decoder to reconstruct from "
        "content alone on a fraction of samples, pressuring content to carry anatomy. "
        "No expectation-rescaling. 0.0 disables (default). Typical values: 0.1–0.5. "
        "Only active when --inject-style-to-decoder is set.",
    )
    parser.add_argument(
        "--style-embed-dim",
        type=int,
        default=None,
        help="Embedding dimension for style codebooks. Defaults to the main --embed-dim.",
    )
    parser.add_argument(
        "--style-nb-entries",
        type=int,
        nargs="+",
        default=None,
        help="Number of codebook entries for style codebooks. Pass a single int to broadcast "
        "to all masked levels, or one int per masked level (length must equal "
        "len(--content-style-levels)). Defaults to the matching content codebook size per level.",
    )
    parser.add_argument(
        "--cb-ema-decay",
        type=float,
        default=0.999,
        help="EMA momentum for codebook running averages (cluster_size and embed_avg). "
        "Higher values (e.g. 0.999) give smoother updates suited for small batches. "
        "Lower values (e.g. 0.99) adapt faster but can be noisy. Default: 0.999.",
    )
    parser.add_argument(
        "--cb-reset-every",
        type=int,
        default=100,
        help="Reset dead codebook entries every N forward passes per codebook. "
        "Dead entries are those with EMA cluster_size below --cb-reset-threshold. "
        "Set to 0 to disable. Default: 100.",
    )
    parser.add_argument(
        "--cb-reset-threshold",
        type=float,
        default=1.0,
        help="EMA cluster_size below this value marks a codebook entry as dead. Default: 1.0.",
    )
    parser.add_argument(
        "--cross-view-negs-only",
        action="store_true",
        default=False,
        help="Use only cross-view negatives in the contrastive loss (InfoNCE and MoCo). "
        "When set, same-view samples are excluded from the negative set, forcing the "
        "model to align representations across views rather than relying on within-view "
        "instance discrimination. Recommended when using --separate-encoders.",
    )
    parser.add_argument(
        "--patch-contrastive",
        action="store_true",
        default=False,
        help="Use patch-level (dense) contrastive alignment instead of global average "
        "pooling. Pools spatial maps into a grid of patches and aligns corresponding "
        "patches across views, preserving spatial correspondence.",
    )
    parser.add_argument(
        "--patch-grid",
        type=int,
        nargs=3,
        default=[4, 5, 4],
        help="Spatial grid size (D, H, W) for patch-level contrastive loss. "
        "Only used when --patch-contrastive is set. Default: 4 5 4 (~80 patches).",
    )
    parser.add_argument(
        "--patch-grid-per-level",
        type=int,
        nargs="+",
        default=None,
        help="Per-level spatial grid for patch-level contrastive loss. Flat list of "
        "3*nb_levels ints (D0 H0 W0 D1 H1 W1 ...). When set, overrides --patch-grid. "
        "Useful when finer levels need smaller patches than coarser ones to avoid "
        "trivially-similar neighbouring voxels. E.g. '--patch-grid-per-level 2 3 2 "
        "3 4 3 4 5 4' for a 3-level model (coarsest → finest).",
    )
    parser.add_argument(
        "--contrastive-level-weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-level weight for the contrastive loss, one per VQ-VAE level. "
        "E.g. '--contrastive-level-weights 3.0 1.0 1.0' upweights level 0 by 3x. "
        "If omitted, all levels are weighted equally (1.0).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory in residual blocks",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile (mode=max-autotune) for kernel fusion (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels_last_3d memory format for the VQ-VAE (faster 3D convs on A100+)",
    )
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="Pre-process and cache all volumes in RAM (avoids repeated disk I/O)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for persistent preprocessed-volume cache (.pt files).  "
            "When set together with --cache-dataset, volumes are written to disk "
            "on the first run and memory-mapped on subsequent runs, cutting startup "
            "from minutes to seconds.  Defaults to None (RAM-only cache)."
        ),
    )
    parser.add_argument(
        "--skip-recon-ratio",
        type=float,
        default=0.0,
        help="Fraction of steps to skip reconstruction (0–1)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Run validation every N training steps and log Val/ losses to TensorBoard. "
        "0 disables periodic validation (default).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps (effective batch = batch_size × N)",
    )
    # Learning rate schedule
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Linear LR warmup steps (0 to disable warmup)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="LR schedule after warmup: cosine annealing or constant",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Minimum LR for cosine annealing (default: decay to zero)",
    )
    # Image preprocessing
    parser.add_argument("--image-spacing", type=float, default=2.0, help="Isotropic voxel spacing in mm")
    parser.add_argument("--crop-margin", type=int, default=0, help="Voxels to crop from each edge")
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Explicit spatial size (D H W) for input volumes after resampling. "
        "Overrides the size derived from --image-spacing and --crop-margin. "
        "Example: --spatial-size 80 96 80",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="gumbel_softmax",
        choices=["ground_truth", "gumbel_softmax", "concat", "soft"],
    )
    parser.add_argument("--n-views", default=2, type=int)
    parser.add_argument("--change-lists", default=[[4, 5, 6, 8, 9, 10]])
    parser.add_argument("--faiss-omp-threads", type=int, default=16)
    parser.add_argument("--subsets", default=[(0, 1), (0, 2), (1, 2), (0, 1, 2)])
    parser.add_argument(
        "--recon-loss-start-step",
        type=int,
        default=0,
        help="Training step at which to start applying the reconstruction loss",
    )
    # MoCo
    parser.add_argument(
        "--inject-style-to-decoder",
        action="store_true",
        help=(
            "Append style embedding dims (those not selected as content by the Gumbel mask "
            "in the embed_dim space) to the final decoder layer before the output conv.  "
            "Requires --content-dim / --total-dim to be set.  Has no effect when "
            "content/style separation is not configured."
        ),
    )
    parser.add_argument(
        "--style-injection-mode",
        type=str,
        default="concat",
        choices=["concat", "film"],
        help=(
            "How style features are injected into the decoder.  "
            "'concat' (default): style is concatenated onto the penultimate feature map "
            "before the final conv — simple but style only influences the last layer.  "
            "'film': Spatial FiLM (Feature-wise Linear Modulation) — style modulates "
            "the decoder feature map via learned per-location scale and shift after every "
            "decoder stage (residual block + each upsampling step), giving the decoder "
            "access to style information at every resolution.  "
            "Requires --inject-style-to-decoder."
        ),
    )
    parser.add_argument(
        "--use-moco",
        action="store_true",
        help="Use MoCo momentum-contrast training for the VQ-VAE encoder",
    )
    parser.add_argument(
        "--moco-queue-size",
        type=int,
        default=4096,
        help="Number of negatives stored per level in the MoCo queue",
    )
    parser.add_argument(
        "--moco-momentum",
        type=float,
        default=0.999,
        help="EMA momentum coefficient for the MoCo momentum encoder",
    )
    parser.add_argument(
        "--mask-warmup-steps",
        type=int,
        default=0,
        help="Number of initial training steps during which the MoCo queue is disabled "
        "and in-batch InfoNCE is used instead, allowing the learned content/style mask "
        "to stabilise before queue negatives are introduced.  Only relevant when "
        "--mask-mode is 'learned' or 'learned_split' AND --use-moco is set. "
        "After warmup the queue is flushed and MoCo resumes normally. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--mask-lr-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the Gumbel mask parameters (channel_logits). "
        "A value < 1 (e.g. 0.1) slows mask evolution relative to the encoder, reducing "
        "staleness in the MoCo queue.  Only relevant for --mask-mode learned/learned_split. "
        "Default: 1.0 (same LR as the encoder).",
    )
    # Evaluation
    parser.add_argument("--eval-dci", action="store_true")
    parser.add_argument("--eval-style", action="store_true")
    parser.add_argument("--grid-search-eval", action="store_true")
    parser.add_argument(
        "--content-dim",
        type=int,
        default=128,
        help="Number of content dimensions (ratio with total-dim determines embed_dim split)",
    )
    parser.add_argument(
        "--total-dim",
        type=int,
        default=512,
        help="Total number of dimensions (ratio with content-dim determines embed_dim split)",
    )
    parser.add_argument(
        "--content-size",
        type=int,
        default=None,
        help="Directly set the number of content channels (out of --vqvae-hidden-channels). "
        "Overrides the ratio derived from --content-dim / --total-dim. "
        "E.g. '--content-size 48 --vqvae-hidden-channels 64' → 48 content, 16 style channels. "
        "Useful for tuning spatial map alignment.",
    )
    parser.add_argument(
        "--use-content-projection",
        action="store_true",
        help="Use content projection in the VQ-VAE encoder",
    )
    parser.add_argument(
        "--narrow-encoder-input",
        action="store_true",
        help="Narrow the encoder input to content channels only (ablation for testing the importance of style information in the encoder)",
    )
    parser.add_argument(
        "--top-level-recon-only",
        action="store_true",
        default=False,
        help="Zero out encoder outputs at non-top levels before the codebook, so "
        "reconstruction depends only on the coarsest (top) level embedding. "
        "Encoder features are still used for the contrastive loss.",
    )
    parser.add_argument(
        "--skip-decoder-concat-levels",
        type=int,
        nargs="+",
        default=None,
        help="Levels whose quantized code contributions are zeroed out in the input "
        "to the FINAL (level-0) decoder, so they do not influence the reconstructed "
        "image. Intermediate decoders are unaffected (their outputs still condition "
        "finer codebooks). The top (coarsest) level cannot be skipped — at least one "
        "level must contribute. "
        "Examples: '--skip-decoder-concat-levels 0' drops only the finest level; "
        "'--skip-decoder-concat-levels 0 1' drops the two finest, leaving only the "
        "top codes to drive reconstruction.",
    )
    # Weights & Biases
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (requires wandb to be installed).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="multiview-crl-sweep",
        help="W&B project name. Default: multiview-crl-sweep.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B team/entity name. Uses default entity if not set.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B group name. Use to bundle seeds/variants of the same experiment "
        "for analysis (e.g. --wandb-group phase1-L0).",
    )
    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training if the monitored loss does not improve for this many "
        "checkpoint intervals. 0 disables early stopping (default). "
        "When --val-every is set, monitors validation loss; otherwise monitors "
        "the rolling average training loss.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in monitored loss to count as progress. "
        "Only used when --early-stopping-patience > 0. Default: 0.0.",
    )
    parser.add_argument(
        "--no-resumable-sampler",
        action="store_true",
        default=False,
        help="Disable the ResumableSampler and fall back to DataLoader(shuffle=True). "
        "Use to A/B-test whether the ResumableSampler (added Apr 25) is responsible "
        "for run-to-run reproducibility regressions vs pre-Apr-25 baselines. With "
        "this flag set, the sample order is drawn from the global torch RNG (as "
        "before), which also restores the random stream consumed by augmentations, "
        "Gumbel noise, dropout, etc. Mid-epoch resume continuity is lost.",
    )
    parser.add_argument(
        "--shared-brain-mask",
        action="store_true",
        default=False,
        help="Intersect the T1 and T2 brain masks into a single shared mask before "
        "applying it to both modalities. Eliminates the modality-specific image "
        "boundary that the patch-contrastive objective can otherwise pick up as a "
        "low-level modality cue (visible as edge-of-skull hot spots on the per-patch "
        "modality probe).",
    )
    parser.add_argument(
        "--asymmetric-aug",
        action="store_true",
        default=False,
        help="Apply independent intensity augmentations per view (T1 and T2 get different "
        "random draws for shift/scale/bias-field/gamma/noise/smooth). Spatial augmentations "
        "remain synchronised across views so anatomical correspondence is preserved for "
        "patch-level contrastive alignment. Motivated by the multi-view identifiability "
        "framework (Yao et al., 2024): an intensity-augmented T1 is effectively a third "
        "view, shrinking the shared-content block to truly modality-invariant anatomy.",
    )
    parser.add_argument(
        "--pass-full-to-next-level",
        action="store_true",
        default=False,
        help="When content/style separation is active, pass the FULL (unmasked) "
        "encoder output to the next encoder level instead of zeroing out style "
        "channels. The content/style split still applies to the codebook input "
        "and the contrastive loss — only the inter-level encoder path is affected. "
        "Incompatible with --narrow-encoder-input and --use-content-projection.",
    )

    return parser


def update_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Populate dataset-specific fields on ``args`` (subsets, content/style indices, etc.).

    Args:
        args: Parsed argument namespace (mutated in-place and returned).

    Returns:
        argparse.Namespace: The updated namespace.
    """
    import logging

    logger = logging.getLogger("multiview_crl")
    logger.info(f"Configuring dataset: {args.dataset_name}")

    # Warn when nb_levels=1 with content/style separation but no style decoder path.
    # In this configuration, style channels receive zero gradient — the codebook
    # only sees content channels, and there are no higher encoder levels to route
    # style through.  --inject-style-to-decoder gives style a gradient path.
    _nb_levels = getattr(args, "vqvae_nb_levels", 3)
    _has_cs = getattr(args, "content_dim", 0) > 0 and getattr(args, "total_dim", 0) > getattr(args, "content_dim", 0)
    _cs_levels = getattr(args, "content_style_levels", [0])
    _all_levels_masked = _has_cs and set(_cs_levels) == set(range(_nb_levels))
    if (_nb_levels == 1 or _all_levels_masked) and _has_cs and not getattr(args, "inject_style_to_decoder", False):
        logger.warning(
            "Content/style separation is active but style channels have NO gradient path! "
            f"(nb_levels={_nb_levels}, content_style_levels={_cs_levels}, "
            "inject_style_to_decoder=False). "
            "The codebook only sees content channels, and there are no unmasked encoder "
            "levels to route style through. Style channels will be dead (zero gradient). "
            "Consider adding --inject-style-to-decoder to give style a reconstruction "
            "gradient path through the decoder."
        )

    # --patch-grid-per-level: flat list → list of (D, H, W) tuples, one per level.
    _pgpl = getattr(args, "patch_grid_per_level", None)
    if _pgpl is not None:
        if not getattr(args, "patch_contrastive", False):
            logger.warning("--patch-grid-per-level is set but --patch-contrastive is not; it will be ignored.")
            args.patch_grid_per_level = None
        elif len(_pgpl) != 3 * _nb_levels:
            raise ValueError(
                f"--patch-grid-per-level expects 3*nb_levels={3 * _nb_levels} ints "
                f"(nb_levels={_nb_levels}), got {len(_pgpl)}."
            )
        else:
            args.patch_grid_per_level = [tuple(_pgpl[3 * i : 3 * i + 3]) for i in range(_nb_levels)]

    # --mask-mode learned_split is incompatible with --inject-style-to-decoder
    # because the number of style channels varies per forward pass.
    if getattr(args, "mask_mode", "onthefly") == "learned_split" and getattr(args, "inject_style_to_decoder", False):
        raise ValueError(
            "--mask-mode learned_split is incompatible with --inject-style-to-decoder "
            "because the number of style channels varies per forward pass. "
            "Use --mask-mode fixed or learned instead."
        )

    # Warn if MoCo is enabled with a negative-free loss (it'll be ignored)
    _cl_type = getattr(args, "contrastive_loss_type", "infonce")
    if _cl_type in ("barlow_twins", "vicreg") and getattr(args, "use_moco", False):
        logger.warning(
            f"--use-moco is set but --contrastive-loss-type is '{_cl_type}' which does not "
            f"use negatives.  MoCo queue and momentum encoder will be disabled."
        )
        args.use_moco = False

    # --content-size: directly set content channels, override ratio-based defaults
    if getattr(args, "content_size", None) is not None:
        hidden_ch = args.vqvae_hidden_channels
        cs = args.content_size
        assert 1 <= cs < hidden_ch, f"--content-size must be in [1, {hidden_ch - 1}], got {cs}"
        ratio = cs / hidden_ch
        # Override content_dim / total_dim to be consistent with the chosen ratio
        args.content_dim = cs
        args.total_dim = hidden_ch
        # Set content_ratios for all content_style_levels
        cs_levels = getattr(args, "content_style_levels", [0])
        args.content_ratios = [ratio] * len(cs_levels)
        logger.info(
            f"  --content-size={cs}: content_ratio={ratio:.3f} "
            f"({cs}/{hidden_ch} channels) applied to levels {cs_levels}"
        )

    if args.dataset_name == "independent3di":
        args.DATASETCLASS = datasets.Indepdenent3DIdent
        setattr(args, "modalities", ["image"])
        logger.info("  -> Using Independent3DIdent dataset (image only)")
    elif args.dataset_name == "causal3di":
        args.DATASETCLASS = datasets.Causal3DIdent
        setattr(args, "modalities", ["image"])
        logger.info("  -> Using Causal3DIdent dataset (image only)")
    elif args.dataset_name == "multimodal3di":
        args.DATASETCLASS = datasets.Multimodal3DIdent
        setattr(args, "modalities", ["image", "text"])
        logger.info("  -> Using Multimodal3DIdent dataset (image + text)")
    elif args.dataset_name == "mpi3d":
        args.DATASETCLASS = datasets.MPI3D
        setattr(args, "modalities", ["image"])
        assert args.n_views == 2, "mpi3d only considers pairs of views: n-views=2"
        setattr(args, "n-views", 2)
        setattr(args, "subsets", [(0, 1)])
        setattr(args, "change_lists", [])
        setattr(args, "collate_random_pair", True)
    elif args.dataset_name == "custom":
        args.DATASETCLASS = datasets.MyCustomDataset
        setattr(args, "modalities", ["image"])
        setattr(args, "n_views", 2)
        setattr(args, "subsets", [(0, 1)])
        logger.info("  -> Using custom dataset (image only, 2 views)")
    elif args.dataset_name == "synthetic":
        args.DATASETCLASS = datasets.SyntheticBrainDataset
        setattr(args, "modalities", ["image"])
        setattr(args, "n_views", 2)
        setattr(args, "subsets", [(0, 1)])
        setattr(args, "content_indices", [list(range(args.content_dim))])
        setattr(args, "style_indices", list(range(args.content_dim, args.total_dim)))
        # Default the model's spatial_size to the synthetic resolution so the
        # VAE decoder isn't sized for ADNI volumes (91,109,91) when the inputs
        # are actually 32x32x32. VQ-VAE infers shape from input and is unaffected.
        if getattr(args, "spatial_size", None) is None:
            res = getattr(args, "synthetic_res", 32)
            setattr(args, "spatial_size", (res, res, res))
            logger.info(f"  -> Auto-set --spatial-size to ({res}, {res}, {res}) from --synthetic-res")
        logger.info("  -> Using synthetic dataset (pseudo-MRI, 2 views)")
        logger.info(f"  -> Content dimensions: 0-{args.content_dim - 1} ({args.content_dim} dims)")
        logger.info(
            f"  -> Style dimensions: {args.content_dim}-{args.total_dim - 1} ({args.total_dim - args.content_dim} dims)"
        )
    elif args.dataset_name in ["adni", "ADNI_registered", "ADNI_stripped", "ADNI_stripped_masks"]:
        args.DATASETCLASS = datasets.MyCustomDataset
        setattr(args, "modalities", ["image"])
        setattr(args, "n_views", 2)
        setattr(args, "subsets", [(0, 1)])
        setattr(args, "content_indices", [list(range(args.content_dim))])
        setattr(args, "style_indices", list(range(args.content_dim, args.total_dim)))
        logger.info("  -> Using ADNI dataset (image only, 2 views)")
        logger.info(f"  -> Content dimensions: 0-{args.content_dim - 1} ({args.content_dim} dims)")
        logger.info(
            f"  -> Style dimensions: {args.content_dim}-{args.total_dim - 1} ({args.total_dim - args.content_dim} dims)"
        )
    else:
        raise ValueError(f"{args.dataset_name=} not supported.")

    if len(args.subsets) == 1 or args.n_views == 2:
        setattr(args, "subsets", [tuple(range(args.n_views))])
        if not hasattr(args, "content_indices") or args.content_indices is None:
            setattr(args, "content_indices", [list(range(args.encoding_size))])
        logger.info(f"  -> Training content encoders with {args.n_views} views")
        logger.info(f"  -> Subsets: {args.subsets}")
        logger.info(f"  -> Content indices: {len(args.content_indices[0])} dimensions")
    else:
        if not hasattr(args, "subsets"):
            subsets, _ = utils.powerset(range(args.n_views))
            setattr(args, "subsets", subsets)

        assert max(set().union(*args.subsets)) < args.n_views, "The given view is outside boundary!"

        if args.selection in ["ground_truth", "gumbel_softmax"]:
            content_indices = compute_gt_idx(args)
            setattr(args, "content_indices", content_indices)
            setattr(args, "encoding_size", len(args.DATASETCLASS.FACTORS["image"]))
        elif args.selection == "concat":
            assert args.encoding_size > len(args.subsets)
            est_content_indices = np.array_split(range(args.encoding_size), len(args.subsets))
            setattr(args, "content_indices", [ind.tolist() for ind in est_content_indices])

        content_union = set().union(*args.content_indices)
        style_indices = [i for i in range(args.encoding_size) if i not in content_union]
        setattr(args, "style_indices", style_indices)

    return args


def compute_gt_idx(args: argparse.Namespace) -> list:
    """
    Compute ground-truth content indices for supervised datasets.

    Args:
        args: Parsed argument namespace.

    Returns:
        list: Per-subset list of content channel indices.
    """
    factors = args.DATASETCLASS.FACTORS["image"].keys()

    if args.dataset_name in ["independent3di", "causal3di"]:
        if args.dataset_name == "independent3di":
            setattr(args, "change_lists", [[4, 5, 6, 8, 9]])
        elif args.dataset_name == "causal3di":
            setattr(args, "change_lists", [[8, 9, 10], [1, 2, 3, 4, 5, 6, 7]])
        content_dict = {}
        indicators = [[True] * len(factors)]
        for _, change in enumerate(args.change_lists):
            indicators.append([z not in change for z in factors])
        for s in args.subsets:
            content_dict[s] = np.where(list(functools.reduce(operator.eq, [np.array(indicators[k]) for k in s])))[
                0
            ].tolist()
        return list(content_dict.values())

    elif args.dataset_name == "multimodal3di":
        setattr(args, "change_lists", [[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        content_dict = {}
        indicators = [[True] * len(factors)]
        for _, change in enumerate(args.change_lists):
            indicators.append([z not in change for z in factors])
        indicators.append([True] * 3)
        for s in args.subsets:
            indicators_copy = indicators.copy()
            if 2 in s:
                indicators_copy = [ind[: len(indicators[-1])] for ind in indicators]
            content_dict[s] = np.where(list(functools.reduce(operator.eq, [np.array(indicators_copy[k]) for k in s])))[
                0
            ].tolist()
        print(content_dict)
        return list(content_dict.values())

    else:
        raise ValueError(f"No ground truth content computed for {args.dataset_name=} yet!")
