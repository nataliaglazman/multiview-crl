"""Encoder-only multi-view contrastive learning on the 3D synthetic data.

3D encoders trained with InfoNCE or Barlow Twins, without reconstruction.
The default ``conv`` architecture uses the VQ-VAE convolutional backbone with
an affine readout. ``--encoder-architecture resnet18`` uses a 3D adaptation of
the upstream image encoder: ResNet-18 -> GAP -> Linear -> LeakyReLU -> Linear.
This architecture option does not change the loss, data, or view-sharing policy.
``--global-pool attention`` replaces the GAP before either readout with multi-head
attention pooling (``models.attention_pool``), which starts as exact GAP.

Identifiability is scored with ``eval.metrics.dci.compute_dci_synthetic`` (per-latent
RidgeCV/GBT R² + block-MCC + content→view leakage): content latents → high,
independent style → ~chance is the block-identification signal.

Example:
    python -m training.main_conv_synthetic --model-id conv_c9 \
        --latent-dim 16 --content-channels 9 --n-content 9 --n-style 3 \
        --train-steps 50000 --eval-every 2000 --contrastive-loss-type infonce
"""

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import eval.metrics.dci as dci
import training.losses as losses
from data.datasets import SyntheticBrainDataset
from models.multiview_encoder import MultiviewConvEncoder
from utils.encoder_runtime import configure_encoder_runtime, select_encoder_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--model-id", type=str, default="conv_synthetic")

    # Model.
    p.add_argument(
        "--encoder-architecture",
        choices=("conv", "resnet18"),
        default="conv",
        help="conv: original VQ-VAE backbone; resnet18: upstream ResNet-18 architecture adapted to 3D",
    )
    p.add_argument(
        "--encoder-head-hidden",
        type=int,
        default=100,
        help="MLP readout hidden width (ResNet, or conv with --conv-readout mlp)",
    )
    p.add_argument("--conv-readout", choices=("linear", "mlp"), default="linear")
    p.add_argument(
        "--separate-spatial-readout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an independent content-only patch MLP, copied from the global head at initialization",
    )
    p.add_argument(
        "--global-pool",
        choices=("gap", "attention"),
        default="gap",
        help="How the global encoding pools the backbone map. attention: multi-head attention pooling with a "
        "Fourier positional encoding, initialized to exact GAP without drawing random numbers, so the "
        "untrained floor matches the GAP run's. Patch readouts still average each bin.",
    )
    p.add_argument(
        "--attention-pool-heads",
        type=int,
        default=4,
        help="Attention heads; each pools its own slice of the backbone channels, which it must divide",
    )
    p.add_argument(
        "--attention-pool-frequencies",
        type=int,
        default=4,
        help="Fourier frequencies per axis in the positional encoding. 0 leaves attention permutation-"
        "invariant over positions, like GAP, so it cannot report where a structure is",
    )
    p.add_argument(
        "--lesion-keypoints",
        type=int,
        default=0,
        help="Lesion branch: spatial-softmax keypoint heads whose (x, y, z) are appended to the content "
        "code and trained by their own InfoNCE (models.keypoint_pool); 0 disables",
    )
    p.add_argument(
        "--lesion-norm",
        choices=("none", "layer"),
        default="none",
        help="With --lesion-keypoints: per-voxel LayerNorm before the keypoint logits",
    )
    p.add_argument(
        "--lesion-frame",
        choices=("brain", "grid"),
        default="brain",
        help="With --lesion-keypoints: coordinates relative to the input brain's centroid/spread, or absolute",
    )
    p.add_argument(
        "--lesion-proj-dim", type=int, default=8, help="With --lesion-keypoints: output size of its loss projector"
    )
    p.add_argument(
        "--lesion-loss-weight", type=float, default=1.0, help="With --lesion-keypoints: weight of its InfoNCE"
    )
    p.add_argument(
        "--lesion-pairing",
        choices=("cross_modal", "within_modality"),
        default="cross_modal",
        help="With --lesion-keypoints: what the branch's own InfoNCE pairs. cross_modal: the T1/FLAIR pair, "
        "like the content block. within_modality: two intensity-augmented copies of FLAIR through the FLAIR "
        "encoder, so the lesion keeps one polarity in both views (it is darker than white matter in T1 and "
        "brighter in FLAIR)",
    )
    p.add_argument(
        "--lesion-input",
        choices=("features", "residual"),
        default="features",
        help="With --lesion-keypoints: what the heads read. residual: each view's brighter/darker-than-normal "
        "residual against a low-rank normative model fitted on training subjects (models.normative_residual)",
    )
    p.add_argument(
        "--lesion-normative-components",
        type=int,
        default=20,
        help="With --lesion-input residual: principal components the normative model explains away",
    )
    p.add_argument(
        "--lesion-normative-subjects",
        type=int,
        default=300,
        help="With --lesion-input residual: training subjects for the normative model (2/3 fit, 1/3 calibrate)",
    )
    p.add_argument(
        "--lesion-head-init",
        choices=("positive", "random", "negative"),
        default="positive",
        help="With --lesion-input residual: signs of the heads' initial weights on the brighter- and "
        "darker-than-normal channels. positive: every head starts attending to anomalies; random: plain "
        "random signs; negative: every head starts avoiding them, the control for whether training finds "
        "the lesion without that head start",
    )
    p.add_argument(
        "--lesion-temperature",
        type=float,
        default=1.0,
        help="With --lesion-keypoints: divides the keypoint logits; below 1 makes the heads sharp",
    )
    p.add_argument(
        "--lesion-decorrelation-weight",
        type=float,
        default=0.0,
        help="With --lesion-keypoints: penalty on the squared correlation between the keypoint coordinates "
        "and the (detached) content block, so the branch carries what the content does not already encode",
    )
    p.add_argument(
        "--lesion-detector",
        choices=("shared", "separate", "separate_conv"),
        default="shared",
        help="Residual branch: shared 1x1 heads, per-view 1x1 heads, or per-view two-layer 16-channel conv detectors",
    )
    p.add_argument(
        "--lesion-branch-frozen",
        action="store_true",
        help="Freeze residual detector and projector; global encoder still trains",
    )
    p.add_argument(
        "--lesion-localization-eval",
        action="store_true",
        help="At each DCI evaluation report every head's native voxel error and hit rate (no fitted probe)",
    )
    p.add_argument("--resnet-norm", choices=("batch", "group"), default="batch")
    p.add_argument(
        "--norm-type",
        choices=("group", "layer"),
        default="group",
        help="Conv encoder norm. group (legacy) pools each sample's statistics over all positions, "
        "so they reach background cells; layer normalizes channels at each voxel independently",
    )
    p.add_argument(
        "--resnet-output-stride",
        type=int,
        choices=(8, 16, 32),
        default=32,
        help="Remove late-stage downsampling for stride 8/16; preserve stem, kernels and channels; no dilation",
    )
    p.add_argument("--latent-dim", type=int, default=16, help="Total encoding size (content + style)")
    p.add_argument("--content-channels", type=int, default=9, help="Content units (set to the true n_content)")
    p.add_argument("--hidden-channels", type=int, default=64, help="conv architecture only")
    p.add_argument("--res-channels", type=int, default=32, help="conv architecture only")
    p.add_argument("--nb-res-layers", type=int, default=2, help="conv architecture only")
    p.add_argument("--downscale-factor", type=int, default=4, help="conv downscale (power of 2)")
    p.add_argument("--no-separate-encoders", action="store_true", help="Share one encoder across both views")

    # Contrastive loss (content alignment − entropy).
    p.add_argument("--contrastive-loss-type", type=str, default="infonce", choices=["infonce", "barlow_twins"])
    p.add_argument("--tau", type=float, default=1.0, help="InfoNCE temperature")
    p.add_argument("--bt-lambda", type=float, default=0.005, help="Barlow Twins off-diagonal weight")
    p.add_argument(
        "--patch-loss-weight",
        type=float,
        default=0.0,
        help="Add this weight times spatial InfoNCE to global InfoNCE; 0 preserves global-only training",
    )
    p.add_argument(
        "--train-patch-grid",
        type=int,
        nargs=3,
        default=[8, 8, 8],
        help="Training-only patch grid; must fit the backbone map when --patch-loss-weight > 0",
    )
    # Same names, default and rule as utils/config.py's VQ-VAE flags.
    p.add_argument(
        "--patch-foreground-mask",
        action="store_true",
        help="Drop always-background positions from the patch InfoNCE: each batch, the brain mask is "
        "pooled to --train-patch-grid and a position is kept if any sample has at least "
        "--patch-foreground-thresh brain there. At res 64, ~77%% of an 8³ grid is background per subject",
    )
    p.add_argument(
        "--patch-foreground-thresh",
        type=float,
        default=0.05,
        help="Brain fraction a patch position needs in at least one batch sample to stay in the patch loss",
    )
    p.add_argument(
        "--cross-view-negs-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="InfoNCE negatives only from the other view (paper aligns across views)",
    )
    p.add_argument(
        "--contrastive-proj-dim",
        type=int,
        default=0,
        help="If > 0, insert an MLP head between the pooled content block and the "
        "contrastive loss. The loss is computed on the head's output while the DCI "
        "probes keep reading the pre-head encoding (the SimCLR/MoCo recipe — the "
        "loss-facing space over-compresses toward view-invariance and loses "
        "linear-probe info). 0 (default) disables the head: the loss acts directly "
        "on the representation being scored.",
    )
    p.add_argument(
        "--contrastive-proj-hidden",
        type=int,
        default=256,
        help="Hidden width of the projection head MLP (Linear -> ReLU -> Linear). "
        "Only used when --contrastive-proj-dim > 0.",
    )

    # Optimisation.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64, help="Per-view batch size")
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=2.0, help="Max grad 2-norm (paper uses 2); 0 disables")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=None, help="Generator seed; defaults to --seed")
    p.add_argument("--model-seed", type=int, default=None, help="Initialization seed; defaults to --seed")
    p.add_argument("--loader-seed", type=int, default=None, help="Independent shuffle seed; defaults to --seed + 10000")
    p.add_argument("--require-new-run", action="store_true", help="Refuse to overwrite an existing run directory")
    p.add_argument("--deterministic", action="store_true", help="Require deterministic PyTorch operations")
    p.add_argument(
        "--deterministic-warn-only",
        action="store_true",
        help="With --deterministic, warn instead of failing for unsupported operations (e.g. CUDA MaxPool3d backward)",
    )
    p.add_argument(
        "--cpu-threads", type=int, default=None, help="Pin CPU rendering/encoding threads; restored by evaluation"
    )
    p.add_argument(
        "--hash-training-inputs", action="store_true", help="Record a streaming hash of all actual training images"
    )
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto keeps the legacy CUDA/CPU selection; mps needs PYTORCH_ENABLE_MPS_FALLBACK=1 exported "
        "before Python starts (ResNet MaxPool3d runs on CPU)",
    )

    # Evaluation pooling — GAP is the paper-faithful default; patch probes whether
    # content survives at spatial resolution (see groupnorm-caps-gap-pooled-mcc).
    p.add_argument(
        "--floor-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one eval before training and keep it as the untrained floor, so every "
        "later per-factor score prints its delta against it. Costs one extra eval.",
    )
    p.add_argument(
        "--best-metric",
        type=str,
        default="block_mcc",
        choices=["block_mcc", "ridge_r2", "none"],
        help="Validation metric that decides which checkpoint is kept as model_best.pt. "
        "Floor-subtracted when --floor-eval gave a floor, since the raw value is mostly "
        "floor. 'none' keeps only the last-step model.pt.",
    )
    p.add_argument("--eval-pooling", type=str, default="gap", choices=["gap", "patch"])
    p.add_argument("--eval-patch-grid", type=int, nargs=3, default=[4, 5, 4])
    p.add_argument(
        "--spatial-recovery-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add frozen backbone/content spatial probes for both views at each evaluation",
    )
    p.add_argument(
        "--spatial-recovery-grids",
        type=int,
        nargs="+",
        help="Cubic probe grids; defaults to GAP plus the training patch grid (or up to 8 for global-only runs)",
    )
    p.add_argument(
        "--spatial-recovery-native",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also probe native maps; more CPU work and temporary disk space",
    )
    p.add_argument("--spatial-recovery-batch-size", type=int, default=4)
    p.add_argument("--spatial-recovery-test-samples", type=int, default=400)
    p.add_argument("--spatial-recovery-seed", type=int, default=1729)

    # Synthetic dataset (forwarded in full to train AND val so the distributions match).
    p.add_argument("--res", type=int, default=32, help="Cubic resolution (power of 2)")
    p.add_argument("--n-content", type=int, default=9, help="True shared content factors")
    p.add_argument("--n-style", type=int, default=3, help="Per-view style factors")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--num-val-samples", type=int, default=400)
    p.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold rendered volumes in RAM (default). The cache is (samples x 2 views x res^3 x 4B) "
        "per split once full -- 23.5 GB for 1000 train + 400 val at res 128, which the OOM killer "
        "takes out on the first eval. Pass --no-cache at high res to re-render each sample instead: "
        "slower per step, constant memory.",
    )
    p.add_argument("--synthetic-mode", type=str, default="pseudo_mri")
    p.add_argument(
        "--synthetic-normalize", type=str, default="per_sample", choices=["per_sample", "shared", "fixed_reference"]
    )
    p.add_argument(
        "--synthetic-clean-content",
        action="store_true",
        help="Zero the unlabeled deformation/fissure nuisance so the named content factors dominate",
    )
    p.add_argument(
        "--synthetic-identifiable-ventricle",
        action="store_true",
        help="The VQ-VAE recipe's ventricle: radius 0.12-0.28 instead of 0.10-0.20, read off the undeformed "
        "radius so no other factor reshapes it, and a fissure with its own label and intensity instead of "
        "ventricle CSF. Off reproduces old encoder-only runs",
    )
    p.add_argument(
        "--synthetic-sulcal-mode",
        choices=("corrugation", "atrophy"),
        default="corrugation",
        help="What sulcal_widening (z_content[8]) does. corrugation reproduces old runs: the signed depth of a "
        "fixed zero-mean corrugation, which cancels under GAP. atrophy widens fixed sulcal clefts, turning 7-28%% "
        "of the cortex into CSF without moving the brain outline, white matter or ventricles",
    )
    p.add_argument("--synthetic-style-scale", type=float, default=1.0)
    p.add_argument("--synthetic-content-scale", type=float, default=1.0)
    p.add_argument(
        "--synthetic-lesion-placement",
        choices=("legacy", "wm_interior"),
        default="legacy",
        help="wm_interior places a full fixed-radius sphere inside final WM labels; legacy reproduces old runs",
    )
    p.add_argument(
        "--synthetic-lesion-radius",
        type=float,
        default=0.1,
        help="Sphere-mode lesion radius in [-1,1] coords (0.1 = 1.6 voxels at res=32)",
    )
    p.add_argument(
        "--synthetic-lesion-intensity",
        choices=("fixed", "styled"),
        default="fixed",
        help="styled passes the lesion through the acquisition gain/bias like every tissue (T1 contrast "
        "0.4*gain); fixed reproduces old runs, whose T1 lesion nearly vanishes at low gain",
    )
    p.add_argument(
        "--synthetic-lesion-target",
        choices=("position", "burden"),
        default="position",
        help="What the lesion content factors are. position: z_content[2:5] place one sphere (lesion_x/y/z). "
        "burden: z_content[2] sets the total volume of --synthetic-lesion-count spheres at nuisance positions, "
        "an amount like ventricle size and the synthetic counterpart of a WMH volume; z_content[3:5] are unused. "
        "Needs --synthetic-lesion-placement wm_interior; --synthetic-lesion-radius is then the largest radius",
    )
    p.add_argument(
        "--synthetic-lesion-count",
        type=int,
        default=4,
        help="With --synthetic-lesion-target burden: lesions per subject sharing the burden",
    )
    p.add_argument(
        "--synthetic-lesion-t1-value",
        type=float,
        default=0.4,
        help="T1 lesion base intensity on the tissue LUT (bg 0, CSF 0.1, WM 0.8, GM 0.5; FLAIR's lesion is "
        "1.0). The default 0.4 is darker than WM but close to GM. 1.6 (above every tissue, the same sign as "
        "FLAIR) and 0.0 (below every tissue) both differ from WM by 0.8, so they separate the lesion's sign "
        "from how distinct it is",
    )
    p.add_argument("--synthetic-n-deformation-grid", type=int, default=4)
    p.add_argument("--synthetic-n-fissure-grid", type=int, default=8)
    p.add_argument("--synthetic-hierarchical-content", action="store_true")
    p.add_argument("--synthetic-causal", action="store_true")
    # Same names, defaults and choices as utils/config.py's copies, so the two entrypoints
    # describe the same generator rather than drifting apart.
    p.add_argument(
        "--synthetic-causal-graph",
        type=str,
        default="chain",
        choices=["chain", "full", "random"],
        help="DAG topology for the content SCM. 'chain' is a single path (n-1 edges, one "
        "parent each) and is the easy case for PC; 'random' gives multi-parent nodes.",
    )
    p.add_argument(
        "--synthetic-causal-edge-prob",
        type=float,
        default=0.5,
        help="Edge probability for random DAG, in [0, 1] (ignored for chain/full)",
    )
    p.add_argument(
        "--synthetic-causal-noise-scale",
        type=float,
        default=0.4,
        help="Additive noise scale in causal mechanisms.",
    )
    p.add_argument(
        "--synthetic-causal-nonlinearity",
        type=str,
        default="leaky_relu",
        choices=["leaky_relu", "none"],
        help="Nonlinearity in causal mechanisms.",
    )
    args = p.parse_args(argv)
    if args.no_cuda and args.device not in ("auto", "cpu"):
        p.error("--no-cuda forces CPU; it cannot be combined with --device cuda/mps")
    for name, default in (("data_seed", args.seed), ("model_seed", args.seed), ("loader_seed", args.seed + 10000)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.train_steps < 1 or args.eval_every < 1 or args.batch_size < 2:
        p.error("Need positive train/evaluation steps and at least two subjects per batch")
    if args.num_train_samples < args.batch_size:
        p.error("--num-train-samples must allow at least one full training batch")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    if args.deterministic_warn_only and not args.deterministic:
        p.error("--deterministic-warn-only requires --deterministic")
    if not 0 < args.synthetic_lesion_radius < float("inf"):
        p.error("--synthetic-lesion-radius must be finite and positive")
    if args.synthetic_lesion_target == "burden":
        if args.synthetic_lesion_placement != "wm_interior":
            p.error("--synthetic-lesion-target burden needs --synthetic-lesion-placement wm_interior")
        if args.synthetic_lesion_count < 1:
            p.error("--synthetic-lesion-count must be positive")
        if args.lesion_keypoints > 0:
            p.error("The lesion branch localises one lesion; it does not apply to --synthetic-lesion-target burden")
    elif args.synthetic_lesion_count != 4:
        p.error("--synthetic-lesion-count only applies with --synthetic-lesion-target burden")
    if not 0 <= args.synthetic_lesion_t1_value < float("inf"):
        p.error("--synthetic-lesion-t1-value must be finite and nonnegative")
    if not 0.0 <= args.synthetic_causal_edge_prob <= 1.0:
        p.error("--synthetic-causal-edge-prob must be between 0 and 1")
    if not 0 <= args.patch_loss_weight < float("inf"):
        p.error("--patch-loss-weight must be finite and nonnegative")
    if args.patch_foreground_mask and args.patch_loss_weight <= 0:
        p.error("--patch-foreground-mask requires a positive --patch-loss-weight")
    if not 0 < args.patch_foreground_thresh <= 1:
        p.error("--patch-foreground-thresh must be in (0, 1]")
    if args.separate_spatial_readout:
        if args.encoder_architecture == "conv" and args.conv_readout != "mlp":
            p.error("--separate-spatial-readout requires an MLP readout (--conv-readout mlp for Conv)")
        if args.patch_loss_weight <= 0:
            p.error("--separate-spatial-readout requires a positive --patch-loss-weight")
        if args.contrastive_proj_dim != 0:
            p.error("--separate-spatial-readout requires --contrastive-proj-dim 0 to avoid a shared loss projector")
    if args.patch_loss_weight > 0:
        if args.contrastive_loss_type != "infonce":
            p.error("Patch training currently supports global + patch InfoNCE only")
        if not 0 < args.tau < float("inf"):
            p.error("Patch InfoNCE requires a finite positive --tau")
        stride = args.resnet_output_stride if args.encoder_architecture == "resnet18" else args.downscale_factor
        if stride < 1:
            p.error("Encoder stride must be positive")
        spatial = (args.res + stride - 1) // stride if args.encoder_architecture == "resnet18" else args.res // stride
        if any(g < 1 or g > spatial for g in args.train_patch_grid):
            p.error(f"--train-patch-grid must fit the {spatial}^3 backbone map")
    if args.encoder_architecture == "resnet18":
        if args.conv_readout != "linear":
            p.error("--conv-readout only applies to --encoder-architecture conv")
        if args.norm_type != "group":
            p.error("--norm-type only applies to --encoder-architecture conv; use --resnet-norm for ResNet")
        if args.encoder_head_hidden <= 0:
            p.error("--encoder-head-hidden must be positive")
        spatial_size = (args.res + args.resnet_output_stride - 1) // args.resnet_output_stride
        if args.eval_pooling == "patch" and any(g < 1 or g > spatial_size for g in args.eval_patch_grid):
            p.error(f"ResNet at --res {args.res} has a {spatial_size}^3 map; --eval-patch-grid must fit it")
    else:
        if args.resnet_norm != "batch" or args.resnet_output_stride != 32:
            p.error("--resnet-norm and --resnet-output-stride only apply to --encoder-architecture resnet18")
        if args.conv_readout == "mlp" and args.encoder_head_hidden <= 0:
            p.error("--encoder-head-hidden must be positive")
    if args.global_pool == "attention":
        channels = 512 if args.encoder_architecture == "resnet18" else args.hidden_channels
        if args.attention_pool_heads < 1 or channels % args.attention_pool_heads:
            p.error(f"--attention-pool-heads must divide the {channels} backbone channels")
        if args.attention_pool_frequencies < 0:
            p.error("--attention-pool-frequencies must be nonnegative")
        stride = args.resnet_output_stride if args.encoder_architecture == "resnet18" else args.downscale_factor
        if stride < 1:
            p.error("Encoder stride must be positive")
        spatial = (args.res + stride - 1) // stride if args.encoder_architecture == "resnet18" else args.res // stride
        if spatial < 2:
            p.error(f"--global-pool attention needs more than one position; the backbone map is {spatial}^3")
    elif (args.attention_pool_heads, args.attention_pool_frequencies) != (4, 4):
        p.error("--attention-pool-heads and --attention-pool-frequencies only apply to --global-pool attention")
    if args.lesion_keypoints < 0:
        p.error("--lesion-keypoints must be nonnegative")
    if (args.lesion_detector != "shared" or args.lesion_branch_frozen) and (
        args.lesion_input != "residual" or args.lesion_keypoints < 1
    ):
        p.error("Separate/frozen detectors require --lesion-input residual and --lesion-keypoints")
    if args.lesion_detector == "separate_conv" and args.lesion_head_init != "positive":
        p.error("--lesion-head-init sign controls only apply to simple residual detectors")
    if args.lesion_localization_eval and (args.lesion_keypoints < 1 or args.synthetic_mode != "pseudo_mri"):
        p.error("--lesion-localization-eval requires a pseudo_mri lesion branch")
    if args.lesion_keypoints > 0:
        if args.contrastive_loss_type != "infonce":
            p.error("The lesion branch trains with its own InfoNCE; use --contrastive-loss-type infonce")
        if not 0 < args.tau < float("inf"):
            p.error("The lesion branch's InfoNCE requires a finite positive --tau")
        if args.lesion_proj_dim < 1:
            p.error("--lesion-proj-dim must be positive")
        if not 0 <= args.lesion_loss_weight < float("inf"):
            p.error("--lesion-loss-weight must be finite and nonnegative")
        if not 0 <= args.lesion_decorrelation_weight < float("inf"):
            p.error("--lesion-decorrelation-weight must be finite and nonnegative")
        if not 0 < args.lesion_temperature < float("inf"):
            p.error("--lesion-temperature must be finite and positive")
        if args.lesion_head_init != "positive" and args.lesion_input != "residual":
            p.error("--lesion-head-init only applies with --lesion-input residual")
        if args.lesion_input == "residual":
            if args.lesion_pairing != "cross_modal":
                p.error(
                    "--lesion-input residual needs cross_modal pairs: augmented intensities break the normative model"
                )
            if args.lesion_normative_components < 1:
                p.error("--lesion-normative-components must be positive")
            if not args.lesion_normative_components + 2 <= (2 * args.lesion_normative_subjects) // 3:
                p.error(
                    "--lesion-normative-subjects must fit more than --lesion-normative-components subjects (2/3 of it)"
                )
            if args.lesion_normative_subjects > args.num_train_samples:
                p.error("--lesion-normative-subjects cannot exceed --num-train-samples")
        stride = args.resnet_output_stride if args.encoder_architecture == "resnet18" else args.downscale_factor
        if stride < 1:
            p.error("Encoder stride must be positive")
        spatial = (args.res + stride - 1) // stride if args.encoder_architecture == "resnet18" else args.res // stride
        if spatial < 2:
            p.error(f"--lesion-keypoints needs more than one position; the backbone map is {spatial}^3")
    elif (
        args.lesion_norm,
        args.lesion_frame,
        args.lesion_proj_dim,
        args.lesion_loss_weight,
        args.lesion_decorrelation_weight,
        args.lesion_temperature,
        args.lesion_pairing,
        args.lesion_input,
        args.lesion_normative_components,
        args.lesion_normative_subjects,
        args.lesion_head_init,
    ) != ("none", "brain", 8, 1.0, 0.0, 1.0, "cross_modal", "features", 20, 300, "positive"):
        p.error(
            "--lesion-norm/-frame/-proj-dim/-loss-weight/-decorrelation-weight only apply with --lesion-keypoints > 0"
        )
    if args.spatial_recovery_eval:
        if args.n_content != 9 or args.synthetic_mode != "pseudo_mri":
            p.error("Spatial recovery requires the nine-factor pseudo_mri recipe")
        if args.num_val_samples < 20 or args.spatial_recovery_test_samples < 10 or args.spatial_recovery_batch_size < 1:
            p.error(
                "Spatial recovery needs >=20 validation subjects, >=10 diagnostic subjects and a positive batch size"
            )
        stride = args.resnet_output_stride if args.encoder_architecture == "resnet18" else args.downscale_factor
        if stride < 1:
            p.error("Encoder stride must be positive")
        spatial = (args.res + stride - 1) // stride if args.encoder_architecture == "resnet18" else args.res // stride
        if args.spatial_recovery_grids is None:
            if args.patch_loss_weight > 0:
                if len(set(args.train_patch_grid)) != 1:
                    p.error("For a non-cubic training grid, specify cubic --spatial-recovery-grids explicitly")
                args.spatial_recovery_grids = [args.train_patch_grid[0]]
            else:
                args.spatial_recovery_grids = [min(8, spatial)]
        args.spatial_recovery_grids = sorted(set([1, *args.spatial_recovery_grids]))
        if any(g < 1 or g > spatial for g in args.spatial_recovery_grids):
            p.error(f"Spatial recovery grids must fit the {spatial}^3 backbone map")
    return args


def cache_gb(res, num_samples):
    """RAM the dataset's in-memory cache will hold once every sample has been rendered."""
    return num_samples * 2 * (res**3) * 4 / 1e9


def make_dataset(args, mode, num_samples):
    """Single factory for train/val/test so the generative distribution is identical."""
    if args.cache:
        gb = cache_gb(args.res, num_samples)
        # The cache fills lazily, so an over-large one survives training and is killed
        # later, on the first eval that starts filling the val split's share of it.
        if gb > 4.0:
            print(
                f"  WARNING: {mode} cache will grow to {gb:.1f} GB in RAM "
                f"({num_samples} samples x 2 views x {args.res}^3 x 4B). "
                f"Pass --no-cache to re-render instead of caching.",
                flush=True,
            )
    return SyntheticBrainDataset(
        mode=mode,
        spatial_size=(args.res, args.res, args.res),
        cache=args.cache,
        synthetic_mode=args.synthetic_mode,
        synthetic_seed=getattr(args, "data_seed", None) if getattr(args, "data_seed", None) is not None else args.seed,
        synthetic_num_samples=num_samples,
        synthetic_n_content=args.n_content,
        synthetic_n_style=args.n_style,
        synthetic_style_scale=args.synthetic_style_scale,
        synthetic_content_scale=args.synthetic_content_scale,
        synthetic_n_deformation_grid=args.synthetic_n_deformation_grid,
        synthetic_n_fissure_grid=args.synthetic_n_fissure_grid,
        synthetic_hierarchical_content=args.synthetic_hierarchical_content,
        synthetic_normalize=args.synthetic_normalize,
        synthetic_causal=args.synthetic_causal,
        synthetic_causal_graph=getattr(args, "synthetic_causal_graph", "chain"),
        synthetic_causal_edge_prob=getattr(args, "synthetic_causal_edge_prob", 0.5),
        synthetic_causal_noise_scale=args.synthetic_causal_noise_scale,
        synthetic_causal_nonlinearity=args.synthetic_causal_nonlinearity,
        synthetic_clean_content=args.synthetic_clean_content,
        synthetic_identifiable_ventricle=getattr(args, "synthetic_identifiable_ventricle", False),
        synthetic_sulcal_mode=getattr(args, "synthetic_sulcal_mode", "corrugation"),
        synthetic_lesion_placement=getattr(args, "synthetic_lesion_placement", "legacy"),
        synthetic_lesion_radius=getattr(args, "synthetic_lesion_radius", 0.1),
        synthetic_lesion_intensity=getattr(args, "synthetic_lesion_intensity", "fixed"),
        synthetic_lesion_target=getattr(args, "synthetic_lesion_target", "position"),
        synthetic_lesion_count=getattr(args, "synthetic_lesion_count", 4),
        synthetic_lesion_t1_value=getattr(args, "synthetic_lesion_t1_value", 0.4),
    )


def contrastive_loss(pooled, model, args, sim_metric, criterion):
    """Content-alignment − entropy on the pooled content block across the two views.

    The content block is selected first and then projected, so with a head every output
    dimension is part of the loss-facing space and counts as content downstream. The head
    is applied here rather than handed to the losses' ``projector`` argument, which
    ``infonce_base_loss`` accepts but never calls. Without a head ``project`` is the
    identity and this is the same tensor the loss selected internally before.
    """
    b = pooled.shape[0] // 2
    hz = torch.stack([pooled[:b], pooled[b:]], dim=0)  # (2, B, C)
    hz = model.project(hz[..., : args.content_channels])
    content_indices = [list(range(hz.shape[-1]))]
    if args.contrastive_loss_type == "barlow_twins":
        loss = losses.barlow_twins_loss(
            hz, estimated_content_indices=content_indices, subsets=[(0, 1)], lambd=args.bt_lambda
        )
    else:
        loss = losses.infonce_loss(
            hz,
            sim_metric=sim_metric,
            criterion=criterion,
            tau=args.tau,
            estimated_content_indices=content_indices,
            subsets=[(0, 1)],
            cross_view_negs_only=args.cross_view_negs_only,
        )
    return loss.squeeze()


def augment_intensity(x):
    """One acquisition-like intensity draw per subject, applied inside the brain only.

    Contrast gain in [0.7, 1.3] about the brain mean, an offset within +/-0.2 brain SD and
    Gaussian noise up to 0.05 brain SD. A positive gain keeps the lesion's polarity, so two
    draws of FLAIR agree on it, unlike T1 vs FLAIR. Voxels outside the brain stay exactly
    zero, so the brain frame's support is unchanged.
    """
    brain = (x != 0).to(x.dtype)
    dims = tuple(range(1, x.dim()))
    count = brain.sum(dim=dims, keepdim=True).clamp_min(1)
    mean = (x * brain).sum(dim=dims, keepdim=True) / count
    sd = (((x - mean) * brain).square().sum(dim=dims, keepdim=True) / count).sqrt()
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    gain = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(0.7, 1.3)
    offset = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(-0.2, 0.2) * sd
    noise = torch.randn_like(x) * torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(0.0, 0.05) * sd
    return (mean + gain * (x - mean) + offset + noise) * brain


def lesion_contrastive_loss(pooled, model, args, sim_metric, criterion, images=None):
    """InfoNCE on the lesion branch's keypoint coordinates alone, through their own projector.

    Kept apart from the content InfoNCE on purpose: a lesion code mixed into the content
    readout is ignored once easier shared factors already identify the subjects (see the
    lesion-branch section of training/ENCODER_ARCHITECTURES.md). The coordinates sit after
    the latent_dim units of the global code; probes read them raw, the loss reads the projection.

    With ``--lesion-pairing within_modality`` the pair is two intensity-augmented copies of the
    FLAIR view (``images`` second half), both through the FLAIR encoder, instead of T1/FLAIR.
    """
    b = pooled.shape[0] // 2
    if getattr(args, "lesion_pairing", "cross_modal") == "within_modality":
        flair = images[b:]
        coords = model.lesion_code(torch.cat([augment_intensity(flair), augment_intensity(flair)]), view_idx=1)
        pair = (coords[:b], coords[b:])
    else:
        block = pooled[:, args.latent_dim :]
        pair = (block[:b], block[b:])
    hz = model.project_lesion(torch.stack(pair, dim=0))
    return losses.infonce_loss(
        hz,
        sim_metric=sim_metric,
        criterion=criterion,
        tau=args.tau,
        estimated_content_indices=[list(range(hz.shape[-1]))],
        subsets=[(0, 1)],
        cross_view_negs_only=args.cross_view_negs_only,
    ).squeeze()


def lesion_decorrelation(pooled, args):
    """Sum of squared correlations between the keypoint coordinates and the content block.

    Both are standardized over the batch (both views). The content side is detached, so it
    keeps what it encodes and only the branch moves. A branch that duplicates a factor the
    content already carries (brain size, captured through any boundary that moves with it)
    is penalized without the factor ever being named.
    """

    def standardize(z):
        return (z - z.mean(0)) / z.std(0).clamp_min(1e-6)

    coords = standardize(pooled[:, args.latent_dim :])
    content = standardize(pooled[:, : args.content_channels].detach())
    return ((coords.T @ content) / coords.shape[0]).square().sum()


def foreground_positions(foreground, grid, threshold):
    """Positions kept by --patch-foreground-mask: some sample has >= threshold brain there.

    ``foreground`` is the batch's brain masks, (images, 1, D, H, W). The rule matches
    main_multimodal's, so both trainers drop the same always-background positions. All
    subjects keep the same positions, and every position is kept if none qualifies.
    """
    with torch.no_grad():
        fraction = torch.nn.functional.adaptive_avg_pool3d(foreground.float(), tuple(grid)).flatten(1)
        keep = (fraction >= threshold).any(dim=0)
    return keep if bool(keep.any()) else torch.ones_like(keep)


def patch_contrastive_loss(patches, model, args, sim_metric, criterion, foreground=None):
    """Registered positions are paired; negatives are subjects at the SAME position.

    Only image-derived features and the input's own brain mask enter this loss; there
    is no lesion sampling or access to generator factor labels. All grid positions
    participate unless --patch-foreground-mask drops the always-background ones.
    """
    if getattr(args, "patch_foreground_mask", False):
        if foreground is None:
            raise ValueError("--patch-foreground-mask needs the batch's brain masks")
        # The launcher's backend check passes raw options, which omit the parser's default.
        keep = foreground_positions(foreground, args.train_patch_grid, getattr(args, "patch_foreground_thresh", 0.05))
        patches = patches[..., keep.to(patches.device)]
    a, b = patches[:, : args.content_channels, :].chunk(2, dim=0)
    hz = torch.stack((a, b), dim=0)  # views, subjects, content channels, positions
    # A configured projector acts on channels, never on the position axis.
    hz = model.project(hz.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    return losses.patch_infonce_loss(
        hz,
        sim_metric=sim_metric,
        criterion=criterion,
        tau=args.tau,
        estimated_content_indices=[list(range(hz.shape[2]))],
        subsets=[(0, 1)],
        cross_view_negs_only=args.cross_view_negs_only,
    ).squeeze()


def training_objective(model, images, args, sim_metric, criterion, masks=None):
    """Keep the old global path exact when local training is disabled.

    ``masks`` are the batch's brain masks in image order, used by --patch-foreground-mask.
    Without them the inputs' nonzero support stands in; the synthetic inputs are zero
    outside the brain mask, so the two agree.
    """
    weight = getattr(args, "patch_loss_weight", 0.0)
    if weight > 0:
        pooled, patches = model.global_and_patch_features(images, args.train_patch_grid, n_views=2)
    else:
        pooled = model(images, pool_only=True, n_views=2)[2][0]
    global_loss = contrastive_loss(pooled, model, args, sim_metric, criterion)
    foreground = None
    if weight > 0 and getattr(args, "patch_foreground_mask", False):
        foreground = masks if masks is not None else images != 0
    patch_loss = (
        patch_contrastive_loss(patches, model, args, sim_metric, criterion, foreground)
        if weight > 0
        else global_loss.new_zeros(())
    )
    weighted = weight * patch_loss
    total = global_loss + weighted if weight > 0 else global_loss
    terms = {"global": global_loss, "patch": patch_loss, "patch_weighted": weighted}
    if getattr(args, "lesion_keypoints", 0) > 0:
        lesion_loss = lesion_contrastive_loss(pooled, model, args, sim_metric, criterion, images)
        terms["lesion"] = lesion_loss
        # The launcher's backend check passes raw options, which omit the parser's default.
        terms["lesion_weighted"] = getattr(args, "lesion_loss_weight", 1.0) * lesion_loss
        total = total + terms["lesion_weighted"]
        decorrelation_weight = getattr(args, "lesion_decorrelation_weight", 0.0)
        if decorrelation_weight > 0:
            terms["lesion_decorrelation"] = lesion_decorrelation(pooled, args)
            total = total + decorrelation_weight * terms["lesion_decorrelation"]
    return (pooled, total, terms)


def effective_rank(feat):
    """Participation ratio of the covariance spectrum: 1 = collapsed to a line, C = isotropic.

    The quantity InfoNCE quietly destroys when alignment is the only pressure on the
    representation. GAP already starts this near 1 on a random encoder, so a run whose
    rank never climbs is discarding information rather than aligning content, and every
    identifiability number it reports will sit at the untrained floor.
    """
    x = feat.detach().float()
    if x.device.type == "mps":
        # This small diagnostic is outside autograd; keep eigvalsh off MPS.
        x = x.cpu()
    x = x - x.mean(dim=0, keepdim=True)
    ev = torch.linalg.eigvalsh(torch.cov(x.T)).clamp(min=0)
    total = ev.sum()
    return float(total**2 / ev.pow(2).sum()) if total > 0 else 0.0


def per_factor_scores(results):
    """Per-factor ridge R² and block-MCC, keyed by block then factor name.

    Both are already computed inside ``compute_dci_synthetic``; this just reads them off
    the block detail dicts. Returned in a plain-dict shape so a step-0 call can be kept
    as the untrained floor and subtracted from every later eval.
    """
    out = {}
    for block in ("content→content", "content→style"):
        detail = results.get(f"{block}/detail")
        if not isinstance(detail, dict):
            continue
        names = detail.get("factor_names") or []
        ridge, mcc, mcc_std, chan = (
            detail.get(k)
            for k in ("per_factor_ridge", "per_factor_mcc", "per_factor_mcc_std", "per_factor_channel_mcc")
        )

        def at(arr, j):
            return float(arr[j]) if arr is not None and j < len(arr) else float("nan")

        out[block] = {
            nm: {"ridge": at(ridge, j), "mcc": at(mcc, j), "mcc_std": at(mcc_std, j), "chan": at(chan, j)}
            for j, nm in enumerate(names)
        }
    return out


def print_per_factor(scores, floor=None, writer=None, step=0):
    """One row per ground-truth factor: which factors the representation actually carries.

    The mean over factors hides exactly the case worth checking — a block that recovers
    two coarse global factors well and every localised one at chance reads as a decent
    average. Floor-subtracted where a step-0 eval is available, because raw per-factor R²
    has the same untrained-floor problem as the block means.
    """
    for block, rows in scores.items():
        if not rows:
            continue
        has_floor = bool(floor) and block in floor
        print(f"    --- per-factor recovery: {block} ---", flush=True)
        head = f"      {'factor':<20s}{'ridge R²':>9s}{'blockMCC':>10s}{'±':>7s}{'chanMCC':>9s}"
        print(head + (f"{'floor':>9s}{'Δ vs floor':>12s}" if has_floor else ""), flush=True)
        for nm, v in rows.items():
            line = (
                f"      {nm:<20s}{v['ridge']:>9.3f}{v['mcc']:>10.3f}{v['mcc_std']:>7.3f}"
                f"{v.get('chan', float('nan')):>9.3f}"
            )
            if has_floor and nm in floor[block]:
                fl = floor[block][nm]["ridge"]
                line += f"{fl:>9.3f}{v['ridge'] - fl:>+12.3f}"
            print(line, flush=True)
            if writer is not None:
                tag = block.replace("→", "_to_")
                writer.add_scalar(f"per_factor/{tag}/{nm}/ridge_r2", v["ridge"], step)
                writer.add_scalar(f"per_factor/{tag}/{nm}/mcc", v["mcc"], step)
                writer.add_scalar(f"per_factor/{tag}/{nm}/channel_mcc", v["chan"], step)


BEST_METRIC_KEYS = {"block_mcc": "content->content/block_mcc", "ridge_r2": "content->content/informativeness_ridge"}


def best_metric_value(flat, floor_flat, name):
    """The scalar model_best.pt is selected on, floor-subtracted where a floor exists.

    Raw block-MCC on this generator is mostly floor — an untrained encoder scores ~0.38 —
    so selecting on it would rank checkpoints partly by how much untrained structure the
    architecture happens to carry. The delta ranks them by what training added.
    """
    v = flat.get(BEST_METRIC_KEYS[name])
    if v is None or not np.isfinite(v):
        return None
    if floor_flat is not None:
        f = floor_flat.get(BEST_METRIC_KEYS[name])
        if f is not None and np.isfinite(f):
            return float(v - f)
    return float(v)


def attention_spread(model, dataset, device, batch_size):
    """Each global-attention head's effective positions as a fraction of the map, per view.

    exp(entropy) of a head's weights over the N backbone positions, divided by N and
    averaged over the first ``batch_size`` subjects. 1 is uniform, i.e. GAP, as at
    initialization; 1/N is a single position. None for a GAP model.
    """
    if model.attention_pool is None:
        return None
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    with torch.no_grad():
        weights = model.attention_maps(torch.cat(batch["image"], dim=0).to(device), n_views=2)
    weights = weights.flatten(2).cpu().double()
    fraction = torch.special.entr(weights).sum(-1).exp() / weights.shape[-1]
    b = fraction.shape[0] // 2
    return {"t1": fraction[:b].mean(0).tolist(), "flair": fraction[b:].mean(0).tolist()}


def holdout_r2(features, targets, ridge=1e-2):
    """Per-target R² of a ridge fit on the first half of the subjects, scored on the second.

    Features are standardized with the fit half's statistics and clipped at 5 SD: a keypoint
    head that barely moves has a near-zero spread, and without the clip one outlying test
    subject dominates the score.
    """
    n = features.shape[0] // 2
    mean, sd = features[:n].mean(0), features[:n].std(0).clip(min=1e-4)

    def design(z):
        return np.concatenate([np.clip((z - mean) / sd, -5, 5), np.ones((len(z), 1))], axis=1)

    fit, test = design(features[:n]), design(features[n:])
    weights = np.linalg.solve(fit.T @ fit + ridge * n * np.eye(fit.shape[1]), fit.T @ targets[:n])
    residual = ((test @ weights - targets[n:]) ** 2).sum(0)
    total = ((targets[n:] - targets[n:].mean(0)) ** 2).sum(0)
    return 1 - residual / np.maximum(total, 1e-12)


def clip_encoder_gradients(model, max_norm):
    """Keep the raw-residual and global paths independent during norm clipping too."""
    if model.normative is None:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    branch = [p for module in model.lesion_modules() for p in module.parameters()]
    branch_ids = {id(p) for p in branch}
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if id(p) not in branch_ids], max_norm)
    return torch.nn.utils.clip_grad_norm_(branch, max_norm)


def lesion_branch_report(model, dataset, device, batch_size, localization=False):
    """What the lesion keypoints alone encode, per view, plus each head's spread. None without the branch.

    Linear R² of every content factor from the raw keypoint coordinates (not the projector),
    so a branch that found the lesion shows lesion_x/y/z, and a branch captured by a factor
    that moves where attention lands shows brain_size or ventricle_size instead. Read it
    against the step-0 row: random heads already carry some position by chance.
    """
    if model.lesion_pool is None:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    coords = {"t1": [], "flair": []}
    factors, spread = [], None
    native, truth, offset = [], [], 0
    with torch.no_grad():
        for batch in loader:
            x = torch.cat(batch["image"], dim=0).to(device)
            code = model(x, pool_only=True, n_views=2)[2][0][:, model.latent_dim :].cpu().double().numpy()
            b = code.shape[0] // 2
            coords["t1"].append(code[:b])
            coords["flair"].append(code[b:])
            factors.append(batch["gt_latents"]["z_content"].double().numpy())
            if spread is None or localization:
                maps = model.lesion_maps(x, n_views=2)
                weights = maps.flatten(2).cpu().double()
                fraction = torch.special.entr(weights).sum(-1).exp() / weights.shape[-1]
                if spread is None:
                    spread = {"t1": fraction[:b].mean(0).tolist(), "flair": fraction[b:].mean(0).tolist()}
                if localization:
                    from eval.lesion.lesion_detector_audit import native_positions, true_centroid

                    positions = native_positions(maps, x.shape[2:]).cpu().numpy()
                    native.append(np.stack([positions[:b], positions[b:]], axis=1))
                    truth.extend(true_centroid(dataset, idx) for idx in range(offset, offset + b))
            offset += b
    factors = np.concatenate(factors)
    names = dci.content_factor_names(factors.shape[1], dci.dataset_lesion_target(dataset))
    r2 = {view: dict(zip(names, holdout_r2(np.concatenate(c), factors).tolist())) for view, c in coords.items()}
    report = {"r2": r2, "effective_fraction": spread}
    if localization:
        from eval.lesion.lesion_detector_audit import localization_metrics

        predictions, targets = np.concatenate(native), np.asarray(truth)
        report["localization"] = {
            view: [localization_metrics(targets, predictions[:, v, head]) for head in range(predictions.shape[2])]
            for v, view in enumerate(("t1", "flair"))
        }
    if model.normative is not None and model.lesion_adapter is None:
        # Each head's logit per unit of brighter- and darker-than-normal residual: whether it
        # attends to anomalies (both positive), avoids them, or reads only one polarity.
        pool = model.lesion_pool
        report["head_weights"] = (pool.logits.weight.detach().flatten(1).cpu() / pool.temperature).tolist()
        report["head_weights_by_view"] = {
            view: (p.logits.weight.detach().flatten(1).cpu() / p.temperature).tolist()
            for view, p in (("t1", pool), ("flair", model.lesion_pool_v1 or pool))
        }
    return report


def evaluate(model, val_dataset, device, args, save_dir, step, writer=None, floor=None):
    print(f"  [eval] synthetic DCI @ step {step} ...", flush=True)
    pooling = "gap" if args.eval_pooling == "gap" else tuple(args.eval_patch_grid)
    results = dci.compute_dci_synthetic(
        encoder=model,
        dataset=val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=0,
        pooling=pooling,
        per_encoder=not args.no_separate_encoders,
    )
    flat = dci.flatten_dci_results(results)

    def show(key):
        for k in flat:
            if k.endswith(key) and np.isfinite(flat[k]):
                print(f"      {k:60s} {flat[k]:.3f}", flush=True)

    print("    --- identifiability summary ---", flush=True)
    show("content->content/block_mcc")
    show("content->content/channel_mcc")
    show("content->content/informativeness_ridge")
    show("content->style/block_mcc")
    show("content->view/acc")

    # DCI was already being computed here every eval and thrown away unprinted.
    print("    --- DCI (GBT importances) ---", flush=True)
    for _k in ("disentanglement", "completeness", "informativeness_test"):
        show(f"content->content/{_k}")

    scores = per_factor_scores(results)
    print_per_factor(scores, floor=floor, writer=writer, step=step)

    # Attention pooling starts as GAP; this shows whether, and per head how far, it left it.
    spread = attention_spread(model, val_dataset, device, args.batch_size)
    if spread is not None:
        print("    --- global attention: effective positions / map size, per head (1 = GAP) ---", flush=True)
        for view, values in spread.items():
            print(f"      {view:<8s}" + "".join(f"{v:8.3f}" for v in values), flush=True)
            if writer is not None:
                for head, value in enumerate(values):
                    writer.add_scalar(f"attention_pool/{view}/head{head}_effective_fraction", value, step)

    # The lesion branch is only useful if its heads found the lesion rather than a factor
    # that moves where attention lands; its coordinates alone say which.
    lesion = lesion_branch_report(model, val_dataset, device, args.batch_size, args.lesion_localization_eval)
    if lesion is not None:
        print("    --- lesion branch alone: linear R² per content factor (T1 / FLAIR) ---", flush=True)
        for name in lesion["r2"]["t1"]:
            print(f"      {name:<24s}{lesion['r2']['t1'][name]:8.3f}{lesion['r2']['flair'][name]:8.3f}", flush=True)
        print("    --- lesion keypoints: effective positions / map size, per head ---", flush=True)
        for view, values in lesion["effective_fraction"].items():
            print(f"      {view:<8s}" + "".join(f"{v:8.3f}" for v in values), flush=True)
        if "head_weights" in lesion:
            print("    --- lesion heads: logit per unit residual z [brighter, darker] ---", flush=True)
            for view, weights in lesion["head_weights_by_view"].items():
                print(f"      {view:<8s}" + " ".join(f"[{b:+.1f} {d:+.1f}]" for b, d in weights), flush=True)
        if "localization" in lesion:
            print("    --- native lesion localisation: mean error (vox) / hit <=3 vox; per head ---", flush=True)
            for view, metrics in lesion["localization"].items():
                print(
                    f"      {view:<8s}"
                    + "  ".join(
                        f"h{k}: {m['mean_error_vox']:.3f} / {m['hit_within_3_vox']:.3f}" for k, m in enumerate(metrics)
                    ),
                    flush=True,
                )
        if writer is not None:
            for view, values in lesion["r2"].items():
                for name, value in values.items():
                    writer.add_scalar(f"lesion_branch/{view}/r2_{name}", value, step)

    if writer is not None:
        for k, v in flat.items():
            if np.isfinite(v):
                writer.add_scalar(f"dci_synthetic/{k}", v, step)

    payload = {k: float(v) for k, v in flat.items()}
    payload["per_factor"] = scores  # nested, so the existing flat keys stay at top level
    if spread is not None:
        payload["attention_effective_fraction"] = spread
    if lesion is not None:
        payload["lesion_branch"] = lesion
    with open(os.path.join(save_dir, f"dci_step{step}.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    return flat, scores


def main():
    args = parse_args()
    configure_encoder_runtime(vars(args))
    device = select_encoder_device(args.device, args.no_cuda)
    save_dir = os.path.join(args.out_dir, args.model_id)
    os.makedirs(save_dir, exist_ok=not args.require_new_run)
    with open(os.path.join(save_dir, "settings.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    print(f"device: {device}", flush=True)
    for k, v in vars(args).items():
        print(f"\t{k}: {v}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = make_dataset(args, "train", args.num_train_samples)
    val_dataset = make_dataset(args, "val", args.num_val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.loader_seed),
    )

    # Dataset construction resets the global RNG. Seed initialization explicitly,
    # while the loader's private generator isolates batch order from architecture.
    torch.manual_seed(args.model_seed)
    model = MultiviewConvEncoder(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        res_channels=args.res_channels,
        nb_res_layers=args.nb_res_layers,
        downscale_factor=args.downscale_factor,
        latent_dim=args.latent_dim,
        content_channels=args.content_channels,
        separate_encoders=not args.no_separate_encoders,
        proj_dim=args.contrastive_proj_dim,
        proj_hidden=args.contrastive_proj_hidden,
        encoder_architecture=args.encoder_architecture,
        encoder_head_hidden=args.encoder_head_hidden,
        conv_readout=args.conv_readout,
        resnet_norm=args.resnet_norm,
        resnet_output_stride=args.resnet_output_stride,
        separate_spatial_readout=args.separate_spatial_readout,
        global_pool=args.global_pool,
        attention_pool_heads=args.attention_pool_heads,
        attention_pool_frequencies=args.attention_pool_frequencies,
        norm_type=args.norm_type,
        lesion_keypoints=args.lesion_keypoints,
        lesion_norm=args.lesion_norm,
        lesion_frame=args.lesion_frame,
        lesion_proj_dim=args.lesion_proj_dim,
        lesion_temperature=args.lesion_temperature,
        lesion_input=args.lesion_input,
        lesion_normative_components=args.lesion_normative_components,
        lesion_input_size=args.res,
        lesion_head_init=args.lesion_head_init,
        lesion_detector=args.lesion_detector,
        lesion_branch_frozen=args.lesion_branch_frozen,
    ).to(device)
    if model.normative is not None:
        # The first training subjects, in order, so a run's normative model is reproducible.
        n = args.lesion_normative_subjects
        images = torch.stack([torch.stack(train_dataset[i]["image"])[:, 0] for i in range(n)])
        model.fit_normative(images[: (2 * n) // 3], images[(2 * n) // 3 :])
        del images
        print(
            f"lesion normative model: {args.lesion_normative_components} components per view from "
            f"{(2 * n) // 3} training subjects, residual SD from {n - (2 * n) // 3} more",
            flush=True,
        )
    pool_name = "attention pool" if model.attention_pool is not None else "GAP"
    if args.encoder_architecture == "resnet18":
        print(
            f"encoder: 3D ResNet-18, {args.resnet_norm} norm, stride {args.resnet_output_stride}, "
            f"{pool_name} -> 512 -> {args.encoder_head_hidden} -> {args.latent_dim}; "
            f"{'separate' if model.separate_encoders else 'shared'} view backbone(s). "
            "The conv-only width, residual-layer and downscale flags are inactive.",
            flush=True,
        )
    elif args.conv_readout == "mlp":
        print(
            f"encoder: conv, {pool_name} -> {args.hidden_channels} -> {args.encoder_head_hidden} -> {args.latent_dim}"
        )
    if model.attention_pool is not None:
        pool = model.attention_pool
        print(
            f"global pool: attention replaces GAP; {pool.num_heads} heads x {pool.channels // pool.num_heads} "
            f"channels, {pool.num_frequencies} Fourier frequencies/axis, "
            f"{sum(p.numel() for p in pool.parameters())} parameters; starts as exact GAP. "
            "Patch readouts still average each bin.",
            flush=True,
        )
    if model.lesion_pool is not None:
        print(
            f"lesion detector: {args.lesion_detector}; "
            f"{'frozen' if args.lesion_branch_frozen else 'trainable'}; "
            f"gradient clipping {'separate from global' if model.normative is not None else 'joint'}",
            flush=True,
        )
        print(
            f"lesion branch: {args.lesion_keypoints} spatial-softmax keypoints ({args.lesion_frame} frame, "
            f"norm {args.lesion_norm}, temperature {args.lesion_temperature:g}, {args.lesion_pairing} pairs, "
            f"reads {args.lesion_input}"
            + (f", {args.lesion_head_init} head init" if args.lesion_input == "residual" else "")
            + ") -> "
            f"{model.lesion_units} "
            f"content units after the {args.latent_dim} "
            f"global units; own InfoNCE x {args.lesion_loss_weight:g} via a {model.lesion_units} -> "
            f"{args.lesion_proj_dim} projector"
            + (
                f"; decorrelation from the content block x {args.lesion_decorrelation_weight:g}"
                if args.lesion_decorrelation_weight > 0
                else ""
            ),
            flush=True,
        )
    if model.projector is not None:
        print(
            f"projection head: {args.content_channels} -> {args.contrastive_proj_hidden} -> "
            f"{args.contrastive_proj_dim} (loss runs here; probes read the {args.latent_dim}-d encoding)",
            flush=True,
        )
    if args.patch_loss_weight > 0:
        readout = (
            f"separate spatial MLP ({args.content_channels} channels, copied global initialization)"
            if args.separate_spatial_readout
            else "shared global/spatial readout"
        )
        positions = (
            f"positions with >= {args.patch_foreground_thresh:g} brain in some batch image"
            if args.patch_foreground_mask
            else "all positions"
        )
        print(
            f"objective: global InfoNCE + {args.patch_loss_weight:g} * patch InfoNCE; "
            f"grid={args.train_patch_grid}; {readout}; both losses update the backbones; {positions}; no target labels",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    torch.save(model.state_dict(), os.path.join(save_dir, "model_init.pt"))
    batch_order = hashlib.sha256()
    input_images = hashlib.sha256() if args.hash_training_inputs else None
    started = time.perf_counter()
    last_loss_terms = None

    def save_progress(step, status):
        payload = {
            "status": status,
            "step": step,
            "subjects_seen_per_view": step * args.batch_size,
            "batch_order_sha256": batch_order.hexdigest(),
            "training_input_sha256": input_images.hexdigest() if input_images is not None else None,
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "loader_seed": args.loader_seed,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "backbone_stride": model.backbone_stride,
            "readout": model.readout_type,
            "separate_spatial_readout": args.separate_spatial_readout,
            "spatial_content_channels": args.content_channels,
            "spatial_readout_parameter_count": (
                sum(p.numel() for p in model.spatial_readout.parameters()) if model.spatial_readout is not None else 0
            ),
            "normalization": model.normalization,
            "global_pool": model.global_pool,
            "attention_pool_parameter_count": (
                sum(p.numel() for p in model.attention_pool.parameters()) if model.attention_pool is not None else 0
            ),
            "optimizer": "AdamW",
            "optimizer_defaults": optimizer.defaults,
            "elapsed_seconds_including_evaluation": time.perf_counter() - started,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "device": str(device),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") if device == "mps" else None,
            "patch_loss_weight": args.patch_loss_weight,
            "train_patch_grid": (args.train_patch_grid if args.patch_loss_weight > 0 else None),
            "last_loss_terms": last_loss_terms,
        }
        if model.lesion_pool is not None:
            payload["lesion_branch_parameter_count"] = sum(
                p.numel() for part in model.lesion_modules() for p in part.parameters()
            )
            payload["lesion_branch_trainable_parameter_count"] = sum(
                p.numel() for part in model.lesion_modules() for p in part.parameters() if p.requires_grad
            )
        path = os.path.join(save_dir, "training_progress.json")
        with open(path + ".tmp", "w") as fp:
            json.dump(payload, fp, indent=2)
        os.replace(path + ".tmp", path)

    save_progress(0, "running")

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
    except Exception:
        writer = None

    # Step-0 eval doubles as the untrained floor: same architecture, same seed, no training.
    # Per-factor R² needs it more than the block means do -- a localised factor can read
    # 0.2 from a random encoder, so the raw number alone cannot say whether it was learned.
    floor = floor_flat = None
    spatial_initial = None
    if args.spatial_recovery_eval:
        from eval.encoder.spatial_recovery_monitor import evaluate_spatial_recovery
    if args.floor_eval:
        model.eval()
        floor_flat, floor = evaluate(model, val_dataset, device, args, save_dir, 0, writer)
        if args.spatial_recovery_eval:
            spatial_initial = evaluate_spatial_recovery(model, vars(args), device, save_dir, 0, writer=writer)
        model.train()

    # Step 0 is the untrained floor, not a candidate: with every delta at or below zero it
    # would win on a tie and save an untrained encoder as "best".
    best = {"value": None, "step": None}

    step = 0
    running = dict(
        loss=0.0, rank=0.0, n=0, global_loss=0.0, patch=0.0, patch_weighted=0.0, lesion=0.0, decorrelation=0.0
    )
    model.train()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            batch_order.update(batch["index"].cpu().numpy().astype("<i8").tobytes())
            images = torch.cat(batch["image"], dim=0)
            if input_images is not None:
                input_images.update(images.contiguous().numpy().tobytes())
            x = images.to(device)  # (2B, 1, res, res, res)
            masks = torch.cat(batch["mask"], dim=0).to(device) if args.patch_foreground_mask else None
            if masks is not None and step == 0:
                kept = int(foreground_positions(masks, args.train_patch_grid, args.patch_foreground_thresh).sum())
                print(f"  patch foreground mask: first batch keeps {kept}/{np.prod(args.train_patch_grid)} positions")

            pooled, loss, terms = training_objective(model, x, args, sim_metric, criterion, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                clip_encoder_gradients(model, args.grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            last_loss_terms = {key: value.detach().item() for key, value in terms.items()}
            last_loss_terms["total"] = loss.detach().item()
            running["global_loss"] += last_loss_terms["global"]
            running["patch"] += last_loss_terms["patch"]
            running["patch_weighted"] += last_loss_terms["patch_weighted"]
            running["lesion"] += last_loss_terms.get("lesion", 0.0)
            running["decorrelation"] += last_loss_terms.get("lesion_decorrelation", 0.0)
            running["rank"] += effective_rank(pooled[: pooled.shape[0] // 2, : args.content_channels])
            running["n"] += 1
            step += 1

            if step % args.log_every == 0:
                n = max(running["n"], 1)
                print(
                    f"step {step:6d} | contrastive {running['loss']/n:.4f} "
                    f"| content eff_rank {running['rank']/n:.2f}/{args.content_channels}"
                    + (
                        f" | global {running['global_loss']/n:.4f} | patch {running['patch']/n:.4f} "
                        f"| weighted patch {running['patch_weighted']/n:.4f}"
                        if args.patch_loss_weight > 0
                        else ""
                    )
                    + (f" | lesion InfoNCE {running['lesion']/n:.4f}" if args.lesion_keypoints > 0 else "")
                    + (
                        f" | lesion decorrelation {running['decorrelation']/n:.4f}"
                        if args.lesion_decorrelation_weight > 0
                        else ""
                    ),
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/contrastive", running["loss"] / n, step)
                    writer.add_scalar("train/content_eff_rank", running["rank"] / n, step)
                    if args.lesion_keypoints > 0:
                        writer.add_scalar("train/lesion_infonce", running["lesion"] / n, step)
                    if args.lesion_decorrelation_weight > 0:
                        writer.add_scalar("train/lesion_decorrelation", running["decorrelation"] / n, step)
                    if args.patch_loss_weight > 0:
                        for tag, key in (
                            ("global_infonce", "global_loss"),
                            ("patch_infonce", "patch"),
                            ("patch_infonce_weighted", "patch_weighted"),
                        ):
                            writer.add_scalar(f"train/{tag}", running[key] / n, step)
                running = dict(
                    loss=0.0,
                    rank=0.0,
                    n=0,
                    global_loss=0.0,
                    patch=0.0,
                    patch_weighted=0.0,
                    lesion=0.0,
                    decorrelation=0.0,
                )

            if step % args.eval_every == 0 or step == args.train_steps:
                model.eval()
                flat, _ = evaluate(model, val_dataset, device, args, save_dir, step, writer, floor=floor)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                save_progress(step, "running")
                if args.spatial_recovery_eval:
                    evaluate_spatial_recovery(
                        model, vars(args), device, save_dir, step, initial=spatial_initial, writer=writer
                    )

                if args.best_metric != "none":
                    value = best_metric_value(flat, floor_flat, args.best_metric)
                    if value is not None and (best["value"] is None or value > best["value"]):
                        best = {"value": value, "step": step}
                        # A bare state_dict, so --checkpoint model_best.pt loads exactly like
                        # model.pt does; the provenance goes in a sidecar rather than wrapping
                        # the tensors in a dict every reader would then have to unwrap.
                        torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
                        with open(os.path.join(save_dir, "best_checkpoint.json"), "w") as fp:
                            json.dump(
                                {
                                    "step": step,
                                    "metric": args.best_metric,
                                    "value": value,
                                    "floor_subtracted": floor_flat is not None,
                                    "raw": flat.get(BEST_METRIC_KEYS[args.best_metric]),
                                },
                                fp,
                                indent=2,
                            )
                        print(f"    new best {args.best_metric} {value:+.4f} -> model_best.pt", flush=True)
                model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    save_progress(step, "complete")
    if best["step"] is not None:
        print(
            f"best {args.best_metric} {best['value']:+.4f} at step {best['step']} -> model_best.pt "
            f"(model.pt is the last step, {step}). Score the best one with "
            f"--checkpoint model_best.pt.",
            flush=True,
        )
        if best["step"] < step:
            # Worth saying out loud: past the peak the run is spending compute making the
            # representation worse, which at this dataset size is the expected shape.
            print(f"  NOTE: peak was {step - best['step']} steps before the end.", flush=True)
    print(f"done. checkpoints + DCI logs in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
