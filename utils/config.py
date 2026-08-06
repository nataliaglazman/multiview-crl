# flake8: noqa
"""Argument parsing and dataset-specific configuration for multiview-CRL."""

import argparse

import data.datasets as datasets


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
        "--dataset-name",
        type=str,
        default="ADNI_registered",
        choices=[
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
    parser.add_argument("--synthetic-n-content", type=int, default=9)
    parser.add_argument("--synthetic-n-style", type=int, default=3)
    parser.add_argument(
        "--synthetic-style-scale",
        type=float,
        default=1.0,
        help="Multiply view-varying style/nuisance magnitudes (gain, bias, noise sigma, bias field). "
        ">1 = nuisance-dominant (reconstruction must spend capacity on nuisance).",
    )
    parser.add_argument(
        "--synthetic-content-scale",
        type=float,
        default=1.0,
        help="Multiply anatomy content effect-sizes (brain/ventricle/cortex/temporal/asymmetry/sulcal). "
        "<1 = subtle content (low reconstruction salience).",
    )
    parser.add_argument(
        "--synthetic-hierarchical-content",
        action="store_true",
        help="Enable hierarchical content latents: a shared global-atrophy "
        "scalar drives regional content dims (brain size, ventricle, cortical "
        "thickness, temporal atrophy, sulcal widening) via fixed couplings + "
        "independent residuals. GT latents include z_global_atrophy and "
        "z_content_residuals for probing.",
    )
    parser.add_argument(
        "--synthetic-normalize",
        type=str,
        default="per_sample",
        choices=["per_sample", "shared", "fixed_reference"],
        help="Normalization for synthetic pseudo-MRI volumes. "
        "'per_sample' z-scores each view independently over its foreground "
        "(erases global style gain/bias). "
        "'shared' z-scores both views using view-1's foreground stats, "
        "preserving the relative intensity difference between views. "
        "'fixed_reference' standardizes every sample/view by dataset-level "
        "constants, preserving absolute global gain/bias in both views.",
    )
    parser.add_argument(
        "--synthetic-res",
        type=int,
        default=32,
        help="Cubic resolution for synthetic volumes. Used as the default "
        "--spatial-size when --dataset_name=synthetic and --spatial-size is unset. "
        "Should be divisible by 8 for the 3-level VQ-VAE.",
    )
    parser.add_argument(
        "--synthetic-causal",
        action="store_true",
        help="Sample content latents from a causal SCM (DAG with nonlinear "
        "mechanisms) instead of i.i.d. Gaussians. Mutually exclusive with "
        "--synthetic-hierarchical-content.",
    )
    parser.add_argument(
        "--synthetic-causal-graph",
        type=str,
        default="chain",
        choices=["chain", "full", "random"],
        help="DAG topology for the content SCM.",
    )
    parser.add_argument(
        "--synthetic-causal-edge-prob",
        type=float,
        default=0.5,
        help="Edge probability for random DAG (ignored for chain/full).",
    )
    parser.add_argument(
        "--synthetic-causal-noise-scale",
        type=float,
        default=0.4,
        help="Additive noise scale in causal mechanisms.",
    )
    parser.add_argument(
        "--synthetic-causal-nonlinearity",
        type=str,
        default="leaky_relu",
        choices=["leaky_relu", "none"],
        help="Nonlinearity in causal mechanisms.",
    )
    parser.add_argument(
        "--synthetic-clean-content",
        action="store_true",
        help="Identifiability-friendly synthetic generation: tanh-squash content factors "
        "(instead of the hard clamp that flattens ~1/3 of N(0,1) values) and zero out the "
        "unlabeled z_deformation/z_fissure nuisance fields so the named factors fully "
        "determine structural variance. Default off = byte-identical to prior runs.",
    )
    parser.add_argument(
        "--synthetic-identifiable-ventricle",
        action="store_true",
        help="Make the ventricle-size factor recoverable: a larger, re-centred cavity "
        "(radius [0.12, 0.28], always clear of the septum split), read off the UNdeformed "
        "radius (the gyral field no longer swamps the tiny central cavity), and the "
        "longitudinal fissure gets its own tissue label/intensity so a pooled probe can separate "
        "ventricle CSF from fissure CSF. Lifts pooled ventricle R^2 from ~0.03 to ~0.92. "
        "Default off = byte-identical to prior runs.",
    )
    parser.add_argument("--model-dir", type=str, default="results")
    parser.add_argument("--model-id", type=str, default=None)
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
        "--deterministic",
        action="store_true",
        help="Make a run bit-reproducible across re-runs with the same config: "
        "seeds MONAI transforms + a numpy-seeding worker_init_fn, disables "
        "cudnn.benchmark, sets cudnn.deterministic, and enables "
        "torch.use_deterministic_algorithms (warn_only). Costs some throughput.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers. For 3D MRI with pin_memory, each worker holds "
        "prefetch_factor batches in pinned memory (~330 MB each). Keep this low (4-8).",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--norm-type",
        type=str,
        default="group",
        choices=["group", "layer"],
        help="Normalization used in the VQ-VAE ENCODER conv blocks: group norm (default) or layer norm.",
    )
    parser.add_argument(
        "--decoder-norm-type",
        type=str,
        default=None,
        choices=["group", "layer"],
        help="Normalization for the DECODER conv blocks. Default: follow --norm-type. Use "
        "'--norm-type layer --decoder-norm-type group' to remove the encoder's global-statistic "
        "broadcast (which makes background positions predict content factors) without breaking "
        "reconstruction: per-voxel channel norm forces every decoder feature voxel to unit scale, "
        "but emitting '0 here, bright there' is a magnitude task, so a layer-normed decoder "
        "reconstructs far worse (measured: brain MSE 0.35 vs 0.003).",
    )
    parser.add_argument(
        "--latent-mask",
        action="store_true",
        help="Zero encoder-output positions whose input footprint contains no foreground. The "
        "reconstruction loss is already brain-masked, so background latents are entirely "
        "unconstrained and the encoder parks brain information in them ('codebook smuggling'): "
        "ablating the FAR background band on a 300k baseline raised brain reconstruction error "
        "+311%% while background error moved 0%%, and the effect GREW with distance from the brain "
        "(+117%% near -> +311%% far) — spare capacity being exploited, not decoder receptive-field "
        "reach. This removes the capacity itself, which also denies the contrastive objective the "
        "per-sample signature it reads off background positions. NOTE: a model trained with this "
        "must also be EVALUATED with masks passed to forward().",
    )
    parser.add_argument(
        "--latent-mask-thresh",
        type=float,
        default=0.0,
        help="Keep a latent position if its pooled foreground fraction exceeds this. Default 0.0 "
        "keeps any position containing at least one foreground voxel — conservative, preserving "
        "the boundary ring the decoder legitimately uses. Raise it to prune the boundary too.",
    )
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
    parser.add_argument(
        "--scale-recon-loss",
        type=float,
        default=1,
        help="Scale factor for the reconstruction loss",
    )
    parser.add_argument(
        "--contrastive-only",
        action="store_true",
        help="Encoder-only ablation: skip the entire quantization + decoding path "
        "(no codebook forward, no decoder, hence no VQ commitment loss and no "
        "reconstruction loss). The encoder is trained purely by the contrastive "
        "objective. Forces return_recon=False on every forward (training, "
        "validation, and accumulation windows) and skips decoder-dependent "
        "logging/eval (decoded-image dumps, cross-reconstruction separation "
        "metrics). Codebook and decoder parameters receive no gradient.",
    )
    parser.add_argument("--scale-contrastive-loss", type=float, default=1)
    parser.add_argument(
        "--contrastive-proj-dim",
        type=int,
        default=0,
        help="If > 0, insert an MLP projection head (one per content/style level) "
        "between the pooled content features and the contrastive loss. The loss is "
        "computed on the head's output, while eval/probes keep reading the pre-head "
        "encoder features (the SimCLR/MoCo/BYOL recipe — the loss-facing space "
        "over-compresses toward view-invariance and loses linear-probe info). "
        "0 (default) disables the head: the loss acts directly on the representation. "
        "Not supported with --use-moco.",
    )
    parser.add_argument(
        "--contrastive-proj-hidden",
        type=int,
        default=256,
        help="Hidden width of the contrastive projection head MLP "
        "(Linear -> ReLU -> Linear). Only used when --contrastive-proj-dim > 0. Default: 256.",
    )
    parser.add_argument(
        "--contrastive-proj-mode",
        default="head",
        choices=("head", "entropy", "bounded"),
        help="Where the projection sits in the contrastive loss. 'head' (default) is the "
        "SimCLR/MoCo recipe: an MLP in front of the WHOLE InfoNCE, so it shapes both the "
        "alignment (numerator) and the uniformity (denominator) terms. 'entropy' is Yao et al. "
        "(ICLR 2024) Defn 3.6: a dimension-preserving map onto a bounded hypercube (tanh, so "
        "(-1,1)^k — an affine reparameterisation of the paper's (0,1)^k that keeps cosine "
        "similarity's full range) applied to the ENTROPY term only, over the FULL representation "
        "(eq. 3.3 carries no content selector on that term and Defn 3.6 sizes t_k by |S_k|), while "
        "alignment stays on the content-selected block. 'bounded' is Thm 3.2 / Defn 3.1 instead: "
        "with a single known content block the encoder itself maps to the unit cube and eq. (3.1) "
        "puts BOTH terms on it, so this is plain InfoNCE on tanh(content) with no extra parameters "
        "— the more faithful choice for a fixed single content/style split, which is the regime "
        "'--mask-mode fixed' puts you in. 'entropy' needs --contrastive-proj-dim > 0 to switch t_k "
        "on (width then forced to the representation width); 'bounded' ignores it. Both require "
        "InfoNCE, no patch-contrastive, no MoCo.",
    )
    parser.add_argument(
        "--tau-entropy",
        type=float,
        default=None,
        help="Separate temperature for the entropy term under --contrastive-proj-mode entropy. "
        "Defaults to --tau. Exists because alignment and entropy then live in different "
        "geometries, so one temperature need not suit both. Hold it fixed across A/B arms.",
    )
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
        "--scale-content-patch-modality-adv",
        type=float,
        default=0.0,
        help="Weight on patch-level gradient-reversal modality classifier from content. "
        "Penalises position-specific modality encoding that the pooled adversarial "
        "term misses. Operates on (B*P, k) content features per level. "
        "Watch ModAdvPatch/acc_L0 → 0.5 means patch-level invariance achieved.",
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
        "--bt-gap-weight",
        type=float,
        default=0.0,
        help="Weight on an ADDITIONAL Barlow Twins term computed on GAP-pooled features, added to "
        "the patch term. Motivation: writing the patch features as z[n,p,c] = s[n,c] + r[n,p,c] "
        "(subject term + interaction), the folded cross-covariance splits exactly into "
        "Cov_subject + Cov_interaction, and on registered volumes the interaction dominates — so "
        "the patch off-diagonal decorrelates within-subject spatial structure rather than subject "
        "identity. Averaging over positions recovers s exactly (r integrates to zero), so a "
        "GAP-pooled term is the only one whose rows are SUBJECTS. 0 disables (default). "
        "NOTE: the GAP term sees only B rows for a d x d matrix, so its off-diagonal has a "
        "sampling floor of about d(d-1)/B — at d=44, B=64 that is ~30 of spurious penalty. "
        "Use batch_size >= 128, or a small weight, or both.",
    )
    parser.add_argument(
        "--bt-gap-lambda",
        type=float,
        default=None,
        help="Off-diagonal weight for the --bt-gap-weight companion term. Defaults to --bt-lambda, "
        "but that is usually WRONG for it: the GAP term's cross-correlation is estimated from only "
        "B rows, so its off-diagonal carries a sampling floor of about d(d-1)/B (~15 at d=44, "
        "B=128) with no real redundancy behind it. At lambda=1 that noise floor is comparable to "
        "the whole useful range of the on-diagonal, so most of the term's gradient goes into "
        "decorrelating noise. Set it low (0.01-0.1) to make the GAP term primarily an ALIGNMENT "
        "term and leave redundancy reduction to the patch term, which has B*P rows to estimate from.",
    )
    parser.add_argument(
        "--bt-patch-stat",
        type=str,
        default="fold",
        choices=["fold", "per_position"],
        help="How Barlow Twins turns patch features into a cross-correlation. 'fold' (default) "
        "flattens (B, C, P) to B*P rows and standardises each channel over all of them. "
        "'per_position' standardises each (channel, position) across the B samples and averages "
        "the per-position cross-correlations. NOTE: with --patch-center-mode set these are very "
        "nearly the same objective (measured: RMS off-diagonal 0.5225 vs 0.5236 on collapsed "
        "input, 0.0642 vs 0.0643 on healthy) — once each (c,p) is zero-mean across samples the "
        "fold already averages the per-position covariances, so this only adds a per-position "
        "variance normalisation. Offered as an ablation; it is NOT a fix for across-subject "
        "collapse. Only used with --patch-contrastive and barlow_twins.",
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
    parser.add_argument(
        "--vqvae-nb-res-layers",
        type=int,
        default=2,
        help="ReZero residual blocks per encoder/decoder level (default 2, the previous "
        "hardcoded value). This is the main driver of the encoder RECEPTIVE FIELD: the "
        "downsampling path alone spans ~18 input voxels, and each block adds two 3^3 convs "
        "at the downsampled resolution, i.e. ~16 more. At scaling_rate 4 that is a radius of "
        "~27 voxels with 2 blocks and ~17 with 1. Lower it when patches need to be local — "
        "adjacent patches overlap by 2*radius/(cell + 2*radius) regardless of patch size, so "
        "shrinking the receptive field is the only lever a finer --patch-grid cannot provide. "
        "Changes the architecture, so checkpoints are not compatible across values.",
    )
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
        "Decoders and Gumbel content masks remain shared; codebooks remain shared too unless "
        "--separate-content-codebooks / --separate-style-codebooks are set. "
        "Consistent with the view-specific encoder identifiability theory (Yao et al., 2024).",
    )
    parser.add_argument(
        "--separate-content-codebooks",
        action="store_true",
        default=False,
        help="Give each view its own content codebook (view-0 and view-1 quantize content "
        "through independent codebooks); decoders stay shared. Pairs naturally with "
        "--separate-encoders. NOTE: content is meant to be the shared, modality-invariant "
        "anatomy, so splitting its codebook removes the common discrete vocabulary that ties "
        "the two views together and weakens identifiability — intended as an ablation, not a "
        "default. Default: shared content codebook.",
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
        "--separate-style-codebooks",
        action="store_true",
        default=False,
        help="Give each view its own style codebook (view-0 and view-1 quantize style through "
        "independent codebooks). Requires --quantize-style; pairs naturally with "
        "--separate-encoders. Since style is modality-specific, this guarantees T1 and T2 style "
        "codes never share entries and gives each modality dedicated style capacity. Content "
        "codebooks remain shared (content is modality-invariant). Default: shared style codebook.",
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
        "--detach-style-injection",
        action="store_true",
        default=False,
        help="Detach style features before decoder injection so the reconstruction loss "
        "cannot backpropagate into the encoder's style channels. Prevents the recon loss "
        "from incentivising content encoding in style channels.",
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
        "Index 0 = level 0 = FINEST level (first encoder); index nb_levels-1 = COARSEST. "
        "Choose grids so that each patch covers one spatial content cell: for a 4^3 "
        "deformation grid on a 64^3 input (cell = 16 input vox), use 4 4 4 at every "
        "level (32^3 features -> 8 vox/patch; 16^3 -> 4 vox/patch; 8^3 -> 2 vox/patch). "
        "E.g. '--patch-grid-per-level 4 4 4 4 4 4 2 2 2' for a 3-level model "
        "(finest → coarsest).",
    )
    parser.add_argument(
        "--patch-foreground-mask",
        action="store_true",
        help="Drop always-background patch positions from the patch-contrastive loss. "
        "Pools the brain mask to the patch grid each batch and keeps only positions with "
        "foreground (>= --patch-foreground-thresh) in at least one batch sample. On the "
        "central-brain synthetic data ~11%% of a 4^3 grid (and ~37%% of an 8^3 grid) is "
        "dead background; those positions inject noise into the contrastive signal "
        "(actively harmful for patch InfoNCE, diluting for Barlow Twins). Requires "
        "--patch-contrastive and brain masks; no effect under MoCo (patch keys unsliced).",
    )
    parser.add_argument(
        "--patch-foreground-thresh",
        type=float,
        default=0.05,
        help="Foreground-fraction threshold for --patch-foreground-mask: a patch position "
        "is kept if its pooled brain-mask fraction is >= this in at least one batch sample. "
        "Default: 0.05.",
    )
    parser.add_argument(
        "--patch-center-mode",
        type=str,
        default="none",
        choices=["none", "position", "double"],
        help="Centre patch features before the patch-InfoNCE similarity. On registered "
        "volumes patch p of two subjects is the same anatomy, so same-position negatives "
        "are largely false, and the cheapest way to solve the discrimination is a "
        "spatially constant per-subject intensity code — which flattens the spatial maps. "
        "'position' subtracts the across-batch mean at each patch position, removing the "
        "shared anatomy so the negatives become genuine. 'double' additionally subtracts "
        "each sample's mean over positions, removing the constant nuisance code so a "
        "uniform channel contributes exactly zero; what is discriminated is then the "
        "subject x location interaction. Expect top-1 accuracy to drop sharply — the task "
        "is genuinely harder. Needs a large batch, and not only because the per-position "
        "mean is a B-sample estimate: centering forces the B residuals to sum to zero, "
        "which is itself discriminative. At B=2 the two residuals are exact antipodes and "
        "the loss collapses to ~0 for free; on random view-consistent features the centred "
        "loss runs 0.0000/0.0005/0.0043/0.0166/0.0436/0.0992 at B=2/4/8/16/32/64 against an "
        "uncentred 0.0009/0.0086/0.0136/0.0246/0.0572/0.1102, i.e. still visibly inflated-easy "
        "at B=8. Prefer B>=32. Applies to every patch objective: InfoNCE centres inside the "
        "loss, Barlow Twins / VICReg centre before folding patches into the batch (without it "
        "their invariance term is satisfied by the shared anatomy at each position). Only used "
        "with --patch-contrastive. Default: none.",
    )
    parser.add_argument(
        "--patch-center-weight",
        action="store_true",
        help="Weight each patch position's contribution to the centred patch-InfoNCE loss "
        "by its mean residual magnitude, measured before L2 normalisation. Without this, "
        "positions where every subject looks alike (background, deep white matter) have a "
        "near-zero residual but are still renormalised to unit vectors, so they contribute "
        "full-magnitude random directions instead of dropping out. Requires "
        "--patch-center-mode != none.",
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
        choices=["concat", "film", "input"],
        help=(
            "How style features are injected into the decoder.  "
            "'concat' (default): style is concatenated onto the penultimate feature map "
            "before the final conv — simple but style only influences the last layer.  "
            "'film': Spatial FiLM (Feature-wise Linear Modulation) — style modulates "
            "the decoder feature map via learned per-location scale and shift after every "
            "decoder stage (residual block + each upsampling step), giving the decoder "
            "access to style information at every resolution.  "
            "'input': style is concatenated onto the decoder INPUT alongside the content "
            "codes, so it flows through the entire decoder from the first conv layer — "
            "style is treated symmetrically with content and reconstruction depends on it "
            "from the start (pairs naturally with --quantize-style).  "
            "Requires --inject-style-to-decoder."
        ),
    )
    parser.add_argument(
        "--style-spatial-size",
        type=int,
        default=0,
        help=(
            "If > 0, average-pool each injected style tensor to an (N, N, N) spatial "
            "grid (clamped per-axis to the current size, so it only ever downsamples) "
            "before it is quantized / injected into the decoder.  Caps the spatial "
            "capacity of the style pathway so style carries the global tissue-contrast "
            "transform rather than anatomy (which stays in the content code); the "
            "decoder's FiLM/concat upsampling restores feature resolution.  "
            "0 (default) keeps full-resolution style (legacy behaviour).  "
            "Requires --inject-style-to-decoder."
        ),
    )
    parser.add_argument(
        "--no-final-recon-norm",
        action="store_true",
        help=(
            "Drop the GroupNorm on the level-0 decoder's final conv (the reconstruction "
            "output).  By default the output is instance-normalized, which pins every "
            "reconstruction to a fixed global mean/std and so cannot reproduce per-sample "
            "intensity — fine when the input is per-sample z-scored, but an unrecoverable "
            "reconstruction error when global gain/bias is preserved into the input "
            "(e.g. --synthetic-normalize fixed_reference).  Pass this to let the decoder "
            "emit absolute intensity.  Changes the model architecture, so it is not "
            "checkpoint-compatible with runs trained without it."
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
        "--dci-every",
        type=int,
        default=0,
        help="If > 0 and --dataset-name is 'synthetic', compute and log DCI "
        "identifiability metrics (content/style vs ground-truth factor recovery) "
        "on the val set every N training steps. 0 disables periodic DCI; the "
        "end-of-run synthetic DCI is still controlled by --eval-dci.",
    )
    parser.add_argument(
        "--no-select-by-synthetic-dci",
        dest="select_by_synthetic_dci",
        action="store_false",
        help="On --dataset-name 'synthetic', do NOT select the best checkpoint by the "
        "run_dci_compare health composite (overall_score) on the GT factors; fall back to "
        "the cross-reconstruction separation_score. By default synthetic runs select on the "
        "composite so the in-training selector matches the offline comparison protocol. "
        "No effect on non-synthetic runs (no GT factors).",
    )
    parser.add_argument(
        "--selection-dci-n-null",
        type=int,
        default=3,
        help="Permutation-null repeats for the synthetic GT selection composite. Higher = "
        "lower-variance null floor but slower. Only used when selecting by synthetic DCI.",
    )
    parser.add_argument(
        "--selection-dci-n-seeds",
        type=int,
        default=2,
        help="CV probe seeds (0..N-1) for the synthetic GT selection composite. Only used "
        "when selecting by synthetic DCI.",
    )
    parser.add_argument(
        "--selection-dci-level",
        type=int,
        default=0,
        help="Encoder level scored for the synthetic GT selection composite. Default 0 (finest).",
    )
    parser.add_argument(
        "--selection-dci-max-samples",
        type=int,
        default=2000,
        help="Cap the val set to this many samples when scoring the synthetic GT selection "
        "composite, to bound periodic cost. 0 = use the full val set. Default 2000 (matches the "
        "offline run_dci_compare --num-samples convention).",
    )
    parser.add_argument(
        "--bt-sim-coeff",
        type=float,
        default=0.0,
        help="Weight on an MSE alignment term added to Barlow Twins, computed on the RAW "
        "(unstandardised) content features. BT's on_diag is a correlation on per-channel "
        "standardised features, so it is invariant to a per-view constant offset and cannot "
        "require the two views to COINCIDE — only to co-vary. Measured consequence: every "
        "content channel identifies the modality at AUC 1.000 under BT, versus 0/44 above "
        "0.7 under VICReg, whose invariance term is exactly this MSE. 0 disables (default), "
        "leaving existing runs bit-identical. Requires --bt-std-coeff > 0.",
    )
    parser.add_argument(
        "--bt-std-coeff",
        type=float,
        default=0.0,
        help="Weight on VICReg's variance hinge relu(1 - std) over the raw content features. "
        "NOT optional when --bt-sim-coeff > 0: MSE alone is minimised by collapsing both "
        "views to zero, and BT's on_diag/off_diag are both scale-invariant, so nothing else "
        "in the loss can detect that. Also a floor-free anti-collapse term — d independent "
        "per-channel estimates, with none of the d(d-1)/B sampling floor that limits the "
        "off-diagonal.",
    )
    parser.add_argument(
        "--selection-info-tolerance",
        type=float,
        default=0.05,
        help="Completeness gate on checkpoint selection: a step whose all-channels capacity "
        "(info_all) has fallen more than this fraction below its own running peak is not eligible "
        "to become the best checkpoint, whatever its overall_score. Guards against the failure "
        "where overall_score keeps climbing on a shrinking representation — three of its four "
        "terms reward what is ABSENT from the content block, so only content_anatomy notices "
        "content being discarded. 0 disables the gate (pre-2026-08 behaviour).",
    )
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
        "Set it equal to --vqvae-hidden-channels for an all-content baseline with no "
        "content/style separation (plain VQ-VAE-2: no Gumbel mask, no style injection). "
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

    _sr = getattr(args, "vqvae_scaling_rates", None)
    if _sr is not None and len(_sr) != _nb_levels:
        if len(_sr) > _nb_levels:
            logger.info(
                f"Truncating --vqvae-scaling-rates from {_sr} to {_sr[:_nb_levels]} "
                f"to match nb_levels={_nb_levels}."
            )
            args.vqvae_scaling_rates = _sr[:_nb_levels]
        else:
            raise ValueError(
                f"--vqvae-scaling-rates has {len(_sr)} entries but nb_levels={_nb_levels}. "
                "Provide at least nb_levels scaling rates."
            )

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

    # --patch-foreground-mask only does anything with patch-contrastive active.
    if getattr(args, "patch_foreground_mask", False) and not getattr(args, "patch_contrastive", False):
        logger.warning("--patch-foreground-mask is set but --patch-contrastive is not; it will be ignored.")
        args.patch_foreground_mask = False

    # --patch-center-mode applies to every patch objective: InfoNCE centres inside
    # _patch_infonce_base, BT/VICReg centre before folding patches into the batch.
    _cl_type_pc = getattr(args, "contrastive_loss_type", "infonce")
    if getattr(args, "patch_center_mode", "none") != "none" and not getattr(args, "patch_contrastive", False):
        logger.warning("--patch-center-mode is set but --patch-contrastive is not; it will be ignored.")
        args.patch_center_mode = "none"
    # --patch-center-weight, by contrast, IS InfoNCE-only: it reweights positions by their
    # pre-normalisation residual magnitude, and BT/VICReg never L2-normalise per position.
    if getattr(args, "patch_center_weight", False) and _cl_type_pc != "infonce":
        logger.warning(
            f"--patch-center-weight is only implemented for --contrastive-loss-type infonce "
            f"(got {_cl_type_pc}); it will be ignored. --patch-center-mode still applies."
        )
        args.patch_center_weight = False
    if getattr(args, "patch_center_weight", False) and getattr(args, "patch_center_mode", "none") == "none":
        logger.warning("--patch-center-weight requires --patch-center-mode != none; it will be ignored.")
        args.patch_center_weight = False
    elif (
        _cl_type_pc == "infonce"
        and getattr(args, "patch_center_mode", "none") != "none"
        and not getattr(args, "patch_center_weight", False)
    ):
        logger.warning(
            "--patch-center-mode is set without --patch-center-weight. Positions where every "
            "subject looks alike have a near-zero residual but are still L2-normalised to unit "
            "vectors, so they contribute chance-level loss that can swamp the informative "
            "positions. Enable --patch-center-weight unless you are deliberately ablating it."
        )

    # --mask-mode learned_split is incompatible with --inject-style-to-decoder
    # because the number of style channels varies per forward pass.
    if getattr(args, "mask_mode", "onthefly") == "learned_split" and getattr(args, "inject_style_to_decoder", False):
        raise ValueError(
            "--mask-mode learned_split is incompatible with --inject-style-to-decoder "
            "because the number of style channels varies per forward pass. "
            "Use --mask-mode fixed or learned instead."
        )

    # --separate-style-codebooks needs a style codebook to separate.
    if getattr(args, "separate_style_codebooks", False):
        if not getattr(args, "quantize_style", False):
            raise ValueError(
                "--separate-style-codebooks requires --quantize-style (there is no style "
                "codebook to give each view otherwise)."
            )
        if not getattr(args, "separate_encoders", False):
            logger.warning(
                "--separate-style-codebooks is set without --separate-encoders. Per-view style "
                "codebooks are most meaningful when each view also has its own encoder; with a "
                "shared encoder the two views' style channels are produced identically."
            )

    # --separate-content-codebooks: ablation that breaks the shared content vocabulary.
    if getattr(args, "separate_content_codebooks", False):
        if not getattr(args, "separate_encoders", False):
            logger.warning(
                "--separate-content-codebooks is set without --separate-encoders. Per-view "
                "content codebooks are most meaningful when each view also has its own encoder."
            )
        logger.warning(
            "--separate-content-codebooks gives each view its own content codebook. Content is "
            "meant to be modality-invariant, so this removes the shared discrete vocabulary that "
            "aligns the two views and is expected to weaken identifiability. Use as an ablation."
        )

    # Warn if MoCo is enabled with a negative-free loss (it'll be ignored)
    _cl_type = getattr(args, "contrastive_loss_type", "infonce")
    if _cl_type in ("barlow_twins", "vicreg") and getattr(args, "use_moco", False):
        logger.warning(
            f"--use-moco is set but --contrastive-loss-type is '{_cl_type}' which does not "
            f"use negatives.  MoCo queue and momentum encoder will be disabled."
        )
        args.use_moco = False

    # The projection head is wired into the in-batch InfoNCE / patch / BT / VICReg
    # paths only. The MoCo path keeps a momentum-encoder + queue of *un-projected*
    # features, so a head there would need a momentum copy and projected enqueues —
    # not implemented. Fail loudly rather than silently ignoring the head.
    if getattr(args, "contrastive_proj_dim", 0) > 0 and getattr(args, "use_moco", False):
        raise ValueError(
            "--contrastive-proj-dim is not supported with --use-moco yet (the MoCo queue stores "
            "un-projected features). Disable one of them."
        )

    # 'entropy' mode splits the InfoNCE numerator from its denominator (see
    # training.losses.split_infonce_loss), so it only exists on the pooled InfoNCE
    # path. Fail loudly rather than silently falling back to the SimCLR head.
    _pmode = getattr(args, "contrastive_proj_mode", "head")
    if _pmode in ("entropy", "bounded"):
        if getattr(args, "contrastive_loss_type", "infonce") != "infonce":
            raise ValueError(
                f"--contrastive-proj-mode {_pmode} requires --contrastive-loss-type infonce "
                "(the alignment/entropy structure is defined by the InfoNCE numerator/denominator). "
                f"Got '{args.contrastive_loss_type}'."
            )
        if _pmode == "bounded" and getattr(args, "patch_contrastive", False):
            raise ValueError(
                "--contrastive-proj-mode bounded is not implemented for --patch-contrastive "
                "(eq. 3.1 puts BOTH terms on one bounded encoding; there is no spatial axis "
                "to split them across). Use --contrastive-proj-mode entropy for patch."
            )
    if _pmode == "entropy" and getattr(args, "contrastive_proj_dim", 0) <= 0:
        raise ValueError(
            "--contrastive-proj-mode entropy needs --contrastive-proj-dim > 0 to switch t_k on "
            "(its width is then forced to the full representation width, per Defn 3.6)."
        )
    if _pmode == "bounded" and getattr(args, "contrastive_proj_dim", 0) > 0:
        logger.warning(
            "--contrastive-proj-mode bounded ignores --contrastive-proj-dim: Thm 3.2 has no "
            "separate projection, only a bounded encoder output. No head will be built."
        )

    # --content-size: directly set content channels, override ratio-based defaults.
    # cs == hidden_ch is the all-content baseline: no style channels, so downstream
    # has_content_style is False (no Gumbel mask, no style injection — plain VQ-VAE-2).
    if getattr(args, "content_size", None) is not None:
        hidden_ch = args.vqvae_hidden_channels
        cs = args.content_size
        assert 1 <= cs <= hidden_ch, f"--content-size must be in [1, {hidden_ch}], got {cs}"
        ratio = cs / hidden_ch
        # Override content_dim / total_dim to be consistent with the chosen ratio
        args.content_dim = cs
        args.total_dim = hidden_ch
        # Set content_ratios for all content_style_levels
        cs_levels = getattr(args, "content_style_levels", [0])
        args.content_ratios = [ratio] * len(cs_levels)
        if cs == hidden_ch:
            logger.info(
                f"  --content-size={cs}: all {hidden_ch} channels are content "
                "(no content/style separation — plain VQ-VAE-2 baseline, no Gumbel "
                "mask or style injection)."
            )
        else:
            logger.info(
                f"  --content-size={cs}: content_ratio={ratio:.3f} "
                f"({cs}/{hidden_ch} channels) applied to levels {cs_levels}"
            )

    args.modalities = ["image"]
    args.n_views = 2
    args.subsets = [(0, 1)]

    if args.dataset_name == "custom":
        args.DATASETCLASS = datasets.MyCustomDataset
        logger.info("  -> Using custom dataset (image only, 2 views)")
    elif args.dataset_name == "synthetic":
        args.DATASETCLASS = datasets.SyntheticBrainDataset
        args.content_indices = [list(range(args.content_dim))]
        args.style_indices = list(range(args.content_dim, args.total_dim))
        if getattr(args, "spatial_size", None) is None:
            res = getattr(args, "synthetic_res", 64)
            args.spatial_size = (res, res, res)
            logger.info(f"  -> Auto-set --spatial-size to ({res}, {res}, {res}) from --synthetic-res")
        logger.info("  -> Using synthetic dataset (pseudo-MRI, 2 views)")
        logger.info(f"  -> Content dimensions: 0-{args.content_dim - 1} ({args.content_dim} dims)")
        logger.info(
            f"  -> Style dimensions: {args.content_dim}-{args.total_dim - 1} ({args.total_dim - args.content_dim} dims)"
        )
    elif args.dataset_name in [
        "adni",
        "ADNI_registered",
        "ADNI_stripped",
        "ADNI_stripped_masks",
    ]:
        args.DATASETCLASS = datasets.MyCustomDataset
        args.content_indices = [list(range(args.content_dim))]
        args.style_indices = list(range(args.content_dim, args.total_dim))
        logger.info("  -> Using ADNI dataset (image only, 2 views)")
        logger.info(f"  -> Content dimensions: 0-{args.content_dim - 1} ({args.content_dim} dims)")
        logger.info(
            f"  -> Style dimensions: {args.content_dim}-{args.total_dim - 1} ({args.total_dim - args.content_dim} dims)"
        )
    else:
        raise ValueError(f"{args.dataset_name=} not supported.")

    if not hasattr(args, "content_indices") or args.content_indices is None:
        args.content_indices = [list(range(args.content_dim))]
    logger.info(f"  -> Subsets: {args.subsets}")
    logger.info(f"  -> Content indices: {len(args.content_indices[0])} dimensions")

    return args
