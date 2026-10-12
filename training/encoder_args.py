"""Command line of the encoder-only trainer, ``training.main_conv_synthetic``, without torch.

Kept out of the trainer so ``scripts/launch.py`` can build and check a run's command on a host
with no torch: ``parse_args`` applies every rule the trainer applies, and ``cli_arguments`` turns
a dict of option values into the arguments that reproduce it.
"""

import argparse

# Both modules import only the standard library at load time, so this stays torch-free.
from eval.causal.latent_causal_discovery import REDUCTIONS as CAUSAL_REDUCTIONS
from eval.causal.run_causal_recovery import DEFAULT_ALPHAS as CAUSAL_ALPHAS
from eval.causal.run_causal_recovery import INDEP_TESTS as CAUSAL_INDEP_TESTS

PROG = "python -m training.main_conv_synthetic"


def build_parser(description=None):
    """The trainer's argument parser, before the cross-option checks in ``parse_args``."""
    p = argparse.ArgumentParser(
        prog=PROG, description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    p.add_argument(
        "--mask-mode",
        choices=("fixed", "learned", "learned_size"),
        default="fixed",
        help="Which latent units are content. fixed: the first --content-channels. learned: a top-k "
        "Gumbel-softmax mask over one learnable logit per unit, k = --content-channels (Yao et al. 2024's "
        "content selector; the VQ-VAE trainer's --mask-mode learned). Each training step draws a mask (see "
        "--mask-estimator); evaluation reads the k largest logits. learned_size: one gate per unit, so the "
        "number of content units is learned too, against --mask-size-weight; evaluation reads the units "
        "with a positive logit and --content-channels is only the starting size. Shared by both views, whose "
        "readout head is shared. Starts at the fixed split (logits +/- half of --mask-init-gap)",
    )
    p.add_argument(
        "--mask-size-weight",
        type=float,
        default=None,
        help="With --mask-mode learned_size (required there): the loss adds this times the expected number "
        "of content units (the sum of the gates' probabilities), so a unit stays content only while it lowers "
        "the InfoNCE by more than this. It sets the size the mask settles at; sweep it and look for a range "
        "where the learned size holds steady",
    )
    p.add_argument(
        "--mask-init-gap",
        type=float,
        default=4.0,
        help="With a learned --mask-mode: initial selected-minus-excluded logit gap. Selected units start at "
        "+gap/2, excluded units at -gap/2. Default 4 reproduces +2/-2; 1 gives +0.5/-0.5. "
        "Must be finite and nonnegative; 0 starts all logits equal (evaluation ties select the first units; "
        "learned_size needs a positive gap). "
        "Only affects fresh initialization; loading a checkpoint restores its learned logits",
    )
    p.add_argument(
        "--mask-estimator",
        choices=("relaxed", "straight_through"),
        default="relaxed",
        help="With a learned --mask-mode: what the loss sees of each draw. relaxed: the continuous Gumbel-softmax "
        "sample, so a left-out unit keeps a small weight and a gradient on whether it would help as content. "
        "straight_through: the binary draw with a straight-through gradient (the VQ-VAE trainer's estimator). "
        "Cosine similarity has zero gradient for a unit zeroed in both views, so a left-out unit then learns "
        "only when a draw includes it. On toy features (tests/test_encoder_gumbel_mask.py) straight_through "
        "swapped a noisy content unit for a shared style unit in 0/10 seeds, relaxed in 10/10",
    )
    p.add_argument(
        "--mask-lr-scale",
        type=float,
        default=1.0,
        help="With a learned --mask-mode: the mask logits' learning rate is --lr times this (no weight decay). "
        "Compare --lr x this x --train-steps with half of --mask-init-gap as a rough movement scale. "
        "Actual Adam updates depend on gradient history",
    )
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
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Generator seed (synthetic) or augmentation seed (real data); defaults to --seed",
    )
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
        default=None,
        choices=["block_mcc", "ridge_r2", "val_loss", "none"],
        help="Validation metric that decides which checkpoint is kept as model_best.pt. "
        "block_mcc and ridge_r2 (synthetic; default block_mcc) are floor-subtracted when "
        "--floor-eval gave a floor, since the raw value is mostly floor. val_loss (real data; "
        "the default there) keeps the lowest training objective on the held-out subjects. "
        "'none' keeps only the last-step model.pt.",
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
    p.add_argument(
        "--causal-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="At each evaluation, run label-free PC on each view's content channels and score the graph "
        "against the generator's SCM (eval.causal.latent_causal_discovery). Needs --synthetic-causal",
    )
    p.add_argument(
        "--causal-eval-grid",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        help="Grid the content channels are read on; 1 1 1 is the model's own global output vector",
    )
    p.add_argument(
        "--causal-eval-reduce",
        choices=CAUSAL_REDUCTIONS,
        default="gap",
        help="Scalar per channel over the grid; only gap applies at grid 1 1 1",
    )
    p.add_argument("--causal-eval-indep-test", choices=CAUSAL_INDEP_TESTS, default="fisherz")
    p.add_argument("--causal-eval-alphas", type=float, nargs="+", default=list(CAUSAL_ALPHAS))
    p.add_argument("--causal-eval-max-cond-set", type=int, default=None, help="Cap PC's conditioning-set size")
    p.add_argument("--causal-eval-null-draws", type=int, default=200, help="Random namings in the null")

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
        "slower per step, constant memory. With real data it holds the preprocessed volumes and masks "
        "instead: on disk under --cache-dir, else in RAM; --no-cache re-reads and resamples every NIfTI.",
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
        "--synthetic-temporal-sigma",
        type=float,
        default=0.18,
        help="Positive Gaussian width of temporal atrophy in [-1,1] coordinates. Smaller values "
        "localize z_content[6] without changing its strength coefficient; 0.18 reproduces old runs",
    )
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

    # Real data. Same names, defaults and meaning as utils/config.py's flags, so the cluster
    # YAMLs' paths apply as written, and a run with the VQ-VAE run's spacing, size, masks and
    # labels reuses its preprocessing cache (data.datasets fingerprints exactly those).
    p.add_argument(
        "--dataset-name",
        choices=("synthetic", "adni", "ADNI_registered", "ADNI_stripped_masks", "custom"),
        default="synthetic",
        help="synthetic: the generator above. Anything else reads paired T1/FLAIR volumes from "
        "<--dataroot>/<--dataset-name>/<Subject>/{t1,t2} with data.datasets.MyCustomDataset. Real data "
        "has no generator factors, so evaluation reports the held-out objective, cross-view subject "
        "retrieval and the content/style probes instead of DCI. See training/ENCODER_ADNI.md",
    )
    p.add_argument("--dataroot", type=str, default=None, help="Real data: the directory holding --dataset-name")
    p.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Real data: labels CSV with Subject and Group columns (e.g. labels_cleaned_3class.csv). Group is "
        "what the split stratifies on and what the diagnosis probe reads",
    )
    p.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Real data: root of the per-subject *_brain_mask files, laid out like the images. By default they "
        "are looked for next to the images. If any subject lacks one, EVERY subject is masked by "
        "thresholding (image > 0) instead",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Real data with --cache: keep the preprocessed volumes here as .pt files, reused by later runs "
        "(and by VQ-VAE runs with the same preprocessing). Without it the cache is held in RAM, "
        "16 bytes per voxel per subject (two views and two masks)",
    )
    p.add_argument("--image-spacing", type=float, default=2.0, help="Real data: isotropic voxel spacing in mm")
    p.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Real data (required): volume size after resampling. Choose sizes the backbone stride divides, "
        "e.g. 96 112 96 at 2 mm as experiments/adni_real.yaml does (a 24x28x24 map at stride 4)",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="Real data (required, > 0): fraction of SUBJECTS held out for validation, subject-level and "
        "stratified by Group (data/splits.py). The labels CSV is read in full, so without a split the "
        "validation subjects are the training subjects",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        help="Real data: fraction of SUBJECTS held out for test. Never loaded by training; split.json lists them",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Real data: seed of the subject split. Keep it fixed across runs you intend to compare",
    )
    p.add_argument(
        "--asymmetric-aug",
        action="store_true",
        help="Real data: draw the intensity augmentation independently per view, then re-apply the brain "
        "mask. Without it both views of a training pair get the SAME random intensity shift (p=0.2), "
        "background included, so a pair shares a non-anatomical offset the cross-view loss can match on. "
        "The spatial augmentation is shared either way, keeping the views registered",
    )
    p.add_argument(
        "--shared-brain-mask",
        action="store_true",
        help="Real data: intersect the T1 and FLAIR brain masks so both views share one outline, removing "
        "the modality-specific mask boundary",
    )
    return p


def parse_args(argv=None, description=None):
    """Parse and check the trainer's arguments; exits through ``argparse`` on a refused combination."""
    p = build_parser(description)
    args = p.parse_args(argv)
    if args.no_cuda and args.device not in ("auto", "cpu"):
        p.error("--no-cuda forces CPU; it cannot be combined with --device cuda/mps")
    for name, default in (("data_seed", args.seed), ("model_seed", args.seed), ("loader_seed", args.seed + 10000)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.train_steps < 1 or args.eval_every < 1 or args.batch_size < 2:
        p.error("Need positive train/evaluation steps and at least two subjects per batch")
    real = args.dataset_name != "synthetic"
    if args.best_metric is None:
        args.best_metric = "val_loss" if real else "block_mcc"
    if real:
        # Generator and factor-evaluation flags would be silently ignored on real volumes;
        # refuse them rather than record settings that did nothing.
        synthetic_only = [k for k in vars(args) if k.startswith(("synthetic_", "spatial_recovery_", "causal_eval"))]
        synthetic_only += ["res", "n_content", "n_style", "num_train_samples", "num_val_samples"]
        synthetic_only += ["eval_pooling", "eval_patch_grid", "lesion_keypoints"]
        changed = [k for k in synthetic_only if getattr(args, k) != p.get_default(k)]
        if changed:
            p.error(
                f"--dataset-name {args.dataset_name} reads real volumes; these flags only configure the synthetic "
                "generator or its factor-based evaluation: " + ", ".join("--" + k.replace("_", "-") for k in changed)
            )
        if not args.dataroot or not args.labels_path:
            p.error("Real data needs --dataroot and --labels-path (experiments/cluster/<cluster>.yaml has both)")
        if args.spatial_size is None or min(args.spatial_size) < 1:
            p.error("Real data needs a positive --spatial-size D H W, e.g. 96 112 96 at --image-spacing 2")
        if not 0 < args.image_spacing < float("inf"):
            p.error("--image-spacing must be finite and positive")
        if not 0 < args.val_frac < 1:
            p.error(
                "Real data needs held-out subjects: --val-frac in (0, 1), e.g. 0.2 with --test-frac 0.1 as in "
                "experiments/adni_real.yaml. Without a split the validation subjects are the training subjects"
            )
        if not 0 <= args.test_frac < 1 or args.val_frac + args.test_frac >= 1:
            p.error("--test-frac must be in [0, 1) and leave training subjects after --val-frac")
        if args.best_metric not in ("val_loss", "none"):
            p.error(f"--best-metric {args.best_metric} scores generator factors; real data selects on val_loss or none")
    else:
        real_only = ["dataroot", "labels_path", "masks_dir", "cache_dir", "image_spacing", "spatial_size"]
        real_only += ["val_frac", "test_frac", "split_seed", "asymmetric_aug", "shared_brain_mask"]
        changed = [k for k in real_only if getattr(args, k) != p.get_default(k)]
        if changed:
            p.error(", ".join("--" + k.replace("_", "-") for k in changed) + " only apply with a real --dataset-name")
        if args.best_metric == "val_loss":
            p.error(
                "--best-metric val_loss is computed on real data only; synthetic runs select on block_mcc or ridge_r2"
            )
        if args.num_train_samples < args.batch_size:
            p.error("--num-train-samples must allow at least one full training batch")
    # The backbone map every grid must fit: the conv encoder floors at each stride-2 step,
    # ResNet's padded strides round up. Cubic for synthetic, the resampled volume otherwise.
    stride = args.resnet_output_stride if args.encoder_architecture == "resnet18" else args.downscale_factor
    if stride < 1:
        p.error("Encoder stride must be positive")
    volume = tuple(args.spatial_size) if real else (args.res,) * 3
    if args.encoder_architecture == "resnet18":
        backbone_map = tuple((size + stride - 1) // stride for size in volume)
    else:
        backbone_map = tuple(size // stride for size in volume)
    map_name = f"{backbone_map[0]}^3" if len(set(backbone_map)) == 1 else "x".join(map(str, backbone_map))
    map_positions = backbone_map[0] * backbone_map[1] * backbone_map[2]
    if args.cpu_threads is not None and args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    if args.deterministic_warn_only and not args.deterministic:
        p.error("--deterministic-warn-only requires --deterministic")
    if not 0 < args.synthetic_lesion_radius < float("inf"):
        p.error("--synthetic-lesion-radius must be finite and positive")
    if not 0 < args.synthetic_temporal_sigma < float("inf"):
        p.error("--synthetic-temporal-sigma must be finite and positive")
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
    if not 0 <= args.mask_init_gap < float("inf"):
        p.error("--mask-init-gap must be finite and nonnegative")
    if args.mask_mode != "fixed":
        if args.mask_mode == "learned" and args.content_channels >= args.latent_dim:
            p.error("--mask-mode learned chooses content among the latent units: --content-channels < --latent-dim")
        if args.contrastive_loss_type != "infonce":
            p.error(
                f"--mask-mode {args.mask_mode} needs --contrastive-loss-type infonce: Barlow Twins standardises "
                "each unit over the batch, which divides the mask back out, so no gradient would reach the logits"
            )
        if args.contrastive_proj_dim != 0:
            p.error(
                f"--mask-mode {args.mask_mode} requires --contrastive-proj-dim 0: the projector reads a fixed set of units"
            )
        if not 0 < args.mask_lr_scale < float("inf"):
            p.error("--mask-lr-scale must be finite and positive")
    elif args.mask_lr_scale != 1.0 or args.mask_estimator != "relaxed" or args.mask_init_gap != 4.0:
        p.error("--mask-lr-scale, --mask-estimator and --mask-init-gap only apply with a learned --mask-mode")
    if args.mask_mode == "learned_size":
        if args.mask_size_weight is None or not 0 <= args.mask_size_weight < float("inf"):
            p.error("--mask-mode learned_size needs a finite, nonnegative --mask-size-weight")
        if args.mask_init_gap == 0:
            p.error("--mask-mode learned_size needs a positive --mask-init-gap: a zero logit is neither side")
        if args.separate_spatial_readout:
            p.error("--mask-mode learned_size cannot drive --separate-spatial-readout, whose head has a fixed width")
    elif args.mask_size_weight is not None:
        p.error("--mask-size-weight only applies with --mask-mode learned_size")
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
        if any(g < 1 or g > size for g, size in zip(args.train_patch_grid, backbone_map)):
            p.error(f"--train-patch-grid must fit the {map_name} backbone map")
    if args.encoder_architecture == "resnet18":
        if args.conv_readout != "linear":
            p.error("--conv-readout only applies to --encoder-architecture conv")
        if args.norm_type != "group":
            p.error("--norm-type only applies to --encoder-architecture conv; use --resnet-norm for ResNet")
        if args.encoder_head_hidden <= 0:
            p.error("--encoder-head-hidden must be positive")
        if args.eval_pooling == "patch" and any(
            g < 1 or g > size for g, size in zip(args.eval_patch_grid, backbone_map)
        ):
            p.error(f"ResNet at --res {args.res} has a {map_name} map; --eval-patch-grid must fit it")
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
        if map_positions < 2:
            p.error(f"--global-pool attention needs more than one position; the backbone map is {map_name}")
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
    causal_defaults = ("grid", "reduce", "indep_test", "alphas", "max_cond_set", "null_draws")
    if args.causal_eval:
        if not args.synthetic_causal:
            p.error("--causal-eval scores against the generator's SCM; it needs --synthetic-causal")
        if args.content_channels < args.n_content:
            p.error("--causal-eval names one channel per factor: --content-channels must be >= --n-content")
        if any(g < 1 or g > size for g, size in zip(args.causal_eval_grid, backbone_map)):
            p.error(f"--causal-eval-grid must fit the {map_name} backbone map")
        if tuple(args.causal_eval_grid) == (1, 1, 1) and args.causal_eval_reduce != "gap":
            p.error("--causal-eval-reduce reduces over a grid; at --causal-eval-grid 1 1 1 only gap applies")
        if not args.causal_eval_alphas or any(not 0 < a < 1 for a in args.causal_eval_alphas):
            p.error("--causal-eval-alphas must lie in (0, 1)")
        if args.causal_eval_null_draws < 1:
            p.error("--causal-eval-null-draws must be positive")
    elif any(getattr(args, f"causal_eval_{k}") != p.get_default(f"causal_eval_{k}") for k in causal_defaults):
        p.error("--causal-eval-* options only apply with --causal-eval")
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
        if map_positions < 2:
            p.error(f"--lesion-keypoints needs more than one position; the backbone map is {map_name}")
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
        spatial = backbone_map[0]  # cubic: spatial recovery is synthetic-only
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


def cli_arguments(options):
    """Command-line arguments that make ``parse_args`` reproduce ``options``, in their order.

    Keys are option names as ``parse_args`` stores them (``model_id``, ``patch_loss_weight``).
    A store-true flag is written when true, a ``--x/--no-x`` flag in either form, a list as one
    option followed by its items, and None not at all. Every other value is written as text. A
    key the trainer does not have raises, so a misspelt setting cannot be dropped silently.
    """
    actions = {action.dest: action for action in build_parser()._actions if action.option_strings}
    arguments = []
    for key, value in options.items():
        if value is None:
            continue
        action = actions.get(key)
        if action is None:
            raise ValueError(f"{PROG} has no option for {key!r}")
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"{key} is a --x/--no-x flag; give true or false, not {value!r}")
            arguments.append(action.option_strings[0 if value else 1])
        elif action.nargs == 0:
            if action.const is not True or not isinstance(value, bool):
                raise ValueError(f"{key} is a store-true flag; give true or false, not {value!r}")
            if value:
                arguments.append(action.option_strings[0])
        else:
            if isinstance(value, bool):
                raise ValueError(f"{key} takes a value, not {value!r}")
            arguments.append(action.option_strings[0])
            arguments.extend(str(item) for item in (value if isinstance(value, (list, tuple)) else [value]))
    return arguments
