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
            "custom",
        ],
    )
    parser.add_argument("--model-dir", type=str, default="results")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=300001)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
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
    parser.add_argument("--vqvae-nb-entries", type=int, default=384)
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
        choices=["learned", "onthefly"],
        help="How the content/style Gumbel mask logits are determined. "
        "'learned': learnable nn.Parameter per level (and per view when --separate-encoders is set). "
        "'onthefly': mask logits computed on-the-fly from average encoder activations, "
        "shared across views (matches the original multiview-crl repo). Default: onthefly.",
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
        help="Use torch.compile for kernel fusion (requires PyTorch 2.0+)",
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
        "--selection",
        type=str,
        default="gumbel_softmax",
        choices=["ground_truth", "gumbel_softmax", "concat", "soft"],
    )
    parser.add_argument("--n-views", default=2, type=int)
    parser.add_argument("--change-lists", default=[[4, 5, 6, 8, 9, 10]])
    parser.add_argument("--faiss-omp-threads", type=int, default=16)
    parser.add_argument("--subsets", default=[(0, 1), (0, 2), (1, 2), (0, 1, 2)])
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
    elif args.dataset_name in ["adni", "ADNI_registered"]:
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
