"""Encoder-only multi-view contrastive learning on the 3D synthetic data or real T1/FLAIR MRI.

3D encoders trained with InfoNCE or Barlow Twins; optional vector-only local cross-view prediction.
The default ``conv`` architecture uses the VQ-VAE convolutional backbone with
an affine readout. ``--encoder-architecture resnet18`` uses a 3D adaptation of
the upstream image encoder: ResNet-18 -> GAP -> Linear -> LeakyReLU -> Linear.
This architecture option does not change the loss, data, or view-sharing policy.
``--global-pool attention`` replaces the GAP before either readout with multi-head
attention pooling (``models.attention_pool``), which starts as exact GAP. ``--global-pool moments``
keeps GAP and adds each channel's first spatial moments to the readout (``models.moment_pool``),
through zero-initialized columns, so it starts as exact GAP too. ``--conv-readout dense`` does not
pool: strided convs to a small grid, flattened into the MLP (``models.dense_readout``).
``--patch-loss-weight`` adds a loss on a grid of registered positions: per-position InfoNCE, or with
``--patch-loss-type barlow_twins|vicreg`` a negative-free loss whose statistics are pooled over positions.
``--pointwise-loss-weight`` adds alignment + joint entropy of a bounded content map, every statistic pooled over
(subject, position) rows; with ``--global-loss-weight 0 --interior-only`` it is the objective of the fully local
encoder (training/ENCODER_POINTWISE.md).
``--mask-mode learned`` replaces the fixed content block (the first ``--content-channels``
units) with a top-k Gumbel-softmax content selector over learnable per-unit logits.
``--stratify-batches oracle|code`` draws every training batch from one k-means cluster of the
training subjects (MCL's batch rule; ``data.cluster_batches``), so InfoNCE's negatives share the
clustered anatomy. Add ``--stratify-sampler neighbourhood --stratify-neighbours 100`` to use
overlapping pools around anchors instead, keeping every subject eligible.

Identifiability is scored with ``eval.metrics.dci.compute_dci_synthetic`` (per-latent
RidgeCV/GBT R² + block-MCC + content→view leakage): content latents → high,
independent style → ~chance is the block-identification signal.

``--dataset-name ADNI_stripped_masks`` (or another real dataset) trains the same models on
paired T1/FLAIR volumes read by ``data.datasets.MyCustomDataset``, with the VQ-VAE trainer's
flag names, preprocessing and cache. Real data has no generator factors, so each evaluation
reports the held-out objective, cross-view subject retrieval and
``eval.metrics.cross_reconstruction``'s content/style probes against the untrained floor.
See training/ENCODER_ADNI.md.

Example:
    python -m training.main_conv_synthetic --model-id conv_c9 \
        --latent-dim 16 --content-channels 9 --n-content 9 --n-style 3 \
        --train-steps 50000 --eval-every 2000 --contrastive-loss-type infonce
"""

import hashlib
import io
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import data.cluster_batches as cluster_batches
import eval.metrics.cross_reconstruction as cross_reconstruction
import eval.metrics.dci as dci
import training.encoder_args as encoder_args
import training.losses as losses
from data.datasets import MyCustomDataset, SyntheticBrainDataset
from data.splits import split_summary, subject_level_split
from eval.adni.encoder_alignment import print_alignment_report
from models.conv_footprint import receptive_field_voxels
from models.multiview_encoder import MultiviewConvEncoder
from models.vector_prediction import vector_prediction_loss
from utils.encoder_runtime import configure_encoder_runtime, select_encoder_device


def parse_args(argv=None):
    return encoder_args.parse_args(argv, description=__doc__)


def cache_gb(res, num_samples):
    """RAM the dataset's in-memory cache will hold once every sample has been rendered."""
    return num_samples * 2 * (res**3) * 4 / 1e9


def make_dataset(args, mode, num_samples):
    """Single factory for train/val/test so the generative distribution is identical."""
    if getattr(args, "dataset_name", "synthetic") != "synthetic":
        return make_real_dataset(args, mode)
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
        synthetic_temporal_sigma=getattr(args, "synthetic_temporal_sigma", 0.18),
        synthetic_content_amp_scale=getattr(args, "synthetic_content_amp_scale", None),
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
        synthetic_lesion_min_radius_vox=getattr(args, "synthetic_lesion_min_radius_vox", None),
    )


def make_real_dataset(args, mode):
    """One subject split of the paired T1/FLAIR volumes, preprocessed as the VQ-VAE trainer does.

    The constructor arguments match training/main_multimodal.py's (crop margin 0), so with the
    same spacing, size, masks and subject list both trainers resolve to one cache directory.
    Only the train split augments; its MONAI transforms are seeded from --data-seed.
    """
    dataset = MyCustomDataset(
        data_dir=os.path.join(args.dataroot, args.dataset_name),
        mode=mode,
        spacing=args.image_spacing,
        crop_margin=0,
        spatial_size=args.spatial_size,
        cache=args.cache,
        cache_dir=args.cache_dir,
        labels_path=args.labels_path,
        masks_dir=args.masks_dir,
        asymmetric_aug=args.asymmetric_aug,
        shared_brain_mask=args.shared_brain_mask,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
    )
    seed_augmentation(dataset, args.data_seed)
    return dataset


def seed_augmentation(dataset, seed):
    """Seed a real dataset's MONAI pipelines, which otherwise share a RandomState seeded at import."""
    for name in ("_aug_transform", "monai_transform"):
        transform = getattr(dataset, name, None)
        if transform is not None:
            transform.set_random_state(seed=seed)


def seed_augmentation_worker(worker_id):
    """DataLoader ``worker_init_fn`` for real data: give each worker its own augmentation stream.

    Workers start from copies of the main process's RandomStates, so unseeded they would all
    replay one sequence. ``torch.initial_seed()`` differs per worker and derives from the
    loader's generator, so the streams still replay from --loader-seed and --data-seed.
    """
    seed_augmentation(torch.utils.data.get_worker_info().dataset, torch.initial_seed() % 2**32)


def describe_real_split(args, train_dataset, val_dataset, save_dir):
    """Print the subject split and write it to split.json for scoring the held-out subjects later.

    Recomputed with the dataset's own function and arguments, then checked against the indices
    each split actually loaded, so the file cannot describe a different split.
    """
    items = train_dataset._all_items
    split = subject_level_split(items, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed)
    if split["train"] != train_dataset._indices or split["val"] != val_dataset._indices:
        raise RuntimeError("The recomputed subject split differs from the one the datasets loaded")
    names = {index: group for group, index in train_dataset.label_map.items()}
    print(f"real data: {args.dataset_name}, subject-level split (seed {args.split_seed})", flush=True)
    for line in split_summary(items, split, label_names=names):
        print(line, flush=True)
    payload = {
        "dataset_name": args.dataset_name,
        "labels_path": args.labels_path,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "split_seed": args.split_seed,
        "label_map": train_dataset.label_map,
        "subjects": {mode: [items[i]["subject"] for i in indices] for mode, indices in split.items()},
    }
    with open(os.path.join(save_dir, "split.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    if len(train_dataset) < args.batch_size or len(val_dataset) < args.batch_size:
        raise SystemExit(
            f"--batch-size {args.batch_size} needs a full batch in both splits; train has {len(train_dataset)} "
            f"scans and val {len(val_dataset)}"
        )
    voxels = args.spatial_size[0] * args.spatial_size[1] * args.spatial_size[2]
    if args.cache and args.cache_dir is None:
        gb = 16 * voxels * (len(train_dataset) + len(val_dataset)) / 1e9
        print(f"  RAM cache: ~{gb:.1f} GB once full; --cache-dir keeps it on disk instead", flush=True)
    if not args.asymmetric_aug:
        print(
            "  NOTE: without --asymmetric-aug both training views share one random intensity shift (p=0.2), "
            "background included",
            flush=True,
        )


STAGE0_DATA_KEYS = ("res", "n_content", "n_style", "num_train_samples", "data_seed")


def stage0_code(args, train_dataset, device):
    """--stratify-batches code: the --stratify-from run's content code of every training subject.

    Both views' content units at the model's own pooling, concatenated, then the true content factors
    of the same subjects. The stage-0 model is rebuilt from its settings, used once and dropped; the
    run being trained starts from its own initialization.
    """
    from eval.protocol.score_checkpoint import build_model, encode_blocks, load_settings

    path = args.stratify_from
    run_dir, checkpoint = (
        (os.path.dirname(path), path) if os.path.isfile(path) else (path, os.path.join(path, "model.pt"))
    )
    if not (os.path.isfile(checkpoint) and os.path.isfile(os.path.join(run_dir, "settings.json"))):
        raise SystemExit(
            f"--stratify-from {path}: need a run directory with settings.json and model.pt, or a checkpoint inside one"
        )
    cfg = load_settings(run_dir)
    differ = [k for k in vars(args) if k in STAGE0_DATA_KEYS or k.startswith("synthetic_")]
    differ = [k for k in differ if cfg.get(k) != getattr(args, k)]
    if differ:
        print(
            "  NOTE: the stage-0 run's data differs from this run's in "
            + ", ".join(f"{k} ({cfg.get(k)} vs {getattr(args, k)})" for k in differ)
            + "; matching uses the frozen stage-0 encoder on this run's training bank",
            flush=True,
        )
    # build_model applies the stage-0 run's runtime settings; this run keeps its own.
    runtime = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    stage0 = build_model(cfg, device, torch.load(checkpoint, map_location="cpu"))
    torch.use_deterministic_algorithms(runtime[0], warn_only=runtime[1])
    torch.set_num_threads(runtime[2])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = runtime[3:]
    blocks = encode_blocks(stage0, train_dataset, device, args.batch_size, stage0.content_channels)
    del stage0
    code = np.concatenate([view.reshape(len(view), -1) for view in blocks["content"]], axis=1)
    source = f"the content code of {checkpoint} ({code.shape[1] // 2} units per view, both views)"
    return code, np.asarray(blocks["z_content"], dtype=np.float64), source


def stratified_batch_sampler(args, train_dataset, device, save_dir):
    """Condition batches on oracle anatomy or a frozen stage-0 code of the training subjects.

    The selected sampler uses k-means clusters or overlapping nearest-neighbour pools. Writes
    the fixed lookup and coverage/spread diagnostics to stratification.json. The summary also
    rides on the sampler as ``summary``.
    """
    names = dci.content_factor_names(args.n_content, args.synthetic_lesion_target)
    anatomy = cluster_batches.anatomy_columns(names)
    clustered = free = None
    started = time.perf_counter()
    if args.stratify_batches == "oracle":
        # Hand-built option namespaces may omit the parser's default.
        chosen = getattr(args, "stratify_factors", None) or [names[j] for j in anatomy]
        missing = [name for name in chosen if name not in names]
        if missing:
            raise SystemExit(f"--stratify-factors {' '.join(missing)}: not among this run's content factors {names}")
        clustered = sorted(names.index(name) for name in chosen)
        free = [j for j in anatomy if j not in clustered]
        factors = cluster_batches.bank_latents(train_dataset)
        features = factors[:, clustered]
        source = "the true anatomy latents (" + ", ".join(names[j] for j in clustered) + ")"
        if free:
            source += " chosen by --stratify-factors, " + ", ".join(names[j] for j in free) + " left free"
        source += ", lesion latents left out"
    else:
        features, factors, source = stage0_code(args, train_dataset, device)
    if getattr(args, "stratify_sampler", "cluster") == "neighbourhood":
        return neighbourhood_batch_sampler(
            args, features, factors, names, anatomy, clustered, source, save_dir, started
        )
    labels = cluster_batches.kmeans_labels(features, args.stratify_clusters, args.data_seed)
    sampler = cluster_batches.ClusterBatchSampler(labels, args.batch_size, args.loader_seed)
    summary = cluster_batches.summarize(
        labels, args.batch_size, factors, names, anatomy, args.stratify_clusters, clustered
    )
    print(
        f"stratified batches: {args.stratify_batches}, every batch from one of K = {args.stratify_clusters} k-means "
        f"clusters of {source}, standardized ({time.perf_counter() - started:.0f} s)",
        flush=True,
    )
    print(
        f"  clusters kept (>= {args.batch_size} subjects): {summary['clusters_kept']} of {summary['clusters']}, "
        f"holding {summary['subjects_in_kept_clusters']} of {summary['subjects']} training subjects; "
        f"{summary['batches_per_epoch']} batches ({summary['subjects_per_epoch']} subjects) per epoch",
        flush=True,
    )
    print("  within-cluster SD / bank SD per content factor (what a batch's negatives keep of it):", flush=True)
    for j, name in enumerate(names):
        role = "clustered" if clustered and j in clustered else ("anatomy" if j in anatomy else "")
        print(f"      {name:<20s}{summary['sd_ratio'][name]:7.3f}  {role}", flush=True)
    theory = f"theory K^(-1/{len(clustered or anatomy)}) = {summary['theory_k_pow_minus_1_over_d']:.3f}"
    if clustered is None:
        print(f"    anatomy mean {summary['sd_ratio_anatomy_mean']:.3f}; {theory}", flush=True)
    else:
        line = f"    clustered mean {summary['sd_ratio_clustered_mean']:.3f}; {theory}"
        if free:
            other = np.mean([summary["sd_ratio"][names[j]] for j in free])
            line += f"; anatomy left free mean {other:.3f}"
        print(line, flush=True)
    payload = {
        "mode": args.stratify_batches,
        "sampler": "cluster",
        "batch_size": args.batch_size,
        "k": args.stratify_clusters,
        "source": source,
        "stratify_from": args.stratify_from,
        "kmeans_seed": args.data_seed,
        "sampler_seed": args.loader_seed,
        **summary,
        "labels": labels.tolist(),
    }
    with open(os.path.join(save_dir, "stratification.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    sampler.summary = summary
    return sampler


def neighbourhood_batch_sampler(args, features, factors, names, anatomy, matched, source, save_dir, started):
    """--stratify-sampler neighbourhood: each batch is an anchor plus B-1 of its fixed nearest neighbours.

    ``matched`` are the oracle's factor columns (None for a code). Prints and writes to
    stratification.json the neighbour pools and, over the run's first anchor pass, subject coverage
    and each factor's within-batch SD over its bank SD; over the whole run, every subject's draw count.
    ``--stratify-balance`` weights the neighbour choice toward subjects drawn less often so far.
    """
    balance = getattr(args, "stratify_balance", 0.0)
    neighbours = cluster_batches.nearest_neighbours(features, args.stratify_neighbours)
    sampler = cluster_batches.NeighbourhoodBatchSampler(neighbours, args.batch_size, args.loader_seed, balance)
    summary = cluster_batches.summarize_neighbourhood(
        neighbours, args.batch_size, args.loader_seed, factors, names, anatomy, matched, balance, args.train_steps
    )
    preview = summary["preview"]
    choice = f"with weights (1 + draws so far)^-{balance:g}" if balance else "uniformly"
    print(
        f"stratified batches: {args.stratify_batches}, anchor + {args.batch_size - 1} sampled {choice} from its "
        f"{args.stratify_neighbours} nearest neighbours in {source}, standardized Euclidean distance "
        f"({time.perf_counter() - started:.0f} s)",
        flush=True,
    )
    print(
        f"  all {summary['subjects']} subjects anchor once per {summary['batches_per_anchor_pass']} batches; "
        f"{summary['batches_per_epoch']} batches ({summary['subject_draws_per_epoch']} subject draws) per "
        "loader epoch; neighbours can recur across batches",
        flush=True,
    )
    print(
        f"  the run's first {preview['batches']} batches: {preview['unique_subjects_first_epoch']} unique subjects "
        f"in the first loader epoch, {preview['unique_subjects']} over the anchor pass",
        flush=True,
    )
    spread = summary["run"]["draw_spread"]
    print(
        f"  draws per subject over the run's {args.train_steps} batches: min/p5/median/p95/max = "
        f"{spread['min']}/{spread['p5']:.0f}/{spread['median']:.0f}/{spread['p95']:.0f}/{spread['max']} "
        f"(mean {spread['mean']:.1f}); {spread['below_quarter_mean']} subjects below a quarter of the mean",
        flush=True,
    )
    ceiling = summary["pool_membership"]
    print(
        f"  ceiling: {ceiling['subjects_capped_below_mean']} subjects sit in too few pools to reach the mean draw "
        f"rate, {ceiling['subjects_capped_below_quarter_mean']} to reach a quarter of it, whatever the balance",
        flush=True,
    )
    print("  RMS within-batch SD / bank SD per content factor (what a batch's negatives keep of it):", flush=True)
    for j, name in enumerate(names):
        value = summary["sd_ratio"][name]
        shown = "    n/a" if value is None else f"{value:7.3f}"
        role = "matched" if matched and j in matched else ("anatomy" if j in anatomy else "")
        print(f"      {name:<20s}{shown}  {role}", flush=True)
    mean = summary["sd_ratio_anatomy_mean" if matched is None else "sd_ratio_matched_mean"]
    if mean is not None:
        print(f"    {'anatomy' if matched is None else 'matched'} mean {mean:.3f}", flush=True)
    payload = {
        "mode": args.stratify_batches,
        "sampler": "neighbourhood",
        "batch_size": args.batch_size,
        "neighbours": args.stratify_neighbours,
        "source": source,
        "stratify_from": args.stratify_from,
        "sampler_seed": args.loader_seed,
        "metric": "euclidean_on_standardized_features",
        "anchor_schedule": "shuffled_full_bank_passes_carried_across_loader_epochs",
        **summary,
        "neighbour_indices": neighbours.tolist(),
    }
    with open(os.path.join(save_dir, "stratification.json"), "w") as fp:
        json.dump(payload, fp, indent=2, allow_nan=False)
    sampler.summary = summary
    return sampler


def contrastive_code(pooled, model, args, mask=None):
    """Select/project the global content code before optional batch standardization."""
    b = pooled.shape[0] // 2
    hz = torch.stack([pooled[:b], pooled[b:]], dim=0)
    if mask is None:
        return model.project(hz[..., : args.content_channels])
    return hz[..., : model.latent_dim] * mask


def standardize_contrastive_code(hz):
    """The per-view, per-dimension batch transform used by --standardize-code."""
    return (hz - hz.mean(1, keepdim=True)) / (hz.std(1, keepdim=True) + 1e-6)


def contrastive_loss(pooled, model, args, sim_metric, criterion, mask=None):
    """Content-alignment − entropy on the pooled content block across the two views.

    The content block is selected first and then projected, so with a head every output
    dimension is part of the loss-facing space and counts as content downstream. The head
    is applied here rather than handed to the losses' ``projector`` argument, which
    ``infonce_base_loss`` accepts but never calls. Without a head ``project`` is the
    identity and this is the same tensor the loss selected internally before.

    ``mask`` is a learned model's (1, latent_dim) content mask for this step
    (``model.sample_content_mask``). The latent units are multiplied by it rather than sliced, so
    the gradient reaches every unit's logit. Under a binary mask the left-out units are exact zeros
    and the cosine similarities equal those of the selected units alone.
    """
    hz = contrastive_code(pooled, model, args, mask)
    if getattr(args, "standardize_code", False):
        # Per view, over the batch, so every unit's spread counts equally in the cosine (--standardize-code).
        hz = standardize_contrastive_code(hz)
    elif getattr(args, "center_code", False):
        hz = hz - hz.mean(1, keepdim=True)
    content_indices = [list(range(hz.shape[-1]))]
    if args.contrastive_loss_type == "barlow_twins":
        # With a mask, Barlow Twins keeps only the selected units (losses.barlow_twins_loss); a
        # zeroed unit would otherwise enter its correlation matrix as 0/0.
        loss = losses.barlow_twins_loss(
            hz,
            estimated_content_indices=content_indices,
            subsets=[(0, 1)],
            soft_content_mask=mask,
            lambd=args.bt_lambda,
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


def lesion_decorrelation(pooled, args, units=None):
    """Sum of squared correlations between the keypoint coordinates and the content block.

    Both are standardized over the batch (both views). The content side is detached, so it
    keeps what it encodes and only the branch moves. A branch that duplicates a factor the
    content already carries (brain size, captured through any boundary that moves with it)
    is penalized without the factor ever being named. ``units`` are a learned mask's content
    units; by default the first ``content_channels``.
    """

    def standardize(z):
        return (z - z.mean(0)) / z.std(0).clamp_min(1e-6)

    coords = standardize(pooled[:, args.latent_dim :])
    if units is None:
        content = standardize(pooled[:, : args.content_channels].detach())
    else:
        content = standardize(pooled[:, units].detach())
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


LOSS_NAMES = {"infonce": "InfoNCE", "barlow_twins": "Barlow Twins", "vicreg": "VICReg"}
# The unweighted parts each patch objective logs as patch_<part>; InfoNCE has none.
PATCH_LOSS_PARTS = {"infonce": (), "barlow_twins": ("on_diag", "off_diag"), "vicreg": ("sim", "var", "cov", "std")}
# What the pointwise loss logs as pointwise_<part> beside its weighted total; only align and entropy are loss terms.
POINTWISE_PARTS = ("align", "entropy", "eff_rank", "residual_rank", "view_r", "saturated", "subject_share")
VECTOR_PREDICTION_PARTS = (
    "mean",
    "t1_to_flair",
    "flair_to_t1",
    "unweighted",
    "unweighted_t1_to_flair",
    "unweighted_flair_to_t1",
    "weight_min",
    "weight_max",
    "foreground_fraction",
)


def patch_contrastive_loss(patches, model, args, sim_metric, criterion, foreground=None, mask=None, interior=None):
    """Registered positions are paired; under InfoNCE, negatives are subjects at the SAME position.

    Only image-derived features and the input's own brain mask enter this loss; there
    is no lesion sampling or access to generator factor labels. All grid positions
    participate unless --patch-foreground-mask drops the always-background ones or
    ``interior`` (--interior-only, (P,) bool on the CPU) those that read padding.
    A learned ``mask`` multiplies the shared head's units, as in ``contrastive_loss``;
    the separate spatial head has only content units, so it is read whole.

    ``--patch-loss-type barlow_twins|vicreg`` replaces the InfoNCE with a negative-free loss whose
    statistics are pooled over positions (``losses.pooled_patch_vicreg_terms`` says why); its
    unweighted terms ride along as ``._patch_terms`` for logging.
    """
    keep = interior
    if getattr(args, "patch_foreground_mask", False):
        if foreground is None:
            raise ValueError("--patch-foreground-mask needs the batch's brain masks")
        # Hand-built option namespaces may omit the parser's default.
        keep = foreground_positions(foreground, args.train_patch_grid, getattr(args, "patch_foreground_thresh", 0.05))
        if interior is not None:
            inside = keep.cpu() & interior
            keep = inside if bool(inside.any()) else interior
    if keep is not None:
        patches = patches[..., keep.to(patches.device)]
    if mask is not None and model.spatial_readout is None:
        a, b = (patches * mask.unsqueeze(-1)).chunk(2, dim=0)
    else:
        a, b = patches[:, : args.content_channels, :].chunk(2, dim=0)
    hz = torch.stack((a, b), dim=0)  # views, subjects, content channels, positions
    # A configured projector acts on channels, never on the position axis.
    hz = model.project(hz.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    # Hand-built option namespaces may omit the parser's default.
    kind = getattr(args, "patch_loss_type", "infonce")
    if kind == "barlow_twins":
        # Folding (subject, position) rows after centring each position makes every channel's variance
        # the across-subject variance averaged over positions; patch_stat="per_position" would instead
        # standardise each position by its own spread and amplify the flat ones.
        loss = losses.barlow_twins_loss(
            hz,
            estimated_content_indices=[list(range(hz.shape[2]))],
            subsets=[(0, 1)],
            lambd=args.bt_lambda,
            center_mode="position",
            patch_stat="fold",
        )
        diag = loss._contrastive_diag
        loss = loss.squeeze()
        loss._patch_terms = {"on_diag": diag["on_diag_loss"], "off_diag": diag["off_diag_loss"]}
        return loss
    if kind == "vicreg":
        terms = losses.pooled_patch_vicreg_terms(hz)
        loss = args.vicreg_sim_coeff * terms["sim"] + args.vicreg_std_coeff * terms["var"]
        loss = loss + args.vicreg_cov_coeff * terms["cov"]
        loss._patch_terms = {key: float(value) for key, value in terms.items()}
        return loss
    # Hand-built option namespaces may omit the parser's default.
    if getattr(args, "standardize_patch_code", False):
        hz = standardize_patch_code(hz, double=getattr(args, "patch_center_mode", "position") == "double")
    return losses.patch_infonce_loss(
        hz,
        sim_metric=sim_metric,
        criterion=criterion,
        tau=args.tau,
        estimated_content_indices=[list(range(hz.shape[2]))],
        subsets=[(0, 1)],
        cross_view_negs_only=args.cross_view_negs_only,
        similarity=getattr(args, "patch_similarity", "cosine"),
    ).squeeze()


def standardize_patch_code(hz, double=False):
    """--standardize-patch-code on a (views, subjects, units, positions) patch code.

    Each view's code is centred over the batch at every position, then each unit is divided by its spread
    pooled over subjects and positions. Within a position, a unit that varies there more than it does elsewhere
    stands out and a unit that is flat there stays near zero; dividing by each position's own spread would
    promote that flat unit's noise to unit variance. The patch InfoNCE then L2-normalises every (subject,
    position) vector, so this does not quiet a position alike in every subject.

    ``double`` (--patch-center-mode double) also removes each subject's mean over positions before the
    scaling, leaving only the subject x position interaction.
    """
    centred = hz - hz.mean(1, keepdim=True)
    if double:
        centred = centred - centred.mean(3, keepdim=True)
    spread = centred.square().sum((1, 3), keepdim=True) / ((hz.shape[1] - 1) * hz.shape[3])
    return centred / (spread.sqrt() + 1e-6)


def pointwise_rows(args, subjects, positions, foreground=None, interior=None):
    """(subject, position) indices of the rows this step's pointwise loss reads, as two CPU tensors.

    Up to --pointwise-rows rows, uniformly without replacement from every subject at every position, so one step
    mixes many of both. ``interior`` (--interior-only, (P,) bool) drops the positions that read padding.
    ``foreground``, the batch's brain masks (2B, 1, D, H, W), drops a subject's cells with under
    --patch-foreground-thresh brain in both of its views. That is per subject, unlike the patch loss's rule: a
    background cell is one constant for every subject, and no entropy term can spread a constant. Drawn on the CPU,
    so the rows replay on any device.
    """
    with torch.no_grad():
        candidates = torch.ones(subjects, positions, dtype=torch.bool)
        if interior is not None:
            candidates = candidates & interior
        if foreground is not None:
            pooled = torch.nn.functional.adaptive_avg_pool3d(foreground.float(), tuple(args.train_patch_grid))
            fraction = pooled.flatten(1).cpu()
            threshold = args.patch_foreground_thresh
            candidates = candidates & ((fraction[:subjects] >= threshold) | (fraction[subjects:] >= threshold))
        rows = candidates.nonzero()
        if len(rows) < 2:
            raise ValueError(f"The pointwise loss has {len(rows)} usable (subject, position) rows; it needs two")
        return rows[torch.randperm(len(rows))[: args.pointwise_rows]].T


def pointwise_exclusions(subject, position, negatives):
    """(M, M) bool on the CPU: pairs of drawn rows that are not negatives under --pointwise-negatives.

    pooled leaves out only the row itself. strict also leaves out rows sharing its subject or its position.
    """
    exclude = torch.eye(len(subject), dtype=torch.bool)
    if negatives == "strict":
        exclude = exclude | (subject[:, None] == subject[None]) | (position[:, None] == position[None])
    return exclude


def subject_share(maps, positions):
    """Share of a (B, C, P) code's variance over (subject, position) rows that lies between subjects at a position.

    Near 0 the code fills its box with a template: what differs is where a cell is, not who the subject is, which
    an entropy term on raw features pooled over positions cannot tell apart (paper/feature-map-formulation.md, 6).
    """
    code = maps[:, :, positions]
    pooled = code.permute(1, 0, 2).flatten(1).var(dim=1).mean()
    return float(code.var(dim=0).mean() / pooled.clamp_min(1e-12))


def view_agreement(first, second):
    """Mean over units of the two views' correlation across (M, C) rows: do the views agree unit by unit."""
    first, second = first - first.mean(0), second - second.mean(0)
    return float(((first * second).sum(0) / (first.norm(dim=0) * second.norm(dim=0)).clamp_min(1e-12)).mean())


def pointwise_loss(patches, args, foreground=None, interior=None):
    """Alignment + joint entropy of the content maps over pooled rows: --pointwise-loss-weight.

    ``patches`` is (2B, latent, P), the map each view's encoder gives on --train-patch-grid, before any standardisation
    or projection. A row is one subject at one position, and every statistic reads rows drawn across both, so nothing
    is computed per position or per subject. The residual inputs subtract each view's batch mean at every position.
    ``losses.pointwise_alignment_entropy`` has the two terms. Its parts and the code's health ride along as
    ``._pointwise_terms`` for logging: eff_rank and residual_rank (channels the rows use, with and without the
    template), view_r (the views' unit-by-unit agreement on the residual), saturated (share of values outside
    [0.01, 0.99]) and subject_share. None of those enters the loss.
    """
    b = patches.shape[0] // 2
    maps = patches[:b, : args.content_channels], patches[b:, : args.content_channels]
    subject, position = pointwise_rows(args, b, maps[0].shape[2], foreground, interior)
    row = subject.to(patches.device), position.to(patches.device)

    def rows(kind):
        views = maps if kind == "raw" else [view - view.mean(0, keepdim=True) for view in maps]
        return [view[row[0], :, row[1]] for view in views]

    terms = losses.pointwise_alignment_entropy(
        rows(args.pointwise_align_input),
        rows(args.pointwise_entropy_input),
        pointwise_exclusions(subject, position, args.pointwise_negatives).to(patches.device),
        args.pointwise_tau,
        args.pointwise_kernel,
        args.cross_view_negs_only,
    )
    loss = terms["align"] + terms["entropy"]
    with torch.no_grad():
        code, residual = rows("raw"), rows("residual")
        values = torch.cat(code)
        kept = position.unique().to(patches.device)
        loss._pointwise_terms = {
            "align": float(terms["align"]),
            "entropy": float(terms["entropy"]),
            "eff_rank": effective_rank(code[0]),
            "residual_rank": effective_rank(residual[0]),
            "view_r": view_agreement(*residual),
            "saturated": float(((values < 0.01) | (values > 0.99)).float().mean()),
            "subject_share": sum(subject_share(view, kept) for view in maps) / 2,
        }
    return loss


def training_objective(model, images, args, sim_metric, criterion, masks=None):
    """Keep the old global path exact when local training is disabled.

    ``masks`` are the batch's brain masks in image order, used by --patch-foreground-mask.
    Without them the inputs' nonzero support stands in; the synthetic inputs are zero
    outside the brain mask, so the two agree.

    A learned content mask is drawn once here, so every term of the step uses one selection.
    """
    weight = getattr(args, "patch_loss_weight", 0.0)
    # Hand-built option namespaces may omit the parser's defaults.
    pointwise_weight = getattr(args, "pointwise_loss_weight", 0.0)
    local = weight > 0 or pointwise_weight > 0
    if local:
        pooled, patches = model.global_and_patch_features(images, args.train_patch_grid, n_views=2)
    else:
        pooled = model(images, pool_only=True, n_views=2)[2][0]
    # Hand-built option namespaces may omit the parser's default.
    content_mask = model.sample_content_mask(hard=getattr(args, "mask_estimator", "relaxed") == "straight_through")
    residual = getattr(model, "residual_heads", None)
    # The primary objective remains the original IMAGE vector's objective. Appended
    # residual units belong only to the explicitly weighted auxiliary objective.
    global_pooled = pooled[:, : model.latent_dim] if residual is not None else pooled
    global_loss = contrastive_loss(global_pooled, model, args, sim_metric, criterion, content_mask)
    foreground = None
    if local and getattr(args, "patch_foreground_mask", False):
        foreground = masks if masks is not None else images != 0
    interior = None
    if getattr(args, "interior_only", False):
        interior = model.interior_positions(images.shape[2:], args.train_patch_grid)
    patch_loss = (
        patch_contrastive_loss(patches, model, args, sim_metric, criterion, foreground, content_mask, interior)
        if weight > 0
        else global_loss.new_zeros(())
    )
    weighted = weight * patch_loss
    global_weighted = getattr(args, "global_loss_weight", 1.0) * global_loss
    total = global_weighted + weighted if weight > 0 else global_weighted
    terms = {"global": global_loss, "patch": patch_loss, "patch_weighted": weighted}
    if residual is not None:
        # Keep the original image objective and add a separately weighted joint objective.
        # A zero global weight reproduces the initial residual-head experiment.
        image_code, residual_code = model.split_residual_codes(pooled)
        auxiliary, residual_terms = residual.joint_loss(
            image_code,
            residual_code,
            raw_weight=args.residual_head_image_alignment_weight,
            residual_weight=args.residual_head_alignment_weight,
            uniform_weight=args.residual_head_uniformity_weight,
        )
        auxiliary_weight = getattr(args, "residual_head_auxiliary_weight", 1.0)
        auxiliary_weighted = auxiliary_weight * auxiliary
        if auxiliary_weight > 0:
            total = total + auxiliary_weighted
        terms["global_weighted"] = global_weighted
        terms["residual_heads"] = auxiliary
        terms["residual_heads_auxiliary_raw"] = auxiliary
        terms["residual_heads_auxiliary_weighted"] = auxiliary_weighted
        terms.update({f"residual_heads_{key}": value for key, value in residual_terms.items()})
    # Not loss terms: the unweighted parts of a negative-free patch loss, for the logs.
    for key, value in getattr(patch_loss, "_patch_terms", {}).items():
        terms[f"patch_{key}"] = global_loss.new_tensor(value)
    if pointwise_weight > 0:
        terms["pointwise"] = pointwise_loss(patches, args, foreground, interior)
        terms["pointwise_weighted"] = pointwise_weight * terms["pointwise"]
        total = total + terms["pointwise_weighted"]
        for key, value in terms["pointwise"]._pointwise_terms.items():
            terms[f"pointwise_{key}"] = global_loss.new_tensor(value)
    prediction_weight = getattr(args, "vector_prediction_weight", 0.0)
    if prediction_weight > 0:
        if model.vector_prediction is None:
            raise ValueError("Vector prediction is enabled but the model has no vector decoder")
        prediction, diagnostics = vector_prediction_loss(
            model.vector_prediction,
            pooled[:, : model.content_channels],
            images,
            masks if masks is not None else images != 0,
            deviation_strength=args.vector_prediction_deviation_strength,
            weight_cap=args.vector_prediction_weight_cap,
        )
        terms["vector_prediction"] = prediction
        terms["vector_prediction_weighted"] = prediction_weight * prediction
        total = total + terms["vector_prediction_weighted"]
        for key, value in diagnostics.items():
            terms[f"vector_prediction_{key}"] = value
    if getattr(args, "lesion_keypoints", 0) > 0:
        lesion_loss = lesion_contrastive_loss(pooled, model, args, sim_metric, criterion, images)
        terms["lesion"] = lesion_loss
        # Hand-built option namespaces may omit the parser's default.
        terms["lesion_weighted"] = getattr(args, "lesion_loss_weight", 1.0) * lesion_loss
        total = total + terms["lesion_weighted"]
        decorrelation_weight = getattr(args, "lesion_decorrelation_weight", 0.0)
        if decorrelation_weight > 0:
            units = model.content_units() if content_mask is not None else None
            terms["lesion_decorrelation"] = lesion_decorrelation(pooled, args, units)
            total = total + decorrelation_weight * terms["lesion_decorrelation"]
    if getattr(model, "mask_mode", "fixed") == "learned_size":
        # Each content unit costs this much, so a unit stays content only while it lowers the
        # InfoNCE by more; the gates' probabilities make the count differentiable.
        terms["mask_size"] = model.expected_content_size()
        terms["mask_size_weighted"] = args.mask_size_weight * terms["mask_size"]
        total = total + terms["mask_size_weighted"]
    if content_mask is not None and model.training:
        # Not a loss term: the mask weight this step's draw put on units outside the evaluated
        # selection; for binary draws, the number of style units swapped in. Eval draws have none.
        selected = torch.zeros_like(content_mask[0], dtype=torch.bool)
        selected[model.content_units()] = True
        terms["mask_leak"] = content_mask.detach()[0][~selected].sum()
    return (pooled, total, terms)


def mask_state(model):
    """A learned content mask as evaluated: its units, every logit, and the selection's margin.

    The margin is the smallest selected logit minus the largest left-out one, i.e. how far the
    selection is from changing. Under a learned size it is the smallest |logit|, how far the
    nearest gate is from flipping, and the size and expected size (sum of the gates'
    probabilities) are reported too. None under the fixed split.
    """
    if model.mask_logits is None:
        return None
    logits = model.mask_logits.detach().float().cpu()
    units = model.content_units()
    if model.mask_mode == "learned_size":
        return {
            "content_units": units,
            "logits": logits.tolist(),
            "margin": float(logits.abs().min()),
            "content_size": len(units),
            "expected_size": float(torch.sigmoid(logits).sum()),
        }
    selected = torch.zeros(len(logits), dtype=torch.bool)
    selected[units] = True
    return {
        "content_units": units,
        "logits": logits.tolist(),
        "margin": float(logits[selected].min() - logits[~selected].max()),
    }


def effective_rank(feat):
    """Covariance participation ratio: variance concentration, not intrinsic dimension.

    A value near one means one direction dominates the covariance. Weak directions
    may still carry information, so recovery probes and raw variance are needed
    alongside this diagnostic to assess a suspected collapse.
    """
    x = feat.detach().float()
    if x.device.type == "mps":
        # This small diagnostic is outside autograd; keep eigvalsh off MPS.
        x = x.cpu()
    x = x - x.mean(dim=0, keepdim=True)
    # A one-unit block (a learned size can shrink to it) has a 0-d covariance.
    ev = torch.linalg.eigvalsh(torch.atleast_2d(torch.cov(x.T))).clamp(min=0)
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


BEST_METRIC_KEYS = {
    "block_mcc": "content->content/block_mcc",
    "ridge_r2": "content->content/informativeness_ridge",
    "val_loss": "val/loss",
}


def best_metric_value(flat, floor_flat, name):
    """The scalar model_best.pt is selected on, floor-subtracted where a floor exists.

    Raw block-MCC on this generator is mostly floor — an untrained encoder scores ~0.38 —
    so selecting on it would rank checkpoints partly by how much untrained structure the
    architecture happens to carry. The delta ranks them by what training added. The
    held-out loss has no such floor, and lower is better, so it is negated instead.
    """
    v = flat.get(BEST_METRIC_KEYS[name])
    if v is None or not np.isfinite(v):
        return None
    if name == "val_loss":
        return -float(v)
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


def residual_head_gradient_report(model, terms, args):
    """Parameter-gradient RMS for each weighted objective, without changing ``.grad`` or model state.

    Each RMS uses all trainable parameters in its group, including zero/unused entries.
    Compare loss terms within one parameter group; differently sized groups are not norm ratios.
    """
    branch = getattr(model, "residual_heads", None)
    if branch is None:
        return None
    groups = {
        "image": tuple(p for p in model.image_parameters() if p.requires_grad),
        "residual": tuple(p for part in (branch.encoders, branch.head) for p in part.parameters() if p.requires_grad),
        "projector": tuple(p for p in branch.projector.parameters() if p.requires_grad),
    }
    parameters = tuple(p for group in groups.values() for p in group)
    auxiliary_weight = getattr(args, "residual_head_auxiliary_weight", 1.0)
    objectives = {
        "image_primary": terms["global_weighted"],
        "image_alignment": auxiliary_weight
        * args.residual_head_image_alignment_weight
        * terms["residual_heads_image_alignment"],
        "residual_alignment": auxiliary_weight
        * args.residual_head_alignment_weight
        * terms["residual_heads_residual_alignment"],
        "uniformity": auxiliary_weight * args.residual_head_uniformity_weight * terms["residual_heads_uniformity"],
        "auxiliary": terms["residual_heads_auxiliary_weighted"],
    }
    report = {}
    for name, objective in objectives.items():
        gradients = (
            torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            if objective.requires_grad and parameters
            else (None,) * len(parameters)
        )
        offset = 0
        for group_name, group in groups.items():
            subset = gradients[offset : offset + len(group)]
            offset += len(group)
            count = sum(parameter.numel() for parameter in group)
            square_sum = sum(
                float(gradient.detach().float().square().sum()) for gradient in subset if gradient is not None
            )
            report[f"{name}/{group_name}"] = (square_sum / count) ** 0.5 if count else 0.0
    return report


def clip_encoder_gradients(model, max_norm):
    """Keep the raw-residual and global paths independent during norm clipping too."""
    if getattr(model, "residual_heads", None) is not None:
        image = [p for p in model.image_parameters() if p.requires_grad]
        if image:
            torch.nn.utils.clip_grad_norm_(image, max_norm)
        return torch.nn.utils.clip_grad_norm_(model.residual_heads.parameters(), max_norm)
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


CAUSAL_VIEWS = ("view1", "view2")


def causal_discovery_eval(model, val_dataset, device, args, save_dir, step, writer=None, initial=None):
    """Label-free PC on each view's content channels, scored against the generator's SCM.

    ``eval.causal.latent_causal_discovery`` applied in-loop: every content channel becomes one
    scalar per subject, PC runs on those alone, and labels enter only afterwards, to name nodes
    through one Hungarian match on ``|corr|``. Each F1 prints beside the random-naming null, the
    step-0 floor (``initial``) and PC on the TRUE factors (the ceiling at this row count). The
    best alpha is chosen against the truth, so read it relative to those rows, not alone.
    """
    from types import SimpleNamespace

    from eval.causal.latent_causal_discovery import evaluate_reduced, reduce_channels
    from eval.protocol.score_checkpoint import encode_blocks

    print(f"  [eval] causal discovery @ step {step} ...", flush=True)
    grid = tuple(args.causal_eval_grid)
    blocks = encode_blocks(
        model, val_dataset, device, args.batch_size, model.content_channels, None if grid == (1, 1, 1) else grid
    )
    adj = blocks["adjacency"]
    if adj is None:
        scm = getattr(getattr(val_dataset, "_inner", val_dataset), "scm", None)
        adj = scm.get("adj") if isinstance(scm, dict) else None
    if adj is None:
        print("    no causal SCM on the validation set; skipped", flush=True)
        return None
    cli = SimpleNamespace(
        reduce=args.causal_eval_reduce,
        alphas=args.causal_eval_alphas,
        indep_test=args.causal_eval_indep_test,
        max_cond_set=args.causal_eval_max_cond_set,
        null_draws=args.causal_eval_null_draws,
    )
    z = np.asarray(blocks["z_content"], dtype=np.float64)
    positions = int(np.prod(grid))
    out = {}
    for view, flat in zip(CAUSAL_VIEWS, blocks["content"]):
        maps = flat.reshape(len(flat), -1, positions)
        try:
            out[view] = evaluate_reduced(reduce_channels(maps, cli.reduce), z, adj, cli, view)
        except Exception as exc:  # a dead representation must not end the run
            out[view] = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
    ceiling = (initial or {}).get("ceiling")
    if ceiling is None:
        ceiling = evaluate_reduced(z, z, adj, cli, "truth-ceiling")
    out["ceiling"] = ceiling

    print(f"    --- label-free PC on content channels ({cli.reduce}, grid {'x'.join(map(str, grid))}) ---", flush=True)
    head = f"      {'':<14s}{'alpha':>7s}{'F1':>7s}{'null F1':>9s}{'SHD':>6s}{'null SHD':>10s}{'match |r|':>11s}"
    print(head + (f"{'floor F1':>10s}{'Δ floor':>9s}" if initial else ""), flush=True)
    for name in (*CAUSAL_VIEWS, "ceiling"):
        row = out[name]
        if "best" not in row:
            print(f"      {name:<14s}{row.get('reason', 'no result')}", flush=True)
            continue
        b = row["best"]
        line = (
            f"      {name:<14s}{b['alpha']:>7.2f}{b['f1']:>7.3f}{b['null']['f1_mean']:>9.3f}"
            f"{b['skeleton_shd']:>6d}{b['null']['shd_mean']:>10.1f}{np.mean(row['match_quality']):>11.3f}"
        )
        floor_row = (initial or {}).get(name) or {}
        if name != "ceiling" and "best" in floor_row:
            line += f"{floor_row['best']['f1']:>10.3f}{b['f1'] - floor_row['best']['f1']:>+9.3f}"
        print(line, flush=True)
        if writer is not None and name != "ceiling":
            writer.add_scalar(f"causal_discovery/{name}/f1", b["f1"], step)
            writer.add_scalar(f"causal_discovery/{name}/f1_over_null", b["f1_over_null"], step)
            writer.add_scalar(f"causal_discovery/{name}/skeleton_shd", b["skeleton_shd"], step)
            writer.add_scalar(f"causal_discovery/{name}/match_quality", float(np.mean(row["match_quality"])), step)
            for sweep in row["alpha_sweep"]:
                writer.add_scalar(f"causal_discovery/{name}/f1_alpha{sweep['alpha']:g}", sweep["f1"], step)

    with open(os.path.join(save_dir, f"causal_step{step}.json"), "w") as fp:
        json.dump(
            dict(protocol=dict(readout="none", naming="hungarian_abs_corr", grid=list(grid)), **out), fp, indent=2
        )
    return out


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
    mask = mask_state(model)
    if mask is not None:
        size = f", size {mask['content_size']}" if "content_size" in mask else ""
        print(
            f"      content units {mask['content_units']} (learned mask{size}, margin {mask['margin']:.2f})", flush=True
        )
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

    prediction = vector_prediction_report(model, val_dataset, args, device)
    if prediction is not None:
        print(
            f"    vector prediction held-out MSE {prediction['mean']:.4f} "
            f"(T1->FLAIR {prediction['t1_to_flair']:.4f}, FLAIR->T1 {prediction['flair_to_t1']:.4f})",
            flush=True,
        )
        for key, value in prediction.items():
            flat[f"vector_prediction/{key}"] = value
    residual_report = None
    if getattr(model, "residual_heads", None) is not None:
        from eval.encoder.residual_head_recovery import residual_head_recovery

        residual_report = residual_head_recovery(model, val_dataset, device, args.batch_size)
        if residual_report.get("status") == "complete":
            print(
                "    --- image / residual / combined: held-out per-factor ridge R² ---",
                flush=True,
            )
            for view, blocks in residual_report["views"].items():
                for factor in residual_report["factor_names"]:
                    values = [blocks[name]["r2"][factor] for name in ("image", "residual", "combined")]
                    formatted = " / ".join("n/a" if value is None else f"{value:.3f}" for value in values)
                    print(f"      {view:<6s} {factor:<26s} {formatted}", flush=True)
                    for name, value in zip(("image", "residual", "combined"), values):
                        if value is not None:
                            flat[f"residual_heads/{view}/{name}/r2_{factor}"] = value
    if writer is not None:
        for k, v in flat.items():
            if np.isfinite(v):
                writer.add_scalar(f"dci_synthetic/{k}", v, step)

    payload = {k: float(v) for k, v in flat.items()}
    payload["per_factor"] = scores  # nested, so the existing flat keys stay at top level
    if mask is not None:
        payload["content_mask"] = mask
    if spread is not None:
        payload["attention_effective_fraction"] = spread
    if lesion is not None:
        payload["lesion_branch"] = lesion
    if residual_report is not None:
        payload["residual_heads"] = residual_report
    with open(os.path.join(save_dir, f"dci_step{step}.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    return flat, scores


@torch.no_grad()
def vector_prediction_report(model, dataset, args, device):
    """Held-out prediction errors, including a short final batch; never refit target buffers."""
    if getattr(args, "vector_prediction_weight", 0.0) <= 0:
        return None
    from eval.encoder.spatial_recovery_monitor import preserve_training_state

    sums, subjects = {}, 0
    with preserve_training_state(model, device):
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.data_seed),
        )
        for batch in loader:
            images = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)
            if model.vector_prediction.target == "patch_distribution" and "mask" not in batch:
                raise ValueError(
                    "Patch-distribution targets require anatomical brain masks; image != 0 excludes T1 lesions"
                )
            masks = (
                torch.cat(batch["mask"], dim=0).as_subclass(torch.Tensor).to(device) if "mask" in batch else images != 0
            )
            pooled = model(images, pool_only=True, n_views=2)[2][0]
            _, terms = vector_prediction_loss(
                model.vector_prediction,
                pooled[:, : model.content_channels],
                images,
                masks,
                deviation_strength=args.vector_prediction_deviation_strength,
                weight_cap=args.vector_prediction_weight_cap,
            )
            count = images.shape[0] // 2
            for key in (
                "mean",
                "t1_to_flair",
                "flair_to_t1",
                "unweighted",
                "unweighted_t1_to_flair",
                "unweighted_flair_to_t1",
            ):
                sums[key] = sums.get(key, 0.0) + float(terms[key]) * count
            subjects += count
    if not subjects:
        raise ValueError("Vector prediction validation needs subjects")
    return {key: value / subjects for key, value in sums.items()}


@torch.no_grad()
def held_out_objective(model, batches, device, args):
    """Average the held-out training objective over full batches only.

    InfoNCE depends on the number of negatives, so omit a short final batch here.
    Alignment and retrieval use every validation row in the representation pass.
    """
    sim_metric, criterion = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    sums, full = {}, 0
    for data in batches:
        x = torch.cat(data["image"], dim=0).as_subclass(torch.Tensor).to(device)
        b = x.shape[0] // 2
        if b == args.batch_size:
            masks = None
            if getattr(args, "vector_prediction_weight", 0.0) > 0 and "mask" in data:
                masks = torch.cat(data["mask"], dim=0).as_subclass(torch.Tensor).to(device)
            pooled, loss, terms = training_objective(model, x, args, sim_metric, criterion, masks)
            for key, value in {"loss": loss, **terms}.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            full += 1
    return {f"val/{key}": value / full for key, value in sums.items()} if full else {}


def evaluate_real(model, val_dataset, device, args, save_dir, step, writer=None, floor=None, previous=None):
    """Real-data counterpart of ``evaluate``: no generator factors, so no DCI.

    Reports the held-out objective, cross-view retrieval and
    ``eval.metrics.cross_reconstruction``'s content/style probes, with subject-disjoint
    diagnosis/demographic probes, each against ``floor``, the step-0
    result. The validation volumes are read once and then held in RAM by that module.
    Returns the flat metrics twice, so the call matches ``evaluate``'s (flat, floor) shape.
    """
    print(f"  [eval] held-out subjects @ step {step} ...", flush=True)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batches = cross_reconstruction._get_cached_batches(loader, len(loader))
    flat = held_out_objective(model, batches, device, args)
    reps = cross_reconstruction._collect_representations(model, loader, args, device, len(loader))
    flat.update(
        cross_reconstruction.evaluate_content_style_separation(
            model, loader, args, device, len(loader), subject_probes=True, representations=reps
        )
    )
    flat = {key: float(value) for key, value in flat.items()}

    print("    --- held-out subjects ---", flush=True)
    mask = mask_state(model)
    if mask is not None:
        size = f", size {mask['content_size']}" if "content_size" in mask else ""
        print(
            f"      content units {mask['content_units']} (learned mask{size}, margin {mask['margin']:.2f})", flush=True
        )
    print_alignment_report(flat, floor=floor, previous=previous)

    spread = attention_spread(model, val_dataset, device, args.batch_size)
    if spread is not None:
        print(
            "    --- global attention: effective positions / map size, per head (1 = GAP) ---",
            flush=True,
        )
        for view, values in spread.items():
            print(f"      {view:<8s}" + "".join(f"{v:8.3f}" for v in values), flush=True)
            if writer is not None:
                for head, value in enumerate(values):
                    writer.add_scalar(
                        f"attention_pool/{view}/head{head}_effective_fraction",
                        value,
                        step,
                    )

    if writer is not None:
        for key, value in flat.items():
            if np.isfinite(value):
                writer.add_scalar(key, value, step)
    finest = reps[min(reps)]
    feature_name = f"alignment_step{step}.npz"
    metadata = {
        "run_dir": os.path.abspath(save_dir),
        "step": step,
        "content_units": model.content_units(),
        "content_channels": int(flat["alignment/content_channels"]),
        "feature_space": "content before projector or loss-side standardization",
        "probe_seed": int(flat["alignment/probe_seed"]),
        "probe_folds": int(flat["alignment/probe_folds"]),
        "protocol": "encoder_modality_gap.diagnose; subject-disjoint CV and fold-fitted scaling",
        "features_file": feature_name,
    }
    np.savez_compressed(
        os.path.join(save_dir, feature_name),
        content_t1=finest["content_v0"],
        content_flair=finest["content_v1"],
        subjects=finest["subjects"],
        metadata_json=np.array(json.dumps(metadata)),
    )
    payload = dict(flat)
    payload["alignment_evaluation"] = metadata
    if mask is not None:
        payload["content_mask"] = mask
    if spread is not None:
        payload["attention_effective_fraction"] = spread
    with open(os.path.join(save_dir, f"separation_step{step}.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    return flat, flat


def build_encoder(args):
    """The run's encoder, on the CPU, initialized from the global torch RNG as it stands."""
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
        dense_readout_downsample=args.dense_readout_downsample,
        dense_readout_grid=args.dense_readout_grid,
        resnet_norm=args.resnet_norm,
        resnet_output_stride=args.resnet_output_stride,
        separate_spatial_readout=args.separate_spatial_readout,
        global_pool=args.global_pool,
        attention_pool_heads=args.attention_pool_heads,
        attention_pool_frequencies=args.attention_pool_frequencies,
        attention_pool_width=args.attention_pool_width,
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
        mask_mode=args.mask_mode,
        mask_init_gap=args.mask_init_gap,
        content_bound=args.content_bound,
        activation=args.activation,
        vector_prediction_grid=(
            tuple(args.vector_prediction_grid) if getattr(args, "vector_prediction_weight", 0.0) > 0 else None
        ),
        vector_prediction_hidden=getattr(args, "vector_prediction_hidden", 128),
        vector_prediction_target=getattr(args, "vector_prediction_target", "spatial"),
        vector_prediction_components=getattr(args, "vector_prediction_components", 8),
        vector_prediction_patch_size=getattr(args, "vector_prediction_patch_size", 3),
        vector_prediction_patch_scales=tuple(getattr(args, "vector_prediction_patch_scales", (1, 2))),
        vector_prediction_patch_projections=getattr(args, "vector_prediction_patch_projections", 8),
        vector_prediction_patch_seed=getattr(args, "vector_prediction_patch_seed", 1729),
        residual_head_units=getattr(args, "residual_head_units", 0),
        residual_head_hidden=getattr(args, "residual_head_hidden", 16),
        residual_head_components=getattr(args, "residual_head_components", 16),
        residual_head_input_size=args.res,
        residual_head_joint_dim=getattr(args, "residual_head_joint_dim", 0),
        residual_head_freeze_image=getattr(args, "residual_head_freeze_image", True),
        residual_head_alignment_mode=getattr(args, "residual_head_alignment_mode", "population"),
    )
    if model.residual_heads is not None and getattr(args, "residual_head_auxiliary_weight", 1.0) == 0:
        # A zero-auxiliary control must also avoid zero-gradient AdamW decay in the
        # residual path through the concatenated exported vector.
        model.residual_heads.requires_grad_(False)
        model.train(model.training)
    return model


def fit_vector_prediction_template(model, train_dataset, args, device):
    """Fit fixed image targets on training subjects without consuming training RNG streams."""
    predictor = model.vector_prediction
    if predictor is None:
        return None
    from eval.encoder.spatial_recovery_monitor import preserve_training_state

    restored = bool(predictor.template_fitted)
    count = min(args.vector_prediction_template_subjects, len(train_dataset))
    if count < 1:
        raise ValueError("Vector prediction needs at least one training subject to fit its template")
    if not restored:
        with preserve_training_state(model, device):
            random.seed(args.data_seed)
            np.random.seed(args.data_seed)
            torch.manual_seed(args.data_seed)
            loader = DataLoader(
                Subset(train_dataset, range(count)),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(args.data_seed),
            )
            predictor.reset_template()
            for batch in loader:
                images = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)
                if predictor.target == "patch_distribution" and "mask" not in batch:
                    raise ValueError(
                        "Patch-distribution targets require anatomical brain masks; image != 0 excludes T1 lesions"
                    )
                masks = (
                    torch.cat(batch["mask"], dim=0).as_subclass(torch.Tensor).to(device)
                    if "mask" in batch
                    else images != 0
                )
                predictor.accumulate_template(images, masks)
            predictor.finalize_template()
    template_hash = hashlib.sha256(predictor.template.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    distribution = predictor.target in ("quantile", "patch_distribution")
    patch_distribution = predictor.target == "patch_distribution"
    descriptor = None
    if patch_distribution:
        from models.patch_distribution import PATCH_QUANTILES

        descriptor = {
            "patch_size": args.vector_prediction_patch_size,
            "scales": list(args.vector_prediction_patch_scales),
            "projections": args.vector_prediction_patch_projections,
            "projection_seed": args.vector_prediction_patch_seed,
            "projection_filters_sha256": hashlib.sha256(
                predictor.patch_filters.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "probabilities": list(PATCH_QUANTILES),
            "geometry": "nonoverlapping average pooling at each scale, then valid stride-1 patch convolution",
            "output_order": "scale, projection, quantile",
            "objective": "standardized projected-quantile MSE; approximate sliced Wasserstein-2",
        }
    return {
        "target": predictor.target,
        "components": predictor.components if distribution else None,
        "interior_cells": predictor.target_interior.sum(1).tolist() if distribution else None,
        "decoder_outputs": predictor.decoders[0][-1].out_features,
        "patch_distribution": descriptor,
        "source": "checkpoint" if restored else "training_subjects",
        "dataset_indices": None if restored else list(range(count)),
        "subjects_per_view": predictor.template_subject_count.detach().cpu().tolist(),
        "grid": list(predictor.grid),
        "foreground_occupancy_threshold": 0.5,
        "template_sha256": template_hash,
        "template_position_coverage": (predictor.template_counts > 0).float().flatten(1).mean(1).cpu().tolist(),
    }


def read_init_from(args):
    """--init-from: the checkpoint's state dict and provenance, refused before the run directory exists.

    The file is read once, so the bytes hashed are the bytes loaded. A missing file, bytes other than
    --init-from-sha256, or another architecture stops the launch without leaving a directory that
    --require-new-run would refuse next time. The architecture check loads strictly into a model this
    run builds, then compares the fixed content mask, which a different --content-channels changes
    without changing any shape. That model draws from the torch RNG before the trainer seeds it, so the
    run's own streams are untouched.
    """
    path = args.init_from
    checkpoint = path if os.path.isfile(path) else os.path.join(path, "model.pt")
    if not os.path.isfile(checkpoint):
        raise SystemExit(f"--init-from {path}: need a checkpoint file, or a run directory with model.pt")
    if getattr(args, "residual_head_units", 0) > 0:
        check_residual_image_settings(args, checkpoint)
    with open(checkpoint, "rb") as fp:
        raw = fp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if args.init_from_sha256 is not None and args.init_from_sha256 != digest:
        raise SystemExit(
            f"--init-from {checkpoint}: its SHA-256 is {digest}, not --init-from-sha256 {args.init_from_sha256}; "
            "the checkpoint changed since that was recorded"
        )
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    model = build_encoder(args)
    content_mask = model.content_mask.clone()
    try:
        model.load_state_dict(state)
    except (RuntimeError, TypeError) as error:
        raise SystemExit(
            f"--init-from {checkpoint}: not this run's architecture; strict load_state_dict refused it:\n{error}"
        ) from None
    if not torch.equal(model.content_mask, content_mask):
        raise SystemExit(
            f"--init-from {checkpoint}: not this run's architecture; its fixed content mask marks units "
            f"{torch.nonzero(model.content_mask[0]).flatten().tolist()} where this run's marks "
            f"{torch.nonzero(content_mask[0]).flatten().tolist()} (--content-channels)"
        )
    provenance = {"path": path, "checkpoint": os.path.abspath(checkpoint), "sha256": digest}
    provenance.update(source_step=None, source_status=None)
    # model.pt is saved with each evaluation, just before training_progress.json records its step.
    progress = os.path.join(os.path.dirname(checkpoint), "training_progress.json")
    if os.path.basename(checkpoint) == "model.pt" and os.path.isfile(progress):
        with open(progress) as fp:
            recorded = json.load(fp)
        provenance.update(source_step=recorded.get("step"), source_status=recorded.get("status"))
    return state, provenance


def check_residual_image_settings(args, checkpoint):
    """Reject behavior changes that strict tensor loading cannot detect, including activation changes."""
    settings_path = os.path.join(os.path.dirname(checkpoint), "settings.json")
    if not os.path.isfile(checkpoint) or not os.path.isfile(settings_path):
        raise SystemExit("Residual-head initialization needs a checkpoint with a sibling settings.json")
    with open(settings_path) as stream:
        source = json.load(stream)
    defaults = vars(encoder_args.parse_args([]))
    architecture = (
        "encoder_architecture",
        "hidden_channels",
        "res_channels",
        "nb_res_layers",
        "downscale_factor",
        "latent_dim",
        "content_channels",
        "no_separate_encoders",
        "conv_readout",
        "encoder_head_hidden",
        "norm_type",
        "activation",
        "global_pool",
        "content_bound",
        "separate_spatial_readout",
        "contrastive_proj_dim",
        "mask_mode",
        "lesion_keypoints",
        "vector_prediction_weight",
    )
    differences = [
        name for name in architecture if source.get(name, defaults.get(name)) != getattr(args, name, defaults.get(name))
    ]
    if differences:
        raise SystemExit("Residual-head image initialization architecture differs: " + ", ".join(differences))
    return source, settings_path


def read_residual_image_init(args):
    """Validate a native image checkpoint before creating an experimental run directory."""
    path = getattr(args, "residual_head_image_init", None)
    if path is None:
        return None, None
    checkpoint = path if os.path.isfile(path) else os.path.join(path, "model.pt")
    source, settings_path = check_residual_image_settings(args, checkpoint)
    if source.get("residual_head_units", 0) > 0:
        raise SystemExit("Use --init-from for a complete two-path checkpoint")
    with open(checkpoint, "rb") as stream:
        raw = stream.read()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    base_args = SimpleNamespace(**vars(args))
    base_args.residual_head_units = 0
    base_args.residual_head_freeze_image = False
    base = build_encoder(base_args)
    expected_mask = base.content_mask.clone()
    try:
        base.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise SystemExit(f"--residual-head-image-init strict image load failed: {error}") from None
    if not torch.equal(base.content_mask, expected_mask):
        raise SystemExit("--residual-head-image-init has a different content mask")
    provenance = {
        "checkpoint": os.path.abspath(checkpoint),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "settings": os.path.abspath(settings_path),
        "initialization": "image pathway only; residual pathway and joint projector initialized afresh",
    }
    return state, provenance


def load_residual_image_state(model, state):
    """Load every original parameter/buffer strictly while retaining the new branch and export mask."""
    full = model.state_dict()
    expected = {name for name in full if not name.startswith("residual_heads.")}
    if set(state) != expected:
        raise ValueError("Image initialization does not contain exactly the original image pathway")
    for name, value in state.items():
        if name != "content_mask":
            full[name] = value
    model.load_state_dict(full, strict=True)


def fit_residual_heads(model, train_dataset, args, device):
    """Fit residual preprocessing and initialize both code gauges using training subjects only."""
    branch = getattr(model, "residual_heads", None)
    if branch is None:
        return None
    from eval.encoder.spatial_recovery_monitor import preserve_training_state

    restored = bool(branch.fitted)
    count = int(branch.fit_subjects) if restored else args.residual_head_subjects
    if not restored:
        with preserve_training_state(model, device):
            loader = DataLoader(
                Subset(train_dataset, range(count)),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(args.data_seed),
            )
            images = torch.cat(
                [torch.stack(batch["image"], dim=1)[:, :, 0].as_subclass(torch.Tensor) for batch in loader]
            )
            image_digest = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
            branch.fit(images)
            raw, residual = ([], []), ([], [])
            with torch.no_grad():
                for start in range(0, count, args.batch_size):
                    part = images[start : start + args.batch_size]
                    paired = torch.cat([part[:, :1], part[:, 1:]]).to(device)
                    code = model(paired, pool_only=True, n_views=2)[2][0]
                    first, second = model.split_residual_codes(code)
                    for view, (a, b) in enumerate(zip(first.chunk(2), second.chunk(2))):
                        raw[view].append(a)
                        residual[view].append(b)
                branch.calibrate_codes(
                    torch.cat([torch.cat(parts) for parts in raw]),
                    torch.cat([torch.cat(parts) for parts in residual]),
                )
    else:
        image_digest = None
    digest = hashlib.sha256()
    for name in ("brain", "mean", "basis", "residual_sd", "interior"):
        digest.update(getattr(branch, name).detach().cpu().contiguous().numpy().tobytes())
    return {
        "source": "checkpoint" if restored else "training_subjects",
        "dataset_split": "train",
        "dataset_indices": list(range(count)),
        "subjects": count,
        "components": branch.components,
        "interior_cells": int(branch.interior.sum()),
        "training_images_sha256": image_digest,
        "preprocessing_sha256": digest.hexdigest(),
    }


def main():
    args = parse_args()
    configure_encoder_runtime(vars(args))
    device = select_encoder_device(args.device, args.no_cuda)
    init_state = init_from = None
    if args.init_from is not None:
        init_state, init_from = read_init_from(args)
        args.init_from_sha256 = init_from["sha256"]
    image_init_state, image_init = read_residual_image_init(args)
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

    real = args.dataset_name != "synthetic"
    train_dataset = make_dataset(args, "train", args.num_train_samples)
    val_dataset = make_dataset(args, "val", args.num_val_samples)
    if real:
        describe_real_split(args, train_dataset, val_dataset, save_dir)
    # Real volumes are augmented on the CPU, so workers matter there; they need their own
    # augmentation streams, and persistence spares a respawn every epoch of ~35 steps.
    workers = {"worker_init_fn": seed_augmentation_worker, "persistent_workers": True}
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.loader_seed),
        **cluster_batches.loader_batching(
            args, lambda: stratified_batch_sampler(args, train_dataset, device, save_dir)
        ),
        **(workers if real and args.num_workers > 0 else {}),
    )
    stratification = getattr(train_loader.batch_sampler, "summary", None)

    # Dataset construction resets the global RNG. Seed initialization explicitly,
    # while the loader's private generator isolates batch order from architecture.
    torch.manual_seed(args.model_seed)
    model = build_encoder(args).to(device)
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
    if init_state is not None:
        # After any normative fit, so every buffer is the source's too.
        model.load_state_dict(init_state)
        del init_state
        step_note = "" if init_from["source_step"] is None else f", recorded at its step {init_from['source_step']}"
        print(
            f"init: warm start from {init_from['checkpoint']} (sha256 {init_from['sha256']}{step_note}); fresh "
            "AdamW and step counter. model_init.pt and the step-0 evaluation are this model, not an untrained floor",
            flush=True,
        )
    if image_init_state is not None:
        load_residual_image_state(model, image_init_state)
        del image_init_state
    residual_fit = fit_residual_heads(model, train_dataset, args, device)
    if residual_fit is not None:
        with open(os.path.join(save_dir, "residual_heads_fit.json"), "w") as stream:
            json.dump({**residual_fit, "image_initialization": image_init}, stream, indent=2)
        print(
            f"residual heads: {args.content_channels} image + {model.residual_units} residual content units; "
            f"joint uniformity in {args.residual_head_joint_dim} dimensions; "
            f"image pathway {'frozen' if args.residual_head_freeze_image else 'trainable'}; "
            f"PCA fitted on {residual_fit['subjects']} training subjects ({residual_fit['source']}). "
            f"Image global weight {args.global_loss_weight:g}; joint auxiliary weight "
            f"{args.residual_head_auxiliary_weight:g}; alignment gauges {args.residual_head_alignment_mode}. "
            "Step 0 is the combined initialization, not a fresh-image floor.",
            flush=True,
        )
    prediction_template = fit_vector_prediction_template(model, train_dataset, args, device)
    if prediction_template is not None:
        with open(os.path.join(save_dir, "vector_prediction_template.json"), "w") as fp:
            json.dump(prediction_template, fp, indent=2)
        print(
            f"vector prediction template: {prediction_template['subjects_per_view']} subjects per view, "
            f"source {prediction_template['source']}, grid {args.vector_prediction_grid}; "
            "fixed during optimization; no labels or validation images",
            flush=True,
        )
    pool_name = {"gap": "GAP", "attention": "attention pool", "moments": "GAP + moments"}[model.global_pool]
    if real:
        # parse_args' rule (conv floors, ResNet rounds up); a probe forward would move BatchNorm statistics.
        stride = model.backbone_stride
        backbone = [
            -(-s // stride) if args.encoder_architecture == "resnet18" else s // stride for s in args.spatial_size
        ]
        print(
            f"input: {'x'.join(map(str, args.spatial_size))} volumes at {args.image_spacing:g} mm -> "
            f"{'x'.join(map(str, backbone))} backbone map (stride {stride})",
            flush=True,
        )
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
    elif args.conv_readout == "dense":
        head = model.to_encoding
        print(
            f"encoder: conv, dense readout: {head.downsample} stride-2 convs -> {head.channels} x {head.grid}^3 -> "
            f"flatten {head.channels * head.grid ** 3} -> {args.encoder_head_hidden} -> {args.latent_dim}; "
            f"{sum(p.numel() for p in head.parameters())} parameters, shared by both views. No pooling and no "
            "patch outputs",
            flush=True,
        )
    if model.attention_pool is not None:
        pool = model.attention_pool
        if pool.mix is None:
            width = f"{pool.channels // pool.num_heads} channels"
        else:
            width = f"{pool.channels} channels (full width, mixed back to {pool.channels})"
        print(
            f"global pool: attention replaces GAP; {pool.num_heads} heads x {width}, "
            f"{pool.num_frequencies} Fourier frequencies/axis, "
            f"{sum(p.numel() for p in pool.parameters())} parameters; starts as exact GAP. "
            "Patch readouts still average each bin.",
            flush=True,
        )
    if model.moment_pool is not None:
        print(
            f"global pool: GAP plus first spatial moments; {3 * model.moment_pool.channels} moment features into "
            f"the readout's first layer through {model.moment_pool.weight.numel()} zero-initialized weights, so it "
            "starts as exact GAP. Patch readouts still average each bin.",
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
    if model.mask_logits is not None:
        mask_lr = args.lr * args.mask_lr_scale
        half_gap = args.mask_init_gap / 2
        estimator = args.mask_estimator.replace("_", "-")
        kind = (
            f"learned size of at most {args.latent_dim} units, one binary-Concrete gate per unit (tau 1, "
            f"{estimator}), {args.mask_size_weight:g} per expected content unit"
            if args.mask_mode == "learned_size"
            else f"learned top-{args.content_channels} of {args.latent_dim} units, Gumbel-softmax (tau 1, {estimator})"
        )
        print(
            f"content mask: {kind}, shared by both views; starts at units "
            f"0-{args.content_channels - 1} (logits +{half_gap:g}/-{half_gap:g}, initial gap {args.mask_init_gap:g}); "
            f"logit lr {mask_lr:g} (--lr x {args.mask_lr_scale:g}), "
            "no weight decay",
            flush=True,
        )
        budget = mask_lr * args.train_steps
        if budget <= half_gap:
            print(
                f"  NOTE: --lr x --mask-lr-scale x --train-steps = {budget:g}, at most half the initial "
                f"logit gap ({args.mask_init_gap:g}). The selection may move slowly; consider a larger "
                "--mask-lr-scale or smaller --mask-init-gap. This is a heuristic; Adam updates depend on gradient history",
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
        patch_objective = {
            "infonce": "",
            "barlow_twins": f" (lambda {args.bt_lambda:g})",
            "vicreg": f" (sim {args.vicreg_sim_coeff:g}, std {args.vicreg_std_coeff:g}, cov {args.vicreg_cov_coeff:g})",
        }[args.patch_loss_type]
        if args.patch_loss_type != "infonce":
            patch_objective += ", centred at each position, statistics pooled over positions"
        elif args.standardize_patch_code:
            subject_mean = " and per subject over positions" if args.patch_center_mode == "double" else ""
            patch_objective += (
                f" ({args.patch_similarity} similarity, code centred at each position{subject_mean} and scaled per "
                "unit over positions)"
            )
        print(
            f"objective: global {LOSS_NAMES[args.contrastive_loss_type]} + {args.patch_loss_weight:g} * patch "
            f"{LOSS_NAMES[args.patch_loss_type]}{patch_objective}; grid={args.train_patch_grid}; {readout}; "
            f"both losses update the backbones; {positions}; no target labels",
            flush=True,
        )
    if args.pointwise_loss_weight > 0:
        negatives = "all but the row itself" if args.pointwise_negatives == "pooled" else "other subject and position"
        foreground = (
            f", cells with >= {args.patch_foreground_thresh:g} brain in that subject"
            if args.patch_foreground_mask
            else ""
        )
        print(
            f"objective: {args.global_loss_weight:g} * global {LOSS_NAMES[args.contrastive_loss_type]} + "
            f"{args.pointwise_loss_weight:g} * pointwise alignment ({args.pointwise_align_input}) + entropy "
            f"({args.pointwise_entropy_input}, {args.pointwise_kernel} kernel, tau {args.pointwise_tau:g}); "
            f"{args.pointwise_rows} (subject, position) rows per step on grid {args.train_patch_grid}{foreground}; "
            f"negatives: {negatives}",
            flush=True,
        )
    elif args.global_loss_weight != 1:
        print(f"objective: the global loss is weighted {args.global_loss_weight:g}", flush=True)
    if args.vector_prediction_weight > 0:
        if args.vector_prediction_target == "spatial":
            target = (
                f"{args.vector_prediction_grid}, separate target-modality decoders; foreground MSE, deviation strength "
                f"{args.vector_prediction_deviation_strength:g}, cap {args.vector_prediction_weight_cap:g}"
            )
        elif args.vector_prediction_target == "quantile":
            target = (
                f"quantiles of the other view's interior residual, minus {args.vector_prediction_components} "
                "population directions, separate target-modality decoders; standardized MSE"
            )
        else:
            target = (
                f"projected quantiles of {args.vector_prediction_patch_size}^3 residual patches on grid "
                f"{args.vector_prediction_grid}, scales {args.vector_prediction_patch_scales}, "
                f"{args.vector_prediction_patch_projections} fixed directions, seed {args.vector_prediction_patch_seed}; "
                f"minus {args.vector_prediction_components} population directions; standardized projected-quantile MSE "
                "(approximate sliced Wasserstein-2), no absolute patch positions"
            )
        print(
            f"objective: + {args.vector_prediction_weight:g} * vector-only cross-view residual prediction; "
            f"{args.content_channels} exported content units -> {args.vector_prediction_hidden} -> {target}; "
            "no spatial bypass, style units or lesion labels",
            flush=True,
        )
    if args.encoder_architecture == "conv":
        print(
            f"receptive field: {receptive_field_voxels(args.downscale_factor, args.nb_res_layers)} voxels per axis at "
            f"map stride {model.backbone_stride}, through the convolutions{' and residual blocks' if args.nb_res_layers else ''}; "
            f"{args.norm_type} norm, {args.activation}, content bound {args.content_bound}",
            flush=True,
        )
    if args.interior_only:
        volume = tuple(args.spatial_size) if real else (args.res,) * 3
        kept = int(model.interior_positions(volume, args.train_patch_grid).sum())
        print(f"interior only: {kept}/{np.prod(args.train_patch_grid)} positions read no padding", flush=True)
        if args.norm_type == "group":
            print("  NOTE: GroupNorm statistics still pool over the whole volume, padding included", flush=True)

    if model.mask_logits is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        # The logits rank units rather than weight anything: decay would pull them toward an even
        # draw. As for the VQ-VAE trainer's mask group, they get their own lr and no decay.
        weights = [p for p in model.parameters() if p is not model.mask_logits]
        mask_group = {"params": [model.mask_logits], "lr": args.lr * args.mask_lr_scale, "weight_decay": 0.0}
        optimizer = torch.optim.AdamW([{"params": weights}, mask_group], lr=args.lr)
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    torch.save(model.state_dict(), os.path.join(save_dir, "model_init.pt"))
    batch_order = hashlib.sha256()
    input_images = hashlib.sha256() if args.hash_training_inputs else None
    started = time.perf_counter()
    last_loss_terms = None
    last_gradient_report = None

    def save_progress(step, status):
        payload = {
            "status": status,
            "step": step,
            "subjects_seen_per_view": step * args.batch_size,
            "batch_order_sha256": batch_order.hexdigest(),
            "training_input_sha256": (input_images.hexdigest() if input_images is not None else None),
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
            "moment_pool_parameter_count": (model.moment_pool.weight.numel() if model.moment_pool is not None else 0),
            "dense_readout_parameter_count": (
                sum(p.numel() for p in model.to_encoding.parameters()) if model.readout_type == "dense" else 0
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
            "mps_fallback": (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") if device == "mps" else None),
            "patch_loss_weight": args.patch_loss_weight,
            "patch_loss_type": (args.patch_loss_type if args.patch_loss_weight > 0 else None),
            "train_patch_grid": (
                args.train_patch_grid if args.patch_loss_weight > 0 or args.pointwise_loss_weight > 0 else None
            ),
            "global_loss_weight": args.global_loss_weight,
            "pointwise_loss_weight": args.pointwise_loss_weight,
            "vector_prediction_weight": args.vector_prediction_weight,
            "vector_prediction_target": args.vector_prediction_target,
            "vector_prediction_template": prediction_template,
            "vector_prediction_parameter_count": (
                sum(p.numel() for p in model.vector_prediction.parameters())
                if model.vector_prediction is not None
                else 0
            ),
            "activation": model.activation,
            "content_bound": model.content_bound,
            "receptive_field_voxels": (
                receptive_field_voxels(args.downscale_factor, args.nb_res_layers)
                if args.encoder_architecture == "conv"
                else None
            ),
            "last_loss_terms": last_loss_terms,
            "residual_head_gradient_report": last_gradient_report,
            "init_from": init_from,
            "residual_heads_fit": residual_fit,
            "residual_heads_image_initialization": image_init,
            "initialization_kind": (
                "protected_image_checkpoint"
                if image_init is not None
                else "checkpoint"
                if init_from is not None
                else "random"
            ),
        }
        if model.mask_logits is not None:
            payload["content_mask"] = {**mask_state(model), "lr": args.lr * args.mask_lr_scale}
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
    if stratification is not None and writer is not None:
        for key in ("clusters_kept", "subjects_per_epoch", "subjects_eligible_as_anchors", "subject_draws_per_epoch"):
            if key in stratification:
                writer.add_scalar(f"stratify/{key}", stratification[key], 0)
        for name, value in stratification["sd_ratio"].items():
            if value is not None:
                writer.add_scalar(f"stratify/sd_ratio/{name}", value, 0)

    # Step-0 eval doubles as the untrained floor: same architecture, same seed, no training.
    # Per-factor R² needs it more than the block means do -- a localised factor can read
    # 0.2 from a random encoder, so the raw number alone cannot say whether it was learned.
    floor = floor_flat = None
    spatial_initial = causal_initial = None
    evaluation = evaluate_real if real else evaluate
    if args.spatial_recovery_eval:
        from eval.encoder.spatial_recovery_monitor import evaluate_spatial_recovery
    if args.floor_eval:
        model.eval()
        floor_flat, floor = evaluation(model, val_dataset, device, args, save_dir, 0, writer)
        if args.spatial_recovery_eval:
            spatial_initial = evaluate_spatial_recovery(model, vars(args), device, save_dir, 0, writer=writer)
        if args.causal_eval:
            causal_initial = causal_discovery_eval(model, val_dataset, device, args, save_dir, 0, writer)
        model.train()

    # Step 0 is the untrained floor, not a candidate: with every delta at or below zero it
    # would win on a tie and save an untrained encoder as "best".
    best = {"value": None, "step": None, "raw": None}
    previous_real = floor_flat if real else None

    step = 0
    patch_parts = PATCH_LOSS_PARTS[args.patch_loss_type] if args.patch_loss_weight > 0 else ()
    pointwise_terms = ()
    if args.pointwise_loss_weight > 0:
        pointwise_terms = ("pointwise", "pointwise_weighted", *(f"pointwise_{part}" for part in POINTWISE_PARTS))
    prediction_terms = ()
    if args.vector_prediction_weight > 0:
        prediction_terms = (
            "vector_prediction",
            "vector_prediction_weighted",
            *(f"vector_prediction_{part}" for part in VECTOR_PREDICTION_PARTS),
        )
    residual_terms = ()
    if getattr(model, "residual_heads", None) is not None:
        residual_terms = (
            "residual_heads",
            "global_weighted",
            "residual_heads_auxiliary_raw",
            "residual_heads_auxiliary_weighted",
            *(
                f"residual_heads_{part}"
                for part in (
                    "image_alignment",
                    "residual_alignment",
                    "uniformity",
                    "total",
                    "joint_variance",
                    "joint_effective_rank",
                    "projected_variance",
                    "image_variance",
                    "residual_variance",
                    "image_raw_variance",
                    "residual_raw_variance",
                    "image_effective_rank",
                    "residual_effective_rank",
                    "projected_effective_rank",
                    "code_sd_min",
                    "projector_weight_norm",
                )
            ),
        )

    def fresh_running():
        return dict(
            loss=0.0,
            rank=0.0,
            n=0,
            global_loss=0.0,
            patch=0.0,
            patch_weighted=0.0,
            lesion=0.0,
            decorrelation=0.0,
            mask_leak=0.0,
            mask_size=0.0,
            **{f"patch_{part}": 0.0 for part in patch_parts},
            **{term: 0.0 for term in pointwise_terms},
            **{term: 0.0 for term in prediction_terms},
            **{term: 0.0 for term in residual_terms},
        )

    running = fresh_running()
    logged_units = model.content_units()
    model.train()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            batch_order.update(batch["index"].cpu().numpy().astype("<i8").tobytes())
            # Real volumes arrive as MONAI MetaTensors; a plain view skips their per-op metadata work.
            images = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor)
            if input_images is not None:
                input_images.update(images.contiguous().numpy().tobytes())
            x = images.to(device)  # (2B, 1, res, res, res)
            masks = (
                torch.cat(batch["mask"], dim=0).as_subclass(torch.Tensor).to(device)
                if args.patch_foreground_mask or args.vector_prediction_weight > 0
                else None
            )
            if args.patch_foreground_mask and masks is not None and step == 0:
                kept = int(foreground_positions(masks, args.train_patch_grid, args.patch_foreground_thresh).sum())
                print(f"  patch foreground mask: first batch keeps {kept}/{np.prod(args.train_patch_grid)} positions")

            pooled, loss, terms = training_objective(model, x, args, sim_metric, criterion, masks)
            if residual_terms and not torch.isfinite(loss):
                raise ValueError(f"Nonfinite residual-head objective at step {step + 1}")

            gradient_every = getattr(args, "residual_head_gradient_every", 0)
            if residual_terms and gradient_every > 0 and (step + 1) % gradient_every == 0:
                gradients = residual_head_gradient_report(model, terms, args)
                last_gradient_report = {
                    "step": step + 1,
                    "parameter_gradient_rms": gradients,
                }
                with open(os.path.join(save_dir, "residual_head_gradients.jsonl"), "a") as stream:
                    stream.write(json.dumps(last_gradient_report) + "\n")
                print(
                    f"  gradient RMS @ {step + 1}: image primary {gradients['image_primary/image']:.3g}, "
                    f"auxiliary {gradients['auxiliary/image']:.3g}; residual auxiliary "
                    f"{gradients['auxiliary/residual']:.3g}; projector auxiliary {gradients['auxiliary/projector']:.3g}",
                    flush=True,
                )
                if writer is not None:
                    for name, value in gradients.items():
                        writer.add_scalar(f"train/gradient_rms/{name}", value, step + 1)

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
            for part in patch_parts:
                running[f"patch_{part}"] += last_loss_terms[f"patch_{part}"]
            for term in pointwise_terms:
                running[term] += last_loss_terms[term]
            for term in prediction_terms:
                running[term] += last_loss_terms[term]
            for term in residual_terms:
                running[term] += last_loss_terms[term]
            running["lesion"] += last_loss_terms.get("lesion", 0.0)
            running["decorrelation"] += last_loss_terms.get("lesion_decorrelation", 0.0)
            running["mask_leak"] += last_loss_terms.get("mask_leak", 0.0)
            running["mask_size"] += last_loss_terms.get("mask_size", 0.0)
            rank_units = model.unit_split(pooled.shape[1])[0] if residual_terms else model.content_units()
            running["rank"] += effective_rank(pooled[: pooled.shape[0] // 2, rank_units])
            running["n"] += 1
            step += 1

            if step % args.log_every == 0:
                n = max(running["n"], 1)
                mask_info = mask_state(model)
                print(
                    f"step {step:6d} | {'total' if prediction_terms or residual_terms else 'contrastive'} {running['loss']/n:.4f} "
                    f"| content eff_rank {running['rank']/n:.2f}/{len(rank_units)}"
                    + (
                        f" | global {running['global_loss']/n:.4f} | patch {running['patch']/n:.4f} "
                        f"| weighted patch {running['patch_weighted']/n:.4f}"
                        + (
                            " (" + ", ".join(f"{part} {running[f'patch_{part}']/n:.4f}" for part in patch_parts) + ")"
                            if patch_parts
                            else ""
                        )
                        if args.patch_loss_weight > 0
                        else ""
                    )
                    + (
                        (f" | global {running['global_loss']/n:.4f}" if args.patch_loss_weight <= 0 else "")
                        + f" | pointwise {running['pointwise']/n:.4f} (align {running['pointwise_align']/n:.4f}, entropy "
                        f"{running['pointwise_entropy']/n:.4f}) | cell eff_rank {running['pointwise_eff_rank']/n:.2f}"
                        f"/{args.content_channels} (residual {running['pointwise_residual_rank']/n:.2f}), view r "
                        f"{running['pointwise_view_r']/n:.3f}, saturated {running['pointwise_saturated']/n:.3f}, subject share "
                        f"{running['pointwise_subject_share']/n:.3f}"
                        if pointwise_terms
                        else ""
                    )
                    + (f" | lesion InfoNCE {running['lesion']/n:.4f}" if args.lesion_keypoints > 0 else "")
                    + (
                        f" | image global {running['global_loss']/n:.4f} "
                        f"(weighted {running['global_weighted']/n:.4f}) "
                        f"| auxiliary {running['residual_heads_auxiliary_raw']/n:.4f} "
                        f"(weighted {running['residual_heads_auxiliary_weighted']/n:.4f}) "
                        f"| image align {running['residual_heads_image_alignment']/n:.4f} "
                        f"| residual align {running['residual_heads_residual_alignment']/n:.4f} "
                        f"| joint MMD {running['residual_heads_uniformity']/n:.4f} "
                        f"| joint rank {running['residual_heads_joint_effective_rank']/n:.2f}"
                        if residual_terms
                        else ""
                    )
                    + (
                        f" | vector prediction {running['vector_prediction']/n:.4f} "
                        f"(T1->FLAIR {running['vector_prediction_t1_to_flair']/n:.4f}, "
                        f"FLAIR->T1 {running['vector_prediction_flair_to_t1']/n:.4f}, "
                        f"unweighted {running['vector_prediction_unweighted']/n:.4f})"
                        if prediction_terms
                        else ""
                    )
                    + (
                        f" | lesion decorrelation {running['decorrelation']/n:.4f}"
                        if args.lesion_decorrelation_weight > 0
                        else ""
                    )
                    + (
                        f" | mask leak {running['mask_leak']/n:.3f}, margin {mask_info['margin']:.2f}"
                        if mask_info is not None
                        else ""
                    )
                    + (
                        f" | content size {mask_info['content_size']} (expected {running['mask_size']/n:.2f})"
                        if mask_info is not None and "content_size" in mask_info
                        else ""
                    ),
                    flush=True,
                )
                if residual_terms:
                    print(
                        f"  within-view ranks: image {running['residual_heads_image_effective_rank']/n:.2f}, "
                        f"residual {running['residual_heads_residual_effective_rank']/n:.2f}, "
                        f"projected {running['residual_heads_projected_effective_rank']/n:.2f}; "
                        f"raw variance: image {running['residual_heads_image_raw_variance']/n:.3g}, "
                        f"residual {running['residual_heads_residual_raw_variance']/n:.3g}; "
                        f"minimum code SD {running['residual_heads_code_sd_min']/n:.3g}",
                        flush=True,
                    )
                if mask_info is not None and mask_info["content_units"] != logged_units:
                    print(f"  content units changed: {logged_units} -> {mask_info['content_units']}", flush=True)
                    logged_units = mask_info["content_units"]
                if writer is not None:
                    writer.add_scalar(
                        ("train/total" if prediction_terms or residual_terms else "train/contrastive"),
                        running["loss"] / n,
                        step,
                    )
                    if mask_info is not None:
                        writer.add_scalar("mask/leak", running["mask_leak"] / n, step)
                        writer.add_scalar("mask/margin", mask_info["margin"], step)
                        if "content_size" in mask_info:
                            writer.add_scalar("mask/content_size", mask_info["content_size"], step)
                            writer.add_scalar("mask/expected_size", running["mask_size"] / n, step)
                        for unit, logit in enumerate(mask_info["logits"]):
                            writer.add_scalar(f"mask/logit_unit{unit}", logit, step)
                    writer.add_scalar("train/content_eff_rank", running["rank"] / n, step)
                    if args.lesion_keypoints > 0:
                        writer.add_scalar("train/lesion_infonce", running["lesion"] / n, step)
                    if args.lesion_decorrelation_weight > 0:
                        writer.add_scalar("train/lesion_decorrelation", running["decorrelation"] / n, step)
                    if args.patch_loss_weight > 0:
                        # InfoNCE runs keep their old tags (global_infonce, patch_infonce, ...).
                        kind = args.patch_loss_type
                        for tag, key in (
                            (f"global_{args.contrastive_loss_type}", "global_loss"),
                            (f"patch_{kind}", "patch"),
                            (f"patch_{kind}_weighted", "patch_weighted"),
                            *((f"patch_{kind}_{part}", f"patch_{part}") for part in patch_parts),
                        ):
                            writer.add_scalar(f"train/{tag}", running[key] / n, step)
                    elif pointwise_terms:
                        writer.add_scalar(
                            f"train/global_{args.contrastive_loss_type}", running["global_loss"] / n, step
                        )
                    for term in pointwise_terms:
                        writer.add_scalar(f"train/{term}", running[term] / n, step)
                    for term in prediction_terms:
                        writer.add_scalar(f"train/{term}", running[term] / n, step)
                    for term in residual_terms:
                        writer.add_scalar(f"train/{term}", running[term] / n, step)
                running = fresh_running()

            if step % args.eval_every == 0 or step == args.train_steps:
                model.eval()
                eval_kwargs = {"previous": previous_real} if real else {}
                flat, _ = evaluation(
                    model, val_dataset, device, args, save_dir, step, writer, floor=floor, **eval_kwargs
                )
                if real:
                    previous_real = flat
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                save_progress(step, "running")
                if args.spatial_recovery_eval:
                    evaluate_spatial_recovery(
                        model, vars(args), device, save_dir, step, initial=spatial_initial, writer=writer
                    )
                if args.causal_eval:
                    causal_discovery_eval(
                        model, val_dataset, device, args, save_dir, step, writer, initial=causal_initial
                    )

                if args.best_metric != "none":
                    value = best_metric_value(flat, floor_flat, args.best_metric)
                    if value is not None and (best["value"] is None or value > best["value"]):
                        best = {"value": value, "step": step, "raw": flat.get(BEST_METRIC_KEYS[args.best_metric])}
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
                                    "floor_subtracted": floor_flat is not None and args.best_metric != "val_loss",
                                    "raw": best["raw"],
                                },
                                fp,
                                indent=2,
                            )
                        shown = f"{best['raw']:.4f}" if args.best_metric == "val_loss" else f"{value:+.4f}"
                        print(f"    new best {args.best_metric} {shown} -> model_best.pt", flush=True)
                model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    save_progress(step, "complete")
    if best["step"] is not None:
        shown = f"{best['raw']:.4f}" if args.best_metric == "val_loss" else f"{best['value']:+.4f}"
        print(
            f"best {args.best_metric} {shown} at step {best['step']} -> model_best.pt "
            f"(model.pt is the last step, {step}). Score the best one with "
            f"--checkpoint model_best.pt.",
            flush=True,
        )
        if best["step"] < step:
            # Worth saying out loud: past the peak the run is spending compute making the
            # representation worse, which at this dataset size is the expected shape.
            print(f"  NOTE: peak was {step - best['step']} steps before the end.", flush=True)
    print(f"done. checkpoints + {'evaluation' if real else 'DCI'} logs in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
