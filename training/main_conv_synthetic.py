"""Encoder-only multi-view contrastive learning on the 3D synthetic data or real T1/FLAIR MRI.

3D encoders trained with InfoNCE or Barlow Twins, without reconstruction.
The default ``conv`` architecture uses the VQ-VAE convolutional backbone with
an affine readout. ``--encoder-architecture resnet18`` uses a 3D adaptation of
the upstream image encoder: ResNet-18 -> GAP -> Linear -> LeakyReLU -> Linear.
This architecture option does not change the loss, data, or view-sharing policy.
``--global-pool attention`` replaces the GAP before either readout with multi-head
attention pooling (``models.attention_pool``), which starts as exact GAP.
``--mask-mode learned`` replaces the fixed content block (the first ``--content-channels``
units) with a top-k Gumbel-softmax content selector over learnable per-unit logits.
``--stratify-batches oracle|code`` draws every training batch from one k-means cluster of the
training subjects (MCL's batch rule; ``data.cluster_batches``), so InfoNCE's negatives share the
clustered anatomy.

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
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import data.cluster_batches as cluster_batches
import eval.metrics.cross_reconstruction as cross_reconstruction
import eval.metrics.dci as dci
import training.encoder_args as encoder_args
import training.losses as losses
from data.datasets import MyCustomDataset, SyntheticBrainDataset
from data.splits import split_summary, subject_level_split
from eval.adni.encoder_alignment import print_alignment_report
from models.multiview_encoder import MultiviewConvEncoder
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
            + "; MCL clusters the code of the bank the stage-0 encoder was trained on",
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
    """--stratify-batches: k-means on the training subjects, then every batch from one cluster.

    oracle clusters the true anatomy latents, read from the generator without rendering a view;
    code clusters a stage-0 run's content code (``stage0_code``). Prints and writes to
    stratification.json the clusters kept and, per true content factor, the SD inside a batch's
    cluster over the SD across the bank: the anatomy should sit near K^(-1/d), the lesion near 1.
    The summary also rides on the sampler as ``summary``.
    """
    names = dci.content_factor_names(args.n_content, args.synthetic_lesion_target)
    anatomy = cluster_batches.anatomy_columns(names)
    started = time.perf_counter()
    if args.stratify_batches == "oracle":
        factors = cluster_batches.bank_latents(train_dataset)
        features = factors[:, anatomy]
        source = "the true anatomy latents (" + ", ".join(names[j] for j in anatomy) + "), lesion latents left out"
    else:
        features, factors, source = stage0_code(args, train_dataset, device)
    labels = cluster_batches.kmeans_labels(features, args.stratify_clusters, args.data_seed)
    sampler = cluster_batches.ClusterBatchSampler(labels, args.batch_size, args.loader_seed)
    summary = cluster_batches.summarize(labels, args.batch_size, factors, names, anatomy, args.stratify_clusters)
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
        role = ("clustered" if args.stratify_batches == "oracle" else "anatomy") if j in anatomy else ""
        print(f"      {name:<20s}{summary['sd_ratio'][name]:7.3f}  {role}", flush=True)
    print(
        f"    anatomy mean {summary['sd_ratio_anatomy_mean']:.3f}; theory K^(-1/{len(anatomy)}) = "
        f"{summary['theory_k_pow_minus_1_over_d']:.3f}",
        flush=True,
    )
    payload = {
        "mode": args.stratify_batches,
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
    b = pooled.shape[0] // 2
    hz = torch.stack([pooled[:b], pooled[b:]], dim=0)  # (2, B, C)
    if mask is None:
        hz = model.project(hz[..., : args.content_channels])
    else:
        hz = hz[..., : model.latent_dim] * mask
    if getattr(args, "standardize_code", False):
        # Per view, over the batch, so every unit's spread counts equally in the cosine (--standardize-code).
        hz = (hz - hz.mean(1, keepdim=True)) / (hz.std(1, keepdim=True) + 1e-6)
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


def patch_contrastive_loss(patches, model, args, sim_metric, criterion, foreground=None, mask=None):
    """Registered positions are paired; negatives are subjects at the SAME position.

    Only image-derived features and the input's own brain mask enter this loss; there
    is no lesion sampling or access to generator factor labels. All grid positions
    participate unless --patch-foreground-mask drops the always-background ones.
    A learned ``mask`` multiplies the shared head's units, as in ``contrastive_loss``;
    the separate spatial head has only content units, so it is read whole.
    """
    if getattr(args, "patch_foreground_mask", False):
        if foreground is None:
            raise ValueError("--patch-foreground-mask needs the batch's brain masks")
        # Hand-built option namespaces may omit the parser's default.
        keep = foreground_positions(foreground, args.train_patch_grid, getattr(args, "patch_foreground_thresh", 0.05))
        patches = patches[..., keep.to(patches.device)]
    if mask is not None and model.spatial_readout is None:
        a, b = (patches * mask.unsqueeze(-1)).chunk(2, dim=0)
    else:
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

    A learned content mask is drawn once here, so every term of the step uses one selection.
    """
    weight = getattr(args, "patch_loss_weight", 0.0)
    if weight > 0:
        pooled, patches = model.global_and_patch_features(images, args.train_patch_grid, n_views=2)
    else:
        pooled = model(images, pool_only=True, n_views=2)[2][0]
    # Hand-built option namespaces may omit the parser's default.
    content_mask = model.sample_content_mask(hard=getattr(args, "mask_estimator", "relaxed") == "straight_through")
    global_loss = contrastive_loss(pooled, model, args, sim_metric, criterion, content_mask)
    foreground = None
    if weight > 0 and getattr(args, "patch_foreground_mask", False):
        foreground = masks if masks is not None else images != 0
    patch_loss = (
        patch_contrastive_loss(patches, model, args, sim_metric, criterion, foreground, content_mask)
        if weight > 0
        else global_loss.new_zeros(())
    )
    weighted = weight * patch_loss
    total = global_loss + weighted if weight > 0 else global_loss
    terms = {"global": global_loss, "patch": patch_loss, "patch_weighted": weighted}
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
    with open(os.path.join(save_dir, f"dci_step{step}.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    return flat, scores


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
            pooled, loss, terms = training_objective(model, x, args, sim_metric, criterion)
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
        print(
            f"objective: global InfoNCE + {args.patch_loss_weight:g} * patch InfoNCE; "
            f"grid={args.train_patch_grid}; {readout}; both losses update the backbones; {positions}; no target labels",
            flush=True,
        )

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
        writer.add_scalar("stratify/clusters_kept", stratification["clusters_kept"], 0)
        writer.add_scalar("stratify/subjects_per_epoch", stratification["subjects_per_epoch"], 0)
        for name, value in stratification["sd_ratio"].items():
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
    running = dict(
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
    )
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
                if args.patch_foreground_mask
                else None
            )
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
            running["mask_leak"] += last_loss_terms.get("mask_leak", 0.0)
            running["mask_size"] += last_loss_terms.get("mask_size", 0.0)
            running["rank"] += effective_rank(pooled[: pooled.shape[0] // 2, model.content_units()])
            running["n"] += 1
            step += 1

            if step % args.log_every == 0:
                n = max(running["n"], 1)
                mask_info = mask_state(model)
                print(
                    f"step {step:6d} | contrastive {running['loss']/n:.4f} "
                    f"| content eff_rank {running['rank']/n:.2f}/{len(model.content_units())}"
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
                if mask_info is not None and mask_info["content_units"] != logged_units:
                    print(f"  content units changed: {logged_units} -> {mask_info['content_units']}", flush=True)
                    logged_units = mask_info["content_units"]
                if writer is not None:
                    writer.add_scalar("train/contrastive", running["loss"] / n, step)
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
                    mask_leak=0.0,
                    mask_size=0.0,
                )

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
