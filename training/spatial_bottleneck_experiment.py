"""Fit a spatial bottleneck to frozen VQ-VAE maps from paired synthetic scans.

python -m training.spatial_bottleneck_experiment --run-dir RUN --out-dir NEW_OUTPUT

The source checkpoint is read-only. Training uses no anatomical labels. The CLI
uses synthetic train/val/test subject splits; extract_content_maps also accepts
any paired-volume DataLoader with the repository's image/mask batch contract.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.spatial_bottleneck import (
    ALIGNMENT_LOSSES,
    SpatialBottleneck,
    deviation_weights,
    infonce_alignment,
    paired_objective,
    position_weights,
    reconstruction_terms,
)
from utils.encoder_runtime import select_encoder_device

RECON_WEIGHTINGS = ("none", "position", "deviation")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def extract_content_maps(
    model, loader, device, level=0, max_bytes=2**30, expected_indices=None, return_foreground=False
):
    """Keep the native lattice, continuous pre-codebook features and fixed columns.

    Returned maps include VQ-VAE's content normalization when present. On-the-fly
    masks are rejected because even evaluation selection depends on the batch.
    No factor labels are read or returned. ``return_foreground`` appends a third
    return value: input-mask occupancy averaged onto the feature grid. Maps are
    never zeroed or otherwise modified by this option.
    """
    if getattr(model, "mask_mode", None) == "onthefly":
        raise ValueError("On-the-fly content masks are batch-dependent; use a fixed or learned-mask checkpoint")
    model.eval().requires_grad_(False)
    bank, foreground_bank, selected, offset = None, None, expected_indices, 0
    for batch in loader:
        images = batch["image"]
        if len(images) != 2 or images[0].shape != images[1].shape:
            raise ValueError("Need exactly two registered views with matching shapes")
        x = torch.cat(images, 0).to(device)
        masks = batch.get("mask")
        if getattr(model, "latent_mask", False) and masks is None:
            raise ValueError("This checkpoint requires input foreground masks")
        if return_foreground and masks is None:
            raise ValueError("Foreground reconstruction requires input foreground masks in every batch")
        mask = torch.cat(masks, 0).to(device) if masks is not None else None
        if return_foreground and (
            len(masks) != 2
            or mask.shape != (len(x), 1, *x.shape[-3:])
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
            or (mask > 1).any()
        ):
            raise ValueError("Need two finite input foreground masks shaped (B,1,D,H,W) in [0,1]")
        output = model(x, return_recon=False, pool_only=False, n_views=2, mask=mask)
        if not 0 <= level < len(output[2]):
            raise ValueError(f"Level {level} is not present in this checkpoint")
        features = output[2][level]
        selection = output[6].get(level)
        if selection is None:
            if getattr(model, "content_channels", None) is not None:
                raise ValueError(f"Level {level} has no explicit content partition; select a masked level")
            indices = [list(range(features.shape[1]))] * 2
        else:
            selections = selection if isinstance(selection, tuple) else (selection, selection)
            indices = [torch.where(s.reshape(-1).bool())[0].cpu().tolist() for s in selections]
        if not indices[0] or len(indices[0]) != len(indices[1]):
            raise ValueError("Both views must have the same nonzero number of content channels")
        if selected is not None and selected != indices:
            raise ValueError("Content channel identity changed between batches or splits")
        selected = indices
        count = len(images[0])
        pair = torch.stack((features[:count, indices[0]], features[count:, indices[1]]), 1).float().cpu()
        if not torch.isfinite(pair).all():
            raise ValueError("Non-finite frozen encoder maps")
        foreground = None
        if return_foreground:
            occupancy = torch.nn.functional.adaptive_avg_pool3d(mask.float(), pair.shape[-3:]).cpu()
            foreground = torch.stack((occupancy[:count], occupancy[count:]), 1)
            if (foreground.sum((2, 3, 4, 5)) <= 0).any():
                raise ValueError("Every subject and view needs nonempty foreground")
        if bank is None:
            required = len(loader.dataset) * pair[0].numel() * pair.element_size()
            if foreground is not None:
                required += len(loader.dataset) * foreground[0].numel() * foreground.element_size()
            if required > max_bytes:
                raise ValueError(
                    f"Map bank needs {required / 2**30:.2f} GiB; reduce sample counts or raise --max-map-gib"
                )
            bank = torch.empty((len(loader.dataset), *pair.shape[1:]), dtype=torch.float32)
            if foreground is not None:
                foreground_bank = torch.empty((len(loader.dataset), *foreground.shape[1:]), dtype=torch.float32)
        bank[offset : offset + count] = pair
        if foreground is not None:
            foreground_bank[offset : offset + count] = foreground
        offset += count
    if bank is None or offset != len(loader.dataset):
        raise ValueError("Extraction requires a complete, non-dropping loader")
    return (bank, selected, foreground_bank) if return_foreground else (bank, selected)


@torch.no_grad()
def evaluate(
    model,
    maps,
    batch_size,
    device,
    weigh=None,
    foreground=None,
    hard_patch_weight=0.0,
    hard_patch_fraction=0.1,
    hard_patch_size=(4, 4, 4),
    spatial_moment_weight=0.0,
    temperature=0.1,
):
    """Unweighted map metrics; ``weigh`` (from ``loss_weighting``) adds the training-weighted error.

    ``alignment_infonce`` scores all subjects at once (every other subject is a negative), so it
    does not depend on ``batch_size`` but sits above the training term by roughly log(N / batch).
    """
    model.eval()
    codes = []
    weighted, used = 0.0, False
    local = foreground is not None or hard_patch_weight > 0
    objective = dict(reconstruction_objective=0.0, reconstruction_mean=0.0, reconstruction_hard=0.0)
    foreground_error, foreground_baseline = 0.0, 0.0
    if foreground is not None and tuple(foreground.shape) != (*maps.shape[:2], 1, *maps.shape[-3:]):
        raise ValueError("Foreground bank must match the map subjects, views and spatial grid")
    errors = torch.zeros(maps.shape[2:], dtype=torch.float64)
    per_view = torch.zeros(2, dtype=torch.float64)
    raw_sum = 0.0
    baseline = 0.0
    map_gap = 0.0
    moment_error = 0.0
    for start in range(0, len(maps), batch_size):
        target = maps[start : start + batch_size].to(device)
        code, reconstruction = model(target.flatten(0, 1))
        if model.moment_head is not None:
            moment_error += torch.nn.functional.mse_loss(
                model.predict_moments(code), model.standardized_moments(target.flatten(0, 1))
            ).item() * len(target)
        residual = reconstruction.reshape_as(target) - target
        error = (residual / model.scale).square()
        weights = None if weigh is None else weigh(target)
        if weights is not None:
            weighted += (error * weights).mean().item() * len(target)
            used = True
        support = None if foreground is None else foreground[start : start + batch_size].to(device)
        if local:
            terms = reconstruction_terms(
                error, weights, support, hard_patch_weight, hard_patch_fraction, hard_patch_size
            )
            for name in objective:
                key = "reconstruction" if name == "reconstruction_objective" else name
                objective[name] += terms[key].item() * len(target)
        if support is not None:
            foreground_error += reconstruction_terms(error, foreground=support)["reconstruction"].item() * len(target)
            template_error = ((target - model.template) / model.scale).square()
            foreground_baseline += reconstruction_terms(template_error, foreground=support)[
                "reconstruction"
            ].item() * len(target)
        errors += error.sum((0, 1)).cpu().double()
        per_view += error.mean((2, 3, 4, 5)).sum(0).cpu().double()
        raw_sum += residual.square().mean().item() * len(target)
        baseline += ((target - model.template) / model.scale).square().mean().item() * len(target)
        map_gap += ((target[:, 0] - target[:, 1]) / model.scale).square().mean().item() * len(target)
        codes.append(code.reshape(len(target), 2, -1).cpu())
    code = torch.cat(codes)
    std = (code.var(0, unbiased=False) + 1e-4).sqrt()
    rec = per_view.mean().item() / len(maps)
    baseline /= len(maps)
    metrics = dict(
        reconstruction=rec,
        reconstruction_per_view=(per_view / len(maps)).tolist(),
        raw_map_mse=raw_sum / len(maps),
        template_normalized_mse=baseline,
        reconstruction_over_template=rec / baseline if baseline > 1e-12 else None,
        alignment=(code[:, 0] - code[:, 1]).square().mean().item(),
        mismatched_alignment=(code[:, 0] - code[:, 1].roll(1, 0)).square().mean().item(),
        alignment_infonce=infonce_alignment(code, temperature).item(),
        alignment_top1=cross_view_top1(code),
        variance=torch.relu(1 - std).mean().item(),
        code_std_per_view=code.std(0, unbiased=False).tolist(),
        original_map_alignment=map_gap / len(maps),
        spatial_moment=moment_error / len(maps) if model.moment_head is not None else 0.0,
    )
    if used:
        metrics["weighted_reconstruction"] = weighted / len(maps)
    if local:
        metrics.update({k: v / len(maps) for k, v in objective.items()})
    if foreground is not None:
        metrics.update(
            foreground_reconstruction=foreground_error / len(maps),
            foreground_template_normalized_mse=foreground_baseline / len(maps),
            foreground_reconstruction_over_template=(
                foreground_error / foreground_baseline if foreground_baseline > 1e-12 else None
            ),
            foreground_occupancy=foreground.mean().item(),
        )
    return metrics, code, (errors / (2 * len(maps))).mean(0).float()


def cross_view_top1(code):
    """Fraction of subjects whose other view is their nearest code by cosine, both directions averaged."""
    z = torch.nn.functional.normalize(code, dim=-1)
    similarity = z[:, 0] @ z[:, 1].T
    target = torch.arange(len(code))
    return (
        0.5 * ((similarity.argmax(1) == target).float().mean() + (similarity.argmax(0) == target).float().mean()).item()
    )


def alignment_options(args):
    # Argument sets written before --alignment-loss existed trained with MSE.
    return dict(
        alignment_loss=getattr(args, "alignment_loss", "mse"), temperature=getattr(args, "infonce_temperature", 0.1)
    )


def objective_score(metrics, args):
    """The training objective on held-out maps, with the reconstruction weighted as in training."""
    infonce = alignment_options(args)["alignment_loss"] == "infonce"
    terms = dict(
        metrics,
        reconstruction=metrics.get(
            "reconstruction_objective", metrics.get("weighted_reconstruction", metrics["reconstruction"])
        ),
        alignment=metrics["alignment_infonce" if infonce else "alignment"],
    )
    score = sum(getattr(args, name + "_weight") * terms[name] for name in ("reconstruction", "alignment", "variance"))
    return score + getattr(args, "spatial_moment_weight", 0.0) * terms["spatial_moment"]


def loss_weighting(model, training, args):
    """Reconstruction weights for a batch of paired maps (None = plain mean), and a record of them.

    Fitted on training maps only, after the model's normalisation; never stored in the model.
    """
    mode = getattr(args, "recon_weighting", "none")
    if mode == "none":
        return (lambda maps: None), dict(mode=mode)
    if mode == "position":
        weights, floored = position_weights(model, training, args.position_floor)
        record = dict(
            mode=mode,
            floor=args.position_floor,
            floored_fraction=floored,
            min_weight=weights.min().item(),
            max_weight=weights.max().item(),
        )
        return (lambda maps: weights.to(maps.device)), record
    return (lambda maps: deviation_weights(model, maps, args.deviation_strength)), dict(
        mode=mode, strength=args.deviation_strength
    )


def reconstruction_options(args):
    return {
        name: getattr(args, name, default)
        for name, default in (("hard_patch_weight", 0.0), ("hard_patch_fraction", 0.1), ("hard_patch_size", (4, 4, 4)))
    }


def fit_readout(training, validation, args, device, training_foreground=None, validation_foreground=None):
    """Validation selects checkpoints but never updates parameters or normalization."""
    if training.shape[1:] != validation.shape[1:]:
        raise ValueError("Training and validation maps must have the same shape")
    if getattr(args, "recon_foreground_only", False) and (training_foreground is None or validation_foreground is None):
        raise ValueError("Foreground reconstruction requires training and validation foreground banks")
    for maps, foreground in ((training, training_foreground), (validation, validation_foreground)):
        if foreground is not None and tuple(foreground.shape) != (*maps.shape[:2], 1, *maps.shape[-3:]):
            raise ValueError("Foreground bank must match the map subjects, views and spatial grid")
    torch.manual_seed(args.seed)
    moment_channels = training.shape[2] if args.spatial_moment_weight > 0 else None
    model = SpatialBottleneck(tuple(training.shape[2:]), args.latent_dim, args.width, args.hidden, moment_channels)
    model.fit_normalization(training)
    weigh, weighting = loss_weighting(model, training, args)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(args.seed + 1)
    options = reconstruction_options(args)
    alignment = alignment_options(args)
    temperature = alignment["temperature"]
    initial, _, _ = evaluate(
        model, validation, args.batch_size, device, weigh, validation_foreground, temperature=temperature, **options
    )
    best_score, best_step = objective_score(initial, args), 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randperm(len(training), generator=rng)[: args.batch_size]
        batch = training[indices].to(device)
        support = None if training_foreground is None else training_foreground[indices].to(device)
        loss, terms = paired_objective(
            model,
            batch,
            args.reconstruction_weight,
            args.alignment_weight,
            args.variance_weight,
            weigh(batch),
            foreground=support,
            spatial_moment_weight=args.spatial_moment_weight,
            **alignment,
            **options,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite bottleneck loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            metrics, _, _ = evaluate(
                model,
                validation,
                args.batch_size,
                device,
                weigh,
                validation_foreground,
                spatial_moment_weight=args.spatial_moment_weight,
                temperature=temperature,
                **options,
            )
            score = objective_score(metrics, args)
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite validation loss at step {step}")
            if score < best_score:
                best_score, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(dict(step=step, train={k: v.item() for k, v in terms.items()}, validation=metrics))
            print(
                f"step {step}/{args.steps}: validation loss={score:.5g}, map MSE={metrics['reconstruction']:.5g}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(
        initial_validation=initial,
        best_step=best_step,
        best_validation_score=best_score,
        history=history,
        recon_weighting=weighting,
        alignment_objective=dict(
            **alignment,
            formula=(
                "symmetric cross-view InfoNCE on cosine similarity / temperature, other subjects' other views "
                "as negatives within each training batch"
                if alignment["alignment_loss"] == "infonce"
                else "mean squared difference between the two views' codes"
            ),
        ),
        reconstruction_objective=dict(
            **options,
            foreground_only=training_foreground is not None,
            formula="mean error + hard_patch_weight * mean of top ceil(fraction * nonempty patches), per subject/view",
            mask="input foreground occupancy averaged onto feature grid; loss only",
        ),
    )


def load_bottleneck(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = SpatialBottleneck(**saved["model_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="vqvae_model.pt")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument(
        "--alignment-loss",
        choices=ALIGNMENT_LOSSES,
        default="infonce",
        help="infonce: symmetric cross-view InfoNCE on cosine similarity, other subjects in the batch as "
        "negatives; mse: squared difference between the two views' codes (the original objective)",
    )
    parser.add_argument("--infonce-temperature", type=float, default=0.1, help="InfoNCE temperature")
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument(
        "--spatial-moment-weight",
        "--spatial-moment-loss-weight",
        type=float,
        default=0.0,
        help="Add a linear-head MSE for standardized first spatial moments of the input maps (0 disables)",
    )
    parser.add_argument(
        "--hard-patch-weight",
        type=float,
        default=0.0,
        help="Add this times the mean error of the hardest patches to mean reconstruction (0 disables)",
    )
    parser.add_argument(
        "--hard-patch-fraction",
        type=float,
        default=0.1,
        help="Fraction of nonempty patches selected separately per subject/view, rounded up (default: 0.1)",
    )
    parser.add_argument(
        "--hard-patch-size",
        type=int,
        nargs=3,
        default=(4, 4, 4),
        metavar=("D", "H", "W"),
        help="Nonoverlapping patch size in feature-map cells; partial edge patches are retained",
    )
    parser.add_argument(
        "--recon-foreground-only",
        action="store_true",
        help="Exclude background from reconstruction and hard-patch selection using input-mask occupancy; loss only",
    )
    parser.add_argument(
        "--recon-weighting",
        choices=RECON_WEIGHTINGS,
        default="none",
        help="Re-weight the training reconstruction error: none (plain mean), position (each map entry in units "
        "of its own spread across training subjects) or deviation (more weight where a map departs from the "
        "template). Checkpoints are selected on the same weighted objective; reported metrics stay unweighted.",
    )
    parser.add_argument(
        "--position-floor",
        type=float,
        default=0.25,
        help="position: floor on an entry's spread, as a fraction of its channel's scale (caps a weight at 1/floor^2)",
    )
    parser.add_argument(
        "--deviation-strength",
        type=float,
        default=1.0,
        help="deviation: a position's weight is 1 + strength * its RMS standardised departure from the template",
    )
    parser.add_argument("--max-map-gib", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        min(args.latent_dim, args.width, args.hidden, args.steps, args.log_every, args.cpu_threads) < 1
        or args.level < 0
    ):
        parser.error("Dimensions, steps and thread counts must be positive; level must be nonnegative")
    if min(args.train_samples, args.val_samples, args.test_samples, args.batch_size) < 2:
        parser.error("At least two subjects are required in every split and training batch")
    if args.batch_size > args.train_samples:
        parser.error("batch-size cannot exceed train-samples")
    for name in ("lr", "reconstruction_weight", "max_map_gib", "infonce_temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    for name in (
        "alignment_weight",
        "variance_weight",
        "deviation_strength",
        "hard_patch_weight",
        "spatial_moment_weight",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative and finite")
    if not 0 < args.position_floor <= 1:
        parser.error("position-floor must lie in (0, 1]")
    if not 0 < args.hard_patch_fraction <= 1:
        parser.error("hard-patch-fraction must lie in (0, 1]")
    if min(args.hard_patch_size) < 1:
        parser.error("hard-patch-size must contain positive integers")
    return args


def build_split_dataset(saved_args, split, count):
    """The synthetic split this CLI uses; the factor evaluator rebuilds splits through it too."""
    from data.datasets import SyntheticBrainDataset

    # Forward all saved generator switches, including field/lesion options that
    # older evaluation helpers do not yet expose. Keep the original SCM seed.
    generator_args = {k: v for k, v in vars(saved_args).items() if k.startswith("synthetic_")}
    generator_args.pop("synthetic_num_samples", None)
    generator_args.pop("synthetic_num_samples_per_mode", None)
    resolution = getattr(saved_args, "synthetic_res", 64)
    spatial_size = getattr(saved_args, "spatial_size", None) or (resolution,) * 3
    return SyntheticBrainDataset(
        mode=split, cache=False, spatial_size=spatial_size, synthetic_num_samples=count, **generator_args
    )


def load_source(args, device):
    from eval.protocol.run_dci_synthetic import load_model_from_run_dir, load_run_args

    saved_args = load_run_args(args.run_dir)
    if str(getattr(saved_args, "dataset_name", "")).lower() != "synthetic":
        raise ValueError("This draft CLI supports synthetic checkpoints; use extract_content_maps with an ADNI loader")
    if getattr(saved_args, "split_encoder_norm", False):
        raise ValueError(
            "The shared checkpoint loader does not restore split_encoder_norm; refusing a mismatched model"
        )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = args.run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    model, saved_args, _ = load_model_from_run_dir(str(args.run_dir), str(checkpoint), device, seed=args.seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("encoders", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Loss-only projectors do not participate in feature extraction. Every other
    # weight/buffer must match, instead of silently retaining initialized weights.
    state = {k: v for k, v in state.items() if not k.startswith("_contrastive_proj_heads.")}
    model.load_state_dict(state, strict=True)
    return model, saved_args, checkpoint


def main(argv=None):
    args = parse_args(argv)
    if args.out_dir.resolve() == args.run_dir.resolve() or args.out_dir.exists():
        raise ValueError("out-dir must be a new directory distinct from the source run")
    torch.set_num_threads(args.cpu_threads)
    device = select_encoder_device(args.device)
    model, saved_args, checkpoint = load_source(args, device)
    source = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_digest(checkpoint),
        settings_sha256=file_digest(args.run_dir / "settings.json"),
        level=args.level,
        representation="continuous encoder maps, after content normalization, before codebook projection",
    )
    banks, foregrounds, splits, indices = {}, {}, {}, None
    remaining = int(args.max_map_gib * 2**30)
    for split, count in (("train", args.train_samples), ("val", args.val_samples), ("test", args.test_samples)):
        dataset = build_split_dataset(saved_args, split, count)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        extracted = extract_content_maps(
            model, loader, device, args.level, remaining, indices, return_foreground=args.recon_foreground_only
        )
        banks[split], indices = extracted[:2]
        if args.recon_foreground_only:
            foregrounds[split] = extracted[2]
            remaining -= foregrounds[split].numel() * foregrounds[split].element_size()
        remaining -= banks[split].numel() * banks[split].element_size()
        splits[split] = dict(subject_ids=[f"{split}:{i}" for i in range(count)], sample_seed=int(dataset._inner.seed))
        print(f"Extracted {split}: {tuple(banks[split].shape)}", flush=True)
    source["content_indices_per_view"] = indices
    del model
    head, training = fit_readout(
        banks["train"], banks["val"], args, device, foregrounds.get("train"), foregrounds.get("val")
    )
    # Preserve whole-map metrics used by existing factor/intervention parity
    # checks, and add the new loss and foreground metrics under separate names.
    options = dict(reconstruction_options(args), temperature=alignment_options(args)["temperature"])
    weigh = None
    if args.recon_foreground_only or args.hard_patch_weight > 0:
        weigh, _ = loss_weighting(head, banks["train"], args)
    validation, _, _ = evaluate(head, banks["val"], args.batch_size, device, weigh, foregrounds.get("val"), **options)
    testing, codes, error_map = evaluate(
        head, banks["test"], args.batch_size, device, weigh, foregrounds.get("test"), **options
    )
    if file_digest(checkpoint) != source["checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint changed during this experiment")
    report = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source=source,
        splits=splits,
        training=training,
        validation=validation,
        test=testing,
        parameters=sum(p.numel() for p in head.parameters()),
        map_shape=list(head.map_shape),
        latent_dim=head.latent_dim,
        label_access="none",
        selection="weighted validation objective; test used only after selection",
        scope="Feature compression and cross-view agreement; not causal-factor identification",
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(
        dict(model_config=head.config, state_dict={k: v.cpu() for k, v in head.state_dict().items()}, source=source),
        args.out_dir / "bottleneck.pt",
    )
    torch.save(
        dict(
            codes=codes,
            gap=banks["test"].mean((3, 4, 5)),
            normalized_error_map=error_map,
            **({"foreground_occupancy": foregrounds["test"]} if args.recon_foreground_only else {}),
            subject_ids=splits["test"]["subject_ids"],
        ),
        args.out_dir / "test_outputs.pt",
    )
    print(f"Saved bottleneck and held-out report to {args.out_dir}", flush=True)
    return report


if __name__ == "__main__":
    main()
