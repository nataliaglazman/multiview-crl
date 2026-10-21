"""Does batch standardization make residual anatomy useful to the global InfoNCE?

Replays the trainer's batching at frozen checkpoints, comparing raw, centered, and
standardized projected content codes. Removes marginal and partial linear factor
components from each batch and measures changes to the actual InfoNCE. These are
associations in code space, not image interventions or estimates of retraining effects.

python -m eval.encoder.code_standardization_probe --run-dir RUN \
    --checkpoints model_init.pt model.pt --batches 8 --device cpu
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import data.cluster_batches as cluster_batches
import training.encoder_args as encoder_args
import training.losses as losses
from eval.metrics.dci import content_factor_names
from eval.protocol.score_checkpoint import build_model, load_settings
from training import main_conv_synthetic as trainer
from utils.encoder_runtime import configure_encoder_runtime, select_encoder_device

MODES = ("raw", "centered", "standardized")
VIEWS = ("t1", "flair")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", default=["model.pt"])
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--factors", nargs="+", help="Default: all active content factors, including lesion controls")
    p.add_argument("--shuffles", type=int, default=8, help="Descriptive within-batch shuffled-factor controls")
    p.add_argument("--seed", type=int, default=1729, help="Null-control seed; batching uses the run's loader seed")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--model-mode", choices=("train", "eval"), default="train")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.batches < 1 or args.shuffles < 1 or args.seed < 0:
        p.error("batches and shuffles must be positive, seed nonnegative")
    if len(set(args.checkpoints)) != len(args.checkpoints):
        p.error("Duplicate checkpoints")
    return args


def training_options(cfg):
    known = {action.dest for action in encoder_args.build_parser()._actions if action.option_strings}
    options = encoder_args.parse_args(encoder_args.cli_arguments({k: v for k, v in cfg.items() if k in known}))
    if options.dataset_name != "synthetic" or options.synthetic_mode != "pseudo_mri":
        raise ValueError("This factor probe requires synthetic pseudo_mri data")
    if options.contrastive_loss_type != "infonce" or options.mask_mode != "fixed":
        raise ValueError("This probe requires global InfoNCE with a fixed content mask")
    if cfg.get("pairing", "cross_modal") != "cross_modal":
        raise ValueError("This probe requires cross_modal pairs; intensity-augmented pairs are not replayed")
    if options.batch_size < 3:
        raise ValueError("Need at least three subjects per batch")
    options.cache = False
    return options, sorted(set(cfg) - known)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "probe_index": index}


def training_loader(options, dataset, run_dir):
    """Replay recorded clusters or neighbour pools without refitting or loading a stage-0 model."""
    labels = None
    generator = torch.Generator().manual_seed(options.loader_seed)
    if options.stratify_batches != "none":
        path = Path(run_dir) / "stratification.json"
        if not path.is_file():
            raise ValueError(f"Missing {path}; need the actual training clusters or neighbour pools")
        saved = json.loads(path.read_text())
        kind = getattr(options, "stratify_sampler", "cluster")
        if (
            saved["mode"] != options.stratify_batches
            or saved.get("sampler", "cluster") != kind
            or saved.get("batch_size", options.batch_size) != options.batch_size
            or saved.get("sampler_seed", options.loader_seed) != options.loader_seed
        ):
            raise ValueError("stratification.json disagrees with settings.json")
        if kind == "neighbourhood":
            neighbours = np.asarray(saved["neighbour_indices"])
            balance = getattr(options, "stratify_balance", 0.0)
            if saved["neighbours"] != options.stratify_neighbours or saved.get("balance", 0.0) != balance:
                raise ValueError("stratification.json disagrees with settings.json")
            if neighbours.shape != (len(dataset), options.stratify_neighbours):
                raise ValueError("Recorded neighbour pools have the wrong subject count or pool size")
            sampler = cluster_batches.NeighbourhoodBatchSampler(
                neighbours, options.batch_size, options.loader_seed, balance
            )
        else:
            labels = np.asarray(saved["labels"])
            if labels.shape != (len(dataset),) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("Recorded cluster labels must be one integer per training subject")
            if saved["k"] != options.stratify_clusters:
                raise ValueError("stratification.json disagrees with settings.json")
            sampler = cluster_batches.ClusterBatchSampler(labels, options.batch_size, options.loader_seed)
        batching = {"batch_sampler": sampler}
    else:
        batching = {"batch_size": options.batch_size, "shuffle": True, "drop_last": True}
    loader = DataLoader(IndexedDataset(dataset), num_workers=0, generator=generator, **batching)
    if len(loader) == 0:
        raise ValueError("No full training batches")
    return loader, labels


def batches(loader, count):
    drawn = 0
    while drawn < count:
        for batch in loader:
            yield batch
            drawn += 1
            if drawn == count:
                break


def transform(code, mode, reference=None):
    if mode == "raw":
        return code
    ref = code if reference is None else reference
    centered = code - ref.mean(1, keepdim=True)
    if mode == "centered":
        return centered
    if mode == "standardized":
        if reference is None:
            return trainer.standardize_contrastive_code(code)
        return centered / (ref.std(1, keepdim=True) + 1e-6)
    raise ValueError(mode)


def code_loss(code, options):
    return losses.infonce_loss(
        code,
        sim_metric=torch.nn.CosineSimilarity(dim=-1),
        criterion=torch.nn.CrossEntropyLoss(),
        tau=options.tau,
        estimated_content_indices=[list(range(code.shape[-1]))],
        subsets=[(0, 1)],
        cross_view_negs_only=options.cross_view_negs_only,
    ).squeeze()


def geometry(code):
    unit = F.normalize(code, dim=-1, eps=1e-8)
    sim = unit[0] @ unit[1].T
    eye = torch.eye(sim.shape[0], dtype=torch.bool)
    truth = torch.arange(len(sim))
    return {
        "cross_view_top1": float(
            ((sim.argmax(1) == truth).double().mean() + (sim.argmax(0) == truth).double().mean()) / 2
        ),
        "positive_cosine": float(sim.diag().mean()),
        "negative_cosine": float(sim[~eye].mean()),
        "cosine_margin": float(sim.diag().mean() - sim[~eye].mean()),
    }


def factor_direction(targets, index, association, permutation=None):
    """Unit subject-space direction, optionally after removing all other active factors.

    SVD handles collinear controls. A negligible unique component is reported as
    unavailable, never inflated to a unit direction. Nulls shuffle the target residual
    and re-project against the same controls; these are descriptive, not causal tests.
    """
    y = targets[:, index] - targets[:, index].mean()
    norm = y.norm()
    if norm <= 1e-12:
        return None, 0.0
    basis = None
    if association == "unique":
        others = torch.cat((targets[:, :index], targets[:, index + 1 :]), dim=1)
        others = others - others.mean(0)
        scale = others.norm(dim=0)
        others = others[:, scale > 1e-12] / scale[scale > 1e-12]
        if others.shape[1]:
            u, s, _ = torch.linalg.svd(others, full_matrices=False)
            basis = u[:, s > s.max() * 1e-8]
            if len(y) - 1 - basis.shape[1] < 3:
                return None, None
            y = y - basis @ (basis.T @ y)
    share = float(y.square().sum() / norm.square())
    if share < 1e-10:
        return None, share
    if permutation is not None:
        y = y[permutation]
        if basis is not None:
            y = y - basis @ (basis.T @ y)
    if y.norm() <= 1e-10 * norm:
        return None, share
    return y / y.norm(), share


def component(code, direction):
    return torch.einsum("b,vbc->vc", direction, code)[:, None, :] * direction[None, :, None]


def removal_metrics(raw, part, mode, options, base_loss, gradient):
    removed = raw - part
    return {
        "delta_loss_recomputed": float(code_loss(transform(removed, mode), options) - base_loss),
        "delta_loss_fixed_stats": float(code_loss(transform(removed, mode, reference=raw), options) - base_loss),
        "first_order_removal": float(-(gradient * part).sum()),
    }


def analyze_batch(raw, targets, names, chosen, options, shuffles=8, seed=1729):
    """All counterfactuals use the same frozen, pre-standardization code on CPU/float64."""
    raw = raw.detach().cpu().double()
    targets = targets.detach().cpu().double()
    if not torch.isfinite(raw).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite codes or targets")
    rng = torch.Generator().manual_seed(seed)
    permutations = [torch.randperm(raw.shape[1], generator=rng) for _ in range(shuffles)]
    baselines, rows, channels = [], [], []
    for view, view_name in enumerate(VIEWS):
        sd = raw[view].std(0)
        for channel in range(raw.shape[-1]):
            channels.append(
                {
                    "view": view_name,
                    "channel": channel,
                    "mean": float(raw[view, :, channel].mean()),
                    "sd": float(sd[channel]),
                    "scale_multiplier": float(1 / (sd[channel] + 1e-6)),
                }
            )
    for mode in MODES:
        differentiable = raw.clone().requires_grad_(True)
        loss = code_loss(transform(differentiable, mode), options)
        gradient = torch.autograd.grad(loss, differentiable)[0].detach()
        base_loss = loss.detach()
        transformed = transform(raw, mode)
        centered = transformed - transformed.mean(1, keepdim=True)
        baselines.append(
            {
                "mode": mode,
                "loss": float(base_loss),
                **geometry(transformed),
                "gradient_norm_pre_standardization": float(gradient.norm()),
            }
        )
        for factor in chosen:
            index = names.index(factor)
            for association in ("marginal", "unique"):
                direction, share = factor_direction(targets, index, association)
                row = {
                    "factor": factor,
                    "association": association,
                    "mode": mode,
                    "target_residual_variance_share": share,
                    "available": direction is not None,
                }
                if direction is None:
                    rows.append(row)
                    continue
                part = component(raw, direction)
                row.update(removal_metrics(raw, part, mode, options, base_loss, gradient))
                # Apply the baseline affine scaling to the removed component, without centering it again.
                scaled = part / (raw.std(1, keepdim=True) + 1e-6) if mode == "standardized" else part
                for v, view in enumerate(VIEWS):
                    energy = centered[v].square().sum()
                    row[f"signal_rms_{view}"] = float(scaled[v].square().mean().sqrt())
                    row[f"variance_share_{view}"] = float(scaled[v].square().sum() / energy) if energy > 1e-24 else None
                nulls = []
                for permutation in permutations:
                    null, _ = factor_direction(targets, index, association, permutation)
                    if null is not None:
                        nulls.append(removal_metrics(raw, component(raw, null), mode, options, base_loss, gradient))
                row["null_count"] = len(nulls)
                for metric in ("delta_loss_recomputed", "delta_loss_fixed_stats", "first_order_removal"):
                    mean = float(np.mean([n[metric] for n in nulls])) if nulls else None
                    row[f"null_{metric}"] = mean
                    row[f"excess_{metric}"] = row[metric] - mean if mean is not None else None
                rows.append(row)
    return baselines, rows, channels


def extract_code(model, images, options, mode):
    """Training-mode forward with all buffers, module modes, and parameter gradients preserved."""
    states = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    try:
        model.train(mode == "train")
        with torch.no_grad():
            pooled = model(images, pool_only=True, n_views=2)[2][0]
            raw = trainer.contrastive_code(pooled, model, options)
            # Check all three transforms against the real trainer, including the actual projector.
            parity = {}
            for transform_mode in MODES:
                args = SimpleNamespace(
                    **{
                        **vars(options),
                        "standardize_code": transform_mode == "standardized",
                        "center_code": transform_mode == "centered",
                    }
                )
                expected = trainer.contrastive_loss(
                    pooled, model, args, torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
                )
                actual = code_loss(transform(raw, transform_mode), args)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                parity[transform_mode] = float((actual - expected).abs())
            return raw.detach().cpu(), parity
    finally:
        with torch.no_grad():
            for buffer, before in buffers:
                buffer.copy_(before)
        for module, training in states:
            module.training = training


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, values in groups.items():
        entry = {**dict(zip(keys, group)), "batches": len(values)}
        metrics = set().union(*(v.keys() for v in values)) - set(keys) - {"batch", "cluster"}
        for metric in sorted(metrics):
            numbers = [
                float(v[metric]) for v in values if isinstance(v.get(metric), (float, int)) and np.isfinite(v[metric])
            ]
            if numbers:
                entry[metric] = {"mean": float(np.mean(numbers)), "std": float(np.std(numbers)), "count": len(numbers)}
        result.append(entry)
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run(args):
    cfg = load_settings(args.run_dir)
    options, unknown = training_options(cfg)
    out = args.out_dir or args.run_dir / "standardization_probe"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise ValueError(f"{out} is nonempty; choose another --out-dir or pass --overwrite")
    checkpoint_paths = [args.run_dir / name for name in args.checkpoints]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    names = content_factor_names(options.n_content, options.synthetic_lesion_target)
    active = [i for i, name in enumerate(names) if not name.startswith("unused_")]
    active_names = [names[i] for i in active]
    chosen = args.factors or active_names
    if len(set(chosen)) != len(chosen) or set(chosen) - set(active_names):
        raise ValueError(f"--factors must be distinct active factors from {active_names}")
    configure_encoder_runtime(cfg)
    device = select_encoder_device(args.device)
    dataset = trainer.make_dataset(options, "train", options.num_train_samples)
    bank = cluster_batches.bank_latents(dataset)[:, active]
    bank_sd = bank.std(0, ddof=1)
    baseline_rows, factor_rows, channel_rows, batch_rows, code_arrays = [], [], [], [], {}
    reference_batches = None
    parity_max = 0.0
    for checkpoint, path in zip(args.checkpoints, checkpoint_paths):
        model = build_model(cfg, device, torch.load(path, map_location="cpu", weights_only=True))
        loader, labels = training_loader(options, dataset, args.run_dir)
        this_batches, codes, factors = [], [], []
        print(
            f"{checkpoint}: {args.batches} batches of {options.batch_size}, model {args.model_mode}, device {device}",
            flush=True,
        )
        for number, batch in enumerate(batches(loader, args.batches)):
            ids = batch["probe_index"].tolist()
            images = torch.cat(batch["image"], dim=0)
            image_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
            this_batches.append((ids, image_hash))
            raw, parity = extract_code(model, images.to(device), options, args.model_mode)
            parity_max = max(parity_max, *parity.values())
            targets = batch["gt_latents"]["z_content"][:, active]
            np.testing.assert_allclose(targets.numpy(), bank[ids], rtol=1e-5, atol=1e-6)
            base, measured, channels = analyze_batch(
                raw, targets, active_names, chosen, options, args.shuffles, args.seed + number
            )
            common = {
                "checkpoint": checkpoint,
                "batch": number,
                "cluster": int(labels[ids[0]]) if labels is not None else None,
            }
            baseline_rows.extend({**common, **row} for row in base)
            factor_rows.extend({**common, **row} for row in measured)
            channel_rows.extend({**common, **row} for row in channels)
            for j, name in enumerate(active_names):
                sd = float(targets[:, j].std())
                batch_rows.append(
                    {
                        **common,
                        "factor": name,
                        "batch_sd": sd,
                        "bank_sd": float(bank_sd[j]),
                        "sd_ratio": sd / float(bank_sd[j]) if bank_sd[j] > 0 else None,
                    }
                )
            codes.append(raw.numpy())
            factors.append(targets.numpy())
            print(
                f"  batch {number + 1}/{args.batches}: " + ", ".join(f"{r['mode']} loss {r['loss']:.4f}" for r in base),
                flush=True,
            )
        if reference_batches is not None and reference_batches != this_batches:
            raise RuntimeError("Checkpoint comparisons did not receive identical subject indices and images")
        reference_batches = this_batches
        code_arrays[f"checkpoint_{len(code_arrays)}"] = np.stack(codes)
        del model
    summaries = aggregate(factor_rows, ("checkpoint", "factor", "association", "mode"))
    # Paired scale effect: subtract centering, which already removes the batch mean.
    lookup = {(r["checkpoint"], r["batch"], r["factor"], r["association"], r["mode"]): r for r in factor_rows}
    effects = []
    for row in factor_rows:
        if row["mode"] != "standardized" or not row["available"]:
            continue
        key = (row["checkpoint"], row["batch"], row["factor"], row["association"], "centered")
        centered = lookup[key]
        effects.append(
            {
                **{k: row[k] for k in ("checkpoint", "batch", "factor", "association")},
                **{
                    metric: row[metric] - centered[metric]
                    for metric in (
                        "excess_delta_loss_recomputed",
                        "excess_delta_loss_fixed_stats",
                        "excess_first_order_removal",
                    )
                    if row.get(metric) is not None and centered.get(metric) is not None
                },
            }
        )
    report = {
        "settings": cfg,
        "unrecognized_settings": unknown,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "code_array_checkpoints": list(args.checkpoints),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): digest(path)
            for path in (
                Path(__file__).resolve(),
                Path(trainer.__file__).resolve(),
                Path(losses.__file__).resolve(),
                Path(cluster_batches.__file__).resolve(),
            )
        },
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_sha256": {name: digest(path) for name, path in zip(args.checkpoints, checkpoint_paths)},
        "settings_sha256": digest(args.run_dir / "settings.json"),
        "stratification_sha256": (
            digest(args.run_dir / "stratification.json") if options.stratify_batches != "none" else None
        ),
        "factor_names": active_names,
        "parity_max_absolute_error": parity_max,
        "batches": [{"indices": ids, "image_sha256": sha} for ids, sha in reference_batches],
        "baseline_summary": aggregate(baseline_rows, ("checkpoint", "mode")),
        "factor_summary": summaries,
        "scaling_effect_summary": aggregate(effects, ("checkpoint", "factor", "association")),
        "scope": "Frozen-checkpoint global InfoNCE, using the start of the training sampler schedule. "
        "Marginal/partial linear associations and shuffled controls, not causal factor interventions. "
        "No claim about retraining or other loss terms. Positive removal delta means the loss uses the component. "
        "Positive standardized-minus-centered excess delta supports increased reliance due to scaling. "
        "Batch SDs are not independent-sample confidence intervals.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for filename, rows in (
        ("baselines.csv", baseline_rows),
        ("factors.csv", factor_rows),
        ("channels.csv", channel_rows),
        ("batch_spread.csv", batch_rows),
        ("scaling_effect.csv", effects),
    ):
        write_csv(out / filename, rows)
    np.savez_compressed(
        out / "codes.npz", **code_arrays, targets=np.stack(factors), indices=np.array([b[0] for b in reference_batches])
    )
    print("\nScaling effect on factor reliance (standardized minus centered; shuffled control subtracted)")
    for entry in report["scaling_effect_summary"]:
        if entry["association"] == "unique":
            value = entry.get("excess_delta_loss_recomputed", {}).get("mean")
            if value is not None:
                print(f"  {entry['checkpoint']:20s} {entry['factor']:22s} {value:+.5f}")
    print(f"Saved {out / 'report.json'}; trainer loss parity max error {parity_max:.3g}")
    return report


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
