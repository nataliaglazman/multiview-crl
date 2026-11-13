"""Continue a frozen warm.py run's shared head, with optional cross-view feature prediction.

The only trainable image path is the source's head. Its exported content units predict the other view's
frozen, unaugmented global average (GAP), through linear training-only decoders. Targets are standardized
and ridge-whitened using training subjects only. Inputs reuse a fixed bank of independent jitter draws;
all arms share that cache, the source head, and the subject/draw sequence. Labels only score the code.

    python frozen_prediction.py SOURCE_RUN OUT STEPS EVERY --cache FEATURES.pt --prediction-weight 1
    python frozen_prediction.py SOURCE_RUN UNUSED STEPS EVERY --cache FEATURES.pt --prepare-only

Checkpoints remain readable by code_audit.py, own_variation.py and precision_audit.py. The latter's price
table prices InfoNCE alone, not the combined loss. Its untrained twin is a fully random network; include
SOURCE_RUN in reports to show this experiment's actual starting representation.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import code_audit
import numpy as np
import torch
import torch.nn.functional as F
import warm
from torch import nn

CACHE_VERSION = 1


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(module, exclude_head=False):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if exclude_head and name.startswith("head."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_features(model, pool):
    model.eval().requires_grad_(False)
    pool.eval().requires_grad_(False)
    pool.head.train().requires_grad_(True)


def fit_whitening(train, ridge):
    """Per-view standardized, regularized whitening; all 64 target coordinates are retained."""
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Whitening ridge must be positive and finite")
    x = train.detach().cpu().double()
    if x.ndim != 3 or x.shape[0] != 2 or x.shape[1] < 2 or not torch.isfinite(x).all():
        raise ValueError("Whitening needs two finite views with at least two training subjects")
    mean = x.mean(1, keepdim=True)
    scale = x.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = (x - mean) / scale
    covariance = z.transpose(1, 2) @ z / z.shape[1]
    values, vectors = torch.linalg.eigh(covariance)
    matrix = (vectors * (values.clamp_min(0) + ridge).rsqrt().unsqueeze(1)) @ vectors.transpose(1, 2)
    return {"mean": mean.float(), "scale": scale.float(), "matrix": matrix.float()}


def whiten(features, stats):
    return ((features - stats["mean"]) / stats["scale"]) @ stats["matrix"]


def fit_code_scale(train_code):
    code = train_code.detach().cpu()
    return {"mean": code.mean(1, keepdim=True), "scale": code.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)}


def prediction_loss(code, targets, decoders, stats):
    """Decoder 0 reads T1 code to predict FLAIR; decoder 1 reads FLAIR code to predict T1."""
    units = stats["mean"].shape[-1]
    normalized = (code[..., :units] - stats["mean"]) / stats["scale"]
    targets = targets.detach()
    losses = [F.mse_loss(decoders[v](normalized[v]), targets[1 - v]) for v in range(2)]
    return (losses[0] + losses[1]) / 2, losses


def draw_batch(rng, subjects, batch, draws):
    return rng.choice(subjects, batch, replace=False), rng.integers(draws, size=(2, batch))


def train_step(pool, decoders, optimizer, pooled, targets, code_stats, weight, tau=0.1, clip=2.0):
    code = torch.stack([pool.head(p) for p in pooled])
    global_loss = warm.hard_negative_infonce(torch.cat(list(code)), pool.units, tau, 0.0, nn.CosineSimilarity(dim=-1))
    auxiliary, directions = prediction_loss(code, targets, decoders, code_stats)
    total = global_loss if weight == 0 else global_loss + weight * auxiliary
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    if not torch.isfinite(total) or not torch.isfinite(norm):
        raise ValueError("Nonfinite loss or gradients")
    optimizer.step()
    return {
        "global": float(global_loss.detach()),
        "prediction": float(auxiliary.detach()),
        "prediction_weighted": float((weight * auxiliary).detach()),
        "t1_to_flair": float(directions[0].detach()),
        "flair_to_t1": float(directions[1].detach()),
        "total": float(total.detach()),
        "gradient_norm": float(norm),
    }


def cache_metadata(source, source_args, draws, seed, batch, test_subjects):
    root = Path(source_args["bank"])
    files = {}
    for split in ("train", "test"):
        names = [source_args["t1key"], "flair", "vox", "zc"]
        if source_args.get("residual") == "True":
            names += [source_args["t1key"] + "_res", "flair_res"]
        for name in names:
            path = root / f"{split}_{name}.npy"
            stat = path.stat()
            files[str(path.resolve())] = [stat.st_size, stat.st_mtime_ns]
    return {
        "version": CACHE_VERSION,
        "source": str(source.resolve()),
        "checkpoint_sha256": file_digest(source / "model.pt"),
        "report_sha256": file_digest(source / "report.json"),
        "draws": draws,
        "seed": seed,
        "feature_batch": batch,
        "test_subjects": test_subjects,
        "files": files,
    }


def read_cache(path, expected):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache.get("metadata") != expected:
        raise ValueError(f"{path}: source or extraction settings changed; use a new cache path")
    return cache


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def augmented_pooled(model, pool, bank, device, batch, rng, shift, degrees):
    rows = [[], []]
    for start in range(0, len(bank.vox), batch):
        x = bank.images(np.arange(start, min(start + batch, len(bank.vox))), device)
        x = warm.jitter(x, shift, degrees, rng, False)
        for view, part in enumerate(pool.pooled(model._encode(x, 2, None))):
            rows[view].append(part.detach().cpu())
    return torch.stack([torch.cat(parts) for parts in rows])


def prepare_cache(source, source_args, model, pool, residual, args):
    metadata = cache_metadata(source, source_args, args.draws, args.seed, args.batch, args.test_subjects)
    if args.cache.exists():
        return read_cache(args.cache, metadata)
    train = warm.Bank(source_args["bank"], "train", source_args["t1key"], residual=residual)
    test = warm.Bank(source_args["bank"], "test", source_args["t1key"], args.test_subjects, residual)
    raw = {}
    for label, bank in (("train", train), ("test", test)):
        print("extract", label, "unaugmented features", flush=True)
        values = warm.features(model, bank, args.device, args.batch, pool)
        raw[label] = {name: torch.from_numpy(np.stack(views)).float() for name, views in values.items()}
    rng = np.random.default_rng(args.seed + 1)
    jittered = []
    for draw in range(args.draws):
        print("extract training jitter draw", draw + 1, "of", args.draws, flush=True)
        jittered.append(
            augmented_pooled(
                model,
                pool,
                train,
                args.device,
                args.batch,
                rng,
                float(source_args.get("jitter", 0)),
                float(source_args.get("rotate", 0)),
            )
        )
    cache = {"metadata": metadata, "train_pooled": torch.stack(jittered), **raw}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(cache, args.cache)
    return cache


@torch.no_grad()
def codes(pool, pooled):
    return torch.stack([pool.head(p)[:, : pool.units] for p in pooled])


def score_codes(code, reference, bank):
    row = {}
    for view, name in enumerate(("t1", "flair")):
        features = code[view].detach().cpu().numpy()
        row[name] = {
            "code_volume_r2": warm.ridge(features, bank.vox),
            "gap_volume_r2": warm.ridge(reference["gap"][view].numpy(), bank.vox),
            "pooled_volume_r2": warm.ridge(reference["pooled"][view].numpy(), bank.vox),
            "code_latent_r2": warm.ridge(features, bank.zc[:, 2]),
            **{f"code_{name}": warm.ridge(features, bank.zc[:, column]) for column, name in warm.FACTORS.items()},
        }
    width = code.shape[-1]
    row["cross_view_code_r"] = float(
        np.abs(np.diag(np.corrcoef(*[view.detach().cpu().numpy().T for view in code])[:width, width:])).mean()
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("steps", type=int)
    parser.add_argument("every", type=int)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--whiten-ridge", type=float, default=0.1)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=warm.pick_device())
    parser.add_argument("--test-subjects", type=int, default=None)
    args = parser.parse_args()
    if min(args.steps, args.every, args.draws) < 1 or args.batch < 2:
        parser.error("steps, every and draws must be positive; batch must be at least 2")
    if not math.isfinite(args.prediction_weight) or args.prediction_weight < 0:
        parser.error("prediction weight must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.whiten_ridge)):
        parser.error("lr and whitening ridge must be finite and positive")
    source_report, model, pool, residual, source_step = code_audit.load_run(args.source, args.device)
    source_args = source_report["args"]
    if source_step != int(source_args["steps"]):
        parser.error("source run must be finished")
    if pool is None or source_args.get("supervise", "None") != "None":
        parser.error("source must have an unsupervised TailCode head")
    if source_args.get("remainder_units", "None") != "None" or source_args.get("shared_jitter") == "True":
        parser.error("source must use a single content block and independent jitter")
    if source_args.get("contrastive_loss_type", "infonce") != "infonce":
        parser.error("source must be the InfoNCE control")
    if source_args.get("critic_weights", "None") != "None":
        parser.error("source must use the unweighted InfoNCE similarity")
    args.test_subjects = (
        int(source_args.get("test_subjects", 1000)) if args.test_subjects is None else args.test_subjects
    )
    if args.test_subjects < 5:
        parser.error("test-subjects must be at least 5 for five-fold reports")
    freeze_features(model, pool)
    frozen_hash = (state_digest(model), state_digest(pool, exclude_head=True))
    initial_head_hash = state_digest(pool.head)
    cache = prepare_cache(args.source, source_args, model, pool, residual, args)
    if args.prepare_only:
        print("CACHE READY", args.cache, flush=True)
        return
    if args.batch > cache["train_pooled"].shape[2]:
        parser.error("batch exceeds the fixed training subject count")
    args.out.mkdir(parents=True, exist_ok=False)
    model.cpu()
    targets_stats = fit_whitening(cache["train"]["gap"], args.whiten_ridge)
    code_stats = fit_code_scale(cache["train"]["code"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        decoders = nn.ModuleList(nn.Linear(pool.units, cache["train"]["gap"].shape[-1]) for _ in range(2))
    decoders.to(args.device)
    parameters = list(pool.head.parameters()) + (list(decoders.parameters()) if args.prediction_weight else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    train_inputs = cache["train_pooled"].to(args.device)
    train_targets = whiten(cache["train"]["gap"], targets_stats).to(args.device)
    test_targets = whiten(cache["test"]["gap"], targets_stats).to(args.device)
    test_inputs = cache["test"]["pooled"].to(args.device)
    code_stats_device = {k: v.to(args.device) for k, v in code_stats.items()}
    test_bank = warm.Bank(source_args["bank"], "test", source_args["t1key"], args.test_subjects, residual)
    report_args = dict(source_args)
    report_args.update({k: str(v) for k, v in vars(args).items()})
    report_args.update(
        seed=source_args.get("seed", "None"),
        head_seed=str(args.seed),
        frozen_backbone="True",
        init=str(args.source),
        uniform="True",
        code_batches="0",
        queue="0",
        patch_weight="0.0",
        hard_negatives="0.0",
        decorrelate="0.0",
        group_dropout="0.0",
        half_views="None",
        half_units="None",
    )
    report = {
        "args": report_args,
        "history": [],
        "source_step": source_step,
        "source_checkpoint_sha256": cache["metadata"]["checkpoint_sha256"],
        "initial_head_sha256": initial_head_hash,
        "cache_metadata": cache["metadata"],
        "target": "opposite-view unaugmented frozen GAP; train-fit standardized ridge whitening",
        "code_scale": "fixed initial source-code mean and SD, training subjects only",
        "trainable": "shared head and training-only linear decoders",
        "prediction_weight_note": "initial experimental weight, not validated",
        "floor_note": "audit untrained twin is a fully random network; source run is the starting reference",
        "price_note": "precision_audit prices the InfoNCE component only",
    }
    rng = np.random.default_rng(args.seed)
    sequence = hashlib.sha256()

    def record(step, losses):
        with torch.no_grad():
            code = codes(pool, test_inputs)
            pred, directions = prediction_loss(code, test_targets, decoders, code_stats_device)
        evaluation = score_codes(code, cache["test"], test_bank)
        report["history"].append(
            {
                "step": step,
                "loss": losses,
                "test": evaluation,
                "spread": None,
                "test_prediction": float(pred),
                "test_t1_to_flair": float(directions[0]),
                "test_flair_to_t1": float(directions[1]),
                "batch_sequence_sha256": sequence.hexdigest(),
            }
        )
        report["frozen_state_unchanged"] = frozen_hash == (state_digest(model), state_digest(pool, exclude_head=True))
        if not report["frozen_state_unchanged"]:
            raise RuntimeError("Frozen backbone or pooling state changed")
        state = {
            "model": model.state_dict(),
            "pool": pool.state_dict(),
            "step": step,
            "prediction_decoders": decoders.state_dict(),
            "target_whitening": targets_stats,
            "prediction_code_scale": code_stats,
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
        }
        atomic_save(state, args.out / "model.pt")
        path = args.out / "report.json"
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(report, indent=1))
        temporary.replace(path)
        print(f"step {step:5d} prediction {float(pred):.4f} | {losses}", flush=True)

    record(0, {})
    started, totals, count = time.time(), {}, 0
    for step in range(1, args.steps + 1):
        indices, draws = draw_batch(rng, train_inputs.shape[2], args.batch, args.draws)
        sequence.update(indices.astype(np.int64).tobytes())
        sequence.update(draws.astype(np.int64).tobytes())
        subjects = torch.as_tensor(indices, device=args.device)
        draw_ids = torch.as_tensor(draws, device=args.device)
        inputs = torch.stack([train_inputs[draw_ids[v], v, subjects] for v in range(2)])
        losses = train_step(
            pool, decoders, optimizer, inputs, train_targets[:, subjects], code_stats_device, args.prediction_weight
        )
        count += 1
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value
        if step % args.every == 0 or step == args.steps:
            record(step, {k: v / count for k, v in totals.items()})
            totals, count = {}, 0
        if step % 500 == 0:
            print(f"step {step:5d} {(time.time() - started) / step:.3f}s/step", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
