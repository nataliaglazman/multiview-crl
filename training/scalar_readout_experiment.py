"""Compare scalar capacity, geometric capacity, and annotation-free learning.

python -m training.scalar_readout_experiment --run-dir RUN --out-dir NEW_OUTPUT --view t1

The source encoder is frozen. --source image instead trains fresh small readouts
from images as an observability control. Existing checkpoints are never modified.
"""

import argparse
import copy
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from eval.encoder.encoder_target_protocol import digest, provenance, save_csv, save_report
from eval.encoder.scalar_readout_audit import audit_arm, write_results
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model, load_settings
from models.scalar_readout import SEMANTIC_NAMES, DescriptorDecoder, ScalarReadout, ssl_objective
from training.scalar_readout_data import (
    RAW_NAMES,
    UNLABELLED_KEYS,
    Banks,
    Extractor,
    intervention_bank,
    observational_bank,
)
from utils.encoder_runtime import select_encoder_device

ARMS = ("free_oracle", "geometric_oracle", "geometric_ssl")


def tensor_rows(bank, key, ids, device):
    return torch.from_numpy(np.array(bank[key][ids], dtype=np.float32)).to(device)


def oracle_objective(model, observations, interventions, ids, pair_ids, scale, device):
    prediction = model(tensor_rows(observations, "features", ids, device))
    target = tensor_rows(observations, "truth", ids, device)
    pair_features = tensor_rows(interventions, "features", pair_ids, device)
    paired = model(pair_features.flatten(0, 1)).reshape(-1, 2, 9)
    paired_target = tensor_rows(interventions, "truth", pair_ids, device)
    level = ((prediction - target) / scale).square().mean()
    endpoint = ((paired - paired_target) / scale).square().mean()
    # Physical centroid responses can change under anatomy interventions. Match
    # the actual target response matrix; do not falsely force it to be diagonal.
    response = (((paired[:, 1] - paired[:, 0]) - (paired_target[:, 1] - paired_target[:, 0])) / scale).square().mean()
    return dict(level=level, endpoint=endpoint, response=response)


def train_oracle(model, observations, interventions, args, device):
    scale = np.asarray(observations["truth"], np.float64).std(0)
    if np.any(scale < 1e-8) or not np.isfinite(scale).all():
        raise ValueError("Training target is constant or non-finite; increase training cohort")
    scale_tensor = torch.as_tensor(scale, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng, history = np.random.default_rng(args.seed + 1), []
    for step in range(1, args.steps + 1):
        ids = rng.choice(len(observations["features"]), args.batch_size, replace=False)
        pair_ids = rng.choice(len(interventions["features"]), args.batch_size, replace=True)
        terms = oracle_objective(model, observations, interventions, ids, pair_ids, scale_tensor, device)
        loss = terms["level"] + 0.5 * terms["endpoint"] + args.response_weight * terms["response"]
        optimize(loss, optimizer, model.parameters())
        if step % args.log_every == 0 or step == args.steps:
            entry = dict(step=step, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()})
            history.append(entry)
            print(f"  step {step}/{args.steps}: {entry}", flush=True)
    return dict(
        history=history, target_scale=scale.tolist(), label_access="training semantic targets and factor interventions"
    )


def optimize(loss, optimizer, parameters):
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(parameters), 5.0, error_if_nonfinite=True)
    optimizer.step()


def train_ssl(model, unlabelled, args, device):
    # This function never receives a label, intervention bank, or validation data.
    if set(unlabelled) != set(UNLABELLED_KEYS):
        raise ValueError("SSL training accepts only the explicit unlabelled bank")
    decoder = DescriptorDecoder(args.descriptor_grid).to(device)
    descriptor = np.asarray(unlabelled["descriptors"], np.float64)
    mean = descriptor.mean((0, 2, 3, 4), keepdims=True).astype(np.float32)
    std = np.maximum(descriptor.std((0, 2, 3, 4), keepdims=True), 1e-6).astype(np.float32)
    mean_t, std_t = [torch.from_numpy(x).to(device) for x in (mean, std)]
    parameters = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    weights = dict(
        reconstruction=args.reconstruction_weight,
        equivariance=args.equivariance_weight,
        photometric=args.photometric_weight,
        variance=args.variance_weight,
    )
    rng, history = np.random.default_rng(args.seed + 1), []
    for step in range(1, args.steps + 1):
        ids = rng.choice(len(unlabelled["features"]), args.batch_size, replace=False)
        batch = {key: tensor_rows(unlabelled, key, ids, device) for key in UNLABELLED_KEYS}
        loss, terms = ssl_objective(model, decoder, batch, mean_t, std_t, weights)
        optimize(loss, optimizer, parameters)
        if step % args.log_every == 0 or step == args.steps:
            entry = dict(step=step, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()})
            history.append(entry)
            print(f"  step {step}/{args.steps}: {entry}", flush=True)
    return decoder, dict(
        history=history,
        descriptor_mean=mean.tolist(),
        descriptor_std=std.tolist(),
        auxiliary_parameters=sum(p.numel() for p in decoder.parameters()),
        label_access="none; image reconstruction, reflection equivariance, photometric consistency",
    )


@torch.no_grad()
def predict(model, bank, args, device, paired=False):
    model.eval()
    output = []
    for start in range(0, len(bank["features"]), args.batch_size):
        x = tensor_rows(bank, "features", slice(start, start + args.batch_size), device)
        y = model(x.flatten(0, 1) if paired else x)
        output.append(y.reshape(-1, 2, 9).cpu().numpy() if paired else y.cpu().numpy())
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite scalar outputs")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source", choices=("backbone", "image"), default="backbone")
    p.add_argument("--view", choices=("t1", "flair"), required=True)
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--train-samples", type=int, default=512)
    p.add_argument("--probe-samples", type=int, default=256)
    p.add_argument("--test-samples", type=int, default=400)
    p.add_argument("--train-pair-subjects", type=int, default=128)
    p.add_argument("--test-pair-subjects", type=int, default=32)
    p.add_argument("--train-eps", nargs="+", type=float, default=[0.5])
    p.add_argument("--eval-eps", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--descriptor-grid", type=int, default=16)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--response-weight", type=float, default=1.0)
    p.add_argument("--reconstruction-weight", type=float, default=1.0)
    p.add_argument("--equivariance-weight", type=float, default=10.0)
    p.add_argument("--photometric-weight", type=float, default=1.0)
    p.add_argument("--variance-weight", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--determinism", choices=("warn", "strict", "off"), default="warn")
    p.add_argument("--cache-dir", type=Path, help="Temporary-bank parent; automatically removed, including on failure")
    args = p.parse_args(argv)
    positive = (
        args.steps,
        args.train_samples,
        args.probe_samples,
        args.test_samples,
        args.train_pair_subjects,
        args.test_pair_subjects,
        args.batch_size,
        args.grid,
        args.descriptor_grid,
        args.width,
        args.log_every,
        args.cpu_threads,
    )
    if min(positive) < 1 or min(args.train_samples, args.probe_samples, args.test_samples) < 8:
        p.error("Need positive sizes and at least eight observations per split")
    if (
        args.batch_size < 2
        or args.batch_size > args.train_samples
        or (any(a.endswith("oracle") for a in args.arms) and args.train_pair_subjects > args.train_samples)
    ):
        p.error("Need 2 <= batch <= training subjects and training-pair subjects <= training subjects")
    weights = [
        args.lr,
        args.response_weight,
        args.reconstruction_weight,
        args.equivariance_weight,
        args.photometric_weight,
        args.variance_weight,
        *args.train_eps,
        *args.eval_eps,
    ]
    if not np.isfinite(weights).all() or min(weights) <= 0:
        p.error("Learning rate, weights and perturbation sizes must be finite and positive")
    if args.seed < 0 or len(set(args.arms)) != len(args.arms):
        p.error("Need a nonnegative seed and no duplicate arms")
    args.train_eps, args.eval_eps = sorted(set(args.train_eps)), sorted(set(args.eval_eps))
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_settings(args.run_dir)
    if cfg.get("synthetic_lesion_mode", "sphere") != "sphere":
        raise ValueError("This coordinate experiment requires sphere lesions")
    args.resolution = int(cfg["res"])
    if args.resolution % args.descriptor_grid or (args.source == "image" and args.resolution % (4 * args.grid)):
        raise ValueError(
            "Descriptor grid must divide resolution; image source also needs resolution divisible by 4*grid"
        )
    if args.train_samples > cfg.get("num_train_samples", args.train_samples):
        raise ValueError("Requested more training subjects than the saved training cohort")
    if cfg.get("content_channels", 9) < 9:
        raise ValueError("Need at least nine content channels for the saved global baseline")
    source_files = {"settings": args.run_dir / "settings.json"}
    if args.source == "backbone":
        source_files["checkpoint"] = args.run_dir / args.checkpoint
    source_hashes = {k: digest(v) for k, v in source_files.items()}
    device = select_encoder_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    runtime = dict(
        cfg,
        cpu_threads=args.cpu_threads,
        deterministic=args.determinism != "off",
        deterministic_warn_only=args.determinism == "warn",
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    report = provenance(cfg, args, device)
    report.update(
        protocol="Scalar readout capacity and learning-signal comparison",
        selection="fixed_final_step",
        target_names=list(SEMANTIC_NAMES),
        raw_factor_names=list(RAW_NAMES),
        source_hashes=source_hashes,
        limitations=[
            "Oracle arms use training targets and intervention identities: positive controls, not unsupervised methods.",
            "Geometric oracle and SSL start from identical readout weights; their supervision and auxiliary decoder differ.",
            "Output 2:5 is physical centroid; the WM placement rule couples centroid to anatomy. Match actual responses, not a forced diagonal.",
            "SSL accesses no factor labels or interventions during optimization. Validation labels are used only afterward to calibrate probes.",
            "One-scalar scores use affine calibration and one-to-one matching. Failure does not exclude nonlinear scalar recovery.",
            "Reconstruction descriptors and reflection equivariance do not guarantee lesion discovery or signed sulcal recovery.",
            "A failed finite-budget oracle is not proof of an information-theoretic impossibility.",
        ],
    )
    save_report(args.out_dir, report)
    banks = None
    start_time = time.monotonic()
    try:
        torch.set_num_threads(args.cpu_threads)
        torch.use_deterministic_algorithms(args.determinism != "off", warn_only=args.determinism == "warn")
        source_model = None
        if args.source == "backbone":
            state = torch.load(source_files["checkpoint"], map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            source_model = build_model(runtime, device, state)
            source_model.requires_grad_(False)
            report["encoder_state_before"] = state_digest(source_model)
        extract = Extractor(source_model, args, device)
        if args.cache_dir:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="scalar_readout_", dir=args.cache_dir) as temporary:
            banks = Banks(temporary)
            try:
                with threadpool_limits(limits=args.cpu_threads):
                    training, train_meta = observational_bank(
                        cfg, "train", args.train_samples, args, extract, banks, augment="geometric_ssl" in args.arms
                    )
                    report["cohorts"] = dict(train=train_meta)
                    train_pairs = None
                    if any(a.endswith("oracle") for a in args.arms):
                        train_pairs, meta = intervention_bank(
                            cfg, "train", args.train_pair_subjects, 0, args.train_eps, args, extract, banks
                        )
                        report["cohorts"]["train_interventions"] = meta
                        save_csv(args.out_dir / "training_intervention_pairs.csv", train_pairs["rows"])
                    channels = training["features"].shape[1]
                    torch.manual_seed(args.seed)
                    geometry = ScalarReadout(
                        channels, args.grid, args.width, args.resolution, True, args.source == "image"
                    )
                    initial_state = copy.deepcopy(geometry.state_dict())
                    readouts = {"initial_geometric": copy.deepcopy(geometry).to(device)}
                    report["arms"] = {}
                    for arm in args.arms:
                        torch.manual_seed(args.seed)
                        model = ScalarReadout(
                            channels,
                            args.grid,
                            args.width,
                            args.resolution,
                            arm != "free_oracle",
                            args.source == "image",
                        ).to(device)
                        if arm != "free_oracle":
                            model.load_state_dict(initial_state)
                        initial_hash = state_digest(model)
                        print(f"Training {arm} ({args.source}, {args.view}, {args.steps} steps)", flush=True)
                        model.train()
                        decoder = None
                        if arm == "geometric_ssl":
                            decoder, history = train_ssl(model, {k: training[k] for k in UNLABELLED_KEYS}, args, device)
                        else:
                            history = train_oracle(model, training, train_pairs, args, device)
                        report["arms"][arm] = dict(
                            **history,
                            initial_state_sha256=initial_hash,
                            readout_parameters=sum(p.numel() for p in model.parameters()),
                        )
                        torch.save(
                            dict(
                                state_dict=model.state_dict(),
                                arguments=report["arguments"],
                                decoder_state_dict=decoder.state_dict() if decoder is not None else None,
                                semantic_names=SEMANTIC_NAMES,
                            ),
                            args.out_dir / f"{arm}.pt",
                        )
                        readouts[arm] = model.eval()
                        save_report(args.out_dir, report)
                    # Held-out observations and interventions are rendered after all optimization.
                    observed = {}
                    for split, count in (("val", args.probe_samples), ("test", args.test_samples)):
                        observed[split], meta = observational_bank(cfg, split, count, args, extract, banks)
                        report["cohorts"][split] = meta
                    pairs, meta = intervention_bank(
                        cfg,
                        "test",
                        args.test_pair_subjects,
                        args.test_samples + 1000,
                        args.eval_eps,
                        args,
                        extract,
                        banks,
                    )
                    report["cohorts"]["test_interventions"] = meta
                    np.savez_compressed(
                        args.out_dir / "observational_truth.npz",
                        **{f"{split}_{key}": observed[split][key] for split in observed for key in ("raw", "truth")},
                    )
                    recovery, movement, responses = [], [], []
                    for arm, model in readouts.items():
                        codes = {split: predict(model, bank, args, device) for split, bank in observed.items()}
                        pair_codes = predict(model, pairs, args, device, paired=True)
                        rr, mm, ss = audit_arm(
                            arm,
                            codes,
                            observed,
                            pair_codes,
                            pairs,
                            args.out_dir,
                            args.seed,
                            direct=arm.endswith("oracle"),
                        )
                        recovery.extend(rr)
                        movement.extend(mm)
                        responses.extend(ss)
                    if source_model is not None:
                        rr, mm, ss = audit_arm(
                            "original_global",
                            {s: observed[s]["reference"] for s in observed},
                            observed,
                            pairs["reference"],
                            pairs,
                            args.out_dir,
                            args.seed,
                        )
                        recovery.extend(rr)
                        movement.extend(mm)
                        responses.extend(ss)
                    write_results(args.out_dir, recovery, movement, responses, pairs)
                    print(
                        "Held-out one-scalar R² (validation-matched units; geometric coordinates are physical):",
                        flush=True,
                    )
                    for r in recovery:
                        if (
                            r["family"] == "semantic"
                            and r["method"] == "scalar"
                            and r["target"] in SEMANTIC_NAMES[2:5] + SEMANTIC_NAMES[8:]
                        ):
                            print(f"  {r['arm']:20s} {r['target']:18s} {r['r2']:+.3f}", flush=True)
            finally:
                banks.close()
        if source_model is not None:
            report["encoder_state_after"] = state_digest(source_model)
            if report["encoder_state_after"] != report["encoder_state_before"]:
                raise RuntimeError("Frozen encoder changed")
        if {k: digest(v) for k, v in source_files.items()} != source_hashes:
            raise RuntimeError("Source files changed during experiment")
        report.update(
            status="complete",
            source_files_unchanged=True,
            temporary_banks_removed=True,
            elapsed_seconds=time.monotonic() - start_time,
        )
        save_report(args.out_dir, report)
        print(f"Saved scalar readout comparison: {args.out_dir}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)


if __name__ == "__main__":
    main()
