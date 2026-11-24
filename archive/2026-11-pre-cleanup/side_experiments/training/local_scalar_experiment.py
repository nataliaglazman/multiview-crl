"""Frozen local-scalar comparison: geometry, signed amplitude and learning signals.

python -m training.local_scalar_experiment --run-dir RUN --out-dir NEW --device cuda
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
from eval.encoder.local_scalar_audit import evaluate_arm, write_results
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model, load_settings
from models.local_scalar_readout import LocalDecoder, LocalReadout, reconstruction_loss
from training.local_scalar_data import UNLABELLED, FrozenExtractor, interventions, observations
from training.local_scalar_objectives import ResidualTarget, oracle_terms, read, tensors, unsupervised_objective
from training.scalar_readout_data import Banks
from training.scalar_readout_experiment import optimize
from utils.encoder_runtime import select_encoder_device

ARMS = ("hybrid_oracle", "free_oracle", "infonce", "decorrelated", "barlow", "residual")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--views", choices=("paired", "t1", "flair"), default="paired")
    p.add_argument("--train-intensity", choices=("saved", "fixed", "styled"), default="saved")
    p.add_argument(
        "--eval-intensities",
        choices=("saved", "fixed", "styled"),
        nargs="+",
        default=["saved"],
    )
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    for name, default in (
        ("steps", 2000),
        ("train-samples", 512),
        ("probe-samples", 256),
        ("test-samples", 400),
        ("train-pair-subjects", 64),
        ("test-pair-subjects", 32),
        ("batch-size", 32),
        ("grid", 8),
        ("heads", 4),
        ("target-channels", 64),
        ("seed", 42),
        ("log-every", 100),
        ("cpu-threads", 1),
        ("bootstrap", 200),
    ):
        p.add_argument(f"--{name}", type=int, default=default)
    for name, default in (
        ("lr", 0.001),
        ("temperature", 0.1),
        ("decorrelation-weight", 0.1),
        ("barlow-lambda", 0.005),
        ("equivariance-weight", 10.0),
        ("photometric-weight", 1.0),
        ("variance-weight", 1.0),
        ("response-weight", 1.0),
        ("ridge", 10.0),
        ("blob-width-vox", 3.0),
    ):
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--train-eps", type=float, nargs="+", default=[0.5])
    p.add_argument("--eval-eps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    p.add_argument("--determinism", choices=("warn", "strict", "off"), default="warn")
    args = p.parse_args(argv)
    for key in (
        "steps",
        "train_samples",
        "probe_samples",
        "test_samples",
        "train_pair_subjects",
        "test_pair_subjects",
        "heads",
        "target_channels",
        "log_every",
        "cpu_threads",
    ):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if (
        min(args.train_samples, args.probe_samples, args.test_samples) < 8
        or not 2 <= args.batch_size <= args.train_samples
    ):
        p.error("Need >=8 subjects per split and 2 <= batch size <= training subjects")
    if args.grid < 2 or args.grid % 2:
        p.error("Grid must be even and >=2 for the coarse residual loss")
    if args.seed < 0 or args.bootstrap < 0 or len(set(args.arms)) != len(args.arms):
        p.error("Invalid seed/bootstrap or duplicate arms")
    if any(a.endswith("oracle") for a in args.arms) and args.train_pair_subjects > args.train_samples:
        p.error("Training intervention subjects must be inside the training cohort")
    positive = [
        args.lr,
        args.temperature,
        args.ridge,
        args.blob_width_vox,
        *args.train_eps,
        *args.eval_eps,
    ]
    nonnegative = [
        args.decorrelation_weight,
        args.barlow_lambda,
        args.equivariance_weight,
        args.photometric_weight,
        args.variance_weight,
        args.response_weight,
    ]
    if not np.isfinite(positive + nonnegative).all() or min(positive) <= 0 or min(nonnegative) < 0:
        p.error("Invalid learning rate, loss weight, ridge or perturbation")
    args.views = ["t1", "flair"] if args.views == "paired" else [args.views]
    args.train_eps, args.eval_eps = sorted(set(args.train_eps)), sorted(set(args.eval_eps))
    return args


def train_arm(head, decoder, target, bank, pairs, arm, args, device):
    oracle = arm.endswith("oracle")
    if not oracle and set(bank) != set(UNLABELLED):
        raise ValueError("Unsupervised training may receive only unlabelled arrays")
    if not oracle and pairs is not None:
        raise ValueError("Unsupervised training cannot receive interventions")
    parameters = list(head.parameters()) + ([] if decoder is None else list(decoder.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    rng, history = np.random.default_rng(args.seed + 1), []
    pair_rng = np.random.default_rng(args.seed + 2)
    if oracle:
        scale = torch.as_tensor(bank["truth"][:, [2, 3, 4, 8]].std(0).clip(1e-5), device=device)
    head.train()
    for step in range(1, args.steps + 1):
        ids = rng.choice(len(bank["features"]), args.batch_size, replace=False)
        if oracle:
            pair_ids = pair_rng.choice(len(pairs["features"]), args.batch_size, replace=True)
            keys = ("features", "support", "truth")
            terms = oracle_terms(
                head,
                target,
                tensors(bank, ids, device, keys),
                tensors(pairs, pair_ids, device, keys),
                scale,
            )
            loss = terms["level"] + 0.5 * terms["endpoint"] + args.response_weight * terms["response"]
        else:
            loss, terms = unsupervised_objective(head, decoder, target, tensors(bank, ids, device), arm, args)
        optimize(loss, optimizer, parameters)
        if step % args.log_every == 0 or step == args.steps:
            row = dict(
                step=step,
                loss=float(loss.detach()),
                **{k: float(v.detach()) for k, v in terms.items()},
            )
            history.append(row)
            print(f"  {arm} {step}/{args.steps}: {row}", flush=True)
    return dict(
        history=history,
        label_access=(
            "training targets and finite factor interventions"
            if oracle
            else "none; images and synthetic augmentations only"
        ),
    )


@torch.inference_mode()
def predict(head, target, bank, args, device, paired=False):
    chunks = {}
    head.eval()
    for start in range(0, len(bank["features"]), args.batch_size):
        batch = tensors(bank, slice(start, start + args.batch_size), device, ("features", "support"))
        if paired:
            batch = {k: a.flatten(0, 1) for k, a in batch.items()}
        result = read(head, batch, target)
        for key, value in result.items():
            if paired:
                value = value.reshape(-1, 2, *value.shape[1:])
            chunks.setdefault(key, []).append(value.cpu().numpy())
    output = {k: np.concatenate(a) for k, a in chunks.items()}
    if not all(np.isfinite(a).all() for a in output.values()):
        raise ValueError("Non-finite local outputs")
    return output


@torch.inference_mode()
def reconstruction_report(target, head, decoder, bank, args, device, path):
    totals, maps, counts = {}, {}, {}
    for start in range(0, len(bank["features"]), args.batch_size):
        batch = tensors(
            bank,
            slice(start, start + args.batch_size),
            device,
            ("features", "support", "global"),
        )
        residual = target.residual(batch)
        code = read(head, batch, target) if decoder is not None else None
        prediction = decoder(code["physical"], code["amplitude"]) if decoder is not None else torch.zeros_like(residual)
        _, _, errors = reconstruction_loss(prediction, residual, batch["support"], target.scales(), return_maps=True)
        for name, error in errors.items():
            mask = torch.nn.functional.adaptive_avg_pool3d(batch["support"].flatten(0, 1), error.shape[-3:]).reshape(
                len(error), len(args.views), *error.shape[-3:]
            )
            sums = (error * mask).sum(0).cpu().numpy()
            mass = mask.sum(0).cpu().numpy()
            maps[name] = maps.get(name, 0) + sums
            counts[name] = counts.get(name, 0) + mass
    for name in maps:
        totals[name] = {
            view: float(maps[name][v].sum() / max(counts[name][v].sum(), 1)) for v, view in enumerate(args.views)
        }
    np.savez_compressed(
        path,
        **{k: np.divide(maps[k], counts[k], out=np.zeros_like(maps[k]), where=counts[k] > 0) for k in maps},
        **{f"{k}_occupancy": counts[k] for k in counts},
    )
    return totals


def main(argv=None):
    args = parse_args(argv)
    cfg = load_settings(args.run_dir)
    if cfg.get("synthetic_lesion_placement") != "wm_interior" or cfg.get("synthetic_lesion_mode", "sphere") != "sphere":
        raise ValueError("This experiment requires wm_interior sphere lesions")
    if cfg.get("lesion_keypoints", 0) or cfg.get("patch_loss_weight", 0):
        raise ValueError("Choose a global-only encoder checkpoint for this controlled comparison")
    args.resolution = int(cfg["res"])
    if args.resolution % args.grid or args.train_samples > cfg.get("num_train_samples", args.train_samples):
        raise ValueError("Grid must divide resolution and training count must fit saved cohort")
    device = select_encoder_device(args.device)
    source = dict(
        settings=args.run_dir / "settings.json",
        checkpoint=args.run_dir / args.checkpoint,
    )
    hashes = {k: digest(v) for k, v in source.items()}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, device)
    report.update(
        protocol="Frozen local scalar readouts",
        selection="fixed final step; no test selection",
        source_hashes=hashes,
        arms={},
        cohorts={},
        limitations=[
            "Oracle arms are supervised capacity controls.",
            "Frozen features, quadratic global prediction and signed residual targets "
            "do not guarantee factor identification.",
            "Geometric coordinates and raw WM-quantile controls are evaluated separately.",
            "Single-head selection and probe calibration use validation only.",
            "Styled visibility controls change the rendered distribution; fixed-intensity evaluation is separate.",
        ],
    )
    save_report(args.out_dir, report)
    threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.monotonic()
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(args.cpu_threads)
        torch.use_deterministic_algorithms(args.determinism != "off", warn_only=args.determinism == "warn")
        state = torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
        encoder = build_model(
            dict(
                cfg,
                cpu_threads=args.cpu_threads,
                deterministic=args.determinism != "off",
                deterministic_warn_only=args.determinism == "warn",
            ),
            device,
            state.get("state_dict", state),
        )
        encoder.eval().requires_grad_(False)
        report["encoder_state_before"] = state_digest(encoder)
        extract = FrozenExtractor(encoder, args.grid, args.views, device)
        train_cfg = dict(cfg)
        if args.train_intensity != "saved":
            train_cfg["synthetic_lesion_intensity"] = args.train_intensity
        if args.cache_dir:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="local_scalar_", dir=args.cache_dir) as temporary, threadpool_limits(
            limits=args.cpu_threads
        ):
            banks = Banks(temporary)
            try:
                training, meta = observations(
                    train_cfg,
                    "train",
                    args.train_samples,
                    args,
                    extract,
                    banks,
                    augment=True,
                )
                report["cohorts"]["train"] = meta
                labelled_pairs = None
                if any(a.endswith("oracle") for a in args.arms):
                    labelled_pairs, meta = interventions(
                        train_cfg,
                        "train",
                        args.train_pair_subjects,
                        0,
                        args.train_eps,
                        args,
                        extract,
                        banks,
                        "train_pairs",
                    )
                    report["cohorts"]["train_interventions"] = meta
                    save_csv(
                        args.out_dir / "training_intervention_pairs.csv",
                        labelled_pairs["rows"],
                    )
                unlabelled = {k: training[k] for k in UNLABELLED}
                target = ResidualTarget(
                    unlabelled,
                    args.target_channels,
                    args.ridge,
                    args.seed,
                    args.batch_size,
                ).to(device)
                target_hash = state_digest(target)
                torch.save(
                    dict(state_dict=target.state_dict(), arguments=report["arguments"]),
                    args.out_dir / "residual_target.pt",
                )
                channels = training["features"].shape[2]
                torch.manual_seed(args.seed)
                initial = LocalReadout(channels, args.grid, args.resolution, args.heads).to(device)
                models = {"initial": copy.deepcopy(initial)}
                decoders = {}
                for arm in args.arms:
                    torch.manual_seed(args.seed)
                    head = LocalReadout(
                        channels,
                        args.grid,
                        args.resolution,
                        args.heads,
                        arm == "free_oracle",
                    ).to(device)
                    if arm != "free_oracle":
                        head.load_state_dict(initial.state_dict())
                    decoder = (
                        LocalDecoder(
                            target.channels,
                            args.grid,
                            args.resolution,
                            args.heads,
                            len(args.views),
                            args.blob_width_vox,
                        ).to(device)
                        if arm == "residual"
                        else None
                    )
                    initial_hash = state_digest(head)
                    print(
                        f"Training {arm}: {args.steps} steps, views={args.views}",
                        flush=True,
                    )
                    oracle = arm.endswith("oracle")
                    history = train_arm(
                        head,
                        decoder,
                        target,
                        training if oracle else unlabelled,
                        labelled_pairs if oracle else None,
                        arm,
                        args,
                        device,
                    )
                    models[arm] = head.eval()
                    if decoder is not None:
                        decoders[arm] = decoder.eval()
                    report["arms"][arm] = dict(
                        **history,
                        initial_state_sha256=initial_hash,
                        readout_parameters=sum(p.numel() for p in head.parameters()),
                        decoder_parameters=(sum(p.numel() for p in decoder.parameters()) if decoder is not None else 0),
                    )
                    torch.save(
                        dict(
                            state_dict=head.state_dict(),
                            decoder_state_dict=(decoder.state_dict() if decoder is not None else None),
                            arguments=report["arguments"],
                            free=arm == "free_oracle",
                            channels=channels,
                            source_hashes=hashes,
                        ),
                        args.out_dir / f"{arm}.pt",
                    )
                    save_report(args.out_dir, report)
                report["reconstruction"] = {
                    "train": {
                        "global_only": reconstruction_report(
                            target,
                            initial,
                            None,
                            unlabelled,
                            args,
                            device,
                            args.out_dir / "train_global_error_maps.npz",
                        )
                    }
                }
                if "residual" in models:
                    report["reconstruction"]["train"]["local_residual"] = reconstruction_report(
                        target,
                        models["residual"],
                        decoders["residual"],
                        unlabelled,
                        args,
                        device,
                        args.out_dir / "train_local_error_maps.npz",
                    )
                saved_intensity = cfg.get("synthetic_lesion_intensity", "fixed")
                intensities = list(dict.fromkeys(saved_intensity if v == "saved" else v for v in args.eval_intensities))
                for intensity in intensities:
                    evaluation = args.out_dir / f"evaluation_{intensity}"
                    evaluation.mkdir()
                    eval_cfg = dict(cfg, synthetic_lesion_intensity=intensity)
                    observed = {}
                    for split, count in (
                        ("val", args.probe_samples),
                        ("test", args.test_samples),
                    ):
                        observed[split], meta = observations(
                            eval_cfg,
                            split,
                            count,
                            args,
                            extract,
                            banks,
                            namespace=f"{intensity}_{split}",
                            contrast=True,
                        )
                        report["cohorts"][f"{intensity}_{split}"] = meta
                    observed["pairs"], meta = interventions(
                        eval_cfg,
                        "test",
                        args.test_pair_subjects,
                        args.test_samples + 1000,
                        args.eval_eps,
                        args,
                        extract,
                        banks,
                        f"{intensity}_pairs",
                    )
                    report["cohorts"][f"{intensity}_interventions"] = meta
                    results = {}
                    for arm, head in models.items():
                        codes = {s: predict(head, target, b, args, device, s == "pairs") for s, b in observed.items()}
                        metrics = evaluate_arm(arm, codes, observed, args, evaluation)
                        for key, rows in metrics.items():
                            results.setdefault(key, []).extend(rows)
                    baseline = {s: {"code": np.asarray(b["global"])} for s, b in observed.items()}
                    for key, rows in evaluate_arm("original_global", baseline, observed, args, evaluation).items():
                        results.setdefault(key, []).extend(rows)
                    write_results(evaluation, results, observed, args)
                    report["reconstruction"][intensity] = {}
                    for split in ("val", "test"):
                        report["reconstruction"][intensity][split] = {
                            "global_only": reconstruction_report(
                                target,
                                initial,
                                None,
                                observed[split],
                                args,
                                device,
                                evaluation / f"{split}_global_error_maps.npz",
                            )
                        }
                        if "residual" in models:
                            report["reconstruction"][intensity][split]["local_residual"] = reconstruction_report(
                                target,
                                models["residual"],
                                decoders["residual"],
                                observed[split],
                                args,
                                device,
                                evaluation / f"{split}_local_error_maps.npz",
                            )
                    print(
                        f"Held-out {intensity}: physical centroid (joint ridge) / signed amplitude (one scalar)",
                        flush=True,
                    )
                    for arm in models:
                        for view in args.views:
                            selected = [
                                r
                                for r in results["recovery"]
                                if r["arm"] == arm and r["view"] == view and r["group"] == "all"
                            ]
                            centroid = np.mean(
                                [
                                    r["r2"]
                                    for r in selected
                                    if r["method"] == "joint_ridge" and r["target"].startswith("centroid_")
                                ]
                            )
                            amplitude = next(
                                r["r2"]
                                for r in selected
                                if r["method"] == "amplitude_scalar" and r["target"] == "sulcal_amplitude"
                            )
                            print(
                                f"  {arm:16s} {view:5s} {centroid:+.3f} / {amplitude:+.3f}",
                                flush=True,
                            )
            finally:
                banks.close()
        report["encoder_state_after"] = state_digest(encoder)
        if report["encoder_state_after"] != report["encoder_state_before"] or state_digest(target) != target_hash:
            raise RuntimeError("Frozen encoder or residual target changed")
        if {k: digest(v) for k, v in source.items()} != hashes:
            raise RuntimeError("Source files changed")
        report.update(
            status="complete",
            source_files_unchanged=True,
            residual_target_unchanged=True,
            temporary_banks_removed=True,
            elapsed_seconds=time.monotonic() - started,
        )
        save_report(args.out_dir, report)
        print(f"Saved local-scalar experiment: {args.out_dir}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn)


if __name__ == "__main__":
    main()
