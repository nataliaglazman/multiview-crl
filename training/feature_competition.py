"""Run the controlled lesion-location versus brain-size experiment.

python -m training.feature_competition --help
"""

import argparse
from pathlib import Path

# Kept equal to training.feature_competition_data.ARMS by a test; this module avoids importing torch.
ARMS = ("A", "B", "Bp", "C", "D")
# supervised: spatial heatmap localizer (does the image hold the location?);
# supervised_gap: the SSL encoder's own architecture trained with labels (can its GAP code carry it?).
KINDS = ("ssl", "supervised", "supervised_gap")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "audit", "train", "evaluate", "summarize", "curves"):
        q = commands.add_parser(name)
        q.add_argument("--bank", type=Path, required=True)
        q.add_argument("--cpu-threads", type=int, default=1)
        if name in ("prepare", "train", "evaluate", "curves"):
            help_text = "Arms to render" if name == "prepare" else "Arms to use (default: every arm in the bank)"
            q.add_argument(
                "--arms", nargs="+", choices=ARMS, default=list(ARMS) if name == "prepare" else None, help=help_text
            )
        if name == "prepare":
            for flag, value in (
                ("res", 64),
                ("train-samples", 2000),
                ("probe-samples", 400),
                ("test-samples", 1000),
                ("pilot-samples", 128),
                ("data-seed", 42),
            ):
                q.add_argument(f"--{flag}", type=int, default=value)
            q.add_argument("--radius", type=float, default=0.1)
            q.add_argument("--high-contrast", type=float, default=1.5)
            q.add_argument("--brain-scale", type=float, default=0.5)
        if name == "audit":
            q.add_argument("--noise-subjects", type=int, default=32)
            q.add_argument("--noise-repeats", type=int, default=32)
            q.add_argument("--displacement", type=int, default=1)
            q.add_argument("--overwrite", action="store_true", help="Replace an existing diagnostic report")
        if name in ("train", "evaluate"):
            q.add_argument("--kind", choices=KINDS, required=True)
            q.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
            q.add_argument("--views", nargs="+", choices=("t1", "flair"), default=["t1", "flair"])
            q.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
            q.add_argument("--batch-size", type=int, default=None)
            q.add_argument(
                "--standardize-code",
                action="store_true",
                help="SSL only: centre and scale the code over the batch, per view, before InfoNCE (own run folder)",
            )
        if name == "train":
            q.add_argument("--steps", type=int, default=10000)
            q.add_argument("--save-every", type=int, default=2000)
            q.add_argument(
                "--save-steps",
                nargs="*",
                type=int,
                default=[],
                help="Extra checkpoint steps, e.g. early learning-curve points",
            )
            q.add_argument("--log-every", type=int, default=100)
            q.add_argument("--lr", type=float, default=0.0001)
            q.add_argument("--hidden-channels", type=int, default=64)
            q.add_argument("--res-channels", type=int, default=32)
            q.add_argument("--resume", action="store_true")
        if name == "evaluate":
            q.add_argument(
                "--checkpoint",
                default="last.pt",
                help="A checkpoint file, step_000000.pt for the untrained floor, or 'all' for every saved step",
            )
            q.add_argument("--spatial-grid", type=int, default=8)
            q.add_argument("--overwrite", action="store_true", help="Recompute evaluations that already exist")
        if name == "curves":
            q.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    a = p.parse_args(argv)
    if a.command in ("train", "evaluate") and a.batch_size is None:
        a.batch_size = 32 if a.command == "train" and a.kind == "ssl" else 8
    for key in (
        "cpu_threads",
        "train_samples",
        "probe_samples",
        "test_samples",
        "pilot_samples",
        "noise_subjects",
        "noise_repeats",
        "displacement",
        "batch_size",
        "steps",
        "save_every",
        "log_every",
        "hidden_channels",
        "res_channels",
        "spatial_grid",
    ):
        if hasattr(a, key) and getattr(a, key) < 1:
            p.error(f"{key} must be positive")
    if a.command == "prepare":
        if a.res < 16 or a.res % 4 or min(a.train_samples, a.probe_samples, a.test_samples) < 8:
            p.error("Need resolution >=16 divisible by 4, and >=8 subjects per main split")
        if not (0 < a.radius < 1 and 1 < a.high_contrast <= 1.5 and 0 < a.brain_scale <= 1) or a.data_seed < 0:
            p.error("Invalid radius, contrast (1 < k <= 1.5), brain scale or data seed")
        if max(a.train_samples, a.probe_samples, a.test_samples, a.pilot_samples) >= 10_000_000:
            p.error("Too many subjects for seed namespaces")
    if a.command == "audit" and not 2 <= a.noise_repeats < 4000:
        p.error("Need 2 <= noise repeats < 4000")
    if (
        a.command in ("prepare", "train", "evaluate", "curves")
        and a.arms is not None
        and len(set(a.arms)) != len(a.arms)
    ):
        p.error("Duplicate arms")
    if a.command in ("train", "evaluate"):
        if any(s < 0 for s in a.seeds) or len(set(a.seeds)) != len(a.seeds):
            p.error("Seeds must be distinct and nonnegative")
        if len(set(a.views)) != len(a.views):
            p.error("Duplicate views")
    if a.command == "train" and not 0 < a.lr < float("inf"):
        p.error("Learning rate must be finite and positive")
    if a.command == "train" and any(s < 1 for s in a.save_steps):
        p.error("save_steps must be positive")
    if a.command in ("train", "evaluate") and a.standardize_code and a.kind != "ssl":
        p.error("--standardize-code applies to --kind ssl only")
    return a


def main(argv=None):
    args = parse_args(argv)

    from utils.encoder_runtime import configure_encoder_runtime

    configure_encoder_runtime(dict(cpu_threads=args.cpu_threads, deterministic=True, deterministic_warn_only=True))
    if args.command in ("prepare", "audit"):
        from training import feature_competition_data as data

        return getattr(data, args.command)(args)
    if args.command in ("train", "evaluate", "curves") and args.arms is None:
        from training.feature_competition_data import load_bank

        args.arms = load_bank(args.bank)["arms"]
    if args.command == "train":
        from training.feature_competition_train import train

        return train(args)
    from eval.synthetic import feature_competition_report

    return getattr(feature_competition_report, args.command)(args)


if __name__ == "__main__":
    main()
