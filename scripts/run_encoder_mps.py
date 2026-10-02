#!/usr/bin/env python3
"""Train one encoder ablation locally on an Apple-silicon GPU (MPS).

Options come from the same recipe and variant files as the Run:ai generator, so the
command differs from the cluster one only in --device, --out-dir and --model-id, plus any
--batch-size/--train-steps/--eval-every override. Before training, one disposable step on
random input checks the backend. --dry-run prints the command without importing torch;
--check runs that step and exits. The trainer's output is also written to
<results-dir>/logs/<model-id>.log.
"""

import argparse
import copy
import gc
import importlib.util
import math
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_encoder_ablation_runai as ablations  # noqa: E402

comparison = ablations.comparison
OVERRIDES = {"batch_size": "b", "train_steps": "t"}


def make_options(args):
    config = comparison.read_json(args.config)
    comparison.validate_config(config)
    if args.seed not in config["seeds"]:
        raise ValueError("Choose a seed from the comparison recipe")
    variants = ablations.launch.load_yaml(ROOT / "experiments/encoder_ablations/variants.yaml")
    variants.update({arch: {"reference": arch, "overrides": {}} for arch in comparison.ARCHITECTURES})
    if args.variant not in variants:
        raise ValueError(f"Unknown variant: {args.variant}")
    options = ablations.training_options(
        config, variants[args.variant], args.variant, args.seed, args.results_dir.resolve()
    )
    recipe = dict(options)
    for key in ("batch_size", "train_steps", "eval_every"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    options["device"] = "mps"
    if min(options["train_steps"], options["eval_every"]) < 1:
        raise ValueError("Training/evaluation steps must be positive")
    if not 2 <= options["batch_size"] <= options["num_train_samples"]:
        raise ValueError("Batch size must be at least 2 and fit the training dataset")
    # A shorter or smaller run must never occupy the directory of the recipe run.
    suffix = "".join(f"_{tag}{options[key]}" for key, tag in OVERRIDES.items() if options[key] != recipe[key])
    options["model_id"] = args.model_id or f"{args.variant}_s{args.seed}_mps{suffix}"
    if Path(options["model_id"]).name != options["model_id"] or options["model_id"] in (".", ".."):
        raise ValueError("--model-id must be a directory name, not a path")
    return options, recipe


def lpips_stand_in():
    """Directory holding a stand-in ``lpips`` package, or None when lpips is installed.

    ``training.losses`` imports LPIPS at module scope for the reconstruction losses, which the
    encoder-only trainer never constructs; the stand-in raises if anything ever does.
    """
    if importlib.util.find_spec("lpips") is not None:
        return None
    directory = Path(tempfile.mkdtemp(prefix="encoder-mps-"))
    (directory / "lpips").mkdir()
    (directory / "lpips" / "__init__.py").write_text(
        "class LPIPS:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise ImportError('lpips is not installed; encoder-only training should never build LPIPS')\n"
    )
    return directory


def check_training_step(options):
    """One disposable step of the real model, loss and AdamW update on random input.

    Fails before any run directory exists if MPS is unavailable, its forward pass disagrees
    with CPU on the same weights, an operator has neither a kernel nor a CPU fallback, the
    batch does not fit, or the step gives non-finite values or no weight update. Writes nothing.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(f"{sys.executable} has no torch; set ENCODER_PYTHON to an environment with it") from error
    from utils.encoder_runtime import select_encoder_device

    device = select_encoder_device("mps")
    print(f"Backend check: torch {torch.__version__}, batch {options['batch_size']}/view on {device}", flush=True)
    loss = _disposable_step(options, device)
    gc.collect()
    torch.mps.empty_cache()
    print(
        f"Backend check passed: MPS matches CPU; finite loss {loss:.4f}, gradients, rank and eval features.",
        flush=True,
    )


def _disposable_step(options, device):
    import torch

    from eval.protocol.score_checkpoint import build_model
    from training.main_conv_synthetic import contrastive_loss, effective_rank

    def forward(model, x):
        pooled = model(x, pool_only=True, n_views=2)[2][0]
        criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())
        return pooled, contrastive_loss(pooled, model, SimpleNamespace(**options), *criteria)

    reference = build_model(options, "cpu").train()
    model = copy.deepcopy(reference).to(device)
    generator = torch.Generator().manual_seed(options["model_seed"])
    x = torch.randn(2 * options["batch_size"], 1, *[options["res"]] * 3, generator=generator)
    # Same weights and the same two subjects per view, so train-mode BatchNorm sees one batch.
    b = options["batch_size"]
    pair = torch.cat([x[:2], x[b : b + 2]])
    with torch.no_grad():
        expected, got = forward(reference, pair), forward(model, pair.to(device))
    try:
        torch.testing.assert_close([t.cpu() for t in got], list(expected), rtol=1e-3, atol=1e-4)
    except AssertionError as error:
        raise RuntimeError(f"Backend check: MPS forward pass differs from CPU\n{error}") from error
    del reference
    x = x.to(device)
    weight = next(model.encoder.parameters())
    before = weight.detach().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"])
    pooled, loss = forward(model, x)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Backend check: non-finite loss")
    loss.backward()
    for part in (model.encoder, model.encoder_v1, model.to_encoding):
        grad = None if part is None else next(part.parameters()).grad
        if part is not None and (grad is None or not torch.isfinite(grad).all().item() or not grad.any().item()):
            raise RuntimeError("Backend check: missing, non-finite or zero gradient")
    if options["grad_clip"] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"])
    optimizer.step()
    if torch.equal(before, weight.detach().cpu()):
        raise RuntimeError("Backend check: AdamW did not update the encoder weights")
    if not math.isfinite(effective_rank(pooled[: options["batch_size"], : options["content_channels"]])):
        raise RuntimeError("Backend check: non-finite effective rank")
    model.eval()
    with torch.no_grad():
        if not torch.isfinite(model(x, pool_only=True, n_views=2)[2][0]).all().item():
            raise RuntimeError("Backend check: non-finite evaluation features")
    return loss.item()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/encoder_comparison.json")
    parser.add_argument("--variant", default="resnet_stride8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, help="Per-view batch; default: the recipe's")
    parser.add_argument("--train-steps", type=int, help="Default: the recipe's")
    parser.add_argument("--eval-every", type=int, help="Default: the recipe's")
    parser.add_argument(
        "--model-id", help="Default: <variant>_s<seed>_mps, plus _b<batch>/_t<steps> where they differ from the recipe"
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/encoder_ablations_mps")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the training command and exit")
    mode.add_argument("--check", action="store_true", help="Run one disposable MPS training step and exit")
    args = parser.parse_args(argv)
    options, recipe = make_options(args)
    command = comparison.training_command(options)
    if args.dry_run:
        print(shlex.join(command))
        return
    run = Path(options["out_dir"]) / options["model_id"]
    if not args.check and run.exists():
        raise ValueError(f"Run already exists: {run}. Choose a new --model-id or --results-dir")
    # torch reads this once, at import; the shell wrapper sets it too.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    for name in ablations.launch.THREAD_ENV_VARS:
        os.environ[name] = str(options["cpu_threads"])
    if options["batch_size"] != recipe["batch_size"]:
        print(
            f"NOTE: batch {options['batch_size']}/view, recipe {recipe['batch_size']}/view. This changes the "
            "InfoNCE negatives, BatchNorm statistics and subjects seen, so the run is not comparable "
            "with the cluster runs.",
            flush=True,
        )
    stand_in = lpips_stand_in()
    try:
        if stand_in is not None:
            print("lpips is not installed; using a stand-in that raises if LPIPS is ever constructed.", flush=True)
            sys.path.insert(0, str(stand_in))
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(stand_in), os.environ.get("PYTHONPATH"))))
        check_training_step(options)
        if args.check:
            return
        log = args.results_dir.resolve() / "logs" / f"{options['model_id']}.log"
        comparison.execute(command, log, options["cpu_threads"])
        _, hashes = comparison.validate_run(run, options, require_receipt=False)
        comparison.write_json(run / "comparison_receipt.json", hashes)
        print(f"Run complete: {run}\nLog: {log}", flush=True)
    finally:
        if stand_in is not None:
            shutil.rmtree(stand_in, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
