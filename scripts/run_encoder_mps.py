#!/usr/bin/env python3
"""Train one encoder ablation locally on MPS (default) or CUDA.

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
    options["device"] = getattr(args, "device", "mps")
    if options["device"] not in ("mps", "cuda"):
        raise ValueError("The local GPU runner requires --device mps or cuda")
    patch_weight = getattr(args, "patch_loss_weight", None)
    patch_grid = getattr(args, "train_patch_grid", None)
    if patch_weight is not None:
        options["patch_loss_weight"] = patch_weight
    if patch_grid is not None:
        options["train_patch_grid"] = patch_grid
    for key in (
        "separate_spatial_readout",
        "global_pool",
        "attention_pool_heads",
        "attention_pool_frequencies",
        "lesion_keypoints",
        "lesion_norm",
        "lesion_frame",
        "lesion_proj_dim",
        "lesion_loss_weight",
        "lesion_decorrelation_weight",
        "lesion_temperature",
        "lesion_pairing",
        "lesion_input",
        "lesion_normative_components",
        "lesion_normative_subjects",
        "lesion_head_init",
        "lesion_detector",
        "lesion_branch_frozen",
        "lesion_localization_eval",
        "spatial_recovery_eval",
        "spatial_recovery_grids",
        "spatial_recovery_native",
        "spatial_recovery_batch_size",
        "spatial_recovery_test_samples",
        "spatial_recovery_seed",
    ):
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    if options.get("spatial_recovery_eval") and options.get("spatial_recovery_grids") is not None:
        options["spatial_recovery_grids"] = sorted(set([1, *options["spatial_recovery_grids"]]))
    lesion_radius = getattr(args, "synthetic_lesion_radius", None)
    if lesion_radius is not None:
        if not 0 < lesion_radius < float("inf"):
            raise ValueError("Lesion radius must be finite and positive")
        options["synthetic_lesion_radius"] = lesion_radius
    lesion_intensity = getattr(args, "synthetic_lesion_intensity", None)
    if lesion_intensity is not None:
        options["synthetic_lesion_intensity"] = lesion_intensity
    for key in ("synthetic_lesion_target", "synthetic_lesion_count", "synthetic_lesion_t1_value"):
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    if getattr(args, "synthetic_identifiable_ventricle", None):
        options["synthetic_identifiable_ventricle"] = True
    sulcal_mode = getattr(args, "synthetic_sulcal_mode", None)
    if sulcal_mode is not None:
        options["synthetic_sulcal_mode"] = sulcal_mode
    if getattr(args, "synthetic_causal", None) is not None:
        options["synthetic_causal"] = args.synthetic_causal
    for key in (
        "synthetic_causal_graph",
        "synthetic_causal_edge_prob",
        "synthetic_causal_noise_scale",
        "synthetic_causal_nonlinearity",
    ):
        if getattr(args, key, None) is not None:
            if not options.get("synthetic_causal", False):
                raise ValueError(f"--{key.replace('_', '-')} requires --synthetic-causal")
            options[key] = getattr(args, key)
    edge_prob = options.get("synthetic_causal_edge_prob", 0.5)
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("--synthetic-causal-edge-prob must be between 0 and 1")
    if getattr(args, "patch_foreground_mask", None):
        options["patch_foreground_mask"] = True
    if getattr(args, "patch_foreground_thresh", None) is not None:
        options["patch_foreground_thresh"] = args.patch_foreground_thresh
    if getattr(args, "norm_type", None) is not None:
        options["norm_type"] = args.norm_type
    if options.get("norm_type", "group") != "group" and options["encoder_architecture"] != "conv":
        raise ValueError("--norm-type applies to the Conv encoder; ResNet variants use resnet_norm")
    factor = getattr(args, "downscale_factor", None)
    if factor is not None:
        if options["encoder_architecture"] != "conv":
            raise ValueError("--downscale-factor applies to the Conv encoder; ResNet variants use resnet_output_stride")
        if factor < 1 or factor & (factor - 1):
            raise ValueError("--downscale-factor must be a power of 2")
        options["downscale_factor"] = factor
    weight = options.get("patch_loss_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Patch loss weight must be finite and nonnegative")
    if options.get("patch_foreground_mask", False) and weight <= 0:
        raise ValueError("The patch foreground mask requires patch training (a positive patch loss weight)")
    if options.get("separate_spatial_readout", False):
        if options["encoder_architecture"] == "conv" and options.get("conv_readout", "linear") != "mlp":
            raise ValueError("Separate spatial readout requires the Conv MLP or ResNet variant")
        if weight <= 0 or options.get("contrastive_proj_dim", 0) != 0:
            raise ValueError("Separate spatial readout requires positive patch weight and no loss projector")
    stride = (
        options["downscale_factor"] if options["encoder_architecture"] == "conv" else options["resnet_output_stride"]
    )
    if options.get("global_pool", "gap") == "attention":
        # As in the trainer: padded ResNet layers round odd sizes up, the conv backbone rounds down.
        res = options["res"]
        native = res // stride if options["encoder_architecture"] == "conv" else (res + stride - 1) // stride
        if native < 2:
            raise ValueError("Attention pooling needs a backbone map with more than one position")
    elif any(key in options for key in ("attention_pool_heads", "attention_pool_frequencies")):
        raise ValueError("Attention-pool heads/frequencies require --global-pool attention")
    lesion_keys = (
        "lesion_detector",
        "lesion_branch_frozen",
        "lesion_localization_eval",
        "lesion_norm",
        "lesion_frame",
        "lesion_proj_dim",
        "lesion_loss_weight",
        "lesion_decorrelation_weight",
        "lesion_temperature",
        "lesion_pairing",
        "lesion_input",
        "lesion_normative_components",
        "lesion_normative_subjects",
        "lesion_head_init",
    )
    if options.get("lesion_keypoints", 0) > 0:
        res = options["res"]
        native = res // stride if options["encoder_architecture"] == "conv" else (res + stride - 1) // stride
        if native < 2:
            raise ValueError("The lesion branch needs a backbone map with more than one position")
        if options["contrastive_loss_type"] != "infonce":
            raise ValueError("The lesion branch trains with its own InfoNCE")
    elif options.get("lesion_keypoints", 0) < 0:
        raise ValueError("Lesion keypoints must be nonnegative")
    elif any(key in options for key in lesion_keys):
        raise ValueError("Lesion norm/frame/projector/weight require --lesion-keypoints")
    if weight > 0:
        options.setdefault("train_patch_grid", [8, 8, 8])
        if options["contrastive_loss_type"] != "infonce":
            raise ValueError("Patch training requires InfoNCE")
        spatial = options["res"] // stride
        if len(options["train_patch_grid"]) != 3 or any(g < 1 or g > spatial for g in options["train_patch_grid"]):
            raise ValueError(f"Training patch grid must fit the {spatial}^3 backbone map")
    if min(options["train_steps"], options["eval_every"]) < 1:
        raise ValueError("Training/evaluation steps must be positive")
    if not 2 <= options["batch_size"] <= options["num_train_samples"]:
        raise ValueError("Batch size must be at least 2 and fit the training dataset")
    # A shorter or smaller run must never occupy the directory of the recipe run.
    suffix = "".join(f"_{tag}{options[key]}" for key, tag in OVERRIDES.items() if options[key] != recipe[key])
    if weight > 0:
        suffix += "_patch" + "x".join(map(str, options["train_patch_grid"])) + f"_w{weight:g}"
    if options.get("patch_foreground_mask", False):
        threshold = options.get("patch_foreground_thresh", 0.05)
        suffix += "_fgmask" + ("" if threshold == 0.05 else f"{threshold:g}")
    if options.get("separate_spatial_readout", False):
        suffix += "_separate_spatial"
    if options.get("global_pool", "gap") == "attention":
        suffix += f"_attnpool_h{options.get('attention_pool_heads', 4)}_f{options.get('attention_pool_frequencies', 4)}"
    if options.get("lesion_keypoints", 0) > 0:
        suffix += f"_lesionkp{options['lesion_keypoints']}"
        if options.get("lesion_detector", "shared") != "shared":
            suffix += f"_ld{options['lesion_detector']}"
        if options.get("lesion_branch_frozen", False):
            suffix += "_lfrozen"
        if options.get("lesion_frame", "brain") != "brain":
            suffix += f"_{options['lesion_frame']}frame"
        if options.get("lesion_norm", "none") != "none":
            suffix += f"_{options['lesion_norm']}lnorm"
        if options.get("lesion_loss_weight", 1.0) != 1.0:
            suffix += f"_lw{options['lesion_loss_weight']:g}"
        if options.get("lesion_decorrelation_weight", 0.0) > 0:
            suffix += f"_dc{options['lesion_decorrelation_weight']:g}"
        if options.get("lesion_temperature", 1.0) != 1.0:
            suffix += f"_temp{options['lesion_temperature']:g}"
        if options.get("lesion_pairing", "cross_modal") != "cross_modal":
            suffix += "_lpwithin"
        if options.get("lesion_input", "features") == "residual":
            suffix += "_lresid"
            if options.get("lesion_normative_components", 20) != 20:
                suffix += f"_nk{options['lesion_normative_components']}"
            if options.get("lesion_head_init", "positive") != "positive":
                suffix += f"_hinit{options['lesion_head_init']}"
    if options.get("norm_type", "group") != recipe.get("norm_type", "group"):
        suffix += f"_{options['norm_type']}norm"
    if options.get("downscale_factor") != recipe.get("downscale_factor"):
        suffix += f"_ds{options['downscale_factor']}"
    if options.get("synthetic_lesion_radius", 0.1) != recipe.get("synthetic_lesion_radius", 0.1):
        suffix += f"_lr{options['synthetic_lesion_radius']:g}"
    if options.get("synthetic_lesion_intensity", "fixed") != recipe.get("synthetic_lesion_intensity", "fixed"):
        suffix += f"_lesion{options['synthetic_lesion_intensity']}"
    if options.get("synthetic_lesion_target", "position") == "burden":
        suffix += "_burden"
        if options.get("synthetic_lesion_count", 4) != 4:
            suffix += f"{options['synthetic_lesion_count']}"
    if options.get("synthetic_lesion_t1_value", 0.4) != 0.4:
        suffix += f"_t1les{options['synthetic_lesion_t1_value']:g}"
    if options.get("synthetic_identifiable_ventricle", False) != recipe.get("synthetic_identifiable_ventricle", False):
        suffix += "_identvent"
    if options.get("synthetic_sulcal_mode", "corrugation") != recipe.get("synthetic_sulcal_mode", "corrugation"):
        suffix += f"_sulcal{options['synthetic_sulcal_mode']}"
    if options.get("synthetic_causal", False) and not recipe.get("synthetic_causal", False):
        graph = options.get("synthetic_causal_graph", "chain")
        suffix += f"_causal{graph}"
        if graph == "random" and edge_prob != 0.5:
            suffix += f"{edge_prob:g}"
        if options.get("synthetic_causal_noise_scale", 0.4) != 0.4:
            suffix += f"_cnoise{options['synthetic_causal_noise_scale']:g}"
        if options.get("synthetic_causal_nonlinearity", "leaky_relu") != "leaky_relu":
            suffix += f"_clin{options['synthetic_causal_nonlinearity']}"
    options["model_id"] = args.model_id or f"{args.variant}_s{args.seed}_{options['device']}{suffix}"
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

    Fails before any run directory exists if the GPU is unavailable, its forward pass disagrees
    with CPU on the same weights, an operator has neither a kernel nor a CPU fallback, the
    batch does not fit, or the step gives non-finite values or no weight update. Writes nothing.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(f"{sys.executable} has no torch; set ENCODER_PYTHON to an environment with it") from error
    from utils.encoder_runtime import configure_encoder_runtime, select_encoder_device

    configure_encoder_runtime(options)
    device = select_encoder_device(options.get("device", "mps"))
    print(f"Backend check: torch {torch.__version__}, batch {options['batch_size']}/view on {device}", flush=True)
    loss = _disposable_step(options, device)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
    print(
        f"Backend check passed: {device.upper()} matches CPU; finite loss {loss:.4f}, gradients, rank and eval features.",
        flush=True,
    )


def compare_backend(expected, got, options):
    """Raise AssertionError unless the GPU forward pass matches CPU on the same weights.

    ``expected``/``got`` are (pooled, loss, global term, patch term). Without a lesion branch
    everything is compared at rtol 1e-3, atol 1e-4. With one, the global code and the
    global/patch terms keep that tolerance, but the keypoint coordinates are a softmax whose
    logits are divided by --lesion-temperature, which amplifies backend rounding (TF32
    convolutions on CUDA) by 1/temperature, so their tolerance is scaled by it. The total loss
    is then not compared: it includes the lesion loss, which under within-modality pairing
    draws random augmentations that differ between the CPU and GPU generators.
    """
    import torch

    got = [t.cpu() for t in got]
    latent = options["latent_dim"]
    if options.get("lesion_keypoints", 0) <= 0:
        torch.testing.assert_close(got, list(expected), rtol=1e-3, atol=1e-4)
        return
    strict = [expected[0][:, :latent], expected[2], expected[3]]
    torch.testing.assert_close([got[0][:, :latent], got[2], got[3]], strict, rtol=1e-3, atol=1e-4)
    scale = 1.0 / options.get("lesion_temperature", 1.0)
    torch.testing.assert_close(got[0][:, latent:], expected[0][:, latent:], rtol=1e-3, atol=1e-4 * scale)


def _disposable_step(options, device):
    import torch

    from eval.protocol.score_checkpoint import build_model
    from training.main_conv_synthetic import effective_rank, training_objective

    def forward(model, x):
        criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())
        pooled, loss, terms = training_objective(model, x, SimpleNamespace(**options), *criteria)
        return pooled, loss, terms["global"], terms["patch"]

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
        compare_backend(expected, got, options)
    except AssertionError as error:
        raise RuntimeError(f"Backend check: {str(device).upper()} forward pass differs from CPU\n{error}") from error
    del reference
    x = x.to(device)
    weight = next(model.encoder.parameters())
    before = weight.detach().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"])
    pooled, loss, _, _ = forward(model, x)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Backend check: non-finite loss")
    loss.backward()
    # An unfitted normative model gives the residual lesion branch all-zero inputs, so its
    # gradient is legitimately zero here; its forward values are still compared above.
    unfitted = getattr(model, "normative", None) is not None and not bool(model.normative.fitted)
    for part in (
        model.encoder,
        model.encoder_v1,
        model.to_encoding,
        model.spatial_readout,
        model.attention_pool,
        None if unfitted else model.lesion_pool,
    ):
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
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, help="Per-view batch; default: the recipe's")
    parser.add_argument("--train-steps", type=int, help="Default: the recipe's")
    parser.add_argument("--eval-every", type=int, help="Default: the recipe's")
    parser.add_argument("--patch-loss-weight", type=float, help="Add spatial InfoNCE; 0 keeps global-only training")
    parser.add_argument("--train-patch-grid", type=int, nargs=3, help="Default for patch training: 8 8 8")
    parser.add_argument("--separate-spatial-readout", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--global-pool", choices=("gap", "attention"), help="Global pooling; default: the recipe's (trainer: gap)"
    )
    parser.add_argument("--attention-pool-heads", type=int, help="With --global-pool attention; trainer default 4")
    parser.add_argument(
        "--attention-pool-frequencies", type=int, help="With --global-pool attention; trainer default 4"
    )
    parser.add_argument(
        "--lesion-keypoints", type=int, help="Lesion keypoint branch heads; the run ID gains _lesionkp<K>"
    )
    parser.add_argument("--lesion-norm", choices=("none", "layer"), help="With --lesion-keypoints; trainer: none")
    parser.add_argument("--lesion-frame", choices=("brain", "grid"), help="With --lesion-keypoints; trainer: brain")
    parser.add_argument("--lesion-proj-dim", type=int, help="With --lesion-keypoints; trainer default 8")
    parser.add_argument("--lesion-loss-weight", type=float, help="With --lesion-keypoints; trainer default 1")
    parser.add_argument(
        "--lesion-decorrelation-weight", type=float, help="With --lesion-keypoints; trainer default 0 (off)"
    )
    parser.add_argument("--lesion-temperature", type=float, help="With --lesion-keypoints; trainer default 1")
    parser.add_argument(
        "--lesion-input", choices=("features", "residual"), help="With --lesion-keypoints; trainer default features"
    )
    parser.add_argument("--lesion-normative-components", type=int, help="With --lesion-input residual; default 20")
    parser.add_argument("--lesion-normative-subjects", type=int, help="With --lesion-input residual; default 300")
    parser.add_argument(
        "--lesion-head-init",
        choices=("positive", "random", "negative"),
        help="With --lesion-input residual; trainer default positive",
    )
    parser.add_argument(
        "--lesion-pairing",
        choices=("cross_modal", "within_modality"),
        help="With --lesion-keypoints; trainer default cross_modal",
    )
    parser.add_argument("--lesion-detector", choices=("shared", "separate", "separate_conv"))
    parser.add_argument("--lesion-branch-frozen", action="store_true", default=None)
    parser.add_argument("--lesion-localization-eval", action="store_true", default=None)
    parser.add_argument("--spatial-recovery-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spatial-recovery-grids", type=int, nargs="+")
    parser.add_argument("--spatial-recovery-native", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spatial-recovery-batch-size", type=int)
    parser.add_argument("--spatial-recovery-test-samples", type=int)
    parser.add_argument("--spatial-recovery-seed", type=int)
    parser.add_argument(
        "--synthetic-lesion-radius",
        type=float,
        help="Sphere lesion radius in [-1,1] coords; default: the recipe's (trainer default 0.1)",
    )
    parser.add_argument(
        "--synthetic-lesion-intensity",
        choices=("fixed", "styled"),
        help="styled puts the lesion on the acquisition gain/bias map; default: the recipe's (trainer default fixed)",
    )
    parser.add_argument(
        "--synthetic-lesion-target",
        choices=("position", "burden"),
        help="burden: z_content[2] is the total lesion volume (adds _burden to the run ID); trainer default position",
    )
    parser.add_argument(
        "--synthetic-lesion-count", type=int, help="With --synthetic-lesion-target burden; trainer default 4"
    )
    parser.add_argument(
        "--synthetic-lesion-t1-value",
        type=float,
        help="T1 lesion base intensity on the tissue LUT (adds _t1les<value> to the run ID); trainer default 0.4",
    )
    parser.add_argument(
        "--synthetic-identifiable-ventricle",
        action="store_true",
        default=None,
        help="The VQ-VAE recipe's larger, undeformed ventricle and separately labelled fissure; adds _identvent "
        "to the run ID",
    )
    parser.add_argument(
        "--synthetic-sulcal-mode",
        choices=("corrugation", "atrophy"),
        help="atrophy widens fixed sulcal clefts instead of the zero-mean corrugation (adds _sulcalatrophy to the "
        "run ID); default: the recipe's (trainer default corrugation)",
    )
    parser.add_argument(
        "--synthetic-causal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw content factors from an SCM (needed for causal-discovery evals); adds _causal<graph> to the "
        "run ID. Default: the recipe's (off)",
    )
    parser.add_argument(
        "--synthetic-causal-graph",
        choices=("chain", "full", "random"),
        help="With --synthetic-causal; trainer default chain",
    )
    parser.add_argument(
        "--synthetic-causal-edge-prob", type=float, help="With --synthetic-causal-graph random; trainer default 0.5"
    )
    parser.add_argument(
        "--synthetic-causal-noise-scale", type=float, help="With --synthetic-causal; trainer default 0.4"
    )
    parser.add_argument(
        "--synthetic-causal-nonlinearity",
        choices=("leaky_relu", "none"),
        help="With --synthetic-causal; trainer default leaky_relu",
    )
    parser.add_argument(
        "--downscale-factor",
        type=int,
        help="Conv encoder downsampling, a power of 2; default: the recipe's (4). Adds _ds<k> to the run ID. "
        "2 keeps small lesions at 2-3 cells instead of ~1, at roughly 8x the backbone compute",
    )
    parser.add_argument(
        "--norm-type",
        choices=("group", "layer"),
        help="Conv encoder norm; default: the recipe's (trainer default group). layer adds _layernorm to the run ID",
    )
    parser.add_argument(
        "--patch-foreground-mask",
        action="store_true",
        help="Drop always-background positions from the patch loss; the run ID gains _fgmask",
    )
    parser.add_argument(
        "--patch-foreground-thresh", type=float, help="Brain fraction a position needs; trainer default 0.05"
    )
    parser.add_argument(
        "--model-id",
        help="Default: <variant>_s<seed>_<device>, with suffixes for batch/steps/patch/lesion overrides",
    )
    parser.add_argument("--results-dir", type=Path, help="Default: results/encoder_ablations_<device>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the training command and exit")
    mode.add_argument("--check", action="store_true", help="Run one disposable GPU training step and exit")
    args = parser.parse_args(argv)
    if args.results_dir is None:
        args.results_dir = ROOT / f"results/encoder_ablations_{args.device}"
    options, recipe = make_options(args)
    command = comparison.training_command(options)
    if args.dry_run:
        print(shlex.join(command))
        return
    run = Path(options["out_dir"]) / options["model_id"]
    if not args.check and run.exists():
        raise ValueError(f"Run already exists: {run}. Choose a new --model-id or --results-dir")
    # Set backend environment before importing torch; shell wrappers set it too.
    if options["device"] == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if options.get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
