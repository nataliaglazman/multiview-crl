#!/usr/bin/env python
"""Embed the synthetic pseudo-MRI views with a pretrained DINOv3 and save the features.

    python -m eval.dinov3_embed_synthetic --out results/dinov3/emb.npz --num-samples 500
    python -m eval.dinov3_embed_synthetic --run-dir results/synthetic/RUN --out emb.npz

    # untrained twin at the same architecture -- the floor every number is read against
    python -m eval.dinov3_embed_synthetic --out emb_floor.npz --random-init --num-samples 500

Score what comes out with ``eval/dinov3_identifiability.py``: per-factor recovery and PC
graph recovery, on the same protocol the VQ-VAE runs are scored on.

Needs ``transformers`` (``python -m pip install transformers``) and network access to the
model repo the first time. DINOv3 weights are gated on the Hub: accept the licence on the
model page and ``hf auth login`` first, or point ``--model-id`` at a local snapshot.

What this script has to decide, and why
---------------------------------------
**3-D volume -> 2-D planes.** DINOv3 is a 2-D ViT, so the volume is reduced to a fixed set
of planes: ``--slices`` evenly spaced positions along each axis in ``--axes``, embedded
independently and then either concatenated (default) or averaged. The generator's lattice
is ``meshgrid(x, y, z, indexing="ij")`` with x the left-right axis (see
``PseudoMRIRenderer.render_structure``), so axis 0 cuts sagittal planes, axis 1 coronal and
axis 2 axial. ``--slice-agg mean`` throws away which plane a feature came from, which is
most of what locates ``lesion_x/y/z``; keep the default ``concat`` if position matters.

**Windowing.** ``--window per_slice`` is the usual DINO recipe (1-99 percentile per slice,
as in ``eval/dino.ipynb``) and it rescales every slice to the same range, which is exactly
the affine intensity map the style factors apply (``lut = base * gain + bias``). Under it
a style-recovery number is bounded by the windowing, not by the encoder -- the same trap
``SyntheticBrainDataset`` warns about for ``--synthetic-normalize per_sample``. The default
``dataset`` estimates one window from a pilot of volumes and applies it to every slice, so
per-sample intensity survives into the ViT input.

**Token pooling.** ``cls_mean`` (default) concatenates the CLS token with the mean patch
token. Mean pooling is permutation-invariant over patches, so in-plane position is not in
it; ``--token-pool grid --grid-size 2`` keeps a 2x2 average of the patch map instead, which
is the analogue of this project's patch pooling and what a position-like factor needs.

Output is a single ``.npz``: the embeddings per view, the ground-truth latents, the SCM
adjacency when the generator is causal, an optional downsampled-voxel baseline, and a
``meta`` JSON string recording every choice above. A sidecar ``<out>.meta.json`` holds the
same metadata in readable form.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The generator's lattice is meshgrid(x, y, z, indexing="ij") with x left-right,
# y anterior-posterior and z inferior-superior, so fixing axis 0 cuts a sagittal
# plane, axis 1 a coronal one and axis 2 an axial one.
AXES = {"sagittal": 0, "coronal": 1, "axial": 2}

# Fallback ViT normalization when the checkpoint ships no preprocessor config.
# Both DINOv2 and DINOv3 use the ImageNet constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Slicing and windowing (pure numpy -- see --self-test)
# --------------------------------------------------------------------------- #


def slice_positions(size, n_slices):
    """``n_slices`` evenly spaced indices into an axis of length ``size``.

    Bin midpoints, so ``n_slices=1`` is the middle plane and no position ever lands on
    the (empty) first or last plane of the volume.
    """
    import numpy as np

    if size < 1 or n_slices < 1:
        raise ValueError("size and n_slices must be >= 1")
    if n_slices > size:
        raise ValueError(f"asked for {n_slices} slices from an axis of length {size}")
    pos = ((np.arange(n_slices) + 0.5) * size / n_slices).astype(int)
    return np.clip(pos, 0, size - 1).tolist()


def slot_names(shape, axes, n_slices):
    """Stable per-slice labels, e.g. ``["axial:10", "axial:32", ...]``.

    The order here IS the order of the concatenated feature blocks, so it is saved with
    the embeddings and used by the scoring script to say where a feature came from.
    """
    names = []
    for axis_name in axes:
        for pos in slice_positions(shape[AXES[axis_name]], n_slices):
            names.append(f"{axis_name}:{pos}")
    return names


def volume_planes(volume, axes, n_slices):
    """Yield the 2-D planes of one volume in ``slot_names`` order."""
    import numpy as np

    for axis_name in axes:
        axis = AXES[axis_name]
        for pos in slice_positions(volume.shape[axis], n_slices):
            yield np.take(volume, pos, axis=axis)


def window_to_uint8(plane, window):
    """Map a plane onto 0-255 with ``window=(lo, hi)``; a degenerate window gives zeros."""
    import numpy as np

    lo, hi = window
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    scaled = (np.asarray(plane, dtype=np.float64) - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def plane_window(plane, percentiles):
    import numpy as np

    return tuple(float(v) for v in np.percentile(np.asarray(plane, dtype=np.float64), percentiles))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _generator_args(cli):
    """The ``args`` object ``build_synthetic_test_set`` reads the generator settings off.

    Either a run's ``settings.json`` (so the embedded distribution is the one that run
    trained on, and the numbers are comparable to its VQ-VAE panel) or this CLI's flags.
    Every field is read there with ``getattr(args, name, default)``, so the defaults for
    anything not exposed here come from that helper rather than being restated.
    """
    from eval.run_dci_synthetic import load_run_args

    if cli.run_dir:
        args = load_run_args(cli.run_dir)
        if cli.res is not None:
            args.synthetic_res = cli.res
            args.spatial_size = (cli.res,) * 3
        return args
    return argparse.Namespace(
        synthetic_mode="pseudo_mri",
        synthetic_res=cli.res if cli.res is not None else 64,
        synthetic_seed=cli.seed,
        synthetic_n_content=cli.n_content,
        synthetic_n_style=cli.n_style,
        synthetic_normalize=cli.normalize,
        synthetic_causal=not cli.no_causal,
        synthetic_causal_graph=cli.causal_graph,
        synthetic_causal_edge_prob=cli.causal_edge_prob,
        synthetic_causal_noise_scale=cli.causal_noise_scale,
        synthetic_causal_nonlinearity=cli.causal_nonlinearity,
        synthetic_content_prior=cli.content_prior,
        synthetic_content_squash=cli.content_squash,
        synthetic_clean_content=cli.clean_content,
        synthetic_identifiable_ventricle=cli.identifiable_ventricle,
        synthetic_style_scale=cli.style_scale,
        synthetic_content_scale=cli.content_scale,
    )


def build_dataset(cli):
    """Build the frozen synthetic test split and report what it actually is."""
    from eval.run_dci_synthetic import build_synthetic_test_set

    args = _generator_args(cli)
    dataset = build_synthetic_test_set(args, cli.num_samples, cache=not cli.no_cache, causal=True)
    inner = getattr(dataset, "_inner", dataset)
    settings = {
        key: value
        for key, value in vars(args).items()
        if key.startswith("synthetic_") and not key.startswith("synthetic_num_")
    }
    settings["synthetic_num_samples"] = len(dataset)
    return dataset, inner, settings


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #


def resolve_normalization(cli):
    """(mean, std, source) for the ViT input, preferring what the checkpoint declares."""
    if cli.image_mean and cli.image_std:
        return tuple(cli.image_mean), tuple(cli.image_std), "cli"
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(cli.model_id, local_files_only=cli.local_files_only)
        mean, std = getattr(processor, "image_mean", None), getattr(processor, "image_std", None)
        if mean and std:
            return tuple(float(v) for v in mean), tuple(float(v) for v in std), "preprocessor_config"
    except Exception as exc:  # noqa: BLE001
        # The constants are the only thing wanted from the processor, so a missing optional
        # dependency (the DINOv3 processors are torchvision-only) or an absent
        # preprocessor_config.json must not stop the run.
        logger.info("Could not read the checkpoint's image processor (%s); using ImageNet constants", exc)
    return IMAGENET_MEAN, IMAGENET_STD, "imagenet_default"


def load_encoder(cli):
    """Load the HF vision encoder, or a seeded random-init twin of the same architecture.

    The random-init weights ARE a measurement here -- they are the floor every trained
    number is read as a gap over -- so ``--model-seed`` fixes them, for the same reason
    ``eval.run_dci_synthetic.load_model_from_run_dir`` seeds its untrained twin.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(cli.model_id, local_files_only=cli.local_files_only)
    if cli.random_init:
        torch.manual_seed(cli.model_seed)
        model = AutoModel.from_config(config)
    else:
        try:
            model = AutoModel.from_pretrained(cli.model_id, local_files_only=cli.local_files_only)
        except Exception as exc:  # noqa: BLE001 - re-raised with the fix for the usual cause
            text = str(exc).lower()
            if any(token in text for token in ("gated", "401", "403", "authoriz", "authentic")):
                raise RuntimeError(
                    f"Could not download {cli.model_id!r}. DINOv3 repos are gated: accept the licence "
                    "on the model page, run `hf auth login` (older CLIs: `huggingface-cli login`), or "
                    "pass --model-id <local snapshot directory>."
                ) from exc
            raise
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[cli.dtype]
    return model.to(device=device, dtype=dtype).eval(), device, dtype, config


def patch_size_of(config):
    patch = getattr(config, "patch_size", None) or getattr(getattr(config, "vision_config", None), "patch_size", None)
    if not patch:
        raise ValueError("Could not read patch_size off the model config; pass --patch-size")
    return int(patch)


def token_prefix(seq_len, n_patches, config):
    """How many leading non-patch tokens the model emits (CLS plus any register tokens).

    Inferred from the sequence length rather than trusted from the config, because it is
    the number that decides whether ``mean`` pooling averages patch tokens or quietly
    averages four DINOv3 register tokens in with them. The config value, when there is
    one, is checked against it.
    """
    prefix = seq_len - n_patches
    if prefix < 1:
        raise ValueError(f"Model returned {seq_len} tokens for {n_patches} patches; cannot locate the patch tokens")
    declared = getattr(config, "num_register_tokens", None)
    if declared is not None and prefix != 1 + int(declared):
        logger.warning(
            "Token prefix inferred as %d but config declares 1 CLS + %d register tokens; using %d",
            prefix,
            int(declared),
            prefix,
        )
    return prefix


def pool_tokens(hidden, prefix, grid_hw, mode, grid_size):
    """(B, T, D) hidden states -> (B, D_out) features under the requested token pooling."""
    import torch
    import torch.nn.functional as F

    cls = hidden[:, 0]
    patches = hidden[:, prefix:]
    if mode == "cls":
        return cls
    if mode == "mean":
        return patches.mean(dim=1)
    if mode == "cls_mean":
        return torch.cat([cls, patches.mean(dim=1)], dim=-1)
    if mode == "grid":
        b, _, d = patches.shape
        maps = patches.transpose(1, 2).reshape(b, d, grid_hw[0], grid_hw[1])
        return F.adaptive_avg_pool2d(maps, grid_size).flatten(1)
    raise ValueError(f"unknown token pooling: {mode}")


def prepare_batch(planes, window, cli, mean, std):
    """Windowed 2-D planes -> a normalized (B, 3, S, S) float tensor.

    Preprocessing is done here rather than through the checkpoint's image processor so
    that it is the same in every environment (the DINOv3 processors are torchvision-only)
    and so the resize is ours: a center crop would cut the brain out of the plane.
    """
    import numpy as np
    import torch
    from PIL import Image

    size = cli.image_size
    batch = np.empty((len(planes), size, size, 3), dtype=np.float32)
    for i, plane in enumerate(planes):
        w = plane_window(plane, cli.window_pct) if cli.window == "per_slice" else window
        image = Image.fromarray(window_to_uint8(plane, w)).convert("RGB").resize((size, size), Image.BICUBIC)
        batch[i] = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
    return (tensor - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def estimate_window(dataset, cli):
    """One intensity window for the whole dataset, from a pilot of volumes.

    Keeps the per-sample intensity differences that style drives -- a per-slice window
    would map every slice onto the same range and delete them before the ViT sees them.
    """
    import numpy as np

    values = []
    for idx in range(min(cli.window_pilot, len(dataset))):
        images = dataset[idx]["image"]
        for view in cli.views:
            values.append(np.asarray(images[view - 1]).ravel())
    pooled = np.concatenate(values)
    if len(pooled) > 4_000_000:
        pooled = pooled[:: len(pooled) // 4_000_000 + 1]
    return plane_window(pooled, cli.window_pct)


def raw_voxel_features(images, grid):
    """Downsampled voxel intensities: the trivial baseline a ViT has to beat."""
    import torch.nn.functional as F

    return F.adaptive_avg_pool3d(images.float(), grid).flatten(1).cpu().numpy()


def extract(dataset, model, device, dtype, config, cli, window, mean, std):
    """Run every requested plane of every sample through the encoder.

    Returns ``(embeddings, latents, raw, slots)``: ``embeddings[view]`` is (N, slots, D)
    before aggregation, ``raw[view]`` the downsampled-voxel baseline, and ``latents`` the
    ground-truth factors, all in dataset order.
    """
    import numpy as np
    import torch

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cli.volume_batch, num_workers=cli.num_workers, shuffle=False
    )
    slots = slot_names(tuple(dataset[0]["image"][0].shape[-3:]), cli.axes, cli.slices)
    patch = cli.patch_size or patch_size_of(config)
    if cli.image_size % patch:
        raise ValueError(f"--image-size {cli.image_size} is not a multiple of the model's patch size {patch}")
    grid_hw = (cli.image_size // patch, cli.image_size // patch)

    embeddings, raw, latents, prefix, done = {}, {}, {}, None, 0
    for batch in loader:
        images = batch["image"]
        for key in ("z_content", "z_style_v1", "z_style_v2", "causal_adj"):
            if key in batch["gt_latents"]:
                latents.setdefault(key, []).append(np.asarray(batch["gt_latents"][key]))
        for view in cli.views:
            volumes = images[view - 1]
            if cli.raw_grid:
                raw.setdefault(view, []).append(raw_voxel_features(volumes, cli.raw_grid))
            planes = [
                plane for volume in volumes.squeeze(1).numpy() for plane in volume_planes(volume, cli.axes, cli.slices)
            ]
            features = []
            for start in range(0, len(planes), cli.batch_size):
                pixels = prepare_batch(planes[start : start + cli.batch_size], window, cli, mean, std)
                with torch.no_grad():
                    hidden = model(pixel_values=pixels.to(device=device, dtype=dtype)).last_hidden_state
                if prefix is None:
                    prefix = token_prefix(hidden.shape[1], grid_hw[0] * grid_hw[1], config)
                    logger.info("Token layout: %d prefix token(s) + %d patch tokens", prefix, grid_hw[0] * grid_hw[1])
                features.append(
                    pool_tokens(hidden.float(), prefix, grid_hw, cli.token_pool, cli.grid_size).cpu().numpy()
                )
            stacked = np.concatenate(features).reshape(len(volumes), len(slots), -1)
            embeddings.setdefault(view, []).append(stacked)
        done += len(images[0])
        logger.info("Embedded %d/%d samples", done, len(dataset))

    return (
        {view: np.concatenate(parts) for view, parts in embeddings.items()},
        {key: np.concatenate(parts) for key, parts in latents.items()},
        {view: np.concatenate(parts) for view, parts in raw.items()},
        slots,
    )


def aggregate(per_slot, how):
    """(N, slots, D) -> (N, F): ``concat`` keeps which plane a feature came from."""
    return per_slot.mean(axis=1) if how == "mean" else per_slot.reshape(len(per_slot), -1)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - provenance is nice to have, never worth failing on
        return None


def save(path, embeddings, latents, raw, slots, meta):
    import numpy as np

    arrays = {"z_content": latents["z_content"].astype(np.float32), "meta": json.dumps(meta, indent=2)}
    for view, features in embeddings.items():
        arrays[f"emb_view{view}"] = features.astype(np.float32)
    for view, features in raw.items():
        arrays[f"raw_view{view}"] = features.astype(np.float32)
    for key in ("z_style_v1", "z_style_v2", "causal_adj"):
        if key in latents:
            arrays[key] = latents[key].astype(np.float32)
    if "causal_adj" in latents:
        # One adjacency per sample comes out of the collate; they are all the same SCM.
        arrays["causal_adj"] = arrays["causal_adj"][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return arrays


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _self_test():
    """Slicing, windowing and token pooling on planted arrays -- no model, no GPU."""
    import numpy as np

    assert slice_positions(64, 1) == [32]
    assert slice_positions(64, 3) == [10, 32, 53]
    assert slice_positions(4, 4) == [0, 1, 2, 3]
    for bad in ((0, 1), (8, 0), (4, 5)):
        try:
            slice_positions(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"slice_positions{bad} should have been rejected")

    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    planes = list(volume_planes(volume, ["sagittal", "axial"], 1))
    assert planes[0].shape == (3, 4) and planes[1].shape == (2, 3)
    assert np.array_equal(planes[0], volume[1]) and np.array_equal(planes[1], volume[:, :, 2])
    assert slot_names(volume.shape, ["sagittal", "axial"], 1) == ["sagittal:1", "axial:2"]

    # A dataset window keeps two volumes that differ only by a gain apart; a per-slice
    # window maps both onto 0-255 and deletes the difference. That IS the style factor.
    dim, bright = np.linspace(0, 1, 64).reshape(8, 8), np.linspace(0, 2, 64).reshape(8, 8)
    shared = plane_window(np.concatenate([dim.ravel(), bright.ravel()]), (1, 99))
    assert window_to_uint8(dim, shared).max() < window_to_uint8(bright, shared).max()
    assert (
        window_to_uint8(dim, plane_window(dim, (1, 99))).max()
        == window_to_uint8(bright, plane_window(bright, (1, 99))).max()
    )
    assert window_to_uint8(np.zeros((4, 4)), (1.0, 1.0)).sum() == 0
    print("  self-test: slicing, slot naming and windowing OK")

    try:
        import torch
    except ImportError:
        print("  self-test PASSED (torch absent; token pooling not exercised)")
        return
    hidden = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    prefix = token_prefix(7, 4, argparse.Namespace(num_register_tokens=2))
    assert prefix == 3
    assert torch.equal(pool_tokens(hidden, prefix, (2, 2), "cls", 1), hidden[:, 0])
    assert torch.equal(pool_tokens(hidden, prefix, (2, 2), "mean", 1), hidden[:, 3:].mean(dim=1))
    assert pool_tokens(hidden, prefix, (2, 2), "cls_mean", 1).shape == (2, 6)
    # grid at the patch grid's own size keeps every token, laid out channel-major.
    assert torch.allclose(pool_tokens(hidden, prefix, (2, 2), "grid", 2), hidden[:, 3:].transpose(1, 2).flatten(1))
    assert torch.allclose(pool_tokens(hidden, prefix, (2, 2), "grid", 1), hidden[:, 3:].mean(dim=1))
    try:
        token_prefix(4, 4, argparse.Namespace())
    except ValueError:
        pass
    else:
        raise AssertionError("token_prefix should reject a sequence with no prefix token")
    print("  self-test: token prefix inference and pooling OK")
    print("  self-test PASSED")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="Where to write the .npz (required unless --self-test)")
    parser.add_argument("--self-test", action="store_true", help="Run the numpy self-test and exit")

    model = parser.add_argument_group("encoder")
    model.add_argument("--model-id", default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="HF id or local path")
    model.add_argument("--random-init", action="store_true", help="Untrained twin of the same architecture (floor)")
    model.add_argument("--model-seed", type=int, default=0, help="Seeds --random-init weights")
    model.add_argument("--local-files-only", action="store_true", help="Never reach for the Hub")
    model.add_argument("--device", help="cpu, cuda, cuda:0, ...; default: CUDA when available")
    model.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    model.add_argument("--image-size", type=int, default=224, help="Square ViT input; must divide by the patch size")
    model.add_argument("--patch-size", type=int, help="Override the patch size read off the model config")
    model.add_argument("--token-pool", default="cls_mean", choices=["cls", "mean", "cls_mean", "grid"])
    model.add_argument("--grid-size", type=int, default=2, help="Patch-map pooling grid for --token-pool grid")
    model.add_argument("--image-mean", type=float, nargs=3, help="Override the checkpoint's normalization mean")
    model.add_argument("--image-std", type=float, nargs=3, help="Override the checkpoint's normalization std")

    view = parser.add_argument_group("slicing")
    view.add_argument("--axes", default="axial,coronal,sagittal", help="Comma-separated: axial, coronal, sagittal")
    view.add_argument("--slices", type=int, default=3, help="Evenly spaced planes per axis (1 = middle plane)")
    view.add_argument("--slice-agg", default="concat", choices=["concat", "mean"])
    view.add_argument("--views", type=int, nargs="+", default=[1, 2], choices=[1, 2], help="1=T1, 2=FLAIR")
    view.add_argument("--window", default="dataset", choices=["dataset", "per_slice"])
    view.add_argument("--window-pct", type=float, nargs=2, default=[1.0, 99.0], help="Window percentiles")
    view.add_argument("--window-pilot", type=int, default=32, help="Volumes used to estimate the dataset window")
    view.add_argument("--raw-grid", type=int, default=8, help="Voxel-baseline pooling grid; 0 disables it")

    data = parser.add_argument_group("generator (ignored where --run-dir supplies them)")
    data.add_argument("--run-dir", help="Take the generator settings from this run's settings.json")
    data.add_argument("--num-samples", type=int, default=500)
    data.add_argument("--res", type=int, help="Cubic volume resolution (default: 64, or the run's)")
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--n-content", type=int, default=9)
    data.add_argument("--n-style", type=int, default=3)
    data.add_argument("--no-causal", action="store_true", help="i.i.d. factors; there is then no graph to recover")
    data.add_argument("--causal-graph", default="chain", choices=["chain", "full", "random"])
    data.add_argument("--causal-edge-prob", type=float, default=0.5)
    data.add_argument("--causal-noise-scale", type=float, default=0.4)
    data.add_argument("--causal-nonlinearity", default="leaky_relu", choices=["leaky_relu", "tanh"])
    data.add_argument("--content-prior", default="normal", choices=["normal", "uniform"])
    data.add_argument("--content-squash", default="auto", choices=["auto", "clamp", "tanh", "none"])
    data.add_argument("--normalize", default="fixed_reference", choices=["per_sample", "shared", "fixed_reference"])
    data.add_argument("--clean-content", action="store_true")
    data.add_argument("--identifiable-ventricle", action="store_true")
    data.add_argument("--style-scale", type=float, default=1.0)
    data.add_argument("--content-scale", type=float, default=1.0)
    data.add_argument("--no-cache", action="store_true", help="Re-render volumes instead of keeping them in RAM")

    run = parser.add_argument_group("execution")
    run.add_argument("--batch-size", type=int, default=64, help="Planes per encoder forward pass")
    run.add_argument("--volume-batch", type=int, default=4, help="Volumes per data-loader batch")
    run.add_argument("--num-workers", type=int, default=0)

    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if cli.self_test:
        _self_test()
        return 0
    if cli.out is None:
        parser.error("--out is required (or pass --self-test)")
    cli.axes = [name.strip() for name in cli.axes.split(",") if name.strip()]
    if not cli.axes or any(name not in AXES for name in cli.axes):
        parser.error(f"--axes must be a comma-separated subset of {sorted(AXES)}")
    cli.views = sorted(dict.fromkeys(cli.views))
    if cli.slices < 1 or cli.batch_size < 1 or cli.volume_batch < 1 or cli.num_samples < 2:
        parser.error("Require --slices >= 1, --batch-size >= 1, --volume-batch >= 1, --num-samples >= 2")
    if not 0 <= cli.window_pct[0] < cli.window_pct[1] <= 100:
        parser.error("--window-pct must be two increasing percentiles in [0, 100]")
    if (cli.image_mean is None) != (cli.image_std is None):
        parser.error("--image-mean and --image-std must be given together")
    if cli.window == "per_slice":
        logger.warning(
            "--window per_slice rescales every plane onto the same range, which is the affine map "
            "style applies (lut = base*gain + bias). Style recovery from these embeddings is then "
            "bounded by the windowing, not by the encoder. Use --window dataset to keep it."
        )

    started = time.time()
    dataset, inner, settings = build_dataset(cli)
    if settings.get("synthetic_normalize") in ("per_sample", "shared") and settings.get("synthetic_n_style", 0):
        logger.warning(
            "The generator's own normalization is %r, which z-scores each volume and removes the "
            "style gain/bias before this script ever sees it. --normalize fixed_reference keeps it.",
            settings["synthetic_normalize"],
        )
    encoder, device, dtype, config = load_encoder(cli)
    mean, std, norm_source = resolve_normalization(cli)
    logger.info("Input normalization from %s: mean=%s std=%s", norm_source, mean, std)
    window = (0.0, 0.0) if cli.window == "per_slice" else estimate_window(dataset, cli)
    if cli.window == "dataset":
        logger.info("Dataset intensity window at percentiles %s: %s", cli.window_pct, window)

    per_slot, latents, raw, slots = extract(dataset, encoder, device, dtype, config, cli, window, mean, std)
    embeddings = {view: aggregate(features, cli.slice_agg) for view, features in per_slot.items()}

    from eval.causal_factor_diagnostics import FACTOR_NAMES
    from eval.dci import STYLE_FACTOR_NAMES

    n_content = latents["z_content"].shape[1]
    n_style = latents["z_style_v1"].shape[1] if "z_style_v1" in latents else 0
    meta = dict(
        model_id=cli.model_id,
        random_init=cli.random_init,
        model_seed=cli.model_seed if cli.random_init else None,
        architecture=type(encoder).__name__,
        dtype=cli.dtype,
        image_size=cli.image_size,
        patch_size=cli.patch_size or patch_size_of(config),
        token_pool=cli.token_pool,
        grid_size=cli.grid_size if cli.token_pool == "grid" else None,
        image_mean=list(mean),
        image_std=list(std),
        normalization_source=norm_source,
        axes=cli.axes,
        slices=cli.slices,
        slice_agg=cli.slice_agg,
        slots=slots,
        views=cli.views,
        window=cli.window,
        window_pct=cli.window_pct,
        window_values=list(window) if cli.window == "dataset" else None,
        raw_grid=cli.raw_grid,
        embedding_dim={f"view{view}": int(features.shape[1]) for view, features in embeddings.items()},
        num_samples=int(len(dataset)),
        causal=bool(getattr(inner, "scm", None)),
        content_factor_names=[FACTOR_NAMES[d] if d < len(FACTOR_NAMES) else f"d{d}" for d in range(n_content)],
        style_factor_names=[STYLE_FACTOR_NAMES[d] if d < len(STYLE_FACTOR_NAMES) else f"s{d}" for d in range(n_style)],
        generator=settings,
        run_dir=cli.run_dir,
        git_sha=git_sha(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=round(time.time() - started, 1),
    )
    arrays = save(cli.out, embeddings, latents, raw, slots, meta)
    print(f"\nSaved {cli.out.resolve()}")
    for key, value in sorted(arrays.items()):
        if key != "meta":
            print(f"  {key:<14} {value.shape}")
    print(f"  {len(slots)} slice(s) per view ({', '.join(slots)}), aggregated by {cli.slice_agg}")
    print(f"\nScore it:\n  python -m eval.dinov3_identifiability --embeddings {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
