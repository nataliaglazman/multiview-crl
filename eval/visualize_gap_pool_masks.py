#!/usr/bin/env python
"""Render learned ``--bt-gap-pool per_channel`` masks from a checkpoint.

Example:
    python -m eval.visualize_gap_pool_masks \
        --checkpoint results/synthetic/RUN/vqvae_model.pt --grid 8 8 8

For a synthetic run, activations are extracted automatically. Add
``--save-activations activations.pt`` to cache them. A pre-extracted tensor (for
example ``(views, samples, C, P)``) can instead be supplied explicitly:
    python -m eval.visualize_gap_pool_masks --checkpoint ... --grid 8 8 8 \
        --activations activations.pt --activation-key hz

The script reads the saved pooler logits directly, so it does not need to build the
model or dataset.  Each row shows the mask's probability mass marginalised along
one anatomical axis; colours are comparable across all displayed channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return data.get("encoders", data)


def _pool_logits(state: dict[str, torch.Tensor], level: int) -> torch.Tensor:
    suffix = f"_bt_gap_pools.L{level}.logits"
    matches = [(key, value) for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        found = [key for key in state if "_bt_gap_pools" in key and key.endswith(".logits")]
        raise KeyError(f"Expected one key ending in {suffix!r}; found pool logits: {found or 'none'}")
    return matches[0][1].float()


def _activation_distribution(value, channels: int, grid: tuple[int, int, int], coarsen: int) -> np.ndarray:
    """Return a per-channel spatial distribution from saved encoder activations.

    Accepted layouts are ``(C, P)``, ``(N, C, P)``, ``(V, N, C, P)``,
    ``(C, D, H, W)``, or ``(N, C, D, H, W)``.  Leading sample/view axes are
    averaged after taking absolute values.  Fine-grid tensors are block-averaged
    to the pooler's coarse grid before being normalised per channel.
    """
    x = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float64)
    fine_p = int(np.prod(grid))
    coarse_grid = tuple(axis // coarsen for axis in grid)
    coarse_p = int(np.prod(coarse_grid))
    if x.ndim >= 2 and x.shape[-1] in (fine_p, coarse_p):
        if x.shape[-2] != channels:
            raise ValueError(f"Expected {channels} channels at axis -2, got activation shape {x.shape}.")
        x = np.abs(x).mean(axis=tuple(range(x.ndim - 2)))
        x = x.reshape(channels, *(grid if x.shape[-1] == fine_p else coarse_grid))
    elif x.ndim in (4, 5) and x.shape[-4] == channels:
        x = np.abs(x).mean(axis=0) if x.ndim == 5 else np.abs(x)
        if tuple(x.shape[1:]) != grid:
            raise ValueError(f"Expected activation spatial grid {grid}, got {tuple(x.shape[1:])}.")
    else:
        raise ValueError("Unsupported activation shape. See --help for accepted layouts.")
    if tuple(x.shape[1:]) == grid and coarsen > 1:
        d, h, w = coarse_grid
        x = x.reshape(channels, d, coarsen, h, coarsen, w, coarsen).mean(axis=(2, 4, 6))
    x = x.reshape(channels, coarse_p)
    return x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)


def _load_activation_distribution(
    path: Path, key: str | None, channels: int, grid: tuple[int, int, int], coarsen: int
) -> np.ndarray:
    loaded = np.load(path) if path.suffix == ".npz" else torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if key is None:
            if len(loaded.files) != 1:
                raise ValueError(f"{path} contains {loaded.files}; pass --activation-key.")
            value = loaded[loaded.files[0]]
        else:
            value = loaded[key]
    elif isinstance(loaded, dict):
        if key is None:
            raise ValueError("Activation .pt is a dictionary; pass --activation-key.")
        value = loaded[key]
    else:
        value = loaded
    return _activation_distribution(value, channels, grid, coarsen)


def _extract_synthetic_activations(
    run_dir: Path, checkpoint: Path, level: int, grid: tuple[int, int, int], samples: int, batch: int
):
    """Extract patch activations from the run's own synthetic evaluation distribution."""
    from eval.bt_term_balance import _as_views
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    run_args = load_run_args(str(run_dir))
    dataset = build_synthetic_test_set(run_args, samples, causal=bool(getattr(run_args, "synthetic_causal", False)))
    model, _args, device = load_model_from_run_dir(str(run_dir), str(checkpoint))
    level_data, _gt, _s1, _s2 = _extract_synthetic_representations(model, dataset, device, batch, 0, pooling=grid)
    c1, c2 = level_data[level][_CONTENT], level_data[level][_CONTENT_V2]
    if c1 is None or c2 is None:
        raise ValueError("This run did not yield two content-view activation maps.")
    return _as_views(c1, c2, int(np.prod(grid)))


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise per-channel Barlow-Twins GAP-pool masks.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--grid", required=True, type=int, nargs=3, metavar=("D", "H", "W"))
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--channels", type=int, nargs="*", help="Channels to show (default: most concentrated).")
    p.add_argument("--n-channels", type=int, default=12, help="Number to show when --channels is omitted.")
    p.add_argument("--output", type=Path, help="Output PNG (default: beside checkpoint).")
    p.add_argument("--activations", type=Path, help="Optional .pt or .npz tensor of encoder activations to compare.")
    p.add_argument("--activation-key", help="Key for a dictionary/.npz activation file.")
    p.add_argument("--run-dir", type=Path, help="Synthetic run directory; defaults to the checkpoint's parent.")
    p.add_argument("--num-samples", type=int, default=200, help="Synthetic subjects used for automatic extraction.")
    p.add_argument("--encode-batch", type=int, default=32, help="Encoder batch size for automatic extraction.")
    p.add_argument("--save-activations", type=Path, help="Save extracted activations as {'hz': tensor} for reuse.")
    args = p.parse_args()

    logits = _pool_logits(_state_dict(args.checkpoint), args.level)
    if logits.ndim != 2:
        raise ValueError(f"Expected (channels, regions) logits, got {tuple(logits.shape)}")
    grid = tuple(args.grid)
    fine_positions = int(np.prod(grid))
    groups = logits.shape[1]
    ratio = fine_positions // groups
    coarsen = round(ratio ** (1 / 3))
    if groups * coarsen**3 != fine_positions or any(axis % coarsen for axis in grid):
        raise ValueError(
            f"Grid {grid} has {fine_positions} positions, incompatible with {groups} saved regions. "
            "Pass the patch grid used for training."
        )

    coarse_grid = tuple(axis // coarsen for axis in grid)
    masks = torch.softmax(logits, dim=1).reshape(logits.shape[0], *coarse_grid).numpy()
    activations = None
    if args.activations:
        activations = _load_activation_distribution(args.activations, args.activation_key, len(masks), grid, coarsen)
        activations = activations.reshape(len(masks), *coarse_grid)
    else:
        run_dir = args.run_dir or args.checkpoint.parent
        hz = _extract_synthetic_activations(
            run_dir, args.checkpoint, args.level, grid, args.num_samples, args.encode_batch
        )
        if args.save_activations:
            torch.save({"hz": hz}, args.save_activations)
            print(f"Saved extracted activations to {args.save_activations}")
        activations = _activation_distribution(hz, len(masks), grid, coarsen).reshape(len(masks), *coarse_grid)
    entropy = -(masks.reshape(len(masks), -1) * np.log(masks.reshape(len(masks), -1) + 1e-12)).sum(1)
    if args.channels is None:
        channels = np.argsort(entropy)[: args.n_channels].tolist()  # most concentrated first
    else:
        channels = args.channels
    if not channels or min(channels) < 0 or max(channels) >= len(masks):
        raise ValueError(f"Choose channel indices in [0, {len(masks) - 1}].")

    # These are marginals, so their maximum can exceed any individual voxel mass.
    panels = 6 if activations is not None else 3
    fig, axes = plt.subplots(len(channels), panels, figsize=(3 * panels, 2.7 * len(channels)), squeeze=False)
    labels = ("sum over D", "sum over H", "sum over W")
    for row, channel in enumerate(channels):
        mask = masks[channel]
        marginals = (mask.sum(0), mask.sum(1), mask.sum(2))
        peak = np.unravel_index(mask.argmax(), mask.shape)
        if activations is not None:
            act = activations[channel]
            marginals = tuple(pair for axis in range(3) for pair in (mask.sum(axis), act.sum(axis)))
            titles = tuple(item for label in labels for item in (f"mask: {label}", f"activation: {label}"))
        else:
            titles = labels
        vmax = max(float(image.max()) for image in marginals)
        for col, (image, label) in enumerate(zip(marginals, titles)):
            axes[row, col].imshow(image, origin="lower", cmap="magma", vmin=0, vmax=vmax)
            axes[row, col].set_title(label if row == 0 else "")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(
            f"ch {channel}\\nH/log G={entropy[channel] / np.log(groups):.2f}\\npeak={tuple(int(x) for x in peak)}"
        )
    fig.suptitle(f"GAP-pool masks: level {args.level}, coarse grid {coarse_grid} (coarsen={coarsen})", y=1.01)
    fig.tight_layout()
    output = args.output or args.checkpoint.with_name(f"gap_pool_masks_L{args.level}.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved {output} (channels: {channels}; logits key has {groups} coarse regions).")


if __name__ == "__main__":
    main()
