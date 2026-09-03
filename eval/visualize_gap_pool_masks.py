#!/usr/bin/env python
"""Render learned ``--bt-gap-pool per_channel`` masks from a checkpoint.

Example:
    python -m eval.visualize_gap_pool_masks \
        --checkpoint results/synthetic/RUN/vqvae_model.pt --grid 8 8 8

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


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise per-channel Barlow-Twins GAP-pool masks.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--grid", required=True, type=int, nargs=3, metavar=("D", "H", "W"))
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--channels", type=int, nargs="*", help="Channels to show (default: most concentrated).")
    p.add_argument("--n-channels", type=int, default=12, help="Number to show when --channels is omitted.")
    p.add_argument("--output", type=Path, help="Output PNG (default: beside checkpoint).")
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
    entropy = -(masks.reshape(len(masks), -1) * np.log(masks.reshape(len(masks), -1) + 1e-12)).sum(1)
    if args.channels is None:
        channels = np.argsort(entropy)[: args.n_channels].tolist()  # most concentrated first
    else:
        channels = args.channels
    if not channels or min(channels) < 0 or max(channels) >= len(masks):
        raise ValueError(f"Choose channel indices in [0, {len(masks) - 1}].")

    # These are marginals, so their maximum can exceed any individual voxel mass.
    vmax = max(float(masks[channel].sum(axis=axis).max()) for channel in channels for axis in range(3))
    fig, axes = plt.subplots(len(channels), 3, figsize=(9, 2.7 * len(channels)), squeeze=False)
    labels = ("sum over D", "sum over H", "sum over W")
    for row, channel in enumerate(channels):
        mask = masks[channel]
        marginals = (mask.sum(0), mask.sum(1), mask.sum(2))
        peak = np.unravel_index(mask.argmax(), mask.shape)
        for col, (image, label) in enumerate(zip(marginals, labels)):
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
