"""How far each cell of an encoder-only run reaches, and how many content channels that reach asks for.

The encoder counterpart of probe/jacobian_spread.py, which reads a VQ-VAE decoder and cannot run on a model without
one. Three numbers for one run, no labels involved:

  hard reach       the input voxels a cell can depend on, from the architecture (models.conv_footprint), plus a
                   scan for operations no receptive field bounds, such as GroupNorm's whole-volume statistics
                   (probe.receptive_field.scan_for_global_ops). The padding-free cells are --interior-only's.
  effective reach  the box around a cell holding a given share of the squared gradient of its content code with respect
                   to the input, by Hutchinson draws. GroupNorm runs repeat it with the statistics frozen, so the
                   gap between the two arms is what the whole-volume path costs. Residual blocks count fully in
                   the hard reach however small their learned weight, so the two reaches can differ a lot.
  channel estimate content channels = channels per site x sites in reach + global factors (paper/theory-framing.md,
                   9.3), counted on the run's generator at each reach. Field sites a box can touch are counted once
                   per latent dimension, so this is an upper bound: a global factor a window cannot see adds
                   nothing there, and the lesion sphere is counted by its few numbers, not per cell.

    python -m eval.encoder.encoder_receptive_field --run-dir results/encoder/runs/<run> --out reach.json
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset
from models.conv_footprint import conv_footprint
from probe.jacobian_spread import freeze_norm_statistics
from probe.receptive_field import scan_for_global_ops

QUANTILES = (0.5, 0.9, 0.95, 0.99)
ANATOMY_FACTORS = (0, 1, 5, 6, 7, 8)  # z_content: brain, ventricle, thickness, temporal, asymmetry, sulcal
FIELDS = ("synthetic_n_deformation_grid", "synthetic_n_fissure_grid")


def sites_in_reach(reach, res, grid):
    """Latent sites of a ``grid``^3 field that a box ``reach`` voxels wide can depend on.

    A site sits every res / grid voxels and trilinear upsampling blends two neighbours along each axis, hence the +1.
    """
    return min(grid, math.ceil(reach * grid / res) + 1) ** 3


def shared_dimensions(cfg, reach):
    """Shared latent dimensions a cell ``reach`` voxels wide can depend on, on the run's pseudo_mri generator.

    z_content's active scalars, the lesion placement numbers the burden target draws outside z_content, and the two
    shared fields, which --synthetic-clean-content multiplies by zero. None on any other generator.
    """
    if cfg.get("dataset_name", "synthetic") != "synthetic" or cfg.get("synthetic_mode", "pseudo_mri") != "pseudo_mri":
        return None
    scale = cfg.get("synthetic_content_amp_scale")
    scalars = sum(1 for factor in ANATOMY_FACTORS if scale is None or scale[factor] > 0)
    burden = cfg.get("synthetic_lesion_target", "position") == "burden"
    scalars += 1 if burden else 3
    placement = 3 * cfg.get("synthetic_lesion_count", 4) if burden else 0
    fields = 0
    if not cfg.get("synthetic_clean_content", False):
        fields = sum(sites_in_reach(reach, cfg["res"], cfg.get(key, default)) for key, default in zip(FIELDS, (4, 8)))
    return dict(scalars=scalars, lesion_placement=placement, field_sites=fields, total=scalars + placement + fields)


def probe_cells(model, mask, count, generator):
    """Up to ``count`` map cells of one subject, half brain or more, reading no padding when any such cell exists."""
    stride = model.backbone_stride
    shape = tuple(size // stride for size in mask.shape[2:])
    brain = F.adaptive_avg_pool3d(mask.float(), shape)[0, 0].cpu() >= 0.5
    inside = brain & model.interior_positions(mask.shape[2:], shape).reshape(shape)
    cells = (inside if bool(inside.any()) else brain).nonzero()
    return cells[torch.randperm(len(cells), generator=generator)[:count]]


def cell_energy(model, encoder, image, cell, draws):
    """Squared gradient of one cell's content code with respect to every input voxel, (D, H, W)."""
    image = image.clone().requires_grad_(True)
    backbone = encoder(image)
    d, h, w = cell.tolist()
    if model.readout_type == "mlp":
        code = model.patch_readout(backbone[0, :, d, h, w])
    else:
        code = model.to_encoding(backbone)[0, :, d, h, w]
    code = code[: model.content_channels]
    energy = torch.zeros_like(image[0, 0])
    for draw in range(draws):
        signs = torch.randint(0, 2, code.shape, device=code.device, dtype=code.dtype) * 2 - 1
        gradient = torch.autograd.grad((signs * code).sum(), image, retain_graph=draw < draws - 1)[0]
        energy += gradient[0, 0].detach().square()
    return energy / draws


def reach_of(energy, cell, footprint):
    """Half-widths in voxels of the boxes around a cell holding each of QUANTILES, and the share outside its hard box."""
    stride, low, high = footprint
    centre = torch.as_tensor([stride * j + (low + high) / 2 for j in cell.tolist()], dtype=torch.float32)
    axes = [torch.arange(size, dtype=torch.float32) - c for size, c in zip(energy.shape, centre)]
    distance = torch.stack(torch.meshgrid(*axes, indexing="ij")).abs().amax(0).flatten().to(energy.device)
    order = torch.argsort(distance)
    share = torch.cumsum(energy.flatten()[order].double(), 0)
    share = share / share[-1]
    index = torch.searchsorted(share, torch.as_tensor(QUANTILES, dtype=torch.float64, device=share.device))
    half_widths = distance[order][index.clamp_max(len(share) - 1)].tolist()
    outside = max(0.0, 1 - float(energy.flatten()[distance <= (high - low) / 2].sum() / energy.sum()))
    return half_widths, outside


def measure(model, dataset, cli, footprint):
    """Effective reach per view and arm, averaged over subjects and cells."""
    frozen_arm = model.normalization == "group"
    arms = ("as built", "norm frozen") if frozen_arm else ("as built",)
    views = [("t1", model.encoder)] + ([("flair", model.encoder_v1)] if model.encoder_v1 is not None else [])
    generator = torch.Generator().manual_seed(cli.seed)
    results = {}
    for v, (view, encoder) in enumerate(views):
        for arm in arms:
            widths, outside = [], []
            for subject in range(cli.subjects):
                image = dataset[subject]["image"][v].as_subclass(torch.Tensor)[None].to(cli.device)
                net = encoder
                if arm == "norm frozen":
                    net = freeze_norm_statistics(encoder, lambda frozen, x=image: frozen(x))
                for cell in probe_cells(model, image != 0, cli.cells, generator):
                    half_widths, beyond = reach_of(cell_energy(model, net, image, cell, cli.draws), cell, footprint)
                    widths.append(half_widths)
                    outside.append(beyond)
            results.setdefault(view, {})[arm] = {
                "half_width_voxels": dict(zip(map(str, QUANTILES), np.mean(widths, axis=0).tolist())),
                "outside_hard_box": float(np.mean(outside)),
                "cells": len(widths),
            }
    return results


def report(model, cfg, effective):
    """Print the run's reach and channel estimate; returns the JSON payload."""
    stride, low, high = conv_footprint(cfg["downscale_factor"], cfg["nb_res_layers"])
    width = high - low + 1
    size = cfg["res"]
    cells = int(model.interior_positions((size,) * 3, (size // stride,) * 3).sum())
    scan = scan_for_global_ops(model.encoder)
    alphas = [float(p) for name, p in model.named_parameters() if name.endswith("alpha")]
    print(
        f"hard reach: {width} voxels per axis, map stride {stride}; {cells}/{(size // stride) ** 3} cells read no padding"
    )
    if alphas:
        print("  residual block weights: " + " ".join(f"{a:+.3f}" for a in alphas))
    if scan["ok"]:
        print("  no operation outside the receptive field: every cell depends only on its box")
    else:
        kinds = sorted({row["class"] for row in scan["offenders"]})
        print(f"  NOT bounded by it: {len(scan['offenders'])} layers ({', '.join(kinds)}) reach every position")
    print("effective reach: half-width in voxels of the box holding each share of a cell's squared gradient")
    print(f"  {'view':<7s}{'arm':<13s}" + "".join(f"{q:>7.0%}" for q in QUANTILES) + f"{'outside hard box':>18s}")
    typical = []
    for view, arms in effective.items():
        for arm, row in arms.items():
            values = list(row["half_width_voxels"].values())
            line = f"  {view:<7s}{arm:<13s}" + "".join(f"{value:>7.1f}" for value in values)
            print(line + f"{row['outside_hard_box']:>18.4f}")
            if arm == "as built":
                typical.append(2 * row["half_width_voxels"]["0.95"])
    effective_width = float(np.mean(typical))
    print(f"content channels: the run has {cfg['content_channels']}")
    estimates = {"hard": shared_dimensions(cfg, width), "effective": shared_dimensions(cfg, effective_width)}
    if estimates["hard"] is None:
        print("  no generator to count on; a real dataset's channels per site are unknown")
    for name, reach in (("hard", width), ("effective", effective_width)):
        row = estimates[name]
        if row is not None:
            print(
                f"  {name:<10s}{reach:>5.0f} voxels: {row['total']} = {row['scalars']} scalars + "
                f"{row['lesion_placement']} lesion placement + {row['field_sites']} field sites"
            )
    return {
        "hard": dict(
            width_voxels=width, stride=stride, low=low, high=high, interior_cells=cells, alphas=alphas, scan=scan
        ),
        "effective": effective,
        "effective_width_voxels": effective_width,
        "content_channels": cfg["content_channels"],
        "shared_dimensions": estimates,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Run directory holding settings.json and the checkpoint")
    p.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename inside --run-dir")
    p.add_argument("--random-init", action="store_true", help="Measure the untrained model instead")
    p.add_argument("--subjects", type=int, default=2, help="Validation subjects per view")
    p.add_argument("--cells", type=int, default=7, help="Cells per subject")
    p.add_argument("--draws", type=int, default=3, help="Hutchinson draws per cell")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="JSON report path")
    cli = p.parse_args(argv)
    cfg = load_settings(cli.run_dir)
    if cfg.get("encoder_architecture", "conv") != "conv":
        p.error("Only the conv backbone has a computed receptive field")
    torch.set_num_threads(cfg.get("cpu_threads") or torch.get_num_threads())
    state = None
    if not cli.random_init:
        state = torch.load(os.path.join(cli.run_dir, cli.checkpoint), map_location="cpu", weights_only=False)
    model = build_model(cfg, cli.device, state)
    footprint = conv_footprint(cfg["downscale_factor"], cfg["nb_res_layers"])
    payload = report(model, cfg, measure(model, make_dataset(cfg, cli.subjects), cli, footprint))
    if cli.out is not None:
        with open(cli.out, "w") as fp:
            json.dump(payload, fp, indent=2)
    return payload


if __name__ == "__main__":
    main()
