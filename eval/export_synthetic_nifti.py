"""Write a sample of synthetic training volumes to NIfTI for inspection.

Renders the same ``SyntheticBrainDataset`` the training loop consumes and saves
each sample's two views as ``.nii.gz``, so the generator's output can be opened
in any MRI viewer.

The generator settings are read from an experiment YAML through
``scripts/launch.py``'s own ``resolve_config`` (defaults <- _base_ <- experiment),
so the export tracks the experiment instead of carrying a second copy of its
settings; ``--set key=value`` overrides individual keys exactly as at launch.

Volumes are written AFTER ``--synthetic-normalize`` is applied, i.e. the arrays
the encoder actually sees. ``--raw`` additionally writes the pre-normalization
renderings, which carry the generator's native intensity contrast.

Usage:
    python -m eval.export_synthetic_nifti --config experiments/synthetic_causal.yaml \
        --num 10 --out /tmp/synthetic_nifti
"""

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np

from eval.causal_factor_diagnostics import FACTOR_NAMES, factor_name

# Style components the pseudo_mri renderer consumes, in order (see
# PseudoMRIRenderer.render_modality: gain and bias form the intensity LUT,
# the third is the noise sigma).
STYLE_NAMES = ("gain", "bias", "noise_sigma")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "synthetic_causal.yaml"


def _launch_module():
    """Import scripts/launch.py by path — ``scripts`` is not a package."""
    spec = importlib.util.spec_from_file_location("_launch", REPO_ROOT / "scripts" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="synthetic_nifti", help="Output directory.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment YAML the settings come from.")
    p.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE", help="Override config keys.")
    p.add_argument("--num", type=int, default=10, help="Number of samples to export.")
    p.add_argument("--start", type=int, default=0, help="First dataset index to export.")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--raw", action="store_true", help="Also write the pre-normalization volumes.")
    p.add_argument("--no-mask", dest="mask", action="store_false", help="Skip the brain-mask volumes.")
    p.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Diagonal of the NIfTI affine. 1.0 is the repo's identity convention; "
        "~2.8 makes a 64^3 volume span a brain-sized field of view in a viewer.",
    )
    return p.parse_args(argv)


def dataset_kwargs(config, split):
    """Pick the synthetic generator settings out of a resolved experiment config."""
    from data.datasets import SyntheticBrainDataset

    # `mode` and `spatial_size` are set here, not passed through, so a config key
    # of either name cannot silently redirect the export to another split or size.
    accepted = set(inspect.signature(SyntheticBrainDataset.__init__).parameters) - {"self", "mode", "spatial_size"}
    kwargs = {k: v for k, v in config.items() if k in accepted and v is not None}

    res = config.get("synthetic_res") or 64
    kwargs["spatial_size"] = (res, res, res)

    # The split is built at its training size on purpose: --synthetic-normalize
    # fixed_reference estimates its centering/scaling constants from the first 64
    # samples of the split, so a short split would shift every exported intensity.
    n_split = config.get(f"synthetic_num_{split}")
    if n_split is not None:
        kwargs["synthetic_num_samples"] = n_split
    return kwargs


def _to_numpy(x):
    return np.asarray(x.detach().cpu().squeeze().float().numpy(), dtype=np.float32)


def _jsonable(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().tolist()
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def main(argv=None):
    args = parse_args(argv)

    import nibabel as nib

    from data.datasets import SyntheticBrainDataset

    launch = _launch_module()
    config = launch.resolve_config(Path(args.config), None, launch.parse_cli_overrides(args.overrides))

    kwargs = dataset_kwargs(config, args.split)
    ds = SyntheticBrainDataset(mode=args.split, **kwargs)

    os.makedirs(args.out, exist_ok=True)
    affine = np.eye(4) * args.voxel_size
    affine[3, 3] = 1.0

    n_content = kwargs.get("synthetic_n_content", len(FACTOR_NAMES))
    manifest = {
        "config": os.path.basename(args.config),
        "split": args.split,
        "voxel_size": args.voxel_size,
        "settings": {k: _jsonable(v) for k, v in sorted(kwargs.items())},
        "z_content_names": [factor_name(d) for d in range(n_content)],
        "z_style_names": list(STYLE_NAMES[: kwargs.get("synthetic_n_style", len(STYLE_NAMES))]),
        "samples": [],
    }

    for n, idx in enumerate(range(args.start, args.start + args.num)):
        raw = ds._inner[idx] if args.raw else None
        item = ds[idx]
        v1, v2 = item["image"]

        stem = f"sample_{idx:04d}"
        volumes = [("view1", v1), ("view2", v2)]
        if raw is not None:
            volumes += [("view1_raw", raw[0]), ("view2_raw", raw[1])]
        if args.mask:
            volumes.append(("brainmask", item["mask"][0]))

        written = {}
        for name, vol in volumes:
            path = os.path.join(args.out, f"{stem}_{name}.nii.gz")
            nib.save(nib.Nifti1Image(_to_numpy(vol), affine), path)
            written[name] = os.path.basename(path)

        latents = {k: _jsonable(v) for k, v in item["gt_latents"].items() if k != "brain_mask"}
        manifest["samples"].append({"index": idx, "files": written, "gt_latents": latents})
        print(f"[{n + 1}/{args.num}] {stem}  shape={_to_numpy(v1).shape}")

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.num} samples to {args.out} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
