#!/usr/bin/env python3
"""extract_factors.py -- dump synthetic ground-truth factors aligned to a
phase0_extract content dump, for the labelled-data maps in eval/recovery_maps.py
(``--mode gt|dci|drift``).

It rebuilds the SAME split the same way ``eval/phase0_extract.py`` does (so subject
order matches the content maps), then pulls a key out of each sample's
``gt_latents`` (default ``z_content``) and writes:

    {out}/factors.npy        (S, K) float32, row s = subject sub-{idx:04d}
    {out}/factors_meta.json  {subjects: [...], factor, K, n_subjects}

``recovery_maps`` reads these from ``<root>`` by default and re-orders rows to its
discovered subject list, so alignment is automatic.

Usage
  python -m eval.extract_factors --run-dir results/<run> --split val
  # -> writes results/<run>/phase0_content_maps/factors.npy (+ meta)
  python -m eval.recovery_maps --mode gt  --root results/<run>/phase0_content_maps --solA view0:0 --out out/gt
  python -m eval.recovery_maps --mode dci --root results/<run>/phase0_content_maps --solA view0:0 --out out/dci

Only SyntheticBrainDataset exposes ``gt_latents``; for ADNI there is no GT, so build
a covariate array (severity d / age, [S, K]) by hand and pass it as --factors.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from eval.phase0_extract import build_dataset_kwargs, load_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Run directory with settings.json (same as phase0_extract).")
    ap.add_argument("--settings", default=None, help="Override path to settings.json.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="Output dir. Default <run-dir>/phase0_content_maps (the content dump).")
    ap.add_argument("--num-samples", type=int, default=None, help="Cap subjects (default: whole split).")
    ap.add_argument("--factor", default="z_content", help="Which gt_latents key to dump (default z_content).")
    a = ap.parse_args()

    args = load_args(a.run_dir, a.settings)
    dataset = args.DATASETCLASS(
        data_dir=getattr(args, "datapath", None),
        mode=a.split,
        spacing=getattr(args, "image_spacing", 2.0),
        crop_margin=getattr(args, "crop_margin", 0),
        spatial_size=getattr(args, "spatial_size", None),
        cache=False,
        **build_dataset_kwargs(args),
    )
    if a.num_samples is not None:
        dataset.num_samples = min(dataset.num_samples, a.num_samples)

    out = a.out or os.path.join(a.run_dir, "phase0_content_maps")
    os.makedirs(out, exist_ok=True)

    subjects, rows = [], []
    for i in range(len(dataset)):
        d = dataset[i]
        latents = d.get("gt_latents")
        if not isinstance(latents, dict) or a.factor not in latents:
            keys = list(latents) if isinstance(latents, dict) else None
            raise SystemExit(
                f"sample {i} has no gt_latents['{a.factor}'] (keys={keys}). This dataset exposes no GT "
                "factors (ADNI?). Build a covariate [S,K] .npy yourself and pass it to recovery_maps --factors."
            )
        z = latents[a.factor]
        z = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
        subjects.append(f"sub-{int(d['index']):04d}")
        rows.append(np.asarray(z, np.float32).ravel())

    K = max(len(r) for r in rows)
    if any(len(r) != K for r in rows):
        lens = set(len(r) for r in rows)
        raise SystemExit(f"ragged factor vectors (lengths {lens}) -- pick a scalar/vector gt_latents key")
    Z = np.stack(rows).astype(np.float32)
    np.save(os.path.join(out, "factors.npy"), Z)
    meta = {"subjects": subjects, "factor": a.factor, "K": int(K), "n_subjects": int(Z.shape[0]), "split": a.split}
    with open(os.path.join(out, "factors_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved factors {Z.shape} (factor='{a.factor}') -> {out}/factors.npy  (+ factors_meta.json)")
    print("now: python -m eval.recovery_maps --mode gt --root", out, "--solA view0:0 --out out/gt")


if __name__ == "__main__":
    main()
