#!/usr/bin/env python3
"""phase0_extract.py -- dump content maps from a trained synthetic VQ-VAE-2 for
the Phase 0 cross-modal identifiability probe (``eval/phase0_probe.py``).

The probe operates on CONTINUOUS, pre-quantisation content maps stored on disk
as ``{root}/{subject}_{modality}_seed{seed}.npy`` (shape ``(C, *spatial)``) plus
a per-subject ``{root}/{subject}_mask.npy`` at feature-grid resolution. This
script produces exactly that layout from a run directory.

What it extracts
  * For every sample of the chosen split it runs each view through the encoder
    (in eval mode, so the content mask is the deterministic top-k) and saves the
    content channels of the finest masked level -- the same continuous features
    the codebook and the contrastive loss see, before quantisation.
  * "modality" maps to the two synthetic views: view 0 -> ``view0`` (the probe's
    ``--t1``), view 1 -> ``view1`` (``--flair``).
  * The brain mask is nearest-downsampled from input resolution to the feature
    grid and saved once per subject.

Each invocation dumps ONE seed (the training seed of the checkpoint). For the
seed-reproducibility ceiling, run it twice -- two checkpoints from two training
seeds -- into the SAME ``--out`` directory with different ``--seed`` tags; the
synthetic split is deterministic, so subjects align across seeds.

Usage
  # one checkpoint -> cross-modal dump
  python -m eval.phase0_extract --run-dir results/<run> --seed 0
  python -m eval.phase0_probe --root results/<run>/phase0_content_maps \
                              --t1 view0 --flair view1

  # add a second seed for the reproducibility ceiling
  python -m eval.phase0_extract --run-dir results/<run_seedA> --seed 0 \
                                --out shared_dump
  python -m eval.phase0_extract --run-dir results/<run_seedB> --seed 1 \
                                --out shared_dump
  python -m eval.phase0_probe --root shared_dump --t1 view0 --flair view1 \
                              --seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import data.datasets as datasets
from training.main_multimodal import build_vqvae

# dataset_name -> dataset class, mirroring utils.config.update_args.
_DATASET_CLASSES = {
    "synthetic": datasets.SyntheticBrainDataset,
    "custom": datasets.MyCustomDataset,
    "adni": datasets.MyCustomDataset,
    "ADNI_registered": datasets.MyCustomDataset,
    "ADNI_stripped": datasets.MyCustomDataset,
    "ADNI_stripped_masks": datasets.MyCustomDataset,
}


def load_args(run_dir: str, settings_path: str | None) -> argparse.Namespace:
    """Reconstruct the training args namespace from a run's ``settings.json``.

    ``settings.json`` is ``args.__dict__`` minus ``DATASETCLASS`` (see
    ``main_multimodal.main``), i.e. already post-``update_args``, so derived
    fields (``content_indices``, ``style_indices``, ``content_ratios``, ...) are
    present. We only re-attach ``DATASETCLASS`` and backfill content/style
    indices for the rare old checkpoint that predates them.
    """
    path = settings_path or os.path.join(run_dir, "settings.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No settings.json at {path}. Pass --settings explicitly, or point "
            f"--run-dir at a directory containing the run's settings.json."
        )
    with open(path) as f:
        d = json.load(f)
    args = argparse.Namespace(**d)

    dataset_name = getattr(args, "dataset_name", "synthetic")
    args.DATASETCLASS = _DATASET_CLASSES.get(dataset_name, datasets.SyntheticBrainDataset)

    # Backfill content/style indices if an old settings.json lacks them.
    if not getattr(args, "content_indices", None):
        cdim = getattr(args, "content_dim", None)
        tdim = getattr(args, "total_dim", None) or getattr(args, "vqvae_hidden_channels", None)
        if cdim is None or tdim is None:
            raise ValueError("settings.json lacks content_indices and content_dim/total_dim to rebuild them.")
        args.content_indices = [list(range(cdim))]
        args.style_indices = list(range(cdim, tdim))
    if not hasattr(args, "style_indices"):
        args.style_indices = []
    return args


def build_dataset_kwargs(args) -> dict:
    """Mirror ``main_multimodal``'s ``dataset_kwargs`` so the split renders
    identically to training/eval."""
    kwargs = {
        "labels_path": getattr(args, "labels_path", None),
        "masks_dir": getattr(args, "masks_dir", None),
        "asymmetric_aug": getattr(args, "asymmetric_aug", False),
        "shared_brain_mask": getattr(args, "shared_brain_mask", False),
    }
    if getattr(args, "dataset_name", None) == "synthetic":
        kwargs.update(
            {
                "synthetic_mode": getattr(args, "synthetic_mode", "pseudo_mri"),
                "synthetic_seed": getattr(args, "synthetic_seed", 42),
                "synthetic_n_content": getattr(args, "synthetic_n_content", 5),
                "synthetic_n_style": getattr(args, "synthetic_n_style", 3),
                "synthetic_normalize": getattr(args, "synthetic_normalize", "per_sample"),
                "synthetic_hierarchical_content": getattr(args, "synthetic_hierarchical_content", False),
                "synthetic_causal": getattr(args, "synthetic_causal", False),
                "synthetic_causal_graph": getattr(args, "synthetic_causal_graph", "chain"),
                "synthetic_causal_edge_prob": getattr(args, "synthetic_causal_edge_prob", 0.5),
                "synthetic_causal_noise_scale": getattr(args, "synthetic_causal_noise_scale", 0.4),
                "synthetic_causal_nonlinearity": getattr(args, "synthetic_causal_nonlinearity", "leaky_relu"),
                "synthetic_identifiable_ventricle": getattr(args, "synthetic_identifiable_ventricle", False),
                "synthetic_num_samples_per_mode": {
                    "train": getattr(args, "synthetic_num_train", 1000),
                    "val": getattr(args, "synthetic_num_val", 100),
                    "test": getattr(args, "synthetic_num_test", 200),
                },
            }
        )
    return kwargs


def _strip_to_vqvae(state_dict: dict, model: torch.nn.Module) -> dict:
    """Map a saved checkpoint state_dict onto a bare ``VQVAE``.

    The checkpoint stores ``encoders[0].state_dict()``. In training the VQ-VAE is
    wrapped in ``DataParallel`` (keys prefixed ``module.``) and optionally in
    ``MoCoEncoder`` (the VQ-VAE lives under ``online.``). Try the plausible
    prefixes and keep the one that best matches the bare model's keys.
    """
    model_keys = set(model.state_dict().keys())
    candidates = ["", "module.", "online.module.", "online.", "module.online.module.", "module.online."]
    best, best_hits = state_dict, -1
    for pfx in candidates:
        if pfx:
            remapped = {k[len(pfx) :]: v for k, v in state_dict.items() if k.startswith(pfx)}
        else:
            remapped = dict(state_dict)
        hits = len(model_keys & set(remapped.keys()))
        if hits > best_hits:
            best, best_hits = remapped, hits
    return best


def load_model(args, checkpoint_path: str, device: str) -> torch.nn.Module:
    model = build_vqvae(args).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["encoders"] if isinstance(ckpt, dict) and "encoders" in ckpt else ckpt
    state = _strip_to_vqvae(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Drop codebook EMA buffers from the "missing" tally — they are registered
    # buffers that always load when present; anything truly absent is reported.
    real_missing = [k for k in missing if "embed" not in k and "cluster_size" not in k]
    if real_missing:
        print(
            f"[warn] {len(real_missing)} model params not found in checkpoint (using init): "
            f"{real_missing[:6]}{' ...' if len(real_missing) > 6 else ''}"
        )
    if unexpected:
        print(f"[info] {len(unexpected)} checkpoint keys unused (wrapper/aux params).")
    if ckpt.get("step") is not None:
        print(f"[info] checkpoint step: {ckpt['step']}")
    return model


def _content_indices(model: torch.nn.Module, est_indices, finest_level: int, hidden_channels: int, device):
    """Resolve the content-channel index tensor for the finest masked level.

    ``est_indices`` is the forward's ``estimated_content_indices`` (a list, one
    entry per subset). ``None`` means an all-content model (no mask) -> every
    channel is content.
    """
    if est_indices is None:
        return torch.arange(hidden_channels, device=device)
    idx = est_indices[0]
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, device=device)
    return idx.to(device).long()


@torch.no_grad()
def extract(args, model, dataset, out_dir, seed, device, batch_size, workers, view_names):
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    hidden = args.vqvae_hidden_channels
    finest_level = min(getattr(args, "content_style_levels", [0]))

    # Pin the content-channel selection from the first batch of each view so the
    # stacked (N, C) matrix has consistent columns across all subjects. For
    # fixed/learned masks this is exact; for "onthefly" the top-k can shift with
    # the batch, so we warn if a later batch disagrees.
    pinned_idx = {}
    grid_shape = None
    n_saved = 0

    for batch in loader:
        images = batch["image"]  # [view0, view1], each (B, 1, D, H, W)
        masks = batch["mask"]  # [mask0, mask1], each (B, 1, D, H, W)
        indices = batch["index"]
        idx_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
        B = len(idx_list)

        for v, vname in enumerate(view_names):
            x = images[v].to(device).float()
            out = model(x, return_recon=False, pool_only=False, n_views=1, view_idx=v)
            enc_feats, est_indices = out[2], out[3]
            feat = enc_feats[finest_level]  # (B, hidden, d, h, w)

            cidx = _content_indices(model, est_indices, finest_level, hidden, device)
            if v not in pinned_idx:
                pinned_idx[v] = cidx
            else:
                same = cidx.numel() == pinned_idx[v].numel() and torch.equal(
                    cidx.sort().values, pinned_idx[v].sort().values
                )
                if not same:
                    print(
                        f"[warn] view{v}: content channels changed between batches "
                        f"(mask_mode={getattr(args, 'mask_mode', 'onthefly')!r}); "
                        f"reusing the first batch's selection for column consistency."
                    )
            content = feat[:, pinned_idx[v]]  # (B, C, d, h, w)
            content_np = content.detach().cpu().numpy().astype(np.float32)

            if grid_shape is None:
                grid_shape = tuple(content.shape[2:])

            for b in range(B):
                subj = f"sub-{idx_list[b]:04d}"
                np.save(os.path.join(out_dir, f"{subj}_{vname}_seed{seed}.npy"), content_np[b])

                # Save the feature-grid mask once per subject (view 0). Synthetic
                # views share a brain mask; deterministic across seeds.
                if v == 0:
                    m = masks[v][b : b + 1].to(device).float()  # (1, 1, D, H, W)
                    m_ds = torch.nn.functional.interpolate(m, size=content.shape[2:], mode="nearest")
                    m_np = m_ds[0, 0].detach().cpu().numpy() > 0.5
                    np.save(os.path.join(out_dir, f"{subj}_mask.npy"), m_np)
        n_saved += B

    meta = {
        "run_settings_dataset": getattr(args, "dataset_name", None),
        "split": dataset.mode,
        "seed_tag": seed,
        "n_subjects": n_saved,
        "content_channels": int(pinned_idx[0].numel()) if 0 in pinned_idx else None,
        "feature_grid": list(grid_shape) if grid_shape else None,
        "finest_masked_level": finest_level,
        "mask_mode": getattr(args, "mask_mode", "onthefly"),
        "view_names": view_names,
        "content_idx_view0": pinned_idx.get(0).tolist() if 0 in pinned_idx else None,
        "content_idx_view1": pinned_idx.get(1).tolist() if 1 in pinned_idx else None,
    }
    with open(os.path.join(out_dir, f"phase0_extract_meta_seed{seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"\nSaved {n_saved} subjects x {len(view_names)} views (C={meta['content_channels']}, "
        f"grid={meta['feature_grid']}) -> {out_dir}"
    )
    # --seeds matches THIS dump so the cross-modal run works without a 2nd seed
    # (the probe's default is [0, 1] and would look for a missing seed-1 file).
    print(
        f"Now run the cross-modal probe:\n  python -m eval.phase0_probe --root {out_dir} "
        f"--t1 {view_names[0]} --flair {view_names[1]} --seeds {seed}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run-dir", required=True, help="Run directory containing settings.json and the checkpoint (vqvae_model.pt)."
    )
    ap.add_argument("--settings", default=None, help="Override path to settings.json.")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path. Default: <run-dir>/vqvae_best.pt if present, else vqvae_model.pt.",
    )
    ap.add_argument("--out", default=None, help="Output dir for the .npy dump. Default: <run-dir>/phase0_content_maps.")
    ap.add_argument(
        "--split", default="val", choices=["train", "val", "test"], help="Dataset split to encode (default: val)."
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed TAG for the dump filenames (use a distinct value per training-seed checkpoint).",
    )
    ap.add_argument("--num-samples", type=int, default=None, help="Cap the number of subjects (default: whole split).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto).")
    ap.add_argument(
        "--view-names",
        nargs=2,
        default=["view0", "view1"],
        help="Names used for the two views in the filenames / probe flags.",
    )
    a = ap.parse_args()

    if a.device:
        device = a.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = load_args(a.run_dir, a.settings)

    checkpoint = a.checkpoint
    if checkpoint is None:
        best = os.path.join(a.run_dir, "vqvae_best.pt")
        rolling = os.path.join(a.run_dir, "vqvae_model.pt")
        checkpoint = best if os.path.exists(best) else rolling
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"No checkpoint at {checkpoint}.")
    print(f"[info] device={device}  checkpoint={checkpoint}")

    out_dir = a.out or os.path.join(a.run_dir, "phase0_content_maps")

    # Build the split dataset, mirroring main_multimodal's construction.
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
    print(f"[info] split={a.split}  subjects={len(dataset)}")

    model = load_model(args, checkpoint, device)
    extract(args, model, dataset, out_dir, a.seed, device, a.batch_size, a.workers, a.view_names)


if __name__ == "__main__":
    main()
