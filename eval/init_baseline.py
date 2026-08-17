"""What does block-MCC read from an UNTRAINED encoder? The zero point for every MCC curve.

A training curve that starts near 1.0 is not evidence that training worked — it may mean the
probe could already read the factors from a random projection of the image. This measures
that zero point directly, so an MCC number can be reported as a GAP over init rather than as
an absolute.

Two readouts, because they behave completely differently:

  GAP     one value per channel. A narrow readout; genuinely hard at init.
  patch   channels x positions. With a 4^3 grid and 48 channels that is 3072 linear features
          spanning the volume, which is enough for a ridge probe to reconstruct any GLOBAL
          geometric factor from an untrained conv stack. This is the pooling the flagship
          config logs, and it is why its curves start high.

A shuffled-label null is computed alongside: it separates "the information is really there at
init" (null ~ 0.15, the CV floor) from "the probe is overfitting" (null rises with the mean).

Usage
-----
    python -m eval.init_baseline --preset new
    python -m eval.init_baseline --preset both --grid 4 --n 400
"""

import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.datasets import SyntheticBrainDataset
from eval.identifiability_metrics import block_mcc

CONTENT_NAMES = [
    "brain_size",
    "ventricle",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "thickness",
    "temporal",
    "lr_asym",
    "sulcal",
]

OLD = dict(
    synthetic_content_prior="normal",
    synthetic_content_squash="auto",
    synthetic_cortex_parameterization="additive",
    synthetic_center_local_deformations=False,
    synthetic_identifiable_ventricle=False,
    synthetic_lesion_radius=0.1,
)
NEW = dict(
    synthetic_content_prior="uniform",
    synthetic_content_squash="none",
    synthetic_cortex_parameterization="patterned",
    synthetic_center_local_deformations=True,
    synthetic_identifiable_ventricle=True,
    synthetic_lesion_radius=0.14,
)


class RandomEncoder(nn.Module):
    """Untrained stand-in for the encoder at step 0.

    Not the VQ-VAE — the point is that the ARCHITECTURE class (strided convs + a
    normalization that mixes global statistics into every position) already carries the
    factors, so the exact weights are irrelevant. Frozen at init.
    """

    def __init__(self, ch=48, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv3d(1, ch, 4, 2, 1),
            nn.GroupNorm(8, ch),
            nn.ReLU(),
            nn.Conv3d(ch, ch, 4, 2, 1),
            nn.GroupNorm(8, ch),
            nn.ReLU(),
            nn.Conv3d(ch, ch, 3, 1, 1),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.net(x)


def collect(overrides, res, n, grid, ch, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = SyntheticBrainDataset(
            mode="train",
            spatial_size=(res,) * 3,
            synthetic_num_samples=n,
            synthetic_n_content=9,
            synthetic_n_style=3,
            synthetic_normalize="fixed_reference",
            synthetic_causal=True,
            synthetic_causal_graph="random",
            synthetic_causal_edge_prob=0.5,
            synthetic_clean_content=True,
            synthetic_seed=42,
            **overrides,
        )
    enc = RandomEncoder(ch=ch, seed=seed).eval()
    gap, patch, Z = [], [], []
    for i in range(n):
        x1, x2, lat = ds._inner[i]
        m = lat["brain_mask"]
        x1, _ = ds.normalize_views(x1, x2, m, m.clone())
        with torch.no_grad():
            f = enc(x1.unsqueeze(0))
        gap.append(f.mean(dim=(2, 3, 4)).squeeze(0).numpy())
        patch.append(F.adaptive_avg_pool3d(f, grid).flatten().numpy())
        Z.append(lat["z_content"].numpy())
    return np.stack(gap), np.stack(patch), np.stack(Z)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="new", choices=["old", "new", "both"])
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--grid", type=int, default=4, help="patch grid per axis")
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    presets = {"old": [("OLD", OLD)], "new": [("NEW", NEW)], "both": [("OLD", OLD), ("NEW", NEW)]}[args.preset]
    print(
        f"UNTRAINED encoder, res={args.res}, n={args.n}, patch grid {args.grid}^3 "
        f"({args.channels * args.grid**3} linear features), ridge probe\n"
    )
    for tag, ov in presets:
        gap, patch, Z = collect(ov, args.res, args.n, args.grid, args.channels, args.seed)
        for fname, X in (("GAP", gap), (f"patch{args.grid}", patch)):
            r = block_mcc(X, Z)
            null = block_mcc(X, Z[np.random.RandomState(0).permutation(len(Z))], seeds=(0,))
            print(f"  {tag} {fname:8s} block_mcc = {r['mean']:.3f}   shuffled null = {null['mean']:.3f}")
            print("      " + "  ".join(f"{n}:{v:.2f}" for n, v in zip(CONTENT_NAMES, r["per_factor"])))
        print()
    print("  Read a trained run's MCC as a GAP over the matching row. A factor already at ~0.99")
    print("  here has no headroom left: at that pooling the metric cannot separate a trained")
    print("  encoder from a random one, whatever the training curve does.")


if __name__ == "__main__":
    main()
