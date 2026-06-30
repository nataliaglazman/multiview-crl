#!/usr/bin/env python
"""CPU smoke test for the patch-contrastive path (incl. --patch-foreground-mask).

Builds a tiny VQ-VAE on a few synthetic volumes and runs ``train_step`` for a
couple of iterations in encoder-only mode (no decoder, so the LPIPS recon loss
is skipped — it needs larger 2D slices than a smoke-sized volume provides).
Exercises the real training code path end-to-end, so it catches integration
breakage before a cluster launch. Validates that the foreground mask actually
changes the loss (drops background patches) rather than just not crashing.

    python scripts/smoke_patch_contrastive.py

Exit code 0 = OK, 1 = failure. Runs on CPU; no GPU or dataset needed.
"""
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from data.datasets import SyntheticBrainDataset  # noqa: E402
from training.losses import barlow_twins_loss  # noqa: E402
from training.main_multimodal import build_vqvae, train_step  # noqa: E402
from utils.config import parse_args, update_args  # noqa: E402

RES, B, GRID = 16, 2, "4"  # 4^3 grid so some corner patches are background


def make_args(extra):
    argv = [
        "--dataset-name",
        "synthetic",
        "--dataroot",
        "/tmp",
        "--synthetic-res",
        str(RES),
        "--vqvae-nb-levels",
        "2",
        "--vqvae-scaling-rates",
        "2",
        "2",
        "--vqvae-hidden-channels",
        "16",
        "--vqvae-res-channels",
        "8",
        "--vqvae-embed-dim",
        "8",
        "--vqvae-nb-entries",
        "16",
        "--content-size",
        "4",
        "--batch-size",
        str(B),
        "--patch-contrastive",
        "--contrastive-loss-type",
        "barlow_twins",
        "--patch-grid",
        GRID,
        GRID,
        GRID,
        "--no-cuda",
        "--contrastive-only",
    ] + extra
    sys.argv = ["smoke"] + argv
    return update_args(parse_args().parse_args())


def build_batch():
    ds = SyntheticBrainDataset(
        mode="train",
        spatial_size=(RES, RES, RES),
        cache=True,
        synthetic_mode="pseudo_mri",
        synthetic_seed=1,
        synthetic_num_samples=B,
        synthetic_n_content=9,
        synthetic_n_style=3,
    )
    cols = {"v0": [], "v1": [], "m0": [], "m1": []}
    for i in range(B):
        d = ds[i]
        cols["v0"].append(d["image"][0])
        cols["v1"].append(d["image"][1])
        cols["m0"].append(d["mask"][0])
        cols["m1"].append(d["mask"][1])

    def st(xs):
        return torch.stack([t.float().reshape(RES, RES, RES) for t in xs]).unsqueeze(1)

    return {"image": [st(cols["v0"]), st(cols["v1"])], "mask": [st(cols["m0"]), st(cols["m1"])]}


def bt_lf(hz, ci, subsets, soft_content_mask=None):
    return barlow_twins_loss(
        hz, estimated_content_indices=ci, subsets=subsets, soft_content_mask=soft_content_mask, lambd=0.005
    )


def run(fg):
    args = make_args(["--patch-foreground-mask"] if fg else [])
    torch.manual_seed(0)
    model = build_vqvae(args).cpu()
    batch = build_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    out = None
    for s in range(2):
        out = train_step(
            batch,
            [model],
            [],
            bt_lf,
            opt,
            list(model.parameters()),
            args,
            patch_loss_func=bt_lf,
            moco_loss_func=None,
            step=s,
        )
    return float(out[0])


def main():
    try:
        loss_off = run(fg=False)
        loss_on = run(fg=True)
    except Exception as e:
        print(f"SMOKE FAILED -> {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print(f"patch BT, no fg-mask : loss={loss_off:.4f}")
    print(f"patch BT, fg-mask    : loss={loss_on:.4f}")
    ok = math.isfinite(loss_off) and math.isfinite(loss_on)
    changed = abs(loss_on - loss_off) > 1e-9
    print(
        f"finite={ok}  fg-mask-changes-loss={changed} "
        f"({'background patches dropped' if changed else 'no background at this grid/res'})"
    )
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
