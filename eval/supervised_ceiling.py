#!/usr/bin/env python
"""Supervised recovery ceiling for the synthetic factors.

Answers one question, from data: *how much of each ground-truth factor is even
present in the rendered image?* A small 3D CNN is trained SUPERVISED to regress
every factor directly from the raw image; its held-out per-factor R^2 is the
ceiling no unsupervised representation (contrastive, recon, or otherwise) can
beat. Reading each factor off the model's pooled representation below that
ceiling is a representation/pooling limit that is worth fixing; a factor whose
*supervised* ceiling is already low is not in the pixels and no method will
recover it — stop chasing it.

Data config is taken from a run's settings.json (via ``load_run_args``) so the
ceiling is measured on exactly the distribution that run trained/eval'd on
(same seed, normalization, scales, clean_content, ...). Content factors and the
view-1 style factors are predicted from the view-1 image; content is shared
across views so a single view is the honest per-image ceiling.

Usage:
    python -m eval.supervised_ceiling --run-dir results/synthetic/<run> \
        --num-samples 3000 --epochs 60 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

from eval.dci import CONTENT_FACTOR_NAMES, STYLE_FACTOR_NAMES
from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FACTOR_NAMES = list(CONTENT_FACTOR_NAMES) + list(STYLE_FACTOR_NAMES)
N_CONTENT = len(CONTENT_FACTOR_NAMES)  # 9
N_STYLE = len(STYLE_FACTOR_NAMES)  # 3


class _ViewFactorDataset(torch.utils.data.Dataset):
    """Wrap the synthetic dataset to yield ``(view-1 image, target vector)``.

    target = [z_content (9) | z_style_v1 (3)]. z_content indexes CONTENT_FACTOR_NAMES
    directly (render_structure: 0=brain_size .. 8=sulcal_widening), and z_style_v1
    indexes STYLE_FACTOR_NAMES directly (0=gain, 1=bias, 2=noise_sigma).
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x = item["image"][0]  # view 1, shape (1, D, H, W)
        gt = item["gt_latents"]
        zc = torch.as_tensor(gt["z_content"], dtype=torch.float32).flatten()[:N_CONTENT]
        zs = torch.as_tensor(gt["z_style_v1"], dtype=torch.float32).flatten()[:N_STYLE]
        # Right-pad defensively if a run used fewer components than the renderer names.
        if zc.numel() < N_CONTENT:
            zc = torch.cat([zc, torch.zeros(N_CONTENT - zc.numel())])
        if zs.numel() < N_STYLE:
            zs = torch.cat([zs, torch.zeros(N_STYLE - zs.numel())])
        return x.float(), torch.cat([zc, zs])


class _SmallCNN3D(nn.Module):
    """~1M-param 3D CNN regressor. Four stride-2 blocks -> GAP -> MLP head.

    Deliberately modest: the ceiling should be a well-trained probe, not an
    overfit one, at a few thousand samples. BatchNorm + early stopping keep it
    honest.
    """

    def __init__(self, n_out, width=16):
        super().__init__()
        c = [1, width, width * 2, width * 4, width * 8]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv3d(c[i], c[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(c[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(c[-1], c[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(c[-1], n_out),
        )

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.head(h)


def _run_split(base_ds, seed):
    n = len(base_ds)
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    n_test = max(1, int(0.15 * n))
    n_val = max(1, int(0.15 * n))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _targets_np(loader):
    ys = []
    for _, y in loader:
        ys.append(y.numpy())
    return np.concatenate(ys, 0)


def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


def main():
    p = argparse.ArgumentParser(description="Supervised per-factor recovery ceiling from the raw synthetic image.")
    p.add_argument("--run-dir", required=True, help="Run dir with settings.json (defines the data distribution).")
    p.add_argument("--num-samples", type=int, default=3000, help="Synthetic samples to render (train+val+test).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=16, help="CNN base width.")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience on val R^2.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="JSON output path (default: <run-dir>/supervised_ceiling.json).")
    cli = p.parse_args()

    if not os.path.exists(os.path.join(cli.run_dir, "settings.json")):
        logger.error("settings.json not found in %s", cli.run_dir)
        sys.exit(1)

    torch.manual_seed(cli.seed)
    device = torch.device(cli.device)

    args = load_run_args(cli.run_dir)
    base = _ViewFactorDataset(build_synthetic_test_set(args, num_samples=cli.num_samples))
    train_idx, val_idx, test_idx = _run_split(base, cli.seed)
    logger.info("splits: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    dl_kw = dict(batch_size=cli.batch_size, num_workers=cli.num_workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(Subset(base, train_idx), shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(Subset(base, val_idx), shuffle=False, **dl_kw)
    test_loader = DataLoader(Subset(base, test_idx), shuffle=False, **dl_kw)

    # Standardize targets on TRAIN stats so the multi-output MSE weights every
    # factor evenly (R^2 is invariant to this, but the optimizer is not).
    y_train = _targets_np(train_loader)
    t_mean = torch.tensor(y_train.mean(0), dtype=torch.float32, device=device)
    t_std = torch.tensor(y_train.std(0) + 1e-6, dtype=torch.float32, device=device)

    model = _SmallCNN3D(n_out=len(FACTOR_NAMES), width=cli.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cli.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    best_val, best_state, since_improved = -np.inf, None, 0
    for epoch in range(cli.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            yt = (y - t_mean) / t_std
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validate on standardized-space R^2 averaged over factors.
        vp, vt = _predict(model, val_loader, device)
        vp = vp * t_std.cpu().numpy() + t_mean.cpu().numpy()  # de-standardize preds
        val_r2 = float(np.mean([r2_score(vt[:, j], vp[:, j]) for j in range(vt.shape[1])]))
        sched.step(val_r2)
        logger.info("epoch %02d  val mean-R2 %.4f", epoch, val_r2)
        if val_r2 > best_val:
            best_val, best_state, since_improved = (
                val_r2,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            since_improved += 1
            if since_improved >= cli.patience:
                logger.info("early stop at epoch %d (best val mean-R2 %.4f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tp, tt = _predict(model, test_loader, device)
    tp = tp * t_std.cpu().numpy() + t_mean.cpu().numpy()
    per_factor = {FACTOR_NAMES[j]: float(r2_score(tt[:, j], tp[:, j])) for j in range(tt.shape[1])}

    print("\n" + "=" * 60)
    print("  SUPERVISED CEILING  (test R^2, raw image -> factor)")
    print("  high => factor IS in the pixels (representation gap is fixable)")
    print("  low  => factor not recoverable from the image (stop chasing)")
    print("=" * 60)
    for kind, names in (("content", CONTENT_FACTOR_NAMES), ("style ", STYLE_FACTOR_NAMES)):
        for name in names:
            r2 = per_factor[name]
            bar = "#" * int(max(0.0, min(1.0, r2)) * 10)
            print(f"  {kind}  {name:18s} {r2:+.3f} [{bar:<10}]")
    print("=" * 60)

    out_path = cli.out or os.path.join(cli.run_dir, "supervised_ceiling.json")
    with open(out_path, "w") as f:
        json.dump(
            {"per_factor": per_factor, "val_mean_r2": best_val, "num_samples": cli.num_samples, "seed": cli.seed},
            f,
            indent=2,
        )
    logger.info("saved ceiling to %s", out_path)


if __name__ == "__main__":
    main()
