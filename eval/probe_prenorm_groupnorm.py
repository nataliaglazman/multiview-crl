#!/usr/bin/env python
"""Prove (or refute) the "GroupNorm caps content-MCC" hypothesis WITHOUT retraining.

The content block is renormalised by ``SplitGroupNorm`` (``content_norms[level]``) just
before it is pooled and probed. GroupNorm subtracts the per-group per-sample (mean, std) —
a *global* statistic, which is where a global-scale factor like brain_size lives. This
script taps the FROZEN model's content features *before* and *after* that norm (via forward
hooks) and probes the GT factors from each, under gap (global) and patch (local) pooling,
plus directly from the (mu, sigma) the norm throws away.

Prediction if GroupNorm is the cap:
  - gap:   pre-norm R²  >>  post-norm R²   (especially the global factors)
  - patch: pre ≈ post                       (GN preserves the spatial *pattern*)
  - the discarded (mu, sigma) alone recovers the global factors

Nothing is retrained — only the feature tap point changes.

Usage:
  python -m eval.probe_prenorm_groupnorm --run-dir runs/<contrastive> --level 0 --num-samples 500
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2


def main():
    ap = argparse.ArgumentParser(description="Pre- vs post-GroupNorm content recovery (no retraining).")
    ap.add_argument("--run-dir", required=True, help="Training run dir with settings.json.")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint path (default: vqvae_best/model.pt).")
    ap.add_argument("--level", type=int, default=0, help="content_norms level to tap.")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch-grid", type=int, default=4, help="Local pooling grid GxGxG.")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    model, run_args, device = load_model_from_run_dir(args.run_dir, args.checkpoint, None)
    inner = model.module if hasattr(model, "module") else model
    model.eval()

    if not hasattr(inner, "content_norms") or str(args.level) not in inner.content_norms:
        raise SystemExit(f"No content_norms[{args.level}] (SplitGroupNorm) on this model — nothing to tap.")
    norm = inner.content_norms[str(args.level)]
    gn = norm.norm_content  # GroupNorm over the content channels
    n_content, n_groups = gn.num_channels, gn.num_groups
    print(f"content_norms[{args.level}].norm_content: {n_content} content channels, {n_groups} groups")

    # Forward hooks: capture the SplitGroupNorm's input (pre-norm) and output (post-norm).
    pre_cap, post_cap = [], []
    h_pre = norm.register_forward_pre_hook(lambda m, inp: pre_cap.append(inp[0].detach()))
    h_post = norm.register_forward_hook(lambda m, inp, out: post_cap.append(out.detach()))

    ds = build_synthetic_test_set(run_args, args.num_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def pool(feat):
        gap = feat.mean(dim=[2, 3, 4])
        patch = F.adaptive_avg_pool3d(feat, args.patch_grid).flatten(1)
        return gap.cpu().numpy(), patch.cpu().numpy()

    def discarded(feat):
        # the (mu, sigma) GroupNorm subtracts: per-group per-sample, over channels-in-group + space
        b = feat.shape[0]
        g = feat.reshape(b, n_groups, n_content // n_groups, -1)
        return torch.cat([g.mean(dim=[2, 3]), g.std(dim=[2, 3])], dim=1).cpu().numpy()

    acc = {k: [] for k in ("pre_gap", "pre_patch", "post_gap", "post_patch", "disc")}
    gts = []
    with torch.no_grad():
        for batch in loader:
            v1, v2 = batch["image"]
            bview = v1.shape[0]
            x = torch.cat([v1, v2], dim=0).to(device)
            pre_cap.clear()
            post_cap.clear()
            model(x, return_recon=False, pool_only=True, n_views=2, subsets=[(0, 1)], patch_grid=None)
            if not pre_cap or not post_cap:
                raise SystemExit("Hook never fired — content_norms not on the forward path for this config.")
            pre = pre_cap[0][:bview, :n_content]  # first call = view 0; content = first c channels
            post = post_cap[0][:bview, :n_content]
            pg, pp = pool(pre)
            qg, qp = pool(post)
            acc["pre_gap"].append(pg)
            acc["pre_patch"].append(pp)
            acc["post_gap"].append(qg)
            acc["post_patch"].append(qp)
            acc["disc"].append(discarded(pre))
            gts.append(batch["gt_latents"]["z_content"].numpy())
    h_pre.remove()
    h_post.remove()

    acc = {k: np.concatenate(v, 0) for k, v in acc.items()}
    GT = np.concatenate(gts, 0)
    names = CONTENT_FACTOR_NAMES[: GT.shape[1]]

    def per_factor(X):
        return np.array([cv_probe_r2(X, GT[:, j], seeds=seeds)["mean"] for j in range(GT.shape[1])])

    cols = [
        ("pre  gap", acc["pre_gap"]),
        ("post gap", acc["post_gap"]),
        ("pre  patch", acc["pre_patch"]),
        ("post patch", acc["post_patch"]),
        ("discard mu,sd", acc["disc"]),
    ]
    table = {label: per_factor(X) for label, X in cols}

    print("\nPer-factor linear R² — what SplitGroupNorm removes (no retraining):")
    head = f"  {'factor':<18}" + "".join(f"{c:>14}" for c, _ in cols)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for j, name in enumerate(names):
        print(f"  {name:<18}" + "".join(f"{table[c][j]:>14.3f}" for c, _ in cols))
    print("  " + "-" * (len(head) - 2))
    print(f"  {'(mean)':<18}" + "".join(f"{np.nanmean(table[c]):>14.3f}" for c, _ in cols))
    print(
        "\nRead: gap pre>>post (esp. global factors) + discard-(mu,sd) recovers them"
        "\n  => GroupNorm removes the global channel signal -> caps gap-pooled MCC."
        "\n  patch pre≈post => the spatial pattern (local factors) survives the norm."
    )


if __name__ == "__main__":
    main()
