"""TEST 3: is the ventricle shortcut carried by the channel MOMENTS (mu,sigma)
that GroupNorm removes, or by the normalized SHAPE it keeps?

Test 1 (data-side) showed the shortcut is intensity COMPOSITION, not mean/std:
raw-image (mu,sigma) don't predict ventricle_size, but the histogram does. This
test asks the same of the MODEL's level-0 features, and adds forward hooks on the
encoder's own GroupNorms to check pre- vs post-norm directly.

Part A - where in the pooled features F (B,C,P) does the signal live:
  GAP   = per-channel mean mu               -> (B,C)        "the channel means"
  hist  = per-channel spatial histogram     -> (B,C*nbins)  "the composition/shape"

  (validated: a per-channel histogram recovers a mean-fixed composition signal at
  R^2 0.99 where GAP gives ~0; a moment/shape split does NOT -- dividing by sigma
  discards composition, which lives in the variance/higher moments.)

  ventricle: GAP low + hist high -> it is COMPOSITION in the features (matches the
  data-side Test 1). GAP high -> the conv stack converted composition into a channel
  mean (which GroupNorm would then strip -- see Part B).

Part B - forward hooks on the last GroupNorm in the level-0 encoder: probe
  ventricle from pre-norm vs post-norm features, and from the discarded (mu,sigma).
  pre ~= post  -> GroupNorm doesn't remove the signal.

Usage:
  python -m eval.groupnorm_carrier_test --run-dir results/synthetic/<run> --num-samples 800
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2

logger = logging.getLogger(__name__)
FACTORS = ("ventricle_size", "brain_size", "lr_asymmetry")


def pca(X, d=128):
    X = np.asarray(X, np.float64)
    if X.shape[1] <= d:
        return X
    return PCA(n_components=d, random_state=0).fit_transform(StandardScaler().fit_transform(X))


def chan_hist(F, nbins=10):
    """Per-channel spatial histogram (composition). F: (N, C, P) -> (N, C*nbins)."""
    N, C, P = F.shape
    lo, hi = np.percentile(F, [1, 99])
    out = np.empty((N, C, nbins), np.float32)
    for i in range(N):
        for c in range(C):
            out[i, c] = np.histogram(F[i, c], bins=nbins, range=(lo, hi))[0]
    return out.reshape(N, C * nbins) / P


def gn_discarded_stats(x, gn):
    """The (mu, sigma) GroupNorm computes per group and removes. x: (B,C,*spatial)."""
    B, C = x.shape[0], x.shape[1]
    G = gn.num_groups
    xg = x.reshape(B, G, C // G, -1)
    mu = xg.mean(dim=(2, 3))  # (B, G)
    sd = xg.std(dim=(2, 3))  # (B, G)
    return torch.cat([mu, sd], dim=1)  # (B, 2G)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    inner = model.module if hasattr(model, "module") else model
    inner = inner.online if hasattr(inner, "online") else inner
    model.eval()

    # Locate the last GroupNorm in the level-0 encoder for the hook (Part B).
    enc0 = inner.encoders[cli.level]
    gns = [m for m in enc0.modules() if isinstance(m, nn.GroupNorm)]
    last_gn = gns[-1] if gns else None
    logger.info("level-%d encoder has %d GroupNorms; hooking the last one", cli.level, len(gns))
    cap = {}
    if last_gn is not None:
        last_gn.register_forward_hook(lambda mod, inp, out: cap.update(pre=inp[0].detach(), post=out.detach(), mod=mod))

    grid = (cli.grid,) * 3
    ds = build_synthetic_test_set(args, cli.num_samples)
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)
    A = {"gap": [], "f16": []}
    Bp = {k: [] for k in ("pre_gap", "pre_patch", "post_gap", "post_patch", "gn_stats")}
    Z = []
    for batch in loader:
        imgs = batch["image"]
        nv = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=nv, patch_grid=grid)
            feats = (out[2] if isinstance(out, tuple) else out)[cli.level]  # (nv*B, C, P)
            f = feats.reshape(nv, -1, *feats.shape[1:])[0]  # view 0: (B, C, P)
            A["gap"].append(f.mean(-1).cpu().numpy())
            A["f16"].append(f.cpu().numpy().astype(np.float16))  # for per-channel histogram
            # Part B: hooked GroupNorm pre/post (full 5D -> pool here)
            if "pre" in cap:
                pre = cap["pre"].reshape(nv, -1, *cap["pre"].shape[1:])[0]  # (B,C,*sp)
                post = cap["post"].reshape(nv, -1, *cap["post"].shape[1:])[0]
                Bp["pre_gap"].append(pre.mean(dim=(2, 3, 4)).cpu().numpy())
                Bp["post_gap"].append(post.mean(dim=(2, 3, 4)).cpu().numpy())
                pg = tuple(min(cli.grid, s) for s in pre.shape[2:])
                Bp["pre_patch"].append(F.adaptive_avg_pool3d(pre, pg).flatten(1).cpu().numpy())
                Bp["post_patch"].append(F.adaptive_avg_pool3d(post, pg).flatten(1).cpu().numpy())
                Bp["gn_stats"].append(
                    gn_discarded_stats(cap["pre"].reshape(nv, -1, *cap["pre"].shape[1:])[0], cap["mod"]).cpu().numpy()
                )
        gt = batch.get("gt_latents") or {}
        v = gt["z_content"]
        Z.append(np.asarray(v).reshape(v.shape[0], -1))

    gap = np.concatenate(A["gap"], 0)
    hist = pca(chan_hist(np.concatenate(A["f16"], 0).astype(np.float32)))
    Bp = {k: np.concatenate(v, 0) for k, v in Bp.items() if v}
    Z = np.concatenate(Z, 0)

    def probe(Xf, fi):
        return cv_probe_r2(Xf, Z[:, fi], kind="ridge")["mean"]

    print(f"\nrun: {cli.run_dir} | level {cli.level} | grid {list(grid)} | N={Z.shape[0]}")
    print("\nPart A - where the signal lives in level-0 features:")
    hdr = f"{'factor':<16}{'GAP(mu)':>10}{'hist(comp)':>12}"
    print(hdr + "\n" + "-" * len(hdr))
    for fname in FACTORS:
        fi = CONTENT_FACTOR_NAMES.index(fname)
        print(f"{fname:<16}{probe(gap, fi):>10.3f}{probe(hist, fi):>12.3f}")
    print("\n  ventricle: GAP low + hist high -> COMPOSITION in the features (matches Test 1).")
    print("  GAP high -> conv stack put composition into a channel mean (GroupNorm would strip it; see Part B).")

    if Bp:
        print("\nPart B - hooked last GroupNorm (pre vs post) + discarded (mu,sigma):")
        hdr = f"{'factor':<16}{'pre_GAP':>10}{'post_GAP':>10}{'pre_patch':>11}{'post_patch':>12}{'gn(mu,sd)':>11}"
        print(hdr + "\n" + "-" * len(hdr))
        for fname in FACTORS:
            fi = CONTENT_FACTOR_NAMES.index(fname)
            print(
                f"{fname:<16}{probe(Bp['pre_gap'], fi):>10.3f}{probe(Bp['post_gap'], fi):>10.3f}"
                f"{probe(pca(Bp['pre_patch']), fi):>11.3f}{probe(pca(Bp['post_patch']), fi):>12.3f}"
                f"{probe(Bp['gn_stats'], fi):>11.3f}"
            )
        print("\n  pre ~= post -> GroupNorm doesn't remove it.  gn(mu,sd) low -> its discarded stats don't carry it.")


if __name__ == "__main__":
    main()
