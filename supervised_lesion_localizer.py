#!/usr/bin/env python3
"""Supervised lesion localisation from ONE synthetic MRI view (T1 or FLAIR).

Control experiment for the encoder-only contrastive runs: can a network that IS given
lesion labels find the lesion from a single view? It trains a small 3D CNN on exactly the
data an encoder-only run saw -- rebuilt from that run's settings.json with
eval.score_checkpoint.make_dataset, the same factory `score_checkpoint --lesion-analysis`
uses -- and scores it on the run's validation subjects against the same two targets:

  centroid  centre of the rendered lesion (voxel coordinates), predicted directly
  latent    the generator's lesion values z_content[2:5], read from the predicted centre
            with the repo's 5-fold x 3-seed ridge probe (eval.identifiability_metrics)

The network outputs a map of lesion logits (not a global average), trained with a
cross-entropy between its softmax over positions and the lesion's occupancy. Validation
subjects are never used for training or model selection; the last step is scored.

Run from the repo root:
    python supervised_lesion_localizer.py --run-dir results/<run> --view both
--device auto picks cuda, then mps (Apple silicon), then cpu. If an op is missing on MPS,
set PYTORCH_ENABLE_MPS_FALLBACK=1.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = slice(8, 56)  # 48^3 around the brain; every white-matter lesion lies inside it at res 64
VIEWS = ("t1", "flair")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Encoder-only run directory holding settings.json")
    p.add_argument("--view", choices=(*VIEWS, "both"), default="both")
    p.add_argument("--train-samples", type=int, default=1500, help="Subjects from the run's train split")
    p.add_argument("--val-samples", type=int, default=None, help="Default: the run's num_val_samples")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Rendering processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repo", default=".", help="Repository root (put on sys.path)")
    p.add_argument("--out", default=None, help="Results JSON (default: <run-dir>/supervised_lesion.json)")
    return p.parse_args(argv)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---- data --------------------------------------------------------------------------------
_DATASETS = {}


def _render_chunk(job):
    repo, cfg, split, n, lo, hi = job
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if (split, n) not in _DATASETS:
        torch.set_num_threads(1)
        from eval.score_checkpoint import make_dataset

        _DATASETS[(split, n)] = make_dataset(cfg, n, mode=split)
    ds = _DATASETS[(split, n)]
    renderer, clean = ds._inner.renderer, ds._inner.clean_content
    rows = []
    for idx in range(lo, hi):
        item = ds[idx]
        lat = item["gt_latents"]
        _, lesion = renderer.render_structure(
            lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=clean
        )
        lesion = (lesion > 0.5).float()
        crop = lesion[CROP, CROP, CROP]
        if int(crop.sum()) != int(lesion.sum()):
            raise RuntimeError(f"{split} subject {idx}: lesion extends outside the {CROP} crop")
        target = F.avg_pool3d(crop[None, None], 2)[0, 0]
        rows.append(
            (
                idx,
                item["image"][0][0][CROP, CROP, CROP].numpy().astype(np.float16),
                item["image"][1][0][CROP, CROP, CROP].numpy().astype(np.float16),
                (target / target.sum()).numpy().astype(np.float32),
                torch.nonzero(lesion).float().mean(0).numpy(),
                lat["z_content"][2:5].numpy().astype(np.float32),
            )
        )
    return rows


def render(cfg, split, n, args):
    jobs = [(os.path.abspath(args.repo), cfg, split, n, lo, min(lo + 25, n)) for lo in range(0, n, 25)]
    with Pool(args.workers) as pool:
        rows = sorted((r for chunk in pool.imap_unordered(_render_chunk, jobs) for r in chunk), key=lambda r: r[0])
    return {
        "t1": np.stack([r[1] for r in rows]),
        "flair": np.stack([r[2] for r in rows]),
        "target": np.stack([r[3] for r in rows]),
        "centroid": np.stack([r[4] for r in rows]),
        "latent": np.stack([r[5] for r in rows]),
    }


def load_or_render(cfg, args):
    cache = os.path.join(args.run_dir, f"supervised_lesion_cache_train{args.train_samples}_val{args.val_samples}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {
            s: {k: z[f"{s}/{k}"] for k in ("t1", "flair", "target", "centroid", "latent")} for s in ("train", "val")
        }
    t0 = time.time()
    data = {"train": render(cfg, "train", args.train_samples, args), "val": render(cfg, "val", args.val_samples, args)}
    np.savez(cache, **{f"{s}/{k}": v for s, d in data.items() for k, v in d.items()})
    print(
        f"rendered {args.train_samples} train + {args.val_samples} val subjects in {time.time() - t0:.0f}s -> {cache}"
    )
    return data


# ---- model -------------------------------------------------------------------------------
class Localizer(nn.Module):
    """48^3 image -> 24^3 map of lesion logits. Plain 3x3x3 convolutions (receptive field 21 voxels)."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        layers = [nn.Conv3d(1, w // 2, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv3d(w // 2, w, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            layers += [nn.Conv3d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv3d(w, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)[:, 0]


def peak_centre(logits, half=2):
    """Argmax cell refined by the softmax-weighted mean of its neighbourhood, in 64^3 voxel coordinates."""
    p = torch.softmax(logits.flatten(), 0).view_as(logits)
    ijk = np.unravel_index(int(torch.argmax(p)), tuple(p.shape))
    sl = tuple(slice(max(c - half, 0), c + half + 1) for c in ijk)
    w = p[sl]
    axes = [torch.arange(s.start, s.start + w.shape[a], dtype=torch.float32) for a, s in enumerate(sl)]
    grids = torch.meshgrid(*axes, indexing="ij")
    cell = torch.stack([(g * w).sum() / w.sum() for g in grids])
    return (2 * cell + 0.5 + CROP.start).numpy()


@torch.no_grad()
def predict(model, X, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i : i + batch].astype(np.float32))[:, None].to(device)
        out += [peak_centre(lg) for lg in model(x).float().cpu()]
    model.train()
    return np.stack(out)


def score(pred, split, radius_vox):
    from eval.identifiability_metrics import cv_probe_r2

    true, lat = split["centroid"], split["latent"]
    err = np.linalg.norm(pred - true, axis=1)
    centroid_r2 = [
        1 - ((pred[:, a] - true[:, a]) ** 2).sum() / ((true[:, a] - true[:, a].mean()) ** 2).sum() for a in range(3)
    ]
    return {
        "median_error_vox": float(np.median(err)),
        "within_one_radius": float((err <= radius_vox).mean()),
        "centroid_r2_xyz": [float(v) for v in centroid_r2],
        "latent_r2_xyz": [cv_probe_r2(pred, lat[:, a])["mean"] for a in range(3)],
    }


def train_view(data, view, args, device, radius_vox):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    X, T = data["train"][view], data["train"]["target"]
    model = Localizer(args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{view}] training {n_params:,}-parameter network on {len(X)} subjects, {args.steps} steps, device {device}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(X), args.batch_size, replace=False)
        x = torch.from_numpy(X[idx].astype(np.float32))[:, None].to(device)
        t = torch.from_numpy(T[idx]).to(device).flatten(1)
        loss = -(t * F.log_softmax(model(x).flatten(1), dim=1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            probe = predict(model, data["val"][view][:100], device)
            err = np.median(np.linalg.norm(probe - data["val"]["centroid"][:100], axis=1))
            print(
                f"[{view}] step {step:4d}  loss {loss.item():.3f}  median error on 100 val subjects {err:5.2f} vox  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return model


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.repo))
    with open(os.path.join(args.run_dir, "settings.json")) as fp:
        cfg = json.load(fp)
    args.val_samples = args.val_samples or cfg.get("num_val_samples", 400)
    radius_vox = 0.1 * (cfg["res"] - 1) / 2  # this trainer's lesion radius is fixed at 0.1
    device = pick_device(args.device)

    data = load_or_render(cfg, args)
    results = {"run_dir": args.run_dir, "placement": cfg.get("synthetic_lesion_placement"), "views": {}}
    ref = score(data["val"]["centroid"], data["val"], radius_vox)
    results["true_centroid_reference"] = {"latent_r2_xyz": ref["latent_r2_xyz"]}
    for view in VIEWS if args.view == "both" else (args.view,):
        model = train_view(data, view, args, device, radius_vox)
        results["views"][view] = score(predict(model, data["val"][view], device), data["val"], radius_vox)

    print(
        f"\n=== supervised lesion localisation, {args.val_samples} held-out val subjects "
        f"(lesion radius {radius_vox:.2f} vox) ==="
    )
    print(
        f"{'view':6s} {'median error':>13s} {'within 1 radius':>16s} {'centroid R² (mean xyz)':>23s} {'latent R² (mean xyz)':>21s}"
    )
    for view, s in results["views"].items():
        print(
            f"{view:6s} {s['median_error_vox']:>9.2f} vox {s['within_one_radius']:>15.1%} "
            f"{np.mean(s['centroid_r2_xyz']):>23.3f} {np.mean(s['latent_r2_xyz']):>21.3f}"
        )
        print(
            f"{'':6s} per axis: centroid {np.round(s['centroid_r2_xyz'], 3).tolist()}  "
            f"latent {np.round(s['latent_r2_xyz'], 3).tolist()}"
        )
    print(f"true centre -> latent reference: {np.round(ref['latent_r2_xyz'], 3).tolist()}")
    out = args.out or os.path.join(args.run_dir, "supervised_lesion.json")
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
