"""Render the vector-prediction pilot's exact recipe into memmaps, with lesion labels, at chosen T1 lesion values.

Same subjects in every array. fixed_reference comes from the training split and is reused for val and test, as
training.encoder_burden_control does. FLAIR does not depend on the T1 lesion value.

    python out/burden_feasibility_20261102/bank.py BANK_DIR [--t1-values 0.0 0.4] [--radius R] [--temporal-sigma S]
        [--nuisance] [--causal] [--field lesion|smooth|off]
--nuisance turns off --synthetic-clean-content, so the deformation and fissure fields add fine anatomy every subject has.
--field replaces the renderer's per-view multiplicative field (1 + 0.15 x noise drawn on a 16^3 grid, so it varies every
~4 voxels, about a lesion's size; real scanner bias fields vary over centimetres). off: none, as after perfect bias
correction; smooth: the same draw averaged onto a 4^3 grid at the same strength, varying over ~16 voxels. Either way the
renderer draws the same random numbers, so the anatomy, lesion, style and noise of every subject are unchanged.
--causal draws the content latents from an SCM over the factors the renderer uses (CAUSAL_EDGES; z_content[3:5] are
unused with the burden target, so they stay unconnected rather than act as hidden mediators) and writes
causal_adj.npy (9 x 9, parent by child).
Writes {split}_{t1_00,t1_04,flair}.npy (float16), {split}_{zc,vox}.npy, {split}_cells.npy (16^3 mean load).
"""

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import training.encoder_args as encoder_args
import training.main_conv_synthetic as T

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
RES = 64
WORKERS = int(os.environ.get("BANK_WORKERS", 8))
T1_KEYS = {0.0: "t1_00", 0.4: "t1_04"}
# --causal: brain -> ventricle -> burden -> temporal <- thickness <- brain, temporal -> sulcal, and lr on its own.
# Indices are z_content's; every edge runs from a lower to a higher index, the order the sampler draws in.
CAUSAL_FACTORS = {0: "brain", 1: "ventricle", 2: "burden", 5: "thickness", 6: "temporal", 7: "lr", 8: "sulcal"}
CAUSAL_EDGES = ((0, 1), (1, 2), (0, 5), (2, 6), (5, 6), (6, 8))


def causal_scm(seed):
    """build_content_scm's structure on CAUSAL_EDGES: each child is leaky_relu of a unit-norm random weighting of its
    parents, plus noise (eval.synthetic.synthetic_dataset.sample_content_from_scm)."""
    adj = np.zeros((9, 9), dtype=bool)
    for parent, child in CAUSAL_EDGES:
        adj[parent, child] = True
    gen = torch.Generator().manual_seed(seed)
    parents, weights = {}, {}
    for d in range(9):
        parents[d] = np.flatnonzero(adj[:, d]).tolist()
        w = torch.randn(len(parents[d]), generator=gen) if parents[d] else torch.tensor([])
        weights[d] = w / (w.norm() + 1e-8) if parents[d] else w
    return {"adj": adj, "parents": parents, "weights": weights, "n_dims": 9}


def set_field(renderer, field):
    """--field off|smooth on one renderer: its _seeded_noise draws only the multiplicative field."""
    if field == "lesion":
        return

    def noise(scale, gen, device):
        size = renderer.res // scale
        grid = torch.randn(1, 1, size, size, size, generator=gen, device=device)
        if field == "off":
            return torch.zeros((renderer.res,) * 3, device=device)
        grid = F.avg_pool3d(grid, 4) * 8.0
        return F.interpolate(grid, size=(renderer.res,) * 3, mode="trilinear", align_corners=False)[0, 0]

    renderer._seeded_noise = noise


def run_args(t1_value, radius=None, sigma=None, nuisance=False, causal=False):
    flags = (HERE / "reference" / "flags.txt").read_text().split()
    args = encoder_args.parse_args(flags)
    args.cache = False
    args.synthetic_lesion_t1_value = t1_value
    if radius is not None:
        args.synthetic_lesion_radius = radius
    if sigma is not None:
        args.synthetic_temporal_sigma = sigma
    if nuisance:
        args.synthetic_clean_content = False
    if causal:
        args.synthetic_causal = True
    return args


def arrays(root, split, n, t1_values, mode="r+"):
    shapes = {
        **{T1_KEYS[v]: ((n, RES, RES, RES), np.float16) for v in t1_values},
        "flair": ((n, RES, RES, RES), np.float16),
        "zc": ((n, 9), np.float32),
        "vox": ((n,), np.float32),
        "cells": ((n, 16, 16, 16), np.float32),
    }
    out = {}
    for key, (shape, dtype) in shapes.items():
        path = f"{root}/{split}_{key}.npy"
        if mode == "w+":
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()
        else:
            out[key] = np.load(path, mmap_mode=mode)
    return out


def chunk(job):
    root, split, start, stop, options = job
    torch.set_num_threads(1)
    n = options["sizes"][split]
    t1_values = options["t1_values"]
    out = arrays(root, split, n, t1_values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for first, t1_value in zip((True, False), t1_values):
            args = run_args(t1_value, options["radius"], options["sigma"], options["nuisance"], options["causal"])
            train = T.make_dataset(args, "train", options["sizes"]["train"])
            if options["causal"]:
                train._inner.scm = causal_scm(args.seed)
            set_field(train._inner.renderer, options["field"])
            train[0]
            ds = T.make_dataset(args, split, n)
            if options["causal"]:
                ds._inner.scm = causal_scm(args.seed)
            set_field(ds._inner.renderer, options["field"])
            ds[0]
            ds._fixed_mean, ds._fixed_scale = train._fixed_mean, train._fixed_scale
            inner = ds._inner
            real = inner.renderer.render_structure
            kept = {}

            def structure(*a, real=real, kept=kept, **k):
                tissue, load = real(*a, **k)
                kept["load"] = load
                return tissue, load

            inner.renderer.render_structure = structure
            for i in range(start, stop):
                item = ds[i]
                out[T1_KEYS[t1_value]][i] = item["image"][0][0].numpy().astype(np.float16)
                if first:
                    load = kept["load"].float()
                    out["flair"][i] = item["image"][1][0].numpy().astype(np.float16)
                    out["zc"][i] = np.asarray(item["gt_latents"]["z_content"], dtype=np.float32)[:9]
                    out["cells"][i] = F.avg_pool3d(load[None, None], 4)[0, 0].numpy()
                    out["vox"][i] = float(load.sum())
    for array in out.values():
        array.flush()
    return split, start, stop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--t1-values", type=float, nargs="+", default=[0.0, 0.4], choices=sorted(T1_KEYS))
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--temporal-sigma", type=float, default=None)
    p.add_argument("--nuisance", action="store_true")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--field", choices=("lesion", "smooth", "off"), default="lesion")
    p.add_argument("--sizes", type=int, nargs=3, default=[2000, 400, 1000], metavar=("TRAIN", "VAL", "TEST"))
    a = p.parse_args()
    os.makedirs(a.root, exist_ok=True)
    options = {
        "sizes": dict(zip(SPLITS, a.sizes)),
        "t1_values": a.t1_values,
        "radius": a.radius,
        "sigma": a.temporal_sigma,
        "nuisance": a.nuisance,
        "causal": a.causal,
        "field": a.field,
    }
    if a.causal:
        np.save(f"{a.root}/causal_adj.npy", causal_scm(run_args(a.t1_values[0]).seed)["adj"])
    jobs = []
    for split, n in options["sizes"].items():
        arrays(a.root, split, n, a.t1_values, "w+")
        step = max(1, n // (WORKERS * 2))
        jobs += [(a.root, split, s, min(s + step, n), options) for s in range(0, n, step)]
    with ProcessPoolExecutor(WORKERS) as pool:
        for done in pool.map(chunk, jobs):
            print("done", done, flush=True)
    print("BANK DONE", flush=True)


if __name__ == "__main__":
    main()
