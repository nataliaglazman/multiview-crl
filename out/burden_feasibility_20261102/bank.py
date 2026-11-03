"""Render the vector-prediction pilot's exact recipe into memmaps, with lesion labels, at chosen T1 lesion values.

Same subjects in every array. fixed_reference comes from the training split and is reused for val and test, as
training.encoder_burden_control does. FLAIR does not depend on the T1 lesion value.

    python out/burden_feasibility_20261102/bank.py BANK_DIR [--t1-values 0.0 0.4] [--radius R] [--temporal-sigma S]
        [--nuisance]
--nuisance turns off --synthetic-clean-content, so the deformation and fissure fields add fine anatomy every subject has.
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


def run_args(t1_value, radius=None, sigma=None, nuisance=False):
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
            args = run_args(t1_value, options["radius"], options["sigma"], options["nuisance"])
            train = T.make_dataset(args, "train", options["sizes"]["train"])
            train[0]
            ds = T.make_dataset(args, split, n)
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
    p.add_argument("--sizes", type=int, nargs=3, default=[2000, 400, 1000], metavar=("TRAIN", "VAL", "TEST"))
    a = p.parse_args()
    os.makedirs(a.root, exist_ok=True)
    options = {
        "sizes": dict(zip(SPLITS, a.sizes)),
        "t1_values": a.t1_values,
        "radius": a.radius,
        "sigma": a.temporal_sigma,
        "nuisance": a.nuisance,
    }
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
