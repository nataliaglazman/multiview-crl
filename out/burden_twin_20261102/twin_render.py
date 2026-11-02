"""Render every subject twice with the run's generator: as trained, and with the lesion load zeroed.

Same tissue, style, bias field and noise, so a twin differs from its subject only through the lesion.
Saves float16 memmaps under ROOT: {split}_{t1,flair,t1free,flairfree,mask,zc,cells,vox}.npy.
Splits: train and val (the run's), val00 (val at T1 lesion value 0.0, T1 images only).

    python twin_render.py ROOT
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_common  # noqa: E402

import training.main_conv_synthetic as T  # noqa: E402

SPLITS = {"train": ("train", 2000, 0.4), "val": ("val", 400, 0.4), "val00": ("val", 400, 0.0)}
WORKERS = 10
RES = 64


def arrays(root, split, mode="r+"):
    n = SPLITS[split][1]
    shapes = {
        "t1": ((n, RES, RES, RES), np.float16),
        "flair": ((n, RES, RES, RES), np.float16),
        "t1free": ((n, RES, RES, RES), np.float16),
        "flairfree": ((n, RES, RES, RES), np.float16),
        "mask": ((n, RES, RES, RES), np.bool_),
        "zc": ((n, 9), np.float32),
        "cells": ((n, 16, 16, 16), np.float32),
        "vox": ((n,), np.float32),
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
    root, split, start, stop = job
    torch.set_num_threads(1)
    mode, n, t1_value = SPLITS[split]
    args = twin_common.run_args()
    args.synthetic_lesion_t1_value = t1_value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = T.make_dataset(args, mode, n)
        inner = ds._inner
        ds[0]  # fits fixed_reference on the lesioned images, as the run's dataset does
        real = inner.renderer.render_structure
        kept = {}
        state = {"free": False}

        def structure(*a, **k):
            tissue, load = real(*a, **k)
            kept["load"] = load
            return tissue, torch.zeros_like(load) if state["free"] else load

        inner.renderer.render_structure = structure
        out = arrays(root, split)
        for i in range(start, stop):
            state["free"] = False
            item = ds[i]
            load = kept["load"].float()
            state["free"] = True
            twin = ds[i]
            out["t1"][i] = item["image"][0][0].numpy().astype(np.float16)
            out["flair"][i] = item["image"][1][0].numpy().astype(np.float16)
            out["t1free"][i] = twin["image"][0][0].numpy().astype(np.float16)
            out["flairfree"][i] = twin["image"][1][0].numpy().astype(np.float16)
            out["mask"][i] = item["mask"][0][0].numpy() > 0
            out["zc"][i] = np.asarray(item["gt_latents"]["z_content"], dtype=np.float32)[:9]
            out["cells"][i] = F.avg_pool3d(load[None, None], 4)[0, 0].numpy()
            out["vox"][i] = float(load.sum())
    for array in out.values():
        array.flush()
    return split, start, stop


def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    jobs = []
    for split, (_, n, _) in SPLITS.items():
        arrays(root, split, "w+")
        step = max(1, n // WORKERS)
        jobs += [(root, split, s, min(s + step, n)) for s in range(0, n, step)]
    with ProcessPoolExecutor(WORKERS) as pool:
        for done in pool.map(chunk, jobs):
            print("done", done, flush=True)


if __name__ == "__main__":
    main()
