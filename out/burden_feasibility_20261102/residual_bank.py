"""Write each view's normative residual beside the bank, for the two-channel arms. No labels are used.

The pipeline residual.py measured: each image z-scored over its own brain, minus the training mean image, minus the
16 largest training PCA directions. Written on the full grid (zero outside the population mask), scaled to unit SD on
training subjects, as {split}_{key}_res.npy. It stands in for models/normative_residual.py so that the input is exactly
what was measured; the module would add a per-voxel SD on calibration subjects.

    python residual_bank.py BANK t1_04 flair [t1_00]
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residual  # noqa: E402

K = 16
SPLITS = ("train", "val", "test")
WORKERS = int(os.environ.get("BANK_WORKERS", 8))


def remove_anatomy(image, mask, mean, basis):
    delta = residual.standard(np.asarray(image, dtype=np.float32), mask) - mean
    return delta - basis @ (basis.T @ delta)


def job(task):
    root, split, key, start, stop, mask, scale = task
    data = np.load(f"{root}/basis_{key}.npz")
    mean, basis = data["mean"], data["basis"][:, :K]
    images = np.load(f"{root}/{split}_{key}.npy", mmap_mode="r")
    out = np.load(f"{root}/{split}_{key}_res.npy", mmap_mode="r+")
    for i in range(start, stop):
        volume = np.zeros(mask.shape, dtype=np.float32)
        volume[mask] = remove_anatomy(images[i], mask, mean, basis) / scale
        out[i] = volume.astype(np.float16)
    out.flush()
    return split, key, start


def main():
    root, keys = sys.argv[1], sys.argv[2:]
    sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
    mask = (sample != 0).mean(0) >= 0.5
    with ProcessPoolExecutor(WORKERS) as pool:
        for key in keys:
            residual.fit_basis(root, key, mask, root)
            data = np.load(f"{root}/basis_{key}.npz")
            images = np.load(f"{root}/train_{key}.npy", mmap_mode="r")
            mean, basis = data["mean"], data["basis"][:, :K]
            scale = float(np.std([remove_anatomy(images[i], mask, mean, basis) for i in range(200)]))
            print(key, "residual SD", round(scale, 4), flush=True)
            jobs = []
            for split in SPLITS:
                n = len(np.load(f"{root}/{split}_vox.npy"))
                np.lib.format.open_memmap(
                    f"{root}/{split}_{key}_res.npy", mode="w+", dtype=np.float16, shape=(n, *mask.shape)
                ).flush()
                jobs += [(root, split, key, s, min(s + 100, n), mask, scale) for s in range(0, n, 100)]
            list(pool.map(job, jobs))
            print("done", key, flush=True)
    print("RESIDUAL DONE", flush=True)


if __name__ == "__main__":
    main()
