"""Write each view's normative residual beside the bank, for the two-channel arms. No labels are used.

The pipeline residual.py measured: each image z-scored over its own brain, minus the training mean image, minus the
16 largest training PCA directions. Written on the full grid (zero outside the population mask), scaled to unit SD on
training subjects, as {split}_{key}_res.npy. It stands in for models/normative_residual.py so that the input is exactly
what was measured; the module would add a per-voxel SD on calibration subjects.

    python residual_bank.py BANK t1_04 flair [t1_00] [--components K] [--source OTHER_BANK]

--components K removes the K largest directions instead of 16 (normative_components.py reads what each K leaves of the
lesion). The stored basis holds residual.py's 64 directions, so K is at most 64.
--source OTHER_BANK first makes BANK a view of OTHER_BANK: every array of OTHER_BANK but its residuals, and its PCA
basis, are linked into BANK, so only BANK's residuals are new and OTHER_BANK is left as it is. Every tool takes a bank
folder, so a residual with another K is a bank of its own.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residual  # noqa: E402

K = 16
SPLITS = ("train", "val", "test")
WORKERS = int(os.environ.get("BANK_WORKERS", 8))


def remove_anatomy(image, mask, mean, basis):
    delta = residual.standard(np.asarray(image, dtype=np.float32), mask) - mean
    return delta - basis @ (basis.T @ delta)


def link_source(source, root):
    """Link every file of bank ``source`` but its residuals into ``root``; files already there are kept."""
    source, root = Path(source).resolve(), Path(root)
    if not source.is_dir() or source == root.resolve():
        raise SystemExit(f"--source must be another bank folder, got {source}")
    root.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        target = root / path.name
        if path.is_file() and not path.name.endswith("_res.npy") and not os.path.lexists(target):
            target.symlink_to(os.path.relpath(path, root.resolve()))  # relative: the folders can move together


def job(task):
    root, split, key, start, stop, mask, scale, components = task
    data = np.load(f"{root}/basis_{key}.npz")
    mean, basis = data["mean"], data["basis"][:, :components]
    images = np.load(f"{root}/{split}_{key}.npy", mmap_mode="r")
    out = np.load(f"{root}/{split}_{key}_res.npy", mmap_mode="r+")
    for i in range(start, stop):
        volume = np.zeros(mask.shape, dtype=np.float32)
        volume[mask] = remove_anatomy(images[i], mask, mean, basis) / scale
        out[i] = volume.astype(np.float16)
    out.flush()
    return split, key, start


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("keys", nargs="+")
    p.add_argument("--components", type=int, default=K, help="PCA directions removed (16)")
    p.add_argument("--source", default=None, help="Another bank whose arrays and basis are linked into this one first")
    a = p.parse_args(argv)
    root, keys, components = a.root, a.keys, a.components
    if a.source is not None:
        link_source(a.source, root)
    sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
    mask = (sample != 0).mean(0) >= 0.5
    with ProcessPoolExecutor(WORKERS) as pool:
        for key in keys:
            residual.fit_basis(root, key, mask, root)
            data = np.load(f"{root}/basis_{key}.npz")
            images = np.load(f"{root}/train_{key}.npy", mmap_mode="r")
            if not 0 <= components <= data["basis"].shape[1]:
                raise SystemExit(
                    f"--components must be 0 to {data['basis'].shape[1]}, the directions basis_{key} holds"
                )
            mean, basis = data["mean"], data["basis"][:, :components]
            scale = float(np.std([remove_anatomy(images[i], mask, mean, basis) for i in range(min(200, len(images)))]))
            print(key, "components", components, "residual SD", round(scale, 4), flush=True)
            jobs = []
            for split in SPLITS:
                n = len(np.load(f"{root}/{split}_{key}.npy", mmap_mode="r"))  # ADNI banks (adni_bank.py) have no vox
                np.lib.format.open_memmap(
                    f"{root}/{split}_{key}_res.npy", mode="w+", dtype=np.float16, shape=(n, *mask.shape)
                ).flush()
                jobs += [(root, split, key, s, min(s + 100, n), mask, scale, components) for s in range(0, n, 100)]
            list(pool.map(job, jobs))
            print("done", key, flush=True)
    print("RESIDUAL DONE", flush=True)


if __name__ == "__main__":
    main()
