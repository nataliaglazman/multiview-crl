"""Is burden hard because it is small, or because it moves? Same contrast for every factor, one change at a time.

Every subject is rendered twice with one latent at -1 and +1 and all else equal, on the recipe's generator, so each
difference image is that factor's effect alone, free of noise. For burden both renders share the lesion's position.
For other factors the lesion is held near zero, because changing anatomy can relocate it.

  voxels     median voxels changed by more than 0.05 (image units)
  energy     median sum of squared change
  repeat     mean cosine between different subjects' effects, at voxels and on the 16^3 cells the encoder works at:
             1 = the same pattern in everyone, 0 = nothing shared. Unit length, so one huge subject cannot dominate.
  dropped    subjects over 20x the median energy: a placement redraw swapped the whole subject between the renders

    python factor_compare.py BANK OUT_JSON
"""

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import bank as B
import numpy as np
import torch

import training.main_conv_synthetic as T

N = 120
ARMS = {
    "burden": (2, 0.18),
    "temporal_s009": (6, 0.09),
    "temporal_s018": (6, 0.18),
    "ventricle": (1, 0.18),
    "sulcal": (8, 0.18),
    "lr": (7, 0.18),
    "thickness": (5, 0.18),
    "brain": (0, 0.18),
}


def chunk(job):
    name, factor, sigma, start, stop = job
    torch.set_num_threads(1)
    args = B.run_args(0.4)
    args.synthetic_temporal_sigma = sigma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = T.make_dataset(args, "train", 2000)
        ds[0]
        inner = ds._inner
        real = inner._draw_pseudo_mri
        state = {"value": 0.0}

        def draw(seed):
            out = real(seed)
            z = out["z_content"].clone()
            if factor != 2:
                z[2] = -3.0
            z[factor] = state["value"]
            return {**out, "z_content": z}

        inner._draw_pseudo_mri = draw
        rows = []
        for i in range(start, stop):
            state["value"] = -1.0
            low = ds[i]["image"]
            state["value"] = 1.0
            high = ds[i]["image"]
            rows.append([(high[v][0] - low[v][0]).numpy().astype(np.float32) for v in range(2)])
    return np.array(rows, dtype=np.float32)


def pairwise_cosine(flat):
    """Mean cosine between different subjects' effects, with no self terms."""
    unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
    total = unit.sum(0)
    return float((total @ total - len(unit)) / (len(unit) * (len(unit) - 1)))


def summary(effect):
    flat = effect.reshape(len(effect), -1)
    energy = (flat**2).sum(1)
    keep = energy < 20 * np.median(energy)  # a placement redraw swaps the whole subject
    cells = torch.nn.functional.avg_pool3d(torch.from_numpy(effect[keep])[:, None], 4)[:, 0].numpy()
    return {
        "voxels": float(np.median((np.abs(flat[keep]) > 0.05).sum(1))),
        "energy": float(np.median(energy[keep])),
        "redraws_dropped": int((~keep).sum()),
        "repeat_voxels": pairwise_cosine(flat[keep]),
        "repeat_cells": pairwise_cosine(cells.reshape(len(cells), -1)),
    }


def main():
    root, out_json = sys.argv[1], sys.argv[2]
    report = {}
    with ProcessPoolExecutor(8) as pool:
        for name, (factor, sigma) in ARMS.items():
            jobs = [(name, factor, sigma, s, min(s + 15, N)) for s in range(0, N, 15)]
            effect = np.concatenate(list(pool.map(chunk, jobs)))
            report[name] = {view: summary(effect[:, v]) for v, view in enumerate(("t1", "flair"))}
            t, f = report[name]["t1"], report[name]["flair"]
            print(
                f"{name:15s} voxels {t['voxels']:7.0f} {f['voxels']:7.0f} | energy {t['energy']:8.2f} {f['energy']:8.2f} | "
                f"repeat vox {t['repeat_voxels']:6.3f} {f['repeat_voxels']:6.3f} | cells {t['repeat_cells']:6.3f} "
                f"{f['repeat_cells']:6.3f} | dropped {t['redraws_dropped']}",
                flush=True,
            )
    cells = np.asarray(np.load(f"{root}/train_cells.npy", mmap_mode="r")) >= 0.1
    counts = cells.reshape(len(cells), -1).sum(0)
    touched = counts[counts > 0]
    report["lesion_exposure"] = {
        "cells_ever_touched": int((counts > 0).sum()),
        "subjects_per_touched_cell_median": float(np.median(touched)),
        "subjects_per_touched_cell_p95": float(np.percentile(touched, 95)),
        "cells_per_subject_median": float(np.median(cells.reshape(len(cells), -1).sum(1))),
    }
    print(report["lesion_exposure"], flush=True)
    with open(out_json, "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main()
