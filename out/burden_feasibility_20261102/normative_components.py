"""Does a richer normative model sharpen the lesion in the residual? No training, no network.

residual_bank.py's residual (each scan z-scored over its own brain, minus the training mean image, minus the K largest
training PCA directions of its own modality) at several K, read as precision_audit.py's image-filter row reads it:
detector.py's fixed blob features over the interior, with the volume the other true latents predict handed to the
readout (ridge or boosting, the better), 5-fold on the test subjects. Labels only enter that readout.

Per view and K: e, the reading error in voxels of burden's own part; e in the largest third of lesions; the leftover
SD (the residual averaged over 4-voxel cells, over interior cells with no lesion; the median subject); the lesion's
contrast over that leftover (the load-weighted mean of the cell residual, over the leftover SD); and the share of the
training variance the K directions remove. Then, per K, how well the two views' readings agree and how much of their
error they share.

The PCA here is exact (the bank's stored basis is a randomized 64-direction fit), so K above 64 can be read, and K = 16
should reproduce the bank's own image-filter row.

    python out/burden_feasibility_20261102/normative_components.py BANK OUT_JSON [--test N] [--t1key t1_04]

The exact basis is cached beside OUT_JSON as pca_{bank}_{key}.npy (about 60 MB each). Workers are started fresh
(spawn), each with one BLAS thread: forked after the parent's PCA, they hung on the PC (stage 37, 16 Nov).
"""

import argparse
import json
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector  # noqa: E402

KS = (0, 16, 24, 32, 48, 64, 128)
BURDEN = 2  # z_content's burden slot, as in precision_audit.py
CLIP = 5.0
WORKERS = int(os.environ.get("BANK_WORKERS", 6))


def standard(image, mask):
    brain = image != 0
    return (image[mask] - image[brain].mean()) / image[brain].std()


def fit_basis(bank, key, mask, stem):
    """Exact PCA of the training subjects' standardized scans: the mean, the max(KS) largest directions as rows, and
    every direction's share of the variance. Cached as ``stem``_{key}.npy and its _mean and _spectrum twins."""
    path = f"{stem}_{key}.npy"
    if os.path.exists(path):
        return path
    images = np.load(f"{bank}/train_{key}.npy", mmap_mode="r")
    rows = np.stack([standard(np.asarray(images[i], dtype=np.float32), mask) for i in range(len(images))])
    mean = rows.mean(0)
    rows -= mean
    values, vectors = np.linalg.eigh((rows @ rows.T).astype(np.float64))
    order = np.argsort(values)[::-1][: max(KS)]
    basis = (vectors[:, order].T.astype(np.float32) @ rows) / np.sqrt(values[order])[:, None].astype(np.float32)
    np.save(f"{stem}_{key}_mean.npy", mean)
    np.save(f"{stem}_{key}_spectrum.npy", np.sort(values)[::-1] / values.sum())
    np.save(path, basis.astype(np.float32))
    return path


def one_thread():
    """Each worker's BLAS on one thread: the workers already fill the cores."""
    threadpool_limits(1)


def job(task):
    bank, key, start, stop, mask, interior, stem = task
    basis = np.load(f"{stem}_{key}.npy", mmap_mode="r")
    mean = np.load(f"{stem}_{key}_mean.npy")
    images = np.load(f"{bank}/test_{key}.npy", mmap_mode="r")
    cells = np.load(f"{bank}/test_cells.npy", mmap_mode="r")
    grid = mask.shape[0] // 4
    inside = interior.reshape(grid, 4, grid, 4, grid, 4).mean((1, 3, 5)) == 1  # cells wholly inside the interior
    features, stats = {k: [] for k in KS}, {k: [] for k in KS}
    for i in range(start, stop):
        delta = standard(np.asarray(images[i], dtype=np.float32), mask) - mean
        coefficients = np.asarray(basis @ delta)
        load = np.asarray(cells[i], dtype=np.float64)
        clean = inside & (load == 0)
        done = 0
        for k in KS:
            if k > done:
                delta = delta - basis[done:k].T @ coefficients[done:k]
                done = k
            volume = np.zeros(mask.shape, dtype=np.float32)
            volume[mask] = delta
            features[k].append(detector.features_of(volume, interior))
            pooled = volume.reshape(grid, 4, grid, 4, grid, 4).mean((1, 3, 5))
            stats[k].append([float(pooled[clean].std()), float((pooled * load).sum() / max(load.sum(), 1e-9))])
    return (
        start,
        {k: np.array(v, dtype=np.float32) for k, v in features.items()},
        {k: np.array(v) for k, v in stats.items()},
    )


def clipped(x):
    return np.clip(x, -CLIP, CLIP)


def predictions(x, y, kind):
    """precision_audit.predictions: out-of-fold predictions of y from x, 5-fold on these rows."""
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), FunctionTransformer(clipped), RidgeCV(alphas=np.logspace(-2, 5, 15)))
    else:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(model, x, y, cv=KFold(5, shuffle=True, random_state=0))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def own_volume(bank, vox):
    """precision_audit.own_volume: the volume every other true latent predicts for the test subjects."""
    train = np.load(f"{bank}/train_zc.npy").astype(np.float64)
    test = np.load(f"{bank}/test_zc.npy").astype(np.float64)[: len(vox)]
    others = [j for j in range(train.shape[1]) if j != BURDEN]
    model = make_pipeline(
        StandardScaler(), SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-4, 3, 15))
    )
    model.fit(train[:, others], np.load(f"{bank}/train_vox.npy").astype(np.float64))
    expected = model.predict(test[:, others])
    return expected, r2(expected, vox)


def read(x, vox, expected):
    """precision_audit.assess_stage's precision readout of features ``x``: own R2, the error in voxels, the error in
    the largest third of lesions, and the out-of-fold predictions."""
    own = vox - expected
    given = np.column_stack([x, expected])
    best = max((predictions(given, own, kind) for kind in ("ridge", "boosted")), key=lambda p: r2(p, own))
    value = r2(best, own)
    third = np.array_split(np.argsort(expected, kind="stable"), 3)[2]
    rho = float(np.corrcoef(best[third], own[third])[0, 1] ** 2)
    return {
        "own_r2": value,
        "error": float(own.std() * np.sqrt(1 / value - 1)) if value > 0 else None,
        "largest": float(own[third].std() * np.sqrt(1 / rho - 1)),
    }, best


def voxels(value):
    return "    -" if value is None else f"{value:5.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank")
    p.add_argument("out_json", type=Path)
    p.add_argument("--test", type=int, default=1000, help="test subjects read (1000)")
    p.add_argument("--t1key", default="t1_04")
    a = p.parse_args()
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    stem = str(a.out_json.parent / f"pca_{Path(a.bank).name}")
    sample = np.asarray(np.load(f"{a.bank}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
    occupancy = (sample != 0).mean(0)
    mask = occupancy >= 0.5
    interior = binary_erosion(occupancy >= 0.9, iterations=2)
    vox = np.load(f"{a.bank}/test_vox.npy").astype(np.float64)[: a.test]
    expected, explained = own_volume(a.bank, vox)
    own = vox - expected
    print(
        f"{Path(a.bank).name}: {len(vox)} test subjects; the other latents explain {explained:.2f} of the volume;"
        f" burden's own part SD {own.std():.1f} vox",
        flush=True,
    )
    report = {"bank": a.bank, "rows": len(vox), "own_sd": float(own.std()), "explained": explained, "views": {}}
    guesses = {}
    keys = (a.t1key, "flair")
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=one_thread) as pool:
        for key in keys:
            fit_basis(a.bank, key, mask, stem)
            spectrum = np.load(f"{stem}_{key}_spectrum.npy")
            print(f"  {key}: basis ready, reading {len(vox)} test subjects", flush=True)
            tasks = [(a.bank, key, s, min(s + 25, len(vox)), mask, interior, stem) for s in range(0, len(vox), 25)]
            done = sorted(pool.map(job, tasks), key=lambda r: r[0])
            report["views"][key] = {}
            print(
                f"  {key:6s} {'K':>4s} | {'e':>5s} | {'largest':>7s} | {'own R2':>6s} | {'leftover SD':>11s} |"
                f" {'lesion / leftover':>17s} | {'variance removed':>16s}"
            )
            for k in KS:
                stats = np.concatenate([d[2][k] for d in done])
                row, guesses[key, k] = read(np.concatenate([d[1][k] for d in done]), vox, expected)
                row.update(
                    leftover_sd=float(np.median(stats[:, 0])),
                    lesion_over_leftover=float(np.median(np.abs(stats[:, 1]) / stats[:, 0])),
                    variance_removed=float(spectrum[:k].sum()),
                )
                report["views"][key][str(k)] = row
                print(
                    f"  {'':6s} {k:4d} | {voxels(row['error'])} | {row['largest']:7.1f} | {row['own_r2']:6.2f} |"
                    f" {row['leftover_sd']:11.3f} | {row['lesion_over_leftover']:17.1f} | {row['variance_removed']:16.2f}",
                    flush=True,
                )
    report["agreement"] = {}
    print(f"  {'views':6s} {'K':>4s} | readings agree (r) | errors shared (r)")
    for k in KS:
        first, second = guesses[keys[0], k], guesses[keys[1], k]
        row = {
            "agreement": float(np.corrcoef(first, second)[0, 1]),
            "shared_error": float(np.corrcoef(first - own, second - own)[0, 1]),
        }
        report["agreement"][str(k)] = row
        print(f"  {'':6s} {k:4d} | {row['agreement']:18.2f} | {row['shared_error']:17.2f}", flush=True)
    print(
        "  e: reading error of burden's own part in voxels, the anatomy part handed to the readout; largest: e in the"
        " largest third of lesions;\n  leftover SD: the residual averaged over 4-voxel cells, over interior cells with"
        " no lesion (median subject);\n  lesion / leftover: the load-weighted mean cell residual over that SD."
    )
    a.out_json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
