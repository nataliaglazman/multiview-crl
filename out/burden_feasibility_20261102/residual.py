"""Does removing population anatomy make the T1 0.4 lesion readable and shared across views, with no labels?

Each image is z-scored over its own brain, the training mean image is subtracted, and the K largest training PCA
directions are removed. No labels. The same fixed blob detector as detector.py then reads the residual volume, and only
the final regressor sees labels, on training subjects; scores are held out. K = 0 is the template residual.

  r2, spearman      burden read from one view
  cross_view_r      correlation of the two views' estimates, the bar InfoNCE needs (~0.7, headrel.py)
  cca_*             the canonical directions between the two views' features, fitted with NO labels: how much burden
                    the most shared directions hold, which is closer to what a cross-view objective can find

    python residual.py BANK OUT_JSON
"""

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector  # noqa: E402

KS = (0, 16, 64)
KEYS = ("t1_04", "t1_00", "flair")
WORKERS = 6


def standard(image, mask):
    brain = image != 0
    return (image[mask] - image[brain].mean()) / image[brain].std()


def fit_basis(bank, key, mask, folder):
    path = f"{folder}/basis_{key}.npz"
    if os.path.exists(path):
        return path
    images = np.load(f"{bank}/train_{key}.npy", mmap_mode="r")
    rows = np.stack([standard(np.asarray(images[i], dtype=np.float32), mask) for i in range(len(images))])
    mean = rows.mean(0)
    _, _, v = torch.svd_lowrank(torch.from_numpy(rows - mean), q=max(KS) + 8, niter=4)
    np.savez(path, mean=mean, basis=v[:, : max(KS)].numpy())
    return path


def job(task):
    bank, split, key, start, stop, mask, interior, basis_path = task
    data = np.load(basis_path)
    mean, basis = data["mean"], data["basis"]
    images = np.load(f"{bank}/{split}_{key}.npy", mmap_mode="r")
    out = {k: [] for k in KS}
    for i in range(start, stop):
        delta = standard(np.asarray(images[i], dtype=np.float32), mask) - mean
        coefficients = basis.T @ delta
        for k in KS:
            residual = delta - basis[:, :k] @ coefficients[:k]
            volume = np.zeros(mask.shape, dtype=np.float32)
            volume[mask] = residual
            out[k].append(detector.features_of(volume, interior))
    return start, {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def table(pool, bank, split, key, mask, interior, basis_path):
    n = len(np.load(f"{bank}/{split}_vox.npy"))
    tasks = [(bank, split, key, s, min(s + 100, n), mask, interior, basis_path) for s in range(0, n, 100)]
    done = sorted(pool.map(job, tasks), key=lambda r: r[0])
    return {k: np.concatenate([r[1][k] for r in done]) for k in KS}


def regress(train, test, y):
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    return model.fit(train, y).predict(test)


def shared_directions(a_train, b_train, a_test, b_test, y_test, count=8):
    """Canonical directions between the views' features, fitted on training rows with no labels."""

    def whiten(x):
        mean, sd = x.mean(0), x.std(0) + 1e-6
        u, s, vt = np.linalg.svd((x - mean) / sd, full_matrices=False)
        keep = s > s[0] * 1e-3
        return lambda z: ((z - mean) / sd) @ vt[keep].T / s[keep], u[:, keep]

    fa, ua = whiten(a_train)
    fb, ub = whiten(b_train)
    left, rho, right = np.linalg.svd(ua.T @ ub, full_matrices=False)
    pa = fa(a_test) @ left[:, :count]
    pb = fb(b_test) @ right.T[:, :count]
    burden = []
    for c in range(count):
        burden.append(
            [float(abs(np.corrcoef(pa[:, c], y_test)[0, 1])), float(abs(np.corrcoef(pb[:, c], y_test)[0, 1]))]
        )
    held = [float(abs(np.corrcoef(pa[:, c], pb[:, c])[0, 1])) for c in range(count)]
    best = int(np.argmax([min(b) for b in burden]))
    return {
        "canonical_r_train": [round(float(r), 3) for r in rho[:count]],
        "canonical_r_test": [round(r, 3) for r in held],
        "corr_with_burden_a_b": [[round(x, 3) for x in b] for b in burden],
        "best_direction": best,
    }


def main():
    bank, out_json = sys.argv[1], sys.argv[2]
    folder = os.path.dirname(os.path.abspath(out_json))
    sample = np.asarray(np.load(f"{bank}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
    occupancy = (sample != 0).mean(0)
    mask = occupancy >= 0.5
    interior = binary_erosion(occupancy >= 0.9, iterations=2)
    print("mask voxels", int(mask.sum()), "interior", int(interior.sum()), flush=True)
    y = {s: np.array(np.load(f"{bank}/{s}_vox.npy"), dtype=np.float64) for s in ("train", "test")}
    features = {}
    report = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for key in KEYS:
            basis_path = fit_basis(bank, key, mask, folder)
            print("basis", key, flush=True)
            features[key] = {s: table(pool, bank, s, key, mask, interior, basis_path) for s in ("train", "test")}
            print("features", key, flush=True)
    estimates = {}
    for key in KEYS:
        estimates[key] = {}
        for k in KS:
            p = regress(features[key]["train"][k], features[key]["test"][k], y["train"])
            estimates[key][k] = p
            report[f"{key}_k{k}"] = {
                "r2": float(1 - ((p - y["test"]) ** 2).sum() / ((y["test"] - y["test"].mean()) ** 2).sum()),
                "spearman": float(spearmanr(p, y["test"]).statistic),
            }
            print(key, k, {a: round(b, 3) for a, b in report[f"{key}_k{k}"].items()}, flush=True)
    for t1 in ("t1_04", "t1_00"):
        for k in KS:
            row = {"cross_view_r": float(np.corrcoef(estimates[t1][k], estimates["flair"][k])[0, 1])}
            row.update(
                shared_directions(
                    features[t1]["train"][k],
                    features["flair"]["train"][k],
                    features[t1]["test"][k],
                    features["flair"]["test"][k],
                    y["test"],
                )
            )
            report[f"{t1}_vs_flair_k{k}"] = row
            print(t1, "k", k, "cross-view r", round(row["cross_view_r"], 3), flush=True)
            print("   canonical test", row["canonical_r_test"], flush=True)
            print("   corr with burden", row["corr_with_burden_a_b"], flush=True)
    with open(out_json, "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
