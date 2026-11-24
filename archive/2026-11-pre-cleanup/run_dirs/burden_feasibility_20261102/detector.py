"""A fixed, label-free blob detector on each view: how much burden sits in the image, and how much do the views share?

No training on images and no lesion masks. Scale-normalised LoG at 4 scales, both polarities, inside a population
interior mask; features are the top-k responses and the mass above fixed thresholds. Only the last step, a regression
from those features to volume, sees labels (train subjects), so scores are held out on the test split.

The cross-view correlation between the two views' estimates, BEFORE and AFTER removing true volume, says how much of
the views' agreement is burden and how independent their errors are. That reliability bounds what any objective that
only compares views can be paid for this factor.

    python detector.py BANK OUT_JSON
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_laplace
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

SIGMAS = (1.0, 1.5, 2.0, 3.0)
TOPK = (1, 3, 10, 30, 100, 300)
KEYS = ("t1_00", "t1_04", "flair")
BINS = ((0, 10), (10, 30), (30, 60), (60, 100), (100, 1e9))


def features_of(image, mask):
    out = []
    for sigma in SIGMAS:
        response = sigma**2 * gaussian_laplace(image.astype(np.float32), sigma)[mask]
        for sign in (1, -1):
            values = sign * response
            ordered = np.partition(values, -TOPK[-1])[-TOPK[-1] :]
            ordered.sort()
            ordered = ordered[::-1]
            spread = values.std()
            out.extend(ordered[k - 1] / spread for k in TOPK)
            out.extend(float((values > t * spread).mean()) for t in (3, 4, 5, 6))
    return out


def job(task):
    bank, split, key, start, stop, mask = task
    images = np.load(f"{bank}/{split}_{key}.npy", mmap_mode="r")
    return start, np.array([features_of(np.asarray(images[i]), mask) for i in range(start, stop)], dtype=np.float32)


def table(pool, bank, split, key, mask):
    n = len(np.load(f"{bank}/{split}_vox.npy"))
    tasks = [(bank, split, key, s, min(s + 100, n), mask) for s in range(0, n, 100)]
    rows = sorted(pool.map(job, tasks), key=lambda r: r[0])
    return np.concatenate([r[1] for r in rows])


def scored(prediction, truth):
    out = {
        "r2": float(1 - ((prediction - truth) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()),
        "spearman": float(spearmanr(prediction, truth).statistic),
    }
    for low, high in BINS:
        keep = (truth >= low) & (truth < high)
        out[f"spearman_{int(low)}"] = float(spearmanr(prediction[keep], truth[keep]).statistic)
    return out


def main():
    bank, out_json = sys.argv[1], sys.argv[2]
    flair = np.load(f"{bank}/train_flair.npy", mmap_mode="r")
    sample = np.asarray(flair[:200], dtype=np.float32)
    occupancy = (sample != sample[0, 0, 0, 0]).mean(0)
    mask = binary_erosion(occupancy >= 0.9, iterations=2)
    print("interior voxels", int(mask.sum()), flush=True)
    truth = {s: np.array(np.load(f"{bank}/{s}_vox.npy"), dtype=np.float64) for s in ("train", "test")}
    report = {"interior_voxels": int(mask.sum())}
    predictions = {}
    with ProcessPoolExecutor(6) as pool:
        for key in KEYS:
            train = table(pool, bank, "train", key, mask)
            test = table(pool, bank, "test", key, mask)
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
            model.fit(train, truth["train"])
            predictions[key] = model.predict(test)
            report[key] = scored(predictions[key], truth["test"])
            print(key, report[key], flush=True)
    y = truth["test"]

    def residual(p):
        # Remove what a smooth function of true volume explains, leaving each view's own error.
        coefficients = np.polyfit(y, p, 3)
        return p - np.polyval(coefficients, y)

    for t1 in ("t1_00", "t1_04"):
        a, b = predictions[t1], predictions["flair"]
        rho = float(np.corrcoef(a, b)[0, 1])
        report[f"cross_view_{t1}"] = {
            "r": rho,
            "error_r": float(np.corrcoef(residual(a), residual(b))[0, 1]),
            "nats_if_gaussian": float(-0.5 * np.log(1 - rho**2)),
        }
        print(t1, report[f"cross_view_{t1}"], flush=True)
    with open(out_json, "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main()
