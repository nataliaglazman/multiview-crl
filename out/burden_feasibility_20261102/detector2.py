"""Which pooling does a fixed blob filter need to read burden?

The same LoG bank as detector.py, pooled three ways, each a separate feature family (held out on the test split):
  gap_whole     mean over the WHOLE 64^3 volume of ReLU(response - t): literally GAP of a rectified fixed conv
  gap_interior  the same mean over a population interior mask only
  top_interior  the top-k responses inside that mask, which sorts and is not a GAP

t is a fixed constant per scale, polarity and view (99, 99.9, 99.97, 99.99, 99.997th percentile of in-brain training
responses), so nothing is normalised per image.

    python detector2.py BANK OUT_JSON
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_laplace
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

SIGMAS = (1.0, 1.5, 2.0, 3.0)
PERCENTILES = (99, 99.9, 99.97, 99.99, 99.997)
TOPK = (1, 3, 10, 30, 100, 300)
KEYS = ("t1_00", "t1_04", "flair")


def responses(image):
    image = image.astype(np.float32)
    for sigma in SIGMAS:
        yield sigma**2 * gaussian_laplace(image, sigma)


def thresholds(task):
    bank, key, mask = task
    images = np.load(f"{bank}/train_{key}.npy", mmap_mode="r")
    pooled = [[] for _ in SIGMAS]
    for i in range(0, 60):
        image = np.asarray(images[i])
        brain = image != 0
        for s, response in enumerate(responses(image)):
            pooled[s].append(response[brain])
    out = {}
    for s in range(len(SIGMAS)):
        values = np.concatenate(pooled[s])
        out[s] = {sign: np.percentile(sign * values, PERCENTILES) for sign in (1, -1)}
    return key, out


def job(task):
    bank, split, key, start, stop, mask, cuts = task
    images = np.load(f"{bank}/{split}_{key}.npy", mmap_mode="r")
    rows = {"gap_whole": [], "gap_interior": [], "top_interior": []}
    for i in range(start, stop):
        whole, interior, top = [], [], []
        for s, response in enumerate(responses(np.asarray(images[i]))):
            for sign in (1, -1):
                signed = sign * response
                inner = signed[mask]
                for t in cuts[s][sign]:
                    whole.append(float(np.maximum(signed - t, 0).mean()))
                    interior.append(float(np.maximum(inner - t, 0).mean()))
                ordered = np.sort(np.partition(inner, -TOPK[-1])[-TOPK[-1] :])[::-1]
                top.extend(float(ordered[k - 1]) for k in TOPK)
        rows["gap_whole"].append(whole)
        rows["gap_interior"].append(interior)
        rows["top_interior"].append(top)
    return start, {k: np.array(v, dtype=np.float32) for k, v in rows.items()}


def table(pool, bank, split, key, mask, cuts):
    n = len(np.load(f"{bank}/{split}_vox.npy"))
    tasks = [(bank, split, key, s, min(s + 100, n), mask, cuts) for s in range(0, n, 100)]
    done = sorted(pool.map(job, tasks), key=lambda r: r[0])
    return {k: np.concatenate([r[1][k] for r in done]) for k in done[0][1]}


def fit(train, test, y_train, y_test):
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    model.fit(train, y_train)
    p = model.predict(test)
    return p, {
        "r2": float(1 - ((p - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()),
        "spearman": float(spearmanr(p, y_test).statistic),
    }


def main():
    bank, out_json = sys.argv[1], sys.argv[2]
    flair = np.load(f"{bank}/train_flair.npy", mmap_mode="r")
    sample = np.asarray(flair[:200], dtype=np.float32)
    mask = binary_erosion((sample != 0).mean(0) >= 0.9, iterations=2)
    y = {s: np.array(np.load(f"{bank}/{s}_vox.npy"), dtype=np.float64) for s in ("train", "test")}
    report = {}
    predictions = {}
    with ProcessPoolExecutor(6) as pool:
        cuts = dict(pool.map(thresholds, [(bank, key, mask) for key in KEYS]))
        for key in KEYS:
            train = table(pool, bank, "train", key, mask, cuts[key])
            test = table(pool, bank, "test", key, mask, cuts[key])
            report[key] = {}
            predictions[key] = {}
            for family in train:
                predictions[key][family], report[key][family] = fit(train[family], test[family], y["train"], y["test"])
            print(key, report[key], flush=True)
    for family in ("gap_whole", "gap_interior", "top_interior"):
        for t1 in ("t1_00", "t1_04"):
            report[f"cross_view_{t1}_{family}"] = float(
                np.corrcoef(predictions[t1][family], predictions["flair"][family])[0, 1]
            )
    print({k: round(v, 3) for k, v in report.items() if k.startswith("cross")}, flush=True)
    with open(out_json, "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main()
