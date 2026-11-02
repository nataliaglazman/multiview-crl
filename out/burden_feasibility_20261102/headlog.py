"""If a selective feature is simply available in GAP, does the real InfoNCE put burden into the 14 units?

Inputs per view are GAP-type vectors, frozen: the trained SSL backbone's 64-channel GAP, the 40 fixed rectified-LoG
whole-volume means of detector2.py, or both. Only the recipe's head is trained, from random, with the run's global
InfoNCE. No labels in training; labels only score, by 5-fold ridge on the test split.

    python headlog.py BANK T1KEY SSL_CKPT OUT_JSON
"""

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector2  # noqa: E402
import headonly  # noqa: E402
import warm  # noqa: E402

import data.cluster_batches as cb  # noqa: E402


def log_features(root, key, folder):
    cache = f"{folder}/logfeat_{key}.npz"
    if os.path.exists(cache):
        data = np.load(cache)
        return {"train": data["train"], "test": data["test"]}
    with ProcessPoolExecutor(6) as pool:
        cuts = dict(pool.map(detector2.thresholds, [(root, key, None)]))
        out = {s: detector2.table(pool, root, s, key, MASK, cuts[key])["gap_whole"] for s in ("train", "test")}
    np.savez(cache, **out)
    return out


from scipy.ndimage import binary_erosion  # noqa: E402

MASK = None


def main():
    global MASK
    root, t1key, ssl, out_json = sys.argv[1:5]
    folder = os.path.dirname(out_json)
    sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
    MASK = binary_erosion((sample != 0).mean(0) >= 0.9, iterations=2)
    train = warm.Bank(root, "train", t1key)
    test = warm.Bank(root, "test", t1key)
    neighbours = cb.nearest_neighbours(train.zc[:, warm.ANATOMY], 50)
    fixed = [log_features(root, t1key, folder), log_features(root, "flair", folder)]
    learned = headonly.backbone_gap(ssl, root, t1key, f"{folder}/gap_{t1key}_ssl.npz")
    sets = {
        "ssl_gap": learned,
        "log": {s: [fixed[v][s] for v in range(2)] for s in ("train", "test")},
        "ssl_gap+log": {
            s: [np.concatenate([learned[s][v], fixed[v][s]], 1) for v in range(2)] for s in ("train", "test")
        },
    }
    report = {}
    for name, gap in sets.items():
        row = {"input_t1": headonly.probes(gap["test"][0], test), "input_flair": headonly.probes(gap["test"][1], test)}
        print(name, "input ridge t1", row["input_t1"], flush=True)
        print(name, "input ridge flair", row["input_flair"], flush=True)
        for label, shared, uniform in (
            ("neighbour_sep", False, False),
            ("uniform_sep", False, True),
            ("neighbour_shared", True, False),
        ):
            loss, c1, c2 = headonly.train_head(gap, neighbours, shared, uniform)
            row[label] = {
                "loss_last200": round(loss, 4),
                "code_t1": headonly.probes(c1, test),
                "code_flair": headonly.probes(c2, test),
            }
            print(name, label, row[label], flush=True)
        report[name] = row
        with open(out_json, "w") as stream:
            json.dump(report, stream, indent=1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
