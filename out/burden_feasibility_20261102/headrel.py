"""How reliable must a burden channel be before the real InfoNCE routes it into the 14 units?

Inputs per view: the trained SSL backbone's 64-channel GAP, plus ONE extra channel. The head is trained from random
with the run's global InfoNCE, no labels, and scored by ridge on test subjects.

The extra channel is either true volume with view-independent noise at cross-view correlation rho, or the honest
blob estimate: the detector's gap_whole features through a regressor fitted with labels, cross-fitted on train so the
training rows are as noisy as test rows. That last one stands for a selective feature a backbone could really hold.

    python headrel.py BANK OUT_JSON
"""

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import headonly  # noqa: E402
import warm  # noqa: E402

import data.cluster_batches as cb  # noqa: E402

FOLDER = os.path.dirname(os.path.abspath(__file__))


def honest(feature, volume):
    """(cross-fitted train estimate, test estimate) of volume from one view's fixed features."""
    make = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    train = cross_val_predict(make(), feature["train"], volume, cv=KFold(5, shuffle=True, random_state=0))
    return train, make().fit(feature["train"], volume).predict(feature["test"])


def arm(job):
    import torch

    torch.set_num_threads(1)
    label, extra, uniform, bank = job
    learned = dict(np.load(f"{FOLDER}/gap_t1_00_ssl.npz"))
    gap = {s: [learned[f"{s}_{v}"] for v in range(2)] for s in ("train", "test")}
    if extra is not None:
        gap = {s: [np.concatenate([gap[s][v], extra[s][v][:, None]], 1) for v in range(2)] for s in gap}
    train = warm.Bank(bank, "train", "t1_00")
    test = warm.Bank(bank, "test", "t1_00")
    neighbours = cb.nearest_neighbours(train.zc[:, warm.ANATOMY], 50)
    loss, c1, c2 = headonly.train_head(gap, neighbours, False, uniform)
    row = {
        "loss_last200": round(loss, 4),
        "code_t1": headonly.probes(c1, test),
        "code_flair": headonly.probes(c2, test),
    }
    print(
        label,
        row["loss_last200"],
        "volume",
        row["code_t1"]["volume"],
        row["code_flair"]["volume"],
        "brain",
        row["code_t1"]["brain"],
        "vent",
        row["code_t1"]["ventricle"],
        flush=True,
    )
    return label, row


def main():
    bank, out_json = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(0)
    y = {s: np.array(np.load(f"{bank}/{s}_vox.npy"), dtype=np.float64) for s in ("train", "test")}
    mean, sd = y["train"].mean(), y["train"].std()
    standard = {s: (v - mean) / sd for s, v in y.items()}
    features = [dict(np.load(f"{FOLDER}/logfeat_{key}.npz")) for key in ("t1_00", "flair")]
    fitted = [honest(f, y["train"]) for f in features]
    blob = {
        "train": [(fitted[v][0] - mean) / sd for v in range(2)],
        "test": [(fitted[v][1] - mean) / sd for v in range(2)],
    }
    report = {
        "blob_cross_view_r_train": float(np.corrcoef(blob["train"][0], blob["train"][1])[0, 1]),
        "blob_cross_view_r_test": float(np.corrcoef(blob["test"][0], blob["test"][1])[0, 1]),
        "blob_r2_test": [
            float(1 - ((blob["test"][v] - standard["test"]) ** 2).mean() / standard["test"].var()) for v in range(2)
        ],
    }
    print(report, flush=True)
    extras = {"none": None, "blob_estimate": blob}
    for rho in (1.0, 0.9, 0.8, 0.7, 0.5):
        keep = np.sqrt(rho)
        extras[f"rho_{rho}"] = {
            s: [keep * standard[s] + np.sqrt(1 - rho) * rng.standard_normal(len(standard[s])) for _ in range(2)]
            for s in ("train", "test")
        }
    jobs = [
        (f"{name}/{batch}", extra, batch == "uniform", bank)
        for name, extra in extras.items()
        for batch in ("neighbour", "uniform")
    ]
    with ProcessPoolExecutor(6) as pool:
        for label, row in pool.map(arm, jobs):
            report[label] = row
            with open(out_json, "w") as stream:
                json.dump(report, stream, indent=1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
