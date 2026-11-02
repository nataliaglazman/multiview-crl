"""Read a dense-arm checkpoint at cell level, then ask which pooling turns its map into burden.

Cell detection: AUROC of lesion cells (load >= 0.1) against all other brain cells, top-cell hit rate. Burden: Spearman
and calibrated R2 of volume against the sum of predicted load, plain and rectified at several thresholds, the top-k
mean, and 5-fold ridge from the 64-d backbone GAP and from a histogram of the predicted map.

    python dense_eval.py BANK RUN T1KEY [SUBJECTS]
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sup  # noqa: E402
import warm  # noqa: E402

from eval.protocol.score_checkpoint import build_model, load_settings  # noqa: E402


def calibrated(x, y):
    if np.ptp(x) == 0:
        return 0.0, 0.0
    return float(spearmanr(x, y).statistic), float(np.corrcoef(x, y)[0, 1] ** 2)


def main():
    bank, run, t1key = sys.argv[1:4]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    device = sup.pick_device()
    cfg = load_settings(sup.REFERENCE)
    model = sup.Dense(build_model(cfg, device)).to(device)
    state = torch.load(f"{run}/model.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    data = sup.Bank(bank, "val", t1key, count)
    brain = np.asarray(data.flair[:count]) != 0
    brain_cells = torch.nn.functional.avg_pool3d(torch.from_numpy(brain.astype(np.float32))[:, None], 4)[:, 0].numpy()
    cells = np.asarray(data.cells[:count])
    maps, gaps = [[], []], [[], []]
    with torch.no_grad():
        for start in range(0, count, 16):
            x = data.images(np.arange(start, min(start + 16, count)), device)
            h = model.encoder._encode(x, 2, None)
            out = model(x).cpu().numpy()
            gap = h.mean((2, 3, 4)).cpu().numpy()
            for view, (m, g) in enumerate(zip(np.split(out, 2), np.split(gap, 2))):
                maps[view].append(m)
                gaps[view].append(g)
    maps = [np.concatenate(m) for m in maps]
    gaps = [np.concatenate(g) for g in gaps]
    y = data.vox[:count]
    report = {"step": state["step"]}
    estimates = {}
    for view, name in enumerate(("t1", "flair")):
        p = maps[view]
        inside = brain_cells >= 0.25
        positive = cells >= 0.1
        row = {
            "cell_auroc": float(roc_auc_score(positive[inside], p[inside])),
            "predicted_lesion_cell_mean": float(p[positive & inside].mean()),
            "predicted_other_cell_mean": float(p[inside & ~positive].mean()),
            "predicted_other_cell_sd": float(p[inside & ~positive].std()),
            "predicted_background_mean": float(p[brain_cells == 0].mean()),
        }
        ranked = np.where(inside, p, -np.inf).reshape(count, -1)
        top = ranked.argmax(1)
        has = positive.reshape(count, -1).any(1)
        row["top_cell_hit"] = float(positive.reshape(count, -1)[np.arange(count), top][has].mean())
        pools = {"sum_all": p.sum((1, 2, 3)), "sum_brain": (p * inside).sum((1, 2, 3))}
        for t in (0.02, 0.05, 0.1, 0.2):
            pools[f"sum_above_{t}"] = (np.maximum(p - t, 0) * inside).sum((1, 2, 3))
        ordered = np.sort(ranked, axis=1)[:, ::-1]
        for k in (1, 8, 32):
            pools[f"top{k}"] = ordered[:, :k].mean(1)
        row["pools"] = {k: [round(a, 3) for a in calibrated(v, y)] for k, v in pools.items()}
        row["gap_ridge_volume"] = round(warm.ridge(gaps[view], y), 3)
        estimates[name] = pools
        report[name] = row
        print(name, json.dumps(row), flush=True)
    report["cross_view_r"] = {
        k: round(float(np.corrcoef(estimates["t1"][k], estimates["flair"][k])[0, 1]), 3)
        for k in estimates["t1"]
        if np.ptp(estimates["t1"][k]) > 0 and np.ptp(estimates["flair"][k]) > 0
    }
    print("cross_view_r", json.dumps(report["cross_view_r"]), flush=True)
    with open(f"{run}/dense_eval_s{state['step']}.json", "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    sys.exit(main())
