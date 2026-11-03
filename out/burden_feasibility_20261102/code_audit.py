"""What does each self-supervised code hold, and what does it tell subjects apart by?

For each warm.py run, on its bank's test subjects, per view:

  read      5-fold R2 of the six anatomy factors, burden, and the lesion centre (x, y, z of the 16^3 lesion load) from
            the 14-unit code, by ridge and by gradient boosting (a factor held nonlinearly reads only in the second),
            and by ridge from the backbone GAP. The SSL histories never scored the lesion centre.
  shared    canonical correlations between the two views' codes, and for each shared direction the gradient-boosted
            R2 of anatomy, of the lesion (centre and burden) and of both, so shared content no known factor names shows
            as a high correlation with a low R2.
  retrieve  cross-view top-1 among 32 test subjects drawn uniformly, among an anchor and 31 of its 50 nearest on the
            six anatomy factors (the training sampler's rule), and among all test subjects, from the code with each
            view standardized over the test set, by cosine.

Only finished runs are audited (checkpoint at the run's last step), so it can run beside training. Runs already in
OUT_JSON at the same step are not recomputed. Prints a table for every run in OUT_JSON.

    python code_audit.py OUT_JSON RUN_DIR [RUN_DIR ...]
"""

import json
import sys
import warnings
from pathlib import Path

import batch_price
import numpy as np
import torch
import warm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

import data.cluster_batches as cb
import training.main_conv_synthetic as T

NAMES = ["brain", "ventricle", "thickness", "temporal", "lr", "sulcal", "burden", "lesion_x", "lesion_y", "lesion_z"]
SHORT = ["brain", "vent", "thick", "temp", "lr", "sulc", "burden", "x", "y", "z"]
GROUPS = {"anatomy": slice(0, 6), "lesion": slice(6, 10), "all": slice(0, 10)}
SETS = 200


def boosted(x, y):
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(model, x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    keep = sd > 1e-8 * max(float(sd.max()), 1e-30)
    return (x[:, keep] - x[:, keep].mean(0)) / sd[keep]


def canonical(a, b):
    """Canonical correlations and both views' unit-variance canonical variates."""

    def basis(x):
        u, s, _ = np.linalg.svd(standardized(x), full_matrices=False)
        return u[:, s > s[0] * 1e-6]

    ua, ub = basis(a), basis(b)
    p, rho, qt = np.linalg.svd(ua.T @ ub, full_matrices=False)
    scale = np.sqrt(len(a))
    return np.clip(rho, 0, 1), ua @ p * scale, ub @ qt.T * scale


def top1(z1, z2, sets):
    z1 = standardized(z1)
    z2 = standardized(z2)
    z1 /= np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12
    z2 /= np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12
    hits = []
    for s in sets:
        sim = z1[s] @ z2[s].T
        truth = np.arange(len(s))
        hits.append((np.mean(sim.argmax(1) == truth) + np.mean(sim.argmax(0) == truth)) / 2)
    return float(np.mean(hits))


def load_run(run, device):
    report = json.loads((run / "report.json").read_text())
    a = report["args"]
    args = warm.run_args()
    torch.manual_seed(args.model_seed)
    model = T.build_encoder(args)
    residual = a.get("residual") == "True"
    if residual:
        warm.widen_input(model)
    state = torch.load(run / "model.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    pool = None
    kind = a.get("pool", "gap")
    if kind != "gap":
        thresholds = () if kind == "gapnorm" else warm.TailCode.THRESHOLDS
        pool = warm.TailCode(warm.interior_cells(a["bank"]), thresholds=thresholds)
        pool.load_state_dict(state["pool"])
        pool = pool.to(device).eval()
    return report, model.to(device).eval(), pool, residual, int(state["step"])


def audit(run, device):
    report, model, pool, residual, step = load_run(run, device)
    a = report["args"]
    bank = warm.Bank(a["bank"], "test", a["t1key"], int(a.get("test_subjects", 1000)), residual)
    n = len(bank.vox)
    centre = batch_price.lesion_centroid(np.load(f"{a['bank']}/test_cells.npy", mmap_mode="r")[:n])
    targets = np.column_stack([bank.zc[:, warm.ANATOMY], bank.vox, centre])
    feats = warm.features(model, bank, device, pool=pool)
    batches = "uniform" if a.get("uniform") == "True" else "code" if a.get("code_batches", "0") != "0" else "neighbour"
    out = {
        "step": step,
        "setting": f"{a.get('pool', 'gap')}{' residual' if residual else ''} {a['t1key']} {batches} batches"
        + (f" queue {a['queue']}" if a.get("queue", "0") != "0" else ""),
        "final_global_loss": report["history"][-1]["loss"].get("global"),
        "read": {},
    }
    for view, name in enumerate(("t1", "flair")):
        code, gap = feats["code"][view], feats["gap"][view]
        out["read"][name] = {
            "code_ridge": [warm.ridge(code, targets[:, j]) for j in range(len(NAMES))],
            "code_boosted": [boosted(code, targets[:, j]) for j in range(len(NAMES))],
            "gap_ridge": [warm.ridge(gap, targets[:, j]) for j in range(len(NAMES))],
        }
    rho, left, right = canonical(*feats["code"])
    shared = []
    for i in range(len(rho)):
        if rho[i] < 0.7:
            break
        variate = (left[:, i] + right[:, i]) / 2
        shared.append({"rho": float(rho[i]), **{g: boosted(targets[:, s], variate) for g, s in GROUPS.items()}})
    out["shared"] = {"canonical_r": rho.round(4).tolist(), "directions": shared}
    rng = np.random.default_rng(0)
    sampler = cb.NeighbourhoodBatchSampler(cb.nearest_neighbours(bank.zc[:, warm.ANATOMY], 50), 32, 0, 0.0)
    near = []
    while len(near) < SETS:
        near.extend(np.asarray(b) for b in sampler)
    sets = {
        "uniform32": [rng.choice(n, 32, replace=False) for _ in range(SETS)],
        "neighbour32": near[:SETS],
        f"all{n}": [np.arange(n)],
    }
    out["top1"] = {k: top1(*feats["code"], v) for k, v in sets.items()}
    return out


def show(name, row):
    print(f"{name}  step {row['step']}, {row['setting']}, last global loss {row['final_global_loss']}")
    head = " ".join(f"{s:>6s}" for s in SHORT)
    print(f"  {'':18s} {head}")
    for view in ("t1", "flair"):
        for kind in ("code_ridge", "code_boosted", "gap_ridge"):
            values = " ".join(f"{v:+6.2f}" for v in row["read"][view][kind])
            print(f"  {view:5s} {kind:12s} {values}")
    rho = np.asarray(row["shared"]["canonical_r"])
    directions = row["shared"]["directions"]
    strong = [d for d in directions if d["rho"] >= 0.9]
    print(
        f"  shared: {int((rho >= 0.9).sum())} directions with r >= 0.9, {int((rho >= 0.7).sum())} with r >= 0.7; "
        "boosted R2 of each from anatomy / lesion / both:"
    )
    print(
        "   " + "  ".join(f"r {d['rho']:.3f}: {d['anatomy']:.2f}/{d['lesion']:.2f}/{d['all']:.2f}" for d in directions)
    )
    if strong:
        mean = {g: np.mean([d[g] for d in strong]) for g in GROUPS}
        print(
            f"   mean over r >= 0.9: anatomy {mean['anatomy']:.2f} lesion {mean['lesion']:.2f} both {mean['all']:.2f}"
        )
    print("  cross-view top-1: " + "  ".join(f"{k} {v:.3f}" for k, v in row["top1"].items()))


def main():
    out_json, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    device = warm.pick_device()
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    for run in runs:
        if not (run / "model.pt").exists():
            continue
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # a checkpoint being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        if step != last or results.get(run.name, {}).get("step") == step:
            continue
        print("auditing", run.name, flush=True)
        results[run.name] = audit(run, device)
        out_json.write_text(json.dumps(results, indent=1))
    print(
        "== code audit: R2 per factor (lesion x, y, z = centre of the lesion load, never scored before), test subjects"
    )
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
