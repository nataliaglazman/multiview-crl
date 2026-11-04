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
  mode      top-1 within sets of 32 (uniform, anatomy neighbourhoods, and neighbourhoods on the code itself, the
            code-batch rule), each set standardized over itself as the training loss does, three ways: test subjects
            with the gauge's running statistics (as every score so far), test subjects with each set's own BatchNorm
            statistics (as in training), and as many training subjects with running statistics. Batch above running
            points at the gauge's train/test mismatch; training subjects above test points at memorization.

Only finished runs are audited (checkpoint at the run's last step), so it can run beside training. Runs already in
OUT_JSON at the same step are not recomputed, except that an entry without the mode check gets it added. Prints a table
for every run in OUT_JSON.

    python code_audit.py OUT_JSON RUN_DIR [RUN_DIR ...]
"""

import copy
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
MODE_SETS = 100
LEVERS = ("decorrelate", "group_dropout", "jitter", "rotate")
OVERRIDES = ("patch_weight", "seed")  # named only when a run set them; otherwise the recipe's values


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
        pool = warm.TailCode(warm.interior_cells(a["bank"]), thresholds=thresholds, gauge=a.get("gauge", "batch"))
        pool.load_state_dict(state["pool"])
        pool = pool.to(device).eval()
    return report, model.to(device).eval(), pool, residual, int(state["step"])


def set_top1(z1, z2):
    """Cross-view top-1 within one set, each view standardized over the set as the training loss does, by cosine."""
    z1, z2 = ((z - z.mean(0)) / (z.std(0) + 1e-8) for z in (z1, z2))
    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12)
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12)
    sim = z1 @ z2.T
    truth = np.arange(len(sim))
    return float((np.mean(sim.argmax(1) == truth) + np.mean(sim.argmax(0) == truth)) / 2)


def mode_sets(anatomy, codes, rng):
    """MODE_SETS sets of 32 per rule: uniform, an anchor and 31 of its 50 nearest on anatomy, and the same on the code
    itself (both views)."""
    sets = {"uniform32": [rng.choice(len(anatomy), 32, replace=False) for _ in range(MODE_SETS)]}
    for name, rows in (("neighbour32", anatomy), ("code32", np.concatenate(codes, 1))):
        sampler = cb.NeighbourhoodBatchSampler(cb.nearest_neighbours(rows, 50), 32, 0, 0.0)
        chosen = []
        while len(chosen) < MODE_SETS:
            chosen.extend(np.asarray(b) for b in sampler)
        sets[name] = chosen[:MODE_SETS]
    return sets


@torch.no_grad()
def batch_codes(model, pool, bank, indices, device):
    """Both views' codes for one set, with whatever mode ``pool``'s gauges are in."""
    h = model._encode(bank.images(indices, device), 2, None)
    code = torch.cat([pool.head(p) for p in pool.pooled(h)], 0)[:, :14].cpu().numpy()
    return np.split(code, 2)


def mode_check(report, model, pool, residual, device):
    a = report["args"]
    n = int(a.get("test_subjects", 1000))
    rng = np.random.default_rng(0)
    out = {}
    for split in ("test", "train"):
        bank = warm.Bank(a["bank"], split, a["t1key"], n, residual)
        codes = warm.features(model, bank, device, pool=pool)["code"]
        sets = mode_sets(bank.zc[:, warm.ANATOMY], codes, rng)
        out[f"{split}_running"] = {
            k: float(np.mean([set_top1(codes[0][s], codes[1][s]) for s in v])) for k, v in sets.items()
        }
        if split == "test":
            out["test_batch"] = None
            if pool is not None and isinstance(pool.gauges[0], torch.nn.BatchNorm1d):
                trained = copy.deepcopy(pool).eval()
                for gauge in trained.gauges:
                    gauge.train()
                out["test_batch"] = {
                    k: float(np.mean([set_top1(*batch_codes(model, trained, bank, s, device)) for s in v]))
                    for k, v in sets.items()
                }
    return out


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
        + (f" queue {a['queue']}" if a.get("queue", "0") != "0" else "")
        + (f" gauge {a['gauge']}" if a.get("gauge", "batch") != "batch" else "")
        + "".join(f" {key.replace('_', ' ')} {a[key]}" for key in LEVERS if float(a.get(key, 0)) != 0)
        + (" jitter shared by both views" if a.get("shared_jitter") == "True" else "")
        + "".join(f" {key.replace('_', ' ')} {a[key]}" for key in OVERRIDES if a.get(key, "None") != "None"),
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
    out["mode_check"] = mode_check(report, model, pool, residual, device)
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
    check = row.get("mode_check")
    if check:
        kinds = list(check["test_running"])
        print("  mode check, top-1 within sets of 32, each set standardized over itself as in training:")
        print(f"    {'':38s} " + " ".join(f"{k:>11s}" for k in kinds))
        labels = {
            "test_running": "test, running statistics (as scored)",
            "test_batch": "test, batch statistics (as trained)",
            "train_running": "train subjects, running statistics",
        }
        for key, label in labels.items():
            values = check.get(key)
            cells = " ".join(f"{values[k]:11.3f}" for k in kinds) if values else "  no BatchNorm gauge in this run"
            print(f"    {label:38s} {cells}")


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
        if step != last:
            continue
        cached = results.get(run.name, {})
        if cached.get("step") == step:
            if "mode_check" in cached:
                continue
            print("mode check", run.name, flush=True)
            report, model, pool, residual, _ = load_run(run, device)
            cached["mode_check"] = mode_check(report, model, pool, residual, device)
        else:
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
