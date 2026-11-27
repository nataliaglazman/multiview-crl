"""Stage 77 diagnosis (26 Nov): does each run's backbone GAP still carry the factors its 8-unit code reads poorly?
Per run and view: 5-fold R2 on the 1000 test subjects of each causal factor, from the code and from the GAP (RBF
kernel ridge on standardized inputs, grid over alpha and gamma, best mean fold score per factor; slightly optimistic
and the same for every run). Reads the exports in causal/pool/ (pool_check.py export). Run from anywhere:
  python gap_content.py [--runs] > causal/stage77/gap_content.txt"""
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

POOL = Path(__file__).resolve().parent / "causal" / "pool"
PREFIX = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_"
FACTORS = {0: "brain", 1: "ventricle", 2: "burden", 5: "thickness", 6: "temporal", 7: "lr", 8: "sulcal"}
GROUPS = {
    "recipe holds": [f"abl_codebatch_predict1_c8_20k_a100_s{s}" for s in (6, 10, 11, 12, 13)],
    "recipe fails": [f"abl_codebatch_predict1_c8_20k_a100_s{s}" for s in (7, 8, 9, 14, 15)],
    "recipe+term": [f"abl_codebatch_predict1_suff1_c8_20k_a100_s{s}" for s in range(16, 26)],
    "lean": [f"uni_suff1_c8_20k_a100_s{s}" for s in range(16, 26)],
}


def standardized(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def r2(features, targets, folds):
    x, y = standardized(features), standardized(targets)
    best = np.full(y.shape[1], -np.inf)
    for alpha in (0.01, 0.1, 1.0):
        for scale in (0.3, 1.0, 3.0):
            pred = np.zeros_like(y)
            for train, test in folds:
                model = KernelRidge(alpha=alpha, kernel="rbf", gamma=scale / x.shape[1]).fit(x[train], y[train])
                pred[test] = model.predict(x[test])
            best = np.maximum(best, 1 - ((y - pred) ** 2).mean(0) / y.var(0))
    return best


def main():
    folds = list(KFold(5, shuffle=True, random_state=0).split(np.zeros(1000)))
    names = list(FACTORS.values())
    print(f"{'group':14s} {'source':5s} " + " ".join(f"{n:>9s}" for n in names))
    for group, runs in GROUPS.items():
        rows = {"code": [], "gap": []}
        for run in runs:
            d = np.load(POOL / f"{PREFIX}{run}.npz")
            y = d["zc_test"][:, list(FACTORS)]
            for source in rows:
                rows[source].append(np.mean([r2(d[f"{source}_test"][v], y, folds) for v in range(2)], axis=0))
        for source, values in rows.items():
            med = np.median(values, axis=0)
            print(f"{group:14s} {source:5s} " + " ".join(f"{m:9.2f}" for m in med), flush=True)
        if "--runs" in sys.argv:
            for run, c, g in zip(runs, rows["code"], rows["gap"]):
                print(f"    {run.rsplit('_', 1)[1]:4s} code " + " ".join(f"{m:5.2f}" for m in c) + "   gap " + " ".join(f"{m:5.2f}" for m in g))


if __name__ == "__main__":
    main()
