"""Can a code read where the fixed-site lesion actually landed? The fixed bank's lesion moves off its site when the
subject's white matter cannot hold it there (bank.py's set_position), so its position is a function of the anatomy.

    python out/burden_feasibility_20261102/lesion_shift.py OUT.json LABEL=BUNDLE... [--subjects 1000] [--cache C.npz]

Re-renders the fixed bank's test subjects with bank.py's own placement (same flags as `bank.py --causal
--lesion-position fixed`; the field and the gain do not move the lesion) and takes each lesion's centre from its load.
The shift is that centre minus FIXED_CENTRE, in voxels. Each bundle (causal_bundle.py's, on the same bank) is checked
row for row against the render, then the shift (x, y, z and its length) is decoded from each view's code by 5-fold
ridge and boosting, beside: the seven true factors (the anatomy fixes the shift, so they are the reference), each factor
alone, and the bundle's untrained twin when `BUNDLE.floor.npz` exists. 'linear beyond the factors' asks whether the
code holds the shift along directions of its own: the code's ridge R2 minus the factors' ridge R2.
"""

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bank as B
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

TARGETS = ("x", "y", "z", "length")


def render(job):
    start, stop, n = job
    import torch

    torch.set_num_threads(1)
    args = B.run_args(0.4, causal=True)
    ds = B.T.make_dataset(args, "test", n)
    ds._inner.scm = B.causal_scm(args.seed)
    B.set_position(ds._inner.renderer, "fixed")
    real = ds._inner.renderer.render_structure
    kept = {}

    def structure(*a, **k):
        tissue, load = real(*a, **k)
        kept["load"] = load
        return tissue, load

    ds._inner.renderer.render_structure = structure
    grid = np.indices((B.RES,) * 3).reshape(3, -1).T.astype(np.float64)
    centres, zc = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(start, stop):
            item = ds[i]
            load = kept["load"].float().cpu().numpy().reshape(-1).astype(np.float64)
            centres.append(load @ grid / load.sum())
            zc.append(np.asarray(item["gt_latents"]["z_content"], dtype=np.float32)[:9])
    return start, np.asarray(centres), np.asarray(zc)


def shifts(n, cache):
    if cache and Path(cache).is_file():
        data = np.load(cache)
        return data["centres"], data["zc"]
    step = max(1, n // (B.WORKERS * 2))
    centres, zc = np.zeros((n, 3)), np.zeros((n, 9), dtype=np.float32)
    with ProcessPoolExecutor(B.WORKERS) as pool:
        for start, c, z in pool.map(render, [(s, min(s + step, n), n) for s in range(0, n, step)]):
            centres[start : start + len(c)], zc[start : start + len(c)] = c, z
    if cache:
        np.savez(cache, centres=centres, zc=zc)
    return centres, zc


def r2(features, y, learner):
    folds = KFold(5, shuffle=True, random_state=0)
    if learner == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 4, 15))
    else:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    predicted = cross_val_predict(model, features, y, cv=folds)
    return float(1 - ((y - predicted) ** 2).mean() / y.var())


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)


def scores(features, target_matrix):
    features = standardized(features)
    return {
        name: {learner: r2(features, target_matrix[:, j], learner) for learner in ("ridge", "boosted")}
        for j, name in enumerate(TARGETS)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("bundles", nargs="+", metavar="LABEL=BUNDLE")
    p.add_argument("--subjects", type=int, default=1000)
    p.add_argument("--cache", help="Save the rendered centres here, or reuse them if the file exists")
    a = p.parse_args()
    centres, zc = shifts(a.subjects, a.cache)
    shift = centres - np.asarray(B.FIXED_CENTRE, dtype=np.float64)
    targets = np.column_stack([shift, np.linalg.norm(shift, axis=1)])
    factors = zc[:, list(B.CAUSAL_FACTORS)].astype(np.float64)
    names = list(B.CAUSAL_FACTORS.values())
    report = {
        "subjects": a.subjects,
        "shift_sd": [float(v) for v in shift.std(0)],
        "on_site": float((np.linalg.norm(shift, axis=1) < 0.5).mean()),
        "factors": scores(factors, targets),
        "single_factor_boosted_length": {
            name: r2(standardized(factors[:, [j]]), targets[:, 3], "boosted") for j, name in enumerate(names)
        },
        "codes": {},
    }
    print(
        f"{a.subjects} test subjects; shift SD x/y/z {' / '.join(f'{v:.2f}' for v in report['shift_sd'])} voxels; "
        f"{report['on_site']:.0%} within half a voxel of the site"
    )
    print(
        "  the shift's length from one factor at a time (boosted R2): "
        + ", ".join(f"{k} {v:.2f}" for k, v in report["single_factor_boosted_length"].items())
    )

    def line(label, got):
        print(f"  {label:42s} " + "  ".join(f"{t} {got[t]['ridge']:+.2f} / {got[t]['boosted']:+.2f}" for t in TARGETS))

    print("\nR2 of the shift, ridge / boosted, 5-fold:")
    line("seven true factors", report["factors"])
    for pair in a.bundles:
        label, path = pair.split("=", 1)
        bundle = np.load(path)
        if not np.allclose(bundle["z_content"], factors, atol=1e-5):
            raise SystemExit(f"{label}: the bundle's rows are not the rendered test subjects")
        floor = Path(path[: -len(".npz")] + ".floor.npz")
        entry = {}
        for view, key in ((1, "T1"), (2, "FLAIR")):
            entry[key] = scores(bundle[f"emb_view{view}"], targets)
            line(f"{label} {key}", entry[key])
            if floor.is_file():
                entry[f"{key} twin"] = scores(np.load(floor)[f"emb_view{view}"], targets)
                line(f"{label} {key} untrained twin", entry[f"{key} twin"])
            beyond = {t: entry[key][t]["ridge"] - report["factors"][t]["ridge"] for t in TARGETS}
            print(f"  {'':42s} linear beyond the factors: " + "  ".join(f"{t} {v:+.2f}" for t, v in beyond.items()))
        report["codes"][label] = entry
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
