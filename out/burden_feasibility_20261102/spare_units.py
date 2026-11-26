"""Where a code's directions go as its width grows (stage 74, item 63).

Per run and view: the code whitened (every direction counts alike), rotated to own_directions.py's axes (best read
from the seven own parts first). Each direction's variance (1) splits four ways:
  own      R2 from the seven own parts by straight lines (5-fold ridge)
  curve    max(own, R2 from the seven raw factors by 5-fold gradient boosting) - own: a curved function of the factors
  shared   of what the factors leave (the boosted 5-fold residual), the share the other view's residualised directions
           read (5-fold ridge), times that leftover: content both views share outside the factors
  private  the rest
Sums over directions count directions: own + curve + shared + private = width.

For the directions past the seventh in codes of --sources-from units or more, the best single factor and the best
pair of factors by boosting, beside all seven: a harmonic of one factor reads high from that factor alone.

    python spare_units.py CODES_DIR OUT_JSON [--sources-from 12]

CODES_DIR holds anchor_check.py's exports (<run>.npz with code_t1, code_flair, zc, own), named <...>_c<W>_20k[_s<seed>].
"""

import argparse
import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from threadpoolctl import threadpool_limits

FOLDS = KFold(5, shuffle=True, random_state=0)
ALPHAS = np.logspace(-3, 4, 15)
COLUMNS = (0, 1, 2, 5, 6, 7, 8)  # anchor_check.LATENT_COLUMNS
NAMES = ("brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal")
PARTS = ("own", "curve", "shared", "private")
PAIRS = list(itertools.combinations(range(len(NAMES)), 2))


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(0)) / x.std(0)


def r2(y, p):
    return 1 - ((y - p) ** 2).mean(0) / y.var(0)


def boosted(x, y):
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        return cross_val_predict(model, x, y, cv=FOLDS)


def axes(code, own):
    """The code whitened and rotated so its directions run from best to worst read from the own parts."""
    c = code - code.mean(0)
    values, vectors = np.linalg.eigh(np.cov(c.T))
    keep = values > 1e-9 * values.max()
    white = c @ vectors[:, keep] / np.sqrt(values[keep])
    pred = np.asarray(cross_val_predict(RidgeCV(alphas=ALPHAS), own, white, cv=FOLDS))
    vals, rot = np.linalg.eigh(np.cov(pred.T))
    order = np.argsort(vals)[::-1]
    d = white @ rot[:, order]
    return d / d.std(0), np.clip(vals[order], 0, 1)


def analyse(path, sources_from):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    m = re.search(r"_c(\d+)_20k(?:_s(\d+))?$", meta["run"])
    width, seed = int(m[1]), int(m[2] or 42)
    own, raw = standardized(z["own"]), standardized(z["zc"][:, COLUMNS])
    views = {}
    for v in ("t1", "flair"):
        d, own_r2 = axes(z[f"code_{v}"], own)
        fit = np.column_stack([boosted(raw, d[:, j]) for j in range(d.shape[1])])
        views[v] = dict(d=d, own=own_r2, factors=np.clip(r2(d, fit), 0, 1), resid=d - fit)
    out = dict(run=meta["run"], width=width, seed=seed, views={}, sources=[])
    for v, w in (("t1", "flair"), ("flair", "t1")):
        a = views[v]
        any_f = np.maximum(a["own"], a["factors"])
        pred = np.asarray(cross_val_predict(RidgeCV(alphas=ALPHAS), views[w]["resid"], a["resid"], cv=FOLDS))
        shared = (1 - any_f) * np.clip(r2(a["resid"], pred), 0, 1)
        out["views"][v] = dict(
            own=a["own"].tolist(),
            curve=(any_f - a["own"]).tolist(),
            shared=shared.tolist(),
            private=(1 - any_f - shared).tolist(),
        )
        if width >= sources_from:
            for j in range(len(NAMES), width):
                y = a["d"][:, j]
                out["sources"].append(
                    dict(
                        view=v,
                        direction=j + 1,
                        all=float(a["factors"][j]),
                        single=[float(r2(y, boosted(raw[:, [f]], y))) for f in range(len(NAMES))],
                        pair=[float(r2(y, boosted(raw[:, list(p)], y))) for p in PAIRS],
                    )
                )
    return out


def report(rows):
    lines = ["directions per width, mean ± sd over seeds x 2 views:"]
    lines.append("width   " + "  ".join(f"{p:>13}" for p in PARTS))
    for k in sorted({r["width"] for r in rows}):
        cells = [r["views"][v] for r in rows if r["width"] == k for v in ("t1", "flair")]
        sums = {p: np.array([sum(c[p]) for c in cells]) for p in PARTS}
        lines.append(f"{k:5d}   " + "  ".join(f"{sums[p].mean():6.2f} ± {sums[p].std():4.2f}" for p in PARTS))
    lines.append("\nper direction, mean over seeds and views (own / curve / shared / private):")
    for k in sorted({r["width"] for r in rows}):
        cells = [r["views"][v] for r in rows if r["width"] == k for v in ("t1", "flair")]
        mean = {p: np.mean([c[p] for c in cells], 0) for p in PARTS}
        lines.append(
            f"{k:3d}: "
            + "  ".join(
                f"{i + 1}:{mean['own'][i]:.2f}/{mean['curve'][i]:.2f}/{mean['shared'][i]:.2f}/{mean['private'][i]:.2f}"
                for i in range(k)
            )
        )
    src = [s for r in rows for s in r["sources"]]
    if src:
        single = np.array([s["single"] for s in src])
        pair = np.array([s["pair"] for s in src])
        whole = np.array([s["all"] for s in src])
        q = lambda x: f"median {np.median(x):.2f} (quartiles {np.percentile(x, 25):.2f}-{np.percentile(x, 75):.2f})"
        lines.append(f"\n{len(src)} directions past the seventh, read by boosting from:")
        lines.append(f"  all seven factors   {q(whole)}")
        lines.append(f"  best pair           {q(pair.max(1))}")
        lines.append(f"  best single factor  {q(single.max(1))}")
        best_pair = collections.Counter(pair.argmax(1)).most_common(6)
        lines.append(
            "  best pair: " + ", ".join(f"{NAMES[PAIRS[b][0]]}+{NAMES[PAIRS[b][1]]} {c}" for b, c in best_pair)
        )
        best_single = collections.Counter(single.argmax(1)).most_common()
        lines.append("  best single: " + ", ".join(f"{NAMES[b]} {c}" for b, c in best_single))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("codes", type=Path)
    p.add_argument("out_json", type=Path)
    p.add_argument("--sources-from", type=int, default=12)
    a = p.parse_args(argv)
    files = sorted(a.codes.glob("*.npz"))
    rows = Parallel(n_jobs=-1)(delayed(analyse)(f, a.sources_from) for f in files)
    a.out_json.write_text(json.dumps(rows, indent=1))
    print(report(rows))


if __name__ == "__main__":
    main()
