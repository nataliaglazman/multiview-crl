"""Which directions of the factors' own variation a code holds, and which pairs of factors it blends.

    python out/burden_feasibility_20261102/own_directions.py OUT.json LABEL=BUNDLE...

Bundles are causal_bundle.py's: each factor's own part (its exogenous noise on a causal bank, the factor itself on an
independent one) beside each view's code. Per code and view, from a 5-fold ridge readout of all seven own parts at once
(each standardized), the explained covariance gives:
  - the spectrum: R2 along each direction of the seven own parts' space, best first. A direction near 0 is one the code
    does not hold at all; its weights name the factors it mixes.
  - every pair of factors: R2 along the pair's best and worst directions, beside each factor's own R2. A pair is blended
    when its worst direction falls well below both factors' own R2: the code keeps one mix of the two and drops the
    other mix. 'blend' is the smaller own R2 minus the worst direction's R2.
  - where the code's directions go, every direction counted alike (the code whitened): R2 of each from the own parts by
    straight lines. A direction under .5 is spent on something else; boosting from the seven raw factors, from the
    single best of them and from brain size alone says whether it is a curved copy of factors the code already holds.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from threadpoolctl import threadpool_limits

FOLDS = KFold(5, shuffle=True, random_state=0)


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)


def explained(code, targets):
    """R2 along the principal directions of the targets' space, best first, and each direction's target weights."""
    predicted = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 15)), code, targets, cv=FOLDS)
    predicted = np.asarray(predicted).reshape(targets.shape)
    whiten = np.linalg.inv(np.linalg.cholesky(np.atleast_2d(np.cov(targets.T))))
    residual = np.atleast_2d(np.cov((targets - predicted).T))
    values, vectors = np.linalg.eigh(whiten @ residual @ whiten.T)  # residual share, smallest first
    weights = whiten.T @ vectors
    return 1 - values, weights / np.linalg.norm(weights, axis=0)


def boosted(features, target):
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):  # many small fits; threads only slow them down here
        predicted = cross_val_predict(model, features, target, cv=FOLDS)
    return float(1 - ((target - predicted) ** 2).mean() / target.var())


def directions_spent(code, own, raw, names):
    """Every direction of the code counted alike: R2 of each whitened direction from the own parts by straight lines,
    best first, and for those under .5 what boosting reads them from."""
    centred = code - code.mean(0)
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    keep = values > 1e-9 * values.max()
    white = centred @ vectors[:, keep] / np.sqrt(values[keep])
    predicted = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 15)), own, white, cv=FOLDS)
    r2, axes = np.linalg.eigh(np.cov(np.asarray(predicted).T))
    order = np.argsort(r2)[::-1]
    elsewhere = []
    for k in order[r2[order] < 0.5]:
        direction = white @ axes[:, k]
        singles = {name: boosted(raw[:, [j]], direction) for j, name in enumerate(names)}
        best = max(singles, key=singles.get)
        elsewhere.append(
            dict(
                own_r2=float(r2[k]),
                raw_boosted=boosted(raw, direction),
                best_single=[best, singles[best]],
                brain_boosted=singles["brain"],
            )
        )
    return dict(own_r2_per_direction=[float(v) for v in r2[order]], elsewhere=elsewhere)


def analyse(code, own, names, raw=None):
    code, own = standardized(code), standardized(own)
    spectrum, weights = explained(code, own)
    single = [float(explained(code, own[:, [j]])[0][0]) for j in range(own.shape[1])]
    pairs = []
    for i, j in itertools.combinations(range(own.shape[1]), 2):
        r2, w = explained(code, own[:, [i, j]])
        pairs.append(
            dict(
                pair=[names[i], names[j]],
                best=float(r2[0]),
                worst=float(r2[1]),
                worst_weights=[float(v) for v in w[:, 1]],
                blend=float(min(single[i], single[j]) - r2[1]),
            )
        )
    pairs.sort(key=lambda p: -p["blend"])
    weakest = weights[:, -1] * np.sign(weights[np.argmax(np.abs(weights[:, -1])), -1])
    return dict(
        spectrum=[float(v) for v in spectrum],
        weakest_weights=dict(zip(names, (float(v) for v in weakest))),
        own_r2=dict(zip(names, single)),
        pairs=pairs,
        directions=None if raw is None else directions_spent(code, own, standardized(raw), names),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("bundles", nargs="+", metavar="LABEL=BUNDLE")
    a = p.parse_args()
    report = {}
    for pair in a.bundles:
        label, path = pair.split("=", 1)
        bundle = np.load(path)
        meta = json.loads(str(bundle["meta"]))
        names = meta["content_factor_names"]
        kind = "causal bank, own parts" if "causal_adj" in bundle else "independent bank, factors"
        report[label] = {"bundle": path, "kind": kind}
        print(f"{label} ({kind}; {len(bundle['z_own'])} test subjects)")
        for view, key in ((1, "T1"), (2, "FLAIR")):
            got = analyse(bundle[f"emb_view{view}"], bundle["z_own"], names, bundle["z_content"])
            report[label][key] = got
            weakest = sorted(got["weakest_weights"].items(), key=lambda kv: -abs(kv[1]))
            print(f"  {key:5s} spectrum " + " ".join(f"{v:.2f}" for v in got["spectrum"]))
            print(
                "        weakest direction mixes "
                + ", ".join(f"{name} {w:+.2f}" for name, w in weakest if abs(w) >= 0.25)
            )
            print("        own R2 " + ", ".join(f"{name} {v:.2f}" for name, v in got["own_r2"].items()))
            vb = next(p for p in got["pairs"] if set(p["pair"]) == {"ventricle", "burden"})
            print(
                f"        ventricle-burden plane: best {vb['best']:.2f}, worst {vb['worst']:.2f}, blend {vb['blend']:+.2f}"
            )
            print(
                "        most blended pairs: "
                + "; ".join(
                    f"{'-'.join(p['pair'])} {p['blend']:+.2f} (worst {p['worst']:.2f})" for p in got["pairs"][:3]
                )
            )
            spent = got["directions"]
            held = sum(v >= 0.5 for v in spent["own_r2_per_direction"])
            print(
                f"        directions: {len(spent['own_r2_per_direction'])}, {held} on the own parts (R2 from them "
                + " ".join(f"{v:.2f}" for v in spent["own_r2_per_direction"])
                + ")"
            )
            for d in spent["elsewhere"]:
                name, value = d["best_single"]
                alone = f"from {name} alone {value:.2f}"
                if name != "brain":
                    alone += f", from brain alone {d['brain_boosted']:.2f}"
                print(
                    f"          one at own R2 {d['own_r2']:.2f}: from the seven factors by boosting "
                    f"{d['raw_boosted']:.2f}, {alone}"
                )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
