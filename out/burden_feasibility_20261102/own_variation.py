"""How much of each causal-bank factor's OWN variation a warm.py run holds: in its code, in TailCode's pooled features
(the head's input) and in the backbone GAP, each against the run's untrained twin.

A factor's own variation is its exogenous noise in the bank's SCM, recovered exactly from the true latents by inverting
eval.synthetic.synthetic_dataset.sample_content_from_scm with the bank's noise scale and nonlinearity (a root is its own
noise). It is independent of the factor's parents by construction, so a code that reads a parent nonlinearly cannot fake
it. The linear-residual partial R2 of compare_bundles --with-graph has no such guarantee: with leaky_relu links, perfect
knowledge of the parents alone scores on it (about 0.12 for burden on stage 15's bank). Both are printed, and the raw
factor for reference.

Two references per target, from fresh SCM draws (no images, no model):
  no own   what perfect knowledge of every OTHER rendered factor scores (gradient boosting), so a score at or below it
           needs no information of the factor's own. A descendant still carries some of a factor's own noise
           (ventricle's through burden, thickness's and temporal's through their children); burden's almost none.
  linear   what a linear readout of all seven true factors scores: the most a code that holds them linearly can reach
           (own variation stays below 1 because the links are nonlinear).

Reading: if the pooled features hold burden's own variation and the code does not, the 14-unit head drops it; if neither
does, the network never learned it.

For a run with remainder units (warm.py --remainder-units), each factor's own variation is also read from the main units
alone and from the remainder units alone, so it shows which part of the code holds it.

    python out/burden_feasibility_20261102/own_variation.py OUT_JSON RUN [RUN ...]

Runs already in OUT_JSON at the same step are reused. Each run's bank needs causal_adj.npy (bank.py --causal).
"""

import argparse
import json
import warnings
from pathlib import Path

import bank as B
import causal_bundle
import code_audit
import numpy as np
import torch
import torch.nn.functional as F
import warm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from eval.synthetic.synthetic_dataset import sample_content_from_scm

KINDS = ("code", "pooled", "gap")
SPLIT = ("main", "rest")  # a run with remainder units: the code's main units, then its remainder units
TARGETS = {
    "own": "own variation: the factor's exogenous noise in the SCM (independent of its parents)",
    "partial": "linear-residual partial R2, compare_bundles' target (a parent read nonlinearly scores on it)",
    "raw": "the factor itself",
}
CLIP = 5.0  # standardized features are clipped at this many SD: one outlying subject once wrecked a 448-wide fold
DRAWS = 20000


def clipped(x):
    return np.clip(x, -CLIP, CLIP)


def ridge(x, y):
    """5-fold R2 of a ridge readout, its penalty chosen inside each training fold."""
    pipe = make_pipeline(StandardScaler(), FunctionTransformer(clipped), RidgeCV(alphas=np.logspace(-2, 5, 15)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(pipe, x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def scm_settings():
    """The bank's SCM, its noise scale and its nonlinearity, as bank.py --causal draws them."""
    args = B.run_args(0.4, causal=True)
    return B.causal_scm(args.seed), args.synthetic_causal_noise_scale, args.synthetic_causal_nonlinearity


def mechanism(signal, nonlinearity):
    if nonlinearity == "leaky_relu":
        return F.leaky_relu(signal, 0.2)
    if nonlinearity == "tanh":
        return torch.tanh(signal)
    return signal


def targets(z, scm, scale, nonlinearity, columns):
    """Each factor in ``columns`` three ways (TARGETS), from full latent vectors ``z``."""
    z = np.asarray(z, dtype=np.float64)
    own, partial = z.copy(), z.copy()
    for d in columns:
        parents = scm["parents"][d]
        if parents:
            signal = torch.as_tensor(z[:, parents]) @ scm["weights"][d].double()
            own[:, d] = (z[:, d] - mechanism(signal, nonlinearity).numpy()) / scale
            partial[:, d] -= LinearRegression().fit(z[:, parents], z[:, d]).predict(z[:, parents])
    return {"own": own[:, columns], "partial": partial[:, columns], "raw": z[:, columns]}


def references(scm, scale, nonlinearity, columns):
    """'no own' and 'linear' for every target, from DRAWS fresh SCM draws, fitted on three quarters, scored on the
    rest."""
    gen = torch.Generator().manual_seed(0)
    z = torch.stack([sample_content_from_scm(scm, gen, scale, nonlinearity) for _ in range(DRAWS)]).double().numpy()
    factors = z[:, columns]
    cut = DRAWS * 3 // 4

    def score(model, x, y):
        pred = model.fit(x[:cut], y[:cut]).predict(x[cut:])
        return float(1 - ((y[cut:] - pred) ** 2).sum() / ((y[cut:] - y[cut:].mean()) ** 2).sum())

    def boosted():
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)

    out = {}
    for name, values in targets(z, scm, scale, nonlinearity, columns).items():
        out[name] = {
            "no_own": [score(boosted(), np.delete(factors, j, 1), values[:, j]) for j in range(len(columns))],
            "linear": [score(LinearRegression(), factors, values[:, j]) for j in range(len(columns))],
        }
    return out


def assess(run, device, scm, scale, nonlinearity, columns):
    out = {}
    for trained in (True, False):
        report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
        a = report["args"]
        root = a["bank"]
        if pool is None:
            raise SystemExit(
                f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm"
            )
        if not Path(f"{root}/causal_adj.npy").exists():
            raise SystemExit(f"{root} has no causal_adj.npy: render it with bank.py --causal")
        if not np.array_equal(np.load(f"{root}/causal_adj.npy"), scm["adj"]):
            raise SystemExit(f"{root}'s causal_adj.npy is not bank.causal_scm's graph")
        if not trained:
            causal_bundle.calibrate(model, pool, warm.Bank(root, "train", a["t1key"], residual=residual), device)
        test = warm.Bank(root, "test", a["t1key"], int(a.get("test_subjects", 1000)), residual)
        feats = warm.features(model, test, device, pool=pool)
        main = getattr(pool, "main", pool.units)
        if main < pool.units:
            feats["main"] = [c[:, :main] for c in feats["code"]]
            feats["rest"] = [c[:, main:] for c in feats["code"]]
        values = targets(test.zc, scm, scale, nonlinearity, columns)
        if trained:
            own = values["own"]
            parents = [[columns.index(p) for p in scm["parents"][d]] for d in columns]
            out.update(
                step=step,
                bank=root,
                rows=len(own),
                own_sd=own.std(0).round(3).tolist(),
                own_parent_corr=max(
                    (
                        abs(float(np.corrcoef(own[:, j], values["raw"][:, p])[0, 1]))
                        for j, ps in enumerate(parents)
                        for p in ps
                    ),
                    default=0.0,
                ),
            )
        out["trained" if trained else "untrained"] = {
            view: {
                kind: {
                    name: [ridge(feats[kind][v], y[:, j]) for j in range(len(columns))] for name, y in values.items()
                }
                for kind in KINDS + SPLIT
                if kind in feats
            }
            for v, view in enumerate(("t1", "flair"))
        }
    return out


def show(name, row, refs, scm, columns):
    names = list(B.CAUSAL_FACTORS.values())
    n = row["rows"]
    sd, corr = np.asarray(row["own_sd"]), row["own_parent_corr"]
    inverted = np.abs(sd - 1).max() <= 4 / np.sqrt(2 * n) and corr <= 4 / np.sqrt(n)
    print(
        f"{name}  step {row['step']}, {n} test subjects; own noise SD {sd.min():.2f}-{sd.max():.2f}, max |corr| with a "
        f"parent {corr:.2f}: " + ("the SCM inverts" if inverted else "WARNING: the SCM does not invert, own is wrong")
    )
    cells = [f"{view} {kind}" for view in ("T1", "FLAIR") for kind in KINDS]
    for target, label in TARGETS.items():
        print(f"  {label}; trained (untrained twin)")
        print(
            f"    {'factor':10s} {'parents':18s} {'no own':>6s} {'linear':>6s} |" + "".join(f"{c:>14s}" for c in cells)
        )
        for j, d in enumerate(columns):
            parents = ",".join(B.CAUSAL_FACTORS[p] for p in scm["parents"][d]) or "-"
            values = "".join(
                f"{row['trained'][view][kind][target][j]:6.2f} ({row['untrained'][view][kind][target][j]:5.2f})"
                for view in ("t1", "flair")
                for kind in KINDS
            )
            ref = refs[target]
            print(f"    {names[j]:10s} {parents:18s} {ref['no_own'][j]:6.2f} {ref['linear'][j]:6.2f} |{values}")
    if "main" in row["trained"]["t1"]:
        print("  own variation from each part of the code, main units and remainder units; trained (untrained twin)")
        parts = [f"{view} {kind}" for view in ("T1", "FLAIR") for kind in SPLIT]
        print(f"    {'factor':10s} {'parents':18s} {'':13s} |" + "".join(f"{c:>14s}" for c in parts))
        for j, d in enumerate(columns):
            parents = ",".join(B.CAUSAL_FACTORS[p] for p in scm["parents"][d]) or "-"
            values = "".join(
                f"{row['trained'][view][kind]['own'][j]:6.2f} ({row['untrained'][view][kind]['own'][j]:5.2f})"
                for view in ("t1", "flair")
                for kind in SPLIT
            )
            print(f"    {names[j]:10s} {parents:18s} {'':13s} |{values}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args()
    device = warm.pick_device()
    scm, scale, nonlinearity = scm_settings()
    columns = list(B.CAUSAL_FACTORS)
    results = json.loads(a.out_json.read_text()) if a.out_json.exists() else {}
    for run in a.runs:
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # no checkpoint yet, or one being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        if step != last or results.get(run.name, {}).get("step") == step:
            continue
        print("assessing", run.name, flush=True)
        with torch.no_grad():
            results[run.name] = assess(run, device, scm, scale, nonlinearity, columns)
        a.out_json.write_text(json.dumps(results, indent=1))
    refs = references(scm, scale, nonlinearity, columns)
    print(
        "== Own variation of each causal-bank factor: 5-fold ridge R2 on test subjects. 'no own' = perfect knowledge of"
        " every other factor, boosted;\n   'linear' = a linear readout of all seven true factors; both from"
        f" {DRAWS} fresh SCM draws."
    )
    for name in sorted(results):
        show(name, results[name], refs, scm, columns)


if __name__ == "__main__":
    main()
