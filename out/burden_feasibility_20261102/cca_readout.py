"""CCA in place of the InfoNCE head: does a readout that ranks shared directions by how well the two views agree keep
what the 14-unit head drops?

For each warm.py run on a causal bank, and for its untrained twin, per feature source (TailCode's pooled features, the
head's input, and the backbone GAP): a regularised CCA between the T1 and FLAIR features is fitted on the training
subjects, and each view's top k canonical variates on the test subjects form a k-unit code. CCA ranks a direction by its
cross-view correlation alone, whatever its variance and whatever else is kept, and view-specific style does not
correlate across views, so the variates are shared content. Features are standardized on training subjects and clipped
at 5 SD; the ridge on each view's covariance is chosen on a held-out fifth of the training subjects.

Each code is scored as own_variation.py scores the run's own code (each factor's own SCM noise, compare_bundles' partial
and the raw factor), beside the run's InfoNCE code, and the canonical correlations are printed as measured on the test
subjects. --bundles DIR also writes every CCA code as a bundle in the shared format, the untrained twin's as its floor,
for decoded discovery (eval.protocol.compare_bundles --with-graph).

    python out/burden_feasibility_20261102/cca_readout.py OUT_JSON RUN [RUN ...] [--bundles DIR]

Runs already in OUT_JSON at the same step are reused.
"""

import argparse
import json
from pathlib import Path

import bank as B
import causal_bundle
import code_audit
import numpy as np
import own_variation as OV
import torch
import warm

KS = (14, 24)
SOURCES = ("pooled", "gap")
RIDGES = (0.01, 0.1, 1.0, 10.0)
SHOWN = 30  # canonical correlations printed per source


def prepare(train, test):
    """Both splits standardized on the training subjects and clipped, as own_variation.ridge treats its features."""
    mean, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return [np.clip((np.asarray(x, dtype=np.float64) - mean) / sd, -OV.CLIP, OV.CLIP) for x in (train, test)]


def inverse_root(c):
    values, vectors = np.linalg.eigh(c)
    return vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-12))) @ vectors.T


def cca(x1, x2, ridge):
    """Each view's canonical directions (columns, strongest first) for centred x1, x2, and the fitted correlations."""
    n = len(x1)
    w1 = inverse_root(x1.T @ x1 / n + ridge * np.eye(x1.shape[1]))
    w2 = inverse_root(x2.T @ x2 / n + ridge * np.eye(x2.shape[1]))
    u, s, vt = np.linalg.svd(w1 @ (x1.T @ x2 / n) @ w2)
    return w1 @ u, w2 @ vt.T, s


def correlations(x1, x2, a1, a2, k):
    z1, z2 = x1 @ a1[:, :k], x2 @ a2[:, :k]
    return [float(np.corrcoef(z1[:, i], z2[:, i])[0, 1]) for i in range(k)]


def fit(x1, x2, k):
    """The ridge whose top-k directions agree best on a held-out fifth of the training subjects, then CCA on all."""
    order = np.random.default_rng(0).permutation(len(x1))
    held, kept = order[: len(x1) // 5], order[len(x1) // 5 :]
    score = {}
    for ridge in RIDGES:
        a1, a2, _ = cca(x1[kept], x2[kept], ridge)
        score[ridge] = float(np.mean(correlations(x1[held], x2[held], a1, a2, k)))
    ridge = max(score, key=score.get)
    a1, a2, _ = cca(x1, x2, ridge)
    return a1, a2, ridge, score


def assess(run, device, scm, scale, nonlinearity, columns, bundles):
    out = {}
    for trained in (True, False):
        report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
        a = report["args"]
        root, t1key = a["bank"], a["t1key"]
        if pool is None:
            raise SystemExit(
                f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm"
            )
        if not Path(f"{root}/causal_adj.npy").exists():
            raise SystemExit(f"{root} has no causal_adj.npy: render it with bank.py --causal")
        train = warm.Bank(root, "train", t1key, residual=residual)
        if not trained:
            causal_bundle.calibrate(model, pool, train, device)
        test = warm.Bank(root, "test", t1key, int(a.get("test_subjects", 1000)), residual)
        seen, scored = warm.features(model, train, device, pool=pool), warm.features(model, test, device, pool=pool)
        codes, side = {"code": scored["code"]}, {"ridge": {}, "ridge_scores": {}, "canonical_r": {}}
        for source in SOURCES:
            (fit1, test1), (fit2, test2) = (prepare(seen[source][v], scored[source][v]) for v in (0, 1))
            a1, a2, ridge, score = fit(fit1, fit2, max(KS))
            side["ridge"][source], side["ridge_scores"][source] = ridge, score
            side["canonical_r"][source] = correlations(test1, test2, a1, a2, min(SHOWN, a1.shape[1]))
            for k in KS:
                codes[f"cca{k}_{source}"] = [test1 @ a1[:, :k], test2 @ a2[:, :k]]
        values = OV.targets(test.zc, scm, scale, nonlinearity, columns)
        side["scores"] = {
            name: {
                view: {
                    target: [OV.ridge(code[v], y[:, j]) for j in range(len(columns))] for target, y in values.items()
                }
                for v, view in enumerate(("t1", "flair"))
            }
            for name, code in codes.items()
        }
        out["trained" if trained else "untrained"] = side
        if trained:
            out.update(step=step, bank=root, rows=len(test.zc))
        if bundles is not None:
            for name, code in codes.items():
                if name == "code":
                    continue
                path = bundles / f"{run.name}.{name}{'' if trained else '.floor'}.npz"
                extra = {"source_run": str(run), "step": step, "untrained": not trained, "features": name}
                causal_bundle.save(path, test, root, t1key, code, extra)
    return out


def show(name, row):
    factors = list(B.CAUSAL_FACTORS.values())
    print(f"{name}  step {row['step']}, {row['rows']} test subjects")
    for source in SOURCES:
        r = row["trained"]["canonical_r"][source]
        floor = row["untrained"]["canonical_r"][source]
        print(
            f"  {source} CCA, ridge {row['trained']['ridge'][source]} (untrained {row['untrained']['ridge'][source]}),"
            f" canonical r on test subjects, trained / untrained:"
        )
        print("    " + " ".join(f"{v:.2f}" for v in r))
        print("    " + " ".join(f"{v:.2f}" for v in floor))
    names = ["code"] + [f"cca{k}_{source}" for source in SOURCES for k in KS]
    for target, label in OV.TARGETS.items():
        print(f"  {label}; trained (untrained twin)")
        for view in ("t1", "flair"):
            print(f"    {view:5s} {'':10s}" + "".join(f"{n:>14s}" for n in names))
            trained, untrained = row["trained"]["scores"], row["untrained"]["scores"]
            for j, factor in enumerate(factors):
                cells = "".join(
                    f"{trained[n][view][target][j]:6.2f} ({untrained[n][view][target][j]:5.2f})" for n in names
                )
                print(f"    {'':5s} {factor:10s}{cells}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--bundles", type=Path, help="Write each CCA code (and its untrained twin's) as a bundle here")
    a = p.parse_args()
    device = warm.pick_device()
    scm, scale, nonlinearity = OV.scm_settings()
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
            results[run.name] = assess(run, device, scm, scale, nonlinearity, columns, a.bundles)
        a.out_json.write_text(json.dumps(results, indent=1))
    print(
        "== CCA codes: each view's top k canonical variates between T1 and FLAIR features, fitted on training subjects;"
        " 5-fold ridge R2 on test subjects"
    )
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
