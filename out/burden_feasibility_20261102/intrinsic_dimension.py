"""Stage 54: the intrinsic dimension of each code, calibrated. No training, and no labels in the estimate.

HANDOFF section 8's optional check. If the code is an embedding of the shared block, its points fill a manifold with
as many dimensions as the block, whatever the code's width: about 10 on the drawn-position bank (seven factors and the
lesion's x, y, z) and 7 on the fixed-position bank. The estimate needs no labels, so it carries to ADNI. It supports
the embedding claim but cannot decide it. Weak factors read noisily look like noise at small scales, and noise adds
dimensions there.

Estimators, on test subjects, each view's code standardized per unit:
  TwoNN (Facco et al. 2017). For each point, the ratio of its second to its first nearest-neighbour distance follows a
    Pareto law whose exponent is the dimension. The estimate is the slope of a line through the origin of
    -log(1 - F(ratio)) on log(ratio), over the 90% smallest ratios. It is repeated on disjoint subsets of N/2, N/4 and
    N/8 subjects. An estimate that rises as N falls is reading noise at small scales; one that holds is the manifold's.
  Levina-Bickel maximum likelihood at k = 10 and 20 neighbours, with MacKay and Ghahramani's averaging.

Rows per run: the code; the code on the unit sphere, as the cosine InfoNCE sees it (one dimension fewer if the norm
carried content); the backbone GAP; and the untrained twin's code and GAP (calibrated as causal_bundle.py does).

Calibration per bank, on the same test subjects (labels build these references only):
  - the true latents: the seven factors, then the seven plus the lesion centre;
  - those ten latents through --embeddings random non-linear maps (64 leaky-ReLU hidden units) into each code width
    present, with reading noise 0, .1 and .3 of each unit's SD;
  - Gaussian noise of each width: what the estimators read at this N when nothing is low-dimensional.

    python out/burden_feasibility_20261102/intrinsic_dimension.py OUT_JSON RUN [RUN ...] [--embeddings 5]

Runs already in OUT_JSON at the same step are reused.
"""

import argparse
import json
from pathlib import Path

import bank as B
import batch_price
import causal_bundle
import code_audit
import numpy as np
import torch
import warm
from sklearn.neighbors import NearestNeighbors

COLUMNS = list(B.CAUSAL_FACTORS)
FRACTIONS = (1, 2, 4, 8)  # TwoNN on disjoint subsets of N / fraction subjects
NEIGHBOURS = (10, 20)  # Levina-Bickel k
NOISE = (0.0, 0.1, 0.3)
COLUMNS_OUT = [f"TwoNN N/{f}" if f > 1 else "TwoNN N" for f in FRACTIONS] + [f"MLE k{k}" for k in NEIGHBOURS]


def neighbour_distances(x, k):
    """(N, k) distances to each point's k nearest other points, nearest first."""
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    return distances[:, 1:]


def two_nn(x, discard=0.1):
    """TwoNN: the slope through the origin of -log(1 - F(mu)) on log(mu), mu = r2 / r1, over the (1 - discard) smallest
    ratios. Points with a duplicate (r1 = 0) are left out."""
    d = neighbour_distances(x, 2)
    d = d[d[:, 0] > 0]
    mu = np.sort(d[:, 1] / d[:, 0])
    keep = int(np.floor(len(mu) * (1 - discard)))
    log_mu = np.log(mu[:keep])
    log_tail = -np.log(1 - np.arange(1, keep + 1) / len(mu))
    return float(log_mu @ log_tail / (log_mu @ log_mu))


def levina_bickel(x, k):
    """The Levina-Bickel estimate at k neighbours, its per-point inverses averaged (MacKay and Ghahramani)."""
    d = neighbour_distances(x, k)
    d = d[d[:, 0] > 0]
    inverse = np.log(d[:, -1:] / d[:, :-1]).sum(1) / (k - 1)
    return float(1 / inverse.mean())


def decimated(x, fraction, seed=0):
    """TwoNN's mean over disjoint random subsets of N / fraction points."""
    if fraction == 1:
        return two_nn(x)
    size = len(x) // fraction
    order = np.random.default_rng(seed).permutation(len(x))
    return float(np.mean([two_nn(x[order[i : i + size]]) for i in range(0, size * fraction, size)]))


def estimates(x):
    """Every column of the table for one (N, d) representation, standardized per unit first."""
    x = code_audit.standardized(x)
    x = x[:, x.any(0)]
    return [decimated(x, f) for f in FRACTIONS] + [levina_bickel(x, k) for k in NEIGHBOURS]


def on_sphere(x):
    x = code_audit.standardized(x)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def embed(latents, width, noise, seed, hidden=64):
    """The latents, standardized, through a random leaky-ReLU map with ``hidden`` units into ``width`` units, each
    standardized, plus Gaussian reading noise of SD ``noise``."""
    rng = np.random.default_rng(seed)
    z = code_audit.standardized(latents)
    h = z @ rng.standard_normal((z.shape[1], hidden)) / np.sqrt(z.shape[1]) + 0.5 * rng.standard_normal(hidden)
    h = np.where(h > 0, h, 0.2 * h)
    y = code_audit.standardized(h @ rng.standard_normal((hidden, width)) / np.sqrt(hidden))
    return y + noise * rng.standard_normal(y.shape)


def calibration(latents7, latents10, widths, embeddings):
    """The reference rows for one bank's test subjects."""
    rows = {"latents (7 factors)": estimates(latents7), "latents (7 + lesion centre)": estimates(latents10)}
    rng = np.random.default_rng(0)
    for width in sorted(widths):
        for noise in NOISE:
            draws = [estimates(embed(latents10, width, noise, seed)) for seed in range(embeddings)]
            rows[f"10 latents -> {width} units, noise {noise:.1f}"] = np.mean(draws, 0).tolist()
        rows[f"Gaussian, {width} units"] = estimates(rng.standard_normal((len(latents10), width)))
    return rows


def codes(run, device, trained):
    """Each view's test code and backbone GAP, for the run or its untrained twin; and the run's bank, step and width."""
    report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
    a = report["args"]
    if pool is None:
        raise SystemExit(f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm")
    if not trained:
        causal_bundle.calibrate(model, pool, warm.Bank(a["bank"], "train", a["t1key"], residual=residual), device)
    test = warm.Bank(a["bank"], "test", a["t1key"], int(a.get("test_subjects", 1000)), residual)
    feats = warm.features(model, test, device, pool=pool)
    return {"code": [c[:, : pool.main] for c in feats["code"]], "gap": feats["gap"]}, a["bank"], step, pool.main


def assess(run, device):
    trained, root, step, width = codes(run, device, True)
    twin, _, _, _ = codes(run, device, False)
    views = []
    for v in range(2):
        views.append(
            {
                "code": estimates(trained["code"][v]),
                "code on the sphere": estimates(on_sphere(trained["code"][v])),
                "backbone GAP": estimates(trained["gap"][v]),
                "twin code": estimates(twin["code"][v]),
                "twin GAP": estimates(twin["gap"][v]),
            }
        )
    return {"step": step, "bank": root, "width": width, "subjects": len(trained["code"][0]), "views": views}


def bank_latents(root, n):
    zc = np.load(f"{root}/test_zc.npy")[:n].astype(np.float64)
    centre = batch_price.lesion_centroid(np.load(f"{root}/test_cells.npy", mmap_mode="r")[:n])
    return zc[:, COLUMNS], np.column_stack([zc[:, COLUMNS], centre])


def line(label, values):
    return f"    {label:34s} " + " ".join(f"{v:9.2f}" for v in values)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--embeddings", type=int, default=5, help="Random non-linear maps per calibration row")
    p.add_argument("--device", default=warm.pick_device())
    a = p.parse_args()
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    runs, banks = results.setdefault("runs", {}), results.setdefault("banks", {})
    for run in a.runs:
        step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
        if runs.get(run.name, {}).get("step") == step:
            print(f"reuse {run.name} (step {step})", flush=True)
            continue
        print(f"{run.name}", flush=True)
        runs[run.name] = assess(run, a.device)
        a.out.write_text(json.dumps(results, indent=1))
    for root in sorted({row["bank"] for row in runs.values()}):
        rows = [row for row in runs.values() if row["bank"] == root]
        n = rows[0]["subjects"]
        widths = sorted({row["width"] for row in rows})
        old = banks.get(root, {})
        if old.get("subjects") != n or old.get("widths") != widths or old.get("embeddings") != a.embeddings:
            print(f"calibration for {root}", flush=True)
            banks[root] = {
                "subjects": n,
                "widths": widths,
                "embeddings": a.embeddings,
                "rows": calibration(*bank_latents(root, n), widths, a.embeddings),
            }
            a.out.write_text(json.dumps(results, indent=1))
    head = " ".join(f"{c:>9s}" for c in COLUMNS_OUT)
    for root, cal in banks.items():
        print(f"{root}: calibration on {cal['subjects']} test subjects")
        print(f"    {'':34s} {head}")
        for label, values in cal["rows"].items():
            print(line(label, values))
        print()
        for name, row in runs.items():
            if row["bank"] != root:
                continue
            print(f"  {name}  step {row['step']}, {row['width']} units")
            for v, view in enumerate(("T1", "FLAIR")):
                print(f"   {view}")
                for label, values in row["views"][v].items():
                    print(line(label, values))
        print()


if __name__ == "__main__":
    main()
