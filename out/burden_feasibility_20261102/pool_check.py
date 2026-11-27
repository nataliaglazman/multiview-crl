"""Stage 75 (no training): merge the runs' codes into one 8-unit code, and read the backbone's leftover past the code.

    python pool_check.py export [RUN...] [--all] [--gauge G]   # GPU: code and GAP of training and test subjects
    python pool_check.py run OUT.json [--restarts 5] [--seed 0] > pool_check.txt
    python pool_check.py detect OUT.json LABEL=RUN... [--stages S...] [--subsets K...] [--rule-k K] [--gauge G]  # 76

At 8 units about half of runs hold all seven own parts. Failing codes lose different directions, and the backbone
still holds what they lose (HANDOFF stage 75). Two label-free uses of that, both resting on what the two views share
being content (style is drawn per view):
  A. merge: per cell of five runs, each run's code is standardized and whitened on its training subjects (both views
     stacked, compress_code.whiten) and the runs' whitened codes are stacked. A linear map to 8 units, shared by the
     views, feeds an MLP that predicts the other view's stacked codes; of several restarts (same held-out subjects)
     the lowest validation loss is kept. The merged code is the linear map's output.
  B. leftover: per code and view, the standardized GAP minus an RBF kernel ridge map from the standardized code (out
     of fold on training subjects). Regularized CCA between the two views' leftovers gives the top shared leftover
     direction and its held-out correlation rho1, read as a failure detector, and a repaired code: the whitened code
     without its most predictable direction (compress_code.redundancy, one step) plus each view's top leftover
     direction.
Fits use training subjects only (HELD_OUT of them for early stopping); scores are own_directions.py's analysis of the
1,000 test subjects, with holds read as holds.py does. Codes, labels and verdicts: causal/stage{64,65,60,70,71,71a}/
holds.txt; own parts from the bundle causal/RUN.npz. The export writes causal/pool/RUN.npz (code and GAP, both views,
both splits), reused while newer than the checkpoint. Rules: HANDOFF stage 75.
--gauge recomputed (stage 78; off by default) reads the code through a copy of the pool whose gauge is recomputed over
the bank's clean training images (warm.recalibrated_pool) in place of the stored running averages, into
causal/pool_recomputed/RUN.npz with both gauges' statistics; detect then reads those.
"""

import argparse
import copy
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from torch import nn

import compress_code
import holds
import own_directions

HERE = Path(__file__).resolve().parent
CAUSAL = HERE / "causal"
POOL = CAUSAL / "pool"
POOL_RECOMPUTED = CAUSAL / "pool_recomputed"
GAUGES = ("stored", "recomputed")
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos"
ARMS = {  # the fixed bank's 8-unit cells, five seeds each
    "e500": "_abl_codebatch_predict1_c8_20k",
    "e100": "_abl_codebatch100_predict1_c8_20k",
    "cont": "_abl_codebatch_predict1_c8_20k_plus10k",
    "red": "_abl_from10red_predict1_c8_10k",
    "pca": "_abl_from10pca_predict1_c8_10k",
    "harm": "_abl_codebatch_predict1_harm1_c8_20k",
    "tau1": "_abl_codebatch_predict1_tau1_c8_20k",
    "proj64": "_abl_codebatch_predict1_proj64_c8_20k",
    "aproj64": "_abl_codebatch_predict1_proj64_c8_20k_a100",
}
STAGES = ("stage64", "stage65", "stage60", "stage70", "stage71", "stage71a")
SEEDS = (42, 2, 3, 4, 5)
NO_REPAIR_RULE = ("tau1",)  # its codes keep 3-4 own directions, so one swap cannot fix them: described only
REFERENCE = "c10_e500_s42"
UNITS = 8
HELD_OUT = 400  # training subjects held out to stop the merge's MLP
CCA_RIDGE = 0.1  # times the mean variance, added to each view's covariance
GAMMAS = (0.25, 0.5, 1.0)  # the RBF's gamma times the code's width, on standardized codes
ALPHAS = (0.01, 0.1, 1.0)
LABEL = re.compile(r"c(\d+)_([a-z0-9]+)_s(\d+)")


def run_name(arm, seed):
    return f"{P}{ARMS[arm]}{'' if int(seed) == 42 else f'_s{seed}'}"


def labels_of(arm):
    return [f"c{UNITS}_{arm}_s{seed}" for seed in SEEDS]


def verdicts(causal=CAUSAL, stages=STAGES):
    """label -> holds, for every code of ARMS, from the stages' holds.txt; a label scored twice must agree."""
    out = {}
    for stage in stages:
        for line in (causal / stage / "holds.txt").read_text().splitlines():
            f = line.split()
            m = LABEL.fullmatch(f[0]) if f else None
            if m and len(f) > 8 and int(m[1]) == UNITS and m[2] in ARMS:
                if out.setdefault(f[0], f[7] == "yes") != (f[7] == "yes"):
                    raise SystemExit(f"{f[0]}: the stages' verdicts differ")
    return out


def own_parts(causal=CAUSAL, stages=STAGES):
    """label -> own_directions' analysis per view, as the stages scored the original codes."""
    out = {}
    for stage in stages:
        path = causal / stage / "own_directions.json"
        if path.exists():
            for label, row in json.loads(path.read_text()).items():
                out.setdefault(label, row)
    return out


def pool_dir(gauge):
    return {"stored": POOL, "recomputed": POOL_RECOMPUTED}[gauge]


def gauge_statistics(pool):
    """Each view's gauge mean and variance, (2, width) each."""
    kept = [(g.running_mean, g.running_var) if isinstance(g, nn.BatchNorm1d) else (g.mean, g.var) for g in pool.gauges]
    return [np.stack([s[i].double().cpu().numpy() for s in kept]) for i in range(2)]


@torch.no_grad()
def export(runs, out_dir=None, gauge="stored"):
    """Each run's code and backbone GAP for its training and test subjects, both views, as (2, n, d) arrays.
    gauge="recomputed": the code through warm.recalibrated_pool's copy of the pool (the gauge recomputed over the
    bank's clean training images), plus both gauges' statistics; the run is left untouched."""
    import code_audit
    import warm

    out_dir = pool_dir(gauge) if out_dir is None else out_dir
    device = warm.pick_device()
    banks = {}
    for run in map(Path, runs):
        out = out_dir / f"{run.name}.npz"
        if out.exists() and out.stat().st_mtime > (run / "model.pt").stat().st_mtime:
            print("reuse", out.name, flush=True)
            continue
        report, model, pool, residual, step = code_audit.load_run(run, device, trained=True)
        if pool is None:
            raise SystemExit(f"{run.name}: the code must come from TailCode's head")
        a = report["args"]
        key = (a["bank"], a["t1key"], residual, int(a.get("test_subjects", 1000)))
        if key not in banks:
            banks[key] = {
                "train": warm.Bank(a["bank"], "train", a["t1key"], residual=residual),
                "test": warm.Bank(a["bank"], "test", a["t1key"], key[3], residual),
            }
        arrays, statistics = {}, {}
        if gauge == "recomputed":
            statistics["stored"] = gauge_statistics(pool)
            pool = warm.recalibrated_pool(model, pool, banks[key]["train"], device)
            statistics["recomputed"] = gauge_statistics(pool)
        for split, bank in banks[key].items():
            feats = warm.features(model, bank, device, pool=pool)
            arrays[f"code_{split}"] = np.stack(feats["code"]).astype(np.float32)
            arrays[f"gap_{split}"] = np.stack(feats["gap"]).astype(np.float32)
        arrays["zc_test"] = np.asarray(banks[key]["test"].zc, dtype=np.float64)
        for name, (mean, var) in statistics.items():
            arrays[f"gauge_{name}_mean"], arrays[f"gauge_{name}_var"] = mean, var
        meta = {"run": run.name, "step": step, "bank": a["bank"], "t1key": a["t1key"], "residual": residual}
        if gauge != "stored":
            meta["gauge"] = gauge
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays, meta=json.dumps(meta))
        print("EXPORT", out.name, {k: v.shape for k, v in arrays.items()}, "step", step, flush=True)
        del model, pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def whitened(train, test):
    """A code (2, n, d) standardized and whitened on its training subjects, both views stacked
    (compress_code.whiten); returns the training and test codes in that frame."""
    mean, sd, white, _ = compress_code.whiten(np.concatenate(train).astype(np.float64))
    return [np.stack([(c - mean) / sd @ white.T for c in split]) for split in (train, test)]


class Bottleneck(nn.Module):
    """A linear map to ``units``, then an MLP back to the stacked codes."""

    def __init__(self, width, units=UNITS, hidden=128):
        super().__init__()
        self.encode = nn.Linear(width, units)
        self.decode = nn.Sequential(
            nn.Linear(units, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, width)
        )

    def forward(self, x):
        return self.decode(self.encode(x))


def cross_view_error(model, x):
    """Mean squared error of each view's prediction of the other view's stacked codes; x is (2, n, width)."""
    return 0.5 * sum(((model(x[v]) - x[1 - v]) ** 2).mean() for v in range(2))


def fit_bottleneck(x, seed, held, units=UNITS, steps=4000, patience=400, lr=2e-3):
    """One restart on training subjects x (2, n, width), stopped on the subjects ``held`` (a boolean mask).
    Returns the model at its best validation loss, and that loss."""
    fit = torch.as_tensor(x[:, ~held], dtype=torch.float32)
    check = torch.as_tensor(x[:, held], dtype=torch.float32)
    torch.manual_seed(seed)
    model = Bottleneck(x.shape[2], units)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, waited = np.inf, None, 0
    for step in range(steps):
        opt.zero_grad()
        cross_view_error(model, fit).backward()
        opt.step()
        if step % 20 == 19:
            with torch.no_grad():
                loss = float(cross_view_error(model, check))
            if loss < best - 1e-6:
                best, state, waited = loss, copy.deepcopy(model.state_dict()), 0
            else:
                waited += 20
                if waited >= patience:
                    break
    model.load_state_dict(state)
    return model, best


def merge(codes_train, codes_test, restarts=5, seed=0, units=UNITS, held_out=HELD_OUT, **fit):
    """Merge runs' codes, given as lists of (2, n, d) training and test arrays. Returns the merged test code
    (2, n_test, units) of the restart with the lowest validation loss, every restart's loss, and every restart's
    merged test code. The held-out subjects are the same for every restart."""
    parts = [whitened(tr, te) for tr, te in zip(codes_train, codes_test)]
    x_train = np.concatenate([p[0] for p in parts], axis=2)
    x_test = torch.as_tensor(np.concatenate([p[1] for p in parts], axis=2), dtype=torch.float32)
    n = x_train.shape[1]
    held = np.isin(np.arange(n), np.random.default_rng(seed).permutation(n)[:held_out])
    fits = [fit_bottleneck(x_train, seed + r, held, units, **fit) for r in range(restarts)]
    with torch.no_grad():
        outputs = [np.stack([m.encode(x_test[v]).numpy() for v in range(2)]) for m, _ in fits]
    losses = [loss for _, loss in fits]
    return outputs[int(np.argmin(losses))], losses, outputs


def standardize(train, *others):
    mean, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return [(x - mean) / sd for x in (train, *others)]


class CodeMap:
    """A map from the code to the GAP: least squares on the code (so a linear function of it leaves no trace, which a
    kernel ridge map alone shrinks), then RBF kernel ridge on what that leaves (curves of what the code holds)."""

    def __init__(self, gamma, alpha):
        self.gamma, self.alpha = gamma, alpha

    def fit(self, code, target):
        design = np.column_stack([np.ones(len(code)), code])
        self.linear = np.linalg.lstsq(design, target, rcond=None)[0]
        self.kernel = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma / code.shape[1])
        self.kernel.fit(code, target - design @ self.linear)
        return self

    def predict(self, code):
        return np.column_stack([np.ones(len(code)), code]) @ self.linear + self.kernel.predict(code)


def kernel_settings(code, target, folds=5):
    """(gamma, alpha) of CodeMap's kernel part, by k-fold CV on these subjects."""
    best, cv = None, KFold(folds, shuffle=True, random_state=0)
    for g, a in itertools.product(GAMMAS, ALPHAS):
        error = 0.0
        for tr, va in cv.split(code):
            error += float(((CodeMap(g, a).fit(code[tr], target[tr]).predict(code[va]) - target[va]) ** 2).sum())
        if best is None or error < best[0]:
            best = (error, g, a)
    return best[1], best[2]


def leftover(code_train, gap_train, code_test, gap_test):
    """One view: the standardized GAP minus CodeMap from the standardized code, out of fold on the training subjects,
    and from the map fitted on all of them on the test subjects. Returns (train, test, settings)."""
    c_tr, c_te = standardize(code_train, code_test)
    g_tr, g_te = standardize(gap_train, gap_test)
    gamma, alpha = kernel_settings(c_tr, g_tr)
    fitted = np.zeros_like(g_tr)
    for tr, va in KFold(5, shuffle=True, random_state=1).split(c_tr):
        fitted[va] = CodeMap(gamma, alpha).fit(c_tr[tr], g_tr[tr]).predict(c_tr[va])
    return g_tr - fitted, g_te - CodeMap(gamma, alpha).fit(c_tr, g_tr).predict(c_te), (gamma, alpha)


def inverse_sqrt(c):
    w, v = np.linalg.eigh(c)
    return (v / np.sqrt(np.maximum(w, 1e-12))) @ v.T


def cca(x, y, ridge=CCA_RIDGE):
    """Regularized CCA of (n, p) x and (n, q) y: the means, and directions a (p, k), b (q, k), best first."""
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    cxx, cyy, cxy = np.cov(xc.T), np.cov(yc.T), xc.T @ yc / (len(x) - 1)
    cxx = cxx + ridge * np.trace(cxx) / len(cxx) * np.eye(len(cxx))
    cyy = cyy + ridge * np.trace(cyy) / len(cyy) * np.eye(len(cyy))
    wx, wy = inverse_sqrt(cxx), inverse_sqrt(cyy)
    u, _, vt = np.linalg.svd(wx @ cxy @ wy)
    k = min(x.shape[1], y.shape[1])
    return mx, my, wx @ u[:, :k], wy @ vt.T[:, :k]


def held_out(x, y, fitted):
    """The fitted CCA's correlations on new subjects, best first as fitted, and both views' variates."""
    mx, my, a, b = fitted
    u, v = (x - mx) @ a, (y - my) @ b
    return np.array([np.corrcoef(u[:, k], v[:, k])[0, 1] for k in range(a.shape[1])]), u, v


def shared_leftover(code_train, gap_train, code_test, gap_test):
    """Per code: each view's leftover, the CCA between them fitted on training subjects, and on test subjects the
    held-out correlations and both views' variates. Also the GAP's own held-out correlations, for comparison."""
    parts = [leftover(code_train[v], gap_train[v], code_test[v], gap_test[v]) for v in range(2)]
    rho, u, v = held_out(parts[0][1], parts[1][1], cca(parts[0][0], parts[1][0]))
    g = [standardize(gap_train[k], gap_test[k]) for k in range(2)]
    gap_rho, _, _ = held_out(g[0][1], g[1][1], cca(g[0][0], g[1][0]))
    return dict(rho=rho, variates=[u, v], gap_rho=gap_rho, settings=[p[2] for p in parts])


def repair(code_train, code_test, variates, seed=0):
    """The whitened code without its most predictable direction (compress_code.redundancy, one step, fitted on the
    training codes) plus each view's given variate. Returns the (2, n_test, d) code, the removed direction's test
    values per view, and its held-out predictability."""
    x_train, x_test = whitened(code_train, code_test)
    n, d = x_train.shape[1], x_train.shape[2]
    basis, log = compress_code.redundancy(np.concatenate(x_train), d - 1, n, seed)
    removed = np.asarray(log[0]["direction"])
    kept = x_test @ basis
    code = np.stack([np.column_stack([kept[v], own_directions.standardized(variates[v])]) for v in range(2)])
    return code, x_test @ removed, log[0]["held_out_r2"]


def holds_of(analysis):
    last = [analysis[v]["spectrum"][-1] for v in holds.VIEWS]
    worst = [holds.pair_worst(analysis[v]) for v in holds.VIEWS]
    return all(x >= holds.HOLD for x in last + worst), last, worst


def score(code, own, names, raw=None):
    """own_directions' analysis of each view of a (2, n, d) code on test subjects (with ``raw``, also where its
    directions go), plus holds, the last values, the ventricle-burden worst values and, with ``raw``, the curves."""
    out = {key: own_directions.analyse(code[v], own, names, raw) for v, key in enumerate(holds.VIEWS)}
    out["holds"], out["last"], out["worst"] = holds_of(out)
    if raw is not None:
        out["curves"] = [sum(d["raw_boosted"] >= 0.5 for d in out[v]["directions"]["elsewhere"]) for v in holds.VIEWS]
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def verdict(count, works, fails):
    return "works" if count >= works else "fails" if count <= fails else "unresolved"


def pair_str(values):
    return "/".join(f"{v:.2f}" for v in values)


def short(row):
    return f"holds {'yes' if row['holds'] else 'no '}  last {pair_str(row['last'])}  worst {pair_str(row['worst'])}"


def misses(row, ref, margin=0.10):
    """Own parts more than ``margin`` below the reference's, as 'factor VIEW value'."""
    out = []
    for view in holds.VIEWS:
        for factor, r2 in ref[view]["own_r2"].items():
            if row[view]["own_r2"][factor] < r2 - margin:
                out.append(f"{factor} {view} {row[view]['own_r2'][factor]:.2f}")
    return out


def light(code, own, names):
    """Holds only: no boosting."""
    s = score(code, own, names)
    return {"holds": s["holds"], "last": s["last"], "worst": s["worst"]}


def run(out_json, restarts, seed, draws):
    held = verdicts()
    originals = own_parts()
    data, missing = {}, []
    for arm in ARMS:
        for label, s in zip(labels_of(arm), SEEDS):
            if label not in held:
                raise SystemExit(f"{label} has no verdict in {STAGES}")
            path = POOL / f"{run_name(arm, s)}.npz"
            if not path.exists():
                missing.append(path.name)
                continue
            data[label] = {k: v for k, v in np.load(path).items() if k != "meta"}
    if missing:
        raise SystemExit(f"run the export first; missing {missing}")
    import bank as B

    bundle = np.load(CAUSAL / f"{run_name('e500', 42)}.npz")
    own, raw = bundle["z_own"].astype(np.float64), bundle["z_content"].astype(np.float64)
    names = json.loads(str(bundle["meta"]))["content_factor_names"]
    for label, d in data.items():
        if not np.allclose(d["zc_test"][:, list(B.CAUSAL_FACTORS)], raw, atol=1e-4):
            raise SystemExit(f"{label}: its test subjects are not the bundle's")
    ref = originals[REFERENCE]
    report = {"verdicts": held, "merge": {}, "leftover": {}}
    rng = np.random.default_rng(seed)

    def merged_of(labels, restarts_here, full):
        code, losses, outputs = merge(
            [data[x]["code_train"] for x in labels], [data[x]["code_test"] for x in labels], restarts_here, seed
        )
        row = score(code, own, names, raw if full else None)
        row["losses"] = losses
        if full:
            row["restarts_hold"] = [light(o, own, names)["holds"] for o in outputs]
        return row

    print("== A. Merge: per cell, the five runs' codes merged into 8 units (rule: holds in >= 7 of the 8 cells with a")
    print("   holder works, <= 4 fails). Holds as holds.py: last spectrum value and ventricle-burden worst >= .40, both")
    print("   views. Curves: directions under .5 own R2 that boosting from the seven factors reads at .5 or more.")
    count = 0
    for arm in ARMS:
        labels = labels_of(arm)
        holders = [x for x in labels if held[x]]
        row = merged_of(labels, restarts, True)
        best = max(holders, key=lambda x: min(originals[x][v]["spectrum"][-1] for v in holds.VIEWS), default=None)
        row["against_best_holder"] = None if best is None else misses(row, originals[best])
        row["against_reference"] = misses(row, ref)
        report["merge"][arm] = row
        count += bool(holders) and row["holds"]
        print(
            f"  {arm:6s} holders {len(holders)}/5 ({' '.join(x.rsplit('_', 1)[1] for x in holders) or '-'}):  "
            f"{short(row)}  curves {pair_str(row['curves']).replace('.00', '')}  restarts holding "
            f"{sum(row['restarts_hold'])}/{len(row['restarts_hold'])}"
        )
        if best is not None:
            print(f"         against its best holder ({best}): {', '.join(row['against_best_holder']) or 'within .10'}")
        print(f"         against {REFERENCE}: {', '.join(row['against_reference']) or 'within .10'}")
    with_holder = sum(any(held[x] for x in labels_of(arm)) for arm in ARMS)
    rule = verdict(count, 7, 4)
    report["merge_rule"] = {"holds": count, "cells": with_holder, "verdict": rule}
    print(f"  RULE: the merged code holds in {count} of the {with_holder} cells with a holder: {rule}")

    print("\n-- described: each cell's failing runs merged alone (3 restarts)")
    report["merge_failures"] = {}
    for arm in ARMS:
        failing = [x for x in labels_of(arm) if not held[x]]
        if len(failing) >= 2:
            row = merged_of(failing, 3, False)
            report["merge_failures"][arm] = row
            print(f"  {arm:6s} {len(failing)} failing runs: {short(row)}")

    print(f"\n-- described: random subsets of 2 and 3 runs per cell ({draws} draws each, 2 restarts)")
    report["merge_subsets"] = {}
    for k in (2, 3):
        rows = []
        for arm in ARMS:
            labels = labels_of(arm)
            for _ in range(draws):
                pick = sorted(rng.choice(len(labels), k, replace=False))
                chosen = [labels[i] for i in pick]
                row = merged_of(chosen, 2, False)
                rows.append({"arm": arm, "runs": chosen, "any_holder": any(held[x] for x in chosen), **row})
        report["merge_subsets"][k] = rows
        print(
            f"  {k} runs: merged holds in {sum(r['holds'] for r in rows)}/{len(rows)}; subsets with a holder "
            f"{sum(r['any_holder'] for r in rows)}/{len(rows)}, of which merged holds in "
            f"{sum(r['holds'] for r in rows if r['any_holder'])}"
        )

    print("\n-- described: the ten projector runs (PC and A100) together, and each seed's PC and A100 runs together")
    ten = labels_of("proj64") + labels_of("aproj64")
    report["merge_projector_ten"] = merged_of(ten, restarts, False)
    print(f"  all ten: {short(report['merge_projector_ten'])}")
    report["merge_same_seed"] = {}
    for s in SEEDS:
        pair = [f"c{UNITS}_proj64_s{s}", f"c{UNITS}_aproj64_s{s}"]
        row = merged_of(pair, 3, False)
        report["merge_same_seed"][s] = row
        both = ", ".join(f"{x} {'holds' if held[x] else 'fails'}" for x in pair)
        print(f"  seed {s:2d} ({both}): {short(row)}")

    print("\n== B. Leftover past the code: per code and view, the GAP minus a kernel ridge map from the code; CCA between")
    print("   the views' leftovers (fitted on training subjects, read on test subjects). rho1: the top direction's")
    print("   held-out correlation. Repair: the code without its most predictable direction, plus the top leftover.")
    rows = {}
    for arm in ARMS:
        for label in labels_of(arm):
            d = data[label]
            s = shared_leftover(d["code_train"], d["gap_train"], d["code_test"], d["gap_test"])
            variates = [s["variates"][v][:, 0] for v in range(2)]
            reads = {
                key: {n: float(np.corrcoef(variates[v], own[:, j])[0, 1] ** 2) for j, n in enumerate(names)}
                for v, key in enumerate(holds.VIEWS)
            }
            code, removed, removed_r2 = repair(d["code_train"], d["code_test"], variates, seed)
            fixed = light(code, own, names)
            std_raw = own_directions.standardized(raw)
            removed_reads = [
                {
                    "factors_boosted": own_directions.boosted(std_raw, own_directions.standardized(removed[v])),
                    "brain_boosted": own_directions.boosted(
                        std_raw[:, [names.index("brain")]], own_directions.standardized(removed[v])
                    ),
                }
                for v in range(2)
            ]
            rows[label] = {
                "arm": arm,
                "holds": held[label],
                "rho": s["rho"][:4].tolist(),
                "gap_rho": s["gap_rho"][:10].tolist(),
                "settings": s["settings"],
                "top_reads": reads,
                "repaired": fixed,
                "removed_predictability": removed_r2,
                "removed_reads": removed_reads,
            }
            top = {key: max(r, key=r.get) for key, r in reads.items()}
            print(
                f"  {label:15s} {'holds' if held[label] else 'fails'}  rho1 {s['rho'][0]:.2f} rho2 {s['rho'][1]:.2f}"
                f"  top reads {top['T1']} {reads['T1'][top['T1']]:.2f} / {top['FLAIR']} "
                f"{reads['FLAIR'][top['FLAIR']]:.2f}  repaired: {short(fixed)}"
            )
            print(
                f"  {'':15s} removed direction: predictable {removed_r2:.2f}, from the factors by boosting "
                f"{pair_str([r['factors_boosted'] for r in removed_reads])}, from brain alone "
                f"{pair_str([r['brain_boosted'] for r in removed_reads])}; GAP's own rho1-8 "
                + " ".join(f"{v:.2f}" for v in s["gap_rho"][:8])
            )
    report["leftover"] = rows
    pos = [r["rho"][0] for r in rows.values() if not r["holds"]]
    neg = [r["rho"][0] for r in rows.values() if r["holds"]]
    pairs = right = 0
    for arm in ARMS:
        cell = [r for r in rows.values() if r["arm"] == arm]
        for h, f in itertools.product([r for r in cell if r["holds"]], [r for r in cell if not r["holds"]]):
            pairs += 1
            right += f["rho"][0] > h["rho"][0]
    value = auc(pos, neg)
    detector = "works" if value >= 0.80 and right >= 0.75 * pairs else "fails" if value <= 0.65 else "unresolved"
    report["detector_rule"] = {"auc": value, "pairs_right": right, "pairs": pairs, "verdict": detector}
    print(
        f"  DETECTOR RULE: AUC of rho1, failures against holders, {value:.2f} over {len(pos)} failures and {len(neg)}"
        f" holders; within-cell pairs on the right side {right}/{pairs}: {detector}"
    )
    ruled = [r for r in rows.values() if r["arm"] not in NO_REPAIR_RULE]
    fixed_failures = sum(r["repaired"]["holds"] for r in ruled if not r["holds"])
    failures = sum(not r["holds"] for r in ruled)
    broken = sum(not r["repaired"]["holds"] for r in ruled if r["holds"])
    holders = sum(r["holds"] for r in ruled)
    repair_rule = (
        "works" if fixed_failures >= 16 and broken <= 1 else "fails" if fixed_failures <= 7 else "unresolved"
    )
    report["repair_rule"] = {
        "failures_holding": fixed_failures,
        "failures": failures,
        "holders_broken": broken,
        "holders": holders,
        "verdict": repair_rule,
    }
    print(
        f"  REPAIR RULE (cells but {', '.join(NO_REPAIR_RULE)}): {fixed_failures} of {failures} failing codes hold after"
        f" the repair; {broken} of {holders} holders stop holding: {repair_rule}"
    )
    described = [r for r in rows.values() if r["arm"] in NO_REPAIR_RULE]
    print(
        f"  described ({', '.join(NO_REPAIR_RULE)}): {sum(r['repaired']['holds'] for r in described)} of"
        f" {len(described)} hold after the repair"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"\nSaved {out_json}")


def stage_verdicts(stages, causal=CAUSAL):
    """label -> holds from the given stages' holds.txt, at any width; a label scored twice must agree."""
    out = {}
    for stage in stages:
        for line in (causal / stage / "holds.txt").read_text().splitlines():
            f = line.split()
            if f and LABEL.fullmatch(f[0]) and len(f) > 8:
                if out.setdefault(f[0], f[7] == "yes") != (f[7] == "yes"):
                    raise SystemExit(f"{f[0]}: the stages' verdicts differ")
    return out


def cell_of(label):
    return label.rsplit("_s", 1)[0]


def selection(rows, k):
    """Every k-subset of ``rows`` (dicts with 'rho1' and 'holds'): how many contain a holder, in how many of those the
    lowest rho1 holds, and a random pick's expected count in the same subsets."""
    with_holder, picked, random_pick = 0, 0, 0.0
    for subset in itertools.combinations(rows, k):
        if any(r["holds"] for r in subset):
            with_holder += 1
            picked += min(subset, key=lambda r: r["rho1"])["holds"]
            random_pick += float(np.mean([r["holds"] for r in subset]))
    return {"subsets": math.comb(len(rows), k), "with_holder": with_holder, "picked": picked, "random": random_pick}


def detect(out_json, pairs, stages=(), ks=(), rule_k=None, pairs_needed=0.8, gauge="stored"):
    """Stage 76: rho1 and holds for LABEL=RUN pairs (RUN a folder or a name under runs/), exporting where needed.
    Holds come from each run's bundle (causal/RUN.npz) by the stages' rule and are checked against ``stages``'
    verdicts. Cells are labels without their seed; pairs and subsets are read within cells. ``gauge`` picks the
    export (stage 78)."""
    import bank as B

    known = stage_verdicts(stages)
    rows = {}
    if gauge != "stored":
        print(f"  codes read with the {gauge} gauge, from {pool_dir(gauge)}", flush=True)
    for pair in pairs:
        label, name = pair.split("=", 1)
        run = Path(name) if "/" in name else HERE / "runs" / name
        path = pool_dir(gauge) / f"{run.name}.npz"
        if not path.exists() or path.stat().st_mtime <= (run / "model.pt").stat().st_mtime:
            export([run], gauge=gauge)
        d = {k: v for k, v in np.load(path).items() if k != "meta"}
        bundle = np.load(CAUSAL / f"{run.name}.npz")
        own, raw = bundle["z_own"].astype(np.float64), bundle["z_content"].astype(np.float64)
        names = json.loads(str(bundle["meta"]))["content_factor_names"]
        if not np.allclose(d["zc_test"][:, list(B.CAUSAL_FACTORS)], raw, atol=1e-4):
            raise SystemExit(f"{label}: its test subjects are not its bundle's")
        held = light(d["code_test"], own, names)
        s = shared_leftover(d["code_train"], d["gap_train"], d["code_test"], d["gap_test"])
        rows[label] = {
            "run": run.name,
            "cell": cell_of(label),
            "holds": held["holds"],
            "stage_holds": known.get(label),
            "last": held["last"],
            "worst": held["worst"],
            "rho1": float(s["rho"][0]),
            "rho2": float(s["rho"][1]),
            "gap_rho": s["gap_rho"][:8].tolist(),
            "settings": s["settings"],
        }
        r = rows[label]
        check = "" if r["stage_holds"] is None else f"  stage verdict {'yes' if r['stage_holds'] else 'no'}"
        if r["stage_holds"] is not None and r["stage_holds"] != r["holds"]:
            check += "  DIFFERS"
        print(
            f"  {label:16s} {'holds' if r['holds'] else 'fails'}  last {pair_str(r['last'])}  worst "
            f"{pair_str(r['worst'])}  rho1 {r['rho1']:.2f} rho2 {r['rho2']:.2f}{check}",
            flush=True,
        )
    report = {"rows": rows, "cells": {}}
    holders = [r["rho1"] for r in rows.values() if r["holds"]]
    failures = [r["rho1"] for r in rows.values() if not r["holds"]]
    right = total = 0
    print()
    for cell in sorted({r["cell"] for r in rows.values()}):
        members = sorted((r for r in rows.values() if r["cell"] == cell), key=lambda r: r["rho1"])
        cell_pairs = [(h, f) for h in members if h["holds"] for f in members if not f["holds"]]
        cell_right = sum(f["rho1"] > h["rho1"] for h, f in cell_pairs)
        right, total = right + cell_right, total + len(cell_pairs)
        entry = {"pairs": len(cell_pairs), "right": cell_right, "lowest_holds": members[0]["holds"]}
        for k in ks:
            if len(members) > k:
                entry[f"k{k}"] = selection(members, k)
        report["cells"][cell] = entry
        order = " < ".join(f"{r['rho1']:.2f}{'h' if r['holds'] else 'f'}" for r in members)
        print(f"  {cell:12s} rho1 by rank (h holds, f fails): {order}; pairs right {cell_right}/{len(cell_pairs)}")
        for k in ks:
            if f"k{k}" in entry:
                e = entry[f"k{k}"]
                print(
                    f"  {'':12s} K = {k}: {e['with_holder']} of {e['subsets']} subsets have a holder; the lowest rho1"
                    f" holds in {e['picked']} of them, a random pick in {e['random']:.1f}"
                )
    enough = len(holders) >= 2 and len(failures) >= 2
    value = auc(failures, holders) if holders and failures else float("nan")
    if not enough:
        detector = "described only (fewer than 2 holders or 2 failures)"
    elif value >= 0.80 and right >= pairs_needed * total:
        detector = "works"
    elif value <= 0.65:
        detector = "fails"
    else:
        detector = "unresolved"
    report["detector"] = {"auc": value, "right": right, "pairs": total, "holders": len(holders),
                          "failures": len(failures), "verdict": detector}
    print(
        f"\n  DETECTOR: AUC {value:.2f} over {len(failures)} failures and {len(holders)} holders; within-cell pairs right"
        f" {right}/{total} (needed {pairs_needed:.0%}): {detector}"
    )
    if rule_k is not None:
        totals = [e[f"k{rule_k}"] for e in report["cells"].values() if f"k{rule_k}" in e]
        with_holder, picked = sum(t["with_holder"] for t in totals), sum(t["picked"] for t in totals)
        random_pick = sum(t["random"] for t in totals)
        share = picked / with_holder if with_holder else float("nan")
        rule = (
            "described only (fewer than 2 holders or 2 failures)"
            if not enough
            else "works" if share >= 0.90 else "fails" if share < 0.70 else "unresolved"
        )
        report["selection"] = {"k": rule_k, "with_holder": with_holder, "picked": picked, "random": random_pick,
                               "share": share, "verdict": rule}
        print(
            f"  SELECTION (K = {rule_k}): the lowest rho1 holds in {picked} of {with_holder} subsets with a holder"
            f" ({share:.0%}; a random pick {random_pick / max(with_holder, 1):.0%}): {rule}"
        )
    cut = [r for r in rows.values()]
    print(
        f"  stage 75's ranges as a cut: holders at or under .56 {sum(r['rho1'] <= 0.56 for r in cut if r['holds'])}/"
        f"{len(holders)}, failures at or over .45 {sum(r['rho1'] >= 0.45 for r in cut if not r['holds'])}/"
        f"{len(failures)}; a cut at .50 classifies {sum((r['rho1'] < 0.50) == r['holds'] for r in cut)}/{len(cut)}"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"\nSaved {out_json}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("runs", nargs="*", type=Path)
    e.add_argument("--all", action="store_true", help="every run of ARMS at SEEDS, from runs/")
    e.add_argument("--gauge", choices=GAUGES, default="stored", help="recomputed: stage 78, into pool_recomputed/")
    r = sub.add_parser("run")
    r.add_argument("out", type=Path)
    r.add_argument("--restarts", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--draws", type=int, default=3, help="random subsets per cell and size (described only)")
    t = sub.add_parser("detect", help="stage 76: rho1 and holds for LABEL=RUN pairs, with pairs and subsets")
    t.add_argument("out", type=Path)
    t.add_argument("pairs", nargs="+", metavar="LABEL=RUN")
    t.add_argument("--stages", nargs="*", default=[], help="stages whose holds.txt verdicts are checked")
    t.add_argument("--subsets", nargs="*", type=int, default=[], help="K values for keep-the-lowest-rho1 subsets")
    t.add_argument("--rule-k", type=int, default=None, help="the K the selection rule reads")
    t.add_argument("--pairs-needed", type=float, default=0.8)
    t.add_argument("--gauge", choices=GAUGES, default="stored", help="which export the codes come from (stage 78)")
    a = p.parse_args()
    if a.command == "export":
        runs = list(a.runs) + ([HERE / "runs" / run_name(arm, s) for arm in ARMS for s in SEEDS] if a.all else [])
        if not runs:
            p.error("export needs runs or --all")
        export(runs, gauge=a.gauge)
    elif a.command == "detect":
        torch.set_num_threads(4)
        detect(a.out, a.pairs, a.stages, a.subsets, a.rule_k, a.pairs_needed, a.gauge)
    else:
        torch.set_num_threads(4)
        run(a.out, a.restarts, a.seed, a.draws)


if __name__ == "__main__":
    main()
