"""Stage 82 (no training; rules written 26 Nov before any of its results; the user asked for it): identify wide, chart
narrow. Can a wide code that holds be brought down to 8 units, without losing an own part, by a nonlinear chart fitted
to rebuild its whitened shared part?

Why. At 8 units InfoNCE trades a weak own part for curves of the strong factors (items 46, 56); no term added to it
has changed that (stages 60, 70, 71, 77, 80). Wide codes hold, their spare units are curves of the seven factors and
their intrinsic dimension is the factors' (items 63, 64). Every compression tried on the bank was linear (stage 65's
redundancy cut and PCA, stage 75's merge), and a linear map cannot fold a curve back onto its factor. A nonlinear
chart can keep a factor once and let its decoder rebuild the factor's curves; on whitened targets a weak direction
counts as much as a strong one; and no contrastive loss competes. Item 27's simulation narrowed a wide code by
rebuilding it and kept its faint latent in 1 of 4 runs, so this may fail for the same finite-sample reason.

Parents: the fixed bank's five 10-unit codes of the frozen recipe (stages 64-65, seeds 42, 2, 3, 4, 5; seed 4's
fails, the other four hold), from their stage 75 exports (causal/pool/RUN.npz: both views' codes for the 2,000
training and 1,000 test subjects). All ten of their canonical directions are shared (held-out cross-view correlation
.87-1.00, checked before these rules), so the chart has to drop two shared directions.

Per parent, label-free, on training subjects only (1,600 to fit, 400 held out; permutation seed 0):
  1. Each view's code standardized; CCA between the views (ridge 1e-3); the shared part = the canonical directions
     whose held-out cross-view correlation beats both .5 and a shuffled-pairing null's 95th percentile. Each view's
     variates restandardized on the fitting subjects, so every direction counts alike.
  2. Nonlinear chart: one MLP encoder (k -> 128 -> 128 -> 8) and decoder (8 -> 128 -> 128 -> k), shared by the views,
     trained to rebuild each view's variates from themselves (mean squared error over all k directions), AdamW (lr
     1e-3, weight decay 1e-4), batches of 256, early stopping on the held-out subjects (patience 300 epochs, at most
     4,000); 5 restarts, the lowest held-out loss kept. The 8-unit code is the encoder's output.
  3. Linear chart (control): the top 8 principal components of the same variates (both views' fitting rows), the best
     linear 8-unit rebuild.
Scored on the 1,000 test subjects as stage 76 (pool_check.score: holds when the own-parts spectrum's last value and
the ventricle-burden worst direction are each at least .40 in both views), own parts from the parent's bundle.

Rules (fixed before any result):
  Works if the nonlinear chart holds in all 4 parents that hold. Fails if it holds in 2 or fewer of the 4. Otherwise
  unresolved.
  Described only: the linear chart's holds (whether the nonlinearity is needed); seed 4 (its parent fails, and a chart
  cannot add content); the own parts against the parent's; the curves each code keeps; stage 65's fine-tuned linear
  compressions of the same parents for reference (redundancy held 3 of 5, PCA 2 of 5).
Expectation, written before: the nonlinear chart holds in 2-3 of the 4, the linear chart in 1-2.
Read 26 Nov: works (4 of 4; linear 0 of 4).

Stage 82b (rules written 26 Nov after stage 82's reading and before any of 82b's results; the user asked for it): the
same chart, unchanged, on other parents and widths.
  Parents, each width its own cell, verdicts rescored here from each parent's bundle:
    c9e500, c9e100: stage 64's 9-unit codes (seeds 42, 3, 4, 5 each);
    c12, c16: stage 74's 12- and 16-unit codes (A100, seeds 42, 3, 4), copied from the cluster into
      causal/stage82/wide_runs/ and exported and bundled here as every other code;
    c10: stage 82's five parents again, the charts kept this time for the checks below.
  Rule, per cell: works if the 8-unit nonlinear chart holds in every parent of the cell that holds; fails if it
  misses two or more; otherwise unresolved. A cell with no holding parent is described only.
  Described only: the linear chart; rho1 (pool_check.shared_leftover, as stage 76) for every parent and chart, and
  whether within each cell the charts that fail read higher rho1 than those that hold (the charts are not trained on
  rho1's quantity, so item 65's caution does not apply); the 10-unit charts at 7 units, the content's own dimension
  (the chart's output is not on a sphere, so it should not need the cosine's extra unit); stage 82's verdicts against
  the c10 rerun (fits are seeded, but the GPU need not repeat them exactly).
Expectation, written before: works at 9 and 12 units; at 16 units the chart has nine spare directions to fold and may
miss one holder. The 7-unit charts hold where the 8-unit ones do, with no curve kept.

Certificate (--certificate; 26 Nov, described only, no rule): two finite-sample quantities per chart, on the 1,000 test
subjects, which neither the CCA, the chart nor its early stopping saw.
  rebuild: the chart's error rebuilding the parent's whitened shared variates (both views), as a share of their
    variance: the mean over directions and the worst single direction, each with a 95% bootstrap upper bound over
    subjects (1,000 draws). "The chart keeps every shared direction of its parent to within the worst bound."
  leftover: rho1 of parent and chart beside the 95th percentile of a shuffled-pairing null (200 draws of the test
    variates' pairing). rho1 above the null says shared content is left once the code is known, at that level.

    python chart_check.py OUT.json [--parents SEED...] [--restarts 5]
    python chart_check.py OUT.json --runs LABEL=RUN... [--units 8] [--save] [--rho] [--certificate]
    python chart_check.py OUT.json --merge PART.json...   # the verdict over parents run as separate processes
"""

import argparse
import json
from pathlib import Path

import holds
import numpy as np
import pool_check
import torch
from torch import nn

import bank as B

HERE = Path(__file__).resolve().parent
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c10_20k"
PARENT_HOLDS = {42: True, 2: True, 3: True, 4: False, 5: True}
UNITS, HELD, RIDGE, AGREE, NULL_DRAWS = 8, 400, 1e-3, 0.5, 200


def run_name(seed):
    return P if seed == 42 else f"{P}_s{seed}"


def cca(a, b, ridge=RIDGE):
    a, b = a - a.mean(0), b - b.mean(0)

    def inverse_root(x):
        c = x.T @ x / len(x)
        c += ridge * np.trace(c) / len(c) * np.eye(len(c))
        w, v = np.linalg.eigh(c)
        return (v / np.sqrt(w)) @ v.T

    wa, wb = inverse_root(a), inverse_root(b)
    u, s, vt = np.linalg.svd(wa @ (a.T @ b / len(a)) @ wb)
    return wa @ u, wb @ vt.T, s


def shared_variates(train, test, rng):
    """Step 1: per view, the shared canonical variates of the standardized code, fitted on the fitting subjects."""
    order = rng.permutation(train.shape[1])
    fit, held = order[:-HELD], order[-HELD:]
    mean, sd = train[:, fit].mean(1, keepdims=True), train[:, fit].std(1, keepdims=True)
    z_train, z_test = (train - mean) / sd, (test - mean) / sd
    wa, wb, rho = cca(z_train[0, fit], z_train[1, fit])
    w = [wa, wb]
    held_v = [z_train[v, held] @ w[v] for v in range(2)]
    agree, null = [], []
    for j in range(wa.shape[1]):
        x, y = held_v[0][:, j], held_v[1][:, j]
        agree.append(float(np.corrcoef(x, y)[0, 1]))
        null.append(float(np.percentile([abs(np.corrcoef(x, y[rng.permutation(len(y))])[0, 1])
                                         for _ in range(NULL_DRAWS)], 95)))
    keep = [j for j in range(len(agree)) if agree[j] > max(AGREE, null[j])]
    u_train = np.stack([z_train[v] @ w[v][:, keep] for v in range(2)])
    u_test = np.stack([z_test[v] @ w[v][:, keep] for v in range(2)])
    m, s = u_train[:, fit].mean(1, keepdims=True), u_train[:, fit].std(1, keepdims=True)
    return (u_train - m) / s, (u_test - m) / s, fit, held, {"train_rho": rho.tolist(), "held_agreement": agree,
                                                             "null95": null, "kept": len(keep)}


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def fit_chart(u_train, fit, held, seed, device, epochs=4000, patience=300, batch=256):  # UNITS: global
    """Step 2: a shared MLP autoencoder k -> 8 -> k on both views' fitting rows; the best state by held-out loss."""
    k = u_train.shape[-1]
    torch.manual_seed(seed)
    enc, dec = mlp([k, 128, 128, UNITS]).to(device), mlp([UNITS, 128, 128, k]).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(np.concatenate([u_train[0, fit], u_train[1, fit]]), dtype=torch.float32, device=device)
    xv = torch.tensor(np.concatenate([u_train[0, held], u_train[1, held]]), dtype=torch.float32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, state, wait = np.inf, None, 0
    for _ in range(epochs):
        enc.train(), dec.train()
        for idx in torch.randperm(len(x), generator=gen).split(batch):
            xb = x[idx.to(device)]
            loss = ((dec(enc(xb)) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        enc.eval(), dec.eval()
        with torch.no_grad():
            val = float(((dec(enc(xv)) - xv) ** 2).mean())
        if val < best - 1e-6:
            best, wait = val, 0
            state = ({k_: v.clone() for k_, v in enc.state_dict().items()},
                     {k_: v.clone() for k_, v in dec.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
    enc.load_state_dict(state[0]), dec.load_state_dict(state[1])
    return enc, dec, best


def encode(enc, u, device):
    with torch.no_grad():
        return np.stack([enc(torch.tensor(u[v], dtype=torch.float32, device=device)).cpu().numpy() for v in range(2)])


def linear_chart(u_train, u_test, fit):
    """Step 3: the top 8 principal components of both views' fitting rows."""
    x = np.concatenate([u_train[0, fit], u_train[1, fit]])
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return np.stack([(u_test[v] - mean) @ vt[:UNITS].T for v in range(2)])


def summarize(score):
    return {"holds": bool(score["holds"]), "last": [float(v) for v in score["last"]],
            "worst": [float(v) for v in score["worst"]], "curves": score.get("curves"),
            "own_r2": {v: {k: float(x) for k, x in score[v]["own_r2"].items()} for v in holds.VIEWS}}


def parent_data(run):
    """A parent's export (code and GAP, both views, training and test subjects) and its bundle, made if missing."""
    import subprocess
    import sys

    export = HERE / "causal" / "pool" / f"{run.name}.npz"
    if not export.exists() or export.stat().st_mtime <= (run / "model.pt").stat().st_mtime:
        pool_check.export([run])
    bundle = HERE / "causal" / f"{run.name}.npz"
    if not bundle.exists():
        subprocess.run([sys.executable, str(HERE / "causal_bundle.py"), str(run), str(bundle)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.load(export), np.load(bundle)


def rho1(code_train, code_test, d, draws=200, seed=0):
    """rho1 (pool_check.shared_leftover) and the 95th percentile of its shuffled-pairing null on the test variates."""
    s = pool_check.shared_leftover(code_train, d["gap_train"].astype(np.float64), code_test,
                                   d["gap_test"].astype(np.float64))
    u, v = s["variates"][0][:, 0], s["variates"][1][:, 0]
    rng = np.random.default_rng(seed)
    null = [abs(np.corrcoef(u, v[rng.permutation(len(v))])[0, 1]) for _ in range(draws)]
    return float(s["rho"][0]), float(np.percentile(null, 95))


def rebuild_certificate(enc, dec, u_test, device, draws=1000, seed=0):
    """The chart's held-out rebuild error per shared direction (both views' test rows), as a share of each direction's
    variance: the mean over directions and the worst one, each with a 95% bootstrap upper bound over subjects."""
    with torch.no_grad():
        errors = []
        for v in range(2):
            x = torch.tensor(u_test[v], dtype=torch.float32, device=device)
            errors.append(((dec(enc(x)) - x) ** 2).cpu().numpy())
    err = np.mean(errors, axis=0)  # (subjects, directions): the two views' errors averaged per subject
    var = np.mean([u_test[v].var(0) for v in range(2)], axis=0)
    rng = np.random.default_rng(seed)
    means, worsts = [], []
    for _ in range(draws):
        rows = rng.integers(0, len(err), len(err))
        per_direction = err[rows].mean(0) / var
        means.append(per_direction.mean())
        worsts.append(per_direction.max())
    per_direction = err.mean(0) / var
    return {"mean": float(per_direction.mean()), "mean_upper95": float(np.percentile(means, 95)),
            "worst": float(per_direction.max()), "worst_upper95": float(np.percentile(worsts, 95)),
            "per_direction": per_direction.tolist()}


def main():
    global UNITS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--parents", nargs="*", type=int, default=[42, 2, 3, 4, 5])
    p.add_argument("--runs", nargs="*", default=None, metavar="LABEL=RUN")
    p.add_argument("--units", type=int, default=UNITS)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--save", action="store_true", help="keep each chart's codes in causal/stage82/charts/")
    p.add_argument("--rho", action="store_true", help="rho1 of every parent and chart (stage 82b)")
    p.add_argument("--certificate", action="store_true", help="the rebuild bound and rho1 against its null")
    p.add_argument("--merge", nargs="*", type=Path, default=None)
    a = p.parse_args()
    if a.merge:
        report = {}
        for part in a.merge:
            report.update(json.loads(part.read_text()))
        a.out.write_text(json.dumps(report, indent=1))
        return verdict(report)
    UNITS = a.units
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    if a.runs:
        parents = [(pair.split("=", 1)[0], pair.split("=", 1)[1]) for pair in a.runs]
    else:
        parents = [(f"c10_s{s}", run_name(s)) for s in a.parents]
    report = {}
    for label, name in parents:
        run = Path(name) if "/" in name else HERE / "runs" / name
        d, bundle = parent_data(run)
        own, raw = bundle["z_own"].astype(np.float64), bundle["z_content"].astype(np.float64)
        names = json.loads(str(bundle["meta"]))["content_factor_names"]
        if not np.allclose(d["zc_test"][:, list(B.CAUSAL_FACTORS)], raw, atol=1e-4):
            raise SystemExit(f"{label}: the export's test subjects are not the bundle's")
        u_train, u_test, fit, held, shared = shared_variates(d["code_train"].astype(np.float64),
                                                             d["code_test"].astype(np.float64),
                                                             np.random.default_rng(0))
        fits = [fit_chart(u_train, fit, held, r, device) for r in range(a.restarts)]
        losses = [loss for _, _, loss in fits]
        enc, dec, _ = fits[int(np.argmin(losses))]
        chart_train, chart_test = encode(enc, u_train, device), encode(enc, u_test, device)
        row = {"run": run.name, "units": UNITS, "shared": shared, "restart_losses": losses,
               "parent": summarize(pool_check.score(d["code_test"].astype(np.float64), own, names, raw)),
               "linear": summarize(pool_check.score(linear_chart(u_train, u_test, fit), own, names, raw)),
               "chart": summarize(pool_check.score(chart_test, own, names, raw))}
        if a.rho or a.certificate:
            row["parent"]["rho1"], row["parent"]["rho1_null95"] = rho1(d["code_train"].astype(np.float64),
                                                                       d["code_test"].astype(np.float64), d)
            row["chart"]["rho1"], row["chart"]["rho1_null95"] = rho1(chart_train, chart_test, d)
        if a.certificate:
            row["certificate"] = rebuild_certificate(enc, dec, u_test, device)
            c = row["certificate"]
            print(f"  certificate: rebuild mean {c['mean']:.4f} (<= {c['mean_upper95']:.4f}), worst direction "
                  f"{c['worst']:.4f} (<= {c['worst_upper95']:.4f}); rho1 parent {row['parent']['rho1']:.2f}, chart "
                  f"{row['chart']['rho1']:.2f} (null95 {row['chart']['rho1_null95']:.2f})", flush=True)
        if a.save:
            folder = HERE / "causal" / "stage82" / "charts"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / f"{label}_u{UNITS}.npz", code_train=chart_train, code_test=chart_test,
                     zc_test=d["zc_test"], meta=json.dumps({"label": label, "parent": run.name, "units": UNITS}))
        report[label] = row
        print(f"{label}: parent {run.name}, shared k {shared['kept']}, {UNITS}-unit chart held-out losses "
              + " ".join(f"{x:.3f}" for x in losses))
        for key in ("parent", "linear", "chart"):
            r = row[key]
            t1, flair = (r["own_r2"][v] for v in holds.VIEWS)
            own_txt = " ".join(f"{n[:4]} {t1[n]:.2f}/{flair[n]:.2f}" for n in names)
            rho_txt = f"  rho1 {r['rho1']:.2f}" if "rho1" in r else ""
            print(f"  {key:6s} {'holds' if r['holds'] else 'fails'}  last {r['last'][0]:+.2f}/{r['last'][1]:+.2f}  "
                  f"worst {r['worst'][0]:+.2f}/{r['worst'][1]:+.2f}  curves {r['curves']}{rho_txt} | own {own_txt}",
                  flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    verdict(report)


def verdict(report):
    """Per cell (the label without its seed): the rule of stage 82 (c10) and 82b (every other cell)."""
    cells = {}
    for label, row in report.items():
        cells.setdefault(label.rsplit("_s", 1)[0], []).append((label, row))
    for cell, rows in sorted(cells.items()):
        print(f"\ncell {cell} ({rows[0][1].get('units', UNITS)}-unit charts)")
        for label, row in sorted(rows):
            marks = "  ".join(f"{k} {'holds' if row[k]['holds'] else 'fails'}"
                              + (f" rho1 {row[k]['rho1']:.2f}" if "rho1" in row[k] else "")
                              for k in ("parent", "linear", "chart"))
            print(f"  {label:12s} {marks}")
        holders = [row for _, row in rows if row["parent"]["holds"]]
        chart = sum(row["chart"]["holds"] for row in holders)
        linear = sum(row["linear"]["holds"] for row in holders)
        if not holders:
            outcome = "described only (no holding parent)"
        else:
            outcome = ("works" if chart == len(holders) else "fails" if len(holders) - chart >= 2 else "unresolved")
        print(f"  parents holding {len(holders)} of {len(rows)}; chart holds in {chart}, linear chart in {linear}:"
              f" {outcome}")
        charts = [(row["chart"]["rho1"], row["chart"]["holds"]) for _, row in rows if "rho1" in row["chart"]]
        if charts and any(h for _, h in charts) and not all(h for _, h in charts):
            pairs = [(f, h) for f, hf in charts if not hf for h, hh in charts if hh]
            print(f"  rho1 of the charts: failing above holding in {sum(f > h for f, h in pairs)} of {len(pairs)} pairs")


if __name__ == "__main__":
    main()
