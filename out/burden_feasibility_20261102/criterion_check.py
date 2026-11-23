"""Can a run be chosen without labels? Label-free signals against ground truth, over every finished warm.py run.

On ADNI no factor's own part can be read, so every choice between recipes there has to rest on something computed
from the images alone. This check asks which such signals rank the synthetic runs the way the ground truth does.

For each run, with no labels (training subjects fit any cross-view map, validation subjects score it):
  cca_sum, cca_count50, gauss_mi   the code's canonical correlations between the two views, fitted on training
                                   subjects and measured on validation subjects: their sum, how many exceed .5 (the
                                   plan's width rule), and -1/2 sum log(1 - rho^2), the Gaussian mutual information
  gauss_mi_gain                    gauss_mi minus the run's untrained twin's (same architecture and init, untrained)
  top1_all, top1_32                cross-view retrieval top-1 among all validation subjects, and among 32
  memorisation                     top-1 among as many training subjects minus top-1 among the validation subjects
  knn_agreement                    overlap (Jaccard) of each subject's 10 nearest neighbours in the two views' codes
  rankme, eff_rank, alpha_dev      the code's spectrum: RankMe (Garrido et al. 2023), participation ratio, and
                                   |alpha - 1| for its power-law decay alpha (alpha-ReQ, Agrawal et al. 2022)
  twonn                            TwoNN intrinsic dimension; no direction is fixed for it, so it is never used to pick
  gap_*                            the same canonical and spectral signals on the backbone GAP
  final_loss                       the last logged global training loss
Spectral signals are each view's, averaged over the views.

Ground truth, on test subjects: each of the seven rendered factors' own part (its SCM noise on a causal bank, the factor
itself on an independent one), read from each view's code by 5-fold ridge and by gradient boosting, averaged over views.
  own_mean, own_weakest, burden_own, own_mean_boosted

Runs are compared only with runs on the same bank and T1 lesion value (grouping "bank"), and again within the same code
width too ("bank_width"). Groups need MIN_GROUP runs. Per signal and truth:
  rho       Spearman correlation within each group, averaged weighted by group size, with a bootstrap 95% interval
            (runs resampled within groups)
  regret    in each group, pick the run the signal ranks best (in its direction fixed above); the share of a random
            pick's shortfall from the best run's truth that this removes, summed over groups (1 = always the best run,
            0 = no better than chance, below 0 = worse)
  banks     in how many groups the correlation has the expected sign
Pass rule, fixed before any result: a signal is usable for choosing runs on ADNI if, under grouping "bank", for both
own_mean and own_weakest its interval lies above 0 and it removes at least half of a random pick's regret.

Supervised runs (sup.py, warm.py --supervise) are skipped. Runs already in OUT_JSON at the same step are reused. With
no RUN, the table is printed from OUT_JSON alone (no torch needed).

    python criterion_check.py OUT_JSON [RUN ...] [--fit-subjects 1000]
    python criterion_check.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

SIGNALS = {  # name: (direction, +1 higher is better, -1 lower, 0 none)
    "cca_sum": 1,
    "cca_count50": 1,
    "gauss_mi": 1,
    "gauss_mi_gain": 1,
    "top1_all": 1,
    "top1_32": 1,
    "memorisation": -1,
    "knn_agreement": 1,
    "rankme": 1,
    "eff_rank": 1,
    "alpha_dev": -1,
    "twonn": 0,
    "gap_cca_sum": 1,
    "gap_gauss_mi_gain": 1,
    "gap_rankme": 1,
    "final_loss": -1,
}
TRUTHS = ("own_mean", "own_weakest", "burden_own", "own_mean_boosted")
PASS_TRUTHS = ("own_mean", "own_weakest")
GROUPINGS = ("bank", "bank_width")
MIN_GROUP = 4
BOOT = 1000
REGRET_SHARE = 0.5
NEIGHBOURS = 10
SETS = 200
SHRINK = 1e-3
FACTOR_NAMES = ["brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal"]
BURDEN = FACTOR_NAMES.index("burden")


def standardized(x, mean=None, sd=None):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0) if mean is None else mean
    sd = x.std(0) if sd is None else sd
    return np.divide(x - mean, sd, out=np.zeros_like(x), where=sd > 1e-12)


def whitener(x):
    c = np.cov(x, rowvar=False)
    c = (1 - SHRINK) * c + SHRINK * np.trace(c) / len(c) * np.eye(len(c))
    w, v = np.linalg.eigh(c)
    keep = w > w.max() * 1e-8
    return v[:, keep] / np.sqrt(w[keep])


def held_out_canonical(fit, held):
    """Canonical directions fitted on ``fit`` = (view 1, view 2), and each pair's correlation on ``held``."""
    (a, b), (ha, hb) = fit, held
    moments = []
    for x in (a, b):
        sd = np.asarray(x, dtype=np.float64).std(0)
        moments.append((np.asarray(x, dtype=np.float64).mean(0), sd, sd > 1e-12))
    a, b = (standardized(x, m, s)[:, k] for x, (m, s, k) in zip((a, b), moments))
    ha, hb = (standardized(x, m, s)[:, k] for x, (m, s, k) in zip((ha, hb), moments))
    wa, wb = whitener(a), whitener(b)
    cross = (a - a.mean(0)).T @ (b - b.mean(0)) / (len(a) - 1)
    u, _, vt = np.linalg.svd(wa.T @ cross @ wb, full_matrices=False)
    va, vb = ha @ (wa @ u), hb @ (wb @ vt.T)
    va, vb = standardized(va), standardized(vb)
    return np.clip((va * vb).mean(0), -1, 1)


def gauss_mi(rho):
    r = np.clip(rho, 0, 0.999)
    return float(-0.5 * np.log(1 - r**2).sum())


def top1(z1, z2, sets):
    """Cross-view top-1 within each set (rows of both views), standardized over the set, by cosine, both directions."""
    hits = []
    for s in sets:
        a, b = standardized(z1[s]), standardized(z2[s])
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        sim = a @ b.T
        truth = np.arange(len(s))
        hits.append((np.mean(sim.argmax(1) == truth) + np.mean(sim.argmax(0) == truth)) / 2)
    return float(np.mean(hits))


def knn_agreement(z1, z2, k=NEIGHBOURS):
    k = min(k, len(z1) - 2)
    sets = []
    for z in (z1, z2):
        z = standardized(z)
        _, idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)
        sets.append(idx[:, 1:])
    return float(np.mean([len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(*sets)]))


def spectrum(z):
    z = standardized(z)
    z = z[:, z.any(0)]
    s = np.linalg.svd(z, compute_uv=False)
    return s[s > s[0] * 1e-8] if len(s) else s


def rankme(z):
    s = spectrum(z)
    p = s / s.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def eff_rank(z):
    lam = spectrum(z) ** 2
    return float(lam.sum() ** 2 / (lam**2).sum())


def alpha(z):
    lam = spectrum(z) ** 2
    i = np.arange(1, len(lam) + 1)
    return float(-np.polyfit(np.log(i), np.log(lam), 1)[0]) if len(lam) > 2 else float("nan")


def two_nn(z, discard=0.1):
    """TwoNN, as intrinsic_dimension.two_nn (kept here so this file needs no torch)."""
    z = standardized(z)
    d, _ = NearestNeighbors(n_neighbors=3).fit(z).kneighbors(z)
    d = d[:, 1:]
    d = d[d[:, 0] > 0]
    mu = np.sort(d[:, 1] / d[:, 0])
    keep = int(np.floor(len(mu) * (1 - discard)))
    log_mu = np.log(mu[:keep])
    log_tail = -np.log(1 - np.arange(1, keep + 1) / len(mu))
    return float(log_mu @ log_tail / (log_mu @ log_mu))


def per_view(fn, views):
    return float(np.mean([fn(v) for v in views]))


def label_free(fit, held, twin_fit=None, twin_held=None, gap_fit=None, gap_held=None, twin_gap=None, seed=0):
    """Every signal for one run: ``fit``/``held`` are (view 1, view 2) codes of training and validation subjects;
    ``twin_*`` the untrained twin's; ``gap_*`` the backbone GAP's (``twin_gap`` = (fit, held) of the twin's GAP)."""
    rho = held_out_canonical(fit, held)
    n = len(held[0])
    rng = np.random.default_rng(seed)
    size = min(32, n)
    out = {
        "cca_sum": float(np.clip(rho, 0, 1).sum()),
        "cca_count50": int((rho > 0.5).sum()),
        "gauss_mi": gauss_mi(rho),
        "top1_all": top1(*held, [np.arange(n)]),
        "top1_32": top1(*held, [rng.choice(n, size, replace=False) for _ in range(SETS)]),
        "memorisation": top1(fit[0][:n], fit[1][:n], [np.arange(min(n, len(fit[0])))]) - top1(*held, [np.arange(n)]),
        "knn_agreement": knn_agreement(*held),
        "rankme": per_view(rankme, held),
        "eff_rank": per_view(eff_rank, held),
        "alpha_dev": per_view(lambda z: abs(alpha(z) - 1), held),
        "twonn": per_view(two_nn, held),
        "canonical_r": np.round(rho, 4).tolist(),
    }
    if twin_fit is not None:
        out["gauss_mi_gain"] = out["gauss_mi"] - gauss_mi(held_out_canonical(twin_fit, twin_held))
    if gap_fit is not None:
        gap_rho = held_out_canonical(gap_fit, gap_held)
        out["gap_cca_sum"] = float(np.clip(gap_rho, 0, 1).sum())
        out["gap_rankme"] = per_view(rankme, gap_held)
        if twin_gap is not None:
            out["gap_gauss_mi_gain"] = gauss_mi(gap_rho) - gauss_mi(held_out_canonical(*twin_gap))
    return out


def truth_summary(ridge_by_view, boosted_by_view):
    ridge = np.mean(ridge_by_view, 0)
    return {
        "own_mean": float(ridge.mean()),
        "own_weakest": float(ridge.min()),
        "burden_own": float(ridge[BURDEN]),
        "own_mean_boosted": float(np.mean(boosted_by_view)),
    }


def supervised(report):
    a = report.get("args", {})
    history = report.get("history") or [{}]
    return "test" in report or "val" in history[0] or a.get("supervise", "None") not in ("None", "False", "")


def assess(run, device, fit_subjects):
    import bank as B
    import causal_bundle
    import code_audit
    import own_variation
    import warm

    columns = list(B.CAUSAL_FACTORS)
    feats = {}
    for trained in (True, False):
        report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
        a = report["args"]
        root, key = a["bank"], a["t1key"]
        train = warm.Bank(root, "train", key, fit_subjects, residual)
        if not trained and pool is not None:
            causal_bundle.calibrate(model, pool, warm.Bank(root, "train", key, residual=residual), device)
        splits = {"fit": train, "held": warm.Bank(root, "val", key, None, residual)}
        if trained:
            splits["test"] = warm.Bank(root, "test", key, int(a.get("test_subjects", 1000)), residual)
            meta = {"step": step, "bank": root, "t1key": key, "report": report, "test": splits["test"]}
        feats[trained] = {k: warm.features(model, bank, device, pool=pool) for k, bank in splits.items()}
        del model, pool
    t, u = feats[True], feats[False]
    signals = label_free(
        t["fit"]["code"],
        t["held"]["code"],
        u["fit"]["code"],
        u["held"]["code"],
        t["fit"]["gap"],
        t["held"]["gap"],
        (u["fit"]["gap"], u["held"]["gap"]),
    )
    history = meta["report"]["history"]
    signals["final_loss"] = next(
        (h["loss"]["global"] for h in reversed(history) if "global" in h.get("loss", {})), None
    )
    test = meta["test"]
    zc = np.asarray(test.zc, dtype=np.float64)
    causal = Path(f"{meta['bank']}/causal_adj.npy").exists()
    own = zc[:, columns]
    if causal:  # own_variation.py's targets, so the truth here is the one its reports print
        scm, scale, nonlinearity = own_variation.scm_settings()
        if not np.array_equal(scm["adj"], np.load(f"{meta['bank']}/causal_adj.npy")):
            raise ValueError(f"{meta['bank']}'s causal_adj.npy is not bank.causal_scm's graph")
        own = own_variation.targets(zc, scm, scale, nonlinearity, columns)["own"]
    codes = t["test"]["code"]
    ridge = [[own_variation.ridge(c, own[:, j]) for j in range(len(columns))] for c in codes]
    boosted = [[code_audit.boosted(c, own[:, j]) for j in range(len(columns))] for c in codes]
    a = meta["report"]["args"]
    return {
        "step": meta["step"],
        "bank": Path(meta["bank"]).name,
        "t1key": meta["t1key"],
        "causal": causal,
        "width": int(codes[0].shape[1]),
        "loss_type": a.get("contrastive_loss_type", "infonce"),
        "signals": signals,
        "own_ridge": np.round(ridge, 4).tolist(),
        "own_boosted": np.round(boosted, 4).tolist(),
        "truth": truth_summary(ridge, boosted),
    }


def group_of(row, grouping):
    key = f"{row['bank']} {row['t1key']}"
    return key if grouping == "bank" else f"{key} c{row['width']}"


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pooled_rho(groups):
    """Size-weighted mean of the within-group Spearman correlations; ``groups`` holds (signal, truth) arrays."""
    pairs = [(spearman(s, t), len(s)) for s, t in groups]
    pairs = [(r, n) for r, n in pairs if np.isfinite(r)]
    if not pairs:
        return float("nan")
    return float(sum(r * n for r, n in pairs) / sum(n for _, n in pairs))


def regret_removed(groups, direction):
    """The share of a random pick's regret that picking by the signal removes, summed over groups."""
    chosen = random = 0.0
    for s, t in groups:
        best = t.max()
        chosen += best - t[np.argmax(direction * s)]
        random += best - t.mean()
    return float(1 - chosen / random) if random > 0 else float("nan")


def analyse(rows, grouping, seed=0):
    """Per signal and truth: rho, its interval, regret removed, groups with the expected sign, and the verdict."""
    by_group = {}
    for name, row in rows.items():
        by_group.setdefault(group_of(row, grouping), []).append(row)
    by_group = {k: v for k, v in by_group.items() if len(v) >= MIN_GROUP}
    rng = np.random.default_rng(seed)
    out = {"groups": {k: len(v) for k, v in by_group.items()}, "signals": {}}
    for signal, direction in SIGNALS.items():
        cell = {}
        for truth in TRUTHS:
            groups, names = [], []
            for group, members in by_group.items():
                kept = [r for r in members if r["signals"].get(signal) is not None and truth in r["truth"]]
                if len(kept) >= MIN_GROUP:
                    names.append(group)
                    groups.append(
                        (
                            np.array([r["signals"][signal] for r in kept], float),
                            np.array([r["truth"][truth] for r in kept]),
                        )
                    )
            if not groups:
                continue
            rho = pooled_rho(groups)
            boot = []
            for _ in range(BOOT):
                drawn = []
                for s, t in groups:
                    i = rng.integers(0, len(s), len(s))
                    drawn.append((s[i], t[i]))
                boot.append(pooled_rho(drawn))
            boot = np.array(boot)
            boot = boot[np.isfinite(boot)]
            low, high = np.quantile(boot, [0.025, 0.975]) if len(boot) else (np.nan, np.nan)
            signed = [spearman(s, t) * (direction or 1) for s, t in groups]
            cell[truth] = {
                "rho": rho,
                "low": float(low),
                "high": float(high),
                "regret_removed": regret_removed(groups, direction) if direction else None,
                "right_sign": int(sum(r > 0 for r in signed if np.isfinite(r))),
                "groups": len(groups),
                "by_group": {k: spearman(s, t) for k, (s, t) in zip(names, groups)},
            }
        if cell:
            cell["usable"] = bool(direction) and all(
                truth in cell
                and direction * cell[truth]["rho"] > 0
                and (cell[truth]["low"] > 0 if direction > 0 else cell[truth]["high"] < 0)
                and cell[truth]["regret_removed"] >= REGRET_SHARE
                for truth in PASS_TRUTHS
            )
            out["signals"][signal] = cell
    return out


def show(rows, analyses):
    print(f"{len(rows)} runs. Truth: own-part R2 by ridge on test subjects, mean of the views.\n")
    print(f"  {'run':70s} {'group':28s} {'own':>5s} {'weak':>5s} {'burd':>5s} {'MI':>6s} {'cca50':>5s} {'top1':>5s}")
    for name, row in sorted(rows.items(), key=lambda kv: (group_of(kv[1], "bank"), -kv[1]["truth"]["own_mean"])):
        t, s = row["truth"], row["signals"]
        print(
            f"  {name[:70]:70s} {group_of(row, 'bank_width')[:28]:28s} {t['own_mean']:5.2f} {t['own_weakest']:5.2f} "
            f"{t['burden_own']:5.2f} {s['gauss_mi']:6.2f} {s['cca_count50']:5d} {s['top1_all']:5.2f}"
        )
    for grouping, result in analyses.items():
        print(f"\nGrouping '{grouping}': {len(result['groups'])} groups of at least {MIN_GROUP} runs")
        for group, n in result["groups"].items():
            print(f"  {n:3d}  {group}")
        if not result["signals"]:
            continue
        head = "".join(f" {t:>28s}" for t in TRUTHS)
        print(f"\n  {'signal':18s} {'dir':>3s}{head}   usable")
        print(f"  {'':22s}" + "".join(f" {'rho [95%]  regret  sign':>28s}" for _ in TRUTHS))
        for signal, cell in result["signals"].items():
            parts = []
            for truth in TRUTHS:
                c = cell.get(truth)
                if c is None:
                    parts.append(f" {'':>28s}")
                    continue
                regret = "   -  " if c["regret_removed"] is None else f"{c['regret_removed']:+6.2f}"
                parts.append(
                    f" {c['rho']:+.2f} [{c['low']:+.2f},{c['high']:+.2f}] {regret} {c['right_sign']}/{c['groups']}"
                )
            print(f"  {signal:18s} {SIGNALS[signal]:+3d}{''.join(parts)}   {'YES' if cell['usable'] else 'no'}")
    print(
        "\nrho: Spearman within groups, size-weighted, bootstrap 95% interval. regret: share of a random pick's shortfall"
        f" removed by picking the signal's best run. sign: groups with the expected sign. usable: under 'bank', for"
        f" {' and '.join(PASS_TRUTHS)}, interval excludes 0 in the expected direction and regret >= {REGRET_SHARE}."
    )


def self_test():
    rng = np.random.default_rng(0)
    n, d = 1500, 12
    shared = rng.standard_normal((n, 4))
    a = np.column_stack([shared, rng.standard_normal((n, d - 4))]) @ rng.standard_normal((d, d))
    b = np.column_stack([shared + 0.3 * rng.standard_normal((n, 4)), rng.standard_normal((n, d - 4))])
    rho = held_out_canonical((a[:1000], b[:1000]), (a[1000:], b[1000:]))
    assert (rho > 0.5).sum() == 4, rho
    assert abs(rho[0] - 1 / np.sqrt(1.09)) < 0.05, rho
    noise = held_out_canonical((a[:1000], rng.standard_normal((1000, d))), (a[1000:], rng.standard_normal((500, d))))
    assert np.abs(noise).max() < 0.2, noise
    z = rng.standard_normal((300, 8))
    assert top1(z, z, [np.arange(300)]) == 1.0 and knn_agreement(z, z) == 1.0
    assert top1(z, rng.standard_normal((300, 8)), [np.arange(300)]) < 0.05
    assert 7 < rankme(rng.standard_normal((5000, 8))) <= 8
    assert 7 < eff_rank(rng.standard_normal((5000, 8))) <= 8
    assert abs(alpha(rng.standard_normal((20000, 10)))) < 0.05
    assert abs(two_nn(rng.standard_normal((3000, 5)) @ rng.standard_normal((5, 12))) - 5) < 0.75

    def fake(quality, group, width=24):
        return {
            "bank": group,
            "t1key": "t1_04",
            "width": width,
            "signals": {
                **{k: float(rng.standard_normal()) for k in SIGNALS},
                "gauss_mi": quality + 0.1 * rng.standard_normal(),
                "memorisation": -quality + 0.1 * rng.standard_normal(),
            },
            "truth": {t: quality + 0.05 * rng.standard_normal() for t in TRUTHS},
        }

    rows = {f"r{g}{i}": fake(rng.random(), f"bank{g}") for g in range(3) for i in range(8)}
    result = analyse(rows, "bank")
    good, bad = result["signals"]["gauss_mi"], result["signals"]["memorisation"]
    assert good["usable"] and good["own_mean"]["rho"] > 0.8 and good["own_mean"]["regret_removed"] > 0.7, good
    assert bad["usable"] and bad["own_mean"]["rho"] < -0.8, bad
    assert not result["signals"]["top1_all"]["usable"]
    assert not result["signals"]["twonn"]["usable"]
    assert set(analyse(rows, "bank_width")["groups"]) == {f"bank{g} t1_04 c24" for g in range(3)}
    print("criterion_check self-test passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("runs", type=Path, nargs="*")
    p.add_argument("--fit-subjects", type=int, default=1000, help="training subjects that fit the cross-view maps")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.out is None:
        p.error("OUT_JSON is required")
    rows = json.loads(a.out.read_text()) if a.out.exists() else {}
    runs = [r for r in a.runs if (r / "model.pt").exists() and (r / "report.json").exists()]
    if runs:
        import warm

        device = warm.pick_device()
    for run in runs:
        report = json.loads((run / "report.json").read_text())
        if supervised(report):
            print(f"skip {run.name}: trained with labels", flush=True)
            continue
        import torch

        step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
        if run.name in rows and rows[run.name]["step"] == step:
            continue
        try:
            rows[run.name] = assess(run, device, a.fit_subjects)
        except Exception as error:  # an older run this loader cannot rebuild: say so and go on
            print(f"skip {run.name}: {type(error).__name__}: {error}", flush=True)
            continue
        a.out.write_text(json.dumps(rows, indent=1))
        print(f"done {run.name}", flush=True)
    if not rows:
        sys.exit("no runs assessed")
    analyses = {g: analyse(rows, g) for g in GROUPINGS}
    show(rows, analyses)
    a.out.with_suffix(".analysis.json").write_text(json.dumps(analyses, indent=1))


if __name__ == "__main__":
    main()
