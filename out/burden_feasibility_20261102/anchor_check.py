"""Can measured anchors choose runs where the label-free signals could not? Stage 67.

Stage 66 found that only cross-view retrieval passed as a label-free way to choose runs, and narrowly. Choosing by how
well a code reads the measured factors' exact own parts carried over to hidden factors far better. ADNI gives neither
exact values nor own parts: it gives noisy measurements (FreeSurfer volumes, the UC Davis WMH volume). This check repeats
that choice with what ADNI would give.

export  (GPU, once per run) each finished warm.py run's content code on its bank's test subjects, with the subjects'
        latents, lesion volumes and each factor's own part, into CODES/<run>.npz (reused while newer than model.pt).
score   (no torch) for every run in CODES:
  truth     each factor's own part read from each view's code (5-fold ridge: own_variation.py's readout, as in stage
            66), averaged over the views
  measured  the anchors as ADNI would measure them: brain, ventricle, thickness, temporal, lr and sulcal as their latent
            values, burden as the rendered lesion volume (the WMH volume), each plus measurement noise at reliability r
            (the share of the measured variance that is true; r = 1 is exact). One noise draw is shared by every run on
            a bank, as one dataset has one set of measurements. DRAWS draws per r.
  raw       score = R2 of the measured anchors from the code, averaged over the anchors and the views
  adjusted  the same, with each anchor first adjusted for the other measured anchors (minus a cross-fitted cubic ridge
            fit on them): the ADNI stand-in for an own part
  exact     reference: the measured factors' exact own parts, no noise (stage 66's comparison)
  signals   reference with --signals (stage 66's criterion.json): retrieval and the backbone's MI gain, label-free
Each score picks runs within each bank (grouping 'bank', then 'bank_width'), judged on the hidden factors' own parts:
Spearman within groups, size-weighted, and the share of a random pick's shortfall that the pick removes
(criterion_check.py). Splits: 'adni' hides lr and sulcal (the rest measured, as FreeSurfer and the WMH volume allow);
'no_wmh' hides burden; and every split hiding one to three factors at r = .9, averaged over the splits.

Pass rule, fixed before any result: anchors are usable to choose runs on ADNI if, for the 'adni' split with adjusted
anchors at r = .9 under grouping 'bank', the bootstrap 95% interval of rho (averaged over draws, runs resampled within
groups) lies above 0 and the share of the shortfall removed (averaged over draws) is at least .5. They are preferred
over retrieval if that share is also higher than top1_32's on the same runs and hidden factors.

With --signals, only runs in both are scored, so every row compares the same runs.

    python anchor_check.py export CODES RUN [RUN ...]
    python anchor_check.py score CODES --out OUT_JSON [--signals STAGE66_JSON] [--draws 10] [--split-draws 3]
    python anchor_check.py --self-test
"""

import argparse
import itertools
import json
import sys
import warnings
import zlib
from pathlib import Path

import criterion_check as C
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

FACTORS = C.FACTOR_NAMES  # brain, ventricle, burden, thickness, temporal, lr, sulcal
LATENT_COLUMNS = (0, 1, 2, 5, 6, 7, 8)  # their columns of the bank's zc (bank.CAUSAL_FACTORS)
BURDEN = C.BURDEN
SPLITS = {"adni": (5, 6), "no_wmh": (BURDEN,)}  # hidden factors
HIDE = tuple(s for k in (1, 2, 3) for s in itertools.combinations(range(len(FACTORS)), k))
RELIABILITIES = (1.0, 0.9, 0.75, 0.5)
SPLIT_R = 0.9
RULE = ("adni", "adjusted", 0.9, "bank")
REFERENCE_SIGNALS = ("top1_32", "top1_all", "gap_gauss_mi_gain")
ALPHAS = np.logspace(-2, 5, 15)
CLIP = 5.0


def clipped(x):
    return np.clip(x, -CLIP, CLIP)


def readout_r2(x, y):
    """5-fold R2 of a ridge readout of every column of ``y`` from ``x``: own_variation.ridge's pipeline (its penalty
    chosen per column inside each training fold), for many columns at once."""
    y = np.asarray(y, dtype=np.float64)
    y = y[:, None] if y.ndim == 1 else y
    pipe = make_pipeline(StandardScaler(), FunctionTransformer(clipped), RidgeCV(alphas=ALPHAS, alpha_per_target=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(pipe, x, y, cv=KFold(5, shuffle=True, random_state=0))
    pred = pred.reshape(y.shape)
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def measured_values(zc, vox):
    """The seven anchors as measured: latent values, burden as the lesion volume."""
    m = np.asarray(zc, dtype=np.float64)[:, LATENT_COLUMNS].copy()
    m[:, BURDEN] = np.asarray(vox, dtype=np.float64)
    return m


def with_noise(m, reliability, rng):
    """``m`` plus independent noise, so that the true share of each column's variance is ``reliability``."""
    if reliability >= 1:
        return m.copy()
    return m + rng.standard_normal(m.shape) * m.std(0) * np.sqrt((1 - reliability) / reliability)


def adjusted(m, measured):
    """Each measured column minus a cross-fitted cubic ridge fit on the other measured columns."""
    out = {}
    for k in measured:
        others = [j for j in measured if j != k]
        if not others:
            out[k] = m[:, k] - m[:, k].mean()
            continue
        pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(3, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = cross_val_predict(pipe, m[:, others], m[:, k], cv=KFold(5, shuffle=True, random_state=1))
        out[k] = m[:, k] - fit
    return out


def plan(draws, split_draws):
    """{(r, draw): splits needing adjusted columns}: the named splits at every r, every hiding split at SPLIT_R."""
    out = {}
    for r in RELIABILITIES:
        for d in range(draws):
            out[(r, d)] = set(SPLITS.values())
    for d in range(split_draws):
        out.setdefault((SPLIT_R, d), set()).update(HIDE)
    return out


def bank_targets(zc, vox, key, draws, split_draws):
    """(matrix, columns): every anchor target for one bank's subjects. Columns are ('raw', r, d, k) and
    ('adjusted', r, d, split, k); one noise draw per (bank, r, d), shared by the raw and every adjusted column."""
    m0 = measured_values(zc, vox)
    blocks, columns = [], []
    for (r, d), splits in sorted(plan(draws, split_draws).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        m = with_noise(m0, r, np.random.default_rng([zlib.crc32(key.encode()), int(round(r * 1000)), d]))
        for k in range(len(FACTORS)):
            columns.append(("raw", r, d, k))
            blocks.append(m[:, k])
        for split in sorted(splits):
            measured = [k for k in range(len(FACTORS)) if k not in split]
            for k, values in adjusted(m, measured).items():
                columns.append(("adjusted", r, d, split, k))
                blocks.append(values)
    return np.column_stack(blocks), columns


def read_entry(entry, targets):
    """Own-part R2 (7) and every anchor column's R2, each the mean of the two views."""
    y = np.column_stack([entry["own"], targets])
    r2 = np.mean([readout_r2(code, y) for code in entry["codes"]], 0)
    return r2[: len(FACTORS)], r2[len(FACTORS) :]


def score(entries, draws, split_draws):
    """Adds 'own_r2' and 'anchor_r2' to every entry; returns the anchor column layout (the same for every bank)."""
    by_bank = {}
    for e in entries:
        by_bank.setdefault((e["bank_root"], len(e["zc"])), []).append(e)
    layout = None
    for (root, n), members in sorted(by_bank.items()):
        first = members[0]
        for e in members[1:]:
            if not (np.allclose(e["zc"], first["zc"]) and np.allclose(e["vox"], first["vox"])):
                raise ValueError(f"{e['run']} and {first['run']} disagree on {root}'s test subjects")
        targets, columns = bank_targets(first["zc"], first["vox"], f"{Path(root).name}:{n}", draws, split_draws)
        if layout is None:
            layout = columns
        assert columns == layout
        for e in members:
            e["own_r2"], e["anchor_r2"] = read_entry(e, targets)
            print(f"read {e['run']}", flush=True)
    return {c: i for i, c in enumerate(layout or [])}


def anchor_score(e, position, variant, r, d, split):
    measured = [k for k in range(len(FACTORS)) if k not in split]
    if variant == "raw":
        idx = [position[("raw", r, d, k)] for k in measured]
    else:
        idx = [position[("adjusted", r, d, split, k)] for k in measured]
    return float(np.mean(e["anchor_r2"][idx]))


def hidden_truth(e, split):
    return float(np.mean(e["own_r2"][list(split)]))


def grouped(entries, grouping):
    g = {}
    for e in entries:
        g.setdefault(C.group_of(e, grouping), []).append(e)
    return [v for v in g.values() if len(v) >= C.MIN_GROUP]


def judge(groups, pick, truth):
    """(pooled rho, share of the shortfall removed) of choosing by ``pick`` within each group."""
    arrays = [(np.array([pick(e) for e in g]), np.array([truth(e) for e in g])) for g in groups]
    return C.pooled_rho(arrays), C.regret_removed(arrays, 1)


def over_draws(groups, picks, truth):
    values = np.array([judge(groups, p, truth) for p in picks])
    return {
        "rho": float(np.nanmean(values[:, 0])),
        "rho_range": [float(np.nanmin(values[:, 0])), float(np.nanmax(values[:, 0]))],
        "removed": float(np.nanmean(values[:, 1])),
        "removed_range": [float(np.nanmin(values[:, 1])), float(np.nanmax(values[:, 1]))],
    }


def bootstrap(groups, picks, truth, seed=0):
    """95% interval of the draw-averaged pooled rho, runs resampled within groups."""
    rng = np.random.default_rng(seed)
    scores = [np.array([[p(e) for e in g] for p in picks]) for g in groups]  # per group: (draws, runs)
    truths = [np.array([truth(e) for e in g]) for g in groups]
    out = []
    for _ in range(C.BOOT):
        idx = [rng.integers(0, len(t), len(t)) for t in truths]
        rhos = [C.pooled_rho([(s[d][i], t[i]) for s, t, i in zip(scores, truths, idx)]) for d in range(len(picks))]
        out.append(np.mean(rhos) if np.isfinite(rhos).any() and np.isfinite(rhos).all() else float("nan"))
    out = np.array(out)
    out = out[np.isfinite(out)]
    return [float(q) for q in np.quantile(out, [0.025, 0.975])] if len(out) else [float("nan")] * 2


def split_name(split):
    named = {v: k for k, v in SPLITS.items()}
    return named.get(split, "hide " + ",".join(FACTORS[k] for k in split))


def assess(entries, position, draws, split_draws, signals=None):
    """Every row of the report, per grouping."""
    out = {}
    for grouping in ("bank", "bank_width"):
        groups = grouped(entries, grouping)
        result = {"groups": {C.group_of(g[0], grouping): len(g) for g in groups}, "splits": {}}
        if not groups:
            out[grouping] = result
            continue

        def rows_for(split, rs, nd):
            truth = lambda e: hidden_truth(e, split)  # noqa: E731
            rows = {}
            for variant in ("raw", "adjusted"):
                for r in rs:
                    picks = [
                        lambda e, v=variant, r=r, d=d: anchor_score(e, position, v, r, d, split) for d in range(nd)
                    ]
                    rows[f"{variant} r={r:.2f}"] = over_draws(groups, picks, truth)
            measured = [k for k in range(len(FACTORS)) if k not in split]
            rows["exact own parts"] = over_draws(groups, [lambda e: float(np.mean(e["own_r2"][measured]))], truth)
            for name in signals or ():
                direction = C.SIGNALS[name]
                rows[name] = over_draws(groups, [lambda e, n=name: direction * e["signals"][n]], truth)
            return rows

        for name, split in SPLITS.items():
            result["splits"][name] = {
                "hidden": [FACTORS[k] for k in split],
                "rows": rows_for(split, RELIABILITIES, draws),
            }
        for k in (1, 2, 3):
            per_split = [rows_for(s, (SPLIT_R,), split_draws) for s in HIDE if len(s) == k]
            if not per_split:
                continue
            result["splits"][f"hide{k}"] = {
                "hidden": f"every split hiding {k}, averaged over {len(per_split)} splits",
                "rows": {
                    row: {key: float(np.nanmean([p[row][key] for p in per_split])) for key in ("rho", "removed")}
                    for row in per_split[0]
                },
            }
        out[grouping] = result
    split = SPLITS[RULE[0]]
    groups = grouped(entries, RULE[3])
    picks = [lambda e, d=d: anchor_score(e, position, RULE[1], RULE[2], d, split) for d in range(draws)]
    cell = out[RULE[3]]["splits"][RULE[0]]["rows"][f"{RULE[1]} r={RULE[2]:.2f}"] if groups else None
    rule = {"cell": list(RULE), "usable": False}
    if cell:
        low, high = bootstrap(groups, picks, lambda e: hidden_truth(e, split))
        rule.update(rho=cell["rho"], low=low, high=high, removed=cell["removed"])
        rule["usable"] = bool(low > 0 and cell["removed"] >= 0.5)
        retrieval = out[RULE[3]]["splits"][RULE[0]]["rows"].get("top1_32")
        if retrieval:
            rule["retrieval_removed"] = retrieval["removed"]
            rule["preferred_over_retrieval"] = bool(rule["usable"] and cell["removed"] > retrieval["removed"])
    out["rule"] = rule
    return out


def show(entries, result, consistency):
    print(f"\n{len(entries)} runs. Truth: the hidden factors' own parts (ridge, test subjects, mean of the views).")
    if consistency is not None:
        verdict = "a match (stage 66 stored 4 decimals)" if consistency <= 5.1e-5 else "NOT a match: other checkpoints?"
        print(f"Own-part R2 against stage 66 on the same runs: largest difference {consistency:.1e}, {verdict}")
    rule = result["rule"]
    print(f"\nRule (fixed beforehand): split '{RULE[0]}', {RULE[1]} anchors, r = {RULE[2]}, grouping '{RULE[3]}'")
    if "rho" in rule:
        print(
            f"  rho {rule['rho']:+.2f} [{rule['low']:+.2f}, {rule['high']:+.2f}], shortfall removed {rule['removed']:+.2f}"
            f" -> usable: {'YES' if rule['usable'] else 'no'}"
        )
        if "retrieval_removed" in rule:
            print(
                f"  top1_32 removes {rule['retrieval_removed']:+.2f} on the same runs -> preferred over retrieval: "
                f"{'YES' if rule['preferred_over_retrieval'] else 'no'}"
            )
    else:
        print("  no group has enough runs")
    for grouping in ("bank", "bank_width"):
        part = result[grouping]
        print(f"\nGrouping '{grouping}': {len(part['groups'])} groups of at least {C.MIN_GROUP} runs")
        for group, n in part["groups"].items():
            print(f"  {n:3d}  {group}")
        for name, block in part["splits"].items():
            hidden = block["hidden"] if isinstance(block["hidden"], str) else "hidden " + ", ".join(block["hidden"])
            print(f"\n  {name}: {hidden}")
            print(f"    {'score':20s} {'rho':>6s} {'range':>15s} {'removed':>8s} {'range':>15s}")
            for row, v in block["rows"].items():
                rho_range = "" if "rho_range" not in v else "[{:+.2f},{:+.2f}]".format(*v["rho_range"])
                rem_range = "" if "removed_range" not in v else "[{:+.2f},{:+.2f}]".format(*v["removed_range"])
                print(f"    {row:20s} {v['rho']:+6.2f} {rho_range:>15s} {v['removed']:+8.2f} {rem_range:>15s}")
    print(
        "\nrho: Spearman within groups, size-weighted, mean over noise draws (range over draws). removed: share of a"
        " random pick's shortfall on the hidden factors that the score's pick removes, mean over draws. raw / adjusted:"
        " noisy measured anchors as read from the code, adjusted for the other measured anchors or not."
    )


def load(codes_dir):
    entries = []
    for path in sorted(Path(codes_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"]))
            entries.append(
                {
                    **meta,
                    "codes": [
                        np.asarray(d["code_t1"], dtype=np.float64),
                        np.asarray(d["code_flair"], dtype=np.float64),
                    ],
                    "zc": np.asarray(d["zc"], dtype=np.float64),
                    "vox": np.asarray(d["vox"], dtype=np.float64),
                    "own": np.asarray(d["own"], dtype=np.float64),
                }
            )
    return entries


def export(run, device, out):
    import bank as B
    import code_audit
    import own_variation
    import warm

    assert tuple(B.CAUSAL_FACTORS) == LATENT_COLUMNS and list(B.CAUSAL_FACTORS.values()) == FACTORS
    report, model, pool, residual, step = code_audit.load_run(run, device, trained=True)
    a = report["args"]
    root, key = a["bank"], a["t1key"]
    test = warm.Bank(root, "test", key, int(a.get("test_subjects", 1000)), residual)
    codes = warm.features(model, test, device, pool=pool)["code"]
    zc = np.asarray(test.zc, dtype=np.float64)
    columns = list(LATENT_COLUMNS)
    own = zc[:, columns]
    causal = Path(f"{root}/causal_adj.npy").exists()
    if causal:  # own_variation.py's targets, as in stage 66
        scm, scale, nonlinearity = own_variation.scm_settings()
        if not np.array_equal(scm["adj"], np.load(f"{root}/causal_adj.npy")):
            raise ValueError(f"{root}'s causal_adj.npy is not bank.causal_scm's graph")
        own = own_variation.targets(zc, scm, scale, nonlinearity, columns)["own"]
    meta = {
        "run": run.name,
        "step": step,
        "bank": Path(root).name,
        "bank_root": str(root),
        "t1key": key,
        "width": int(codes[0].shape[1]),
        "loss_type": a.get("contrastive_loss_type", "infonce"),
        "causal": causal,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        code_t1=np.asarray(codes[0], dtype=np.float32),
        code_flair=np.asarray(codes[1], dtype=np.float32),
        zc=zc,
        vox=np.asarray(test.vox, dtype=np.float64),
        own=np.asarray(own, dtype=np.float64),
        meta=json.dumps(meta),
    )


def self_test():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((300, 12)), rng.standard_normal((300, 5))
    y[:, 0] += x[:, :3].sum(1)
    many = readout_r2(x, y)
    for j in range(5):
        pipe = make_pipeline(StandardScaler(), FunctionTransformer(clipped), RidgeCV(alphas=ALPHAS))
        pred = cross_val_predict(pipe, x, y[:, j], cv=KFold(5, shuffle=True, random_state=0))
        one = 1 - ((y[:, j] - pred) ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum()
        assert abs(one - many[j]) < 1e-9, (j, one, many[j])
    m = rng.standard_normal((20000, 3))
    noisy = with_noise(m, 0.5, rng)
    assert all(abs(np.corrcoef(m[:, k], noisy[:, k])[0, 1] ** 2 - 0.5) < 0.03 for k in range(3))
    parent, own = rng.standard_normal(2000), rng.standard_normal(2000)
    child = np.where(parent > 0, parent, 0.2 * parent) + 0.5 * own
    adj = adjusted(np.column_stack([parent, child]), [0, 1])[1]
    assert np.corrcoef(adj, own)[0, 1] > 0.95
    print("anchor_check self-test passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    sub = p.add_subparsers(dest="command")
    e = sub.add_parser("export", help="each run's test codes and targets into CODES (GPU)")
    e.add_argument("codes", type=Path)
    e.add_argument("runs", type=Path, nargs="*")
    s = sub.add_parser("score", help="choose runs by noisy anchors and score the choice (no torch)")
    s.add_argument("codes", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--signals", type=Path, help="stage 66's criterion.json, for the label-free reference rows")
    s.add_argument("--draws", type=int, default=10, help="measurement-noise draws per reliability")
    s.add_argument("--split-draws", type=int, default=3, help="draws for the splits hiding one to three factors")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.command == "export":
        runs = [r for r in a.runs if (r / "model.pt").exists() and (r / "report.json").exists()]
        if not runs:
            return
        import torch
        import warm

        device = warm.pick_device()
        for run in runs:
            out = a.codes / f"{run.name}.npz"
            if out.exists() and out.stat().st_mtime > (run / "model.pt").stat().st_mtime:
                continue
            if C.supervised(json.loads((run / "report.json").read_text())):
                print(f"skip {run.name}: trained with labels", flush=True)
                continue
            try:
                export(run, device, out)
            except Exception as error:  # an older run this loader cannot rebuild: say so and go on
                print(f"skip {run.name}: {type(error).__name__}: {error}", flush=True)
                continue
            print(f"exported {run.name}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
        return
    if a.command != "score":
        p.error("choose export or score")
    entries = load(a.codes)
    signals, consistency = None, None
    if a.signals:
        stage66 = json.loads(a.signals.read_text())
        kept = [e for e in entries if e["run"] in stage66 and stage66[e["run"]]["step"] == e["step"]]
        print(f"{len(kept)} of {len(entries)} runs are in {a.signals} at the same step; scoring those", flush=True)
        entries = kept
        for e in entries:
            e["signals"] = stage66[e["run"]]["signals"]
        signals = [s for s in REFERENCE_SIGNALS if all(e["signals"].get(s) is not None for e in entries)]
    if not entries:
        sys.exit("no runs to score")
    position = score(entries, a.draws, a.split_draws)
    if a.signals:
        consistency = max(
            float(np.abs(np.mean(stage66[e["run"]]["own_ridge"], 0) - e["own_r2"]).max()) for e in entries
        )
    result = assess(entries, position, a.draws, a.split_draws, signals)
    show(entries, result, consistency)
    runs = {
        e["run"]: {
            "bank": e["bank"],
            "t1key": e["t1key"],
            "width": e["width"],
            "step": e["step"],
            "own_r2": np.round(e["own_r2"], 4).tolist(),
        }
        for e in entries
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"runs": runs, **result, "consistency_with_stage66": consistency}, indent=1))


if __name__ == "__main__":
    main()
