"""Stage 78 (no training): do TailCode's stored gauge statistics change any synthetic verdict? Rules: HANDOFF stage 78.

    python gauge_check.py neighbourhood OUT.json RUN...      # GPU, described only: where the stored averages come from
    python gauge_check.py summary OUT.json --part A STORED.json RECOMPUTED.json [--part B ...] \
        [--recorded REPORT.json...] [--neighbourhood NB.json] > OUT.txt

The gauge is TailCode's per-view BatchNorm between the pooled features and the head. Every stage so far read the code
through its stored running averages; `pool_check.py export/detect --gauge recomputed` reads it with the gauge
recomputed over the bank's clean training images. summary sets the two side by side for every code of each part (from
both detect reports and both exports): the hold verdict and its margin (the smallest of the own-parts spectrum's last
value and the ventricle-burden worst direction, T1 and FLAIR), rho1, cross-view top-1 among all test subjects
(code_audit.top1, as holds.py reads it), how linearly the recomputed code follows from the stored one, and per view
how far the stored averages sit from the recomputed ones (offsets in recomputed SDs and log variance ratios, BatchNorm's
eps added to every variance); then the stage's rules on both gauges. --recorded checks the stored reading against the
stages' own reports (stage 76's detect json, stage 75's pool_check.json).
neighbourhood: every training subject's pooled features (clean images, the network frozen), batched as training
batches them (each subject with its code neighbours in the stored code, as warm.py rebuilds them), and the batches'
average mean and variance against the stored averages and the population's.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import pool_check

EPS = 1e-5  # nn.BatchNorm1d's eps
GAP_WIDTH = 64  # TailCode's pooled features: the GAP's 64 channels, then the tails
MARGIN_MOVE = 0.05
PARTS = {"all": slice(None), "gap": slice(0, GAP_WIDTH), "tails": slice(GAP_WIDTH, None)}


def distance(mean_a, var_a, mean_b, var_b):
    """Per feature, a against b: the offset in b's SDs and log(a's variance / b's)."""
    return (mean_a - mean_b) / np.sqrt(var_b + EPS), np.log((var_a + EPS) / (var_b + EPS))


def summarized(offset, log_ratio):
    """(2, width) offsets and log ratios -> per view and feature group: mean and median |offset|, mean log ratio and
    mean |log ratio|."""
    out = []
    for v in range(2):
        row = {}
        for part, cols in PARTS.items():
            o, r = np.abs(offset[v, cols]), log_ratio[v, cols]
            row[part] = {
                "mean_abs_offset": float(o.mean()),
                "median_abs_offset": float(np.median(o)),
                "mean_log_ratio": float(r.mean()),
                "mean_abs_log_ratio": float(np.abs(r).mean()),
            }
        out.append(row)
    return out


def margin(row):
    return float(min(row["last"] + row["worst"]))


def linear_fit(source, target):
    """Per unit, R2 of the target code from the source code by least squares with an intercept, (2, n, d) each."""
    out = []
    for v in range(2):
        design = np.column_stack([np.ones(len(source[v])), source[v]])
        fitted = design @ np.linalg.lstsq(design, target[v], rcond=None)[0]
        out.append(1 - ((target[v] - fitted) ** 2).sum(0) / ((target[v] - target[v].mean(0)) ** 2).sum(0))
    return np.stack(out)


def verdict_rule(rows):
    """The stage's verdict rule over rows (label -> {'stored': {...}, 'recomputed': {...}} with holds and margin)."""
    flips = sorted(x for x, r in rows.items() if r["stored"]["holds"] != r["recomputed"]["holds"])
    moved = sorted(x for x, r in rows.items() if abs(r["recomputed"]["margin"] - r["stored"]["margin"]) >= MARGIN_MOVE)
    return {
        "outcome": "verdicts move" if flips else "values move" if moved else "no effect",
        "to_hold": [x for x in flips if rows[x]["recomputed"]["holds"]],
        "to_fail": [x for x in flips if not rows[x]["recomputed"]["holds"]],
        "margins_moved": moved,
        "holds": {g: sum(r[g]["holds"] for r in rows.values()) for g in pool_check.GAUGES},
        "codes": len(rows),
    }


def rho_rule(rows, gauge, part, k=3):
    """Stages 75's and 76's rules on one gauge's holds and rho1: part 'A' needs the AUC and the keep-the-lowest-rho1
    selection over k-subsets of each cell; part 'B' the AUC and 3/4 of within-cell pairs."""
    members = [{"rho1": r[gauge]["rho1"], "holds": r[gauge]["holds"], "cell": r["cell"]} for r in rows.values()]
    holders = [m["rho1"] for m in members if m["holds"]]
    failures = [m["rho1"] for m in members if not m["holds"]]
    enough = len(holders) >= 2 and len(failures) >= 2
    value = pool_check.auc(failures, holders) if holders and failures else float("nan")
    right = pairs = with_holder = picked = 0
    random_pick = 0.0
    for cell in sorted({m["cell"] for m in members}):
        inside = [m for m in members if m["cell"] == cell]
        cell_pairs = [(h, f) for h in inside if h["holds"] for f in inside if not f["holds"]]
        pairs += len(cell_pairs)
        right += sum(f["rho1"] > h["rho1"] for h, f in cell_pairs)
        if len(inside) > k:
            s = pool_check.selection(inside, k)
            with_holder, picked, random_pick = (
                with_holder + s["with_holder"],
                picked + s["picked"],
                random_pick + s["random"],
            )
    share = picked / with_holder if with_holder else float("nan")
    if not enough:
        outcome = "described only (fewer than 2 holders or 2 failures)"
    elif part == "A":
        outcome = (
            "stands" if value >= 0.80 and share >= 0.90 else "fails" if value <= 0.65 or share < 0.70 else "unresolved"
        )
    else:
        outcome = "stands" if value >= 0.80 and right >= 0.75 * pairs else "fails" if value <= 0.65 else "unresolved"
    return {
        "auc": value,
        "pairs": pairs,
        "right": right,
        "holders": len(holders),
        "failures": len(failures),
        "k": k,
        "with_holder": with_holder,
        "picked": picked,
        "random": random_pick,
        "share": share,
        "outcome": outcome,
    }


def recorded_rho1(paths):
    """label -> rho1 from the stages' own reports: detect's {'rows': {label: {'rho1'}}} or stage 75's
    {'leftover': {label: {'rho': [...]}}}."""
    out = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        for label, row in report.get("rows", {}).items():
            out[label] = row["rho1"]
        for label, row in report.get("leftover", {}).items():
            out[label] = row["rho"][0]
    return out


def neighbourhood(out_json, runs):
    """Described only: each run's stored averages against the average within-batch statistics of code-neighbourhood
    batches, and both against the population's, on clean training images."""
    import code_audit
    import torch
    import warm

    import data.cluster_batches as cb

    report = json.loads(out_json.read_text()) if out_json.exists() else {}
    device = warm.pick_device()
    defaults = warm.run_args()
    banks = {}
    for run in map(Path, runs):
        if run.name in report:
            print("reuse", run.name, flush=True)
            continue
        info, model, pool, residual, step = code_audit.load_run(run, device, trained=True)
        a = info["args"]
        key = (a["bank"], a["t1key"], residual)
        if key not in banks:
            banks[key] = warm.Bank(a["bank"], "train", a["t1key"], residual=residual)
        train, size = banks[key], defaults.batch_size
        pooled = [[], []]
        with torch.no_grad():
            for start in range(0, len(train), size):
                maps = model._encode(train.images(np.arange(start, min(start + size, len(train))), device), 2, None)
                for v, part in enumerate(maps.chunk(2)):
                    pooled[v].append(pool.ungauged(part, v, update=False).double().cpu().numpy())
        pooled = np.stack([np.concatenate(p) for p in pooled])
        code = np.load(pool_check.POOL / f"{run.name}.npz")["code_train"]
        count = (
            defaults.stratify_neighbours if a.get("code_neighbours", "None") == "None" else int(a["code_neighbours"])
        )
        sampler = cb.NeighbourhoodBatchSampler(cb.nearest_neighbours(np.concatenate(list(code), 1), count), size, 0)
        batches = []
        while len(batches) < len(train):
            batches.extend(np.asarray(b) for b in sampler)
        batches = batches[: len(train)]
        means = np.stack([pooled[:, b].mean(1) for b in batches])
        variances = np.stack([pooled[:, b].var(1, ddof=1) for b in batches])
        hood = means.mean(0), variances.mean(0)
        population = pooled.mean(1), pooled.var(1)
        stored = pool_check.gauge_statistics(pool)
        report[run.name] = {
            "step": step,
            "neighbours": count,
            "batches": len(batches),
            "stored_vs_neighbourhood": summarized(*distance(*stored, *hood)),
            "neighbourhood_vs_population": summarized(*distance(*hood, *population)),
            "stored_vs_population": summarized(*distance(*stored, *population)),
            "batch_mean_offset_rms": [
                float(np.sqrt(((means[:, v] - population[0][v]) ** 2 / (population[1][v] + EPS)).mean()))
                for v in range(2)
            ],
        }
        ratios = {
            k: log_ratios(report[run.name][k]) for k in ("stored_vs_neighbourhood", "neighbourhood_vs_population")
        }
        print(
            f"{run.name}: log variance ratio T1/FLAIR, stored/neighbourhood {ratios['stored_vs_neighbourhood']},"
            f" neighbourhood/population {ratios['neighbourhood_vs_population']},"
            f" stored/population {log_ratios(report[run.name]['stored_vs_population'])}",
            flush=True,
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=1) + "\n")
        del model, pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pair(values, fmt="{:.2f}"):
    return "/".join(fmt.format(v) for v in values)


def log_ratios(views):
    """T1/FLAIR mean log variance ratio over all features, from summarized()'s output."""
    return pair([x["all"]["mean_log_ratio"] for x in views], "{:+.2f}")


def summary(out_json, parts, recorded=(), neighbourhood_json=None):
    import code_audit

    known = recorded_rho1(recorded)
    hoods = json.loads(Path(neighbourhood_json).read_text()) if neighbourhood_json else {}
    report = {"parts": {}, "rules": {}}
    every = {}
    for part, (stored_json, recomputed_json) in parts.items():
        reads = {
            g: json.loads(Path(p).read_text())["rows"]
            for g, p in zip(pool_check.GAUGES, (stored_json, recomputed_json))
        }
        if set(reads["stored"]) != set(reads["recomputed"]):
            raise SystemExit(f"part {part}: the two reports score different codes")
        rows = {}
        for label, s in reads["stored"].items():
            r = reads["recomputed"][label]
            if s["run"] != r["run"]:
                raise SystemExit(f"{label}: the two reports read different runs")
            exports = {g: np.load(pool_check.pool_dir(g) / f"{s['run']}.npz") for g in pool_check.GAUGES}
            for key in ("gap_train", "gap_test", "zc_test"):
                if np.abs(exports["stored"][key] - exports["recomputed"][key]).max() > 1e-4:
                    raise SystemExit(f"{label}: {key} differs between the exports; the gauge sits after the GAP")
            e = exports["recomputed"]
            row = {"run": s["run"], "cell": s["cell"], "stage_holds": s.get("stage_holds")}
            for g, x in (("stored", s), ("recomputed", r)):
                code = exports[g]["code_test"]
                row[g] = {
                    "holds": x["holds"],
                    "last": x["last"],
                    "worst": x["worst"],
                    "margin": margin(x),
                    "rho1": x["rho1"],
                    "rho2": x["rho2"],
                    "top1": code_audit.top1(code[0], code[1], [np.arange(code.shape[1])]),
                }
            row["recorded_rho1"] = known.get(label)
            row["linear_r2"] = linear_fit(exports["stored"]["code_test"], e["code_test"]).tolist()
            row["distance"] = summarized(
                *distance(
                    e["gauge_stored_mean"], e["gauge_stored_var"], e["gauge_recomputed_mean"], e["gauge_recomputed_var"]
                )
            )
            row["neighbourhood"] = hoods.get(s["run"])
            rows[label] = row
        report["parts"][part] = rows
        report["rules"][part] = {
            "verdicts": verdict_rule(rows),
            **{g: rho_rule(rows, g, part) for g in pool_check.GAUGES},
        }
        every.update(rows)
    report["rules"]["all"] = {"verdicts": verdict_rule(every)}
    for g in pool_check.GAUGES:
        report["rules"]["all"][g] = {
            "pairs": sum(report["rules"][p][g]["pairs"] for p in parts),
            "right": sum(report["rules"][p][g]["right"] for p in parts),
        }
    from scipy.stats import spearmanr

    s1 = [r["stored"]["rho1"] for r in every.values()]
    r1 = [r["recomputed"]["rho1"] for r in every.values()]
    report["described"] = {
        "rho1_spearman": float(spearmanr(s1, r1)[0]),
        "rho1_largest_change": float(np.max(np.abs(np.subtract(r1, s1)))),
        "top1_mean_change": float(np.mean([r["recomputed"]["top1"] - r["stored"]["top1"] for r in every.values()])),
        "recorded_rho1_largest_difference": max(
            (abs(r["stored"]["rho1"] - r["recorded_rho1"]) for r in every.values() if r["recorded_rho1"] is not None),
            default=None,
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, default=float) + "\n")
    show(report)
    print(f"\nSaved {out_json}")
    return report


def show(report):
    print("Stage 78: each code read with the gauge's stored averages (s) and with the gauge recomputed over the 2,000")
    print("clean training images (r). Holds: the spectrum's last value and the ventricle-burden worst direction each")
    print(
        ">= .40 in both views; margin = the smallest of the four. top-1: among all test subjects. Distances, T1/FLAIR,"
    )
    print(
        "over the 448 pooled features: |offset| = mean |stored - recomputed mean| in recomputed SDs; log var = mean of"
    )
    print("log(stored / recomputed variance), below 0 = stored narrower. lin R2: the recomputed code's least-squares")
    print("fit from the stored one, worst unit, T1/FLAIR.")
    for part, rows in report["parts"].items():
        print(f"\n== part {part} ({len(rows)} codes)")
        print(
            f"  {'code':16s} {'holds s>r':10s} {'margin s > r':16s} {'rho1 s > r':11s} {'top-1 s > r':12s}"
            f" {'|offset|':10s} {'log var':12s} {'lin R2':9s}"
        )
        for label, row in rows.items():
            s, r = row["stored"], row["recomputed"]
            flag = "  FLIP" if s["holds"] != r["holds"] else ""
            if row["stage_holds"] is not None and row["stage_holds"] != s["holds"]:
                flag += "  stored differs from the stage's verdict"
            d = row["distance"]
            print(
                f"  {label:16s} {'yes' if s['holds'] else 'no '} > {'yes' if r['holds'] else 'no '}  "
                f"{s['margin']:5.2f} > {r['margin']:5.2f}  {s['rho1']:.2f} > {r['rho1']:.2f}  "
                f"{s['top1']:.3f} > {r['top1']:.3f}  {pair([x['all']['mean_abs_offset'] for x in d]):10s} "
                f"{pair([x['all']['mean_log_ratio'] for x in d], '{:+.2f}'):12s} "
                f"{pair([min(v) for v in row['linear_r2']]):9s}{flag}"
            )
        rules = report["rules"][part]
        v = rules["verdicts"]
        print(
            f"  VERDICTS: {v['outcome']}; holds {v['holds']['stored']} stored, {v['holds']['recomputed']} recomputed of"
            f" {v['codes']}; flips to hold {v['to_hold'] or 'none'}, to fail {v['to_fail'] or 'none'}; margins moved"
            f" by >= {MARGIN_MOVE:.2f}: {v['margins_moved'] or 'none'}"
        )
        for g in pool_check.GAUGES:
            x = rules[g]
            selection = (
                f"; lowest rho1 of {x['k']} holds in {x['picked']}/{x['with_holder']} subsets with a holder"
                f" (random {x['random']:.1f})"
                if x["with_holder"]
                else ""
            )
            print(
                f"  RHO1 ({g}): AUC {x['auc']:.2f} over {x['failures']} failures and {x['holders']} holders;"
                f" within-cell pairs right {x['right']}/{x['pairs']}{selection}: {x['outcome']}"
            )
    a = report["rules"]["all"]
    print(f"\n== all codes: VERDICTS {a['verdicts']['outcome']}")
    for g in pool_check.GAUGES:
        print(f"  within-cell pairs right ({g}): {a[g]['right']}/{a[g]['pairs']}")
    d = report["described"]
    print(
        f"  rho1 stored against recomputed: Spearman {d['rho1_spearman']:.2f}, largest change"
        f" {d['rho1_largest_change']:.2f}; top-1 changes by {d['top1_mean_change']:+.3f} on average"
    )
    if d["recorded_rho1_largest_difference"] is not None:
        print(
            "  stored rho1 against the stages' own reports: largest difference"
            f" {d['recorded_rho1_largest_difference']:.4f}"
        )
    hoods = [(label, row["neighbourhood"]) for rows in report["parts"].values() for label, row in rows.items()]
    hoods = [(label, h) for label, h in hoods if h]
    if hoods:
        print("\n== described: where the stored averages come from (log variance ratio, all features, T1/FLAIR)")
        print(
            f"  {'code':16s} {'stored/hood':12s} {'hood/pop':12s} {'stored/pop':12s} {'one batch mean, RMS offset':s}"
        )
        for label, h in hoods:
            print(
                f"  {label:16s} {log_ratios(h['stored_vs_neighbourhood']):12s}"
                f" {log_ratios(h['neighbourhood_vs_population']):12s} {log_ratios(h['stored_vs_population']):12s}"
                f" {pair(h['batch_mean_offset_rms'])} SDs"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    n = sub.add_parser("neighbourhood")
    n.add_argument("out", type=Path)
    n.add_argument("runs", nargs="+", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("out", type=Path)
    s.add_argument("--part", nargs=3, action="append", required=True, metavar=("NAME", "STORED", "RECOMPUTED"))
    s.add_argument("--recorded", nargs="*", default=[])
    s.add_argument("--neighbourhood", default=None)
    a = p.parse_args()
    if a.command == "neighbourhood":
        neighbourhood(a.out, a.runs)
    else:
        summary(a.out, {name: (stored, recomputed) for name, stored, recomputed in a.part}, a.recorded, a.neighbourhood)


if __name__ == "__main__":
    main()
