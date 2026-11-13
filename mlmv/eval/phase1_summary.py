"""Gate 1 tables from finished Phase 1 runs (never re-scores; reads each run's ``eval/leakage.json``).

    python -m mlmv.eval.phase1_summary --runs results/mlmv/p1_* --out results/mlmv/phase1_summary.md

Tables: the leakage matrix per run and modality (best of ridge / MLP R^2), the pass checks, site per block,
the f_L curve (noisy, tau^2 = 0.5), the noisy tau^2 trend, the crowding-out test (w from S, centring on vs
off), the one-modality lesion check and the counterfactual site change per block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FACTORS = ("g", "site", "r", "s", "w", "lesion")
LEVELS = ("G", "L", "S")
MODALITIES = ("t1", "flair")


def load(run_dirs):
    runs = {}
    for d in run_dirs:
        path = Path(d) / "eval" / "leakage.json"
        if path.exists():
            runs[Path(d).name] = json.loads(path.read_text())
    return runs


def best(result, modality, block, factor):
    for r in result["rows"]:
        if (r["modality"], r["block"], r["factor"]) == (modality, block, factor):
            return r["best"]
    return float("nan")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f3(x):
    return f"{x:.3f}"


def summarise(runs, thresholds=None) -> str:
    parts = ["# Phase 1 summary", ""]
    for name, res in sorted(runs.items()):
        parts += [
            f"## {name}",
            f"variant {res['variant']}, objective {res['objective']}, centring {res['centring']}, "
            f"f_L {res['fraction']}, tau^2 {res['noise_tau2']}, {res['n_subjects']} test subjects",
            "",
        ]
        for m in MODALITIES:
            rows = [[lvl] + [f3(best(res, m, lvl, f)) for f in FACTORS] for lvl in LEVELS]
            parts += [f"Leakage matrix ({m}, best R^2)", "", table(["block", *FACTORS], rows), ""]
        checks = res["checks"]
        failed = [c for c in checks if not c["passed"]]
        parts += [
            f"Pass checks: {len(checks) - len(failed)}/{len(checks)}"
            + (
                ""
                if not failed
                else "; failed: "
                + ", ".join(
                    f"{c['modality']} {c['block']}<-{c['factor']} {c['value']:.3f} (needs {c['rule']})" for c in failed
                )
            ),
            "",
        ]
        site = [
            [r["modality"], r["block"], f3(r["best"]), f3(r["site_balanced_accuracy"])]
            for r in res["rows"]
            if r["factor"] == "site"
        ]
        parts += [
            "Site per block (chance accuracy 1/6)",
            "",
            table(["modality", "block", "R^2 (embedding)", "balanced acc"], site),
            "",
        ]
        cf = [[c["modality"], c["block"], f3(c["relative_l2_change"])] for c in res["counterfactual"]]
        parts += [
            "Counterfactual: same latents, different site",
            "",
            table(["modality", "block", "relative L2 change"], cf),
            "",
        ]

    noisy = {n: r for n, r in runs.items() if r["variant"] == "noisy" and r["objective"] == "barlow" and r["centring"]}
    curve = sorted((r["fraction"], n, r) for n, r in noisy.items() if abs(r["noise_tau2"] - 0.5) < 1e-9)
    if curve:
        rows = [[fr, n] + [f3(best(r, m, "L", "r")) for m in MODALITIES] for fr, n, r in curve]
        vals = [best(r, "t1", "L", "r") for _, _, r in curve]
        rising = all(b >= a for a, b in zip(vals, vals[1:]))
        parts += [
            "## f_L curve (noisy, tau^2 = 0.5): R^2 of r from L",
            "",
            table(["f_L", "run", "t1", "flair"], rows),
            "",
            f"Rises with f_L (t1): {rising}",
            "",
        ]
    trend = sorted((r["noise_tau2"], n, r) for n, r in noisy.items() if r["fraction"] == 1.0)
    if trend:
        rows = [
            [t, n] + [f3(best(r, m, "L", "r")) for m in MODALITIES] + [f3(best(r, "t1", "G", "g"))] for t, n, r in trend
        ]
        parts += [
            "## Noisy sharing: R^2 of r from L and g from G against tau^2 (f_L = 1)",
            "",
            table(["tau^2", "run", "L<-r t1", "L<-r flair", "G<-g t1"], rows),
            "",
        ]
    on = [
        r
        for r in runs.values()
        if r["variant"] == "exact" and r["objective"] == "barlow" and r["centring"] and r["fraction"] == 1.0
    ]
    off = [r for r in runs.values() if r["variant"] == "exact" and r["objective"] == "barlow" and not r["centring"]]
    if on and off:
        rows, gaps = [], []
        for m in MODALITIES:
            a, b = best(on[0], m, "S", "w"), best(off[0], m, "S", "w")
            gaps.append(a - b)
            rows.append([m, f3(a), f3(b), f3(a - b)])
        parts += [
            "## Crowding out: R^2 of w from S",
            "",
            table(["modality", "centring on", "centring off", "difference"], rows),
            "",
            f"Passes (difference >= 0.1 in both modalities): {all(g >= 0.1 for g in gaps)}",
            "",
        ]
    lesion = [(n, r) for n, r in runs.items() if r["variant"] == "one_modality_lesion"]
    for n, r in lesion:
        rows = [[m, f3(best(r, m, "S", "lesion"))] for m in MODALITIES]
        parts += [
            f"## One-modality lesion ({n}): R^2 of the lesion from S (expected <= 0.1, no partner view)",
            "",
            table(["modality", "R^2"], rows),
            "",
        ]
    return "\n".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument("--out", default=None, help="markdown output path")
    args = p.parse_args(argv)
    text = summarise(load(args.runs))
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
