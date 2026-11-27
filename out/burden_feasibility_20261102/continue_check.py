"""Stage 86 (rules in HANDOFF section 9, written before the run): stage 81's two failing 8-unit codes, continued
60,000 more steps from their 60,000-step models on Run:ai. Is seed 4's lost direction slow learning or a stable
trade? As long_check.py (stage 81), with its own folder, start and rules.

    python continue_check.py points RUN...         # checkpoint pseudo-runs in causal/stage86/at/; prints LABEL=PATH
    python continue_check.py summary DETECT.json   # trajectories, own parts and curves, and the rules
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AT = HERE / "causal" / "stage86" / "at"
START = 60_000
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k_plus40k_a100"
STARTS = {42: HERE / "causal" / "stage81" / "at" / f"{P}_at{START}",  # stage 81's scored 60,000-step pseudo-runs
          4: HERE / "causal" / "stage81" / "at" / f"{P}_s4_at{START}"}
STABLE_TRADE = 0.10  # seed 4's last value stays under this in both views at every checkpoint
CURVES_AT = (60_000, 90_000, 120_000)


def seed_of(run_name):
    match = re.search(r"_s(\d+)$", run_name)
    return int(match.group(1)) if match else 42


def points(runs, at=AT, starts=STARTS, start=START):
    """One folder per kept checkpoint (report.json copied, model.pt a link), named RUN_atTOTAL; returns LABEL=PATH
    pairs, each seed's 60,000-step start first (stage 81's pseudo-run, already bundled)."""
    at.mkdir(parents=True, exist_ok=True)
    pairs = []
    for run in map(Path, runs):
        seed = seed_of(run.name)
        pairs.append(f"cont_at{start}_s{seed}={starts[seed]}")
        for checkpoint in sorted(run.glob("model_*.pt")):
            step = int(checkpoint.stem.split("_")[1])
            if step == 0:
                continue
            total = start + step
            folder = at / f"{run.name}_at{total}"
            folder.mkdir(exist_ok=True)
            shutil.copyfile(run / "report.json", folder / "report.json")
            link = folder / "model.pt"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(checkpoint.resolve())
            pairs.append(f"cont_at{total}_s{seed}={folder}")
    return pairs


def rules(series, start=START):
    """series: seed -> [(total, holds, last (T1, FLAIR))] sorted by total. Seed 4: repaired / stable trade /
    unresolved; seed 42: stable / lost / unstable."""
    out = {}
    for seed, rows in series.items():
        after = [r for r in rows if r[0] > start]
        held = [h for _, h, _ in after]
        if seed == 4:
            if any(all(held[i:]) for i in range(len(held))):
                out[seed] = "repaired"
            elif all(max(last) < STABLE_TRADE for _, _, last in after):
                out[seed] = "stable trade"
            else:
                out[seed] = "unresolved"
        else:
            out[seed] = "stable" if held and all(held) else "lost" if after and not held[-1] else "unstable"
    return out


def summary(detect_json):
    import pool_check  # torch and the bank only here

    rows = json.loads(Path(detect_json).read_text())["rows"]
    by = {}
    for label, row in rows.items():
        total, seed = re.fullmatch(r"cont_at(\d+)_s(\d+)", label).groups()
        by.setdefault(int(seed), {})[int(total)] = row
    series = {}
    for seed in sorted(by):
        print(f"seed {seed}: step, holds, last T1/FLAIR, ventricle-burden worst T1/FLAIR, rho1 | T1 own parts | the lost "
              f"direction's main weights (T1)")
        series[seed] = []
        for total, r in sorted(by[seed].items()):
            d = np.load(pool_check.POOL / f"{r['run']}.npz")
            b = np.load(pool_check.CAUSAL / f"{r['run']}.npz")
            names = json.loads(str(b["meta"]))["content_factor_names"]
            raw = b["z_content"].astype(np.float64) if total in CURVES_AT else None
            s = pool_check.score(d["code_test"].astype(np.float64), b["z_own"].astype(np.float64), names, raw)
            t1 = s["T1"]
            top = sorted(t1["weakest_weights"].items(), key=lambda kv: -abs(kv[1]))[:3]
            curves = f"  curves {s['curves']}" if raw is not None else ""
            print(f"  {total:6d}  {'holds' if r['holds'] else 'fails'}  last {r['last'][0]:+.2f}/{r['last'][1]:+.2f}  "
                  f"worst {r['worst'][0]:+.2f}/{r['worst'][1]:+.2f}  rho1 {r['rho1']:.2f} | "
                  + " ".join(f"{n[:4]} {t1['own_r2'][n]:.2f}" for n in names) + " | "
                  + " ".join(f"{n[:4]} {v:+.2f}" for n, v in top) + curves, flush=True)
            series[seed].append((total, bool(r["holds"]), tuple(r["last"])))
        print()
    verdicts = rules(series)
    complete = all(len([t for t, _, _ in series.get(s, []) if t > START]) == 6 for s in (42, 4))
    for seed, v in sorted(verdicts.items()):
        print(f"seed {seed}: {v}")
    reading = {"repaired": "both stage 64 failures were slow learning", "stable trade":
               "at least one matched-width failure is not slow learning"}.get(verdicts.get(4), "unresolved")
    print(f"STAGE 86: {reading}" + ("" if complete else " (incomplete: fewer than 6 checkpoints per seed)"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    pts = sub.add_parser("points")
    pts.add_argument("runs", nargs="+", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("detect_json", type=Path)
    a = p.parse_args()
    if a.what == "points":
        print(" ".join(points(a.runs)))
    else:
        summary(a.detect_json)


if __name__ == "__main__":
    main()
