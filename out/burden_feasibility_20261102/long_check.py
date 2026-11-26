"""Stage 81 (rules written 26 Nov before any of its results; the user asked for it): does longer training repair a
failing 8-unit code?

Why. The user asked whether the 8-unit failures need longer training. Stage 65's continuation (10,000 more steps)
changed no verdict, but seed 42's own-parts spectrum's last value crept from .01 to .23 / .17 (item 52); nothing has
run past 30,000 steps, and no checkpoints were kept, so the trend is unknown. Item 65's refitted heads peaked within
2,000-4,000 steps and then drifted down, which points the other way.

What. Stage 64's two failing from-scratch codes (seeds 42 and 4; the frozen recipe, code batches every 500, trained
to 20,000 steps on the PC: `..._fixedpos_abl_codebatch_predict1_c8_20k` and `..._s4`) continued for 40,000 steps on
Run:ai A100s with the same flags and seed, as stage 65's continuation was (`warm.py --init-run`: the network and
TailCode are loaded; the optimizer, the momentum target, its whitening and the code batches start afresh, and the
step counter restarts). An evaluation and a kept checkpoint every 5,000 steps: 25,000 to 60,000 in all. Runs
`..._c8_20k_plus40k_a100` and `..._c8_20k_plus40k_a100_s4`; each checkpoint is scored as stage 76 (causal_bundle.py,
pool_check.py detect: holds, the spectrum's last value, the ventricle-burden worst direction, rho1), beside the
20,000-step start.

Rules (fixed before any result):
  A seed is repaired if its code holds (the last value and the worst direction each at least .40 in both views) at
  some checkpoint and at every later one up to 60,000.
  Works (longer training is a lever) if at least one of the two seeds is repaired. Fails if neither seed holds at any
  checkpoint. Otherwise (a hold that does not last to 60,000): unstable, described.
  Described only: each seed's trajectory; rho1; the 30,000-step checkpoint against stage 65's PC continuation (s42
  last .23 / .17, s4 .01 / .02); a different GPU changes the trajectory as a new seed would, so they need not match.
Expectation, written before: fails. Seed 42's last value creeps up and stays under .40; seed 4 stays near 0.

    python long_check.py points RUN...         # checkpoint pseudo-runs in causal/stage81/at/, one per kept checkpoint
    python long_check.py summary DETECT.json   # trajectories and the rules
"""

import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
AT = HERE / "causal" / "stage81" / "at"
START = 20_000
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k"
SOURCES = {42: P, 4: f"{P}_s4"}


def seed_of(run_name):
    match = re.search(r"_s(\d+)$", run_name)
    return int(match.group(1)) if match else 42


def points(runs):
    """One folder per kept checkpoint (report.json copied, model.pt a link), named RUN_atTOTAL; prints LABEL=PATH."""
    AT.mkdir(parents=True, exist_ok=True)
    pairs = []
    for run in map(Path, runs):
        seed = seed_of(run.name)
        pairs.append(f"long_at{START}_s{seed}={HERE / 'runs' / SOURCES[seed]}")
        for checkpoint in sorted(run.glob("model_*.pt")):
            step = int(checkpoint.stem.split("_")[1])
            if step == 0:
                continue  # the start itself, scored from the source run
            total = START + step
            folder = AT / f"{run.name}_at{total}"
            folder.mkdir(exist_ok=True)
            shutil.copyfile(run / "report.json", folder / "report.json")
            link = folder / "model.pt"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(checkpoint.resolve())
            pairs.append(f"long_at{total}_s{seed}={folder}")
    print(" ".join(pairs))


def summary(detect_json):
    rows = json.loads(Path(detect_json).read_text())["rows"]
    by = {}
    for label, row in rows.items():
        total, seed = re.fullmatch(r"long_at(\d+)_s(\d+)", label).groups()
        by.setdefault(int(seed), {})[int(total)] = row
    verdicts = {}
    for seed in sorted(by):
        print(f"seed {seed}: step, holds, last T1/FLAIR, ventricle-burden worst T1/FLAIR, rho1")
        series = sorted(by[seed].items())
        for total, r in series:
            print(f"  {total:6d}  {'holds' if r['holds'] else 'fails'}  last {r['last'][0]:+.2f}/{r['last'][1]:+.2f}  "
                  f"worst {r['worst'][0]:+.2f}/{r['worst'][1]:+.2f}  rho1 {r['rho1']:.2f}")
        held = [r["holds"] for total, r in series if total > START]
        ever = any(held)
        lasting = any(all(held[i:]) for i in range(len(held))) if held else False
        verdicts[seed] = "repaired" if lasting else "unstable" if ever else "never holds"
        print(f"  seed {seed}: {verdicts[seed]} ({len(held)} checkpoints after {START})\n")
    complete = all(len([t for t in by.get(s, {}) if t > START]) == 8 for s in SOURCES)
    if any(v == "repaired" for v in verdicts.values()):
        verdict = "works"
    elif all(v == "never holds" for v in verdicts.values()):
        verdict = "fails"
    else:
        verdict = "unstable"
    print(f"STAGE 81: {verdict}" + ("" if complete else " (incomplete: fewer than 8 checkpoints per seed)"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("points")
    a.add_argument("runs", nargs="+", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("detect_json", type=Path)
    args = p.parse_args()
    points(args.runs) if args.command == "points" else summary(args.detect_json)


if __name__ == "__main__":
    main()
