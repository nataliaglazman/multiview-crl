#!/usr/bin/env python
"""Did content's amplitude growth SUPPRESS style, or just outpace it?

Compares, across runs, the ``Encoder/*_rms`` series (pre-SplitGroupNorm block
amplitude) and the ``Norm/*`` gamma series (``gamma_max`` / ``eff_dim`` /
``gamma_spread``) that ``main_multimodal`` writes.

The question: between a recon-only and a contrastive run on the same config, content
per-channel RMS went 1.88 -> 29.5 while style went 1.23 -> 0.30. Style did not merely
fail to grow, it *shrank*. Two explanations predict different traces:

  * **Competition.** With ``--norm-type layer`` every channel at a voxel shares one
    normalization denominator, so growth in content divides style down. The two curves
    MIRROR: style falls as content rises, correlation strongly negative.
  * **Indifference.** Content grew and style was simply never trained. Style is flat
    regardless of what content does, correlation near zero.

Only ``Encoder/*_rms`` can tell them apart. ``Norm/*_gamma_max`` reads SplitGroupNorm,
which z-scores content and style independently a line later, so the two blocks are
already separated there and no competition can show up — a positive correlation on the
gamma series is not evidence against the mechanism, it is the wrong measurement point.
The mirror test therefore runs on both and labels which is which.

A strongly negative correlation on ``Encoder/*_rms`` is the evidence that would justify
``--split-encoder-norm``, which gives each block its own denominator inside the encoder.

Runs of different length cannot be compared at their endpoints; pass ``--at-step`` to
read every series at a common step (the script suggests one when lengths differ).

These tags are TensorBoard-only for gamma_max — the ``wandb.log`` call beside it sends
just ``eff_dim`` and ``gamma_spread`` — so this reads the event files directly.

Usage:
  python -m eval.norm_gamma_trace --run-dirs results/synthetic/<A> results/synthetic/<B> \
      --names contrastive baseline
  python -m eval.norm_gamma_trace --run-dirs <A> <B> --names contrastive baseline \
      --at-step 26000 --csv gamma_trace.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

SERIES = (
    ("Encoder", "rms"),  # pre-SplitGroupNorm block amplitude — where competition would show
    ("Norm", "gamma_max"),  # SplitGroupNorm gamma — downstream, blocks already separated
    ("Norm", "eff_dim"),
    ("Norm", "gamma_spread"),
)
TAG_PREFIXES = ("Norm/", "Encoder/")


def load_scalars(run_dir):
    """Return {tag: (steps, values)} for the Norm/* scalars in one run directory.

    Deduplicates by step keeping the LAST value, because a resumed run replays steps
    into a second event file and the raw series is then non-monotonic.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = os.path.join(run_dir, "tensorboard")
    if not os.path.isdir(log_dir):
        log_dir = run_dir  # allow pointing straight at an event directory
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()

    out = {}
    for tag in acc.Tags().get("scalars", []):
        if not tag.startswith(TAG_PREFIXES):
            continue
        by_step = {}
        for ev in acc.Scalars(tag):
            by_step[ev.step] = ev.value
        steps = np.array(sorted(by_step), dtype=np.int64)
        out[tag] = (steps, np.array([by_step[s] for s in steps], dtype=np.float64))
    return out


def _spearman(a, b):
    """Rank correlation without pulling in scipy."""

    def rank(x):
        r = np.empty(len(x), dtype=np.float64)
        r[np.argsort(x, kind="stable")] = np.arange(len(x))
        return r

    return _pearson(rank(a), rank(b))


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def levels_in(tags):
    """Encoder levels that have a content or encoder-RMS series, in order."""
    lvls = set()
    for t in tags:
        for pfx in ("Norm/content_gamma_max_L", "Encoder/content_rms_L"):
            if t.startswith(pfx):
                lvls.add(t.rsplit("_L", 1)[1])
    return sorted(lvls)


def value_at(steps, vals, at):
    """Value at the last recorded step <= ``at``. Returns (step, value) or None.

    Runs of different length cannot be compared at their endpoints — a baseline
    stopped at 26k has not had the chance to do what a 174k run did — so every
    cross-run claim needs a common step.
    """
    idx = np.searchsorted(steps, at, side="right") - 1
    if idx < 0:
        return None
    return int(steps[idx]), float(vals[idx])


def report(name, run_dir, tags, rows, at_step=None):
    if not tags:
        logger.warning("%s: no Norm/* or Encoder/* scalars under %s", name, run_dir)
        return
    lvls = levels_in(tags)
    if not lvls:
        logger.warning(
            "%s: no content_gamma_max_L* or content_rms_L* — this run predates the "
            "logging blocks in main_multimodal. Available: %s",
            name,
            sorted(tags),
        )
        return

    print(f"\n=== {name}  ({run_dir}) ===")
    for lvl in lvls:
        print(f"\n  level {lvl}")
        for group, metric in SERIES:
            for block in ("content", "style"):
                tag = f"{group}/{block}_{metric}_L{lvl}"
                if tag not in tags:
                    continue
                steps, vals = tags[tag]
                growth = vals[-1] / vals[0] if abs(vals[0]) > 1e-12 else float("nan")
                row = {
                    "run": name,
                    "level": lvl,
                    "block": block,
                    "metric": f"{group}/{metric}",
                    "first": vals[0],
                    "last": vals[-1],
                    "min": vals.min(),
                    "max": vals.max(),
                    "n": len(vals),
                    "at_step": "",
                    "at_step_value": "",
                }
                line = (
                    f"    {block:<8}{group + '/' + metric:<20}"
                    f"first={vals[0]:12.4f}  last={vals[-1]:12.4f}  "
                    f"x{growth:9.2f}   n={len(vals)} steps {steps[0]}..{steps[-1]}"
                )
                if at_step is not None:
                    got = value_at(steps, vals, at_step)
                    if got is None:
                        line += f"   @{at_step}: no data yet"
                    else:
                        row["at_step"], row["at_step_value"] = got
                        line += f"   @{got[0]}={got[1]:.4f}"
                print(line)
                rows.append(row)

        # The mirror test, at BOTH layers.
        #
        # Encoder/*_rms is the one that can see the hypothesised competition: it is the
        # pre-normalisation amplitude, where a shared per-voxel denominator would make
        # the blocks trade off. Norm/*_gamma_max is downstream of SplitGroupNorm, which
        # already normalises each block separately, so it cannot show competition and a
        # positive correlation there says nothing either way.
        for group, metric, note in (
            ("Encoder", "rms", "pre-SplitGroupNorm — CAN see competition"),
            ("Norm", "gamma_max", "post-SplitGroupNorm — blocks already separated, cannot see it"),
        ):
            c_tag = f"{group}/content_{metric}_L{lvl}"
            s_tag = f"{group}/style_{metric}_L{lvl}"
            if c_tag not in tags or s_tag not in tags:
                continue
            cs, cv = tags[c_tag]
            ss, sv = tags[s_tag]
            common = np.intersect1d(cs, ss)
            if at_step is not None:
                common = common[common <= at_step]
            print(f"\n    mirror test on {group}/*_{metric}  ({note})")
            if len(common) < 3:
                print(f"      only {len(common)} shared steps, need >= 3")
                continue
            c = cv[np.searchsorted(cs, common)]
            s = sv[np.searchsorted(ss, common)]
            pear, spear = _pearson(c, s), _spearman(c, s)
            if group != "Encoder":
                verdict = "uninformative about competition by construction — see the note above"
            elif spear < -0.5:
                verdict = (
                    "MIRROR — style falls as content rises; the shared-denominator "
                    "story holds, --split-encoder-norm is the fix"
                )
            elif spear > 0.5:
                verdict = "TRACKS TOGETHER — both grow; no competition, so --split-encoder-norm " "would not help"
            else:
                verdict = (
                    "FLAT/INDEPENDENT — style is indifferent to content's growth; it is " "untrained, not suppressed"
                )
            print(f"      over {len(common)} shared steps: pearson={pear:+.3f} spearman={spear:+.3f}")
            print(f"      -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Run directories (each containing tensorboard/).")
    ap.add_argument("--names", nargs="+", default=None, help="Labels, one per run dir. Defaults to the basenames.")
    ap.add_argument("--csv", default=None, help="Optional path to write the summary table.")
    ap.add_argument(
        "--at-step",
        type=int,
        default=None,
        help="Also report every series at the last recorded step <= this, and restrict the "
        "mirror test to steps up to it. Use when runs have different lengths — comparing a "
        "26k-step baseline against a 174k-step run at their endpoints measures training "
        "duration, not the thing you changed. Pass 'auto' behaviour by omitting this: the "
        "shortest run's final step is suggested for you.",
    )
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in cli.run_dirs]
    if len(names) != len(cli.run_dirs):
        raise SystemExit(f"--names has {len(names)} entries but --run-dirs has {len(cli.run_dirs)}")

    loaded = {name: load_scalars(run_dir) for name, run_dir in zip(names, cli.run_dirs)}

    # Endpoint comparisons across runs of different length are not comparisons.
    last_steps = {}
    for name, tags in loaded.items():
        ends = [int(steps[-1]) for steps, _ in tags.values() if len(steps)]
        if ends:
            last_steps[name] = max(ends)
    at_step = cli.at_step
    if len(last_steps) > 1 and min(last_steps.values()) * 1.1 < max(last_steps.values()):
        shortest = min(last_steps, key=last_steps.get)
        print("WARNING: runs differ in length — " + ", ".join(f"{n}={s}" for n, s in last_steps.items()))
        if at_step is None:
            print(
                f"         endpoint values are NOT comparable. Re-run with "
                f"--at-step {last_steps[shortest]} (where '{shortest}' ends) for a fair one."
            )

    rows = []
    for name, run_dir in zip(names, cli.run_dirs):
        report(name, run_dir, loaded[name], rows, at_step=at_step)

    if cli.csv and rows:
        with open(cli.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary written to {cli.csv}")


if __name__ == "__main__":
    main()
