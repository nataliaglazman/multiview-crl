#!/usr/bin/env python
"""Did content's amplitude growth SUPPRESS style, or just outpace it?

Reads the ``Norm/*_gamma_max`` / ``eff_dim`` / ``gamma_spread`` series that
``main_multimodal`` writes for SplitGroupNorm and compares them across runs.

The question this settles: between a recon-only and a contrastive run on the same
config, content per-channel RMS went 1.9 -> 29.5 while style went 1.2 -> 0.30. Style
did not merely fail to grow, it *shrank*. Two explanations predict different traces:

  * **Competition.** With ``--norm-type layer`` all channels at a voxel share one
    normalization denominator, so growth in content divides style down. Then the two
    gamma_max curves MIRROR: style falls as content rises, correlation strongly
    negative.
  * **Indifference.** Content grew and style was simply never trained. Then style is
    flat regardless of what content does, correlation near zero.

A strongly negative correlation is evidence for ``--split-encoder-norm``, which gives
each block its own denominator inside the encoder.

Note these tags go to TensorBoard only — the ``wandb.log`` call alongside them sends
just ``eff_dim`` and ``gamma_spread``, so ``gamma_max`` is absent from W&B.

Usage:
  python -m eval.norm_gamma_trace --run-dirs results/synthetic/<A> results/synthetic/<B> \
      --names contrastive baseline
  python -m eval.norm_gamma_trace --run-dirs results/synthetic/<A> --csv gamma_trace.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

METRICS = ("gamma_max", "eff_dim", "gamma_spread")


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
        if not tag.startswith("Norm/"):
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
    """Encoder levels that have a content gamma series, in order."""
    lvls = set()
    for t in tags:
        if t.startswith("Norm/content_gamma_max_L"):
            lvls.add(t.rsplit("_L", 1)[1])
    return sorted(lvls)


def report(name, run_dir, rows):
    tags = load_scalars(run_dir)
    if not tags:
        logger.warning("%s: no Norm/* scalars under %s", name, run_dir)
        return
    lvls = levels_in(tags)
    if not lvls:
        logger.warning(
            "%s: found Norm/* tags but no content_gamma_max_L* — this run predates the "
            "gamma logging block in main_multimodal. Available: %s",
            name,
            sorted(tags),
        )
        return

    print(f"\n=== {name}  ({run_dir}) ===")
    for lvl in lvls:
        print(f"\n  level {lvl}")
        for metric in METRICS:
            for block in ("content", "style"):
                tag = f"Norm/{block}_{metric}_L{lvl}"
                if tag not in tags:
                    continue
                steps, vals = tags[tag]
                growth = vals[-1] / vals[0] if abs(vals[0]) > 1e-12 else float("nan")
                print(
                    f"    {block:<8}{metric:<14}"
                    f"first={vals[0]:12.4f}  last={vals[-1]:12.4f}  "
                    f"x{growth:9.2f}   min={vals.min():10.4f} max={vals.max():10.4f}  "
                    f"n={len(vals)} steps {steps[0]}..{steps[-1]}"
                )
                rows.append(
                    {
                        "run": name,
                        "level": lvl,
                        "block": block,
                        "metric": metric,
                        "first": vals[0],
                        "last": vals[-1],
                        "min": vals.min(),
                        "max": vals.max(),
                        "n": len(vals),
                    }
                )

        # The mirror test.
        c_tag, s_tag = f"Norm/content_gamma_max_L{lvl}", f"Norm/style_gamma_max_L{lvl}"
        if c_tag in tags and s_tag in tags:
            cs, cv = tags[c_tag]
            ss, sv = tags[s_tag]
            common = np.intersect1d(cs, ss)
            if len(common) >= 3:
                c = cv[np.searchsorted(cs, common)]
                s = sv[np.searchsorted(ss, common)]
                pear, spear = _pearson(c, s), _spearman(c, s)
                verdict = (
                    "MIRROR — style falls as content rises; consistent with a shared "
                    "normalization denominator (try --split-encoder-norm)"
                    if spear < -0.5
                    else (
                        "TRACKS TOGETHER — both move the same way; not a competition signature"
                        if spear > 0.5
                        else "FLAT/INDEPENDENT — style is indifferent to content's growth"
                    )
                )
                print(
                    f"\n    mirror test over {len(common)} shared steps: " f"pearson={pear:+.3f} spearman={spear:+.3f}"
                )
                print(f"    -> {verdict}")
            else:
                print(f"\n    mirror test: only {len(common)} shared steps, need >= 3")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Run directories (each containing tensorboard/).")
    ap.add_argument("--names", nargs="+", default=None, help="Labels, one per run dir. Defaults to the basenames.")
    ap.add_argument("--csv", default=None, help="Optional path to write the summary table.")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in cli.run_dirs]
    if len(names) != len(cli.run_dirs):
        raise SystemExit(f"--names has {len(names)} entries but --run-dirs has {len(cli.run_dirs)}")

    rows = []
    for name, run_dir in zip(names, cli.run_dirs):
        report(name, run_dir, rows)

    if cli.csv and rows:
        with open(cli.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary written to {cli.csv}")


if __name__ == "__main__":
    main()
