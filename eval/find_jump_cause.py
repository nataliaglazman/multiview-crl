"""Rank EVERY logged scalar by how specifically it changes at a jump — hypothesis-free.

Chasing a step change by nominating one metric at a time is slow and biased toward whatever
you already suspect. This does the opposite: locate the jump on a target curve, then score
all ~100 logged scalars by whether THEY also change at that exact step, and rank them.

The scoring matters. A raw before/after delta ranks every monotone curve at the top, because
a steadily-rising scalar has a large delta everywhere. So each scalar's delta at the jump is
compared against its OWN delta distribution at every other candidate step:

    specificity = |delta at jump| / median(|delta| at all other steps)

A scalar that drifts smoothly scores ~1 however large its drift. A scalar that is flat
everywhere and steps once, at the jump, scores high. That is the signature of a trigger.

Usage:
    python -m eval.find_jump_cause <run_dir_or_tb_logdir>
    python -m eval.find_jump_cause <logdir> --target Loss/Recon
    python -m eval.find_jump_cause <logdir> --step 31000        # if you know the jump step
    python -m eval.find_jump_cause <logdir> --window 4000 --top 30
"""

import argparse
import glob
import os

import numpy as np


def _load_scalars(logdir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise SystemExit(
            "Needs tensorboard (`pip install tensorboard`). It is normally present wherever "
            f"training ran, since SummaryWriter requires it. ({e})"
        )
    files = [p for p in glob.glob(os.path.join(logdir, "**", "*"), recursive=True) if "tfevents" in os.path.basename(p)]
    if not files:
        raise SystemExit(f"No tfevents files under {logdir}")
    series = {}
    for f in sorted(files):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            ev = acc.Scalars(tag)
            s = np.array([e.step for e in ev], dtype=np.float64)
            v = np.array([e.value for e in ev], dtype=np.float64)
            if tag in series:  # concatenate across event files (resumes)
                s = np.concatenate([series[tag][0], s])
                v = np.concatenate([series[tag][1], v])
            o = np.argsort(s)
            series[tag] = (s[o], v[o])
    return series


def _window_delta(steps, vals, at, w):
    """mean(after) - mean(before) around `at`, or None if either side is too thin."""
    before = vals[(steps >= at - w) & (steps < at)]
    after = vals[(steps > at) & (steps <= at + w)]
    if len(before) < 2 or len(after) < 2:
        return None
    return float(np.mean(after) - np.mean(before))


def _specificity(steps, vals, at, w, n_probe=40):
    """|delta at `at`| relative to the scalar's own typical |delta| elsewhere."""
    d = _window_delta(steps, vals, at, w)
    if d is None:
        return None
    lo, hi = steps.min() + w, steps.max() - w
    if hi <= lo:
        return None
    others = []
    for p in np.linspace(lo, hi, n_probe):
        if abs(p - at) < w:  # don't let the jump itself pollute its own reference
            continue
        o = _window_delta(steps, vals, p, w)
        if o is not None:
            others.append(abs(o))
    if len(others) < 5:
        return None
    ref = float(np.median(others))
    return abs(d), d, (abs(d) / ref if ref > 1e-12 else float("inf")), ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir", help="Run directory or TensorBoard log directory (searched recursively)")
    ap.add_argument("--target", default="Loss/Recon", help="Scalar whose jump to locate (default: Loss/Recon)")
    ap.add_argument("--step", type=float, default=None, help="Jump step, if you already know it from the curves")
    ap.add_argument("--window", type=float, default=None, help="Half-width in steps (default: 5%% of the run)")
    ap.add_argument("--top", type=int, default=25, help="How many scalars to print")
    ap.add_argument("--min-points", type=int, default=8, help="Skip scalars with fewer points than this")
    args = ap.parse_args()

    series = _load_scalars(args.logdir)
    print(f"Loaded {len(series)} scalars from {args.logdir}")

    if args.target not in series:
        cand = [t for t in series if "recon" in t.lower()]
        raise SystemExit(f"--target {args.target!r} not found. Recon-ish tags present: {cand[:10]}")
    ts, tv = series[args.target]
    span = ts.max() - ts.min()
    w = args.window if args.window else max(1.0, 0.05 * span)

    # Locate the jump: the step at which the target's before/after delta is most extreme.
    if args.step is not None:
        jump = args.step
    else:
        best, jump = -np.inf, None
        for p in np.linspace(ts.min() + w, ts.max() - w, 200):
            d = _window_delta(ts, tv, p, w)
            if d is not None and d > best:  # a JUMP UP in the target
                best, jump = d, p
        if jump is None:
            raise SystemExit("Could not locate a jump; pass --step explicitly.")
    print(f"Jump located at step ~{jump:.0f} on {args.target} (window +-{w:.0f} steps)")
    d0 = _window_delta(ts, tv, jump, w)
    print(f"  {args.target}: {d0:+.6g} across the jump\n")

    rows = []
    for tag, (s, v) in series.items():
        if len(s) < args.min_points:
            continue
        r = _specificity(s, v, jump, w)
        if r is None:
            continue
        absd, d, spec, ref = r
        before = float(np.mean(v[(s >= jump - w) & (s < jump)]))
        rows.append((spec, tag, d, before, ref))

    rows.sort(key=lambda r: -r[0])
    print("Scalars ranked by SPECIFICITY of their change at the jump")
    print("(specificity ~1 = changes no more here than anywhere else -> ignore;")
    print(" high = flat elsewhere, steps HERE -> candidate trigger)\n")
    print(f"  {'spec':>7}  {'delta at jump':>14}  {'pre-jump value':>14}  tag")
    for spec, tag, d, before, ref in rows[: args.top]:
        print(f"  {spec:>7.2f}  {d:>+14.6g}  {before:>14.6g}  {tag}")
    print(f"\n  ({len(rows)} scalars scored; showing top {min(args.top, len(rows))})")
    print("  Read the top of this list as CANDIDATES, not causes — a scalar downstream of the")
    print("  failure steps at the jump too. But anything that does NOT appear here is ruled out")
    print("  as a trigger, which is most of the list.")


if __name__ == "__main__":
    main()
