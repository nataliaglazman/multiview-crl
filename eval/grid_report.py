#!/usr/bin/env python
"""One table for a grid of runs: did rank recover, and what did it cost?

Built for ``scripts/launch_rank_grid.sh``, but it works on any set of run directories.
Reads TensorBoard only — no checkpoints, no GPU, no forward pass — so it can be run against
a grid that is still in flight.

Four metrics, chosen so no single one can be gamed by the others:

    content_rank    the collapse itself. Reported as last AND delta-from-first, because the
                    absolute number means little (an UNTRAINED encoder of this architecture
                    already sits near 6 at GAP) while the direction is the whole question.
    mcc_cc_gap      null-subtracted block-MCC, content block -> content factors. The one
                    logged number that rises only when content genuinely carries MORE factor
                    information: unlike content_purity and separation it cannot be improved
                    by emptying the block, and unlike rank it cannot be improved by pumping
                    nuisance directions. Scored at STATS pooling, so it is blind to lesion_*.
    mcc_pool_patch  the same at PATCH pooling — the only one that can see lesion_x/y/z, and
                    therefore what lowering bt_patch_weight is actually spending.
    info_all        all-channels capacity. Falls when information leaves the model entirely,
                    which is the failure the other three can miss; used here as a GATE rather
                    than ranked, matching --selection-info-tolerance in the training loop.

A cell whose info_all has fallen below where it started is marked [!] — its other numbers
describe a shrinking representation and should not be read as an improvement. That failure
is real and measured on this project: overall_score climbed 0.49 -> 0.53 between two
checkpoints whose effective rank fell 37.9 -> 28.3.

Usage:
    python -m eval.grid_report results/synthetic/rank-grid-*
    python -m eval.grid_report results/synthetic/rank-grid-* --csv grid.csv
    python -m eval.grid_report results/synthetic/rank-grid-* --level 0
"""

from __future__ import annotations

import argparse
import csv
import os

from eval.loss_breakdown import _load_scalars

# (column label, tb tag template, higher_is_better)
_METRICS = [
    ("content_rank", "selection/content_rank", True),
    ("mcc_cc_gap", "selection/mcc_cc_gap", True),
    ("mcc_pool_patch", "selection/mcc_by_pool/patch", True),
    ("info_all", "selection/info_all", True),
]

# Context columns: not scored, but they say WHY a cell moved.
_CONTEXT = [
    ("gap_off_diag", "Contrastive/gap_off_diag_loss_L{level}"),
    ("gap_on_diag", "Contrastive/gap_on_diag_loss_L{level}"),
    ("dead_frac", "Contrastive/gap_dead_frac_L{level}"),
    ("recon_l1", "Recon/Loss-MAE-Reconstruction"),
]


def _first_last(series, tag):
    """(first, last) value of ``tag``, or (None, None) if it was never logged.

    First rather than a fixed step, because the selection metrics only start once
    ``dci_every`` fires; on a 12k-step run that is step ~1, which is the untrained floor and
    exactly the right baseline for "did training move this".
    """
    if tag not in series:
        return None, None
    steps, vals = series[tag]
    if len(steps) == 0:
        return None, None
    return float(vals[0]), float(vals[-1])


def collect(run_dir, level):
    series = _load_scalars(run_dir)
    row = {"run": os.path.basename(os.path.normpath(run_dir))}
    for label, tag, _hib in _METRICS:
        row[label] = _first_last(series, tag)
    for label, tmpl in _CONTEXT:
        row[label] = _first_last(series, tmpl.format(level=level))
    return row


def _fmt(pair, width=13):
    if pair is None or pair[1] is None:
        return f"{'—':>{width}}"
    first, last = pair
    return f"{last:>7.3f}{last - first:>+6.2f}"[:width].rjust(width)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="Run directories (TensorBoard events are searched recursively)")
    ap.add_argument("--level", type=int, default=0, help="Encoder level for the Contrastive/* context columns")
    ap.add_argument("--csv", default=None, help="Also write the table here")
    cli = ap.parse_args()

    rows = []
    for d in cli.run_dirs:
        if not os.path.isdir(d):
            print(f"  skipping (not a directory): {d}")
            continue
        try:
            rows.append(collect(d, cli.level))
        except SystemExit as e:  # _load_scalars raises this when there are no tfevents yet
            print(f"  skipping {os.path.basename(d)}: {e}")
    if not rows:
        raise SystemExit("No run directories with TensorBoard data.")

    name_w = max(len(r["run"]) for r in rows) + 2
    cols = [c[0] for c in _METRICS] + [c[0] for c in _CONTEXT]

    print(f"\n{'=' * (name_w + 13 * len(cols) + 4)}")
    print("  each cell is  LAST  and  Δ from first logged value")
    print(f"{'=' * (name_w + 13 * len(cols) + 4)}")
    print(f"  {'run':<{name_w}}" + "".join(f"{c:>13}" for c in cols))
    print("  " + "-" * (name_w + 13 * len(cols)))
    for r in sorted(rows, key=lambda x: x["run"]):
        # The gate, not a ranking term: a cell that lost capacity is describing a smaller
        # representation, so its rank and MCC columns are not comparable with the others.
        info = r.get("info_all")
        gated = info is not None and info[1] is not None and info[1] < info[0]
        print(f"  {r['run']:<{name_w}}" + "".join(_fmt(r[c]) for c in cols) + ("   [!] info_all fell" if gated else ""))
    print()
    print("  content_rank    Δ > 0 = the collapse stopped. The absolute value is not the target:")
    print("                  an untrained encoder of this architecture already reads ~6 at GAP.")
    print("  mcc_cc_gap      the completeness term — cannot be improved by emptying the block.")
    print("  mcc_pool_patch  what bt_patch_weight is buying; the only column that sees lesion_*.")
    print("  [!]             info_all below its own start: read nothing else in that row as a win.")
    print("  gap_off_diag    sqrt() of it is the RMS cross-channel correlation (bt_normalize_terms).")

    if cli.csv:
        with open(cli.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run"] + [f"{c}_{s}" for c in cols for s in ("first", "last")])
            for r in sorted(rows, key=lambda x: x["run"]):
                out = [r["run"]]
                for c in cols:
                    pair = r.get(c) or (None, None)
                    out.extend(pair)
                w.writerow(out)
        print(f"\n  wrote {cli.csv}")


if __name__ == "__main__":
    main()
