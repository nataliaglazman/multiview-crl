"""Per-cell count of the codes that hold all seven own parts, and of those that match the reference code (stage 64).

    python holds.py OUT LABEL=RUN ... [--ref LABEL]

Labels are c{units}_{arm}_s{seed}; an arm e{N} is neighbourhoods rebuilt every N steps (others name a start, e.g.
red, pca, cont). Reads OUT's own_directions.json (by label), precision_audit.json and
code_audit.json (by run), tabular.json (by label) and discovery_decoded_view{1,2}.json (by run). Rules fixed in
HANDOFF.md before the results:
  holds    the own-parts spectrum's last value and the ventricle-burden plane's worst direction are each >= .40 in
           both views
  matches  every own part >= the reference's less .10, volume error <= the reference's plus 1 voxel, |ATE bias| <=
           the reference's plus .03, decoded-graph F1 >= the reference's less .05, all in both views (one-sided:
           doing better than the reference never fails)
"""

import argparse
import json
import re
from math import comb
from collections import defaultdict
from pathlib import Path

VIEWS = ("T1", "FLAIR")
HOLD = 0.40


def load(path):
    return json.loads(Path(path).read_text())


def pair_worst(view):
    for row in view["pairs"]:
        if sorted(row["pair"]) == ["burden", "ventricle"]:
            return row["worst"]
    raise KeyError("no ventricle-burden pair")


def measures(out, label, run):
    own = load(out / "own_directions.json")[label]
    precision = load(out / "precision_audit.json")[run]
    tabular = load(out / "tabular.json")["summary"]
    decoded = [load(out / f"discovery_decoded_view{v}.json")["graph_stability"][run] for v in (1, 2)]
    m = {
        "last": [own[v]["spectrum"][-1] for v in VIEWS],
        "worst": [pair_worst(own[v]) for v in VIEWS],
        "own": [own[v]["own_r2"] for v in VIEWS],
        "volume": [precision["trained"][v.lower()]["code"]["error"] for v in VIEWS],
        "bias": [abs(tabular[f"{label} / {v}"]["bias"]) for v in VIEWS],
        "graph": [d["f1_mean"] for d in decoded],
        "top1": load(out / "code_audit.json")[run]["top1"]["all1000"],
    }
    m["holds"] = all(x >= HOLD for x in m["last"] + m["worst"])
    return m


def fisher(a, n1, b, n2):
    """Two-sided Fisher exact p for a of n1 against b of n2."""
    total, k = n1 + n2, a + b
    def prob(x):
        return comb(n1, x) * comb(n2, k - x) / comb(total, k)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, k - n2), min(k, n1) + 1) if prob(x) <= observed * (1 + 1e-9)))


def arm_key(arm):
    """Rebuild intervals first, longest first, then named arms alphabetically."""
    match = re.fullmatch(r"e(\d+)", arm)
    return (0, -int(match[1]), "") if match else (1, 0, arm)


def arm_name(arm):
    match = re.fullmatch(r"e(\d+)", arm)
    return f"every {int(match[1]):3}" if match else arm


def misses(m, ref):
    out = []
    for i, view in enumerate(VIEWS):
        for factor, r2 in ref["own"][i].items():
            if m["own"][i][factor] < r2 - 0.10:
                out.append(f"{factor} {view} {m['own'][i][factor]:.2f}")
        if m["volume"][i] is None:  # the audit gives no error when the code reads no volume at all
            out.append(f"volume {view} none")
        elif m["volume"][i] > ref["volume"][i] + 1:
            out.append(f"volume {view} {m['volume'][i]:.1f}")
        if m["bias"][i] > ref["bias"][i] + 0.03:
            out.append(f"ATE bias {view} {m['bias'][i]:.3f}")
        if m["graph"][i] < ref["graph"][i] - 0.05:
            out.append(f"graph {view} {m['graph'][i]:.3f}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("codes", nargs="+", help="LABEL=RUN, label c{units}_{arm}_s{seed}")
    parser.add_argument("--ref", default="c10_e500_s42")
    parser.add_argument("--compare", nargs="+", default=[], help="CELL:CELL pairs, cells c{units}_{arm}, paired by seed")
    args = parser.parse_args()
    runs = dict(code.split("=", 1) for code in args.codes)
    rows = {label: measures(args.out, label, run) for label, run in runs.items()}
    ref = rows[args.ref]
    cell = {}
    for label in rows:
        match = re.fullmatch(r"c(\d+)_([a-z]+\d*)_s(\d+)", label)
        if not match:
            raise SystemExit(f"label {label} is not c{{units}}_{{arm}}_s{{seed}}")
        cell[label] = (int(match[1]), match[2], int(match[3]))
        rows[label]["misses"] = misses(rows[label], ref)

    print(f"holds: last spectrum value and ventricle-burden worst each >= {HOLD:.2f} in both views")
    print(f"matches: against {args.ref}, own parts -.10, volume error +1 voxel, |ATE bias| +.03, decoded graph -.05")
    print()
    print(f"{'code':15} {'last T1/FL':>11} {'worst T1/FL':>12} {'volume':>11} {'|bias|':>11} {'graph':>11}"
          f" {'top-1':>6}  holds  matches")
    order = sorted(rows, key=lambda label: (cell[label][0], arm_key(cell[label][1]), cell[label][2]))
    for label in order:
        m = rows[label]
        volume = ["none" if v is None else f"{v:.1f}" for v in m["volume"]]
        print(f"{label:15} {m['last'][0]:5.2f}/{m['last'][1]:<5.2f} {m['worst'][0]:6.2f}/{m['worst'][1]:<5.2f}"
              f" {volume[0]:>5}/{volume[1]:<5} {m['bias'][0]:5.3f}/{m['bias'][1]:<5.3f}"
              f" {m['graph'][0]:5.3f}/{m['graph'][1]:<5.3f} {m['top1']:6.3f}  {'yes' if m['holds'] else 'no ':5}"
              f"  {'yes' if not m['misses'] else 'no: ' + ', '.join(m['misses'])}")

    def count(labels):
        labels = list(labels)
        held = sum(rows[x]["holds"] for x in labels)
        matched = sum(not rows[x]["misses"] for x in labels)
        return f"holds {held}/{len(labels)}, matches {matched}/{len(labels)}"

    narrow = [x for x in rows if cell[x][0] < 10]
    groups = defaultdict(list)
    for label in rows:
        groups[cell[label][:2]].append(label)
    print("\nper cell (units, arm):")
    for key in sorted(groups, key=lambda k: (k[0], arm_key(k[1]))):
        seeds = ", ".join(str(cell[x][2]) for x in sorted(groups[key], key=lambda x: cell[x][2]))
        print(f"  {key[0]} units, {arm_name(key[1])}: {count(groups[key])}  (seeds {seeds})")
    print("pooled:")
    for units in sorted({cell[x][0] for x in narrow}):
        print(f"  {units} units, every arm: {count(x for x in narrow if cell[x][0] == units)}")
    for arm in sorted({cell[x][1] for x in narrow}, key=arm_key):
        print(f"  {arm_name(arm)}, every width: {count(x for x in narrow if cell[x][1] == arm)}")

    def compare(name, first, second):
        for what, ok in (("holds", lambda x: rows[x]["holds"]), ("matches", lambda x: not rows[x]["misses"])):
            a, b = sum(map(ok, first)), sum(map(ok, second))
            if first and second:
                diff = a / len(first) - b / len(second)
                print(f"  {name}, {what}: {a}/{len(first)} against {b}/{len(second)}, rate difference {diff:+.2f},"
                      f" Fisher p {fisher(a, len(first), b, len(second)):.2f}")

    widths = sorted({cell[x][0] for x in narrow})
    if len(widths) == 2:
        compare(f"{widths[1]} units against {widths[0]}", [x for x in narrow if cell[x][0] == widths[1]],
                [x for x in narrow if cell[x][0] == widths[0]])
    intervals = sorted({cell[x][1] for x in narrow if re.fullmatch(r"e\d+", cell[x][1])}, key=arm_key, reverse=True)
    if len(intervals) == 2:
        compare(f"{arm_name(intervals[0])} against {arm_name(intervals[1])}",
                [x for x in narrow if cell[x][1] == intervals[0]], [x for x in narrow if cell[x][1] == intervals[1]])
    for pair in args.compare:
        first, second = (re.fullmatch(r"c(\d+)_([a-z]+\d*)", side) for side in pair.split(":"))
        a_rows = [x for x in rows if cell[x][:2] == (int(first[1]), first[2])]
        b_rows = [x for x in rows if cell[x][:2] == (int(second[1]), second[2])]
        compare(f"{first[0]} against {second[0]}", a_rows, b_rows)
        seeds = sorted({cell[x][2] for x in a_rows} & {cell[x][2] for x in b_rows})
        by_seed = {cell[x][2]: x for x in a_rows}, {cell[x][2]: x for x in b_rows}
        only = [sum(rows[by_seed[i][s]]["holds"] and not rows[by_seed[1 - i][s]]["holds"] for s in seeds) for i in (0, 1)]
        print(f"    paired by seed ({len(seeds)} seeds): {first[0]} alone holds in {only[0]}, {second[0]} alone in {only[1]}")

    print("\ncross-view top-1 over all 1,000 subjects, holders against the rest, per width:")
    for units in sorted({cell[x][0] for x in narrow}):
        labels = [x for x in narrow if cell[x][0] == units]
        held = [rows[x]["top1"] for x in labels if rows[x]["holds"]]
        rest = [rows[x]["top1"] for x in labels if not rows[x]["holds"]]
        best = max(labels, key=lambda x: rows[x]["top1"])
        if held and rest:
            overlap = sum(r >= min(held) for r in rest) + sum(h <= max(rest) for h in held)
            verdict = (f"holders {min(held):.3f}-{max(held):.3f}, the rest {min(rest):.3f}-{max(rest):.3f}: "
                       + ("separates" if min(held) > max(rest) else f"overlaps ({overlap} codes on the wrong side)"))
        else:
            verdict = "cannot be read: " + ("no code holds" if not held else "every code holds")
        print(f"  {units} units: {verdict}")
        print(f"    highest top-1 is {best} ({rows[best]['top1']:.3f}): holds {'yes' if rows[best]['holds'] else 'no'},"
              f" matches {'yes' if not rows[best]['misses'] else 'no'}")


if __name__ == "__main__":
    main()
