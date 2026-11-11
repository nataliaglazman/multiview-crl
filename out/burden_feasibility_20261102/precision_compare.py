"""precision_audit.py's volume-reading error for the same runs under several conditions, side by side.

    python out/burden_feasibility_20261102/precision_compare.py LABEL=OUT_JSON [LABEL=OUT_JSON ...]

Each OUT_JSON is one precision_audit.py condition: raw test scans (the default), --jitter-draws K or --split train. Per
run: the reading error e in voxels at each stage and view, e in the largest third of lesions for the map and the code,
and how well the two views' readings agree. '-' where a file lacks the run or the stage, or nothing is read.
"""

import argparse
import json
from pathlib import Path

STAGES = ("image", "map", "gap", "pooled", "code")
VIEWS = ("t1", "flair")


def voxels(value, width):
    if value is None:
        return f"{'-':>{width}s}"
    return f"{'>99':>{width}s}" if value > 99 else f"{value:{width}.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("columns", nargs="+", metavar="LABEL=OUT_JSON")
    a = p.parse_args()
    columns = []
    for item in a.columns:
        label, sep, path = item.partition("=")
        if not sep:
            p.error(f"expected LABEL=OUT_JSON, got {item!r}")
        columns.append((label, json.loads(Path(path).read_text())))
    width = max(8, *(len(label) + 2 for label, _ in columns))
    print(
        "e: volume-reading error in voxels, own R2 = SD^2 / (SD^2 + e^2); big: e in the largest third of lesions;"
        " agree: correlation of the T1 and FLAIR readings"
    )
    for name in sorted({name for _, data in columns for name in data}):
        rows = [data.get(name) for _, data in columns]

        def line(title, cells):
            print(f"  {title:16s}" + "".join(cells))

        def cell(row, stage, view, key):
            found = row["trained"][view].get(stage) if row else None
            if found is None:
                return None
            return found["error"] if key is None else found[key][-1]

        print(f"\n{name}")
        line("", [f"{label:>{width}s}" for label, _ in columns])
        line("step", [f"{(str(row['step']) if row else '-'):>{width}s}" for row in rows])
        line("subjects", [f"{(str(row['rows']) if row else '-'):>{width}s}" for row in rows])
        line("own-part SD", [voxels(row["own_sd"] if row else None, width) for row in rows])
        for stage in STAGES:
            for view in VIEWS:
                line(f"{stage} {view}", [voxels(cell(row, stage, view, None), width) for row in rows])
        for stage in ("map", "code"):
            for view in VIEWS:
                line(f"{stage} {view} big", [voxels(cell(row, stage, view, "error_by_size"), width) for row in rows])
        for stage in STAGES:
            agree = [row.get("cross_view", {}).get(stage) if row else None for row in rows]
            line(f"agree {stage}", [f"{g['reading_r']:+{width}.2f}" if g else f"{'-':>{width}s}" for g in agree])


if __name__ == "__main__":
    main()
