"""Table of every arm's ridge R^2 at one step, per readout and view, next to that readout's untrained floor (step 0).

    python out/regional_pilot_20261028/compare.py [STEP]

Reads ARM.jsonl here, plus ARM_post.jsonl (checkpoints re-scored with readouts added later); the xpred pilot's base
arm (global InfoNCE + patch BT at 16^3) is added where it logged the same readout (the global code only).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = (
    "gap",
    "region2",
    "region3",
    "gap_region2",
    "region4",
    "region8",
    "region16",
    "region16dot",
    "region16_oracle",
    "gap_region16",
    "gap_region16_double",
    "gap01_region16_double",
    "gap03_region16_double",
    "region16_double",
    "gap_region16_double_sepadam",
    "wide64_sepadam",
    "predcode_sepadam",
    "lin_gap_region16_double",
    "lin_region16_double",
)
# Arms with a 1x1-conv readout instead of the MLP head: their untrained floor is their own step 0.
LINEAR = ("lin_gap_region16_double", "lin_region16_double")
READOUTS = ("code", "r2", "r3", "r4", "r8")
COLUMNS = (
    "brain_size",
    "ventricle_size",
    "lesion_burden",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
    "gain (own style)",
)


def load(*paths):
    scores = {}
    for path in paths:
        if os.path.exists(path):
            for row in map(json.loads, open(path)):
                scores.setdefault(row["step"], {}).update(row["scores"])
    return scores


def main(step):
    logs = {arm: load(f"{HERE}/{arm}.jsonl", f"{HERE}/{arm}_post.jsonl") for arm in ARMS}
    logs["base (+BT16)"] = load(f"{HERE}/../xpred_pilot_20261028/base.jsonl")
    width = max(map(len, logs))
    header = f"{'':>{width}}" + " ".join(f"{c[:10]:>10}" for c in COLUMNS)
    groups = (("floor", [arm for arm in logs if arm not in LINEAR]), ("floor (linear)", list(LINEAR)))
    for readout in READOUTS:
        for view in ("t1", "flair"):
            key = f"{readout}_{view}"
            titled = False
            for label, arms in groups:
                floor = next((logs[arm][0][key] for arm in arms if key in logs[arm].get(0, {})), None)
                rows = [(arm, logs[arm][step][key]) for arm in arms if key in logs[arm].get(step, {})]
                if floor is None or not rows:
                    continue
                if not titled:
                    print(f"\n{key} at step {step} (floor = untrained, same readout)\n{header}")
                    titled = True
                print(f"{label:>{width}}" + " ".join(f"{floor[c]:>10.3f}" for c in COLUMNS))
                for arm, row in rows:
                    print(f"{arm:>{width}}" + " ".join(f"{row[c]:>10.3f}" for c in COLUMNS))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
