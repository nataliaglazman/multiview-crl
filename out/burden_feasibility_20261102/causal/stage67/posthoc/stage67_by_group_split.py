"""Stage 67's rule cell (adni split, adjusted anchors, r = .9, grouping 'bank') broken down by bank group.

Rebuilds only the rule's columns with anchor_check's own functions and seeds, so the pooled values must equal
anchor.json's rule (checked), then reports rho and the share of the shortfall removed within each group, beside the
exact own parts and retrieval (top1_32, top1_all).
    python stage67_by_group.py CODES STAGE66_JSON ANCHOR_JSON
"""

import json
import sys
import zlib
from pathlib import Path

import anchor_check as A
import criterion_check as C
import numpy as np

codes, stage66_path, anchor_path = sys.argv[1:4]
stage66 = json.loads(Path(stage66_path).read_text())
anchor = json.loads(Path(anchor_path).read_text())
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "adni"
split, r, draws = A.SPLITS[SPLIT], 0.9, 10
measured = [k for k in range(len(A.FACTORS)) if k not in split]

entries = [e for e in A.load(codes) if e["run"] in stage66 and stage66[e["run"]]["step"] == e["step"]]
for e in entries:
    e["signals"] = stage66[e["run"]]["signals"]
print(f"{len(entries)} runs", flush=True)

by_bank = {}
for e in entries:
    by_bank.setdefault((e["bank_root"], len(e["zc"])), []).append(e)
for (root, n), members in sorted(by_bank.items()):
    first = members[0]
    m0 = A.measured_values(first["zc"], first["vox"])
    key = f"{Path(root).name}:{n}"
    cols = []
    for d in range(draws):  # the same generator bank_targets uses for (bank, r, d)
        m = A.with_noise(m0, r, np.random.default_rng([zlib.crc32(key.encode()), int(round(r * 1000)), d]))
        adj = A.adjusted(m, measured)
        cols += [adj[k] for k in measured]
    targets = np.column_stack(cols)
    for e in members:
        y = np.column_stack([e["own"], targets])
        r2 = np.mean([A.readout_r2(code, y) for code in e["codes"]], 0)
        e["own_r2"] = r2[: len(A.FACTORS)]
        e["picks"] = r2[len(A.FACTORS) :].reshape(draws, len(measured)).mean(1)
    print(f"read {key}: {len(members)} runs", flush=True)

truth = lambda e: float(np.mean(e["own_r2"][list(split)]))  # noqa: E731
rows = {
    "adjusted r=.9": [lambda e, d=d: e["picks"][d] for d in range(draws)],
    "exact own parts": [lambda e: float(np.mean(e["own_r2"][measured]))],
    "top1_32": [lambda e: C.SIGNALS["top1_32"] * e["signals"]["top1_32"]],
    "top1_all": [lambda e: C.SIGNALS["top1_all"] * e["signals"]["top1_all"]],
}
groups = A.grouped(entries, "bank")
pooled = A.over_draws(groups, rows["adjusted r=.9"], truth)
ref = anchor["bank"]["splits"][SPLIT]["rows"]["adjusted r=0.90"]
rule = dict(rho=ref["rho"], removed=ref["removed"])
print(f"\npooled rho {pooled['rho']:+.4f} (anchor.json {rule['rho']:+.4f}), "
      f"removed {pooled['removed']:+.4f} (anchor.json {rule['removed']:+.4f})")
assert abs(pooled["rho"] - rule["rho"]) < 1e-6 and abs(pooled["removed"] - rule["removed"]) < 1e-6, "no match"

print(f"\n{'group':38s} {'n':>3s} " + " ".join(f"{name:>22s}" for name in rows))
print(f"{'':38s} {'':>3s} " + " ".join(f"{'rho  removed':>22s}" for _ in rows))
for g in sorted(groups, key=len, reverse=True):
    cells = []
    for picks in rows.values():
        o = A.over_draws([g], picks, truth)
        cells.append(f"{o['rho']:+.2f}  {o['removed']:+.2f}")
    t = np.array([truth(e) for e in g])
    print(f"{C.group_of(g[0], 'bank'):38s} {len(g):3d} " + " ".join(f"{c:>22s}" for c in cells)
          + f"   hidden own parts {t.min():.2f}-{t.max():.2f}")
