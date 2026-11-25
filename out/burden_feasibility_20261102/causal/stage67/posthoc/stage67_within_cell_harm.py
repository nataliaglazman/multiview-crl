"""Post hoc, after stage 67's reading: within one cell (units, arm) of the fixed-position bank, does stage 67's rule
score (adni split, adjusted anchors, r = .9, mean over 10 draws) separate the seeds that hold from those that fail?
Holds verdicts from causal/stage64/holds.txt and causal/stage65/holds.txt.
    python stage67_within_cell.py CODES
"""

import json
import re
import sys
import zlib
from itertools import product
from pathlib import Path

import anchor_check as A
import numpy as np
from scipy.stats import spearmanr

codes = Path(sys.argv[1])
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos"
ARMS = {
    "c8_e500": "_abl_codebatch_predict1_c8_20k",
    "c8_e100": "_abl_codebatch100_predict1_c8_20k",
    "c9_e500": "_abl_codebatch_predict1_c9_20k",
    "c9_e100": "_abl_codebatch100_predict1_c9_20k",
    "c10_e500": "_abl_codebatch_predict1_c10_20k",
    "c8_cont": "_abl_codebatch_predict1_c8_20k_plus10k",
    "c8_pca": "_abl_from10pca_predict1_c8_10k",
    "c8_red": "_abl_from10red_predict1_c8_10k",
    "c8_harm": "_abl_codebatch_predict1_harm1_c8_20k",
}
holds = {}
for stage in ("stage64", "stage65", "stage60"):
    for line in (codes.parent / stage / "holds.txt").read_text().splitlines():
        f = line.split()
        if len(f) > 8 and re.fullmatch(r"c\d+_[a-z0-9]+_s\d+", f[0]) and f[0].rsplit("_s", 1)[0] in ARMS:
            last = min(float(v) for v in f[1].split("/"))
            holds[f[0]] = (last, f[7] == "yes")

split, r, draws = A.SPLITS["adni"], 0.9, 10
measured = [k for k in range(len(A.FACTORS)) if k not in split]
rows = []
targets = None
for label in sorted(holds):
    arm, seed = label.rsplit("_s", 1)
    path = codes / f"{P}{ARMS[arm]}{'' if seed == '42' else '_s' + seed}.npz"
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        e = {**meta, "codes": [np.asarray(d["code_t1"], float), np.asarray(d["code_flair"], float)],
             "zc": np.asarray(d["zc"], float), "vox": np.asarray(d["vox"], float),
             "own": np.asarray(d["own"], float)}
    if targets is None:
        key = f"{Path(e['bank_root']).name}:{len(e['zc'])}"
        m0 = A.measured_values(e["zc"], e["vox"])
        cols = []
        for d in range(draws):
            m = A.with_noise(m0, r, np.random.default_rng([zlib.crc32(key.encode()), int(round(r * 1000)), d]))
            adj = A.adjusted(m, measured)
            cols += [adj[k] for k in measured]
        targets, zc0 = np.column_stack(cols), e["zc"]
    assert np.allclose(e["zc"], zc0), label
    y = np.column_stack([e["own"], targets])
    r2 = np.mean([A.readout_r2(c, y) for c in e["codes"]], 0)
    own = r2[: len(A.FACTORS)]
    score = r2[len(A.FACTORS) :].reshape(draws, len(measured)).mean(1).mean()
    rows.append((arm, int(seed), score, float(own[list(split)].mean()), float(own.mean()), float(own.min()),
                 *holds[label]))

print(f"{'cell':9s} {'seed':>4s} {'anchors':>8s} {'hidden':>7s} {'own mean':>9s} {'own min':>8s} {'last':>6s} holds")
pairs = right = 0
tops = []
for arm in ARMS:
    cell = sorted([x for x in rows if x[0] == arm], key=lambda x: -x[2])
    if not cell:
        continue
    for x in cell:
        print(f"{x[0]:9s} {x[1]:4d} {x[2]:8.3f} {x[3]:7.2f} {x[4]:9.2f} {x[5]:8.2f} {x[6]:6.2f} {'yes' if x[7] else 'no'}")
    rho = spearmanr([x[2] for x in cell], [x[6] for x in cell]).correlation
    for a, b in product(cell, cell):
        if a[7] and not b[7]:
            pairs += 1
            right += a[2] > b[2]
    tops.append(cell[0][7])
    print(f"  {arm}: Spearman(anchors, last) {rho:+.2f}; top pick holds: {'yes' if cell[0][7] else 'no'};"
          f" cell holds {sum(x[7] for x in cell)}/{len(cell)}\n")
print(f"holder above non-holder in the same cell: {right}/{pairs} pairs")
print(f"anchors' top pick holds in {sum(tops)}/{len(tops)} cells; codes hold in "
      f"{sum(x[7] for x in rows)}/{len(rows)}")
