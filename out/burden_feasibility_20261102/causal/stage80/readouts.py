"""Stage 80, three failing backbones (described only): each code's readouts, source against its two refitted heads.

Per code and view: R2 of each factor's own part and of the factor itself from the code (own_variation.ridge, 5-fold,
the 1,000 test subjects of the code's bundle), and from detect's JSON the hold verdict, the spectrum's last value, the
ventricle-burden worst direction and rho1.

    python readouts.py DETECT.json SEED...
"""

import json
import sys
from pathlib import Path

import numpy as np
import own_variation

HERE = Path(__file__).resolve().parent
CAUSAL = HERE.parent
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k"
NAMES = ["brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal"]


def source(seed):
    return P if seed == 42 else f"{P}_s{seed}" if 2 <= seed <= 5 else f"{P}_a100_s{seed}"


def read(name):
    b = np.load(CAUSAL / f"{name}.npz")
    own, raw = b["z_own"].astype(np.float64), b["z_content"].astype(np.float64)
    out = {}
    for view, key in (("T1", "emb_view1"), ("FLAIR", "emb_view2")):
        code = b[key].astype(np.float64)
        out[view] = {"own": [own_variation.ridge(code, own[:, j]) for j in range(7)],
                     "raw": [own_variation.ridge(code, raw[:, j]) for j in range(7)]}
    return out


def main():
    detect = json.loads(Path(sys.argv[1]).read_text())["rows"]
    seeds = [int(s) for s in sys.argv[2:]]
    print("own-part R2 from the code, T1 / FLAIR; holds: last spectrum value and ventricle-burden worst >= .40 in both views")
    print(f"{'code':12s} {'verdict':7s} {'rho1':>5s}  " + "  ".join(f"{n:>11s}" for n in NAMES))
    for seed in seeds:
        for arm, label, name in (("source", f"c8_src_s{seed}", source(seed)),
                                 ("control", f"c8_ctrl_s{seed}", f"{source(seed)}_refit80_control"),
                                 ("suff", f"c8_suff_s{seed}", f"{source(seed)}_refit80_suff")):
            r = read(name)
            d = detect.get(label, {})
            verdict = "-" if not d else ("holds" if d["holds"] else "fails")
            rho = f"{d['rho1']:.2f}" if d else "-"
            cells = "  ".join(f"{r['T1']['own'][j]:5.2f}/{r['FLAIR']['own'][j]:4.2f}" for j in range(7))
            print(f"s{seed:<3d} {arm:7s} {verdict:7s} {rho:>5s}  {cells}")
            if d:
                print(f"{'':21s} last {d['last'][0]:.2f}/{d['last'][1]:.2f}  ventricle-burden worst "
                      f"{d['worst'][0]:.2f}/{d['worst'][1]:.2f}")
        print()
    print("whole factors, R2 from the code, T1 / FLAIR")
    for seed in seeds:
        for arm, name in (("source", source(seed)), ("control", f"{source(seed)}_refit80_control"),
                          ("suff", f"{source(seed)}_refit80_suff")):
            r = read(name)
            print(f"s{seed:<3d} {arm:7s} " + "  ".join(f"{r['T1']['raw'][j]:5.2f}/{r['FLAIR']['raw'][j]:4.2f}"
                                                     for j in range(7)))


if __name__ == "__main__":
    main()
