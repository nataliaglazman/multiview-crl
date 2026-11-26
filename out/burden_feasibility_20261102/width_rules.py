"""E-W rules 2 and 3 on stage 74's width sweep (paper/experiment-plan-2026-11-24.md). No torch, no training.

Rule 1 (the held-out loss elbow) is width_sweep.py's and failed (HANDOFF item 63). The plan names two more label-free
rules without numbers. Their numbers were fixed here, before either was computed on any stage 74 code (26 Nov):

Rule 2, intrinsic dimension of the widest code.
  - Each 16-unit code (3 seeds x 2 views), standardized per unit, on the 1,000 test subjects: TwoNN at full N
    (intrinsic_dimension.py's "TwoNN N" column, the same estimator).
  - Calibration offset c = 7 - TwoNN of the bank's seven true factors on the same subjects (stage 56 read 6.37).
  - |C| estimate = mean TwoNN over seeds and views + c; width k_id = round(|C| estimate) + 1.
  - Pass on this bank if |k_id - 8| <= 1.
Rule 3, retrieval margin.
  - Per run: cross-view top-1 among all 1,000 test subjects (code_audit.top1: each view standardized, cosine, both
    directions averaged). Mean over seeds per width.
  - k_ret = the smallest width whose mean top-1 is at least top1(16) - .05 * (top1(16) - top1(4)): the same 5% of the
    range as rule 1's elbow, taken from the widest code as the plan words it.
  - Pass on this bank if |k_ret - 8| <= 1.
A rule sets ADNI's width only if it passes on all three banks (fixed causal, fixed independent, drawn causal); stage 74
covers the first only, so this can fail a rule but not pass it.

Other banks (26 Nov, after the fixed bank's result, before any other bank's): the same rules with that bank's |C|
(--latents) as target |C| + 1 and offset |C| - TwoNN of its true latents. On the drawn causal bank (stage 79) |C| is 10:
the seven factors and the lesion centre, the load-weighted centroid of each test subject's 16^3 lesion cells, as
intrinsic_dimension.py calibrates it (--bank gives the bank's test_cells.npy; --latents 10 requires it). The ends of
rule 3's range stay 4 and 16 units. --borrowed-offset C also gives rule 2 with another bank's offset (stage 79: the
fixed bank's .63), as ADNI would have to borrow one; it transfers if that also lands within one unit of |C| + 1.

Described, not judged: TwoNN at every width, on the unit sphere, and decimated to N/2, N/4, N/8; rule 3 at absolute
margins .01, .02 and .05; each rule per seed.

    python width_rules.py OUT_JSON CODES... [--latents 7] [--bank ROOT]   (CODES: export .npz files or folders of them)
    python width_rules.py --self-test
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

LATENT_COLUMNS = (0, 1, 2, 5, 6, 7, 8)  # anchor_check.LATENT_COLUMNS: the seven factors' columns of zc
FACTORS = 7
WIDEST, NARROWEST = 16, 4
FRAC = 0.05
FRACTIONS = (1, 2, 4, 8)


def standardized(x):  # code_audit.standardized
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    keep = (sd > 1e-8 * max(float(sd.max()), 1e-30)) & (np.ptp(x, axis=0) > 0)
    return np.divide(x - x.mean(0), sd, out=np.zeros_like(x), where=keep)


def two_nn(x, discard=0.1):  # intrinsic_dimension.two_nn
    d = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)[0][:, 1:]
    d = d[d[:, 0] > 0]
    mu = np.sort(d[:, 1] / d[:, 0])
    keep = int(np.floor(len(mu) * (1 - discard)))
    log_mu = np.log(mu[:keep])
    log_tail = -np.log(1 - np.arange(1, keep + 1) / len(mu))
    return float(log_mu @ log_tail / (log_mu @ log_mu))


def decimated(x, fraction, seed=0):  # intrinsic_dimension.decimated
    if fraction == 1:
        return two_nn(x)
    size = len(x) // fraction
    order = np.random.default_rng(seed).permutation(len(x))
    return float(np.mean([two_nn(x[order[i : i + size]]) for i in range(0, size * fraction, size)]))


def twonn_row(x, sphere=False):
    x = standardized(x)
    x = x[:, x.any(0)]
    if sphere:
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return [decimated(x, f) for f in FRACTIONS]


def top1_all(z1, z2):  # code_audit.top1 with one set of every subject
    z1, z2 = standardized(z1), standardized(z2)
    z1 /= np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12
    z2 /= np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12
    sim = z1 @ z2.T
    truth = np.arange(len(z1))
    return float((np.mean(sim.argmax(1) == truth) + np.mean(sim.argmax(0) == truth)) / 2)


def k_retrieval(widths, top1, margin):
    """Smallest width whose top-1 is at least the widest's minus margin."""
    target = top1[widths.index(max(widths))] - margin
    return next(k for k, t in zip(widths, top1) if t >= target)


def run_key(name):
    width = int(re.search(r"_c(\d+)_20k", name).group(1))
    seed = re.search(r"_s(\d+)$", name)
    return width, int(seed.group(1)) if seed else 42


def export_files(codes):
    files = []
    for c in map(Path, codes):
        files += sorted(c.glob("*.npz")) if c.is_dir() else [c]
    return files


def lesion_centroid(cells):  # batch_price.lesion_centroid, without its torch import
    load = np.asarray(cells, dtype=np.float64).reshape(len(cells), -1)
    grid = np.stack(np.meshgrid(*[np.arange(16)] * 3, indexing="ij"), -1).reshape(-1, 3)
    total = load.sum(1, keepdims=True)
    centre = load @ grid / np.maximum(total, 1e-12)
    centre[total[:, 0] <= 0] = centre[total[:, 0] > 0].mean(0)
    return centre


def true_latents(zc, latents, bank):
    """The bank's shared block on the export's subjects: the seven factors, plus the lesion centre for |C| 10."""
    if latents == FACTORS:
        return zc[:, LATENT_COLUMNS]
    if latents != FACTORS + 3 or bank is None:
        raise ValueError("--latents is 7 (the factors) or 10 (plus the lesion centre, which needs --bank)")
    stored = np.load(f"{bank}/test_zc.npy")[: len(zc)]
    if not np.allclose(stored, zc):
        raise ValueError(f"{bank}: its test subjects are not the exports'")
    centre = lesion_centroid(np.load(f"{bank}/test_cells.npy", mmap_mode="r")[: len(zc)])
    return np.column_stack([zc[:, LATENT_COLUMNS], centre])


def measure(codes, latents=FACTORS, bank=None):
    rows, zc0 = [], None
    for npz in export_files(codes):
        d = np.load(npz)
        width, seed = run_key(npz.stem)
        zc = np.asarray(d["zc"], dtype=np.float64)
        if zc0 is None:
            zc0 = zc
        elif not np.allclose(zc0, zc):
            raise ValueError(f"{npz.name}: different test subjects from the first export")
        views = [d["code_t1"], d["code_flair"]]
        rows.append(
            {
                "run": npz.stem,
                "width": width,
                "seed": seed,
                "twonn": [twonn_row(v) for v in views],
                "twonn_sphere": [twonn_row(v, sphere=True) for v in views],
                "top1": top1_all(*views),
            }
        )
        print(
            f"{npz.stem}: width {width} seed {seed} TwoNN {rows[-1]['twonn'][0][0]:.2f}/{rows[-1]['twonn'][1][0]:.2f}"
            f" top-1 {rows[-1]['top1']:.3f}",
            flush=True,
        )
    if zc0 is None:
        raise ValueError("no exports given")
    true = twonn_row(true_latents(zc0, latents, bank))
    return {"rows": rows, "true": true, "latents": latents, "subjects": len(zc0)}


def score(m, borrowed=None):
    by = defaultdict(list)
    for r in m["rows"]:
        by[r["width"]].append(r)
    widths = sorted(by)
    if WIDEST not in by or NARROWEST not in by:
        raise ValueError(f"needs widths {NARROWEST} and {WIDEST}")
    latents = m["latents"]
    offset = latents - m["true"][0]
    out = {"widths": widths, "latents": latents, "offset": offset, "true": m["true"], "per_width": {}}
    for k in widths:
        rs = by[k]
        tw = np.array([v for r in rs for v in r["twonn"]])  # (seeds x views, fractions)
        ts = np.array([v for r in rs for v in r["twonn_sphere"]])
        out["per_width"][k] = {
            "seeds": sorted(r["seed"] for r in rs),
            "twonn": tw.mean(0).tolist(),
            "twonn_sd": tw[:, 0].std().item(),
            "twonn_sphere": ts.mean(0).tolist(),
            "top1": float(np.mean([r["top1"] for r in rs])),
            "top1_sd": float(np.std([r["top1"] for r in rs])),
        }
    # rule 2
    d_hat = out["per_width"][WIDEST]["twonn"][0]
    c_hat = d_hat + offset
    k_id = int(round(c_hat)) + 1
    per_seed_id = {r["seed"]: int(round(np.mean([v[0] for v in r["twonn"]]) + offset)) + 1 for r in by[WIDEST]}
    out["rule2"] = {
        "twonn_widest": d_hat,
        "c_hat": c_hat,
        "k_id": k_id,
        "per_seed": per_seed_id,
        "pass": abs(k_id - (latents + 1)) <= 1,
    }
    if borrowed is not None:
        k_b = int(round(d_hat + borrowed)) + 1
        out["rule2"]["borrowed"] = {"offset": borrowed, "k_id": k_b, "pass": abs(k_b - (latents + 1)) <= 1}
    # rule 3
    top1 = [out["per_width"][k]["top1"] for k in widths]
    margin = FRAC * (top1[widths.index(WIDEST)] - top1[widths.index(NARROWEST)])
    k_ret = k_retrieval(widths, top1, margin)
    seeds = sorted({r["seed"] for r in m["rows"]})
    per_seed_ret = {}
    for s in seeds:
        t = {r["width"]: r["top1"] for r in m["rows"] if r["seed"] == s}
        ws = sorted(t)
        tt = [t[k] for k in ws]
        per_seed_ret[s] = k_retrieval(ws, tt, FRAC * (t[WIDEST] - t[NARROWEST]))
    out["rule3"] = {
        "margin": margin,
        "k_ret": k_ret,
        "per_seed": per_seed_ret,
        "absolute": {a: k_retrieval(widths, top1, a) for a in (0.01, 0.02, 0.05)},
        "pass": abs(k_ret - (latents + 1)) <= 1,
    }
    return out


def report(s):
    t, target = s["true"], s["latents"] + 1
    lines = [
        f"{s['latents']} true latents, same subjects: TwoNN N {t[0]:.2f} (N/2 {t[1]:.2f}, N/4 {t[2]:.2f}, N/8"
        f" {t[3]:.2f}); offset c = {s['offset']:.2f}",
        "",
        "width  seeds   TwoNN N (sd)   N/2    N/4    N/8   | sphere N  | top-1 all (sd)",
    ]
    for k in s["widths"]:
        p = s["per_width"][k]
        t = p["twonn"]
        lines.append(
            f"{k:5d}  {len(p['seeds']):5d}   {t[0]:5.2f} ({p['twonn_sd']:.2f})   {t[1]:5.2f}  {t[2]:5.2f}"
            f"  {t[3]:5.2f} |  {p['twonn_sphere'][0]:5.2f}    |  {p['top1']:.3f} ({p['top1_sd']:.3f})"
        )
    r2, r3 = s["rule2"], s["rule3"]
    lines += [
        "",
        f"rule 2: TwoNN of the {WIDEST}-unit code {r2['twonn_widest']:.2f} + c {s['offset']:.2f} = |C| estimate"
        f" {r2['c_hat']:.2f} -> k_id {r2['k_id']} (per seed {r2['per_seed']}); target {target}:"
        f" {'PASS on this bank' if r2['pass'] else 'FAIL'}",
        f"rule 3: margin {r3['margin']:.3f} below the {WIDEST}-unit top-1 -> k_ret {r3['k_ret']} (per seed"
        f" {r3['per_seed']}); target {target}: {'PASS on this bank' if r3['pass'] else 'FAIL'}",
        f"        absolute margins: {r3['absolute']}",
    ]
    if "borrowed" in r2:
        b = r2["borrowed"]
        lines.insert(
            -2,
            f"rule 2, borrowed offset {b['offset']:.2f}: |C| estimate {r2['twonn_widest'] + b['offset']:.2f} -> k_id"
            f" {b['k_id']}; target {target}: {'transfers' if b['pass'] else 'does not transfer'}",
        )
    return "\n".join(lines)


def self_test():
    rng = np.random.default_rng(0)
    for d in (3, 6):
        x = rng.standard_normal((2000, d))
        assert abs(two_nn(x) - d) < 0.6, (d, two_nn(x))
    z = rng.standard_normal((300, 8))
    assert top1_all(z, z) == 1.0
    assert top1_all(z, rng.standard_normal((300, 8))) < 0.05
    assert k_retrieval([4, 8, 16], [0.2, 0.89, 0.9], 0.02) == 8
    assert k_retrieval([4, 8, 16], [0.2, 0.85, 0.9], 0.02) == 16
    assert run_key("x_c12_20k_s3") == (12, 3) and run_key("x_c8_20k") == (8, 42)
    cells = np.zeros((2, 16, 16, 16))
    cells[0, 1, 2, 3] = 1
    cells[1, 5, 5, 5] = cells[1, 7, 5, 5] = 1
    assert np.allclose(lesion_centroid(cells), [[1, 2, 3], [6, 5, 5]])
    zc = rng.standard_normal((5, 9))
    assert true_latents(zc, 7, None).shape == (5, 7)
    try:
        true_latents(zc, 10, None)
        raise AssertionError("|C| 10 without a bank must fail")
    except ValueError:
        pass
    print("self-test passed")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("out", nargs="?")
    p.add_argument("codes", nargs="*")
    p.add_argument("--latents", type=int, default=FACTORS, help="The bank's |C|: 7, or 10 with the lesion centre")
    p.add_argument("--bank", help="The bank's folder, for the lesion centre (--latents 10)")
    p.add_argument("--borrowed-offset", type=float, help="Also score rule 2 with this offset (another bank's)")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    m = measure(a.codes, a.latents, a.bank)
    s = score(m, a.borrowed_offset)
    text = report(s)
    print(text)
    Path(a.out).write_text(json.dumps({"measured": m, "scored": s}, indent=1))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
