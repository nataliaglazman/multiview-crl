"""Stage 72 (no training): can a seventh shared local direction be measured without labels?

Among similar subjects a curve moves with the factor it bends, so it adds no local direction, while an independent
factor that both views see adds one. A code that holds all seven own parts should show seven local directions the two
views share; a code that lost one, six and a curve's leftover. (Suggested by another assistant, adapted in HANDOFF.md
stage 72.)

    python local_directions.py OUT.json [--sizes 32 64 128] [--shuffles 5] [--seed 0] > local_directions.txt

The pass rule reads k = 32 and 64 (RULE_SIZES); 128 is described only. It was added after the synthetic tests and
before any trained code was scored: at k = 32 the held-out profile is noise-limited (16 subjects a half for 8
directions), and 128 shows whether the signal is there once that noise is smaller.

Reads every fixed-position causal-bank code with a hold verdict (causal/stage{64,65,60,70}/holds.txt) from its bundle
(the 1,000 test subjects, both views). Per code:
  - whitened by the covariance of the two views' mean code;
  - around each subject a neighbourhood of k subjects (the subject and its k - 1 nearest), split in two at random by
    one draw per subject that every code shares;
  - on one half, the shared directions: eigenvectors of the symmetrized T1-FLAIR cross-covariance of the deviations
    from that half's mean, ranked by eigenvalue (independent view noise does not enter a cross-covariance); on the
    other half, the shared covariance along each (the mean product of the views' deviations) and its correlation;
    both ways round, averaged over neighbourhoods, never pooled across them (a curve's local slope changes from one
    neighbourhood to the next, so pooling would make it a direction again);
  - s7: the seventh direction's held-out shared covariance (seven content factors, at every width).
Neighbourhoods 'own': the code's own, both views side by side, each unit standardized (training's rule,
data.cluster_batches.nearest_neighbours); 'fixed': the seven true factors standardized, the same for every code (a
diagnostic: it uses labels). Null: the same with the T1-FLAIR pairs shuffled within each neighbourhood. Calibration
rows: the bank's true own parts plus independent view noise (reliability .8), as a 7 + 1 code (the seven parts and
brain's part squared) and a 6 + 2 code (ventricle and burden merged into their sum, plus temporal's part squared).
Pass rule fixed in HANDOFF.md before the results.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import data.cluster_batches as cb

HERE = Path(__file__).resolve().parent
P = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos"
ARMS = {
    "e500": "_abl_codebatch_predict1_c{u}_20k",
    "e100": "_abl_codebatch100_predict1_c{u}_20k",
    "cont": "_abl_codebatch_predict1_c8_20k_plus10k",
    "pca": "_abl_from10pca_predict1_c8_10k",
    "red": "_abl_from10red_predict1_c8_10k",
    "harm": "_abl_codebatch_predict1_harm1_c8_20k",
    "tau1": "_abl_codebatch_predict1_tau1_c8_20k",
}
STAGES = ("stage64", "stage65", "stage60", "stage70")
SEVENTH = 6  # index of the seventh direction
RULE_SIZES = (32, 64)  # the pass rule's neighbourhood sizes; any other size is described only
LABEL = re.compile(r"c(\d+)_([a-z0-9]+)_s(\d+)")


def verdicts(causal):
    """label -> (holds, the spectrum's last value (lower view), top-1 over 1,000), one verdict per code."""
    out = {}
    for stage in STAGES:
        for line in (causal / stage / "holds.txt").read_text().splitlines():
            f = line.split()
            if len(f) > 8 and LABEL.fullmatch(f[0]):
                row = (f[7] == "yes", min(float(v) for v in f[1].split("/")), float(f[6]))
                if f[0] in out and out[f[0]][0] != row[0]:
                    raise SystemExit(f"{f[0]}: the stages' verdicts differ")
                out.setdefault(f[0], row)
    return out


def bundle_path(causal, label):
    units, arm, seed = LABEL.fullmatch(label).groups()
    return causal / f"{P}{ARMS[arm].format(u=units)}{'' if seed == '42' else '_s' + seed}.npz"


def order(label):
    units, arm, seed = LABEL.fullmatch(label).groups()
    return int(units), list(ARMS).index(arm), int(seed)


def whiten(t1, flair, floor=1e-8):
    mean = (t1 + flair) / 2
    w, v = np.linalg.eigh(np.cov(mean, rowvar=False))
    m = v / np.sqrt(np.maximum(w, w.max() * floor))
    centre = mean.mean(0)
    return (t1 - centre) @ m, (flair - centre) @ m


def neighbourhoods(features, k):
    """(N, k): each subject, then its k - 1 nearest by training's rule (each feature standardized, Euclidean)."""
    return np.column_stack([np.arange(len(features)), cb.nearest_neighbours(features, k - 1)])


def splits(n, k, seed):
    """Two (N, k // 2) arrays of positions within each neighbourhood: one random split per subject."""
    order_ = np.argsort(np.random.default_rng([seed, k]).random((n, k)), axis=1)
    return order_[:, : k // 2], order_[:, k // 2 : 2 * (k // 2)]


def centred(v, idx):
    part = v[idx]
    return part - part.mean(1, keepdims=True)


def profile(x, y, gx, gy, split):
    """Held-out shared covariance and correlation along each local shared direction, averaged over neighbourhoods.
    x, y: (N, d) whitened views; gx, gy: (N, k) the members whose T1 and FLAIR codes pair up (gy = gx unshuffled)."""
    covs, cors = [], []
    for fit, held in (split, split[::-1]):
        xf, yf = centred(x, np.take_along_axis(gx, fit, 1)), centred(y, np.take_along_axis(gy, fit, 1))
        c = np.einsum("nhi,nhj->nij", xf, yf) / xf.shape[1]
        directions = np.linalg.eigh((c + c.transpose(0, 2, 1)) / 2)[1][:, :, ::-1]  # largest first
        xh, yh = centred(x, np.take_along_axis(gx, held, 1)), centred(y, np.take_along_axis(gy, held, 1))
        px, py = np.einsum("nhi,nij->nhj", xh, directions), np.einsum("nhi,nij->nhj", yh, directions)
        cov = (px * py).mean(1)
        covs.append(cov.mean(0))
        cors.append((cov / np.sqrt((px**2).mean(1) * (py**2).mean(1) + 1e-12)).mean(0))
    return np.mean(covs, 0), np.mean(cors, 0)


def shuffled(groups, rng):
    """The same neighbourhoods with each one's members permuted, for the FLAIR side of the pairs."""
    return np.take_along_axis(groups, np.argsort(rng.random(groups.shape), axis=1), 1)


def score(t1, flair, factors, sizes, shuffles, seed):
    x, y = whiten(t1, flair)
    out = {}
    for kind, features in (("own", np.concatenate([t1, flair], 1)), ("fixed", factors)):
        for k in sizes:
            groups = neighbourhoods(features, k)
            split = splits(len(x), k, seed)
            cov, cor = profile(x, y, groups, groups, split)
            rng = np.random.default_rng([seed, k, 1])
            null = [float(profile(x, y, groups, shuffled(groups, rng), split)[0][SEVENTH]) for _ in range(shuffles)]
            out[f"{kind}{k}"] = {"cov": cov.round(5).tolist(), "cor": cor.round(4).tolist(), "null": null}
    return out


def calibration(own, rng, reliability=0.8):
    """Two codes with known structure from the bank's own parts (brain, ventricle, burden, thickness, temporal, lr,
    sulcal), each view with its own noise: 7 + 1 and 6 + 2."""
    z = (own - own.mean(0)) / own.std(0)

    def square(v):
        s = v**2
        return (s - s.mean()) / s.std()

    holder = np.column_stack([z, square(z[:, 0])])
    merged = (z[:, 1] + z[:, 2]) / np.sqrt(2)
    failure = np.column_stack([z[:, [0, 3, 4, 5, 6]], merged, square(z[:, 0]), square(z[:, 4])])
    sd = np.sqrt((1 - reliability) / reliability)
    return {
        name: [code + sd * rng.standard_normal(code.shape) for _ in range(2)]
        for name, code in (("cal_7+1", holder), ("cal_6+2", failure))
    }


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def s7(row, key):
    return row[key]["cov"][SEVENTH]


def report(rows, sizes):
    codes = [x for x in rows if LABEL.fullmatch(x)]
    narrow = [x for x in codes if order(x)[0] == 8]
    keys = [f"{kind}{k}" for kind in ("own", "fixed") for k in sizes]
    first = f"own{sizes[0]}"
    print("s7: the seventh local shared direction's held-out covariance (whitened code units); null: largest of the")
    print(f"shuffled values at {first}; r7: its held-out correlation at {first}.\n")
    print(f"{'code':15} {'holds':5} {'last':>5} {'top-1':>6} " + " ".join(f"{k:>8}" for k in keys)
          + f" {'null max':>8} {'r7':>6}")
    for label in sorted(codes, key=order):
        r = rows[label]
        print(f"{label:15} {'yes' if r['holds'] else 'no':5} {r['last']:5.2f} {r['top1']:6.3f} "
              + " ".join(f"{s7(r, k):8.4f}" for k in keys)
              + f" {max(r[first]['null']):8.4f} {r[first]['cor'][SEVENTH]:6.3f}")
    for name in ("cal_7+1", "cal_6+2"):
        r = rows[name]
        print(f"{name:15} {'':5} {'':>5} {'':>6} " + " ".join(f"{s7(r, k):8.4f}" for k in keys)
              + f" {max(r[first]['null']):8.4f} {r[first]['cor'][SEVENTH]:6.3f}")

    held = [x for x in narrow if rows[x]["holds"]]
    lost = [x for x in narrow if not rows[x]["holds"]]
    need = len(held) - 1
    for kind, title in (("own", "Pass rule (own neighbourhoods"), ("fixed", "The same rule on fixed neighbourhoods (labels")):
        print(f"\n{title}; {len(narrow)} 8-unit codes, {len(held)} hold; k = {', '.join(map(str, RULE_SIZES))}):")
        passes = []
        for k in sizes:
            value = auc([s7(rows[x], f"{kind}{k}") for x in held], [s7(rows[x], f"{kind}{k}") for x in lost])
            judged = k in RULE_SIZES
            if judged:
                passes.append(value >= 0.80)
            mark = ("pass" if value >= 0.80 else "fail") if judged else "described only"
            print(f"  AUC of s7, holders against failures, k = {k}: {value:.3f} ({mark})")
        key = f"{kind}{RULE_SIZES[0]}"
        real = sum(s7(rows[x], key) > max(rows[x][key]["null"]) for x in held)
        passes.append(real >= need)
        print(f"  holders whose s7 at k = {RULE_SIZES[0]} exceeds every shuffled value: {real}/{len(held)} (need {need})")
        print(f"  verdict: {'PASS' if all(passes) else 'FAIL'}")

    print(f"\nPer cell at {first}, holder-failure pairs on the wrong side (holder's s7 not above the failure's):")
    for units in sorted({order(x)[0] for x in codes}):
        for arm in ARMS:
            cell = [x for x in codes if order(x)[0] == units and LABEL.fullmatch(x)[2] == arm]
            yes = [x for x in cell if rows[x]["holds"]]
            no = [x for x in cell if not rows[x]["holds"]]
            if not cell:
                continue
            wrong = sum(s7(rows[a], first) <= s7(rows[b], first) for a in yes for b in no)
            pairs = len(yes) * len(no)
            print(f"  {units} units {arm:5}: holds {len(yes)}/{len(cell)}"
                  + (f", wrong side {wrong}/{pairs}" if pairs else ", no pair"))
    for units in (9, 10):
        cell = [x for x in codes if order(x)[0] == units]
        yes = [s7(rows[x], first) for x in cell if rows[x]["holds"]]
        no = [s7(rows[x], first) for x in cell if not rows[x]["holds"]]
        if yes and no:
            print(f"  {units} units pooled: AUC {auc(yes, no):.3f} ({len(yes)} hold, {len(no)} fail)")

    values = [s7(rows[x], first) for x in narrow]
    print(f"\nSpearman over the 8-unit codes at {first}: s7 with the spectrum's last value "
          f"{spearmanr(values, [rows[x]['last'] for x in narrow])[0]:+.2f}, with top-1 over 1,000 "
          f"{spearmanr(values, [rows[x]['top1'] for x in narrow])[0]:+.2f}")
    print(f"\nProfiles at {first} (held-out shared covariance, directions 1-8):")
    for name in ("c8_e500_s2", "c8_e500_s42", "cal_7+1", "cal_6+2"):
        if name in rows:
            print(f"  {name:12} " + " ".join(f"{v:.3f}" for v in rows[name][first]["cov"][:8]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="JSON with every code's profiles and nulls")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    causal = HERE / "causal"
    marks = verdicts(causal)
    rows, reference = {}, None
    for label in sorted(marks, key=order):
        with np.load(bundle_path(causal, label), allow_pickle=False) as d:
            digest = json.loads(str(d["meta"]))["factor_digest"]
            t1, flair = np.asarray(d["emb_view1"], float), np.asarray(d["emb_view2"], float)
            factors, own = np.asarray(d["z_content"], float), np.asarray(d["z_own"], float)
        if reference is None:
            reference = digest, factors, own
        if digest != reference[0]:
            raise SystemExit(f"{label}: its bundle holds other test subjects")
        holds, last, top1 = marks[label]
        rows[label] = {"holds": holds, "last": last, "top1": top1, **score(t1, flair, factors, a.sizes, a.shuffles, a.seed)}
    for name, (v1, v2) in calibration(reference[2], np.random.default_rng([a.seed, 7])).items():
        rows[name] = score(v1, v2, reference[1], a.sizes, a.shuffles, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    report(rows, a.sizes)


if __name__ == "__main__":
    main()
