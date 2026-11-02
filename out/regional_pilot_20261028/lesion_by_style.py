"""Are lesions harder to find in some acquisitions? The cell probe's top-cell hit, split by each view's gain, bias and noise.

The probe is cellcode.probe_between's, kept out of fold by subject, returning each subject's hit instead of the mean, so the
overall rate must reproduce cellcode.py. Each row splits the 400 validation subjects into thirds of one quantity of the scored
view and gives the hit rate per third, and high third minus low third with a subject bootstrap 95% interval. Lesion size is
the reference: the lesion's total load over cells.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/lesion_by_style.py BANK_DIR OUT_JSON CHECKPOINT
"""

import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402

L, R, pilot, posthoc, T = C.L, C.R, C.pilot, C.posthoc, C.T
VIEWS = ("t1", "flair")
RESAMPLES = 2000


def subject_hits(fit, score, brain, lesion):
    """probe_between's top-cell hit per subject (nan for a subject never scored)."""
    n = fit.shape[0]
    fit = fit.reshape(n, fit.shape[1], -1).transpose(0, 2, 1)
    score = score.reshape(n, score.shape[1], -1).transpose(0, 2, 1)
    lesion = lesion.reshape(n, -1)
    brain = brain.reshape(n, -1) & (lesion >= posthoc.LESION_FRACTION).any(0)
    rng = np.random.default_rng(0)
    fitted, labels, groups = [], [], []
    for i in range(n):
        pos = np.flatnonzero(brain[i] & (lesion[i] >= posthoc.LESION_FRACTION))
        neg = np.flatnonzero(brain[i] & (lesion[i] == 0))
        neg = rng.choice(neg, size=min(len(neg), 40), replace=False)
        for cells, label in ((pos, 1), (neg, 0)):
            fitted.append(fit[i, cells])
            labels += [label] * len(cells)
            groups += [i] * len(cells)
    X, y, g = np.concatenate(fitted), np.array(labels), np.array(groups)
    hits = np.full(n, np.nan)
    for train, test in GroupKFold(5).split(X, y, g):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        probe.fit(X[train], y[train])
        for i in np.unique(g[test]):
            inside = np.flatnonzero(brain[i])
            top = inside[np.argmax(probe.decision_function(score[i, inside]))]
            hits[i] = lesion[i, top] > 0
    return hits


def split(hits, value):
    """Hit rate per third of value, and high third minus low third with a bootstrap interval."""
    edges = np.quantile(value, [1 / 3, 2 / 3])
    third = np.digitize(value, edges)
    rates = [float(np.nanmean(hits[third == t])) for t in range(3)]
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(RESAMPLES):
        pick = rng.integers(0, len(hits), len(hits))
        h, t = hits[pick], third[pick]
        gaps.append(np.nanmean(h[t == 2]) - np.nanmean(h[t == 0]))
    low, high = np.quantile(gaps, [0.025, 0.975])
    return {"thirds": rates, "gap": rates[2] - rates[0], "interval": [float(low), float(high)]}


def main():
    bank_dir, out_json, state = sys.argv[1], sys.argv[2], sys.argv[3]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    bank = pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    lesion = np.load(L.CACHE)
    keep = T.foreground_positions(bank.mask[:, None], C.CELLS, C.FOREGROUND)
    out = C.cell_maps(C.build(state, pilot.user_args(), device), bank, device)
    units = C.standardised(out["code_t1"][:, : C.CONTENT], out["code_flair"][:, : C.CONTENT], keep, True)[0]
    size = lesion.reshape(len(lesion), -1).sum(1)
    style = {
        "gain": lambda view: np.clip(bank.zs[:, view, 0], -1, 1),
        "bias": lambda view: np.clip(bank.zs[:, view, 1], -1, 1),
        "noise": lambda view: np.abs(bank.zs[:, view, 2]),
        "lesion size": lambda view: size,
    }
    results = {}
    for v, view in enumerate(VIEWS):
        image = bank.targets[v][:, 0].numpy()[:, None]
        image = image - image.mean(0, keepdims=True)
        h = out[view].numpy()
        reads = {
            "image": (image, image),
            "map": (h - h.mean(0, keepdims=True), h - h.mean(0, keepdims=True)),
            "code": (units[v], units[v]),
            "transfer": (units[1 - v], units[v]),
        }
        for name, (fit, score) in reads.items():
            hits = subject_hits(fit, score, brain, lesion)
            top = C.probe_between(fit, score, brain, lesion)[1]
            print(f"{view} {name}: overall {100 * np.nanmean(hits):.1f}% (probe_between {100 * top:.1f}%)", flush=True)
            for factor, values in style.items():
                row = split(hits, values(v))
                results[f"{view} {name} {factor}"] = row
                thirds = " / ".join(f"{100 * r:4.1f}" for r in row["thirds"])
                print(
                    f"   {factor:<12} low/mid/high {thirds}   gap {100 * row['gap']:+5.1f} [{100 * row['interval'][0]:+.1f}, {100 * row['interval'][1]:+.1f}]",
                    flush=True,
                )
            json.dump(results, open(out_json, "w"), indent=1)
    print("correlation of lesion size with gain, bias, noise, per view:")
    for v, view in enumerate(VIEWS):
        print(
            " ",
            view,
            [round(float(np.corrcoef(size, style[k](v))[0, 1]), 3) for k in ("gain", "bias", "noise")],
            flush=True,
        )


if __name__ == "__main__":
    main()
