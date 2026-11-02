"""Which prediction target would make a saved map's one-number guess read the lesion? No training, no labels in any fit.

For each target a 1x1x1 linear predictor is fitted by least squares on the first N_FIT train subjects' kept cells, as
xpred_cells.py does, from each view's map and from its 14 cell units. Lesion labels only score the guess on validation
subjects, through the cell probe, on the guess minus the target's population mean.

  raw     the other view's z-scored cell, what the prediction term trains on
  pcaK    raw minus its projection on the K largest population directions of that view's cell images, fitted on all train
          subjects: what the K biggest differences between subjects do not explain
  hp      raw minus the mean of the kept cells in its 3x3x3 neighbourhood
  boost   raw minus the checkpoint's own guess from the cell units: what the code could not predict (needs a predictor)

Rows per target and view:
  share   the lesion cells' share of the target's energy over kept cells: how much a squared error can pay for the lesion
  target  the probe on the target itself: what a perfect guess would read
  map, code  the probe on the guess from the map and from the 14 units

hard cells: with a stored predictor, how often a subject's worst 1% of kept cells by squared error include a lesion cell,
and the lesion's share of those cells: whether a loss on the worst cells would land on the lesion.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/residual_targets.py BANK_DIR OUT_JSON CHECKPOINT ...
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402

L, R, pilot, posthoc = C.L, C.R, C.pilot, C.posthoc
VIEWS = ("t1", "flair")
N_FIT = 800
BATCH = 32
RANKS = (16, 64, 256)


def population_basis(bank, view):
    """A view's per-cell population mean, the cells any subject keeps, and its largest population directions."""
    keep, y = bank.keep[:, 0], bank.targets[view][:, 0]
    mean = (y * keep).sum(0) / keep.sum(0).clamp_min(1)
    cells = keep.any(0)
    deviation = ((y - mean) * keep)[:, cells].double()
    return mean, cells, torch.linalg.svd(deviation, full_matrices=False)[2][: max(RANKS)]


def residual(y, keep, mean, cells, basis, k):
    deviation = ((y - mean) * keep)[:, cells].double()
    left = deviation - deviation @ basis[:k].T @ basis[:k]
    out = torch.zeros_like(y)
    out[:, cells] = left.float()
    return out * keep


def high_pass(y, keep):
    weight = keep.float()[:, None]
    local = F.avg_pool3d((y * keep)[:, None], 3, 1, 1) / F.avg_pool3d(weight, 3, 1, 1).clamp_min(1e-6)
    return (y - local[:, 0]) * keep


def targets_for(bank, models):
    """{kind: per input view, (n, 16, 16, 16) the other view's target}."""
    keep = bank.keep[:, 0]
    out = {
        "raw": [bank.targets[1 - v][:, 0] for v in range(2)],
        "hp": [high_pass(bank.targets[1 - v][:, 0], keep) for v in range(2)],
    }
    for k in RANKS:
        out[f"pca{k}"] = [residual(bank.targets[1 - v][:, 0], keep, *models[1 - v], k) for v in range(2)]
    return out


def cell_units(model, h):
    codes = R.region_codes(model, h, C.CELLS[0])[:, : C.CONTENT]
    return codes.reshape(*codes.shape[:2], *C.CELLS)


def stored_guess(model, predictors, bank, device):
    """Per view: (n, 16, 16, 16) the checkpoint's own guess of the other view, from the cell units."""
    out = [[], []]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bank), BATCH):
            idx = list(range(start, min(start + BATCH, len(bank))))
            units = cell_units(model, model._encode(bank.batch(idx, device)[0], 2, None))
            for v in range(2):
                out[v].append(predictors[v](units[v * len(idx) : (v + 1) * len(idx)])[:, 0].cpu())
    return [torch.cat(part) for part in out]


def fit(model, bank, device, targets):
    """{input: {kind: per view, least-squares weights and bias}} over the first N_FIT subjects' kept cells."""
    grams = {name: [0.0, 0.0] for name in ("map", "code")}
    moments = {name: {kind: [0.0, 0.0] for kind in targets} for name in grams}
    model.eval()
    with torch.no_grad():
        for start in range(0, N_FIT, BATCH):
            idx = list(range(start, start + BATCH))
            x, _, _, keep = bank.batch(idx, device)
            h = model._encode(x, 2, None)
            inputs = {"map": h.cpu(), "code": cell_units(model, h).cpu()}
            keep = keep.cpu()
            for v in range(2):
                part = slice(v * BATCH, (v + 1) * BATCH)
                inside = keep[part][:, 0]
                for name, values in inputs.items():
                    X = values[part].movedim(1, -1)[inside].double().numpy()
                    X = np.hstack((X, np.ones((len(X), 1))))
                    grams[name][v] = grams[name][v] + X.T @ X
                    for kind, target in targets.items():
                        moments[name][kind][v] = moments[name][kind][v] + X.T @ target[v][idx][inside].double().numpy()
    weights = {}
    for name in grams:
        weights[name] = {}
        for kind in targets:
            weights[name][kind] = []
            for v in range(2):
                size = len(grams[name][v])
                ridge = 1e-6 * np.trace(grams[name][v]) / size * np.eye(size)
                weights[name][kind].append(np.linalg.solve(grams[name][v] + ridge, moments[name][kind][v]))
    return weights


def population(target, keep):
    return ((target * keep).sum(0) / keep.sum(0).clamp_min(1)).numpy()


def hard_cells(guess, true, brain, lesion):
    """How a loss on each subject's worst 1% of kept cells would land, per subject and per cell."""
    n = len(true)
    error = ((guess - true) ** 2).reshape(n, -1)
    inside = brain.reshape(n, -1)
    positive = lesion.reshape(n, -1) >= posthoc.LESION_FRACTION
    found, share = [], []
    for i in range(n):
        cells = np.flatnonzero(inside[i])
        worst = cells[np.argsort(error[i, cells])[-max(1, len(cells) // 100) :]]
        found.append(positive[i, worst].any())
        share.append(positive[i, worst].mean())
    return float(np.mean(found)), float(np.mean(share))


def main():
    bank_dir, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = pilot.user_args()
    train, val = pilot.Bank(bank_dir, "train"), pilot.Bank(bank_dir, "val")
    brain = val.keep[:, 0].numpy()
    lesion = np.load(L.CACHE)
    started = time.time()
    models = [population_basis(train, v) for v in range(2)]
    fixed_train, fixed_val = targets_for(train, models), targets_for(val, models)
    print(f"population model ({time.time() - started:.0f} s)", flush=True)
    keep_train = train.keep[:, 0]
    results = {}
    for state in states:
        name = os.path.splitext(os.path.basename(state))[0]
        started = time.time()
        model = C.build(state, args, device)
        saved = torch.load(state, map_location="cpu").get("predictors")
        fit_targets, val_targets = dict(fixed_train), dict(fixed_val)
        predictors = None
        if saved is not None:
            predictors = torch.nn.ModuleList(torch.nn.Conv3d(C.CONTENT, 1, 1) for _ in range(2)).to(device)
            predictors.load_state_dict(saved)
            guesses = {
                "train": stored_guess(model, predictors, train, device),
                "val": stored_guess(model, predictors, val, device),
            }
            fit_targets["boost"] = [(fixed_train["raw"][v] - guesses["train"][v]) * keep_train for v in range(2)]
            val_targets["boost"] = [(fixed_val["raw"][v] - guesses["val"][v]) * val.keep[:, 0] for v in range(2)]
        weights = fit(model, train, device, fit_targets)
        maps = C.cell_maps(model, val, device)
        row = {}
        on = lesion >= posthoc.LESION_FRACTION
        for kind in fit_targets:
            for v, view in enumerate(VIEWS):
                true = val_targets[kind][v].numpy()
                centre = population(fit_targets[kind][v], keep_train)
                energy = (true - centre) ** 2
                entry = {"share": float(energy[brain & on].sum() / energy[brain].sum())}
                deviation = (true - centre)[:, None]
                entry["target"] = C.probe_between(deviation, deviation, brain, lesion)
                for input_name, values in (("map", maps[view]), ("code", maps[f"code_{view}"][:, : C.CONTENT])):
                    w = weights[input_name][kind][v]
                    guess = values.movedim(1, -1).numpy() @ w[:-1].astype(np.float32) + w[-1]
                    deviation = (guess - centre)[:, None]
                    entry[input_name] = C.probe_between(deviation, deviation, brain, lesion)
                row[f"{kind} {view}"] = entry
        if saved is not None:
            for v, view in enumerate(VIEWS):
                row[f"hard cells {view}"] = hard_cells(
                    guesses["val"][v].numpy(), fixed_val["raw"][v].numpy(), brain, lesion
                )
        results[name] = row
        print(name, f"({time.time() - started:.0f} s)", flush=True)
        for key, entry in row.items():
            if isinstance(entry, dict):
                print(
                    f"  {key:<14} share {100 * entry['share']:5.1f}%   target {entry['target'][0]:.3f} / {100 * entry['target'][1]:4.1f}   map {entry['map'][0]:.3f} / {100 * entry['map'][1]:4.1f}   code {entry['code'][0]:.3f} / {100 * entry['code'][1]:4.1f}",
                    flush=True,
                )
            else:
                print(
                    f"  {key:<14} a lesion cell in the worst 1%: {100 * entry[0]:.0f}% of subjects; lesion share of those cells {100 * entry[1]:.0f}%",
                    flush=True,
                )
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
