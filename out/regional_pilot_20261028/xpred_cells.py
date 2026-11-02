"""Does cross-view prediction hold the lesion where the InfoNCE arms do not? Read what a prediction objective trains.

cellcode.py runs unchanged on the xpred pilot's checkpoints and gives their map, cell-code and transfer rows. This adds
the one number per cell the prediction term trains: the other view's z-scored cell image, guessed from this view's map
by a 1x1x1 conv. Each row reads that guess minus the cell's population mean (the train subjects' mean of the target).

  stored      the checkpoint's own predictor, on the map, or on the 14 cell units in a predcode arm
  refit       a 1x1x1 conv on the map, fitted here by least squares on the first N_FIT train subjects' kept cells. No
              labels, and it needs no prediction term, so every arm gets the same row. On an xpred checkpoint it should
              land near stored
  refit code  the same fit on the 14 cell units: what a linear read of each arm's code can guess of the other view
  actual      the other view's real cell value, the same way: what a perfect guess would read

Each row gives the cell probe's AUROC and top cell (cellcode.probe_between on the one number), and at lesion cells the
slope and correlation of the guess against the target's real deviation from the population mean (the xpred pilot's
slope table: 0.09 / 0.25 at xpred.pt, 0.37 / 0.66 at xpred_w5_s4000.pt). r2 is over all kept cells of the val subjects, of
the target and of its deviation from the population mean.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/xpred_cells.py BANK_DIR OUT_JSON init|CHECKPOINT ...

States already in OUT_JSON are skipped.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402

L, R, pilot, posthoc = C.L, C.R, C.pilot, C.posthoc
N_FIT = 800
BATCH = 32
CHANNELS = 64
VIEWS = ("t1", "flair")


def population(bank, target):
    """Each cell's mean of one target over the subjects that keep it."""
    keep = bank.keep[:, 0].float()
    return ((bank.targets[target][:, 0] * keep).sum(0) / keep.sum(0).clamp_min(1)).numpy()


def cell_units(model, h):
    """(n, 14, 16, 16, 16): every cell's content units, a predcode arm's predictor input."""
    codes = R.region_codes(model, h, C.CELLS[0])[:, : C.CONTENT]
    return codes.reshape(*codes.shape[:2], *C.CELLS)


def fit_predictors(model, bank, device, read):
    """Per view: least-squares weights and bias of a 1x1x1 conv from read(map) to the other view's cell target."""
    grams, moments = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, N_FIT, BATCH):
            x, _, targets, keep = bank.batch(list(range(start, start + BATCH)), device)
            inputs = read(model._encode(x, 2, None)).cpu()
            targets, keep = targets.cpu(), keep.cpu()
            rows, outcomes = [], []
            for v in range(2):
                part = slice(v * BATCH, (v + 1) * BATCH)
                inside = keep[part][:, 0]
                X = inputs[part].movedim(1, -1)[inside].double().numpy()
                rows.append(np.hstack((X, np.ones((len(X), 1)))))
                outcomes.append(targets[part][:, 0][inside].double().numpy())
            grams = grams + np.stack([X.T @ X for X in rows])
            moments = moments + np.stack([X.T @ y for X, y in zip(rows, outcomes)])
    size = grams.shape[1]
    ridge = 1e-6 * np.trace(grams, axis1=1, axis2=2)[:, None, None] / size * np.eye(size)
    return list(np.linalg.solve(grams + ridge, moments[..., None])[..., 0])


def stored_predictors(state):
    """Per view: the checkpoint's predictor as weights and bias, or None if it was not trained with one."""
    saved = torch.load(state, map_location="cpu").get("predictors")
    if saved is None:
        return None
    return [np.r_[saved[f"{v}.weight"].flatten().numpy(), saved[f"{v}.bias"].numpy()] for v in range(2)]


def guess(weights, maps, view):
    """(n, 16, 16, 16) one number per cell from the input these weights fit: the 64-channel map or the 14 cell units."""
    inputs = maps[view] if len(weights) == CHANNELS + 1 else maps[f"code_{view}"][:, : C.CONTENT]
    return inputs.movedim(1, -1).numpy() @ weights[:-1].astype(np.float32) + weights[-1]


def read(pred, true, centre, brain, lesion):
    deviation, real = pred - centre, true - centre
    auroc, top = C.probe_between(deviation[:, None], deviation[:, None], brain, lesion)
    on = brain & (lesion >= posthoc.LESION_FRACTION)
    error = ((pred - true)[brain] ** 2).sum()
    return {
        "AUROC": auroc,
        "top cell": top,
        "slope": float(np.polyfit(real[on], deviation[on], 1)[0]),
        "corr": float(np.corrcoef(real[on], deviation[on])[0, 1]),
        "r2": float(1 - error / ((true[brain] - true[brain].mean()) ** 2).sum()),
        "r2 deviation": float(1 - error / (real[brain] ** 2).sum()),
    }


def main():
    bank_dir, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = pilot.user_args()
    train, val = pilot.Bank(bank_dir, "train"), pilot.Bank(bank_dir, "val")
    brain = val.keep[:, 0].numpy()
    lesion = np.load(L.CACHE)
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for state in states:
        name = state if state.startswith("init") else os.path.splitext(os.path.basename(state))[0]
        if name in results:
            continue
        started = time.time()
        model = C.build(state, args, device)
        predictors = {
            "refit": fit_predictors(model, train, device, lambda h: h),
            "refit code": fit_predictors(model, train, device, lambda h: cell_units(model, h)),
        }
        stored = None if state.startswith("init") else stored_predictors(state)
        if stored is not None:
            predictors["stored"] = stored
        maps = C.cell_maps(model, val, device)
        row = {}
        for v, view in enumerate(VIEWS):
            target = 1 - v  # the T1 map guesses FLAIR, the FLAIR map guesses T1
            true = val.targets[target][:, 0].numpy()
            centre = population(train, target)
            for kind, weights in predictors.items():
                row[f"{view} {kind}"] = read(guess(weights[v], maps, view), true, centre, brain, lesion)
            row[f"{VIEWS[target]} actual"] = read(true, true, centre, brain, lesion)
        results[name] = row
        print(name, f"({time.time() - started:.0f} s)", json.dumps(row, default=lambda v: round(v, 3)), flush=True)
        for key, values in row.items():
            print(f"  {key:<16}" + " ".join(f"{k} {v:.3f}" for k, v in values.items()), flush=True)
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
