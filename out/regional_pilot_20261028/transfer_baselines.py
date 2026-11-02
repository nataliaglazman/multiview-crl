"""Is the cell code's transfer special, or would any 14 numbers of the map do the same? Baselines for cellcode.py's rows.

  map                 the centred 64-channel map, as cellcode.py's "h centred" (its rows must come back exactly)
  map transfer        the probe fitted on one view's centred map, scored on the other view's held-out subjects
  random14 transfer   the same on 14 random orthonormal directions, one basis for both views, DRAWS draws
  cca14, cca14 transfer
                      the 14 most correlated pairs of directions of the two centred maps, fitted with no labels on the
                      first N_FIT train subjects' kept cells: the 14 numbers these maps agree on most

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/transfer_baselines.py BANK_DIR OUT_JSON init|CHECKPOINT ...

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

L, pilot = C.L, C.pilot
N_FIT = 400
VIEWS = ("t1", "flair")


class First:
    """The first n subjects of a bank, for cell_maps."""

    def __init__(self, bank, n):
        self.bank, self.n = bank, n

    def __len__(self):
        return self.n

    def batch(self, idx, device):
        return self.bank.batch(idx, device)


def whitener(values):
    cov = values.T @ values / len(values)
    eigenvalues, vectors = np.linalg.eigh(cov + 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov)))
    return vectors / np.sqrt(eigenvalues)


def canonical(x, y):
    """(64, 14) directions per view and the 14 correlations, from each view's (cells, 64) centred values."""
    wx, wy = whitener(x), whitener(y)
    u, s, vt = np.linalg.svd(wx.T @ (x.T @ y / len(x)) @ wy)
    return wx @ u[:, : C.CONTENT], wy @ vt.T[:, : C.CONTENT], s[: C.CONTENT]


def project(centred, basis):
    return np.moveaxis(np.moveaxis(centred, 1, -1) @ basis, -1, 1)


def fit_cca(model, train, device):
    maps = C.cell_maps(model, First(train, N_FIT), device)
    inside = train.keep[:N_FIT, 0]
    cells = []
    for view in VIEWS:
        centred = maps[view] - maps[view].mean(0, keepdim=True)
        cells.append(centred.movedim(1, -1)[inside].double().numpy())
    return canonical(*cells)


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
        bases_t1, bases_flair, correlations = fit_cca(model, train, device)
        out = C.cell_maps(model, val, device)
        centred = [out[view].numpy() - out[view].numpy().mean(0, keepdims=True) for view in VIEWS]
        agreed = [project(centred[0], bases_t1), project(centred[1], bases_flair)]
        row = {"cca14 correlations": [round(float(c), 3) for c in correlations]}

        def put(key, values):
            row[f"{key} AUROC"], row[f"{key} top cell"] = values

        for v, view in enumerate(VIEWS):
            other = 1 - v
            put(f"{view} map", C.probe_between(centred[v], centred[v], brain, lesion))
            put(f"{view} map transfer", C.probe_between(centred[other], centred[v], brain, lesion))
            draws = [
                C.probe_between(
                    C.random_directions(centred[other], d), C.random_directions(centred[v], d), brain, lesion
                )
                for d in range(C.DRAWS)
            ]
            put(f"{view} random14 transfer", tuple(np.mean(draws, 0)))
            row[f"{view} random14 transfer top cell sd"] = float(np.std([top for _, top in draws], ddof=1))
            put(f"{view} cca14", C.probe_between(agreed[v], agreed[v], brain, lesion))
            put(f"{view} cca14 transfer", C.probe_between(agreed[other], agreed[v], brain, lesion))
        results[name] = row
        print(name, f"({time.time() - started:.0f} s)", flush=True)
        for key in ("map", "map transfer", "random14 transfer", "cca14", "cca14 transfer"):
            print(
                f"  {key:<20}"
                + "   ".join(
                    f"{view} {row[f'{view} {key} AUROC']:.3f} / {100 * row[f'{view} {key} top cell']:.1f}"
                    for view in VIEWS
                ),
                flush=True,
            )
        print("  correlations", row["cca14 correlations"], flush=True)
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
