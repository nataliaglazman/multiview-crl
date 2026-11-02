"""Is the one burden lesion in each regional arm's 16^3 backbone map? The xpred pilot's posthoc readouts, per checkpoint.

  burden R^2   5-fold ridge of the burden latent from each view's 14 content units (code), 64-channel backbone GAP,
               or per-channel max over brain cells (max)
  cell AUROC   one linear probe shared by every map cell, separating lesion cells (>= 10% lesion) from lesion-free cells
               of other subjects at the same placement-region cells; 5-fold grouped by subject
  top cell     share of subjects whose highest-scoring brain cell under that probe touches the lesion (chance ~0.03)
All from out/xpred_pilot_20261028/posthoc.py, so the numbers sit next to its init/base/xpred rows. Every checkpoint has
the same architecture as that pilot's model, whatever its training grid, except the lin_* arms' 1x1-conv readout, which
is read off the checkpoint. Their backbone starts from the same weights, so the init row is their cell-probe floor too.

    python out/regional_pilot_20261028/lesion.py BANK_DIR OUT_JSON init|CHECKPOINT ...
"""

import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "out/xpred_pilot_20261028")
import pilot  # noqa: E402
import posthoc  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_lesion_cells.npy")


def checkpoint_args(args, weights):
    """The pilot's arguments as a checkpoint's model was built: the 1x1-conv readout (its to_encoding is one Conv3d),
    or its head's own width."""
    args = copy.copy(args)
    if weights is not None and "to_encoding.weight" in weights:
        args.conv_readout = "linear"
    elif weights is not None:
        args.latent_dim = weights["to_encoding.2.weight"].shape[0]
    return args


def main():
    bank_dir, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args = pilot.user_args()
    bank = pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    if not os.path.exists(CACHE):
        np.save(CACHE, posthoc.lesion_cells(args, len(bank)))
    lesion = np.load(CACHE)
    burden = bank.zc[:, 2]
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for state in states:
        weights = None if state == "init" else torch.load(state, map_location="cpu")["model"]
        model, _ = pilot.build(checkpoint_args(args, weights), "base", device)
        if weights is not None:
            model.load_state_dict(weights)
        out = posthoc.maps(model, bank, device)
        row = {}
        for view in ("t1", "flair"):
            h = out[view]
            flat = h.reshape(len(h), h.shape[1], -1)
            masked = np.where(brain.reshape(len(h), 1, -1), flat, -np.inf)
            row[f"{view} burden R2 code"] = pilot.ridge_r2(out[f"code_{view}"].astype(np.float64), burden)
            row[f"{view} burden R2 GAP"] = pilot.ridge_r2(h.mean((2, 3, 4)).astype(np.float64), burden)
            row[f"{view} burden R2 max"] = pilot.ridge_r2(masked.max(2).astype(np.float64), burden)
            row[f"{view} cell AUROC"], row[f"{view} top cell"], row["top cell chance"] = posthoc.cell_probe(
                h, brain, lesion
            )
        name = "init" if state == "init" else os.path.splitext(os.path.basename(state))[0]
        results[name] = row
        print(name, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
