"""Is the wide arm's lesion fall overfitting? The region and global losses and the cell probes on 400 train subjects
against the 400 val subjects, for the same checkpoints. The val probe rows must reproduce cellcode.json."""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402

L, R, T, pilot, posthoc = C.L, C.R, C.T, C.pilot, C.posthoc
BANK = sys.argv[1]
OUT = sys.argv[2]
N = 400
B = 32
HERE = "out/regional_pilot_20261028"
STATES = (
    ("gap_region16_double_sepadam", None),
    ("gap_region16_double_sepadam", 500),
    ("gap_region16_double_sepadam", 2000),
    ("wide64_sepadam", 500),
    ("wide64_sepadam", 1000),
    ("wide64_sepadam", 2000),
)


class Part:
    """The first N subjects of a bank."""

    def __init__(self, bank):
        self.bank, self.mask, self.keep = bank, bank.mask[:N], bank.keep[:N]

    def __len__(self):
        return N

    def batch(self, idx, device):
        return self.bank.batch(idx, device)


def losses(arm, model, part, args, device):
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    region, global_ = [], []
    with torch.no_grad():
        for start in range(0, N - B + 1, B):
            x, masks, _, _ = part.batch(list(range(start, start + B)), device)
            terms = R.objective(arm, model, x, args, sim, crit, masks)
            region.append(terms["patch"].item())
            global_.append(terms["global"].item())
    return float(np.mean(region)), float(np.mean(global_))


def probes(model, part, lesion, device):
    out = C.cell_maps(model, part, device)
    width = C.CONTENT if out["code_t1"].shape[1] == 18 else out["code_t1"].shape[1]
    keep = T.foreground_positions(part.mask[:, None], C.CELLS, C.FOREGROUND)
    brain = part.keep[:, 0].numpy()
    units = C.standardised(out["code_t1"][:, :width], out["code_flair"][:, :width], keep, True)[0]
    row = {}
    for v, view in enumerate(("t1", "flair")):
        h = out[view].numpy()
        centred = h - h.mean(0, keepdims=True)
        row[f"{view} map"] = C.probe_between(centred, centred, brain, lesion)
        row[f"{view} own"] = C.probe_between(units[v], units[v], brain, lesion)
        row[f"{view} transfer"] = C.probe_between(units[1 - v], units[v], brain, lesion)
    return row


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    splits = {
        "train": (Part(pilot.Bank(BANK, "train")), np.load(f"{HERE}/train_lesion_cells.npy")[:N]),
        "val": (Part(pilot.Bank(BANK, "val")), np.load(f"{HERE}/val_lesion_cells.npy")[:N]),
    }
    results = {}
    for arm, step in STATES:
        name = f"{arm} s{step or 0}"
        args = R.arm_args(arm)
        model = T.build_encoder(args)
        if step:
            model.load_state_dict(torch.load(f"{HERE}/{arm}_s{step}.pt", map_location="cpu")["model"])
        model.to(device).train()
        results[name] = {}
        for split, (part, lesion) in splits.items():
            region, global_ = losses(arm, model, part, args, device)
            row = {"region loss": region, "global loss": global_, **probes(model, part, lesion, device)}
            results[name][split] = row
            print(name, split, json.dumps({k: np.round(v, 3).tolist() for k, v in row.items()}), flush=True)
        json.dump(results, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
