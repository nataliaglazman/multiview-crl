"""Is the one burden lesion in the pointwise arms' cells? The regional pilot's probe on the same subjects.

Reads the xpred pilot's bank of 400 validation subjects, which has this recipe's generator, and scores each state with
the probe behind out/regional_pilot_20261028/cellcode.json, so these rows sit beside its region16 and sepadam rows:

  h            the 64-channel backbone map
  h centred    the map minus each cell's mean over subjects
  code         each cell's content units
  code centred those minus each cell's mean over subjects
  transfer     the probe fitted on the other view's centred units and scored on this view's held-out subjects

AUROC and top cell as in posthoc.cell_probe: only cells where some subject has a lesion take part, and top cell is the
share of subjects whose best-scoring brain cell touches the lesion. Burden is a 5-fold ridge from the mean cell code
and from each unit's largest value over the subject's brain cells.

    python out/pointwise_pilot_20261102/score.py BANK_DIR OUT_JSON NAME=RUN_DIR[:CHECKPOINT] ...
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402

from eval.protocol.score_checkpoint import build_model, load_settings  # noqa: E402

GRID = (16, 16, 16)
PILOT_ROWS = ("init", "region16_s2000", "gap_region16_double_sepadam_s2000")


def maps_of(model, bank, device, content, batch=32):
    """Per view: (n, 64, 16, 16, 16) backbone maps and (n, content, 16, 16, 16) cell codes."""
    out = {"t1": [], "flair": [], "code_t1": [], "code_flair": []}
    with torch.no_grad():
        for start in range(0, len(bank), batch):
            idx = list(range(start, min(start + batch, len(bank))))
            h = model._encode(bank.batch(idx, device)[0], 2, None)
            codes = model.to_encoding(h)[:, :content]
            b = len(idx)
            out["t1"].append(h[:b].cpu().half())
            out["flair"].append(h[b:].cpu().half())
            out["code_t1"].append(codes[:b].cpu())
            out["code_flair"].append(codes[b:].cpu())
    return {key: torch.cat(value).float().numpy() for key, value in out.items()}


def score(out, bank, brain, lesion):
    burden = bank.zc[:, 2]
    row = {}

    def put(name, values):
        row[f"{name} AUROC"], row[f"{name} top cell"] = values

    centred = {key: out[key] - out[key].mean(0, keepdims=True) for key in out}
    for view, other in (("t1", "flair"), ("flair", "t1")):
        h, code = out[view], out[f"code_{view}"]
        put(f"{view} h", C.probe_between(h, h, brain, lesion))
        put(f"{view} h centred", C.probe_between(centred[view], centred[view], brain, lesion))
        put(f"{view} code", C.probe_between(code, code, brain, lesion))
        mine = centred[f"code_{view}"]
        put(f"{view} code centred", C.probe_between(mine, mine, brain, lesion))
        put(f"{view} transfer", C.probe_between(centred[f"code_{other}"], mine, brain, lesion))
        flat = code.reshape(len(code), code.shape[1], -1)
        masked = np.where(brain.reshape(len(code), 1, -1), flat, -np.inf)
        row[f"{view} burden R2 mean"] = C.pilot.ridge_r2(code.mean((2, 3, 4)).astype(np.float64), burden)
        row[f"{view} burden R2 max"] = C.pilot.ridge_r2(masked.max(2).astype(np.float64), burden)
    return row


def main():
    bank_dir, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device(os.environ.get("DEVICE", "cpu"))
    bank = C.pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    lesion = np.load(C.L.CACHE)
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for state in states:
        name, spec = state.split("=", 1)
        run_dir, checkpoint = (spec.split(":") + ["model.pt"])[:2]
        started = time.time()
        cfg = load_settings(run_dir)
        model = build_model(cfg, device, torch.load(os.path.join(run_dir, checkpoint), map_location="cpu"))
        results[name] = score(maps_of(model, bank, device, cfg["content_channels"]), bank, brain, lesion)
        print(f"{name} ({time.time() - started:.0f} s)", flush=True)
        json.dump(results, open(out_json, "w"), indent=1)
    stored = json.load(open("out/regional_pilot_20261028/cellcode.json"))
    rows = {**{key: stored[key] for key in PILOT_ROWS}, **results}
    columns = (
        "h centred AUROC",
        "h centred top cell",
        "code AUROC",
        "code top cell",
        "transfer AUROC",
        "transfer top cell",
    )
    for view in ("t1", "flair"):
        print(
            f"\n{view:<40s}"
            + "".join(f"{c.replace('AUROC', 'AUC').replace(' top cell', ' top'):>20s}" for c in columns)
        )
        for name, row in rows.items():
            cells = []
            for column in columns:
                key = f"{view} {column}"
                if column.startswith("transfer"):
                    key = f"{view} position {column}" if key not in row else key
                cells.append(f"{row[key]:>20.3f}" if key in row else f"{'-':>20s}")
            print(f"{name:<40s}" + "".join(cells))


if __name__ == "__main__":
    main()
