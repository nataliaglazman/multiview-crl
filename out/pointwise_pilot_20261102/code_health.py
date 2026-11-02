"""What the cell code holds at a checkpoint, read on the template residual as well as on the raw rows.

The trainer's cell eff_rank reads raw rows, whose covariance the template dominates, so it can fall while the
subject-varying code stays wide. Over the xpred bank's 400 validation subjects and their brain cells, per view:

  rank raw        participation ratio of the rows' covariance: units in use, template included
  rank residual   the same after subtracting each cell's mean over subjects: units the subjects vary along
  subject share   share of the rows' variance left in that residual
  saturated       share of values outside [0.01, 0.99]
  unit r          mean over units of the T1-FLAIR correlation of the residuals: do the views agree unit by unit
  lesion cos      cosine between the two views' mean residual response at lesion cells: are they the same direction
  lesion size     length of that response in pooled residual SDs

    python out/pointwise_pilot_20261102/code_health.py BANK_DIR NAME=RUN_DIR[:CHECKPOINT] ...
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "out/pointwise_pilot_20261102")
sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402
import score as S  # noqa: E402

from eval.protocol.score_checkpoint import build_model, load_settings  # noqa: E402


def participation(rows):
    values = np.linalg.eigvalsh(np.cov(rows.T)).clip(min=0)
    return float(values.sum() ** 2 / np.square(values).sum())


def health(out, brain, lesion):
    n = len(brain)
    keep = brain.reshape(n, -1)
    touched = lesion.reshape(n, -1) >= C.posthoc.LESION_FRACTION
    row = {}
    residuals = {}
    for view in ("t1", "flair"):
        code = out[f"code_{view}"].reshape(n, out[f"code_{view}"].shape[1], -1)
        rows = code.transpose(0, 2, 1)[keep]
        residual = (code - code.mean(0, keepdims=True)).transpose(0, 2, 1)
        residuals[view] = residual
        kept = residual[keep]
        row[f"{view} rank raw"] = participation(rows)
        row[f"{view} rank residual"] = participation(kept)
        row[f"{view} subject share"] = float(kept.var(0).sum() / rows.var(0).sum())
        row[f"{view} saturated"] = float(((rows < 0.01) | (rows > 0.99)).mean())
    a, b = residuals["t1"][keep], residuals["flair"][keep]
    row["unit r"] = float(np.mean([np.corrcoef(a[:, u], b[:, u])[0, 1] for u in range(a.shape[1])]))
    sites = keep & touched
    response = {view: residuals[view][sites].mean(0) for view in residuals}
    x, y = response["t1"], response["flair"]
    row["lesion cos"] = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    for view, kept in (("t1", a), ("flair", b)):
        row[f"{view} lesion size"] = float(np.linalg.norm(response[view]) / np.sqrt(kept.var(0).mean()))
    return row


def main():
    bank_dir, states = sys.argv[1], sys.argv[2:]
    device = torch.device(os.environ.get("DEVICE", "cpu"))
    bank = C.pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    lesion = np.load(C.L.CACHE)
    results = {}
    for state in states:
        name, spec = state.split("=", 1)
        run_dir, checkpoint = (spec.split(":") + ["model.pt"])[:2]
        cfg = load_settings(run_dir)
        model = build_model(cfg, device, torch.load(os.path.join(run_dir, checkpoint), map_location="cpu"))
        results[name] = health(S.maps_of(model, bank, device, cfg["content_channels"]), brain, lesion)
        print(name, json.dumps({key: round(value, 3) for key, value in results[name].items()}), flush=True)


if __name__ == "__main__":
    main()
