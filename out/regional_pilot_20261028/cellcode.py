"""Is the lesion in the 16^3 cell codes the region term trains, or only in the backbone map under them?

lesion.py's cell probe reads the backbone map, before the head. This runs that probe (same cells, folds and metrics as
posthoc.cell_probe) on what the region term aligns: each cell's 14 content units after the shared head.

  h            the 64-channel map, as lesion.py reads it (its rows must come back exactly)
  h centred    the map minus each cell's mean over subjects
  random14     14 random orthonormal directions of h centred, DRAWS draws: what any 14 numbers of this map keep
  code         the 14 units
  position     the units through standardize_patch_code, over all val subjects on the kept cells
  double       the same with each subject's mean over kept cells removed too (--patch-center-mode double)
  position L2, double L2
               those with each cell's vector scaled to unit length: what the loss's cosine sees
  transfer     the probe fitted on one view's centred units and scored on the other view's held-out subjects, for each
               centring: a lesion signal the two views share survives, one only the fitted view holds does not

A head wider than the pilot's 18 outputs (wide64_sepadam) reads every output as its code, the region term's width, and
adds a "NAME first14" row with only the global term's units.

    python out/regional_pilot_20261028/cellcode.py BANK_DIR OUT_JSON init|init_linear|init_wide|CHECKPOINT ...

States already in OUT_JSON are skipped, so a stopped run can be restarted with the same command.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "out/regional_pilot_20261028")
import regional as R  # noqa: E402

import lesion as L  # noqa: E402

pilot, posthoc, T = L.pilot, L.posthoc, R.T
CELLS = (16, 16, 16)
FOREGROUND = 0.05  # --patch-foreground-thresh
DRAWS = 5
CONTENT = 14  # --content-channels: the units the global term reads


def build(state, args, device):
    """The model at a checkpoint, or at init (init_linear: the 1x1-conv readout; init_wide: a 64-output head)."""
    weights = None if state.startswith("init") else torch.load(state, map_location="cpu")["model"]
    args = L.checkpoint_args(args, weights)
    if state == "init_linear":
        args.conv_readout = "linear"
    elif state == "init_wide":
        args.latent_dim = 64
    model, _ = pilot.build(args, "base", device)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def cell_maps(model, bank, device, batch=32):
    """Per view: (n, 64, 16, 16, 16) backbone maps, as posthoc.maps stores them, and every head output of every cell,
    (n, outputs, 16, 16, 16)."""
    model.eval()
    out = {"t1": [], "flair": [], "code_t1": [], "code_flair": []}
    with torch.no_grad():
        for start in range(0, len(bank), batch):
            idx = list(range(start, min(start + batch, len(bank))))
            h = model._encode(bank.batch(idx, device)[0], 2, None)
            codes = R.region_codes(model, h, CELLS[0])
            codes = codes.reshape(len(h), codes.shape[1], *CELLS)
            b = len(idx)
            out["t1"].append(h[:b].cpu().half())
            out["flair"].append(h[b:].cpu().half())
            out["code_t1"].append(codes[:b].cpu())
            out["code_flair"].append(codes[b:].cpu())
    return {key: torch.cat(value).float() for key, value in out.items()}


def standardised(code_t1, code_flair, keep, double):
    """Both views' cell codes as the region term sees them, on the 16^3 grid (zero on dropped cells), and those scaled
    to unit length per cell."""
    hz = torch.stack((code_t1, code_flair)).flatten(3)[..., keep]  # views, subjects, units, kept cells
    hz = T.standardize_patch_code(hz, double=double)
    outputs = []
    for values in (hz, torch.nn.functional.normalize(hz, dim=2)):
        grid = torch.zeros(*values.shape[:3], np.prod(CELLS))
        grid[..., keep] = values
        outputs.append(grid.reshape(*values.shape[:3], *CELLS).numpy())
    return outputs


def probe_between(fit, score, brain, lesion):
    """posthoc.cell_probe with the probe fitted on one map and scored on another; fit is score reproduces it."""
    n = fit.shape[0]
    fit = fit.reshape(n, fit.shape[1], -1).transpose(0, 2, 1)  # subjects, cells, channels
    score = score.reshape(n, score.shape[1], -1).transpose(0, 2, 1)
    lesion = lesion.reshape(n, -1)
    brain = brain.reshape(n, -1) & (lesion >= posthoc.LESION_FRACTION).any(0)
    rng = np.random.default_rng(0)
    fitted, scored, labels, groups = [], [], [], []
    for i in range(n):
        pos = np.flatnonzero(brain[i] & (lesion[i] >= posthoc.LESION_FRACTION))
        neg = np.flatnonzero(brain[i] & (lesion[i] == 0))
        neg = rng.choice(neg, size=min(len(neg), 40), replace=False)
        for cells, label in ((pos, 1), (neg, 0)):
            fitted.append(fit[i, cells])
            scored.append(score[i, cells])
            labels += [label] * len(cells)
            groups += [i] * len(cells)
    X, Y, y, g = np.concatenate(fitted), np.concatenate(scored), np.array(labels), np.array(groups)
    scores, hits = np.zeros(len(y)), []
    for train, test in GroupKFold(5).split(X, y, g):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        probe.fit(X[train], y[train])
        scores[test] = probe.decision_function(Y[test])
        for i in np.unique(g[test]):
            inside = np.flatnonzero(brain[i])
            top = inside[np.argmax(probe.decision_function(score[i, inside]))]
            hits.append(bool(lesion[i, top] > 0))
    return float(roc_auc_score(y, scores)), float(np.mean(hits))


def random_directions(centred, draw):
    """Each cell's 64 channels on 14 random orthonormal directions."""
    basis = np.linalg.qr(np.random.default_rng(draw).standard_normal((centred.shape[1], 14)))[0]
    return np.moveaxis(np.moveaxis(centred, 1, -1) @ basis, -1, 1)


def score(out, keep, brain, lesion):
    """Every row of the module docstring for one state's maps."""
    row = {}

    def put(name, values):
        row[f"{name} AUROC"], row[f"{name} top cell"] = values

    centrings = {}
    for mode, double in (("position", False), ("double", True)):
        centrings[mode] = standardised(out["code_t1"], out["code_flair"], keep, double)
    for v, view in enumerate(("t1", "flair")):
        h = out[view].numpy()
        put(f"{view} h", probe_between(h, h, brain, lesion))
        centred = h - h.mean(0, keepdims=True)
        put(f"{view} h centred", probe_between(centred, centred, brain, lesion))
        draws = [probe_between(*(random_directions(centred, d),) * 2, brain, lesion) for d in range(DRAWS)]
        put(f"{view} random14", tuple(np.mean(draws, 0)))
        row[f"{view} random14 top cell sd"] = float(np.std([top for _, top in draws], ddof=1))
        code = out[f"code_{view}"].numpy()
        put(f"{view} code", probe_between(code, code, brain, lesion))
        for mode, (units, unit_length) in centrings.items():
            put(f"{view} {mode}", probe_between(units[v], units[v], brain, lesion))
            put(f"{view} {mode} L2", probe_between(unit_length[v], unit_length[v], brain, lesion))
            put(f"{view} {mode} transfer", probe_between(units[1 - v], units[v], brain, lesion))
    return row


def main():
    bank_dir, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args = pilot.user_args()
    bank = pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    lesion = np.load(L.CACHE)
    keep = T.foreground_positions(bank.mask[:, None], CELLS, FOREGROUND)
    print(f"{int(keep.sum())}/{np.prod(CELLS)} cells kept over the val split", flush=True)
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for state in states:
        name = state if state.startswith("init") else os.path.splitext(os.path.basename(state))[0]
        if name in results:
            continue
        started = time.time()
        out = cell_maps(build(state, args, device), bank, device)
        outputs = out["code_t1"].shape[1]
        widths = {"": CONTENT} if outputs == args.latent_dim else {"": outputs, " first14": CONTENT}
        for suffix, width in widths.items():
            codes = {key: out[key][:, :width] for key in ("code_t1", "code_flair")}
            results[name + suffix] = score({**out, **codes}, keep, brain, lesion)
        print(
            f"{name} ({time.time() - started:.0f} s)",
            json.dumps({k: round(v, 3) for k, v in results[name].items()}),
            flush=True,
        )
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
