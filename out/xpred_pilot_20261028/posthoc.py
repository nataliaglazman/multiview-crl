"""Where does each pilot arm keep the lesion? Readouts of the T1 and FLAIR backbone maps on the 400 val subjects.

States: the shared init (model seed 42) and each arm's final checkpoint (pilot.py's {arm}.pt).
  burden R^2   5-fold ridge of the burden latent from each view's 14 content units (code), 64-channel backbone GAP,
               or per-channel max over brain cells (max)
  cell AUROC   a linear probe on single map cells (64 channels) separating lesion cells (>= 10% lesion) from lesion-free
               cells, 5-fold grouped by subject, inside the placement region (cells where some subject has a lesion),
               so a white-matter prior can't score; no tissue labels
  top cell     share of subjects whose highest-scoring region cell under that probe (held-out fold) touches the lesion,
               against its chance level (lesion cells / region cells, averaged over subjects)

    python out/xpred_pilot_20261028/posthoc.py BANK_DIR OUT_DIR ARM [ARM ...]
"""

import json
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "out/xpred_pilot_20261028")
import pilot  # noqa: E402

import training.main_conv_synthetic as T  # noqa: E402

LESION_FRACTION = 0.1


def lesion_cells(args, n, mode="val"):
    """(n, 16, 16, 16) lesion fraction of every cell, from the generator's own lesion load."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = T.make_dataset(args, mode, n)
        inner = ds._inner
        cells = []
        for i in range(n):
            d = inner._draw_pseudo_mri(inner.sample_seed_for(i))
            with torch.no_grad():
                _, load = inner.renderer.render_structure(
                    d["z_content"], d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
                )
            cells.append(F.avg_pool3d(load[None, None].float(), pilot.CELL)[0, 0])
    return torch.stack(cells).numpy()


def maps(model, bank, device, batch=32):
    """Per view: (n, 64, 16, 16, 16) backbone maps and (n, 14) content codes."""
    model.eval()
    out = {"t1": [], "flair": [], "code_t1": [], "code_flair": []}
    with torch.no_grad():
        for start in range(0, len(bank), batch):
            idx = list(range(start, min(start + batch, len(bank))))
            x, _, _, _ = bank.batch(idx, device)
            h = model._encode(x, 2, None)
            code = model._global_code(h, x, 2)[:, model.content_units()]
            b = len(idx)
            out["t1"].append(h[:b].cpu().half())
            out["flair"].append(h[b:].cpu().half())
            out["code_t1"].append(code[:b].cpu())
            out["code_flair"].append(code[b:].cpu())
    return {key: torch.cat(value).float().numpy() for key, value in out.items()}


def cell_probe(h, brain, lesion):
    """Grouped-CV linear probe on single cells: AUROC, top-cell hit rate and that rate's chance level.

    Only cells where some subject has a lesion take part (the placement region), so "this is deep white
    matter" can't score: a negative is a lesion-free subject at a cell that holds lesions in others.
    """
    n = h.shape[0]
    feats = h.reshape(n, h.shape[1], -1).transpose(0, 2, 1)  # subjects, cells, channels
    lesion = lesion.reshape(n, -1)
    brain = brain.reshape(n, -1) & (lesion >= LESION_FRACTION).any(0)
    rng = np.random.default_rng(0)
    rows, labels, groups = [], [], []
    for i in range(n):
        pos = np.flatnonzero(brain[i] & (lesion[i] >= LESION_FRACTION))
        neg = np.flatnonzero(brain[i] & (lesion[i] == 0))
        neg = rng.choice(neg, size=min(len(neg), 40), replace=False)
        for cells, label in ((pos, 1), (neg, 0)):
            rows.append(feats[i, cells])
            labels += [label] * len(cells)
            groups += [i] * len(cells)
    X, y, g = np.concatenate(rows), np.array(labels), np.array(groups)
    scores, hits, subjects = np.zeros(len(y)), [], []
    for train, test in GroupKFold(5).split(X, y, g):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        probe.fit(X[train], y[train])
        scores[test] = probe.decision_function(X[test])
        for i in np.unique(g[test]):
            inside = np.flatnonzero(brain[i])
            top = inside[np.argmax(probe.decision_function(feats[i, inside]))]
            hits.append(bool(lesion[i, top] > 0))
            subjects.append(i)
    chance = float(np.mean([(brain[i] & (lesion[i] > 0)).sum() / max(brain[i].sum(), 1) for i in range(n)]))
    return float(roc_auc_score(y, scores)), float(np.mean(hits)), chance


def main():
    bank_dir, out_dir, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = pilot.user_args()
    bank = pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    lesion = lesion_cells(args, len(bank))
    burden = bank.zc[:, 2]
    results = {}
    for state in ["init", *arms]:
        model, predictors = pilot.build(args, "xpred" if state == "xpred" else "base", device)
        if state != "init":
            saved = torch.load(f"{out_dir}/{state}.pt", map_location="cpu")
            model.load_state_dict(saved["model"])
        out = maps(model, bank, device)
        row = {}
        for view in ("t1", "flair"):
            h = out[view]
            gap = h.mean((2, 3, 4))
            flat = h.reshape(len(h), h.shape[1], -1)
            masked = np.where(brain.reshape(len(h), 1, -1), flat, -np.inf)
            row[f"{view} burden R2 code"] = pilot.ridge_r2(out[f"code_{view}"].astype(np.float64), burden)
            row[f"{view} burden R2 GAP"] = pilot.ridge_r2(gap.astype(np.float64), burden)
            row[f"{view} burden R2 max"] = pilot.ridge_r2(masked.max(2).astype(np.float64), burden)
            row[f"{view} cell AUROC"], row[f"{view} top cell"], row["top cell chance"] = cell_probe(h, brain, lesion)
        results[state] = row
        print(state, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
    json.dump(results, open(f"{out_dir}/posthoc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
