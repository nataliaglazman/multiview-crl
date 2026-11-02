"""Is burden in the feature map but lost by whole-map pooling? Burden readouts from one checkpoint's maps (val, 400).

  GAP              ridge on the 64-channel average over the whole map (what the code is built from)
  detect, count    a linear probe first scores every brain cell for "lesion here" (trained on lesion-cell labels of
                   other subjects, 5-fold grouped by subject); then per subject: sum of the probe's probabilities
                   over brain cells, its max and its sum above zero -> ridge. A diagnostic upper bound: it uses labels
  own prediction   label-free, prediction checkpoints only: the predictor's guess of the OTHER view at every cell minus
                   that cell's population mean (train bank), signed so the lesion is positive; per subject: sum of the
                   positive part over brain cells, sum above 0.25, max -> ridge

    python out/xpred_pilot_20261028/readout.py BANK_DIR CHECKPOINT
"""

import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "out/xpred_pilot_20261028")
import pilot  # noqa: E402
import posthoc  # noqa: E402


def detect_and_count(h, brain, lesion):
    n = h.shape[0]
    feats = h.reshape(n, h.shape[1], -1).transpose(0, 2, 1)
    brain, lesion = brain.reshape(n, -1), lesion.reshape(n, -1)
    rng = np.random.default_rng(0)
    out = np.zeros((n, 3))
    subjects = np.arange(n)
    for train, test in GroupKFold(5).split(subjects, groups=subjects):
        rows, labels = [], []
        for i in train:
            pos = np.flatnonzero(brain[i] & (lesion[i] >= posthoc.LESION_FRACTION))
            neg = np.flatnonzero(brain[i] & (lesion[i] == 0))
            neg = rng.choice(neg, size=min(len(neg), 60), replace=False)
            rows += [feats[i, pos], feats[i, neg]]
            labels += [1] * len(pos) + [0] * len(neg)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        probe.fit(np.concatenate(rows), np.array(labels))
        for i in test:
            score = probe.decision_function(feats[i, brain[i]])
            out[i] = [(1 / (1 + np.exp(-score))).sum(), score.max(), np.clip(score, 0, None).sum()]
    return out


def main():
    bank_dir, checkpoint = sys.argv[1], sys.argv[2]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = pilot.user_args()
    train, val = pilot.Bank(bank_dir, "train"), pilot.Bank(bank_dir, "val")
    saved = torch.load(checkpoint, map_location="cpu")
    has_predictors = saved.get("predictors") is not None
    model, predictors = pilot.build(args, "xpred" if has_predictors else "base", device)
    model.load_state_dict(saved["model"])
    out = posthoc.maps(model, val, device)
    brain = val.keep[:, 0].numpy()
    lesion = posthoc.lesion_cells(args, len(val))
    burden = val.zc[:, 2]
    print(checkpoint)
    for view in ("t1", "flair"):
        h = out[view]
        row = {"GAP": pilot.ridge_r2(h.mean((2, 3, 4)).astype(np.float64), burden)}
        row["detect, count"] = pilot.ridge_r2(detect_and_count(h, brain, lesion), burden)
        if has_predictors:
            predictors.load_state_dict(saved["predictors"])
            other = 1 if view == "t1" else 0  # the T1 map predicts FLAIR, the FLAIR map predicts T1
            keep_train = train.keep[:, 0].float()
            population = ((train.targets[other][:, 0] * keep_train).sum(0) / keep_train.sum(0).clamp_min(1)).numpy()
            head = predictors[0 if view == "t1" else 1].cpu()
            with torch.no_grad():
                pred = head(torch.from_numpy(h).float())[:, 0].numpy()
            dev = (pred - population) * (1 if view == "t1" else -1) * brain  # the lesion brightens FLAIR, darkens T1
            feats = np.stack(
                [np.clip(dev, 0, None).sum((1, 2, 3)), np.clip(dev - 0.25, 0, None).sum((1, 2, 3)), dev.max((1, 2, 3))],
                1,
            )
            row["own prediction (label-free)"] = pilot.ridge_r2(feats.astype(np.float64), burden)
        print(f"  {view:>5}: " + "   ".join(f"{k} {v:+.2f}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
