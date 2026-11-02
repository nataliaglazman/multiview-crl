"""Is the cross-view prediction reaching the lesion, and is it still pushing on it? (val subjects, one checkpoint)

  predicted lesion    at lesion cells (>= 10% lesion) and inside the placement region elsewhere: the T1 map's
                      prediction of FLAIR minus that cell's population mean (train bank), against the same for the
                      true FLAIR. slope = least-squares slope of predicted on true deviation at lesion cells
                      (1 = the T1 encoder predicts the lesion's full brightening, 0 = not at all)
  push on lesion      share of a term's squared gradient w.r.t. the T1 map that lands on lesion cells, divided by
                      their share of brain cells (1 = no focus)

    python out/xpred_pilot_20261028/focus.py BANK_DIR CHECKPOINT [N_SUBJECTS]
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "out/xpred_pilot_20261028")
import pilot  # noqa: E402
import posthoc  # noqa: E402

import training.main_conv_synthetic as T  # noqa: E402


def main():
    bank_dir, checkpoint = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 192
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = pilot.user_args()
    train, val = pilot.Bank(bank_dir, "train"), pilot.Bank(bank_dir, "val")
    keep_train = train.keep[:, 0].float()
    population = (train.targets[1][:, 0] * keep_train).sum(0) / keep_train.sum(0).clamp_min(1)  # FLAIR, per cell
    lesion = posthoc.lesion_cells(args, len(val))[:n]
    saved = torch.load(checkpoint, map_location="cpu")
    has_predictors = saved.get("predictors") is not None
    model, predictors = pilot.build(args, "xpred" if has_predictors else "base", device)
    model.load_state_dict(saved["model"])
    if has_predictors:
        predictors.load_state_dict(saved["predictors"])
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    pred_dev, true_dev, region_pred = [], [], []
    push = {name: [0.0, 0.0] for name in (("global", "patch", "xpred") if has_predictors else ("global", "patch"))}
    region = (lesion >= posthoc.LESION_FRACTION).any(0)
    for start in range(0, n, 32):
        idx = list(range(start, min(start + 32, n)))
        b = len(idx)
        x, masks, targets, keep = val.batch(idx, device)
        _, _, terms = T.training_objective(model, x, args, sim, crit, masks)
        h = model._last_map
        xpred = pilot.cross_prediction(model, predictors, targets, keep) if has_predictors else None
        les = torch.from_numpy(lesion[start : start + b] >= posthoc.LESION_FRACTION).to(device)
        brain = keep[:b, 0]
        for name, term in (("global", terms["global"]), ("patch", terms["patch"]), ("xpred", xpred))[: len(push)]:
            g = torch.autograd.grad(term, h, retain_graph=True)[0][:b].square().sum(1)
            push[name][0] += float(g[les & brain].sum() / g[brain].sum()) * b
            push[name][1] += float((les & brain).sum() / brain.sum()) * b
        if not has_predictors:
            continue
        with torch.no_grad():
            pred = predictors[0](h[:b])[:, 0].cpu()
        truth = targets[:b, 0].cpu()
        pop = population.unsqueeze(0)
        les_cpu, brain_cpu = les.cpu(), brain.cpu()
        pred_dev.append((pred - pop)[les_cpu & brain_cpu])
        true_dev.append((truth - pop)[les_cpu & brain_cpu])
        elsewhere = (
            brain_cpu
            & ~les_cpu
            & torch.from_numpy(region).unsqueeze(0)
            & torch.from_numpy(lesion[start : start + b] == 0)
        )
        region_pred.append((pred - pop)[elsewhere])
    print(f"{checkpoint}: {n} subjects")
    if has_predictors:
        p, t, r = (torch.cat(v).numpy() for v in (pred_dev, true_dev, region_pred))
        slope = float((p * t).sum() / (t * t).sum())
        print(
            f"  {len(p)} lesion cells; true FLAIR deviation at lesion cells {t.mean():+.2f}; predicted from T1 {p.mean():+.2f} "
            f"(slope {slope:.2f}, corr {np.corrcoef(p, t)[0, 1]:.2f}); predicted elsewhere in the region {r.mean():+.2f}"
        )
    for name, (share, cells) in push.items():
        print(f"  push on lesion cells, {name}: {share / cells:.2f}x their share of brain cells")


if __name__ == "__main__":
    main()
