"""Checks for region16_oracle before its long run, and how much the filtered negatives weigh in the softmax.

  labels    builds train_lesion_cells.npy if missing, and shows it lines up with the bank: lesion load against the burden
            latent, and FLAIR at lesion cells against the same cells of other subjects
  overlap   the share of lesion anchors with a lesioned negative at that cell, on the train labels at B=32
  formula   filtered_region_loss on small random codes against InfoNCE written out row by row
  parity    region16 and region16_oracle with no lesions at all, three training steps from the same init: must be equal
  weight    at init and at region16_s2000: of a lesion anchor's softmax over negatives, the share held by lesioned
            negatives, against their share of the count (ratio > 1: they are harder than the average negative)

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/oracle_check.py BANK_DIR
"""

import os
import sys
import types

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "out/regional_pilot_20261028")
import regional as R  # noqa: E402

import lesion as L  # noqa: E402

pilot, posthoc, T = L.pilot, L.posthoc, R.T
HERE = os.path.dirname(os.path.abspath(__file__))
B = 32
WEIGHT_BATCHES = 12


def labels(bank):
    path = f"{HERE}/train_lesion_cells.npy"
    if not os.path.exists(path):
        np.save(path, posthoc.lesion_cells(pilot.user_args(), len(bank), "train"))
    cells = np.load(path)
    print(f"labels: {cells.shape}, burden rho {spearmanr(cells.sum((1, 2, 3)), bank.zc[:, 2]).correlation:.4f}")
    flair = torch.nn.functional.avg_pool3d(bank.flair[:, None].float(), pilot.CELL)[:, 0].numpy()
    deviation = flair - flair.mean(0, keepdims=True)
    on = cells >= 0.1
    print(f"  FLAIR at lesion cells vs the cell's mean: own subject {deviation[on].mean():+.3f}, ", end="")
    print(f"next subject's lesion cells {deviation[np.roll(on, 1, axis=0)].mean():+.3f}")
    return torch.from_numpy(cells)


def overlap(cells):
    on = (cells > 0).flatten(1).numpy()
    rng = np.random.default_rng(0)
    hit = total = negatives = 0
    for _ in range(4000):
        count = on[rng.choice(len(on), B, replace=False)].sum(0)
        count = count[count > 0]
        hit += (count * (count > 1)).sum()
        total += count.sum()
        negatives += (count * (count - 1)).sum()
    print(
        f"overlap: {hit / total:.3f} of lesion anchors, {negatives / total:.2f} lesioned negatives each (val: 0.66, 1.30)"
    )


def formula():
    torch.manual_seed(0)
    b, c, tau = 6, 14, 0.1
    args = types.SimpleNamespace(train_patch_grid=(2, 2, 2), patch_foreground_thresh=0.05, content_channels=c, tau=tau)
    patches = torch.randn(2 * b, c + 4, 8)
    masks = torch.ones(2 * b, 1, 8, 8, 8)
    lesion = torch.rand(b, 2, 2, 2) < 0.4
    got = R.filtered_region_loss(patches, args, torch.nn.CrossEntropyLoss(), masks, lesion).item()
    a, d = patches[:, :c].chunk(2)
    hz = torch.nn.functional.normalize(T.standardize_patch_code(torch.stack((a, d))), dim=2, eps=1e-6)
    on = lesion.flatten(1)
    want = 0.0
    for first, second in ((hz[0], hz[1]), (hz[1], hz[0])):
        for p in range(8):
            for i in range(b):
                candidates = [j for j in range(b) if j == i or not (on[i, p] and on[j, p])]
                s = torch.stack([(first[i, :, p] * second[j, :, p]).sum() / tau for j in candidates])
                want -= (s[candidates.index(i)] - torch.logsumexp(s, 0)).item() / (8 * b)
    print(f"formula: filtered {got:.6f}, written out {want:.6f}")
    assert abs(got - want) < 1e-4


def parity(bank, device):
    idxs = [list(range(k * B, (k + 1) * B)) for k in range(3)]
    runs = {}
    for arm in ("region16", "region16_oracle"):
        args = R.arm_args(arm)
        torch.manual_seed(args.model_seed)
        model = T.build_encoder(args).to(device).train()
        optimizers = R.optimizers_for(arm, model, args)
        sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        values = []
        for idx in idxs:
            x, masks, _, _ = bank.batch(idx, device)
            none = torch.zeros(B, 16, 16, 16, dtype=torch.bool)
            terms = R.objective(arm, model, x, args, sim, crit, masks, none)
            values.append(terms["patch"].item())
            R.update(model, terms, R.ARMS[arm][1], optimizers, args.grad_clip)
        runs[arm] = values
    print(f"parity: region16 {runs['region16']}\n        oracle   {runs['region16_oracle']}")
    assert np.allclose(runs["region16"], runs["region16_oracle"], atol=1e-5)


def weight(bank, cells, device):
    args = R.arm_args("region16")
    crit = torch.nn.CrossEntropyLoss()
    on_all = cells > 0
    for state in ("init", f"{HERE}/region16_s2000.pt", f"{HERE}/region16_oracle_s2000.pt"):
        torch.manual_seed(args.model_seed)
        model = T.build_encoder(args).to(device).train()
        if state != "init":
            model.load_state_dict(torch.load(state, map_location="cpu")["model"])
        mass, count, hit, plain, filtered = [], [], [], 0.0, 0.0
        order = np.random.default_rng(0).permutation(len(bank))
        for k in range(WEIGHT_BATCHES):
            idx = order[k * B : (k + 1) * B].tolist()
            x, masks, _, _ = bank.batch(idx, device)
            lesion = on_all[idx]
            with torch.no_grad():
                _, patches = model.global_and_patch_features(x, args.train_patch_grid, n_views=2)
                batch_plain = R.filtered_region_loss(patches, args, crit, masks, torch.zeros_like(lesion)).item()
                batch_filtered = R.filtered_region_loss(patches, args, crit, masks, lesion).item()
                assert batch_filtered <= batch_plain, "dropping negatives can only lower InfoNCE"
                plain += batch_plain
                filtered += batch_filtered
                keep = T.foreground_positions(masks, args.train_patch_grid, args.patch_foreground_thresh)
                a, b = patches[..., keep.to(device)][:, : args.content_channels].chunk(2, dim=0)
                hz = torch.nn.functional.normalize(T.standardize_patch_code(torch.stack((a, b))), dim=2, eps=1e-6)
                scores = (torch.einsum("bcp,dcp->pbd", hz[0], hz[1]) / args.tau).cpu()
                bad = R.same_cell_negatives(lesion, keep.cpu())
                negatives = scores.masked_fill(torch.eye(B, dtype=torch.bool), -1e4).softmax(2)  # over the 31 negatives
                anchors = bad.any(2)
                positive = scores.softmax(2).diagonal(dim1=1, dim2=2)
                mass.append(negatives[anchors].mul(bad[anchors]).sum(1))
                count.append(bad[anchors].sum(1).float() / (B - 1))
                hit.append(positive[anchors])
        mass, count, hit = torch.cat(mass), torch.cat(count), torch.cat(hit)
        name = os.path.basename(state).removesuffix(".pt")
        print(
            f"weight {name}: {len(mass)} anchors, lesioned negatives hold {mass.mean():.3f} of the negatives' softmax "
            f"at {count.mean():.3f} of the count, ratio {mass.mean() / count.mean():.2f}; "
            f"positive's own softmax {hit.mean():.3f}; region loss {plain / WEIGHT_BATCHES:.4f} plain, {filtered / WEIGHT_BATCHES:.4f} filtered"
        )


def main():
    bank_dir = sys.argv[1]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    bank = pilot.Bank(bank_dir, "train")
    cells = labels(bank)
    overlap(cells)
    formula()
    parity(bank, device)
    weight(bank, cells, device)


if __name__ == "__main__":
    main()
