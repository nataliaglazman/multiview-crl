"""If GAP already holds burden, does the run's own InfoNCE route it into the 14 content units?

Backbones are frozen at sup.py checkpoints; GAP features are computed once. Then only the recipe's head (64 -> 100 -> 18,
first 14 units content) is trained from random with the run's global InfoNCE on neighbourhood or uniform batches. No
labels in training. Labels only score: 5-fold ridge on the test split, from GAP and from the 14 units.

Sources:
  dense:CKPT            the dense arm's backbone (sees the lesion; was never asked for anatomy)
  ssl:CKPT              a self-supervised backbone (knows anatomy; never saw burden in GAP)
  both:CKPT_DENSE,CKPT  the two GAP vectors concatenated per view, so anatomy and burden are both on offer

    python headonly.py BANK T1KEY OUT_JSON NAME=SOURCE [...]
"""

import copy
import json
import os
import sys
import warnings

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warm  # noqa: E402

import data.cluster_batches as cb  # noqa: E402
import training.losses as losses  # noqa: E402
import training.main_conv_synthetic as T  # noqa: E402

TAU = 0.1
SIM, CRIT = nn.CosineSimilarity(dim=-1), nn.CrossEntropyLoss()
ANATOMY = warm.ANATOMY
FACTORS = warm.FACTORS
STEPS = 4000


def infonce(z1, z2):
    hz = T.standardize_contrastive_code(torch.stack((z1, z2)))
    return losses.infonce_loss(
        hz,
        sim_metric=SIM,
        criterion=CRIT,
        tau=TAU,
        estimated_content_indices=[list(range(hz.shape[-1]))],
        subsets=[(0, 1)],
        cross_view_negs_only=True,
    ).squeeze()


def backbone_gap(checkpoint, bank_root, t1key, cache):
    if os.path.exists(cache):
        data = np.load(cache)
        return {k: [data[f"{k}_0"], data[f"{k}_1"]] for k in ("train", "test")}
    args = warm.run_args()
    torch.manual_seed(args.model_seed)
    model = T.build_encoder(args)
    state = torch.load(checkpoint, map_location="cpu")["model"]
    if any(k.startswith("encoder.encoder.") for k in state):
        state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.encoder")}
    print(" load", model.load_state_dict(state, strict=False).unexpected_keys[:3], flush=True)
    device = warm.pick_device()
    model = model.to(device).eval()
    out = {}
    for split in ("train", "test"):
        out[split] = warm.features(model, warm.Bank(bank_root, split, t1key), device)["gap"]
    np.savez(cache, **{f"{k}_{v}": out[k][v] for k in out for v in range(2)})
    return out


def probes(x, bank):
    row = {"volume": warm.ridge(x, bank.vox), "latent": warm.ridge(x, bank.zc[:, 2])}
    for column, factor in FACTORS.items():
        row[factor] = warm.ridge(x, bank.zc[:, column])
    return {k: round(v, 3) for k, v in row.items()}


def source_gap(spec, root, t1key, folder):
    kind, value = spec.split(":", 1)
    parts = value.split(",")
    gaps = [
        backbone_gap(
            p, root, t1key, f"{folder}/gap_{t1key}_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)}.npz"
        )
        for p in parts
    ]
    if len(gaps) == 1:
        return gaps[0]
    return {s: [np.concatenate([g[s][v] for g in gaps], 1) for v in range(2)] for s in ("train", "test")}


def train_head(gap, neighbours, shared, uniform, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = [g.mean(0) for g in gap["train"]]
    sd = [np.maximum(g.std(0), 1e-3) for g in gap["train"]]  # a tail never reached in training must not explode in test
    train = [torch.from_numpy((g - m) / s).float() for g, m, s in zip(gap["train"], mean, sd)]
    test = [torch.from_numpy((g - m) / s).float() for g, m, s in zip(gap["test"], mean, sd)]
    head = nn.Sequential(nn.Linear(train[0].shape[1], 100), nn.LeakyReLU(), nn.Linear(100, 18))
    other = head if shared else copy.deepcopy(head)
    parameters = list(head.parameters()) + ([] if shared else list(other.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    sampler = cb.NeighbourhoodBatchSampler(neighbours, 32, 10042, 0.0)

    def batches():
        while True:
            if uniform:
                yield rng.choice(len(train[0]), 32, replace=False)
            else:
                yield from sampler

    source = batches()
    final = []
    for step in range(STEPS):
        index = torch.as_tensor(np.asarray(next(source)))
        loss = infonce(head(train[0][index])[:, :14], other(train[1][index])[:, :14])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step >= STEPS - 200:
            final.append(float(loss))
    with torch.no_grad():
        return float(np.mean(final)), head(test[0])[:, :14].numpy(), other(test[1])[:, :14].numpy()


def main():
    root, t1key, out_json, specs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    folder = os.path.dirname(out_json)
    train = warm.Bank(root, "train", t1key)
    test = warm.Bank(root, "test", t1key)
    neighbours = cb.nearest_neighbours(train.zc[:, ANATOMY], 50)
    report = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for item in specs:
        name, spec = item.split("=", 1)
        gap = source_gap(spec, root, t1key, folder)
        row = {"gap_t1": probes(gap["test"][0], test), "gap_flair": probes(gap["test"][1], test)}
        print(name, "GAP t1", row["gap_t1"], flush=True)
        print(name, "GAP flair", row["gap_flair"], flush=True)
        for label, shared, uniform in (
            ("neighbour_sepheads", False, False),
            ("uniform_sepheads", False, True),
            ("neighbour_shared", True, False),
        ):
            loss, c1, c2 = train_head(gap, neighbours, shared, uniform)
            cell = {"loss_last200": round(loss, 4), "code_t1": probes(c1, test), "code_flair": probes(c2, test)}
            row[label] = cell
            print(name, label, cell, flush=True)
        report[name] = row
        with open(out_json, "w") as stream:
            json.dump(report, stream, indent=1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
