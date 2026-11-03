"""Which shared content does each batch scheme pay for? The run's InfoNCE on codes built from the true latents.

A code holds the six anatomy factors, the lesion (centroid x, y, z of the 16^3 lesion load, plus burden), or both;
each unit is seen by the two views with independent noise at cross-view r. Schemes:

  neighbour32       the SSL arms' batches: an anchor and 31 of its 50 nearest training subjects on the six anatomy
                    factors, standardized per batch (the run's loss)
  uniform32         32 distinct subjects drawn uniformly, same loss
  *_gauge           the same batches under a population gauge instead of per-batch standardization
  *_queue1024       plus 1024 other subjects as negatives, population gauge, as warm.py --queue

    python batch_price.py BANK OUT_JSON
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

import data.cluster_batches as cb
import training.losses as losses
import training.main_conv_synthetic as T

ANATOMY = [0, 1, 5, 6, 7, 8]
BATCHES = 300
TAU = 0.1
SIM, CRIT = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()


def lesion_centroid(cells):
    """(n, 3) load-weighted centroid of each subject's 16^3 lesion load, in cell units; the mean where there is none."""
    load = np.asarray(cells, dtype=np.float64).reshape(len(cells), -1)
    grid = np.stack(np.meshgrid(*[np.arange(16)] * 3, indexing="ij"), -1).reshape(-1, 3)
    total = load.sum(1, keepdims=True)
    centre = load @ grid / np.maximum(total, 1e-12)
    centre[total[:, 0] <= 0] = centre[total[:, 0] > 0].mean(0)
    return centre


def standard(x):
    return (x - x.mean(0)) / x.std(0)


def run_loss(z1, z2):
    hz = T.standardize_contrastive_code(torch.stack((z1, z2)))
    value = losses.infonce_loss(
        hz,
        sim_metric=SIM,
        criterion=CRIT,
        tau=TAU,
        estimated_content_indices=[list(range(hz.shape[-1]))],
        subsets=[(0, 1)],
        cross_view_negs_only=True,
    )
    return float(value.squeeze())


def gauge_loss(z1, z2, batch, others):
    """warm.queue_infonce on population-standardized codes: batch keys plus ``others`` as negatives."""
    total = 0.0
    q_rows, k_rows = (F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)), (
        F.normalize(z2, dim=-1),
        F.normalize(z1, dim=-1),
    )
    for q, k in zip(q_rows, k_rows):
        logits = torch.cat([q[batch] @ k[batch].T, q[batch] @ k[others].T], 1) / TAU
        total += float(F.cross_entropy(logits, torch.arange(len(batch))))
    return total / 2


def views(columns, rho, rng):
    """Two (n, d) views of standardized columns, each unit with cross-view correlation ``rho``."""
    keep = np.sqrt(rho)
    out = [keep * columns + np.sqrt(1 - rho) * rng.standard_normal(columns.shape) for _ in range(2)]
    return [torch.from_numpy(standard(v)).float() for v in out]


def main():
    bank, out_json = sys.argv[1], sys.argv[2]
    zc = np.array(np.load(f"{bank}/train_zc.npy"), dtype=np.float64)
    vox = np.array(np.load(f"{bank}/train_vox.npy"), dtype=np.float64)
    centre = lesion_centroid(np.load(f"{bank}/train_cells.npy", mmap_mode="r"))
    n = len(zc)
    rng = np.random.default_rng(0)
    sampler = cb.NeighbourhoodBatchSampler(cb.nearest_neighbours(zc[:, ANATOMY], 50), 32, 0, 0.0)
    near = []
    while len(near) < BATCHES:
        near.extend(sampler)
    near = [np.asarray(b) for b in near[:BATCHES]]
    uniform = [rng.choice(n, 32, replace=False) for _ in range(BATCHES)]
    others = {}
    for label, sets in (("neighbour32", near), ("uniform32", uniform)):
        others[label] = [rng.choice(np.setdiff1d(np.arange(n), b), 1024, replace=False) for b in sets[:100]]

    blocks = {
        "anatomy": standard(zc[:, ANATOMY]),
        "lesion_position": standard(centre),
        "lesion": standard(np.column_stack([centre, vox])),
        "anatomy+lesion": standard(np.column_stack([zc[:, ANATOMY], centre, vox])),
        "brain+lesion": standard(np.column_stack([zc[:, 0], centre, vox])),
    }
    report = {
        "within_neighbourhood_sd_over_population": {
            "anatomy": float(
                np.mean([zc[b][:, ANATOMY].std(0, ddof=1).mean() for b in near]) / zc[:, ANATOMY].std(0).mean()
            ),
            "lesion_position": float(np.mean([centre[b].std(0, ddof=1).mean() for b in near]) / centre.std(0).mean()),
            "burden": float(np.mean([vox[b].std(ddof=1) for b in near]) / vox.std()),
        },
        "log_B": {"32": float(np.log(32)), "32+1024": float(np.log(32 + 1024))},
        "loss": {},
    }
    print(report["within_neighbourhood_sd_over_population"], flush=True)
    for rho in (0.99, 0.9):
        table = {}
        for name, columns in blocks.items():
            z1, z2 = views(columns, rho, rng)
            row = {}
            for label, sets in (("neighbour32", near), ("uniform32", uniform)):
                row[label] = float(np.mean([run_loss(z1[b], z2[b]) for b in sets]))
                row[f"{label}_gauge"] = float(
                    np.mean([gauge_loss(z1, z2, torch.as_tensor(b), torch.as_tensor(b[:0])) for b in sets])
                )
                row[f"{label}_queue1024"] = float(
                    np.mean(
                        [
                            gauge_loss(z1, z2, torch.as_tensor(b), torch.as_tensor(o))
                            for b, o in zip(sets, others[label])
                        ]
                    )
                )
            table[name] = row
            print(f"r {rho} {name:16s}", {k: round(v, 3) for k, v in row.items()}, flush=True)
        report["loss"][str(rho)] = table
    with open(out_json, "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main()
