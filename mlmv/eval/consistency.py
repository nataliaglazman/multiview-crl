"""Cross-split consistency of G: linear CKA between G pooled over two disjoint random region sets.

Each split partitions the eligible eroded regions into two sets (volume share within ``ratio``); for every
subject G is pooled over set 1 and over set 2 of the same modality, and CKA compares the two ``[n, d_G]``
matrices. The spec's Phase 4 metric takes the median over 20 splits; training logs use fewer.
"""

from __future__ import annotations

import numpy as np
import torch

from mlmv.data.geometry import draw_region_sets
from mlmv.eval.metrics import linear_cka

MODALITIES = ("t1", "flair")


@torch.no_grad()
def g_region_sums(model, source, split: str, indices, geometry, batch: int = 8):
    """Per modality ``[n, R_e, d_G]`` sums and ``[R_e]`` counts of G over the eligible eroded regions."""
    model.eval()
    dev = next(model.parameters()).device
    eroded = torch.from_numpy(geometry.eroded).flatten().to(dev)
    member = torch.stack([(eroded == r).float() for r in geometry.eligible], 1)  # [P, R_e]
    sums = {m: [] for m in MODALITIES}
    for start in range(0, len(indices), batch):
        data = source.batch(split, list(indices[start : start + batch]))
        for m in MODALITIES:
            g = model.blocks(data[m].to(dev), m)["G"].float().flatten(2)  # [n, d, P]
            sums[m].append((g @ member).permute(0, 2, 1).cpu())
    return {m: torch.cat(v).numpy() for m, v in sums.items()}, member.sum(0).cpu().numpy()


def g_cross_split_cka(sums, counts, geometry, ratio, n_splits: int = 20, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    index = {r: j for j, r in enumerate(geometry.eligible)}
    out = {}
    for m, s in sums.items():
        values = []
        for _ in range(n_splits):
            a, b = draw_region_sets(geometry.eligible, counts, ratio, gen)
            ia, ib = [index[int(r)] for r in a], [index[int(r)] for r in b]
            pa = s[:, ia].sum(1) / counts[ia].sum()
            pb = s[:, ib].sum(1) / counts[ib].sum()
            values.append(linear_cka(pa, pb))
        out[m] = dict(median=float(np.median(values)), values=[float(v) for v in values])
    return out
