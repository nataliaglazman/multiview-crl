"""Two checks on twin_score.py for one state.

  fisher   the best linear direction the lesion's own effect allows: Sigma_background^-1 cov(subject - twin, burden),
           fitted on the train bank, scored on val. SD of the lesion effect / SD of the background along it, and the
           burden R^2 a probe would get reading that direction from the real code. Background is the twins' spread
           across all subjects, then within anatomy neighbourhoods.
  rings    where the patch loss changes when lesions are removed, by distance from the subject's own lesion.

    python twin_followup.py ROOT OUT_JSON STATE
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_common  # noqa: E402
import twin_score as S  # noqa: E402

import data.cluster_batches as cb  # noqa: E402
import training.main_conv_synthetic as T  # noqa: E402

EDGES = [0, 1.5, 3, 5, 8, 100]


def fisher(tr_x, tr_free, va_x, va_free, y_tr, y_va, neighbours):
    delta, background = tr_x - tr_free, tr_free
    sigma = np.cov(background, rowvar=False)
    sigma += 1e-6 * np.trace(sigma) / len(sigma) * np.eye(len(sigma))
    c = ((delta - delta.mean(0)) * (y_tr - y_tr.mean())[:, None]).mean(0)
    w = np.linalg.solve(sigma, c)
    within = background - background[neighbours].mean(1)
    sigma_w = np.cov(within, rowvar=False)
    sigma_w += 1e-6 * np.trace(sigma_w) / len(sigma_w) * np.eye(len(sigma_w))
    w_within = np.linalg.solve(sigma_w, c)
    va_delta = va_x - va_free
    return {
        "effect_over_background_all": float((va_delta @ w).std() / (va_free @ w).std()),
        "burden_R2_real_code_along_it": float(np.corrcoef(va_x @ w, y_va)[0, 1] ** 2),
        "effect_over_background_within_neighbourhood": float(
            (tr_x - tr_free).dot(w_within).std() / within.dot(w_within).std()
        ),
    }


@torch.no_grad()
def rings(model, bank, batches, args, device):
    total = np.zeros(len(EDGES) - 1)
    rows_in = np.zeros(len(EDGES) - 1)
    coords = None
    for batch in batches:
        idx = np.asarray(batch)
        images = torch.cat([S.tensor(bank["t1"], idx, device), S.tensor(bank["flair"], idx, device)])
        twins = torch.cat([S.tensor(bank["t1free"], idx, device), S.tensor(bank["flairfree"], idx, device)])
        mask = S.tensor(bank["mask"], idx, device)
        _, patches = model.global_and_patch_features(images, S.GRID, n_views=2)
        _, patches_free = model.global_and_patch_features(twins, S.GRID, n_views=2)
        keep = T.foreground_positions(torch.cat([mask, mask]), S.GRID, args.patch_foreground_thresh)
        rows, _ = S.patch_rows(patches, keep, args.tau)
        rows_free, _ = S.patch_rows(patches_free, keep, args.tau)
        change = (rows_free - rows).cpu().numpy()  # (P, B); positive = the lesion lowered this row's loss
        if coords is None:
            g = np.stack(np.meshgrid(*[np.arange(16.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
            coords = g
        kept = coords[keep.cpu().numpy()]
        cells = np.asarray(bank["cells"][idx]).reshape(len(idx), -1)
        centre = (cells @ coords) / cells.sum(1, keepdims=True)
        dist = np.linalg.norm(kept[:, None] - centre[None], axis=-1)  # (P, B)
        for b in range(len(EDGES) - 1):
            inside = (dist >= EDGES[b]) & (dist < EDGES[b + 1])
            total[b] += change[inside].sum()
            rows_in[b] += inside.sum()
    return [
        {
            "cells_from_lesion": f"{EDGES[b]}-{EDGES[b + 1]}",
            "share_of_rows": float(rows_in[b] / rows_in.sum()),
            "mean_loss_drop_per_row": float(total[b] / max(rows_in[b], 1)),
            "share_of_total_drop": float(total[b] / total.sum()),
        }
        for b in range(len(EDGES) - 1)
    ]


def main():
    root, out_json, state = sys.argv[1:4]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = twin_common.run_args()
    train, val = S.load(root, "train"), S.load(root, "val")
    z = np.asarray(train["zc"], dtype=np.float64)
    y_tr, y_va = z[:, S.BURDEN], np.asarray(val["zc"][:, S.BURDEN], dtype=np.float64)
    neighbours = cb.nearest_neighbours(z[:, S.ANATOMY], args.stratify_neighbours)
    sampler = cb.NeighbourhoodBatchSampler(neighbours, args.batch_size, args.loader_seed, 0.0)
    batches = []
    while len(batches) < S.PATCH_BATCHES:
        batches.extend(sampler)
    batches = batches[: S.PATCH_BATCHES]
    rng = np.random.default_rng(0)
    uniform = [rng.choice(2000, args.batch_size, replace=False).tolist() for _ in range(S.PATCH_BATCHES)]

    model = S.build(args, state, device)
    ft, fv = S.features(model, train, device), S.features(model, val, device)
    out = {"volume_sd_voxels": float(np.std(train["vox"]))}
    for v, view in enumerate(("T1", "FLAIR")):
        out.setdefault("fisher", {})[view] = {
            "code": fisher(
                ft["code"][v, :, :14].numpy().astype(np.float64),
                ft["codefree"][v, :, :14].numpy().astype(np.float64),
                fv["code"][v, :, :14].numpy().astype(np.float64),
                fv["codefree"][v, :, :14].numpy().astype(np.float64),
                y_tr,
                y_va,
                neighbours,
            ),
            "gap": fisher(
                ft["gap"][v].numpy().astype(np.float64),
                ft["gapfree"][v].numpy().astype(np.float64),
                fv["gap"][v].numpy().astype(np.float64),
                fv["gapfree"][v].numpy().astype(np.float64),
                y_tr,
                y_va,
                neighbours,
            ),
        }
    out["rings_neighbourhood"] = rings(model, train, batches, args, device)
    out["rings_uniform"] = rings(model, train, uniform, args, device)
    with open(out_json, "w") as fp:
        json.dump(out, fp, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
