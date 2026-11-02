"""What does one burden lesion contribute to the run's two losses and to the probe?

Each state is scored on the run's generator, with lesion-free twins (twin_render.py):
  sizes     lesion voxels, share of brain, and the probe ceiling (volume -> burden latent)
  global    the run's global InfoNCE on real batches, lesioned vs all twins; the lesion's share of the standardized code
  patch     per-row patch InfoNCE (parity-checked against T.patch_contrastive_loss) at lesion rows vs the same rows in twins
  probe     5-fold ridge of the burden latent from code / GAP, then from (subject - twin), the same features with
            anatomy, style and everything else subtracted exactly, and from (code - its neighbourhood's mean)

    python twin_score.py ROOT OUT_JSON NAME=STATE [NAME=STATE ...]     (STATE = init or a checkpoint path)
"""

import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_common  # noqa: E402

import data.cluster_batches as cb  # noqa: E402
import training.main_conv_synthetic as T  # noqa: E402

N = {"train": 2000, "val": 400, "val00": 400}
GRID = [16, 16, 16]
ANATOMY = [0, 1, 5, 6, 7, 8]
BURDEN = 2
BATCHES = 200  # real batches for the global term
PATCH_BATCHES = 24  # real batches for the patch term
LESION_ROW = 0.1


def load(root, split):
    return {
        k: np.load(f"{root}/{split}_{k}.npy", mmap_mode="r")
        for k in ("t1", "flair", "t1free", "flairfree", "mask", "zc", "cells", "vox")
    }


def tensor(array, idx, device):
    return torch.from_numpy(np.asarray(array[idx], dtype=np.float32))[:, None].to(device)


def build(args, state, device):
    torch.manual_seed(args.model_seed)
    model = T.build_encoder(args)
    if state != "init":
        weights = torch.load(state, map_location="cpu")["model"]
        missing = model.load_state_dict(weights, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            print("  load:", missing, flush=True)
    return model.to(device).eval()


@torch.no_grad()
def forward(model, t1, flair, device):
    """Backbone GAP and head output per view, for view-major (t1, flair) batches."""
    x = torch.cat([t1, flair])
    h = model._encode(x, 2, None)
    gap = h.mean((2, 3, 4))
    code = model._global_code(h, x, 2)
    return h, gap, code


@torch.no_grad()
def features(model, bank, device, batch=32, t1key="t1", t1free="t1free"):
    n = len(bank["zc"])
    out = {k: [] for k in ("gap", "gapfree", "code", "codefree")}
    for start in range(0, n, batch):
        idx = np.arange(start, min(start + batch, n))
        _, gap, code = forward(model, tensor(bank[t1key], idx, device), tensor(bank["flair"], idx, device), device)
        _, gapf, codef = forward(
            model, tensor(bank[t1free], idx, device), tensor(bank["flairfree"], idx, device), device
        )
        out["gap"].append(gap.cpu())
        out["code"].append(code.cpu())
        out["gapfree"].append(gapf.cpu())
        out["codefree"].append(codef.cpu())
    # (2, n, width): view, subject, unit
    return {k: torch.cat(v).reshape(n, 2, -1).transpose(0, 1).float() if False else _unstack(v) for k, v in out.items()}


def _unstack(chunks):
    """Each chunk is view-major (2b, width); returns (2, n, width)."""
    views = [[], []]
    for c in chunks:
        a, b = c.chunk(2, dim=0)
        views[0].append(a)
        views[1].append(b)
    return torch.stack([torch.cat(views[0]), torch.cat(views[1])])


def ridge(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def snr(delta, free, y):
    """Along the direction that best reads the burden from the lesion's own effect: its SD over the background SD."""
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(delta, y)
    w = pipe[-1].coef_ / pipe[0].scale_
    return float((delta @ w).std() / (free @ w).std())


def coherence(delta):
    """Mean cosine between different subjects' lesion effects: 1 if every lesion moves the feature the same way."""
    unit = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
    gram = unit @ unit.T
    n = len(unit)
    return float((gram.sum() - n) / (n * (n - 1)))


def probe(feat, y, views=(0, 1)):
    rows = {}
    for v, name in zip(views, ("T1", "FLAIR")):
        code, free = feat["code"][v, :, :14].numpy(), feat["codefree"][v, :, :14].numpy()
        gap, gapfree = feat["gap"][v].numpy(), feat["gapfree"][v].numpy()
        rows[name] = {
            "code_R2": ridge(code, y),
            "gap_R2": ridge(gap, y),
            "code_minus_twin_R2": ridge(code - free, y),
            "gap_minus_twin_R2": ridge(gap - gapfree, y),
            "lesion_vs_background_code": snr(code - free, free, y),
            "lesion_vs_background_gap": snr(gap - gapfree, gapfree, y),
            "coherence_code": coherence(code - free),
            "coherence_gap": coherence(gap - gapfree),
        }
    return rows


def global_share(feat, batches, model, args, sim, crit):
    """Global InfoNCE with the batch's lesions vs all twins, and the lesion's share of the standardized code."""
    with_lesion, without, share = [], [], []
    for batch in batches:
        idx = torch.as_tensor(batch)
        code = feat["code"][:, idx]  # (2, B, 18)
        free = feat["codefree"][:, idx]
        with_lesion.append(float(T.contrastive_loss(torch.cat([code[0], code[1]]), model, args, sim, crit, None)))
        without.append(float(T.contrastive_loss(torch.cat([free[0], free[1]]), model, args, sim, crit, None)))
        c, f = code[:, :, :14], free[:, :, :14]
        mean, sd = c.mean(1, keepdim=True), c.std(1, keepdim=True) + 1e-6
        zc, zf = (c - mean) / sd, (f - mean) / sd
        share.append(float((zc - zf).square().sum() / zc.square().sum()))
    return {
        "infonce_lesion": float(np.mean(with_lesion)),
        "infonce_twins": float(np.mean(without)),
        "lesion_share_of_code_variance": float(np.mean(share)),
    }


def patch_rows(patches, keep, tau):
    """Per-row patch InfoNCE of (2B, units, positions) patches: (P, B) of T1->FLAIR + FLAIR->T1, as training computes."""
    patches = patches[..., keep]
    a, b = patches[:, :14, :].chunk(2, dim=0)
    hz = T.standardize_patch_code(torch.stack((a, b), dim=0))
    hz_n = F.normalize(hz, dim=2, eps=1e-6)
    scores = torch.einsum("bcp,dcp->pbd", hz_n[0], hz_n[1]) / tau
    B = scores.shape[1]
    d = torch.arange(B, device=scores.device)
    forward_rows = -F.log_softmax(scores, dim=2)[:, d, d]
    backward_rows = -F.log_softmax(scores.transpose(1, 2), dim=2)[:, d, d]
    return forward_rows + backward_rows, hz_n


@torch.no_grad()
def patch_analysis(model, bank, batches, args, sim, crit, device):
    grid = tuple(GRID)
    stats = {k: [] for k in ("rows", "lesion_rows", "loss", "loss_twin", "les_ce", "les_ce_twin", "other_ce", "share")}
    posles, posfree = [], []
    parity = []
    for batch in batches:
        idx = np.asarray(batch)
        t1, fl = tensor(bank["t1"], idx, device), tensor(bank["flair"], idx, device)
        t1f, flf = tensor(bank["t1free"], idx, device), tensor(bank["flairfree"], idx, device)
        mask = tensor(bank["mask"], idx, device)
        images, twins = torch.cat([t1, fl]), torch.cat([t1f, flf])
        _, patches = model.global_and_patch_features(images, GRID, n_views=2)
        _, patches_free = model.global_and_patch_features(twins, GRID, n_views=2)
        masks = torch.cat([mask, mask])
        keep = T.foreground_positions(masks, GRID, args.patch_foreground_thresh)
        rows, hz = patch_rows(patches, keep, args.tau)
        rows_free, hz_free = patch_rows(patches_free, keep, args.tau)
        official = float(T.patch_contrastive_loss(patches, model, args, sim, crit, masks, None, None))
        parity.append(abs(official - float(rows.mean())))
        cells = torch.from_numpy(np.asarray(bank["cells"][idx])).to(device).flatten(1)[:, keep]  # (B, P)
        lesion = (cells >= LESION_ROW).T  # (P, B)
        stats["rows"].append(int(lesion.numel()))
        stats["lesion_rows"].append(int(lesion.sum()))
        stats["loss"].append(float(rows.sum()))
        stats["loss_twin"].append(float(rows_free.sum()))
        stats["les_ce"].append(float(rows[lesion].sum()))
        stats["les_ce_twin"].append(float(rows_free[lesion].sum()))
        stats["other_ce"].append(float(rows[~lesion].sum()))
        p, b = lesion.nonzero(as_tuple=True)
        posles.append((hz[0, b, :, p] * hz[1, b, :, p]).sum(1).cpu())
        posfree.append((hz_free[0, b, :, p] * hz_free[1, b, :, p]).sum(1).cpu())
    rows, lrows = sum(stats["rows"]), sum(stats["lesion_rows"])
    return {
        "parity_max_abs": max(parity),
        "lesion_rows_share": lrows / rows,
        "patch_infonce_lesion": sum(stats["loss"]) / rows,
        "patch_infonce_twins": sum(stats["loss_twin"]) / rows,
        "ce_lesion_rows": sum(stats["les_ce"]) / lrows,
        "ce_same_rows_twin": sum(stats["les_ce_twin"]) / lrows,
        "ce_other_rows": sum(stats["other_ce"]) / (rows - lrows),
        "share_of_loss_at_lesion_rows": sum(stats["les_ce"]) / sum(stats["loss"]),
        "positive_cos_lesion_rows": float(torch.cat(posles).mean()),
        "positive_cos_same_rows_twin": float(torch.cat(posfree).mean()),
    }


def main():
    root, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = twin_common.run_args()
    train, val, val00 = load(root, "train"), load(root, "val"), load(root, "val00")
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()

    y_train, y_val = np.asarray(train["zc"][:, BURDEN]), np.asarray(val["zc"][:, BURDEN])
    report = {}

    # Twins differ only near the lesion.
    probe_i = 3
    diff = np.abs(np.asarray(val["t1"][probe_i], float) - np.asarray(val["t1free"][probe_i], float))
    report["twin_check"] = {
        "t1_changed_voxels": int((diff > 1e-3).sum()),
        "lesion_voxels": float(val["vox"][probe_i]),
    }
    brain = np.asarray(train["mask"]).reshape(2000, -1).sum(1)
    vox = np.asarray(train["vox"])
    report["sizes"] = {
        "lesion_voxels_p5_p50_p95": [float(q) for q in np.percentile(vox, [5, 50, 95])],
        "lesion_voxels_min_max": [float(vox.min()), float(vox.max())],
        "brain_voxels_median": float(np.median(brain)),
        "lesion_percent_of_brain_p5_p50_p95": [float(q) for q in np.percentile(100 * vox / brain, [5, 50, 95])],
        "probe_ceiling_volume_to_latent_R2": ridge(vox[:, None], y_train),
    }

    z = np.asarray(train["zc"], dtype=np.float64)
    neighbours = cb.nearest_neighbours(z[:, ANATOMY], args.stratify_neighbours)
    sampler = cb.NeighbourhoodBatchSampler(neighbours, args.batch_size, args.loader_seed, 0.0)
    neighbour_batches = []
    while len(neighbour_batches) < BATCHES:
        neighbour_batches.extend(sampler)
    neighbour_batches = neighbour_batches[:BATCHES]
    rng = np.random.default_rng(0)
    uniform_batches = [rng.choice(2000, args.batch_size, replace=False).tolist() for _ in range(BATCHES)]
    sd = lambda batches, col: float(
        np.sqrt(z[np.asarray(batches)][..., col].var(axis=1, ddof=1).mean()) / z[:, col].std()
    )
    report["batches"] = {
        "anatomy_sd_ratio": float(np.mean([sd(neighbour_batches, c) for c in ANATOMY])),
        "burden_sd_ratio": sd(neighbour_batches, BURDEN),
    }

    for item in states:
        name, state = item.split("=", 1)
        print("state", name, flush=True)
        model = build(args, state, device)
        row = {}
        ftrain = features(model, train, device)
        fval = features(model, val, device)
        row["probe_val"] = probe(fval, y_val)
        for key in ("gap", "code"):
            row.setdefault("neighbourhood_centred", {})[key] = {}
        for v, view in enumerate(("T1", "FLAIR")):
            code = ftrain["code"][v, :, :14].numpy()
            centred = code - code[neighbours].mean(1)
            row["neighbourhood_centred"]["code"][view] = {
                "code_R2_train_bank": ridge(code, y_train),
                "code_minus_neighbourhood_mean_R2": ridge(centred, y_train),
                "code_minus_twin_R2": ridge(code - ftrain["codefree"][v, :, :14].numpy(), y_train),
            }
        del row["neighbourhood_centred"]["gap"]
        row["global_neighbourhood"] = global_share(ftrain, neighbour_batches, model, args, sim, crit)
        row["global_uniform"] = global_share(ftrain, uniform_batches, model, args, sim, crit)
        row["patch_neighbourhood"] = patch_analysis(
            model, train, neighbour_batches[:PATCH_BATCHES], args, sim, crit, device
        )
        row["patch_uniform"] = patch_analysis(model, train, uniform_batches[:PATCH_BATCHES], args, sim, crit, device)
        f00 = features(model, val00, device)
        row["probe_val_t1_value_0"] = probe(f00, y_val, views=(0,))
        report[name] = row
        with open(out_json, "w") as fp:
            json.dump(report, fp, indent=2)
        print(json.dumps(row, indent=1), flush=True)


if __name__ == "__main__":
    main()
