"""Leakage matrix on the synthetic benchmark: R^2 of each true factor from each learned block.

Readouts (per modality, from the template-subtracted blocks of a frozen model):

* g and site per subject, from the block pooled over every position (site: R^2 on its 2-d embedding, which is
  what enters the renderer, plus logistic balanced accuracy);
* r_R per region, from the block pooled over R (the generator's region, unmerged and uneroded); one probe per
  region, averaged;
* s, w and the lesion indicator per position, from the block averaged over each lattice cell
  (``positions_per_subject`` random cells per subject).

With the brain renderer, subject pooling covers the brain only, cells are drawn only where the generator has a
region, and the lesion target is the rendered lesion fraction of each cell.

Each R^2 is the mean over subject-grouped folds of ridge and of a 2-layer MLP; ``best`` is the larger.

A reference row (block ``raw``, modality ``t1+flair``) runs the same probes on simple summaries of the images
themselves, at the same readouts: per cell, the eight sub-cell means and the 3^3 neighbourhood of cell means;
per region, the mean, sd and 10/50/90th percentiles inside the region; per subject, all regions' summaries.
It shows how readable each factor is from the image without any training. It is a reference, not a bound: a
trained block can beat it.

Also the counterfactual: the same latents re-rendered with a different site, and the relative L2 change of each
block's code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mlmv.eval.metrics import cv_balanced_accuracy, cv_r2

LEVELS = ("G", "L", "S")
MODALITIES = ("t1", "flair")
CELL_FACTORS = ("s", "w", "lesion")


def _gather_cells(x, pick):
    """``x [n, c, *L]``, ``pick [n, k]`` flat cell indices -> ``[n, k, c]``."""
    flat = x.flatten(2)
    return torch.gather(flat, 2, pick[:, None].expand(-1, flat.shape[1], -1)).permute(0, 2, 1)


def raw_image_features(images, pick, gen, region_masks):
    """Reference features from the images themselves (both modalities concatenated).

    ``images`` [(t1, flair)] each ``[n, 1, X, Y, Z]``; returns cell ``[n, k, f]``, region ``[n, R, f]`` and
    subject ``[n, R * f]`` arrays (see the module docstring).
    """
    m = gen.m
    cell_parts, region_parts = [], []
    for img in images:
        img = img.float()
        cells = F.avg_pool3d(img, m)  # [n, 1, *L]
        if m % 2 == 0:
            sub = F.avg_pool3d(img, m // 2)[:, 0]  # [n, 2Lx, 2Ly, 2Lz]
            n, X, Y, Z = sub.shape
            sub = (
                sub.reshape(n, X // 2, 2, Y // 2, 2, Z // 2, 2)
                .permute(0, 2, 4, 6, 1, 3, 5)
                .reshape(n, 8, *cells.shape[2:])
            )
            cell_parts.append(_gather_cells(sub, pick))
        padded = F.pad(cells, (1, 1, 1, 1, 1, 1), mode="replicate")
        Lx, Ly, Lz = cells.shape[2:]
        neigh = torch.cat(
            [padded[:, :, i : i + Lx, j : j + Ly, k : k + Lz] for i in range(3) for j in range(3) for k in range(3)], 1
        )
        cell_parts.append(_gather_cells(neigh, pick))
        stats = []
        for mask in region_masks:
            vals = img[:, 0][:, mask]  # [n, voxels]
            q = torch.quantile(vals, torch.tensor([0.1, 0.5, 0.9], device=vals.device), dim=1).T
            stats.append(torch.cat([vals.mean(1, keepdim=True), vals.std(1, keepdim=True), q], 1))
        region_parts.append(torch.stack(stats, 1))  # [n, R, 5]
    region = torch.cat(region_parts, 2)
    return torch.cat(cell_parts, 2), region, region.flatten(1)


@torch.no_grad()
def collect_codes(model, source, split: str, indices, positions_per_subject: int, batch: int = 8, seed: int = 0):
    """Pooled codes and matching ground truth for subjects ``indices`` of ``split``.

    Returns ``codes[modality][level][readout]``, plus ``codes["raw"][readout]`` for the image reference.
    """
    model.eval()
    gen_model = source.gen
    labels = torch.from_numpy(source.feature_labels).flatten()
    regions = sorted(int(r) for r in labels.unique() if r > 0)
    dev = gen_model.device
    onehot = torch.stack([(labels == r).float() for r in regions]).to(dev)
    onehot = onehot / onehot.sum(1, keepdim=True)  # [R, P]
    mask = source.feature_mask
    pool = None if mask is None else torch.from_numpy(np.asarray(mask, np.float32)).flatten().to(dev)
    image_regions = torch.from_numpy(gen_model.image_regions_np).to(dev)
    region_masks = [image_regions == r for r in regions]
    candidates = torch.nonzero(gen_model.regions.flatten() > 0)[:, 0].to(dev)  # cells inside a region
    rng = torch.Generator().manual_seed(seed)
    codes = {m: {lvl: {"subject": [], "region": [], "cell": []} for lvl in LEVELS} for m in MODALITIES}
    codes["raw"] = {"subject": [], "region": [], "cell": []}
    truth = {k: [] for k in ("g", "site", "site_emb", "r", *CELL_FACTORS)}
    for start in range(0, len(indices), batch):
        chunk = list(indices[start : start + batch])
        data = source.batch(split, chunk)
        lat = data["latents"]
        n = len(chunk)
        draws = [torch.randperm(len(candidates), generator=rng)[:positions_per_subject] for _ in range(n)]
        pick = candidates[torch.stack(draws).to(dev)]
        for m in MODALITIES:
            blocks = model.blocks(data[m].to(next(model.parameters()).device), m)
            for lvl, f in blocks.items():
                f = f.float().to(dev)
                flat = f.flatten(2)
                subject = flat.mean(2) if pool is None else (flat * pool).sum(2) / pool.sum()
                codes[m][lvl]["subject"].append(subject.cpu())
                codes[m][lvl]["region"].append((flat @ onehot.T).permute(0, 2, 1).cpu())
                codes[m][lvl]["cell"].append(_gather_cells(F.adaptive_avg_pool3d(f, gen_model.L), pick).cpu())
        raw_cell, raw_region, raw_subject = raw_image_features(
            [data["t1"].to(dev), data["flair"].to(dev)], pick, gen_model, region_masks
        )
        codes["raw"]["cell"].append(raw_cell.cpu())
        codes["raw"]["region"].append(raw_region.cpu())
        codes["raw"]["subject"].append(raw_subject.cpu())
        truth["g"].append(lat["g"].cpu())
        truth["site"].append(lat["site"].cpu())
        truth["site_emb"].append(lat["site_emb"].cpu())
        truth["r"].append(lat["r"][:, [r - 1 for r in regions]].cpu())
        for k in CELL_FACTORS:
            truth[k].append(_gather_cells(lat[k], pick).cpu())
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    out = {m: {lvl: {k: cat(v) for k, v in codes[m][lvl].items()} for lvl in LEVELS} for m in MODALITIES}
    out["raw"] = {k: cat(v) for k, v in codes["raw"].items()}
    return out, {k: cat(v) for k, v in truth.items()}, regions


def readout_rows(modality, block, c, truth, eval_cfg, device="cpu"):
    """One block's (or the image reference's) row per factor."""
    n = truth["g"].shape[0]
    subj = np.arange(n)
    rows = []

    def row(factor, res, **extra):
        rows.append(dict(modality=modality, block=block, factor=factor, **res, **extra))

    row("g", cv_r2(c["subject"], truth["g"], subj, eval_cfg, device=device))
    row(
        "site",
        cv_r2(c["subject"], truth["site_emb"], subj, eval_cfg, device=device),
        site_balanced_accuracy=cv_balanced_accuracy(c["subject"], truth["site"], subj, eval_cfg.cv_folds),
    )
    per_region = [
        cv_r2(c["region"][:, j], truth["r"][:, j], subj, eval_cfg, device=device) for j in range(truth["r"].shape[1])
    ]
    res = {k: float(np.mean([p[k] for p in per_region])) for k in ("ridge", "mlp", "best")}
    row("r", res, r_best_min_region=float(np.min([p["best"] for p in per_region])))
    k = c["cell"].shape[1]
    groups = np.repeat(subj, k)
    x = c["cell"].reshape(n * k, -1)
    for factor in CELL_FACTORS:
        row(factor, cv_r2(x, truth[factor].reshape(n * k, -1), groups, eval_cfg, device=device))
    return rows


def leakage_rows(codes, truth, eval_cfg, device="cpu", reference: bool = True):
    rows = []
    for m in MODALITIES:
        for lvl in LEVELS:
            rows += readout_rows(m, lvl, codes[m][lvl], truth, eval_cfg, device)
    if reference and "raw" in codes:
        rows += readout_rows("t1+flair", "raw", codes["raw"], truth, eval_cfg, device)
    return rows


def pass_checks(rows, thresholds, variant: str):
    """The spec's pass criteria per modality (they are defined for the ``exact`` variant)."""
    best = {(r["modality"], r["block"], r["factor"]): r["best"] for r in rows}
    rules = [
        ("G", "g", ">=", thresholds.g_r2_min),
        ("G", "r", "<=", thresholds.leak_r2_max),
        ("G", "s", "<=", thresholds.leak_r2_max),
        ("L", "r", ">=", thresholds.r_r2_min),
        ("L", "s", "<=", thresholds.leak_r2_max),
        ("S", "s", ">=", thresholds.s_r2_min),
    ]
    if variant == "one_modality_lesion":
        rules.append(("S", "lesion", "<=", thresholds.leak_r2_max))
    checks = []
    for m in MODALITIES:
        for block, factor, op, thr in rules:
            value = best[(m, block, factor)]
            ok = value >= thr if op == ">=" else value <= thr
            checks.append(
                dict(modality=m, block=block, factor=factor, rule=f"{op} {thr}", value=value, passed=bool(ok))
            )
    return checks


@torch.no_grad()
def counterfactual_change(model, source, split: str, indices, seed: int = 0):
    """Relative L2 change of each block's code when only the site is changed (mean over subjects)."""
    model.eval()
    g = source.gen
    lat = g.latents(split, list(indices))
    rng = torch.Generator().manual_seed(seed)
    shift = 1 + torch.randint(g.cfg.n_sites - 1, (len(indices),), generator=rng).to(lat["site"].device)
    new_site = (lat["site"] + shift) % g.cfg.n_sites
    before, after = g.render(lat), g.render(lat, site=new_site)
    dev = next(model.parameters()).device
    out = []
    for k, m in enumerate(MODALITIES):
        b0, b1 = model.blocks(before[k].to(dev), m), model.blocks(after[k].to(dev), m)
        for lvl in LEVELS:
            num = (b1[lvl] - b0[lvl]).float().flatten(1).norm(dim=1)
            den = b0[lvl].float().flatten(1).norm(dim=1).clamp_min(1e-12)
            out.append(dict(modality=m, block=lvl, relative_l2_change=float((num / den).mean()), n=len(indices)))
    return out


def evaluate_leakage(model, source, cfg, out_dir, split: str = "test", device="cpu", n_counterfactual: int = 64):
    """Write ``leakage.csv``, ``counterfactual.csv`` and ``leakage.json`` under ``out_dir``; return the json dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(source.sizes[split], cfg.eval.max_subjects)
    codes, truth, regions = collect_codes(model, source, split, range(n), cfg.eval.positions_per_subject, seed=cfg.seed)
    rows = leakage_rows(codes, truth, cfg.eval, device=device)
    checks = pass_checks(rows, cfg.eval.thresholds, cfg.synth.variant)
    cf = counterfactual_change(model, source, split, range(min(n, n_counterfactual)), seed=cfg.seed)
    for name, table in (("leakage.csv", rows), ("counterfactual.csv", cf), ("checks.csv", checks)):
        with open(out_dir / name, "w", newline="") as fp:
            fields = sorted(
                {k for r in table for k in r}, key=lambda k: list(table[0]).index(k) if k in table[0] else 99
            )
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    result = dict(
        variant=cfg.synth.variant,
        renderer=cfg.synth.renderer,
        split=split,
        n_subjects=n,
        regions=regions,
        rows=rows,
        checks=checks,
        counterfactual=cf,
        fraction=cfg.sampler.L.fraction,
        centring=cfg.loss.centring,
        objective=cfg.loss.objective,
        noise_tau2=cfg.synth.noise_tau2,
    )
    (out_dir / "leakage.json").write_text(json.dumps(result, indent=2))
    return result


def format_matrix(rows, modality: str = "t1") -> str:
    """Plain-text matrix: blocks as rows (plus the raw-image reference when present), factors as columns."""
    factors = ("g", "site", "r", *CELL_FACTORS)
    lines = ["block " + " ".join(f"{f:>7}" for f in factors)]
    for lvl, mod in [(lvl, modality) for lvl in LEVELS] + [("raw", "t1+flair")]:
        vals = {r["factor"]: r["best"] for r in rows if r["modality"] == mod and r["block"] == lvl}
        if vals:
            lines.append(f"{lvl:5} " + " ".join(f"{vals[f]:7.3f}" for f in factors))
    return "\n".join(lines)
