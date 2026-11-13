"""Leakage matrix on the synthetic benchmark: R^2 of each true factor from each learned block.

Readouts (per modality, from the template-subtracted blocks of a frozen model):

* g and site per subject, from the block pooled over every position (site: R^2 on its 2-d embedding, which is
  what enters the renderer, plus logistic balanced accuracy);
* r_R per region, from the block pooled over R (the generator's region, unmerged and uneroded); one probe per
  region, averaged;
* s, w and the lesion indicator per position, from the block averaged over each lattice cell
  (``positions_per_subject`` random cells per subject).

Each R^2 is the mean over subject-grouped folds of ridge and of a 2-layer MLP; ``best`` is the larger.
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


@torch.no_grad()
def collect_codes(model, source, split: str, indices, positions_per_subject: int, batch: int = 8, seed: int = 0):
    """Pooled codes and matching ground truth for subjects ``indices`` of ``split``."""
    model.eval()
    gen_model = source.gen
    labels = torch.from_numpy(source.feature_labels).flatten()
    regions = sorted(int(r) for r in labels.unique() if r > 0)
    dev = gen_model.device
    onehot = torch.stack([(labels == r).float() for r in regions]).to(dev)
    onehot = onehot / onehot.sum(1, keepdim=True)  # [R, P]
    L = gen_model.L
    cells = L**3
    rng = torch.Generator().manual_seed(seed)
    codes = {m: {lvl: {"subject": [], "region": [], "cell": []} for lvl in LEVELS} for m in MODALITIES}
    truth = {k: [] for k in ("g", "site", "site_emb", "r", *CELL_FACTORS)}
    for start in range(0, len(indices), batch):
        chunk = list(indices[start : start + batch])
        data = source.batch(split, chunk)
        lat = data["latents"]
        n = len(chunk)
        pick = torch.stack([torch.randperm(cells, generator=rng)[:positions_per_subject] for _ in range(n)]).to(dev)
        for m in MODALITIES:
            blocks = model.blocks(data[m].to(next(model.parameters()).device), m)
            for lvl, f in blocks.items():
                f = f.float().to(dev)
                codes[m][lvl]["subject"].append(f.mean((2, 3, 4)).cpu())
                codes[m][lvl]["region"].append((f.flatten(2) @ onehot.T).permute(0, 2, 1).cpu())
                pooled = F.adaptive_avg_pool3d(f, L).flatten(2)  # [n, d, L^3]
                codes[m][lvl]["cell"].append(
                    torch.gather(pooled, 2, pick[:, None].expand(-1, f.shape[1], -1)).permute(0, 2, 1).cpu()
                )
        truth["g"].append(lat["g"].cpu())
        truth["site"].append(lat["site"].cpu())
        truth["site_emb"].append(lat["site_emb"].cpu())
        truth["r"].append(lat["r"][:, [r - 1 for r in regions]].cpu())
        for k in CELL_FACTORS:
            v = lat[k].flatten(2)
            truth[k].append(torch.gather(v, 2, pick[:, None].expand(-1, v.shape[1], -1)).permute(0, 2, 1).cpu())
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    codes = {m: {lvl: {k: cat(v) for k, v in d.items()} for lvl, d in md.items()} for m, md in codes.items()}
    return codes, {k: cat(v) for k, v in truth.items()}, regions


def leakage_rows(codes, truth, eval_cfg, device="cpu"):
    rows = []
    n = truth["g"].shape[0]
    subj = np.arange(n)
    for m in MODALITIES:
        for lvl in LEVELS:
            c = codes[m][lvl]
            row = lambda factor, res, **extra: rows.append(
                dict(modality=m, block=lvl, factor=factor, **res, **extra)
            )  # noqa: E731
            row("g", cv_r2(c["subject"], truth["g"], subj, eval_cfg, device=device))
            site_r2 = cv_r2(c["subject"], truth["site_emb"], subj, eval_cfg, device=device)
            row(
                "site",
                site_r2,
                site_balanced_accuracy=cv_balanced_accuracy(c["subject"], truth["site"], subj, eval_cfg.cv_folds),
            )
            per_region = [
                cv_r2(c["region"][:, j], truth["r"][:, j], subj, eval_cfg, device=device)
                for j in range(truth["r"].shape[1])
            ]
            res = {k: float(np.mean([p[k] for p in per_region])) for k in ("ridge", "mlp", "best")}
            row("r", res, r_best_min_region=float(np.min([p["best"] for p in per_region])))
            k = c["cell"].shape[1]
            groups = np.repeat(subj, k)
            x = c["cell"].reshape(n * k, -1)
            for factor in CELL_FACTORS:
                y = truth[factor].reshape(n * k, -1)
                row(factor, cv_r2(x, y, groups, eval_cfg, device=device))
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
    """Plain-text matrix: blocks as rows, factors as columns, ``best`` R^2."""
    factors = ("g", "site", "r", *CELL_FACTORS)
    lines = ["block " + " ".join(f"{f:>7}" for f in factors)]
    for lvl in LEVELS:
        vals = {r["factor"]: r["best"] for r in rows if r["modality"] == modality and r["block"] == lvl}
        lines.append(f"{lvl:5} " + " ".join(f"{vals[f]:7.3f}" for f in factors))
    return "\n".join(lines)
