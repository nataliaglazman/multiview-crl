"""Evaluation (spec section 5): global metrics M1-M4 and local metrics M5-M10 on the fixed test set.

Regressors are fitted on test subjects [0, n_fit) and scored on [n_fit, n_test). Early stopping and the ridge
penalty are chosen on a 10% subject-level split of the fit half. R^2 is variance-weighted over outputs.

    PYTHONPATH=. python -m experiments.gl_ident.evaluate --run-dir experiments/gl_ident/results/runs/<run>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from experiments.gl_ident.config import RunConfig, load_config
from experiments.gl_ident.generator import SyntheticGenerator, cpu_generator
from experiments.gl_ident.losses import crop_windows, gather_sites
from experiments.gl_ident.models import TwoView, normalise

# ----------------------------------------------------------------------------------------------- regressors


def r2_vw(y, yhat):
    return float(1 - (y - yhat).square().sum() / (y - y.mean(0)).square().sum())


def _group_split(groups, frac, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    val_groups = rng.permutation(uniq)[: max(1, int(round(frac * len(uniq))))]
    val = np.isin(groups, val_groups)
    return np.flatnonzero(~val), np.flatnonzero(val)


def _standardise(fit, score):
    mu, sd = fit.mean(0), fit.std(0).clamp_min(1e-8)
    return (fit - mu) / sd, (score - mu) / sd


def ridge_predict(Xf, Yf, Xp, groups, seed=0, alphas=10.0 ** np.arange(-6, 3)):
    """Ridge with the penalty chosen on a subject-level 10% split of the fit rows; predictions for Xp."""
    Xf, Xp = _standardise(Xf.double(), Xp.double())
    Yf = Yf.double()
    ymu = Yf.mean(0)
    tr, va = _group_split(groups, 0.1, seed)
    eye = torch.eye(Xf.shape[1], dtype=Xf.dtype)

    def solver(X, Y):
        gram, cross, n = X.T @ X, X.T @ (Y - ymu), X.shape[0]
        return lambda alpha: torch.linalg.solve(gram + alpha * n * eye, cross)

    on_train = solver(Xf[tr], Yf[tr])
    scores = [r2_vw(Yf[va], Xf[va] @ on_train(a) + ymu) for a in alphas]
    alpha = alphas[int(np.argmax(scores))]
    return (Xp @ solver(Xf, Yf)(alpha) + ymu).float()


def mlp_predict(Xf, Yf, Xp, groups, ecfg, seed=0, device="cpu"):
    """2 x hidden ReLU MLP, Adam, early stopping on a subject-level 10% split of the fit rows; predictions for Xp."""
    Xf, Xp = _standardise(Xf.float(), Xp.float())
    ymu, ysd = Yf.float().mean(0), Yf.float().std(0).clamp_min(1e-8)
    Yfs = (Yf.float() - ymu) / ysd
    tr, va = _group_split(groups, 0.1, seed)
    Xtr, Ytr = Xf[tr].to(device), Yfs[tr].to(device)
    Xva, Yva = Xf[va].to(device), Yfs[va].to(device)
    torch.manual_seed(seed)
    h = ecfg.mlp_hidden
    net = nn.Sequential(nn.Linear(Xf.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, Yf.shape[1]))
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = int(np.clip(len(tr) // 20, 32, 512))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(ecfg.mlp_max_epochs):
        perm = torch.randperm(len(tr), generator=gen).to(device)
        net.train()
        for i in range(0, len(tr), batch):
            idx = perm[i : i + batch]
            loss = F.mse_loss(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(F.mse_loss(net(Xva), Yva))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= ecfg.mlp_patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        pred = torch.cat([net(Xp[i : i + 8192].to(device)).cpu() for i in range(0, len(Xp), 8192)])
    return pred * ysd + ymu


def _predictors(ecfg, device, seed=0):
    return dict(
        ridge=lambda Xf, Yf, Xp, g: ridge_predict(Xf, Yf, Xp, g, seed),
        mlp=lambda Xf, Yf, Xp, g: mlp_predict(Xf, Yf, Xp, g, ecfg, seed, device),
    )


def ridge_r2(Xf, Yf, Xs, Ys, groups, seed=0):
    return r2_vw(Ys.float(), ridge_predict(Xf, Yf, Xs, groups, seed))


def regress(X, Y, groups, n_fit_rows, ecfg, device, seed=0):
    """{'ridge': R^2, 'mlp': R^2} for Y <- X, fit on the first n_fit_rows rows, scored on the rest."""
    Xf, Xs, Yf, Ys = X[:n_fit_rows], X[n_fit_rows:], Y[:n_fit_rows], Y[n_fit_rows:]
    g = groups[:n_fit_rows]
    return {k: r2_vw(Ys.float(), f(Xf, Yf, Xs, g)) for k, f in _predictors(ecfg, device, seed).items()}


def residual_leak(gamma, Z, ghat, n_fit, ecfg, device, folds=5):
    """Not in the spec. How much of g-hat the nuisance block Z explains beyond gamma.

    e = g-hat - f(gamma), with f cross-fitted (K folds) on the fit half so its residuals are not shrunk by
    overfitting, and f fitted on the whole fit half for the score half. Then Z -> e is fitted on the fit half
    and scored on the score half. ``r2`` is that R^2 (about 0 or below when nothing leaks); ``share`` rescales it
    to a fraction of g-hat's variance on the score half.
    """
    out = {}
    rows = np.arange(n_fit)
    split = np.array_split(np.random.default_rng(0).permutation(n_fit), folds)
    ghat, gamma, Z = ghat.float(), gamma.float(), Z.float()
    for name, predict in _predictors(ecfg, device).items():
        e_fit = torch.empty_like(ghat[:n_fit])
        for idx in split:
            tr = np.setdiff1d(rows, idx)
            e_fit[idx] = ghat[idx] - predict(gamma[tr], ghat[tr], gamma[idx], np.arange(len(tr)))
        e_score = ghat[n_fit:] - predict(gamma[:n_fit], ghat[:n_fit], gamma[n_fit:], rows)
        r2 = r2_vw(e_score, predict(Z[:n_fit], e_fit, Z[n_fit:], rows))
        g_s = ghat[n_fit:]
        share = r2 * float((e_score - e_score.mean(0)).square().sum() / (g_s - g_s.mean(0)).square().sum())
        out[name] = dict(r2=r2, share=share)
    return out


# ----------------------------------------------------------------------------------------------- encoding


def window_starts(n_sites, w, dim, n_windows, seed):
    """Tiling starts (0, w, 2w, ...)^dim; if there are more than n_windows, a fixed random subset."""
    tiles = torch.tensor(list(itertools.product(range(0, n_sites - w + 1, w), repeat=dim)))
    if len(tiles) > n_windows:
        tiles = tiles[torch.randperm(len(tiles), generator=torch.Generator().manual_seed(seed))[:n_windows]]
    return tiles


@torch.no_grad()
def encode(model, cfg: RunConfig, x1, x2, chunk=250):
    """local: per-site codes (n, d, *N) per view. global/pooled: per-view subject vectors (n, d)."""
    sim, g = cfg.loss.similarity, cfg.generator
    model.eval()
    dev = next(model.parameters()).device
    outs = [[], []]
    starts = None
    if model.kind == "global":
        starts = window_starts(g.n_sites, cfg.model.window, g.dim, cfg.eval.n_windows, cfg.eval.seed)
    for i in range(0, x1.shape[0], chunk):
        for k, x in enumerate((x1, x2)):
            xb = x[i : i + chunk].to(dev)
            if model.kind == "local":
                outs[k].append(normalise(model(xb, k), sim, dim=1).cpu())
            elif model.kind == "global":
                per_window = []
                for s in starts:
                    crop = crop_windows(xb, s.expand(xb.shape[0], -1), cfg.model.window, g.stride)
                    per_window.append(normalise(model(crop, k), sim, dim=1))
                outs[k].append(torch.stack(per_window, 1).mean(1).cpu())
            else:
                outs[k].append(normalise(model(xb, k), sim, dim=1).cpu())
    return torch.cat(outs[0]), torch.cat(outs[1])


def neighbourhood(field, sites, r):
    """field (n, d, *N), sites (n, S, dim) -> (n, S, d * (2r+1)^dim): the circular B_r(u) neighbourhood."""
    dim, n_grid = sites.shape[-1], field.shape[2]
    parts = []
    for off in itertools.product(range(-r, r + 1), repeat=dim):
        shifted = (sites + torch.tensor(off)) % n_grid
        parts.append(gather_sites(field, shifted))
    return torch.cat(parts, dim=-1)


def channel_corr(a, b):
    """Per-channel Pearson correlation; a, b (rows, d)."""
    a, b = a.double() - a.double().mean(0), b.double() - b.double().mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)


# ----------------------------------------------------------------------------------------------- metrics


def global_metrics(g1, g2, lat, cfg: RunConfig, device):
    """M1-M4 (+ per-channel cross-view correlation of the two per-view means)."""
    ecfg, gc = cfg.eval, cfg.generator
    n, nf = g1.shape[0], ecfg.n_fit
    groups = np.arange(n)
    ghat = 0.5 * (g1 + g2)
    gamma = lat["gamma"].cpu()

    def pcs(field, k):
        flat = field.cpu().reshape(n, -1).numpy()
        flat = flat.astype(np.float64)
        return torch.from_numpy(PCA(k, svd_solver="full").fit(flat[:nf]).transform(flat)).float()

    Z = torch.cat([pcs(lat["c"], 32), pcs(lat["s1"], 16), pcs(lat["s2"], 16)], dim=1)
    nuisance = torch.cat([gamma, Z], dim=1)
    nu_low = F.adaptive_avg_pool2d(lat["nu"].cpu(), 2) if gc.dim == 2 else F.adaptive_avg_pool3d(lat["nu"].cpu(), 2)
    nu_low = nu_low.reshape(n, -1)

    out = dict(
        M1=regress(ghat, gamma, groups, nf, ecfg, device),
        M2_pure=regress(gamma, ghat, groups, nf, ecfg, device),
        M2_full=regress(nuisance, ghat, groups, nf, ecfg, device),
        M4=regress(ghat, nu_low, groups, nf, ecfg, device),
    )
    out["M2_gap"] = {k: out["M2_full"][k] - out["M2_pure"][k] for k in ("ridge", "mlp")}
    out["M2_resid"] = residual_leak(gamma, Z, ghat, nf, ecfg, device)
    fwd, bwd = regress(g1, g2, groups, nf, ecfg, device), regress(g2, g1, groups, nf, ecfg, device)
    out["M3"] = {k: 0.5 * (fwd[k] + bwd[k]) for k in fwd}
    cc = channel_corr(g1, g2)
    out["C_ii_global"] = dict(mean=float(cc.mean()), min=float(cc.min()))
    return out


def local_metrics(z1, z2, lat, cfg: RunConfig, device):
    """M5-M10 per view (and the mean over views)."""
    ecfg, gc = cfg.eval, cfg.generator
    R = gc.rho_f + cfg.model.n_g
    r_max = ecfg.r_max if ecfg.r_max is not None else R + 2
    n = z1.shape[0]
    S = ecfg.sites_per_subject
    sites = torch.randint(0, gc.n_sites, (n, S, gc.dim), generator=cpu_generator("eval-sites", ecfg.seed))
    groups = np.repeat(np.arange(n), S)
    nfr = ecfg.n_fit * S
    rows = lambda t: t.reshape(n * S, -1)  # noqa: E731
    c, gamma = lat["c"].cpu(), lat["gamma"].cpu()
    gamma_rows = gamma[:, None, :].expand(-1, S, -1).reshape(n * S, -1)
    c_here = rows(gather_sites(c, sites))
    nu_here = rows(gather_sites(lat["nu"].cpu(), sites))
    c_nb = {r: rows(neighbourhood(c, sites, r)) for r in range(r_max + 1)}

    per_view = {}
    for k, z in ((1, z1), (2, z2)):
        z_here = rows(gather_sites(z, sites))
        m = {}
        for r in range(r_max + 1):
            m[f"M5_r{r}"] = regress(torch.cat([c_nb[r], gamma_rows], 1), z_here, groups, nfr, ecfg, device)
            m[f"M7_r{r}"] = regress(rows(neighbourhood(z, sites, r)), c_here, groups, nfr, ecfg, device)
        s_nb = rows(neighbourhood(lat[f"s{k}"].cpu(), sites, R))
        with_style = regress(torch.cat([c_nb[R], gamma_rows, s_nb], 1), z_here, groups, nfr, ecfg, device)
        m["M5_style_rR"] = with_style
        m["M6"] = {key: with_style[key] - m[f"M5_r{R}"][key] for key in with_style}
        m["M8"] = {key: dict(r0=m["M5_r0"][key], rR=m[f"M5_r{R}"][key]) for key in with_style}
        m["M10"] = regress(z_here, nu_here, groups, nfr, ecfg, device)
        per_view[f"view{k}"] = m

    out = {"R": R, "r_max": r_max, "per_view": per_view}
    for key in per_view["view1"]:
        if key == "M8":
            continue
        out[key] = {reg: 0.5 * (per_view["view1"][key][reg] + per_view["view2"][key][reg]) for reg in ("ridge", "mlp")}
    out["M8"] = {reg: dict(r0=out["M5_r0"][reg], rR=out[f"M5_r{R}"][reg]) for reg in ("ridge", "mlp")}
    flat = lambda z: z.movedim(1, -1).reshape(-1, z.shape[1])  # noqa: E731
    cc = channel_corr(flat(z1), flat(z2))
    out["M9"] = dict(mean=float(cc.mean()), min=float(cc.min()), per_channel=[float(v) for v in cc])
    return out


def test_data(gen: SyntheticGenerator):
    lat = gen.split("test")
    x1, x2 = gen.split_images("test", lat)
    return lat, x1, x2


def evaluate(model, cfg: RunConfig, gen: SyntheticGenerator, data=None, device="cpu"):
    lat, x1, x2 = data if data is not None else test_data(gen)
    z1, z2 = encode(model, cfg, x1, x2)
    if model.kind == "local":
        return local_metrics(z1, z2, lat, cfg, device)
    return global_metrics(z1, z2, lat, cfg, device)


def headline(metrics: dict, kind: str) -> dict:
    """Flat {name: value} for the summary CSV."""
    out = {}
    if kind == "local":
        for key, val in metrics.items():
            if key.startswith(("M5", "M6", "M7", "M10")):
                out.update({f"{key}_{reg}": v for reg, v in val.items()})
        out["M9_mean"], out["M9_min"] = metrics["M9"]["mean"], metrics["M9"]["min"]
        out["R"] = metrics["R"]
    else:
        for key in ("M1", "M2_pure", "M2_full", "M2_gap", "M3", "M4"):
            out.update({f"{key}_{reg}": v for reg, v in metrics[key].items()})
        for reg, val in metrics["M2_resid"].items():
            out[f"M2_resid_r2_{reg}"], out[f"M2_resid_share_{reg}"] = val["r2"], val["share"]
        out["C_ii_global_mean"] = metrics["C_ii_global"]["mean"]
        out["C_ii_global_min"] = metrics["C_ii_global"]["min"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    from experiments.gl_ident.train import resolve_device

    run = Path(args.run_dir)
    cfg = load_config(run / "config.yaml")
    device = resolve_device(args.device)
    gen = SyntheticGenerator(cfg.generator, device=device)
    model = TwoView(cfg).to(device)
    model.load_state_dict(torch.load(run / args.checkpoint, map_location=device)["model"])
    metrics = evaluate(model, cfg, gen, device=device)
    (run / f"metrics_{Path(args.checkpoint).stem}.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(headline(metrics, cfg.model.kind), indent=1))


if __name__ == "__main__":
    main()
