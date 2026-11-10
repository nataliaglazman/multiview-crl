"""Guard checks G1-G6 on the generator; run before any training.

    PYTHONPATH=. python -m experiments.gl_ident.guards --config experiments/gl_ident/configs/e1.yaml

Writes ``results/guards.md`` (relative to this package) and exits non-zero if any guard fails.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import torch

from experiments.gl_ident.config import load_config
from experiments.gl_ident.generator import SyntheticGenerator, cpu_generator

RESULTS = Path(__file__).parent / "results"
RANK_TOL = 1e-4


def _crop_latents(lat, n):
    sl = (slice(None), slice(None)) + (slice(0, n),) * (lat["c"].ndim - 2)
    return {k: (v if k == "gamma" else v[sl]) for k, v in lat.items()}


def _fields_fn(mixer, shapes, view_out=None):
    """z (flat: c, s, gamma) -> mixer pixels (optionally cropped), for jacobians."""
    sizes = [int(torch.tensor(s).prod()) for s in shapes]

    def f(z):
        c, s, g = (part.reshape(shape) for part, shape in zip(torch.split(z, sizes), shapes))
        out = mixer(c.unsqueeze(0), s.unsqueeze(0), g.unsqueeze(0))[0, 0]
        return (out if view_out is None else out[view_out]).reshape(-1)

    return f


def _flat(lat, k, i):
    return torch.cat([lat["c"][i].reshape(-1), lat[f"s{k}"][i].reshape(-1), lat["gamma"][i]])


def g1_whole_field(gen, n_sites, draws=16):
    """min/max singular value of d x_k / d (c, s_k, gamma) on a small periodic grid."""
    lat = _crop_latents(gen.sample_latents(draws, cpu_generator("guard-g1")), n_sites)
    lat = {k: v.double().cpu() for k, v in lat.items()}
    out = {}
    for k in (1, 2):
        mixer = copy.deepcopy(gen.mixers[k - 1]).double().cpu()
        shapes = [tuple(lat["c"].shape[1:]), tuple(lat[f"s{k}"].shape[1:]), tuple(lat["gamma"].shape[1:])]
        f = _fields_fn(mixer, shapes)
        ratios = []
        for i in range(draws):
            J = torch.func.jacrev(f)(_flat(lat, k, i))
            sv = torch.linalg.svdvals(J)
            ratios.append(float(sv.min() / sv.max()))
        out[f"view{k}"] = dict(min_ratio=min(ratios), max_ratio=max(ratios), shape=list(J.shape))
    out["pass"] = all(out[f"view{k}"]["min_ratio"] > RANK_TOL for k in (1, 2))
    return out


def _support_columns(cfg, start, w, d_fields):
    """Column indices (into the flat c, s, gamma vector) of the window's latent support plus gamma."""
    n, dim, r = cfg.n_sites, cfg.dim, cfg.rho_f
    axes = [[(start[a] + o) % n for o in range(-r, w + r)] for a in range(dim)]
    sites = [sum(idx * n ** (dim - 1 - a) for a, idx in enumerate(site)) for site in itertools.product(*axes)]
    cols, offset = [], 0
    for channels in d_fields:  # c then s: (channels, *grid) flattened channel-major
        for ch in range(channels):
            cols += [offset + ch * n**dim + s for s in sites]
        offset += channels * n**dim
    cols += list(range(offset, offset + cfg.d_gamma))
    return cols, len(sites)


def g2_window(gen, draws=16, w=None):
    """Full column rank of window pixels w.r.t. gamma and the latents of the window's support."""
    cfg = gen.cfg
    w = w or 6
    m, dim = cfg.stride, cfg.dim
    lat = {k: v.double().cpu() for k, v in gen.sample_latents(draws, cpu_generator("guard-g2")).items()}
    start = (5,) * dim
    pix = tuple(slice(start[a] * m, (start[a] + w) * m) for a in range(dim))
    out = {}
    for k in (1, 2):
        mixer = copy.deepcopy(gen.mixers[k - 1]).double().cpu()
        shapes = [tuple(lat["c"].shape[1:]), tuple(lat[f"s{k}"].shape[1:]), tuple(lat["gamma"].shape[1:])]
        cols, n_support = _support_columns(cfg, start, w, (cfg.d_c, cfg.d_s))
        f = _fields_fn(mixer, shapes, view_out=pix)
        ratios, outside = [], []
        for i in range(draws):
            J = torch.func.jacrev(f)(_flat(lat, k, i))
            mask = torch.ones(J.shape[1], dtype=torch.bool)
            mask[cols] = False
            outside.append(float(J[:, mask].abs().max()))
            sv = torch.linalg.svdvals(J[:, cols])
            ratios.append(float(sv.min() / sv.max()))
        out[f"view{k}"] = dict(min_ratio=min(ratios), max_ratio=max(ratios), max_abs_outside_support=max(outside))
    n_pix = (w * m) ** dim
    n_lat = len(cols)
    out.update(pixels=n_pix, latents=n_lat, support_sites=n_support, counts_ok=n_pix >= n_lat)
    out["pass"] = out["counts_ok"] and all(out[f"view{k}"]["min_ratio"] > RANK_TOL for k in (1, 2))
    return out


def _site_distance(a, b, n):
    return max(min(abs(x - y), n - abs(x - y)) for x, y in zip(a, b))


def g3_locality(gen, n_sites_tested=8, tol=1e-6):
    """Perturb c (then s_k) at one site: only pixel blocks within rho_f may change."""
    cfg = gen.cfg
    n, m, dim = cfg.n_sites, cfg.stride, cfg.dim
    lat = {k: v.double().cpu()[:1] for k, v in gen.sample_latents(2, cpu_generator("guard-g3")).items()}
    rng = cpu_generator("guard-g3-sites")
    results, ok = [], True
    for k in (1, 2):
        mixer = copy.deepcopy(gen.mixers[k - 1]).double().cpu()
        base = mixer(lat["c"], lat[f"s{k}"], lat["gamma"])
        for field in ("c", f"s{k}"):
            for _ in range(n_sites_tested):
                site = tuple(int(v) for v in torch.randint(0, n, (dim,), generator=rng))
                pert = {key: v.clone() for key, v in lat.items()}
                direction = torch.randn(pert[field].shape[1], generator=rng, dtype=torch.float64)
                pert[field][(0, slice(None)) + site] += direction / direction.norm()
                diff = (mixer(pert["c"], pert[f"s{k}"], pert["gamma"]) - base).abs()[0, 0]
                blocks = diff.reshape(*sum(([n, m] for _ in range(dim)), [])).amax(dim=tuple(range(1, 2 * dim, 2)))
                changed = [tuple(int(i) for i in idx) for idx in (blocks > tol).nonzero()]
                far = max((_site_distance(c, site, n) for c in changed), default=0)
                ok &= far <= cfg.rho_f
                results.append(dict(view=k, field=field, site=site, changed_blocks=len(changed), max_distance=far))
    return dict(
        pass_=ok,
        max_distance=max(r["max_distance"] for r in results),
        n_tests=len(results),
        changed_blocks=sorted({r["changed_blocks"] for r in results}),
    )


def g4_translation(gen, lambda_nu=0.3):
    """Shift every latent field by one site -> the image shifts by m pixels (noise-free; nuisance on)."""
    cfg = gen.cfg
    lat = gen.sample_latents(8, cpu_generator("guard-g4"))
    x = gen.observe(lat, None, snr=float("inf"), lambda_nu=lambda_nu)
    worst = 0.0
    for a in range(cfg.dim):
        axis = 2 + a
        shifted = {k: (v if k == "gamma" else torch.roll(v, 1, dims=axis)) for k, v in lat.items()}
        xs = gen.observe(shifted, None, snr=float("inf"), lambda_nu=lambda_nu)
        for v, vs in zip(x, xs):
            worst = max(worst, float((torch.roll(v, cfg.stride, dims=axis) - vs).abs().max()))
    return dict(max_abs_diff=worst, lambda_nu=lambda_nu, pass_=worst < 1e-5)


def _corr(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def g5_noise(gen, n=256):
    """Noise residuals: uncorrelated across views and between neighbouring pixels."""
    lat = gen.sample_latents(n, cpu_generator("guard-g5"))
    clean = [gen.observe(lat, None, snr=float("inf"), lambda_nu=0.0)[k] for k in (0, 1)]
    out = {}
    for noise, snr in (("gaussian", 10.0), ("rician", 5.0)):
        x = gen.observe(lat, cpu_generator("guard-g5-noise"), snr=snr, lambda_nu=0.0, noise=noise)
        r = [xi - ci for xi, ci in zip(x, clean)]
        cross = _corr(r[0], r[1])
        neighbour = max(
            abs(_corr(rk.narrow(2 + a, 0, rk.shape[2 + a] - 1), rk.narrow(2 + a, 1, rk.shape[2 + a] - 1)))
            for rk in r
            for a in range(gen.cfg.dim)
        )
        out[f"{noise}_snr{snr:g}"] = dict(cross_view=cross, max_neighbour=neighbour)
    out["pass_"] = all(abs(v["cross_view"]) < 0.01 and v["max_neighbour"] < 0.01 for v in out.values())
    return out


def g6_budget(cfg):
    need, have = cfg.d_c + cfg.d_s + cfg.d_gamma, cfg.stride**cfg.dim
    return dict(need=need, have=have, pass_=need <= have)


def c_correlation_by_distance(gen, max_d=10, n=512):
    """Not a guard: corr of c(u) and c(u + d e_1) (same channel), averaged over channels."""
    c = gen.sample_latents(n, cpu_generator("guard-ccorr"))["c"]
    return {
        d: round(sum(_corr(c[:, ch], torch.roll(c[:, ch], d, dims=1)) for ch in range(c.shape[1])) / c.shape[1], 4)
        for d in range(max_d + 1)
    }


def run_guards(cfg, device="cpu"):
    gcfg = cfg.generator
    budget = g6_budget(gcfg)
    if not budget["pass_"]:
        return dict(G6=budget)
    gen = SyntheticGenerator(gcfg, device=device)
    small = 8 if gcfg.dim == 2 else 6
    out = dict(
        G1=g1_whole_field(gen, small),
        G2=g2_window(gen, w=cfg.model.window),
        G3=g3_locality(gen),
        G4=g4_translation(gen),
        G5=g5_noise(gen),
        G6=budget,
        c_corr_by_distance=c_correlation_by_distance(gen),
    )
    for key in ("G1", "G2"):
        out[key]["pass_"] = out[key].pop("pass")
    return out


def passed(out) -> bool:
    return all(v["pass_"] for k, v in out.items() if k.startswith("G"))


def write_markdown(out, cfg, path):
    g = cfg.generator
    lines = [
        "# Guard checks",
        "",
        f"Generator: dim {g.dim}, image {g.image_size}, m {g.stride}, grid {g.n_sites}, "
        f"(d_gamma, d_c, d_s) = ({g.d_gamma}, {g.d_c}, {g.d_s}), hidden {g.hidden}, weights_seed {g.weights_seed}.",
        f"Rank tolerance {RANK_TOL:g} on min/max singular value.",
        "",
        "| guard | result | pass |",
        "|---|---|---|",
    ]
    G1, G2, G3, G4, G5, G6 = (out[k] for k in ("G1", "G2", "G3", "G4", "G5", "G6"))
    small = 8 if g.dim == 2 else 6
    lines.append(
        f"| G1 whole-field injectivity ({small}^{g.dim} grid, 16 draws, J {G1['view1']['shape']}) | "
        f"min sv ratio view1 {G1['view1']['min_ratio']:.3e}, view2 {G1['view2']['min_ratio']:.3e} | {G1['pass_']} |"
    )
    lines.append(
        f"| G2 window injectivity (w = {cfg.model.window}; {G2['pixels']} pixels >= {G2['latents']} latents "
        f"= {G2['support_sites']} sites x {g.d_c + g.d_s} + {g.d_gamma}) | min sv ratio view1 {G2['view1']['min_ratio']:.3e}, "
        f"view2 {G2['view2']['min_ratio']:.3e}; max abs Jacobian outside support "
        f"{max(G2['view1']['max_abs_outside_support'], G2['view2']['max_abs_outside_support']):.1e} | {G2['pass_']} |"
    )
    lines.append(
        f"| G3 locality ({G3['n_tests']} single-site perturbations of c and s_k, tol 1e-6) | farthest changed block "
        f"{G3['max_distance']} site(s) away (rho_f = {g.rho_f}); changed-block counts {G3['changed_blocks']} | {G3['pass_']} |"
    )
    lines.append(
        f"| G4 translation equivariance (one-site roll, lambda_nu = {G4['lambda_nu']}) | max abs diff "
        f"{G4['max_abs_diff']:.2e} | {G4['pass_']} |"
    )
    g5 = "; ".join(
        f"{k}: cross-view {v['cross_view']:+.4f}, max neighbour {v['max_neighbour']:.4f}"
        for k, v in G5.items()
        if k != "pass_"
    )
    lines.append(f"| G5 noise independence (256 subjects, residual x - clean) | {g5} | {G5['pass_']} |")
    lines.append(f"| G6 channel budget | {G6['need']} <= {G6['have']} | {G6['pass_']} |")
    corr = out["c_corr_by_distance"]
    lines += [
        "",
        "Not a guard (context for the window results): correlation of c between sites d apart along one axis,",
        "mean over channels, 512 subjects.",
        "",
        "| d | " + " | ".join(str(d) for d in corr) + " |",
        "|---" * (len(corr) + 1) + "|",
        "| corr | " + " | ".join(f"{v:.3f}" for v in corr.values()) + " |",
        "",
        f"All guards pass: **{passed(out)}**",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="*", default=[str(Path(__file__).parent / "configs" / "e1.yaml")])
    p.add_argument("--out", default=str(RESULTS / "guards.md"))
    args = p.parse_args(argv)
    cfg = load_config(*args.config)
    out = run_guards(cfg)
    write_markdown(out, cfg, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=str))
    print(Path(args.out).read_text())
    return 0 if passed(out) else 1


if __name__ == "__main__":
    sys.exit(main())
