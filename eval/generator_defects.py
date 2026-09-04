"""Five *generator-side* identifiability defects, each converted into a measurement.

None of these need a checkpoint. They are properties of the data-generating process, so
they bound what ANY encoder can recover — which makes them the right thing to rule out
before reading another per-factor R^2 as a model result.

  1. SQUASH        The renderer passes every content factor through `_sq`: a hard
                   clamp to [-1, 1] normally, tanh under `--synthetic-clean-content`.
                   z ~ N(0,1) puts ~32% of draws outside [-1,1], and the clamp maps all
                   of them to the same value — non-injective, so those samples are
                   unrecoverable in principle. tanh is injective and fixes that for a
                   NONLINEAR probe, but the headline metrics (`cv_probe_r2`, `block_mcc`)
                   default to `kind="ridge"` = LINEAR, and tanh is just as nonlinear as
                   the clamp is lossy, so the linear ceiling barely moves. Reported here
                   for both probe classes, plus a U(-1,1) content prior where `_sq` is
                   the identity and no ceiling exists at all.

  2. AMPLITUDE     Several factors move a tissue boundary by ~1 voxel (voxel = 2/res =
                   0.0625 at res=32) before a 3^3 average-pool blur and Rician noise.
                   Whether that is "too small" depends entirely on the probe: it is
                   measured here both per-voxel (what a local/conv feature sees) and as
                   a matched filter over the whole volume (the optimal-probe bound).
                   The noise floor is measured, not assumed white — the bias field and
                   the blur correlate it, so the sqrt(N) white-noise gain is an overestimate.

  3. STYLE NORM    `lut = base*gain + bias` is affine in intensity, and `per_sample`
                   normalization z-scores each volume over its foreground, which removes
                   an affine map. So gain/bias should be largely erased and only the
                   noise-sigma dim should survive. Measured as the post-normalization
                   render sensitivity per style dim, under each `--synthetic-normalize`
                   mode.

  4. INERT DIMS    In `lesion_mode="field"` the lesion is drawn from `z_lesion` and
                   `z_content[2:5]` are never read (the renderer's own comment says so).
                   In `lesion_mode="sphere"` they are live but coarse. Settled by
                   sweeping those dims and diffing renders in both modes.

  5. DEGENERACY    `radii_gm = radii_wm + cortical_thickness` = 0.65 + 0.1*s(z0) +
                   0.06*s(z5), so brain_size and cortical_thickness act IDENTICALLY on
                   the outer GM boundary and separate only at the WM/GM boundary. The
                   prediction is mutual confusability, not individual weakness — a
                   rendering artifact that would otherwise read as a model failure.
                   Measured as the cosine between per-dim render-Jacobian directions and
                   the unique (non-explainable-by-others) variance share of each dim.

Usage
-----
    python -m eval.generator_defects --tests all
    python -m eval.generator_defects --tests squash --causal --clean-content
    python -m eval.generator_defects --tests amplitude style --normalize per_sample

Defaults mirror the *shipped defaults* (`utils/config.py`), NOT the flagship configs:
`--synthetic-clean-content` is off and `--synthetic-normalize per_sample`. Pass
`--preset flagship` to measure what `experiments/synthetic_causal.yaml` actually trains on.
"""

import argparse
import warnings

import numpy as np
import torch

from data.datasets import SyntheticBrainDataset
from eval.synthetic_dataset import sample_content_from_scm

CONTENT_NAMES = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]
STYLE_NAMES = ["gain", "bias", "noise_sigma"]

# Renderer amplitudes, per content dim, as written in render_structure. Used only to
# report the boundary displacement in voxels — the measurements below never assume them.
CONTENT_AMPS = [0.1, 0.05, None, None, None, 0.06, 0.12, 0.08, 0.06]


def build_dataset(args, normalize=None, quiet=False):
    """`quiet=True` for tests that deliberately enumerate the flagged configurations.

    Tests 3 and 4 construct `per_sample`/`shared` and `lesion_mode="field"` on purpose —
    that IS the measurement — so `SyntheticBrainDataset`'s warnings about them would fire
    on every row and interleave with the table. They stay on everywhere else.
    """
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    if quiet:
        warnings.simplefilter("ignore", UserWarning)
    try:
        return _build_dataset(args, normalize)
    finally:
        ctx.__exit__(None, None, None)


def _build_dataset(args, normalize=None):
    return SyntheticBrainDataset(
        mode="train",
        spatial_size=(args.res,) * 3,
        synthetic_num_samples=max(args.n_samples, 64),
        synthetic_n_content=args.n_content,
        synthetic_n_style=3,
        synthetic_normalize=normalize or args.normalize,
        synthetic_causal=args.causal,
        synthetic_clean_content=args.clean_content,
        synthetic_lesion_mode=args.lesion_mode,
        synthetic_identifiable_ventricle=args.identifiable_ventricle,
        synthetic_content_scale=args.content_scale,
        synthetic_style_scale=args.style_scale,
        synthetic_seed=args.seed,
        synthetic_content_prior=args.content_prior,
        synthetic_content_squash=args.content_squash,
        synthetic_content_amp_scale=args.content_amp_scale,
        synthetic_lesion_radius=args.lesion_radius,
        synthetic_cortex_parameterization=args.cortex_parameterization,
        synthetic_center_local_deformations=args.center_local_deformations,
        synthetic_csf_t1_intensity=getattr(args, "csf_t1_intensity", 0.1),
    )


def draw_content(inner, idx):
    """The content vector `_pseudo_mri_item` would draw for `idx`, without rendering.

    Duplicates four lines of the sampler, so `verify_content_sampler` asserts it against
    the real item on every run — a silent drift here would corrupt every ceiling below.
    """
    gen = torch.Generator().manual_seed(inner.sample_seed_for(idx))
    uniform = getattr(inner, "content_prior", "normal") == "uniform"
    if inner.causal:
        z = sample_content_from_scm(inner.scm, gen, inner.causal_noise_scale, inner.causal_nonlinearity)
        if uniform:
            z = inner._pit_to_uniform(
                z, lambda g: sample_content_from_scm(inner.scm, g, inner.causal_noise_scale, inner.causal_nonlinearity)
            )
        return z
    if inner.hierarchical_content:
        z = inner._sample_hierarchical_content(gen)[0]
        return inner._pit_to_uniform(z, lambda g: inner._sample_hierarchical_content(g)[0]) if uniform else z
    if uniform:
        return torch.rand(inner.n_content, generator=gen) * 2.0 - 1.0
    return torch.randn(inner.n_content, generator=gen)


def verify_content_sampler(inner, n=8):
    for i in range(n):
        got = inner[i][2]["z_content"]
        want = draw_content(inner, i)
        if not torch.allclose(got, want):
            raise AssertionError(f"draw_content drifted from _pseudo_mri_item at idx={i}: {got} vs {want}")


def render(ds, lat, sample_seed, normalize=True, **overrides):
    """Render + normalize one view pair, exactly as training would see it.

    `normalize=False` returns the raw render. Note this is NOT reachable by passing a
    bogus `synthetic_normalize`: `normalize_views` falls through to `_znorm_nonzero`
    (i.e. per_sample) on any unrecognised mode.
    """
    inner = ds._inner
    z = {k: lat.get(k) for k in ("z_content", "z_deformation", "z_fissure", "z_style_v1", "z_style_v2", "z_lesion")}
    z.update(overrides)
    x1, x2, mask = inner.render_pseudo_mri(
        z["z_content"],
        z["z_deformation"],
        z["z_fissure"],
        z["z_style_v1"],
        z["z_style_v2"],
        sample_seed,
        z_lesion=z.get("z_lesion"),
    )
    if normalize:
        x1, x2 = ds.normalize_views(x1, x2, mask, mask.clone())
    return x1, x2, mask


# ─────────────────────────────── 1. squash ceiling ───────────────────────────────


def _clamp_ceilings(z):
    """(linear R^2, nonlinear R^2, saturated fraction) for recovering z from clamp(z)."""
    c = np.clip(z, -1, 1)
    sat_hi, sat_lo = z > 1, z < -1
    lin = np.corrcoef(z, c)[0, 1] ** 2 if c.std() > 0 else 0.0
    # The optimal nonlinear inverse is E[z | clamp(z)]: identity on the interior,
    # the conditional mean of each saturated tail. Residual variance is what is lost.
    resid = np.zeros_like(z)
    for m in (sat_hi, sat_lo):
        if m.any():
            resid[m] = z[m] - z[m].mean()
    nonlin = 1.0 - resid.var() / z.var()
    return lin, nonlin, float(sat_hi.mean() + sat_lo.mean())


def test_squash(args):
    ds = build_dataset(args)
    inner = ds._inner
    verify_content_sampler(inner)
    Z = np.stack([draw_content(inner, i).numpy() for i in range(args.n_squash)])

    active = inner.renderer.content_squash
    if active == "auto":
        active = f"tanh (auto)" if inner.clean_content else "clamp (auto)"
    print(
        f"\n[1] SQUASH CEILING   n={args.n_squash}  active _sq = {active}  "
        f"(clean_content={inner.clean_content}, prior={inner.content_prior})"
    )
    print("     ceiling = best achievable R^2(z_true | squashed z), i.e. an upper bound on any probe\n")
    print(
        f"    {'dim':<3} {'name':<20} {'std':>6} {'|z|>1':>7} | {'clamp lin':>9} {'clamp non':>9} | "
        f"{'tanh lin':>8} {'tanh non':>8} | {'U(-1,1)':>8}"
    )
    for d in range(Z.shape[1]):
        z = Z[:, d]
        lin_c, non_c, sat = _clamp_ceilings(z)
        t = np.tanh(z)
        lin_t = np.corrcoef(z, t)[0, 1] ** 2
        star = "  <=" if sat > 0.15 else ""
        print(
            f"    {d:<3} {CONTENT_NAMES[d] if d < len(CONTENT_NAMES) else '?':<20} {z.std():>6.2f} {sat:>6.1%} | "
            f"{lin_c:>9.3f} {non_c:>9.3f} | {lin_t:>8.3f} {'1.000':>8} | {'1.000':>8}{star}"
        )
    print("\n    tanh is injective -> nonlinear ceiling is exactly 1.0; its LINEAR ceiling is no")
    print("    better than the clamp's. cv_probe_r2 / block_mcc default to kind='ridge' (linear),")
    print("    so switching clamp->tanh does not lift the headline ceiling. A U(-1,1) content")
    print("    prior makes _sq the identity and removes the ceiling for every probe class.")


# ─────────────────────────── 2/3. render sensitivity ────────────────────────────


def _noise_basis(ds, lat, seeds, normalize=True):
    """Renders of ONE fixed latent draw under K different rendering seeds."""
    return torch.stack([render(ds, lat, s, normalize=normalize)[0].flatten() for s in seeds])


def _sensitivity(ds, lat, sample_seed, key, resample, noise_stack, mask, normalize=True):
    """RMS and matched-filter SNR of resampling one latent component.

    Signal: the render difference from redrawing that component with everything else,
    INCLUDING the rendering seed, held fixed. Noise: the spread of `noise_stack`, i.e.
    the same latents rendered under different seeds (Rician + bias field). The matched
    filter projects onto the signal direction and uses the EMPIRICAL noise spread along
    it, so spatial correlation from the blur and the bias field is accounted for.
    """
    base = render(ds, lat, sample_seed, normalize=normalize)[0].flatten()
    deltas = []
    for alt in resample:
        pert = render(ds, lat, sample_seed, normalize=normalize, **{key: alt})[0].flatten()
        deltas.append(pert - base)
    delta = torch.stack(deltas)

    # Full volume for BOTH terms, deliberately. Restricting to the base brain mask would
    # discard exactly the voxels a boundary-moving factor lights up (they are background
    # before the perturbation and tissue after), which is most of brain_size's signal.
    # Under every real normalize mode the background is exactly 0 in both renders, so the
    # extra voxels contribute nothing to either term.
    sig_rms = float(delta.pow(2).mean().sqrt())
    noise_delta = noise_stack - noise_stack.mean(0, keepdim=True)
    noise_rms = float(noise_delta.pow(2).mean().sqrt())

    u = delta.mean(0)
    nu = u.norm()
    if nu < 1e-12:
        return sig_rms, noise_rms, 0.0, 0.0
    u = u / nu
    proj_sig = float((delta @ u).abs().mean())
    proj_noise = float((noise_delta @ u).std())
    mf = proj_sig / proj_noise if proj_noise > 1e-12 else float("inf")
    return sig_rms, noise_rms, sig_rms / noise_rms if noise_rms > 0 else float("inf"), mf


def _run_sensitivity(ds, args, which, normalize=True):
    """which: 'content' or 'style'. Returns list of (dim, name, rms_ratio, mf_snr)."""
    inner = ds._inner
    key = "z_content" if which == "content" else "z_style_v1"
    names = CONTENT_NAMES if which == "content" else STYLE_NAMES
    ndim = inner.n_content if which == "content" else inner.n_style

    rows = []
    per_dim = {d: ([], []) for d in range(ndim)}
    for i in range(args.n_samples):
        lat = inner[i][2]
        seed = inner.sample_seed_for(i)
        mask = render(ds, lat, seed, normalize=normalize)[2]
        noise_stack = _noise_basis(ds, lat, [seed + 7919 * k for k in range(args.n_noise)], normalize=normalize)
        for d in range(ndim):
            alts = []
            for r in range(args.n_resample):
                alt = lat[key].clone()
                other = draw_content(inner, 10_000 + i * 97 + r * 13) if which == "content" else None
                if other is not None:
                    alt[d] = other[d]
                else:
                    g = torch.Generator().manual_seed(seed + 31 * r + d)
                    alt[d] = torch.randn(1, generator=g)[0]
                alts.append(alt)
            _, _, ratio, mf = _sensitivity(ds, lat, seed, key, alts, noise_stack, mask, normalize=normalize)
            per_dim[d][0].append(ratio)
            per_dim[d][1].append(mf)
    for d in range(ndim):
        rows.append(
            (d, names[d] if d < len(names) else "?", float(np.mean(per_dim[d][0])), float(np.mean(per_dim[d][1])))
        )
    return rows


def test_amplitude(args):
    ds = build_dataset(args)
    vox = 2.0 / args.res
    print(
        f"\n[2] AMPLITUDE vs NOISE FLOOR   res={args.res} (voxel={vox:.4f})  normalize={ds.synthetic_normalize}  "
        f"n={args.n_samples} x {args.n_resample} resamples, {args.n_noise} noise draws"
    )
    print("     rms_ratio  = per-voxel signal / per-voxel rendering noise  (what a LOCAL feature sees)")
    print("     mf_snr     = matched filter over the volume, empirical noise spread (OPTIMAL probe bound)\n")
    rows = _run_sensitivity(ds, args, "content")
    print(f"    {'dim':<3} {'name':<20} {'ampl(vox)':>10} {'rms_ratio':>10} {'mf_snr':>9}   verdict")
    for d, name, ratio, mf in rows:
        amp = CONTENT_AMPS[d] if d < len(CONTENT_AMPS) else None
        astr = f"{amp * args.content_scale / vox:>10.2f}" if amp else f"{'-':>10}"
        if mf < 3:
            v = "BELOW noise floor"
        elif ratio < 0.5:
            v = "sub-voxel; needs pooling"
        else:
            v = "ok"
        print(f"    {d:<3} {name:<20} {astr} {ratio:>10.3f} {mf:>9.1f}   {v}")
    print("\n    ampl(vox) is the boundary displacement at |s(z)|=1, in voxels.")


def _style_blindness(ds, args, normalize=True):
    """How much of the input survives swapping the style draw at FIXED anatomy?

    The direct form of the question. If per-sample normalization removes the affine
    intensity map that `lut = base*gain + bias` applies, then two renders of the same
    anatomy under different style draws collapse onto (nearly) the same normalized
    volume, and `resid` -> 0. Reported relative to the volume's own contrast, so it is
    a fraction, not an intensity.
    """
    inner = ds._inner
    resids, corrs = [], []
    for i in range(args.n_samples):
        lat = inner[i][2]
        seed = inner.sample_seed_for(i)
        g = torch.Generator().manual_seed(seed + 555)
        sa = torch.randn(inner.n_style, generator=g)
        sb = torch.randn(inner.n_style, generator=g)
        xa = render(ds, lat, seed, normalize=normalize, z_style_v1=sa)[0].flatten()
        xb = render(ds, lat, seed, normalize=normalize, z_style_v1=sb)[0].flatten()
        resids.append(float((xa - xb).norm() / (xa - xa.mean()).norm()))
        corrs.append(float(torch.corrcoef(torch.stack([xa, xb]))[0, 1]))
    return float(np.mean(resids)), float(np.mean(corrs))


def test_style(args):
    print(f"\n[3] STYLE UNDER NORMALIZATION   n={args.n_samples}")
    print("     Same measurement as [2], per style dim, under each --synthetic-normalize mode.\n")
    for mode in ("raw (none)", "per_sample", "shared", "fixed_reference"):
        norm = mode != "raw (none)"
        ds = build_dataset(args, normalize=mode if norm else "per_sample", quiet=True)
        rows = _run_sensitivity(ds, args, "style", normalize=norm)
        cells = "  ".join(f"{n}: rms {r:.3f} / mf {m:>8.1f}" for _, n, r, m in rows)
        print(f"    {mode:<16} {cells}")

    print("\n     Direct form — same anatomy, two style draws, same rendering seed:")
    print(f"    {'mode':<16} {'residual/contrast':>18} {'corr(x_a, x_b)':>16}   style information left in the input")
    for mode in ("raw (none)", "per_sample", "shared", "fixed_reference"):
        norm = mode != "raw (none)"
        ds = build_dataset(args, normalize=mode if norm else "per_sample", quiet=True)
        resid, corr = _style_blindness(ds, args, normalize=norm)
        verdict = "erased" if resid < 0.1 else ("mostly erased" if resid < 0.25 else "preserved")
        print(f"    {mode:<16} {resid:>18.4f} {corr:>16.5f}   {verdict}")

    print("\n    per_sample z-scores each volume over its foreground, which is exactly an affine")
    print("    map in intensity — the same family as lut = base*gain + bias.")
    print("    CAVEAT on the noise_sigma column: sigma is a VARIANCE parameter, and the matched")
    print("    filter is a linear detector sharing the rendering seed with the base render, so it")
    print("    reads a fixed noise realization rather than sigma. Compare gain/bias across rows;")
    print("    take noise_sigma from the residual test above, not from its mf.")


# ───────────────────────────── 4. inert lesion dims ─────────────────────────────


def test_inert(args):
    print(f"\n[4] ARE z_content[2:5] LIVE?   sweeping the lesion-position dims in both lesion modes\n")
    print(f"    {'lesion_mode':<14} {'max|dx|':>10} {'mean|dx|':>10} {'voxels changed':>15}   verdict")
    for mode in ("sphere", "field"):
        a = argparse.Namespace(**vars(args))
        a.lesion_mode = mode
        ds = build_dataset(a, quiet=True)
        inner = ds._inner
        worst_max, worst_mean, worst_n = 0.0, 0.0, 0
        for i in range(min(args.n_samples, 8)):
            lat = inner[i][2]
            seed = inner.sample_seed_for(i)
            base = render(ds, lat, seed)[0]
            for sign in (-1.0, 1.0):
                zc = lat["z_content"].clone()
                zc[2:5] = sign * 1.0
                alt = render(ds, lat, seed, z_content=zc)[0]
                d = (alt - base).abs()
                worst_max = max(worst_max, float(d.max()))
                worst_mean = max(worst_mean, float(d.mean()))
                worst_n = max(worst_n, int((d > 1e-6).sum()))
        verdict = "INERT — dims are dead" if worst_max < 1e-6 else "live"
        print(f"    {mode:<14} {worst_max:>10.5f} {worst_mean:>10.6f} {worst_n:>15d}   {verdict}")

    # Positional resolution in sphere mode: how many distinguishable lesion centres per axis.
    a = argparse.Namespace(**vars(args))
    a.lesion_mode = "sphere"
    ds = build_dataset(a, quiet=True)
    r = ds._inner.renderer
    vox = 2.0 / args.res
    # Read the geometry off the renderer, not off the defaults: --lesion-radius moves BOTH
    # the ball and the WM margin, and they pull in opposite directions.
    rad = r.lesion_radius
    reach = (0.5 - (rad + 0.02)) / (3**0.5)
    print(f"\n    sphere mode: lesion radius {rad / vox:.1f} vox, centre travel +/-{reach / vox:.1f} vox per axis")
    print(f"    -> ~{2 * reach / vox:.1f} voxels of travel. Note the TRADE-OFF: the WM margin tracks the")
    print("       radius, so a bigger lesion carries more contrast energy (higher SNR) but travels")
    print("       LESS, and its extreme positions overlap more — which is why the +/-1 sweep above")
    print(f"       can change fewer voxels at a larger radius. lesion_mode={r.lesion_mode} in this run.")


# ─────────────────────────── 5. brain_size / thickness ──────────────────────────


def test_degeneracy(args):
    ds = build_dataset(args)
    inner = ds._inner
    delta = args.fd_delta
    print(
        f"\n[5] JACOBIAN DEGENERACY   central differences (+/-{delta}), noise averaged over "
        f"{args.n_noise} seeds, n={min(args.n_samples, 6)} base draws\n"
    )

    ndim = inner.n_content
    J = []
    for d in range(ndim):
        cols = []
        for i in range(min(args.n_samples, 6)):
            lat = inner[i][2]
            seed = inner.sample_seed_for(i)
            acc = 0.0
            for k in range(args.n_noise):
                s = seed + 7919 * k
                zp, zm = lat["z_content"].clone(), lat["z_content"].clone()
                zp[d] += delta
                zm[d] -= delta
                acc = acc + (render(ds, lat, s, z_content=zp)[0] - render(ds, lat, s, z_content=zm)[0]).flatten()
            cols.append(acc / args.n_noise)
        J.append(torch.stack(cols).mean(0))
    J = torch.stack(J)
    Jn = J / J.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = (Jn @ Jn.T).numpy()

    print("    cosine between per-dim render directions (|cos| ~ 1 => mutually confusable)")
    hdr = "    " + " " * 22 + " ".join(f"{i:>6}" for i in range(ndim))
    print(hdr)
    for d in range(ndim):
        row = " ".join(f"{C[d, j]:>6.2f}" for j in range(ndim))
        print(f"    {d} {CONTENT_NAMES[d] if d < len(CONTENT_NAMES) else '?':<20} {row}")

    print("\n    unique variance share: 1 - R^2 of each direction regressed on all others")
    Jm = J.numpy()
    for d in range(ndim):
        others = np.delete(Jm, d, axis=0).T
        y = Jm[d]
        if np.linalg.norm(y) < 1e-12:
            print(f"    {d} {CONTENT_NAMES[d]:<20} {'n/a (zero direction)':>22}")
            continue
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ coef
        unique = float((resid @ resid) / (y @ y))
        flag = "  <= mostly explained by other dims" if unique < 0.25 else ""
        print(f"    {d} {CONTENT_NAMES[d] if d < len(CONTENT_NAMES) else '?':<20} {unique:>8.3f}{flag}")

    i0, i5 = 0, 5
    if ndim > 5:
        print(f"\n    predicted pair: brain_size vs cortical_thickness  cos = {C[i0, i5]:+.3f}")
        print("    radii_gm = radii_wm + thickness, so both shift the OUTER boundary; only the")
        print("    WM/GM boundary separates them.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tests", nargs="+", default=["all"], choices=["all", "squash", "amplitude", "style", "inert", "degeneracy"]
    )
    p.add_argument(
        "--preset",
        choices=["defaults", "flagship"],
        default="defaults",
        help="'flagship' = experiments/synthetic_causal.yaml (causal, clean_content, fixed_reference)",
    )
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--n-content", type=int, default=9)
    p.add_argument("--n-samples", type=int, default=8, help="base latent draws")
    p.add_argument("--n-resample", type=int, default=4, help="redraws per dim, for the signal term")
    p.add_argument("--n-noise", type=int, default=8, help="rendering seeds, for the noise term")
    p.add_argument("--n-squash", type=int, default=20000)
    p.add_argument("--fd-delta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", default="per_sample", choices=["per_sample", "shared", "fixed_reference"])
    p.add_argument("--causal", action="store_true")
    p.add_argument("--clean-content", action="store_true")
    p.add_argument("--lesion-mode", default="sphere", choices=["sphere", "field"])
    p.add_argument("--identifiable-ventricle", action="store_true")
    p.add_argument("--csf-t1-intensity", type=float, default=0.1)
    p.add_argument("--content-scale", type=float, default=1.0)
    p.add_argument("--style-scale", type=float, default=1.0)
    p.add_argument("--content-prior", default="normal", choices=["normal", "uniform"])
    p.add_argument("--content-squash", default="auto", choices=["auto", "clamp", "tanh", "none"])
    p.add_argument("--content-amp-scale", type=float, nargs="+", default=None)
    p.add_argument("--lesion-radius", type=float, default=0.1)
    p.add_argument(
        "--cortex-parameterization", default="additive", choices=["additive", "nested", "midsurface", "patterned"]
    )
    p.add_argument("--center-local-deformations", action="store_true")
    args = p.parse_args()

    if args.preset == "flagship":
        args.causal = True
        args.clean_content = True
        args.normalize = "fixed_reference"

    tests = ["squash", "amplitude", "style", "inert", "degeneracy"] if "all" in args.tests else args.tests
    print(
        f"preset={args.preset}  causal={args.causal}  clean_content={args.clean_content}  "
        f"normalize={args.normalize}  lesion_mode={args.lesion_mode}  content_scale={args.content_scale}"
    )
    for t in tests:
        {
            "squash": test_squash,
            "amplitude": test_amplitude,
            "style": test_style,
            "inert": test_inert,
            "degeneracy": test_degeneracy,
        }[t](args)


if __name__ == "__main__":
    main()
