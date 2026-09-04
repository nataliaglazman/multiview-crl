#!/usr/bin/env python
"""Does each content factor LOOK THE SAME in both views? Checkpoint-free.

The question behind it: a contrastive run and its architecture-matched recon-only
baseline (both @ step 22001, `--causal iid --probe-dim 64`) split cleanly on WHICH
content factors survive:

    kept / improved   brain_size, cortical_thickness, temporal_atrophy (0.281 -> 0.800),
                      lr_asymmetry (0.463 -> 0.879), sulcal_widening
    destroyed         ventricle_size (0.596 -> -0.012), lesion_x/y/z (~0.12 -> ~-0.02)

Total factor information is unchanged (mean 0.398 vs 0.382), so the objective
REALLOCATES rather than loses. `temporal_atrophy` is a localized Gaussian bump and it
more than doubled, so the split is not local-vs-global. The candidate explanation is
that it is view-consistency: the kept factors displace tissue BOUNDARIES, whose location
is view-invariant, while the destroyed ones are read from intensity contrast that the
modality LUT modulates (`Renderer.render_modality`):

    tissue      T1     FLAIR
    WM         0.80     0.40
    GM         0.50     0.80
    CSF        0.10     0.10
    lesion     0.40     1.00      <- 0.40 BELOW its WM surround in T1, 0.60 ABOVE in FLAIR

A cross-view objective forces content to agree across views. Evidence that inverts
between views cannot be carried by a view-shared LINEAR code -- it needs a rectifying
detector (|contrast|, not signed contrast) and nothing in Barlow Twins' correlation
structure asks for one. Reconstruction is under no such constraint: each view is decoded
from its own code, so "dark blob" and "bright blob" both work.

What is measured
----------------
Resample one latent dim with everything else -- INCLUDING the rendering seed -- held
fixed, so the Rician noise and bias field cancel in the difference and the delta is
essentially pure signal (same trick as `generator_defects._sensitivity`). Then compare
the two views' render deltas:

    cos(dx_T1, dx_FLAIR)   sign/shape agreement of the evidence
    ||dx_FLAIR|| / ||dx_T1||   amplitude asymmetry

Reading it
----------
    cos ~ +1                 view-consistent. A shared linear detector works. Expected
                             for every boundary-displacing (geometric) factor -- these
                             are the positive controls, and if they do NOT come out near
                             +1 the measurement itself is wrong.
    cos ~ -1                 the evidence INVERTS between views. No view-shared linear
                             detector exists. This is the hypothesis under test.
    cos ~ 0                  the two views carry unrelated evidence.
    ratio far from 1         same sign, but one view carries the factor far more
                             strongly; a shared code has to compromise.

Pre-registered predictions, so this cannot be read after the fact
----------------------------------------------------------------
    lesion_x/y/z    cos ~ -1.  The LUT reverses the lesion's contrast sign. CONFIRMS.
    ventricle_size  cos ~ +1, ratio ~ 0.43.  CSF-minus-WM is -0.70 in T1 and -0.30 in
                    FLAIR: same sign, 2.3x weaker. So the sign-flip story does NOT
                    explain ventricle_size's loss, and a +1 here means the hypothesis
                    covers the lesion half ONLY and something else is needed for the
                    ventricle -- amplitude asymmetry is the next candidate, not a
                    conclusion.
    the other five  cos ~ +1.  Positive control.

A style-dim control block runs too: perturbing `z_style_v1` must leave view 2 untouched
(||dx_FLAIR|| ~ 0). If it does not, the harness is wrong and nothing above is readable.

Usage
-----
    python -m eval.view_consistency --preset flagship
    python -m eval.view_consistency --preset flagship --res 32 --n-samples 4   # faster
    python -m eval.view_consistency --preset flagship --independent-style
"""

from __future__ import annotations

import argparse

import torch

from eval.generator_defects import (
    CONTENT_NAMES,
    STYLE_NAMES,
    build_dataset,
    draw_content,
    render,
    verify_content_sampler,
)


def _deltas(ds, lat, sample_seed, key, alt, normalize=True):
    """Render deltas in BOTH views from resampling one latent component."""
    b1, b2, _ = render(ds, lat, sample_seed, normalize=normalize)
    p1, p2, _ = render(ds, lat, sample_seed, normalize=normalize, **{key: alt})
    return (p1 - b1).flatten(), (p2 - b2).flatten()


def _cos_and_ratio(d1, d2):
    n1, n2 = d1.norm(), d2.norm()
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan"), float(n2 / n1) if n1 > 1e-12 else float("nan")
    return float(torch.dot(d1, d2) / (n1 * n2)), float(n2 / n1)


def _scan(ds, args, key, names, n_dims):
    """Per-dim (mean cos, std cos, median amplitude ratio) across latents x resamples."""
    inner = ds._inner
    rows = []
    for dim in range(n_dims):
        cosines, ratios = [], []
        for i in range(args.n_samples):
            seed = inner.sample_seed_for(i)
            lat = _latents(inner, i, args)
            for j in range(args.n_resample):
                alt = _resampled(inner, i, j, key, dim, lat)
                d1, d2 = _deltas(ds, lat, seed, key, alt, normalize=not args.raw)
                c, r = _cos_and_ratio(d1, d2)
                if c == c:  # drop NaN (a dim with no render effect at all)
                    cosines.append(c)
                    ratios.append(r)
        if not cosines:
            rows.append((names[dim], float("nan"), float("nan"), float("nan")))
            continue
        c = torch.tensor(cosines)
        rows.append((names[dim], float(c.mean()), float(c.std()), float(torch.tensor(ratios).median())))
    return rows


def _latents(inner, idx, args):
    """The latent dict for sample idx (same source as `_run_sensitivity`), with the two
    views' style matched by default.

    Matching the style vectors isolates the MODALITY LUT: with independent style draws a
    low cosine could be the style difference rather than the LUT, and the LUT is the
    claim under test. --independent-style restores the natural draw.
    """
    lat = dict(inner[idx][2])
    if not args.independent_style:
        lat["z_style_v2"] = lat["z_style_v1"].clone()
    return lat


def _resampled(inner, idx, rep, key, dim, lat):
    """`lat[key]` with component `dim` redrawn, everything else held.

    Index arithmetic mirrors `generator_defects._run_sensitivity` so the redraws come
    from the same family of latents that the amplitude test uses.
    """
    alt = lat[key].clone()
    if key == "z_content":
        alt[dim] = draw_content(inner, 10_000 + idx * 97 + rep * 13)[dim]
    else:
        g = torch.Generator().manual_seed(inner.sample_seed_for(idx) + 31 * rep + dim)
        alt[dim] = torch.randn(1, generator=g)[0]
    return alt


def _print(title, rows, note=None):
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")
    print(f"  {'factor':<20}{'cos(T1, FLAIR)':>18}{'+/- sd':>10}{'||FLAIR||/||T1||':>20}")
    print(f"  {'-' * 68}")
    for name, mean_c, sd_c, ratio in rows:
        flag = ""
        if mean_c == mean_c:
            flag = "  <- INVERTS" if mean_c < -0.3 else ("" if mean_c > 0.7 else "  <- unrelated")
        print(f"  {name:<20}{mean_c:>18.3f}{sd_c:>10.3f}{ratio:>20.3f}{flag}")
    if note:
        print(f"  {note}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=["defaults", "flagship"], default="flagship")
    p.add_argument("--res", type=int, default=64, help="Training resolution is 64; 32 is ~8x faster.")
    p.add_argument("--n-content", type=int, default=9)
    p.add_argument("--n-samples", type=int, default=8, help="base latent draws")
    p.add_argument("--n-resample", type=int, default=4, help="redraws per dim")
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
    p.add_argument(
        "--independent-style",
        action="store_true",
        help="Draw each view's style independently (the natural draw). Default matches them, "
        "so a low cosine is attributable to the modality LUT rather than to the style gap.",
    )
    p.add_argument("--raw", action="store_true", help="Skip normalize_views; measure the raw render.")
    p.add_argument("--no-style-control", action="store_true", help="Skip the style sanity block.")
    args = p.parse_args()

    if args.preset == "flagship":
        args.causal = True
        args.clean_content = True
        args.normalize = "fixed_reference"

    ds = build_dataset(args)
    verify_content_sampler(ds._inner)
    print(
        f"preset={args.preset}  res={args.res}  causal={args.causal}  clean_content={args.clean_content}  "
        f"normalize={args.normalize}  lesion_mode={args.lesion_mode}  lesion_radius={args.lesion_radius}  "
        f"style={'independent' if args.independent_style else 'matched across views'}"
    )

    _print(
        "CROSS-VIEW CONSISTENCY OF THE RENDER JACOBIAN  (content factors)",
        _scan(ds, args, "z_content", CONTENT_NAMES, args.n_content),
        note="cos ~ +1 view-consistent | cos ~ -1 evidence inverts between views | ratio = amplitude asymmetry",
    )

    if not args.no_style_control:
        _print(
            "CONTROL: perturbing z_style_v1 must not move view 2",
            _scan(ds, args, "z_style_v1", STYLE_NAMES, ds._inner.n_style),
            note="||FLAIR||/||T1|| ~ 0 expected; anything else means the harness is wrong.",
        )


if __name__ == "__main__":
    main()
