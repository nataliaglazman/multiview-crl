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

RESULT (res 64, flagship preset, 20 Aug 2026) -- the sign hypothesis is DEAD
----------------------------------------------------------------------------
    factor                cos      ratio    model delta (contrastive - baseline)
    ventricle_size      +0.992     0.431          -0.608
    lesion_y            -0.317     4.129          -0.173
    lesion_x            -0.409     2.697          -0.134
    lesion_z            -0.306     3.964          -0.128
    brain_size          +0.031     1.047          +0.028
    cortical_thickness  +0.654     0.807          +0.086
    sulcal_widening     +0.483     0.765          +0.137
    lr_asymmetry        +0.283     0.814          +0.416
    temporal_atrophy    +0.018     0.894          +0.519

`ventricle_size` is the MOST view-consistent factor measured (+0.992) and it is the one
most completely destroyed; `brain_size` and `temporal_atrophy` are near zero and are kept
and improved. Spearman(cos, model delta) = +0.150. Sign consistency does not order the
outcome, so the "the lesion's contrast reverses, therefore a view-shared code cannot carry
it" story is falsified even though the reversal is real.

What DOES order it is the AMPLITUDE RATIO. Spearman(|log2 ratio|, model delta) = -0.700,
and the separation is clean with no overlap: every kept factor has |log2 ratio| <= 0.386,
every destroyed one >= 1.214 -- a 3x gap. A cross-view objective has to emit the same code
from evidence that is 2.3x weaker (ventricle) or 4x stronger (lesion) in one view than the
other; the factors it keeps are the ones whose evidence arrives at comparable strength in
both. Caveat on the statistics: the three lesion dims are one factor group, not three
independent ones, so the honest count is 2 destroyed groups vs 5 kept, and n=9 overstates
the rank correlation's significance. The clean separation is the stronger evidence.

The measurement validates against the LUT to three digits: CSF-minus-WM is -0.70 in T1 and
-0.30 in FLAIR, predicting a ventricle ratio of 0.3/0.7 = 0.4286 -- measured 0.431.

Why the "geometric" positive controls do NOT come out near +1
-------------------------------------------------------------
Because the WM/GM edge itself inverts: WM-minus-GM is +0.30 in T1 and -0.40 in FLAIR,
while the outer GM/background edge does not (+0.50 vs +0.80). A factor that moves both
boundaries -- brain_size moves radii_wm and the outer shell together -- gets one inverting
and one agreeing contribution, and they partially cancel to a near-zero cosine with a large
spread (brain_size +0.031 +/- 0.387). So a mid-range cosine here means "mixed evidence",
not "weak evidence", and the cosine column should not be read as a quality score.

Reading it
----------
    ratio far from 1         the factor's evidence arrives at very different strength in
                             the two views. THIS is the column that predicts what a
                             contrastive objective discards.
    cos ~ -1                 the evidence inverts between views. Real, and true of the
                             lesion dims -- but it does not predict the outcome.
    cos ~ 0 with large sd    mixed: the factor moves both an inverting and an agreeing
                             boundary. Not the same as weak.

A style-dim control block runs too: perturbing `z_style_v1` must leave view 2 untouched.
Its cosine is NaN by construction (view 2 does not move at all, so the angle is undefined)
and the verdict is carried by ||dx_FLAIR||/||dx_T1|| ~ 0 in the ratio column.

Usage
-----
    python -m eval.view_consistency --preset flagship
    python -m eval.view_consistency --preset flagship --res 32 --n-samples 4   # faster
    python -m eval.view_consistency --preset flagship --independent-style
"""
from __future__ import annotations

import argparse
from math import log2

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
                # Accumulated SEPARATELY: the cosine is undefined when either view does
                # not move, but the ratio is exactly the number that case is asking for.
                # The style control lives here — view 2 must not move, so its cosine is
                # NaN by construction and only the ~0 ratio carries the verdict.
                if c == c:
                    cosines.append(c)
                if r == r:
                    ratios.append(r)
        c = torch.tensor(cosines) if cosines else None
        rows.append(
            (
                names[dim],
                float(c.mean()) if c is not None else float("nan"),
                float(c.std()) if c is not None and len(cosines) > 1 else float("nan"),
                float(torch.tensor(ratios).median()) if ratios else float("nan"),
            )
        )
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
        # Flag on the AMPLITUDE, not the cosine: measured on this generator, |log2 ratio|
        # separates the factors a contrastive run keeps from the ones it destroys with no
        # overlap (kept <= 0.386, lost >= 1.214), while the cosine does not order them at
        # all. See the module docstring.
        flag = ""
        if ratio == ratio and ratio > 0 and abs(log2(ratio)) > 1.0:
            flag = "  <- VIEW-ASYMMETRIC"
        if mean_c == mean_c and mean_c < -0.3:
            flag += "  (inverts)"
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
        note="ratio is the column that predicts what a contrastive objective discards; "
        "cos ~ 0 with a large sd means MIXED evidence (see the module docstring), not weak.",
    )

    if not args.no_style_control:
        _print(
            "CONTROL: perturbing z_style_v1 must not move view 2",
            _scan(ds, args, "z_style_v1", STYLE_NAMES, ds._inner.n_style),
            note="cos is NaN by construction here (view 2 does not move, so the angle is undefined); "
            "the verdict is ||FLAIR||/||T1|| ~ 0. Anything else means the harness is wrong.",
        )


if __name__ == "__main__":
    main()
