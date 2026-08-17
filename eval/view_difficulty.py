#!/usr/bin/env python
"""How hard is the cross-view alignment task, as a property of the DATA?

No model, no checkpoint. Every other diagnostic in this project asks what a trained
encoder did; this asks whether the benchmark can distinguish encoders at all.

The question matters because of how the synthetic generator works: ``render_structure``
runs ONCE per subject and then two modality intensity maps are applied on top. If the two
views are therefore the same geometry under a recolouring, any geometry-sensitive feature
is view-invariant for free, cross-view alignment is solved at initialisation-plus-epsilon,
and no weighting of a contrastive objective can make it bind. That would be a property of
the benchmark, not of the objective — and it is not something coefficient tuning can fix.

The headline number is the COLOUR-MAP FRACTION: how much of view 2 is predictable from
view 1 by a pointwise intensity map, with no spatial context at all.

    R^2 ~ 1.0   view 2 IS view 1 recoloured. A pointwise nonlinearity suffices, so any
                encoder that responds to geometry is already invariant. The alignment task
                is close to vacuous and the benchmark cannot test the claim.
    R^2 ~ 0.5   substantial genuinely view-specific structure. Alignment is a real task.

Three maps are fitted, and the differences between them are as informative as the values:

  * ``linear, per subject``     — the weakest map. Excess over this is contrast nonlinearity.
  * ``pointwise, SHARED``       — one map for the whole dataset. If this alone is ~1.0, even
                                  the style factors do nothing per-subject, and there is
                                  nothing for a style code to carry.
  * ``pointwise, per subject``  — the strongest pointwise map. The gap between this and the
                                  shared one is exactly what a per-sample style code has to
                                  encode, which is a direct read on how much work the style
                                  pathway should be doing.

Every map is fitted on half the brain voxels and scored on the other half, so a per-subject
map with many bins cannot look good by memorising.

Also reported:

  * MONOTONICITY of the shared map. A genuine T1/T2-like contrast inverts the tissue
    ordering (CSF dark in one, bright in the other). A monotone map is a much weaker
    transformation and an easier alignment problem.
  * EDGE AGREEMENT between the views' gradient magnitudes. Near 1 means the geometry is
    literally identical and only intensity differs.

Usage
-----
    python -m eval.view_difficulty --run-dir results/synthetic/RUN
    python -m eval.view_difficulty --run-dir results/synthetic/RUN --split train
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

# torch only appears in the data-loading shim, so the estimators below are testable
# on plain numpy (same convention as eval.run_dci_compare).

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_pointwise_map(x_fit, y_fit, n_bins, edges=None):
    """Optimal pointwise map x -> y as a binned conditional mean.

    This is the best ANY function of the voxel intensity alone can do (in squared error),
    so it upper-bounds what a pointwise recolouring explains. Empty bins fall back to the
    global mean rather than 0, which would otherwise inject a spurious residual wherever
    the fit half happened not to cover an intensity range.
    """
    if edges is None:
        lo, hi = float(np.min(x_fit)), float(np.max(x_fit))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(x_fit, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=y_fit, minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins)
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), y_fit.mean())
    return means, edges


def apply_pointwise_map(x, means, edges):
    n_bins = len(means)
    return means[np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)]


def colour_map_scores(v1, v2, n_bins=48, seed=0):
    """Held-out R^2 of predicting v2 from v1 under three pointwise maps.

    ``v1``/``v2``: ``(N, K)`` matched brain-voxel samples, one row per subject.
    """
    v1, v2 = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    n, k = v1.shape
    rng = np.random.RandomState(seed)
    # Split VOXELS, not subjects: the per-subject map must not be scored on the voxels it
    # was fitted on, or more bins would always look better.
    perm = rng.permutation(k)
    fit_i, ev_i = perm[: k // 2], perm[k // 2 :]

    lin_pred = np.empty((n, len(ev_i)))
    pt_pred = np.empty((n, len(ev_i)))
    for i in range(n):
        xf, yf = v1[i, fit_i], v2[i, fit_i]
        xe = v1[i, ev_i]
        a, b = np.polyfit(xf, yf, 1) if np.std(xf) > 1e-12 else (0.0, float(yf.mean()))
        lin_pred[i] = a * xe + b
        means, edges = fit_pointwise_map(xf, yf, n_bins)
        pt_pred[i] = apply_pointwise_map(xe, means, edges)

    shared_means, shared_edges = fit_pointwise_map(v1[:, fit_i].ravel(), v2[:, fit_i].ravel(), n_bins)
    shared_pred = apply_pointwise_map(v1[:, ev_i], shared_means, shared_edges)

    truth = v2[:, ev_i]
    return {
        "linear_per_subject": _r2(truth, lin_pred),
        "pointwise_shared": _r2(truth, shared_pred),
        "pointwise_per_subject": _r2(truth, pt_pred),
        "shared_map": shared_means,
        "shared_edges": shared_edges,
    }


def monotonicity(means):
    """Fraction of adjacent steps in the shared map that increase.

    1.0 or 0.0 means a monotone recolouring — a weak transformation. Values near 0.5 mean
    the map reorders intensities, which is what a real T1/T2 contrast change does and what
    makes alignment a genuine problem.
    """
    d = np.diff(np.asarray(means, dtype=np.float64))
    d = d[np.abs(d) > 1e-12]
    return float((d > 0).mean()) if len(d) else float("nan")


def edge_agreement(g1, g2):
    """Pearson correlation of the two views' gradient magnitudes over brain voxels.

    Near 1 means the edges sit in the same places with the same relative strength, i.e. the
    geometry is shared and only intensity differs.
    """
    a, b = np.asarray(g1, dtype=np.float64).ravel(), np.asarray(g2, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _grad_mag(vol):
    gx, gy, gz = np.gradient(np.asarray(vol, dtype=np.float64))
    return np.sqrt(gx**2 + gy**2 + gz**2)


def verdict(s, mono, edge):
    pps = s["pointwise_per_subject"]
    print("\n  Verdict")
    if pps > 0.95:
        print(f"    COLOUR MAP. {100 * pps:.1f}% of view 2 is a pointwise recolouring of view 1,")
        print("    so the two views share their geometry almost exactly. Any geometry-sensitive")
        print("    feature is view-invariant for free and cross-view alignment is close to")
        print("    vacuous — the benchmark cannot distinguish a good encoder from a mediocre one,")
        print("    and no contrastive coefficient will make the objective bind. Fix the DATA:")
        print("    a harder style transformation, or move to the real paired scans.")
    elif pps > 0.8:
        print(f"    MOSTLY COLOUR MAP ({100 * pps:.1f}%). Alignment is easy but not trivial;")
        print("    expect a contrastive objective to converge early. Worth hardening the style")
        print("    transformation before drawing conclusions about the objective.")
    else:
        print(f"    GENUINE TASK ({100 * pps:.1f}% pointwise-explainable). There is real")
        print("    view-specific structure to be invariant to, so an objective that converges")
        print("    early is a property of the objective, not of the data.")

    gap = pps - s["pointwise_shared"]
    print(f"\n    Per-subject gain over one shared map: {gap:+.4f}")
    if gap < 0.01:
        print("      Style has essentially NOTHING per-subject to encode — one global recolouring")
        print("      fits every subject. A style code cannot be expected to develop, and")
        print("      style_sufficiency is measuring a factor with almost no effect on the image.")
    else:
        print("      This is the per-sample variation a style code has to carry; compare it")
        print("      against style_sufficiency to see how much of it the model actually captured.")

    if np.isfinite(mono):
        print(f"\n    Shared-map monotonicity: {mono:.2f}")
        if mono > 0.95 or mono < 0.05:
            print("      MONOTONE — the contrast change preserves tissue ordering. That is much")
            print("      weaker than a real T1/T2 inversion (CSF dark in one, bright in the other)")
            print("      and makes the alignment problem correspondingly easier.")
    if np.isfinite(edge):
        print(f"\n    Edge agreement between views: {edge:.4f}")
        if edge > 0.9:
            print("      The geometry is effectively identical across views — edges in the same")
            print("      places at the same relative strength.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Run whose settings.json defines the generator")
    ap.add_argument("--split", choices=("test", "train"), default="test")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--voxels-per-sample", type=int, default=4000)
    ap.add_argument("--n-bins", type=int, default=48)
    ap.add_argument("--edge-samples", type=int, default=16, help="Volumes used for the edge-agreement number")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    args_ = load_run_args(cli.run_dir)
    if cli.split == "train":
        # build_synthetic_test_set hardcodes mode="test"; the training split differs only by
        # _SPLIT_OFFSETS on the seed, so ask the dataset class directly to get the volumes
        # the model actually trained on.
        from data.datasets import SyntheticBrainDataset

        res = int(getattr(args_, "synthetic_res", 64))
        ds = SyntheticBrainDataset(
            mode="train",
            spatial_size=getattr(args_, "spatial_size", None) or (res, res, res),
            cache=False,
            synthetic_num_samples=cli.num_samples,
            **{
                k: getattr(args_, k)
                for k in (
                    "synthetic_mode",
                    "synthetic_seed",
                    "synthetic_n_content",
                    "synthetic_n_style",
                    "synthetic_style_scale",
                    "synthetic_content_scale",
                    "synthetic_normalize",
                    "synthetic_clean_content",
                    "synthetic_causal",
                    "synthetic_causal_graph",
                    "synthetic_causal_edge_prob",
                    "synthetic_content_prior",
                    "synthetic_content_squash",
                    "synthetic_content_amp_scale",
                    "synthetic_lesion_radius",
                    "synthetic_cortex_parameterization",
                    "synthetic_center_local_deformations",
                )
                if hasattr(args_, k)
            },
        )
    else:
        ds = build_synthetic_test_set(args_, cli.num_samples, cache=False, causal=True)

    print(f"\n{'=' * 78}\nVIEW DIFFICULTY — {cli.split} split, {cli.num_samples} subjects\n{'=' * 78}")
    print(
        f"  clean_content={getattr(args_, 'synthetic_clean_content', False)}  "
        f"normalize={getattr(args_, 'synthetic_normalize', '?')}  "
        f"n_style={getattr(args_, 'synthetic_n_style', '?')}"
    )

    rng = np.random.RandomState(cli.seed)
    v1s, v2s, edge_vals = [], [], []
    for i in range(min(cli.num_samples, len(ds))):
        item = ds[i]
        a = np.asarray(item["image"][0]).squeeze()
        b = np.asarray(item["image"][1]).squeeze()
        m = np.asarray(item["mask"][0]).squeeze() > 0.5
        idx = np.flatnonzero(m.ravel())
        if len(idx) < cli.voxels_per_sample:
            continue
        pick = rng.choice(idx, cli.voxels_per_sample, replace=False)
        v1s.append(a.ravel()[pick])
        v2s.append(b.ravel()[pick])
        if len(edge_vals) < cli.edge_samples:
            edge_vals.append(edge_agreement(_grad_mag(a)[m], _grad_mag(b)[m]))

    v1, v2 = np.stack(v1s), np.stack(v2s)
    print(f"  {v1.shape[0]} subjects x {v1.shape[1]} brain voxels, held-out halves\n")

    s = colour_map_scores(v1, v2, n_bins=cli.n_bins, seed=cli.seed)
    print(f"  {'map from view 1 -> view 2':<32}{'held-out R^2':>14}")
    print("  " + "-" * 46)
    for key, label in (
        ("linear_per_subject", "linear, per subject"),
        ("pointwise_shared", "pointwise, SHARED across all"),
        ("pointwise_per_subject", "pointwise, per subject"),
    ):
        print(f"  {label:<32}{s[key]:>14.4f}")

    verdict(s, monotonicity(s["shared_map"]), float(np.nanmean(edge_vals)) if edge_vals else float("nan"))
    print()


if __name__ == "__main__":
    main()
