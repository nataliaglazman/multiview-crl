"""Ground-truth ensemble covariance; called by factor_structure_audit.

Scalars have a between-subject covariance matrix, not a spatial kernel. Genuine
latent fields are sampled on the same voxel lattice after the renderer's exact
interpolation, before their structural multipliers. No per-image standardization
is introduced by this audit. Hälvä applicability is assessed separately.
"""

import hashlib
import json
from itertools import combinations

import numpy as np
import torch

from eval.encoder.encoder_target_protocol import save_csv
from eval.lesion.checkpoint_lesion_analysis import json_safe
from eval.metrics.dci import CONTENT_FACTOR_NAMES
from eval.synthetic.synthetic_dataset import CLEAN_NUISANCE_SCALE

PAPER = "https://proceedings.mlr.press/v238/halva24a.html"


def covariance(samples):
    samples = np.asarray(samples, np.float64)
    centered = samples - samples.mean(0)
    return centered.T @ centered / (len(samples) - 1)


def correlation(cov):
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    denominator = np.outer(std, std)
    return np.divide(cov, denominator, out=np.full_like(cov, np.nan), where=denominator > 1e-15)


def scale_normalize(cov):
    scale = float(np.trace(cov) / len(cov))
    return cov / scale if scale > 1e-15 else np.full_like(cov, np.nan)


def kernel_distance(a, b):
    """Compare shape after removing overall variance; amplitude alone is ambiguous."""
    a, b = scale_normalize(a), scale_normalize(b)
    denominator = (np.linalg.norm(a) + np.linalg.norm(b)) / 2
    return float(np.linalg.norm(a - b) / denominator) if denominator > 0 else np.nan


def lag_profiles(samples, grid):
    """Average K(u,u+h) along each axis; spatial pooling is descriptive, not stationarity."""
    x = np.asarray(samples, np.float64).reshape(-1, grid, grid, grid)
    x = x - x.mean(0, keepdims=True)  # ensemble mean at every location, not each field's spatial mean
    correction = len(x) / (len(x) - 1)
    result = np.zeros((3, grid))
    result[:, 0] = np.mean(x * x) * correction
    for axis in range(3):
        for lag in range(1, grid):
            a, b = [slice(None)] * 4, [slice(None)] * 4
            a[axis + 1], b[axis + 1] = slice(None, -lag), slice(lag, None)
            result[axis, lag] = np.mean(x[tuple(a)] * x[tuple(b)]) * correction
    scale = result[0, 0]
    return result, result / scale if scale > 1e-15 else np.full_like(result, np.nan)


def applicability(inner, components, resampled):
    checks = [
        dict(
            requirement="Spatially indexed components",
            status="partial",
            finding="z_deformation/z_fissure (and z_lesion in field mode) are fields. The nine named content controls are scalars per subject; their 9x9 covariance is not a spatial kernel.",
        ),
        dict(
            requirement="Components influence observations",
            status="violated" if inner.clean_content else "needs_check",
            finding=f"clean_content={inner.clean_content}; deformation/fissure structural multipliers are {components['z_deformation']['structural_multiplier']:g} and {components['z_fissure']['structural_multiplier']:g}. Zero multipliers make those raw fields unrecoverable from images.",
        ),
        dict(
            requirement="Injective pointwise mixing",
            status="violated",
            finding="This renderer thresholds continuous geometry into tissue labels, masks fields by tissue, and then spatially blurs the image. It is not the paper's injective f(s(u)) at each spatial location. Sphere lesion position is also quantized.",
        ),
        dict(
            requirement="Independent spatial components",
            status="not_established",
            finding=f"Causal content={inner.causal}; hierarchical content={inner.hierarchical_content}; {resampled} audited subjects needed anatomy redraws. WM-fit acceptance can condition the prior. Small empirical cross-correlations do not prove independence.",
        ),
        dict(
            requirement="GP or TP component law",
            status="not_established" if inner.field_prior == "iid" else "violated_as_exact_sampler",
            finding=(
                "iid native Gaussian grids undergo linear interpolation; their image-grid covariance may be degenerate. This alone does not establish the remaining assumptions."
                if inner.field_prior == "iid"
                else "sample_gp_field centers and divides by each realization's own spatial standard deviation. Consequently the implemented 'gp'/'tp' names do not certify the exact GP/TP process law in the paper."
            ),
        ),
        dict(
            requirement="Observation noise model",
            status="violated_as_exact_model",
            finding="Magnitude noise is applied before spatial smoothing, giving signal-dependent and spatially correlated residuals; the paper models additive observation noise independent of the components.",
        ),
        dict(
            requirement="Distinct covariance kernels (GP case)",
            status="empirical_diagnostic_only",
            finding="Raw and trace-normalized kernel estimates, lag profiles and split-half variability are provided. Finite-sample differences do not prove distinct population kernels; TP identifiability is not governed by this GP-only condition.",
        ),
    ]
    return dict(
        paper=PAPER,
        theorem_guarantee_established=False,
        conclusion="The current renderer does not satisfy the exact Hälvä observation model. Kernel differences can motivate a spatial prior, but cannot certify its identifiability theorem here.",
        checks=checks,
    )


@torch.inference_mode()
def kernel_audit(ds, args, out):
    inner = ds._inner
    indices = np.rint(np.linspace(0, ds.res - 1, args.kernel_grid)).astype(int)
    selection = np.ix_(indices, indices, indices)
    fields, native, scalars, raw_hash = {}, {}, [], hashlib.sha256()
    native_shapes, ids = {}, []
    resampled = 0
    for n, idx in enumerate(range(args.subject_offset, args.subject_offset + args.kernel_samples)):
        # Same accepted subject as __getitem__, without rendering unneeded acquisitions.
        seed = inner.sample_seed_for(idx)
        latent = inner._draw_pseudo_mri(seed)
        ids.append(idx)
        scalars.append(latent["z_content"].numpy())
        resampled += int(inner._accepted_attempt.get(idx, 0) > 0)
        raw_hash.update(np.asarray([idx, seed], np.int64).tobytes())
        raw_hash.update(scalars[-1].tobytes())
        for name in ("z_deformation", "z_fissure", "z_lesion"):
            value = latent[name]
            if value is None:
                continue
            native_shapes[name] = list(value.shape)
            native.setdefault(name, []).append(value.numpy().reshape(-1))
            expanded = inner.renderer._upsample_field(value, "cpu").numpy()[selection]
            fields.setdefault(name, []).append(expanded.reshape(-1))
            raw_hash.update(value.contiguous().numpy().tobytes())
        if (n + 1) % 32 == 0 or n + 1 == args.kernel_samples:
            print(f"  latent kernels: {n + 1}/{args.kernel_samples} subjects", flush=True)
    scalars = np.asarray(scalars, np.float64)
    archive = dict(
        subject_ids=np.array(ids),
        voxel_indices=indices,
        scalar_names=np.array(CONTENT_FACTOR_NAMES),
        scalar_covariance=covariance(scalars),
        scalar_correlation=correlation(covariance(scalars)),
    )
    components, profiles, estimates, bootstraps = {}, [], {}, {}
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(args.kernel_samples, size=(args.kernel_bootstrap, args.kernel_samples))
    split = np.random.default_rng(args.seed).permutation(args.kernel_samples)
    mid = len(split) // 2
    for name in fields:
        samples = np.asarray(fields[name], np.float64)
        fields[name] = samples
        cov = covariance(samples)
        native_cov = covariance(np.asarray(native[name]))
        if name in ("z_deformation", "z_fissure"):
            multiplier = (0.1 if name == "z_deformation" else 0.05) * (
                CLEAN_NUISANCE_SCALE if inner.clean_content else 1.0
            )
        else:
            multiplier = None  # sigmoid and tissue masking; there is no constant multiplier
        info = dict(
            native_shape=native_shapes[name],
            n_subjects=len(samples),
            structural_multiplier=multiplier,
            active_structural_path=multiplier != 0,
            average_raw_variance=float(np.diag(cov).mean()),
            spatial_variance_cv=float(np.diag(cov).std() / np.diag(cov).mean()) if np.diag(cov).mean() else np.nan,
            split_half_kernel_distance=kernel_distance(
                covariance(samples[split[:mid]]), covariance(samples[split[mid:]])
            ),
        )
        components[name] = info
        archive[name + "_covariance"] = cov.astype(np.float32)
        archive[name + "_native_covariance"] = native_cov.astype(np.float32)
        archive[name + "_mean"] = samples.mean(0).astype(np.float32)
        if multiplier is not None:
            archive[name + "_effective_covariance"] = (cov * multiplier**2).astype(np.float32)
        point_cov, point = lag_profiles(samples, args.kernel_grid)
        estimates[name] = point
        bootstrap = np.array([lag_profiles(samples[draw], args.kernel_grid)[1] for draw in draws])
        bootstraps[name] = bootstrap
        for axis, label in enumerate(("x", "y", "z", "mean_axes")):
            values = point[axis] if axis < 3 else point.mean(0)
            covariance_values = point_cov[axis] if axis < 3 else point_cov.mean(0)
            boot = bootstrap[:, axis] if axis < 3 and len(bootstrap) else bootstrap.mean(1) if len(bootstrap) else None
            for lag, value in enumerate(values):
                bounds = (
                    np.quantile(boot[:, lag], [0.025, 0.975])
                    if boot is not None and len(boot) >= 10
                    else (np.nan, np.nan)
                )
                profiles.append(
                    dict(
                        component=name,
                        active_structural_path=info["active_structural_path"],
                        axis=label,
                        grid_lag=lag,
                        mean_distance_renderer_units=float(
                            np.mean(indices[lag:] - indices[: len(indices) - lag]) * 2 / (ds.res - 1)
                        ),
                        covariance=float(covariance_values[lag]),
                        normalized_covariance=float(value),
                        normalized_ci95_low=float(bounds[0]),
                        normalized_ci95_high=float(bounds[1]),
                    )
                )
    comparisons, differences = [], []
    for a, b in combinations(fields, 2):
        x, y = fields[a] - fields[a].mean(0), fields[b] - fields[b].mean(0)
        cross = x.T @ y / (len(x) - 1)
        va, vb = np.diag(archive[a + "_covariance"]), np.diag(archive[b + "_covariance"])
        denominator = np.sqrt(np.outer(va, vb))
        corr = np.divide(cross, denominator, out=np.full_like(cross, np.nan), where=denominator > 1e-15)
        comparisons.append(
            dict(
                component_a=a,
                component_b=b,
                trace_normalized_kernel_distance=kernel_distance(
                    archive[a + "_covariance"], archive[b + "_covariance"]
                ),
                within_a_split_half_distance=components[a]["split_half_kernel_distance"],
                within_b_split_half_distance=components[b]["split_half_kernel_distance"],
                cross_component_correlation_rms=(
                    float(np.sqrt(np.nanmean(corr**2))) if np.isfinite(corr).any() else np.nan
                ),
                cross_component_max_abs_correlation=float(np.nanmax(abs(corr))) if np.isfinite(corr).any() else np.nan,
            )
        )
        for lag in range(args.kernel_grid):
            point = float((estimates[a] - estimates[b]).mean(0)[lag])
            draws_delta = (bootstraps[a] - bootstraps[b]).mean(1)[:, lag] if len(bootstraps[a]) else []
            bounds = np.quantile(draws_delta, [0.025, 0.975]) if len(draws_delta) >= 10 else (np.nan, np.nan)
            differences.append(
                dict(
                    component_a=a,
                    component_b=b,
                    grid_lag=lag,
                    normalized_profile_difference=point,
                    difference_ci95_low=float(bounds[0]),
                    difference_ci95_high=float(bounds[1]),
                )
            )
    checks = applicability(inner, components, resampled)
    archive["component_names"] = np.array(list(fields))
    np.savez_compressed(out / "latent_covariances.npz", **archive)
    save_csv(out / "kernel_profiles.csv", profiles)
    save_csv(out / "kernel_comparisons.csv", comparisons)
    save_csv(out / "kernel_profile_differences.csv", differences)
    save_csv(out / "halva_checks.csv", checks["checks"])
    scalar_rows = [
        dict(
            factor_a=a,
            factor_b=b,
            covariance=float(archive["scalar_covariance"][i, j]),
            correlation=float(archive["scalar_correlation"][i, j]),
        )
        for i, a in enumerate(CONTENT_FACTOR_NAMES)
        for j, b in enumerate(CONTENT_FACTOR_NAMES)
    ]
    save_csv(out / "scalar_content_covariance.csv", scalar_rows)
    result = dict(
        n_subjects=args.kernel_samples,
        subject_offset=args.subject_offset,
        grid=args.kernel_grid,
        voxel_indices=indices.tolist(),
        latent_sha256=raw_hash.hexdigest(),
        components=components,
        field_prior=inner.field_prior,
        components_standardized_by_sampler=inner.field_prior != "iid",
        accepted_subjects_requiring_redraw=resampled,
        comparisons=comparisons,
        applicability=checks,
        notes=[
            "K(u,v) is unbiased ensemble covariance across subjects after subtracting the ensemble mean at each location. No per-realization centering or normalization is added by this audit.",
            "Common-grid fields use exact renderer interpolation then sample the listed voxel indices. Native covariance matrices are also saved, but differing native grids are not directly compared.",
            "Lag profiles average covariance over location and axes as labeled; this is not a stationarity assumption or a fitted squared-exponential kernel.",
            "Trace normalization removes overall variance before comparing kernel shape. Split-half distances illustrate estimation variability, not a formal equality test.",
            "Profile confidence intervals resample whole subjects and recompute their ensemble means; they are pointwise, not simultaneous or corrected for multiple comparisons.",
            "The scalar 9x9 covariance concerns different factors across subjects. Broadcasting scalars across space would manufacture constant, degenerate kernels and is not done.",
            "Effective covariance is multiplier squared times raw covariance for deformation/fissure. In clean-content runs it is zero, even when raw kernels differ.",
            "Sample covariance rank is bounded by n_subjects-1. Finite-sample rank deficiency or differences alone do not establish population identifiability.",
        ],
    )
    (out / "kernel_report.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    kernel_plot(archive, profiles, comparisons, out)
    print("  Hälvä theorem guarantee is not established for this renderer; see halva_checks.csv.", flush=True)
    return result


def kernel_plot(archive, profiles, comparisons, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    im = axes[0].imshow(archive["scalar_correlation"], vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_xticks(range(9), CONTENT_FACTOR_NAMES, rotation=65, ha="right", fontsize=8)
    axes[0].set_yticks(range(9), CONTENT_FACTOR_NAMES, fontsize=8)
    axes[0].set_title("Scalar content correlation\nThis is not a spatial kernel", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.6)
    names = archive["component_names"].tolist()
    for name in names:
        rows = [r for r in profiles if r["component"] == name and r["axis"] == "mean_axes"]
        x = [r["mean_distance_renderer_units"] for r in rows]
        y = [r["normalized_covariance"] for r in rows]
        (line,) = axes[1].plot(
            x, y, "o-", label=name + (" (image path OFF)" if not rows[0]["active_structural_path"] else "")
        )
        axes[1].fill_between(
            x,
            [r["normalized_ci95_low"] for r in rows],
            [r["normalized_ci95_high"] for r in rows],
            color=line.get_color(),
            alpha=0.15,
        )
    axes[1].set_title("Raw interpolated latent fields\nCovariance / mean point variance", fontsize=10)
    axes[1].set_xlabel("Mean axial separation (renderer coordinates)")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].legend(fontsize=8)
    distances = np.zeros((len(names), len(names)))
    for row in comparisons:
        a, b = names.index(row["component_a"]), names.index(row["component_b"])
        distances[a, b] = distances[b, a] = row["trace_normalized_kernel_distance"]
    im = axes[2].imshow(distances, vmin=0, vmax=2, cmap="viridis")
    axes[2].set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    axes[2].set_yticks(range(len(names)), names, fontsize=8)
    for (a, b), value in np.ndenumerate(distances):
        axes[2].text(b, a, f"{value:.2f}", ha="center", va="center", color="white" if value < 1 else "black")
    axes[2].set_title("Trace-normalized kernel distance\nCompare split-half variability in CSV", fontsize=10)
    fig.colorbar(im, ax=axes[2], shrink=0.6)
    fig.savefig(out / "kernel_diagnostics.png", dpi=140)
    plt.close(fig)
