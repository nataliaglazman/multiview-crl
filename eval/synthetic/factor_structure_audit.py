"""Finite factor-response maps, subspace overlap, and ground-truth spatial kernels.

python -m eval.synthetic.factor_structure_audit --run-dir RUN --out-dir NEW_OUTPUT

Generator diagnostics only: no checkpoint, encoder training or fitted probe.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from data.datasets import SyntheticBrainDataset
from eval.encoder.encoder_target_protocol import VIEWS, digest, provenance, save_csv, save_report
from eval.metrics.dci import CONTENT_FACTOR_NAMES as FACTORS
from eval.synthetic.synthetic_dataset import LesionPlacementError
from eval.ventricle.ventricle_routing import normalization_affine

ANATOMY = (0, 1, 5, 6, 7)
BLOCKS = {
    "lesion_vs_anatomy": ((2, 3, 4), ANATOMY),
    "sulcal_vs_anatomy": ((8,), ANATOMY),
    "lesion_sulcal_vs_anatomy": ((2, 3, 4, 8), ANATOMY),
    "lesion_vs_anatomy_and_sulcal": ((2, 3, 4), (*ANATOMY, 8)),
}


def restore_dataset(cfg, count, split="test"):
    """Restore all supported generator options, including the GP/TP field options."""
    accepted = inspect.signature(SyntheticBrainDataset.__init__).parameters
    kwargs = {k: v for k, v in cfg.items() if k.startswith("synthetic_") and k in accepted}
    res = int(cfg["res"]) if cfg.get("res") is not None else int(min(cfg.get("spatial_size") or [64]))
    seed = cfg.get("data_seed")
    if seed is None:
        seed = cfg.get("synthetic_seed", cfg.get("seed", 42))
    kwargs.update(
        synthetic_seed=seed,
        synthetic_num_samples=count,
        synthetic_num_samples_per_mode=None,
        synthetic_n_content=cfg.get("n_content", cfg.get("synthetic_n_content", 9)),
        synthetic_n_style=cfg.get("n_style", cfg.get("synthetic_n_style", 3)),
        synthetic_style_alignment_pairs=False,
    )
    if kwargs.get("synthetic_mode", "pseudo_mri") != "pseudo_mri" or kwargs["synthetic_n_content"] != 9:
        raise ValueError("This audit expects the nine-factor pseudo_mri generator")
    return SyntheticBrainDataset(mode=split, spatial_size=(res,) * 3, cache=False, **kwargs), kwargs


@torch.inference_mode()
def context(ds, idx):
    raw0, raw1, lat = ds._inner[idx]
    mask = lat["brain_mask"]
    norm = ds.normalize_views(raw0, raw1, mask, mask.clone())
    return dict(
        index=idx,
        lat=lat,
        seed=ds._inner.sample_seed_for(idx),
        affines=[normalization_affine(a, b, mask) for a, b in zip((raw0, raw1), norm)],
    )


@torch.inference_mode()
def render(ds, ctx, content):
    """Original placement rule, RNG and normalization affine; endpoint's own foreground."""
    inner, lat = ds._inner, ctx["lat"]
    tissue, lesion = inner.renderer.render_structure(
        content,
        lat["z_deformation"],
        lat["z_fissure"],
        "cpu",
        clean=inner.clean_content,
        z_lesion=lat.get("z_lesion"),
    )
    images = []
    for v, modality in enumerate(("T1", "FLAIR")):
        raw = inner.renderer.render_modality(
            tissue, lesion, lat[f"z_style_v{v + 1}"], modality, ctx["seed"] * 2 + v, "cpu"
        )
        gain, bias = ctx["affines"][v]
        images.append(((raw * gain + bias) * (tissue > 0)[None]).numpy()[0])
    mass = float(lesion.sum())
    center = ((lesion[..., None] * inner.renderer.coords).sum((0, 1, 2)) / mass).numpy() if mass else np.full(3, np.nan)
    images = np.stack(images)
    if not np.isfinite(images).all():
        raise ValueError("Non-finite rendered image")
    return images, dict(lesion_mass=mass, centroid=center, foreground_voxels=int((tissue > 0).sum()))


def region_indices(shape, grid):
    if grid < 1 or grid > min(shape):
        raise ValueError("region-grid must fit the image")
    a, b, c = [np.arange(n) * grid // n for n in shape]
    return ((a[:, None, None] * grid + b[None, :, None]) * grid + c[None, None, :]).ravel()


def energy_count(energy, fraction=0.9):
    energy = np.sort(np.asarray(energy).ravel())[::-1]
    return int(np.searchsorted(np.cumsum(energy), fraction * energy.sum()) + 1) if energy.sum() else 0


def footprint(delta, eps, regions, atol):
    d = np.asarray(delta, np.float64).ravel()
    energy = d * d
    total = energy.sum()
    active = bool(np.max(np.abs(d)) > atol)
    patch_energy = np.bincount(regions, weights=energy)
    return dict(
        active=active,
        pair_delta_rms=float(np.sqrt(energy.mean())),
        fd_rms=float(np.sqrt(energy.mean()) / (2 * eps)),
        changed_voxel_fraction=float(np.mean(np.abs(d) > atol)),
        energy90_voxel_fraction=energy_count(energy) / len(d) if active else np.nan,
        effective_regions=float(total**2 / (patch_energy @ patch_energy)) if active else np.nan,
        energy90_regions=energy_count(patch_energy) if active else np.nan,
        gap_survival=float(abs(d.sum()) / np.abs(d).sum()) if active else np.nan,
    )


def inverse_and_basis(gram, rtol):
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    keep = values > rtol**2 * max(1.0, float(values.max(initial=0)))
    basis = vectors[:, keep] / np.sqrt(values[keep])
    return basis @ basis.T, basis


def response_geometry(maps, active, valid, rtol=1e-5):
    """Small Gram matrices avoid repeated voxel-space SVDs. Zero directions stay zero."""
    flat = np.asarray(maps, np.float64).reshape(len(maps), -1)
    norms = np.linalg.norm(flat, axis=1)
    normalized = np.divide(flat, norms[:, None], out=np.zeros_like(flat), where=np.asarray(active)[:, None])
    gram = normalized @ normalized.T

    def residual_fraction(k, refs):
        if not active[k] or not all(valid[j] for j in refs):
            return np.nan
        if not refs:
            return 1.0
        inv, _ = inverse_and_basis(gram[np.ix_(refs, refs)], rtol)
        cross = gram[k, list(refs)]
        return float(np.clip(1 - cross @ inv @ cross, 0, 1))

    per_factor = []
    for k in range(len(maps)):
        refs = tuple(j for j in ANATOMY if j != k)
        others = tuple(j for j in range(len(maps)) if j != k)
        per_factor.append(
            dict(
                residual_energy_vs_anatomy=residual_fraction(k, refs),
                residual_energy_vs_all_others=residual_fraction(k, others),
                anatomy_reference_complete=bool(all(valid[j] for j in refs)),
                all_other_references_complete=bool(all(valid[j] for j in others)),
                anatomy_active_directions=int(sum(active[j] for j in refs)),
            )
        )
    blocks = []
    for name, (target, refs) in BLOCKS.items():
        row = dict(
            block=name,
            complete=bool(all(valid[j] for j in (*target, *refs))),
            target_dimensions=len(target),
            reference_dimensions=len(refs),
            target_rank=np.nan,
            reference_rank=np.nan,
            residual_target_rank=np.nan,
            target_full_rank=False,
            min_principal_angle_deg=np.nan,
            min_residual_singular_value=np.nan,
        )
        if row["complete"]:
            a, b, cross = gram[np.ix_(target, target)], gram[np.ix_(refs, refs)], gram[np.ix_(target, refs)]
            _, qa = inverse_and_basis(a, rtol)
            invb, qb = inverse_and_basis(b, rtol)
            residual = a - cross @ invb @ cross.T
            eigen = np.maximum(np.linalg.eigvalsh((residual + residual.T) / 2), 0)
            rank = int(np.sum(eigen > rtol**2 * max(1.0, float(np.linalg.eigvalsh(a).max(initial=0)))))
            row.update(
                target_rank=qa.shape[1],
                reference_rank=qb.shape[1],
                residual_target_rank=rank,
                target_full_rank=qa.shape[1] == len(target),
                min_residual_singular_value=float(np.sqrt(eigen.min())),
            )
            if qa.shape[1] and qb.shape[1]:
                cosine = np.linalg.svd(qa.T @ cross @ qb, compute_uv=False).max()
                row["min_principal_angle_deg"] = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
        blocks.append(row)
    cosines = gram.copy()
    cosines[~np.asarray(active), :] = np.nan
    cosines[:, ~np.asarray(active)] = np.nan
    return per_factor, blocks, cosines, normalized, gram, norms


def residual_map(k, refs, normalized, gram, norms, rtol):
    inv, _ = inverse_and_basis(gram[np.ix_(refs, refs)], rtol)
    coefficients = inv @ gram[list(refs), k]
    return (normalized[k] - coefficients @ normalized[list(refs)]) * norms[k]


def write_nifti(path, array):
    import nibabel as nib

    image = nib.Nifti1Image(np.asarray(array, np.float32), np.eye(4))
    image.set_qform(np.eye(4), code=0)
    image.set_sform(np.eye(4), code=2)
    image.header.set_xyzt_units("unknown")
    image.header["descrip"] = b"Synthetic index space; signed response; no physical orientation"
    nib.save(image, path)


def response_audit(ds, args, out):
    rows, block_rows, cosine_rows = [], [], []
    regions = region_indices((ds.res,) * 3, args.region_grid)
    hashes = hashlib.sha256()
    nifti = out / "nifti"
    if args.nifti_subjects:
        nifti.mkdir()
    for n, idx in enumerate(range(args.subject_offset, args.subject_offset + args.num_samples)):
        ctx = context(ds, idx)
        base, _ = render(ds, ctx, ctx["lat"]["z_content"])
        repeated, _ = render(ds, ctx, ctx["lat"]["z_content"])
        if not np.array_equal(base, repeated):
            raise ValueError("Identical-input renderer replay changed")
        hashes.update(base.tobytes())
        if n < args.nifti_subjects:
            for v, view in enumerate(VIEWS):
                write_nifti(nifti / f"subject{idx}_{view}_baseline.nii.gz", base[v])
        for eps in args.eps:
            maps = np.zeros((2, 9, ds.res, ds.res, ds.res), np.float32)
            valid = np.ones(9, bool)
            metadata = []
            for k, factor in enumerate(FACTORS):
                z = ctx["lat"]["z_content"]
                minus, plus = z.clone(), z.clone()
                minus[k] -= eps
                plus[k] += eps
                info = dict(
                    subject_id=idx,
                    eps=eps,
                    factor=factor,
                    factor_index=k,
                    latent_minus=float(minus[k]),
                    latent_plus=float(plus[k]),
                    lesion_displacement_vox=np.nan,
                    foreground_voxel_delta=np.nan,
                    render_error="",
                )
                try:
                    a, ma = render(ds, ctx, minus)
                    b, mb = render(ds, ctx, plus)
                    maps[:, k] = b - a
                    hashes.update(a.tobytes())
                    hashes.update(b.tobytes())
                    info.update(
                        lesion_displacement_vox=float(
                            np.linalg.norm(mb["centroid"] - ma["centroid"]) * (ds.res - 1) / 2
                        ),
                        foreground_voxel_delta=mb["foreground_voxels"] - ma["foreground_voxels"],
                    )
                except LesionPlacementError as error:
                    valid[k] = False  # Never redraw the subject or reinterpret failure as zero response.
                    info["render_error"] = str(error)
                metadata.append(info)
            for v, view in enumerate(VIEWS):
                metrics = [footprint(m, eps, regions, args.response_atol) for m in maps[v]]
                active = [valid[k] and m["active"] for k, m in enumerate(metrics)]
                factors, blocks, cosines, normalized, gram, norms = response_geometry(
                    maps[v], active, valid, args.svd_rtol
                )
                for k in range(9):
                    if not valid[k]:
                        metrics[k] = {key: False if key == "active" else np.nan for key in metrics[k]}
                    rows.append(
                        dict(
                            **metadata[k],
                            view=view,
                            status="render_failed" if not valid[k] else "ok" if active[k] else "zero_response",
                            **metrics[k],
                            **factors[k],
                        )
                    )
                    for j in range(9):
                        cosine_rows.append(
                            dict(
                                subject_id=idx,
                                eps=eps,
                                view=view,
                                factor_a=FACTORS[k],
                                factor_b=FACTORS[j],
                                cosine=float(cosines[k, j]),
                            )
                        )
                    if n < args.nifti_subjects and valid[k]:
                        prefix = f"subject{idx}_{view}_eps{eps:g}_{FACTORS[k]}"
                        write_nifti(nifti / f"{prefix}_delta.nii.gz", maps[v, k])
                        refs = tuple(j for j in ANATOMY if j != k)
                        if k in (2, 3, 4, 8) and all(valid[j] for j in refs):
                            residual = residual_map(k, refs, normalized, gram, norms, args.svd_rtol)
                            write_nifti(nifti / f"{prefix}_residual_vs_anatomy.nii.gz", residual.reshape((ds.res,) * 3))
                block_rows.extend(dict(subject_id=idx, eps=eps, view=view, **b) for b in blocks)
        print(f"  image responses: {n + 1}/{args.num_samples} subjects", flush=True)
    if args.nifti_subjects:
        (nifti / "README.txt").write_text(
            "delta = image(z+eps) - image(z-eps), in fixed normalized input units; divide by 2eps for finite derivatives.\n"
            "residual_vs_anatomy removes projection onto brain_size, ventricle_size, cortical_thickness, temporal_atrophy, lr_asymmetry responses.\n"
            "Use a symmetric signed colormap for response maps; baseline images use greyscale.\n"
            "Identity affine, unknown spatial units, axes preserved; no anatomical orientation or millimetre scale is implied.\n"
        )
    summary = []
    keys = (
        "pair_delta_rms",
        "energy90_voxel_fraction",
        "effective_regions",
        "gap_survival",
        "residual_energy_vs_anatomy",
        "residual_energy_vs_all_others",
        "lesion_displacement_vox",
    )
    for eps in args.eps:
        for view in VIEWS:
            for factor in FACTORS:
                selected = [r for r in rows if (r["eps"], r["view"], r["factor"]) == (eps, view, factor)]
                row = dict(
                    eps=eps,
                    view=view,
                    factor=factor,
                    n_subjects=len(selected),
                    n_failed=sum(r["status"] == "render_failed" for r in selected),
                    n_zero=sum(r["status"] == "zero_response" for r in selected),
                )
                for key in keys:
                    values = np.array([r[key] for r in selected], float)
                    values = values[np.isfinite(values)]
                    row[key + "_n"] = len(values)
                    for q, label in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
                        row[key + "_" + label] = float(np.quantile(values, q)) if len(values) else np.nan
                summary.append(row)
    for name, table in (
        ("responses", rows),
        ("blocks", block_rows),
        ("cosines", cosine_rows),
        ("response_summary", summary),
    ):
        save_csv(out / f"{name}.csv", table)
    response_plot(summary, args.eps, out)
    return dict(
        input_sha256=hashes.hexdigest(),
        response_summary=summary,
        n_failed_pairs=sum(r["status"] == "render_failed" for r in rows) // 2,
    )


def response_plot(summary, epsilons, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    metrics = (
        ("energy90_voxel_fraction_median", "Voxels containing 90% of effect energy (%)"),
        ("residual_energy_vs_anatomy_median", "Effect energy outside anatomy span (%)"),
        ("residual_energy_vs_all_others_median", "Effect energy outside all other factors (%)"),
    )
    for v, view in enumerate(VIEWS):
        for j, (key, title) in enumerate(metrics):
            matrix = (
                np.array(
                    [
                        [
                            next(r[key] for r in summary if (r["view"], r["factor"], r["eps"]) == (view, f, e))
                            for e in epsilons
                        ]
                        for f in FACTORS
                    ]
                )
                * 100
            )
            ax = axes[v, j]
            im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="viridis")
            ax.set_xticks(range(len(epsilons)), [f"±{e:g}" for e in epsilons])
            ax.set_yticks(range(9), FACTORS)
            ax.set_title(f"{view.upper()} — {title}", fontsize=10)
            for (row, col), value in np.ndenumerate(matrix):
                ax.text(
                    col,
                    row,
                    f"{value:.1f}" if np.isfinite(value) else "n/a",
                    ha="center",
                    va="center",
                    color="white" if not np.isfinite(value) or value < 50 else "black",
                    fontsize=8,
                )
            fig.colorbar(im, ax=ax, shrink=0.65)
    fig.suptitle(
        "Median per-subject finite responses; zeros/failed renders counted separately in CSV\nAnatomy is a chosen reference block, not a proven global partition",
        fontsize=11,
    )
    fig.savefig(out / "response_summary.png", dpi=140)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--audit-report", type=Path, help="Existing report.json containing saved settings")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--subject-offset", type=int, default=1000)
    p.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    p.add_argument("--region-grid", type=int, default=4)
    p.add_argument("--response-atol", type=float, default=1e-6)
    p.add_argument("--svd-rtol", type=float, default=1e-5)
    p.add_argument("--nifti-subjects", type=int, default=1)
    p.add_argument("--kernel-samples", type=int, default=256)
    p.add_argument("--kernel-grid", type=int, default=8)
    p.add_argument("--kernel-bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--cpu-threads", type=int, default=1)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--kernels-only", action="store_true")
    mode.add_argument("--skip-kernels", action="store_true")
    args = p.parse_args(argv)
    if min(args.num_samples, args.cpu_threads) < 1 or args.subject_offset < 0 or args.nifti_subjects < 0:
        p.error("Need positive sample/thread counts and nonnegative offset/NIfTI count")
    if args.kernel_samples < 8 or args.kernel_grid < 2 or args.kernel_bootstrap < 0:
        p.error("Need >=8 kernel subjects, grid>=2 and nonnegative bootstrap")
    if (
        not all(np.isfinite(e) and e > 0 for e in args.eps)
        or not np.isfinite(args.response_atol)
        or args.response_atol < 0
    ):
        p.error("Need finite positive eps and nonnegative response-atol")
    if not np.isfinite(args.svd_rtol) or not 0 < args.svd_rtol < 1:
        p.error("svd-rtol must lie in (0,1)")
    args.eps = sorted(set(args.eps))
    return args


def main(argv=None):
    args = parse_args(argv)
    source = args.audit_report or args.run_dir / "settings.json"
    source_hash = digest(source)
    data = json.loads(source.read_text())
    cfg = data["settings"] if args.audit_report else data
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = provenance(cfg, args, "cpu")
    report.update(
        source_settings_sha256=source_hash,
        evaluation_only=True,
        target_names=FACTORS,
        scope="Generator finite responses and latent spatial covariance; not encoder identifiability.",
        notes=[
            "Central differences vary raw content controls with all other controls, acquisition random draws and the original normalization affine fixed.",
            "The endpoint uses its own foreground mask and the native lesion placement. Anatomy changes may consequently reposition the lesion; measured displacement is recorded.",
            "Anatomy reference = brain_size, ventricle_size, cortical_thickness, temporal_atrophy, lr_asymmetry. This is a hypothesis, not a learned or proven partition.",
            "Maps are analyzed per subject before aggregation. Opposing effects across subjects are never averaged away before projection.",
            "Projection energy is uncentered Euclidean image energy, not probe R², a noise-whitened information bound, or an identifiability guarantee.",
            "Principal angles concern active directions. Read ranks and zero counts: missing/quantized directions do not establish a separable full block.",
            "Placement failures are retained and invalidate projections requiring those reference columns; subjects are never redrawn.",
            "Finite-step and thresholded-renderer effects need not converge to a useful smooth Jacobian. Compare multiple eps values.",
        ],
    )
    save_report(args.out_dir, report)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.cpu_threads)
        count = max(64, args.subject_offset + max(args.num_samples, args.kernel_samples))
        ds, effective = restore_dataset(cfg, count)
        report["generator_arguments"] = effective
        if args.region_grid > ds.res or args.kernel_grid > ds.res:
            raise ValueError("Analysis grids must fit the image resolution")
        with threadpool_limits(limits=args.cpu_threads):
            if not args.kernels_only:
                report.update(response_audit(ds, args, args.out_dir))
            if not args.skip_kernels:
                from eval.synthetic.latent_spatial_kernels import kernel_audit

                report["kernels"] = kernel_audit(ds, args, args.out_dir)
        if digest(source) != source_hash:
            raise ValueError("Source settings changed during audit")
        report.update(status="complete", source_settings_unchanged=True)
        save_report(args.out_dir, report)
        print(f"Saved factor structure audit: {args.out_dir}")
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    main()
