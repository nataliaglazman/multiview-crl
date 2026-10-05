"""Relate rendered lesion contrast to existing frozen-probe errors; export NIfTI examples.

python -m eval.encoder.encoder_lesion_contrast --audit-dir MOVEMENT_AUDIT --out-dir NEW_OUTPUT

Reuses encoder_lesion_intervention predictions. No checkpoint, feature extraction,
probe fitting or encoder training. Exact image replay is required before analysis.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.stats import rankdata

from eval.diagnostics.style_path_audit import effective_style
from eval.encoder.encoder_lesion_intervention import movement_metrics
from eval.encoder.encoder_target_protocol import VIEWS, dataset, digest, provenance, save_csv, save_report
from eval.lesion.lesion_routing import render_pair
from eval.ventricle.ventricle_routing import normalization_affine

PREDICTION = re.compile(r"(t1|flair)_g([1-9][0-9]*)_(backbone|projected)_(ridge|rbf)_(observed|shuffled)")


def load_audit(directory):
    """Require explicit pair identity and matching initial/trained evaluation images."""
    report = json.loads((directory / "report.json").read_text())
    if report.get("status") != "complete" or not report.get("evaluation_only"):
        raise ValueError("Need a completed encoder lesion-movement audit")
    cfg, args = report["settings"], report["arguments"]
    if cfg.get("synthetic_lesion_placement") != "wm_interior":
        raise ValueError("Contrast against healthy white matter requires wm_interior lesions")
    with (directory / "pairs.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    expected = [
        (idx, axis)
        for idx in range(args["subject_offset"], args["subject_offset"] + args["num_samples"])
        for axis in args["axes"]
    ]
    actual = [(int(row["subject_id"]), row["intervention_axis"]) for row in pairs]
    if actual != expected or not pairs:
        raise ValueError("pairs.csv does not match the saved subject/axis order")
    if any(not np.isclose(float(row["eps"]), args["eps"], rtol=0, atol=1e-12) for row in pairs):
        raise ValueError("pairs.csv intervention size differs from report")
    truth = np.asarray([[[float(row[f"centroid_{end}_{axis}"]) for axis in "xyz"] for end in "ab"] for row in pairs])
    subjects = np.asarray([idx for idx, _ in actual])
    axes = np.asarray([axis for _, axis in actual])
    arms = list(report["cohorts"])
    if "trained" not in arms or set(arms) - {"trained", "initial"}:
        raise ValueError("Expected trained and optional initial arms")
    paths = [directory / name for name in ("report.json", "pairs.csv")]
    predictions = {}
    for arm in arms:
        cohort = report["cohorts"][arm]["moves"]
        if cohort["input_sha256"] != report["cohorts"]["trained"]["moves"]["input_sha256"]:
            raise ValueError("Initial/trained image cohorts differ")
        if cohort["ids"] != sorted(set(subjects.tolist())) or cohort["n_pairs"] != len(pairs):
            raise ValueError("Pair cohort identity/count differs from report")
        path = directory / f"{arm}_predictions.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as bank:
            if not np.array_equal(bank["subject_id"], subjects) or not np.array_equal(bank["intervention_axis"], axes):
                raise ValueError("Prediction subject/axis order differs from pairs.csv")
            if not np.array_equal(bank["truth"], truth):
                raise ValueError("Prediction ground truth differs from pairs.csv")
            found = set(bank.files) - {"truth", "subject_id", "intervention_axis"}
            expected_keys = {
                f"{r['view']}_g{r['grid']}_{r['stage']}_{r['probe']}_{r['condition']}"
                for r in report["summary"]
                if r["arm"] == arm and r["intervention_axis"] == "all"
            }
            if not found or found != expected_keys:
                raise ValueError("Prediction keys differ from the saved audit summary")
            for key in sorted(found):
                if not PREDICTION.fullmatch(key) or bank[key].shape != truth.shape or not np.isfinite(bank[key]).all():
                    raise ValueError(f"Invalid prediction array: {arm}/{key}")
                predictions[arm, key] = bank[key].copy()
    if not np.isfinite(truth).all():
        raise ValueError("Non-finite ground truth")
    return report, pairs, truth, predictions, paths


@torch.inference_mode()
def render_reference(ds, idx):
    """Remove lesion only; preserve acquisition RNG and the original normalization affine."""
    inner = ds._inner
    raw0, raw1, lat = inner[idx]
    mask = lat["brain_mask"]
    normalized = ds.normalize_views(raw0, raw1, mask, mask.clone())
    affines = [normalization_affine(raw, norm, mask) for raw, norm in zip((raw0, raw1), normalized)]
    tissue, lesion = inner.renderer.render_structure(
        lat["z_content"],
        lat["z_deformation"],
        lat["z_fissure"],
        "cpu",
        clean=inner.clean_content,
        z_lesion=lat.get("z_lesion"),
    )
    references, styles = [], []
    for v, modality in enumerate(("T1", "FLAIR")):
        z = lat[f"z_style_v{v + 1}"]
        raw = inner.renderer.render_modality(
            tissue,
            torch.zeros_like(lesion),
            z,
            modality,
            view_seed=inner.sample_seed_for(idx) * 2 + v,
            device="cpu",
        )
        norm_gain, norm_bias = affines[v]
        references.append((raw * norm_gain + norm_bias) * mask)
        gain, bias, sigma = effective_style(z, inner.renderer.style_scale)
        styles.append(
            dict(
                acquisition_gain=float(gain),
                acquisition_bias=float(bias),
                preblur_noise_sigma=float(sigma),
                normalized_preblur_noise_sigma=float(abs(norm_gain) * sigma),
                normalization_gain=float(norm_gain),
                normalization_bias=float(norm_bias),
            )
        )
    return dict(images=references, tissue=tissue.numpy(), styles=styles)


def contrast_metrics(image, reference, lesion, tissue):
    """Primary contrast is the signed effect of the lesion inside its exact support."""
    image, reference = np.asarray(image), np.asarray(reference)
    if not lesion.any() or not np.all(tissue[lesion] == 2):
        raise ValueError("Expected a nonempty lesion entirely in white matter")
    if not np.isfinite(image).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite rendered image")
    difference = image.astype(np.float64) - reference
    affected = maximum_filter(lesion, size=3)
    if (~affected).any() and np.abs(difference[~affected]).max() > 2e-6:
        raise ValueError("Lesion-free reference differs outside the lesion/blur support")
    shell = maximum_filter(lesion, size=7) & ~affected & (tissue == 2)
    signed = float(difference[lesion].mean())
    local = float(image[lesion].mean() - image[shell].mean()) if shell.sum() >= 8 else np.nan
    return dict(
        matched_contrast=abs(signed),
        signed_matched_contrast=signed,
        lesion_effect_rms=float(np.sqrt(np.mean(difference[lesion] ** 2))),
        local_wm_contrast=abs(local),
        signed_local_wm_contrast=local,
        lesion_mean=float(image[lesion].mean()),
        reference_wm_mean=float(reference[lesion].mean()),
        shell_wm_mean=float(image[shell].mean()) if shell.sum() >= 8 else np.nan,
        lesion_voxels=int(lesion.sum()),
        shell_voxels=int(shell.sum()),
    )


def replay_contrast(ds, source, pairs, truth):
    image_hash, rows = hashlib.sha256(), []
    current, reference = None, None
    for i, pair in enumerate(pairs):
        idx, axis = int(pair["subject_id"]), pair["intervention_axis"]
        if idx != current:
            reference = render_reference(ds, idx)
            current = idx
        sample = render_pair(ds, idx, axis, float(pair["eps"]))
        actual = 2 * np.asarray(sample["centroids"]) / (ds.res - 1) - 1
        if not np.array_equal(actual, truth[i]):
            raise ValueError(f"Rendered centroids differ from saved truth for subject {idx}/{axis}")
        for end, lesion in zip("ab", sample["lesions"]):
            image_hash.update(torch.stack(sample[end]).contiguous().numpy().tobytes())
            for v, view in enumerate(VIEWS):
                row = dict(
                    subject_id=idx,
                    intervention_axis=axis,
                    endpoint=end,
                    view=view,
                    image_sha256=hashlib.sha256(sample[end][v].contiguous().numpy().tobytes()).hexdigest(),
                )
                row.update(
                    contrast_metrics(
                        sample[end][v].numpy()[0], reference["images"][v].numpy()[0], lesion, reference["tissue"]
                    )
                )
                row.update(reference["styles"][v])
                rows.append(row)
        if (i + 1) % 12 == 0 or i + 1 == len(pairs):
            print(f"  contrast: replayed {i + 1}/{len(pairs)} image pairs", flush=True)
    expected = source["cohorts"]["trained"]["moves"]["input_sha256"]
    if image_hash.hexdigest() != expected:
        raise ValueError(
            "Rendered image SHA256 differs from the movement audit. Refusing to associate new images with old predictions. "
            "Use the original renderer, settings and PyTorch environment (including CPU thread count)."
        )
    return rows, expected


def mean_finite(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def subject_contrasts(rows):
    result = {}
    for idx, view in sorted({(r["subject_id"], r["view"]) for r in rows}):
        block = [r for r in rows if (r["subject_id"], r["view"]) == (idx, view)]
        result[idx, view] = {
            key: mean_finite([r[key] for r in block])
            for key in (
                "matched_contrast",
                "local_wm_contrast",
                "normalized_preblur_noise_sigma",
                "lesion_voxels",
            )
        }
    return result


def contrast_groups(values):
    """Cut on contrast alone. Do not split ties to manufacture low/high groups."""
    values = np.asarray(values)
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    if cuts[0] == cuts[1]:
        groups = np.where(values < cuts[0], "low", np.where(values > cuts[1], "high", "middle"))
    else:
        groups = np.where(values <= cuts[0], "low", np.where(values > cuts[1], "high", "middle"))
    return groups, cuts.tolist()


def rank_correlation(x, y, controls=None):
    """Spearman or partial Spearman; undefined, small or rank-deficient cases stay NaN."""
    x, y = rankdata(x).astype(float), rankdata(y).astype(float)
    covariates = np.empty((len(x), 0)) if controls is None else np.asarray(controls, dtype=float)
    covariates = covariates[:, np.ptp(covariates, axis=0) > 1e-12]
    design = np.column_stack((np.ones(len(x)), *[rankdata(c) for c in covariates.T]))
    if len(x) < max(8, design.shape[1] + 3) or np.linalg.matrix_rank(design) < design.shape[1]:
        return np.nan
    x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denominator, -1, 1)) if min(np.linalg.norm(x), np.linalg.norm(y)) > 1e-8 else np.nan


def association(x, y, controls, bootstrap, seed):
    x, y, controls = np.asarray(x), np.asarray(y), np.asarray(controls)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(1)
    x, y, controls = x[valid], y[valid], controls[valid]
    result = dict(n_subjects=len(x))
    for name, covariates in (("spearman", None), ("partial_spearman", controls)):
        point = rank_correlation(x, y, covariates) if len(x) else np.nan
        estimates = []
        if np.isfinite(point):
            rng = np.random.default_rng(seed)
            for _ in range(bootstrap):
                selected = rng.integers(len(x), size=len(x))
                value = rank_correlation(x[selected], y[selected], None if covariates is None else covariates[selected])
                if np.isfinite(value):
                    estimates.append(value)
        low, high = (
            np.quantile(estimates, [0.025, 0.975]) if len(estimates) >= max(10, 0.8 * bootstrap) else (np.nan, np.nan)
        )
        result.update(
            {
                name: point,
                name + "_ci95_low": float(low),
                name + "_ci95_high": float(high),
                name + "_valid_bootstraps": len(estimates),
            }
        )
    return result


def analyze(pairs, truth, predictions, contrasts, resolution, bootstrap, seed):
    ids = np.array([int(row["subject_id"]) for row in pairs])
    unique = np.unique(ids)
    subjects = subject_contrasts(contrasts)
    groups, cuts = {}, {}
    for view in VIEWS:
        labels, cuts[view] = contrast_groups([subjects[idx, view]["matched_contrast"] for idx in unique])
        groups[view] = dict(zip(unique, labels))
    scale = (resolution - 1) / 2
    delta = (truth[:, 1] - truth[:, 0]) * scale
    distance = np.linalg.norm(delta, axis=1)
    moved = np.sum((truth[:, 1] - truth[:, 0]) ** 2, axis=1) > 1e-16
    subject_rows, group_rows, correlations = [], [], []
    for (arm, key), predicted in predictions.items():
        view, grid, stage, probe, condition = PREDICTION.fullmatch(key).groups()
        info = dict(arm=arm, view=view, grid=int(grid), stage=stage, probe=probe, condition=condition)
        endpoint_error = np.linalg.norm((predicted - truth) * scale, axis=2).mean(1)
        movement_error = np.linalg.norm((predicted[:, 1] - predicted[:, 0]) * scale - delta, axis=1)
        block = []
        for idx in unique:
            chosen = ids == idx
            active = chosen & moved
            block.append(
                dict(
                    **info,
                    subject_id=int(idx),
                    own_contrast_group=groups[view][idx],
                    t1_contrast_group=groups["t1"][idx],
                    **subjects[idx, view],
                    n_pairs=int(chosen.sum()),
                    n_moved=int(active.sum()),
                    true_displacement_vox=mean_finite(distance[active]),
                    endpoint_error_vox=mean_finite(endpoint_error[chosen]),
                    movement_error_vox=mean_finite(movement_error[active]),
                    relative_movement_error=mean_finite(movement_error[active] / distance[active]),
                )
            )
        subject_rows.extend(block)
        # Group by the same T1 subjects for BOTH modalities, as well as each view's own contrast.
        for grouping in ("own_view", "t1_matched"):
            labels = groups[view if grouping == "own_view" else "t1"]
            for group in ("low", "middle", "high"):
                selected = np.array([labels[idx] == group for idx in ids])
                members = [r for r in block if labels[r["subject_id"]] == group]
                metrics = movement_metrics(truth[selected], predicted[selected], ids[selected], scale, bootstrap, seed)
                group_rows.append(
                    dict(
                        **info,
                        grouping=grouping,
                        contrast_group=group,
                        n_subjects=len(members),
                        mean_matched_contrast=mean_finite([r["matched_contrast"] for r in members]),
                        all_endpoint_error_vox=mean_finite(endpoint_error[selected]),
                        **metrics,
                    )
                )
        for error_name in ("endpoint_error_vox", "relative_movement_error"):
            correlations.append(
                dict(
                    **info,
                    error_metric=error_name,
                    **association(
                        [r["matched_contrast"] for r in block],
                        [r[error_name] for r in block],
                        [[r["true_displacement_vox"], r["normalized_preblur_noise_sigma"]] for r in block],
                        bootstrap,
                        seed,
                    ),
                )
            )
    # Paired view differences use the same subject, not separate contrast-selected cohorts.
    indexed = {
        (r["arm"], r["grid"], r["stage"], r["probe"], r["condition"], r["subject_id"], r["view"]): r
        for r in subject_rows
    }
    paired = []
    for identity, t1 in indexed.items():
        if identity[-1] != "t1" or (*identity[:-1], "flair") not in indexed:
            continue
        flair = indexed[*identity[:-1], "flair"]
        paired.append(
            dict(
                **{
                    k: t1[k]
                    for k in (
                        "arm",
                        "grid",
                        "stage",
                        "probe",
                        "condition",
                        "subject_id",
                        "t1_contrast_group",
                        "n_moved",
                    )
                },
                t1_contrast=t1["matched_contrast"],
                flair_contrast=flair["matched_contrast"],
                t1_endpoint_error_vox=t1["endpoint_error_vox"],
                flair_endpoint_error_vox=flair["endpoint_error_vox"],
                t1_minus_flair_endpoint_error_vox=t1["endpoint_error_vox"] - flair["endpoint_error_vox"],
                t1_relative_movement_error=t1["relative_movement_error"],
                flair_relative_movement_error=flair["relative_movement_error"],
            )
        )
    return (
        dict(subject_errors=subject_rows, contrast_groups=group_rows, associations=correlations, paired_views=paired),
        subjects,
        groups,
        cuts,
    )


def export_nifti(ds, pairs, contrasts, subjects, groups, per_group, output):
    """Export extremes by subject mean T1 contrast, independent of prediction errors."""
    if not per_group:
        return []
    import nibabel as nib

    directory = output / "nifti"
    directory.mkdir()
    entries, pending = [], []
    lookup = {(r["subject_id"], r["intervention_axis"], r["endpoint"], r["view"]): r for r in contrasts}
    for group in ("low", "high"):
        chosen = sorted(
            [idx for idx, label in groups["t1"].items() if label == group],
            key=lambda idx: (subjects[idx, "t1"]["matched_contrast"], idx),
            reverse=group == "high",
        )[:per_group]
        for idx in chosen:
            idx = int(idx)  # torch.Generator.manual_seed requires a Python integer
            pair = next(p for p in pairs if int(p["subject_id"]) == idx)  # first saved axis, both endpoints
            axis = pair["intervention_axis"]
            sample, reference = render_pair(ds, idx, axis, float(pair["eps"])), render_reference(ds, idx)
            prefix = f"{group}_t1_subject{idx}_{axis}"
            maps = {"white_matter_mask": (reference["tissue"] == 2).astype(np.uint8)}
            for end, lesion in zip("ab", sample["lesions"]):
                maps[f"{end}_lesion_mask"] = lesion.astype(np.uint8)
                for v, view in enumerate(VIEWS):
                    array = sample[end][v].numpy()[0]
                    # Export the exact verified endpoint, not a visually similar redraw.
                    metric = contrast_metrics(array, reference["images"][v].numpy()[0], lesion, reference["tissue"])
                    if hashlib.sha256(array.tobytes()).hexdigest() != lookup[idx, axis, end, view]["image_sha256"]:
                        raise ValueError("NIfTI replay differs from measured images")
                    maps[f"{end}_{view}"] = array
                    entries.append(
                        dict(
                            group=group,
                            subject_id=int(idx),
                            intervention_axis=axis,
                            endpoint=end,
                            view=view,
                            subject_t1_contrast=subjects[idx, "t1"]["matched_contrast"],
                            **metric,
                            image=f"{prefix}_{end}_{view}.nii.gz",
                            lesion_mask=f"{prefix}_{end}_lesion_mask.nii.gz",
                            reference=f"{prefix}_no_lesion_{view}.nii.gz",
                            centroid_i=float(sample["centroids"]["ab".index(end)][0]),
                            centroid_j=float(sample["centroids"]["ab".index(end)][1]),
                            centroid_k=float(sample["centroids"]["ab".index(end)][2]),
                        )
                    )
            for v, view in enumerate(VIEWS):
                maps[f"no_lesion_{view}"] = reference["images"][v].numpy()[0]
            pending.extend((prefix, name, array) for name, array in maps.items())
    # Shared display windows within each modality; no independent per-image rescaling.
    windows = {}
    for view in VIEWS:
        values = [a[a != 0] for _, name, a in pending if name.endswith("_" + view)]
        if values:
            windows[view] = np.quantile(np.concatenate(values), [0.01, 0.99]).tolist()
    for prefix, name, array in pending:
        image = nib.Nifti1Image(np.asarray(array, dtype=np.uint8 if name.endswith("mask") else np.float32), np.eye(4))
        image.set_qform(np.eye(4), code=0)
        image.set_sform(np.eye(4), code=2)
        image.header.set_xyzt_units("unknown")
        image.header["descrip"] = b"Synthetic voxel axes; model-normalized intensity; no physical orientation"
        view = next((v for v in VIEWS if name.endswith("_" + v)), None)
        lo, hi = windows[view] if view else (0, 1)
        image.header["cal_min"], image.header["cal_max"] = lo, hi
        nib.save(image, directory / f"{prefix}_{name}.nii.gz")
    save_csv(directory / "examples.csv", entries)
    (directory / "display_windows.json").write_text(json.dumps(windows, indent=2) + "\n")
    (directory / "README.txt").write_text(
        "Examples selected by subject mean T1 matched contrast, never by probe error.\n"
        "Low/high are this cohort's contrast groups; export takes the most extreme subjects in each.\n"
        "Each subject uses its first saved intervention axis and both -eps (a) / +eps (b) endpoints.\n"
        "T1/FLAIR share the exact lesion positions. no_lesion_* keeps anatomy, style, noise draws and normalization fixed.\n"
        "Inputs preserve the encoder's normalized intensities without clipping or rescaling.\n"
        "Use the SAME display window for low/high within a modality (display_windows.json; also NIfTI cal_min/max).\n"
        "Per-image auto-contrast in a viewer can hide the contrast difference. Overlay *_lesion_mask to find the lesion.\n"
        "examples.csv lists per-image contrasts and zero-based i,j,k centroids for navigation.\n"
        "Synthetic index-space: identity affine, unit voxel spacing, unknown units; no millimetres or anatomical orientation implied.\n"
        "These subjects differ in anatomy and acquisition; this is an illustrative comparison, not a causal contrast manipulation.\n"
    )
    return entries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-dir", type=Path, required=True, help="Completed encoder_lesion_intervention output")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; source audit is read-only")
    parser.add_argument(
        "--bootstrap", type=int, default=200, help="Whole-subject bootstrap draws; 0 disables intervals"
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--nifti-per-group", type=int, default=3, help="Low/high T1 examples each; 0 skips NIfTI exports"
    )
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.nifti_per_group < 0:
        parser.error("bootstrap and nifti-per-group must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Hash before reading, then verify again before rendering and at completion.
    paths = [args.audit_dir / name for name in ("report.json", "pairs.csv")]
    paths.extend(sorted(args.audit_dir.glob("*_predictions.npz")))
    hashes = {p.name: digest(p) for p in paths}
    source, pairs, truth, predictions, used = load_audit(args.audit_dir)
    if any(hashes.get(p.name) != digest(p) for p in used):
        raise ValueError("Source audit changed while loading")
    if args.nifti_per_group:
        import nibabel  # noqa: F401 -- fail before work if export dependency is missing
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = source["settings"]
    report = provenance(cfg, args, "cpu")
    report.update(
        source_audit_sha256=hashes,
        source_checkpoint_sha256=source.get("checkpoint_sha256"),
        evaluation_only=True,
        probes_refit=False,
        target_names=["physical_lesion_centroid"],
        notes=[
            "Primary contrast = absolute mean lesion effect within its mask, against a lesion-free reference with identical acquisition draws and normalization.",
            "Contrast is measured in normalized input units after bias field, magnitude noise and blur. It is not a calibrated contrast-to-noise ratio.",
            "Secondary local contrast compares lesion mean to a healthy WM shell, excluding one blur voxel; fewer than 8 shell voxels gives undefined local contrast.",
            "Each subject contributes one mean contrast/error per view. Axes/endpoints are not independent subjects. No-motion pairs remain in endpoint errors but not movement metrics.",
            "Within-view contrast thirds and T1-defined matched-subject thirds use images only. Ties are not split artificially.",
            "Negative contrast/error Spearman correlation supports lower error at higher contrast. Partial ranks adjust for mean nonzero move distance and preblur noise sigma in normalized units.",
            "Subjects with no nonzero moves are excluded from associations requiring movement-distance adjustment. Constant/rank-deficient or fewer than 8 subjects yields undefined associations.",
            "Bootstrap intervals condition on the saved checkpoint/probes and are exploratory, without multiple-comparison correction.",
            "This tests physical centroid errors on controlled lesion-move endpoints, not raw lesion-control R² or observational-cohort R².",
            "Contrast/error association cannot establish that contrast alone explains T1/FLAIR differences; anatomy and acquisition still covary. No identifiability guarantee.",
        ],
    )
    save_report(args.out_dir, report)
    old_threads = torch.get_num_threads()
    old_determinism = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        if cfg.get("cpu_threads") is not None:
            torch.set_num_threads(int(cfg["cpu_threads"]))
        if cfg.get("deterministic", False):
            torch.use_deterministic_algorithms(True, warn_only=cfg.get("deterministic_warn_only", False))
        original = source["arguments"]
        ds = dataset(cfg, max(64, original["subject_offset"] + original["num_samples"]), "test")
        contrasts, image_hash = replay_contrast(ds, source, pairs, truth)
        print("  Image replay verified; analyzing saved predictions (no probe refitting).", flush=True)
        tables, subjects, groups, cuts = analyze(
            pairs, truth, predictions, contrasts, ds.res, args.bootstrap, args.seed
        )
        examples = export_nifti(ds, pairs, contrasts, subjects, groups, args.nifti_per_group, args.out_dir)
        save_csv(args.out_dir / "image_contrasts.csv", contrasts)
        for name, rows in tables.items():
            save_csv(args.out_dir / f"{name}.csv", rows)
        if hashes != {p.name: digest(p) for p in paths}:
            raise ValueError("Source audit changed during analysis")
        report.update(
            status="complete",
            source_audit_unchanged=True,
            input_sha256=image_hash,
            image_replay_verified=True,
            contrast_tertile_cutoffs=cuts,
            n_subjects=len(groups["t1"]),
            n_pairs=len(pairs),
            n_nifti_examples=len(examples),
            associations=tables["associations"],
            contrast_groups=tables["contrast_groups"],
        )
        save_report(args.out_dir, report)
        print("\nTrained ridge: higher contrast -> lower error gives negative rho (subject-level)")
        print("view  grid stage        error                         rho   adjusted   n")
        for row in tables["associations"]:
            if row["arm"] == "trained" and row["probe"] == "ridge" and row["condition"] == "observed":
                print(
                    f"{row['view']:5s} {row['grid']:4d} {row['stage']:12s} {row['error_metric']:27s} "
                    f"{row['spearman']:+6.3f} {row['partial_spearman']:+10.3f} {row['n_subjects']:3d}"
                )
        print(f"Saved contrast test and {len(examples)} T1/FLAIR endpoint examples: {args.out_dir}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(args.out_dir, report)
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism[0], warn_only=old_determinism[1])


if __name__ == "__main__":
    main()
