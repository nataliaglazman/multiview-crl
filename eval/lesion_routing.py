"""Locate lesion information in a frozen VQVAE's decoder pathways.

    python -m eval.lesion_routing --run-dir /path/to/vqvae/run --num-samples 64

Move one lesion coordinate by -/+eps while fixing all other anatomy, acquisition
parameters, noise seeds and the original normalization affine. Decode cA/sA,
cB/sA, cA/sB, cB/sB using actual decoder-bound tensors. Score the signed image
response where the rendered lesion changes, not a fitted coordinate probe.

This tests decoder reliance, not whether a block contains independently decodable
coordinates. Hybrids and coordinate interventions can be outside the training
distribution, especially for causal runs. Read joint fidelity before routing.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from eval.lesion_reconstruction import json_safe
from eval.lesion_reconstruction import make_dataset as make_lesion_dataset
from eval.ventricle_routing import (
    REPLAY_RMS_ATOL,
    REPLAY_RMS_RTOL,
    REPLAY_SIGNAL_FRACTION,
    decode_swaps,
    normalization_affine,
    score_swaps,
    state_digest,
    verify_checkpoint,
)

logger = logging.getLogger(__name__)
VIEWS = ("t1", "flair")


def make_dataset(args, n, causal="match", split="test"):
    from data.datasets import SyntheticBrainDataset

    # Older checkouts silently filter unknown renderer flags. Do not audit a
    # different lesion-placement rule from the one explicitly saved by training.
    placement = getattr(args, "synthetic_lesion_placement", "legacy")
    if (
        placement != "legacy"
        and "synthetic_lesion_placement" not in inspect.signature(SyntheticBrainDataset.__init__).parameters
    ):
        raise ValueError(f"This checkout cannot reproduce synthetic_lesion_placement={placement!r}")
    return make_lesion_dataset(args, n, causal, split)


def render_pair(ds, idx, axis="x", eps=0.5):
    """Only one z_content[2:5] entry changes; retain actual mask centroids too."""
    import torch

    if axis not in ("x", "y", "z") or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Need axis x/y/z and finite positive eps")
    inner = ds._inner
    raw_a, raw_b, lat = inner[idx]
    mask = lat["brain_mask"]
    normalized = ds.normalize_views(raw_a, raw_b, mask, mask.clone())
    affines = [normalization_affine(raw, norm, mask) for raw, norm in zip((raw_a, raw_b), normalized)]
    coordinate = 2 + ("x", "y", "z").index(axis)
    states, tissues, lesions, controls = [], [], [], []
    for sign in (-1, 1):
        content = lat["z_content"].clone()
        content[coordinate] += sign * eps
        render_args = (content, lat["z_deformation"], lat["z_fissure"])
        raw0, raw1, new_mask = inner.render_pseudo_mri(
            *render_args,
            lat["z_style_v1"],
            lat["z_style_v2"],
            inner.sample_seed_for(idx),
            z_lesion=lat.get("z_lesion"),
        )
        if not torch.equal(mask, new_mask):
            raise ValueError("Lesion-coordinate intervention changed foreground support")
        tissue, lesion = inner.renderer.render_structure(
            *render_args,
            device="cpu",
            clean=inner.clean_content,
            z_lesion=lat.get("z_lesion"),
        )
        tissues.append(tissue)
        lesions.append(lesion.numpy() > 0)
        controls.append(content.numpy()[2:5].copy())
        states.append([(raw * gain + bias) * mask for raw, (gain, bias) in zip((raw0, raw1), affines)])
    if not torch.equal(*tissues):
        raise ValueError("Lesion-coordinate intervention also changed the anatomical tissue map")
    support = lesions[0] != lesions[1]
    roi = maximum_filter(support, size=3)
    for view in range(2):
        outside = (states[1][view] - states[0][view]).numpy()[0][~roi]
        if outside.size and np.max(np.abs(outside)) > 2e-6:
            raise ValueError("Input change extends beyond the lesion/blur support; intervention is confounded")
    centroids = [np.argwhere(a).mean(0) if a.any() else np.full(3, np.nan) for a in lesions]
    return {
        "index": idx,
        "axis": axis,
        "eps": eps,
        "a": states[0],
        "b": states[1],
        "mask": mask,
        "support": support,
        "lesions": lesions,
        "centroids": centroids,
        "controls": controls,
    }


def score_pair(sample, view, decoded, diagnostics, batch_index):
    xa, xb = [sample[k][view].numpy()[0] for k in ("a", "b")]
    foreground = sample["mask"].numpy()[0] > 0
    row = {
        "index": sample["index"],
        "axis": sample["axis"],
        "eps": sample["eps"],
        "view": VIEWS[view],
    }
    for state, lesion, centroid, control in zip("ab", sample["lesions"], sample["centroids"], sample["controls"]):
        row[f"lesion_{state}_voxels"] = int(lesion.sum())
        for j, axis in enumerate("xyz"):
            row[f"centroid_{state}_{axis}"] = float(centroid[j])
            row[f"latent_{state}_{axis}"] = float(control[j])
    row["centroid_displacement_vox"] = float(np.linalg.norm(sample["centroids"][1] - sample["centroids"][0]))
    row.update(score_swaps(xa, xb, decoded, sample["support"], foreground))
    row["changed_lesion_voxels"] = row.pop("changed_tissue_voxels")
    row.update({key: float(values[batch_index]) for key, values in diagnostics.items()})
    row["both_lesions_nonempty"] = bool(all(a.any() for a in sample["lesions"]))
    row["valid_routing"] = bool(row["valid_input"] and row["endpoint_signal_resolved"] and row["both_lesions_nonempty"])
    roi = maximum_filter(sample["support"], size=3) & foreground
    outside = foreground & ~roi
    delta = decoded["bb"] - decoded["aa"]
    energy = float(np.square(delta[foreground]).sum())
    row["joint_energy_in_affected_fraction"] = float(np.square(delta[roi]).sum() / energy) if energy > 1e-20 else np.nan
    row["joint_outside_rms"] = float(np.sqrt(np.square(delta[outside]).mean())) if outside.any() else np.nan
    for key in list(row):
        if key.endswith("_gap_delta_rms"):
            native = row[key.replace("_gap_delta_rms", "_delta_rms")]
            row[key.replace("_gap_delta_rms", "_gap_to_native_rms")] = row[key] / native if native > 1e-20 else np.nan
    return row


def summarize(rows):
    """Paired means; each axis/view contains independent subjects, not pooled repeats."""
    summary = {}
    routing = (
        "joint_gain",
        "joint_cosine",
        "joint_relative_error",
        "content_mean_gain",
        "style_mean_gain",
        "content_at_style_a_gain",
        "content_at_style_b_gain",
        "style_at_content_a_gain",
        "style_at_content_b_gain",
        "interaction_rms_ratio",
        "joint_energy_in_affected_fraction",
        "joint_outside_rms",
    )
    for axis, view in sorted({(r["axis"], r["view"]) for r in rows}):
        selected = [r for r in rows if (r["axis"], r["view"]) == (axis, view)]
        valid = [r for r in selected if r["valid_routing"]]
        keys = (
            *routing,
            "aa_roi_mae",
            "bb_roi_mae",
            "endpoint_replay_rms",
            "endpoint_error_to_input_ratio",
        )
        keys += tuple(
            k for k in selected[0] if k.endswith(("_delta_rms", "_gap_to_native_rms", "_code_change_fraction"))
        )
        metrics = {}
        for key in keys:
            source = valid if key in routing or key.endswith(("_delta_rms", "_gap_to_native_rms")) else selected
            values = np.asarray([r[key] for r in source], float)
            values = values[np.isfinite(values)]
            ci = None
            if len(values) and key in routing:
                rng = np.random.default_rng(1729)
                ci = np.quantile(
                    values[rng.integers(len(values), size=(1000, len(values)))].mean(1),
                    [0.025, 0.975],
                ).tolist()
            metrics[key] = {
                "n": len(values),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "mean_ci95": ci,
            }
        summary[f"{axis}/{view}"] = {
            "n": len(selected),
            "nonempty_pairs": sum(r["both_lesions_nonempty"] for r in selected),
            "measurable": sum(r["valid_input"] for r in selected),
            "resolved": len(valid),
            "metrics": metrics,
        }
    return summary


def save_example(sample, view, decoded, directory, nifti=False):
    """Show BOTH modalities and both lesion planes; export synthetic voxel coordinates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xa, xb = [sample[k][view].numpy()[0] for k in ("a", "b")]
    images = [
        xa,
        xb,
        decoded["aa"],
        decoded["ba"],
        decoded["ab"],
        decoded["bb"],
        xb - xa,
        decoded["bb"] - decoded["aa"],
    ]
    names = (
        "input_a",
        "input_b",
        "recon_aa",
        "content_b_style_a",
        "content_a_style_b",
        "recon_bb",
        "input_delta",
        "joint_delta",
    )
    planes = sorted({int(round(c[2])) for c in sample["centroids"] if np.isfinite(c).all()})
    if not planes:
        planes = [xa.shape[2] // 2]
    fg = sample["mask"].numpy()[0] > 0
    lo, hi = np.quantile(np.concatenate([xa[fg], xb[fg]]), [0.01, 0.99])
    limit = max(float(np.abs(images[-2]).max()), 1e-8)
    fig, axes = plt.subplots(len(planes), 8, figsize=(20, 2.8 * len(planes)), squeeze=False)
    for row, plane in enumerate(planes):
        for col, (array, name) in enumerate(zip(images, names)):
            ax = axes[row, col]
            ax.imshow(
                array[:, :, plane].T,
                origin="lower",
                cmap="gray" if col < 6 else "coolwarm",
                vmin=lo if col < 6 else -limit,
                vmax=hi if col < 6 else limit,
            )
            ax.set_title(name.replace("_", " "), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"voxel z={plane}")
    prefix = f"sample{sample['index']:04d}_{sample['axis']}_{VIEWS[view]}"
    fig.suptitle(f"{prefix}: A/B differ only in one lesion coordinate; shared image/difference scales")
    fig.tight_layout()
    fig.savefig(directory / f"{prefix}.png", dpi=110)
    plt.close(fig)
    if nifti:
        import nibabel as nib

        for name, array in zip(names, images):
            nib.save(
                nib.Nifti1Image(np.asarray(array, np.float32), np.eye(4)),
                directory / f"{prefix}_{name}.nii.gz",
            )
        for state, array in zip("ab", sample["lesions"]):
            nib.save(
                nib.Nifti1Image(array.astype(np.uint8), np.eye(4)),
                directory / f"{prefix}_mask_{state}.nii.gz",
            )


def audit(
    model,
    ds,
    device,
    axes=("x", "y", "z"),
    eps=0.5,
    batch_size=2,
    examples=0,
    directory=None,
    nifti=False,
):
    if model.training:
        raise ValueError("Call model.eval() before the diagnostic")
    if not axes or any(a not in "xyz" or len(a) != 1 for a in axes) or len(set(axes)) != len(axes):
        raise ValueError("Choose distinct axes from x, y, z")
    if batch_size < 1 or len(ds) < 1 or examples < 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Need positive sample/batch counts and eps, and nonnegative examples")
    if examples and directory is None:
        raise ValueError("An output directory is needed to save examples")
    before = state_digest(model)
    rows = []
    try:
        for axis in axes:
            for start in range(0, len(ds), batch_size):
                samples = [render_pair(ds, i, axis, eps) for i in range(start, min(start + batch_size, len(ds)))]
                decoded, diagnostics = decode_swaps(model, samples, device, measure_gap=True)
                for b, sample in enumerate(samples):
                    for view in range(2):
                        images = {key: values[b] for key, values in decoded[view].items()}
                        rows.append(score_pair(sample, view, images, diagnostics[view], b))
                        if sample["index"] < examples:
                            save_example(sample, view, images, directory, nifti)
                logger.info(
                    "Lesion %s: %d/%d subjects",
                    axis,
                    min(start + batch_size, len(ds)),
                    len(ds),
                )
    finally:
        if before != state_digest(model):
            raise RuntimeError("Diagnostic changed a registered parameter or buffer")
    return rows, summarize(rows)


def print_summary(summary):
    def fmt(value):
        return "n/a" if value is None else f"{value:.4g}"

    print("\nPaired mean lesion-movement responses (identity gain=1; no response=0)")
    print(
        "axis/view   nonempty measurable resolved/total  joint gain/cos/error       content/style gain    interaction"
    )
    for key, group in summary.items():
        m = group["metrics"]
        val = lambda name: fmt(m[name]["mean"])
        joint = "/".join(val(k) for k in ("joint_gain", "joint_cosine", "joint_relative_error"))
        gains = "/".join(val(k) for k in ("content_mean_gain", "style_mean_gain"))
        print(
            f"{key:<11} {group['nonempty_pairs']:>6} {group['measurable']:>10} {group['resolved']:>5}/{group['n']:<5}"
            f" {joint:<26} {gains:<21} {val('interaction_rms_ratio')}"
        )
    print("\nDecoder-bound latent response RMS: native / GAP (sensitivity, not decodability)")
    for key, group in summary.items():
        m = group["metrics"]
        for name in m:
            if name.endswith("_gap_delta_rms") and "_post_" in name:
                native = name.replace("_gap_delta_rms", "_delta_rms")
                print(
                    f"{key:<11} {name.removesuffix('_gap_delta_rms'):<20} {fmt(m[native]['mean'])} / {fmt(m[name]['mean'])}"
                )
    print(
        "\nRead joint fidelity and the T1/FLAIR images first. Large interaction makes a single-path label misleading."
    )
    print(
        "Native sensitivity with a small GAP response shows cancellation under pooling; it is not proof of coordinate recovery."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-samples", type=int, default=64, help="Subjects per tested axis")
    p.add_argument("--axes", choices=("x", "y", "z"), nargs="+", default=["x", "y", "z"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Subjects; forward uses four volumes per subject",
    )
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument(
        "--examples",
        type=int,
        default=2,
        help="First N subjects per axis, both modalities",
    )
    p.add_argument(
        "--save-nifti",
        action="store_true",
        help="Also save example volumes; identity affine, synthetic voxel coordinates",
    )
    p.add_argument(
        "--old-generator",
        action="store_true",
        help="Use the pre-7ac56a3 renderer, as in ventricle_routing",
    )
    p.add_argument("--out-dir", default=None)
    cli = p.parse_args()
    if (
        min(cli.num_samples, cli.batch_size, cli.cpu_threads) < 1
        or cli.examples < 0
        or not np.isfinite(cli.eps)
        or cli.eps <= 0
    ):
        p.error("Need positive counts/eps and nonnegative examples")
    if len(set(cli.axes)) != len(cli.axes):
        p.error("Axes must be distinct")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir

    torch.set_num_threads(cli.cpu_threads)
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device=cli.device, seed=0)
    checkpoint = Path(cli.run_dir) / "vqvae_model.pt" if cli.checkpoint is None else Path(cli.checkpoint)
    if cli.checkpoint is not None and checkpoint.name == cli.checkpoint:
        checkpoint = Path(cli.run_dir) / checkpoint
    provenance = verify_checkpoint(model, checkpoint)
    ds = make_dataset(args, cli.num_samples, cli.causal, cli.split)
    if cli.old_generator:
        from eval.legacy_renderer import use_legacy_renderer

        use_legacy_renderer(ds)
    directory = Path(
        cli.out_dir or Path(cli.run_dir) / ("lesion_routing_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "arguments": vars(cli),
        "run_settings": vars(args),
        "checkpoint": provenance,
        "protocol": __doc__,
        "render_resolution": ds.res,
        "normalization": "Original sample affine frozen for both endpoints; anatomy and view noise fixed",
        "interpretation": "All content levels exchanged together within each modality. Gains are signed projections onto "
        "the input movement in changed-lesion support dilated by one voxel. Averaged conditional content/style gains "
        "sum to joint gain, but do not uniquely allocate interactions. Missing lesions and unresolved replay are excluded "
        "from routing summaries; all rows remain in responses.csv. Latent RMS is scale-dependent sensitivity, not "
        "information or a fitted probe. Multi-level content codes can already contain coarse style conditioning. "
        "This isolates rendered lesion coordinates without propagating an SCM intervention to descendants.",
        "replay_limits": {
            "rms_atol": REPLAY_RMS_ATOL,
            "rms_rtol": REPLAY_RMS_RTOL,
            "signal_fraction": REPLAY_SIGNAL_FRACTION,
        },
        "nifti_coordinates": "Identity affine: synthetic voxel indices, no patient-space orientation or physical spacing",
    }
    report_path = directory / "summary.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    rows, summary = audit(
        model,
        ds,
        device,
        cli.axes,
        cli.eps,
        cli.batch_size,
        cli.examples,
        directory,
        cli.save_nifti,
    )
    with (directory / "responses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update(status="complete", summary=summary, registered_state_unchanged=True)
    report_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print_summary(summary)
    print(f"Saved {directory}\nNo registered model parameter or buffer changed; no checkpoint was written.")


if __name__ == "__main__":
    main()
