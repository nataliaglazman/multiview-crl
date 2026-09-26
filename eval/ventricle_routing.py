"""Test which decoder pathway carries a controlled ventricle-size change.

    python -m eval.ventricle_routing --run-dir /path/to/run --num-samples 64

A/B change only z_content[1] by -/+eps. Other factors, fields, acquisition
noise and the original sample's normalization affine stay fixed. Decode AA,
BA, AB, BB (content donor first), within each modality. Gains project decoder
changes onto the actual input change in the dilated ventricular boundary ROI.
Identity response has gain 1; no response has gain 0. Both donor contexts are
reported, since nonlinear interactions can prevent a unique allocation.

This is decoder reliance under intervention, not a fitted information probe or
proof of training causality. Read joint reconstruction fidelity first. Hybrids
and perturbed samples may be off-distribution. Multi-level content codes can
already contain conditioning from coarser style-dependent reconstructions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from eval.lesion_reconstruction import json_safe

logger = logging.getLogger(__name__)

REPLAY_RMS_ATOL = 1e-6
REPLAY_RMS_RTOL = 1e-4
REPLAY_SIGNAL_FRACTION = 0.01


def state_digest(model):
    """Include every registered parameter and buffer, including integer counters."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_checkpoint(model, path):
    """Fail if the shared permissive loader left evaluated state at initialization."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stored = checkpoint.get("encoders", checkpoint)
    stored = {k.removeprefix("module."): v for k, v in stored.items()}
    mismatched = [
        key
        for key, value in model.state_dict().items()
        if key not in stored or not torch.equal(value.detach().cpu(), stored[key].cpu())
    ]
    if mismatched:
        raise ValueError(f"Loaded model does not exactly match checkpoint state: {mismatched[:8]}")
    return {
        "path": str(path.resolve()),
        "step": checkpoint.get("step"),
        "model_state_sha256": state_digest(model),
    }


@contextlib.contextmanager
def stable_replay_math():
    """Use consistent CUDA precision/algorithms and restore the caller's flags."""
    import torch

    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled, benchmark=False, deterministic=True, allow_tf32=False
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def assess_endpoint_replay(replay, reference, input_delta, roi):
    """Check both endpoints separately, then calibrate error to the local signal.

    A pointwise allclose rejects isolated near-zero voxels even when the overall
    replay agrees. RMS checks catch substantive mismatches without that failure.
    The sum of endpoint error norms bounds their contribution to a difference's
    projected gain. It does not bound unmeasured numerical errors in hybrids.
    """
    replay, reference, input_delta = [np.asarray(a, dtype=np.float64) for a in (replay, reference, input_delta)]
    roi = np.asarray(roi, bool)
    if replay.shape != reference.shape or replay.shape != (2, *input_delta.shape) or roi.shape != input_delta.shape:
        raise ValueError("Invalid endpoint shapes")
    if not all(np.isfinite(a).all() for a in (replay, reference, input_delta)):
        raise ValueError("Non-finite endpoint values")
    error = replay - reference
    rms = np.sqrt(np.mean(error.reshape(2, -1) ** 2, axis=1))
    reference_rms = np.sqrt(np.mean(reference.reshape(2, -1) ** 2, axis=1))
    limits = REPLAY_RMS_ATOL + REPLAY_RMS_RTOL * reference_rms
    if np.any(rms > limits):
        raise ValueError(
            "Decoder endpoint replay has a substantial mismatch; swaps would be invalid. "
            f"max_abs={np.abs(error).max():.6g}, endpoint_rms={rms.tolist()}, "
            f"allowed_rms={limits.tolist()}. Try --device cpu to separate CUDA numerics from a replay defect."
        )
    signal_norm = np.linalg.norm(input_delta[roi])
    error_bound = sum(np.linalg.norm(e[roi]) for e in error)
    ratio = float(error_bound / signal_norm) if signal_norm ** 2 > 1e-12 else np.nan
    return {
        "endpoint_replay_max_abs": float(np.abs(error).max()),
        "endpoint_replay_rms": float(rms.max()),
        "endpoint_error_to_input_ratio": ratio,
        "endpoint_signal_resolved": bool(np.isfinite(ratio) and ratio <= REPLAY_SIGNAL_FRACTION),
    }


def make_dataset(args, n, causal="match", split="test"):
    from data.datasets import SyntheticBrainDataset

    accepted = inspect.signature(SyntheticBrainDataset.__init__).parameters
    placement = getattr(args, "synthetic_lesion_placement", "legacy")
    if placement != "legacy" and "synthetic_lesion_placement" not in accepted:
        raise ValueError(f"This checkout cannot reproduce synthetic_lesion_placement={placement!r}")
    kw = {k: v for k, v in vars(args).items() if k.startswith("synthetic_") and k in accepted}
    if kw.get("synthetic_mode", "pseudo_mri") != "pseudo_mri":
        raise ValueError("Ventricle routing requires synthetic_mode=pseudo_mri")
    if kw.get("synthetic_n_content", 9) < 2:
        raise ValueError("Ventricle routing requires z_content[1]")
    if causal == "iid":
        kw.update(synthetic_causal=False, synthetic_hierarchical_content=False)
    kw.update(synthetic_num_samples=n, synthetic_num_samples_per_mode=None)
    # SyntheticBrainDataset takes resolution via spatial_size, not synthetic_res.
    # Omitting it silently selects 32, even for checkpoints trained at 64.
    res = getattr(args, "synthetic_res", 64)
    spatial_size = getattr(args, "spatial_size", None) or (res, res, res)
    ds = SyntheticBrainDataset(mode=split, spatial_size=spatial_size, cache=False, **kw)
    logger.info("Ventricle routing renders at %d³ (requested spatial_size=%s)", ds.res, spatial_size)
    return ds


def normalization_affine(raw, normalized, mask):
    """Recover and verify the original foreground affine in float64."""
    import torch

    x, y = raw[mask > 0].double(), normalized[mask > 0].double()
    if x.numel() < 2 or not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise ValueError("Empty or non-finite normalization reference")
    centered = x - x.mean()
    energy = centered.square().sum()
    if float(energy) < 1e-12:
        raise ValueError("Degenerate normalization reference")
    gain = (centered * (y - y.mean())).sum() / energy
    bias = y.mean() - gain * x.mean()
    if not torch.allclose(x * gain + bias, y, atol=2e-6, rtol=2e-6):
        raise ValueError("Normalizer is not affine; cannot freeze it safely")
    return gain.float(), bias.float()


def render_pair(ds, idx, eps=0.25):
    """Return normalized low/high views, fixed mask and actual boundary support."""
    import torch

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    inner = ds._inner
    raw0, raw1, lat = inner[idx]
    mask = lat["brain_mask"]
    normalized = ds.normalize_views(raw0, raw1, mask, mask.clone())
    affines = [normalization_affine(x, y, mask) for x, y in zip((raw0, raw1), normalized)]
    states, tissues, lesions = [], [], []
    for sign in (-1, 1):
        content = lat["z_content"].clone()
        content[1] += sign * eps
        args = (content, lat["z_deformation"], lat["z_fissure"])
        a, b, new_mask = inner.render_pseudo_mri(
            *args,
            lat["z_style_v1"],
            lat["z_style_v2"],
            inner.sample_seed_for(idx),
            z_lesion=lat.get("z_lesion"),
        )
        if not torch.equal(new_mask, mask):
            raise ValueError("Ventricle intervention changed foreground support; fixed-mask test is invalid")
        tissue, lesion = inner.renderer.render_structure(
            *args, device="cpu", clean=inner.clean_content, z_lesion=lat.get("z_lesion")
        )
        tissues.append(tissue.numpy())
        lesions.append(lesion.numpy())
        states.append([(x * gain + bias) * mask for x, (gain, bias) in zip((a, b), affines)])
    support = tissues[0] != tissues[1]
    roi = maximum_filter(support, size=3)
    outside_max = max(
        float(np.abs((states[1][v] - states[0][v]).numpy()[0][~roi]).max()) if (~roi).any() else 0.0 for v in range(2)
    )
    # wm_interior places lesions using the final tissue map. Fixed lesion latents
    # therefore do not guarantee a fixed rendered lesion when ventricles grow.
    changed_lesion = int(np.count_nonzero(lesions[0] != lesions[1]))
    return {
        "a": states[0],
        "b": states[1],
        "mask": mask,
        "support": support,
        "tissues": tissues,
        "lesion_changed_voxels": changed_lesion,
        "input_outside_roi_max_abs": outside_max,
        "isolated_intervention": changed_lesion == 0 and outside_max <= 2e-6,
        "index": idx,
        "eps": eps,
        "z_low": float(lat["z_content"][1]) - eps,
        "z_high": float(lat["z_content"][1]) + eps,
    }


def score_swaps(xa, xb, decoded, support, foreground):
    """Factorial effects along the input intervention, without fitting a probe."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    y = {k: np.asarray(decoded[k], float) for k in ("aa", "ba", "ab", "bb")}
    support, foreground = np.asarray(support, bool), np.asarray(foreground, bool)
    if xa.ndim != 3 or any(a.shape != xa.shape for a in (xb, support, foreground, *y.values())):
        raise ValueError("Expected matching 3-D images and masks")
    if any(not np.isfinite(a).all() for a in (xa, xb, *y.values())) or not foreground.any():
        raise ValueError("Non-finite images or empty foreground")
    # The renderer uses a 3x3x3 blur: one voxel of dilation covers its support.
    roi = maximum_filter(support, size=3) & foreground
    target = (xb - xa)[roi]
    energy = float(target @ target)
    result = {
        "changed_tissue_voxels": int(support.sum()),
        "roi_voxels": int(roi.sum()),
        "input_response_rms": float(np.sqrt(energy / max(1, roi.sum()))),
        "valid_input": bool(energy > 1e-12),
        "aa_foreground_mae": float(np.abs(y["aa"] - xa)[foreground].mean()),
        "bb_foreground_mae": float(np.abs(y["bb"] - xb)[foreground].mean()),
    }
    effects = {
        "content_at_style_a": y["ba"] - y["aa"],
        "content_at_style_b": y["bb"] - y["ab"],
        "style_at_content_a": y["ab"] - y["aa"],
        "style_at_content_b": y["bb"] - y["ba"],
        "joint": y["bb"] - y["aa"],
        "interaction": y["bb"] - y["ba"] - y["ab"] + y["aa"],
    }
    effects["content_mean"] = (effects["content_at_style_a"] + effects["content_at_style_b"]) / 2
    effects["style_mean"] = (effects["style_at_content_a"] + effects["style_at_content_b"]) / 2
    for name, effect in effects.items():
        local = effect[roi]
        result[name + "_gain"] = float(local @ target / energy) if energy > 1e-12 else np.nan
        result[name + "_rms_ratio"] = float(np.linalg.norm(local) / np.sqrt(energy)) if energy > 1e-12 else np.nan
    joint = effects["joint"][roi]
    denom = np.sqrt(energy) * np.linalg.norm(joint)
    result["joint_cosine"] = float(joint @ target / denom) if denom > 1e-12 else np.nan
    result["joint_relative_error"] = (
        float(np.linalg.norm(joint - target) / np.sqrt(energy)) if energy > 1e-12 else np.nan
    )
    result["aa_roi_mae"] = float(np.abs(y["aa"] - xa)[roi].mean()) if roi.any() else np.nan
    result["bb_roi_mae"] = float(np.abs(y["bb"] - xb)[roi].mean()) if roi.any() else np.nan
    return result


def decode_swaps(model, samples, device, *, measure_gap=False):
    """One view-major forward pins each view's mask across both donor states."""
    import torch

    if model.training:
        raise ValueError("Call model.eval() before the diagnostic; codebooks must not update")
    if not model.inject_style_to_decoder:
        raise ValueError("This test requires an injected style pathway")
    n = len(samples)
    if not n:
        raise ValueError("Need at least one sample")
    # [v0: A subjects, B subjects; v1: A subjects, B subjects]
    x = torch.cat([torch.stack([s[state][v] for s in samples]) for v in range(2) for state in ("a", "b")]).to(device)
    masks = torch.cat([torch.stack([s["mask"] for s in samples])] * 4).to(device)
    with torch.inference_mode(), stable_replay_math():
        # Capture actual q outputs, not embed_code(ids): forward's straight-through
        # x + (q - x).detach() can incur cancellation when encoder magnitudes grow.
        # Do not alter the quantizer or training arithmetic to fix an eval replay.
        captured, handles = {}, []

        def capture(key):
            def hook(module, inputs, output):
                if key in captured:
                    raise ValueError(f"Content codebook {key} ran more than once; replay mapping is ambiguous")
                captured[key] = output[0].detach()

            return hook

        split = model.separate_content_codebooks and model.codebooks_v1 is not None
        try:
            for level, cb in enumerate(model.codebooks):
                handles.append(cb.register_forward_hook(capture((level, 0))))
            if split:
                for level, cb in enumerate(model.codebooks_v1):
                    handles.append(cb.register_forward_hook(capture((level, 1))))
            out, pre_style = model(
                x,
                return_recon=True,
                pool_only=False,
                n_views=2,
                subsets=[(0, 1)],
                mask=masks,
                return_style_features=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        quantized = {
            level: torch.cat([captured[level, 0], captured[level, 1]]) if split else captured[level, 0]
            for level in range(model.nb_levels)
        }
        post_style = model._last_style_spatials
        if not post_style:
            raise ValueError("No decoder-bound style features; the run may have an all-content split")
        # Forward accumulates IDs coarsest-first; decode_codes indexes finest-first.
        codes = list(reversed(out[5]))
        if len(codes) != model.nb_levels or any(c is None for c in codes):
            raise ValueError("Missing content codes")

        # Replay all 4N rows, preserving the forward's batch geometry. GPU convolution
        # kernels can change with batch size, producing another numerical discrepancy.
        replay = model.decode_codes(quantized_codes=quantized, styles=dict(post_style), target_spatial_size=x.shape[2:])
        if replay.shape != x.shape or not bool(torch.isfinite(replay).all()):
            raise ValueError("Invalid endpoint reconstruction")
        reference = out[0]
        replay_np, reference_np = replay[:, 0].cpu().numpy(), reference[:, 0].cpu().numpy()
        endpoint_checks = []
        for view in range(2):
            checks = []
            for b, sample in enumerate(samples):
                a_idx, b_idx = view * 2 * n + b, view * 2 * n + n + b
                roi = maximum_filter(sample["support"], size=3) & (sample["mask"].numpy()[0] > 0)
                delta = (sample["b"][view] - sample["a"][view]).numpy()[0]
                checks.append(
                    assess_endpoint_replay(replay_np[[a_idx, b_idx]], reference_np[[a_idx, b_idx]], delta, roi)
                )
            endpoint_checks.append({k: np.asarray([r[k] for r in checks]) for k in checks[0]})

        # Exchange A/B content donors within each modality; style stays at its row.
        # Original A rows now decode BA, and original B rows decode AB. Preserve
        # tensor memory format as well as batch shape for comparable decoder calls.
        order = torch.arange(len(x), device=x.device).reshape(2, 2, n).flip(1).reshape(-1)
        swapped = {}
        for level, q in quantized.items():
            swapped[level] = torch.empty_like(q)
            swapped[level].copy_(q[order])
        hybrid = model.decode_codes(quantized_codes=swapped, styles=dict(post_style), target_spatial_size=x.shape[2:])
        if hybrid.shape != x.shape or not bool(torch.isfinite(hybrid).all()):
            raise ValueError("Invalid hybrid reconstruction")
        recon, diagnostics = [], []
        for view in range(2):
            start = view * 2 * n
            slices = {"a": slice(start, start + n), "b": slice(start + n, start + 2 * n)}
            recon.append(
                {
                    "aa": replay[slices["a"], 0].cpu().numpy(),
                    "bb": replay[slices["b"], 0].cpu().numpy(),
                    "ba": hybrid[slices["a"], 0].cpu().numpy(),
                    "ab": hybrid[slices["b"], 0].cpu().numpy(),
                }
            )
            view_diag = endpoint_checks[view]

            def response(name, features):
                diff = features[slices["b"]] - features[slices["a"]]
                view_diag[name + "_delta_rms"] = diff.flatten(1).square().mean(1).sqrt().cpu().numpy()
                if measure_gap:
                    # Compare the actual spatial response with the response left
                    # after global pooling. This is sensitivity, not a probe score.
                    pooled = diff.mean(tuple(range(2, diff.ndim)))
                    view_diag[name + "_gap_delta_rms"] = pooled.square().mean(1).sqrt().cpu().numpy()

            for level, feature in enumerate(out[2]):
                mask = out[6].get(level)
                if isinstance(mask, tuple):
                    mask = mask[view]
                idx = (
                    torch.ones(feature.shape[1], device=feature.device, dtype=torch.bool)
                    if mask is None
                    else mask.flatten().bool()
                )
                if idx.any():
                    response(f"content_pre_L{level}", feature[:, idx])
                response(f"content_post_L{level}", quantized[level])
                view_diag[f"content_L{level}_code_change_fraction"] = (
                    (codes[level][slices["a"]] != codes[level][slices["b"]]).flatten(1).float().mean(1).cpu().numpy()
                )
            for level in post_style:
                response(f"style_pre_L{level}", pre_style[level])
                response(f"style_post_L{level}", post_style[level])
            diagnostics.append(view_diag)
    return recon, diagnostics


def audit(model, ds, device, eps=0.25, batch_size=2, examples=4, *, directory=None, nifti=False):
    """Match lesion routing, retaining the legacy return value and median fields."""
    if model.training:
        raise ValueError("Call model.eval() before the diagnostic")
    if batch_size < 1 or len(ds) < 1 or examples < 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Need positive batch size/sample count/eps and nonnegative examples")
    if nifti and directory is None:
        raise ValueError("An output directory is needed to save NIfTI examples")
    before = state_digest(model)
    try:
        return _audit(model, ds, device, eps, batch_size, examples, directory, nifti)
    finally:
        if before != state_digest(model):
            raise RuntimeError("Diagnostic changed a registered parameter or buffer")


def _audit(model, ds, device, eps, batch_size, examples, directory, nifti):
    rows, panels = [], []
    for start in range(0, len(ds), batch_size):
        samples = [render_pair(ds, i, eps) for i in range(start, min(start + batch_size, len(ds)))]
        decoded, responses = decode_swaps(model, samples, device, measure_gap=True)
        for b, sample in enumerate(samples):
            for view, modality in enumerate(("t1", "flair")):
                xa, xb = [sample[k][view].numpy()[0] for k in ("a", "b")]
                ys = {k: v[b] for k, v in decoded[view].items()}
                row = {k: sample[k] for k in ("index", "z_low", "z_high")}
                row.update(
                    {
                        k: sample[k]
                        for k in ("lesion_changed_voxels", "input_outside_roi_max_abs", "isolated_intervention")
                    }
                )
                row["eps"] = eps
                row["modality"] = modality
                row.update(score_swaps(xa, xb, ys, sample["support"], sample["mask"].numpy()[0]))
                row.update({k: float(v[b]) for k, v in responses[view].items()})
                row["valid_routing"] = (
                    row["valid_input"] and bool(row["endpoint_signal_resolved"]) and row["isolated_intervention"]
                )
                foreground = sample["mask"].numpy()[0] > 0
                roi = maximum_filter(sample["support"], size=3) & foreground
                outside = foreground & ~roi
                delta = ys["bb"] - ys["aa"]
                energy = float(np.square(delta[foreground]).sum())
                row["joint_energy_in_affected_fraction"] = (
                    float(np.square(delta[roi]).sum() / energy) if energy > 1e-20 else np.nan
                )
                row["joint_outside_rms"] = float(np.sqrt(np.square(delta[outside]).mean())) if outside.any() else np.nan
                for key in list(row):
                    if key.endswith("_gap_delta_rms"):
                        native = row[key.replace("_gap_delta_rms", "_delta_rms")]
                        row[key.replace("_gap_delta_rms", "_gap_to_native_rms")] = (
                            row[key] / native if native > 1e-20 else np.nan
                        )
                rows.append(row)
                if directory is not None and sample["index"] < examples:
                    save_example(sample, modality, xa, xb, ys, Path(directory), nifti)
                if len(panels) < examples and view == 1 and sample["support"].any():
                    z = int(np.argmax(sample["support"].sum((0, 1))))
                    panels.append(
                        (sample["index"], z, [im[:, :, z] for im in (xa, xb, ys["aa"], ys["ba"], ys["ab"], ys["bb"])])
                    )
        logger.info("Scored %d/%d subjects", min(start + batch_size, len(ds)), len(ds))
    summary = {}
    for view in ("t1", "flair"):
        selected = [r for r in rows if r["modality"] == view]
        metrics = {}
        for key in selected[0]:
            if key in (
                "index",
                "modality",
                "valid_input",
                "valid_routing",
                "endpoint_signal_resolved",
                "z_low",
                "z_high",
                "eps",
                "isolated_intervention",
            ):
                continue
            routing_metric = key.endswith(("_gain", "_rms_ratio")) or key in (
                "joint_cosine",
                "joint_relative_error",
                "joint_energy_in_affected_fraction",
                "joint_outside_rms",
            )
            needs_valid = routing_metric or key.endswith(("_delta_rms", "_gap_to_native_rms"))
            values = np.asarray([r[key] for r in selected if not needs_valid or r["valid_routing"]], float)
            values = values[np.isfinite(values)]
            ci = None
            if len(values) and routing_metric:
                rng = np.random.default_rng(1729)
                ci = np.quantile(
                    values[rng.integers(len(values), size=(1000, len(values)))].mean(1), [0.025, 0.975]
                ).tolist()
            metrics[key] = {
                "median": float(np.median(values)) if len(values) else None,
                "mean": float(values.mean()) if len(values) else None,
                "mean_ci95": ci,
                "n_valid": len(values),
            }
        summary[view] = {
            "n": len(selected),
            "n_valid_input": sum(r["valid_input"] for r in selected),
            "n_valid_routing": sum(r["valid_routing"] for r in selected),
            "n_isolated_intervention": sum(r["isolated_intervention"] for r in selected),
            "n_lesion_changed": sum(r["lesion_changed_voxels"] > 0 for r in selected),
            "metrics": metrics,
        }
    return rows, summary, panels


def save_example(sample, modality, xa, xb, decoded, directory, nifti=False):
    """Both modalities, all donor combinations and signed responses on one scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = {
        "input_a": xa,
        "input_b": xb,
        "recon_aa": decoded["aa"],
        "content_b_style_a": decoded["ba"],
        "content_a_style_b": decoded["ab"],
        "recon_bb": decoded["bb"],
        "input_delta": xb - xa,
        "joint_delta": decoded["bb"] - decoded["aa"],
    }
    support = sample["support"]
    z = int(np.argmax(support.sum((0, 1)))) if support.any() else xa.shape[2] // 2
    fg = sample["mask"].numpy()[0] > 0
    lo, hi = np.quantile(np.concatenate([xa[fg], xb[fg]]), [0.01, 0.99])
    limit = max(float(np.abs(xb - xa).max()), 1e-8)
    fig, axes = plt.subplots(1, len(arrays), figsize=(20, 3))
    for col, (ax, (name, array)) in enumerate(zip(axes, arrays.items())):
        ax.imshow(
            array[:, :, z].T,
            origin="lower",
            cmap="gray" if col < 6 else "coolwarm",
            vmin=lo if col < 6 else -limit,
            vmax=hi if col < 6 else limit,
        )
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.axis("off")
    prefix = f"sample{sample['index']:04d}_ventricle_{modality}"
    note = "isolated" if sample["isolated_intervention"] else "CONFOUNDED: excluded from routing summary"
    fig.suptitle(f"{prefix}: z={z}, ventricle size -/+{sample['eps']:g}; {note}")
    fig.tight_layout()
    fig.savefig(directory / f"{prefix}.png", dpi=110)
    plt.close(fig)
    if nifti:
        import nibabel as nib

        arrays.update(
            affected_mask=maximum_filter(support, size=3) & fg,
            changed_tissue_mask=support,
            tissue_a=sample["tissues"][0],
            tissue_b=sample["tissues"][1],
        )
        for name, array in arrays.items():
            nib.save(nib.Nifti1Image(np.asarray(array, np.float32), np.eye(4)), directory / f"{prefix}_{name}.nii.gz")


def save_panels(panels, path):
    if not panels:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 6, figsize=(15, 2.6 * len(panels)), squeeze=False)
    for row, (idx, z, images) in zip(axes, panels):
        lo, hi = np.quantile(np.stack(images[:2]), [0.01, 0.99])
        for ax, im, title in zip(row, images, ("Input A", "Input B", "cA / sA", "cB / sA", "cA / sB", "cB / sB")):
            ax.imshow(im.T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"{title} · #{idx} z={z}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2, help="Subjects; forward contains four volumes per subject")
    p.add_argument("--eps", type=float, default=0.25, help="Low/high are original z_content[1] minus/plus eps")
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--old-generator", action="store_true", help="Use the pre-7ac56a3 renderer for older checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--examples", type=int, default=2, help="First N subjects, both modalities")
    p.add_argument("--save-nifti", action="store_true", help="Save example volumes in synthetic voxel coordinates")
    p.add_argument("--out-dir", default=None)
    cli = p.parse_args()
    if (
        min(cli.num_samples, cli.batch_size, cli.cpu_threads) < 1
        or cli.examples < 0
        or not np.isfinite(cli.eps)
        or cli.eps <= 0
    ):
        p.error("Need positive samples/batch size/threads/eps and nonnegative examples")
    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
    directory = Path(cli.out_dir or Path(cli.run_dir) / f"ventricle_routing_{datetime.now():%Y%m%d_%H%M%S_%f}")
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "arguments": vars(cli),
        "run_settings": vars(args),
        "checkpoint": provenance,
        "protocol": __doc__,
        "normalization": "Original sample foreground affine frozen for A and B",
        "generator": "legacy_pre_7ac56a3" if cli.old_generator else "current",
        "render_resolution": ds.res,
        "summary_statistic": "Paired subject means, matching lesion_routing; medians retained for older comparisons",
        "nifti_coordinates": "Identity affine: synthetic voxel indices, no patient orientation or physical spacing",
        "replay_validation": {
            "endpoint_rms_atol": REPLAY_RMS_ATOL,
            "endpoint_rms_rtol": REPLAY_RMS_RTOL,
            "max_endpoint_error_to_input_ratio": REPLAY_SIGNAL_FRACTION,
            "cuda_math": "TF32 off; cuDNN benchmark off, deterministic on; caller flags restored",
        },
        "interpretation": "Compare content_mean_gain and style_mean_gain only alongside joint_gain, joint_cosine and joint_relative_error. "
        "Means add to joint gain per sample; medians need not. Weak joint fidelity makes routing inconclusive. "
        "Latent RMS values depend on scale/width and are sensitivity diagnostics, not information scores. "
        "Input-invisible interventions have null gains and remain in coverage counts. "
        "Pairs with changed rendered lesions or input changes outside the ventricular/blur ROI are confounded and "
        "excluded from routing summaries. All other factor values, including SCM descendants, are held fixed; "
        "this is not a propagated SCM intervention. "
        "Rows with valid_routing=false retain raw scores but are excluded from routing metric summaries. "
        "Endpoint replay error calibration does not bound all hybrid numerical errors. No checkpoints are written.",
    }
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    rows, summary, panels = audit(
        model, ds, device, cli.eps, cli.batch_size, cli.examples, directory=directory, nifti=cli.save_nifti
    )
    for filename in ("responses.csv", "samples.csv"):
        with (directory / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_panels(panels, directory / "examples.png")
    report.update(status="complete", summary=summary, registered_state_unchanged=True)
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print("\nPaired mean ventricle-size responses (identity gain=1; no response=0)")
    for view, result in summary.items():
        print(f"{view}: measurable input {result['n_valid_input']}/{result['n']}")
        print(
            f"  isolated interventions: {result['n_isolated_intervention']}/{result['n']} "
            f"(lesion changed in {result['n_lesion_changed']})"
        )
        print(f"  isolated and resolved above endpoint replay error: {result['n_valid_routing']}/{result['n']}")
        for name in (
            "joint_gain",
            "joint_cosine",
            "joint_relative_error",
            "content_mean_gain",
            "style_mean_gain",
            "interaction_rms_ratio",
            "joint_energy_in_affected_fraction",
            "joint_outside_rms",
            "aa_roi_mae",
            "bb_roi_mae",
            "endpoint_replay_rms",
            "endpoint_error_to_input_ratio",
        ):
            value = result["metrics"][name]
            print(f"  {name}: {value['mean']} (n={value['n_valid']})")
    print("\nDecoder-bound latent response RMS: native / GAP (sensitivity, not decodability)")
    for view, result in summary.items():
        for name, value in result["metrics"].items():
            if name.endswith("_gap_delta_rms") and "_post_" in name:
                native = result["metrics"][name.replace("_gap_delta_rms", "_delta_rms")]
                print(f"{view:<6} {name.removesuffix('_gap_delta_rms'):<20} {native['mean']} / {value['mean']}")
    print("Gains: identity response=1, no response=0. Check joint fidelity before assigning a pathway.")
    print("Small GAP response with native sensitivity shows pooling attenuation, not absence of information.")
    print(f"Saved {directory}\nNo registered model parameter or buffer changed; no checkpoint was written.")


if __name__ == "__main__":
    main()
