"""Projection conditioning and controlled ventricular sensitivity, without training.

See VENTRICLE_QUANTIZER_AUDIT.md. This measures feature/code responses, not
decoder reliance, mutual information, or a causal effect of the training loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.content_path_probe import stage_maps, state_digest
from eval.lesion_reconstruction import json_safe, make_dataset
from eval.pooling_probe import VIEWS
from eval.style_path_audit import validate_model

LOG = logging.getLogger(__name__)
REGIONS = ("all", "affected_bins", "neighborhood", "outside")


def projection_diagnostics(conv):
    """SVD of a pointwise affine projection; rank tolerance at float32 precision."""
    if tuple(conv.kernel_size) != (1, 1, 1) or conv.groups != 1 or tuple(conv.stride) != (1, 1, 1):
        raise ValueError("Expected an ungrouped pointwise codebook projection")
    weight = conv.weight.detach().double().cpu().reshape(conv.out_channels, conv.in_channels)
    if not torch.isfinite(weight).all():
        raise ValueError("Non-finite projection weights")
    singular = torch.linalg.svdvals(weight)
    tol = max(weight.shape) * torch.finfo(torch.float32).eps * float(singular[0])
    rank = int((singular > tol).sum())
    rank64 = int(torch.linalg.matrix_rank(weight))
    full = rank == weight.shape[1]
    return {
        "input_channels": weight.shape[1],
        "output_channels": weight.shape[0],
        "singular_values": singular.tolist(),
        "float32_rank_tolerance": tol,
        "rank_float32": rank,
        "rank_float64": rank64,
        "input_nullity_float32": weight.shape[1] - rank,
        "full_column_rank_float32": full,
        "condition_number": float(singular[0] / singular[-1]) if full else None,
        "pseudoinverse_rtol": max(weight.shape) * torch.finfo(torch.float32).eps,
    }


def codebook_for(model, block, view):
    if block == "content":
        return model.codebooks_v1[0] if view and model.separate_content_codebooks else model.codebooks[0]
    if not model.quantize_style:
        return None
    return model.style_codebooks_v1["0"] if view and model.separate_style_codebooks else model.style_codebooks["0"]


def fixed_affine(raw, normalized, mask):
    x, y = raw[mask > 0].double(), normalized[mask > 0].double()
    xc = x - x.mean()
    if float(xc.square().sum()) <= 1e-12:
        raise ValueError("Cannot identify normalization affine from constant foreground")
    scale = (xc * (y - y.mean())).sum() / xc.square().sum()
    offset = y.mean() - scale * x.mean()
    replay = (raw.double() * scale + offset) * mask
    if not torch.allclose(replay, normalized.double(), rtol=2e-5, atol=2e-5):
        raise ValueError("Normalizer is not foreground-affine; cannot freeze it safely")
    return scale.float(), offset.float()


def render_pair(ds, idx, eps):
    """z1 +/- eps, holding every other realized latent and render seed fixed.

    This is a direct latent intervention, not propagation through SCM descendants.
    Paired variants use the unperturbed sample's affine and foreground mask.
    """
    inner = ds._inner
    a, b, lat = inner[idx]
    mask = lat["brain_mask"]
    normalized = ds.normalize_views(a, b, mask, mask)
    affines = [fixed_affine(raw, norm, mask) for raw, norm in zip((a, b), normalized)]

    def render(z):
        return inner.render_pseudo_mri(
            z,
            lat["z_deformation"],
            lat["z_fissure"],
            lat["z_style_v1"],
            lat["z_style_v2"],
            inner.sample_seed_for(idx),
            z_lesion=lat.get("z_lesion"),
        )

    replay = render(lat["z_content"])
    if not torch.equal(a, replay[0]) or not torch.equal(b, replay[1]):
        raise ValueError("Original render is not reproducible with stored latents and seeds")
    images, tissues, lesions, masks = [], [], [], []
    for delta in (-eps, eps):
        z = lat["z_content"].clone()
        z[1] += delta
        raw = render(z)
        images.append([(v * scale + offset) * mask for v, (scale, offset) in zip(raw[:2], affines)])
        tissue, lesion = inner.renderer.render_structure(
            z, lat["z_deformation"], lat["z_fissure"], "cpu", clean=inner.clean_content, z_lesion=lat.get("z_lesion")
        )
        tissues.append(tissue)
        lesions.append(lesion)
        masks.append(raw[2])
    if not torch.equal(lesions[0], lesions[1]):
        raise ValueError("Ventricle intervention unexpectedly changed the lesion load")
    changed = (tissues[0] != tissues[1]).unsqueeze(0) & mask.bool()
    # The renderer blurs with a 3x3x3 average: expand changed tissue support by one voxel.
    support = F.max_pool3d(changed.float().unsqueeze(0), 3, 1, 1)[0].bool() & mask.bool()
    outside_max = max(
        float((hi - lo)[~support].abs().max()) if (~support).any() else 0.0 for lo, hi in zip(images[0], images[1])
    )
    if outside_max > 2e-6:
        raise ValueError("Rendered response escaped the predicted changed-tissue/blur support")
    return {
        "low": images[0],
        "high": images[1],
        "mask": mask,
        "support": support,
        "changed_tissue_voxels": int(changed.sum()),
        "support_voxels": int(support.sum()),
        "foreground_mask_changed_voxels": int((masks[0] != masks[1]).sum()),
        "z1_low": float(lat["z_content"][1]) - eps,
        "z1_high": float(lat["z_content"][1]) + eps,
        "input_outside_support_max_abs": outside_max,
    }


def encode(model, samples, which, device):
    count = len(samples)
    x = torch.cat([torch.stack([s[which][v] for s in samples]) for v in range(2)]).to(device)
    mask = torch.cat([torch.stack([s["mask"] for s in samples])] * 2).to(device)
    stages, partitions = stage_maps(model, x, mask)
    ids = {"content": model._last_id_outputs[0].detach().cpu().clone()}
    if model.quantize_style:
        ids["style"] = model._last_style_id_outputs[0].detach().cpu().clone()
    # CPU storage keeps three paired/replay snapshots off GPU while running the next batch.
    return {k: v.detach().cpu() for k, v in stages.items()}, ids, partitions


def regions_at(support, shape, halo):
    affected = F.adaptive_max_pool3d(support.float().unsqueeze(0), shape)[0, 0].bool()
    if halo:
        near = F.max_pool3d(affected.float()[None, None], 2 * halo + 1, 1, halo)[0, 0].bool()
    else:
        near = affected.clone()
    return {"all": torch.ones_like(affected), "affected_bins": affected, "neighborhood": near, "outside": ~near}


def rms(x):
    return float(x.double().square().mean().sqrt()) if x.numel() else float("nan")


def response_metrics(
    pre_low, pre_high, q_low, q_high, ids_low, ids_high, replay_pre, replay_q, replay_ids, region, signal_floor=1e-8
):
    """Same-embedding-space finite differences; ratios are not information fractions."""
    n = int(region.sum())
    if n == 0:
        return {"sites": 0, "resolved_pre": False, "resolved_quant": False}
    dp = (pre_high - pre_low)[:, region].double()
    dq = (q_high - q_low)[:, region].double()
    noise_p = rms((replay_pre - pre_low)[:, region])
    noise_q = rms((replay_q - q_low)[:, region])
    p_rms, q_rms = rms(dp), rms(dq)
    valid_p = p_rms > max(signal_floor, 10 * noise_p)
    valid_q = q_rms > max(signal_floor, 10 * noise_q)
    p_energy, q_energy = float(dp.square().sum()), float(dq.square().sum())
    dot = float((dp * dq).sum())
    error = torch.cat([(q_low - pre_low)[:, region], (q_high - pre_high)[:, region]], dim=0)
    return {
        "sites": n,
        "pre_delta_rms": p_rms,
        "quant_delta_rms": q_rms,
        "pre_replay_rms": noise_p,
        "quant_replay_rms": noise_q,
        "resolved_pre": valid_p,
        "resolved_quant": valid_q,
        "changed_code_fraction": float((ids_low[region] != ids_high[region]).float().mean()),
        "replay_changed_code_fraction": float((ids_low[region] != replay_ids[region]).float().mean()),
        "quant_to_pre_rms_ratio": q_rms / p_rms if valid_p else None,
        "quant_response_gain": dot / p_energy if valid_p else None,
        "quant_response_cosine": dot / np.sqrt(p_energy * q_energy) if valid_p and valid_q else None,
        "quant_response_relative_error": rms(dq - dp) / p_rms if valid_p else None,
        "endpoint_quantization_error_rms": rms(error),
        "quantization_error_to_response_ratio": rms(error) / p_rms if valid_p else None,
    }


def projection_response(conv, low_input, high_input, low_pre, high_pre, region):
    """Check the real affine and pseudoinvert the response using float32-rank tolerance."""
    if not region.any():
        return {}
    weight = conv.weight.detach().double().cpu().flatten(1)
    bias = conv.bias.detach().double().cpu() if conv.bias is not None else torch.zeros(weight.shape[0])
    before = (high_input - low_input)[:, region].double()
    after = (high_pre - low_pre)[:, region].double()
    err = max(
        rms(weight @ value[:, region].double() + bias[:, None] - pre[:, region].double())
        for value, pre in ((low_input, low_pre), (high_input, high_pre))
    )
    scale = max(rms(low_pre[:, region]), rms(high_pre[:, region]))
    if err > 2e-5 * max(1.0, scale):
        raise ValueError("Captured pre-quant features do not match the codebook affine")
    recovered = torch.linalg.pinv(weight, rtol=max(weight.shape) * torch.finfo(torch.float32).eps) @ after
    before_rms = rms(before)
    return {
        "projection_input_delta_rms": before_rms,
        "projection_forward_error_rms": err,
        "projection_response_norm_ratio": float(after.norm() / before.norm()) if before_rms > 1e-8 else None,
        "projection_inverse_response_relative_error": (
            rms(recovered - before) / before_rms if before_rms > 1e-8 else None
        ),
    }


def usage_stats(counts):
    total = int(counts.sum())
    p = counts[counts > 0] / total if total else np.array([])
    return {
        "assignments": total,
        "active_codes": int((counts > 0).sum()),
        "codebook_entries": len(counts),
        "perplexity": float(np.exp(-(p * np.log(p)).sum())) if total else None,
        "unobserved_codes": int((counts == 0).sum()),
        "counts": counts.tolist(),
    }


def summarize(rows):
    summaries = []
    keys = list(dict.fromkeys((r["eps"], r["view"], r["block"], r["region"]) for r in rows))
    metrics = (
        "pre_delta_rms",
        "quant_delta_rms",
        "changed_code_fraction",
        "quant_to_pre_rms_ratio",
        "quant_response_gain",
        "quant_response_cosine",
        "quant_response_relative_error",
        "projection_inverse_response_relative_error",
        "endpoint_quantization_error_rms",
        "pre_energy_fraction",
        "quant_energy_fraction",
    )
    for eps, view, block, region in keys:
        subset = [r for r in rows if (r["eps"], r["view"], r["block"], r["region"]) == (eps, view, block, region)]
        valid = [r for r in subset if r["input_measurable"] and r["sites"] > 0]
        entry = {
            "eps": eps,
            "view": view,
            "block": block,
            "region": region,
            "n": len(subset),
            "n_measurable_nonempty": len(valid),
            "n_resolved_pre": sum(r["resolved_pre"] for r in valid),
            "n_resolved_quant": sum(r["resolved_quant"] for r in valid),
        }
        for metric in metrics:
            values = [r[metric] for r in valid if r.get(metric) is not None and np.isfinite(r[metric])]
            entry[metric] = {
                "n": len(values),
                "median": float(np.median(values)) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }
        summaries.append(entry)
    return summaries


def audit(model, ds, device, epsilons, batch_size, halo, examples):
    rows, usage, saved, partitions = [], {}, {}, None
    for eps in epsilons:
        for start in range(0, len(ds), batch_size):
            samples = [render_pair(ds, i, eps) for i in range(start, min(start + batch_size, len(ds)))]
            low, low_ids, part = encode(model, samples, "low", device)
            high, high_ids, part_high = encode(model, samples, "high", device)
            repeat, repeat_ids, part_repeat = encode(model, samples, "low", device)
            if not np.array_equal(part, part_high) or not np.array_equal(part, part_repeat):
                raise ValueError("Channel selection changed across intervention/replay")
            if partitions is not None and not np.array_equal(partitions, part):
                raise ValueError("Channel selection changed across batches")
            partitions = part
            count = len(samples)
            for v, view in enumerate(VIEWS):
                for b, sample in enumerate(samples):
                    index = v * count + b
                    dx = sample["high"][v] - sample["low"][v]
                    for block in low_ids:
                        cb = codebook_for(model, block, v)
                        pl, ph = low[(block, "pre_quant")][index], high[(block, "pre_quant")][index]
                        ql, qh = low[(block, "decoder_input")][index], high[(block, "decoder_input")][index]
                        il, ih, ir = low_ids[block][index], high_ids[block][index], repeat_ids[block][index]
                        if il.shape != pl.shape[1:] or ql.shape != pl.shape:
                            raise ValueError("Assignment and embedding shapes disagree")
                        source = "post_norm" if block == "content" else "bottleneck"
                        src_low, src_high = low[(block, source)][index], high[(block, source)][index]
                        regions = regions_at(sample["support"], pl.shape[1:], halo)
                        total_pre = float((ph.double() - pl.double()).square().sum())
                        total_q = float((qh.double() - ql.double()).square().sum())
                        for name, region in regions.items():
                            metrics = response_metrics(
                                pl,
                                ph,
                                ql,
                                qh,
                                il,
                                ih,
                                repeat[(block, "pre_quant")][index],
                                repeat[(block, "decoder_input")][index],
                                ir,
                                region,
                            )
                            metrics.update(projection_response(cb.conv_in, src_low, src_high, pl, ph, region))
                            metrics["pre_energy_fraction"] = (
                                float((ph[:, region].double() - pl[:, region].double()).square().sum()) / total_pre
                                if total_pre > 1e-16
                                else None
                            )
                            metrics["quant_energy_fraction"] = (
                                float((qh[:, region].double() - ql[:, region].double()).square().sum()) / total_q
                                if total_q > 1e-16
                                else None
                            )
                            rows.append(
                                {
                                    "index": start + b,
                                    "eps": eps,
                                    "view": view,
                                    "block": block,
                                    "region": name,
                                    "input_delta_rms": rms(dx),
                                    "input_measurable": rms(dx) > 1e-8,
                                    **{
                                        k: sample[k]
                                        for k in (
                                            "z1_low",
                                            "z1_high",
                                            "changed_tissue_voxels",
                                            "support_voxels",
                                            "foreground_mask_changed_voxels",
                                            "input_outside_support_max_abs",
                                        )
                                    },
                                    **metrics,
                                }
                            )
                            key = f"eps{eps:g}_{view}_{block}_{name}"
                            hist = usage.setdefault(key, np.zeros(cb.n_embed, dtype=np.int64))
                            if rms(dx) > 1e-8:
                                hist += np.bincount(torch.cat([il[region], ih[region]]).numpy(), minlength=cb.n_embed)
                        if start + b < examples:
                            prefix = f"eps{eps:g}_sample{start+b}_{view}_{block}"
                            for name, value in {
                                "pre_response_norm": (ph - pl).square().sum(0).sqrt(),
                                "quant_response_norm": (qh - ql).square().sum(0).sqrt(),
                                "changed_codes": il != ih,
                                "affected_bins": regions["affected_bins"],
                                "neighborhood": regions["neighborhood"],
                                "input_delta": dx,
                                "input_low": sample["low"][v],
                                "input_support": sample["support"],
                            }.items():
                                saved[f"{prefix}_{name}"] = value.numpy()
            LOG.info("eps=%g: audited %d/%d pairs, including numerical replay", eps, start + count, len(ds))
    return rows, {k: usage_stats(v) for k, v in usage.items()}, saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.25])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--halo", type=int, default=1, help="Neighborhood dilation in native feature-map sites")
    parser.add_argument("--causal", choices=("iid", "match"), default="iid")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args(argv)
    if min(cli.num_samples, cli.batch_size, cli.threads) < 1 or min(cli.halo, cli.examples) < 0:
        parser.error("Samples/batch/threads must be positive; halo/examples nonnegative")
    if any(not np.isfinite(e) or e <= 0 for e in cli.eps):
        parser.error("--eps requires positive finite perturbations")
    cli.eps = list(dict.fromkeys(cli.eps))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.threads)
    from eval.run_dci_synthetic import load_model_from_run_dir

    model, args, device = load_model_from_run_dir(
        cli.run_dir, cli.checkpoint, torch.device(cli.device) if cli.device else None
    )
    checkpoint = Path(cli.checkpoint or "vqvae_model.pt")
    if checkpoint.parent == Path("."):
        checkpoint = Path(cli.run_dir) / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.get("encoders", state).items()}, strict=True)
    step = state.get("step")
    del state
    model.eval().requires_grad_(False)
    validate_model(model)
    before = state_digest(model)
    projections = {
        f"{view}_{block}": projection_diagnostics(cb.conv_in)
        for v, view in enumerate(VIEWS)
        for block in ("content", "style")
        if (cb := codebook_for(model, block, v)) is not None
    }
    ds = make_dataset(args, cli.num_samples, cli.causal, "test")
    output = Path(cli.output_dir or Path(cli.run_dir) / f"ventricle_quantizer_audit_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    rows, usage, examples = audit(model, ds, device, cli.eps, cli.batch_size, cli.halo, cli.examples)
    if state_digest(model) != before:
        raise ValueError("Registered model state changed; refusing to report a frozen audit")
    summary = summarize(rows)
    report = {
        "config": vars(cli),
        "dataset_settings": vars(args),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": step,
        "model_state_unchanged": True,
        "state_sha256": before,
        "projection": projections,
        "summary": summary,
        "code_usage": usage,
        "notes": [
            "Only z_content[1] is changed; other realized latents and noise seeds are fixed.",
            "Original normalization affine and brain mask are held fixed; variants may be off-distribution.",
            "Regions map changed tissue plus renderer blur to spatial bins; they are not encoder receptive fields.",
            "Style is audited only when quantized. Spatially global style has no meaningful regional localization.",
            "Response gains/norm ratios are not information-retention fractions or decoder-use measures.",
            "Usage counts include both paired endpoints; unobserved codes are not necessarily dead codes.",
            "Null/rank conclusions depend on tolerance; raw singular values and paired rows are saved.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    with (output / "responses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    if examples:
        np.savez_compressed(output / "examples.npz", **examples)
    print("\nProjection SVD (float32 rank criterion)")
    for name, info in projections.items():
        cond = info["condition_number"]
        print(
            f"{name:14s} rank={info['rank_float32']}/{info['input_channels']}  "
            f"condition={cond if cond is not None else 'not full column rank'}"
        )
    print("\nMedians over measurable pairs; inspect replay/error columns in responses.csv")
    print("eps   view   block    region          valid resolved  changed_codes  quant/pre RMS  cosine")

    def fmt(value):
        return "n/a" if value is None else f"{value:.4g}"

    for item in summary:
        print(
            f"{item['eps']:<5g} {item['view']:6s} {item['block']:8s} {item['region']:15s} "
            f"{item['n_measurable_nonempty']:5d} {item['n_resolved_pre']:8d}  "
            f"{fmt(item['changed_code_fraction']['median']):>12s}  "
            f"{fmt(item['quant_to_pre_rms_ratio']['median']):>13s}  {fmt(item['quant_response_cosine']['median'])}"
        )
    print(f"\nSaved {output}\nNo registered model parameter or buffer changed.")


if __name__ == "__main__":
    main()
