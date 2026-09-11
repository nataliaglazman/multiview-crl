"""Lesion-specific cross-view alignment, using matched lesion-on/off images.

    python -m eval.lesion_alignment --run-dir /path/to/run --causal both

Measures delta_h = h(lesion-on) - h(lesion-off) at native probe maps,
native pre-normalization alignment sources, and actual pooled/projected BT inputs.
Content selection and foreground patch filtering follow training. The on/off pair
shares anatomy, style, rendering noise and the original normalization affine.

This is an eval-mode sensitivity diagnostic, not a training-gradient attribution.
BT terms are recomputed with the saved coefficients and instantaneous correlations;
the historical EMA state and training-mode stochastic masks are not replayed.
Lesion-free images may be outside the training distribution. Near-zero responses
are reported as undefined cosine, never as successful alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.lesion_reconstruction import json_safe, make_dataset, render_pair

logger = logging.getLogger(__name__)


def safe_ratio(a, b):
    return a / b if b > 1e-20 else float("nan")


def pair_metrics(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    aa, bb = float(a.square().sum()), float(b.square().sum())
    dot = float((a * b).sum())
    return {
        "cosine": safe_ratio(dot, (aa * bb) ** 0.5),
        "relative_mse": safe_ratio(float((a - b).square().sum()), aa + bb),
        "t1_rms": float(a.square().mean().sqrt()),
        "flair_rms": float(b.square().mean().sqrt()),
        "t1_over_flair_rms": safe_ratio(aa**0.5, bb**0.5),
        "mse": float((a - b).square().mean()),
    }


def response_rows(on, off, indices, stage, distribution):
    delta = on - off
    centered = on - on.mean(dim=1, keepdim=True)
    rows = []
    for i, idx in enumerate(indices):
        row = {"distribution": distribution, "index": int(idx), "stage": stage}
        for prefix, a, b in (
            ("delta", delta[0, i], delta[1, i]),
            ("on", on[0, i], on[1, i]),
            ("subject_centered_on", centered[0, i], centered[1, i]),
        ):
            row.update({f"{prefix}_{k}": v for k, v in pair_metrics(a, b).items()})
        row["delta_wrong_subject_cosine"] = pair_metrics(delta[0, i], delta[1, (i + 1) % len(indices)])["cosine"]
        for v, name in enumerate(("t1", "flair")):
            row[f"delta_to_subject_variation_{name}"] = safe_ratio(
                float(delta[v, i].square().sum()), float(centered[v, i].square().sum())
            )
        rows.append(row)
    return rows


def grid_for(args, level):
    if not getattr(args, "patch_contrastive", False):
        return None, None
    all_grids = getattr(args, "patch_grid_per_level", None)
    spec = all_grids if all_grids is not None else tuple(args.patch_grid)
    return spec, tuple(all_grids[level] if all_grids is not None else args.patch_grid)


def selected_channels(masks, level, channels):
    selected = masks.get(level)
    if selected is None:
        return [torch.arange(channels)] * 2
    if not isinstance(selected, tuple):
        selected = (selected, selected)
    result = [torch.where(m.detach().cpu().reshape(-1).bool())[0] for m in selected]
    if not len(result[0]) or len(result[0]) != len(result[1]):
        raise ValueError("Need nonempty, equal-width content blocks across views")
    return result


def select_views(features, indices):
    b = features.shape[0] // 2
    return torch.stack([features[v * b : (v + 1) * b, indices[v]] for v in range(2)])


def extract_stages(model, images, masks, args, device, level):
    """Use real forward pool outputs; capture their pre-normalization source separately."""
    x = torch.cat([torch.stack([s[v] for s in images]) for v in range(2)]).to(device)
    m = torch.cat([torch.stack(masks)] * 2).to(device)
    sources = []
    norm = (
        getattr(model, "content_norms", {}).get(str(level))
        if isinstance(getattr(model, "content_norms", None), dict)
        else None
    )
    if hasattr(model, "content_norms") and str(level) in model.content_norms:
        norm = model.content_norms[str(level)]
    hook = (
        norm.register_forward_pre_hook(lambda mod, inp: sources.append(inp[0].detach().cpu().clone()))
        if norm is not None
        else None
    )
    try:
        with torch.inference_mode():
            native = model(x, return_recon=False, pool_only=False, n_views=2, subsets=[(0, 1)], mask=m)
    finally:
        if hook is not None:
            hook.remove()
    if level >= len(native[2]):
        raise ValueError(f"No encoder level {level}")
    maps = native[2][level].detach().cpu()
    indices = selected_channels(native[6], level, maps.shape[1])
    stages = {"native_probe": select_views(maps, indices)}
    source = torch.cat(sources, dim=0) if sources else maps
    stages["native_alignment_source"] = select_views(source, indices)
    spec, grid = grid_for(args, level)
    with torch.inference_mode():
        pooled = model(x, return_recon=False, pool_only=True, n_views=2, subsets=[(0, 1)], mask=m, patch_grid=spec)
    pooled_indices = selected_channels(pooled[6], level, maps.shape[1])
    if any(not torch.equal(a, b) for a, b in zip(indices, pooled_indices)):
        raise ValueError("Channel selection changed between native and pooled forwards")
    features = pooled[2][level].detach().cpu()
    # Verify that the captured source is the exact map pooled by this checkpoint.
    expected = F.adaptive_avg_pool3d(source, grid).flatten(2) if grid else source.mean((2, 3, 4))
    torch.testing.assert_close(features, expected, atol=2e-5, rtol=2e-5)
    keep = None
    if grid and getattr(args, "patch_foreground_mask", False):
        fraction = F.adaptive_avg_pool3d(m.cpu(), grid).flatten(1)
        keep = (fraction >= float(getattr(args, "patch_foreground_thresh", 0.05))).any(0)
        if not bool(keep.any()):
            keep = torch.ones_like(keep)
        features = features[..., keep]
    content = select_views(features, indices)
    stages["pooled_content"] = content
    heads = getattr(model, "_contrastive_proj_heads", {})
    if f"L{level}" in heads:
        head = heads[f"L{level}"]
        a = content.permute(0, 1, 3, 2) if grid else content
        with torch.inference_mode():
            projected = head(a.reshape(-1, a.shape[-1]).to(device)).cpu().reshape(*a.shape[:-1], -1)
        content = projected.permute(0, 1, 3, 2).contiguous() if grid else projected
    stages["loss_patch" if grid else "loss_global"] = content
    if grid and float(getattr(args, "bt_gap_weight", 0)) > 0:
        # Training's companion averages retained patches AFTER projection/filtering.
        stages["loss_gap"] = content.mean(-1)
    for stage, values in stages.items():
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Non-finite features at {stage}")
    return stages, indices, keep


def bt_terms(hz, args, stage, level):
    from training.losses import barlow_twins_loss

    gap = stage == "loss_gap"

    def setting(name, default):
        base = getattr(args, name, default)
        override = getattr(args, name.replace("bt_", "bt_gap_", 1), None) if gap else None
        return base if override is None else override

    patch = stage == "loss_patch"
    lambd, sim, std = setting("bt_lambda", 0.005), setting("bt_sim_coeff", 0.0), setting("bt_std_coeff", 0.0)
    with torch.inference_mode():
        value = barlow_twins_loss(
            hz,
            estimated_content_indices=[list(range(hz.shape[2]))],
            subsets=[(0, 1)],
            lambd=lambd,
            sim_coeff=sim,
            std_coeff=std,
            center_mode=(getattr(args, "patch_center_mode", "none") or "none") if patch else "none",
            patch_stat=getattr(args, "bt_patch_stat", "fold"),
            sim_normalize=getattr(args, "bt_sim_normalize", False),
            normalize_terms=getattr(args, "bt_normalize_terms", False),
        )
    arm_weight = (
        float(getattr(args, "bt_gap_weight", 0))
        if gap
        else float(getattr(args, "bt_patch_weight", 1))
        if patch
        else 1.0
    )
    levels = getattr(args, "contrastive_level_weights", None)
    weight = (
        arm_weight
        * float(getattr(args, "scale_contrastive_loss", 1))
        * (levels[level] if levels and level < len(levels) else 1)
    )
    return {
        "instantaneous_weighted_total": float(value) * weight,
        "arm_scale": weight,
        "sim_coefficient": sim,
        "std_coefficient": std,
        "lambda": lambd,
        **value._contrastive_diag,
    }


def audit(model, ds, args, device, level, batch_size, distribution):
    rows, losses, channels = [], [], []
    groups = np.array_split(np.arange(len(ds)), int(np.ceil(len(ds) / batch_size)))
    for batch, indices in enumerate(groups):
        samples = [render_pair(ds, int(i)) for i in indices]
        masks = [s[2] for s in samples]
        on, selected, keep = extract_stages(model, [s[0] for s in samples], masks, args, device, level)
        off, off_selected, off_keep = extract_stages(model, [s[1] for s in samples], masks, args, device, level)
        if any(not torch.equal(a, b) for a, b in zip(selected, off_selected)):
            raise ValueError("Lesion removal changes content channel selection; cannot compare fixed channel responses")
        if keep is not None and not torch.equal(keep, off_keep):
            raise ValueError("Foreground patch selection changed under lesion removal")
        for stage in on:
            rows.extend(response_rows(on[stage], off[stage], indices, stage, distribution))
            if stage.startswith("loss_"):
                before, after = bt_terms(on[stage], args, stage, level), bt_terms(off[stage], args, stage, level)
                losses.append(
                    {
                        "distribution": distribution,
                        "batch": batch,
                        "n": len(indices),
                        "stage": stage,
                        "on": before,
                        "off": after,
                        "on_minus_off_weighted_total": before["instantaneous_weighted_total"]
                        - after["instantaneous_weighted_total"],
                    }
                )
                delta = on[stage] - off[stage]
                for c in range(delta.shape[2]):
                    channels.append(
                        {
                            "distribution": distribution,
                            "batch": batch,
                            "stage": stage,
                            "channel": c,
                            **pair_metrics(delta[0, :, c], delta[1, :, c]),
                        }
                    )
        logger.info("%s: measured %d/%d subjects", distribution, int(indices[-1]) + 1, len(ds))
    summary = {}
    for stage in on:
        subset = [r for r in rows if r["stage"] == stage]
        metrics = {}
        for key in subset[0]:
            if key in ("distribution", "index", "stage"):
                continue
            vals = np.array([r[key] for r in subset], dtype=float)
            valid = vals[np.isfinite(vals)]
            metrics[key] = {
                "median": float(np.median(valid)) if len(valid) else None,
                "mean": float(valid.mean()) if len(valid) else None,
                "n_valid": len(valid),
            }
        summary[stage] = metrics
    return rows, losses, channels, summary


def load_model(run_dir, checkpoint, device):
    from eval.run_dci_synthetic import load_model_from_run_dir, load_run_args

    args = load_run_args(run_dir)
    if getattr(args, "contrastive_loss_type", "barlow_twins") != "barlow_twins" or getattr(args, "use_moco", False):
        raise ValueError("This diagnostic currently reproduces the Barlow Twins feature path")
    if getattr(args, "contrastive_proj_mode", "head") != "head":
        raise ValueError("This BT diagnostic supports the standard projection head, not entropy/bounded modes")
    if getattr(args, "split_encoder_norm", False):
        raise ValueError(
            "Shared checkpoint loader does not restore split_encoder_norm; cannot faithfully score this checkpoint"
        )
    model, args, device = load_model_from_run_dir(run_dir, checkpoint, device=device, seed=0)
    path = Path(checkpoint or "vqvae_model.pt")
    if path.parent == Path("."):
        path = Path(run_dir) / path
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in saved.get("encoders", saved).items()}
    if getattr(args, "contrastive_proj_dim", 0) > 0:
        heads = torch.nn.ModuleDict()
        for level in getattr(args, "content_style_levels", [0]):
            width = model.content_channels_per_level.get(level)
            if width is None:
                continue
            head = torch.nn.Sequential(
                torch.nn.Linear(width, getattr(args, "contrastive_proj_hidden", 256)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(getattr(args, "contrastive_proj_hidden", 256), args.contrastive_proj_dim),
            )
            prefix = f"_contrastive_proj_heads.L{level}."
            head.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
            heads[f"L{level}"] = head
        model._contrastive_proj_heads = heads.to(device)
        logger.info("Restored contrastive heads omitted by the shared evaluation loader")
    expected = model.state_dict()
    missing = set(dict(model.named_parameters())) - state.keys()
    unexpected_encoder = [
        k
        for k in state
        if k.startswith(
            ("encoders.", "encoders_v1.", "content_norms.", "content_projections.", "_contrastive_proj_heads.")
        )
        and k not in expected
    ]
    if missing or unexpected_encoder:
        raise ValueError(
            f"Checkpoint architecture mismatch: missing parameters {sorted(missing)[:5]}, unexpected encoder/head state {unexpected_encoder[:5]}"
        )
    model.eval()
    return model, args, device


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--causal", choices=["iid", "match", "both"], default="iid")
    p.add_argument("--num-samples", type=int, default=64, help="Subjects per evaluation distribution")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--out-dir", default=None)
    cli = p.parse_args()
    if cli.num_samples < 4 or cli.batch_size < 2 or cli.level < 0 or cli.cpu_threads < 1:
        p.error("Need >=4 samples, batch size >=2, nonnegative level and positive CPU threads")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.cpu_threads)
    model, args, device = load_model(cli.run_dir, cli.checkpoint, cli.device)
    rows, losses, channels, summaries = [], [], [], {}
    for distribution in ("iid", "match") if cli.causal == "both" else (cli.causal,):
        ds = make_dataset(args, cli.num_samples, distribution, "test")
        r, l, c, summary = audit(model, ds, args, device, cli.level, cli.batch_size, distribution)
        rows.extend(r)
        losses.extend(l)
        channels.extend(c)
        summaries[distribution] = summary
    directory = Path(cli.out_dir or Path(cli.run_dir) / f"lesion_alignment_{cli.causal}_L{cli.level}")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": vars(cli),
        "run_settings": vars(args),
        "summary": summaries,
        "batch_bt_terms": losses,
        "stage_definitions": {
            "native_probe": "Post-content_norms native encoder maps, as used by lesion_probe",
            "native_alignment_source": "Native maps before content_norms; verified against forward pooling",
            "pooled_content": "Forward-pooled content after foreground patch filtering, before optional head",
            "loss_patch/global/gap": "Actual BT input after head; GAP averages retained projected patches",
        },
        "limitations": "Eval mode; instantaneous BT correlations, not historical EMA; no lesion-free training guarantee. Raw feature magnitudes are coordinate-dependent; cosine alone ignores magnitude. Batch size affects centering and foreground filtering.",
    }
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    write_csv(directory / "samples.csv", rows)
    write_csv(directory / "channels.csv", channels)
    for distribution, stages in summaries.items():
        print(f"\n{distribution}: medians across subjects; Δh = lesion-on − lesion-off")
        print("  stage                       Δcos   wrong-subj  T1/FLAIR RMS   relative MSE   on-cos")
        for stage, metrics in stages.items():

            def val(key):
                v = metrics[key]["median"]
                return f"{v:.3f}" if v is not None else "undefined"

            print(
                f"  {stage:27s} {val('delta_cosine'):>8s} {val('delta_wrong_subject_cosine'):>10s}"
                f" {val('delta_t1_over_flair_rms'):>13s} {val('delta_relative_mse'):>14s} {val('on_cosine'):>8s}"
            )
    print("\nMatched responses: Δcos near 1, T1/FLAIR RMS near 1, relative MSE near 0.")
    print("High on-cos with poor Δcos means overall similarity misses lesion-specific disagreement.")
    print("Inspect absolute response RMS in samples.csv; tiny/zero responses do not establish shared encoding.")
    print("BT on/off loss changes are not an additive attribution to lesions; correlations are recomputed.")
    print(f"Saved {directory}")


if __name__ == "__main__":
    main()
