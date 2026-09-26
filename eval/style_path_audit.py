"""Frozen style-path readouts and controlled gain/bias decoder swaps.

    python -m eval.style_path_audit --run-dir results/synthetic/MY_RUN

Readouts use natural samples, with disjoint subject train/validation/test sets.
Targets are the renderer's effective gain, bias and noise sigma, NOT signed noise
latents. Raw style is captured immediately before _bottleneck_style, pooled style
immediately after it, and injected style at the actual decoder call. Full spatial
tensors are flattened (no patch averaging/PCA); raw GAP is an additional control.
These are linear probes of availability, not proofs of absence or decoder usage.

Swaps separately set gain or bias latent to +/- intervention on the same anatomy,
holding all other latents, render seeds and the original normalization affine fixed.
The decoder is replayed on exact tensors, not code-ID lookup approximations. Both
content-held-fixed directions, content-only changes, and joint fidelity are scored
against the rendered image effect. This supports attribution at the decoder input,
not causal claims about the training loss. Frozen normalization can make variants
OOD for per-sample-normalized runs. Only single-level models with stable channel
masks are supported, to avoid changing cross-level decoder conditioning.

No representation training, optimizer update, or checkpoint write occurs.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.lesion_reconstruction import json_safe

LOG = logging.getLogger(__name__)
VIEWS = ("t1", "flair")
TARGETS = ("gain", "bias", "noise_sigma")


def effective_style(z, scale):
    """Matches PseudoMRIRenderer.render_modality, including clipping/absolute value."""
    z = F.pad(z.flatten(), (0, max(0, 3 - z.numel())))
    return np.array(
        [
            max(0.05, 1 + float(z[0].clamp(-1, 1)) * 0.3 * scale),
            float(z[1].clamp(-1, 1)) * 0.1 * scale,
            0.01 + abs(float(z[2])) * 0.05 * scale,
        ]
    )


def make_dataset(args, n, causal, split):
    from data.datasets import SyntheticBrainDataset

    accepted = inspect.signature(SyntheticBrainDataset.__init__).parameters
    kw = {k: v for k, v in vars(args).items() if k.startswith("synthetic_") and k in accepted}
    if kw.get("synthetic_mode", "pseudo_mri") != "pseudo_mri":
        raise ValueError("Style-path audit requires pseudo_mri rendering")
    if kw.get("synthetic_n_style", 3) < 3:
        raise ValueError("Need three style latents for gain/bias/noise readouts")
    if causal == "iid":
        kw.update(synthetic_causal=False, synthetic_hierarchical_content=False)
    kw.update(synthetic_num_samples=n, synthetic_num_samples_per_mode=None)
    res = getattr(args, "synthetic_res", 64)
    return SyntheticBrainDataset(
        mode=split, spatial_size=getattr(args, "spatial_size", None) or (res,) * 3, cache=False, **kw
    )


def render_sample(ds, idx, intervention=None):
    """Natural image plus optional gain/bias pairs; preserve field-lesion latents too."""
    inner = ds._inner
    r0, r1, lat = inner[idx]
    mask = lat["brain_mask"]
    original = list(ds.normalize_views(r0, r1, mask, mask))
    targets = np.stack([effective_style(lat[f"z_style_v{v + 1}"], inner.renderer.style_scale) for v in range(2)])
    if intervention is None:
        return original, mask, targets
    affine = []
    for raw, normalized in zip((r0, r1), original):
        x, y = raw[mask > 0].double(), normalized[mask > 0].double()
        xc = x - x.mean()
        if float(xc.square().sum()) < 1e-12:
            raise ValueError("Cannot recover normalization from a constant input")
        a = (xc * (y - y.mean())).sum() / xc.square().sum()
        b = y.mean() - a * x.mean()
        if not torch.allclose(a * x + b, y, atol=2e-5, rtol=2e-5):
            raise ValueError("Normalizer is not affine; cannot freeze it safely")
        affine.append((a.float(), b.float()))
    pairs = {}
    for factor in range(2):
        variants = []
        for value in (-intervention, intervention):
            styles = [lat[f"z_style_v{v + 1}"].clone() for v in range(2)]
            for z in styles:
                z[factor] = value
            raw = inner.render_pseudo_mri(
                lat["z_content"],
                lat["z_deformation"],
                lat["z_fissure"],
                *styles,
                inner.sample_seed_for(idx),
                z_lesion=lat.get("z_lesion"),
            )[:2]
            variants.append([(x * a + b) * mask for x, (a, b) in zip(raw, affine)])
        pairs[TARGETS[factor]] = variants
    return original, mask, targets, pairs


def validate_model(model):
    if model.nb_levels != 1 or list(model.content_style_levels) != [0]:
        raise ValueError("Exact style audit currently requires one VQ level with content/style at level 0")
    if not model.inject_style_to_decoder:
        raise ValueError("This run does not inject style into its decoder")
    if getattr(model, "mask_mode", "onthefly") not in ("fixed", "learned", "learned_split"):
        raise ValueError("Need stable channel masks; batch-dependent onthefly masks invalidate comparisons")
    if any(m.training for m in model.modules()):
        raise ValueError("Model must be in eval mode (including codebooks)")


@contextmanager
def capture_path(model):
    """Temporary taps at the actual bottleneck and decoder boundary; always removed."""
    captured = {}
    original = model._bottleneck_style
    had_override = "_bottleneck_style" in model.__dict__

    def bottleneck(style):
        if "raw" in captured:
            raise ValueError("Expected one style bottleneck call for the single level")
        captured["raw"] = style.detach().clone()
        pooled = original(style)
        captured["pooled"] = pooled.detach().clone()
        return pooled

    def decoder_tap(module, args, kwargs):
        if "style" not in kwargs or kwargs["style"] is None:
            raise ValueError("Decoder did not receive a style tensor")
        captured["content"] = args[0].detach().clone()
        captured["injected"] = kwargs["style"].detach().clone()

    model._bottleneck_style = bottleneck
    handle = model.decoders[0].register_forward_pre_hook(decoder_tap, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()
        if had_override:
            model._bottleneck_style = original
        else:
            del model._bottleneck_style


def encode(model, images, masks, device):
    x = torch.cat([torch.stack([s[v] for s in images]) for v in range(2)]).to(device)
    mask = torch.cat([torch.stack(masks)] * 2).to(device)
    with torch.inference_mode(), capture_path(model) as path:
        out = model(x, n_views=2, subsets=[(0, 1)], pool_only=True, return_recon=True, mask=mask)
    if set(path) != {"raw", "pooled", "content", "injected"} or path["raw"].shape[1] == 0:
        raise ValueError("Missing or empty style pathway")
    path["output"] = out[0].detach().clone()
    path["ids"] = {k: v.detach().clone() for k, v in model._last_style_id_outputs.items()}
    if not all(torch.isfinite(v).all() for v in path.values() if isinstance(v, torch.Tensor)):
        raise ValueError("Non-finite model output")
    return path


def replay(model, content, style, spatial):
    with torch.inference_mode():
        y = model.decoders[0](content, style=style)
        if tuple(y.shape[2:]) != tuple(spatial):
            y = F.interpolate(y, size=spatial, mode="trilinear", align_corners=False)
    if not torch.isfinite(y).all():
        raise ValueError("Non-finite decoder replay")
    return y


def check_endpoint(reference, replayed, atol=1e-5, rtol=1e-4):
    """RMS criterion tolerates CUDA elementwise noise; error is also scored per subject."""
    error = float((reference.double() - replayed.double()).square().mean().sqrt())
    scale = float(reference.double().square().mean().sqrt())
    if error > atol + rtol * scale:
        raise ValueError(f"Decoder endpoint replay failed: rms={error:.6g}, reference_rms={scale:.6g}")
    return error


def response_metrics(target, effect):
    target, effect = np.asarray(target, float).ravel(), np.asarray(effect, float).ravel()
    energy = np.dot(target, target)
    if energy <= 1e-16:
        return dict(gain=np.nan, cosine=np.nan, relative_error=np.nan)
    return dict(
        gain=float(np.dot(target, effect) / energy),
        cosine=float(np.dot(target, effect) / max(np.sqrt(energy) * np.linalg.norm(effect), 1e-16)),
        relative_error=float(np.linalg.norm(effect - target) / np.sqrt(energy)),
    )


def swap_batch(model, samples, device, factor):
    masks = [s[1] for s in samples]
    low = [s[3][factor][0] for s in samples]
    high = [s[3][factor][1] for s in samples]
    a, b = encode(model, low, masks, device), encode(model, high, masks, device)
    spatial = a["output"].shape[2:]
    aa = replay(model, a["content"], a["injected"], spatial)
    bb = replay(model, b["content"], b["injected"], spatial)
    check_endpoint(a["output"], aa)
    check_endpoint(b["output"], bb)
    ab = replay(model, a["content"], b["injected"], spatial)
    ba = replay(model, b["content"], a["injected"], spatial)
    # Effects average both background settings. Their sum equals the joint effect.
    effects = {
        "joint": bb - aa,
        "style_at_low_content": ab - aa,
        "style_at_high_content": bb - ba,
        "style_mean": ((ab - aa) + (bb - ba)) / 2,
        "content_mean": ((ba - aa) + (bb - ab)) / 2,
    }
    rows = []
    for v, view in enumerate(VIEWS):
        for i in range(len(samples)):
            j = v * len(samples) + i
            mask = masks[i].numpy() > 0
            dx = (high[i][v] - low[i][v]).numpy()[mask]
            input_rms = float(np.sqrt(np.mean(dx.astype(float) ** 2)))
            mask_device = torch.as_tensor(mask, device=aa.device)
            err = max(
                float((a["output"][j] - aa[j])[mask_device].double().square().mean().sqrt()),
                float((b["output"][j] - bb[j])[mask_device].double().square().mean().sqrt()),
            )
            row = dict(
                view=view,
                factor=factor,
                input_rms=input_rms,
                endpoint_rms=err,
                resolved=bool(input_rms > max(1e-8, 10 * err)),
            )
            for name, effect in effects.items():
                row.update(
                    {f"{name}_{k}": val for k, val in response_metrics(dx, effect[j].cpu().numpy()[mask]).items()}
                )
            interaction = (bb[j] - ba[j] - ab[j] + aa[j]).cpu().numpy()[mask]
            row["interaction_rms_ratio"] = (
                float(np.sqrt(np.mean(interaction**2)) / input_rms) if input_rms > 1e-8 else np.nan
            )
            for stage in ("raw", "pooled", "injected", "content"):
                row[f"{stage}_change_rms"] = float((b[stage][j] - a[stage][j]).double().square().mean().sqrt())
            if a["ids"]:
                row["style_code_changed_fraction"] = float((a["ids"][0][j] != b["ids"][0][j]).float().mean())
            rows.append(row)
    return rows


def r2(y, prediction):
    denom = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - prediction) ** 2) / denom) if denom > 1e-12 else np.nan


def fit_probe(x, y, split, seed=0):
    """Dual linear ridge: feature standardization is fit without test subjects.

    Hyperparameter selection is per target on validation subjects, followed by a
    train+validation refit. A separately tuned shuffled-label fit is the null.
    Constant features predict the training mean. No dimensionality reduction.
    """
    from scipy.linalg import cho_factor, cho_solve

    train, val, test = split
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    alphas = np.logspace(-6, 2, 9)

    def kernels(fit, evaluate):
        a, b = x[fit], x[evaluate]
        mu, sd = a.mean(0), a.std(0)
        keep = sd > 1e-10
        if not keep.any():
            return np.zeros((len(fit), len(fit))), np.zeros((len(evaluate), len(fit))), 0
        a = (a[:, keep] - mu[keep]) / sd[keep]
        b = (b[:, keep] - mu[keep]) / sd[keep]
        return a @ a.T / keep.sum(), b @ a.T / keep.sum(), int(keep.sum())

    kt, kv, _ = kernels(train, val)
    fit = np.concatenate([train, val])
    kf, ke, varying = kernels(fit, test)
    shuffled = np.random.default_rng(seed).permutation(y[fit])
    null_y = y.copy()
    null_y[fit] = shuffled
    result = {"dimensions": x.shape[1], "varying_fit_columns": varying}
    predictions = {}
    for name, labels in (("true", y), ("shuffled", null_y)):
        mean = labels[train].mean()
        candidates = [
            kv @ cho_solve(cho_factor(kt + alpha * np.eye(len(train))), labels[train] - mean) + mean for alpha in alphas
        ]
        best = int(np.argmin([np.mean((labels[val] - pred) ** 2) for pred in candidates]))
        mean = labels[fit].mean()
        pred = ke @ cho_solve(cho_factor(kf + alphas[best] * np.eye(len(fit))), labels[fit] - mean) + mean
        result[f"{name}_r2"] = r2(y[test], pred)
        result[f"{name}_alpha"] = float(alphas[best])
        predictions[name] = pred
    return result, predictions


def collect_features(model, ds, device, batch_size):
    features, targets, ids = {}, [], {v: [] for v in VIEWS}
    shapes = {}
    for start in range(0, len(ds), batch_size):
        samples = [render_sample(ds, i) for i in range(start, min(start + batch_size, len(ds)))]
        path = encode(model, [s[0] for s in samples], [s[1] for s in samples], device)
        stages = {k: path[k] for k in ("raw", "pooled", "injected")}
        stages["raw_gap"] = path["raw"].mean((2, 3, 4))
        for stage, tensor in stages.items():
            shapes[stage] = list(tensor.shape[1:])
            for v, view in enumerate(VIEWS):
                key = f"{view}/{stage}"
                values = tensor[v * len(samples) : (v + 1) * len(samples)].flatten(1).cpu().numpy().copy()
                features.setdefault(key, []).append(values)
        for v, view in enumerate(VIEWS):
            if path["ids"]:
                ids[view].append(
                    path["ids"][0][v * len(samples) : (v + 1) * len(samples)].cpu().numpy().reshape(len(samples), -1)
                )
        targets.extend(s[2] for s in samples)
        LOG.info("Captured style stages for %d/%d subjects", start + len(samples), len(ds))
    features = {k: np.concatenate(v) for k, v in features.items()}
    usage = {}
    for view, parts in ids.items():
        if parts:
            codes = np.concatenate(parts)
            _, counts = np.unique(codes, return_counts=True)
            p = counts / counts.sum()
            usage[view] = dict(
                unique_codes=len(counts),
                unique_subject_code_patterns=len(np.unique(codes, axis=0)),
                code_perplexity=float(np.exp(-(p * np.log(p)).sum())),
                assignments_per_subject=codes.shape[1],
            )
    return features, np.stack(targets), shapes, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-samples", type=int, default=256, help="Natural subjects for frozen probes")
    parser.add_argument("--swap-samples", type=int, default=64, help="Additional, disjoint intervention subjects")
    parser.add_argument("--batch-size", type=int, default=4, help="Subjects per batch, each with two views")
    parser.add_argument("--intervention", type=float, default=0.5, help="Set gain/bias latent to +/- this value (0,1]")
    parser.add_argument("--causal", choices=("iid", "match"), default="iid")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--out-dir")
    cli = parser.parse_args()
    if (
        cli.num_samples < 40
        or cli.swap_samples < 1
        or cli.batch_size < 1
        or cli.cpu_threads < 1
        or not 0 < cli.intervention <= 1
    ):
        parser.error("Need >=40 probe subjects, positive swap samples/batch/threads and intervention in (0,1]")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.cpu_threads)
    from eval.run_dci_synthetic import load_model_from_run_dir

    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device=cli.device, seed=cli.seed)
    validate_model(model)
    LOG.info(
        "Style spatial size=%s; quantize_style=%s; normalization=%s",
        model.style_spatial_size,
        model.quantize_style,
        args.synthetic_normalize,
    )
    ds = make_dataset(args, cli.num_samples, cli.causal, cli.split)
    features, targets, shapes, usage = collect_features(model, ds, device, cli.batch_size)
    order = np.random.default_rng(cli.seed).permutation(cli.num_samples)
    split = np.split(order, [int(0.6 * cli.num_samples), int(0.8 * cli.num_samples)])
    probes, predictions, dispersion = {}, {}, {}
    for key, x in features.items():
        view, stage = key.split("/")
        vi = VIEWS.index(view)
        sd = x.astype(float).std(0)
        dispersion[key] = dict(
            mean_subject_std=float(sd.mean()), max_subject_std=float(sd.max()), varying_columns=int((sd > 1e-10).sum())
        )
        for t, target in enumerate(TARGETS):
            result, pred = fit_probe(x, targets[:, vi, t], split, seed=cli.seed + t)
            name = f"{key}/{target}"
            probes[name] = result
            predictions[name + "/truth"] = targets[split[2], vi, t]
            predictions.update({name + "/" + k: v for k, v in pred.items()})
        LOG.info("Probed %s (%d descriptors)", key, x.shape[1])
    del features
    swap_ds = make_dataset(args, cli.num_samples + cli.swap_samples, cli.causal, cli.split)
    rows = []
    for start in range(cli.num_samples, len(swap_ds), cli.batch_size):
        indices = list(range(start, min(start + cli.batch_size, len(swap_ds))))
        samples = [render_sample(swap_ds, i, cli.intervention) for i in indices]
        for factor in TARGETS[:2]:
            batch = swap_batch(model, samples, device, factor)
            for j, row in enumerate(batch):
                row["subject"] = indices[j % len(indices)]
            rows.extend(batch)
        LOG.info("Swapped %d/%d subjects", indices[-1] + 1 - cli.num_samples, cli.swap_samples)
    swaps = {}
    for view in VIEWS:
        for factor in TARGETS[:2]:
            group = [r for r in rows if r["view"] == view and r["factor"] == factor]
            valid = [r for r in group if r["resolved"]]
            metrics = (
                {
                    k: float(np.nanmedian([r[k] for r in valid]))
                    for k in group[0]
                    if k not in ("view", "factor", "resolved", "subject")
                }
                if valid
                else {}
            )
            swaps[f"{view}/{factor}"] = dict(n=len(group), n_resolved=len(valid), medians=metrics)
    directory = Path(
        cli.out_dir or Path(cli.run_dir) / ("style_path_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(
        arguments=vars(cli),
        run_settings=vars(args),
        stage_shapes=shapes,
        code_usage=usage,
        dispersion=dispersion,
        probes=probes,
        swaps=swaps,
        subject_split=[s.tolist() for s in split],
        definitions={
            "injected": "Actual decoder style, post-quantization when enabled; otherwise pooled style",
            "probe": "Held-out raw R2, train-only standardized linear ridge; tuned shuffled-label control",
            "targets": "Effective renderer gain, bias, nonnegative noise sigma",
            "swaps": "Median per-subject effects versus rendered high-minus-low image, within brain mask",
            "resolved": "Input RMS > max(1e-8, 10 * endpoint replay RMS), both within brain mask",
            "limitations": "Linear probes can miss nonlinear codes. Frozen-affine variants may be OOD. Read joint fidelity before routing; no claim about training causality.",
        },
    )
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(directory / "probe_predictions.npz", **predictions)
    with (directory / "swaps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nFrozen style-target probes: held-out R2 (shuffled-label R2)")
    for key in probes:
        m = probes[key]
        print(f"{key:35s} {m['true_r2']:+.3f} ({m['shuffled_r2']:+.3f})  d={m['dimensions']}")
    print("\nControlled decoder response: gain=1 reproduces input effect; gain=0 is no aligned effect")
    for key, group in swaps.items():
        m = group["medians"]
        print(
            f"{key:12s} resolved={group['n_resolved']}/{group['n']} "
            f"joint={m.get('joint_gain', np.nan):+.3f} style={m.get('style_mean_gain', np.nan):+.3f} "
            f"content={m.get('content_mean_gain', np.nan):+.3f} "
            f"joint_cos={m.get('joint_cosine', np.nan):.3f} joint_relerr={m.get('joint_relative_error', np.nan):.3f}"
        )
    print("Style-code usage:", json.dumps(usage))
    print("Read joint cosine/error and endpoint error before attributing a pathway; low probe R2 is inconclusive.")
    print(f"Saved {directory}\nNo representation training or checkpoint update.")


if __name__ == "__main__":
    main()
