"""Frozen ventricular interventions and exact content/style decoder swaps.

See VENTRICLE_DECODER_AUDIT.md. No training or anatomical supervision is added.
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

from eval.content_path_probe import state_digest
from eval.lesion_reconstruction import json_safe, make_dataset
from eval.style_path_audit import check_endpoint, encode, replay, validate_model
from eval.ventricle_quantizer_audit import render_pair

LOG = logging.getLogger(__name__)
VIEWS = ("t1", "flair")


def effects_from_endpoints(ll, lh, hl, hh):
    """First letter is content, second style; average effects over both contexts."""
    return {
        "joint": hh - ll,
        "content_at_low_style": hl - ll,
        "content_at_high_style": hh - lh,
        "style_at_low_content": lh - ll,
        "style_at_high_content": hh - hl,
        "content_mean": ((hl - ll) + (hh - lh)) / 2,
        "style_mean": ((lh - ll) + (hh - hl)) / 2,
        "interaction": hh - hl - lh + ll,
    }


def rms(x):
    return float(x.double().square().mean().sqrt()) if x.numel() else None


def score_response(target, effects, endpoint_error, repeat_error, region):
    """Resolve effects against numerical error, without imposing a fidelity cutoff.

    Outside the changed input support, only absolute RMS is meaningful. Missing
    or unresolvable denominators produce None, never a favorable zero score.
    """
    dx = target[region].double()
    noise = max(rms(endpoint_error[region]) or 0, rms(repeat_error[region]) or 0)
    signal = rms(dx)
    measurable = signal is not None and signal > 1e-8
    resolved = measurable and signal > 10 * noise
    row = {
        "voxels": int(region.sum()),
        "input_rms": signal,
        "measurable_input": measurable,
        "resolved_input": resolved,
        "endpoint_replay_rms": rms(endpoint_error[region]),
        "repeat_forward_rms": rms(repeat_error[region]),
        "numerical_error_to_input_ratio": noise / signal if measurable else None,
    }
    for name, effect in effects.items():
        dy = effect[region].double()
        size = rms(dy)
        row[f"{name}_rms"] = size
        row[f"{name}_rms_ratio"] = size / signal if resolved else None
        row[f"{name}_gain"] = float((dx * dy).sum() / dx.square().sum()) if resolved else None
        row[f"{name}_relative_error"] = rms(dy - dx) / signal if resolved else None
        row[f"{name}_cosine"] = (
            float((dx * dy).sum() / (dx.norm() * dy.norm())) if resolved and size > max(1e-8, 10 * noise) else None
        )
    row["attribution_sum_error_rms"] = rms((effects["content_mean"] + effects["style_mean"] - effects["joint"])[region])
    return row


def swap_batch(model, samples, device):
    masks = [s["mask"] for s in samples]
    low = encode(model, [s["low"] for s in samples], masks, device)
    high = encode(model, [s["high"] for s in samples], masks, device)
    repeated = encode(model, [s["low"] for s in samples], masks, device)
    spatial = low["output"].shape[2:]
    decoded = {}
    for key, c, s in (("ll", low, low), ("lh", low, high), ("hl", high, low), ("hh", high, high)):
        decoded[key] = replay(model, c["content"], s["injected"], spatial).cpu()
    # Check each subject/view separately; a batch mean must not hide one bad replay.
    for i in range(len(samples) * 2):
        check_endpoint(low["output"][i].cpu(), decoded["ll"][i])
        check_endpoint(high["output"][i].cpu(), decoded["hh"][i])
    # Pointwise upper envelopes retain errors localized around the ventricles.
    endpoint_error = torch.maximum(
        (low["output"].cpu() - decoded["ll"]).abs(),
        (high["output"].cpu() - decoded["hh"]).abs(),
    )
    repeat_error = (repeated["output"].cpu() - low["output"].cpu()).abs()
    return decoded, endpoint_error, repeat_error


def reconstruction_row(prediction, target, region):
    error = (prediction - target)[region].double()
    return {
        "voxels": int(region.sum()),
        "mae": float(error.abs().mean()) if error.numel() else None,
        "rmse": rms(error),
    }


def audit(model, ds, device, epsilons, batch_size=4, examples=2):
    validate_model(model)
    if model.mask_mode not in ("fixed", "learned"):
        raise ValueError("Ventricle swaps require fixed or hard learned channel masks")
    rows, reconstruction, saved = [], [], {}
    for start in range(0, len(ds), batch_size):
        indices = list(range(start, min(start + batch_size, len(ds))))
        natural, masks = [], []
        for idx in indices:
            a, b, lat = ds._inner[idx]
            mask = lat["brain_mask"]
            natural.append(list(ds.normalize_views(a, b, mask, mask)))
            masks.append(mask)
        original = encode(model, natural, masks, device)["output"].cpu()
        for eps in epsilons:
            samples = [render_pair(ds, idx, eps) for idx in indices]
            decoded, endpoint_error, repeat_error = swap_batch(model, samples, device)
            effects = effects_from_endpoints(**decoded)
            for v, view in enumerate(VIEWS):
                for i, (idx, sample) in enumerate(zip(indices, samples)):
                    j = v * len(samples) + i
                    regions = {
                        "brain": sample["mask"].bool(),
                        "affected": sample["support"].bool(),
                        "outside": sample["mask"].bool() & ~sample["support"].bool(),
                    }
                    dx = sample["high"][v] - sample["low"][v]
                    brain_energy = float(effects["joint"][j][regions["brain"]].double().square().sum())
                    for region_name, region in regions.items():
                        row = dict(
                            index=idx, seed=ds._inner.sample_seed_for(idx), eps=eps, view=view, region=region_name
                        )
                        row.update(
                            score_response(
                                dx, {k: x[j] for k, x in effects.items()}, endpoint_error[j], repeat_error[j], region
                            )
                        )
                        row.update(
                            {
                                k: sample[k]
                                for k in (
                                    "z1_low",
                                    "z1_high",
                                    "changed_tissue_voxels",
                                    "foreground_mask_changed_voxels",
                                )
                            }
                        )
                        row["joint_brain_energy_fraction"] = (
                            float(effects["joint"][j][region].double().square().sum()) / brain_energy
                            if brain_energy > 1e-16
                            else None
                        )
                        rows.append(row)
                        for endpoint, pred, target in (
                            ("natural", original[j], natural[i][v]),
                            ("low", decoded["ll"][j], sample["low"][v]),
                            ("high", decoded["hh"][j], sample["high"][v]),
                        ):
                            reconstruction.append(
                                dict(
                                    index=idx,
                                    eps=eps,
                                    view=view,
                                    region=region_name,
                                    endpoint=endpoint,
                                    **reconstruction_row(pred, target, region),
                                )
                            )
                    if idx < examples:
                        prefix = f"eps{eps:g}_sample{idx}_{view}"
                        maps = dict(
                            natural_input=natural[i][v],
                            natural_recon=original[j],
                            input_low=sample["low"][v],
                            input_high=sample["high"][v],
                            support=sample["support"],
                            **{k: x[j] for k, x in decoded.items()},
                            **{k: x[j] for k, x in effects.items()},
                        )
                        saved[prefix] = {k: x.numpy() for k, x in maps.items()}
            LOG.info("eps=%g: audited %d/%d subjects", eps, start + len(indices), len(ds))
    return rows, reconstruction, saved


def estimate(values):
    x = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return dict(n=0, mean=None, median=None, ci95=None)
    draws = np.random.default_rng(0).choice(x, size=(500, len(x)), replace=True).mean(1)
    return dict(
        n=len(x),
        mean=float(x.mean()),
        median=float(np.median(x)),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist() if len(x) > 1 else None,
    )


def summarize(rows, reconstruction):
    summaries = []
    metrics = [k for k in rows[0] if k.endswith(("_gain", "_cosine", "_relative_error", "_rms", "_rms_ratio"))]
    metrics += ["numerical_error_to_input_ratio", "joint_brain_energy_fraction"]
    for eps, view, region in sorted({(r["eps"], r["view"], r["region"]) for r in rows}):
        group = [r for r in rows if (r["eps"], r["view"], r["region"]) == (eps, view, region)]
        rec = [r for r in reconstruction if (r["eps"], r["view"], r["region"]) == (eps, view, region)]
        summaries.append(
            dict(
                eps=eps,
                view=view,
                region=region,
                n=len(group),
                n_measurable=sum(r["measurable_input"] for r in group),
                n_resolved=sum(r["resolved_input"] for r in group),
                metrics={k: estimate([r[k] for r in group]) for k in metrics},
                reconstruction={
                    endpoint: {k: estimate([r[k] for r in rec if r["endpoint"] == endpoint]) for k in ("mae", "rmse")}
                    for endpoint in ("natural", "low", "high")
                },
            )
        )
    return summaries


def save_panels(saved, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for prefix, maps in saved.items():
        support = maps["support"][0]
        z = int(support.sum((0, 1)).argmax()) if support.any() else support.shape[-1] // 2
        inputs = [
            maps[k][0, :, :, z]
            for k in ("natural_input", "natural_recon", "input_low", "input_high", "ll", "lh", "hl", "hh")
        ]
        delta = maps["input_high"] - maps["input_low"]
        responses = [x[0, :, :, z] for x in (delta, maps["joint"], maps["content_mean"], maps["style_mean"])]
        lo, hi = min(x.min() for x in inputs), max(x.max() for x in inputs)
        limit = max(max(abs(x).max() for x in responses), 1e-8)
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
        titles = (
            "Natural input",
            "Natural reconstruction",
            "Input low",
            "Input high",
            "Decode low/low",
            "Low content / high style",
            "High content / low style",
            "Decode high/high",
            "Input change",
            "Joint change",
            "Content mean change",
            "Style mean change",
        )
        for k, (ax, data, title) in enumerate(zip(axes.flat, inputs + responses, titles)):
            ax.imshow(
                data.T,
                origin="lower",
                cmap="gray" if k < 8 else "coolwarm",
                vmin=lo if k < 8 else -limit,
                vmax=hi if k < 8 else limit,
            )
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{prefix}, z={z}; shared image scale; shared change scale")
        fig.savefig(output / f"{prefix}.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.25])
    parser.add_argument("--causal", choices=("match", "iid"), default="match")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    cli = parser.parse_args(argv)
    if min(cli.num_samples, cli.batch_size, cli.threads) < 1 or cli.examples < 0:
        parser.error("Samples, batch size and threads must be positive; examples nonnegative")
    if any(not np.isfinite(e) or e <= 0 for e in cli.eps):
        parser.error("eps must be positive and finite")
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
    before = state_digest(model)
    ds = make_dataset(args, cli.num_samples, cli.causal, "test")
    rows, reconstruction, saved = audit(model, ds, device, cli.eps, cli.batch_size, cli.examples)
    if state_digest(model) != before:
        raise ValueError("Registered model state changed; refusing to report a frozen audit")
    output = Path(cli.output_dir or Path(cli.run_dir) / f"ventricle_decoder_audit_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows, reconstruction)
    report = dict(
        config=vars(cli),
        dataset_settings=vars(args),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_step=step,
        model_state_unchanged=True,
        state_sha256=before,
        summary=summary,
        notes=[
            "Only z_content[1] changes; other realized latents, appearance and render seeds stay fixed.",
            "The natural sample's normalization affine and brain mask are held fixed. Variants can be off-distribution.",
            "Affected region is changed tissue dilated for renderer blur, not an encoder receptive field.",
            "Resolved means input exceeds numerical error, not that ventricular reconstruction is faithful.",
            "Content/style mean effects average both contexts and sum to the joint effect; they are not information fractions.",
            "Cosine is undefined for unresolved or zero output responses; no response has gain 0 and relative error 1.",
            "Reconstruction and absolute RMS summaries include all nonempty regions; normalized effects require resolved input.",
            "CIs bootstrap subjects within this checkpoint; they do not represent training-seed uncertainty.",
        ],
    )
    (output / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    for filename, records in (("responses.csv", rows), ("reconstruction.csv", reconstruction)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if saved:
        np.savez_compressed(
            output / "examples.npz", **{f"{p}_{k}": x for p, maps in saved.items() for k, x in maps.items()}
        )
        save_panels(saved, output)
    print("\nPaired mean responses (identity gain=1, no response=0)")
    print("eps  view   region    measurable/resolved   joint gain/cos/error    content/style gain   natural MAE")
    fmt = lambda x: "n/a" if x is None else f"{x:.4g}"
    for s in summary:
        m = s["metrics"]
        get = lambda k: fmt(m[k]["mean"])
        print(
            f"{s['eps']:<4g} {s['view']:6s} {s['region']:9s} {s['n_measurable']:3d}/{s['n_resolved']:<3d} of {s['n']:<3d}  "
            f"{get('joint_gain')}/{get('joint_cosine')}/{get('joint_relative_error')}    "
            f"{get('content_mean_gain')}/{get('style_mean_gain')}    {fmt(s['reconstruction']['natural']['mae']['mean'])}"
        )
    print("\nReconstruction and response controls (paired means)")
    print(
        "eps  view   region    low/high MAE    style RMS/input   interaction/input   numerical/input   joint energy fraction"
    )
    for s in summary:
        rec, metrics = s["reconstruction"], s["metrics"]
        print(
            f"{s['eps']:<4g} {s['view']:6s} {s['region']:9s} "
            f"{fmt(rec['low']['mae']['mean'])}/{fmt(rec['high']['mae']['mean'])}    "
            f"{fmt(metrics['style_mean_rms_ratio']['mean'])}    "
            f"{fmt(metrics['interaction_rms_ratio']['mean'])}    "
            f"{fmt(metrics['numerical_error_to_input_ratio']['mean'])}    "
            f"{fmt(metrics['joint_brain_energy_fraction']['mean'])}"
        )
    print(
        "Check joint fidelity and endpoint reconstructions before assigning a pathway.\n"
        "Outside-region absolute responses, replay errors and confidence intervals are saved in the reports."
    )
    print(f"Saved {output}\nNo registered model parameter or buffer changed.")


if __name__ == "__main__":
    main()
