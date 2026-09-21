"""Route a lesion-on/off intervention through content and style separately.

A = lesion absent; B = lesion present. Decode AA, BA, AB, BB (content donor
first) within each modality. All other factors and the original normalization
affine stay fixed. Uses the ventricular diagnostic's actual decoder tensors,
endpoint replay checks and factorial scoring; no probes or training updates.
See LESION_ROUTING.md for usage and interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from eval.lesion_reconstruction import json_safe, locate, make_dataset, render_pair
from eval.reconstruction_attribution import frozen_checkpoint
from eval.ventricle_routing import (
    REPLAY_RMS_ATOL,
    REPLAY_RMS_RTOL,
    REPLAY_SIGNAL_FRACTION,
    decode_swaps,
    score_swaps,
    stable_replay_math,
)

logger = logging.getLogger(__name__)
VIEWS = ("t1", "flair")
DISPLAY = (
    "joint_gain",
    "joint_cosine",
    "joint_relative_error",
    "content_at_style_a_gain",
    "content_at_style_b_gain",
    "style_at_content_a_gain",
    "style_at_content_b_gain",
    "content_mean_gain",
    "style_mean_gain",
    "interaction_rms_ratio",
    "endpoint_replay_rms",
    "endpoint_error_to_input_ratio",
)


def lesion_pair(ds, index):
    on, off, mask, support, centroid, latent = render_pair(ds, index)
    return {"a": off, "b": on, "mask": mask, "support": support, "index": index, "centroid": centroid, "latent": latent}


def response_images(xa, xb, y):
    """Full spatial effects; donor convention matches score_swaps exactly."""
    effects = {
        "input": xb - xa,
        "joint": y["bb"] - y["aa"],
        "content_at_style_a": y["ba"] - y["aa"],
        "content_at_style_b": y["bb"] - y["ab"],
        "style_at_content_a": y["ab"] - y["aa"],
        "style_at_content_b": y["bb"] - y["ba"],
    }
    effects["content_mean"] = (effects["content_at_style_a"] + effects["content_at_style_b"]) / 2
    effects["style_mean"] = (effects["style_at_content_a"] + effects["style_at_content_b"]) / 2
    return effects


def summarize(rows, draws=500, seed=0):
    """Mean and median on the same resolved subjects; no automatic route verdict."""
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive.")
    result = {}
    excluded = {"index", "modality", "valid_input", "valid_routing", "endpoint_signal_resolved"}
    for view in VIEWS:
        selected = [r for r in rows if r["modality"] == view]
        metrics = {}
        if not selected:
            raise ValueError(f"No rows for {view}.")
        for key in selected[0]:
            if key in excluded:
                continue
            route = key.endswith(("_gain", "_rms_ratio", "_location_error_vox")) or key in (
                "joint_cosine",
                "joint_relative_error",
            )
            values = np.asarray([r[key] for r in selected if not route or r["valid_routing"]], dtype=float)
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if len(values) else None
            metric = {"n_valid": len(values), "mean": mean, "median": float(np.median(values)) if len(values) else None}
            if key in DISPLAY and len(values):
                # Same seed gives paired resamples for metrics with identical coverage.
                rng = np.random.default_rng(seed)
                means = [float(values[rng.integers(len(values), size=len(values))].mean()) for _ in range(draws)]
                low, high = np.quantile(means, [0.025, 0.975])
                metric.update(mean_ci_low=float(low), mean_ci_high=float(high))
            metrics[key] = metric
        result[view] = {
            "n": len(selected),
            "n_valid_input": sum(r["valid_input"] for r in selected),
            "n_valid_routing": sum(r["valid_routing"] for r in selected),
            "n_empty_lesions": sum(r["lesion_voxels"] == 0 for r in selected),
            "metrics": metrics,
        }
    return result


def audit(model, ds, device, batch_size=2, examples=3, draws=500):
    if batch_size < 1 or len(ds) < 1 or examples < 0:
        raise ValueError("Need positive batch/sample counts and nonnegative examples.")
    radius = ds._inner.renderer.lesion_radius * (ds.res - 1) / 2
    rows, panels = [], []
    for start in range(0, len(ds), batch_size):
        samples = [lesion_pair(ds, i) for i in range(start, min(start + batch_size, len(ds)))]
        decoded, diagnostics = decode_swaps(model, samples, device)
        for b, sample in enumerate(samples):
            foreground = sample["mask"].numpy()[0] > 0
            for view, modality in enumerate(VIEWS):
                xa, xb = [sample[state][view].numpy()[0] for state in ("a", "b")]
                ys = {key: images[b] for key, images in decoded[view].items()}
                row = {"index": sample["index"], "modality": modality, "lesion_voxels": int(sample["support"].sum())}
                row.update(score_swaps(xa, xb, ys, sample["support"], foreground))
                row.update({key: float(value[b]) for key, value in diagnostics[view].items()})
                row["valid_routing"] = bool(row["valid_input"] and row["endpoint_signal_resolved"])
                effects = response_images(xa, xb, ys)
                for name, effect in effects.items():
                    # Search the whole foreground, not the target ROI. A zero response
                    # stays missing; weak but nonzero response locations need gain context.
                    location = locate(effect * foreground, radius, response=True)
                    row[f"{name}_location_error_vox"] = float(np.linalg.norm(location - sample["centroid"]))
                rows.append(row)
                if sample["index"] < examples and sample["support"].any():
                    z = int(np.argmax(sample["support"].sum((0, 1))))
                    names = (
                        "input",
                        "content_at_style_a",
                        "content_at_style_b",
                        "style_at_content_a",
                        "style_at_content_b",
                        "joint",
                    )
                    panels.append(
                        {
                            "index": sample["index"],
                            "view": modality,
                            "slice": z,
                            "centroid": sample["centroid"],
                            "images": [(effects[name] * foreground)[:, :, z] for name in names],
                        }
                    )
        logger.info("Scored %d/%d lesion pairs", min(start + batch_size, len(ds)), len(ds))
    return rows, summarize(rows, draws), panels


def save_panels(panels, path):
    if not panels:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = (
        "Input on − off",
        "Content | style OFF",
        "Content | style ON",
        "Style | content OFF",
        "Style | content ON",
        "Joint on − off",
    )
    fig, axes = plt.subplots(len(panels), 6, figsize=(16, 2.6 * len(panels)), squeeze=False)
    for axes_row, panel in zip(axes, panels):
        scale = max(float(np.max(np.abs(panel["images"]))), 1e-8)
        for axis, image, title in zip(axes_row, panel["images"], titles):
            axis.imshow(image.T, origin="lower", cmap="coolwarm", vmin=-scale, vmax=scale)
            axis.plot(panel["centroid"][0], panel["centroid"][1], "+", color="lime", markersize=7)
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        axes_row[0].set_ylabel(f"{panel['view']} #{panel['index']}\nz={panel['slice']}")
    fig.suptitle("Lesion routing: signed responses share one scale per row; cross marks rendered centroid")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def validate(args):
    if int(getattr(args, "vqvae_nb_levels", 1)) != 1:
        raise ValueError("This lesion routing test supports one VQ level; multilevel conditioning can mix pathways.")
    if getattr(args, "mask_mode", None) != "fixed":
        raise ValueError("Use a fixed content/style split for comparable donor swaps.")
    if not getattr(args, "inject_style_to_decoder", False):
        raise ValueError("The checkpoint needs an injected style pathway.")
    if getattr(args, "split_encoder_norm", False):
        raise ValueError("The shared loader cannot restore split_encoder_norm faithfully.")
    if getattr(args, "contrastive_only", False):
        raise ValueError("This checkpoint did not train its decoder; reconstruction routing is not meaningful.")
    if getattr(args, "synthetic_lesion_mode", "sphere") != "sphere":
        raise ValueError("This diagnostic uses sphere-lesion on/off pairs, not field lesions.")


def run(cli):
    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir, load_run_args

    if min(cli.num_samples, cli.batch_size, cli.cpu_threads, cli.bootstrap) < 1 or cli.examples < 0:
        raise ValueError("Need positive sample/batch/thread/bootstrap counts and nonnegative examples.")
    args = load_run_args(cli.run_dir)
    validate(args)
    directory = (
        Path(cli.out_dir)
        if cli.out_dir
        else Path(cli.run_dir) / datetime.now().strftime("lesion_routing_%Y%m%d_%H%M%S_%f")
    )
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")
    checkpoint = Path(cli.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(cli.run_dir) / checkpoint
    checkpoint = checkpoint.resolve()
    torch.set_num_threads(cli.cpu_threads)
    model, args, device = load_model_from_run_dir(cli.run_dir, str(checkpoint), device=cli.device, seed=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = state.get("step")
    weights = state.get("encoders", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in weights.items()}, strict=True)
    del state, weights
    ds = make_dataset(args, cli.num_samples, cli.causal, cli.split)
    with stable_replay_math(), frozen_checkpoint(model):
        rows, summary, panels = audit(model, ds, device, cli.batch_size, cli.examples, cli.bootstrap)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "arguments": vars(cli),
        "run_settings": vars(args),
        "render_resolution": ds.res,
        "summary": summary,
        "protocol": __doc__,
        "donors": {"a": "lesion absent", "b": "lesion present"},
        "normalization": "Original lesion-on foreground affine frozen for both donors",
        "replay_validation": {
            "rms_atol": REPLAY_RMS_ATOL,
            "rms_rtol": REPLAY_RMS_RTOL,
            "max_error_to_input_ratio": REPLAY_SIGNAL_FRACTION,
        },
        "interpretation": [
            "Read joint gain/cosine/relative error before assigning a pathway; no automatic exclusivity verdict.",
            "Content-only and style-only changes are measured in BOTH donor contexts; inspect interactions.",
            "Mean content and style gains add to joint gain per subject and in their means over identical subjects, not necessarily in medians.",
            "Gain projects an output change onto the local input change; it is not a percentage of encoded information.",
            "Unresolved replay or invisible-input rows are excluded from routing summaries and retained in samples.csv.",
            "Bootstrap intervals are conditional on this checkpoint and resolved subjects, not training-seed uncertainty.",
            "Latent RMS values depend on scale/width; they are sensitivity diagnostics, not information scores.",
            "On/off and hybrid inputs can be outside the training distribution; this measures local decoder reliance, not ordinary-image probe accuracy.",
            "No anatomical labels train the model, no optimizer steps occur, and no checkpoint is written.",
        ],
    }
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    with (directory / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_panels(panels, directory / "responses.png")
    print("\nLesion routing: A=OFF, B=ON. Gains are relative to the input lesion response.")
    for view, result in summary.items():
        print(
            f"\n{view}: measurable input {result['n_valid_input']}/{result['n']}; replay-resolved {result['n_valid_routing']}/{result['n']}"
        )
        for name in DISPLAY:
            metric = result["metrics"][name]
            if metric["mean"] is None:
                print(f"  {name}: undefined (n=0)")
            else:
                print(
                    f"  {name}: mean={metric['mean']:.5g} [{metric['mean_ci_low']:.5g}, {metric['mean_ci_high']:.5g}] "
                    f"median={metric['median']:.5g} (n={metric['n_valid']})"
                )
    print(f"\nCheck joint fidelity and donor-context effects before assigning a route.\nSaved {directory}")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2, help="Subjects; forward holds four volumes per subject.")
    p.add_argument("--causal", choices=["iid", "match"], default="iid")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--examples", type=int, default=3, help="Subjects shown in both modalities; 0 disables plots.")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--out-dir")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(parser().parse_args())
