r"""Ventricle-specific feature alignment and balanced Barlow Twins mismatches.

    python -m eval.ventricle_alignment --run-dir /path/to/run \
        --num-samples 64 --batch-size 8 --loss-batch-size 64 --eps 0.25 --causal match

Delta h is ventricle-high minus ventricle-low. The mismatch swaps FLAIR endpoints
WITHIN each subject, preserving both views' complete feature multisets. A separate
whole-subject shuffle provides context for the factor-specific loss difference.
See VENTRICLE_ALIGNMENT.md for interpretation and EMA/batch-size limitations.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from eval.lesion_alignment import (
    bt_terms,
    extract_stages,
    foreground_positions,
    load_model,
    pair_metrics,
    response_rows,
    write_csv,
)
from eval.lesion_reconstruction import json_safe
from eval.ventricle_routing import make_dataset, render_pair, stable_replay_math, state_digest, verify_checkpoint

LOG = logging.getLogger(__name__)


def balanced_features(low, high):
    """Every condition has the same endpoint multiset in each modality."""
    if low.shape != high.shape or low.ndim not in (3, 4) or low.shape[0] != 2 or low.shape[1] < 2:
        raise ValueError("Balanced comparison needs matching (2, N>=2, C[, P]) tensors")
    matched = torch.cat((low, high), dim=1)
    factor = matched.clone()
    factor[1] = torch.cat((high[1], low[1]), dim=0)
    subject = matched.clone()
    subject[1] = torch.cat((low[1].roll(1, dims=0), high[1].roll(1, dims=0)), dim=0)
    return {"matched": matched, "ventricle_mismatched": factor, "subject_mismatched": subject}


def pairing_losses(low, high, args, stage, level):
    """Actual BT terms; EMA conditions each start from the same cloned reference.

    The optional reference is the hypothetical settled correlation of this
    matched batch, NOT the unrecorded historical training EMA.
    """
    conditions = balanced_features(low, high)
    decay = float(getattr(args, "bt_corr_ema", 0) or 0)
    if not 0 <= decay < 1:
        raise ValueError("bt_corr_ema must be in [0,1)")
    references = [("instantaneous", None, 0.0)]
    if decay:
        reference = {}
        bt_terms(conditions["matched"], args, stage, level, corr_ema=reference, corr_ema_decay=decay)
        for value in reference.values():
            value["c"] = value["c"] / (1 - decay)
            value["t"] = max(1000, int(np.ceil(40 / (1 - decay))))
        references.append(("ema_matched_reference", reference, decay))
    records = []
    for mode, reference, momentum in references:
        terms = {
            name: bt_terms(features, args, stage, level, corr_ema=copy.deepcopy(reference), corr_ema_decay=momentum)
            for name, features in conditions.items()
        }
        row = {"mode": mode, "ema_decay": momentum, "n_subjects": low.shape[1], "loss_rows": 2 * low.shape[1]}
        for name, values in terms.items():
            for key, value in values.items():
                row[f"{name}_{key}"] = value
        for name in ("ventricle_mismatched", "subject_mismatched"):
            for key, value in terms[name].items():
                if isinstance(value, (int, float)):
                    row[f"{name}_minus_matched_{key}"] = value - terms["matched"][key]
        records.append(row)
    return records


def render_sample(ds, idx, eps):
    pair = render_pair(ds, idx, eps)
    inputs = {}
    measurable = []
    for view, name in enumerate(("t1", "flair")):
        delta = (pair["b"][view] - pair["a"][view])[pair["mask"].bool()].double()
        value = float(delta.square().mean().sqrt()) if delta.numel() else float("nan")
        inputs[f"input_{name}_rms"] = value
        measurable.append(np.isfinite(value) and value > 1e-8)
    isolated = bool(pair["isolated_intervention"])
    changed = int(np.count_nonzero(pair["support"]))
    valid = isolated and changed > 0 and all(measurable)
    reason = (
        "lesion_geometry_changed"
        if pair["lesion_changed_voxels"]
        else "nonlocal_input_change"
        if not isolated
        else "no_measurable_input_change"
        if not valid
        else ""
    )
    metadata = {
        "index": int(idx),
        "eps": float(eps),
        "z_low": pair["z_low"],
        "z_high": pair["z_high"],
        "changed_tissue_voxels": changed,
        "lesion_changed_voxels": pair["lesion_changed_voxels"],
        "input_outside_roi_max_abs": pair["input_outside_roi_max_abs"],
        "isolated_intervention": isolated,
        "valid_input": valid,
        "exclusion_reason": reason,
        **inputs,
    }
    # Do not retain full tissue/lesion maps once intervention validity is checked.
    return {"low": pair["a"], "high": pair["b"], "mask": pair["mask"], "metadata": metadata}


def encode_group(model, samples, args, device, level, encode_batch):
    """Use one foreground selection and channel ordering across every chunk/state."""
    masks = [s["mask"] for s in samples]
    keep = foreground_positions(masks, args, level)
    selected = None
    endpoints = {}
    for endpoint in ("low", "high"):
        pieces = {}
        for start in range(0, len(samples), encode_batch):
            chunk = samples[start : start + encode_batch]
            stages, channels, actual_keep = extract_stages(
                model,
                [s[endpoint] for s in chunk],
                [s["mask"] for s in chunk],
                args,
                device,
                level,
                keep_override=keep,
            )
            if selected is None:
                selected = channels
            if any(not torch.equal(a, b) for a, b in zip(selected, channels)):
                raise ValueError("Content channel selection changed across endpoints/encoding chunks")
            if (keep is None) != (actual_keep is None) or (keep is not None and not torch.equal(keep, actual_keep)):
                raise ValueError("Foreground selection changed across endpoints/encoding chunks")
            for stage, values in stages.items():
                pieces.setdefault(stage, []).append(values)
        endpoints[endpoint] = {stage: torch.cat(parts, dim=1) for stage, parts in pieces.items()}
    return endpoints, selected, keep


def summarize_samples(rows):
    summary = {}
    for stage in dict.fromkeys(r["stage"] for r in rows):
        group = [r for r in rows if r["stage"] == stage]
        metrics = {}
        for key in group[0]:
            if key in ("distribution", "index", "stage", "batch"):
                continue
            values = np.asarray([r[key] for r in group], dtype=float)
            finite = values[np.isfinite(values)]
            metrics[key] = {
                "median": float(np.median(finite)) if len(finite) else None,
                "mean": float(finite.mean()) if len(finite) else None,
                "n_valid": int(len(finite)),
            }
        summary[stage] = metrics
    return summary


def audit(model, ds, args, device, *, eps=0.25, level=0, batch_size=8, loss_batch_size=64, distribution="match"):
    if model.training:
        raise ValueError("Call model.eval() before the alignment diagnostic")
    if batch_size < 1 or loss_batch_size < 4 or loss_batch_size % 2:
        raise ValueError("Need positive encoding batch size and even loss batch size >=4")
    before = state_digest(model)
    rows, losses, channels, interventions = [], [], [], []
    capacity = loss_batch_size // 2
    pending = []
    batch = 0

    def measure(samples):
        nonlocal batch
        endpoints, selected, keep = encode_group(model, samples, args, device, level, batch_size)
        indices = [s["metadata"]["index"] for s in samples]
        for stage in endpoints["low"]:
            low, high = endpoints["low"][stage], endpoints["high"][stage]
            # Reuse the lesion metric definitions, but name the endpoint correctly.
            for row in response_rows(high, low, indices, stage, distribution):
                renamed = {}
                for key, value in row.items():
                    if key.startswith("on_"):
                        key = "high_" + key.removeprefix("on_")
                    elif key.startswith("subject_centered_on_"):
                        key = "subject_centered_high_" + key.removeprefix("subject_centered_on_")
                    renamed[key] = value
                rows.append({**renamed, "batch": batch})
            if stage.startswith("loss_"):
                if len(samples) >= 2:
                    for loss in pairing_losses(low, high, args, stage, level):
                        losses.append({"distribution": distribution, "batch": batch, "stage": stage, **loss})
                delta = high - low
                for channel in range(delta.shape[2]):
                    channels.append(
                        {
                            "distribution": distribution,
                            "batch": batch,
                            "stage": stage,
                            "channel": channel,
                            **pair_metrics(delta[0, :, channel], delta[1, :, channel]),
                        }
                    )
        for sample in samples:
            sample["metadata"].update(
                batch=batch,
                included_in_pairing_loss=len(samples) >= 2,
                loss_rows=2 * len(samples),
                retained_positions=int(keep.sum()) if keep is not None else None,
                selected_t1_channels=selected[0].tolist(),
                selected_flair_channels=selected[1].tolist(),
            )
        LOG.info(
            "%s: scored %d subjects, %d endpoint rows per loss (encode batch %d)",
            distribution,
            len(samples),
            2 * len(samples),
            batch_size,
        )
        batch += 1

    try:
        with stable_replay_math():
            for idx in range(len(ds)):
                sample = render_sample(ds, idx, eps)
                sample["metadata"].update(
                    distribution=distribution,
                    batch=None,
                    included_in_pairing_loss=False,
                    loss_rows=0,
                    retained_positions=None,
                    selected_t1_channels=[],
                    selected_flair_channels=[],
                )
                interventions.append(sample["metadata"])
                if sample["metadata"]["valid_input"]:
                    pending.append(sample)
                if len(pending) == capacity:
                    measure(pending)
                    pending = []
            if pending:
                measure(pending)
    finally:
        if state_digest(model) != before:
            raise RuntimeError("A registered model parameter or buffer changed during the audit")
    return {
        "samples": rows,
        "losses": losses,
        "channels": channels,
        "interventions": interventions,
        "summary": summarize_samples(rows),
        "counts": {
            "requested": len(ds),
            "valid_input": sum(r["valid_input"] for r in interventions),
            "confounded": sum(not r["isolated_intervention"] for r in interventions),
            "in_pairing_loss": sum(r["included_in_pairing_loss"] for r in interventions),
            "loss_batches": len({r["batch"] for r in losses}),
        },
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument(
        "--batch-size",
        "--encode-batch",
        dest="batch_size",
        type=int,
        default=8,
        help="Subjects per encoder forward, independent of logical BT batch size",
    )
    p.add_argument(
        "--loss-batch-size",
        type=int,
        help="Even endpoint rows per view (default: saved training batch size); half as many subjects",
    )
    p.add_argument("--eps", type=float, default=0.25, help="Half step in z_content[1], not a voxel/mm increment")
    p.add_argument("--causal", choices=["iid", "match", "both"], default="match")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--device")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--out-dir", type=Path)
    return p


def print_result(distribution, result):
    counts = result["counts"]
    print(
        f"\n{distribution}: {counts['valid_input']}/{counts['requested']} isolated measurable interventions; "
        f"{counts['confounded']} confounded; {counts['in_pairing_loss']} subjects in pairing losses"
    )
    print("Medians across subjects; Δh = ventricle-high − ventricle-low")
    print("  stage                       Δcos   wrong-subj  T1/FLAIR RMS   relative MSE   high-cos")
    for stage, metrics in result["summary"].items():

        def val(key):
            value = metrics[key]["median"]
            return f"{value:.3f}" if value is not None else "undefined"

        print(
            f"  {stage:27s} {val('delta_cosine'):>8s} {val('delta_wrong_subject_cosine'):>10s}"
            f" {val('delta_t1_over_flair_rms'):>13s} {val('delta_relative_mse'):>14s} {val('high_cosine'):>9s}"
        )
    print("\nBalanced pairing losses (configured weights; means over batches weighted by subject count)")
    print("  mode                   stage        batches rows     matched     Δventricle     Δsubject")
    for mode, stage in dict.fromkeys((r["mode"], r["stage"]) for r in result["losses"]):
        subset = [r for r in result["losses"] if r["mode"] == mode and r["stage"] == stage]
        weights = [r["n_subjects"] for r in subset]
        values = [
            float(np.average([r[key] for r in subset], weights=weights))
            for key in (
                "matched_weighted_total",
                "ventricle_mismatched_minus_matched_weighted_total",
                "subject_mismatched_minus_matched_weighted_total",
            )
        ]
        sizes = sorted({r["loss_rows"] for r in subset})
        label = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]}"
        print(
            f"  {mode:22s} {stage:12s} {len(subset):7d} {label:>5s}"
            f" {values[0]:11.5g} {values[1]:+14.5g} {values[2]:+12.5g}"
        )
    if not result["losses"]:
        print("  No eligible multi-subject batch; inspect interventions.csv.")


def main(cli=None):
    p = parser()
    cli = p.parse_args() if cli is None else cli
    if cli.num_samples < 4 or cli.batch_size < 1 or cli.level < 0 or cli.cpu_threads < 1:
        p.error("Need >=4 samples, positive encode batch/CPU threads, and a nonnegative level")
    if not np.isfinite(cli.eps) or cli.eps <= 0:
        p.error("--eps must be finite and positive")
    if cli.loss_batch_size is not None and (cli.loss_batch_size < 4 or cli.loss_batch_size % 2):
        p.error("--loss-batch-size must be even and >=4")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(cli.cpu_threads)
    model, args, device = load_model(cli.run_dir, cli.checkpoint, cli.device)
    checkpoint = Path(cli.checkpoint or "vqvae_model.pt")
    if checkpoint.parent == Path("."):
        checkpoint = Path(cli.run_dir) / checkpoint
    provenance = verify_checkpoint(model, checkpoint)
    loss_batch_size = cli.loss_batch_size if cli.loss_batch_size is not None else int(getattr(args, "batch_size", 64))
    if loss_batch_size < 4 or loss_batch_size % 2:
        p.error("Saved training batch size must be even and >=4; supply --loss-batch-size")
    results = {}
    for distribution in ("iid", "match") if cli.causal == "both" else (cli.causal,):
        ds = make_dataset(args, cli.num_samples, distribution, "test")
        results[distribution] = audit(
            model,
            ds,
            args,
            device,
            eps=cli.eps,
            level=cli.level,
            batch_size=cli.batch_size,
            loss_batch_size=loss_batch_size,
            distribution=distribution,
        )
    directory = cli.out_dir or Path(cli.run_dir) / (
        f"ventricle_alignment_{cli.causal}_eps{cli.eps:g}_L{cli.level}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": vars(cli),
        "run_settings": vars(args),
        "checkpoint": provenance,
        "resolved_loss_batch_size": loss_batch_size,
        "registered_state_unchanged": True,
        "summary": {key: value["summary"] for key, value in results.items()},
        "counts": {key: value["counts"] for key, value in results.items()},
        "batch_bt_terms": [r for result in results.values() for r in result["losses"]],
        "protocol": {
            "intervention": "z_content[1] +/- eps with other latents, noise, mask and normalization fixed; not SCM propagation",
            "exclusions": "Changed rendered lesion geometry, nonlocal input changes, or no measurable input response in either view",
            "pairing": "T1 [low,high]; matched FLAIR [low,high]; ventricle mismatch FLAIR [high,low] within subject",
            "subject_control": "Cyclic subject permutation of FLAIR, keeping low/high endpoint identity",
            "ema": "One step from a hypothetical settled matched-batch correlation, cloned for each condition; NOT training history",
            "batching": "Loss rows are twice the subject count; final batch may be smaller. Mask fixed over the logical subject batch",
            "stages": "Native probe after content_norms; native alignment source before it; pooled_content before head; loss stages after head",
        },
        "limitations": (
            "Eval-mode diagnostic, not a training-gradient or retention test. Endpoint pairs are dependent and can be off-distribution. "
            "Raw cosine is not the BT loss. Check absolute response RMS. Counterfactual loss differences are not additive factor attribution. "
            "A positive mismatch penalty does not prevent both views from discarding the factor. Only the requested level is evaluated."
        ),
    }
    # Paths in arguments are serialized explicitly; json_safe handles non-finite metrics.
    report["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    for name, key in (
        ("samples", "samples"),
        ("pairing_losses", "losses"),
        ("channels", "channels"),
        ("interventions", "interventions"),
    ):
        write_csv(directory / f"{name}.csv", [r for result in results.values() for r in result[key]])
    for distribution, result in results.items():
        print_result(distribution, result)
    print(
        "\nΔventricle = mismatched − matched: positive penalizes wrong ventricular pairing; negative favors it in this batch."
    )
    print("Inspect component deltas in pairing_losses.csv; terms may cancel. This is not a factor-retention guarantee.")
    print(
        "EMA rows use a hypothetical matched reference, not historical training state. No uncertainty estimate from these batch means."
    )
    print(
        "Inspect absolute Δh RMS in samples.csv and exclusions in interventions.csv. No registered parameter or buffer changed."
    )
    print(f"Saved {directory}")


if __name__ == "__main__":
    main()
