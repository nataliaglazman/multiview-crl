#!/usr/bin/env python
"""Which content channel carries the lesion, and does the probe agree?

    python -m eval.lesion_channel_probe --run-dir /path/to/run \
        --channels-csv /path/to/lesion_alignment_iid_L0/channels.csv

``lesion_alignment`` reports one cross-view alignment number per content channel, and
a block-level number that averages them. When a single channel is well aligned and the
rest are incoherent, the block number understates what is encoded. This script answers
the follow-up: name that channel, then check whether the lesion probe agrees that it is
where the lesion lives.

It joins two things per channel:

- alignment, read from ``channels.csv``: the energy-weighted cross-view cosine of the
  lesion response, and the channel's share of the block's total response energy.
- decodability, fitted here with ``lesion_probe``'s own protocol: held-out R^2 for the
  lesion coordinates from that ONE channel's spatial columns ("alone"), and from the
  block with that channel removed ("without"). Alone says the channel is sufficient;
  without says it is necessary. A concentrated code scores high alone and costs a lot
  when removed.

``channels.csv`` indexes positions WITHIN the content block, not encoder channels. The
mapping comes from the forward content mask, so the table prints both.

The probe protocol -- split, standardization, kernel/alpha selection on validation,
shuffled-label null -- is imported from ``eval.lesion_probe``, never re-implemented
here. Only the column subsetting differs. Per-view scores are separate fits; the
alignment cosine is a single cross-view number, so it has no per-view column.

Failure of a finite-sample probe is not proof of information loss, and a channel that
scores well alone may still be redundant with its neighbours. ``--rank-only`` prints
the alignment ranking without loading a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

# lesion_alignment writes one row per (distribution, batch, stage, channel). Rank on the
# stage the loss actually consumes, in this order of preference.
STAGE_PREFERENCE = ("loss_patch", "loss_global", "pooled_content", "native_probe")


def pick_stage(available, requested=None):
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Stage {requested!r} not in channels.csv; have {sorted(available)}")
        return requested
    for stage in STAGE_PREFERENCE:
        if stage in available:
            return stage
    return sorted(available)[0]


def rank_channels(channels_csv, distribution="iid", stage=None):
    """Energy-weighted cosine and energy share per content-block position.

    Energy weighting matters: a channel whose response is numerically tiny has a
    meaningless cosine, and a plain mean over channels lets it vote like the rest.
    """
    with Path(channels_csv).open() as f:
        rows = [r for r in csv.DictReader(f) if r["distribution"] == distribution]
    if not rows:
        raise ValueError(f"No rows for distribution {distribution!r} in {channels_csv}")
    stage = pick_stage({r["stage"] for r in rows}, stage)
    per = {}
    for r in (x for x in rows if x["stage"] == stage):
        energy = float(r["t1_rms"]) ** 2 + float(r["flair_rms"]) ** 2
        cosine = float(r["cosine"]) if r["cosine"] not in (None, "") else float("nan")
        if not math.isfinite(energy):
            continue
        slot = per.setdefault(int(r["channel"]), {"energy": 0.0, "weighted": 0.0, "cosines": []})
        slot["energy"] += energy
        if math.isfinite(cosine):
            if not -1.001 <= cosine <= 1.001:
                raise ValueError(
                    f"channels.csv has cosine {cosine} at stage {stage}, channel {r['channel']}; a cosine "
                    "cannot leave [-1, 1], so this file is not lesion_alignment output"
                )
            slot["weighted"] += cosine * energy
            slot["cosines"].append(cosine)
    total = sum(v["energy"] for v in per.values())
    if total <= 0:
        raise ValueError(f"Zero total response energy at stage {stage}; nothing to rank")
    out = []
    for channel, v in sorted(per.items()):
        out.append(
            {
                "block_position": channel,
                "cosine": v["weighted"] / v["energy"] if v["energy"] > 0 else None,
                "cosine_unweighted": statistics.fmean(v["cosines"]) if v["cosines"] else None,
                "energy_share": v["energy"] / total,
            }
        )
    return stage, out


def content_channels_from_probe_summary(path):
    """Encoder channel indices per view, as lesion_probe records them."""
    data = json.loads(Path(path).read_text())
    got = data.get("content_channels_by_view")
    if not got:
        raise ValueError(f"{path} has no content_channels_by_view")
    return [list(map(int, view)) for view in got]


def encoder_label(content_channels, position):
    """Print the encoder channel for a block position, collapsing views when equal."""
    if content_channels is None:
        return "?"
    try:
        per_view = [view[position] for view in content_channels]
    except IndexError:
        return "?"
    return str(per_view[0]) if len(set(per_view)) == 1 else "/".join(map(str, per_view))


def best_test_r2(rows, target="latent", condition="observed"):
    """Mean held-out R^2 for the kernel/alpha that validation selected.

    Selection is on validation only, across probe kinds as well as hyperparameters --
    picking the kind by test score would be tuning against the held-out set.
    """
    import numpy as np

    candidates = [r for r in rows if r.get("condition") == condition and r.get("target") == target]
    if not candidates:
        return None, None
    row = min(candidates, key=lambda r: r["validation_mse_standardized"])
    scores = row["test"].get("r2_xyz")
    if scores is None:
        return None, row["probe"]
    return float(np.mean(scores)), row["probe"]


def probe_columns(arrays, view, columns, targets, splits, seed):
    from eval.lesion_probe import block_gram, fit_probes

    gram, width = block_gram(arrays[view], columns, splits[0])
    if width == 0:
        return {"feature_count": 0, "r2": None, "null_r2": None, "probe": None}
    rows, _ = fit_probes(gram, width, targets, splits, seed)
    r2, kind = best_test_r2(rows)
    null, _ = best_test_r2(rows, condition="shuffled")
    return {"feature_count": width, "r2": r2, "null_r2": null, "probe": kind}


def probe_channels(arrays, partitions, shape, targets, splits, seed, leave_one_out=True):
    """Full block, each channel alone, and the block without each channel."""
    import numpy as np

    sites = int(np.prod(shape[1:]))
    content = [np.flatnonzero(mask) for mask in partitions]
    width = min(len(c) for c in content)
    result = {"content_channels_by_view": [c.tolist() for c in content], "views": {}}
    for view, name in enumerate(("t1", "flair")):
        block = np.flatnonzero(np.repeat(partitions[view], sites))
        logger.info("%s: full content block, %d columns", name, len(block))
        per_view = {"full_block": probe_columns(arrays, view, block, targets, splits, seed), "channels": {}}
        for position in range(width):
            encoder_channel = int(content[view][position])
            own = np.arange(encoder_channel * sites, (encoder_channel + 1) * sites)
            entry = {
                "encoder_channel": encoder_channel,
                "alone": probe_columns(arrays, view, own, targets, splits, seed),
            }
            if leave_one_out:
                entry["without"] = probe_columns(arrays, view, np.setdiff1d(block, own), targets, splits, seed)
            per_view["channels"][position] = entry
            logger.info(
                "%s: block position %d (encoder channel %d) alone R2=%s",
                name,
                position,
                encoder_channel,
                "n/a" if entry["alone"]["r2"] is None else f"{entry['alone']['r2']:.3f}",
            )
        result["views"][name] = per_view
    return result


def fmt_r2(value):
    return "  n/a" if value is None else f"{value:+.3f}"


def print_report(stage, ranking, probes, content_channels):
    print("\n" + "=" * 92)
    print(f"Content channel: alignment (stage {stage}) vs lesion decodability")
    print("=" * 92)
    if probes:
        print("\nFull content block, held-out R2 for the lesion coordinates:")
        for name, view in probes["views"].items():
            full = view["full_block"]
            print(
                f"  {name:6s} R2 {fmt_r2(full['r2'])}   null {fmt_r2(full['null_r2'])}"
                f"   probe {full['probe']}   {full['feature_count']} columns"
            )

    print("\nPer channel, ordered by share of the lesion response energy:")
    header = "  blk  enc ch     dcos  energy%"
    if probes:
        header += "    R2 alone T1/FLAIR    R2 without T1/FLAIR"
    print(header)
    order = sorted(ranking, key=lambda r: -r["energy_share"])
    for row in order:
        position = row["block_position"]
        cosine = "n/a" if row["cosine"] is None else f"{row['cosine']:+.3f}"
        line = (
            f"  {position:3d}  {encoder_label(content_channels, position):6s}"
            f"  {cosine:>7s}  {100 * row['energy_share']:6.2f}%"
        )
        if probes:
            cells = []
            for kind in ("alone", "without"):
                pair = []
                for name in ("t1", "flair"):
                    entry = probes["views"][name]["channels"].get(position, {}).get(kind)
                    pair.append(fmt_r2(entry["r2"] if entry else None))
                cells.append(" / ".join(pair))
            line += f"    {cells[0]:17s}    {cells[1]}"
        print(line)

    if not probes:
        print("\n  dcos +1 = the two views move the same way when a lesion appears; energy% is the")
        print("  channel's share of the block's total lesion response. Run without --rank-only to")
        print("  add the probe columns.")
        return

    print("\n  alone   = R2 from this channel's spatial columns only (is it sufficient?)")
    print("  without = R2 from the block with this channel removed (is it necessary?)")
    for name, view in probes["views"].items():
        full = view["full_block"]["r2"]
        best = max(
            (c for c in view["channels"].values() if c["alone"]["r2"] is not None),
            key=lambda c: c["alone"]["r2"],
            default=None,
        )
        if full is None or best is None:
            continue
        share = 100 * best["alone"]["r2"] / full if abs(full) > 1e-9 else float("nan")
        print(
            f"\n  {name}: best single channel is encoder channel {best['encoder_channel']}"
            f" at R2 {best['alone']['r2']:+.3f}, {share:.0f}% of the full block's {full:+.3f}."
        )
        if "without" in best and best["without"]["r2"] is not None:
            print(f"     Removing it takes the block to {best['without']['r2']:+.3f}.")
    print("\n  A channel that is well aligned AND decodable alone is where the lesion is encoded.")
    print("  Aligned but not decodable, or decodable but not aligned, means the block number is")
    print("  mixing two different things and neither channel alone carries the factor.")


def _self_test_stdlib():
    """Ranking and index mapping, with no numpy or model."""
    import tempfile

    rows = []
    # Block position 3 dominates the energy and is the only aligned one.
    plan = {0: (0.02, 0.05), 1: (-0.10, 0.05), 2: (0.04, 0.05), 3: (0.62, 1.00), 4: (-0.05, 0.05)}
    for batch in range(4):
        for position, (cosine, energy) in plan.items():
            rms = (energy / 2) ** 0.5
            for stage in ("loss_patch", "loss_gap"):
                rows.append(
                    {
                        "distribution": "iid",
                        "batch": batch,
                        "stage": stage,
                        "channel": position,
                        "cosine": cosine if stage == "loss_patch" else 0.01,
                        "relative_mse": 0.9,
                        "t1_rms": rms,
                        "flair_rms": rms,
                        "t1_over_flair_rms": 1.0,
                        "mse": 0.01,
                    }
                )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "channels.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        stage, ranking = rank_channels(path)
        _, gap = rank_channels(path, stage="loss_gap")
        summary = Path(tmp) / "probe.json"
        summary.write_text(json.dumps({"content_channels_by_view": [[5, 9, 14, 17, 23], [5, 9, 14, 19, 23]]}))
        mapping = content_channels_from_probe_summary(summary)

    top = max(ranking, key=lambda r: r["energy_share"])
    checks = [
        ("prefers the stage the loss consumes", stage == "loss_patch"),
        ("finds block position 3 as top energy", top["block_position"] == 3),
        ("recovers its cosine", abs(top["cosine"] - 0.62) < 1e-9),
        ("energy share sums to 1", abs(sum(r["energy_share"] for r in ranking) - 1.0) < 1e-9),
        ("top channel holds ~83% of energy", abs(top["energy_share"] - 1.0 / 1.2) < 1e-9),
        ("--stage override is honoured", abs(gap[3]["cosine"] - 0.01) < 1e-9),
        ("maps block 3 to encoder channel 17/19", encoder_label(mapping, 3) == "17/19"),
        ("collapses equal views to one label", encoder_label(mapping, 0) == "5"),
        ("unknown mapping degrades to ?", encoder_label(None, 3) == "?" and encoder_label(mapping, 99) == "?"),
    ]
    print("self-test (stdlib: ranking and channel mapping)")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print_report(stage, ranking, None, mapping)
    return all(ok for _, ok in checks)


def _self_test_numpy():
    """The probe layer, on planted features where one channel holds the target."""
    try:
        import numpy as np

        from eval.lesion_probe import split_subjects
    except ImportError as exc:
        print(f"\nself-test (probe layer): SKIPPED, needs numpy/scipy/sklearn ({exc})")
        return True

    rng = np.random.default_rng(0)
    n, sites, total_channels = 120, 8, 6
    signal_channel, content = 2, np.array([True, True, True, True, False, False])
    targets = rng.normal(size=(n, 6))
    arrays = []
    for _ in range(2):
        x = rng.normal(size=(n, total_channels * sites)) * 0.3
        # Plant the lesion coordinates in one channel only, as a linear spatial code.
        block = signal_channel * sites
        for axis in range(3):
            x[:, block + axis] += targets[:, 3 + axis] * 3.0
            x[:, block + 3 + axis] += targets[:, axis] * 0.05
        arrays.append(x)
    splits = split_subjects(n, 0)
    got = probe_channels(arrays, [content, content], (total_channels, sites), targets, splits, 0)

    t1 = got["views"]["t1"]
    planted = t1["channels"][signal_channel]
    others = [c for p, c in t1["channels"].items() if p != signal_channel]
    checks = [
        ("planted channel decodes alone", planted["alone"]["r2"] > 0.7),
        ("other channels do not", max(c["alone"]["r2"] for c in others) < 0.2),
        ("removing it collapses the block", planted["without"]["r2"] < 0.2),
        ("keeping it does not", min(c["without"]["r2"] for c in others) > 0.7),
        ("full block decodes", t1["full_block"]["r2"] > 0.7),
        ("shuffled null is near zero", abs(t1["full_block"]["null_r2"]) < 0.3),
        ("encoder channel index is reported", planted["encoder_channel"] == signal_channel),
        ("style channels are excluded", t1["full_block"]["feature_count"] <= 4 * sites),
    ]
    print("\nself-test (probe layer: planted single-channel code)")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return all(ok for _, ok in checks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Training run directory, as lesion_probe takes it")
    p.add_argument("--channels-csv", help="channels.csv from lesion_alignment; default: alongside --run-dir")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--causal", choices=["iid", "match"], default="iid")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--stage", default=None, help="channels.csv stage to rank by; default: what the loss consumes")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--grid", type=int, default=0, help="0=native spatial map; positive value=pooled grid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--temp-dir", default=None, help="Scratch parent for temporary spatial feature arrays")
    p.add_argument("--no-leave-one-out", action="store_true", help="Skip the block-without-channel fits")
    p.add_argument("--probe-summary", default=None, help="lesion_probe summary.json, for --rank-only channel names")
    p.add_argument("--rank-only", action="store_true", help="Alignment ranking only; no model, no probe")
    p.add_argument("--self-test", action="store_true")
    cli = p.parse_args()
    if cli.self_test:
        ok = _self_test_stdlib()
        ok = _self_test_numpy() and ok
        if not ok:
            raise SystemExit("self-test failed")
        print("\nall checks passed")
        return
    if not cli.channels_csv and not cli.run_dir:
        p.error("Need --channels-csv, or --run-dir to find it")
    if cli.num_samples < 30 or min(cli.batch_size, cli.cpu_threads) < 1 or min(cli.level, cli.grid) < 0:
        p.error("Need >=30 samples, positive batch size/threads, nonnegative level/grid")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    channels_csv = cli.channels_csv or (
        Path(cli.run_dir) / f"lesion_alignment_{cli.causal}_L{cli.level}" / "channels.csv"
    )
    stage, ranking = rank_channels(channels_csv, "iid" if cli.causal == "iid" else "match", cli.stage)

    if cli.rank_only:
        mapping = content_channels_from_probe_summary(cli.probe_summary) if cli.probe_summary else None
        print_report(stage, ranking, None, mapping)
        return
    if not cli.run_dir:
        p.error("--run-dir is required unless --rank-only")

    import tempfile

    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    from eval.lesion_probe import extract_features, split_subjects
    from eval.lesion_reconstruction import json_safe, make_dataset
    from eval.run_dci_synthetic import load_model_from_run_dir

    torch.set_num_threads(cli.cpu_threads)
    splits = split_subjects(cli.num_samples, cli.seed)
    with threadpool_limits(limits=cli.cpu_threads):
        model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device=cli.device, seed=cli.seed)
        ds = make_dataset(args, cli.num_samples, cli.causal, cli.split)
        with tempfile.TemporaryDirectory(prefix="lesion-channel-", dir=cli.temp_dir) as tmp:
            arrays, partitions, shape, targets, ids = extract_features(
                model, ds, device, tmp, cli.batch_size, cli.level, cli.grid
            )
            if len(ranking) != int(min(np.count_nonzero(m) for m in partitions)):
                raise ValueError(
                    f"channels.csv has {len(ranking)} content channels but this checkpoint selects "
                    f"{int(min(np.count_nonzero(m) for m in partitions))}; the CSV is from a different run or level"
                )
            probes = probe_channels(arrays, partitions, shape, targets, splits, cli.seed, not cli.no_leave_one_out)
            del arrays

    print_report(stage, ranking, probes, probes["content_channels_by_view"])
    directory = Path(cli.out_dir or Path(cli.run_dir) / f"lesion_channel_probe_{cli.causal}_L{cli.level}")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": vars(cli),
        "channels_csv": str(channels_csv),
        "alignment_stage": stage,
        "alignment_ranking": ranking,
        "probes": probes,
        "protocol": "Column subsets of eval.lesion_probe; its split, scaling, selection and null are unchanged",
        "limitations": "Finite-sample probe. A channel scoring well alone may still be redundant with others; "
        "alone and without together separate sufficiency from necessity.",
    }
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    with (directory / "channels.csv").open("w", newline="") as f:
        fields = ["block_position", "encoder_channel_t1", "encoder_channel_flair", "cosine", "energy_share"]
        fields += [f"r2_{k}_{v}" for k in ("alone", "without") for v in ("t1", "flair")]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranking:
            position = row["block_position"]
            out = {
                "block_position": position,
                "cosine": row["cosine"],
                "energy_share": row["energy_share"],
            }
            for view in ("t1", "flair"):
                channel = probes["views"][view]["channels"].get(position, {})
                out[f"encoder_channel_{view}"] = channel.get("encoder_channel")
                for kind in ("alone", "without"):
                    out[f"r2_{kind}_{view}"] = (channel.get(kind) or {}).get("r2")
            writer.writerow(out)
    print(f"\nSaved {directory}")


if __name__ == "__main__":
    main()
