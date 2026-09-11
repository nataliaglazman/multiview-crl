#!/usr/bin/env python
"""One readable page from a ``lesion_alignment`` output directory.

    python -m eval.lesion_alignment_report results/.../lesion_alignment_iid_L0

``lesion_alignment`` prints stage medians and writes the per-subject rows. This reads
those rows back and answers the questions the medians alone cannot:

1. Is dcos actually above its own wrong-subject null, per subject, with a CI?
2. Is the lesion response big enough for the cosine to mean anything? A cosine
   between two near-zero vectors is noise, not shared encoding, and this is the
   caveat the table prints but does not resolve.
3. How much of the high on-cos is just a constant offset shared by both views?
4. What does each pooling step cost, paired within subject?
5. Does the BT objective register the lesion at all?
6. Is the aligned part spread over the content block or concentrated in a few channels?

Inputs are ``samples.csv`` (required), ``summary.json`` and ``channels.csv`` (optional;
their sections are skipped if absent). Pure stdlib, no torch: ``--self-test`` runs the
whole report on planted rows with known answers.

Nothing is re-scored here. Every number is an aggregate of what lesion_alignment wrote.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

# Raw-image reference for the T1/FLAIR response-magnitude ratio. Lesions sit in white
# matter and render with OPPOSITE sign in the two views (eval/synthetic_dataset.py):
# T1 WM 0.8 -> lesion 0.4 (-0.4), FLAIR WM 0.4 -> lesion 1.0 (+0.6). So the inputs
# themselves have dcos ~ -1 and a magnitude ratio of ~0.4/0.6, up to the per-view
# normalization scales. A view-invariant encoder has to undo both.
INPUT_RMS_RATIO = 0.4 / 0.6
INPUT_COSINE = -1.0


def finite(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, ""):
            continue
        v = float(v)
        if math.isfinite(v):
            out.append(v)
    return out


def med(values):
    return statistics.median(values) if values else None


def boot_ci(values, seed=0, reps=2000, alpha=0.05):
    """Percentile bootstrap CI for the median."""
    if len(values) < 3:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(reps))
    return draws[int(alpha / 2 * reps)], draws[min(reps - 1, int((1 - alpha / 2) * reps))]


def sign_p(diffs):
    """Two-sided exact sign test that the paired differences are centred on zero."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(rows, a, b):
    """Per-subject differences a-b, keeping only subjects where both are finite."""
    out = []
    for r in rows:
        x, y = r.get(a), r.get(b)
        if x in (None, "") or y in (None, ""):
            continue
        x, y = float(x), float(y)
        if math.isfinite(x) and math.isfinite(y):
            out.append(x - y)
    return out


def pct_of_spread(rows, view):
    """Lesion response as a percentage of the between-subject spread, same units.

    lesion_alignment stores an energy ratio; the square root is the amplitude ratio,
    which is the one that is comparable to a cosine's tolerance for noise.
    """
    vals = [v for v in finite(rows, f"delta_to_subject_variation_{view}") if v >= 0]
    return 100 * math.sqrt(med(vals)) if vals else None


def fmt(v, width=7, places=3, suffix=""):
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{places}f}{suffix}"


def verdict(dcos, gap, p, resp_ok):
    if not resp_ok:
        return "UNDEFINED (response too small)"
    if dcos is None:
        return "n/a"
    if p > 0.05 or abs(gap) < 0.02:
        return "at null (no shared encoding)"
    if dcos < 0:
        return "ANTI-ALIGNED"
    if dcos < 0.5:
        return "weak"
    if dcos < 0.8:
        return "partial"
    return "aligned"


def stage_order(rows):
    """Pipeline order, as lesion_alignment wrote it. Never sort these alphabetically:
    the whole signal is the trend from the native map down to what the loss sees."""
    seen = []
    for r in rows:
        if r["stage"] not in seen:
            seen.append(r["stage"])
    return seen


def in_order(names, stages):
    return sorted(names, key=lambda s: (stages.index(s) if s in stages else len(stages), s))


def clean(v):
    return None if v is None or not math.isfinite(v) else v


def report(directory, min_response_pct=1.0, seed=0):
    directory = Path(directory)
    with (directory / "samples.csv").open() as f:
        samples = list(csv.DictReader(f))
    if not samples:
        raise ValueError(f"No rows in {directory / 'samples.csv'}")
    summary = {}
    if (directory / "summary.json").exists():
        summary = json.loads((directory / "summary.json").read_text())
    channels = []
    if (directory / "channels.csv").exists():
        with (directory / "channels.csv").open() as f:
            channels = list(csv.DictReader(f))

    args = summary.get("arguments", {})
    print("=" * 92)
    print(f"lesion_alignment report   {directory}")
    if args:
        print(f"  level {args.get('level')}   subjects/dist {args.get('num_samples')}   batch {args.get('batch_size')}")
    print("=" * 92)

    out = {"directory": str(directory), "distributions": {}}
    for dist in sorted({r["distribution"] for r in samples}):
        drows = [r for r in samples if r["distribution"] == dist]
        stages = stage_order(drows)
        result = {}
        print(f"\n######## distribution: {dist}   ({len({r['index'] for r in drows})} subjects)\n")

        print("[1] LESION RESPONSE ALIGNED ACROSS VIEWS?   dcos vs its wrong-subject null; want +1")
        print("    stage                        dcos  95% CI            null  dcos-null   sign p   n  verdict")
        for stage in stages:
            rows = [r for r in drows if r["stage"] == stage]
            d = finite(rows, "delta_cosine")
            lo, hi = boot_ci(d, seed=seed)
            gaps = paired(rows, "delta_cosine", "delta_wrong_subject_cosine")
            g, p = med(gaps), sign_p(gaps)
            t1 = pct_of_spread(rows, "t1")
            fl = pct_of_spread(rows, "flair")
            ok = t1 is not None and fl is not None and min(t1, fl) >= min_response_pct
            ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "n/a"
            v = verdict(med(d), g if g is not None else 0.0, p, ok)
            null = med(finite(rows, "delta_wrong_subject_cosine"))
            print(
                f"    {stage:24s}{'n/a' if med(d) is None else f'{med(d):+.3f}':>9s}  {ci:16s}"
                f"{'n/a' if null is None else f'{null:+.3f}':>7s}"
                f"  {'n/a' if g is None else f'{g:+.3f}':>9s}   {p:7.1e} {len(d):3d}  {v}"
            )
            result[stage] = {
                "n": len(d),
                "delta_cosine_median": med(d),
                "delta_cosine_ci": [lo, hi],
                "wrong_subject_median": med(finite(rows, "delta_wrong_subject_cosine")),
                "dcos_minus_null_median": g,
                "sign_p": p,
                "response_pct_of_subject_spread": {"t1": t1, "flair": fl},
                "verdict": v,
            }

        print(f"\n[2] IS THE RESPONSE BIG ENOUGH TO TRUST THE COSINE?   (< {min_response_pct}% => cosine is noise)")
        print("    stage                    T1 resp  FLAIR resp   T1/FLAIR RMS   rel MSE")
        for stage in stages:
            rows = [r for r in drows if r["stage"] == stage]
            t1, fl = pct_of_spread(rows, "t1"), pct_of_spread(rows, "flair")
            ratio = med(finite(rows, "delta_t1_over_flair_rms"))
            flag = "  <-- negligible" if t1 is not None and fl is not None and min(t1, fl) < min_response_pct else ""
            print(
                f"    {stage:24s}{fmt(t1, 7, 2, '%')}{fmt(fl, 11, 2, '%')}{fmt(ratio, 15)}"
                f"{fmt(med(finite(rows, 'delta_relative_mse')), 10)}{flag}"
            )
            result[stage]["t1_over_flair_rms_median"] = ratio
            result[stage]["relative_mse_median"] = med(finite(rows, "delta_relative_mse"))
        print(
            f"    raw-image reference: T1/FLAIR RMS ~{INPUT_RMS_RATIO:.2f} and dcos ~{INPUT_COSINE:+.0f}"
            " (the lesion renders dark on T1,"
        )
        print("    bright on FLAIR). A ratio below the reference means the encoder widens that asymmetry.")

        print("\n[3] OVERALL SIMILARITY, RAW vs SUBJECT-CENTRED   (raw on-cos is inflated by a shared offset)")
        print("    stage                     on-cos  centred on-cos       dcos")
        for stage in stages:
            rows = [r for r in drows if r["stage"] == stage]
            raw = med(finite(rows, "on_cosine"))
            cen = med(finite(rows, "subject_centered_on_cosine"))
            print(f"    {stage:24s}{fmt(raw, 8)}{fmt(cen, 16)}{fmt(med(finite(rows, 'delta_cosine')), 11)}")
            result[stage]["on_cosine_median"] = raw
            result[stage]["subject_centered_on_cosine_median"] = cen

        if len(stages) > 1:
            print(f"\n[4] WHAT EACH STAGE COSTS   (paired within subject, vs {stages[0]})")
            for stage in stages[1:]:
                base = {r["index"]: r for r in drows if r["stage"] == stages[0]}
                diffs = [
                    float(r["delta_cosine"]) - float(base[r["index"]]["delta_cosine"])
                    for r in drows
                    if r["stage"] == stage and r["index"] in base
                    if math.isfinite(float(r["delta_cosine"]))
                    and math.isfinite(float(base[r["index"]]["delta_cosine"]))
                ]
                if diffs:
                    print(f"    -> {stage:24s} dcos {med(diffs):+.3f}   sign p {sign_p(diffs):7.1e}   n {len(diffs)}")
                    result[stage]["dcos_change_vs_first_stage"] = med(diffs)

        bt = [t for t in summary.get("batch_bt_terms", []) if t["distribution"] == dist]
        if bt:
            print("\n[5] DOES THE BT OBJECTIVE REGISTER THE LESION?   (mean over batches)")
            print("    stage                     loss on   loss off     change   % of loss   pos-sim on/off")
            for stage in in_order({t["stage"] for t in bt}, stages):
                ts = [t for t in bt if t["stage"] == stage]
                on = statistics.fmean(t["on"]["instantaneous_weighted_total"] for t in ts)
                off = statistics.fmean(t["off"]["instantaneous_weighted_total"] for t in ts)
                rel = 100 * (on - off) / abs(on) if abs(on) > 1e-20 else float("nan")
                ps_on = statistics.fmean(t["on"].get("pos_sim_mean", float("nan")) for t in ts)
                ps_off = statistics.fmean(t["off"].get("pos_sim_mean", float("nan")) for t in ts)
                print(
                    f"    {stage:24s}{on:9.4f}{off:11.4f}{on - off:+11.4f}{rel:11.2f}%"
                    f"   {ps_on:+.3f} / {ps_off:+.3f}"
                )
                result.setdefault(stage, {})["bt"] = {
                    "loss_on": clean(on),
                    "loss_off": clean(off),
                    "change": clean(on - off),
                    "percent_of_loss": clean(rel),
                    "pos_sim_on": clean(ps_on),
                    "pos_sim_off": clean(ps_off),
                }
            print("    A change near 0% means the loss is blind to the lesion; it is a sensitivity, not")
            print("    an additive attribution (correlations are recomputed, EMA state is not replayed).")

        crows = [r for r in channels if r["distribution"] == dist]
        if crows:
            print("\n[6] IS THE ALIGNED PART CONCENTRATED IN A FEW CHANNELS?   (energy-weighted over batches)")
            print("    stage                    chans  wmean cos  chans for 80% energy   top-3 channel cos")
            for stage in in_order({r["stage"] for r in crows}, stages):
                per = {}
                for r in (x for x in crows if x["stage"] == stage):
                    e = float(r["t1_rms"]) ** 2 + float(r["flair_rms"]) ** 2
                    c = float(r["cosine"]) if r["cosine"] not in (None, "") else float("nan")
                    if not math.isfinite(e):
                        continue
                    slot = per.setdefault(int(r["channel"]), [0.0, 0.0, 0])
                    slot[0] += e
                    slot[2] += 1
                    if math.isfinite(c):
                        slot[1] += c * e
                total = sum(v[0] for v in per.values())
                if not per or total <= 0:
                    continue
                wmean = sum(v[1] for v in per.values()) / total
                energies = sorted((v[0] for v in per.values()), reverse=True)
                cum, n80 = 0.0, 0
                for e in energies:
                    cum += e
                    n80 += 1
                    if cum >= 0.8 * total:
                        break
                top = sorted(per.items(), key=lambda kv: -kv[1][0])[:3]
                cos_str = ", ".join(f"{v[1] / v[0]:+.2f}" if v[0] > 0 else " n/a" for _, v in top)
                print(f"    {stage:24s}{len(per):6d}{wmean:+11.3f}{n80:22d}   {cos_str}")
                result.setdefault(stage, {})["channels"] = {
                    "n_channels": len(per),
                    "energy_weighted_cosine": wmean,
                    "channels_for_80pct_energy": n80,
                }
        out["distributions"][dist] = result

    print("\n" + "=" * 92)
    print("dcos is the number: +1 = the two views move the same way when a lesion appears, 0 = no")
    print("shared lesion code, negative = they move oppositely. Check [2] before reading [1]; check")
    print("[3] before believing a high on-cos; [4] tells you whether pooling or the encoder is at fault.")
    print("=" * 92)
    return out


def _self_test():
    """Run the whole report on planted rows whose answers are known in advance."""
    import tempfile

    rng = random.Random(0)
    n = 40
    rows, chans = [], []
    # (stage, target dcos, response as fraction of subject spread)
    plan = [("shared_big", 0.95, 0.30), ("null_big", 0.0, 0.30), ("anti_big", -0.90, 0.30), ("shared_tiny", 0.95, 1e-4)]
    for stage, target, frac in plan:
        for i in range(n):
            jitter = rng.gauss(0, 0.03)
            rows.append(
                {
                    "distribution": "iid",
                    "index": i,
                    "stage": stage,
                    "delta_cosine": max(-1.0, min(1.0, target + jitter)),
                    "delta_relative_mse": 1 - target,
                    "delta_t1_rms": frac,
                    "delta_flair_rms": frac,
                    "delta_t1_over_flair_rms": 0.5,
                    "delta_mse": 0.1,
                    "on_cosine": 0.99,
                    "subject_centered_on_cosine": 0.40,
                    "delta_wrong_subject_cosine": rng.gauss(0, 0.03),
                    "delta_to_subject_variation_t1": frac**2,
                    "delta_to_subject_variation_flair": frac**2,
                }
            )
        # One dominant channel carrying the alignment, nine near-zero ones.
        for c in range(10):
            chans.append(
                {
                    "distribution": "iid",
                    "batch": 0,
                    "stage": stage,
                    "channel": c,
                    "cosine": target if c == 0 else rng.gauss(0, 0.5),
                    "t1_rms": 1.0 if c == 0 else 0.02,
                    "flair_rms": 1.0 if c == 0 else 0.02,
                    "relative_mse": 0.5,
                    "t1_over_flair_rms": 1.0,
                    "mse": 0.1,
                }
            )
    bt = [
        {
            "distribution": "iid",
            "batch": 0,
            "n": n,
            "stage": s,
            "on": {"instantaneous_weighted_total": 10.0, "pos_sim_mean": 0.8},
            "off": {"instantaneous_weighted_total": 10.0 - d, "pos_sim_mean": 0.8},
        }
        for s, d in (("shared_big", 2.0), ("null_big", 0.001))
    ]

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name, data in (("samples.csv", rows), ("channels.csv", chans)):
            with (d / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
        (d / "summary.json").write_text(json.dumps({"arguments": {"level": 0, "num_samples": n}, "batch_bt_terms": bt}))
        got = report(d)["distributions"]["iid"]

    print("\nself-test")
    checks = [
        ("shared_big  -> aligned", got["shared_big"]["verdict"] == "aligned"),
        ("null_big    -> at null", got["null_big"]["verdict"] == "at null (no shared encoding)"),
        ("anti_big    -> ANTI-ALIGNED", got["anti_big"]["verdict"] == "ANTI-ALIGNED"),
        ("shared_tiny -> UNDEFINED despite dcos 0.95", got["shared_tiny"]["verdict"].startswith("UNDEFINED")),
        (
            "dcos CI brackets the planted 0.95",
            got["shared_big"]["delta_cosine_ci"][0] < 0.95 < got["shared_big"]["delta_cosine_ci"][1],
        ),
        ("null stage is not above its null", got["null_big"]["sign_p"] > 0.05),
        ("response % recovered as 30%", abs(got["shared_big"]["response_pct_of_subject_spread"]["t1"] - 30.0) < 0.1),
        ("paired stage delta negative for anti", got["anti_big"]["dcos_change_vs_first_stage"] < -1.5),
        ("BT sees shared_big (20%)", abs(got["shared_big"]["bt"]["percent_of_loss"] - 20.0) < 0.1),
        ("BT blind to null_big (~0%)", abs(got["null_big"]["bt"]["percent_of_loss"]) < 0.1),
        ("one channel holds 80% of energy", got["shared_big"]["channels"]["channels_for_80pct_energy"] == 1),
        ("energy-weighted cos follows it", abs(got["shared_big"]["channels"]["energy_weighted_cosine"] - 0.95) < 0.05),
    ]
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not all(ok for _, ok in checks):
        raise SystemExit("self-test failed")
    print("  all checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", nargs="?", help="A lesion_alignment_<causal>_L<level> output directory")
    p.add_argument("--min-response-pct", type=float, default=1.0, help="Below this, cosines are called undefined")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    p.add_argument("--out", default=None, help="Also write the aggregates as JSON")
    p.add_argument("--self-test", action="store_true", help="Run on planted rows; needs no run directory")
    cli = p.parse_args()
    if cli.self_test:
        _self_test()
        return
    if not cli.directory:
        p.error("Need a directory, or --self-test")
    result = report(cli.directory, cli.min_response_pct, cli.seed)
    if cli.out:
        Path(cli.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {cli.out}")


if __name__ == "__main__":
    main()
