"""Show what the optimiser ACTUALLY receives from every loss term, over training.

TensorBoard logs the Barlow Twins terms UNWEIGHTED (losses.py: "Logged unweighted so the
four terms can be balanced from what they actually are"). So `Contrastive/off_diag_loss_L0`
is the raw off-diagonal sum, not its contribution to the loss — you have to multiply by
scale_contrastive_loss * bt_{patch,gap}_weight * bt_{,gap_}lambda to get that. Same for the
reconstruction pieces. This applies every coefficient from the run's own settings.json and
prints the weighted contributions, so "what is my model optimising" is answerable directly.

It also reconciles: the weighted parts are summed and compared against the logged
Loss/Total. A residual near zero means the accounting below is complete; a large residual
means a term is missing from this script and the percentages are not trustworthy.

Usage:
    python -m eval.loss_breakdown <run_dir>
    python -m eval.loss_breakdown <run_dir> --steps 1,2000,20000,40000
    python -m eval.loss_breakdown <run_dir> --csv breakdown.csv
"""

import argparse
import glob
import json
import os

import numpy as np


def _load_scalars(logdir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise SystemExit(f"Needs tensorboard (pip install tensorboard). {e}")
    files = [p for p in glob.glob(os.path.join(logdir, "**", "*"), recursive=True) if "tfevents" in os.path.basename(p)]
    if not files:
        raise SystemExit(f"No tfevents files under {logdir}")
    series = {}
    for f in sorted(files):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            ev = acc.Scalars(tag)
            s = np.array([e.step for e in ev], float)
            v = np.array([e.value for e in ev], float)
            if tag in series:
                s = np.concatenate([series[tag][0], s])
                v = np.concatenate([series[tag][1], v])
            o = np.argsort(s)
            series[tag] = (s[o], v[o])
    return series


def _at(series, tag, step, tol):
    """Value of `tag` nearest `step`, or None if absent / too far away."""
    if tag not in series:
        return None
    s, v = series[tag]
    i = int(np.argmin(np.abs(s - step)))
    return float(v[i]) if abs(s[i] - step) <= tol else None


def _settings(run_dir, override):
    path = override or os.path.join(run_dir, "settings.json")
    if not os.path.exists(path):
        raise SystemExit(f"No settings.json at {path} (pass --settings). Coefficients come from it.")
    with open(path) as f:
        return json.load(f)


def build_terms(cfg, level):
    """(display name, tb tag, multiplier) for every term that reaches the loss.

    Multipliers mirror the call chain exactly:
      contrastive: scale_contrastive_loss * bt_{patch,gap}_weight * <term coefficient>
      recon:       scale_recon_loss                (the 0.002 perceptual factor is ALREADY
                                                    inside Loss-Perceptual-Reconstruction)
      commitment:  counted inside BaselineLoss (weight scale_recon_loss) UNLESS
                   --single-count-commitment, PLUS vq_commitment_weight as Loss/VQ.
    """
    g = cfg.get
    sc = float(g("scale_contrastive_loss", 1.0) or 0.0)
    sr = float(g("scale_recon_loss", 1.0) or 0.0)
    pw = float(g("bt_patch_weight", 1.0))
    gw = float(g("bt_gap_weight", 0.0) or 0.0)
    lam = float(g("bt_lambda", 0.005))
    glam = g("bt_gap_lambda", None)
    glam = lam if glam is None else float(glam)
    sim = float(g("bt_sim_coeff", 0.0) or 0.0)
    std = float(g("bt_std_coeff", 0.0) or 0.0)
    gstd = g("bt_gap_std_coeff", None)
    gstd = std if gstd is None else float(gstd)
    lvlw = g("contrastive_level_weights", None)
    lw = float(lvlw[level]) if lvlw and level < len(lvlw) else 1.0
    single = bool(g("single_count_commitment", False))
    vqw = float(g("vq_commitment_weight", 0.25))

    L = f"_L{level}"
    terms = [
        ("BT patch  on_diag", f"Contrastive/on_diag_loss{L}", sc * lw * pw * 1.0),
        ("BT patch  off_diag", f"Contrastive/off_diag_loss{L}", sc * lw * pw * lam),
        ("BT patch  sim", f"Contrastive/sim_loss{L}", sc * lw * pw * sim),
        ("BT patch  var hinge", f"Contrastive/var_loss{L}", sc * lw * pw * std),
        ("BT gap    on_diag", f"Contrastive/gap_on_diag_loss{L}", sc * lw * gw * 1.0),
        ("BT gap    off_diag", f"Contrastive/gap_off_diag_loss{L}", sc * lw * gw * glam),
        ("BT gap    sim", f"Contrastive/gap_sim_loss{L}", sc * lw * gw * sim),
        ("BT gap    var hinge", f"Contrastive/gap_var_loss{L}", sc * lw * gw * gstd),
        ("recon     pixel L1", "Recon/Loss-MAE-Reconstruction", sr),
        ("recon     perceptual", "Recon/Loss-Perceptual-Reconstruction", sr),
    ]
    commit_w = vqw + (0.0 if single else sr)
    terms.append(("VQ        commitment", f"Recon/Loss-MSE-VQ0_Commitment_Cost", commit_w))
    return terms, commit_w, single


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--steps", default=None, help="Comma-separated steps (default: 6 spread over the run)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--tol", type=float, default=None, help="Max step distance when matching a tag")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = _settings(args.run_dir, args.settings)
    series = _load_scalars(args.run_dir)
    terms, commit_w, single = build_terms(cfg, args.level)

    if "Loss/Total" not in series:
        raise SystemExit("Loss/Total not found — is this a training log directory?")
    steps_all = series["Loss/Total"][0]
    tol = args.tol if args.tol is not None else max(1.0, 0.01 * (steps_all.max() - steps_all.min()))
    if args.steps:
        steps = [float(x) for x in args.steps.split(",")]
    else:
        steps = list(np.linspace(steps_all.min(), steps_all.max(), 6))

    print(f"\nrun: {args.run_dir}")
    print(
        f"commitment effective weight = {commit_w:g}"
        f"  ({'single-count' if single else 'DOUBLE-COUNTED: scale_recon_loss + vq_commitment_weight'})"
    )

    rows = []
    for st in steps:
        vals, total_logged = {}, _at(series, "Loss/Total", st, tol)
        for name, tag, mult in terms:
            raw = _at(series, tag, st, tol)
            vals[name] = (raw, mult, None if raw is None else raw * mult)
        present = {k: v[2] for k, v in vals.items() if v[2] is not None}
        ssum = sum(present.values())
        print(
            f"\n{'='*82}\nstep ~{st:.0f}    Loss/Total (logged) = "
            f"{'n/a' if total_logged is None else f'{total_logged:.4f}'}\n{'='*82}"
        )
        print(f"  {'term':<24}{'raw (tb)':>12}{'x coeff':>11}{'= weighted':>13}{'% of sum':>10}")
        for name, (raw, mult, w) in vals.items():
            if raw is None:
                print(f"  {name:<24}{'--- not logged ---':>36}")
                continue
            print(f"  {name:<24}{raw:>12.5g}{mult:>11.5g}{w:>13.5g}{100*w/ssum if ssum else 0:>9.2f}%")
        print(f"  {'-'*80}\n  {'SUM of weighted parts':<24}{'':>23}{ssum:>13.5g}")
        missing = [n for n, (raw, _, _) in vals.items() if raw is None]
        if total_logged is not None:
            resid = total_logged - ssum
            if missing:
                # Runs from before the Recon/* logging have no pixel/perceptual/commitment
                # tags, so the residual IS those terms — not an accounting error. Say so,
                # and attribute it, rather than crying MISMATCH at a complete breakdown.
                flag = f"= the {len(missing)} unlogged term(s): {', '.join(m.split()[-1] for m in missing)}"
            elif abs(resid) <= 0.02 * max(abs(total_logged), 1e-9):
                flag = "OK — breakdown is complete"
            else:
                flag = "MISMATCH — a logged term is unaccounted for"
            print(f"  {'Loss/Total logged':<24}{'':>23}{total_logged:>13.5g}")
            print(f"  {'residual':<24}{'':>23}{resid:>13.5g}   {flag}")
            if missing and abs(total_logged) > 1e-12:
                print(f"  {'-> contrastive share':<24}{'':>23}{100*ssum/total_logged:>12.1f}%")
                print(f"  {'-> residual share':<24}{'':>23}{100*resid/total_logged:>12.1f}%")
        r = {"step": st, "Loss/Total": total_logged, "sum_weighted": ssum}
        r.update({k: v[2] for k, v in vals.items()})
        rows.append(r)

    if args.csv:
        import csv as _csv

        cols = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
