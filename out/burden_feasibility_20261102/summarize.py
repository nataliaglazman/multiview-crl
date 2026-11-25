"""Print every finished or running arm of run_pc.sh as compact text, small enough to paste back.

    python out/burden_feasibility_20261102/summarize.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def supervised(name, report):
    history = report["history"]
    if not history or "val" not in history[0]:
        return False
    last = history[-1]
    rise = next(
        (h["step"] for h in history if max(h["val"]["t1"]["pearson2"], h["val"]["flair"]["pearson2"]) > 0.1), None
    )
    source = report.get("test") or last["val"]
    label = "test" if "test" in report else f"val@{last['step']}"
    t, f = source["t1"], source["flair"]
    train = last["train400"]
    print(
        f"{name:22s} {label:9s} calib R2 {t['pearson2']:.3f} {f['pearson2']:.3f} | rho {t['spearman']:+.3f} {f['spearman']:+.3f} | "
        f"train400 {train['t1']['pearson2']:.3f} {train['flair']['pearson2']:.3f} | step {last['step']} | first > 0.1: {rise}"
    )
    bins = [k for k in t if k.startswith("spearman_")]
    if "test" in report and bins:
        print("   by size, T1 / FLAIR: " + "  ".join(f"{k[9:]}+ {t[k]:+.2f}/{f.get(k, 0):+.2f}" for k in bins))
    return True


def ssl(name, report):
    history = report["history"]
    if not history or "test" not in history[0]:
        return False
    print(name)
    for row in history:
        t, f = row["test"]["t1"], row["test"]["flair"]
        pooled = f" pool {t['pooled_volume_r2']:+.3f} {f['pooled_volume_r2']:+.3f} |" if "pooled_volume_r2" in t else ""
        loss = row["loss"].get("global")
        extra = "".join(f" {k} {row['loss'][k]:.3f}" for k in ("patch", "decor", "remainder") if k in row["loss"])
        spread = row.get("spread")
        spread = f" | nbhd spread anat {spread['anatomy']:.2f} pos {spread['lesion_position']:.2f}" if spread else ""
        rest = row.get("remainder_spread")
        if rest:
            spread += (
                f" | remainder nbhd anat {rest['anatomy']:.2f} pos {rest['lesion_position']:.2f}"
                f" burden {rest['burden']:.2f}"
            )
        print(
            f"  s{row['step']:5d} code {t['code_volume_r2']:+.3f} {f['code_volume_r2']:+.3f} | GAP {t['gap_volume_r2']:+.3f} "
            f"{f['gap_volume_r2']:+.3f} |{pooled} brain {t['code_brain']:.2f} vent {t['code_ventricle']:.2f} thick "
            f"{t['code_thickness']:.2f} temp {t['code_temporal']:.2f} lr {t['code_lr']:.2f} sulc {t['code_sulcal']:.2f} | "
            f"global {'' if loss is None else format(loss, '.3f')}{extra}{spread}"
        )
    return True


def real(name, report):
    """A warm.py --real (ADNI) run: label-free readouts on validation subjects; the step-0 row is the untrained floor,
    except in a run that only evaluates another run's checkpoint (--init-run with 0 steps)."""
    start = report.get("args", {}).get("init_run", "None")
    print(name if start == "None" else f"{name} (evaluates {start})")
    for row in report["history"]:
        h = row.get("heldout")
        if not h:
            continue
        loss = row["loss"].get("global")
        predict = f" predict {row['loss']['predict']:.3f}" if "predict" in row["loss"] else ""
        print(
            f"  s{row['step']:5d} top-1 val {h['top1']['mean']:.3f} train {h['train_top1']['mean']:.3f} "
            f"@32 {h['top1_32']:.3f} | GAP {h['gap_top1']:.3f} | held-out {h['heldout_loss']:.3f} | rank "
            f"{h['effective_rank']['t1']:.1f} {h['effective_rank']['flair']:.1f} | unit r {h['cross_view_code_r']:.2f} | "
            f"modality {h['modality_probe']['normalized_linear']:.2f} {h['modality_probe']['normalized_rbf']:.2f} "
            f"{h['modality_probe']['mean_centered_rbf']:.2f} | global {'' if loss is None else format(loss, '.3f')}{predict}"
        )
        parts = []
        for kind, r in h.get("readouts", {}).items():
            among = f"{r['top1']['mean']:.3f} " if "top1" in r else ""
            centred = f" c {r['modality_probe']['mean_centered_rbf']:.2f}" if "modality_probe" in r else ""
            parts.append(f"{kind} {among}@32 {r['top1_32']:.3f}{centred}")
        if parts:
            print("         " + " | ".join(parts))


def dense(path):
    report = load(path)
    if report is None:
        return
    print(os.path.basename(os.path.dirname(path)), "step", report["step"])
    for view in ("t1", "flair"):
        row = report[view]
        keys = ("sum_all", "sum_above_0.1", "sum_above_0.2", "top8")
        print(
            f"  {view:5s} cell AUROC {row['cell_auroc']:.3f} top cell {row['top_cell_hit']:.2f} GAP ridge {row['gap_ridge_volume']:+.3f} | "
            + " ".join(f"{k} {row['pools'][k][1]:.3f}" for k in keys)
        )
    print("  cross-view r", report.get("cross_view_r"))


def main():
    names = sorted(os.path.basename(p) for p in glob.glob(f"{HERE}/runs/*"))
    reports = {n: load(f"{HERE}/runs/{n}/report.json") for n in names}
    reports = {n: r for n, r in reports.items() if r}
    print("== supervised: held-out calibrated R2 and Spearman, T1 then FLAIR (dense arms are read below instead)")
    for name, report in reports.items():
        if not name.startswith(("ssl", "dense")):
            supervised(name, report)
    print("\n== dense-label detectors")
    for path in sorted(glob.glob(f"{HERE}/runs/dense*/dense_eval_s*.json")):
        dense(path)
    print("\n== self-supervised, no labels: burden R2 from the 14 units, T1 then FLAIR, on test subjects")
    for name, report in reports.items():
        if name.startswith("ssl"):
            ssl(name, report)
    adni = sorted(glob.glob(f"{HERE}/adni/runs/*/report.json"))
    if adni:
        print(
            "\n== ADNI (warm.py --real), validation subjects: T1<->FLAIR top-1 among all, on as many training subjects"
            " and in sets of 32; backbone GAP top-1; held-out loss; effective rank T1 FLAIR; matching-unit r; modality"
            " probe linear, RBF, mean-centred RBF (chance .5). Readout rows (warm.py gauge_readouts): the code with one"
            " difference from the loss's undone, as top-1 among all, @32 and centred probe (c); batch_* use each set's"
            " own gauge statistics, *_jittered training's jitter, recalibrated the gauge recomputed on clean training"
            " images"
        )
        for path in adni:
            report = load(path)
            if report:
                real(os.path.basename(os.path.dirname(path)), report)
    report = load(f"{HERE}/bank_nuisance/residual.json")
    if report:
        print("\n== fixed filter on the normative residual, nuisance fields on: R2, then cross-view r with FLAIR")
        for k in (0, 16, 64):
            shared = report[f"t1_04_vs_flair_k{k}"]
            best = shared["corr_with_burden_a_b"][shared["best_direction"]]
            print(
                f"  K {k:2d} T1 0.4 R2 {report[f't1_04_k{k}']['r2']:.3f} FLAIR {report[f'flair_k{k}']['r2']:.3f} | "
                f"cross-view r {shared['cross_view_r']:.3f} | best label-free direction, corr with burden {best}"
            )
    if os.path.exists(f"{HERE}/audit.txt"):
        print()
        with open(f"{HERE}/audit.txt") as stream:
            print(
                "".join(line for line in stream if not line.startswith(("auditing", "skipping", "mode check"))), end=""
            )


if __name__ == "__main__":
    main()
