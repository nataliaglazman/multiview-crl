"""Diagnosis, sex and race probes at an encoder-only ADNI run's best checkpoint, against its init.

Reads what ``training.main_conv_synthetic`` already logged on real data, so nothing is
re-scored: either the run directory (``separation_step<N>.json`` + ``best_checkpoint.json``)
or the trainer's stdout (SLURM ``.out``, ``runai logs``). The step-0 evaluation is the init:
the same weights as ``model_init.pt``, scored by the same subject-disjoint probes
(``eval.adni.encoder_probes``) on the same held-out subjects as every later step.

A run directory gives balanced and plain accuracy for T1, FLAIR and pooled content. Stdout
prints per-view rows only, to 3 decimals: balanced accuracy, plus plain accuracy in logs from
trainers before 14 Oct 2026 (the ``REAL_SUMMARY`` table; later ones print the ``↑`` report).

    python -m eval.adni.label_probes_vs_init results/encoder_adni/runs/encoder_adni_conv_mlp_s42
    python -m eval.adni.label_probes_vs_init results/encoder_adni/runs --csv out/label_probes.csv
    python -m eval.adni.label_probes_vs_init /scratch/users/$USER/12345.out --step last
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

TARGETS = (("diagnosis", "diagnosis"), ("gender", "sex"), ("race", "race"))
READOUTS = (("_v0", "T1"), ("_v1", "FLAIR"), ("", "pooled"))
VIEW_SUFFIX = {"T1": "_v0", "FLAIR": "_v1"}

EVAL_RE = re.compile(r"\[eval\] held-out subjects @ step (\d+)")
LABEL_ROW_RE = re.compile(
    r"^\s*(?:↑ )?(diagnosis|gender|race) (balanced acc|probe), (T1|FLAIR) content\s+(n/a|nan|[-+0-9.eE]+)"
)
ROW_KIND = {"balanced acc": "_balanced_acc", "probe": "_acc"}
LABEL_STATS_RE = re.compile(
    r"^\s*(\w+): (\d+) pairs / (\d+) subjects, (\d+) classes, (\d+) folds;.*"
    r"majority=([-+0-9.na]+), balanced chance=([-+0-9.na]+)"
)
NEW_BEST_RE = re.compile(r"new best (\S+) (\S+) -> model_best\.pt")
SETTING_RE = re.compile(r"^\t(model_id|best_metric): (.*)$")


def _float(text):
    return float("nan") if text in ("n/a", "nan") else float(text)


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fp:
        return json.load(fp)


def _get(flat, key):
    """Unsuffixed key, else the level-0 copy cross_reconstruction also writes."""
    value = flat.get(key, flat.get(f"{key}_L0"))
    return float(value) if value is not None else float("nan")


def _choose_step(steps, requested, best_step, source):
    if requested == "best":
        if best_step is None:
            raise ValueError(f"{source}: no best checkpoint recorded (--best-metric none?); pass --step last or N")
        step = best_step
    elif requested == "last":
        step = max(s for s in steps if s > 0) if any(s > 0 for s in steps) else None
        if step is None:
            raise ValueError(f"{source}: no evaluation after step 0")
    else:
        step = int(requested)
    if step not in steps:
        raise ValueError(f"{source}: no evaluation at step {step} (have {sorted(steps)})")
    return step


def load_run_dir(run_dir, requested="best"):
    """Init (step 0) and chosen-step metrics from a run directory's separation JSONs."""
    files = {
        int(re.search(r"separation_step(\d+)\.json$", path).group(1)): path
        for path in glob.glob(os.path.join(run_dir, "separation_step*.json"))
    }
    if 0 not in files:
        raise ValueError(f"{run_dir}: no separation_step0.json (run without --floor-eval?), so no init to compare")
    best = _read_json(os.path.join(run_dir, "best_checkpoint.json"), {})
    progress = _read_json(os.path.join(run_dir, "training_progress.json"), {})
    settings = _read_json(os.path.join(run_dir, "settings.json"), {})
    step = _choose_step(files, requested, best.get("step"), run_dir)
    return {
        "name": os.path.basename(os.path.normpath(run_dir)),
        "source": "run dir",
        "path": run_dir,
        "step": step,
        "last_step": max(files),
        "status": progress.get("status"),
        "best_step": best.get("step"),
        "best_metric": best.get("metric", settings.get("best_metric")),
        "best_raw": best.get("raw"),
        "init": _read_json(files[0]),
        "chosen": _read_json(files[step]),
    }


def parse_log(path, requested="best"):
    """The same, from trainer stdout: the per-view label rows and each probe's coverage line."""
    evals, settings, current = {}, {}, None
    best_step = best_raw = best_metric = None
    with open(path, errors="replace") as fp:
        for line in fp:
            line = line.rstrip("\n")
            match = SETTING_RE.match(line)
            if match:
                settings[match.group(1)] = match.group(2).strip()
                continue
            match = EVAL_RE.search(line)
            if match:
                current = int(match.group(1))
                if current == 0 and evals:
                    # A restarted job re-runs the floor; everything before it belongs to the old attempt.
                    print(f"  NOTE: {path} holds more than one run; reading the last.", file=sys.stderr)
                    evals, best_step, best_raw, best_metric = {}, None, None, None
                evals[current] = {}
                continue
            if current is None:
                continue
            match = LABEL_ROW_RE.match(line)
            if match:
                target, kind, view, value = match.groups()
                evals[current][f"content/{target}_probe{ROW_KIND[kind]}{VIEW_SUFFIX[view]}"] = _float(value)
                continue
            match = LABEL_STATS_RE.match(line)
            if match:
                target, _, subjects, classes, folds, majority, balanced = match.groups()
                prefix = f"content/{target}_probe"
                evals[current].update(
                    {
                        f"{prefix}_subjects": float(subjects),
                        f"{prefix}_classes": float(classes),
                        f"{prefix}_folds": float(folds),
                        f"{prefix}_chance": _float(majority),
                        f"{prefix}_balanced_chance": _float(balanced),
                    }
                )
                continue
            match = NEW_BEST_RE.search(line)
            if match:
                best_step, best_raw, best_metric = current, _float(match.group(2)), match.group(1)
    if 0 not in evals:
        raise ValueError(f"{path}: no step-0 evaluation in the log (run without --floor-eval?), so no init to compare")
    if not any(key.endswith("_balanced_acc_v0") for key in evals[0]):
        raise ValueError(f"{path}: no diagnosis/demographic probe rows; is this an encoder ADNI run's stdout?")
    step = _choose_step(evals, requested, best_step, path)
    return {
        "name": settings.get("model_id", os.path.basename(path)),
        "source": "stdout log",
        "path": path,
        "step": step,
        "last_step": max(evals),
        "status": None,
        "best_step": best_step,
        "best_metric": best_metric or settings.get("best_metric"),
        "best_raw": best_raw,
        "init": evals[0],
        "chosen": evals[step],
    }


def load(path, requested="best"):
    if os.path.isdir(path):
        return load_run_dir(path, requested)
    return parse_log(path, requested)


def expand(paths):
    """Run dirs and log files as given; any other directory is searched for run dirs."""
    out = []
    for path in paths:
        if not os.path.isdir(path) or os.path.exists(os.path.join(path, "separation_step0.json")):
            out.append(path)
            continue
        found = sorted(
            os.path.dirname(hit) for hit in glob.glob(os.path.join(path, "**", "separation_step0.json"), recursive=True)
        )
        if not found:
            raise ValueError(f"{path}: not a run dir and no run dirs (separation_step0.json) underneath")
        out.extend(found)
    return out


def rows(run):
    """One row per target x readout: init vs chosen step, with the probe's coverage and chance."""
    init, chosen, out = run["init"], run["chosen"], []
    for key, target in TARGETS:
        prefix = f"content/{key}_probe"
        subjects = _get(chosen, f"{prefix}_subjects")
        if np.isfinite(_get(init, f"{prefix}_subjects")) and _get(init, f"{prefix}_subjects") != subjects:
            print(
                f"  WARNING: {run['name']}: {target} probe covers different subjects at init and step {run['step']}",
                file=sys.stderr,
            )
        class_counts = [
            float(v)
            for k, v in chosen.items()
            if re.fullmatch(rf"{re.escape(prefix)}_class_\w+_subjects", k) and float(v) >= 2
        ]
        for suffix, readout in READOUTS:
            row = {
                "run": run["name"],
                "source": run["source"],
                "step": run["step"],
                "init_step": 0,
                "selected_on": run["best_metric"] if run["step"] == run["best_step"] else None,
                "target": target,
                "content": readout,
                "subjects": subjects,
                "classes": _get(chosen, f"{prefix}_classes"),
                "smallest_class": min(class_counts) if class_counts else float("nan"),
                "balanced_chance": _get(chosen, f"{prefix}_balanced_chance"),
                "majority": _get(chosen, f"{prefix}_chance"),
                "init_balanced_acc": _get(init, f"{prefix}_balanced_acc{suffix}"),
                "best_balanced_acc": _get(chosen, f"{prefix}_balanced_acc{suffix}"),
                "init_acc": _get(init, f"{prefix}_acc{suffix}"),
                "best_acc": _get(chosen, f"{prefix}_acc{suffix}"),
            }
            if not np.isfinite([row["init_balanced_acc"], row["best_balanced_acc"]]).any():
                continue
            row["delta_balanced_acc"] = row["best_balanced_acc"] - row["init_balanced_acc"]
            row["delta_acc"] = row["best_acc"] - row["init_acc"]
            p = row["best_acc"] if np.isfinite(row["best_acc"]) else row["best_balanced_acc"]
            row["se"] = float(np.sqrt(p * (1 - p) / subjects)) if subjects > 0 and np.isfinite(p) else float("nan")
            out.append(row)
    return out


def _cell(value, width, fmt=".3f"):
    if value is None or not np.isfinite(value):
        return f"{'—':>{width}}"
    sign, fmt = ("+", fmt[1:]) if fmt.startswith("+") else ("", fmt)
    return f"{value:>{sign}{width}{fmt}}"


def print_run(run, table):
    which = f"step {run['step']}"
    if run["step"] == run["best_step"]:
        raw = f" = {run['best_raw']:.4f}" if run["best_raw"] is not None else ""
        which = f"best checkpoint, step {run['step']} (selected on {run['best_metric']}{raw}, not on these probes)"
    elif run["best_step"] is not None:
        which += f" (best checkpoint is step {run['best_step']})"
    status = f", {run['status']}" if run["status"] not in (None, "complete") else ""
    print(f"\n{run['name']}  [{run['source']}: {run['path']}]")
    print(f"  {which}; last eval step {run['last_step']}{status}")
    print("  init = step-0 eval of the untrained encoder (model_init.pt), same subjects and folds")
    print(f"  {'':<32}{'balanced accuracy':^33}{'accuracy':^35}".rstrip())
    print(
        f"  {'target':<10}{'content':<8}{'subj':>5}{'cls':>4}{'min':>5}"
        f"{'chance':>8}{'init':>8}{'best':>8}{'Δ':>9}"
        f"{'majority':>10}{'init':>8}{'best':>8}{'Δ':>9}{'±SE':>7}"
    )
    previous = None
    for row in table:
        target = row["target"] if row["target"] != previous else ""
        previous = row["target"]
        print(
            f"  {target:<10}{row['content']:<8}{_cell(row['subjects'], 5, '.0f')}{_cell(row['classes'], 4, '.0f')}"
            f"{_cell(row['smallest_class'], 5, '.0f')}"
            f"{_cell(row['balanced_chance'], 8)}{_cell(row['init_balanced_acc'], 8)}"
            f"{_cell(row['best_balanced_acc'], 8)}{_cell(row['delta_balanced_acc'], 9, '+.3f')}"
            f"{_cell(row['majority'], 10)}{_cell(row['init_acc'], 8)}{_cell(row['best_acc'], 8)}"
            f"{_cell(row['delta_acc'], 9, '+.3f')}{_cell(row['se'], 7)}"
        )


def print_summary(runs, tables):
    """Δ balanced accuracy vs init, one line per run, when several are compared."""
    columns = [(target, readout) for _, target in TARGETS for _, readout in READOUTS]
    width = max(len(run["name"]) for run in runs) + 2
    print("\nΔ balanced accuracy vs init, per run (best checkpoint unless --step)")
    print(f"  {'run':<{width}}{'step':>7}" + "".join(f"{f'{t[:4]} {r}':>13}" for t, r in columns))
    for run, table in zip(runs, tables):
        by_key = {(row["target"], row["content"]): row["delta_balanced_acc"] for row in table}
        print(
            f"  {run['name']:<{width}}{run['step']:>7}"
            + "".join(_cell(by_key.get(column), 13, "+.3f") for column in columns)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="run dirs, parent dirs to search for run dirs, or trainer stdout logs")
    parser.add_argument("--step", default="best", help="'best' (model_best.pt's step), 'last', or an eval step")
    parser.add_argument("--csv", help="write every row (long format) to this CSV")
    args = parser.parse_args(argv)
    if args.step not in ("best", "last") and not args.step.isdigit():
        parser.error("--step must be 'best', 'last' or an integer step")

    runs, tables = [], []
    for path in expand(args.paths):
        try:
            run = load(path, args.step)
        except ValueError as error:
            print(f"skipping: {error}", file=sys.stderr)
            continue
        runs.append(run)
        tables.append(rows(run))
        print_run(run, tables[-1])
    if not runs:
        raise SystemExit("no readable runs")
    if len(runs) > 1:
        print_summary(runs, tables)
    print(
        "\nsubj: labelled held-out subjects; cls: classes probed; min: subjects in the smallest probed class "
        "(a class with one subject is dropped). chance: 1/cls; majority: share of the largest class."
    )
    print(
        "±SE: binomial standard error of one accuracy at that many subjects. Init and best are scored on the "
        "same subjects, so it is a rough yardstick for Δ, not a test; |Δ| under ~2 SE is within probe noise."
    )
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fields = next(list(table[0]) for table in tables if table)
        with open(args.csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for table in tables:
                writer.writerows(table)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
