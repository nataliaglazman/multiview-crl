"""Run-dir and stdout readers agree, pick the best checkpoint's step, and compare it to step 0."""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest

import numpy as np

from eval.adni import label_probes_vs_init as lp
from eval.adni.encoder_alignment import alignment_metrics, print_alignment_report
from eval.adni.encoder_probes import evaluate_subject_probes

N = 36
DIAGNOSIS = np.arange(N) % 3
GENDER = np.where(np.arange(N) % 2, 1, 2)
RACE = np.where(np.arange(N) % 4 == 0, 4, 5)
RACE[:2] = -1


def flat(informative, seed):
    rng = np.random.default_rng(seed)
    signal = np.column_stack([DIAGNOSIS == k for k in range(3)] + [GENDER == 1, RACE == 4]).astype(float)
    reps = {}
    for view in (0, 1):
        noise = rng.normal(size=(N, signal.shape[1] + 3))
        reps[f"content_v{view}"] = noise + (3 * np.pad(signal, ((0, 0), (0, 3))) if informative else 0)
    reps.update(
        subjects=np.array([f"s{i}" for i in range(N)]),
        labels=DIAGNOSIS,
        demographics={"gender": GENDER, "race": RACE},
    )
    metrics = alignment_metrics(reps)
    metrics.update(evaluate_subject_probes(reps))
    metrics["val/loss"] = 5.0 - seed
    return metrics


STEPS = {0: flat(False, 0), 500: flat(True, 1), 1000: flat(True, 2)}


def stdout_log(best_step=500, steps=STEPS):
    """What main_conv_synthetic prints around evaluate_real, through the real report printer."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print("\tmodel_id: encoder_adni_test")
        print("\tbest_metric: val_loss")
        previous = None
        for step, metrics in steps.items():
            print(f"  [eval] held-out subjects @ step {step} ...")
            print_alignment_report(metrics, floor=steps[0] if step else None, previous=previous)
            previous = metrics
            if step == best_step:
                print(f"    new best val_loss {metrics['val/loss']:.4f} -> model_best.pt")
        print(f"best val_loss {steps[best_step]['val/loss']:.4f} at step {best_step} -> model_best.pt")
    return out.getvalue()


def old_stdout_log(best_step=500, steps=STEPS):
    """The REAL_SUMMARY table trainers printed before 14 Oct 2026 (fbc9891..1ad585a): no arrows, plain acc too."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print("\tmodel_id: encoder_adni_test")
        for step, metrics in steps.items():
            print(f"  [eval] held-out subjects @ step {step} ...")
            print("    --- held-out subjects ---")
            for target in ("diagnosis", "gender", "race"):
                for kind, key in (("probe", "acc"), ("balanced acc", "balanced_acc")):
                    for view, name in enumerate(("T1", "FLAIR")):
                        value, floor = (m[f"content/{target}_probe_{key}_v{view}"] for m in (metrics, steps[0]))
                        line = f"      {f'{target} {kind}, {name} content':<40s}{value:>9.3f}"
                        print(line + (f"{floor:>9.3f}{value - floor:>+12.3f}" if step else ""))
            print("      modality probe, content (0.5 = none)       0.612    0.500      +0.112")
            for target in ("diagnosis", "gender", "race"):
                p = f"content/{target}_probe"
                print(
                    f"      {target}: {metrics[p + '_n']:.0f} pairs / {metrics[p + '_subjects']:.0f} subjects, "
                    f"{metrics[p + '_classes']:.0f} classes, {metrics[p + '_folds']:.0f} folds; "
                    f"excluded missing=0, rare=0, conflicting=0; "
                    f"majority={metrics[p + '_chance']:.3f}, balanced chance={metrics[p + '_balanced_chance']:.3f}"
                )
            if step == best_step:
                print(f"    new best val_loss {metrics['val/loss']:.4f} -> model_best.pt")
    return out.getvalue()


class LabelProbesVsInitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.run_dir = os.path.join(self.root, "runs", "encoder_adni_test")
        os.makedirs(self.run_dir)
        for step, metrics in STEPS.items():
            with open(os.path.join(self.run_dir, f"separation_step{step}.json"), "w") as fp:
                json.dump(metrics, fp)
        with open(os.path.join(self.run_dir, "best_checkpoint.json"), "w") as fp:
            json.dump({"step": 500, "metric": "val_loss", "value": -4.0, "raw": 4.0}, fp)
        self.log = os.path.join(self.root, "job.out")
        with open(self.log, "w") as fp:
            fp.write(stdout_log())

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_dir_compares_best_step_with_step_zero(self):
        run = lp.load(self.run_dir)
        self.assertEqual((run["step"], run["best_step"], run["last_step"]), (500, 500, 1000))
        table = {(r["target"], r["content"]): r for r in lp.rows(run)}
        self.assertEqual(len(table), 9)
        row = table[("diagnosis", "T1")]
        self.assertEqual(row["init_balanced_acc"], STEPS[0]["content/diagnosis_probe_balanced_acc_v0"])
        self.assertEqual(row["best_balanced_acc"], STEPS[500]["content/diagnosis_probe_balanced_acc_v0"])
        self.assertAlmostEqual(row["delta_balanced_acc"], row["best_balanced_acc"] - row["init_balanced_acc"])
        self.assertGreater(row["delta_balanced_acc"], 0.3)
        self.assertEqual(table[("race", "pooled")]["subjects"], N - 2)
        self.assertEqual(table[("sex", "FLAIR")]["smallest_class"], N / 2)
        self.assertEqual(table[("race", "pooled")]["selected_on"], "val_loss")

    def test_stdout_log_matches_run_dir_to_printed_precision(self):
        from_dir = {(r["target"], r["content"]): r for r in lp.rows(lp.load(self.run_dir))}
        run = lp.load(self.log)
        self.assertEqual((run["name"], run["step"], run["best_step"]), ("encoder_adni_test", 500, 500))
        self.assertEqual(run["best_metric"], "val_loss")
        from_log = {(r["target"], r["content"]): r for r in lp.rows(run)}
        self.assertEqual(set(from_log), {(t, v) for _, t in lp.TARGETS for v in ("T1", "FLAIR")})
        for key, row in from_log.items():
            for field in ("init_balanced_acc", "best_balanced_acc", "balanced_chance", "majority"):
                self.assertAlmostEqual(row[field], from_dir[key][field], delta=5e-4, msg=f"{key} {field}")
            self.assertEqual(row["subjects"], from_dir[key]["subjects"])
            self.assertTrue(np.isnan(row["init_acc"]))

    def test_old_stdout_format_also_gives_plain_accuracy(self):
        from_dir = {(r["target"], r["content"]): r for r in lp.rows(lp.load(self.run_dir))}
        with open(self.log, "w") as fp:
            fp.write(old_stdout_log())
        run = lp.load(self.log)
        self.assertEqual((run["step"], run["best_step"], run["best_metric"]), (500, 500, "val_loss"))
        from_log = {(r["target"], r["content"]): r for r in lp.rows(run)}
        self.assertEqual(set(from_log), {(t, v) for _, t in lp.TARGETS for v in ("T1", "FLAIR")})
        for key, row in from_log.items():
            for field in ("init_balanced_acc", "best_balanced_acc", "init_acc", "best_acc", "majority"):
                self.assertAlmostEqual(row[field], from_dir[key][field], delta=5e-4, msg=f"{key} {field}")

    def test_step_choice(self):
        self.assertEqual(lp.load(self.run_dir, "last")["step"], 1000)
        self.assertEqual(lp.load(self.log, "last")["step"], 1000)
        self.assertEqual(lp.load(self.log, "1000")["step"], 1000)
        with self.assertRaises(ValueError):
            lp.load(self.run_dir, "700")
        os.remove(os.path.join(self.run_dir, "best_checkpoint.json"))
        with self.assertRaises(ValueError):
            lp.load(self.run_dir)

    def test_no_init_is_an_error_not_a_silent_baseline(self):
        os.remove(os.path.join(self.run_dir, "separation_step0.json"))
        with self.assertRaises(ValueError):
            lp.load(self.run_dir)
        with open(self.log, "w") as fp:
            fp.write(stdout_log().split("  [eval] held-out subjects @ step 500")[0].replace("@ step 0", "@ step 5"))
        with self.assertRaises(ValueError):
            lp.load(self.log)

    def test_restarted_log_reads_the_last_attempt(self):
        with open(self.log, "w") as fp:
            fp.write(stdout_log(best_step=1000) + stdout_log(best_step=500, steps={0: STEPS[0], 500: STEPS[500]}))
        with contextlib.redirect_stderr(io.StringIO()):
            run = lp.load(self.log)
        self.assertEqual((run["best_step"], run["last_step"]), (500, 500))

    def test_parent_dir_search_summary_and_csv(self):
        other = os.path.join(self.root, "runs", "encoder_adni_other")
        os.makedirs(other)
        for name in os.listdir(self.run_dir):
            with open(os.path.join(self.run_dir, name)) as src, open(os.path.join(other, name), "w") as dst:
                dst.write(src.read())
        self.assertEqual(lp.expand([os.path.join(self.root, "runs")]), sorted([other, self.run_dir]))
        out_csv = os.path.join(self.root, "out", "probes.csv")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            lp.main([os.path.join(self.root, "runs"), self.log, "--csv", out_csv])
        self.assertIn("Δ balanced accuracy vs init, per run", printed.getvalue())
        with open(out_csv) as fp:
            written = list(csv.DictReader(fp))
        self.assertEqual(len(written), 9 + 9 + 6)
        self.assertEqual({row["step"] for row in written}, {"500"})


if __name__ == "__main__":
    unittest.main()
