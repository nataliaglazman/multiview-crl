"""burden_recovery reads real run directories, scores a matched pair and refuses an unmatched one."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from eval.encoder import burden_recovery
from training import main_conv_synthetic as trainer

SMALL = (
    "--res 16 --hidden-channels 8 --res-channels 4 --nb-res-layers 0 --latent-dim 7 --content-channels 4 "
    "--conv-readout mlp --encoder-head-hidden 8 --batch-size 2 --num-train-samples 4 --num-val-samples 20 "
    "--best-metric none --cpu-threads 1 --train-steps 2 --eval-every 2 --log-every 1 --no-cache --no-floor-eval "
    "--patch-loss-weight 0.5 --train-patch-grid 4 4 4 --tau 0.1 --synthetic-clean-content "
    "--synthetic-normalize fixed_reference --synthetic-lesion-target burden --synthetic-lesion-count 1 "
    "--synthetic-lesion-placement wm_interior --synthetic-lesion-t1-value 0 --hash-training-inputs "
    "--require-new-run --device cpu"
).split()


def train(root, name, *extra):
    flags = SMALL + ["--out-dir", str(root), "--model-id", name, *extra]
    with patch.object(sys, "argv", ["trainer", *flags]), patch.object(
        trainer, "evaluate", return_value=({}, {})
    ), contextlib.redirect_stdout(io.StringIO()):
        trainer.main()
    return str(root / name)


class BurdenRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.control = train(root, "control")
        cls.intervention = train(
            root,
            "intervention",
            "--vector-prediction-weight",
            "0.3",
            "--vector-prediction-grid",
            "4",
            "4",
            "4",
            "--vector-prediction-hidden",
            "8",
            "--vector-prediction-template-subjects",
            "3",
        )
        cls.other_lr = train(root, "other_lr", "--lr", "0.001")

    def run_eval(self, *run_dirs, extra=()):
        argv = ["--run-dirs", *run_dirs, "--num-samples", "24", "--probe-grid", "2", "2", "2"]
        argv += ["--device", "cpu", "--batch-size", "4", "--bootstrap", "20", *extra]
        # A 16-voxel brain has no 90%-brain cell on this grid.
        with contextlib.redirect_stdout(io.StringIO()) as output, patch.object(burden_recovery, "INTERIOR", 0.5):
            report = burden_recovery.run(burden_recovery.parse_args(argv))
        return report, output.getvalue()

    def test_matched_pair_is_scored_against_each_run_floor_and_the_control(self):
        report, output = self.run_eval(self.control, self.intervention)
        self.assertEqual(report["unmatched"], {})
        self.assertEqual(report["subjects"], 24)
        self.assertEqual([run["step"] for run in report["runs"]], [2, 2])
        expected = {
            f"{view}/{readout}/{probe}"
            for view in burden_recovery.VIEWS
            for readout in burden_recovery.READOUTS
            for probe in ("ridge", "kernel") + (() if readout == "code" else ("quantile",))
        }
        for result in report["runs"]:
            self.assertEqual(set(result["trained"]), expected)
            self.assertEqual(set(result["floor"]), expected)
            self.assertEqual(result["trained"]["t1/code/ridge"]["all"]["n"], 24)
        self.assertNotIn("minus_control", report["runs"][0])
        self.assertEqual(set(report["runs"][1]["minus_control"]), expected)
        self.assertIn("B-A", output)
        self.assertEqual(set(report["image"]), set(burden_recovery.VIEWS))
        # The decoder's forked RNG leaves both runs on the same untrained encoder.
        first, second = (json.dumps(report["runs"][i]["floor"], sort_keys=True) for i in range(2))
        self.assertEqual(first, second)

    def test_unmatched_pair_is_refused_and_names_the_difference(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            burden_recovery.run(burden_recovery.parse_args(["--run-dirs", self.control, self.other_lr]))
        self.assertIn("setting lr", output.getvalue())

    def test_allow_unmatched_scores_the_pair_and_records_why_it_is_not_matched(self):
        report, _ = self.run_eval(self.control, self.other_lr, extra=["--allow-unmatched"])
        self.assertIn(self.other_lr, report["unmatched"])
        self.assertTrue(any("setting lr" in problem for problem in report["unmatched"][self.other_lr]))

    def test_a_grid_without_interior_cells_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "finer grid"), contextlib.redirect_stdout(io.StringIO()):
            burden_recovery.run(
                burden_recovery.parse_args(
                    ["--run-dirs", self.control, "--num-samples", "4", "--probe-grid", "2", "2", "2", "--device", "cpu"]
                )
            )

    def test_toy_probes(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            burden_recovery.self_test()
        self.assertIn("self-test passed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
