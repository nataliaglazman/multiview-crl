"""Stdlib checks for paired commands, provenance and final comparison safeguards."""

import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import compare_encoders as comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = comparison.read_json(comparison.ROOT / "experiments/encoder_comparison.json")

    def test_pair_changes_only_architecture_and_name(self):
        comparison.validate_config(self.config)
        a, b = [
            comparison.run_options(self.config, Path("/tmp/comparison"), 142, arch) for arch in comparison.ARCHITECTURES
        ]
        self.assertEqual({key for key in a if a[key] != b[key]}, {"encoder_architecture", "model_id"})
        self.assertEqual((a["seed"], a["model_seed"], a["data_seed"], a["loader_seed"]), (142, 142, 42, 10142))
        command = comparison.training_command(a)
        self.assertIn("--no-cache", command)
        self.assertIn("--cross-view-negs-only", command)
        self.assertIn("--deterministic", command)
        self.assertIn("--deterministic-warn-only", command)
        self.assertNotIn("--no-separate-encoders", command)
        self.assertNotIn("--synthetic-causal", command)

    def test_invalid_grid_and_nonpaired_protocol_are_rejected(self):
        for section, key, value in (
            ("evaluation", "patch_grid", 4),
            ("shared", "best_metric", "block_mcc"),
            ("shared", "batch_size", 3000),
            ("shared", "data_seed", 9),
            ("evaluation", "probe_samples", 200),
            ("shared", "deterministic", False),
        ):
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(ValueError):
                comparison.validate_config(config)

    def test_plan_is_read_only_and_manifest_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", new_callable=io.StringIO):
            output = Path(tmp) / "new"
            comparison.main(["plan", "--output-dir", str(output), "--only-seed", "42"])
            self.assertFalse(output.exists())
            with patch.object(comparison, "source_hashes", return_value={"model.py": "one"}):
                comparison.ensure_manifest(output, self.config, create=True)
                comparison.ensure_manifest(output, self.config)
                changed = copy.deepcopy(self.config)
                changed["shared"]["lr"] *= 2
                with self.assertRaisesRegex(ValueError, "differs"):
                    comparison.ensure_manifest(output, changed)
            with patch.object(comparison, "source_hashes", return_value={"model.py": "two"}):
                with self.assertRaisesRegex(ValueError, "differs"):
                    comparison.ensure_manifest(output, self.config)

    def fixture(self, output, architecture, value):
        options = comparison.run_options(self.config, output, 42, architecture)
        run = Path(options["out_dir"]) / options["model_id"]
        run.mkdir(parents=True)
        comparison.write_json(run / "settings.json", options)
        progress = {
            "status": "complete",
            "step": options["train_steps"],
            "batch_order_sha256": "same-order",
            "training_input_sha256": "same-training-images",
            "cpu_threads": 1,
            "numpy_version": "test",
            "torch_version": "test",
            "optimizer_defaults": {"lr": options["lr"]},
            "deterministic_algorithms": options["deterministic"],
            "deterministic_warn_only": options["deterministic_warn_only"],
        }
        comparison.write_json(run / "training_progress.json", progress)
        hashes = {}
        for name in ("model.pt", "model_init.pt"):
            (run / name).write_bytes((architecture + name).encode())
            hashes[name] = comparison.file_hash(run / name)
        comparison.write_json(run / "comparison_receipt.json", hashes)
        comparison.write_json(
            run / "evaluation/complete.json", {"checkpoint_hashes": hashes, "evaluation": self.config["evaluation"]}
        )
        report = {
            "status": "complete",
            "checkpoint_sha256": hashes["model.pt"],
            "cohorts": {"val": [1, 2]},
            "probe_split": {"fit_validation_ids": [1], "tune_validation_ids": [2]},
            "state": {"original": {split: {"input_sha256": "same-images"} for split in ("train", "val", "test")}},
            "probes": [
                {
                    "arm": "original",
                    "stage": "content",
                    "condition": "observed",
                    "view": "t1",
                    "target": "sulcal_widening",
                    "probe": "ridge",
                    "test_r2": value,
                }
            ],
        }
        comparison.write_json(run / "evaluation/global_path/report.json", report)
        return run

    def test_evaluation_checks_all_selected_runs_before_launching(self):
        for state in ("missing", "incomplete", "complete"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp).resolve()
                conv = self.fixture(output, "conv", 0.1)
                if state != "missing":
                    resnet = self.fixture(output, "resnet18", 0.2)
                    if state == "incomplete":
                        progress = comparison.read_json(resnet / "training_progress.json")
                        progress.update(status="running", step=0)
                        comparison.write_json(resnet / "training_progress.json", progress)
                with patch.object(comparison, "source_hashes", return_value={}), patch.object(
                    comparison, "evaluate_run"
                ) as evaluate:
                    comparison.ensure_manifest(output, self.config, create=True)
                    command = ["evaluate", "--output-dir", str(output), "--only-seed", "42"]
                    if state == "complete":
                        comparison.main(command)
                        self.assertEqual([call.args[0] for call in evaluate.call_args_list], [conv, resnet])
                    else:
                        with self.assertRaisesRegex(ValueError, "Evaluation not started"):
                            comparison.main(command)
                        evaluate.assert_not_called()

    def test_summary_reports_paired_difference_and_rejects_broken_pairing(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", new_callable=io.StringIO):
            output = Path(tmp)
            self.fixture(output, "conv", -0.02)
            run = self.fixture(output, "resnet18", 0.68)
            comparison.summarize(output, self.config, [42], "evaluation")
            with (output / "summary_evaluation/paired_comparison.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["resnet_minus_conv_mean"]), 0.70)
            self.assertEqual(row["paired_seeds"], "1")
            self.assertEqual(row["paired_difference_sd"], "")
            progress = comparison.read_json(run / "training_progress.json")
            progress["batch_order_sha256"] = "different-order"
            comparison.write_json(run / "training_progress.json", progress)
            with self.assertRaisesRegex(ValueError, "subject order differed"):
                comparison.summarize(output, self.config, [42], "evaluation")
            progress["batch_order_sha256"] = "same-order"
            progress["deterministic_warn_only"] = False
            comparison.write_json(run / "training_progress.json", progress)
            with self.assertRaisesRegex(ValueError, "runtime or optimizer defaults differed"):
                comparison.summarize(output, self.config, [42], "evaluation")
            progress["deterministic_warn_only"] = True
            comparison.write_json(run / "training_progress.json", progress)
            path = run / "evaluation/global_path/report.json"
            report = comparison.read_json(path)
            report["state"]["original"]["test"]["input_sha256"] = "different-images"
            comparison.write_json(path, report)
            with self.assertRaisesRegex(ValueError, "test images differed"):
                comparison.summarize(output, self.config, [42], "evaluation")
            (run / "model.pt").write_bytes(b"overwritten")
            with self.assertRaisesRegex(ValueError, "changed after training"):
                comparison.validate_run(run, comparison.run_options(self.config, output, 42, "resnet18"))


if __name__ == "__main__":
    unittest.main()
