"""Diagnosis must expose drift without changing the existing experiment."""

import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import compare_encoders as comparison
from scripts import diagnose_encoder_comparison as diagnostic


class ComparisonDiagnosisTests(unittest.TestCase):
    def test_drift_and_saved_runs_are_checked_without_writing(self):
        config = comparison.read_json(comparison.ROOT / "experiments/encoder_comparison.json")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            comparison.write_json(
                output / "comparison_manifest.json",
                {"schema_version": 1, "config": config, "sources": {"training/main.py": "old", "removed.py": "old"}},
            )
            options = comparison.run_options(config, output, 42, "resnet18")
            run = Path(options["out_dir"]) / options["model_id"]
            comparison.write_json(run / "settings.json", options)
            comparison.write_json(run / "training_progress.json", {"status": "complete", "step": 10000})
            for name in ("model.pt", "model_init.pt"):
                (run / name).write_bytes(name.encode())
            comparison.write_json(
                run / "comparison_receipt.json",
                {name: comparison.file_hash(run / name) for name in ("model.pt", "model_init.pt")},
            )
            before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            current = copy.deepcopy(config)
            current["shared"]["batch_size"] = 2
            with patch.object(comparison, "source_hashes", return_value={"training/main.py": "new", "added.py": "new"}):
                report = diagnostic.diagnose(output, current, 42)
            self.assertIn("shared.batch_size: saved=32, current=2", report["configuration_differences"])
            self.assertEqual(len(report["source_differences"]), 3)
            self.assertEqual(report["runs"][1]["saved_recipe_and_checkpoint_receipt"], "match")
            self.assertFalse(report["runs"][0]["model.pt"])
            self.assertEqual(before, {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()})
            (run / "model.pt").write_bytes(b"changed")
            report = diagnostic.diagnose(output, config, 42)
            self.assertIn("changed after training", report["runs"][1]["saved_recipe_and_checkpoint_receipt"])

    def test_empty_output_is_rejected_and_missing_output_is_not_created(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                diagnostic.main(["--output-dir", ""])
            self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                diagnostic.main(["--output-dir", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
