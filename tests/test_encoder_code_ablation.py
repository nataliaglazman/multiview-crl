"""Matched saved-recipe launches, including a misleading burden folder name."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_encoder_code_ablation as ablation
from training import encoder_args


class CodeAblationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.reference = Path(temporary.name) / "burden_k60_s42"
        self.reference.mkdir()
        self.cfg = vars(encoder_args.parse_args(["--standardize-code", "--synthetic-lesion-target", "position"]))
        self.settings = self.reference / "settings.json"
        self.settings.write_text(json.dumps(self.cfg))

    def options(self, *extra):
        return ablation.parse_args(["--reference-run", str(self.reference), *extra])

    def test_only_interventions_and_output_settings_change_and_source_is_untouched(self):
        before = self.settings.read_bytes()
        reference, planned = ablation.plan_runs(self.options("--modes", *ablation.MODES))
        allowed = {"center_code", "standardize_code", "require_new_run", "out_dir", "model_id"}
        self.assertEqual(len(planned), 3)
        for mode, cfg in planned:
            self.assertEqual(
                {k: v for k, v in cfg.items() if k not in allowed},
                {k: v for k, v in reference.items() if k not in allowed},
            )
            self.assertEqual(cfg["synthetic_lesion_target"], "position")
            self.assertEqual(cfg["center_code"], mode == "centered")
            self.assertEqual(cfg["standardize_code"], mode == "standardized")
        self.assertEqual(self.settings.read_bytes(), before)

    def test_replicate_keeps_generator_seed_and_matches_model_and_loader_seeds(self):
        _, planned = ablation.plan_runs(self.options("--seed", "43", "--device", "cuda"))
        for _, cfg in planned:
            self.assertEqual((cfg["seed"], cfg["model_seed"], cfg["loader_seed"]), (43, 43, 10043))
            self.assertEqual(cfg["data_seed"], self.cfg["data_seed"])
            self.assertEqual(cfg["device"], "cuda")
            self.assertFalse(cfg["no_cuda"])

    def test_existing_output_and_unknown_saved_option_are_refused_before_launch(self):
        _, planned = ablation.plan_runs(self.options())
        cfg = planned[-1][1]
        (Path(cfg["out_dir"]) / cfg["model_id"]).mkdir()
        with self.assertRaises(FileExistsError):
            ablation.plan_runs(self.options())
        self.settings.write_text(json.dumps({**self.cfg, "unknown_training_flag": True}))
        with self.assertRaises(ValueError):
            ablation.plan_runs(self.options())

    def test_dry_run_does_not_start_training(self):
        with patch.object(ablation.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as output:
            ablation.main(["--reference-run", str(self.reference), "--dry-run"])
        run.assert_not_called()
        self.assertIn("lesion target=position", output.getvalue())
        self.assertIn("--center-code", output.getvalue())


if __name__ == "__main__":
    unittest.main()
