"""Real frozen probes, state isolation, and unchanged training with monitoring on."""

import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.encoder import encoder_spatial_target_audit as audit
from eval.encoder import spatial_recovery_monitor as monitor
from eval.encoder.encoder_target_protocol import TARGETS
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model
from scripts import run_encoder_mps
from tests.test_encoder_mps_runner import runner_args
from training import main_conv_synthetic as trainer


def arguments():
    return [
        "--res",
        "16",
        "--hidden-channels",
        "8",
        "--res-channels",
        "4",
        "--nb-res-layers",
        "1",
        "--latent-dim",
        "12",
        "--conv-readout",
        "mlp",
        "--encoder-head-hidden",
        "8",
        "--patch-loss-weight",
        "1",
        "--train-patch-grid",
        "2",
        "2",
        "2",
        "--tau",
        "0.1",
        "--num-train-samples",
        "8",
        "--num-val-samples",
        "20",
        "--batch-size",
        "2",
        "--train-steps",
        "2",
        "--eval-every",
        "1",
        "--log-every",
        "1",
        "--device",
        "cpu",
        "--no-cache",
        "--best-metric",
        "none",
        "--hash-training-inputs",
        "--synthetic-clean-content",
        "--synthetic-normalize",
        "fixed_reference",
        "--synthetic-lesion-placement",
        "wm_interior",
        "--spatial-recovery-eval",
        "--spatial-recovery-test-samples",
        "10",
        "--spatial-recovery-batch-size",
        "2",
    ]


class SpatialRecoveryMonitorTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_configuration_and_launcher_flags(self):
        self.assertFalse(trainer.parse_args([]).spatial_recovery_eval)
        cfg = trainer.parse_args(arguments())
        self.assertEqual(cfg.spatial_recovery_grids, [1, 2])
        self.assertEqual(trainer.parse_args(arguments() + ["--patch-loss-weight", "0"]).spatial_recovery_grids, [1, 4])
        for extra in (
            ["--spatial-recovery-grids", "5"],
            ["--spatial-recovery-grids", "0"],
            ["--spatial-recovery-test-samples", "9"],
            ["--spatial-recovery-batch-size", "0"],
            ["--num-val-samples", "19"],
            ["--n-content", "10"],
            ["--train-patch-grid", "2", "3", "2"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(arguments() + extra)
        options, _ = run_encoder_mps.make_options(
            runner_args(
                variant="conv_mlp",
                spatial_recovery_eval=True,
                spatial_recovery_grids=[8, 8],
                spatial_recovery_native=True,
                spatial_recovery_test_samples=40,
            )
        )
        parsed = trainer.parse_args(run_encoder_mps.comparison.cli_arguments(options))
        for key, value in options.items():
            self.assertEqual(getattr(parsed, key), value, key)

    def test_real_probes_preserve_state_match_initial_and_delete_feature_banks(self):
        cfg = vars(trainer.parse_args(arguments() + ["--spatial-recovery-native"]))
        model = build_model(cfg, "cpu").train()
        # Preserve a mixed module mode and existing gradients, not just model.training.
        model.to_encoding.eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = state_digest(model)
        modes = [module.training for module in model.modules()]
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            initial = monitor.evaluate_spatial_recovery(model, cfg, "cpu", tmp, 0)
            trained = monitor.evaluate_spatial_recovery(model, cfg, "cpu", tmp, 1, initial=initial)
            self.assertEqual(trained["status"], "complete")
            self.assertEqual(set(row["view"] for row in trained["probes"]), {"t1", "flair"})
            self.assertEqual(set(row["grid"] for row in trained["probes"]), {1, 2, 4})
            self.assertEqual(set(row["stage"] for row in trained["probes"]), {"backbone", "projected"})
            self.assertEqual(set(row["target"] for row in trained["probes"]), set(TARGETS))
            self.assertEqual(set(row["probe"] for row in trained["probes"]), {"ridge", "rbf"})
            self.assertEqual(set(row["condition"] for row in trained["probes"]), {"observed", "shuffled"})
            self.assertTrue(all(row["delta_initial"] == 0 for row in trained["probes"]))
            self.assertEqual(trained["cohorts"], initial["cohorts"])
            self.assertNotEqual(trained["cohorts"]["val"]["input_sha256"], trained["cohorts"]["test"]["input_sha256"])
            self.assertFalse(list(Path(tmp).rglob("*.npy")))
            self.assertFalse(list(Path(tmp).rglob(".features-*")))
            self.assertTrue((Path(tmp) / "spatial_recovery/step_00000001/probes.csv").exists())
            self.assertIn("amp_null", output.getvalue())
        self.assertEqual(state_digest(model), before)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))
        self.assertEqual(random.getstate(), python_state)
        actual_np = np.random.get_state()
        self.assertEqual(actual_np[0], numpy_state[0])
        np.testing.assert_array_equal(actual_np[1], numpy_state[1])
        self.assertEqual(actual_np[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_probe_failure_restores_training_state_and_cleans_temporary_files(self):
        cfg = vars(trainer.parse_args(arguments()))
        model = build_model(cfg, "cpu").train()
        before, rng = state_digest(model), torch.get_rng_state()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(
            audit, "score_banks", side_effect=RuntimeError("probe failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                monitor.evaluate_spatial_recovery(model, cfg, "cpu", tmp, 0)
            report = json.loads((Path(tmp) / "spatial_recovery/step_00000000/report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(list(Path(tmp).rglob("*.npy")))
        self.assertTrue(model.training)
        self.assertEqual(state_digest(model), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_monitoring_does_not_change_real_training_weights_or_batch_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(trainer, "evaluate", return_value=({}, {})):
            results = {}
            for enabled in (False, True):
                name = "enabled" if enabled else "disabled"
                argv = ["trainer", *arguments(), "--out-dir", tmp, "--model-id", name, "--require-new-run"]
                if not enabled:
                    argv += ["--no-spatial-recovery-eval"]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
                directory = Path(tmp) / name
                results[enabled] = (
                    torch.load(directory / "model.pt", weights_only=True),
                    json.loads((directory / "training_progress.json").read_text()),
                )
                if enabled:
                    for step in (0, 1, 2):
                        self.assertTrue((directory / f"spatial_recovery/step_{step:08d}/probes.csv").exists())
                else:
                    self.assertFalse((directory / "spatial_recovery").exists())
                self.assertFalse((directory / "model_best.pt").exists())
            for key in results[False][0]:
                torch.testing.assert_close(results[False][0][key], results[True][0][key], rtol=0, atol=0)
            for key in ("batch_order_sha256", "training_input_sha256", "last_loss_terms"):
                self.assertEqual(results[False][1][key], results[True][1][key])

    def test_offline_audit_can_discard_large_features(self):
        cfg = vars(trainer.parse_args(arguments() + ["--synthetic-lesion-radius", "0.2"]))
        model = build_model(cfg, "cpu")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "settings.json").write_text(json.dumps(cfg))
            for name in ("model.pt", "model_init.pt"):
                torch.save(model.state_dict(), run / name)
            out = Path(tmp) / "audit"
            audit.main(
                [
                    "--run-dir",
                    str(run),
                    "--out-dir",
                    str(out),
                    "--test-samples",
                    "10",
                    "--grids",
                    "1",
                    "2",
                    "--device",
                    "cpu",
                    "--discard-features",
                ]
            )
            self.assertFalse((out / "features").exists())
            self.assertTrue((out / "probes.csv").exists())
            self.assertEqual(json.loads((out / "report.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
