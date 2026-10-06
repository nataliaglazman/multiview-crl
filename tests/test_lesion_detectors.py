"""Modality routing, controlled initialisation, native geometry and a real audit."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.lesion import lesion_detector_audit as audit
from eval.protocol.score_checkpoint import build_model
from scripts import compare_lesion_detectors as comparison
from tests.test_lesion_branch import brain_images, config, trainer


def residual_config(**changes):
    return config(
        conv_readout="mlp",
        lesion_keypoints=2,
        lesion_input="residual",
        lesion_normative_components=3,
        res=16,
        **changes
    )


def fitted_model(**changes):
    model = build_model(residual_config(**changes), "cpu")
    model.normative.brain.fill_(True)
    model.normative.fitted.fill_(True)
    return model


class LesionDetectorTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_init_preserves_global_projector_and_rng_and_shared_defaults(self):
        base = build_model(residual_config(), "cpu")
        rng = torch.get_rng_state()
        x = brain_images()
        for kind in ("separate", "separate_conv"):
            model = build_model(residual_config(lesion_detector=kind), "cpu")
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, tensor in base.state_dict().items():
                if not key.startswith("lesion_pool"):
                    torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
            model.normative.brain.fill_(True)
            model.normative.fitted.fill_(True)
            restored = build_model(residual_config(lesion_detector=kind), "cpu", model.state_dict())
            torch.testing.assert_close(model.lesion_maps(x, n_views=2), restored.lesion_maps(x, n_views=2))
        shared, separate = fitted_model(), fitted_model(lesion_detector="separate")
        torch.testing.assert_close(shared.lesion_code(x), separate.lesion_code(x))
        torch.testing.assert_close(separate.lesion_pool.logits.weight, separate.lesion_pool_v1.logits.weight)
        self.assertIsNot(separate.lesion_pool.logits.weight, separate.lesion_pool_v1.logits.weight)

    def test_view_major_and_single_view_routing_with_independent_gradients(self):
        for kind in ("separate", "separate_conv"):
            model, x = fitted_model(lesion_detector=kind), brain_images()
            with torch.no_grad():
                model.lesion_pool_v1.logits.weight.mul_(-3)
            together = model.lesion_maps(x, n_views=2)
            torch.testing.assert_close(together[:2], model.lesion_maps(x[:2], view_idx=0))
            torch.testing.assert_close(together[2:], model.lesion_maps(x[2:], view_idx=1))
            model.lesion_code(x[2:], view_idx=1).square().sum().backward()
            self.assertIsNone(model.lesion_pool.logits.weight.grad)
            self.assertGreater(model.lesion_pool_v1.logits.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            if kind == "separate_conv":
                self.assertIsNone(model.lesion_adapter[0].weight.grad)
                self.assertGreater(model.lesion_adapter_v1[0].weight.grad.abs().sum().item(), 0)

    def test_frozen_branch_stays_fixed_while_global_trains(self):
        model = fitted_model(lesion_branch_frozen=True).train()
        x = brain_images()
        before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("lesion_")}
        global_before = model.to_encoding[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        code = model(x, pool_only=True, n_views=2)[2][0]
        code.square().sum().backward()
        trainer.clip_encoder_gradients(model, 1)
        optimizer.step()
        for k, v in before.items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        self.assertFalse(torch.equal(global_before, model.to_encoding[0].weight))
        self.assertTrue(all(not p.requires_grad for part in model.lesion_modules() for p in part.parameters()))

    def test_branch_gradient_does_not_change_global_gradient_clipping(self):
        global_grads = []
        for scale in (1, 10000):
            model = fitted_model()
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            for part in model.lesion_modules():
                for p in part.parameters():
                    p.grad.mul_(scale)
            trainer.clip_encoder_gradients(model, 0.5)
            global_grads.append(model.to_encoding[0].weight.grad.clone())
        torch.testing.assert_close(*global_grads, rtol=0, atol=0)

    def test_voxel_coordinates_account_for_bin_centres_and_nondivisible_grids(self):
        weights = torch.zeros(1, 1, 4, 4, 4)
        weights[0, 0, 1, 2, 3] = 1
        torch.testing.assert_close(audit.native_positions(weights, (16, 16, 16)), torch.tensor([[[5.5, 9.5, 13.5]]]))
        # Bin intervals for size 10/grid 4: [0,3), [2,5), [5,8), [7,10).
        torch.testing.assert_close(audit.native_positions(weights, (10, 10, 10)), torch.tensor([[[3.0, 6.0, 8.0]]]))

    def test_localization_hit_rate_and_validation_head_selection(self):
        truth = np.zeros((2, 3))
        predictions = np.array([[0, 3, 0], [0, 4, 0]])
        metrics = audit.localization_metrics(truth, predictions)
        self.assertEqual(metrics["mean_error_vox"], 3.5)
        self.assertEqual(metrics["hit_within_3_vox"], 0.5)
        val = {"head_0": np.zeros((2, 2, 3)), "head_1": np.ones((2, 2, 3))}
        selected = audit.select_heads(truth, val)
        test = {"head_0": np.ones((2, 2, 3)) * 100, "head_1": np.zeros((2, 2, 3))}
        audit.add_selected(test, selected)
        self.assertTrue((test["selected_validation_head"] == 100).all())

    def test_movement_uses_actual_vector_displacement_and_counts_no_move(self):
        truth = np.array([[[0.0, 0, 0], [2, 3, 1]], [[0, 0, 0], [0, 0, 0]]])
        perfect = audit.movement_metrics(truth, truth, [0, 1], 1, bootstrap=0)
        static = audit.movement_metrics(truth, np.zeros_like(truth), [0, 1], 1, bootstrap=0)
        self.assertEqual(perfect["movement_skill"], 1)
        self.assertEqual(static["movement_skill"], 0)
        self.assertEqual(perfect["n_moved"], 1)
        self.assertEqual(perfect["n_pairs"], 2)

    def test_four_arm_plan_and_cli_restore(self):
        args = comparison.parse_args(["plan", "--output-dir", "/tmp/lesion-plan", "--device", "cpu"])
        recipe = comparison.comparison.read_json(args.config)
        for arm in comparison.ARMS:
            options = comparison.training_options(args, recipe, arm, 42)
            parsed = trainer.parse_args(comparison.comparison.cli_arguments(options))
            self.assertEqual(parsed.lesion_branch_frozen, arm == "frozen_shared")
            self.assertTrue(parsed.lesion_localization_eval)
            self.assertEqual(parsed.data_seed, 42)
            self.assertEqual(parsed.lesion_input, "residual")
        for extra in (["--lesion-detector", "separate"], ["--lesion-branch-frozen"], ["--lesion-localization-eval"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(extra)

    def test_real_training_direct_report_and_initial_vs_trained_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--out-dir",
                tmp,
                "--model-id",
                "separate_conv",
                "--device",
                "cpu",
                "--require-new-run",
                "--res",
                "16",
                "--hidden-channels",
                "64",
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
                "--lesion-keypoints",
                "2",
                "--lesion-input",
                "residual",
                "--lesion-detector",
                "separate_conv",
                "--lesion-normative-components",
                "3",
                "--lesion-normative-subjects",
                "9",
                "--lesion-localization-eval",
                "--synthetic-lesion-placement",
                "wm_interior",
                "--tau",
                "0.1",
                "--lr",
                "0.01",
                "--batch-size",
                "2",
                "--train-steps",
                "2",
                "--eval-every",
                "2",
                "--num-train-samples",
                "10",
                "--num-val-samples",
                "20",
                "--no-cache",
                "--best-metric",
                "none",
                "--synthetic-clean-content",
                "--synthetic-normalize",
                "fixed_reference",
                "--cpu-threads",
                "1",
            ]
            with patch.object(sys, "argv", ["trainer", *argv]), patch.object(
                trainer.dci, "compute_dci_synthetic", return_value={}
            ), patch.object(trainer.dci, "flatten_dci_results", return_value={}), contextlib.redirect_stdout(
                io.StringIO()
            ):
                trainer.main()
                run = Path(tmp) / "separate_conv"
                report = json.loads((run / "dci_step2.json").read_text())["lesion_branch"]
                self.assertEqual(len(report["localization"]["t1"]), 2)
                args = audit.parse_args(
                    [
                        "--run-dir",
                        str(run),
                        "--out-dir",
                        str(run / "audit"),
                        "--device",
                        "cpu",
                        "--num-samples",
                        "4",
                        "--validation-samples",
                        "4",
                        "--movement-subjects",
                        "2",
                        "--bootstrap",
                        "4",
                    ]
                )
                result = audit.run_audit(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(set(result["initial"]["selected_heads"]), {"t1", "flair"})
            self.assertTrue((run / "audit/predictions.npz").exists())
            self.assertTrue(all(row["n_nonempty"] == 4 for row in result["localization"]))
            with np.load(run / "audit/predictions.npz") as bank:
                np.testing.assert_array_equal(bank["initial_truth"], bank["trained_truth"])
                np.testing.assert_array_equal(
                    bank["initial_residual_peak_unsigned"], bank["trained_residual_peak_unsigned"]
                )
            for row in result["localization"]:
                if row["method"].startswith("residual_peak"):
                    self.assertEqual(row["delta_mean_error_vox_vs_initial"], 0)


if __name__ == "__main__":
    unittest.main()
