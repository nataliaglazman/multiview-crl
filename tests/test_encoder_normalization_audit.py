"""Normalization domains, forward routing, frozen state, and held-out audit smoke tests."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval.diagnostics.pooling_probe import fit_readouts
from eval.encoder import encoder_normalization_audit as audit
from eval.encoder.encoder_target_protocol import dataset, digest, sample_targets
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.lesion.lesion_probe import block_gram
from eval.protocol.score_checkpoint import build_model
from models.vqvae import ChannelLayerNorm3d
from training.main_conv_synthetic import parse_args


def config(norm="group", **overrides):
    cfg = vars(
        parse_args(
            [
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
                "--num-val-samples",
                "20",
                "--device",
                "cpu",
                "--no-cache",
                "--synthetic-clean-content",
                "--synthetic-normalize",
                "fixed_reference",
                "--synthetic-lesion-placement",
                "wm_interior",
            ]
        )
    )
    cfg.update(norm_type=norm, **overrides)
    return cfg


class NormalizationAuditTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_statistics_reconstruct_normalized_tensor_on_correct_axes(self):
        torch.manual_seed(5)
        x = torch.randn(3, 8, 4, 3, 2)
        for module in (torch.nn.GroupNorm(4, 8), ChannelLayerNorm3d(8)):
            with self.subTest(module=type(module).__name__):
                mean, scale = audit.norm_statistics(module, x)
                if isinstance(module, torch.nn.GroupNorm):
                    self.assertEqual(mean.shape, (3, 4))
                    y = (x.reshape(3, 4, -1) - mean[..., None]) / scale[..., None]
                    expected = y.reshape_as(x)
                    weights, biases = module.weight, module.bias
                else:
                    self.assertEqual(mean.shape, (3, 1, 4, 3, 2))
                    expected = (x - mean) / scale
                    weights, biases = module.norm.weight, module.norm.bias
                with torch.no_grad():
                    weights.copy_(torch.linspace(0.2, 1.6, 8))
                    biases.copy_(torch.linspace(-0.3, 0.4, 8))
                expected = expected * weights[None, :, None, None, None] + biases[None, :, None, None, None]
                torch.testing.assert_close(module(x), expected, atol=1e-6, rtol=1e-5)

    def test_capture_matches_forward_for_both_norms_and_view_routings(self):
        for norm, shared in (("group", False), ("layer", False), ("layer", True)):
            with self.subTest(norm=norm, shared=shared):
                model = build_model(config(norm, no_separate_encoders=shared), "cpu")
                x = torch.randn(4, 1, 16, 16, 16)
                before = state_digest(model)
                hooks_before = sum(len(m._forward_hooks) for m in model.modules())
                features, metadata = audit.capture(model, x, "layers.0.1")
                # The hook must retain negative post-norm values even though ReLU is in-place.
                self.assertLess(features["norm_post_spatial"].min(), -0.1)
                with torch.inference_mode():
                    h = model._encode(x, 2, None)
                    code = model(x, pool_only=True, n_views=2)[2][0]
                np.testing.assert_allclose(features["backbone_spatial"], h.flatten(1).numpy())
                np.testing.assert_allclose(features["backbone_gap"], h.mean((2, 3, 4)).numpy())
                np.testing.assert_allclose(features["global_content"], code[:, :9].numpy())
                np.testing.assert_allclose(features["global_all"], code[:, :12].numpy())
                self.assertEqual(len(metadata["normalizers"]), 1 if shared else 2)
                self.assertEqual(state_digest(model), before)
                self.assertEqual(sum(len(m._forward_hooks) for m in model.modules()), hooks_before)
                with self.assertRaisesRegex(ValueError, "exceeds native"):
                    audit.capture(model, x, spatial_grid=100)
                self.assertEqual(sum(len(m._forward_hooks) for m in model.modules()), hooks_before)

    def test_chunked_kernel_and_probe_detect_spatial_information_lost_by_gap(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=(90, 1))
        x = np.column_stack((y[:, 0], -y[:, 0], np.ones(90)))
        splits = (np.arange(45), np.arange(45, 60), np.arange(60, 90))
        gram, width = audit.bank_gram(x[:60], x[60:], splits[0], chunk_size=1)
        expected, expected_width = block_gram(x, np.arange(3), splits[0])
        np.testing.assert_allclose(gram, expected, atol=1e-10)
        self.assertEqual(width, expected_width)
        rows, _ = fit_readouts(gram, width, y, splits, 1729, ("factor",))
        scores = {(r["probe"], r["condition"]): r["test_r2"] for r in rows}
        self.assertGreater(scores["ridge", "observed"], 0.99)
        self.assertLess(scores["ridge", "shuffled"], 0.2)
        pooled = x[:, :2].mean(1, keepdims=True)
        gram, width = audit.bank_gram(pooled[:60], pooled[60:], splits[0])
        self.assertEqual(width, 0)
        rows, _ = fit_readouts(gram, width, y, splits, 1729, ("factor",))
        self.assertTrue(all(row["test_r2"] <= 0 for row in rows))

    def test_linear_readout_attention_and_lesion_branch_are_separate(self):
        for readout, pool in (("linear", "gap"), ("mlp", "attention")):
            with self.subTest(readout=readout, pool=pool):
                model = build_model(config("layer", conv_readout=readout, global_pool=pool, lesion_keypoints=2), "cpu")
                x = torch.randn(4, 1, 16, 16, 16)
                features, metadata = audit.capture(model, x)
                with torch.inference_mode():
                    code = model(x, pool_only=True, n_views=2)[2][0]
                    h = model._encode(x, 2, None)
                    if pool == "attention":
                        np.testing.assert_allclose(features["actual_pool"], model.attention_pool(h).numpy())
                    else:
                        self.assertNotIn("actual_pool", features)
                self.assertEqual(features["global_content"].shape, (4, 9))
                self.assertEqual(features["global_all"].shape, (4, 12))
                self.assertEqual(features["lesion_branch"].shape, (4, 6))
                np.testing.assert_allclose(features["lesion_branch"], code[:, 12:].numpy())
                self.assertEqual(metadata["lesion_branch_units"], 6)

    def test_position_targets_match_existing_protocol_and_burden_uses_placement(self):
        for target in ("position", "burden"):
            ds = dataset(config(synthetic_lesion_target=target), 20, "val")
            latents = ds[0]["gt_latents"]
            y = audit.targets_for_sample(ds._inner, latents)
            self.assertEqual(y.shape, (14,))
            self.assertTrue(np.isfinite(y).all())
            if target == "position":
                np.testing.assert_array_equal(y, sample_targets(ds._inner, latents)[0])
            else:
                self.assertIn("z_lesion", latents)

    def test_real_two_checkpoint_audit_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            before = {}
            for norm in ("group", "layer"):
                run = root / norm
                run.mkdir()
                cfg = config(norm)
                (run / "settings.json").write_text(json.dumps(cfg))
                torch.save(build_model(cfg, "cpu").state_dict(), run / "model.pt")
                before[norm] = digest(run / "model.pt")
            report = audit.main(
                [
                    "--run",
                    f"gn={root / 'group'}",
                    "--run",
                    f"ln={root / 'layer'}",
                    "--out-dir",
                    str(root / "audit"),
                    "--device",
                    "cpu",
                    "--test-samples",
                    "10",
                    "--batch-size",
                    "3",
                    "--spatial-grid",
                    "2",
                    "--bootstrap-draws",
                    "10",
                ]
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(row["arm"] for row in report["probes"]), {"gn", "ln"})
            self.assertEqual(set(row["condition"] for row in report["probes"]), {"observed", "shuffled"})
            self.assertEqual(set(row["probe"] for row in report["probes"]), {"ridge", "rbf"})
            self.assertEqual(set(row["target"] for row in report["probes"]), set(audit.TARGETS))
            self.assertTrue(all(np.isfinite(row["test_r2"]) for row in report["probes"]))
            self.assertEqual(len(report["probe_split"]["fit_validation_ids"]), 15)
            self.assertEqual(len(report["probe_split"]["tune_validation_ids"]), 5)
            self.assertFalse(list((root / "audit").rglob("*.npy")))
            self.assertFalse(list((root / "audit").glob(".features-*")))
            self.assertTrue((root / "audit" / "contrasts.csv").exists())
            self.assertTrue(
                any(row["candidate_arm"] == "ln" and row["reference_arm"] == "gn" for row in report["contrasts"])
            )
            with np.load(root / "audit" / "gn_predictions.npz") as predictions:
                self.assertEqual(predictions["truth"].shape, (10, 14))
            for norm in ("group", "layer"):
                self.assertEqual(digest(root / norm / "model.pt"), before[norm])
            reference = report["cohorts"]["gn"]
            changed = json.loads(json.dumps(report["cohorts"]["ln"]))
            changed["test"]["input_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "unmatched comparison"):
                audit.matched_cohorts(reference, changed)

    def test_contrasts_have_candidate_minus_reference_sign(self):
        truth = np.linspace(-1, 1, 20)[:, None]
        predictions = {"gn": {}, "ln": {}}
        for arm, prediction in (("gn", np.zeros_like(truth)), ("ln", truth)):
            for view in audit.VIEWS:
                for probe in ("ridge", "rbf"):
                    predictions[arm][view, "global_content", f"{probe}_observed"] = prediction
        rows = audit.contrast_rows(predictions, truth, ("factor",), SimpleNamespace(seed=3, bootstrap_draws=20))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(abs(row["delta_r2"] - 1) < 1e-8 for row in rows))


if __name__ == "__main__":
    unittest.main()
