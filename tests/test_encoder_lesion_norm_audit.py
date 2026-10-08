"""Controlled lesion moves, native normalization mechanics, and held-out probes."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from eval.encoder import encoder_lesion_norm_audit as audit
from eval.encoder.encoder_target_protocol import dataset, digest
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model
from models.vqvae import ChannelLayerNorm3d
from tests.test_encoder_normalization_audit import config


def options(**changes):
    return SimpleNamespace(
        **{
            "num_samples": 3,
            "subject_offset": 0,
            "axes": ["x", "y", "z"],
            "eps": 0.5,
            "batch_size": 2,
            "spatial_grid": 2,
            "seed": 1729,
            "bootstrap": 10,
            "resolution": 16,
            **changes,
        }
    )


class LesionNormAuditTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_local_positive_scaling_is_suppressed_by_ln_but_survives_gn(self):
        a = torch.arange(1.0, 5.0).reshape(1, 4, 1, 1, 1).expand(1, 4, 3, 3, 3).clone()
        b = a.clone()
        b[:, :, 1, 1, 1] *= 2
        x = torch.cat((a, b))
        ln, gn = ChannelLayerNorm3d(4), torch.nn.GroupNorm(2, 4)
        local = audit.layer_response(ln, x, ln(x))[0]
        global_ = audit.layer_response(gn, x, gn(x))[0]
        self.assertGreater(local["fixed_stats_delta_rms"], 0.01)
        self.assertLess(local["adaptive_to_fixed_response"], 1e-4)
        self.assertGreater(global_["adaptive_to_fixed_response"], 0.5)
        self.assertAlmostEqual(local["shift_direction_fraction"] + local["scale_direction_fraction"], 1)
        self.assertAlmostEqual(local["orthogonal_direction_fraction"], 0)

    def test_fixed_statistics_control_matches_explicit_affine_transform(self):
        torch.manual_seed(7)
        x = torch.randn(4, 8, 3, 3, 3)
        for module in (torch.nn.GroupNorm(4, 8), ChannelLayerNorm3d(8)):
            with self.subTest(module=type(module).__name__):
                base_module = module.norm if isinstance(module, ChannelLayerNorm3d) else module
                with torch.no_grad():
                    base_module.weight.copy_(torch.linspace(0.2, 1.6, 8))
                    base_module.bias.copy_(torch.linspace(-1, 1, 8))
                rows = audit.layer_response(module, x, module(x))
                a, b = x[::2].double(), x[1::2].double()
                if isinstance(module, torch.nn.GroupNorm):
                    mean = a.reshape(2, 4, -1).mean(-1, keepdim=True)
                    scale = (a.reshape(2, 4, -1).var(-1, unbiased=False, keepdim=True) + module.eps).sqrt()
                    transform = lambda value: ((value.reshape(2, 4, -1) - mean) / scale).reshape_as(value)
                else:
                    mean = a.mean(1, keepdim=True)
                    scale = (a.var(1, unbiased=False, keepdim=True) + module.norm.eps).sqrt()
                    transform = lambda value: (value - mean) / scale
                expected = (transform(b) - transform(a)) * base_module.weight.detach().double()[
                    None, :, None, None, None
                ]
                np.testing.assert_allclose(
                    [r["fixed_stats_delta_rms"] for r in rows], expected.square().mean((1, 2, 3, 4)).sqrt()
                )
                for row in rows:
                    self.assertAlmostEqual(
                        sum(
                            row[k]
                            for k in (
                                "shift_direction_fraction",
                                "scale_direction_fraction",
                                "orthogonal_direction_fraction",
                            )
                        ),
                        1,
                    )

    def test_capture_preserves_pre_relu_values_view_order_and_state(self):
        for shared in (False, True):
            model = build_model(config("layer", no_separate_encoders=shared), "cpu")
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            x = torch.randn(8, 1, 16, 16, 16)
            before = state_digest(model)
            self.assertEqual(
                audit.resolve_layers(model, ["early", "pre_residual"]), ["layers.0.1", "layers.1.1", "layers.3"]
            )
            # Early maps are 8^3, larger than the final 4^3 backbone map.
            values, metadata, native = audit.capture(model, x, "layers.0.1", 8, with_response=True)
            self.assertLess(values["norm_post_spatial"].min(), -0.1)
            self.assertEqual(values["norm_pre_spatial"].shape[0], 8)
            self.assertEqual(len(native), 4)
            self.assertEqual(metadata["normalizers"][0]["type"], "ChannelLayerNorm3d")
            self.assertEqual(state_digest(model), before)
            self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))
            self.assertFalse(any(m._forward_hooks for m in model.modules()))
            with self.assertRaisesRegex(ValueError, "exceeds tapped"):
                audit.capture(model, x, "layers.0.1", 9)
            self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_real_pairs_replay_and_no_moves(self):
        cfg = config("layer")
        model, ds = build_model(cfg, "cpu"), dataset(cfg, 20, "test")
        args = options()
        for eps in (0.5, 1e-12):
            args.eps = eps
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                pairs = audit.extract_pairs(model, ds, args, "cpu", Path(tmp) / "pairs", "layers.0.1")
                try:
                    self.assertEqual(pairs[1].shape, (18, 3))
                    self.assertEqual(len(pairs[4]), 18)
                    for replay in pairs[3].values():
                        np.testing.assert_array_equal(replay, 0)
                    if eps < 1e-10:
                        self.assertFalse(any(row["moved"] for row in pairs[2]))
                        self.assertTrue(all(row["post_native_delta_rms"] == 0 for row in pairs[4]))
                        for feature in pairs[0].values():
                            np.testing.assert_array_equal(feature[::2], feature[1::2])
                    else:
                        self.assertTrue(any(row["moved"] for row in pairs[2]))
                finally:
                    audit.close_arrays(pairs[0])

    def test_intervention_targets_do_not_influence_probe_fitting(self):
        rng = np.random.default_rng(44)
        val, endpoints = rng.normal(size=(40, 3)), rng.normal(size=(12, 3))
        reference = {(view, stage): val for view in audit.VIEWS for stage in audit.STAGES}
        arrays = {(view, stage): endpoints for view in audit.VIEWS for stage in audit.STAGES}
        rows = [dict(subject_id=i, intervention_axis="x", moved=True) for i in range(6)]
        replay = {key: np.zeros(6) for key in arrays}
        pairs = (arrays, endpoints.copy(), rows, replay, [], {})
        info = dict(run="a", checkpoint="trained", normalization="GroupNorm", layer="layers.0.1", probe_grid=2)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(axes=["x"])
            first = audit.score(reference, val, pairs, args, info, Path(tmp))
            pairs[1][:] += 100
            second = audit.score(reference, val, pairs, args, {**info, "run": "b"}, Path(tmp))
            keys = ("view", "stage", "probe", "condition", "target", "alpha", "gamma", "validation_mse_standardized")
            self.assertEqual([[r[k] for k in keys] for r in first[1]], [[r[k] for k in keys] for r in second[1]])
            with np.load(Path(tmp) / "a__trained__layers.0.1_predictions.npz") as a, np.load(
                Path(tmp) / "b__trained__layers.0.1_predictions.npz"
            ) as b:
                for key in a.files:
                    if key != "truth":
                        np.testing.assert_array_equal(a[key], b[key])

    def create_runs(self, root):
        # Deliberately misleading labels: actual class must determine the report.
        for label, norm in (("gn", "layer"), ("ln", "group")):
            run = root / label
            run.mkdir()
            cfg = config(norm)
            (run / "settings.json").write_text(json.dumps(cfg))
            model = build_model(cfg, "cpu")
            for filename in ("model.pt", "model_init.pt"):
                torch.save(model.state_dict(), run / filename)
        return [
            "--run",
            f"gn={root/'gn'}",
            "--run",
            f"ln={root/'ln'}",
            "--out-dir",
            str(root / "audit"),
            "--num-samples",
            "2",
            "--subject-offset",
            "0",
            "--axes",
            "x",
            "--spatial-grid",
            "2",
            "--norm-layers",
            "early",
            "--bootstrap",
            "4",
            "--device",
            "cpu",
        ]

    def test_two_runs_initial_and_trained_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            argv = self.create_runs(root)
            before = {p: digest(p) for folder in (root / "gn", root / "ln") for p in folder.iterdir()}
            with patch.object(torch.Tensor, "backward", side_effect=AssertionError("No training")):
                report = audit.main(argv)
            self.assertEqual(report["status"], "complete")
            self.assertEqual({r["checkpoint"] for r in report["summary"]}, {"trained", "initial"})
            self.assertTrue(
                all(
                    r["normalization"] == ("ChannelLayerNorm3d" if r["run"] == "gn" else "GroupNorm")
                    for r in report["summary"]
                )
            )
            self.assertEqual(len({cohort["moves"]["input_sha256"] for cohort in report["cohorts"].values()}), 1)
            self.assertEqual(before, {p: digest(p) for p in before})
            self.assertFalse(list((root / "audit").rglob("*.npy")))
            self.assertFalse(list((root / "audit").glob(".features-*")))
            for name in ("summary.csv", "normalization_response.csv", "sensitivity.csv", "pairs.csv"):
                self.assertTrue((root / "audit" / name).exists())
            for name in ("gn", "ln"):
                with np.load(root / "audit" / f"{name}__trained__layers.0.1_predictions.npz") as a, np.load(
                    root / "audit" / f"{name}__initial__layers.0.1_predictions.npz"
                ) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])

    def test_failure_cleans_banks_and_marks_report_failed(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            argv = self.create_runs(root)
            with patch.object(audit, "score", side_effect=RuntimeError("intentional failure")):
                with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                    audit.main(argv)
            self.assertEqual(json.loads((root / "audit/report.json").read_text())["status"], "failed")
            self.assertFalse(list((root / "audit").glob(".features-*")))


if __name__ == "__main__":
    unittest.main()
