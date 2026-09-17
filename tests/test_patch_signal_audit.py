"""Scientific controls for checkpoint-free patch/loss sensitivity.

Load the unchanged loss functions without unrelated LPIPS/ADNI dependencies.
The production CLI imports training.losses normally.
"""

import ast
import contextlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from eval import patch_signal_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def source_losses():
    path = ROOT / "training/losses.py"
    tree = ast.parse(path.read_text())
    names = {"barlow_twins_loss", "stats_pool", "_center_patch_features", "_merge_diags", "_whiten_batch"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = types.ModuleType("training.losses")
    module.__dict__.update(torch=torch, F=F, contextlib=contextlib, np=np)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class PatchSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.losses = source_losses()
        spec = importlib.util.spec_from_file_location("routing_fixtures", ROOT / "tests/test_ventricle_routing.py")
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        cls.datasets = fixtures.load_without_monai("data/datasets.py")

    def settings(self, **kw):
        result = audit.DEFAULTS.copy()
        result.update(bt_corr_ema=0, bt_lambda=0, bt_sim_coeff=1, bt_std_coeff=0)
        result.update(kw)
        return result

    def measure(self, common, target, shuffled, settings=None, arm="patch"):
        with patch.dict("sys.modules", {"training.losses": self.losses}):
            return audit.measure_loss(
                common,
                target,
                shuffled,
                torch.ones(target.shape[-1], dtype=torch.bool),
                settings or self.settings(),
                arm,
            )

    def test_density_mse_has_exact_occupancy_scaling(self):
        rows = []
        for k in (1, 8, 64):
            signal = torch.zeros(64, 64)
            signal[:, :k] = 1
            c, t, s = audit.planted_features(signal, 4, "mixed", 1, 2)
            row = self.measure(c, t, s)
            self.assertAlmostEqual(row["mismatch_delta_sim_loss"], float((s - t).square().mean()), places=6)
            rows.append(row)
        self.assertAlmostEqual(rows[1]["mismatch_delta_sim_loss"] / rows[0]["mismatch_delta_sim_loss"], 8, places=5)
        self.assertAlmostEqual(rows[2]["mismatch_delta_sim_loss"] / rows[0]["mismatch_delta_sim_loss"], 64, places=5)

    def test_dedicated_channel_correlation_resists_dilution(self):
        values = []
        for k in (1, 64):
            signal = torch.zeros(64, 64)
            signal[:, :k] = 1
            c, t, s = audit.planted_features(signal, 4, "dedicated", 1, 2)
            values.append(self.measure(c, t, s)["mismatch_delta_on_diag_loss"])
        self.assertGreater(values[0], 0.1)
        self.assertAlmostEqual(values[0], values[1], places=4)

    def test_zero_signal_is_a_true_null(self):
        c, t, s = audit.planted_features(torch.zeros(8, 64), 4, "mixed", 1, 0)
        row = self.measure(c, t, s, self.settings(bt_corr_ema=0.99))
        for key in (
            "mismatch_delta",
            "both_removed_delta",
            "mismatch_directional_gradient",
            "ema_one_step_mismatch_delta",
            "ema_one_step_directional_gradient",
        ):
            self.assertEqual(row[key], 0)

    def test_gradient_matches_finite_difference_and_ema_keeps_mse_gradient(self):
        c, t, s = audit.planted_features(torch.randn(16, 16), 4, "mixed", 1, 0)
        settings = self.settings(bt_corr_ema=0.99)
        row = self.measure(c, t, s, settings)
        kwargs = audit.loss_settings(settings, "patch")

        def objective(r):
            return float(
                self.losses.barlow_twins_loss(
                    torch.stack((c + t, c + t + r * (s - t))),
                    estimated_content_indices=[list(range(4))],
                    subsets=[(0, 1)],
                    **kwargs
                )
            )

        finite_difference = (objective(1.001) - objective(0.999)) / 0.002
        self.assertAlmostEqual(finite_difference, row["mismatch_directional_gradient"], delta=2e-4)
        # In the converged matched reference, correlation gradients are strongly
        # attenuated; the un-EMA'd raw MSE derivative remains 2*mean(delta²).
        expected = 2 * float((s - t).square().mean())
        self.assertAlmostEqual(row["ema_one_step_directional_gradient"], expected, delta=2e-4)

    def test_gap_and_stats_reuse_the_actual_pooling(self):
        c, t, s = audit.planted_features(torch.randn(16, 8), 4, "mixed", 1, 0)
        for pooling in ("gap", "stats"):
            settings = self.settings(bt_gap_pooling=pooling)
            row = self.measure(c, t, s, settings, "gap")
            hz = torch.stack((c + t, c + s))
            pooled = hz.mean(-1) if pooling == "gap" else self.losses.stats_pool(hz)[0]
            expected = float(self.losses.barlow_twins_loss(pooled, **audit.loss_settings(settings, "gap")))
            self.assertAlmostEqual(row["mismatch_loss"], expected, places=6)

    def test_per_position_selects_channels_not_patch_indices(self):
        c, t, s = audit.planted_features(torch.randn(16, 64), 4, "mixed", 1, 0)
        result = self.measure(c, t, s, self.settings(bt_patch_stat="per_position"))
        self.assertTrue(np.isfinite(result["mismatch_loss"]))

    def test_input_energy_lifting_and_signed_cancellation(self):
        x = torch.ones(2, 1, 8, 8, 8)
        x[1, :, ::2] = -1
        pooled = audit.pool(x, 2)
        ratio = pooled.square().sum(1) * 64 / x.square().flatten(1).sum(1)
        torch.testing.assert_close(ratio, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(audit.pool(x.abs(), 2), torch.ones(2, 8))

    def test_renderer_both_factors_and_cli_never_load_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = audit.DEFAULTS.copy()
            settings.update(synthetic_res=16, synthetic_lesion_radius=0.2)
            config = Path(tmp) / "settings.json"
            config.write_text(json.dumps(settings))
            out = Path(tmp) / "report"
            cli = audit.parser().parse_args(
                [
                    "--settings",
                    str(config),
                    "--num-samples",
                    "6",
                    "--batch-size",
                    "4",
                    "--grids",
                    "4",
                    "--out-dir",
                    str(out),
                ]
            )
            with patch.dict(
                "sys.modules", {"data.datasets": self.datasets, "training.losses": self.losses}
            ), patch.object(
                torch, "load", side_effect=AssertionError("No checkpoint allowed")
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                report = audit.main(cli)
            saved = json.loads((out / "summary.json").read_text())
            self.assertEqual({r["factor"] for r in saved["geometry"]}, {"ventricle", "lesion"})
            self.assertEqual({r["view"] for r in saved["loss"]}, {"t1", "flair"})
            for r in saved["geometry"]:
                for field in ("pooled_energy_retained", "masked_energy_retained"):
                    if r[field]["mean"] is not None:
                        self.assertLessEqual(r[field]["max"], 1 + 1e-6)
                        self.assertGreaterEqual(r[field]["min"], 0)
            self.assertEqual(len(report["density_control"]), 16)
            for name in ("geometry.csv", "loss.csv", "density_control.csv"):
                self.assertTrue((out / name).is_file())
            with self.assertRaises(FileExistsError):
                audit.main(cli)

    def test_rejects_single_subject_tail(self):
        cli = audit.parser().parse_args(["--num-samples", "5", "--batch-size", "4"])
        with self.assertRaisesRegex(ValueError, "at least two"):
            audit.main(cli)


if __name__ == "__main__":
    unittest.main()
