"""Style-block scoring in eval.score_checkpoint: one forward, rendered targets, leakage probes, CLI report."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval import score_checkpoint
from eval.dci import CONTENT_FACTOR_NAMES, STYLE_FACTOR_NAMES
from eval.style_path_audit import effective_style


def config(**kwargs):
    cfg = dict(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        downscale_factor=4,
        latent_dim=12,
        content_channels=9,
        seed=42,
        res=32,
        n_content=9,
        n_style=3,
        synthetic_normalize="fixed_reference",
        synthetic_clean_content=True,
        synthetic_mode="pseudo_mri",
        num_val_samples=24,
    )
    cfg.update(kwargs)
    return cfg


class StyleBlockTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_blocks_come_from_one_forward_and_encode_is_unchanged(self):
        cfg = config()
        ds = score_checkpoint.make_val_dataset(cfg, 12)
        model = score_checkpoint.build_model(cfg, "cpu")
        batch = next(iter(DataLoader(ds, batch_size=12)))
        for grid, cells in ((None, 1), ((2, 2, 2), 8)):
            blocks = score_checkpoint.encode_blocks(model, ds, "cpu", 4, 9, grid)
            with torch.no_grad():
                code = model(torch.cat(batch["image"]), pool_only=True, n_views=2, patch_grid=grid)[2][0]
            code = code.reshape(24, 12, -1).numpy()
            for view, rows in enumerate((slice(0, 12), slice(12, 24))):
                content, style = blocks["content"][view], blocks["style"][view]
                self.assertEqual((content.shape, style.shape), ((12, 9 * cells), (12, 3 * cells)))
                np.testing.assert_allclose(content, code[rows, :9].reshape(12, -1), atol=1e-6)
                np.testing.assert_allclose(style, code[rows, 9:].reshape(12, -1), atol=1e-6)
            legacy = score_checkpoint.encode(model, ds, "cpu", 4, 9, grid)
            np.testing.assert_array_equal(legacy[0], blocks["content"][0])
            np.testing.assert_array_equal(legacy[1], blocks["content"][1])
            np.testing.assert_array_equal(legacy[2], blocks["z_content"])

    def test_style_targets_are_the_parameters_each_view_was_rendered_with(self):
        cfg = config(synthetic_style_scale=1.5)
        ds = score_checkpoint.make_val_dataset(cfg, 8)
        blocks = score_checkpoint.encode_blocks(score_checkpoint.build_model(cfg, "cpu"), ds, "cpu", 4, 9)
        self.assertEqual(ds._inner.renderer.style_scale, 1.5)
        for view in range(2):
            expected = np.stack([effective_style(ds[i]["gt_latents"][f"z_style_v{view + 1}"], 1.5) for i in range(8)])
            np.testing.assert_allclose(blocks["style_targets"][view], expected)
        self.assertFalse(np.allclose(blocks["style_targets"][0], blocks["style_targets"][1]))

    def test_samples_without_a_renderer_have_no_style_targets(self):
        model = score_checkpoint.build_model(config(), "cpu")
        samples = [
            dict(
                image=[torch.randn(1, 32, 32, 32), torch.randn(1, 32, 32, 32)], gt_latents={"z_content": torch.randn(9)}
            )
            for _ in range(3)
        ]
        blocks = score_checkpoint.encode_blocks(model, samples, "cpu", 2, 9)
        self.assertIsNone(blocks["style_targets"])
        self.assertEqual(blocks["style"][0].shape, (3, 3))

    def test_planted_style_is_recovered_and_leaks_are_flagged(self):
        rng = np.random.default_rng(0)
        names, style_names = CONTENT_FACTOR_NAMES[:9], STYLE_FACTOR_NAMES
        z, targets = rng.normal(size=(200, 9)), rng.normal(size=(200, 3))
        style = targets @ rng.normal(size=(3, 3)) + 0.01 * rng.normal(size=(200, 3))
        content = z + 0.01 * rng.normal(size=(200, 9))
        clean = score_checkpoint.style_recovery(style, content, targets, z, names, style_names)
        self.assertEqual(clean["n_style_channels"], 3)
        self.assertGreater(clean["style_to_style"]["mean"], 0.95)
        self.assertLess(clean["content_to_style"]["mean"], 0.1)
        self.assertLess(clean["style_to_content"]["mean"], 0.1)
        # A content factor stored in the style units, and style stored in the content units.
        leaky = score_checkpoint.style_recovery(
            np.hstack([style, z[:, :1]]), np.hstack([content, targets[:, 1:2]]), targets, z, names, style_names
        )
        self.assertGreater(leaky["style_to_content"]["per_factor"][names[0]], 0.95)
        self.assertGreater(leaky["content_to_style"]["per_factor"]["bias"], 0.95)
        targets[:, 2] = 0.01  # a style factor that never varies has no defined R²
        constant = score_checkpoint.style_recovery(style, content, targets, z, names, style_names)
        self.assertTrue(np.isnan(constant["style_to_style"]["per_factor"]["noise_sigma"]))
        self.assertTrue(np.isfinite(constant["style_to_style"]["mean"]))

    def test_cli_report_scores_both_views_against_their_floor(self):
        for latent_dim, status in ((12, "ok"), (9, "no_style_units")):
            cfg = config(latent_dim=latent_dim)
            with tempfile.TemporaryDirectory() as temp:
                run = Path(temp)
                (run / "settings.json").write_text(json.dumps(cfg))
                torch.save(score_checkpoint.build_model(cfg, "cpu").state_dict(), run / "model.pt")
                argv = [
                    "score_checkpoint",
                    "--run-dir",
                    temp,
                    "--batch-size",
                    "8",
                    "--no-cuda",
                    "--no-graph",
                    "--no-dci",
                ]
                stream = io.StringIO()
                with patch("sys.argv", argv), contextlib.redirect_stdout(stream):
                    score_checkpoint.main()
                report = json.loads((run / "score_report.json").read_text())
            self.assertEqual(report["style_status"], status)
            if status != "ok":
                self.assertNotIn("style", report)
                self.assertIn("no style units", stream.getvalue())
                continue
            self.assertIn("style -> gain", stream.getvalue())
            for key in ("style", "style_floor"):
                self.assertEqual(set(report[key]), {"view1", "view2"})
                for view in report[key].values():
                    self.assertEqual(view["n_style_channels"], 3)
                    self.assertEqual(set(view["style_to_style"]["per_factor"]), set(STYLE_FACTOR_NAMES))
                    self.assertEqual(len(view["style_to_content"]["per_factor"]), 9)
            # The checkpoint IS the untrained twin here, so trained and floor must agree exactly.
            self.assertEqual(report["style"], report["style_floor"])


if __name__ == "__main__":
    unittest.main()
