"""Lesion burden: total lesion volume as the content factor, behind --synthetic-lesion-target burden."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from eval.metrics import dci
from eval.protocol import score_checkpoint
from eval.synthetic.synthetic_dataset import PseudoMRIRenderer

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from scripts import run_encoder_mps as runner  # noqa: E402
from training import main_conv_synthetic as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BURDEN = ["--synthetic-lesion-placement", "wm_interior", "--synthetic-lesion-target", "burden"]
SMALL = ["--res", "32", "--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]


class BurdenRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = PseudoMRIRenderer(res=64, lesion_placement="wm_interior", lesion_target="burden")
        cls.quantiles = torch.rand(4, 3, generator=torch.Generator().manual_seed(0))

    def render(self, burden, quantiles=None):
        z = torch.zeros(9)
        z[2] = burden
        return self.renderer.render_structure(
            z,
            torch.zeros(4, 4, 4),
            torch.zeros(8, 8, 8),
            "cpu",
            clean=True,
            z_lesion=self.quantiles if quantiles is None else quantiles,
        )

    def test_total_volume_is_linear_in_the_burden_and_stays_in_white_matter(self):
        fractions, totals = [], []
        for value in torch.linspace(-3, 3, 13):
            tissue, load = self.render(value)
            self.assertTrue(bool(torch.all(tissue[load > 0] == 2)))
            self.assertLessEqual(load.max().item(), 1.0)
            fractions.append((torch.tanh(value).item() + 1) / 2)
            totals.append(load.sum().item())
        self.assertTrue(bool(np.all(np.diff(totals) > 0)))
        self.assertGreater(np.corrcoef(fractions, totals)[0, 1], 0.999)
        full = 4 * 4 / 3 * np.pi * 0.1**3 / (2 / 63) ** 3
        self.assertAlmostEqual(totals[-1] / (full * fractions[-1]), 1.0, delta=0.05)

    def test_zero_burden_renders_nothing_and_positions_ignore_the_burden(self):
        self.assertEqual(self.render(torch.tensor(-1e6))[1].sum().item(), 0.0)
        _, small = self.render(torch.tensor(-1.0))
        _, large = self.render(torch.tensor(1.5))
        self.assertGreater(small.sum().item(), 0.0)
        self.assertTrue(bool(torch.all(large[small > 0] > 0)))
        _, moved = self.render(torch.tensor(1.5), torch.rand(4, 3, generator=torch.Generator().manual_seed(1)))
        self.assertFalse(torch.equal(moved > 0, large > 0))
        torch.testing.assert_close(moved.sum(), large.sum())

    def test_invalid_options_fail(self):
        with self.assertRaisesRegex(ValueError, "wm_interior"):
            PseudoMRIRenderer(res=32, lesion_target="burden")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PseudoMRIRenderer(res=32, lesion_placement="wm_interior", lesion_target="burden", lesion_count=0)
        with self.assertRaisesRegex(ValueError, "position\\|burden"):
            PseudoMRIRenderer(res=32, lesion_placement="wm_interior", lesion_target="volume")
        with self.assertRaisesRegex(ValueError, "z_lesion"):
            self.renderer.render_structure(
                torch.zeros(9), torch.zeros(4, 4, 4), torch.zeros(8, 8, 8), "cpu", clean=True
            )


class BurdenPipelineTests(unittest.TestCase):
    def test_dataset_keeps_every_other_latent_and_reports_burden_names(self):
        position = trainer.make_dataset(
            trainer.parse_args(SMALL + ["--synthetic-lesion-placement", "wm_interior"]), "val", 6
        )
        burden = trainer.make_dataset(trainer.parse_args(SMALL + BURDEN), "val", 6)
        compared = 0
        for i in range(6):
            a, b = position[i]["gt_latents"], burden[i]["gt_latents"]
            self.assertEqual(tuple(b["z_lesion"].shape), (4, 3))
            if position._inner._accepted_attempt.get(i, 0) == 0 and burden._inner._accepted_attempt.get(i, 0) == 0:
                for key in ("z_content", "z_deformation", "z_fissure", "z_style_v1", "z_style_v2"):
                    torch.testing.assert_close(a[key], b[key])
                compared += 1
        self.assertGreater(compared, 0)
        self.assertEqual(dci.dataset_lesion_target(burden), "burden")
        self.assertEqual(dci.dataset_lesion_target(torch.utils.data.Subset(burden, [0])), "burden")
        self.assertEqual(dci.dataset_lesion_target(position), "position")
        self.assertEqual(dci.content_factor_names(9, "burden")[2:5], ["lesion_burden", "unused_3", "unused_4"])
        self.assertEqual(dci.content_factor_names(9), dci.CONTENT_FACTOR_NAMES)
        cfg = vars(trainer.parse_args(SMALL + BURDEN))
        self.assertEqual(score_checkpoint.make_dataset(cfg, 2)._inner.lesion_target, "burden")

    def test_cli_and_runner(self):
        parsed = trainer.parse_args(BURDEN)
        self.assertEqual((parsed.synthetic_lesion_target, parsed.synthetic_lesion_count), ("burden", 4))
        for argv in (
            ["--synthetic-lesion-target", "burden"],
            ["--synthetic-lesion-count", "3"],
            BURDEN + ["--synthetic-lesion-count", "0"],
            BURDEN + ["--lesion-keypoints", "2"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(argv)

        def runner_args(**changes):
            values = dict(
                config=ROOT / "experiments/encoder_comparison.json",
                variant="conv_mlp",
                seed=42,
                batch_size=None,
                train_steps=None,
                eval_every=None,
                model_id=None,
                results_dir=Path("/tmp/encoder_ablations_mps"),
            )
            values.update(changes)
            return types.SimpleNamespace(**values)

        options, _ = runner.make_options(runner_args(synthetic_lesion_target="burden"))
        self.assertEqual(options["model_id"], "conv_mlp_s42_mps_burden")
        parsed = trainer.parse_args(runner.comparison.cli_arguments(options))
        self.assertEqual(parsed.synthetic_lesion_target, "burden")
        six, _ = runner.make_options(runner_args(synthetic_lesion_target="burden", synthetic_lesion_count=6))
        self.assertEqual(six["model_id"], "conv_mlp_s42_mps_burden6")
        self.assertEqual(trainer.parse_args(runner.comparison.cli_arguments(six)).synthetic_lesion_count, 6)


if __name__ == "__main__":
    unittest.main()
