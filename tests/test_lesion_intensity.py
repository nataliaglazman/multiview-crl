"""Lesion on the acquisition gain/bias map: renderer, encoder and VQ-VAE trainers, evaluation and launchers."""

import ast
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, maximum_filter

from data.datasets import SyntheticBrainDataset
from eval.encoder.encoder_lesion_contrast import contrast_metrics
from eval.encoder.encoder_target_protocol import dataset
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.synthetic_dataset import PseudoMRIRenderer
from scripts import generate_conv_patch_slurm, run_encoder_mps
from tests.test_encoder_mps_runner import runner_args, trainer_parse_args
from tests.test_encoder_target_followups import config

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = [-1.0, -1.0, 0.5], [1.0, 1.0, 0.5]  # gain 0.7, bias -0.1 / gain 1.3, bias +0.1


def trainer_make_dataset():
    # The real factory, without importing the trainer (training.losses needs LPIPS).
    path = ROOT / "training/main_conv_synthetic.py"
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_dataset"]
    namespace = {"SyntheticBrainDataset": SyntheticBrainDataset}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["make_dataset"]


def vqvae_dataset_kwargs(args):
    # The dataset_kwargs block of training.main_multimodal.main, without importing the trainer.
    path = ROOT / "training/main_multimodal.py"
    main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["dataset_kwargs"]
    )
    block = main.body[start : start + 2]  # noqa: E203
    assert isinstance(block[1], ast.If) and "dataset_kwargs.update" in ast.unparse(block[1].body[0])
    namespace = {"args": args}
    exec(compile(ast.Module(body=block, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["dataset_kwargs"]


class LesionIntensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        ds = dataset(config(res=32, synthetic_lesion_radius=0.14), 64, "test")
        lat = ds._inner[0][2]
        cls.tissue, cls.lesion = ds._inner.renderer.render_structure(
            lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=True
        )
        cls.mask = cls.lesion.numpy() > 0

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def render(self, mode, style, view, lesion=None):
        renderer = PseudoMRIRenderer(res=32, lesion_intensity=mode)
        load = self.lesion if lesion is None else lesion
        return renderer.render_modality(self.tissue, load, torch.tensor(style), view, 7, "cpu").numpy()[0]

    def test_neutral_style_is_unchanged_and_fixed_lesion_ignores_style(self):
        # A 3x3x3-blurred lesion-core voxel reads only lesion voxels.
        core = binary_erosion(self.mask, np.ones((3, 3, 3)))
        self.assertTrue(core.any())
        for view in ("T1", "FLAIR"):
            # At gain 1 and bias 0 the styled lesion takes its legacy value, bit for bit.
            np.testing.assert_array_equal(
                self.render("styled", [0.0, 0.0, 0.5], view), self.render("fixed", [0.0, 0.0, 0.5], view)
            )
            low, high = self.render("fixed", LOW, view), self.render("fixed", HIGH, view)
            np.testing.assert_array_equal(low[core], high[core])
            low, high = self.render("styled", LOW, view), self.render("styled", HIGH, view)
            self.assertGreater(np.abs(low[core] - high[core]).min(), 0.1)

    def test_styled_contrast_follows_gain_and_ignores_bias(self):
        zero = torch.zeros_like(self.lesion)

        def contrast(mode, gain, bias, view):
            style = [gain, bias, 0.0]
            image, reference = self.render(mode, style, view), self.render(mode, style, view, lesion=zero)
            return contrast_metrics(image, reference, self.mask, self.tissue.numpy())["signed_matched_contrast"]

        for view, sign in (("T1", -1), ("FLAIR", 1)):
            styled = {(g, b): contrast("styled", g, b, view) for g in (-1.0, 1.0) for b in (-1.0, 1.0)}
            fixed = {(g, b): contrast("fixed", g, b, view) for g in (-1.0, 1.0) for b in (-1.0, 1.0)}
            self.assertTrue(all(np.sign(c) == sign for c in [*styled.values(), *fixed.values()]))
            for g in (-1.0, 1.0):
                self.assertAlmostEqual(styled[g, -1.0] / styled[g, 1.0], 1.0, delta=0.02)
                # The defect being fixed: with a fixed lesion, bias alone moves the contrast.
                self.assertGreater(abs(fixed[g, -1.0] / fixed[g, 1.0] - 1.0), 0.2)
            self.assertAlmostEqual(styled[1.0, 1.0] / styled[-1.0, 1.0], 1.3 / 0.7, delta=0.03)

    def test_raw_render_changes_only_inside_lesion_and_blur_ring(self):
        outside = ~maximum_filter(self.mask, size=3)
        for view in ("T1", "FLAIR"):
            styled, fixed = (self.render(mode, [-0.8, -0.6, 0.4], view) for mode in ("styled", "fixed"))
            np.testing.assert_array_equal(styled[outside], fixed[outside])
            self.assertGreater(np.abs(styled - fixed)[~outside].max(), 0.05)

    def test_flag_reaches_training_and_encoder_evaluation_factories(self):
        parse = trainer_parse_args()
        base = ["--res", "16", "--no-cache", "--synthetic-clean-content", "--synthetic-lesion-placement", "wm_interior"]
        base += ["--synthetic-normalize", "fixed_reference"]
        self.assertEqual(parse(base).synthetic_lesion_intensity, "fixed")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(base + ["--synthetic-lesion-intensity", "bright"])
        args = parse(base + ["--synthetic-lesion-intensity", "styled"])
        settings = json.loads(json.dumps(vars(args)))  # what training saves as settings.json
        trained = trainer_make_dataset()(args, "val", 4)
        scored = make_dataset(settings, 4, "val")
        followup = dataset(settings, 4, "val")
        for ds in (trained, scored, followup):
            self.assertEqual(ds._inner.renderer.lesion_intensity, "styled")
        for a, b in zip(trained._inner[0][:2], scored._inner[0][:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # settings.json files written before the flag existed restore the legacy renderer.
        del settings["synthetic_lesion_intensity"]
        self.assertEqual(make_dataset(settings, 4, "val")._inner.renderer.lesion_intensity, "fixed")
        with self.assertRaisesRegex(ValueError, "lesion_intensity"):
            SyntheticBrainDataset(spatial_size=(16,) * 3, synthetic_num_samples=2, synthetic_lesion_intensity="bright")

    def test_flag_reaches_vqvae_training_and_evaluation(self):
        from eval.protocol.run_dci_synthetic import build_synthetic_test_set, load_run_args
        from utils.config import parse_args

        parser = parse_args()
        base = ["--dataset-name", "synthetic", "--synthetic-res", "16", "--synthetic-clean-content"]
        self.assertEqual(parser.parse_args(base).synthetic_lesion_intensity, "fixed")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base + ["--synthetic-lesion-intensity", "bright"])
        args = parser.parse_args(base + ["--synthetic-lesion-intensity", "styled"])
        trained = SyntheticBrainDataset(mode="test", spatial_size=(16,) * 3, cache=False, **vqvae_dataset_kwargs(args))
        settings = json.loads(json.dumps(vars(args), default=str))  # what training saves as settings.json
        with tempfile.TemporaryDirectory() as run_dir:
            Path(run_dir, "settings.json").write_text(json.dumps(settings))
            scored = build_synthetic_test_set(load_run_args(run_dir), 4, causal=False, cache=False)
            for ds in (trained, scored):
                self.assertEqual(ds._inner.renderer.lesion_intensity, "styled")
            for a, b in zip(trained._inner[0][:2], scored._inner[0][:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # settings.json files written before the flag existed restore the legacy renderer.
            del settings["synthetic_lesion_intensity"]
            Path(run_dir, "settings.json").write_text(json.dumps(settings))
            legacy = load_run_args(run_dir)
            self.assertEqual(vqvae_dataset_kwargs(legacy)["synthetic_lesion_intensity"], "fixed")
            scored = build_synthetic_test_set(legacy, 4, causal=False, cache=False)
            self.assertEqual(scored._inner.renderer.lesion_intensity, "fixed")

    def test_launchers_add_only_the_flag_and_a_distinct_run_id(self):
        parse = trainer_parse_args()
        for device in ("cuda", "mps"):
            common = dict(variant="conv_mlp", patch_loss_weight=1, train_patch_grid=[8] * 3, device=device)
            fixed, _ = run_encoder_mps.make_options(runner_args(**common))
            styled, _ = run_encoder_mps.make_options(runner_args(**common, synthetic_lesion_intensity="styled"))
            self.assertEqual(
                {k for k in fixed.keys() | styled.keys() if fixed.get(k) != styled.get(k)},
                {"synthetic_lesion_intensity", "model_id"},
            )
            self.assertEqual(styled["model_id"], fixed["model_id"] + "_lesionstyled")
            preview = subprocess.check_output(
                [
                    "bash",
                    str(ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"),
                    "--synthetic-lesion-intensity",
                    "styled",
                    "--dry-run",
                ],
                env={**os.environ, "ENCODER_PYTHON": sys.executable},
                text=True,
            )
            parsed = parse(shlex.split(preview)[3:])
            self.assertEqual(parsed.synthetic_lesion_intensity, "styled")
            self.assertTrue(parsed.model_id.endswith("_lesionstyled"))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            original = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            before = original.read_bytes()
            generate_conv_patch_slurm.main(["--output-dir", tmp, "--synthetic-lesion-intensity", "styled"])
            self.assertEqual(before, original.read_bytes())
            styled_script = Path(tmp) / "encoder_conv_mlp_patch_lesionstyled_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(styled_script)], check=True)
            configs = []
            for script in (original, styled_script):
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], text=True)
                configs.append(vars(parse(shlex.split(preview)[3:])))
            self.assertEqual(
                {k for k in configs[0] if configs[0][k] != configs[1][k]}, {"synthetic_lesion_intensity", "model_id"}
            )
            self.assertEqual(configs[1]["model_id"], configs[0]["model_id"].replace("_w1_s42", "_w1_lesionstyled_s42"))


if __name__ == "__main__":
    unittest.main()
