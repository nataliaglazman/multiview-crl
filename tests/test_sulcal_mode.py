"""--synthetic-sulcal-mode atrophy: renderer, encoder-only trainer, evaluation targets and launchers."""

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

import torch

from data.datasets import SyntheticBrainDataset
from eval.encoder.encoder_target_protocol import dataset, sample_targets
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.synthetic_dataset import SULCAL_CLEFT_HALF_WIDTH, PseudoMRIRenderer, sulcal_cleft_distance
from scripts import generate_conv_patch_slurm, run_encoder_mps
from tests.test_encoder_mps_runner import runner_args, trainer_parse_args
from tests.test_encoder_target_followups import config
from tests.test_lesion_intensity import trainer_make_dataset

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (torch.zeros(4, 4, 4), torch.zeros(8, 8, 8), "cpu")
LUT = {"T1": torch.tensor([0.0, 0.1, 0.8, 0.5, 0.3]), "FLAIR": torch.tensor([0.0, 0.1, 0.4, 0.8, 0.3])}


def anatomy(sulcal, size=0.3):
    return torch.tensor([size, -0.2, 0.1, 0.4, -0.5, 0.2, 0.6, -0.4, sulcal])


class SulcalModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.atrophy = PseudoMRIRenderer(res=48, sulcal_mode="atrophy")
        cls.corrugation = PseudoMRIRenderer(res=48)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def render(self, renderer, z):
        return renderer.render_structure(z, *FIELDS, clean=True)

    def test_default_is_corrugation_and_unknown_modes_raise(self):
        self.assertEqual(self.corrugation.sulcal_mode, "corrugation")
        self.assertNotIn("cleft_distance", dict(self.corrugation.named_buffers()))
        self.assertNotIn("cleft_distance", self.atrophy.state_dict())  # derived from coords, never saved
        with self.assertRaisesRegex(ValueError, "sulcal_mode"):
            PseudoMRIRenderer(res=8, sulcal_mode="widen")
        with self.assertRaisesRegex(ValueError, "sulcal_mode"):
            SyntheticBrainDataset(spatial_size=(16,) * 3, synthetic_num_samples=2, synthetic_sulcal_mode="widen")

    def test_atrophy_turns_only_grey_matter_into_csf(self):
        reference, lesion = self.render(self.atrophy, anatomy(-2.0))
        csf = []
        for sulcal in (-2.0, -0.5, 0.5, 2.0):
            tissue, load = self.render(self.atrophy, anatomy(sulcal))
            torch.testing.assert_close(load, lesion, rtol=0, atol=0)
            # The outline, white matter, ventricles and fissure stay put; wider clefts only take grey matter.
            self.assertTrue(torch.equal(tissue > 0, reference > 0))
            self.assertTrue(torch.equal(tissue == 2, reference == 2))
            changed = tissue != reference
            self.assertTrue(set(tissue[changed].tolist()) <= {1} and set(reference[changed].tolist()) <= {3})
            csf.append(int((tissue == 1).sum()))
        self.assertTrue(all(a < b for a, b in zip(csf, csf[1:])), csf)
        # The corrugation moves the brain outline instead.
        a, _ = self.render(self.corrugation, anatomy(-1.0))
        b, _ = self.render(self.corrugation, anatomy(1.0))
        self.assertFalse(torch.equal(a > 0, b > 0))

    def test_atrophy_darkens_both_views_on_average_unlike_the_corrugation(self):
        # GM is brighter than CSF in T1 (0.5 vs 0.1) and FLAIR (0.8 vs 0.1), so the noiseless brain mean
        # falls with z8; the zero-mean corrugation leaves it almost where it was.
        for view, lut in LUT.items():
            shift = {}
            for name, renderer in (("atrophy", self.atrophy), ("corrugation", self.corrugation)):
                means = []
                for sulcal in (-2.0, 0.0, 2.0):
                    tissue, _ = self.render(renderer, anatomy(sulcal))
                    means.append(float(lut[tissue][tissue > 0].mean()))
                shift[name] = means[2] - means[0]
                if name == "atrophy":
                    self.assertTrue(means[0] > means[1] > means[2], (view, means))
            self.assertGreater(abs(shift["atrophy"]), 3 * abs(shift["corrugation"]), (view, shift))

    def test_cleft_share_follows_z8_not_brain_size(self):
        # Clefts depend on direction only. (Far from every cleft, where the gyroid's gradient vanishes,
        # the first-order distance is huge and unstable, which no cleft width ever reaches.)
        near, scaled = sulcal_cleft_distance(self.atrophy.coords), sulcal_cleft_distance(2.5 * self.atrophy.coords)
        for width in (0.012, 0.031, 0.05):
            self.assertTrue(torch.equal(near < width, scaled < width))
        torch.testing.assert_close(near[near < 0.1], scaled[near < 0.1], rtol=0, atol=1e-5)

        def share(sulcal, size):
            tissue, _ = self.render(self.atrophy, anatomy(sulcal, size))
            ribbon, _ = self.render(self.corrugation, anatomy(0.0, size))  # zero corrugation: no clefts
            return float(((tissue == 1) & (ribbon == 3)).sum() / (ribbon == 3).sum())

        self.assertLess(abs(share(0.5, -1.0) - share(0.5, 1.0)), 0.02)
        self.assertGreater(share(1.0, 0.3) - share(-1.0, 0.3), 0.1)

    def test_evaluation_targets_hold_the_cleft_half_width(self):
        for mode in ("corrugation", "atrophy"):
            ds = dataset(config(res=32, synthetic_sulcal_mode=mode), 4, "val")
            latents = ds._inner[0][2]
            target, _ = sample_targets(ds._inner, latents)
            value = float(torch.tanh(latents["z_content"][8]))
            amplitude, magnitude = float(target[-2]), float(target[-1])
            if mode == "atrophy":
                mid, half = SULCAL_CLEFT_HALF_WIDTH
                self.assertAlmostEqual(amplitude, mid + half * value, places=6)
                self.assertEqual(amplitude, magnitude)
            else:
                self.assertAlmostEqual(amplitude, 0.06 * value, places=6)
                self.assertAlmostEqual(magnitude, abs(0.06 * value), places=6)

    def test_flag_reaches_training_and_encoder_evaluation_factories(self):
        parse = trainer_parse_args()
        base = ["--res", "16", "--no-cache", "--synthetic-clean-content", "--synthetic-lesion-placement", "wm_interior"]
        base += ["--synthetic-normalize", "fixed_reference"]
        self.assertEqual(parse(base).synthetic_sulcal_mode, "corrugation")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(base + ["--synthetic-sulcal-mode", "widen"])
        args = parse(base + ["--synthetic-sulcal-mode", "atrophy"])
        settings = json.loads(json.dumps(vars(args)))  # what training saves as settings.json
        trained = trainer_make_dataset()(args, "val", 4)
        scored = make_dataset(settings, 4, "val")
        followup = dataset(settings, 4, "val")
        for ds in (trained, scored, followup):
            self.assertEqual(ds._inner.renderer.sulcal_mode, "atrophy")
        for a, b in zip(trained._inner[0][:2], scored._inner[0][:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # settings.json files written before the flag existed restore the corrugation.
        del settings["synthetic_sulcal_mode"]
        self.assertEqual(make_dataset(settings, 4, "val")._inner.renderer.sulcal_mode, "corrugation")

    def test_launchers_add_only_the_flag_and_a_distinct_run_id(self):
        parse = trainer_parse_args()
        for device in ("cuda", "mps"):
            common = dict(variant="conv_mlp", patch_loss_weight=1, train_patch_grid=[8] * 3, device=device)
            plain, _ = run_encoder_mps.make_options(runner_args(**common))
            flagged, _ = run_encoder_mps.make_options(runner_args(**common, synthetic_sulcal_mode="atrophy"))
            self.assertEqual(
                {k for k in plain.keys() | flagged.keys() if plain.get(k) != flagged.get(k)},
                {"synthetic_sulcal_mode", "model_id"},
            )
            self.assertEqual(flagged["model_id"], plain["model_id"] + "_sulcalatrophy")
            preview = subprocess.check_output(
                [
                    "bash",
                    str(ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"),
                    "--synthetic-sulcal-mode",
                    "atrophy",
                    "--dry-run",
                ],
                env={**os.environ, "ENCODER_PYTHON": sys.executable},
                text=True,
            )
            parsed = parse(shlex.split(preview)[3:])
            self.assertEqual(parsed.synthetic_sulcal_mode, "atrophy")
            self.assertTrue(parsed.model_id.endswith("_sulcalatrophy"))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            original = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            before = original.read_bytes()
            generate_conv_patch_slurm.main(["--output-dir", tmp, "--synthetic-sulcal-mode", "atrophy"])
            self.assertEqual(before, original.read_bytes())
            flagged_script = Path(tmp) / "encoder_conv_mlp_patch_sulcalatrophy_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(flagged_script)], check=True)
            configs = []
            for script in (original, flagged_script):
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], text=True)
                configs.append(vars(parse(shlex.split(preview)[3:])))
            self.assertEqual(
                {k for k in configs[0] if configs[0][k] != configs[1][k]}, {"synthetic_sulcal_mode", "model_id"}
            )
            self.assertEqual(configs[1]["model_id"], configs[0]["model_id"].replace("_w1_s42", "_w1_sulcalatrophy_s42"))


if __name__ == "__main__":
    unittest.main()
