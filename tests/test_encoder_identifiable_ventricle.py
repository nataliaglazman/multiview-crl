"""--synthetic-identifiable-ventricle on the encoder-only path: trainer, evaluation factories and launchers."""

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

from eval.encoder.encoder_target_protocol import dataset
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.synthetic_dataset import PseudoMRIRenderer
from scripts import generate_conv_patch_slurm, run_encoder_mps
from tests.test_encoder_mps_runner import runner_args, trainer_parse_args
from tests.test_lesion_intensity import trainer_make_dataset

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (torch.zeros(4, 4, 4), torch.zeros(8, 8, 8), "cpu")


class IdentifiableVentricleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_flag_gives_an_undeformed_larger_ventricle_and_a_separate_fissure(self):
        ventricles = {}
        for flag in (False, True):
            renderer = PseudoMRIRenderer(res=48, identifiable_ventricle=flag)
            lateral = renderer.coords[..., 0].abs() > 0.05  # the ventricle split; the fissure lies at |x| < 0.03
            for asymmetry in (-1.0, 1.0):
                z = torch.zeros(9)
                z[7] = asymmetry  # L-R asymmetry shifts the deformed radius
                tissue, _ = renderer.render_structure(z, *FIELDS, clean=True)
                ventricles[flag, asymmetry] = (tissue == 1) & lateral
                self.assertEqual(bool((tissue == 4).any()), flag)
        self.assertTrue(torch.equal(ventricles[True, -1.0], ventricles[True, 1.0]))
        self.assertFalse(torch.equal(ventricles[False, -1.0], ventricles[False, 1.0]))
        # At z1 = 0 the radius is 0.20 instead of 0.15.
        self.assertGreater(int(ventricles[True, 1.0].sum()), int(ventricles[False, 1.0].sum()))

    def test_flag_reaches_training_and_encoder_evaluation_factories(self):
        parse = trainer_parse_args()
        base = ["--res", "16", "--no-cache", "--synthetic-clean-content", "--synthetic-lesion-placement", "wm_interior"]
        base += ["--synthetic-normalize", "fixed_reference"]
        self.assertFalse(parse(base).synthetic_identifiable_ventricle)
        args = parse(base + ["--synthetic-identifiable-ventricle"])
        settings = json.loads(json.dumps(vars(args)))  # what training saves as settings.json
        trained = trainer_make_dataset()(args, "val", 4)
        scored = make_dataset(settings, 4, "val")
        followup = dataset(settings, 4, "val")  # the encoder follow-ups used to refuse this setting
        for ds in (trained, scored, followup):
            self.assertTrue(ds._inner.renderer.identifiable_ventricle)
        for a, b in zip(trained._inner[0][:2], scored._inner[0][:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # settings.json files written before the flag existed restore the old ventricle.
        del settings["synthetic_identifiable_ventricle"]
        self.assertFalse(make_dataset(settings, 4, "val")._inner.renderer.identifiable_ventricle)

    def test_launchers_add_only_the_flag_and_a_distinct_run_id(self):
        parse = trainer_parse_args()
        for device in ("cuda", "mps"):
            common = dict(variant="conv_mlp", patch_loss_weight=1, train_patch_grid=[8] * 3, device=device)
            plain, _ = run_encoder_mps.make_options(runner_args(**common))
            flagged, _ = run_encoder_mps.make_options(runner_args(**common, synthetic_identifiable_ventricle=True))
            self.assertEqual(
                {k for k in plain.keys() | flagged.keys() if plain.get(k) != flagged.get(k)},
                {"synthetic_identifiable_ventricle", "model_id"},
            )
            self.assertEqual(flagged["model_id"], plain["model_id"] + "_identvent")
            preview = subprocess.check_output(
                [
                    "bash",
                    str(ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"),
                    "--synthetic-identifiable-ventricle",
                    "--dry-run",
                ],
                env={**os.environ, "ENCODER_PYTHON": sys.executable},
                text=True,
            )
            parsed = parse(shlex.split(preview)[3:])
            self.assertTrue(parsed.synthetic_identifiable_ventricle)
            self.assertTrue(parsed.model_id.endswith("_identvent"))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            original = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            before = original.read_bytes()
            generate_conv_patch_slurm.main(["--output-dir", tmp, "--synthetic-identifiable-ventricle"])
            self.assertEqual(before, original.read_bytes())
            flagged_script = Path(tmp) / "encoder_conv_mlp_patch_identvent_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(flagged_script)], check=True)
            configs = []
            for script in (original, flagged_script):
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], text=True)
                configs.append(vars(parse(shlex.split(preview)[3:])))
            self.assertEqual(
                {k for k in configs[0] if configs[0][k] != configs[1][k]},
                {"synthetic_identifiable_ventricle", "model_id"},
            )
            self.assertEqual(configs[1]["model_id"], configs[0]["model_id"].replace("_w1_s42", "_w1_identvent_s42"))


if __name__ == "__main__":
    unittest.main()
