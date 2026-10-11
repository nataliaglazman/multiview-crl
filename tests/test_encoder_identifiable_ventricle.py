"""--synthetic-identifiable-ventricle on the encoder-only path: trainer, evaluation factories and launchers."""

import json
import unittest
from pathlib import Path

import torch

import training.encoder_args as encoder_args
from eval.encoder.encoder_target_protocol import dataset
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.synthetic_dataset import PseudoMRIRenderer
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
        parse = encoder_args.parse_args
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


if __name__ == "__main__":
    unittest.main()
