"""--synthetic-lesion-margin: white matter around a wm_interior lesion, from the CLI to every evaluation factory."""

import argparse
import ast
import contextlib
import inspect
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

import training.encoder_args as encoder_args
from data.datasets import SyntheticBrainDataset
from eval.encoder.encoder_target_protocol import dataset
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.synthetic_dataset import LesionPlacementError, PseudoMRIRenderer

ROOT = Path(__file__).resolve().parents[1]
# The anatomy of the real encoder-only recipe: larger ventricle, sulcal clefts, every factor at full range.
RECIPE = dict(res=64, lesion_placement="wm_interior", identifiable_ventricle=True, sulcal_mode="atrophy")
CUBE = np.ones((3, 3, 3), dtype=bool)


def trainer_make_dataset():
    # The real factory, without importing the trainer (training.losses needs LPIPS).
    path = ROOT / "training/main_conv_synthetic.py"
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_dataset"]
    namespace = {"SyntheticBrainDataset": SyntheticBrainDataset}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["make_dataset"]


def structures(renderer, n, seed=0):
    """``n`` seeded anatomies; None where the lesion has no room (the dataset would redraw that subject)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        z = torch.randn(9, generator=g)
        try:
            out.append(renderer.render_structure(z, torch.zeros(4, 4, 4), torch.zeros(8, 8, 8), "cpu", clean=True))
        except LesionPlacementError:
            out.append(None)
    return out


def ring_outside_white_matter(tissue, load):
    support = load.numpy() > 0
    return bool(np.any(tissue.numpy()[binary_dilation(support, CUBE)] != 2))


class MarginRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_margin_keeps_the_smoothing_ring_in_white_matter_without_dropping_a_subject(self):
        plain = structures(PseudoMRIRenderer(**RECIPE), 12)
        ringed = structures(PseudoMRIRenderer(lesion_margin=1, **RECIPE), 12)
        # The margin never costs a subject its lesion, so exactly the same subjects are kept.
        self.assertEqual([s is None for s in ringed], [s is None for s in plain])
        touched = [ring_outside_white_matter(*s) for s in plain if s is not None]
        still = [ring_outside_white_matter(*s) for s in ringed if s is not None]
        self.assertGreater(sum(touched), 4)
        self.assertLess(sum(still), sum(touched))
        fallback = 0
        for before, after in zip(plain, ringed):
            if before is None:
                continue
            torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)  # anatomy is untouched
            self.assertEqual(int(after[1].sum()), int(before[1].sum()))  # one complete sphere, same radius
            if torch.equal(before[1], after[1]):
                fallback += 1  # no room for the margin: the brain keeps its usual lesion
            else:
                self.assertFalse(ring_outside_white_matter(*after))
        self.assertGreaterEqual(fallback, 1)

    def test_widest_margin_falls_back_to_what_the_brain_has_room_for(self):
        r = PseudoMRIRenderer(res=8, lesion_placement="wm_interior", lesion_margin=10)
        block = np.zeros((9, 9, 9), dtype=bool)
        block[1:8, 1:8, 1:8] = True
        admissible, margin = r._widest_margin(block)
        self.assertEqual(margin, 3)
        self.assertEqual(np.argwhere(admissible).tolist(), [[4, 4, 4]])
        r.lesion_margin = 2
        admissible, margin = r._widest_margin(block)
        self.assertEqual((margin, int(admissible.sum())), (2, 27))

    def test_zero_margin_is_the_default_placement(self):
        for before, after in zip(
            structures(PseudoMRIRenderer(**RECIPE), 4, seed=3),
            structures(PseudoMRIRenderer(lesion_margin=0, **RECIPE), 4, seed=3),
        ):
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_margins_are_refused(self):
        for margin in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                PseudoMRIRenderer(lesion_placement="wm_interior", lesion_margin=margin)
        with self.assertRaises(ValueError):
            PseudoMRIRenderer(lesion_margin=1)
        with self.assertRaises(ValueError):
            PseudoMRIRenderer(lesion_placement="wm_interior", lesion_target="burden", lesion_margin=1)


class MarginFactoryTests(unittest.TestCase):
    BASE = ["--res", "16", "--no-cache", "--synthetic-clean-content", "--synthetic-lesion-placement", "wm_interior"]

    def test_cli_validates_the_margin(self):
        parse = encoder_args.parse_args
        self.assertEqual(parse(self.BASE).synthetic_lesion_margin, 0)
        self.assertEqual(parse(self.BASE + ["--synthetic-lesion-margin", "1"]).synthetic_lesion_margin, 1)
        for bad in (
            ["--synthetic-lesion-margin", "-1"],
            ["--synthetic-lesion-placement", "legacy", "--synthetic-lesion-margin", "1"],
            ["--synthetic-lesion-target", "burden", "--synthetic-lesion-margin", "1"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(self.BASE + bad)

    def test_training_and_every_evaluation_factory_rebuild_the_margin(self):
        args = encoder_args.parse_args(self.BASE + ["--synthetic-lesion-margin", "1"])
        settings = json.loads(json.dumps(vars(args)))  # what training saves as settings.json
        trained = trainer_make_dataset()(args, "val", 3)
        scored = make_dataset(settings, 3, "val")
        monitored = dataset(settings, 3, "val")  # the spatial-recovery monitor's cohort
        for ds in (trained, scored, monitored):
            self.assertEqual(ds._inner.renderer.lesion_margin, 1)
        for a, b in zip(trained._inner[0][:2], scored._inner[0][:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Settings saved before the flag existed rebuild without a margin.
        del settings["synthetic_lesion_margin"]
        self.assertEqual(make_dataset(settings, 2, "val")._inner.renderer.lesion_margin, 0)
        # Generic factories copy every synthetic_* setting the dataset accepts; explicit ones name it.
        self.assertIn("synthetic_lesion_margin", inspect.signature(SyntheticBrainDataset.__init__).parameters)
        from eval.protocol.run_dci_synthetic import build_synthetic_test_set

        eval_args = argparse.Namespace(
            synthetic_res=16, synthetic_n_content=9, synthetic_lesion_placement="wm_interior", synthetic_lesion_margin=1
        )
        ds = build_synthetic_test_set(eval_args, num_samples=2, causal=False, cache=False)
        self.assertEqual(ds._inner.renderer.lesion_margin, 1)


if __name__ == "__main__":
    unittest.main()
