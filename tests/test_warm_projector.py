"""warm.py --projector: a head shared by both views between the content units and the global InfoNCE (stage 71)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import warm

import training.main_conv_synthetic as T

BASE = ["warm.py", "/nonexistent/bank", "OUT", "cold", "2", "1", "--pool", "tail", "--patch-weight", "0"]


class ProjectorOptionTests(unittest.TestCase):
    def test_options_reject_what_the_head_cannot_use(self):
        out = "/nonexistent/stage71_out"
        cases = [
            ["--projector", "0"],
            ["--projector", "-1"],
            ["--projector", "64", "--pool", "gap"],
            ["--projector", "64", "--patch-weight", "1"],
            ["--projector", "64", "--contrastive-loss-type", "barlow_twins"],
            ["--projector", "64", "--queue", "64"],
            ["--projector", "64", "--hard-negatives", "1"],
            ["--projector", "64", "--supervise", "volume"],
            ["--projector", "64", "--tau", "1"],
            ["--projector", "64", "--harmonic-weight", "1"],
        ]
        for extra in cases:
            argv = [out if x == "OUT" else x for x in BASE] + extra
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "--projector"):
                    warm.main()
        self.assertFalse(Path(out).exists())

    def test_the_head_is_announced(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [str(Path(tmp) / "run") if x == "OUT" else x for x in BASE] + ["--projector", "64", "--device", "cpu"]
            printed = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(printed):
                with self.assertRaises(Exception):  # the bank does not exist, after the head is announced
                    warm.main()
        self.assertIn("projector head between the content units and the global InfoNCE, hidden 64", printed.getvalue())

    def test_without_the_option_nothing_is_said(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [str(Path(tmp) / "run") if x == "OUT" else x for x in BASE] + ["--device", "cpu"]
            printed = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(printed):
                with self.assertRaises(Exception):
                    warm.main()
        self.assertNotIn("projector", printed.getvalue())


class ProjectorHeadTests(unittest.TestCase):
    def pair(self, rng, n=32):
        scale = torch.tensor([3.0, 0.5, 10.0, 1.0, 2.0, 0.1, 5.0, 1.5])
        return torch.as_tensor(rng.normal(size=(2, n, 8)), dtype=torch.float32) * scale + 4.0

    def test_at_the_start_the_loss_sees_the_recipes_code(self):
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        head = warm.Projector(8, 16).train()
        pair = self.pair(rng)
        out = head(pair)
        # The InfoNCE standardizes per view over the batch, which no per-unit positive affine map changes.
        torch.testing.assert_close(
            T.standardize_contrastive_code(out), T.standardize_contrastive_code(pair), atol=1e-4, rtol=1e-4
        )

    def test_every_part_of_the_head_learns(self):
        torch.manual_seed(0)
        head = warm.Projector(8, 16).train()
        pair = self.pair(np.random.default_rng(1)).requires_grad_(True)
        head(pair).square().sum().backward()
        self.assertGreater(float(pair.grad.abs().sum()), 0)
        self.assertGreater(float(head.linear.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.mlp[-1].weight.grad.abs().sum()), 0)  # the zero output layer still learns

    def test_the_gauge_tracks_the_population_and_eval_leaves_it(self):
        rng = np.random.default_rng(2)
        population = self.pair(rng, n=2000)
        head = warm.Projector(8, 16, momentum=0.01).train()
        for _ in range(3000):
            head.gauged(population[:, rng.choice(2000, 32, replace=False)])
        sd = (head.square - head.mean.square()).sqrt()
        torch.testing.assert_close(head.mean, population.mean(1), atol=0.15 * float(population.std(1).max()), rtol=0)
        torch.testing.assert_close(sd, population.std(1), atol=0, rtol=0.1)
        head.eval()
        before = head.mean.clone(), head.square.clone()
        head(population[:, :32] * 100)
        torch.testing.assert_close(head.mean, before[0])
        torch.testing.assert_close(head.square, before[1])

    def test_linearity_is_one_for_a_linear_head_and_lower_for_a_bent_one(self):
        rng = np.random.default_rng(3)
        torch.manual_seed(0)
        head = warm.Projector(8, 16).train()
        codes = [rng.normal(size=(500, 8)) for _ in range(2)]
        head(torch.as_tensor(np.stack(codes), dtype=torch.float32))  # sets the gauge
        self.assertTrue(all(r > 0.999 for r in warm.projector_linearity(head, codes, "cpu")))
        with torch.no_grad():
            head.mlp[0].weight.mul_(3.0)
            head.mlp[-1].weight.normal_(0, 3.0)
        self.assertTrue(all(r < 0.95 for r in warm.projector_linearity(head, codes, "cpu")))
        self.assertTrue(head.training)  # restored



class WideProjectorTests(unittest.TestCase):
    """--projector-out (stage 83): the head's output wider than the code, the InfoNCE on every output unit."""

    def test_options_reject_a_narrow_or_lone_output(self):
        out = "/nonexistent/stage83_out"
        cases = [
            ["--projector-out", "16", "--content-units", "8"],  # no head
            ["--projector", "64", "--projector-out", "16"],  # no --content-units
            ["--projector", "64", "--projector-out", "6", "--content-units", "8"],  # narrower than the code
        ]
        for extra in cases:
            argv = [out if x == "OUT" else x for x in BASE] + extra
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "--projector-out"):
                    warm.main()
        self.assertFalse(Path(out).exists())

    def test_the_wide_head_starts_with_the_code_and_varies_everywhere(self):
        rng = np.random.default_rng(4)
        torch.manual_seed(0)
        head = warm.Projector(8, 16, out=16).train()
        pair = ProjectorHeadTests.pair(None, rng)
        out = head(pair)
        self.assertEqual(tuple(out.shape), (2, 32, 16))
        torch.testing.assert_close(
            T.standardize_contrastive_code(out[..., :8]), T.standardize_contrastive_code(pair), atol=1e-4, rtol=1e-4
        )
        self.assertTrue(bool((out[..., 8:].std(1) > 1e-3).all()))  # no constant unit for the standardization

    def test_at_the_codes_width_the_head_is_stage_71s(self):
        torch.manual_seed(0)
        narrow = warm.Projector(8, 16)
        torch.manual_seed(0)
        same = warm.Projector(8, 16, out=8)
        for a, b in zip(narrow.state_dict().values(), same.state_dict().values()):
            torch.testing.assert_close(a, b)

    def test_the_wide_loss_reads_every_output_unit(self):
        rng = np.random.default_rng(5)
        torch.manual_seed(0)
        head = warm.Projector(8, 16, out=16).train()
        pair = ProjectorHeadTests.pair(None, rng).requires_grad_(True)
        sim = torch.nn.CosineSimilarity(dim=-1)
        loss = warm.hard_negative_infonce(torch.cat(list(head(pair))), 16, 0.1, 0.0, sim)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.linear.weight.grad[8:].abs().sum()), 0)  # the extra outputs carry gradient


if __name__ == "__main__":
    unittest.main()
