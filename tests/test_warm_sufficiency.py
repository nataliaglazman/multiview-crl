"""Stage 77's sufficiency term in warm.py: what the two views' targets still share once each code has explained its own."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import warm


def shared_targets(n, rng, private=4):
    """Both views' targets share one component and three others; each view adds private noise."""
    shared, other = rng.standard_normal((n, 1)), rng.standard_normal((n, 3))
    views = [np.column_stack([shared, other, rng.standard_normal((n, private))]) for _ in range(2)]
    return torch.as_tensor(np.stack(views), dtype=torch.float32), shared, other


def fitted_value(code, targets, rng, steps=1500, batch=64):
    """The term's running value on a fixed code, after fitting its explainers batch by batch."""
    z = torch.as_tensor(code, dtype=torch.float32)
    term = warm.Sufficiency(z.shape[1], targets.shape[-1], hidden=64)
    opt = torch.optim.Adam(term.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.as_tensor(rng.choice(len(z), batch, replace=False))
        pair, t = torch.stack([z[idx], z[idx]]), targets[:, idx]
        term(pair, t)
        opt.zero_grad()
        term.fit(pair, t).backward()
        opt.step()
    return term.value()[0]


class WarmSufficiencyTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)

    def test_options_need_tailcode_infonce_and_no_conflicting_term(self):
        base = ["warm.py", "bank", "/nonexistent/stage77_out", "cold", "2", "1", "--patch-weight", "0"]
        cases = [
            ["--pool", "gap", "--sufficiency-weight", "1"],
            ["--pool", "tail", "--sufficiency-weight", "-1"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--sufficiency-hidden", "0"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--projector", "64"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--tau", "1"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--harmonic-weight", "1"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--predict-weight", "1", "--predict-batches", "uniform"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--contrastive-loss-type", "barlow_twins"],
            ["--pool", "tail", "--sufficiency-weight", "1", "--patch-weight", "1"],
        ]
        for extra in cases:
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra):
                with self.assertRaisesRegex(ValueError, "--sufficiency"):
                    warm.main()
        self.assertFalse(Path("/nonexistent/stage77_out").exists())

    def test_term_reads_a_shared_part_the_code_lacks(self):
        targets, shared, other = shared_targets(4000, self.rng)
        lacks = fitted_value(other, targets, self.rng)
        holds = fitted_value(np.column_stack([shared, other]), targets, self.rng)
        self.assertGreater(lacks, 0.6)
        self.assertLess(holds, 0.15)

    def test_a_curve_does_not_stand_in_for_the_missing_part(self):
        targets, shared, other = shared_targets(4000, self.rng)
        curved = np.column_stack([other, other[:, :1] ** 2 - 1, np.tanh(other[:, 1:2] * other[:, 2:3])])
        self.assertGreater(fitted_value(curved, targets, self.rng), 0.6)

    def test_gradients_reach_the_code_from_the_term_and_the_explainers_from_their_fit(self):
        targets, _, other = shared_targets(64, self.rng)
        term = warm.Sufficiency(3, targets.shape[-1], hidden=16)
        pair = torch.as_tensor(np.stack([other, other]), dtype=torch.float32).requires_grad_(True)
        term(pair, targets)  # fills the running matrix, so the next call has a direction
        term(pair, targets).backward()
        self.assertGreater(float(pair.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.grad is None for p in term.parameters()))
        pair.grad = None
        term.fit(pair, targets).backward()
        self.assertIsNone(pair.grad)
        self.assertTrue(all(p.grad is not None for p in term.parameters()))

    def test_values_stay_canonical_when_the_first_batches_are_rank_deficient(self):
        term = warm.Sufficiency(8, 64, hidden=16)  # batches of 32 subjects for 64 target channels, as in training
        for _ in range(5):
            term(torch.randn(2, 32, 8), torch.randn(2, 32, 64))
        self.assertLessEqual(max(term.value()), 1.0)
        self.assertLess(float(term(torch.randn(2, 32, 8), torch.randn(2, 32, 64))), 5.0)

    def test_running_moments_whiten_correlated_batches(self):
        moments = warm.RunningMoments(momentum=1.0, ridge=0.0)
        x = torch.as_tensor(self.rng.standard_normal((5000, 3)) @ np.array([[2.0, 0, 0], [1, 1, 0], [0, 0, 0.5]]))
        moments.update(x)
        white = moments(x)
        torch.testing.assert_close(torch.cov(white.T, correction=0), torch.eye(3, dtype=white.dtype), atol=1e-6, rtol=0)
        self.assertEqual(warm.Sufficiency(3, 4).value(), [])


if __name__ == "__main__":
    unittest.main()
