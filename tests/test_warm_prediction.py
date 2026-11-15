"""Stage 33's end-to-end cross-view prediction term in warm.py: running whitening, view pairing, the momentum copy."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import frozen_prediction as P
import warm


class WarmPredictionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_running_whitening_starts_as_the_frozen_fit(self):
        sample = torch.randn(2, 70, 8) @ torch.randn(8, 8)
        sample[..., -1] = 3.0
        running = warm.TargetWhitening(sample, 0.1)
        fitted = P.fit_whitening(sample, 0.1)
        torch.testing.assert_close(running(sample), P.whiten(sample, fitted), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running(sample)[..., -1], torch.zeros(2, 70))

    def test_spread_counts_the_directions_the_targets_use(self):
        for value in warm.TargetWhitening(torch.randn(2, 4000, 6), 0.1).spread():
            self.assertGreater(value, 5.5)
        for value in warm.TargetWhitening(torch.randn(2, 500, 1).repeat(1, 1, 6), 0.1).spread():
            self.assertAlmostEqual(value, 1.0, places=2)

    def test_update_moves_the_statistics_by_the_momentum(self):
        running = warm.TargetWhitening(torch.randn(2, 64, 5), 0.1, momentum=0.25)
        before = running.mean.clone()
        batch = torch.randn(2, 32, 5) + 4
        running.update(batch)
        torch.testing.assert_close(running.mean, before + 0.25 * (batch.double().mean(1, keepdim=True) - before))
        whitened = running(batch)
        self.assertTrue(torch.isfinite(whitened).all())
        self.assertEqual(whitened.dtype, batch.dtype)

    def test_prediction_reads_the_opposite_view_and_never_trains_the_target(self):
        decoders = nn.ModuleList(nn.Linear(3, 3, bias=False) for _ in range(2))
        with torch.no_grad():
            for decoder in decoders:
                decoder.weight.copy_(torch.eye(3))
        pair = torch.randn(2, 9, 3, requires_grad=True)
        standardized = (pair - pair.mean(1, keepdim=True)) / (pair.std(1, keepdim=True) + 1e-6)
        self.assertLess(float(warm.cross_view_prediction(pair, standardized.detach().flip(0), decoders)), 1e-10)
        self.assertGreater(float(warm.cross_view_prediction(pair, standardized.detach(), decoders)), 0.1)
        targets = torch.randn(2, 9, 3, requires_grad=True)
        warm.cross_view_prediction(pair, targets, decoders).backward()
        self.assertIsNone(targets.grad)
        self.assertGreater(float(pair.grad.abs().sum()), 0.0)
        self.assertTrue(all(decoder.weight.grad is not None for decoder in decoders))

    def test_momentum_copy_moves_toward_the_online_backbone(self):
        online, kept = nn.Linear(4, 4), nn.Linear(4, 4)
        before = kept.weight.detach().clone()
        warm.momentum_update(online, kept, 0.996)
        torch.testing.assert_close(kept.weight.detach(), 0.996 * before + 0.004 * online.weight.detach())

    def test_options_need_a_weight_tailcode_and_infonce(self):
        base = ["warm.py", "bank", "/nonexistent/stage33_out", "cold", "2", "1", "--patch-weight", "0"]
        cases = [
            ["--pool", "tail", "--predict-momentum", "0.9"],
            ["--pool", "gap", "--predict-weight", "1"],
            ["--pool", "tail", "--predict-weight", "1", "--supervise", "volume"],
            ["--pool", "tail", "--predict-weight", "1", "--contrastive-loss-type", "barlow_twins"],
            ["--pool", "tail", "--predict-weight", "-1"],
            ["--pool", "tail", "--predict-weight", "1", "--predict-momentum", "1"],
            ["--pool", "tail", "--predict-weight", "1", "--predict-ridge", "0"],
        ]
        for extra in cases:
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra):
                with self.assertRaisesRegex(ValueError, "--predict"):
                    warm.main()
        self.assertFalse(Path("/nonexistent/stage33_out").exists())


if __name__ == "__main__":
    unittest.main()
