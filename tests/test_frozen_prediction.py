"""Frozen-feature prediction must train through the shared code without changing its teacher."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import frozen_prediction as P
import warm


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = 4
        self.gauge = nn.BatchNorm1d(6)
        self.head = nn.Sequential(nn.Linear(6, 9), nn.LeakyReLU(), nn.Linear(9, 6))


class FrozenPredictionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_whitening_is_finite_and_heldout_cannot_refit_it(self):
        train = torch.randn(2, 70, 8)
        train[..., -1] = 7
        stats = P.fit_whitening(train, 0.1)
        frozen = copy.deepcopy(stats)
        transformed = P.whiten(train, stats)
        self.assertTrue(torch.isfinite(transformed).all())
        torch.testing.assert_close(transformed.mean(1), torch.zeros(2, 8), atol=1e-6, rtol=0)
        torch.testing.assert_close(transformed[..., -1], torch.zeros(2, 70))
        P.whiten(torch.full((2, 20, 8), 1e5), stats)
        for name in stats:
            self.assertTrue(torch.equal(stats[name], frozen[name]))
        self.assertFalse(torch.equal(stats["mean"], P.fit_whitening(train + 10, 0.1)["mean"]))
        with self.assertRaises(ValueError):
            P.fit_whitening(train, 0)

    def test_prediction_uses_opposite_view_and_only_content(self):
        decoders = nn.ModuleList(nn.Linear(3, 3, bias=False) for _ in range(2))
        with torch.no_grad():
            for decoder in decoders:
                decoder.weight.copy_(torch.eye(3))
        code = torch.randn(2, 7, 5, requires_grad=True)
        stats = {"mean": torch.zeros(2, 1, 3), "scale": torch.ones(2, 1, 3)}
        perfect_targets = code.detach().flip(0)[..., :3]
        loss, _ = P.prediction_loss(code, perfect_targets, decoders, stats)
        self.assertEqual(float(loss), 0.0)
        targets = (perfect_targets + 1).requires_grad_(True)
        loss, _ = P.prediction_loss(code, targets, decoders, stats)
        loss.backward()
        self.assertIsNone(targets.grad)
        self.assertGreater(float(code.grad[..., :3].abs().sum()), 0)
        self.assertEqual(float(code.grad[..., 3:].abs().sum()), 0)
        self.assertGreater(float(decoders[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(decoders[1].weight.grad.abs().sum()), 0)

    def fixture(self):
        model = nn.Sequential(nn.Linear(6, 6), nn.BatchNorm1d(6))
        pool = Pool()
        P.freeze_features(model, pool)
        pooled = torch.randn(2, 16, 6)
        targets = torch.randn(2, 16, 8)
        stats = P.fit_code_scale(P.codes(pool, pooled))
        decoders = nn.ModuleList(nn.Linear(pool.units, 8) for _ in range(2))
        return model, pool, pooled, targets, stats, decoders

    def test_training_changes_only_head_and_decoders(self):
        model, pool, pooled, targets, stats, decoders = self.fixture()
        before = (P.state_digest(model), P.state_digest(pool, exclude_head=True))
        head = P.state_digest(pool.head)
        predictor = P.state_digest(decoders)
        optimizer = torch.optim.AdamW(list(pool.head.parameters()) + list(decoders.parameters()), lr=1e-3)
        P.train_step(pool, decoders, optimizer, pooled, targets, stats, weight=1)
        self.assertEqual(before, (P.state_digest(model), P.state_digest(pool, exclude_head=True)))
        self.assertNotEqual(head, P.state_digest(pool.head))
        self.assertNotEqual(predictor, P.state_digest(decoders))
        self.assertFalse(model.training)
        self.assertFalse(pool.training)
        self.assertFalse(pool.gauge.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_weight_zero_exactly_matches_plain_infonce_update(self):
        _, pool, pooled, targets, stats, decoders = self.fixture()
        control = copy.deepcopy(pool)
        optimizer = torch.optim.AdamW(pool.head.parameters(), lr=1e-3)
        baseline_optimizer = torch.optim.AdamW(control.head.parameters(), lr=1e-3)
        P.train_step(pool, decoders, optimizer, pooled, targets, stats, weight=0)
        code = torch.cat([control.head(part) for part in pooled])
        loss = warm.hard_negative_infonce(code, control.units, 0.1, 0, nn.CosineSimilarity(dim=-1))
        baseline_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(control.head.parameters(), 2.0)
        baseline_optimizer.step()
        self.assertEqual(P.state_digest(pool), P.state_digest(control))
        self.assertTrue(all(p.grad is None for p in decoders.parameters()))

    def test_batch_draws_do_not_depend_on_decoder_initialization(self):
        first, second = np.random.default_rng(42), np.random.default_rng(42)
        expected = [P.draw_batch(first, 100, 32, 8) for _ in range(4)]
        torch.manual_seed(921)
        nn.Linear(24, 64)
        for reference in expected:
            actual = P.draw_batch(second, 100, 32, 8)
            for left, right in zip(reference, actual):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(len(np.unique(actual[0])), 32)

    def test_stale_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.pt"
            expected = {"checkpoint_sha256": "one", "draws": 8}
            P.atomic_save({"metadata": expected, "example": torch.zeros(1)}, path)
            P.read_cache(path, expected)
            for key, replacement in (("checkpoint_sha256", "two"), ("draws", 4)):
                with self.assertRaisesRegex(ValueError, "changed"):
                    P.read_cache(path, {**expected, key: replacement})


if __name__ == "__main__":
    unittest.main()
