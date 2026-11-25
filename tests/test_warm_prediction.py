"""Stage 33's end-to-end cross-view prediction term in warm.py: running whitening, view pairing, the momentum copy."""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import frozen_prediction as P
import warm


class PredictionBank:
    def __init__(self, subjects=40):
        self.vox = np.zeros(subjects)
        self.images_by_view = torch.randn(2, subjects, 1, 4, 4, 4)

    def __len__(self):  # warm.Bank counts its subjects this way since --real
        return len(self.vox)

    def images(self, indices, device):
        return torch.cat([view[indices] for view in self.images_by_view]).to(device)


class PredictionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.views = nn.ModuleList(nn.Conv3d(1, 3, 1) for _ in range(2))
        self.inputs = []

    def _encode(self, images, views, view_idx):
        self.inputs.append(images.detach().clone())
        return torch.cat([layer(part).tanh() for layer, part in zip(self.views, images.chunk(2))])


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

    def test_default_maps_are_the_linear_maps(self):
        maps = warm.prediction_decoders(10, 64)
        self.assertEqual([type(m) for m in maps], [nn.Linear, nn.Linear])
        self.assertEqual(sorted(maps.state_dict()), ["0.bias", "0.weight", "1.bias", "1.weight"])

    def test_hidden_map_starts_as_its_linear_map(self):
        maps = warm.prediction_decoders(3, 5, hidden=8)
        code = torch.randn(7, 3)
        for m in maps:
            self.assertIsInstance(m, warm.HiddenDecoder)
            torch.testing.assert_close(m(code), m.linear(code))

    def test_hidden_map_lets_one_unit_predict_two_directions(self):
        code = torch.linspace(-2, 2, 200)[:, None]
        target = torch.cat([code, code.square() - code.square().mean()], 1)
        errors = {}
        for hidden in (0, 32):
            decoder = warm.prediction_decoders(1, 2, hidden)[0]
            optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
            for _ in range(1500):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(decoder(code), target)
                loss.backward()
                optimizer.step()
            errors[hidden] = float(loss)
        self.assertGreater(errors[0], 0.3)  # the square's variance is ~1.4: a line misses it
        self.assertLess(errors[32], 0.05)

    def test_momentum_copy_moves_toward_the_online_backbone(self):
        online, kept = nn.Linear(4, 4), nn.Linear(4, 4)
        before = kept.weight.detach().clone()
        warm.momentum_update(online, kept, 0.996)
        torch.testing.assert_close(kept.weight.detach(), 0.996 * before + 0.004 * online.weight.detach())

    def prediction_fixture(self):
        bank = PredictionBank()
        model = PredictionEncoder()
        teacher = copy.deepcopy(model).eval()
        pool = warm.TailCode(np.ones(64, dtype=bool), channels=3, latent=14, units=10)
        pool(model._encode(bank.images(np.arange(8), "cpu"), 2, None))
        return bank, model, teacher, pool

    def test_uniform_prediction_pairs_clean_teacher_with_jittered_online_views(self):
        bank, model, teacher, pool = self.prediction_fixture()
        indices, clean, moved = warm.draw_prediction_batch(
            bank, 8, "cpu", np.random.default_rng(45), np.random.default_rng(46), 1, 3
        )
        self.assertEqual(len(np.unique(indices)), 8)
        for view in range(2):
            torch.testing.assert_close(clean.chunk(2)[view], bank.images_by_view[view, indices])
        self.assertFalse(torch.equal(clean, moved))
        pair = warm.prediction_code(model, pool, moved, 10)
        gap = warm.target_gap(teacher, clean)
        torch.testing.assert_close(model.inputs[-1], moved)
        torch.testing.assert_close(teacher.inputs[-1], clean)
        self.assertEqual(tuple(pair.shape), (2, 8, 10))
        self.assertFalse(gap.requires_grad)
        decoders = warm.prediction_decoders(10, 3)
        warm.cross_view_prediction(pair, gap, decoders).backward()
        for layer in model.views:
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
        self.assertGreater(float(pool.head[-1].weight.grad[:10].abs().sum()), 0)
        self.assertEqual(float(pool.head[-1].weight.grad[10:].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is not None for p in decoders.parameters()))

    def test_uniform_extra_pass_keeps_running_buffers_and_batch_normalization(self):
        bank, model, _, pool = self.prediction_fixture()
        buffers = {name: value.clone() for name, value in pool.named_buffers()}
        inputs, outputs = [], []

        def capture(_, args, output):
            inputs.append(args[0])
            outputs.append(output)

        handle = pool.head.register_forward_hook(capture)
        try:
            pair = warm.prediction_code(model, pool, bank.images(np.arange(8, 16), "cpu") + 3, 10)
        finally:
            handle.remove()
        for name, value in pool.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]), name)
        for view in range(2):
            torch.testing.assert_close(inputs[view].mean(0), torch.zeros_like(inputs[view][0]), atol=2e-5, rtol=0)
            torch.testing.assert_close(pair[view], outputs[view][:, :10])
        self.assertTrue(pool.training)
        self.assertTrue(all(gauge.training for gauge in pool.gauges))

    def test_uniform_subject_draws_do_not_depend_on_augmentation_draws(self):
        bank = PredictionBank()
        first, second = np.random.default_rng(45), np.random.default_rng(45)
        aug_first, aug_second = np.random.default_rng(46), np.random.default_rng(99)
        for _ in range(4):
            expected = warm.draw_prediction_batch(bank, 8, "cpu", first, aug_first, 0, 0)
            actual = warm.draw_prediction_batch(bank, 8, "cpu", second, aug_second, 1, 3)
            np.testing.assert_array_equal(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            self.assertEqual(len(np.unique(actual[0])), 8)
        repeated = [np.random.default_rng(45), np.random.default_rng(46)]
        again = [np.random.default_rng(45), np.random.default_rng(46)]
        for _ in range(3):
            left = warm.draw_prediction_batch(bank, 8, "cpu", *repeated, 1, 3)
            right = warm.draw_prediction_batch(bank, 8, "cpu", *again, 1, 3)
            np.testing.assert_array_equal(left[0], right[0])
            torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)

    def test_prediction_jitter_is_independent_between_views(self):
        bank = PredictionBank()
        bank.images_by_view[1].copy_(bank.images_by_view[0])
        _, clean, moved = warm.draw_prediction_batch(
            bank, 8, "cpu", np.random.default_rng(45), np.random.default_rng(46), 1, 3
        )
        torch.testing.assert_close(*clean.chunk(2))
        self.assertFalse(torch.equal(*moved.chunk(2)))

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
            ["--pool", "tail", "--predict-hidden", "64"],
            ["--pool", "tail", "--predict-weight", "1", "--predict-hidden", "-1"],
        ]
        for extra in cases:
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra):
                with self.assertRaisesRegex(ValueError, "--predict"):
                    warm.main()
        self.assertFalse(Path("/nonexistent/stage33_out").exists())

    def test_uniform_prediction_rejects_unsupported_options_before_writing(self):
        base = [
            "warm.py",
            "bank",
            "/nonexistent/stage45_out",
            "cold",
            "2",
            "1",
            "--pool",
            "tail",
            "--patch-weight",
            "0",
            "--predict-weight",
            "1",
            "--predict-batches",
            "uniform",
        ]
        cases = [
            ["--predict-weight", "0"],
            ["--gauge", "running"],
            ["--patch-weight", "1"],
            ["--uniform", "--remainder-units", "2"],
            ["--half-views", "x", "--half-units", "4"],
            ["--group-dropout", "0.1"],
            ["--jitter", "1", "--shared-jitter"],
        ]
        for extra in cases:
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra):
                with self.assertRaisesRegex(ValueError, "--predict"):
                    warm.main()
        self.assertFalse(Path("/nonexistent/stage45_out").exists())


if __name__ == "__main__":
    unittest.main()
