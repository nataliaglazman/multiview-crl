"""Loss math and the boundary between aligned content and unaligned style."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.dino_partition import make_partition, split_embeddings
from training import finetune_dino as train
from utils.config import parse_dino_finetune_args


class PartitionTests(unittest.TestCase):
    def options(self, **changes):
        cli = SimpleNamespace(
            style_fraction=0.25,
            backbone="3dino",
            token_pool="cls",
            grid_size=2,
            axes=["axial", "coronal"],
            slices=2,
            slice_agg="concat",
            objective="infonce",
            temperature=0.1,
            barlow_lambda=0.0051,
            barlow_eps=1e-5,
            vicreg_sim_coeff=25.0,
            vicreg_std_coeff=25.0,
            vicreg_cov_coeff=1.0,
        )
        vars(cli).update(changes)
        return cli

    def test_default_cls_is_768_content_256_style_and_zero_restores_full_embedding(self):
        x = torch.arange(2048).reshape(2, 1024)
        partition = make_partition(1024, 1024, self.options())
        c, s = split_embeddings(x, partition)
        torch.testing.assert_close(c, x[:, :768])
        torch.testing.assert_close(s, x[:, 768:])
        c, s = split_embeddings(x, make_partition(1024, 1024, self.options(style_fraction=0)))
        torch.testing.assert_close(c, x)
        self.assertEqual(s.shape, (2, 0))

    def test_each_channel_keeps_its_role_across_grid_cls_mean_and_slices(self):
        for backend in ("3dino", "dinov3"):
            for pooling in ("cls", "mean", "cls_mean", "grid"):
                for aggregation in ("mean", "concat"):
                    spatial = 2 ** (3 if backend == "3dino" else 2) if pooling == "grid" else 1
                    repeats = (2 if pooling == "cls_mean" else 1) * (
                        4 if backend == "dinov3" and aggregation == "concat" else 1
                    )
                    shape = (2, repeats, 8, spatial)
                    x = np.broadcast_to(np.arange(8)[None, None, :, None], shape).copy().reshape(2, -1)
                    cli = self.options(backbone=backend, token_pool=pooling, slice_agg=aggregation)
                    spec = make_partition(x.shape[1], 8, cli)
                    content, style = split_embeddings(x, spec)
                    self.assertEqual(set(np.unique(content)), set(range(6)))
                    self.assertEqual(set(np.unique(style)), {6, 7})
                    self.assertEqual(style.size, x.size // 4)

    def test_both_losses_are_independent_of_style_values_and_have_no_style_output_gradient(self):
        for objective in ("infonce", "barlow", "vicreg"):
            torch.manual_seed(5)
            cli = self.options(objective=objective)
            first, second = [torch.randn(8, 16, requires_grad=True) for _ in range(2)]
            spec = make_partition(16, 16, cli)
            projector = torch.nn.Linear(12, 5)
            a, b = split_embeddings(first, spec), split_embeddings(second, spec)
            loss, _ = train.alignment_loss(a[0], b[0], projector, cli)
            changed = second.detach().clone()
            changed[:, 12:] += 1000
            same, _ = train.alignment_loss(a[0], split_embeddings(changed, spec)[0], projector, cli)
            torch.testing.assert_close(loss, same)
            loss.backward()
            for x in (first, second):
                self.assertGreater(x.grad[:, :12].norm().item(), 0)
                self.assertEqual(x.grad[:, 12:].abs().sum().item(), 0)
            self.assertGreater(projector.weight.grad.norm().item(), 0)

    def test_invalid_cli_and_partition_dimensions_fail(self):
        for option, value in (
            ("--style-fraction", "1"),
            ("--style-fraction", "nan"),
            ("--barlow-lambda", "-1"),
            ("--barlow-eps", "0"),
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_dino_finetune_args(["--output-dir", "unused", option, value])
        with self.assertRaises(ValueError):
            make_partition(7, 8, self.options())
        spec = make_partition(8, 8, self.options())
        with self.assertRaises(ValueError):
            split_embeddings(np.ones((3, 9)), spec)


class BarlowTests(unittest.TestCase):
    def test_matches_explicit_numpy_correlation_and_is_symmetric(self):
        rng = np.random.RandomState(31)
        x, y = rng.randn(12, 5), rng.randn(12, 5)
        normalize = lambda a: (a - a.mean(0)) / np.sqrt(a.var(0) + 1e-5)
        corr = normalize(x).T @ normalize(y) / len(x)
        expected = ((np.diag(corr) - 1) ** 2).sum() + 0.0051 * (corr[~np.eye(5, dtype=bool)] ** 2).sum()
        loss, metrics = train.barlow_twins(torch.tensor(x), torch.tensor(y))
        reverse, _ = train.barlow_twins(torch.tensor(y), torch.tensor(x))
        self.assertAlmostEqual(loss.item(), expected, places=5)
        torch.testing.assert_close(loss, reverse)
        self.assertIn("barlow_off_diag", metrics)

    def test_whitened_pairs_win_over_shuffled_pairs_and_constants_are_finite(self):
        x = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        aligned, _ = train.barlow_twins(x, x)
        shuffled, _ = train.barlow_twins(x, x.roll(1, 0))
        self.assertLess(aligned.item(), 1e-8)
        self.assertGreater(shuffled.item(), 1)
        a, b = [torch.ones(4, 3, requires_grad=True) for _ in range(2)]
        constant, _ = train.barlow_twins(a, b)
        self.assertEqual(constant.item(), 3)
        constant.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(ValueError):
            train.barlow_twins(a[:1], b[:1])


if __name__ == "__main__":
    unittest.main()
