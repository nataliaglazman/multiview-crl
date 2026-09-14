"""Stats pooling for the GAP companion: same statistic the report scores, and it keeps
localised factors that the spatial mean erases.

Run: python -m unittest tests.test_stats_pool -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from eval.dci import _pool_and_split_view
from eval.identifiability_metrics import cv_probe_r2
from training.losses import stats_pool

B, C, P = 256, 6, 64


class TestMatchesTheReportsPooling(unittest.TestCase):
    """The loss must train on the statistic eval/dci.py scores, or the report measures
    something the objective never optimised. Pinned by calling the eval code, not by
    copying its expression -- a copy would not notice if the eval side changed."""

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.vol = torch.randn(B, C, 4, 4, 4, generator=g)  # (B, C, D, H, W)
        self.flat = self.vol.reshape(1, B, C, -1)  # (n_views=1, B, C, P)
        self.content = [1, 2]

    def test_pooled_content_block_equals_eval_dci(self):
        mine, idx, _ = stats_pool(self.flat, [self.content], None)
        got = mine[0][:, idx[0]].numpy()
        want, _ = _pool_and_split_view(self.vol, self.content, None, False, True)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_layout_is_stat_major(self):
        pooled, _, _ = stats_pool(self.flat, None, None)
        z = self.flat[0]
        for k, want in enumerate((z.mean(2), z.std(2), z.amax(2), z.amin(2))):
            torch.testing.assert_close(pooled[0][:, k * C : (k + 1) * C], want)

    def test_mask_expansion_selects_the_same_columns_as_the_indices(self):
        mask = torch.zeros(1, C)
        mask[0, self.content] = 1.0
        pooled, idx, mask_out = stats_pool(self.flat, [self.content], mask)
        by_mask = pooled[0][:, mask_out.reshape(-1).bool()]
        by_index = pooled[0][:, idx[0]]
        self.assertEqual(by_mask.shape[1], 4 * len(self.content))
        torch.testing.assert_close(
            by_mask.sort(dim=1).values,
            by_index.sort(dim=1).values,
        )

    def test_rejects_shapes_it_cannot_pool(self):
        with self.assertRaises(ValueError):
            stats_pool(torch.randn(B, C, P))  # missing the view axis
        with self.assertRaises(ValueError):
            stats_pool(torch.randn(1, B, C, 1))  # unbiased std undefined at P=1


class TestKeepsLocalisedFactors(unittest.TestCase):
    """The reason for the flag. A factor confined to one position contributes ~1/P of a
    channel's mean and is gone from GAP, while clearly moving that channel's spread and
    extremes. A global factor must survive both, so this is not a trade."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.local = rng.standard_normal(B)
        self.glob = rng.standard_normal(B)
        z = rng.standard_normal((B, C, P)) * 0.3 + self.glob[:, None, None]
        z[:, 2, 17] += self.local  # one channel, one position
        t = torch.from_numpy(z).float().unsqueeze(0)
        self.gap = t[0].mean(-1).numpy()
        self.stats = stats_pool(t, None, None)[0][0].numpy()

    def _r2(self, X, y):
        return cv_probe_r2(X, y, seeds=(0,))["mean"]

    def test_gap_loses_the_localised_factor(self):
        self.assertLess(self._r2(self.gap, self.local), 0.15)

    def test_stats_keeps_it(self):
        self.assertGreater(self._r2(self.stats, self.local), 0.50)

    def test_neither_loses_the_global_factor(self):
        self.assertGreater(self._r2(self.gap, self.glob), 0.95)
        self.assertGreater(self._r2(self.stats, self.glob), 0.95)


if __name__ == "__main__":
    unittest.main()
