"""Whitened alignment: does Sigma^(-1/2) actually redistribute the gradient to the tail?

The claim --bt-sim-whiten rests on is that the MSE alignment term is variance-weighted, so a
factor holding a small share of the embedding's variance commands almost none of the
gradient, and that whitening fixes it where per-channel normalisation does not.  That is a
measurable claim, so it is measured here rather than asserted.

Run: python -m unittest tests.test_bt_whitening -v
"""

from __future__ import annotations

import unittest

import torch

from training.losses import _whiten_batch

B, C, EPS = 128, 12, 1e-3


def _factors(seed=0):
    """A dominant factor and a weak one, mixed across all channels, plus per-view noise."""
    g = torch.Generator().manual_seed(seed)
    strong = torch.randn(B, generator=g)
    weak = torch.randn(B, generator=g) * 0.1  # 10x smaller spread
    w_s = torch.randn(C, generator=g)
    w_w = torch.randn(C, generator=g)
    w_s, w_w = w_s / w_s.norm(), w_w / w_w.norm()
    return strong, weak, w_s, w_w


def _view(strong, weak, w_s, w_w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.outer(strong, w_s) + torch.outer(weak, w_w) + torch.randn(B, C, generator=g) * 0.05


def _raw(a, b):
    return ((a - b) ** 2).mean()


def _std(a, b):
    den = 0.5 * (a.var(0, unbiased=False) + b.var(0, unbiased=False))
    return (((a - b) ** 2) / (2.0 * den + 1e-8)).mean()


def _whitened(a, b, eps=EPS):
    mu = 0.5 * (a.mean(0, keepdim=True) + b.mean(0, keepdim=True))
    return ((_whiten_batch(a, eps, mean=mu) - _whiten_batch(b, eps, mean=mu)) ** 2).mean()


class TestWhitenBatch(unittest.TestCase):
    def test_output_has_identity_covariance(self):
        x = torch.randn(B, C) @ torch.randn(C, C)  # full rank, badly conditioned
        w = _whiten_batch(x, 1e-8)
        cov = (w - w.mean(0)).T @ (w - w.mean(0)) / (B - 1)
        self.assertLess((cov - torch.eye(C)).abs().max().item(), 1e-3)

    def test_pooled_mean_keeps_offset_visible_and_own_mean_hides_it(self):
        """Centring each view separately would make the MSE blind to a per-view offset --
        which is the one thing the sim term exists to catch, since on_diag is already
        blind to it."""
        x = torch.randn(B, C)
        y = x + 3.0  # pure constant offset, identical covariance
        own = ((_whiten_batch(x, EPS) - _whiten_batch(y, EPS)) ** 2).mean()
        mu = 0.5 * (x.mean(0, keepdim=True) + y.mean(0, keepdim=True))
        pooled = ((_whiten_batch(x, EPS, mean=mu) - _whiten_batch(y, EPS, mean=mu)) ** 2).mean()
        self.assertLess(own.item(), 1e-6)
        self.assertGreater(pooled.item(), 1.0)

    def test_rejects_shapes_it_cannot_estimate(self):
        with self.assertRaises(ValueError):
            _whiten_batch(torch.randn(4, 8, 3), EPS)  # patch fold, not (N, d)
        with self.assertRaises(ValueError):
            _whiten_batch(torch.randn(C, C), EPS)  # too few rows for a d x d covariance

    def test_gradients_are_finite(self):
        x = torch.randn(B, C, requires_grad=True)
        _whiten_batch(x, EPS).pow(2).mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class TestGradientShare(unittest.TestCase):
    """The actual claim: decorrelating the WEAK factor between views must move the loss
    materially, relative to doing the same to the strong one.

    Averaged over DRAWS perturbations. A single draw is not a usable statistic -- adding
    independent noise to one view can move it CLOSER to the other by chance, so the
    per-draw delta changes sign and the ratio is meaningless. Measured over 8 seeds, one
    draw gave shares from -5.04 to +1.05; averaged over 32 it gives 0.35 to 0.95.
    """

    DRAWS = 32

    def setUp(self):
        self.strong, self.weak, self.w_s, self.w_w = _factors()
        self.a = _view(self.strong, self.weak, self.w_s, self.w_w, 1)
        self.b = _view(self.strong, self.weak, self.w_s, self.w_w, 2)
        self.g = torch.Generator().manual_seed(9)

    def _mean_delta(self, fn, vec, scale):
        """Mean rise in the loss from DECORRELATING one factor in view b.

        Decorrelate, do not rescale: whitening normalises amplitude away, so a scaling
        perturbation is not a valid probe of what the loss can still see.
        """
        base = fn(self.a, self.b)
        deltas = []
        for _ in range(self.DRAWS):
            noise = torch.randn(B, generator=self.g) * scale * 0.10
            deltas.append((fn(self.a, self.b + torch.outer(noise, vec)) - base).item())
        return sum(deltas) / len(deltas)

    def _share(self, fn):
        weak = self._mean_delta(fn, self.w_w, self.weak.std())
        strong = self._mean_delta(fn, self.w_s, self.strong.std())
        return weak / strong

    def test_raw_mse_almost_ignores_the_weak_factor(self):
        self.assertLess(self._share(_raw), 0.05)

    def test_per_channel_normalisation_does_not_fix_it(self):
        """sim_normalize equalises CHANNELS, and when every channel is a mixture of the same
        few factors that leaves the mixture untouched. Measured across seeds: 0.006-0.089,
        i.e. no reliable improvement on raw. It is doing its OTHER documented job (keeping
        the term scale-free against the variance hinge), not this one."""
        self.assertLess(self._share(_std), 0.10)

    def test_whitening_raises_the_weak_factor_share_by_an_order_of_magnitude(self):
        share_raw = self._share(_raw)
        share_white = self._share(_whitened)
        self.assertGreater(share_white, 5 * share_raw)
        self.assertGreater(share_white, 0.10)

    def test_large_eps_decays_back_toward_plain_mse(self):
        """eps is a real hyperparameter, not a numerical formality: enough shrinkage and the
        term stops equalising directions at all."""
        mid = self._share(lambda a, b: _whitened(a, b, eps=1e-2))
        big = self._share(lambda a, b: _whitened(a, b, eps=10.0))
        self.assertGreater(mid, big)


if __name__ == "__main__":
    unittest.main()
