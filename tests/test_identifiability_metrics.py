"""Shared-probe tests: the equalities that let two scripts' numbers be compared."""

import unittest
import warnings

import numpy as np

from eval.identifiability_metrics import BATCHABLE_PROBES, cv_probe_r2, cv_probe_r2_multi

PROBE_KINDS = ("ridge", "kernel", "mlp")


def planted_targets(seed=19, n=120, d=12):
    """One near-linear target, one weak nonlinear target, and a permuted copy of each.

    The nonlinear column is the case that separated the two code paths: a shared
    ``GridSearchCV`` alpha or a shared MLP is chosen partly for the OTHER columns in the
    batch, and a weak target is where that costs the most.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    linear = X[:, 0] + 0.05 * rng.randn(n)
    nonlinear = 0.2 * np.sin(3 * X[:, 1]) + rng.randn(n)
    return X, np.column_stack([linear, nonlinear, linear[rng.permutation(n)], nonlinear[rng.permutation(n)]])


class BatchedProbeEquivalenceTests(unittest.TestCase):
    """``cv_probe_r2_multi`` column j must equal ``cv_probe_r2`` on column j.

    ``eval.identifiability_report`` fits one target at a time and
    ``eval.dinov3_identifiability`` / ``eval.run_dci_compare`` batch them, so a VQ-VAE
    number and a DINO number are only on one scale while this holds.  It used to hold for
    ridge alone: kernel differed by 0.14 and MLP by 0.41 on the arrays below.
    """

    def setUp(self):
        warnings.simplefilter("ignore")
        self.X, self.Y = planted_targets()

    def test_every_probe_kind_matches_the_single_target_probe(self):
        for kind in PROBE_KINDS:
            batched = cv_probe_r2_multi(self.X, self.Y, n_splits=3, seeds=(0,), kind=kind)["mean"]
            for t in range(self.Y.shape[1]):
                with self.subTest(kind=kind, target=t):
                    separate = cv_probe_r2(self.X, self.Y[:, t], n_splits=3, seeds=(0,), kind=kind)["mean"]
                    self.assertAlmostEqual(separate, batched[t], places=10)

    def test_adding_null_columns_cannot_move_a_real_score(self):
        # The failure this rules out is subtle: n_null is a reporting knob, so a batch that
        # let it change the real R2 made the headline depend on how many nulls were asked for.
        for kind in PROBE_KINDS:
            with self.subTest(kind=kind):
                without = cv_probe_r2_multi(self.X, self.Y[:, :2], n_splits=3, seeds=(0,), kind=kind)["mean"]
                with_nulls = cv_probe_r2_multi(self.X, self.Y, n_splits=3, seeds=(0,), kind=kind)["mean"]
                np.testing.assert_allclose(without, with_nulls[:2], atol=1e-10)

    def test_only_ridge_is_declared_batchable(self):
        # A kind added to this tuple without a column-wise-identical multi-output fit would
        # silently reintroduce the mismatch, so pin the membership rather than the behaviour.
        self.assertEqual(tuple(BATCHABLE_PROBES), ("ridge",))

    def test_degenerate_inputs_return_nan_per_column(self):
        for X in (np.zeros((80, 0)), np.zeros((8, 5))):
            with self.subTest(shape=X.shape):
                out = cv_probe_r2_multi(X, self.Y[: len(X)], n_splits=5, seeds=(0,))
                self.assertEqual(out["mean"].shape, (self.Y.shape[1],))
                self.assertTrue(np.isnan(out["mean"]).all())


if __name__ == "__main__":
    unittest.main()
