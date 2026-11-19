"""Stage 54's intrinsic dimension (out/burden_feasibility_20261102/intrinsic_dimension.py): TwoNN and Levina-Bickel
on manifolds of known dimension, and the calibration embeddings."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
import intrinsic_dimension as ID


def subspace(d, width=24, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) @ rng.standard_normal((d, width))


class EstimatorTests(unittest.TestCase):
    def test_two_nn_reads_a_linear_subspace(self):
        for d in (3, 7, 10):
            self.assertAlmostEqual(ID.two_nn(subspace(d)), d, delta=0.15 * d)

    def test_levina_bickel_reads_a_linear_subspace(self):
        for d in (3, 7):
            self.assertAlmostEqual(ID.levina_bickel(subspace(d), 20), d, delta=0.15 * d)

    def test_a_curved_surface_reads_two(self):
        rng = np.random.default_rng(1)
        t = 1.5 * np.pi * (1 + 2 * rng.random(3000))
        roll = np.column_stack([t * np.cos(t), 20 * rng.random(3000), t * np.sin(t)])
        self.assertAlmostEqual(ID.two_nn(roll), 2, delta=0.3)

    def test_decimation_keeps_a_noiseless_estimate(self):
        x = subspace(5)
        for fraction in ID.FRACTIONS:
            self.assertAlmostEqual(ID.decimated(x, fraction), 5, delta=1.0)

    def test_duplicates_are_left_out(self):
        x = subspace(4)
        self.assertTrue(np.isfinite(ID.two_nn(np.concatenate([x, x[:50]]))))


class CalibrationTests(unittest.TestCase):
    def test_a_noiseless_embedding_keeps_the_latent_dimension(self):
        latents = np.random.default_rng(2).standard_normal((1500, 6))
        embedded = ID.embed(latents, 24, 0.0, seed=0)
        self.assertEqual(embedded.shape, (1500, 24))
        self.assertAlmostEqual(ID.estimates(embedded)[0], 6, delta=1.2)

    def test_reading_noise_raises_the_small_scale_estimate(self):
        latents = np.random.default_rng(3).standard_normal((1000, 4))
        clean = ID.estimates(ID.embed(latents, 24, 0.0, seed=0))[0]
        noisy = ID.estimates(ID.embed(latents, 24, 0.3, seed=0))[0]
        self.assertGreater(noisy, clean + 2)

    def test_the_sphere_removes_the_norm(self):
        x = ID.on_sphere(subspace(3, width=10))
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
