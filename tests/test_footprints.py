"""Stage 54's footprints (out/burden_feasibility_20261102/footprints.py): the cell Grams of a code's input gradient,
the joint diagonalisation that makes footprints sparse, and the locality measures."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ortho_group

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
import footprints as FP

SIZE = 16  # a 16^3 toy volume, so a 4^3 grid of cells


def planted(seed=0, k=6, matrices=60):
    """Symmetric matrices that one orthogonal basis diagonalises exactly, and that basis."""
    rng = np.random.default_rng(seed)
    basis = ortho_group.rvs(k, random_state=seed)
    mats = np.stack([basis @ np.diag(rng.exponential(size=k)) @ basis.T for _ in range(matrices)])
    return mats, basis


def tensor_of(mats):
    flat = mats.reshape(len(mats), -1)
    k = mats.shape[1]
    return (flat.T @ flat).reshape(k, k, k, k)


def toy_code(seed=0, factors=4, channels=2):
    """A linear code of ``factors`` latents, each read from its own disjoint block of cells, mixed by a random
    invertible matrix: (code function, mixing matrix, per-factor voxel weights)."""
    rng = np.random.default_rng(seed)
    grid = SIZE // FP.CELL
    cells = rng.permutation(grid**3)[: factors * 3].reshape(factors, 3)  # three cells per factor
    weights = np.zeros((factors, channels, SIZE, SIZE, SIZE))
    for j in range(factors):
        for c in cells[j]:
            x, y, z = np.unravel_index(c, (grid,) * 3)
            block = rng.standard_normal((channels, FP.CELL, FP.CELL, FP.CELL))
            weights[j, :, x * 4 : x * 4 + 4, y * 4 : y * 4 + 4, z * 4 : z * 4 + 4] = block
    mixing = rng.standard_normal((factors, factors)) + 2 * np.eye(factors)
    full = torch.tensor((mixing @ weights.reshape(factors, -1)), dtype=torch.float32)

    def code(x):
        return x.flatten(1) @ full.T

    return code, mixing, weights


class JacobiTests(unittest.TestCase):
    def test_recovers_a_planted_basis(self):
        mats, basis = planted()
        start = ortho_group.rvs(6, random_state=99)  # start away from the answer
        found, _ = FP.jacobi(FP.rotate_tensor(tensor_of(mats), start))
        overlap = np.abs((start @ found).T @ basis)
        np.testing.assert_allclose(overlap.max(1), 1, atol=1e-6)  # a permutation, up to sign

    def test_energy_is_the_diagonal_energy_and_never_falls(self):
        rng = np.random.default_rng(1)
        mats = np.stack([(lambda m: m @ m.T)(rng.standard_normal((5, 5))) for _ in range(30)])
        tensor = tensor_of(mats)
        basis, energy = FP.joint_diagonalise(tensor, restarts=3)
        direct = sum(np.square(np.diag(basis.T @ m @ basis)).sum() for m in mats)
        self.assertAlmostEqual(energy, direct, places=6)
        self.assertGreaterEqual(energy, FP.diagonal_energy(tensor) - 1e-9)
        np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-10)

    def test_rotating_the_tensor_rotates_the_matrices(self):
        mats, _ = planted(2, k=4, matrices=10)
        rotation = ortho_group.rvs(4, random_state=3)
        expected = tensor_of(np.stack([rotation.T @ m @ rotation for m in mats]))
        np.testing.assert_allclose(FP.rotate_tensor(tensor_of(mats), rotation), expected, atol=1e-10)


class GramTests(unittest.TestCase):
    def test_jacobian_is_the_code_weights(self):
        code, mixing, weights = toy_code()
        x = torch.randn(3, 2, SIZE, SIZE, SIZE)
        jac = FP.jacobian(code, x, 4)
        expected = np.einsum("kj,jcxyz->kcxyz", mixing, weights)
        for b in range(3):
            np.testing.assert_allclose(jac[b].numpy(), expected, atol=1e-4)

    def test_whitened_grams_unmix_disjoint_factors(self):
        """Factors read from disjoint cells, linearly mixed: both rotations return the factors."""
        code, mixing, _ = toy_code()
        grams = FP.Grams(4, 2, (SIZE // FP.CELL) ** 3)
        for _ in range(2):
            grams.add(FP.jacobian(code, torch.randn(4, 2, SIZE, SIZE, SIZE), 4))
        pooled, moment = grams.result()
        rot = FP.rotations(pooled, moment, restarts=2)
        self.assertEqual(rot["kept"], 4)
        for kind in FP.KINDS:
            unmixed = rot[kind] @ mixing  # directions in factor coordinates
            unmixed = unmixed / np.abs(unmixed).max(1, keepdims=True)
            np.testing.assert_allclose(np.sort(np.abs(unmixed), 1)[:, :-1], 0, atol=1e-3)
            np.testing.assert_allclose(FP.effective_cells(FP.footprints(rot[kind], pooled.sum(0))), 3, atol=0.5)
        units = FP.effective_cells(FP.footprints(np.eye(4), pooled.sum(0)))
        self.assertTrue(np.all(units > 4))  # the mixed units read from several factors' cells

    def test_subject_cells_match_a_direct_count(self):
        rng = np.random.default_rng(4)
        per_subject = np.stack(
            [np.stack([(lambda m: m @ m.T)(rng.standard_normal((3, 2))) for _ in range(10)]) for _ in range(5)]
        )  # (subjects, cells, K, K)
        flat = per_subject.reshape(5 * 10, 9)
        moment, total = flat.T @ flat / 5, per_subject.sum(1).mean(0)
        w = rng.standard_normal((2, 3))
        values = np.einsum("nd,scde,ne->nsc", w, per_subject, w)
        share = values / values.sum((1, 2), keepdims=True)
        direct = 1 / np.square(share).sum((1, 2)) / 5
        np.testing.assert_allclose(FP.subject_cells(w, total, moment), direct, rtol=1e-10)

    def test_a_footprint_every_direction_shares_has_no_divergence(self):
        """Cell Grams that are one matrix scaled per cell: every direction reads like the whole code (KL 0), the mean
        footprint is that scale, and it is the same for any rotation."""
        rng = np.random.default_rng(5)
        common = (lambda m: m @ m.T)(rng.standard_normal((4, 4)))
        scale = rng.exponential(size=20)
        grams = scale[:, None, None] * common
        white = FP.whitener(grams.sum(0))
        mean = FP.mean_share(grams, white)
        np.testing.assert_allclose(mean, scale / scale.sum(), atol=1e-12)
        directions = rng.standard_normal((6, 4))
        np.testing.assert_allclose(FP.divergence(FP.footprints(directions, grams), mean), 0, atol=1e-12)
        rotated = ortho_group.rvs(4, random_state=6).T @ white
        np.testing.assert_allclose(FP.mean_share(grams, rotated), mean, atol=1e-12)

    def test_effective_cells_of_a_flat_footprint(self):
        footprint = np.zeros((1, 64))
        footprint[0, :8] = 2.5
        self.assertAlmostEqual(float(FP.effective_cells(footprint)[0]), 8.0)


if __name__ == "__main__":
    unittest.main()
