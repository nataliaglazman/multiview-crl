"""out/burden_feasibility_20261102/heterogeneity_ica.py: joint diagonalisation, and own parts separated from a mixed code
only when their variances change across levels."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import ortho_group

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import bank as B  # noqa: E402
import heterogeneity_ica as H  # noqa: E402


class JointDiagonalisationTests(unittest.TestCase):
    def test_recovers_a_shared_rotation(self):
        rng = np.random.default_rng(0)
        rotation = ortho_group.rvs(6, random_state=1)
        matrices = [rotation @ np.diag(rng.uniform(0.2, 3.0, 6)) @ rotation.T for _ in range(4)]
        v = H.joint_diagonalize(matrices)
        overlap = np.abs(v.T @ rotation)
        np.testing.assert_allclose(np.sort(overlap.max(1)), np.ones(6), atol=1e-6)
        for m in matrices:
            d = v.T @ m @ v
            self.assertLess(np.abs(d - np.diag(np.diag(d))).max(), 1e-6)


class SeparationTests(unittest.TestCase):
    def test_own_parts_need_the_variance_to_change(self):
        scm = B.causal_scm(B.run_args(0.4, causal=True).seed)
        flat = H.run(scm, 5, 0.0, 1.0, 0.0, 0.3, 0.05, seed=0)
        varying = H.run(scm, 5, 1.0, 1.0, 0.0, 0.3, 0.05, seed=0)
        own = slice(0, len(B.CAUSAL_FACTORS))
        self.assertLess(flat["jd"][own].mean(), 0.4)  # Gaussian sources with equal profiles: no better than chance
        self.assertGreater(varying["jd"][own].mean(), 0.7)
        self.assertGreater(varying["jd"][own].mean(), varying["ica"][own].mean() + 0.1)
        burden = list(B.CAUSAL_FACTORS.values()).index("burden")
        self.assertTrue(varying["found"][burden, H.ANCHORS.index("temporal")])  # burden's noise reaches temporal

    def test_ancestry_of_the_bank_graph(self):
        reach = H.ancestry(B.causal_scm(B.run_args(0.4, causal=True).seed))
        names = list(B.CAUSAL_FACTORS.values())
        expected = {
            "brain": {"brain", "ventricle", "thickness", "temporal"},
            "ventricle": {"ventricle", "temporal"},
            "burden": {"temporal"},
            "thickness": {"thickness", "temporal"},
            "temporal": {"temporal"},
            "lr": set(),
            "sulcal": set(),
        }
        for name, anchors in expected.items():
            self.assertEqual({H.ANCHORS[i] for i in np.flatnonzero(reach[names.index(name)])}, anchors, name)


if __name__ == "__main__":
    unittest.main()
