"""Stage 67's noisy-anchor check (out/burden_feasibility_20261102/anchor_check.py): the readout, the measurement model,
the adjustment, and choosing runs on fake banks whose hidden factors do or do not follow the measured ones."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
import anchor_check as A
import criterion_check as C

C.BOOT = 100
A.HIDE = tuple(s for s in A.HIDE if len(s) == 1) + ((5, 6),)


def fake_entries(hidden_follows, banks=2, runs=6, n=300, seed=0):
    """Runs whose code reads every factor through noise set by a per-run quality; the hidden factors (lr, sulcal) share
    the run's quality if ``hidden_follows``, else get their own."""
    rng = np.random.default_rng(seed)
    entries = []
    for b in range(banks):
        z = rng.standard_normal((n, 7))
        zc = np.zeros((n, 9))
        zc[:, list(A.LATENT_COLUMNS)] = z
        vox = 50 * (1 + np.tanh(z[:, A.BURDEN]))
        mix = rng.standard_normal((7, 10))
        for r in range(runs):
            quality = np.full(7, rng.random())
            if not hidden_follows:
                quality[[5, 6]] = rng.random()
            noise = 2 * (1 - quality)
            codes = [(z + rng.standard_normal(z.shape) * noise) @ mix for _ in range(2)]
            entries.append(
                {
                    "run": f"b{b}r{r}",
                    "step": 1,
                    "bank": f"bank{b}",
                    "bank_root": f"/banks/bank{b}",
                    "t1key": "t1_04",
                    "width": 10,
                    "codes": codes,
                    "zc": zc,
                    "vox": vox,
                    "own": z,
                }
            )
    return entries


class PieceTests(unittest.TestCase):
    def test_self_test(self):
        A.self_test()

    def test_measured_burden_is_the_lesion_volume(self):
        zc = np.arange(18, dtype=float).reshape(2, 9)
        m = A.measured_values(zc, [7.0, 9.0])
        self.assertEqual(m[:, A.BURDEN].tolist(), [7.0, 9.0])
        self.assertEqual(m[:, 0].tolist(), [0.0, 9.0])
        self.assertEqual(m[:, 3].tolist(), [5.0, 14.0])

    def test_noise_draws_are_shared_within_a_bank(self):
        rng = np.random.default_rng(1)
        zc, vox = rng.standard_normal((120, 9)), rng.random(120)
        first, columns = A.bank_targets(zc, vox, "bankX:120", draws=2, split_draws=1)
        again, _ = A.bank_targets(zc, vox, "bankX:120", draws=2, split_draws=1)
        other, _ = A.bank_targets(zc, vox, "bankY:120", draws=2, split_draws=1)
        np.testing.assert_array_equal(first, again)
        noisy = columns.index(("raw", 0.5, 0, 0))
        exact = columns.index(("raw", 1.0, 0, 0))
        self.assertFalse(np.allclose(first[:, noisy], other[:, noisy]))
        np.testing.assert_array_equal(first[:, exact], other[:, exact])


class ChoiceTests(unittest.TestCase):
    def run_check(self, hidden_follows):
        entries = fake_entries(hidden_follows)
        position = A.score(entries, draws=2, split_draws=1)
        return A.assess(entries, position, draws=2, split_draws=1)

    def test_anchors_choose_when_hidden_factors_follow(self):
        result = self.run_check(True)
        self.assertTrue(result["rule"]["usable"], result["rule"])
        self.assertGreater(result["rule"]["rho"], 0.7)
        exact = result["bank"]["splits"]["adni"]["rows"]["exact own parts"]
        self.assertGreater(exact["removed"], 0.7)

    def test_anchors_cannot_choose_when_hidden_factors_do_not_follow(self):
        follows = self.run_check(True)["rule"]["rho"]
        alone = self.run_check(False)["rule"]["rho"]
        self.assertLess(alone, follows - 0.3)


if __name__ == "__main__":
    unittest.main()
