"""local_directions.py (stage 72): the seventh shared local direction, on codes with known structure."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import local_directions as L


def views(code, rng, sd=0.5):
    return [code + sd * rng.standard_normal(code.shape) for _ in range(2)]


def square(v):
    s = v**2
    return (s - s.mean()) / s.std()


class LocalDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.factors = rng.standard_normal((1000, 7))
        z = cls.factors
        holder = np.column_stack([z, square(z[:, 0])])  # seven factors and one curve
        merged = (z[:, 1] + z[:, 2]) / np.sqrt(2)
        failure = np.column_stack([z[:, [0, 3, 4, 5, 6]], merged, square(z[:, 0]), square(z[:, 4])])
        cls.holder, cls.failure = views(holder, rng), views(failure, rng)  # reliability .8 per unit
        cls.clean = views(holder, rng, 0.2), views(failure, rng, 0.2)  # reliability .96

    def s7(self, pair, kind="own", k=32, shuffles=0):
        return L.score(*pair, self.factors, [k], shuffles, 0)[f"{kind}{k}"]

    def test_a_held_seventh_factor_beats_a_second_curve(self):
        # Where the estimate is not noise-limited: wide neighbourhoods, or clean views at the batch's size. At k = 32
        # with reliability .8 the held-out profile flattens (16 subjects a half for 8 directions) and the gap shrinks
        # to about 1.4 times: a property of the measure that stage 72 reports, not a bug.
        # Fixed neighbourhoods (the true factors) leave the curve coordinates free inside each neighbourhood, so a
        # curve leaves more behind there than in the code's own neighbourhoods, which bound every unit.
        cases = [("own", 128, self.holder, self.failure, 1.8), ("fixed", 128, self.holder, self.failure, 1.5)]
        cases += [("own", 32, *self.clean, 2.0), ("fixed", 32, *self.clean, 1.5)]
        for kind, k, held_pair, lost_pair, ratio in cases:
            with self.subTest(kind=kind, k=k):
                held = self.s7(held_pair, kind, k)["cov"][L.SEVENTH]
                lost = self.s7(lost_pair, kind, k)["cov"][L.SEVENTH]
                self.assertGreater(held, ratio * lost)

    def test_shuffling_the_pairs_removes_the_signal(self):
        row = self.s7(self.holder, shuffles=3)
        self.assertTrue(all(abs(n) < 0.2 * row["cov"][L.SEVENTH] for n in row["null"]))

    def test_view_noise_alone_shares_nothing(self):
        rng = np.random.default_rng(1)
        noise = [rng.standard_normal((1000, 8)) for _ in range(2)]
        cov = self.s7(noise)["cov"]
        self.assertLess(max(abs(v) for v in cov), 0.05)

    def test_the_halves_split_each_neighbourhood_without_overlap(self):
        a, b = L.splits(50, 32, 0)
        self.assertEqual(a.shape, (50, 16))
        for row_a, row_b in zip(a, b):
            self.assertFalse(set(row_a) & set(row_b))
        np.testing.assert_array_equal(L.splits(50, 32, 0)[0], a)  # one draw, shared by every code

    def test_each_neighbourhood_starts_with_its_subject(self):
        groups = L.neighbourhoods(self.factors, 32)
        self.assertEqual(groups.shape, (1000, 32))
        np.testing.assert_array_equal(groups[:, 0], np.arange(1000))
        self.assertTrue(all(len(set(row)) == 32 for row in groups))

    def test_bundle_names(self):
        here = Path("c")
        self.assertEqual(L.bundle_path(here, "c8_e500_s42").name, f"{L.P}_abl_codebatch_predict1_c8_20k.npz")
        self.assertEqual(L.bundle_path(here, "c9_e100_s3").name, f"{L.P}_abl_codebatch100_predict1_c9_20k_s3.npz")
        self.assertEqual(L.bundle_path(here, "c8_harm_s2").name, f"{L.P}_abl_codebatch_predict1_harm1_c8_20k_s2.npz")
        self.assertEqual(L.bundle_path(here, "c8_red_s5").name, f"{L.P}_abl_from10red_predict1_c8_10k_s5.npz")

    def test_auc(self):
        self.assertEqual(L.auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(L.auc([1], [1]), 0.5)
        self.assertEqual(L.auc([1, 2], [3, 4]), 0.0)


if __name__ == "__main__":
    unittest.main()
