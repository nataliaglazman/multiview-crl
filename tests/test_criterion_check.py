"""Stage 62's label-free criterion check (out/burden_feasibility_20261102/criterion_check.py): the signals on data with
known shared structure, and the ranking analysis on runs whose truth a signal does or does not follow."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
import criterion_check as C

C.BOOT = 200


class SignalTests(unittest.TestCase):
    def test_self_test(self):
        C.self_test()

    def test_held_out_canonical_is_not_inflated_by_width(self):
        rng = np.random.default_rng(1)
        a, b = rng.standard_normal((1400, 64)), rng.standard_normal((1400, 64))
        rho = C.held_out_canonical((a[:1000], b[:1000]), (a[1000:], b[1000:]))
        self.assertLess(np.clip(rho, 0, 1).sum(), 3.0)
        self.assertEqual(int((rho > 0.5).sum()), 0)

    def test_dead_units_are_ignored(self):
        rng = np.random.default_rng(2)
        z = rng.standard_normal((500, 6))
        dead = np.column_stack([z, np.zeros(500)])
        self.assertAlmostEqual(C.rankme(dead), C.rankme(z), places=6)
        rho = C.held_out_canonical((dead[:400], dead[:400]), (dead[400:], dead[400:]))
        self.assertEqual(len(rho), 6)

    def test_memorisation_sign(self):
        rng = np.random.default_rng(3)
        fit = rng.standard_normal((200, 8))
        held = rng.standard_normal((100, 8))
        signals = C.label_free((fit, fit), (held, held + 3 * rng.standard_normal((100, 8))))
        self.assertGreater(signals["memorisation"], 0.5)


class AnalysisTests(unittest.TestCase):
    def rows(self, groups=3, size=6, small=False):
        rng = np.random.default_rng(4)
        rows = {}
        for g in range(groups):
            for i in range(size if not (small and g == 0) else 2):
                q = rng.random()
                rows[f"g{g}r{i}"] = {
                    "bank": f"b{g}",
                    "t1key": "t1_04",
                    "width": 24 if i % 2 else 10,
                    "signals": {**{k: float(rng.standard_normal()) for k in C.SIGNALS}, "cca_sum": q},
                    "truth": {t: q for t in C.TRUTHS},
                }
        return rows

    def test_small_groups_are_left_out(self):
        result = C.analyse(self.rows(small=True), "bank")
        self.assertEqual(sorted(result["groups"]), ["b1 t1_04", "b2 t1_04"])

    def test_perfect_signal_removes_all_regret(self):
        cell = C.analyse(self.rows(), "bank")["signals"]["cca_sum"]
        self.assertAlmostEqual(cell["own_mean"]["rho"], 1.0)
        self.assertAlmostEqual(cell["own_mean"]["regret_removed"], 1.0)
        self.assertEqual(cell["own_mean"]["right_sign"], 3)
        self.assertTrue(cell["usable"])
        self.assertEqual(sorted(cell["own_mean"]["by_group"]), ["b0 t1_04", "b1 t1_04", "b2 t1_04"])

    def test_width_grouping_splits_groups(self):
        result = C.analyse(self.rows(size=8), "bank_width")
        self.assertEqual(len(result["groups"]), 6)


if __name__ == "__main__":
    unittest.main()
