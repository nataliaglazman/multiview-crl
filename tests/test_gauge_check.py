"""Stage 78's pieces that need no run: distances, the verdict and rho1 rules, the gauge statistics and the linear fit."""

import sys
import unittest
from pathlib import Path

import numpy as np
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import gauge_check as G
import pool_check as PC


def row(holds, margin, rho1, cell="c8_x"):
    return {"holds": holds, "last": [margin, 0.9], "worst": [0.9, 0.9], "margin": margin, "rho1": rho1, "cell": cell}


def pair_of(stored, recomputed, cell="c8_x"):
    return {"stored": stored, "recomputed": recomputed, "cell": cell}


class Distances(unittest.TestCase):
    def test_offset_in_recomputed_sds_and_log_ratio(self):
        width = 448
        mean_b, var_b = np.zeros((2, width)), np.full((2, width), 4.0)
        mean_a, var_a = mean_b + 1.0, var_b * np.exp(-1.0)
        summary = G.summarized(*G.distance(mean_a, var_a, mean_b, var_b))
        for view in summary:
            for part in G.PARTS:
                self.assertAlmostEqual(view[part]["mean_abs_offset"], 0.5, places=5)
                self.assertAlmostEqual(view[part]["mean_log_ratio"], -1.0, places=5)
                self.assertAlmostEqual(view[part]["mean_abs_log_ratio"], 1.0, places=5)

    def test_eps_keeps_a_constant_feature_finite(self):
        offset, ratio = G.distance(np.full((2, 3), 1e-4), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertTrue(np.isfinite(offset).all() and np.isfinite(ratio).all())
        self.assertTrue(np.allclose(ratio, 0))


class Rules(unittest.TestCase):
    def test_verdict_outcomes(self):
        same = {
            "a": pair_of(row(True, 0.6, 0.3), row(True, 0.58, 0.3)),
            "b": pair_of(row(False, 0.1, 0.6), row(False, 0.12, 0.6)),
        }
        self.assertEqual(G.verdict_rule(same)["outcome"], "no effect")
        moved = dict(same, c=pair_of(row(True, 0.7, 0.3), row(True, 0.5, 0.3)))
        self.assertEqual(G.verdict_rule(moved)["outcome"], "values move")
        self.assertEqual(G.verdict_rule(moved)["margins_moved"], ["c"])
        flipped = dict(
            same,
            d=pair_of(row(False, 0.38, 0.5), row(True, 0.41, 0.5)),
            e=pair_of(row(True, 0.42, 0.4), row(False, 0.3, 0.4)),
        )
        rule = G.verdict_rule(flipped)
        self.assertEqual(rule["outcome"], "verdicts move")
        self.assertEqual((rule["to_hold"], rule["to_fail"]), (["d"], ["e"]))
        self.assertEqual(rule["holds"], {"stored": 2, "recomputed": 2})

    def test_rho1_rule_part_a(self):
        good = {f"s{i}": pair_of(row(i < 5, 0.5, 0.3 + 0.05 * i), row(i < 5, 0.5, 0.3 + 0.05 * i)) for i in range(10)}
        rule = G.rho_rule(good, "recomputed", "A")
        self.assertEqual((rule["auc"], rule["right"], rule["pairs"]), (1.0, 25, 25))
        self.assertEqual((rule["picked"], rule["with_holder"], rule["outcome"]), (110, 110, "stands"))
        reversed_ = {
            k: pair_of(r["stored"], dict(r["recomputed"], rho1=1 - r["recomputed"]["rho1"])) for k, r in good.items()
        }
        self.assertEqual(G.rho_rule(reversed_, "recomputed", "A")["outcome"], "fails")
        one = {k: pair_of(r["stored"], dict(r["recomputed"], holds=k == "s0")) for k, r in good.items()}
        self.assertTrue(G.rho_rule(one, "recomputed", "A")["outcome"].startswith("described only"))

    def test_rho1_rule_part_b_needs_three_quarters_of_pairs(self):
        rows = {}
        for c in range(4):
            for i in range(5):
                rho = 0.3 + 0.05 * i if c < 3 else 0.6 - 0.05 * i  # the last cell is reversed
                rows[f"c{c}_s{i}"] = pair_of(row(i < 2, 0.5, rho, f"c{c}"), row(i < 2, 0.5, rho, f"c{c}"), f"c{c}")
        rule = G.rho_rule(rows, "stored", "B")
        self.assertEqual((rule["right"], rule["pairs"]), (18, 24))
        self.assertAlmostEqual(rule["auc"], 0.75)  # 72 of 96 holder-failure pairs across cells
        self.assertEqual(rule["outcome"], "unresolved")


class Exports(unittest.TestCase):
    def test_pool_dir_keeps_the_stored_default(self):
        self.assertEqual(PC.pool_dir("stored"), PC.POOL)
        self.assertNotEqual(PC.pool_dir("recomputed"), PC.POOL)

    def test_gauge_statistics_reads_each_view(self):
        class Pool(nn.Module):
            def __init__(self):
                super().__init__()
                self.gauges = nn.ModuleList(nn.BatchNorm1d(5, affine=False) for _ in range(2))

        pool = Pool()
        for v, g in enumerate(pool.gauges):
            g.running_mean.fill_(v + 1.0)
            g.running_var.fill_(v + 3.0)
        mean, var = PC.gauge_statistics(pool)
        self.assertEqual(mean.shape, (2, 5))
        self.assertTrue(np.allclose(mean[:, 0], [1, 2]) and np.allclose(var[:, 0], [3, 4]))

    def test_linear_fit(self):
        rng = np.random.default_rng(0)
        code = rng.standard_normal((2, 500, 8))
        affine = np.stack([c @ rng.standard_normal((8, 8)) + 3 for c in code])
        self.assertTrue(np.allclose(G.linear_fit(code, affine), 1))
        self.assertLess(G.linear_fit(code, rng.standard_normal((2, 500, 8))).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
