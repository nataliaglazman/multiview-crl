"""Stage 84's cross-seed completeness (out/burden_feasibility_20261102/panel_check.py) on synthetic codes.

Run from the repo root with PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:.
"""

import unittest

import numpy as np
from sklearn.kernel_ridge import KernelRidge

import panel_check as PC

N_TRAIN, N_TEST = 1000, 500


def views_of(latent, rng, width=6, noise=0.1, private=0):
    """A content-only code of the latent (a fixed random non-linear map), with each view's own noise and, optionally,
    view-private coordinates the other view does not share."""
    a = rng.normal(size=(latent.shape[1], width))
    b = rng.normal(size=(latent.shape[1], width))
    clean = latent @ a + 0.5 * np.tanh(latent @ b)
    views = []
    for _ in range(2):
        v = clean + noise * rng.normal(size=clean.shape)
        if private:
            v = np.column_stack([v, rng.normal(size=(len(v), private))])
        views.append(v)
    code = np.stack(views)
    return code[:, :N_TRAIN], code[:, N_TRAIN:]


class PanelCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(N_TRAIN + N_TEST, 4))
        cls.codes = {
            "x": views_of(latent, rng),
            "y": views_of(latent, rng),
            "lacks": views_of(latent[:, :3], rng),  # misses the fourth latent
            "private": views_of(latent, rng, private=2),  # complete, plus view-private coordinates
        }
        pairs = [(a, b) for a in cls.codes for b in cls.codes if a != b]
        cls.q, cls.by_view, cls.ceiling, cls.kept, _ = PC.pair_matrix(cls.codes, pairs)

    def test_kernel_matches_sklearn(self):
        rng = np.random.default_rng(1)
        x_train, x_test = rng.normal(size=(300, 5)), rng.normal(size=(100, 5))
        y = np.sin(x_train[:, :2]) + x_train[:, 2:4]
        model = PC.LooKernel(x_train, x_test, gammas=(0.5,), alphas=(0.1,))
        ours, setting = model.predict(y)
        self.assertEqual(setting, (0.5, 0.1))
        z_train, z_test = PC.standardized(x_train, x_test)
        design = np.column_stack([np.ones(300), z_train])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        kr = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5 / 5).fit(z_train, y - design @ coef)
        theirs = np.column_stack([np.ones(100), z_test]) @ coef + kr.predict(z_test)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_complete_codes_determine_each_other(self):
        for a, b in (("x", "y"), ("y", "x"), ("x", "lacks"), ("y", "lacks")):
            self.assertGreater(self.q[(a, b)], 0.7, (a, b))

    def test_a_code_missing_a_part_cannot_determine_one_that_holds_it(self):
        self.assertLess(self.q[("lacks", "x")], 0.3)
        self.assertLess(self.q[("lacks", "y")], 0.3)

    def test_view_private_content_does_not_count(self):
        # its two private coordinates are not shared, so its target keeps only the clean code's 6 linear directions
        self.assertEqual(self.kept["x"], 6)
        self.assertEqual(self.kept["private"], 6)
        # complete codes read .64-.91 of each other and the code missing a latent reads below 0 (seed 0)
        self.assertGreater(self.q[("x", "private")], 0.5)
        self.assertGreater(self.q[("private", "x")], 0.5)
        self.assertLess(self.q[("lacks", "private")], 0.2)

    def test_scores_pairs_and_selection(self):
        held = {"x": True, "y": True, "lacks": False, "private": True}
        members = list(self.codes)
        scores = PC.panel_scores(self.q, members)
        self.assertEqual(PC.pairs_right(scores, held), (3, 3))
        with_holder, right, _ = PC.selection(self.q, members, held, 2)
        self.assertEqual((with_holder, right), (6, 6))

    def test_rule_width_takes_the_smallest_width_within_the_bound(self):
        self.assertEqual(PC.rule_width({6: 0.25, 7: 0.03, 8: 0.01}), 7)
        self.assertEqual(PC.rule_width({6: 0.08, 7: 0.05, 8: 0.03}), 6)
        self.assertEqual(PC.rule_width({9: 0.4, 10: 0.2, 11: 0.15}), 11)  # none within the bound: the widest

    def test_pipeline_picks_charts_and_counts(self):
        labels = ["a", "b", "c", "d"]
        held = {"a": True, "b": True, "c": False, "d": False}
        good = {"a": 0.9, "b": 0.85, "c": 0.4, "d": 0.3}  # each code's q against any other
        q = {(x, y): good[x] for x in labels for y in labels if x != y}
        charts = {"a": {6: (False, 0.3), 7: (True, 0.02), 8: (True, 0.01)},
                  "b": {6: (False, 0.2), 7: (False, 0.2), 8: (True, 0.05)},  # its rule width is 8
                  "c": {6: (False, 0.05), 7: (False, 0.04), 8: (False, 0.03)},
                  "d": {6: (False, 0.3), 7: (False, 0.2), 8: (False, 0.2)}}
        rows, width = PC.pipeline_cell(labels, q, held, charts)
        self.assertEqual(width, {"a": 7, "b": 8, "c": 6, "d": 8})
        self.assertEqual(len(rows), 4)
        picks = {tuple(r["seeds"]): (r["pick"], r["success"]) for r in rows}
        self.assertEqual(picks[("a", "b", "c")], ("a", True))
        self.assertEqual(picks[("b", "c", "d")], ("b", True))
        self.assertEqual(picks[("a", "c", "d")], ("a", True))
        self.assertEqual(sum(r["success"] for r in rows), 4)

    def test_completeness_is_a_worst_direction_ratio(self):
        e_ref = np.diag([0.9, 0.5])
        self.assertAlmostEqual(PC.completeness(np.diag([0.9, 0.5]), e_ref), 1.0)
        self.assertAlmostEqual(PC.completeness(np.diag([0.9, 0.1]), e_ref), 0.2)
        self.assertAlmostEqual(PC.completeness(np.diag([0.45, 0.5]), e_ref), 0.5)


if __name__ == "__main__":
    unittest.main()
