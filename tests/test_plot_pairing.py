"""The two-arm pairing figures: what they compare, and what they refuse to claim."""

import csv
import json
import os
import tempfile
import unittest
import warnings

import numpy as np

from eval import plot_pairing as pp
from eval.plot_identifiability import THEME

T = THEME["light"]


def report(cross_gap, within_gap, cross_floor=None, within_floor=None, std=0.0, names=None):
    names = names or [f"f{i}" for i in range(len(cross_gap))]

    def arm(gaps, floors):
        floors = floors if floors is not None else [0.0] * len(gaps)
        rows = {
            n: dict(gap=float(g), std=float(std), floor_gap=float(fl), delta_floor=float(g - fl))
            for n, g, fl in zip(names, gaps, floors)
        }
        rows["_block"] = {"probe_features": 24}
        return rows

    return {
        "results": {
            "cross": {"content": arm(cross_gap, cross_floor), "style": {}},
            "within": {"content": arm(within_gap, within_floor), "style": {}},
        }
    }


class PairingTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_rows_come_back_sorted_by_the_difference(self):
        r = report([0.2, 0.9, 0.5], [0.5, 0.1, 0.5], names=["a", "b", "c"])
        order = [row[0] for row in pp.paired(r, "cross", "within", "content", "gap")]
        self.assertEqual(order, ["b", "c", "a"])  # +0.8, 0.0, -0.3

    def test_a_factor_only_one_arm_scored_is_dropped(self):
        # Not zero-filled: a missing factor is an unscored factor, and a zero would draw as
        # a tie between two arms, one of which was never measured.
        r = report([0.5, 0.5], [0.5, 0.5], names=["a", "b"])
        del r["results"]["within"]["content"]["b"]
        self.assertEqual([row[0] for row in pp.paired(r, "cross", "within", "content", "gap")], ["a"])

    def test_seed_spreads_add_in_quadrature(self):
        r = report([0.5], [0.4], std=0.03)
        self.assertAlmostEqual(pp.paired(r, "cross", "within", "content", "gap")[0][5], np.sqrt(2) * 0.03, places=9)

    def test_delta_floor_measures_against_each_arms_own_floor(self):
        r = report([0.9], [0.9], cross_floor=[0.6], within_floor=[0.2])
        gap = pp.paired(r, "cross", "within", "content", "gap")[0]
        dfl = pp.paired(r, "cross", "within", "content", "delta_floor")[0]
        self.assertAlmostEqual(gap[1] - gap[2], 0.0)  # equal raw recovery
        self.assertAlmostEqual(dfl[1] - dfl[2], -0.4, places=9)  # but within gained more

    def test_a_metric_sign_flip_is_reported(self):
        # The case that matters: cross recovers more in absolute terms, but only because its
        # untrained floor was already higher. The two metrics then name different winners and
        # neither is quotable, so the script has to say so.
        r = report([0.90], [0.85], cross_floor=[0.50], within_floor=[0.30], names=["vent"])
        self.assertEqual(pp.sign_disagreements(r, "cross", "within", "content"), ["vent"])

    def test_agreeing_metrics_report_nothing(self):
        r = report([0.9, 0.8], [0.5, 0.4], cross_floor=[0.2, 0.2], within_floor=[0.2, 0.2])
        self.assertEqual(pp.sign_disagreements(r, "cross", "within", "content"), [])

    def test_arms_default_to_the_two_that_were_scored(self):
        self.assertEqual(pp.pick_arms(report([0.5], [0.4]), None, None), ("cross", "within"))

    def test_three_arms_must_be_named(self):
        r = report([0.5], [0.4])
        r["results"]["third"] = r["results"]["cross"]
        with self.assertRaises(SystemExit):
            pp.pick_arms(r, None, None)
        self.assertEqual(pp.pick_arms(r, "third", "within"), ("third", "within"))

    def test_an_unknown_or_repeated_arm_is_refused(self):
        r = report([0.5], [0.4])
        for bad in (("nope", "within"), ("cross", "cross")):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    pp.pick_arms(r, *bad)


class FigureOutputTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_each_figure_ships_a_csv_twin(self):
        r = report([0.9, 0.4], [0.5, 0.6], cross_floor=[0.1, 0.1], within_floor=[0.1, 0.1], std=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            for fn, name in ((pp.fig_recovery, "recovery.png"), (pp.fig_advantage, "advantage.png")):
                path = os.path.join(tmp, name)
                self.assertTrue(fn(r, "cross", "within", "content", "gap", T, path))
                self.assertTrue(os.path.exists(path))
                self.assertTrue(os.path.exists(path.replace(".png", ".csv")))

    def test_the_advantage_csv_names_no_winner_it_cannot_separate(self):
        # separated=no means the two arms' seed spreads reach across zero. Leaving a name in
        # the winner column there is how an unsupported claim gets quoted out of the table.
        r = report([0.50, 0.90], [0.48, 0.40], std=0.20, names=["tied", "clear"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            pp.fig_advantage(r, "cross", "within", "content", "gap", T, path)
            rows = {row["factor"]: row for row in csv.DictReader(open(path.replace(".png", ".csv")))}
        self.assertEqual(rows["tied"]["separated"], "no")
        self.assertEqual(rows["tied"]["winner"], "")
        self.assertEqual(rows["clear"]["separated"], "yes")
        self.assertEqual(rows["clear"]["winner"], "cross")

    def test_an_empty_block_draws_nothing_rather_than_an_empty_axes(self):
        r = report([0.5], [0.4])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.png")
            self.assertFalse(pp.fig_recovery(r, "cross", "within", "style", "gap", T, path))
            self.assertFalse(os.path.exists(path))

    def test_both_themes_render(self):
        r = report([0.9, 0.4], [0.5, 0.6], std=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("light", "dark"):
                path = os.path.join(tmp, f"{mode}.png")
                self.assertTrue(pp.fig_advantage(r, "cross", "within", "content", "gap", THEME[mode], path))

    def test_main_writes_both_figures_for_both_blocks_it_has(self):
        r = report([0.9, 0.4], [0.5, 0.6], cross_floor=[0.1, 0.1], within_floor=[0.1, 0.1], std=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "compare.json")
            with open(src, "w") as fh:
                json.dump(r, fh)
            out = os.path.join(tmp, "fig")
            self.assertEqual(pp.main(["--json", src, "--out", out]), 0)
            self.assertEqual(
                sorted(os.listdir(out)),
                ["pairing_advantage.csv", "pairing_advantage.png", "pairing_recovery.csv", "pairing_recovery.png"],
            )

    def test_self_test_runs_without_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pp.main(["--self-test", "--out", tmp]), 0)


if __name__ == "__main__":
    unittest.main()
