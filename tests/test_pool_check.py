"""Stage 75's merge and leftover checks (pool_check.py) on synthetic codes whose content is known."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import pool_check as PC

NAMES = ["brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal"]
CURVES = (lambda o: o[:, 0] ** 2 - 1, lambda o: o[:, 0] ** 3 / 3 - o[:, 0])


def r2(features, target):
    predicted = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), features, target, cv=KFold(5))
    return float(1 - ((target - predicted) ** 2).mean() / target.var())


class Synthetic:
    """Seven independent own parts for training and test subjects; a run's code holds some of them linearly and
    curves of brain, behind one fixed mixing shared by the views, with independent noise per view."""

    def __init__(self, n_train=1200, n_test=600, seed=0):
        self.rng = np.random.default_rng(seed)
        self.own = [self.rng.standard_normal((n, 7)) for n in (n_train, n_test)]

    def code(self, keep, curves, noise=0.15):
        mix = self.rng.standard_normal((len(keep) + curves, len(keep) + curves))
        out = []
        for own in self.own:
            x = np.column_stack([own[:, j] for j in keep] + [c(own) for c in CURVES[:curves]])
            out.append(np.stack([(x + noise * self.rng.standard_normal(x.shape)) @ mix for _ in range(2)]))
        return out  # train (2, n, d), test (2, n, d)

    def gap(self, width=24, style=5, noise=0.3):
        """Backbone-like features per view: every own part, brain squared, and the view's own style, mixed."""
        out = [[], []]
        for v in range(2):
            content = self.rng.standard_normal((8, width))
            looks = self.rng.standard_normal((style, width))
            for split, own in enumerate(self.own):
                x = np.column_stack([own, CURVES[0](own)]) @ content
                x += self.rng.standard_normal((len(own), style)) @ looks + noise * self.rng.standard_normal(x.shape)
                out[split].append(x)
        return [np.stack(s) for s in out]


class PoolCheckTests(unittest.TestCase):
    def setUp(self):
        self.s = Synthetic()

    def merged(self, runs, **kw):
        kw = dict(restarts=2, seed=0, held_out=200, steps=2500, patience=300) | kw
        code, losses, outputs = PC.merge([r[0] for r in runs], [r[1] for r in runs], **kw)
        self.assertEqual(code.shape, (2, 600, 8))
        self.assertEqual(len(losses), kw["restarts"])
        return code

    def test_merge_keeps_what_the_holders_hold(self):
        holders = [self.s.code(range(7), 1) for _ in range(2)]
        failures = [self.s.code([j for j in range(7) if j != lost], 2) for lost in (2, 4, 6)]
        code = self.merged(holders + failures)
        for j in range(7):
            self.assertGreater(r2(code[0], self.s.own[1][:, j]), 0.8, NAMES[j])

    def test_merge_recovers_what_no_single_run_holds(self):
        failures = [self.s.code([j for j in range(7) if j != lost], 2) for lost in (2, 4, 6)]
        code = self.merged(failures)
        for j in (2, 4, 6):
            self.assertGreater(r2(code[1], self.s.own[1][:, j]), 0.7, NAMES[j])

    def test_merge_cannot_invent_a_part_no_run_holds(self):
        runs = [self.s.code(range(6), 2) for _ in range(3)]
        code = self.merged(runs)
        self.assertLess(r2(code[0], self.s.own[1][:, 6]), 0.1)

    def test_leftover_finds_the_part_a_code_lost(self):
        gap = self.s.gap()
        failure, holder = self.s.code(range(6), 2), self.s.code(range(7), 1)
        lost = PC.shared_leftover(failure[0], gap[0], failure[1], gap[1])
        kept = PC.shared_leftover(holder[0], gap[0], holder[1], gap[1])
        self.assertGreater(lost["rho"][0], 0.8)
        self.assertGreater(lost["rho"][0], kept["rho"][0] + 0.2)
        for v in range(2):
            self.assertGreater(np.corrcoef(lost["variates"][v][:, 0], self.s.own[1][:, 6])[0, 1] ** 2, 0.7)
        self.assertGreater(kept["gap_rho"][0], 0.9)  # the GAP itself shares content across views

    def test_style_alone_leaves_no_shared_leftover(self):
        rng = np.random.default_rng(3)
        x = [rng.standard_normal((n, 10)) for n in (800, 400)]
        y = [rng.standard_normal((n, 10)) for n in (800, 400)]
        rho, _, _ = PC.held_out(x[1], y[1], PC.cca(x[0], y[0]))
        self.assertLess(abs(rho[0]), 0.2)

    def test_repair_swaps_a_curve_for_the_lost_part(self):
        gap = self.s.gap()
        failure = self.s.code(range(6), 2)
        lost = PC.shared_leftover(failure[0], gap[0], failure[1], gap[1])
        variates = [lost["variates"][v][:, 0] for v in range(2)]
        code, removed, predictable = PC.repair(failure[0], failure[1], variates)
        self.assertEqual(code.shape, (2, 600, 8))
        self.assertGreater(predictable, 0.8)
        curves = np.column_stack([c(self.s.own[1]) for c in CURVES])
        self.assertGreater(r2(curves, removed[0]), 0.8)  # the removed direction is a curve
        for j in range(7):
            self.assertGreater(r2(code[0], self.s.own[1][:, j]), 0.6, NAMES[j])

    def test_repair_keeps_a_holder_whole(self):
        gap = self.s.gap()
        holder = self.s.code(range(7), 1)
        kept = PC.shared_leftover(holder[0], gap[0], holder[1], gap[1])
        code, removed, _ = PC.repair(holder[0], holder[1], [kept["variates"][v][:, 0] for v in range(2)])
        self.assertGreater(np.corrcoef(removed[0], CURVES[0](self.s.own[1]))[0, 1] ** 2, 0.8)
        for j in range(7):
            self.assertGreater(r2(code[1], self.s.own[1][:, j]), 0.8, NAMES[j])

    def test_holds_reads_the_ventricle_burden_plane(self):
        own = self.s.own[1]
        noise = np.random.default_rng(5).standard_normal((2, len(own), 1))
        whole = np.stack([np.column_stack([own, noise[v]]) for v in range(2)])
        self.assertTrue(PC.score(whole, own, NAMES)["holds"])
        merged = np.column_stack([own[:, [0]], own[:, 1] + own[:, 2], own[:, 3:]])
        blended = np.stack([np.column_stack([merged, noise[v], -noise[v]]) for v in range(2)])
        self.assertFalse(PC.score(blended, own, NAMES)["holds"])

    def test_names_and_verdicts(self):
        self.assertTrue(PC.run_name("aproj64", 2).endswith("_abl_codebatch_predict1_proj64_c8_20k_a100_s2"))
        self.assertTrue(PC.run_name("aproj64", 42).endswith("_proj64_c8_20k_a100"))
        self.assertEqual(PC.labels_of("e500")[0], "c8_e500_s42")
        row = "{} 0.56/0.56 0.56/0.56 12.6/11.9 0.079/0.072 0.835/0.838 0.834 {} no: x"
        with tempfile.TemporaryDirectory() as tmp:
            for stage, verdict in (("a", "yes"), ("b", "yes")):
                (Path(tmp) / stage).mkdir()
                lines = [row.format("c8_e500_s2", verdict), row.format("c10_e500_s2", "no"), "per cell: holds 1/1"]
                (Path(tmp) / stage / "holds.txt").write_text("\n".join(lines))
            self.assertEqual(PC.verdicts(Path(tmp), ("a", "b")), {"c8_e500_s2": True})
            (Path(tmp) / "b" / "holds.txt").write_text(row.format("c8_e500_s2", "no"))
            with self.assertRaises(SystemExit):
                PC.verdicts(Path(tmp), ("a", "b"))

    def test_stage_verdicts_read_every_width(self):
        row = "{} 0.56/0.56 0.56/0.56 12.6/11.9 0.079/0.072 0.835/0.838 0.834 {} no: x"
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a").mkdir()
            lines = [row.format("c9_e100_s3", "no"), row.format("c10_e500_s2", "yes"), "pooled: 8 units"]
            (Path(tmp) / "a" / "holds.txt").write_text("\n".join(lines))
            self.assertEqual(PC.stage_verdicts(["a"], Path(tmp)), {"c9_e100_s3": False, "c10_e500_s2": True})
        self.assertEqual(PC.cell_of("c8_ae500_s12"), "c8_ae500")

    def test_selection_counts_the_lowest_rho1_against_a_random_pick(self):
        rows = [{"rho1": 0.3, "holds": True}, {"rho1": 0.6, "holds": False}, {"rho1": 0.5, "holds": False},
                {"rho1": 0.7, "holds": True}]
        out = PC.selection(rows, 2)
        self.assertEqual(out["subsets"], 6)
        self.assertEqual(out["with_holder"], 5)  # all but the two failures together
        self.assertEqual(out["picked"], 3)  # the .3 holder wins its three subsets; the .7 holder loses both of its others
        self.assertAlmostEqual(out["random"], 3.0)  # .5 + .5 + 1 + .5 + .5


if __name__ == "__main__":
    unittest.main()
