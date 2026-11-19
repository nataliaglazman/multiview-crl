"""A fixed, matched cohort through the tabular benchmark's public entry point."""

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.causal import tabular_benchmark
from eval.causal.effect_estimation import estimate_cate, make_folds, proximal_ate
from eval.causal.tabular_scm import simulate_tabular


def cohort():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(1500, 3))
    z[:, 1] += 0.7 * z[:, 0]
    z[:, 2] += 0.7 * z[:, 1]
    adjacency = np.eye(3, k=1, dtype=bool)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return z, adjacency, z @ rotation, rng.normal(size=z.shape)


def own_noise():
    return np.random.default_rng(12).normal(size=(1500, 3))


def bundle(path, z, adjacency, features, own=None):
    extra = {} if own is None else {"z_own": own}
    np.savez(path, z_content=z, causal_adj=adjacency, emb_view1=features, emb_view2=features[:, ::-1], **extra)


class TabularBenchmarkTests(unittest.TestCase):
    def test_potential_outcomes_on_fixed_rows(self):
        z, adjacency, _, _ = cohort()
        args = dict(seed=2, effect=-0.75, heterogeneity=0.4, image_adjacency=adjacency)
        scm = simulate_tabular(z, **args)
        self.assertEqual(scm.covariates.shape, (len(z), 3))
        self.assertAlmostEqual(scm.ate, -0.75)
        self.assertGreater(np.std(scm.y1 - scm.y0), 0.3)
        self.assertTrue(np.all((scm.propensity >= 0.1) & (scm.propensity <= 0.9)))
        np.testing.assert_array_equal(scm.outcome, np.where(scm.treatment, scm.y1, scm.y0))
        np.testing.assert_array_equal(scm.outcome, simulate_tabular(z, **args).outcome)
        np.testing.assert_array_equal(scm.adjacency[:3, :3], adjacency)
        self.assertTrue(scm.effect_adjacency[:3, -2:].all())

    def test_discovery_edges_survive_conditioning_on_every_other_column(self):
        z, adjacency, _, _ = cohort()
        for seed in range(6):
            scm = simulate_tabular(z, seed=seed, image_adjacency=adjacency)
            truth = scm.adjacency | scm.adjacency.T
            self.assertEqual(scm.adjacency[:3, -2].sum(), 1)
            self.assertEqual(scm.adjacency[:3, -1].sum(), 1)
            self.assertFalse((scm.adjacency[:-2, -2] & scm.adjacency[:-2, -1]).any())
            values = np.column_stack((z, scm.covariates, scm.discovery_treatment, scm.discovery_outcome))
            precision = np.linalg.inv(np.corrcoef(values.T))
            partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
            for column in (-2, -1):
                parents = np.flatnonzero(truth[:-2, column])
                self.assertGreater(np.abs(partial[parents, column]).min(), 0.1, (seed, column))
        null = simulate_tabular(z, seed=0, confounding=0.0, image_adjacency=adjacency)
        self.assertFalse(null.adjacency[:5, -2:].any())
        self.assertTrue(null.adjacency[5, -1] and null.adjacency[-2, -1])

    def test_tabular_causes_enter_through_own_noise_alone(self):
        z, adjacency, _, _ = cohort()
        own = own_noise()
        descendants = {0: {0, 1, 2}, 1: {1, 2}, 2: {2}}
        for seed in range(6):
            plain = simulate_tabular(z, seed=seed, image_adjacency=adjacency)
            scm = simulate_tabular(z, seed=seed, image_adjacency=adjacency, own_noise=own)
            for field in (
                "outcome",
                "propensity",
                "y1",
                "discovery_treatment",
                "discovery_outcome",
                "effect_adjacency",
            ):
                np.testing.assert_array_equal(getattr(scm, field), getattr(plain, field))
            self.assertEqual(plain.causes.shape, (len(z), 0))
            self.assertEqual(scm.causes.shape, (len(z), 2))
            self.assertEqual(len(set(scm.cause_targets)), 2)
            self.assertEqual(scm.adjacency.shape, (10, 10))
            np.testing.assert_array_equal(scm.adjacency[np.ix_([*range(6), 8, 9], [*range(6), 8, 9])], plain.adjacency)
            values = np.column_stack((z, scm.covariates, scm.causes, scm.discovery_treatment, scm.discovery_outcome))
            precision = np.linalg.inv(np.corrcoef(values.T))
            partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
            for i, (target, strength) in enumerate(zip(scm.cause_targets, scm.cause_strengths)):
                cause = 6 + i
                self.assertEqual(np.flatnonzero(scm.adjacency[cause]).tolist(), [target])
                self.assertFalse(scm.adjacency[:, cause].any())
                self.assertTrue(0.5 <= abs(strength) <= 0.8)
                self.assertGreater(abs(partial[cause, target]), 0.1, (seed, i))
                self.assertGreater(abs(np.corrcoef(scm.causes[:, i], own[:, target])[0, 1]), 0.45)
                for other in set(range(3)) - descendants[target]:
                    self.assertLess(abs(np.corrcoef(scm.causes[:, i], z[:, other])[0, 1]), 0.1, (seed, i, other))
        null = simulate_tabular(z, seed=0, confounding=0.0, image_adjacency=adjacency, own_noise=own)
        self.assertFalse(null.adjacency[6:8].any())
        with self.assertRaisesRegex(ValueError, "own_noise"):
            simulate_tabular(z, own_noise=own[:, :2])

    def test_rejects_reordered_floor(self):
        z, adjacency, dense, floor = cohort()
        with tempfile.TemporaryDirectory() as scratch:
            a, b = Path(scratch) / "dense.npz", Path(scratch) / "floor.npz"
            bundle(a, z, adjacency, dense)
            bundle(b, z[::-1], adjacency, floor[::-1])
            with self.assertRaisesRegex(ValueError, "not row-aligned"):
                tabular_benchmark.load_arms({"dense": a}, {"dense": b}, "1", True)

    @unittest.skipUnless(importlib.util.find_spec("causallearn"), "causal-learn is needed for the graph panel")
    def test_cli_direct_adjustment_and_held_out_graph(self):
        z, adjacency, dense, floor = cohort()
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            root = Path(scratch)
            a, b, out = root / "dense.npz", root / "floor.npz", root / "report.json"
            bundle(a, z, adjacency, dense)
            bundle(b, z, adjacency, floor)
            tabular_benchmark.main(
                [
                    "--bundles",
                    f"dense={a}",
                    "--floors",
                    f"dense={b}",
                    "--view",
                    "each",
                    "--seeds",
                    "0",
                    "1",
                    "--tables-dir",
                    scratch,
                    "--out",
                    str(out),
                ]
            )
            report = json.loads(out.read_text())
            self.assertEqual(report["n_subjects"], len(z))
            self.assertEqual(len(report["discovery_held_out_rows"]), len(z) // 2)
            self.assertEqual(report["summary"]["oracle factors"]["image_tabular_f1"], 1.0)
            self.assertEqual(np.shape(report["runs"][1]["discovery_truth"]), (8, 8))
            for view in ("T1", "FLAIR"):
                key = f"dense / {view}"
                self.assertLess(report["summary"][key]["mae_delta_vs_tabular"], -0.1)
                self.assertLess(report["summary"][key]["mae_delta_vs_untrained"], -0.1)
                self.assertGreater(min(report["readout_r2"][key].values()), 0.99)
                for task in report["runs"]:
                    self.assertAlmostEqual(task["effects"][key]["ate"], task["effects"]["oracle factors"]["ate"], 6)
                    self.assertEqual(task["discovery"][key]["n_samples"], len(z) // 2)
            with (root / "observed_0.csv").open(newline="") as handle, np.load(root / "truth_0.npz") as truth:
                observed = list(csv.DictReader(handle))
                self.assertEqual(len(observed), len(z))
                self.assertNotIn("y1", observed[0])
                np.testing.assert_array_equal([int(row["row"]) for row in observed], truth["row"])
                np.testing.assert_allclose(
                    [float(row["outcome"]) for row in observed],
                    np.where([int(row["treatment"]) for row in observed], truth["y1"], truth["y0"]),
                )

    @unittest.skipUnless(importlib.util.find_spec("causallearn"), "causal-learn is needed for the graph panel")
    def test_cli_scores_edges_into_and_out_of_the_image(self):
        z, adjacency, dense, floor = cohort()
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            root = Path(scratch)
            a, b, out = root / "dense.npz", root / "floor.npz", root / "report.json"
            bundle(a, z, adjacency, dense, own_noise())
            bundle(b, z, adjacency, floor)
            args = ["--bundles", f"dense={a}", "--floors", f"dense={b}", "--seeds", "0", "1", "2", "--out", str(out)]
            tabular_benchmark.main(args)
            report = json.loads(out.read_text())
            self.assertEqual(report["discovery_nodes"][6:8], ["tabular_cause_1", "tabular_cause_2"])
            oracle = report["summary"]["oracle factors"]
            self.assertGreater(oracle["image_in_f1"], 0.9)
            self.assertGreater(oracle["image_out_f1"], 0.8)
            self.assertGreater(oracle["image_in_oriented"], 0.5)
            for run in report["runs"]:
                self.assertEqual(np.shape(run["discovery_truth"]), (10, 10))
                self.assertEqual(len(run["tabular_causes"]), 2)
                counts = run["discovery"]["oracle factors"]["orientation"]["into_image"]
                self.assertEqual((counts["true"], counts["found"]), (2, 2))
                self.assertEqual(counts["reversed"], 0)
            untrained = report["summary"]["dense / T1 (untrained)"]
            self.assertLess(untrained["image_in_f1"], 0.5)
            tabular_benchmark.main([*args, "--no-tabular-causes"])
            self.assertNotIn("image_in_f1", json.loads(out.read_text())["summary"]["oracle factors"])

    def test_proximal_ate_corrects_what_one_noisy_proxy_misses(self):
        rng = np.random.default_rng(3)
        n = 20000
        u = rng.normal(size=(n, 3))
        score = u @ np.array([0.6, -0.5, 0.4])
        score = (score - score.mean()) / score.std()
        x = rng.normal(size=(n, 2))
        a = (rng.random(n) < 1 / (1 + np.exp(-(score + 0.5 * x[:, 0])))).astype(int)
        y = a * (1.0 + 0.5 * score) + score + 0.5 * x[:, 0] + rng.normal(size=n)
        z, w = (u @ rng.normal(size=(3, 6)) + 0.8 * rng.normal(size=(n, 6)) for _ in range(2))
        naive = np.linalg.lstsq(np.column_stack((np.ones(n), a, x, w)), y, rcond=None)[0][1]
        self.assertGreater(naive - 1.0, 0.15)  # W's measurement error hides part of the confounder
        for interactions in (False, True):
            result = proximal_ate(a, y, x, z, w, interactions=interactions)
            self.assertEqual(result["components"], 3)
            self.assertLess(abs(result["ate"] - 1.0), 0.05)
            self.assertLess(result["ci_low"], 1.0)
            self.assertGreater(result["ci_high"], 1.0)
        unrelated = proximal_ate(a, y, x, rng.normal(size=(n, 6)), rng.normal(size=(n, 6)))
        self.assertEqual(unrelated["components"], 0)

    def test_a_modifier_changes_only_the_effect(self):
        z, adjacency, _, _ = cohort()
        modifier = own_noise()[:, 1]
        plain = simulate_tabular(z, seed=4, heterogeneity=0.8, image_adjacency=adjacency)
        scm = simulate_tabular(z, seed=4, heterogeneity=0.8, image_adjacency=adjacency, modifier=modifier)
        np.testing.assert_array_equal(scm.treatment, plain.treatment)
        np.testing.assert_array_equal(scm.y0, plain.y0)
        standard = (modifier - modifier.mean()) / modifier.std()
        np.testing.assert_allclose(scm.y1 - scm.y0, 1.0 + 0.8 * standard)
        with self.assertRaisesRegex(ValueError, "modifier"):
            simulate_tabular(z, modifier=np.zeros(len(z)))

    def test_individual_effects_from_the_factors_not_from_the_tabular_columns(self):
        z, adjacency, _, _ = cohort()
        scm = simulate_tabular(z, seed=1, heterogeneity=1.0, image_adjacency=adjacency)
        effect = scm.y1 - scm.y0
        folds = make_folds(scm.treatment, 5, 0)
        w = scm.covariates

        def r2(estimated):
            return 1 - np.mean((estimated - effect) ** 2) / effect.var()

        oracle = estimate_cate(scm.treatment, scm.outcome, np.column_stack((w, z)), folds, tabular_width=3)
        tabular = estimate_cate(scm.treatment, scm.outcome, w, folds, tabular_width=3)
        self.assertGreater(r2(oracle["dr_learner"]), 0.9)
        self.assertGreater(r2(oracle["t_learner"]), 0.9)
        self.assertLess(r2(tabular["dr_learner"]), 0.1)  # the effect varies with the image alone
        both = np.column_stack((scm.outcome, scm.outcome + effect))  # several outcomes share one fit
        joint = estimate_cate(scm.treatment, both, np.column_stack((w, z)), folds, tabular_width=3)
        np.testing.assert_allclose(joint["dr_learner"][:, 0], oracle["dr_learner"], atol=1e-10)
        np.testing.assert_allclose(joint["mu1"][:, 0], oracle["mu1"], atol=1e-10)

    def test_cli_counterfactual_panel(self):
        z, adjacency, dense, floor = cohort()
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()) as printed:
            root = Path(scratch)
            a, b, out = root / "dense.npz", root / "floor.npz", root / "report.json"
            bundle(a, z, adjacency, dense, own_noise())
            bundle(b, z, adjacency, floor)
            args = ["--bundles", f"dense={a}", "--floors", f"dense={b}", "--effects-only", "--seeds", "0", "1"]
            args += ["--counterfactual", "--counterfactual-modifiers", "score", "own", "--out", str(out)]
            tabular_benchmark.main(args)
            report = json.loads(out.read_text())
            panel = report["counterfactual"]
            names = report["image_factors"]
            self.assertEqual(panel["modifiers"], ["image score"] + [f"{name} own part" for name in names])
            summary = panel["summary"]
            score = {key: rows["image score"]["dr_learner_r2"] for key, rows in summary.items()}
            self.assertGreater(score["oracle factors"], 0.9)
            self.assertGreater(score["dense / T1"], 0.9)  # a rotation of the factors holds the image score
            self.assertLess(score["dense / T1 (untrained)"], 0.2)
            self.assertLess(score["tabular"], 0.1)
            self.assertGreater(summary["dense / T1"]["image score"]["dr_learner_r2_delta_vs_untrained"], 0.6)
            for label in panel["modifiers"][1:]:
                # The toy SCM is linear and own_noise() is its exogenous noise, so a rotation of the factors holds
                # each own part linearly; the untrained floor holds none.
                self.assertGreater(summary["oracle own parts"][label]["dr_learner_r2"], 0.9)
                self.assertGreater(summary["dense / T1"][label]["dr_learner_r2"], 0.85)
                self.assertLess(summary["dense / T1 (untrained)"][label]["dr_learner_r2"], 0.1)
            for run in report["runs"]:
                row = run["counterfactual"]["oracle factors"]["image score"]
                self.assertAlmostEqual(row["counterfactual_rmse"]["abducted"], row["dr_learner"]["pehe"], 10)
                self.assertGreater(row["counterfactual_rmse"]["mean_prediction"], 0.9)  # the subject's noise stays
            self.assertIn("own-part R²", printed.getvalue())
            no_own = ["--bundles", f"floor={b}", "--effects-only", "--counterfactual", "--out", str(out)]
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                tabular_benchmark.main([*no_own, "--counterfactual-modifiers", "own"])  # no bundle stores z_own
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                tabular_benchmark.main([*args, "--counterfactual-heterogeneity", "0"])

    def test_cli_proximal_arms_remove_what_one_noisy_view_leaves(self):
        z, adjacency, _, _ = cohort()
        rng = np.random.default_rng(5)
        views = [z @ rng.normal(size=(3, 6)) + rng.normal(size=(len(z), 6)) for _ in range(2)]
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            root = Path(scratch)
            a, b, out = root / "noisy.npz", root / "floor.npz", root / "report.json"
            np.savez(a, z_content=z, causal_adj=adjacency, emb_view1=views[0], emb_view2=views[1])
            np.savez(
                b,
                z_content=z,
                causal_adj=adjacency,
                emb_view1=rng.normal(size=(len(z), 6)),
                emb_view2=rng.normal(size=(len(z), 6)),
            )
            args = ["--bundles", f"noisy={a}", "--floors", f"noisy={b}", "--effects-only", "--proximal"]
            args += ["--out", str(out)]
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                tabular_benchmark.main(args)  # one view only
            tabular_benchmark.main([*args, "--view", "each"])
            summary = json.loads(out.read_text())["summary"]
            proximal = [summary[f"noisy / proximal {name}"] for name in ("T1→FLAIR", "FLAIR→T1")]
            self.assertEqual([arm["components"] for arm in proximal], [3, 3])
            self.assertEqual(summary["noisy / proximal T1→FLAIR (untrained)"]["components"], 0)
            # Over the ten tasks each view alone leaves confounding its noise hides; the proxies remove it.
            self.assertGreater(min(summary[f"noisy / {view}"]["bias"] for view in ("T1", "FLAIR")), 0.08)
            self.assertLess(max(abs(arm["bias"]) for arm in proximal), 0.04)
            self.assertLess(proximal[0]["mae_delta_vs_untrained"], -0.3)


if __name__ == "__main__":
    unittest.main()
