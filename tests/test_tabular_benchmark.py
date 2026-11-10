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
from eval.causal.tabular_scm import simulate_tabular


def cohort():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(1500, 3))
    z[:, 1] += 0.7 * z[:, 0]
    z[:, 2] += 0.7 * z[:, 1]
    adjacency = np.eye(3, k=1, dtype=bool)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return z, adjacency, z @ rotation, rng.normal(size=z.shape)


def bundle(path, z, adjacency, features):
    np.savez(path, z_content=z, causal_adj=adjacency, emb_view1=features, emb_view2=features[:, ::-1])


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


if __name__ == "__main__":
    unittest.main()
