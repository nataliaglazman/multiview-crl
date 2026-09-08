"""Batch causal-panel tests, including real PC recovery on a planted chain."""

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval import causal_factor_diagnostics as diagnostics
from eval import run_causal_recovery as recovery


class ScoringTests(unittest.TestCase):
    def test_skeleton_counts_ignore_direction_and_diagonal(self):
        truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        estimate = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
        score = recovery.skeleton_metrics(estimate, truth)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 1, 1))
        self.assertEqual(score["f1"], 0.5)
        self.assertEqual(score["skeleton_shd"], 2)
        self.assertFalse(score["exact_match"])
        self.assertTrue(recovery.skeleton_metrics(truth.T, truth)["exact_match"])
        empty = recovery.skeleton_metrics(np.zeros((3, 3)), np.zeros((3, 3)))
        self.assertEqual(empty["f1"], 0.0)
        self.assertTrue(empty["exact_match"])

    def test_real_pc_recovers_planted_chain_and_parent_only_probe_loses_residual(self):
        rng = np.random.RandomState(41)
        z = rng.randn(1800, 3)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        result = recovery.evaluate_arrays(z, z, truth)
        self.assertTrue(result["best"]["exact_match"])
        self.assertGreater(result["partial_r2_mean"], 0.99)
        self.assertEqual(len(result["alpha_sweep"]), 4)
        tied = [row for row in result["alpha_sweep"] if row["f1"] == result["best"]["f1"]]
        self.assertEqual(result["best"]["alpha"], tied[-1]["alpha"])
        # Extra independent columns avoid a singular decoded covariance matrix.
        X = np.column_stack([z[:, 0], rng.randn(len(z), 10)])
        parent_only = recovery.evaluate_arrays(X, z, truth, alphas=[0.05])
        child = parent_only["factors"][1]
        self.assertGreater(child["raw_r2"], 0.5)
        self.assertLess(child["partial_r2"], 0.1)
        self.assertEqual(parent_only["factors"][0]["gap"], 0.0)

    def test_invalid_inputs_do_not_become_graph_scores(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 5)
        z = rng.randn(40, 3)
        truth = np.zeros((3, 3))
        for features, targets, adj in [
            (X[:10], z[:10], truth),
            (X[:, :0], z, truth),
            (X, z, truth[:2, :2]),
            (X, z[:20], truth),
            (X * np.nan, z, truth),
        ]:
            with self.subTest(shape=features.shape):
                with self.assertRaises(ValueError):
                    recovery.evaluate_arrays(features, targets, adj)

    def test_complete_collapse_retains_factor_scores_when_pc_is_undefined(self):
        rng = np.random.RandomState(2)
        result = recovery.evaluate_arrays(np.zeros((80, 5)), rng.randn(80, 3), np.zeros((3, 3)))
        self.assertIsNone(result["best"])
        self.assertEqual(result["graph_status"], "unavailable")
        self.assertEqual(len(result["factors"]), 3)
        self.assertTrue(all("constant" in row["error"] for row in result["alpha_sweep"]))

    def test_planted_single_factor_loss_is_ranked_first(self):
        rng = np.random.RandomState(52)
        z = rng.randn(700, 3)
        X = np.column_stack([z, rng.randn(700, 7)])
        before = recovery.evaluate_arrays(X, z, np.zeros((3, 3)), alphas=[0.05])
        X[:, 1] = rng.randn(700)
        after = recovery.evaluate_arrays(X, z, np.zeros((3, 3)), alphas=[0.05], factor_rescue=True)
        before.update(run_dir="/early", status="ok")
        after.update(run_dir="/late", status="ok")
        rows, _ = diagnostics.build_factor_rows([before, after])
        late = sorted([r for r in rows if r["run_dir"] == "/late"], key=lambda r: r["delta_partial_r2"])
        self.assertEqual(late[0]["dim"], 1)
        self.assertLess(late[0]["delta_partial_r2"], -0.9)
        self.assertGreater(late[0]["partial_drop_share"], 0.99)
        self.assertLess(late[0]["readout_test_r2"], 0.1)
        self.assertEqual(len(after["factor_rescue"]["factors"]), 3)
        baseline = after["factor_rescue"]["baseline"]
        for rescue in after["factor_rescue"]["factors"]:
            self.assertEqual(rescue["shd_reduction"], baseline["skeleton_shd"] - rescue["skeleton_shd"])


class FactorReportTests(unittest.TestCase):
    def runs(self):
        before = dict(
            run_dir="/early",
            status="ok",
            num_samples=100,
            level=0,
            pooling=[4, 4, 4],
            causal_settings={"synthetic_seed": 42},
            true_dag=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
            true_skeleton=[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            factors=[dict(dim=d, parents=[] if d == 0 else [d - 1], raw_r2=0.8, partial_r2=0.7) for d in range(3)],
        )
        before["alpha_sweep"] = [dict(alpha=0.05, adjacency=before["true_skeleton"])]
        after = copy.deepcopy(before)
        after["run_dir"] = "/late"
        after["factors"][0]["partial_r2"] = 0.9  # Improving factors don't mask losses.
        after["factors"][1]["partial_r2"] = 0.1
        after["factors"][2]["partial_r2"] = 0.5
        after["alpha_sweep"][0]["adjacency"] = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
        return [before, after]

    def test_decline_contributions_and_incident_edge_errors(self):
        rows, reference = diagnostics.build_factor_rows(self.runs())
        self.assertEqual(reference, "/early")
        late = rows[3:]
        self.assertAlmostEqual(late[1]["partial_drop_share"], 0.75)
        self.assertAlmostEqual(late[2]["partial_drop_share"], 0.25)
        self.assertEqual(late[0]["partial_drop_share"], 0)
        self.assertAlmostEqual(sum(r["mean_partial_delta_contribution"] for r in late), -0.2)
        self.assertEqual([r["incident_shd"] for r in late], [2, 1, 1])
        self.assertEqual(late[0]["false_neighbors"], [2])
        self.assertEqual(late[0]["missing_neighbors"], [1])
        self.assertEqual(late[0]["delta_incident_shd"], 2)

    def test_metadata_mismatch_blocks_deltas_and_missing_alpha_stays_unavailable(self):
        runs = self.runs()
        runs[1]["causal_settings"]["synthetic_seed"] = 43
        rows, _ = diagnostics.build_factor_rows(runs, alpha=0.1)
        self.assertNotIn("delta_partial_r2", rows[-1])
        self.assertNotIn("incident_shd", rows[-1])
        self.assertIn("incompatible", rows[-1]["comparison_status"])

    def test_reference_selection_and_legacy_json_replay_without_evaluation(self):
        runs = self.runs()
        rows, ref = diagnostics.build_factor_rows(runs, reference_run="late")
        self.assertEqual(ref, "/late")
        self.assertAlmostEqual(rows[1]["delta_partial_r2"], 0.6)
        with self.assertRaises(ValueError):
            diagnostics.build_factor_rows(runs, reference_run="unknown")
        for run in runs:
            run.update(best=dict(f1=1.0, precision=1.0, recall=1.0, skeleton_shd=0), partial_r2_mean=0.7)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "causal_recovery.json"
            source.write_text(json.dumps(dict(protocol={"original": True}, runs=runs)))
            with patch.object(
                recovery, "evaluate_run", side_effect=AssertionError("Must not evaluate")
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recovery.main(["--from-json", str(source)]), 0)
            with (Path(tmp) / "causal_recovery_factors.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)
            self.assertIn("ventricle_size", (Path(tmp) / "causal_recovery_factors.txt").read_text())
            self.assertEqual(json.loads(source.read_text())["protocol"], {"original": True})


class BatchTests(unittest.TestCase):
    def test_file_relative_globs_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                (root / name).mkdir()
            file = root / "runs.txt"
            file.write_text("# runs\n\na\nb\na\nmissing\n")
            found = recovery.collect_runs([str(root / "[ab]")], file)
            self.assertEqual(found, [(root / name).resolve() for name in ("a", "b", "missing")])

    def test_skip_and_error_are_saved_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skip = root / "noncausal"
            skip.mkdir()
            (skip / "settings.json").write_text('{"synthetic_causal": false}')
            output = root / "output"
            with contextlib.redirect_stdout(io.StringIO()), self.assertLogs(recovery.logger, level="ERROR"):
                code = recovery.main(["--run-dirs", str(root / "missing"), str(skip), "--output-dir", str(output)])
            self.assertEqual(code, 1)
            payload = json.loads((output / "causal_recovery.json").read_text())
            self.assertEqual([row["status"] for row in payload["runs"]], ["error", "skipped"])
            with (output / "causal_recovery.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["f1"], "")

    def test_success_exports_full_sweep_and_factor_details(self):
        rng = np.random.RandomState(1)
        z = rng.randn(100, 3)
        score = recovery.evaluate_arrays(z, z, np.zeros((3, 3)), alphas=[0.05])
        score.update(status="ok", level=0, pooling=(4, 4, 4))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(recovery, "evaluate_run", return_value=score), contextlib.redirect_stdout(io.StringIO()):
                code = recovery.main(["--run-dirs", tmp, "--output-dir", tmp])
            self.assertEqual(code, 0)
            result = json.loads((Path(tmp) / "causal_recovery.json").read_text())["runs"][0]
            self.assertEqual(len(result["factors"]), 3)
            self.assertEqual(result["alpha_sweep"][0]["alpha"], 0.05)


class ExtractionTests(unittest.TestCase):
    def test_view_channel_alignment_fixed_mask_and_hook_cleanup(self):
        import torch

        class ToyEncoder(torch.nn.Module):
            def __init__(self, separate, fail=False):
                super().__init__()
                self.encoders = torch.nn.ModuleList([torch.nn.Identity()])
                self.encoders_v1 = torch.nn.ModuleList([torch.nn.Identity()]) if separate else None
                self.separate_encoders = separate
                self.calls = 0
                self.fail = fail

            def forward(self, x, **kwargs):
                self.calls += 1
                if self.fail and self.calls > 1:
                    raise RuntimeError("injected failure")
                if self.separate_encoders:
                    self.encoders[0](x[: len(x) // 2])
                    self.encoders_v1[0](x[len(x) // 2 :])
                else:
                    self.encoders[0](x)
                # Later batches deliberately change the mask. Sample 0 defines it.
                mask = torch.tensor([1.0, 0.0]) if self.calls == 1 else torch.tensor([0.0, 1.0])
                return (None,) * 6 + ({0: (mask, mask)},)

        dataset = [
            dict(
                image=[torch.full((2, 2, 2, 2), float(i)), torch.full((2, 2, 2, 2), 99.0)],
                gt_latents=dict(z_content=torch.tensor([float(i), float(i + 1)])),
            )
            for i in range(5)
        ]
        for separate in (False, True):
            for pooling in ("gap", (2, 2, 2)):
                model = ToyEncoder(separate)
                X, z = recovery.extract_content(model, dataset, "cpu", 0, pooling, 2, 0)
                np.testing.assert_array_equal(X[:, 0], np.arange(5))
                np.testing.assert_array_equal(z[:, 0], X[:, 0])
                self.assertEqual(X.shape[1], 1 if pooling == "gap" else 8)
                self.assertEqual(len(model.encoders[0]._forward_hooks), 0)
        model = ToyEncoder(False, fail=True)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            recovery.extract_content(model, dataset, "cpu", 0, "gap", 2, 0)
        self.assertEqual(len(model.encoders[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
