"""Batch causal-panel tests, including real PC recovery on a planted chain."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
            (X * 0, z, truth),
        ]:
            with self.subTest(shape=features.shape):
                with self.assertRaises(ValueError):
                    recovery.evaluate_arrays(features, targets, adj)


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
