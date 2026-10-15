"""Bundle channels go straight to PC; the mixing null changes only the chosen basis."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval.causal import latent_causal_discovery as causal
from eval.dino.dinov3_embed_synthetic import save
from eval.metrics.bundle_identity import identity_record
from eval.metrics.identifiability_metrics import channel_mcc


def planted():
    rng = np.random.default_rng(31)
    z = rng.normal(size=(600, 3))
    z[:, 1] += 0.9 * z[:, 0]
    z[:, 2] += 0.9 * z[:, 1]
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    x = np.column_stack([z[:, [1, 0]], rng.normal(size=(len(z), 1)), z[:, 2]])
    return x, z, adj


def options(**overrides):
    return argparse.Namespace(
        **dict(
            dict(
                reduce="none",
                alphas=[0.01, 0.05],
                indep_test="fisherz",
                max_cond_set=None,
                null_draws=10,
                top_var_channels=None,
            ),
            **overrides,
        )
    )


def bundle(path, x, z, adj, rotated=(0, 1, 2)):
    latents = {"z_content": z}
    if adj is not None:
        latents["causal_adj"] = adj
    generator = {"synthetic_seed": 31, "synthetic_res": 16}
    meta = dict(
        content_factor_names=["first", "second", "lesion"],
        rotated_channels=list(rotated),
        **identity_record(latents, generator, len(z), "match"),
    )
    save(path, {1: x, 2: x[:, ::-1]}, latents, {}, [], meta)


class MixingNullTests(unittest.TestCase):
    def setUp(self):
        self.x, self.z, self.adj = planted()
        self.cli = options()

    def test_mcc_is_the_existing_channel_metric_and_pc_sees_the_original_columns(self):
        with patch.object(causal, "discover", wraps=causal.discover) as discover:
            result = causal.evaluate_reduced(self.x, self.z, self.adj, self.cli)
        np.testing.assert_array_equal(discover.call_args.args[0], self.x)
        self.assertAlmostEqual(result["channel_mcc"], channel_mcc(self.x, self.z)["mean"])
        self.assertEqual(result["assignment"], [1, 0, 3])
        self.assertEqual(result["best"]["f1"], 1.0)

    def test_random_mixing_keeps_lesion_fixed_and_preserves_the_blocks_linear_span(self):
        before = self.x.copy()
        with patch.object(causal, "evaluate_reduced", wraps=causal.evaluate_reduced) as evaluate:
            result = causal.mixing_null(self.x, self.z, self.adj, self.cli, [0, 1, 2], draws=3, seed=7)
        np.testing.assert_array_equal(before, self.x)
        self.assertEqual(evaluate.call_count, 3)
        centered = self.x[:, :3] - self.x[:, :3].mean(0)
        for call in evaluate.call_args_list:
            mixed = call.args[0]
            np.testing.assert_array_equal(mixed[:, 3], self.x[:, 3])
            self.assertFalse(np.allclose(mixed[:, :3], self.x[:, :3]))
            reconstruction = mixed[:, :3] @ np.linalg.lstsq(mixed[:, :3], centered, rcond=None)[0]
            np.testing.assert_allclose(reconstruction, centered, atol=1e-10)
        self.assertEqual(result["columns"], [0, 1, 2])
        self.assertEqual(result["seed"], 7)
        self.assertEqual(set(result["f1_by_alpha"]), {"0.01", "0.05"})
        self.assertEqual(len(result["f1"]), 3)
        self.assertEqual(result, causal.mixing_null(self.x, self.z, self.adj, self.cli, [0, 1, 2], draws=3, seed=7))

    def test_top_variance_selection_remaps_the_mixable_indices(self):
        x = self.x * [4, 1, 0.01, 10]
        with patch.object(causal, "mixing_null", return_value=None) as mixed:
            row = causal.score_row(x, self.z, self.adj, options(top_var_channels=3), "m", [0, 1, 2], mixing=2)
        self.assertEqual(row["top_var_channels"], [3, 0, 1])
        self.assertEqual(mixed.call_args.args[4], [1, 2])
        np.testing.assert_array_equal(mixed.call_args.args[0][:, 0], x[:, 3])

    def test_zero_or_one_mixable_channel_has_no_basis_null(self):
        for columns in ([], [0]):
            self.assertIsNone(causal.mixing_null(self.x, self.z, self.adj, self.cli, columns, draws=2))

    def test_malformed_metadata_is_rejected_before_channel_selection(self):
        for columns in ([0, 0], [-1, 0], [0, 4], [0.5, 1], [[0, 1]], [True, False]):
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, "rotated_channels"):
                causal.score_row(self.x, self.z, self.adj, options(top_var_channels=3), "m", columns, mixing=2)


class BundleCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.x, self.z, self.adj = planted()
        self.path = self.root / "main.npz"
        bundle(self.path, self.x, self.z, self.adj)

    def run_cli(self, *args):
        out = self.root / "report.json"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = causal.main(
                ["--bundles", f"m={self.path}", "--alphas", "0.05", "--null-draws", "10", "--out", str(out), *args]
            )
        self.assertEqual(code, 0)
        return json.loads(out.read_text()), stream.getvalue()

    def test_bundles_floor_ceiling_and_mixing_work_without_an_encoder_or_readout(self):
        floor = self.root / "floor.npz"
        bundle(floor, np.random.default_rng(1).normal(size=self.x.shape), self.z, self.adj)
        with patch.object(causal, "evaluate_run", side_effect=AssertionError("must not load a run")), patch.object(
            causal, "reduce_channels", side_effect=AssertionError("must not reduce bundle columns")
        ):
            report, text = self.run_cli(
                "--bundle-floors", f"m={floor}", "--ceiling", "--mixing-null", "2", "--reduce", "std"
            )
        rows = {r["label"]: r for r in report["runs"]}
        self.assertEqual(set(rows), {"m", "m-floor", "truth-ceiling"})
        self.assertEqual(rows["m"]["reduce"], "none")
        self.assertEqual(rows["truth-ceiling"]["reduce"], "none")
        self.assertEqual(rows["m"]["n_channels"], 4)
        self.assertEqual(rows["m"]["mixing_null"]["columns"], [0, 1, 2])
        self.assertNotIn("mixing_null", rows["m-floor"])
        self.assertNotIn("mixing_null", rows["truth-ceiling"])
        self.assertEqual(rows["truth-ceiling"]["channel_mcc"], 1.0)
        self.assertEqual(report["protocol"]["readout"], "none")
        for phrase in ("mix F1", "mix MCC", "best-F1 alpha", "lesion 1.00"):
            self.assertIn(phrase, text)

    def test_bundle_view_two_reaches_pc(self):
        report, _ = self.run_cli("--bundle-view", "2")
        row = report["runs"][0]
        self.assertEqual(row["assignment"], [2, 3, 0])
        self.assertEqual(row["bundle_view"], "2")

    def test_mismatched_rows_and_floors_are_refused(self):
        other = self.root / "other.npz"
        bundle(other, self.x, self.z[::-1], self.adj)
        for args in (("--bundles", f"m={self.path}", f"other={other}"), ("--bundle-floors", f"m={other}")):
            with self.subTest(args=args), self.assertRaisesRegex(SystemExit, "not row-aligned"):
                self.run_cli(*args)

    def test_explicit_row_mismatch_override_warns_about_the_single_ceiling(self):
        other = self.root / "other.npz"
        bundle(other, self.x[::-1], self.z[::-1], self.adj)
        with self.assertLogs(causal.logger, level="WARNING") as logs:
            report, _ = self.run_cli(
                "--bundles", f"m={self.path}", f"other={other}", "--allow-row-mismatch", "--ceiling"
            )
        self.assertEqual(len(report["runs"]), 3)
        self.assertIn("different subjects", " ".join(logs.output))

    def test_label_collisions_cannot_hide_a_mismatched_floor(self):
        for args in (
            ("--bundles", f"m={self.path}", f"m-floor={self.path}", "--bundle-floors", f"m={self.path}"),
            ("--bundles", f"truth-ceiling={self.path}", "--ceiling"),
            ("--bundle-floors", f"missing={self.path}"),
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli(*args)

    def test_noncausal_bundle_is_explicitly_skipped(self):
        bundle(self.path, self.x, self.z, None)
        report, text = self.run_cli("--ceiling", "--mixing-null", "2")
        self.assertEqual(len(report["runs"]), 1)
        self.assertEqual(report["runs"][0]["status"], "skipped")
        self.assertIn("no causal_adj", text)


if __name__ == "__main__":
    unittest.main()
