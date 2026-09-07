"""Per-factor report regression tests, using planted information in the STYLE block."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval import identifiability_report as report


class PerFactorDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(19)
        n = 240
        gt = rng.randn(n, 2)
        bias = rng.randn(n, 1)
        # Content encodes brain size only. GAP style encodes ventricle size and bias;
        # patch style is noise, so a pooling mix-up must fail the recovery test.
        content = gt[:, :1] + 0.02 * rng.randn(n, 1)
        style = np.hstack([gt[:, 1:], bias]) + 0.02 * rng.randn(n, 2)
        noise = rng.randn(n, 2)

        def score(c, s):
            reprs = {
                "gap": {0: (c, s, None, None, {})},
                "patch": {0: (c, noise, None, None, {})},
            }
            seeds = (0,)
            null_rng = np.random.RandomState(0)
            names = ["brain_size", "ventricle_size"]
            return {
                "name": "planted",
                "n_samples": n,
                "level": 0,
                "poolings": "gap,2x2x2",
                "probe_dim": 0,
                "per_factor": report.per_factor_scores(reprs, 0, gt, names, seeds, 1, null_rng),
                "leakage": report.leakage_scores(reprs, 0, gt, bias, names, ["bias"], seeds, 1, null_rng),
                "mcc": {},
                "mcc_per_factor_pooling": "patch",
            }

        cls.result = score(content, style)
        cls.floor = score(rng.randn(n, 1), rng.randn(n, 2))

    def test_ventricle_recovery_uses_style_and_correct_pooling(self):
        with patch.object(report, "cv_probe_r2", side_effect=AssertionError("must reuse fitted scores")):
            rows = report.per_factor_decoding_rows(self.result, self.floor)
        lookup = {(r["target"], r["factor"], r["pooling"]): r for r in rows}
        ventricle = lookup["content", "ventricle_size", "gap"]
        self.assertGreater(ventricle["style_r2"], 0.98)
        self.assertLess(ventricle["content_r2"], 0.1)
        self.assertGreater(ventricle["style_learned"], 0.8)
        self.assertLess(lookup["content", "ventricle_size", "patch"]["style_r2"], 0.1)
        self.assertGreater(lookup["content", "brain_size", "gap"]["content_r2"], 0.98)
        self.assertGreater(lookup["style", "bias", "gap"]["style_r2"], 0.98)
        self.assertLess(lookup["style", "bias", "gap"]["content_r2"], 0.1)
        expected = (
            self.result["leakage"]["cells"]["style→content"]["ventricle_size"]["r2"]
            - self.floor["leakage"]["cells"]["style→content"]["ventricle_size"]["r2"]
        )
        self.assertAlmostEqual(ventricle["style_learned"], expected)

    def test_missing_floor_or_style_stays_missing(self):
        no_style = {"per_factor": self.result["per_factor"], "leakage": {"cells": {}}}
        rows = report.per_factor_decoding_rows(no_style)
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(np.isnan(row["style_r2"]))
            self.assertTrue(np.isnan(row["style_learned"]))
            self.assertTrue(np.isnan(row["content_learned"]))

    def test_legacy_assigned_score_is_not_reused_at_another_pooling(self):
        current = {"per_factor": {"ventricle_size": self.result["per_factor"]["ventricle_size"]}}
        floor = {"per_factor": {"ventricle_size": {"pooling": "gap", "r2": 0.2, "r2_raw": 0.1}}}
        rows = {r["pooling"]: r for r in report.per_factor_decoding_rows(current, floor)}
        self.assertTrue(np.isfinite(rows["gap"]["content_learned"]))
        self.assertTrue(np.isnan(rows["patch"]["content_learned"]))

    def test_saved_report_replays_without_model_or_probe_work(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.json"
            new = Path(directory) / "new.json"
            old.write_text(json.dumps({"run": self.result, "floor": self.floor, "floor_std": None}))
            output = io.StringIO()
            with (
                patch("sys.argv", ["identifiability_report", "--from-json", str(old), "--out", str(new)]),
                patch.object(report, "score_run", side_effect=AssertionError("no model work during replay")),
                patch.object(report, "cv_probe_r2", side_effect=AssertionError("no probe fits during replay")),
                contextlib.redirect_stdout(output),
            ):
                report.main()
            text = output.getvalue()
            self.assertIn("PER-FACTOR DECODING FROM CONTENT AND STYLE", text)
            self.assertIn("from content", text)
            self.assertIn("from style", text)
            self.assertIn("ventricle_size", text)
            self.assertIn("STYLE FACTORS (prediction targets)", text)
            saved = json.loads(new.read_text())
            self.assertEqual(saved["run"], self.result)
            self.assertEqual(saved["floor"], self.floor)
            self.assertEqual(len(saved["per_factor_decoding"]), 6)


if __name__ == "__main__":
    unittest.main()
