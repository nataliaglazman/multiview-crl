"""warm.py --tau: the global InfoNCE's temperature, the term scaled by tau over the recipe's (stage 70)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import warm

BASE = ["warm.py", "/nonexistent/bank", "OUT", "cold", "2", "1", "--pool", "tail", "--patch-weight", "0"]


class TauOptionTests(unittest.TestCase):
    def test_options_reject_what_the_temperature_cannot_use(self):
        out = "/nonexistent/stage70_out"
        cases = [
            ["--tau", "0"],
            ["--tau", "-1"],
            ["--tau", "nan"],
            ["--tau", "inf"],
            ["--tau", "1", "--pool", "gap"],
            ["--tau", "1", "--patch-weight", "1"],
            ["--tau", "1", "--contrastive-loss-type", "barlow_twins"],
            ["--tau", "1", "--queue", "64"],
            ["--tau", "1", "--hard-negatives", "1"],
            ["--tau", "1", "--supervise", "volume"],
        ]
        for extra in cases:
            argv = [out if x == "OUT" else x for x in BASE] + extra
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "--tau"):
                    warm.main()
        self.assertFalse(Path(out).exists())

    def test_the_term_is_scaled_by_tau_over_the_recipes(self):
        recipe = warm.run_args().tau
        self.assertEqual(recipe, 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            argv = [str(Path(tmp) / "run") if x == "OUT" else x for x in BASE] + ["--tau", "1", "--device", "cpu"]
            printed = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(printed):
                with self.assertRaises(Exception):  # the bank does not exist, after the temperature is set
                    warm.main()
        self.assertIn(f"global InfoNCE temperature 1.0 scaled by {1.0 / recipe}", printed.getvalue())

    def test_without_the_option_nothing_is_said_or_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [str(Path(tmp) / "run") if x == "OUT" else x for x in BASE] + ["--device", "cpu"]
            printed = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(printed):
                with self.assertRaises(Exception):
                    warm.main()
        self.assertNotIn("temperature", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
