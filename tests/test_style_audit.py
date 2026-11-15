"""Stage 34's style audit (out/burden_feasibility_20261102/style_audit.py): targets, the verified redraw, the reading."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import style_audit as S


class FakeGenerator:
    """Content latents that depend on the bank.py setting, style latents that depend only on the subject."""

    def __init__(self, setting):
        self.offset = 10.0 * setting["causal"] + 100.0 * setting["nuisance"]

    def sample_seed_for(self, i):
        return i

    def _draw_pseudo_mri(self, seed):
        rng = np.random.default_rng(seed)
        return {
            "z_content": rng.normal(size=12) + self.offset,
            "z_style_v1": rng.normal(size=3),
            "z_style_v2": rng.normal(size=3),
        }


def fake_bank(directory, setting, n):
    generator = FakeGenerator(setting)
    content = [S.draw(generator, i)[0] for i in range(n)]
    np.save(Path(directory) / "test_zc.npy", np.stack(content))


class StyleAuditTests(unittest.TestCase):
    def test_targets_are_what_the_renderer_applies(self):
        z = np.array([[2.0, -3.0, -0.5], [0.2, 0.4, 0.7]])
        np.testing.assert_allclose(S.style_targets(z), [[1.0, -1.0, 0.5], [0.2, 0.4, 0.7]])

    def test_redraw_keeps_the_setting_that_reproduces_the_bank(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            S, "generator", lambda s, total: FakeGenerator(s)
        ):
            fake_bank(root, {"causal": False, "nuisance": False}, 9)
            setting, (t1, flair) = S.redraw(root, 6, workers=1)
        self.assertEqual(setting, {"causal": False, "nuisance": False})
        expected = [S.draw(FakeGenerator(setting), i) for i in range(6)]
        np.testing.assert_array_equal(t1, np.stack([e[1] for e in expected]))
        np.testing.assert_array_equal(flair, np.stack([e[2] for e in expected]))

    def test_redraw_refuses_a_bank_no_setting_reproduces(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            S, "generator", lambda s, total: FakeGenerator(s)
        ):
            np.save(Path(root) / "test_zc.npy", np.zeros((9, 9), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "no bank.py setting"):
                S.redraw(root, 6, workers=1)

    def test_redraw_refuses_any_subject_that_differs(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            S, "generator", lambda s, total: FakeGenerator(s)
        ):
            fake_bank(root, {"causal": True, "nuisance": False}, 9)
            stored = np.load(Path(root) / "test_zc.npy")
            stored[5, 3] += 0.01  # past the subjects that pick the setting
            np.save(Path(root) / "test_zc.npy", stored)
            with self.assertRaisesRegex(ValueError, "differ .* for 1 of 6"):
                S.redraw(root, 6, workers=1)

    def test_bank_styles_redraws_once_then_reads_the_cache(self):
        latents = [np.random.default_rng(v).normal(size=(5, 3)) for v in range(2)]
        setting = {"causal": True, "nuisance": False}
        with tempfile.TemporaryDirectory() as out, mock.patch.object(S, "redraw", return_value=(setting, latents)) as m:
            cache = Path(out) / "style_bank_5.npz"
            first = S.bank_styles("bank", 5, cache)
            second = S.bank_styles("bank", 5, cache)
        self.assertEqual(m.call_count, 1)
        for got in (first, second):
            self.assertEqual(got[0], setting)
            for view in range(2):
                np.testing.assert_allclose(got[1][view], S.style_targets(latents[view]))

    def test_reading_tells_the_own_view_from_the_null(self):
        rng = np.random.default_rng(0)
        styles = [rng.normal(size=(300, 3)) for _ in range(2)]
        feats = {
            stage: [
                np.column_stack([styles[v] + 0.05 * rng.normal(size=(300, 3)), rng.normal(size=(300, 4))])
                for v in range(2)
            ]
            for stage in ("code", "gap")
        }
        # Gradient boosting stands in as ridge: with torch loaded, its OpenMP pool can hang on macOS.
        with mock.patch.object(S.code_audit, "boosted", side_effect=S.own_variation.ridge) as boosted:
            table = S.read(feats, styles)
        self.assertEqual(boosted.call_count, 2 * 2 * len(S.STYLE))  # the code only: two views, own and other
        for view in S.VIEWS:
            self.assertEqual(set(table[view]), {"code", "code_boosted", "gap"})
            for stage, cell in table[view].items():
                self.assertTrue(all(r > 0.8 for r in cell["own"]), (view, stage, cell))
                self.assertTrue(all(r < 0.1 for r in cell["other"]), (view, stage, cell))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            S.show(
                "run",
                {
                    "step": 2,
                    "bank": "x/bank",
                    "setting": {"causal": True, "nuisance": False},
                    "read": table,
                    "floor": table,
                },
            )
        self.assertIn("bank.py --causal", printed.getvalue())
        self.assertEqual(printed.getvalue().count("code_boosted"), 4)


if __name__ == "__main__":
    unittest.main()
