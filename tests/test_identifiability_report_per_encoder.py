"""--per-encoder plumbing: encoder 2's blocks and view-2 style labels reach the scorers.

The scoring semantics (own PCA, z_style_v2 targets, encoder 1 unchanged) are asserted on
planted data by ``eval.protocol.identifiability_report --self-test``; these tests cover the paths
around it: extraction -> score_run / score_live, the CLI flag, warnings, and JSON replay.
"""

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval.protocol import identifiability_report as report
from eval.protocol import run_dci_compare as compare
from eval.protocol import run_dci_synthetic as synthetic

N = 120
NAMES, STYLE_NAMES = ["brain_size", "ventricle_size"], ["bias"]


def _planted(with_v2=True, with_level1=False):
    rng = np.random.RandomState(5)
    gt, gs1, gs2 = rng.randn(N, 2), rng.randn(N, 1), rng.randn(N, 1)

    def enc(z):
        return z @ rng.randn(z.shape[1], 4) + 0.05 * rng.randn(N, 4)

    info = {"content_names": NAMES, "style_names": STYLE_NAMES}
    # Each encoder's style block encodes ITS view's style factor only, so scoring encoder 2
    # against view 1's labels would read ~0 instead of ~1.
    blocks = (enc(gt), enc(gs1), enc(gt), enc(gs2)) if with_v2 else (enc(gt), enc(gs1), None, None)
    levels = {0: (*blocks, info)}
    if with_level1:
        # A coarser level with no content mask: all channels are content, no style block.
        levels[1] = (enc(gt), None, enc(gt) if with_v2 else None, None, info)
    extractor = types.SimpleNamespace(_extract_synthetic_representations=lambda *a, **kw: (levels, gt, gs1, gs2))
    return extractor


def _score_run(run_args, with_v2=True, **kw):
    common = dict(
        run_dir="unused",
        dataset=types.SimpleNamespace(),
        poolings=[("gap", "gap")],
        level=0,
        seeds=(0,),
        n_null=1,
        batch_size=8,
        num_workers=0,
        device="cpu",
        probe_dim=0,
        per_encoder=True,
    )
    common.update(kw)
    with (
        patch.dict("sys.modules", {"eval.metrics.dci": _planted(with_v2)}),
        patch.object(compare, "_resolve_checkpoint", return_value="unused"),
        patch.object(synthetic, "load_model_from_run_dir", return_value=(object(), run_args, "cpu")),
    ):
        return report.score_run(**common)


def _style_r2(res):
    return res["leakage"]["cells"]["style→style"]["bias"]["r2"]


SEPARATE = types.SimpleNamespace(separate_encoders=True, mask_mode="fixed")


class ScoreRunTests(unittest.TestCase):
    def test_encoder2_is_scored_against_view2_style(self):
        res = _score_run(SEPARATE)
        enc2 = res["encoder2"]
        self.assertEqual((res["encoder"], enc2["encoder"]), (1, 2))
        self.assertGreater(_style_r2(res), 0.9)
        self.assertGreater(_style_r2(enc2), 0.9)
        self.assertGreater(enc2["per_factor"]["brain_size"]["r2"], 0.9)
        self.assertTrue(res["leakage"]["view"])
        self.assertEqual(enc2["leakage"]["view"], {})
        self.assertTrue(enc2["name"].endswith("[enc2]"))

    def test_encoder1_is_unchanged_by_per_encoder(self):
        both, only1 = _score_run(SEPARATE), _score_run(SEPARATE, per_encoder=False)
        self.assertNotIn("encoder2", only1)
        for key in ("per_factor", "mcc", "leakage"):
            self.assertEqual(json.dumps(both[key], sort_keys=True), json.dumps(only1[key], sort_keys=True))

    def test_missing_view2_is_logged_not_raised(self):
        with self.assertLogs(report.logger, "WARNING") as logs:
            res = _score_run(SEPARATE, with_v2=False)
        self.assertNotIn("encoder2", res)
        self.assertIn("no view-2 features", "\n".join(logs.output))

    def test_warns_once_on_shared_encoder_and_learned_mask(self):
        cases = {
            "shares ONE encoder": types.SimpleNamespace(separate_encoders=False),
            "mis-split": types.SimpleNamespace(separate_encoders=True, mask_mode="learned"),
        }
        for message, run_args in cases.items():
            with self.subTest(message), self.assertLogs(report.logger, "WARNING") as logs:
                _score_run(run_args)
            self.assertIn(message, "\n".join(logs.output))
            # The floor twins load the same run; they must not repeat the warning.
            with self.assertNoLogs(report.logger, "WARNING"):
                _score_run(run_args, random_init=True)
        with self.assertNoLogs(report.logger, "WARNING"):
            _score_run(SEPARATE)


class ScoreLiveTests(unittest.TestCase):
    def test_live_probe_scores_both_encoders_and_tags_encoder2(self):
        with patch.dict("sys.modules", {"eval.metrics.dci": _planted()}):
            kw = dict(dataset=None, device="cpu", poolings=[("gap", "gap")], seeds=(0,), n_null=1, probe_dim=0)
            res = report.score_live(object(), per_encoder=True, **kw)
            only1 = report.score_live(object(), **kw)
        self.assertGreater(_style_r2(res["encoder2"]), 0.9)
        tags = report.live_metrics(res)
        self.assertIn("iid_enc2_r2/bias/gap/from_style", tags)
        self.assertIn("iid_enc2_mcc/gap", tags)
        self.assertFalse(any(k.startswith("iid_enc2_view") for k in tags))
        self.assertEqual({k: v for k, v in tags.items() if not k.startswith("iid_enc2_")}, report.live_metrics(only1))


class ScoreLiveLevelsTests(unittest.TestCase):
    def test_every_level_scored_from_one_pass_and_level0_unchanged(self):
        planted = _planted(with_level1=True)
        calls = []
        extract = planted._extract_synthetic_representations
        planted._extract_synthetic_representations = lambda *a, **kw: calls.append(1) or extract(*a, **kw)
        kw = dict(dataset=None, device="cpu", poolings=[("gap", "gap")], seeds=(0,), n_null=1, probe_dim=0)
        with patch.dict("sys.modules", {"eval.metrics.dci": planted}):
            res = report.score_live_levels(object(), **kw)
            self.assertEqual(len(calls), 1)
            only0 = report.score_live(object(), **kw)
        self.assertEqual(sorted(res), [0, 1])
        self.assertEqual(report.live_metrics(res[0]), report.live_metrics(only0))
        tags1 = report.live_metrics(res[1], prefix=report.live_level_prefix(1))
        self.assertIn("iid_l1_r2/brain_size/gap/from_content", tags1)
        self.assertFalse(any(k.endswith("from_style") for k in tags1))
        self.assertEqual(report.live_level_prefix(0), "iid")

    def test_missing_level_raises(self):
        kw = dict(dataset=None, device="cpu", poolings=[("gap", "gap")], seeds=(0,), n_null=1, probe_dim=0)
        with patch.dict("sys.modules", {"eval.metrics.dci": _planted()}), self.assertRaises(RuntimeError):
            report.score_live_levels(object(), levels=[2], **kw)


class CliTests(unittest.TestCase):
    def invoke(self, flags):
        with (
            patch("sys.argv", ["report", "--run-dir", "unused", *flags]),
            patch.object(synthetic, "load_run_args", return_value=object()),
            patch.object(synthetic, "build_synthetic_test_set", return_value=None),
            patch.object(report, "score_run", return_value={}) as score,
            patch.object(report, "print_report"),
        ):
            report.main()
        return score

    def test_flag_reaches_checkpoint_and_every_floor_draw(self):
        score = self.invoke(["--per-encoder", "--floor-seeds", "2"])
        self.assertEqual(score.call_count, 3)
        self.assertTrue(all(call.kwargs["per_encoder"] for call in score.call_args_list))
        self.assertFalse(self.invoke(["--no-floor"]).call_args.kwargs["per_encoder"])

    def test_json_replay_prints_both_encoders_without_fits(self):
        res = _score_run(SEPARATE)
        floor = _score_run(SEPARATE, random_init=True)
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "in.json", Path(directory) / "out.json"
            report.write_report_json(str(source), res, floor)
            self.assertIn("per_factor_decoding_encoder2", json.loads(source.read_text()))
            output = io.StringIO()
            with (
                patch("sys.argv", ["report", "--from-json", str(source), "--out", str(dest)]),
                patch.object(report, "score_run", side_effect=AssertionError("no model work during replay")),
                patch.object(report, "cv_probe_r2", side_effect=AssertionError("no probe fits during replay")),
                contextlib.redirect_stdout(output),
            ):
                report.main()
            text = output.getvalue()
            saved = json.loads(dest.read_text())
        self.assertEqual(text.count("IDENTIFIABILITY REPORT"), 2)
        self.assertIn("ENCODER 2 of 2", text)
        self.assertIn("6. ENCODER 1 vs ENCODER 2", text)
        self.assertIn("printed once, in", text)
        self.assertIn("Both representation blocks are from view 2", text)
        rows = {(r["target"], r["factor"], r["pooling"]): r for r in saved["per_factor_decoding_encoder2"]}
        self.assertGreater(rows["style", "bias", "gap"]["style_r2"], 0.9)


if __name__ == "__main__":
    unittest.main()
