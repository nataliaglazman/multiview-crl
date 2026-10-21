"""eval.encoder.lesion_side_probe: the toy probes, and a real run scored against the training log's own probe."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.encoder import lesion_side_probe as probe

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
import eval.metrics.dci as dci  # noqa: E402
from eval.lesion.lesion_detector_audit import true_centroid  # noqa: E402
from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset  # noqa: E402
from training import main_conv_synthetic as trainer  # noqa: E402

TINY = ["--res", "32", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--content-channels", "9", "--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1"]
TINY += ["--best-metric", "none", "--batch-size", "2", "--lr", "0.05", "--log-every", "1", "--device", "cpu"]
TINY += ["--num-train-samples", "8", "--num-val-samples", "64", "--no-cache", "--require-new-run"]
TINY += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
TINY += ["--synthetic-lesion-placement", "wm_interior", "--train-steps", "1", "--eval-every", "1"]


def flatten(scores, prefix=""):
    """A probe's nested score dict as (dotted key, value) pairs."""
    for key, value in scores.items():
        if isinstance(value, dict):
            yield from flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def train(out_dir, model_id):
    argv = ["trainer", "--out-dir", str(out_dir), "--model-id", model_id, *TINY]
    with patch.object(sys, "argv", argv), patch.object(trainer, "evaluate", return_value=({}, {})):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
    return Path(out_dir) / model_id


class LesionSideProbeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def test_toy_codes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            probe.self_test()

    def test_a_run_is_scored_with_the_logs_probe_on_its_own_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = train(tmp, "side")
            args = probe.parse_args(["--run-dirs", str(run), "--device", "cpu", "--batch-size", "8"])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = probe.run(args)
            cfg = load_settings(run)
            model = build_model(cfg, "cpu", torch.load(run / "model.pt", weights_only=True))
            dataset = make_dataset(cfg, cfg["num_val_samples"])
            level, factors, _, _ = dci._extract_synthetic_representations(model, dataset, "cpu", batch_size=8)
            centres = probe.lesion_centres(dataset)
            reference = [true_centroid(dataset, i) for i in range(3)]

        self.assertEqual([c["label"] for c in report["columns"]], ["A model_init.pt", "A model.pt"])
        data = report["data_sets"][0]
        self.assertEqual((data["subjects"], data["without_lesion"]), (64, 0))
        self.assertTrue(0 < data["right_share"] < 1)
        # The training log's per-factor ridge (content→content is T1, content_v2→content FLAIR).
        trained = report["columns"][1]["scores"]
        for view, block in (("t1", level[0][0]), ("flair", level[0][2])):
            _, per_factor = dci._ridge_informativeness(block, factors)
            for j, name in enumerate(probe.LESION, start=2):
                with self.subTest(view=view, factor=name):
                    self.assertAlmostEqual(trained[view]["all_subjects"][name], per_factor[j], places=10)
        np.testing.assert_allclose(centres[:3], np.array(reference), atol=1e-9)
        self.assertIn("side accuracy", output.getvalue())
        self.assertIn("centre z R², within hemisphere", output.getvalue())
        for key in ("centre_y_within_hemisphere", "centre_z_within_hemisphere"):
            self.assertTrue(np.isfinite(trained["flair"][key]["mean"]), key)
        # The anatomy column is the true anatomy latents, lesion_x/y/z removed, through the same probes.
        side = probe.hemisphere(centres, cfg["res"])
        expected = dict(flatten(probe.score_code(probe.anatomy_latents(factors), factors[:, 2:5], centres, side)))
        got = dict(flatten(data["anatomy_only"]))
        self.assertEqual(got.keys(), expected.keys())
        np.testing.assert_allclose(list(got.values()), list(expected.values()), rtol=0, atol=1e-10, equal_nan=True)
        self.assertIn("anatomy only", output.getvalue())

    def test_a_run_without_one_lesion_position_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "burden"
            run.mkdir()
            (run / "settings.json").write_text(
                json.dumps({"dataset_name": "synthetic", "n_content": 9, "synthetic_lesion_target": "burden"})
            )
            with self.assertRaises(SystemExit) as raised:
                probe.run(probe.parse_args(["--run-dirs", str(run), "--device", "cpu"]))
        self.assertIn("not one lesion's position", str(raised.exception.code))


if __name__ == "__main__":
    unittest.main()
