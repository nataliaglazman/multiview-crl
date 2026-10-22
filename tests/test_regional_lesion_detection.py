"""eval.encoder.regional_lesion_detection: the toy detector, and a real run's cells checked against its images."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.encoder import regional_lesion_detection as detection

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from eval.encoder.lesion_side_probe import lesion_centres  # noqa: E402
from eval.protocol.score_checkpoint import load_settings, make_dataset  # noqa: E402
from training import main_conv_synthetic as trainer  # noqa: E402

TINY = ["--res", "32", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--content-channels", "9", "--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1"]
TINY += ["--best-metric", "none", "--batch-size", "2", "--lr", "0.05", "--log-every", "1", "--device", "cpu"]
TINY += ["--num-train-samples", "8", "--num-val-samples", "64", "--no-cache", "--require-new-run"]
TINY += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
TINY += ["--synthetic-lesion-placement", "wm_interior", "--train-steps", "1", "--eval-every", "1"]
TINY += ["--patch-loss-weight", "1", "--train-patch-grid", "4", "4", "4"]


def train(out_dir, model_id):
    argv = ["trainer", "--out-dir", str(out_dir), "--model-id", model_id, *TINY]
    with patch.object(sys, "argv", argv), patch.object(trainer, "evaluate", return_value=({}, {})):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
    return Path(out_dir) / model_id


class RegionalLesionDetectionTests(unittest.TestCase):
    def test_toy_detector(self):
        with contextlib.redirect_stdout(io.StringIO()):
            detection.self_test()

    def test_a_run_is_scored_against_its_own_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = train(tmp, "run")
            args = detection.parse_args(["--run-dirs", str(run), "--device", "cpu", "--batch-size", "8"])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = detection.run(args)
            cfg = load_settings(run)
            dataset = make_dataset(cfg, cfg["num_val_samples"])
            fractions = detection.lesion_fractions(dataset, [4, 4, 4])
            centres = lesion_centres(dataset)
            flair = []
            for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
                x = torch.cat(batch["image"], dim=0)
                flair.append(detection.image_cells(x, [4, 4, 4])[x.shape[0] // 2 :].transpose(1, 2).numpy())
            flair = np.concatenate(flair)
            aligned = detection.detect(flair, fractions)
            shuffled = detection.detect(flair, fractions[np.random.default_rng(0).permutation(len(fractions))])

        # The rendered centre's cell holds lesion: the cells and the lesion masks share one x-major layout.
        cell = (centres // (cfg["res"] / 4)).astype(int)
        index = cell[:, 0] * 16 + cell[:, 1] * 4 + cell[:, 2]
        self.assertTrue((fractions[np.arange(len(index)), index] > 0).all())
        scores = report["data_sets"][0]["scores"]
        self.assertEqual(set(scores), {"t1", "flair"})
        expected = {"image cells", "A init regional", "A init map", "A regional", "A map"}
        self.assertEqual(set(scores["flair"]), expected)
        # The FLAIR lesion is the brightest thing in its cell, so the image alone finds it; giving every subject
        # another subject's lesion map must lose that, or the labels were never tied to the images.
        image = scores["flair"]["image cells"]
        self.assertAlmostEqual(image["auc"], aligned["auc"], places=10)
        self.assertGreater(aligned["auc"], shuffled["auc"] + 0.15)
        self.assertGreater(aligned["exact"], 3 * aligned["chance_exact"])
        self.assertLess(shuffled["exact"], 2 * shuffled["chance_exact"])
        for row in scores["t1"].values():
            self.assertTrue(0 <= row["hit"] <= 1 and np.isfinite(row["auc"]))
        self.assertIn("lesion-cell detection", output.getvalue())


if __name__ == "__main__":
    unittest.main()
