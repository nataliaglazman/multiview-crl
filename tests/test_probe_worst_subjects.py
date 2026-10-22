"""eval.encoder.probe_worst_subjects: ranking on toy data, and a real run's worst subjects written to NIfTI."""

import contextlib
import csv
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from eval.encoder import probe_worst_subjects as worst

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


def train(out_dir, model_id):
    argv = ["trainer", "--out-dir", str(out_dir), "--model-id", model_id, *TINY]
    with patch.object(sys, "argv", argv), patch.object(trainer, "evaluate", return_value=({}, {})):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
    return Path(out_dir) / model_id


class ProbeWorstSubjectsTests(unittest.TestCase):
    def test_toy_ranking(self):
        with contextlib.redirect_stdout(io.StringIO()):
            worst.self_test()

    def test_a_run_writes_its_worst_and_best_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = train(tmp, "run")
            out = Path(tmp) / "out"
            argv = ["--run-dir", str(run), "--target", "lesion y", "--readout", "regional", "--view", "flair"]
            argv += ["--worst", "2", "--best", "1", "--probe-grid", "2", "2", "2", "--batch-size", "8"]
            args = worst.parse_args([*argv, "--device", "cpu", "--out", str(out)])
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = worst.run(args)
            cfg = load_settings(run)
            dataset = make_dataset(cfg, cfg["num_val_samples"])
            centres = lesion_centres(dataset)
            with open(out / "subjects.csv") as fp:
                rows = list(csv.DictReader(fp))
            res = cfg["res"]
            for subject in manifest["subjects"]:
                files = {name: out / path for name, path in subject["files"].items()}
                self.assertEqual(set(files), set(worst.VOLUMES))
                t1 = nib.load(files["t1"]).get_fdata()
                self.assertEqual(t1.shape, (res, res, res))
                # The volumes are what the encoder sees, and the lesion mask sits at the scored centre.
                np.testing.assert_allclose(t1, dataset[subject["index"]]["image"][0].squeeze().numpy(), atol=1e-6)
                lesion = nib.load(files["lesion"]).get_fdata()
                np.testing.assert_allclose(np.argwhere(lesion > 0).mean(0), centres[subject["index"]], atol=1e-9)
                self.assertTrue(set(np.unique(nib.load(files["tissue"]).get_fdata())) >= {0, 1, 2, 3})
            with self.assertRaises(SystemExit):
                worst.run(args)  # the output directory is no longer empty
            args.out, args.target = str(Path(tmp) / "other"), "lesion_q"
            with self.assertRaises(SystemExit) as refused:
                worst.run(args)

        self.assertEqual(len(rows), cfg["num_val_samples"])
        self.assertEqual([int(r["rank"]) for r in rows], list(range(1, len(rows) + 1)))
        badness = [abs(float(r["error"])) for r in rows]
        self.assertEqual(badness, sorted(badness, reverse=True))  # worst first
        by_index = {int(r["index"]): r for r in rows}
        for r in rows[:5]:
            self.assertAlmostEqual(float(r["truth"]), centres[int(r["index"])][1])  # "lesion y" is the centre's y
            self.assertAlmostEqual(float(r["error"]), float(r["prediction"]) - float(r["truth"]))
        groups = [(s["group"], s["index"]) for s in manifest["subjects"]]
        self.assertEqual(
            groups,
            [("worst", int(rows[0]["index"])), ("worst", int(rows[1]["index"])), ("best", int(rows[-1]["index"]))],
        )
        self.assertEqual(manifest["score_kind"], "R²")
        self.assertTrue(np.isfinite(manifest["held_out_score"]))
        self.assertIn(by_index[int(rows[0]["index"])]["side"], {"left", "right"})
        self.assertIn("choose from", str(refused.exception.code))


if __name__ == "__main__":
    unittest.main()
