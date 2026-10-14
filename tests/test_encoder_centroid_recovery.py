"""The rendered lesion centroid in the encoder-only per-factor recovery eval, next to lesion_x/y/z."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.encoder.encoder_target_protocol import sample_targets
from eval.metrics import dci
from models.multiview_encoder import MultiviewConvEncoder
from training import main_conv_synthetic as trainer

BASE = ["--res", "16", "--no-cache", "--synthetic-clean-content", "--synthetic-lesion-placement", "wm_interior"]


def dataset(n, extra=()):
    return trainer.make_dataset(trainer.parse_args(BASE + list(extra)), "val", n)


class Writer:
    def __init__(self):
        self.tags = {}

    def add_scalar(self, tag, value, step):
        self.tags[tag] = value


class CentroidTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        cls.ds = dataset(20)
        cls.model = MultiviewConvEncoder(
            hidden_channels=64,
            res_channels=4,
            nb_res_layers=1,
            downscale_factor=4,
            latent_dim=12,
            content_channels=9,
            conv_readout="mlp",
            encoder_head_hidden=8,
        ).eval()

    def test_centroids_are_each_subjects_rendered_lesion_and_computed_once(self):
        centroids = trainer.lesion_centroid_targets(self.ds)
        self.assertEqual(centroids.shape, (20, 3))
        for i in (0, 7, 19):
            target, _ = sample_targets(self.ds._inner, self.ds[i]["gt_latents"])
            np.testing.assert_allclose(centroids[i], target[9:12], atol=1e-6)
        self.assertIs(trainer.lesion_centroid_targets(self.ds), centroids)
        burden = dataset(2, ["--synthetic-lesion-target", "burden"])
        self.assertIsNone(trainer.lesion_centroid_targets(burden))

    def test_extra_targets_use_the_factor_probes_and_change_no_aggregate(self):
        gt = np.stack([self.ds[i]["gt_latents"]["z_content"].numpy() for i in range(len(self.ds))])
        centroids = trainer.lesion_centroid_targets(self.ds)
        kwargs = dict(encoder=self.model, dataset=self.ds, device="cpu", batch_size=4, per_encoder=True)
        plain = dci.compute_dci_synthetic(**kwargs)
        scored = dci.compute_dci_synthetic(**kwargs, extra_targets=(trainer.CENTROID_NAMES, centroids))
        # No aggregate (block MCC, DCI, ridge informativeness, view leakage) moves.
        a, b = dci.flatten_dci_results(plain), dci.flatten_dci_results(scored)
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_equal(a[key], b[key], err_msg=key)
        for block in ("content→content", "content_v2→content"):
            detail = scored[f"{block}/detail"]
            self.assertEqual(detail["extra_factor_names"], list(trainer.CENTROID_NAMES))
            for key in ("per_extra_ridge", "per_extra_mcc", "per_extra_mcc_std", "per_extra_channel_mcc"):
                self.assertEqual(len(detail[key]), 3)
            self.assertEqual(len(detail["per_factor_mcc"]), 9)  # the FLAIR block keeps per-factor MCC too
        self.assertNotIn("extra_factor_names", scored["content→style/detail"])
        # Fed a factor as an extra target, the extra ridge probe reproduces that factor's row exactly. (Block MCC
        # can differ: a factor row's readout is Hungarian-matched among all nine factors.)
        copy = dci.compute_dci_synthetic(**kwargs, extra_targets=(("again",), gt[:, 2:3]))
        detail = copy["content→content/detail"]
        self.assertEqual(detail["per_extra_ridge"][0], detail["per_factor_ridge"][2])
        with self.assertRaises(ValueError):
            dci.compute_dci_synthetic(**kwargs, extra_targets=(("short",), centroids[:5, :1]))

        scores = trainer.per_factor_scores(scored)
        self.assertEqual(list(scores), ["content→content", "content→style", "content_v2→content"])
        self.assertEqual(list(scores["content→content"])[-3:], list(trainer.CENTROID_NAMES))
        self.assertTrue(scores["content→content"]["centroid_x"]["extra"])
        writer, out = Writer(), io.StringIO()
        with contextlib.redirect_stdout(out):
            trainer.print_per_factor(scores, floor=scores, writer=writer, step=3)
        text = out.getvalue()
        self.assertIn("content_v2→content (FLAIR encoder)", text)
        self.assertEqual(text.count("rendered lesion centroid: where the lesion is"), 2)
        self.assertEqual(text.count("lesion x/y/z mean ridge R²: generator draws"), 2)
        self.assertIn("per_factor/content_to_content/centroid_x/ridge_r2", writer.tags)
        drawn = np.mean([scores["content→content"][f"lesion_{a}"]["ridge"] for a in "xyz"])
        self.assertAlmostEqual(writer.tags["per_factor/content_to_content/lesion_xyz_mean/ridge_r2"], drawn)
        self.assertIn("per_factor/content_v2_to_content/centroid_xyz_mean/ridge_r2", writer.tags)

    def test_training_eval_prints_and_saves_centroid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                *("--out-dir", tmp, "--model-id", "centroid_smoke", "--require-new-run", "--device", "cpu"),
                *BASE,
                *("--hidden-channels", "64", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"),
                *("--conv-readout", "mlp", "--encoder-head-hidden", "8", "--batch-size", "2", "--train-steps", "2"),
                *("--eval-every", "2", "--log-every", "1", "--num-train-samples", "8", "--num-val-samples", "20"),
                *("--best-metric", "none", "--synthetic-normalize", "fixed_reference"),
            ]
            with patch.object(sys, "argv", ["trainer", *argv]), contextlib.redirect_stdout(io.StringIO()) as output:
                trainer.main()
            run = Path(tmp) / "centroid_smoke"
            for step in (0, 2):
                per_factor = json.loads((run / f"dci_step{step}.json").read_text())["per_factor"]
                for block in ("content→content", "content_v2→content"):
                    self.assertEqual(list(per_factor[block])[-3:], list(trainer.CENTROID_NAMES))
            self.assertIn("rendered centroid", output.getvalue())


if __name__ == "__main__":
    unittest.main()
