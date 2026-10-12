"""Temporal width must localize images and survive training/evaluation round trips."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.datasets import SyntheticBrainDataset
from eval.encoder.encoder_target_protocol import dataset
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.legacy_renderer import use_legacy_renderer
from eval.synthetic.preview_temporal_atrophy import render_comparison
from eval.synthetic.synthetic_dataset import PseudoMRIRenderer
from tests.test_lesion_intensity import trainer_make_dataset, vqvae_dataset_kwargs
from training.encoder_args import parse_args


class TemporalSigmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_narrower_width_localizes_tissue_changes_and_zero_atrophy_is_unchanged(self):
        z = torch.zeros(9)
        fields = (torch.zeros(4, 4, 4), torch.zeros(8, 8, 8), "cpu")
        broad = PseudoMRIRenderer(res=64)
        self.assertEqual(broad.temporal_sigma, 0.18)
        reference, _ = broad.render_structure(z, *fields, clean=True)
        counts = []
        previous = None
        for sigma in (0.18, 0.12, 0.10):
            renderer = PseudoMRIRenderer(res=64, temporal_sigma=sigma)
            unchanged, _ = renderer.render_structure(z, *fields, clean=True)
            self.assertTrue(torch.equal(unchanged, reference))
            atrophied = z.clone()
            atrophied[6] = 1.0
            tissue, _ = renderer.render_structure(atrophied, *fields, clean=True)
            changed = tissue != reference
            self.assertTrue(torch.equal(changed, changed.flip(0)))  # bilateral support
            if previous is not None:
                self.assertFalse((changed & ~previous).any())
            previous = changed
            counts.append(int(changed.sum()))
        self.assertTrue(counts[0] > counts[1] > counts[2] > 0, counts)

    def test_invalid_widths_are_rejected(self):
        for sigma in (0, -0.1, float("nan"), float("inf")):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "temporal_sigma"):
                    PseudoMRIRenderer(res=16, temporal_sigma=sigma)
                with self.assertRaisesRegex(ValueError, "temporal_sigma"):
                    SyntheticBrainDataset(spatial_size=(16,) * 3, synthetic_temporal_sigma=sigma)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--synthetic-temporal-sigma", str(sigma)])

    def test_width_survives_encoder_training_and_saved_evaluation(self):
        base = ["--res", "16", "--no-cache", "--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
        args = parse_args(base + ["--synthetic-temporal-sigma", "0.10"])
        settings = json.loads(json.dumps(vars(args)))
        trained = trainer_make_dataset()(args, "val", 4)
        scored = make_dataset(settings, 4, "val")
        followup = dataset(settings, 4, "val")
        for ds in (trained, scored, followup):
            self.assertEqual(ds._inner.renderer.temporal_sigma, 0.10)
            for a, b in zip(trained._inner[0][:2], ds._inner[0][:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        del settings["synthetic_temporal_sigma"]
        restored = make_dataset(settings, 4, "val")
        self.assertEqual(restored._inner.renderer.temporal_sigma, 0.18)
        self.assertFalse(torch.equal(trained._inner[0][0], restored._inner[0][0]))

    def test_width_survives_vqvae_training_and_saved_evaluation(self):
        from eval.protocol.run_dci_synthetic import build_synthetic_test_set, load_run_args
        from utils.config import parse_args as vq_parser

        base = ["--dataset-name", "synthetic", "--synthetic-res", "16", "--synthetic-n-content", "9"]
        base += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
        base += ["--synthetic-num-train", "4", "--synthetic-num-val", "4", "--synthetic-num-test", "4"]
        parser = vq_parser()
        self.assertEqual(parser.parse_args(base).synthetic_temporal_sigma, 0.18)
        args = parser.parse_args(base + ["--synthetic-temporal-sigma", "0.12"])
        settings = json.loads(json.dumps(vars(args)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "settings.json")
            path.write_text(json.dumps(settings))
            restored = load_run_args(folder)
            for split in ("train", "val", "test"):
                trained = SyntheticBrainDataset(mode=split, spatial_size=(16,) * 3, **vqvae_dataset_kwargs(args))
                scored = build_synthetic_test_set(restored, 4, causal=False, cache=False, split=split)
                self.assertEqual(scored._inner.renderer.temporal_sigma, 0.12)
                for a, b in zip(trained._inner[0][:2], scored._inner[0][:2]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            del settings["synthetic_temporal_sigma"]
            path.write_text(json.dumps(settings))
            old = build_synthetic_test_set(load_run_args(folder), 4, causal=False, cache=False)
            self.assertEqual(old._inner.renderer.temporal_sigma, 0.18)

    def test_legacy_renderer_refuses_to_silently_ignore_width(self):
        ds = SyntheticBrainDataset(
            spatial_size=(16,) * 3, synthetic_temporal_sigma=0.12, synthetic_normalize="fixed_reference"
        )
        with self.assertRaisesRegex(ValueError, "temporal_sigma"):
            use_legacy_renderer(ds, verify=False)

    def test_preview_holds_acquisition_fixed_and_reports_real_label_changes(self):
        import numpy as np

        for centered in (False, True):
            images, tissues, metadata = render_comparison(res=32, atrophy_value=0, centered=centered)
            for image, tissue in zip(images, tissues):
                np.testing.assert_array_equal(image, images[0])
                np.testing.assert_array_equal(tissue, tissues[0])
            self.assertEqual(metadata["changed_tissue_voxels_vs_z6_zero"], [0, 0, 0])
        images, tissues, metadata = render_comparison(res=32)
        counts = (tissues[1:] != tissues[0]).sum(axis=(1, 2, 3)).tolist()
        self.assertEqual(metadata["changed_tissue_voxels_vs_z6_zero"], counts)
        self.assertGreater(counts[0], counts[-1])
        self.assertGreater(float(np.abs(images[1] - images[0]).sum()), 0)


if __name__ == "__main__":
    unittest.main()
