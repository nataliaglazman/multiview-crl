"""Frozen real-model feature probes, renderer targets, controls, and CLI reports."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval import score_checkpoint
from eval.checkpoint_lesion_analysis import (
    batch_features,
    json_safe,
    lesion_targets,
    run_analysis,
    score_features,
    state_digest,
)


def config(**kwargs):
    cfg = dict(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        downscale_factor=4,
        latent_dim=12,
        content_channels=9,
        seed=42,
        res=32,
        n_content=9,
        n_style=3,
        synthetic_normalize="fixed_reference",
        synthetic_clean_content=True,
        synthetic_mode="pseudo_mri",
        num_val_samples=24,
    )
    cfg.update(kwargs)
    return cfg


class CheckpointLesionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_targets_are_rendered_centroids_not_latent_coordinates(self):
        ds = score_checkpoint.make_val_dataset(config(res=64, synthetic_lesion_placement="wm_interior"), 3)
        for idx in range(3):
            _, _, latents = ds._inner[idx]
            targets, mass = lesion_targets(ds._inner, latents)
            _, load = ds._inner.renderer.render_structure(
                latents["z_content"], latents["z_deformation"], latents["z_fissure"], "cpu", clean=True
            )
            index_centroid = torch.nonzero(load).float().mean(0).numpy()
            expected = 2 * index_centroid / 63 - 1
            np.testing.assert_allclose(targets[3:], expected, atol=2e-6)
            np.testing.assert_array_equal(targets[:3], latents["z_content"][2:5])
            self.assertFalse(np.allclose(targets[:3], targets[3:]))
            self.assertEqual(mass, float(load.sum()))
        with patch.object(ds._inner.renderer, "render_structure", return_value=(load * 0, load * 0)):
            targets, mass = lesion_targets(ds._inner, latents)
            self.assertEqual(mass, 0)
            self.assertTrue(np.isnan(targets[3:]).all())

    def test_capture_matches_actual_gap_and_patch_paths_and_keeps_state(self):
        for shared in (False, True):
            model = score_checkpoint.build_model(config(no_separate_encoders=shared), "cpu")
            before = state_digest(model)
            x = torch.randn(4, 1, 32, 32, 32)
            features, shape, channels = batch_features(model, x, [1, 4])
            self.assertEqual(shape, (8, 8, 8))
            self.assertEqual(channels, 8)
            with torch.inference_mode():
                h = model._encode(x, 2, None)
                for grid in (1, 4):
                    actual = model(x, pool_only=True, n_views=2, patch_grid=[grid] * 3)[2][0][:, :9].flatten(1)
                    np.testing.assert_allclose(features[(grid, "projected")], actual.numpy(), atol=1e-6)
                    expected = torch.nn.functional.adaptive_avg_pool3d(h, (grid,) * 3).flatten(1).numpy()
                    np.testing.assert_allclose(features[(grid, "backbone")], expected)
            self.assertEqual(before, state_digest(model))
            self.assertFalse(model.encoder._forward_hooks)
            self.assertFalse(model.to_encoding._forward_hooks)
            with self.assertRaisesRegex(ValueError, "must fit"):
                batch_features(model, x, [16])
            self.assertFalse(model.encoder._forward_hooks)
            model.train()
            with self.assertRaisesRegex(ValueError, "model.eval"):
                batch_features(model, x, [1])

    def test_resnet_head_is_applied_after_bin_pooling(self):
        model = score_checkpoint.build_model(config(encoder_architecture="resnet18", no_separate_encoders=True), "cpu")
        x = torch.randn(2, 1, 64, 64, 64)
        before = state_digest(model)
        features, shape, channels = batch_features(model, x, [1, 2])
        self.assertEqual((shape, channels), ((2, 2, 2), 512))
        with torch.inference_mode():
            for grid in (1, 2):
                actual = model(x, pool_only=True, n_views=2, patch_grid=[grid] * 3)[2][0][:, :9].flatten(1)
                np.testing.assert_allclose(features[(grid, "projected")], actual.numpy(), atol=1e-6)
        self.assertEqual(before, state_digest(model))

    def test_planted_information_loss_and_shuffled_controls(self):
        rng = np.random.default_rng(6)
        targets = rng.normal(size=(120, 6))
        permutations = [rng.permutation(120) for _ in range(3)]
        full = score_features(targets, targets, permutations, seeds=(0,), folds=3)
        lost = score_features(targets[:, 3:], targets, permutations, seeds=(0,), folds=3)
        for name, values in full.items():
            self.assertGreater(values["r2"], 0.99)
            self.assertLess(values["shuffled_mean"], 0.15)
            self.assertEqual(len(values["shuffled_r2"]), 3)
            if name.startswith("latent"):
                self.assertLess(lost[name]["r2"], 0.15)
            else:
                self.assertGreater(lost[name]["r2"], 0.99)
        # The number of control targets cannot affect hyperparameter selection for a real target.
        fewer = score_features(targets, targets, permutations[:1], seeds=(0,), folds=3)
        for name in full:
            self.assertAlmostEqual(full[name]["r2"], fewer[name]["r2"], places=10)
        targets[:, 0] = 1
        undefined = score_features(targets[:, 3:], targets, permutations, seeds=(0,), folds=3)
        self.assertEqual(undefined["latent_x"]["status"], "constant_in_test_fold")
        self.assertIsNone(json_safe(undefined)["latent_x"]["r2"])

    def test_extraction_targets_align_and_no_floor_mode_excludes_missing_lesions(self):
        cfg = config()
        ds = score_checkpoint.make_val_dataset(cfg, 24)
        model = score_checkpoint.build_model(cfg, "cpu")
        original = lesion_targets
        calls = 0

        def missing_first(inner, latents):
            nonlocal calls
            target, mass = original(inner, latents)
            calls += 1
            if calls == 1:
                target[3:] = np.nan
                mass = 0
            return target, mass

        with contextlib.redirect_stdout(io.StringIO()), patch(
            "eval.checkpoint_lesion_analysis.lesion_targets", missing_first
        ):
            report = run_analysis(model, None, ds, "cpu", 4, grids=[1], n_shuffles=1)
        self.assertEqual(report["n_valid"], 23)
        self.assertEqual(report["retained_subject_ids"], list(range(1, 24)))
        self.assertEqual({row["arm"] for row in report["rows"]}, {"trained"})
        self.assertEqual(report["subjects"]["valid"][0], False)
        self.assertTrue(all(row["delta_untrained"] is None for row in report["rows"]))
        self.assertEqual(set(report["permuted_subject_ids"][0]), set(range(1, 24)))
        # Confirm each retained target is from the exact sample whose image was encoded.
        for idx in (1, 8, 23):
            expected, _ = original(ds._inner, ds[idx]["gt_latents"])
            np.testing.assert_array_equal(report["subjects"]["targets"][idx], expected)

    def test_analysis_only_cli_saves_controls_and_preserves_checkpoint(self):
        cfg = config()
        model = score_checkpoint.build_model(cfg, "cpu")
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "settings.json").write_text(json.dumps(cfg))
            checkpoint = run / "model_best.pt"
            torch.save(model.state_dict(), checkpoint)
            original = checkpoint.read_bytes()
            argv = [
                "score_checkpoint",
                "--run-dir",
                str(run),
                "--checkpoint",
                "model_best.pt",
                "--lesion-analysis-only",
                "--lesion-grids",
                "1",
                "4",
                "--lesion-shuffles",
                "2",
                "--batch-size",
                "4",
                "--no-cuda",
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), patch.object(
                score_checkpoint, "recovery", side_effect=AssertionError("regular probes should be skipped")
            ):
                score_checkpoint.main()
            self.assertEqual(original, checkpoint.read_bytes())
            (output,) = run.glob("score_lesion_*.json")
            report = json.loads(output.read_text())
            self.assertEqual(report["checkpoint"], "model_best.pt")
            analysis = report["lesion_analysis"]
            self.assertEqual(
                analysis["state"]["trained"]["input_sha256"], analysis["state"]["untrained"]["input_sha256"]
            )
            self.assertEqual(len(analysis["rows"]), 96)
            self.assertTrue(all(value["unchanged"] for value in analysis["state"].values()))
            for row in analysis["rows"]:
                if row["arm"] == "trained":
                    self.assertAlmostEqual(row["delta_untrained"], 0, places=8)
            csv_path = output.with_name(output.stem + "_lesion_scores.csv")
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 96)
            self.assertIn("shuffled_r2_1", rows[0])
            self.assertIn("delta_backbone", rows[0])
            self.assertTrue(output.with_name(output.stem + "_lesion_targets.csv").exists())

    def test_cli_validation_and_defaults(self):
        self.assertFalse(score_checkpoint.parse_args([]).lesion_analysis)
        args = score_checkpoint.parse_args(["--lesion-analysis-only", "--no-floor"])
        self.assertTrue(args.lesion_analysis)
        self.assertTrue(args.no_floor)
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (["--lesion-shuffles", "0"], ["--lesion-grids", "0"], ["--batch-size", "0"]):
                with self.assertRaises(SystemExit):
                    score_checkpoint.parse_args(["--lesion-analysis", *flags])


if __name__ == "__main__":
    unittest.main()
