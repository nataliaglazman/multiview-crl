"""Patch descriptor configuration survives trainer/exported checkpoint reconstruction."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from eval.protocol.score_checkpoint import build_model
from models.patch_distribution import PATCH_QUANTILES
from training import main_conv_synthetic as trainer
from training.encoder_args import cli_arguments, parse_args


def config(**changes):
    return dict(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=0,
        downscale_factor=4,
        latent_dim=7,
        content_channels=4,
        conv_readout="mlp",
        encoder_head_hidden=8,
        norm_type="layer",
        res=16,
        seed=42,
        **changes,
    )


class PatchSubjects(torch.utils.data.Dataset):
    """Small image/mask-only splits with enough brain support for both patch scales."""

    def __init__(self, count, offset=0):
        self.count, self.offset = count, offset

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(1234 + self.offset + index)
        base = torch.linspace(-1, 1, 16**3).reshape(1, 16, 16, 16)
        images = tuple(base + 0.1 * torch.randn(base.shape, generator=generator) for _ in range(2))
        return {"image": images, "mask": tuple(torch.ones_like(image) for image in images), "index": index}


class PatchDistributionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_new_cli_roundtrips_and_rejects_invalid_descriptor_geometry(self):
        options = [
            "--res",
            "64",
            "--vector-prediction-weight",
            "0.1",
            "--vector-prediction-target",
            "patch_distribution",
            "--vector-prediction-grid",
            "32",
            "32",
            "32",
        ]
        args = parse_args(options)
        self.assertEqual(vars(parse_args(cli_arguments(vars(args)))), vars(args))
        self.assertEqual(args.vector_prediction_patch_size, 3)
        self.assertEqual(args.vector_prediction_patch_scales, [1, 2])
        self.assertEqual(args.vector_prediction_patch_projections, 8)
        self.assertEqual(args.vector_prediction_patch_seed, 1729)
        for extra in (
            ["--vector-prediction-patch-size", "0"],
            ["--vector-prediction-patch-size", "2"],
            ["--vector-prediction-patch-scales", "0"],
            ["--vector-prediction-patch-scales", "1", "1"],
            ["--vector-prediction-patch-projections", "0"],
            ["--vector-prediction-patch-size", "1"],
            ["--vector-prediction-patch-seed", "-1"],
            ["--vector-prediction-patch-seed", str(2**63)],
            ["--vector-prediction-grid", "4", "4", "4"],
            ["--vector-prediction-template-subjects", "8"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(options + extra)

    def test_checkpoint_reconstructs_nondefault_patch_descriptor_and_state(self):
        settings = config(
            vector_prediction_weight=0.1,
            vector_prediction_target="patch_distribution",
            vector_prediction_grid=[8, 8, 8],
            vector_prediction_components=2,
            vector_prediction_patch_size=3,
            vector_prediction_patch_scales=[1, 2],
            vector_prediction_patch_projections=3,
            vector_prediction_patch_seed=37,
        )
        original = build_model(settings, "cpu")
        predictor = original.vector_prediction
        self.assertEqual(
            (predictor.patch_size, tuple(predictor.patch_scales), predictor.patch_projections, predictor.patch_seed),
            (3, (1, 2), 3, 37),
        )
        self.assertEqual(predictor.decoders[0][-1].out_features, 2 * 3 * len(PATCH_QUANTILES))
        restored = build_model(settings, "cpu", original.state_dict())
        for name, value in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            build_model({**settings, "vector_prediction_patch_projections": 4}, "cpu", original.state_dict())

    def test_enabled_and_disabled_descriptor_preserve_encoder_initialization_and_rng(self):
        legacy = build_model(config(), "cpu")
        legacy_rng = torch.get_rng_state().clone()
        settings = config(
            vector_prediction_weight=0.1,
            vector_prediction_target="patch_distribution",
            vector_prediction_grid=[8, 8, 8],
        )
        enabled = build_model(settings, "cpu")
        torch.testing.assert_close(torch.get_rng_state(), legacy_rng, rtol=0, atol=0)
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(enabled.state_dict()[name], value, rtol=0, atol=0)
        disabled = build_model({**settings, "vector_prediction_weight": 0.0}, "cpu", legacy.state_dict())
        self.assertIsNone(disabled.vector_prediction)
        self.assertEqual(set(disabled.state_dict()), set(legacy.state_dict()))

    def test_two_step_trainer_predicts_patches_validates_and_restores_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            flags = (
                "--res 16 --hidden-channels 8 --res-channels 4 --nb-res-layers 0 --latent-dim 7 "
                "--content-channels 4 --conv-readout mlp --encoder-head-hidden 8 --batch-size 2 "
                "--num-train-samples 6 --num-val-samples 4 --best-metric none --cpu-threads 1 "
                "--vector-prediction-weight 0.1 --vector-prediction-target patch_distribution "
                "--vector-prediction-grid 8 8 8 --vector-prediction-components 2 "
                "--vector-prediction-template-subjects 4 --vector-prediction-hidden 8 "
                "--vector-prediction-patch-projections 3 --vector-prediction-patch-seed 37 "
                "--train-steps 2 --eval-every 2 --log-every 1 --no-floor-eval --no-cache "
                "--device cpu --require-new-run --model-id patch_distribution_smoke"
            ).split() + ["--out-dir", tmp]
            validation = []

            def evaluate(model, dataset, device, args, *unused, **kwargs):
                validation.append(trainer.vector_prediction_report(model, dataset, args, device))
                return {}, {}

            def dataset(args, mode, count):
                return PatchSubjects(count, offset=0 if mode == "train" else 100)

            with patch.object(sys, "argv", ["trainer", *flags]), patch.object(
                trainer, "make_dataset", side_effect=dataset
            ), patch.object(trainer, "evaluate", side_effect=evaluate), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                trainer.main()
            directory = Path(tmp) / "patch_distribution_smoke"
            settings = json.loads((directory / "settings.json").read_text())
            progress = json.loads((directory / "training_progress.json").read_text())
            template = json.loads((directory / "vector_prediction_template.json").read_text())
            self.assertEqual((progress["status"], progress["step"]), ("complete", 2))
            self.assertEqual(progress["vector_prediction_target"], "patch_distribution")
            self.assertEqual(template["subjects_per_view"], [4, 4])
            descriptor = template["patch_distribution"]
            self.assertEqual(descriptor["probabilities"], list(PATCH_QUANTILES))
            self.assertEqual(descriptor["scales"], [1, 2])
            self.assertEqual(descriptor["projection_seed"], 37)
            self.assertEqual(len(descriptor["projection_filters_sha256"]), 64)
            self.assertIn("valid stride-1", descriptor["geometry"])
            self.assertEqual(template["decoder_outputs"], 2 * 3 * len(PATCH_QUANTILES))
            terms = progress["last_loss_terms"]
            self.assertGreater(terms["vector_prediction"], 0)
            self.assertIn("vector_prediction_t1_to_flair", terms)
            self.assertIn("vector_prediction_flair_to_t1", terms)
            self.assertAlmostEqual(terms["vector_prediction_weighted"], 0.1 * terms["vector_prediction"], places=5)
            self.assertIn("approximate sliced Wasserstein-2", output.getvalue())
            self.assertEqual(len(validation), 1)
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in validation[0].values()))
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            initial = torch.load(directory / "model_init.pt", map_location="cpu", weights_only=True)
            restored = build_model(settings, "cpu", state)
            build_model(settings, "cpu", initial)
            self.assertEqual(restored.vector_prediction.target, "patch_distribution")
            self.assertTrue(
                any(
                    not torch.equal(state[f"vector_prediction.{name}"], initial[f"vector_prediction.{name}"])
                    for name, _ in restored.vector_prediction.named_parameters()
                )
            )
            for name in (
                "template",
                "template_counts",
                "target_components",
                "target_mean",
                "target_sd",
                "patch_filters",
            ):
                torch.testing.assert_close(
                    state[f"vector_prediction.{name}"], initial[f"vector_prediction.{name}"], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
