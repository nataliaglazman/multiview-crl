"""Vector-only auxiliary training, fixed training targets and trainer integration."""

import contextlib
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from eval.protocol.score_checkpoint import build_model
from models.vector_prediction import prepare_vector_targets, vector_prediction_loss
from training import main_conv_synthetic as trainer

SMALL = (
    "--res 16 --hidden-channels 8 --res-channels 4 --nb-res-layers 0 --latent-dim 7 "
    "--content-channels 4 --conv-readout mlp --encoder-head-hidden 8 --batch-size 2 "
    "--num-train-samples 4 --num-val-samples 20 --best-metric none --cpu-threads 1"
).split()
PREDICTION = (
    "--vector-prediction-weight 0.3 --vector-prediction-grid 4 4 4 "
    "--vector-prediction-hidden 8 --vector-prediction-template-subjects 3"
).split()


def arguments(extra=(), **changes):
    args = trainer.parse_args(SMALL + list(extra))
    for name, value in changes.items():
        setattr(args, name, value)
    return args


class ImageSubjects(Dataset):
    """Image/mask-only subjects, with deliberate RNG use and observable row access."""

    def __init__(self, count=5, offset=0):
        self.count, self.offset, self.read = count, offset, []
        self.base = torch.linspace(-1, 1, 16**3).reshape(1, 16, 16, 16)

    def __len__(self):
        return self.count

    def images(self, index):
        value = index + self.offset
        return (
            self.base + 0.03 * value * self.base.square(),
            self.base.flip(-1) - 0.05 * value * self.base.sin(),
        )

    def __getitem__(self, index):
        self.read.append(index)
        random.random()
        np.random.random()
        torch.rand(1)
        views = self.images(index)
        return {"image": views, "mask": tuple(torch.ones_like(x) for x in views), "index": index}


def paired(dataset, indices):
    return torch.cat([torch.stack([dataset.images(i)[view] for i in indices]) for view in range(2)])


class EncoderVectorPredictionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())

    def fitted_model(self, args):
        model = build_model(vars(args), "cpu").train()
        images = torch.randn(4, 1, 16, 16, 16)
        masks = torch.ones_like(images)
        model.vector_prediction.accumulate_template(images, masks)
        model.vector_prediction.finalize_template()
        return model

    def assert_rng_equal(self, expected):
        python_state, numpy_state, torch_state = expected
        self.assertEqual(random.getstate(), python_state)
        actual = np.random.get_state()
        self.assertEqual(actual[0], numpy_state[0])
        np.testing.assert_array_equal(actual[1], numpy_state[1])
        self.assertEqual(actual[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_parser_rejects_invalid_or_incompatible_prediction_options(self):
        for extra in (
            ["--vector-prediction-weight", "-1"],
            ["--vector-prediction-weight", "nan"],
            ["--vector-prediction-weight", "inf"],
            ["--vector-prediction-hidden", "0"],
            ["--vector-prediction-template-subjects", "0"],
            ["--vector-prediction-grid", "0", "4", "4"],
            ["--vector-prediction-weight-cap", "0.9"],
            ["--vector-prediction-weight-cap", "nan"],
            ["--vector-prediction-deviation-strength", "inf"],
            ["--vector-prediction-target", "histogram"],
            ["--vector-prediction-components", "0"],
            PREDICTION + ["--vector-prediction-target", "quantile", "--vector-prediction-components", "3"],
            PREDICTION + ["--vector-prediction-grid", "17", "4", "4"],
            PREDICTION + ["--vector-prediction-grid", "1", "1", "1"],
            PREDICTION + ["--mask-mode", "learned"],
            PREDICTION + ["--lesion-keypoints", "1"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(extra)

    def test_disabled_feature_preserves_legacy_outputs_losses_and_gradients(self):
        for patch_weight in (0.0, 0.5):
            with self.subTest(patch_weight=patch_weight):
                args = arguments(
                    patch_loss_weight=patch_weight,
                    train_patch_grid=[2, 2, 2],
                    tau=0.1,
                )
                legacy_args = SimpleNamespace(
                    **{name: value for name, value in vars(args).items() if not name.startswith("vector_prediction_")}
                )
                legacy = build_model(vars(args), "cpu").train()
                current = copy.deepcopy(legacy)
                images = torch.randn(4, 1, 16, 16, 16)
                if patch_weight:
                    expected, cells = legacy.global_and_patch_features(images, args.train_patch_grid, n_views=2)
                else:
                    expected = legacy(images, pool_only=True, n_views=2)[2][0]
                expected_loss = trainer.contrastive_loss(expected, legacy, legacy_args, *self.criteria)
                if patch_weight:
                    expected_loss = expected_loss + patch_weight * trainer.patch_contrastive_loss(
                        cells, legacy, legacy_args, *self.criteria
                    )
                with patch.object(trainer, "vector_prediction_loss", side_effect=AssertionError("disabled")):
                    actual, actual_loss, terms = trainer.training_objective(current, images, args, *self.criteria)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                self.assertFalse(any(key.startswith("vector_prediction") for key in terms))
                actual_loss.backward()
                expected_loss.backward()
                self.assertEqual(set(dict(legacy.named_parameters())), set(dict(current.named_parameters())))
                for name, parameter in legacy.named_parameters():
                    other = dict(current.named_parameters())[name]
                    if parameter.grad is None:
                        self.assertIsNone(other.grad)
                    else:
                        torch.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0)

    def test_prediction_reads_exported_content_without_style_or_target_gradients(self):
        args = arguments(PREDICTION, global_loss_weight=0.0)
        model = self.fitted_model(args)
        exported = torch.randn(4, 7, requires_grad=True)
        images = torch.randn(4, 1, 16, 16, 16, requires_grad=True)
        seen = []
        hook = model.vector_prediction.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]))
        self.addCleanup(hook.remove)
        with patch.object(model, "forward", return_value=(None, None, [exported])), patch.object(
            model.encoder, "forward", side_effect=AssertionError("spatial bypass")
        ), patch.object(model.encoder_v1, "forward", side_effect=AssertionError("spatial bypass")):
            pooled, total, terms = trainer.training_objective(
                model, images, args, *self.criteria, masks=torch.ones_like(images)
            )
        self.assertIs(pooled, exported)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], exported[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(total, 0.3 * terms["vector_prediction"])
        terms["vector_prediction"].backward()
        self.assertGreater(exported.grad[:, :4].abs().sum().item(), 0)
        self.assertEqual(exported.grad[:, 4:].abs().sum().item(), 0)
        self.assertIsNone(images.grad)

    def test_prediction_gradients_reach_both_encoders_and_shared_global_head_in_one_pass(self):
        args = arguments(PREDICTION, patch_loss_weight=0.5, train_patch_grid=[2, 2, 2], tau=0.1)
        model = self.fitted_model(args)
        self.assertIs(model.patch_readout, model.to_encoding)
        images = torch.randn(4, 1, 16, 16, 16)
        with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as first, patch.object(
            model.encoder_v1, "forward", wraps=model.encoder_v1.forward
        ) as second:
            _, total, terms = trainer.training_objective(
                model, images, args, *self.criteria, masks=torch.ones_like(images)
            )
        self.assertEqual((first.call_count, second.call_count), (1, 1))
        torch.testing.assert_close(
            total, terms["global"] + terms["patch_weighted"] + terms["vector_prediction_weighted"]
        )
        terms["vector_prediction"].backward()
        for name in ("encoder", "encoder_v1", "to_encoding", "vector_prediction"):
            gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        self.assertEqual(model.to_encoding[-1].weight.grad[4:].abs().sum().item(), 0)

    def test_training_template_uses_only_first_train_subjects_and_preserves_training_state(self):
        args = arguments(PREDICTION)
        model = build_model(vars(args), "cpu").train()
        model.encoder.eval()
        model.vector_prediction.decoders[0].eval()
        modes = [module.training for module in model.modules()]
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        data = ImageSubjects()
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        with patch.object(model.encoder, "forward", side_effect=AssertionError("must fit from images only")):
            metadata = trainer.fit_vector_prediction_template(model, data, args, "cpu")
        self.assert_rng_equal(rng)
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        self.assertEqual(data.read, [0, 1, 2])
        self.assertEqual(metadata["dataset_indices"], [0, 1, 2])
        self.assertEqual(metadata["subjects_per_view"], [3, 3])
        self.assertEqual(metadata["source"], "training_subjects")
        images = paired(data, [0, 1, 2])
        target, _ = prepare_vector_targets(images, torch.ones_like(images), (4, 4, 4))
        expected = target.reshape(2, 3, 1, 4, 4, 4).mean(1)
        torch.testing.assert_close(model.vector_prediction.template, expected, rtol=1e-6, atol=1e-6)
        state = copy.deepcopy(model.state_dict())
        data.read.clear()
        restored = trainer.fit_vector_prediction_template(model, data, args, "cpu")
        self.assertEqual(restored["source"], "checkpoint")
        self.assertEqual(data.read, [])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_validation_report_never_fits_templates_and_includes_partial_final_batch(self):
        args = arguments(PREDICTION)
        model = build_model(vars(args), "cpu").train()
        trainer.fit_vector_prediction_template(model, ImageSubjects(), args, "cpu")
        model.vector_prediction.standardized(torch.randn(8, 4))
        validation = ImageSubjects(count=5, offset=20)
        model.encoder.eval()
        modes = [module.training for module in model.modules()]
        state = copy.deepcopy(model.state_dict())
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        with patch.object(
            model.vector_prediction, "accumulate_template", side_effect=AssertionError("validation fit")
        ), patch.object(
            model.vector_prediction, "reset_template", side_effect=AssertionError("validation reset")
        ), patch.object(
            model.vector_prediction, "finalize_template", side_effect=AssertionError("validation finalize")
        ):
            report = trainer.vector_prediction_report(model, validation, args, "cpu")
        self.assert_rng_equal(rng)
        self.assertEqual(validation.read, [0, 1, 2, 3, 4])
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        model.eval()
        images = paired(validation, range(5))
        with torch.no_grad():
            content = model(images, pool_only=True, n_views=2)[2][0][:, :4]
            _, expected = vector_prediction_loss(
                model.vector_prediction,
                content,
                images,
                torch.ones_like(images),
                args.vector_prediction_deviation_strength,
                args.vector_prediction_weight_cap,
            )
        for name, value in report.items():
            self.assertAlmostEqual(value, float(expected[name]), places=6)

    def test_tiny_quantile_run_trains_on_the_arm_and_records_its_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            flags = (
                SMALL
                + PREDICTION
                + (
                    "--vector-prediction-target quantile --vector-prediction-components 2 "
                    "--train-steps 2 --eval-every 2 --log-every 1 --no-cache --no-floor-eval "
                    "--synthetic-clean-content --synthetic-normalize fixed_reference "
                    "--synthetic-lesion-target burden --synthetic-lesion-count 1 "
                    "--synthetic-lesion-placement wm_interior --synthetic-lesion-t1-value 0 "
                    "--require-new-run --device cpu --model-id quantile_smoke"
                ).split()
                + ["--out-dir", tmp]
            )
            # A 16-voxel brain has no 90%-brain cell on this grid.
            with patch.object(sys, "argv", ["trainer", *flags]), patch.object(
                trainer, "evaluate", return_value=({}, {})
            ), patch("models.vector_prediction.INTERIOR", 0.5), contextlib.redirect_stdout(io.StringIO()) as output:
                trainer.main()
            directory = Path(tmp) / "quantile_smoke"
            template = json.loads((directory / "vector_prediction_template.json").read_text())
            self.assertEqual((template["target"], template["components"]), ("quantile", 2))
            self.assertTrue(all(cells > 0 for cells in template["interior_cells"]))
            terms = json.loads((directory / "training_progress.json").read_text())["last_loss_terms"]
            self.assertGreater(terms["vector_prediction"], 0)
            self.assertEqual(terms["vector_prediction"], terms["vector_prediction_unweighted"])
            self.assertIn("quantiles of the other view", output.getvalue())
            settings = json.loads((directory / "settings.json").read_text())
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            restored = build_model(settings, "cpu", state)
            self.assertEqual(restored.vector_prediction.target, "quantile")
            self.assertTrue(restored.vector_prediction.target_interior.any())

    def test_tiny_training_run_logs_replayable_fixed_targets_and_updates_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            flags = (
                SMALL
                + PREDICTION
                + (
                    "--patch-loss-weight 0.5 --train-patch-grid 2 2 2 --global-loss-weight 0.7 --tau 0.1 "
                    "--train-steps 2 --eval-every 1 --log-every 1 --no-cache --no-floor-eval "
                    "--synthetic-clean-content --synthetic-normalize fixed_reference "
                    "--synthetic-lesion-target burden --synthetic-lesion-count 1 "
                    "--synthetic-lesion-placement wm_interior --synthetic-lesion-t1-value 0 "
                    "--require-new-run --device cpu --model-id vector_smoke"
                ).split()
                + ["--out-dir", tmp]
            )
            with patch.object(sys, "argv", ["trainer", *flags]), patch.object(
                trainer, "evaluate", return_value=({}, {})
            ) as evaluate, contextlib.redirect_stdout(io.StringIO()) as output:
                trainer.main()
            directory = Path(tmp) / "vector_smoke"
            settings = json.loads((directory / "settings.json").read_text())
            progress = json.loads((directory / "training_progress.json").read_text())
            template = json.loads((directory / "vector_prediction_template.json").read_text())
            self.assertEqual((progress["status"], progress["step"]), ("complete", 2))
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(template["subjects_per_view"], [3, 3])
            self.assertEqual(template["dataset_indices"], [0, 1, 2])
            self.assertEqual(progress["vector_prediction_template"], template)
            terms = progress["last_loss_terms"]
            self.assertAlmostEqual(
                terms["total"],
                0.7 * terms["global"] + terms["patch_weighted"] + terms["vector_prediction_weighted"],
                places=5,
            )
            self.assertAlmostEqual(terms["vector_prediction_weighted"], 0.3 * terms["vector_prediction"], places=6)
            for part in trainer.VECTOR_PREDICTION_PARTS:
                self.assertIn(f"vector_prediction_{part}", terms)
            self.assertIn("vector-only cross-view residual prediction", output.getvalue())
            self.assertIn("T1->FLAIR", output.getvalue())
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            initial = torch.load(directory / "model_init.pt", map_location="cpu", weights_only=True)
            restored = build_model(settings, "cpu", state)
            build_model(settings, "cpu", initial)
            decoder_parameters = [
                f"vector_prediction.{name}" for name, _ in restored.vector_prediction.named_parameters()
            ]
            self.assertTrue(any(not torch.equal(state[name], initial[name]) for name in decoder_parameters))
            for name, _ in restored.vector_prediction.named_buffers():
                key = f"vector_prediction.{name}"
                if name.startswith("template"):
                    torch.testing.assert_close(state[key], initial[key], rtol=0, atol=0)
            self.assertTrue(bool(state["vector_prediction.input_gauge_fitted"]))
            self.assertFalse(bool(initial["vector_prediction.input_gauge_fitted"]))
            self.assertEqual(restored.vector_prediction.template_subject_count.tolist(), [3, 3])
            self.assertTrue(bool(restored.vector_prediction.template_fitted))


if __name__ == "__main__":
    unittest.main()
