"""Protected image initialization and the real residual-head training loop."""

import contextlib
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from eval.protocol.score_checkpoint import build_model
from training import main_conv_synthetic as trainer

SMALL = (
    "--res 16 --hidden-channels 8 --res-channels 4 --nb-res-layers 0 --latent-dim 7 "
    "--content-channels 4 --conv-readout mlp --encoder-head-hidden 8 --batch-size 2 "
    "--num-train-samples 8 --num-val-samples 20 --best-metric none --cpu-threads 1 "
    "--no-cache --no-floor-eval --device cpu --synthetic-mode pseudo_mri"
).split()
HEADS = (
    "--residual-head-units 3 --residual-head-hidden 4 --residual-head-components 2 "
    "--residual-head-subjects 6 --residual-head-joint-dim 4 --global-loss-weight 0"
).split()


class Subjects(Dataset):
    def __init__(self, count, offset=0):
        self.count = count
        self.read = []
        self.images = torch.randn(count, 2, 16, 16, 16, generator=torch.Generator().manual_seed(31 + offset))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.read.append(index)
        # Real synthetic rendering changes RNGs too; fitting must restore them.
        random.random()
        np.random.random()
        torch.rand(1)
        views = self.images[index]
        return {"image": (views[:1], views[1:]), "index": index}


class ResidualHeadTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def source(self, directory):
        args = trainer.parse_args(SMALL)
        model = trainer.build_encoder(args)
        directory.mkdir()
        (directory / "settings.json").write_text(json.dumps(vars(args)))
        torch.save(model.state_dict(), directory / "model.pt")
        return args, model

    def enabled(self, source=None):
        extra = ["--no-residual-head-freeze-image"] if source is None else ["--residual-head-image-init", str(source)]
        return trainer.parse_args(SMALL + HEADS + extra)

    def test_source_load_is_strict_and_preserves_the_original_image_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original = self.source(root / "source")
            args = self.enabled(root / "source")
            state, provenance = trainer.read_residual_image_init(args)
            model = trainer.build_encoder(args)
            trainer.load_residual_image_state(model, state)
            trainer.fit_residual_heads(model, Subjects(8), args, "cpu")
            model.eval()
            original.eval()
            images = torch.randn(4, 1, 16, 16, 16)
            with torch.no_grad():
                expected = original(images, pool_only=True, n_views=2)[2][0]
                actual = model(images, pool_only=True, n_views=2)[2][0][:, : args.latent_dim]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(provenance["sha256"]), 64)
            for name in ("activation", "norm_type", "downscale_factor"):
                changed = copy.copy(args)
                setattr(
                    changed,
                    name,
                    {"activation": "gelu", "norm_type": "layer", "downscale_factor": 2}[name],
                )
                with self.subTest(name=name), self.assertRaisesRegex(SystemExit, name):
                    trainer.read_residual_image_init(changed)

    def test_fit_reads_training_rows_only_preserves_rng_and_restores_without_refitting(
        self,
    ):
        args = self.enabled()
        model = trainer.build_encoder(args)
        subjects = Subjects(8)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        report = trainer.fit_residual_heads(model, subjects, args, "cpu")
        self.assertEqual(subjects.read, list(range(6)))
        self.assertEqual(report["dataset_split"], "train")
        self.assertEqual(report["dataset_indices"], list(range(6)))
        self.assertTrue(model.residual_heads.code_started.all())
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)
        state = copy.deepcopy(model.state_dict())
        restored = trainer.fit_residual_heads(model, subjects, args, "cpu")
        self.assertEqual(subjects.read, list(range(6)))
        self.assertEqual(restored["source"], "checkpoint")
        self.assertEqual(restored["preprocessing_sha256"], report["preprocessing_sha256"])
        for name, value in state.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_full_checkpoint_initialization_rejects_a_changed_image_function(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = self.enabled()
            model = trainer.build_encoder(args)
            trainer.fit_residual_heads(model, Subjects(8), args, "cpu")
            (directory / "settings.json").write_text(json.dumps(vars(args)))
            torch.save(model.state_dict(), directory / "model.pt")
            args.init_from = str(directory)
            args.residual_head_freeze_image = True
            args.residual_head_alignment_mode = "batch"  # Objective-only setting; checkpoint tensors stay compatible.
            state, _ = trainer.read_init_from(args)
            self.assertTrue(state["residual_heads.fitted"])
            args.activation = "gelu"
            with self.assertRaisesRegex(SystemExit, "activation"):
                trainer.read_init_from(args)

    def test_zero_global_weight_preserves_original_auxiliary_only_objective(
        self,
    ):
        args = self.enabled()
        model = trainer.build_encoder(args)
        subjects = Subjects(8)
        trainer.fit_residual_heads(model, subjects, args, "cpu")
        model.train()
        images = torch.cat([subjects.images[:2, :1], subjects.images[:2, 1:]])
        criteria = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        _, loss, terms = trainer.training_objective(model, images, args, *criteria)
        torch.testing.assert_close(loss, terms["residual_heads"])
        torch.testing.assert_close(
            loss,
            terms["residual_heads_image_alignment"]
            + terms["residual_heads_residual_alignment"]
            + terms["residual_heads_uniformity"],
        )
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.image_parameters()))
        self.assertTrue(
            any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.residual_heads.encoders.parameters())
        )
        trainer.clip_encoder_gradients(model, 1.0)

    def test_weighted_primary_patch_and_auxiliary_terms_are_added(self):
        args = self.enabled()
        args.global_loss_weight = 0.7
        args.residual_head_auxiliary_weight = 0.05
        args.patch_loss_weight = 0.2
        args.train_patch_grid = [2, 2, 2]
        model = trainer.build_encoder(args)
        subjects = Subjects(8)
        trainer.fit_residual_heads(model, subjects, args, "cpu")
        images = torch.cat([subjects.images[:2, :1], subjects.images[:2, 1:]])
        _, loss, terms = trainer.training_objective(
            model,
            images,
            args,
            torch.nn.CosineSimilarity(dim=-1),
            torch.nn.CrossEntropyLoss(),
        )
        torch.testing.assert_close(terms["global_weighted"], 0.7 * terms["global"])
        torch.testing.assert_close(terms["residual_heads_auxiliary_raw"], terms["residual_heads_total"])
        torch.testing.assert_close(
            terms["residual_heads_auxiliary_weighted"],
            0.05 * terms["residual_heads_auxiliary_raw"],
        )
        torch.testing.assert_close(
            loss,
            0.7 * terms["global"] + 0.2 * terms["patch"] + 0.05 * terms["residual_heads"],
        )

    def test_zero_auxiliary_weight_matches_legacy_image_parameter_gradients(self):
        args = self.enabled()
        args.global_loss_weight = 1.0
        args.residual_head_auxiliary_weight = 0.0
        model = trainer.build_encoder(args)
        subjects = Subjects(8)
        trainer.fit_residual_heads(model, subjects, args, "cpu")
        baseline_args = trainer.parse_args(SMALL)
        baseline = trainer.build_encoder(baseline_args)
        state = {name: value for name, value in model.state_dict().items() if not name.startswith("residual_heads.")}
        state["content_mask"] = baseline.content_mask.clone()
        baseline.load_state_dict(state, strict=True)
        model.train()
        baseline.train()
        self.assertFalse(model.residual_heads.training)
        residual_state = {name: value.clone() for name, value in model.residual_heads.state_dict().items()}
        images = torch.cat([subjects.images[:2, :1], subjects.images[:2, 1:]])
        criteria = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        _, actual, terms = trainer.training_objective(model, images, args, *criteria)
        _, expected, _ = trainer.training_objective(baseline, images, baseline_args, *criteria)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradients = trainer.residual_head_gradient_report(model, terms, args)
        self.assertGreater(gradients["image_primary/image"], 0)
        self.assertEqual(gradients["image_primary/residual"], 0)
        self.assertEqual(gradients["image_primary/projector"], 0)
        self.assertEqual(gradients["auxiliary/image"], 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        actual.backward()
        expected.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.residual_heads.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        for name, value in model.residual_heads.state_dict().items():
            torch.testing.assert_close(value, residual_state[name], rtol=0, atol=0)
        expected_parameters = dict(baseline.named_parameters())
        for name, parameter in model.named_parameters():
            if not name.startswith("residual_heads."):
                torch.testing.assert_close(parameter.grad, expected_parameters[name].grad, rtol=0, atol=0)

    def test_frozen_image_primary_has_no_gradients_and_diagnostics_do_not_change_backward(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            self.source(source)
            args = self.enabled(source)
            args.global_loss_weight = 1.0
            model = trainer.build_encoder(args)
            subjects = Subjects(8)
            trainer.fit_residual_heads(model, subjects, args, "cpu")
            images = torch.cat([subjects.images[:2, :1], subjects.images[:2, 1:]])
            _, loss, terms = trainer.training_objective(
                model,
                images,
                args,
                torch.nn.CosineSimilarity(dim=-1),
                torch.nn.CrossEntropyLoss(),
            )
            before = {name: value.clone() for name, value in model.state_dict().items()}
            rng = torch.get_rng_state()
            gradients = trainer.residual_head_gradient_report(model, terms, args)
            self.assertEqual(gradients["image_primary/image"], 0)
            self.assertEqual(gradients["auxiliary/image"], 0)
            self.assertGreater(gradients["auxiliary/residual"], 0)
            self.assertGreater(gradients["uniformity/projector"], 0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            loss.backward()
            self.assertTrue(all(parameter.grad is None for parameter in model.image_parameters()))

    def test_real_training_loop_saves_replayable_protected_checkpoint_and_provenance(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, source = self.source(root / "source")
            flags = (
                SMALL
                + HEADS
                + [
                    "--residual-head-image-init",
                    str(root / "source"),
                    "--out-dir",
                    str(root),
                    "--model-id",
                    "protected",
                    "--train-steps",
                    "2",
                    "--eval-every",
                    "2",
                    "--log-every",
                    "1",
                    "--require-new-run",
                    "--residual-head-gradient-every",
                    "1",
                    "--residual-head-alignment-mode",
                    "batch",
                    "--residual-head-auxiliary-weight",
                    "0.1",
                    "--global-loss-weight",
                    "1",
                ]
            )
            with patch.object(sys, "argv", ["trainer", *flags]), patch.object(
                trainer,
                "make_dataset",
                side_effect=lambda args, split, count: Subjects(count, split == "val"),
            ), patch.object(trainer, "evaluate", return_value=({}, {})), contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            directory = root / "protected"
            progress = json.loads((directory / "training_progress.json").read_text())
            self.assertEqual((progress["status"], progress["step"]), ("complete", 2))
            self.assertEqual(progress["initialization_kind"], "protected_image_checkpoint")
            self.assertIn("sha256", progress["residual_heads_image_initialization"])
            self.assertIn("residual_heads_uniformity", progress["last_loss_terms"])
            self.assertEqual(progress["residual_head_gradient_report"]["step"], 2)
            self.assertEqual(
                len((directory / "residual_head_gradients.jsonl").read_text().splitlines()),
                2,
            )
            self.assertAlmostEqual(
                progress["last_loss_terms"]["total"],
                progress["last_loss_terms"]["global_weighted"]
                + progress["last_loss_terms"]["residual_heads_auxiliary_weighted"],
                places=6,
            )
            self.assertAlmostEqual(
                progress["last_loss_terms"]["residual_heads_auxiliary_weighted"],
                0.1 * progress["last_loss_terms"]["residual_heads_auxiliary_raw"],
                places=6,
            )
            settings = json.loads((directory / "settings.json").read_text())
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            initial = torch.load(directory / "model_init.pt", map_location="cpu", weights_only=True)
            model = build_model(settings, "cpu", state)
            self.assertEqual(model.residual_heads.alignment_mode, "batch")
            build_model(settings, "cpu", initial)
            for name, value in source.state_dict().items():
                if name != "content_mask":
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)
            self.assertTrue(
                any(
                    not torch.equal(initial[name], state[name])
                    for name, _ in model.residual_heads.encoders.named_parameters(prefix="residual_heads.encoders")
                )
            )


if __name__ == "__main__":
    unittest.main()
