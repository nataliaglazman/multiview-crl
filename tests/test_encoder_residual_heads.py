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
    "--residual-head-subjects 6 --residual-head-joint-dim 4"
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
            state, _ = trainer.read_init_from(args)
            self.assertTrue(state["residual_heads.fitted"])
            args.activation = "gelu"
            with self.assertRaisesRegex(SystemExit, "activation"):
                trainer.read_init_from(args)

    def test_joint_objective_replaces_global_infonce_and_trains_both_unfrozen_paths(
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
            self.assertEqual(
                progress["last_loss_terms"]["total"],
                progress["last_loss_terms"]["residual_heads"],
            )
            settings = json.loads((directory / "settings.json").read_text())
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            initial = torch.load(directory / "model_init.pt", map_location="cpu", weights_only=True)
            model = build_model(settings, "cpu", state)
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
