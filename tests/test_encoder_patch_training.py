"""Global/patch objective wiring, gradients, checkpoint replay and launch recipes."""

import contextlib
import copy
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eval.protocol.score_checkpoint import build_model
from scripts import generate_conv_patch_slurm, run_encoder_mps
from training import main_conv_synthetic as trainer

ROOT = Path(__file__).resolve().parents[1]


class EncoderPatchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())

    def config(self, **changes):
        cfg = vars(
            trainer.parse_args(
                [
                    "--res",
                    "16",
                    "--hidden-channels",
                    "8",
                    "--res-channels",
                    "4",
                    "--nb-res-layers",
                    "1",
                    "--latent-dim",
                    "12",
                    "--conv-readout",
                    "mlp",
                    "--encoder-head-hidden",
                    "8",
                    "--patch-loss-weight",
                    "0.5",
                    "--train-patch-grid",
                    "2",
                    "2",
                    "2",
                    "--tau",
                    "0.1",
                    "--best-metric",
                    "none",
                    "--batch-size",
                    "2",
                ]
            )
        )
        cfg.update(changes)
        return cfg

    def test_validation_distinguishes_training_and_evaluation_grids(self):
        legacy = trainer.parse_args([])
        self.assertEqual(legacy.patch_loss_weight, 0)
        for args in (
            ["--patch-loss-weight", "-1"],
            ["--patch-loss-weight", "nan"],
            ["--patch-loss-weight", "inf"],
            ["--patch-loss-weight", "1", "--train-patch-grid", "0", "2", "2"],
            ["--patch-loss-weight", "1", "--train-patch-grid", "9", "8", "8"],
            ["--patch-loss-weight", "1", "--encoder-architecture", "resnet18"],
            ["--patch-loss-weight", "1", "--contrastive-loss-type", "barlow_twins"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(args)
        args = trainer.parse_args(["--patch-loss-weight", "1", "--eval-pooling", "gap"])
        self.assertEqual(args.eval_pooling, "gap")
        self.assertEqual(args.train_patch_grid, [8, 8, 8])

    def test_single_backbone_pass_matches_both_existing_readouts(self):
        for readout in ("linear", "mlp"):
            with self.subTest(readout=readout):
                cfg = self.config(conv_readout=readout)
                model = build_model(cfg, "cpu")
                images = torch.randn(4, 1, 16, 16, 16)
                expected_global = model(images, pool_only=True, n_views=2)[2][0]
                expected_patch = model(images, pool_only=True, n_views=2, patch_grid=[2] * 3)[2][0]
                with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as a, patch.object(
                    model.encoder_v1, "forward", wraps=model.encoder_v1.forward
                ) as b:
                    global_, patches = model.global_and_patch_features(images, [2] * 3)
                    self.assertEqual((a.call_count, b.call_count), (1, 1))
                torch.testing.assert_close(global_, expected_global, rtol=0, atol=0)
                torch.testing.assert_close(patches, expected_patch, rtol=0, atol=0)
                restored = build_model(cfg, "cpu", model.state_dict())
                torch.testing.assert_close(restored(images, pool_only=True, n_views=2)[2][0], global_, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "must fit"):
                    model.global_and_patch_features(images, [5] * 3)

    def test_zero_weight_preserves_global_outputs_loss_and_gradients_exactly(self):
        cfg = self.config(patch_loss_weight=0.0, train_patch_grid=[999] * 3)
        old = build_model(cfg, "cpu").train()
        new = copy.deepcopy(old)
        images = torch.randn(4, 1, 16, 16, 16)
        expected = old(images, pool_only=True, n_views=2)[2][0]
        expected_loss = trainer.contrastive_loss(expected, old, SimpleNamespace(**cfg), *self.criteria)
        with patch.object(new, "global_and_patch_features", side_effect=AssertionError("Should not extract patches")):
            actual, actual_loss, terms = trainer.training_objective(new, images, SimpleNamespace(**cfg), *self.criteria)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        self.assertEqual(terms["patch"].item(), 0)
        actual_loss.backward()
        expected_loss.backward()
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_patch_pairs_are_subjects_at_same_position_and_ignore_style(self):
        cfg = self.config()
        model = build_model(cfg, "cpu")
        features = torch.randn(6, 12, 5, requires_grad=True)
        loss = trainer.patch_contrastive_loss(features, model, SimpleNamespace(**cfg), *self.criteria)
        a, b = F.normalize(features[:, :9], dim=1).chunk(2)
        expected = (
            sum(
                F.cross_entropy(a[:, :, p] @ b[:, :, p].T / cfg["tau"], torch.arange(3))
                + F.cross_entropy(b[:, :, p] @ a[:, :, p].T / cfg["tau"], torch.arange(3))
                for p in range(5)
            )
            / 5
        )
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(features.grad[:, :9].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, 9:].abs().sum().item(), 0)
        # A position-only code, identical across subjects, cannot identify the subject.
        atlas = torch.randn(1, 12, 5).expand(6, -1, -1)
        atlas_loss = trainer.patch_contrastive_loss(atlas, model, SimpleNamespace(**cfg), *self.criteria)
        torch.testing.assert_close(atlas_loss, 2 * torch.log(torch.tensor(3.0)))

    def test_local_gradients_reach_both_backbones_readout_and_optional_projector(self):
        for readout, projector in (("linear", 0), ("mlp", 0), ("mlp", 5)):
            with self.subTest(readout=readout, projector=projector):
                cfg = self.config(conv_readout=readout, contrastive_proj_dim=projector, contrastive_proj_hidden=8)
                model = build_model(cfg, "cpu").train()
                _, total, terms = trainer.training_objective(
                    model, torch.randn(4, 1, 16, 16, 16), SimpleNamespace(**cfg), *self.criteria
                )
                torch.testing.assert_close(total, terms["global"] + 0.5 * terms["patch"])
                terms["patch"].backward()
                for module in (model.encoder, model.encoder_v1, model.to_encoding, model.projector):
                    if module is None:
                        continue
                    grad = next(module.parameters()).grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)

    def test_real_training_writes_replayable_checkpoint_and_separate_loss_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [
                "--out-dir",
                tmp,
                "--model-id",
                "patch_smoke",
                "--require-new-run",
                "--device",
                "cpu",
                "--res",
                "16",
                "--hidden-channels",
                "8",
                "--res-channels",
                "4",
                "--nb-res-layers",
                "1",
                "--latent-dim",
                "12",
                "--conv-readout",
                "mlp",
                "--encoder-head-hidden",
                "8",
                "--patch-loss-weight",
                "0.5",
                "--train-patch-grid",
                "2",
                "2",
                "2",
                "--tau",
                "0.1",
                "--batch-size",
                "2",
                "--train-steps",
                "2",
                "--eval-every",
                "2",
                "--log-every",
                "1",
                "--num-train-samples",
                "8",
                "--num-val-samples",
                "20",
                "--no-cache",
                "--no-floor-eval",
                "--best-metric",
                "none",
                "--hash-training-inputs",
                "--synthetic-clean-content",
                "--synthetic-normalize",
                "fixed_reference",
            ]
            # The real renderer/loader/training loop run; unrelated expensive label probes are omitted.
            with patch.object(sys, "argv", ["trainer", *args]), patch.object(
                trainer, "evaluate", return_value=({}, {})
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                trainer.main()
            directory = Path(tmp) / "patch_smoke"
            cfg = json.loads((directory / "settings.json").read_text())
            progress = json.loads((directory / "training_progress.json").read_text())
            self.assertEqual((progress["status"], progress["step"]), ("complete", 2))
            self.assertEqual(cfg["patch_loss_weight"], 0.5)
            self.assertEqual(progress["train_patch_grid"], [2, 2, 2])
            terms = progress["last_loss_terms"]
            self.assertAlmostEqual(terms["total"], terms["global"] + terms["patch_weighted"], places=5)
            self.assertAlmostEqual(terms["patch_weighted"], 0.5 * terms["patch"], places=5)
            self.assertIn("weighted patch", output.getvalue())
            state = torch.load(directory / "model.pt", weights_only=True)
            initial = torch.load(directory / "model_init.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(state[k], initial[k]) for k in state))
            restored = build_model(cfg, "cpu", state)
            self.assertEqual(restored(torch.zeros(2, 1, 16, 16, 16), pool_only=True, n_views=2)[2][0].shape, (2, 12))
            self.assertFalse((directory / "model_best.pt").exists())

    def test_local_and_slurm_launchers_use_matched_recipe_and_separate_outputs(self):
        local = ROOT / "experiments/generated/encoder_conv_mlp_patch_s42.mps.sh"
        subprocess.run(["bash", "-n", str(local)], check=True)
        env = {**os.environ, "ENCODER_PYTHON": sys.executable}
        preview = subprocess.check_output(["bash", str(local), "--dry-run"], env=env, text=True)
        args = trainer.parse_args(shlex.split(preview)[3:])
        self.assertEqual((args.conv_readout, args.patch_loss_weight, args.best_metric), ("mlp", 1, "none"))
        self.assertEqual(args.train_patch_grid, [8, 8, 8])
        self.assertEqual(args.batch_size, 32)
        self.assertIn("patch8x8x8_w1", args.model_id)
        options = self.config(batch_size=2, model_seed=42, lr=1e-4, grad_clip=2)
        self.assertGreater(run_encoder_mps._disposable_step(options, "cpu"), 0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            script = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(script)], check=True)
            preview = subprocess.check_output(["bash", str(script), "--dry-run"], env=env, text=True)
            cluster = trainer.parse_args(shlex.split(preview)[3:])
            self.assertEqual(cluster.out_dir, "/scratch/users/k24058220/encoder_patch_slurm_bio/runs")
            self.assertEqual(cluster.patch_loss_weight, 1)
            ignored = {"device", "out_dir", "model_id"}
            self.assertEqual(
                {k: v for k, v in vars(args).items() if k not in ignored},
                {k: v for k, v in vars(cluster).items() if k not in ignored},
            )


if __name__ == "__main__":
    unittest.main()
