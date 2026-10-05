"""Encoder-only --patch-foreground-mask: the VQ trainer's rule, the real loop and the launchers."""

import contextlib
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

from eval.protocol.score_checkpoint import build_model
from scripts import generate_conv_patch_slurm, run_encoder_mps
from tests.test_encoder_mps_runner import runner_args
from training import main_conv_synthetic as trainer

ROOT = Path(__file__).resolve().parents[1]
BASE = ["--res", "16", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
BASE += ["--conv-readout", "mlp", "--encoder-head-hidden", "8", "--patch-loss-weight", "0.5"]
BASE += ["--train-patch-grid", "2", "2", "2", "--tau", "0.1", "--best-metric", "none", "--batch-size", "2"]


class PatchForegroundMaskTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())

    def test_flags_default_off_and_require_patch_training(self):
        args = trainer.parse_args(BASE)
        self.assertFalse(args.patch_foreground_mask)
        self.assertEqual(args.patch_foreground_thresh, 0.05)
        for bad in (
            ["--patch-loss-weight", "0", "--patch-foreground-mask"],
            ["--patch-foreground-thresh", "0"],
            ["--patch-foreground-thresh", "1.5"],
        ):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(BASE + bad)

    def test_positions_follow_the_vq_rule_and_never_empty(self):
        masks = torch.zeros(4, 1, 8, 8, 8)
        masks[0, 0, :4, :4, :4] = 1  # bin 0 fully brain in one image
        masks[2, 0, 4:, 4:, 4:5] = 1  # a quarter of bin 7 in another
        self.assertEqual(trainer.foreground_positions(masks, (2, 2, 2), 0.05).tolist(), [True] + [False] * 6 + [True])
        self.assertEqual(trainer.foreground_positions(masks, (2, 2, 2), 0.5).tolist(), [True] + [False] * 7)
        self.assertTrue(trainer.foreground_positions(torch.zeros(4, 1, 8, 8, 8), (2, 2, 2), 0.05).all())

    def test_masked_loss_uses_only_kept_positions(self):
        cfg = vars(trainer.parse_args(BASE + ["--patch-foreground-mask"]))
        args, plain = SimpleNamespace(**cfg), SimpleNamespace(**{**cfg, "patch_foreground_mask": False})
        model = build_model(cfg, "cpu")
        patches = torch.randn(4, 12, 8)  # (T1 then FLAIR images, latent units, 2³ positions)
        masks = torch.zeros(4, 1, 8, 8, 8)
        masks[:, :, :4, :4, :4] = 1  # bin 0 is brain in every image
        masks[1, :, 4:, :4, :4] = 1  # bin 4 is brain in one image only
        loss = trainer.patch_contrastive_loss(patches, model, args, *self.criteria, foreground=masks)
        expected = trainer.patch_contrastive_loss(patches[..., [0, 4]], model, plain, *self.criteria)
        torch.testing.assert_close(loss, expected)
        changed = patches.clone()
        changed[..., [1, 2, 3, 5, 6, 7]] = torch.randn(4, 12, 6)  # always-background bins
        torch.testing.assert_close(
            trainer.patch_contrastive_loss(changed, model, args, *self.criteria, foreground=masks), loss
        )
        changed[0, :, 4] = torch.randn(12)
        moved = trainer.patch_contrastive_loss(changed, model, args, *self.criteria, foreground=masks)
        self.assertFalse(torch.allclose(moved, loss))
        with self.assertRaisesRegex(ValueError, "brain masks"):
            trainer.patch_contrastive_loss(patches, model, args, *self.criteria)

    def test_objective_unchanged_when_off_and_image_support_matches_masks(self):
        cfg = vars(trainer.parse_args(BASE))
        model = build_model(cfg, "cpu").eval()
        masks = torch.zeros(4, 1, 16, 16, 16)
        masks[..., :6, :6, :6] = 1  # only bin 0 holds brain
        images = torch.randn(4, 1, 16, 16, 16) * masks  # zero outside the mask, like the dataset inputs
        off = trainer.training_objective(model, images, SimpleNamespace(**cfg), *self.criteria)[1]
        off_with_masks = trainer.training_objective(model, images, SimpleNamespace(**cfg), *self.criteria, masks)[1]
        torch.testing.assert_close(off, off_with_masks)
        on = SimpleNamespace(**{**cfg, "patch_foreground_mask": True})
        with_masks = trainer.training_objective(model, images, on, *self.criteria, masks)[1]
        torch.testing.assert_close(with_masks, trainer.training_objective(model, images, on, *self.criteria)[1])
        self.assertFalse(torch.allclose(with_masks, off))

    def test_real_training_records_the_flag_and_reports_kept_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--out-dir", tmp, "--model-id", "fg_smoke", "--require-new-run", "--device", "cpu", *BASE]
            args += ["--patch-foreground-mask", "--train-steps", "2", "--eval-every", "2", "--log-every", "1"]
            args += ["--num-train-samples", "8", "--num-val-samples", "20", "--no-cache", "--no-floor-eval"]
            args += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
            with patch.object(sys, "argv", ["trainer", *args]), patch.object(
                trainer, "evaluate", return_value=({}, {})
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                trainer.main()
            run = Path(tmp) / "fg_smoke"
            cfg = json.loads((run / "settings.json").read_text())
            progress = json.loads((run / "training_progress.json").read_text())
        self.assertTrue(cfg["patch_foreground_mask"])
        self.assertEqual(cfg["patch_foreground_thresh"], 0.05)
        self.assertEqual((progress["status"], progress["step"]), ("complete", 2))
        self.assertRegex(output.getvalue(), r"first batch keeps \d+/8 positions")

    def test_launchers_add_only_the_flag_and_a_distinct_run_id(self):
        for device in ("cuda", "mps"):
            common = dict(variant="conv_mlp", patch_loss_weight=1, train_patch_grid=[8] * 3, device=device)
            plain, _ = run_encoder_mps.make_options(runner_args(**common))
            masked, _ = run_encoder_mps.make_options(runner_args(**common, patch_foreground_mask=True))
            self.assertEqual(
                {k for k in plain.keys() | masked.keys() if plain.get(k) != masked.get(k)},
                {"patch_foreground_mask", "model_id"},
            )
            self.assertEqual(masked["model_id"], plain["model_id"] + "_fgmask")
            stricter, _ = run_encoder_mps.make_options(
                runner_args(**common, patch_foreground_mask=True, patch_foreground_thresh=0.2)
            )
            self.assertTrue(stricter["model_id"].endswith("_fgmask0.2"))
            with self.assertRaises(ValueError):
                run_encoder_mps.make_options(runner_args(variant="conv_mlp", device=device, patch_foreground_mask=True))
            parsed = trainer.parse_args(run_encoder_mps.comparison.training_command(stricter)[3:])
            self.assertTrue(parsed.patch_foreground_mask)
            self.assertEqual(parsed.patch_foreground_thresh, 0.2)
            preview = subprocess.check_output(
                [
                    "bash",
                    str(ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"),
                    "--patch-loss-weight",  # the wrappers' own default may be global-only
                    "1",
                    "--patch-foreground-mask",
                    "--dry-run",
                ],
                env={**os.environ, "ENCODER_PYTHON": sys.executable},
                text=True,
            )
            self.assertTrue(trainer.parse_args(shlex.split(preview)[3:]).patch_foreground_mask)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            original = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            before = original.read_bytes()
            generate_conv_patch_slurm.main(["--output-dir", tmp, "--patch-foreground-mask"])
            self.assertEqual(before, original.read_bytes())
            masked_script = Path(tmp) / "encoder_conv_mlp_patch_fgmask_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(masked_script)], check=True)
            configs = []
            for script in (original, masked_script):
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], text=True)
                configs.append(vars(trainer.parse_args(shlex.split(preview)[3:])))
            self.assertEqual(
                {k for k in configs[0] if configs[0][k] != configs[1][k]}, {"patch_foreground_mask", "model_id"}
            )
            self.assertTrue(configs[1]["model_id"].endswith("_fgmask_s42"))


if __name__ == "__main__":
    unittest.main()
