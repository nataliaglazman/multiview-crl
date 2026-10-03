"""Untied content readouts: matched initialization, isolated gradients and real probes."""

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

import numpy as np
import torch

from eval.encoder.encoder_spatial_target_audit import capture
from eval.protocol.score_checkpoint import build_model
from scripts import generate_conv_patch_slurm, run_encoder_mps
from tests.test_encoder_mps_runner import runner_args
from tests.test_spatial_recovery_monitor import arguments
from training import main_conv_synthetic as trainer

ROOT = Path(__file__).resolve().parents[1]


class SeparateSpatialReadoutTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        # With only 8 channels, 8-group normalization gives one channel/group;
        # its GAP is constant at initialization. Use 16 to exercise global gradients.
        self.cfg = vars(trainer.parse_args(arguments() + ["--hidden-channels", "16", "--separate-spatial-readout"]))
        self.criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())

    def test_copied_initialization_preserves_legacy_state_rng_and_global_outputs(self):
        legacy_cfg = {k: v for k, v in self.cfg.items() if k != "separate_spatial_readout"}
        shared = build_model(legacy_cfg, "cpu")
        rng = torch.get_rng_state()
        separate = build_model(self.cfg, "cpu")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(shared.spatial_readout)
        self.assertEqual(separate.spatial_readout[-1].out_features, 9)
        for key, value in shared.state_dict().items():
            torch.testing.assert_close(value, separate.state_dict()[key], rtol=0, atol=0)
        for global_, local in zip(separate.to_encoding.parameters(), separate.spatial_readout.parameters()):
            self.assertNotEqual(global_.data_ptr(), local.data_ptr())
            torch.testing.assert_close(global_[: local.shape[0]], local, rtol=0, atol=0)
        images = torch.randn(4, 1, 16, 16, 16)
        with torch.no_grad():
            expected_global, expected_patch = shared.global_and_patch_features(images, [2] * 3)
            with patch.object(separate.encoder, "forward", wraps=separate.encoder.forward) as a, patch.object(
                separate.encoder_v1, "forward", wraps=separate.encoder_v1.forward
            ) as b:
                global_, spatial = separate.global_and_patch_features(images, [2] * 3)
            self.assertEqual((a.call_count, b.call_count), (1, 1))
            self.assertEqual(spatial.shape, (4, 9, 8))
            torch.testing.assert_close(global_, expected_global, rtol=0, atol=0)
            torch.testing.assert_close(spatial, expected_patch[:, :9])
        # Old settings without the new flag still load strict, unchanged state dicts.
        build_model(legacy_cfg, "cpu", shared.state_dict())
        with self.assertRaises(RuntimeError):
            build_model(self.cfg, "cpu", shared.state_dict())

    def test_losses_update_only_their_own_head_and_both_backbones(self):
        model = build_model(self.cfg, "cpu").train()
        images = torch.randn(4, 1, 16, 16, 16)
        args = SimpleNamespace(**self.cfg)
        for term, active, inactive in (
            ("global", model.to_encoding, model.spatial_readout),
            ("patch", model.spatial_readout, model.to_encoding),
        ):
            model.zero_grad(set_to_none=True)
            _, _, losses = trainer.training_objective(model, images, args, *self.criteria)
            losses[term].backward()
            self.assertTrue(all(p.grad is None for p in inactive.parameters()))
            for module in (active, model.encoder, model.encoder_v1):
                grads = [p.grad for p in module.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _, total, losses = trainer.training_objective(model, images, args, *self.criteria)
        torch.testing.assert_close(total, losses["global"] + losses["patch"])
        total.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.to_encoding[0].weight, model.spatial_readout[0].weight))
        restored = build_model(self.cfg, "cpu", model.state_dict())
        model.eval()
        for expected, actual in zip(
            model.global_and_patch_features(images, [2] * 3), restored.global_and_patch_features(images, [2] * 3)
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(run_encoder_mps._disposable_step(self.cfg, "cpu"), 0)

    def test_spatial_probes_read_new_head_and_keep_gap_global(self):
        model = build_model(self.cfg, "cpu")
        images = torch.randn(4, 1, 16, 16, 16)
        with torch.no_grad():
            expected_gap = model(images, pool_only=True, n_views=2)[2][0]
            # Make the local mapping unmistakably different from the copied global one.
            model.spatial_readout[-1].weight.zero_()
            model.spatial_readout[-1].bias.fill_(7)
            values, _, _ = capture(model, images, [1, 2], include_native=True)
            np.testing.assert_array_equal(values[1, "projected"], expected_gap[:, :9].numpy())
            self.assertIn((1, "style"), values)
            for grid in (2, 4):
                actual = model(images, pool_only=True, n_views=2, patch_grid=[grid] * 3)
                self.assertEqual(actual[2][0].shape, (4, 9, grid**3))
                self.assertEqual(actual[6][0].shape, (1, 9))
                self.assertNotIn((grid, "style"), values)
                np.testing.assert_array_equal(values[grid, "projected"], np.full((4, 9 * grid**3), 7))
                np.testing.assert_array_equal(values[grid, "projected"], actual[2][0].flatten(1).numpy())
            native = model(images, n_views=2)[2][0]
            self.assertEqual(native.shape, (4, 9, 4, 4, 4))
            self.assertTrue((native == 7).all())

    def test_invalid_combinations_fail_early(self):
        for extra in (
            ["--conv-readout", "linear"],
            ["--patch-loss-weight", "0"],
            ["--contrastive-proj-dim", "5"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(arguments() + ["--separate-spatial-readout", *extra])
        for changes in (dict(conv_readout="linear"), dict(contrastive_proj_dim=5)):
            with self.assertRaises(ValueError):
                build_model({**self.cfg, **changes}, "cpu")

    def test_launchers_change_only_readout_and_run_id(self):
        for device in ("cuda", "mps"):
            common = dict(variant="conv_mlp", patch_loss_weight=1, train_patch_grid=[8] * 3, device=device)
            shared, _ = run_encoder_mps.make_options(runner_args(**common))
            separate, _ = run_encoder_mps.make_options(runner_args(**common, separate_spatial_readout=True))
            self.assertEqual(
                {k for k in shared.keys() | separate.keys() if shared.get(k) != separate.get(k)},
                {"separate_spatial_readout", "model_id"},
            )
            self.assertEqual(separate["model_id"], shared["model_id"] + "_separate_spatial")
            env = {**os.environ, "ENCODER_PYTHON": sys.executable}
            preview = subprocess.check_output(
                [
                    "bash",
                    str(ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"),
                    "--separate-spatial-readout",
                    "--dry-run",
                ],
                env=env,
                text=True,
            )
            parsed = trainer.parse_args(shlex.split(preview)[3:])
            self.assertTrue(parsed.separate_spatial_readout)
            self.assertTrue(parsed.spatial_recovery_eval)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            original = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            before = original.read_bytes()
            generate_conv_patch_slurm.main(["--output-dir", tmp, "--separate-spatial-readout"])
            self.assertEqual(before, original.read_bytes())
            separate_script = Path(tmp) / "encoder_conv_mlp_patch_separate_spatial_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(separate_script)], check=True)
            configs = []
            for script in (original, separate_script):
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], text=True)
                configs.append(vars(trainer.parse_args(shlex.split(preview)[3:])))
            self.assertEqual(
                {k for k in configs[0] if configs[0][k] != configs[1][k]}, {"separate_spatial_readout", "model_id"}
            )
            self.assertTrue(configs[1]["out_dir"].startswith("/scratch/users/k24058220/"))

    def test_real_training_monitor_and_checkpoint_with_matched_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(trainer, "evaluate", return_value=({}, {})):
            progress = []
            for separate in (False, True):
                name = "separate" if separate else "shared"
                argv = [
                    "trainer",
                    *arguments(),
                    "--hidden-channels",
                    "16",
                    "--out-dir",
                    tmp,
                    "--model-id",
                    name,
                    "--eval-every",
                    "2",
                ]
                argv += ["--separate-spatial-readout"] if separate else ["--no-spatial-recovery-eval"]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
                directory = Path(tmp) / name
                progress.append(json.loads((directory / "training_progress.json").read_text()))
                self.assertEqual((progress[-1]["status"], progress[-1]["step"]), ("complete", 2))
                if separate:
                    cfg = json.loads((directory / "settings.json").read_text())
                    state = torch.load(directory / "model.pt", weights_only=True)
                    initial = torch.load(directory / "model_init.pt", weights_only=True)
                    restored = build_model(cfg, "cpu", state)
                    self.assertEqual(restored.spatial_readout[-1].out_features, 9)
                    self.assertTrue(progress[-1]["separate_spatial_readout"])
                    for head in ("to_encoding", "spatial_readout"):
                        self.assertFalse(torch.equal(state[f"{head}.0.weight"], initial[f"{head}.0.weight"]))
                    for step in (0, 2):
                        report = json.loads((directory / f"spatial_recovery/step_{step:08d}/report.json").read_text())
                        self.assertTrue(report["encoder_unchanged"])
                        self.assertEqual(report["projected_readouts"], {"grid_1": "global", "larger_grids": "spatial"})
                        for row in report["probes"]:
                            if row["stage"] == "projected":
                                self.assertEqual(row["readout_head"], "global" if row["grid"] == 1 else "spatial")
                                self.assertEqual(row["dimensions"], 9 * row["grid"] ** 3)
                    self.assertFalse(list(directory.rglob("*.npy")))
                    self.assertFalse((directory / "model_best.pt").exists())
            for key in ("batch_order_sha256", "training_input_sha256"):
                self.assertEqual(progress[0][key], progress[1][key])


if __name__ == "__main__":
    unittest.main()
