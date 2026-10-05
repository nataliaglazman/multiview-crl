"""Encoder-only --norm-type: GroupNorm vs per-voxel LayerNorm in the Conv encoder, end to end."""

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
from unittest.mock import patch

import torch

from eval.protocol.score_checkpoint import build_model
from models.multiview_encoder import MultiviewConvEncoder
from models.vqvae import ChannelLayerNorm3d
from scripts import generate_conv_patch_slurm, run_encoder_mps
from tests.test_encoder_mps_runner import runner_args
from tests.test_encoder_target_followups import config
from training import main_conv_synthetic as trainer

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(hidden_channels=8, res_channels=4, nb_res_layers=1, latent_dim=12, content_channels=9)
SMALL.update(conv_readout="mlp", encoder_head_hidden=8)
BASE = ["--res", "16", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
BASE += ["--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1", "--best-metric", "none"]
BASE += ["--batch-size", "2"]


def norms(module):
    return [m for m in module.modules() if isinstance(m, (torch.nn.GroupNorm, ChannelLayerNorm3d))]


class ConvNormTypeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def build(self, **changes):
        torch.manual_seed(3)
        return MultiviewConvEncoder(**{**SMALL, **changes}).eval()

    def test_default_is_group_and_layer_changes_only_the_norms(self):
        legacy, group, layer = self.build(), self.build(norm_type="group"), self.build(norm_type="layer")
        self.assertEqual(list(legacy.state_dict()), list(group.state_dict()))
        for key, value in legacy.state_dict().items():
            torch.testing.assert_close(value, group.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(isinstance(m, torch.nn.GroupNorm) for m in norms(group.encoder)))
        self.assertTrue(norms(layer.encoder) and all(isinstance(m, ChannelLayerNorm3d) for m in norms(layer.encoder)))
        self.assertEqual((group.normalization, layer.normalization), ("group", "layer"))
        # Norm layers draw no random numbers: every convolution and the head start identical.
        shared = group.state_dict().keys() & layer.state_dict().keys()
        self.assertIn("encoder.layers.0.0.weight", shared)
        for key in shared:
            torch.testing.assert_close(group.state_dict()[key], layer.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.build(norm_type="batch")
        with self.assertRaises(ValueError):
            MultiviewConvEncoder(latent_dim=12, content_channels=9, encoder_architecture="resnet18", norm_type="layer")

    def test_layer_norm_keeps_background_cells_independent_of_the_brain(self):
        # ReZero starts at alpha = 0, so the backbone output is the final norm's output.
        # Different brain content, not just a rescaled brain: GroupNorm divides out a uniform scale.
        brain, other = torch.zeros(2, 1, 1, 64, 64, 64)
        brain[..., 24:40, 24:40, 24:40] = torch.randn(1, 1, 16, 16, 16)
        other[..., 24:40, 24:40, 24:40] = torch.randn(1, 1, 16, 16, 16)
        for norm, background_moves in (("group", True), ("layer", False)):
            with self.subTest(norm=norm), torch.no_grad():
                model = self.build(norm_type=norm)
                before, after = model.encoder(brain), model.encoder(other)
                corner = (before - after)[..., :2, :2, :2].abs().max().item()  # far outside the receptive field
                self.assertEqual(corner > 1e-4, background_moves)
                if norm == "layer":
                    torch.testing.assert_close(before.mean(1), torch.zeros_like(before.mean(1)), atol=1e-5, rtol=0)
                else:  # one channel per group here: every channel averages to zero over space
                    torch.testing.assert_close(before.mean((2, 3, 4)), torch.zeros(1, 8), atol=1e-5, rtol=0)

    def test_build_model_restores_the_norm_and_rejects_mismatched_checkpoints(self):
        group_cfg, layer_cfg = config(), config(norm_type="layer")
        group, layer = build_model(group_cfg, "cpu"), build_model(
            layer_cfg, "cpu", build_model(layer_cfg, "cpu").state_dict()
        )
        self.assertEqual((group.normalization, layer.normalization), ("group", "layer"))
        with self.assertRaises(RuntimeError):  # parameter names differ, so a mismatch cannot load silently
            build_model(group_cfg, "cpu", layer.state_dict())

    def test_trainer_flag_records_and_trains_with_layer_norm(self):
        self.assertEqual(trainer.parse_args(BASE).norm_type, "group")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.parse_args(["--encoder-architecture", "resnet18", "--norm-type", "layer"])
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--out-dir", tmp, "--model-id", "ln_smoke", "--require-new-run", "--device", "cpu", *BASE]
            args += ["--norm-type", "layer", "--train-steps", "2", "--eval-every", "2", "--log-every", "1"]
            args += ["--num-train-samples", "8", "--num-val-samples", "20", "--no-cache", "--no-floor-eval"]
            args += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
            with patch.object(sys, "argv", ["trainer", *args]), patch.object(
                trainer, "evaluate", return_value=({}, {})
            ), contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            run = Path(tmp) / "ln_smoke"
            cfg = json.loads((run / "settings.json").read_text())
            progress = json.loads((run / "training_progress.json").read_text())
            restored = build_model(cfg, "cpu", torch.load(run / "model.pt", weights_only=True))
        self.assertEqual(
            (cfg["norm_type"], progress["normalization"], progress["status"]), ("layer", "layer", "complete")
        )
        self.assertEqual(restored.normalization, "layer")

    def test_launchers_add_only_norm_type_and_a_distinct_run_id(self):
        for device in ("cuda", "mps"):
            common = dict(variant="conv_mlp", device=device)
            plain, _ = run_encoder_mps.make_options(runner_args(**common))
            layer, _ = run_encoder_mps.make_options(runner_args(**common, norm_type="layer"))
            self.assertEqual(
                {k for k in plain.keys() | layer.keys() if plain.get(k) != layer.get(k)}, {"norm_type", "model_id"}
            )
            self.assertEqual(layer["model_id"], plain["model_id"] + "_layernorm")
            with self.assertRaises(ValueError):
                run_encoder_mps.make_options(runner_args(device=device, norm_type="layer"))  # ResNet variant
            preview = subprocess.check_output(
                [
                    "bash",
                    str(ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"),
                    "--norm-type",
                    "layer",
                    "--dry-run",
                ],
                env={**os.environ, "ENCODER_PYTHON": sys.executable},
                text=True,
            )
            parsed = trainer.parse_args(shlex.split(preview)[3:])
            self.assertEqual(parsed.norm_type, "layer")
            self.assertTrue(parsed.model_id.endswith("_layernorm"))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generate_conv_patch_slurm.main(["--output-dir", tmp])
            original = Path(tmp) / "encoder_conv_mlp_patch_s42.slurm_bio.sh"
            before = original.read_bytes()
            generate_conv_patch_slurm.main(["--output-dir", tmp, "--norm-type", "layer"])
            self.assertEqual(before, original.read_bytes())
            layer_script = Path(tmp) / "encoder_conv_mlp_patch_layernorm_s42.slurm_bio.sh"
            subprocess.run(["bash", "-n", str(layer_script)], check=True)
            configs = []
            for script in (original, layer_script):
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], text=True)
                configs.append(vars(trainer.parse_args(shlex.split(preview)[3:])))
            self.assertEqual({k for k in configs[0] if configs[0][k] != configs[1][k]}, {"norm_type", "model_id"})


if __name__ == "__main__":
    unittest.main()
