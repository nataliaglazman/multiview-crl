"""Dense readout: shapes, position kept where GAP loses it, global-only outputs, gradients, replay and eval paths."""

import contextlib
import gc
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from eval.encoder.encoder_generalization_audit import capture_batch
from eval.encoder.encoder_normalization_audit import capture
from eval.protocol.score_checkpoint import build_model
from models.dense_readout import DenseReadout3d

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from training import main_conv_synthetic as trainer  # noqa: E402


def config(**changes):
    cfg = dict(
        hidden_channels=64,
        res_channels=4,
        nb_res_layers=1,
        downscale_factor=4,
        latent_dim=12,
        content_channels=9,
        encoder_head_hidden=7,
        seed=17,
        conv_readout="dense",
    )
    cfg.update(changes)
    return cfg


def paired_images(size=64):
    x = torch.randn(4, 1, size, size, size, generator=torch.Generator().manual_seed(3))
    x[2:] = 1.5 * x[2:] + 0.2
    return x


def infonce(model, x):
    a, b = F.normalize(model(x, pool_only=True, n_views=2)[2][0][:, :9], dim=1).chunk(2)
    return F.cross_entropy(a @ b.T / 0.1, torch.arange(a.shape[0]))


class DenseReadoutModuleTests(unittest.TestCase):
    def test_default_head_takes_a_16_cubed_map_to_4_cubed_and_reads_it_whole(self):
        head = DenseReadout3d(64, 100, 12)
        h = torch.randn(3, 64, 16, 16, 16)
        self.assertEqual(head.features(h).shape, (3, 64 * 4**3))
        self.assertEqual(head(h).shape, (3, 12))
        convs = 2 * (64 * 64 * 4**3 + 64)
        norms = 2 * 2 * 64
        mlp = (64 * 4**3 * 100 + 100) + (100 * 12 + 12)
        self.assertEqual(sum(p.numel() for p in head.parameters()), convs + norms + mlp)

    def test_a_moving_blob_changes_the_dense_input_but_not_gap(self):
        # The same blob at two places: GAP is identical, the flattened grid is not.
        def blob(position):
            h = torch.zeros(1, 4, 16, 16, 16)
            h[0, 0, position[0] : position[0] + 3, position[1] : position[1] + 3, position[2] : position[2] + 3] = 1.0
            return h

        a, b = blob((2, 3, 4)), blob((10, 9, 8))
        torch.testing.assert_close(a.mean((2, 3, 4)), b.mean((2, 3, 4)), atol=0, rtol=0)
        head = DenseReadout3d(4, 8, 3).eval()
        with torch.no_grad():
            self.assertGreater((head.features(a) - head.features(b)).abs().max().item(), 1e-3)
        # Without downsampling, the flattened grid is the map itself: every position its own input.
        exact = DenseReadout3d(4, 8, 3, downsample=0, grid=16)
        torch.testing.assert_close(exact.features(a), a.flatten(1), atol=0, rtol=0)

    def test_a_larger_or_non_cubic_map_is_averaged_to_the_grid(self):
        head = DenseReadout3d(4, 8, 3).eval()
        h = torch.randn(2, 4, 22, 27, 22)
        with torch.no_grad():
            reduced = head.down(h)
            self.assertEqual(tuple(reduced.shape[2:]), (5, 6, 5))
            torch.testing.assert_close(head.features(h), F.adaptive_avg_pool3d(reduced, (4, 4, 4)).flatten(1))
            # A map that already lands on the grid is read as is.
            h = torch.randn(2, 4, 16, 16, 16)
            torch.testing.assert_close(head.features(h), head.down(h).flatten(1), atol=0, rtol=0)

    def test_bad_maps_and_arguments_fail(self):
        head = DenseReadout3d(4, 8, 3)
        with self.assertRaisesRegex(ValueError, "smaller than its 4"):
            head.features(torch.zeros(1, 4, 8, 16, 16))
        with self.assertRaisesRegex(ValueError, "Expected a"):
            head.features(torch.zeros(1, 3, 16, 16, 16))
        for kwargs in (dict(downsample=-1), dict(grid=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DenseReadout3d(4, 8, 3, **kwargs)
        with self.assertRaises(ValueError):
            DenseReadout3d(4, 0, 3)


class DenseReadoutEncoderTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def test_backbones_match_the_pooled_model_and_only_the_global_code_exists(self):
        mlp = build_model(config(conv_readout="mlp"), "cpu")
        dense = build_model(config(), "cpu")
        self.assertEqual(dense.readout_type, "dense")
        self.assertIsInstance(dense.to_encoding, DenseReadout3d)
        mlp_state, state = mlp.state_dict(), dense.state_dict()
        backbone = [k for k in mlp_state if k.startswith(("encoder.", "encoder_v1."))]
        self.assertTrue(backbone)
        for key in backbone:
            torch.testing.assert_close(state[key], mlp_state[key], atol=0, rtol=0)
        self.assertEqual(
            {k for k in state if not k.startswith(("encoder.", "encoder_v1."))} - {"content_mask"},
            {k for k in state if k.startswith("to_encoding.")},
        )
        x = paired_images()
        with torch.no_grad():
            out = dense(x, pool_only=True, n_views=2)
            self.assertEqual(out[2][0].shape, (4, 12))
            self.assertEqual(out[3], [list(range(9))])
            h = dense._encode(x, 2, None)
            torch.testing.assert_close(out[2][0], dense.to_encoding(h), atol=0, rtol=0)
        for kwargs in (dict(pool_only=True, patch_grid=[2, 2, 2]), dict(pool_only=True, patch_grid=[1, 1, 1])):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "no patch"):
                dense(x, n_views=2, **kwargs)
        with self.assertRaisesRegex(ValueError, "no patch"):
            dense(x, pool_only=False, n_views=2)
        with self.assertRaisesRegex(ValueError, "no patch"):
            dense.global_and_patch_features(x, [2, 2, 2])
        with self.assertRaisesRegex(ValueError, "no spatial head"):
            dense.patch_readout

    def test_dense_excludes_pools_spatial_heads_and_resnet(self):
        for changes in (
            dict(global_pool="attention"),
            dict(global_pool="moments"),
            dict(separate_spatial_readout=True),
            dict(encoder_architecture="resnet18"),
            dict(encoder_head_hidden=0),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_model(config(**changes), "cpu")

    def test_lesion_units_follow_the_dense_code(self):
        model = build_model(config(lesion_keypoints=2, res=64), "cpu")
        with torch.no_grad():
            code = model(paired_images(), pool_only=True, n_views=2)[2][0]
        self.assertEqual(code.shape, (4, 12 + 6))

    def test_contrastive_gradient_trains_head_and_both_backbones_and_checkpoint_replays(self):
        cfg = config()
        model = build_model(cfg, "cpu").train()
        x = paired_images()
        infonce(model, x).backward()
        for name in ("to_encoding.down.0.weight", "to_encoding.mlp.0.weight", "encoder.layers.0.0.weight"):
            grad = dict(model.named_parameters())[name].grad
            with self.subTest(name=name):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder_v1.layers[0][0].weight.grad.abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.step()
            optimizer.zero_grad()
            infonce(model, x).backward()
        model.eval()
        restored = build_model(cfg, "cpu", model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(
                restored(x, pool_only=True, n_views=2)[2][0], model(x, pool_only=True, n_views=2)[2][0], atol=0, rtol=0
            )
        for other in (config(conv_readout="mlp"), config(dense_readout_grid=2)):
            with self.subTest(other=other), self.assertRaises(RuntimeError):
                build_model(other, "cpu", model.state_dict())

    def test_audits_read_the_flattened_grid_as_the_readout_input(self):
        model = build_model(config(), "cpu").eval()
        x = paired_images()
        with torch.no_grad():
            h = model._encode(x, 2, None)
            grid = model.to_encoding.features(h)
            hidden = F.leaky_relu(model.to_encoding.mlp[0](grid))
            code = model(x, pool_only=True, n_views=2)[2][0]
        stages = capture_batch(model, x)
        np.testing.assert_allclose(stages["backbone"], grid.numpy(), atol=1e-6)
        np.testing.assert_allclose(stages["hidden"], hidden.numpy(), atol=1e-6)
        np.testing.assert_allclose(stages["content"], code[:, :9].numpy(), atol=1e-6)
        features, _ = capture(model, x)
        np.testing.assert_allclose(features["actual_pool"], grid.numpy(), atol=1e-6)

    def test_cli_accepts_dense_and_refuses_what_needs_patch_outputs(self):
        # The parser's default --res 32 leaves an 8^3 map, too small for the default 2 steps to 4^3; the recipe is 64.
        dense = ["--conv-readout", "dense", "--res", "64"]
        args = trainer.parse_args(dense)
        self.assertEqual((args.conv_readout, args.dense_readout_downsample, args.dense_readout_grid), ("dense", 2, 4))
        trainer.parse_args(["--conv-readout", "dense", "--res", "32", "--dense-readout-downsample", "1"])
        for argv in (
            ["--dense-readout-grid", "2"],
            ["--conv-readout", "mlp", "--dense-readout-downsample", "1"],
            [*dense, "--encoder-architecture", "resnet18"],
            [*dense, "--global-pool", "attention"],
            [*dense, "--global-pool", "moments"],
            [*dense, "--patch-loss-weight", "1"],
            [*dense, "--eval-pooling", "patch"],
            [*dense, "--spatial-recovery-eval"],
            ["--conv-readout", "dense"],
            [*dense, "--dense-readout-downsample", "-1"],
            [*dense, "--dense-readout-grid", "0"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(argv)

    def test_real_training_reports_the_dense_readout_and_writes_a_replayable_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                *("--out-dir", tmp, "--model-id", "dense_smoke", "--require-new-run", "--device", "cpu"),
                *("--res", "16", "--hidden-channels", "64", "--res-channels", "4", "--nb-res-layers", "1"),
                *("--latent-dim", "12", "--conv-readout", "dense", "--encoder-head-hidden", "8"),
                *("--dense-readout-downsample", "1", "--dense-readout-grid", "2"),
                *("--tau", "0.1", "--lr", "0.05", "--batch-size", "2", "--train-steps", "2", "--eval-every", "2"),
                *("--log-every", "1", "--num-train-samples", "8", "--num-val-samples", "20", "--no-cache"),
                *("--best-metric", "none", "--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"),
            ]
            # The real loop and evaluate(); only the expensive factor probes are stubbed.
            with patch.object(sys, "argv", ["trainer", *argv]), patch.object(
                trainer.dci, "compute_dci_synthetic", return_value={}
            ), patch.object(trainer.dci, "flatten_dci_results", return_value={}), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                trainer.main()
            run = Path(tmp) / "dense_smoke"
            parameters = (64 * 64 * 4**3 + 64) + 2 * 64 + (64 * 2**3 * 8 + 8) + (8 * 12 + 12)
            self.assertIn(
                "encoder: conv, dense readout: 1 stride-2 convs -> 64 x 2^3 -> flatten 512 -> 8 -> 12; "
                f"{parameters} parameters",
                output.getvalue(),
            )
            progress = json.loads((run / "training_progress.json").read_text())
            self.assertEqual(progress["readout"], "dense")
            self.assertEqual(progress["dense_readout_parameter_count"], parameters)
            cfg = json.loads((run / "settings.json").read_text())
            state = torch.load(run / "model.pt", weights_only=True)
            restored = build_model(cfg, "cpu", state)
            self.assertIsInstance(restored.to_encoding, DenseReadout3d)
            initial = torch.load(run / "model_init.pt", weights_only=True)
            self.assertFalse(torch.equal(state["to_encoding.mlp.0.weight"], initial["to_encoding.mlp.0.weight"]))


if __name__ == "__main__":
    unittest.main()
