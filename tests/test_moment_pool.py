"""Moment pooling: exact GAP start, what the moments read, gradients, checkpoint replay and eval paths."""

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
from eval.protocol.score_checkpoint import build_model
from models.moment_pool import MomentPool3d

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from training import main_conv_synthetic as trainer  # noqa: E402

VARIANTS = (
    dict(encoder_architecture="conv"),
    dict(encoder_architecture="conv", conv_readout="mlp"),
    dict(encoder_architecture="resnet18", resnet_output_stride=8),
)


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
    )
    cfg.update(changes)
    return cfg


def paired_images(size=16):
    x = torch.randn(4, 1, size, size, size, generator=torch.Generator().manual_seed(3))
    x[2:] = 1.5 * x[2:] + 0.2
    return x


def infonce(model, x):
    a, b = F.normalize(model(x, pool_only=True, n_views=2)[2][0][:, :9], dim=1).chunk(2)
    return F.cross_entropy(a @ b.T / 0.1, torch.arange(a.shape[0]))


class MomentPoolModuleTests(unittest.TestCase):
    def test_moments_are_each_channels_coordinate_weighted_mean(self):
        h = torch.randn(2, 3, 4, 5, 6, generator=torch.Generator().manual_seed(0))
        grids = torch.meshgrid(
            torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 5), torch.linspace(-1, 1, 6), indexing="ij"
        )
        expected = torch.cat([(h * axis).mean((2, 3, 4)) for axis in grids], dim=1)
        torch.testing.assert_close(MomentPool3d(3, 7).moments(h), expected)

    def test_moments_read_where_a_fixed_blob_sits_where_gap_cannot(self):
        # The same blob at two positions: GAP is unchanged, and the moment of its channel moves by
        # mass x displacement / positions, i.e. linearly with the blob's position.
        def blob(position):
            h = torch.zeros(1, 2, 8, 8, 8)
            h[0, 0, position[0] : position[0] + 2, position[1] : position[1] + 2, position[2] : position[2] + 2] = 1.0
            return h

        pool = MomentPool3d(2, 4)
        a, b = blob((1, 2, 3)), blob((4, 2, 1))
        torch.testing.assert_close(a.mean((2, 3, 4)), b.mean((2, 3, 4)), atol=0, rtol=0)
        step = 2.0 / 7  # coordinate spacing of an 8-position axis
        moved = pool.moments(b) - pool.moments(a)  # [x of ch0, ch1, y of ch0, ch1, z of ch0, ch1]
        torch.testing.assert_close(moved[0], torch.tensor([8 * 3 * step / 512, 0, 0, 0, 8 * -2 * step / 512, 0]))

    def test_zero_weight_contributes_nothing_and_draws_no_random_numbers(self):
        rng = torch.get_rng_state()
        pool = MomentPool3d(16, 5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(pool.weight.shape, (5, 48))
        out = pool(torch.randn(3, 16, 2, 3, 4))
        self.assertEqual(out.shape, (3, 5))
        self.assertFalse(out.any())

    def test_single_position_axis_sits_at_zero_and_bad_shapes_fail(self):
        h = torch.randn(2, 4, 1, 3, 3)
        self.assertFalse(MomentPool3d(4, 2).moments(h)[:, :4].any())
        with self.assertRaisesRegex(ValueError, "Expected a"):
            MomentPool3d(4, 2).moments(torch.zeros(1, 3, 2, 2, 2))
        with self.assertRaises(ValueError):
            MomentPool3d(0, 2)


class MomentPoolEncoderTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def test_initial_model_is_the_gap_model_exactly_without_drawing_random_numbers(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                size = 32 if variant["encoder_architecture"] == "resnet18" else 16
                gap = build_model(config(**variant), "cpu")
                rng = torch.get_rng_state()
                pooled = build_model(config(global_pool="moments", **variant), "cpu")
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertIsNone(gap.moment_pool)
                self.assertIsNone(pooled.attention_pool)
                state, gap_state = pooled.state_dict(), gap.state_dict()
                self.assertEqual(set(state) - set(gap_state), {"moment_pool.weight"})
                for key, value in gap_state.items():
                    torch.testing.assert_close(state[key], value, atol=0, rtol=0)
                width = 512 if variant["encoder_architecture"] == "resnet18" else 64
                out = 7 if pooled.readout_type == "mlp" else 12
                self.assertEqual(state["moment_pool.weight"].shape, (out, 3 * width))
                self.assertFalse(state["moment_pool.weight"].any())
                x = paired_images(size)
                with torch.no_grad():
                    for kwargs in (
                        dict(pool_only=True),
                        dict(pool_only=True, patch_grid=[2, 2, 2]),
                        dict(pool_only=False),
                    ):
                        torch.testing.assert_close(
                            pooled(x, n_views=2, **kwargs)[2][0], gap(x, n_views=2, **kwargs)[2][0], atol=0, rtol=0
                        )
                    for actual, expected in zip(
                        pooled.global_and_patch_features(x, [2, 2, 2]), gap.global_and_patch_features(x, [2, 2, 2])
                    ):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "no attention maps"):
                    pooled.attention_maps(x, n_views=2)

    def test_contrastive_gradient_trains_the_moment_columns_and_checkpoint_replays(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                size = 32 if variant["encoder_architecture"] == "resnet18" else 16
                cfg = config(global_pool="moments", **variant)
                model = build_model(cfg, "cpu").train()
                x = paired_images(size)
                infonce(model, x).backward()
                grad = model.moment_pool.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
                for _ in range(3):
                    optimizer.step()
                    optimizer.zero_grad()
                    infonce(model, x).backward()
                model.eval()
                self.assertGreater(model.moment_pool.weight.detach().abs().max().item(), 0)
                restored = build_model(cfg, "cpu", model.state_dict())
                with torch.no_grad():
                    code = model(x, pool_only=True, n_views=2)[2][0]
                    torch.testing.assert_close(restored(x, pool_only=True, n_views=2)[2][0], code, atol=0, rtol=0)
                    # Once the columns move, the global code reads the moments: zeroing them changes it.
                    restored.moment_pool.weight.zero_()
                    self.assertGreater((restored(x, pool_only=True, n_views=2)[2][0] - code).abs().max().item(), 0)
                gap_cfg = config(**variant)
                with self.assertRaises(RuntimeError):
                    build_model(gap_cfg, "cpu", model.state_dict())
                with self.assertRaises(RuntimeError):
                    build_model(cfg, "cpu", build_model(gap_cfg, "cpu").state_dict())

    def test_generalization_audit_reads_gap_and_moments_as_the_readout_input(self):
        model = build_model(config(global_pool="moments", conv_readout="mlp"), "cpu")
        with torch.no_grad():
            model.moment_pool.weight.normal_(generator=torch.Generator().manual_seed(1))
        model.eval()
        x = paired_images()
        with torch.no_grad():
            h = model._encode(x, 2, None)
            readout_input = torch.cat([h.mean((2, 3, 4)), model.moment_pool.moments(h)], dim=1)
            hidden = F.leaky_relu(model.to_encoding[0](h.mean((2, 3, 4))) + model.moment_pool(h))
            code = model(x, pool_only=True, n_views=2)[2][0]
        stages = capture_batch(model, x)
        np.testing.assert_allclose(stages["backbone"], readout_input.numpy(), atol=1e-6)
        np.testing.assert_allclose(stages["hidden"], hidden.numpy(), atol=1e-6)
        np.testing.assert_allclose(stages["content"], code[:, :9].numpy(), atol=1e-6)

    def test_cli_accepts_moments_and_keeps_attention_options_to_attention(self):
        args = trainer.parse_args(["--global-pool", "moments"])
        self.assertEqual(args.global_pool, "moments")
        trainer.parse_args(["--global-pool", "moments", "--encoder-architecture", "resnet18", "--res", "64"])
        for argv in (
            ["--global-pool", "moments", "--attention-pool-heads", "8"],
            ["--global-pool", "moments", "--attention-pool-width", "full"],
            ["--global-pool", "moments", "--res", "4"],
            ["--global-pool", "moments", "--encoder-architecture", "resnet18", "--res", "32"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(argv)

    def test_real_training_reports_the_moment_pool_and_writes_a_replayable_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                *("--out-dir", tmp, "--model-id", "moments_smoke", "--require-new-run", "--device", "cpu"),
                *("--res", "16", "--hidden-channels", "64", "--res-channels", "4", "--nb-res-layers", "1"),
                *("--latent-dim", "12", "--conv-readout", "mlp", "--encoder-head-hidden", "8"),
                *("--global-pool", "moments"),
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
            run = Path(tmp) / "moments_smoke"
            self.assertIn("encoder: conv, GAP + moments -> 64 -> 8 -> 12", output.getvalue())
            self.assertIn("global pool: GAP plus first spatial moments; 192 moment features", output.getvalue())
            progress = json.loads((run / "training_progress.json").read_text())
            self.assertEqual(progress["global_pool"], "moments")
            self.assertEqual(progress["moment_pool_parameter_count"], 8 * 192)
            cfg = json.loads((run / "settings.json").read_text())
            state = torch.load(run / "model.pt", weights_only=True)
            restored = build_model(cfg, "cpu", state)
            self.assertIsNotNone(restored.moment_pool)
            initial = torch.load(run / "model_init.pt", weights_only=True)
            self.assertFalse(initial["moment_pool.weight"].any())
            self.assertTrue(state["moment_pool.weight"].any())


if __name__ == "__main__":
    unittest.main()
