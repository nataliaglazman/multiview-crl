"""Attention pooling: exact GAP start, position recovery, gradients, checkpoint replay and eval paths."""

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
from eval.lesion.checkpoint_lesion_analysis import batch_features
from eval.protocol.score_checkpoint import build_model
from models.attention_pool import AttentionPool3d, fourier_positions

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from scripts import run_encoder_mps as runner  # noqa: E402
from training import main_conv_synthetic as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    dict(encoder_architecture="conv"),
    dict(encoder_architecture="conv", conv_readout="mlp"),
    dict(encoder_architecture="resnet18", resnet_output_stride=8),
)


def config(**changes):
    # Width 64 gives 2 channels per GroupNorm group. At 1 channel per group the
    # conv GAP is ~0 for every subject at initialization, so no gradient reaches the pool.
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


class AttentionPoolModuleTests(unittest.TestCase):
    def test_soft_argmax_recovers_the_position_that_gap_cancels(self):
        # A fixed "lesion" feature moves between two positions of an otherwise empty map.
        # GAP and content-only attention are blind to the move; with positions, a head
        # locked onto the blob reports W_pos phi(position) in its channel slice.
        def blob(position):
            h = torch.zeros(1, 8, 4, 4, 4)
            h[0, 0, position[0], position[1], position[2]] = 50.0
            return h

        a, b = blob((1, 1, 1)), blob((2, 3, 0))
        torch.testing.assert_close(a.mean((2, 3, 4)), b.mean((2, 3, 4)), atol=0, rtol=0)
        located = AttentionPool3d(8, num_heads=2, num_frequencies=2)
        content_only = AttentionPool3d(8, num_heads=2, num_frequencies=0)
        with torch.no_grad():
            for pool in (located, content_only):
                pool.query.copy_(20.0 * F.one_hot(torch.tensor([0, 0]), 8))
            # Positions only enter head 1's channels, so head 0 reports pure content.
            located.position[4:] = torch.randn(4, 12, generator=torch.Generator().manual_seed(0))
            phi = fourier_positions((4, 4, 4), 2)
            for h, position in ((a, (1, 1, 1)), (b, (2, 3, 0))):
                pooled = located(h)[0]
                index = np.ravel_multi_index(position, (4, 4, 4))
                torch.testing.assert_close(pooled[:4], h[0, :4].flatten(1)[:, index], atol=1e-4, rtol=0)
                torch.testing.assert_close(pooled[4:], (located.position @ phi[index])[4:], atol=1e-4, rtol=0)
            self.assertGreater((located(a) - located(b)).norm().item(), 0.5)
            torch.testing.assert_close(content_only(a), content_only(b), atol=1e-5, rtol=0)

    def test_zero_parameters_give_uniform_maps_and_gap(self):
        pool = AttentionPool3d(16, num_heads=4, num_frequencies=3)
        h = torch.randn(3, 16, 2, 3, 5)
        maps = pool.maps(h)
        self.assertEqual(maps.shape, (3, 4, 2, 3, 5))
        torch.testing.assert_close(maps, torch.full_like(maps, 1 / 30))
        torch.testing.assert_close(pool(h), h.mean((2, 3, 4)))
        with torch.no_grad():
            pool.query.normal_()
            pool.position.normal_()
        torch.testing.assert_close(pool.maps(h).flatten(2).sum(-1), torch.ones(3, 4))

    def test_fourier_features_are_distinct_and_monotone_along_each_axis(self):
        phi = fourier_positions((5, 4, 3), 3)
        self.assertEqual(phi.shape, (60, 18))
        self.assertEqual(torch.unique(phi, dim=0).shape[0], 60)
        lowest_x = phi[:, 0].reshape(5, 4, 3)[:, 0, 0]  # sin(pi x / 2) at the cell centres
        self.assertTrue(torch.all(lowest_x[1:] > lowest_x[:-1]))
        torch.testing.assert_close(lowest_x, -lowest_x.flip(0))

    def test_invalid_shapes_fail(self):
        for channels, heads, frequencies in ((10, 4, 4), (8, 0, 4), (8, 2, -1)):
            with self.subTest(channels=channels, heads=heads, frequencies=frequencies):
                with self.assertRaises(ValueError):
                    AttentionPool3d(channels, heads, frequencies)
        with self.assertRaisesRegex(ValueError, "Expected 8 channels"):
            AttentionPool3d(8, 2)(torch.zeros(1, 4, 2, 2, 2))


class AttentionPoolEncoderTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def test_initial_model_matches_gap_model_without_drawing_random_numbers(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                size = 32 if variant["encoder_architecture"] == "resnet18" else 16
                gap = build_model(config(**variant), "cpu")
                rng = torch.get_rng_state()
                pooled = build_model(config(global_pool="attention", **variant), "cpu")
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertIsNone(gap.attention_pool)
                state, gap_state = pooled.state_dict(), gap.state_dict()
                self.assertEqual(set(state) - set(gap_state), {"attention_pool.query", "attention_pool.position"})
                for key, value in gap_state.items():
                    torch.testing.assert_close(state[key], value, atol=0, rtol=0)
                x = paired_images(size)
                with torch.no_grad():
                    torch.testing.assert_close(
                        pooled(x, pool_only=True, n_views=2)[2][0], gap(x, pool_only=True, n_views=2)[2][0]
                    )
                    # Patch and unpooled readouts never touch the global pool.
                    for kwargs in (dict(pool_only=True, patch_grid=[2, 2, 2]), dict(pool_only=False)):
                        torch.testing.assert_close(
                            pooled(x, n_views=2, **kwargs)[2][0], gap(x, n_views=2, **kwargs)[2][0], atol=0, rtol=0
                        )
                    for actual, expected in zip(
                        pooled.global_and_patch_features(x, [2, 2, 2]), gap.global_and_patch_features(x, [2, 2, 2])
                    ):
                        torch.testing.assert_close(actual, expected)
                    maps = pooled.attention_maps(x, n_views=2)
                torch.testing.assert_close(maps, torch.full_like(maps, 1 / maps[0, 0].numel()))
                with self.assertRaisesRegex(ValueError, "no attention maps"):
                    gap.attention_maps(x, n_views=2)

    def test_contrastive_gradient_moves_attention_off_gap_and_checkpoint_replays(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                size = 32 if variant["encoder_architecture"] == "resnet18" else 16
                cfg = config(global_pool="attention", **variant)
                model = build_model(cfg, "cpu").train()
                x = paired_images(size)
                infonce(model, x).backward()
                for parameter in (model.attention_pool.query, model.attention_pool.position):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
                for _ in range(3):
                    optimizer.step()
                    optimizer.zero_grad()
                    infonce(model, x).backward()
                model.eval()
                with torch.no_grad():
                    maps = model.attention_maps(x, n_views=2)
                self.assertGreater((maps - 1 / maps[0, 0].numel()).abs().max().item(), 1e-4)
                restored = build_model(cfg, "cpu", model.state_dict())
                with torch.no_grad():
                    torch.testing.assert_close(
                        restored(x, pool_only=True, n_views=2)[2][0],
                        model(x, pool_only=True, n_views=2)[2][0],
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(restored.attention_maps(x, n_views=2), maps, atol=0, rtol=0)
                gap_cfg = config(**variant)
                with self.assertRaises(RuntimeError):
                    build_model(gap_cfg, "cpu", model.state_dict())
                with self.assertRaises(RuntimeError):
                    build_model(cfg, "cpu", build_model(gap_cfg, "cpu").state_dict())

    def test_eval_readers_take_the_attention_pooled_code(self):
        model = build_model(config(global_pool="attention", conv_readout="mlp"), "cpu")
        with torch.no_grad():
            model.attention_pool.query.normal_(generator=torch.Generator().manual_seed(1))
            model.attention_pool.position.normal_(generator=torch.Generator().manual_seed(2))
        model.eval()
        x = paired_images()
        with torch.no_grad():
            h = model._encode(x, 2, None)
            pooled = model.attention_pool(h)
            code = model(x, pool_only=True, n_views=2)[2][0]
            patches = model(x, pool_only=True, n_views=2, patch_grid=[2, 2, 2])[2][0]
        self.assertGreater((pooled - h.mean((2, 3, 4))).abs().max().item(), 1e-3)
        stages = capture_batch(model, x)
        np.testing.assert_allclose(stages["backbone"], pooled.numpy(), atol=1e-6)
        np.testing.assert_allclose(stages["content"], code[:, :9].numpy(), atol=1e-6)
        features, _, _ = batch_features(model, x, [1, 2])
        np.testing.assert_allclose(features[(1, "projected")], code[:, :9].numpy(), atol=1e-6)
        np.testing.assert_allclose(features[(2, "projected")], patches[:, :9].flatten(1).numpy(), atol=1e-6)

    def test_cli_validates_attention_options(self):
        args = trainer.parse_args([])
        self.assertEqual((args.global_pool, args.attention_pool_heads, args.attention_pool_frequencies), ("gap", 4, 4))
        args = trainer.parse_args(["--global-pool", "attention", "--attention-pool-heads", "8"])
        self.assertEqual((args.global_pool, args.attention_pool_heads), ("attention", 8))
        trainer.parse_args(["--global-pool", "attention", "--encoder-architecture", "resnet18", "--res", "64"])
        trainer.parse_args(["--global-pool", "attention", "--attention-pool-frequencies", "0"])
        for argv in (
            ["--global-pool", "attention", "--attention-pool-heads", "3"],
            ["--global-pool", "attention", "--attention-pool-heads", "0"],
            ["--global-pool", "attention", "--attention-pool-frequencies", "-1"],
            ["--attention-pool-heads", "8"],
            ["--attention-pool-frequencies", "2"],
            ["--global-pool", "attention", "--encoder-architecture", "resnet18", "--res", "32"],
            ["--global-pool", "attention", "--res", "4"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(argv)

    def test_local_runner_passes_attention_options_through(self):
        def runner_args(**changes):
            values = dict(
                config=ROOT / "experiments/encoder_comparison.json",
                variant="conv_mlp",
                seed=42,
                batch_size=None,
                train_steps=None,
                eval_every=None,
                model_id=None,
                results_dir=Path("/tmp/encoder_ablations_mps"),
            )
            values.update(changes)
            return types.SimpleNamespace(**values)

        options, recipe = runner.make_options(runner_args(global_pool="attention", attention_pool_heads=8))
        self.assertEqual(options["model_id"], "conv_mlp_s42_mps_attnpool_h8_f4")
        self.assertNotIn("global_pool", recipe)
        parsed = trainer.parse_args(runner.comparison.cli_arguments(options))
        self.assertEqual(
            (parsed.global_pool, parsed.attention_pool_heads, parsed.attention_pool_frequencies), ("attention", 8, 4)
        )
        self.assertNotIn("global_pool", runner.make_options(runner_args())[0])
        for changes in (dict(attention_pool_heads=8), dict(attention_pool_frequencies=0)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "require --global-pool attention"):
                runner.make_options(runner_args(**changes))

    def test_real_training_logs_attention_spread_and_writes_replayable_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                *("--out-dir", tmp, "--model-id", "attention_smoke", "--require-new-run", "--device", "cpu"),
                *("--res", "16", "--hidden-channels", "64", "--res-channels", "4", "--nb-res-layers", "1"),
                *("--latent-dim", "12", "--conv-readout", "mlp", "--encoder-head-hidden", "8"),
                *("--global-pool", "attention", "--attention-pool-heads", "2", "--attention-pool-frequencies", "2"),
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
            run = Path(tmp) / "attention_smoke"
            self.assertIn("global pool: attention replaces GAP; 2 heads x 32 channels", output.getvalue())
            self.assertIn("global attention: effective positions", output.getvalue())
            floor = json.loads((run / "dci_step0.json").read_text())["attention_effective_fraction"]
            trained = json.loads((run / "dci_step2.json").read_text())["attention_effective_fraction"]
            for view in ("t1", "flair"):
                self.assertEqual(len(floor[view]), 2)
                for value in floor[view]:
                    self.assertAlmostEqual(value, 1.0, places=6)
                self.assertTrue(all(0 < value < 0.9999 for value in trained[view]))
            progress = json.loads((run / "training_progress.json").read_text())
            self.assertEqual(progress["global_pool"], "attention")
            self.assertEqual(progress["attention_pool_parameter_count"], 2 * 64 + 64 * 12)
            cfg = json.loads((run / "settings.json").read_text())
            state = torch.load(run / "model.pt", weights_only=True)
            restored = build_model(cfg, "cpu", state)
            self.assertEqual(restored.attention_pool.num_heads, 2)
            initial = torch.load(run / "model_init.pt", weights_only=True)
            self.assertFalse(torch.equal(state["attention_pool.query"], initial["attention_pool.query"]))
            self.assertFalse(initial["attention_pool.query"].any())


if __name__ == "__main__":
    unittest.main()
