"""Lesion keypoint branch: equivariant coordinates, brain frame, unchanged base model, content layout, trainer path."""

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

from eval.lesion.checkpoint_lesion_analysis import batch_features
from eval.protocol.score_checkpoint import build_model, encode_blocks
from models.keypoint_pool import KeypointPool3d, brain_frame, cell_centres

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


def ball(size, radius, centre=(0.0, 0.0, 0.0), soft=0.0):
    axis = (torch.arange(size) * 2 + 1) / size - 1
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    distance = (grid - torch.tensor(centre)).norm(dim=-1)
    if soft:
        return torch.sigmoid((radius - distance) / soft)
    return (distance < radius).float()


def brain_images(size=16):
    """Two subjects x two views, exactly zero outside a ball so the brain frame has a support."""
    x = torch.randn(4, 1, size, size, size, generator=torch.Generator().manual_seed(3)) * ball(size, 0.7)
    x[2:] = 1.5 * x[2:] + 0.2 * ball(size, 0.7)
    return x


def blob_map(position, size=8, channels=4, value=50.0):
    h = torch.zeros(1, channels, size, size, size)
    h[0, 0, position[0], position[1], position[2]] = value
    return h


def detector(pool, channel=0, weight=1.0):
    with torch.no_grad():
        pool.logits.weight.zero_()
        pool.logits.weight[:, channel] = weight
        pool.logits.bias.zero_()


class KeypointPoolModuleTests(unittest.TestCase):
    def test_coordinates_follow_a_blob_that_gap_cannot_see(self):
        a, b = blob_map((1, 2, 3)), blob_map((2, 4, 3))
        torch.testing.assert_close(a.mean((2, 3, 4)), b.mean((2, 3, 4)), atol=0, rtol=0)
        pool = KeypointPool3d(4, num_keypoints=2, frame="grid")
        detector(pool)
        centres = cell_centres((8, 8, 8))
        for h, position in ((a, (1, 2, 3)), (b, (2, 4, 3))):
            coords = pool(h).view(2, 3)
            expected = centres[torch.tensor(position) @ torch.tensor([64, 8, 1])]
            torch.testing.assert_close(coords, expected.expand(2, 3), atol=1e-4, rtol=0)
        # Moving the blob by (1, 2, 0) cells moves the coordinate by that many cell widths (2 / 8).
        torch.testing.assert_close(pool(b) - pool(a), torch.tensor([[0.25, 0.5, 0.0] * 2]), atol=1e-4, rtol=0)

    def test_brain_frame_tracks_brain_scale_and_position(self):
        grid = cell_centres((32, 32, 32))
        small = ball(32, 0.4, soft=0.02)[None, None]
        large = ball(32, 0.6, soft=0.02)[None, None]
        shifted = ball(32, 0.4, centre=(0.2, 0.0, 0.0), soft=0.02)[None, None]
        centre_s, spread_s = brain_frame(small, grid)
        centre_l, spread_l = brain_frame(large, grid)
        centre_t, spread_t = brain_frame(shifted, grid)
        torch.testing.assert_close(centre_s, torch.zeros(1, 3), atol=1e-4, rtol=0)
        torch.testing.assert_close(spread_l / spread_s, torch.full((1, 3), 1.5), atol=0.03, rtol=0)
        torch.testing.assert_close(centre_t, torch.tensor([[0.2, 0.0, 0.0]]), atol=1e-3, rtol=0)
        torch.testing.assert_close(spread_t, spread_s, atol=1e-3, rtol=0)

    def test_brain_frame_reports_the_same_coordinate_for_a_scaled_brain(self):
        # A keypoint at 0.625 R along x for brains of radius 0.45 and 0.75 on a 32^3 grid: both
        # points are cell centres (9/32 and 15/32). The absolute coordinate differs by 5/3, the
        # brain-frame one should not.
        pool = KeypointPool3d(4, num_keypoints=1, frame="brain")
        detector(pool)
        relative = []
        for radius, index in ((0.45, 20), (0.75, 23)):
            support = ball(32, radius, soft=0.01)[None, None]
            h = torch.zeros(1, 4, 32, 32, 32)
            h[0, 0, index, 16, 16] = 50.0
            relative.append(pool(h, support)[0, 0].item())
        self.assertAlmostEqual(relative[0], relative[1], delta=0.06)

    def test_an_unfocused_head_carries_no_brain_size_in_the_brain_frame(self):
        # Over the whole grid an even head reports the grid centre, whose brain-relative
        # position depends on the brain's size; confined to the brain it reports its centroid.
        pool = KeypointPool3d(4, num_keypoints=2, frame="brain")
        detector(pool, weight=0.0)
        h = torch.ones(1, 4, 32, 32, 32)
        for radius, centre in ((0.45, (0.0, 0.0, 0.0)), (0.75, (0.0, 0.0, 0.0)), (0.45, (0.2, -0.1, 0.0))):
            support = ball(32, radius, centre=centre, soft=0.01)[None, None]
            torch.testing.assert_close(pool(h, support), torch.zeros(1, 6), atol=1e-4, rtol=0)
            maps = pool.maps(h, support)
            self.assertLess(maps[0, 0][support[0, 0] < 1e-3].sum().item(), 1e-3)

    def test_temperature_sharpens_the_heads_without_scaling_the_brain_mask(self):
        h = torch.randn(1, 6, 8, 8, 8, generator=torch.Generator().manual_seed(6))
        support = ball(8, 0.8, soft=0.05)[None, None]
        spreads = []
        for temperature in (1.0, 0.1, 0.01):
            pool = KeypointPool3d(6, num_keypoints=2, frame="brain", temperature=temperature)
            with torch.no_grad():
                pool.logits.weight.copy_(torch.randn(2, 6, 1, 1, 1, generator=torch.Generator().manual_seed(7)) * 0.05)
            maps = pool.maps(h, support).flatten(2)
            spreads.append(torch.special.entr(maps).sum(-1).exp().mean().item())
        self.assertGreater(spreads[0], spreads[1])
        self.assertGreater(spreads[1], spreads[2])
        # Uniform logits: a temperature must not turn partial-volume edge cells into a hard mask.
        flat = KeypointPool3d(6, num_keypoints=1, frame="brain", temperature=0.01)
        detector(flat, weight=0.0)
        torch.testing.assert_close(flat.maps(h, support).flatten(), support.flatten() / support.sum())
        for temperature in (0.0, -1.0, float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                KeypointPool3d(6, temperature=temperature)

    def test_layer_norm_removes_a_per_voxel_gain(self):
        h = torch.randn(2, 6, 4, 4, 4, generator=torch.Generator().manual_seed(1))
        normed = KeypointPool3d(6, num_keypoints=3, norm="layer", frame="grid")
        raw = KeypointPool3d(6, num_keypoints=3, norm="none", frame="grid")
        with torch.no_grad():
            raw.logits.weight.copy_(normed.logits.weight)
        torch.testing.assert_close(normed(3.0 * h), normed(h), atol=1e-4, rtol=0)
        self.assertGreater((raw(3.0 * h) - raw(h)).abs().max().item(), 1e-4)

    def test_random_init_gives_subject_dependent_coordinates(self):
        # Zero logits would put every subject at the grid centre, where the branch's InfoNCE has no gradient.
        pool = KeypointPool3d(6, num_keypoints=2, frame="grid")
        h = torch.randn(3, 6, 4, 4, 4, generator=torch.Generator().manual_seed(2))
        self.assertGreater(pool(h).std(0).min().item(), 1e-5)
        torch.testing.assert_close(pool.maps(h).flatten(2).sum(-1), torch.ones(3, 2))

    def test_invalid_options_fail(self):
        for kwargs in (dict(num_keypoints=0), dict(norm="batch"), dict(frame="image")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KeypointPool3d(4, **kwargs)
        with self.assertRaisesRegex(ValueError, "needs the brain support"):
            KeypointPool3d(4, frame="brain")(torch.zeros(1, 4, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "Expected 4 channels"):
            KeypointPool3d(4, frame="grid")(torch.zeros(1, 3, 2, 2, 2))


class LesionBranchEncoderTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def test_branch_leaves_the_base_model_unchanged_and_appends_content_units(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                size = 32 if variant["encoder_architecture"] == "resnet18" else 16
                base = build_model(config(**variant), "cpu")
                branched = build_model(config(lesion_keypoints=4, **variant), "cpu")
                state, base_state = branched.state_dict(), base.state_dict()
                self.assertEqual(
                    set(state) - set(base_state),
                    {
                        "lesion_pool.logits.weight",
                        "lesion_pool.logits.bias",
                        "lesion_projector.weight",
                        "lesion_projector.bias",
                    },
                )
                for key, value in base_state.items():
                    if key != "content_mask":
                        torch.testing.assert_close(state[key], value, atol=0, rtol=0)
                x = brain_images(size)
                with torch.no_grad():
                    out, base_out = branched(x, pool_only=True, n_views=2), base(x, pool_only=True, n_views=2)
                    code, base_code = out[2][0], base_out[2][0]
                    self.assertEqual(code.shape, (4, 12 + 12))
                    torch.testing.assert_close(code[:, :12], base_code, atol=0, rtol=0)
                    self.assertEqual(out[3], [list(range(9)) + list(range(12, 24))])
                    self.assertEqual(
                        torch.nonzero(out[6][0][0]).flatten().tolist(), list(range(9)) + list(range(12, 24))
                    )
                    # Patch and unpooled readouts have no lesion units and keep the old content split.
                    for kwargs in (dict(pool_only=True, patch_grid=[2, 2, 2]), dict(pool_only=False)):
                        patched, base_patched = branched(x, n_views=2, **kwargs), base(x, n_views=2, **kwargs)
                        torch.testing.assert_close(patched[2][0], base_patched[2][0], atol=0, rtol=0)
                        self.assertEqual(patched[3], base_patched[3])
                        torch.testing.assert_close(patched[6][0], base_patched[6][0], atol=0, rtol=0)
                    pooled, patches = branched.global_and_patch_features(x, [2, 2, 2])
                    base_pooled, base_patches = base.global_and_patch_features(x, [2, 2, 2])
                    torch.testing.assert_close(pooled[:, :12], base_pooled, atol=1e-6, rtol=0)
                    torch.testing.assert_close(pooled, code, atol=1e-6, rtol=0)
                    torch.testing.assert_close(patches, base_patches, atol=0, rtol=0)
                    maps = branched.lesion_maps(x, n_views=2)
                self.assertEqual(maps.shape[:2], (4, 4))
                with self.assertRaisesRegex(ValueError, "no lesion branch"):
                    base.lesion_maps(x, n_views=2)

    def test_only_the_lesion_loss_trains_the_branch_and_checkpoints_replay(self):
        cfg = config(conv_readout="mlp", lesion_keypoints=3)
        model = build_model(cfg, "cpu").train()
        args = trainer.parse_args(
            ["--conv-readout", "mlp", "--latent-dim", "12", "--tau", "0.1", "--lesion-keypoints", "3"]
        )
        x = brain_images()
        sim, ce = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        pooled, total, terms = trainer.training_objective(model, x, args, sim, ce)
        self.assertEqual(set(terms), {"global", "patch", "patch_weighted", "lesion", "lesion_weighted"})
        torch.testing.assert_close(total, terms["global"] + terms["lesion"])
        terms["global"].backward(retain_graph=True)
        untouched = model.lesion_pool.logits.weight.grad
        self.assertTrue(untouched is None or not untouched.any())
        total.backward()
        for parameter in (model.lesion_pool.logits.weight, model.lesion_projector.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        restored = build_model(cfg, "cpu", model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(
                restored(x, pool_only=True, n_views=2)[2][0], model.eval()(x, pool_only=True, n_views=2)[2][0]
            )
        with self.assertRaises(RuntimeError):
            build_model(config(conv_readout="mlp"), "cpu", model.state_dict())

    def test_lesion_loss_sees_the_coordinates_at_unit_scale(self):
        # An untrained head moves its coordinates by ~1e-2 between subjects; the projector must not care.
        model = build_model(config(lesion_keypoints=2), "cpu")
        coords = torch.randn(2, 6, 6, generator=torch.Generator().manual_seed(4))
        scale, shift = torch.rand(6) * 0.01 + 1e-3, torch.randn(6)
        # float32 keeps ~4 digits of a 1e-3 variation on an O(1) offset, hence the tolerance.
        torch.testing.assert_close(
            model.project_lesion(coords * scale + shift), model.project_lesion(coords), atol=1e-3, rtol=1e-3
        )

    def test_scorers_count_the_keypoints_as_content_not_style(self):
        model = build_model(config(conv_readout="mlp", lesion_keypoints=2), "cpu")
        x = brain_images()
        with torch.no_grad():
            code = model(x, pool_only=True, n_views=2)[2][0]
        content, style = torch.cat([code[:, :9], code[:, 12:]], dim=1).numpy(), code[:, 9:12].numpy()
        features, _, _ = batch_features(model, x, [1, 2])
        np.testing.assert_allclose(features[(1, "projected")], content, atol=1e-6)
        np.testing.assert_allclose(features[(1, "style")], style, atol=1e-6)
        self.assertEqual(features[(2, "projected")].shape[1], 9 * 8)
        subjects = [{"image": [x[i], x[2 + i]], "gt_latents": {"z_content": torch.zeros(9)}} for i in range(2)]
        blocks = encode_blocks(model, subjects, "cpu", batch_size=2, content_channels=9)
        for view in range(2):
            np.testing.assert_allclose(blocks["content"][view], content[2 * view : 2 * view + 2], atol=1e-6)
            np.testing.assert_allclose(blocks["style"][view], style[2 * view : 2 * view + 2], atol=1e-6)

    def test_local_runner_passes_lesion_options_through(self):
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

        options, recipe = runner.make_options(runner_args(lesion_keypoints=4, lesion_frame="grid"))
        self.assertEqual(options["model_id"], "conv_mlp_s42_mps_lesionkp4_gridframe")
        sharp, _ = runner.make_options(runner_args(lesion_keypoints=4, lesion_temperature=0.02))
        self.assertEqual(sharp["model_id"], "conv_mlp_s42_mps_lesionkp4_temp0.02")
        self.assertEqual(trainer.parse_args(runner.comparison.cli_arguments(sharp)).lesion_temperature, 0.02)
        within, _ = runner.make_options(runner_args(lesion_keypoints=4, lesion_pairing="within_modality"))
        self.assertEqual(within["model_id"], "conv_mlp_s42_mps_lesionkp4_lpwithin")
        parsed = trainer.parse_args(runner.comparison.cli_arguments(within))
        self.assertEqual(parsed.lesion_pairing, "within_modality")
        self.assertNotIn("lesion_keypoints", recipe)
        parsed = trainer.parse_args(runner.comparison.cli_arguments(options))
        self.assertEqual((parsed.lesion_keypoints, parsed.lesion_frame, parsed.lesion_norm), (4, "grid", "none"))
        self.assertNotIn("lesion_keypoints", runner.make_options(runner_args())[0])
        with self.assertRaisesRegex(ValueError, "require --lesion-keypoints"):
            runner.make_options(runner_args(lesion_norm="layer"))

    def test_decorrelation_moves_only_the_branch(self):
        model = build_model(config(conv_readout="mlp", lesion_keypoints=2), "cpu").train()
        argv = ["--conv-readout", "mlp", "--latent-dim", "12", "--tau", "0.1", "--lesion-keypoints", "2"]
        args = trainer.parse_args([*argv, "--lesion-decorrelation-weight", "2"])
        sim, ce = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        pooled, total, terms = trainer.training_objective(model, brain_images(), args, sim, ce)
        torch.testing.assert_close(
            total, terms["global"] + terms["lesion"] + 2 * terms["lesion_decorrelation"], atol=1e-5, rtol=0
        )
        terms["lesion_decorrelation"].backward()
        head = model.to_encoding[-1].weight.grad
        self.assertTrue(head is None or not head.any())
        self.assertGreater(model.lesion_pool.logits.weight.grad.abs().sum().item(), 0)
        # Zero for coordinates uncorrelated with the content block, as large as K x C for duplicates.
        content = torch.randn(64, 9, generator=torch.Generator().manual_seed(5))
        duplicate = torch.cat([content, torch.zeros(64, 3), content[:, :6]], dim=1)
        self.assertAlmostEqual(trainer.lesion_decorrelation(duplicate, args).item(), 6.0, delta=0.6)
        self.assertNotIn(
            "lesion_decorrelation",
            trainer.training_objective(model, brain_images(), trainer.parse_args(argv), sim, ce)[2],
        )

    def test_intensity_augmentation_keeps_the_background_and_the_lesion_polarity(self):
        x = ball(16, 0.7)[None, None].repeat(3, 1, 1, 1, 1)
        x[:, :, 8, 8, 8] = 3.0  # a bright "lesion" voxel inside the brain
        torch.manual_seed(0)
        a, b = trainer.augment_intensity(x), trainer.augment_intensity(x)
        outside = x == 0
        self.assertTrue(torch.equal(a[outside], torch.zeros_like(a[outside])))
        self.assertFalse(torch.equal(a, b))
        for view in (a, b):
            rest = view[(x != 0) & (x < 2)].view(3, -1).mean(1)
            self.assertTrue(torch.all(view[:, 0, 8, 8, 8] > rest + 1.0))

    def test_within_modality_pairs_are_two_flair_draws_through_the_flair_encoder(self):
        model = build_model(config(conv_readout="mlp", lesion_keypoints=2), "cpu").train()
        argv = ["--conv-readout", "mlp", "--latent-dim", "12", "--tau", "0.1", "--lesion-keypoints", "2"]
        args = trainer.parse_args([*argv, "--lesion-pairing", "within_modality"])
        x = brain_images()
        calls, original = [], model.lesion_code

        def spy(batch, view_idx=None):
            calls.append((view_idx, batch.detach().clone()))
            return original(batch, view_idx=view_idx)

        model.lesion_code = spy
        sim, ce = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        pooled, total, terms = trainer.training_objective(model, x, args, sim, ce)
        self.assertEqual(len(calls), 1)
        view_idx, batch = calls[0]
        self.assertEqual((view_idx, batch.shape[0]), (1, 4))
        outside = x[2:] == 0
        for half in (batch[:2], batch[2:]):
            self.assertTrue(torch.equal(half[outside], torch.zeros_like(half[outside])))
            self.assertFalse(torch.equal(half, x[2:]))
        self.assertFalse(torch.equal(batch[:2], batch[2:]))
        self.assertEqual(pooled.shape, (4, 12 + 6))
        total.backward()
        self.assertGreater(model.lesion_pool.logits.weight.grad.abs().sum().item(), 0)

    def test_cli_validates_lesion_options(self):
        args = trainer.parse_args([])
        self.assertEqual(
            (args.lesion_keypoints, args.lesion_norm, args.lesion_frame, args.lesion_proj_dim, args.lesion_loss_weight),
            (0, "none", "brain", 8, 1.0),
        )
        trainer.parse_args(["--lesion-keypoints", "4", "--lesion-norm", "layer", "--lesion-frame", "grid"])
        for argv in (
            ["--lesion-keypoints", "-1"],
            ["--lesion-norm", "layer"],
            ["--lesion-frame", "grid"],
            ["--lesion-loss-weight", "2"],
            ["--lesion-keypoints", "2", "--lesion-proj-dim", "0"],
            ["--lesion-keypoints", "2", "--lesion-loss-weight", "-1"],
            ["--lesion-keypoints", "2", "--lesion-decorrelation-weight", "-1"],
            ["--lesion-decorrelation-weight", "1"],
            ["--lesion-temperature", "0.1"],
            ["--lesion-pairing", "within_modality"],
            ["--lesion-keypoints", "2", "--lesion-temperature", "0"],
            ["--lesion-keypoints", "2", "--contrastive-loss-type", "barlow_twins"],
            ["--lesion-keypoints", "2", "--res", "4"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(argv)

    def test_real_training_reports_what_the_lesion_branch_encodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                *("--out-dir", tmp, "--model-id", "lesion_smoke", "--require-new-run", "--device", "cpu"),
                *("--res", "16", "--hidden-channels", "64", "--res-channels", "4", "--nb-res-layers", "1"),
                *("--latent-dim", "12", "--conv-readout", "mlp", "--encoder-head-hidden", "8"),
                *("--lesion-keypoints", "2", "--lesion-proj-dim", "4"),
                *("--tau", "0.1", "--lr", "0.05", "--batch-size", "2", "--train-steps", "2", "--eval-every", "2"),
                *("--log-every", "1", "--num-train-samples", "8", "--num-val-samples", "20", "--no-cache"),
                *("--best-metric", "none", "--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"),
            ]
            with patch.object(sys, "argv", ["trainer", *argv]), patch.object(
                trainer.dci, "compute_dci_synthetic", return_value={}
            ), patch.object(trainer.dci, "flatten_dci_results", return_value={}), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                trainer.main()
            run = Path(tmp) / "lesion_smoke"
            log = output.getvalue()
            self.assertIn(
                "lesion branch: 2 spatial-softmax keypoints (brain frame, norm none, temperature 1, cross_modal pairs)",
                log,
            )
            self.assertIn("lesion InfoNCE", log)
            self.assertIn("lesion branch alone: linear R² per content factor", log)
            report = json.loads((run / "dci_step2.json").read_text())["lesion_branch"]
            for view in ("t1", "flair"):
                self.assertIn("lesion_x", report["r2"][view])
                self.assertEqual(len(report["effective_fraction"][view]), 2)
            progress = json.loads((run / "training_progress.json").read_text())
            self.assertEqual(progress["lesion_branch_parameter_count"], (64 * 2 + 2) + (6 * 4 + 4))
            self.assertIn("lesion", progress["last_loss_terms"])
            cfg = json.loads((run / "settings.json").read_text())
            restored = build_model(cfg, "cpu", torch.load(run / "model.pt", weights_only=True))
            self.assertEqual(restored.lesion_pool.num_keypoints, 2)


if __name__ == "__main__":
    unittest.main()
