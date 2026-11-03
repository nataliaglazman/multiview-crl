"""Independent residual content: legacy compatibility, export layout and protected image path."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eval.protocol.score_checkpoint import build_floor_model, build_model, encode_blocks
from training import encoder_args


def config(**changes):
    return dict(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=0,
        downscale_factor=4,
        latent_dim=7,
        content_channels=4,
        conv_readout="mlp",
        encoder_head_hidden=8,
        norm_type="layer",
        res=16,
        seed=42,
        **changes,
    )


def enabled_config(**changes):
    settings = config(
        residual_head_units=3,
        residual_head_hidden=4,
        residual_head_components=2,
        residual_head_joint_dim=4,
        residual_head_freeze_image=True,
    )
    settings.update(changes)
    return settings


def fit(model):
    images = torch.randn(8, 2, 16, 16, 16, generator=torch.Generator().manual_seed(17))
    model.residual_heads.fit(images)
    return images


class Subjects(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": (self.images[index, :1], self.images[index, 1:]),
            "gt_latents": {"z_content": torch.tensor([float(index), 1.0])},
        }


class ResidualHeadCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_disabled_settings_preserve_legacy_state_rng_and_global_features(self):
        legacy = build_model(config(), "cpu")
        expected_rng = torch.get_rng_state()
        disabled = build_model(config(residual_head_units=0), "cpu", legacy.state_dict())
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertIsNone(disabled.residual_heads)
        self.assertFalse(any(key.startswith("residual_heads.") for key in disabled.state_dict()))
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        images = torch.randn(4, 1, 16, 16, 16)
        with torch.no_grad():
            expected = legacy(images, pool_only=True, n_views=2)[2][0]
            actual = disabled(images, pool_only=True, n_views=2)[2][0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_strict_checkpoint_restores_residual_fit_and_exact_global_outputs(self):
        original = build_model(enabled_config(), "cpu")
        images = fit(original)
        original.eval()
        batch = torch.cat([images[:3, :1], images[:3, 1:]])
        with torch.no_grad():
            expected = original(batch, pool_only=True, n_views=2)[2][0]
        stream = io.BytesIO()
        torch.save(original.state_dict(), stream)
        stream.seek(0)
        restored = build_model(enabled_config(), "cpu", torch.load(stream, weights_only=True))
        self.assertTrue(restored.residual_heads.fitted)
        with torch.no_grad():
            actual = restored(batch, pool_only=True, n_views=2)[2][0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(set(restored.state_dict()), set(original.state_dict()))
        for name, buffer in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], buffer, rtol=0, atol=0)

    def test_floor_requires_saved_fitted_initial_model(self):
        model = build_model(enabled_config(), "cpu")
        fit(model)
        with tempfile.TemporaryDirectory() as run_dir:
            with self.assertRaisesRegex(ValueError, "model_init.pt"):
                build_floor_model(enabled_config(), "cpu", run_dir)
            torch.save(model.state_dict(), Path(run_dir) / "model_init.pt")
            floor = build_floor_model(enabled_config(), "cpu", run_dir)
            for name, expected in model.state_dict().items():
                torch.testing.assert_close(floor.state_dict()[name], expected, rtol=0, atol=0)

    def test_image_style_then_residual_layout_and_view_routing_and_global_only_eval(
        self,
    ):
        base = build_model(config(), "cpu")
        model = build_model(enabled_config(), "cpu")
        images = fit(model)[:4]
        model.eval()
        batch = torch.cat([images[:, :1], images[:, 1:]])
        with torch.no_grad():
            output = model(batch, pool_only=True, n_views=2)
            code = output[2][0]
            expected_image = base(batch, pool_only=True, n_views=2)[2][0]
            first = model(images[:, :1], pool_only=True, view_idx=0)[2][0]
            second = model(images[:, 1:], pool_only=True, view_idx=1)[2][0]
        self.assertEqual(code.shape, (8, 10))
        torch.testing.assert_close(code[:, :7], expected_image, rtol=0, atol=0)
        torch.testing.assert_close(code, torch.cat([first, second]), rtol=0, atol=0)
        self.assertEqual(model.unit_split(10), ([0, 1, 2, 3, 7, 8, 9], [4, 5, 6]))
        self.assertEqual(model.content_units(), [0, 1, 2, 3, 7, 8, 9])
        self.assertEqual(
            model.content_units(),
            torch.nonzero(model.content_mask[0]).flatten().tolist(),
        )
        self.assertEqual(model.unit_split(7), ([0, 1, 2, 3], [4, 5, 6]))
        torch.testing.assert_close(
            output[6][0],
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        )
        raw, residual = model.split_residual_codes(code)
        torch.testing.assert_close(raw, code[:, :4])
        torch.testing.assert_close(residual, code[:, 7:])
        blocks = encode_blocks(model, Subjects(images), "cpu", 2, 4)
        np.testing.assert_allclose(blocks["content"][0], first[:, [0, 1, 2, 3, 7, 8, 9]].numpy(), atol=3e-7)
        np.testing.assert_allclose(blocks["style"][1], second[:, 4:7].numpy(), atol=3e-7)
        with self.assertRaisesRegex(ValueError, "global codes"):
            encode_blocks(model, Subjects(images), "cpu", 2, 4, patch_grid=(1, 1, 1))

    def test_frozen_image_path_keeps_parameters_gradients_and_running_statistics(self):
        model = build_model(enabled_config(), "cpu")
        images = fit(model)
        # The supported conv path has no running statistics; a tracking layer verifies the
        # protection also covers normalization buffers when a checkpoint path adds one.
        model.encoder = torch.nn.Sequential(model.encoder, torch.nn.BatchNorm3d(8))
        model.encoder_v1 = torch.nn.Sequential(model.encoder_v1, torch.nn.BatchNorm3d(8))
        for parameter in model.image_parameters():
            parameter.requires_grad_(False)
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.residual_heads.training)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.encoder_v1.training)
        self.assertFalse(model.to_encoding.training)
        before = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("residual_heads.")
        }
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        batch = torch.cat([images[:3, :1], images[:3, 1:]])
        code = model(batch, pool_only=True, n_views=2)[2][0]
        code[:, -3:].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.image_parameters()))
        self.assertTrue(any(p.grad is not None for p in model.residual_heads.parameters()))
        optimizer.step()
        for name, expected in before.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_unfrozen_paths_both_receive_gradients(self):
        model = build_model(enabled_config(residual_head_freeze_image=False), "cpu")
        self.assertFalse(model.residual_heads.frozen_image)
        images = fit(model)
        model.train()
        code = model(torch.cat([images[:3, :1], images[:3, 1:]]), pool_only=True, n_views=2)[2][0]
        raw, residual = model.split_residual_codes(code)
        (raw.square().mean() + residual.square().mean()).backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.encoder_v1.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.residual_heads.parameters()))

    def test_cli_requires_explicit_joint_dimension_and_protected_initialization(self):
        enabled = ["--conv-readout", "mlp", "--residual-head-units", "3"]
        valid = [
            *enabled,
            "--residual-head-joint-dim",
            "4",
            "--residual-head-image-init",
            "/tmp/image.pt",
        ]
        args = encoder_args.parse_args(valid)
        self.assertTrue(args.residual_head_freeze_image)
        self.assertEqual(args.residual_head_joint_dim, 4)
        fine_tune = encoder_args.parse_args(
            [
                *enabled,
                "--residual-head-joint-dim",
                "4",
                "--no-residual-head-freeze-image",
            ]
        )
        self.assertFalse(fine_tune.residual_head_freeze_image)
        encoder_args.parse_args([*enabled, "--residual-head-joint-dim", "4", "--init-from", "/tmp/full.pt"])
        invalid = (
            enabled,
            [*enabled, "--residual-head-joint-dim", "4"],
            [*valid, "--residual-head-joint-dim", "13"],
            [*valid, "--eval-pooling", "patch"],
            [*valid, "--spatial-recovery-eval"],
            [
                *valid,
                "--causal-eval",
                "--synthetic-causal",
                "--causal-eval-grid",
                "2",
                "2",
                "2",
            ],
            [*valid, "--residual-head-uniformity-weight", "0"],
            [
                *valid,
                "--residual-head-subjects",
                "16",
                "--residual-head-components",
                "16",
            ],
            ["--residual-head-joint-dim", "4"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                encoder_args.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
