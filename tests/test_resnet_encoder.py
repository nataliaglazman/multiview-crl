"""Upstream architecture parity, pooling order, learning and checkpoint compatibility."""

import argparse
import ast
import copy
import gc
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eval.score_checkpoint import build_model, encode
from models.multiview_encoder import MultiviewConvEncoder
from models.vqvae import Encoder

ROOT = Path(__file__).resolve().parents[1]


def functions_from_file(relative_path, names, namespace):
    # Avoid unrelated perceptual-loss dependencies (LPIPS); execute the actual
    # parser / contrastive loss functions used by the training entry point.
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ResNetEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def tearDown(self):
        gc.collect()

    def model(self, **kwargs):
        config = dict(encoder_architecture="resnet18", latent_dim=12, content_channels=9, separate_encoders=False)
        config.update(kwargs)
        return MultiviewConvEncoder(**config)

    def test_matches_torchvision_resnet18_on_depth_constant_input(self):
        # A centre-plane inflation of every 2D kernel must reproduce the upstream
        # network on identical depth slices, including residual shortcuts and pooling.
        try:
            from torchvision.models import resnet18
        except ImportError:
            self.skipTest("torchvision is needed only for independent reference comparison")
        reference = nn.Sequential(resnet18(weights=None, num_classes=100), nn.LeakyReLU(), nn.Linear(100, 12))
        reference[0].conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        reference.eval()
        model = self.model().eval()
        sources = dict(reference[0].named_modules())
        with torch.no_grad():
            for name, module in model.encoder.named_modules():
                if isinstance(module, nn.Conv3d):
                    module.weight.zero_()
                    module.weight[:, :, module.kernel_size[0] // 2].copy_(sources[name].weight)
                    self.assertEqual(module.stride[-2:], sources[name].stride)
                    self.assertEqual(module.padding[-2:], sources[name].padding)
                elif isinstance(module, nn.BatchNorm3d):
                    module.load_state_dict(sources[name].state_dict())
            model.to_encoding[0].load_state_dict(reference[0].fc.state_dict())
            model.to_encoding[2].load_state_dict(reference[2].state_dict())
            x2d = torch.randn(2, 1, 64, 64)
            x3d = x2d.unsqueeze(2).expand(-1, -1, 32, -1, -1).contiguous()
            actual = model(x3d, pool_only=True)[2][0]
            torch.testing.assert_close(actual, reference(x2d), atol=2e-5, rtol=1e-4)
        self.assertEqual([len(getattr(model.encoder, f"layer{i}")) for i in range(1, 5)], [2, 2, 2, 2])

    def test_gap_precedes_nonlinear_head_and_patch_semantics(self):
        model = self.model().eval()
        with torch.no_grad():
            x = torch.randn(1, 1, 64, 64, 64)
            h = model.encoder(x)
            self.assertEqual(h.shape, (1, 512, 2, 2, 2))
            actual = model(x, pool_only=True)[2][0]
            torch.testing.assert_close(actual, model.to_encoding(h.mean((2, 3, 4))))
            patches = model(x, pool_only=True, patch_grid=[2, 2, 2])[2][0]
            self.assertEqual(patches.shape, (1, 12, 8))
            torch.testing.assert_close(patches, model.to_encoding(h.flatten(2).transpose(1, 2)).transpose(1, 2))
            torch.testing.assert_close(actual, model(x, pool_only=True, patch_grid=[1, 1, 1])[2][0].squeeze(-1))
            self.assertEqual(model(x, pool_only=False)[2][0].shape, (1, 12, 2, 2, 2))
            for grid in ([4, 4, 4], [0, 1, 1], []):
                with self.assertRaisesRegex(ValueError, "must fit its spatial map"):
                    model(x, pool_only=True, patch_grid=grid)

            # Deterministic counterexample: averaging after LeakyReLU is different.
            synthetic_map = torch.zeros(1, 512, 1, 1, 2)
            synthetic_map[0, 0, 0, 0] = torch.tensor([-1.0, 1.0])
            for layer in (model.to_encoding[0], model.to_encoding[2]):
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight[0, 0] = 1.0
            with patch.object(model, "_encode", return_value=synthetic_map):
                global_code = model(x, pool_only=True)[2][0]
                spatial_codes = model(x, pool_only=False)[2][0]
            self.assertEqual(float(global_code[0, 0]), 0.0)
            self.assertAlmostEqual(float(spatial_codes[0, 0].mean()), 0.495, places=6)

    def test_infonce_updates_both_backbones_and_readout(self):
        loss_ns = functions_from_file(
            "training/losses.py", {"_merge_diags", "infonce_loss", "infonce_base_loss"}, {"torch": torch}
        )
        train_ns = functions_from_file(
            "training/main_conv_synthetic.py",
            {"contrastive_loss"},
            {"torch": torch, "losses": SimpleNamespace(infonce_loss=loss_ns["infonce_loss"])},
        )
        model = self.model(separate_encoders=True).train()
        args = SimpleNamespace(content_channels=9, contrastive_loss_type="infonce", tau=0.1, cross_view_negs_only=True)
        x = torch.randn(4, 1, 32, 32, 32)
        old_weight = model.encoder.conv1.weight.detach().clone()
        # Differentiate the two views after their identical initialization.
        x[2:] = x[2:] * 1.5 + 0.25
        codes = model(x, pool_only=True, n_views=2)[2][0]
        loss = train_ns["contrastive_loss"](codes, model, args, nn.CosineSimilarity(dim=-1), nn.CrossEntropyLoss())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (
            model.encoder.conv1.weight,
            model.encoder_v1.conv1.weight,
            model.to_encoding[0].weight,
            model.to_encoding[2].weight,
        ):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        # Only the content coordinates have a direct contrastive target.
        self.assertEqual(float(model.to_encoding[2].weight.grad[9:].abs().sum()), 0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # training entry point's default
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        self.assertFalse(torch.equal(old_weight, model.encoder.conv1.weight))
        self.assertEqual(int(model.encoder.bn1.num_batches_tracked), 1)
        self.assertEqual(int(model.encoder_v1.bn1.num_batches_tracked), 1)
        model.eval()
        with torch.no_grad():
            paired = model(x, pool_only=True, n_views=2)[2][0]
            for view in (0, 1):
                alone = model(x[view * 2 : (view + 1) * 2], pool_only=True, view_idx=view)[2][0]
                torch.testing.assert_close(paired[view * 2 : (view + 1) * 2], alone, atol=1e-6, rtol=1e-5)

    def test_saved_settings_rebuild_trained_and_untrained_evaluation(self):
        cfg = dict(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            downscale_factor=4,
            latent_dim=12,
            content_channels=9,
            no_separate_encoders=True,
            encoder_architecture="resnet18",
            encoder_head_hidden=37,
            contrastive_proj_dim=5,
            contrastive_proj_hidden=11,
            seed=17,
        )
        model = build_model(cfg, "cpu")
        self.assertEqual(model.to_encoding[0].out_features, 37)
        self.assertEqual(model.project(torch.zeros(2, 9)).shape, (2, 5))
        with torch.no_grad():
            model.encoder.bn1.running_mean.add_(0.2)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_model(cfg, "cpu", torch.load(stream, weights_only=True))
        del stream
        samples = [
            dict(
                image=[torch.randn(1, 32, 32, 32), torch.randn(1, 32, 32, 32)], gt_latents={"z_content": torch.randn(9)}
            )
            for _ in range(3)
        ]
        original = encode(model, samples, "cpu", 2, 9)
        replay = encode(restored, samples, "cpu", 2, 9)
        for a, b in zip(original[:3], replay[:3]):
            torch.testing.assert_close(torch.from_numpy(a), torch.from_numpy(b), atol=0, rtol=0)
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_legacy_settings_and_checkpoint_keep_exact_original_behavior(self):
        cfg = dict(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            downscale_factor=4,
            latent_dim=12,
            content_channels=9,
            seed=17,
        )
        torch.manual_seed(cfg["seed"])
        old = nn.Module()
        old.encoder = Encoder(1, 8, 4, 1, 4, False)
        old.encoder_v1 = copy.deepcopy(old.encoder)
        old.to_encoding = nn.Conv3d(8, 12, 1)
        old.register_buffer("content_mask", torch.tensor([[1.0] * 9 + [0.0] * 3]))
        old.eval()
        fresh = build_model(cfg, "cpu")
        self.assertEqual(fresh.encoder_architecture, "conv")
        self.assertEqual(old.state_dict().keys(), fresh.state_dict().keys())
        for a, b in zip(old.state_dict().values(), fresh.state_dict().values()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        restored = build_model(cfg, "cpu", old.state_dict())
        with torch.no_grad():
            x = torch.randn(4, 1, 16, 16, 16)
            expected = old.to_encoding(torch.cat([old.encoder(x[:2]), old.encoder_v1(x[2:])])).mean((2, 3, 4))
            torch.testing.assert_close(restored(x, pool_only=True, n_views=2)[2][0], expected, atol=0, rtol=0)

    def test_cli_defaults_and_new_flags(self):
        ns = functions_from_file("training/main_conv_synthetic.py", {"parse_args"}, {"argparse": argparse})
        parse_args = ns["parse_args"]
        self.assertEqual(parse_args([]).encoder_architecture, "conv")
        args = parse_args(["--encoder-architecture", "resnet18", "--res", "64", "--no-separate-encoders"])
        self.assertEqual(args.encoder_head_hidden, 100)
        self.assertTrue(args.no_separate_encoders)
        self.assertEqual(args.contrastive_proj_dim, 0)
        with patch("sys.stderr", new_callable=io.StringIO):
            for extra in (
                ["--encoder-head-hidden", "0"],
                ["--eval-pooling", "patch", "--eval-patch-grid", "4", "4", "4"],
            ):
                with self.assertRaises(SystemExit):
                    parse_args(["--encoder-architecture", "resnet18", "--res", "64", *extra])


if __name__ == "__main__":
    unittest.main()
