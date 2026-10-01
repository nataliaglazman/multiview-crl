"""Real ablation forward/backward, saved architecture replay and probe paths."""

import gc
import unittest

import numpy as np
import torch
from torch import nn

from eval.checkpoint_lesion_analysis import batch_features
from eval.encoder_generalization_audit import capture_batch
from eval.score_checkpoint import build_model
from models.resnet3d import ResNet18Features3d

VARIANTS = (
    dict(encoder_architecture="conv", conv_readout="mlp"),
    dict(encoder_architecture="resnet18", resnet_norm="group"),
    dict(encoder_architecture="resnet18", resnet_output_stride=8),
)


class EncoderAblationTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def config(self, **kwargs):
        return dict(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            downscale_factor=4,
            latent_dim=12,
            content_channels=9,
            encoder_head_hidden=7,
            seed=17,
            **kwargs,
        )

    def test_each_variant_trains_both_views_and_replays_saved_checkpoint(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = self.config(**variant)
                model = build_model(cfg, "cpu").train()
                x = torch.randn(4, 1, 32, 32, 32)
                x[2:] = 1.5 * x[2:] + 0.2
                content = model(x, pool_only=True, n_views=2)[2][0][:, :9]
                a, b = nn.functional.normalize(content, dim=1).chunk(2)
                loss = nn.functional.cross_entropy(a @ b.T / 0.1, torch.arange(2))
                loss.backward()
                for part in (model.encoder, model.encoder_v1, model.to_encoding):
                    weight = next(part.parameters())
                    self.assertTrue(torch.isfinite(weight.grad).all())
                    self.assertGreater(weight.grad.abs().sum().item(), 0)
                torch.optim.SGD(model.parameters(), lr=1e-4).step()
                model.eval()
                restored = build_model(cfg, "cpu", state_dict=model.state_dict())
                with torch.inference_mode():
                    expected = model(x, pool_only=True, n_views=2)[2][0]
                    actual = restored(x, pool_only=True, n_views=2)[2][0]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(restored.readout_type, "mlp")
                stages = capture_batch(restored, x)
                self.assertEqual(stages["hidden"].shape, (4, 7))
                np.testing.assert_allclose(stages["content"], actual[:, :9], atol=1e-6)
                grid = 1 if restored.backbone_stride == 32 else 2
                features, _, _ = batch_features(restored, x, [1, grid])
                with torch.inference_mode():
                    patches = restored(x, pool_only=True, n_views=2, patch_grid=[grid] * 3)[2][0]
                np.testing.assert_allclose(
                    features[(grid, "projected")],
                    patches[:, :9].flatten(1),
                    atol=1e-6,
                )
                del model, restored, loss, content, a, b, expected, actual, weight, part, patches
                gc.collect()

    def test_resnet_changes_keep_initial_weights_and_advertised_map_sizes(self):
        torch.manual_seed(9)
        baseline = ResNet18Features3d().eval()
        for options, spatial in (({"norm": "group"}, 2), ({"output_stride": 8}, 8)):
            with self.subTest(options=options):
                torch.manual_seed(9)
                variant = ResNet18Features3d(**options).eval()
                for (name, expected), (actual_name, actual) in zip(
                    baseline.named_parameters(), variant.named_parameters()
                ):
                    self.assertEqual(name, actual_name)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(variant.conv1.stride, (2, 2, 2))
                self.assertEqual(variant.maxpool.stride, 2)
                if options.get("norm") == "group":
                    self.assertFalse(any(isinstance(m, nn.BatchNorm3d) for m in variant.modules()))
                    self.assertEqual(sum(isinstance(m, nn.GroupNorm) for m in variant.modules()), 20)
                else:
                    self.assertEqual(variant.layer3[0].conv1.stride, (1, 1, 1))
                    self.assertEqual(variant.layer4[0].conv1.stride, (1, 1, 1))
                with torch.inference_mode():
                    self.assertEqual(variant(torch.zeros(1, 1, 64, 64, 64)).shape, (1, 512, spatial, spatial, spatial))
                del variant

    def test_conv_mlp_changes_only_readout_and_pools_before_nonlinearity(self):
        baseline = build_model(self.config(encoder_architecture="conv"), "cpu")
        variant = build_model(self.config(encoder_architecture="conv", conv_readout="mlp"), "cpu")
        for name in ("encoder", "encoder_v1"):
            for a, b in zip(getattr(baseline, name).parameters(), getattr(variant, name).parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        with torch.inference_mode():
            x = torch.randn(2, 1, 32, 32, 32)
            h = variant.encoder(x)
            actual = variant(x, pool_only=True)[2][0]
            torch.testing.assert_close(actual, variant.to_encoding(h.mean((2, 3, 4))))
            self.assertEqual(variant(x, pool_only=False)[2][0].shape, (2, 12, 8, 8, 8))
            with self.assertRaisesRegex(ValueError, "must fit"):
                variant(x, pool_only=True, patch_grid=[16] * 3)


if __name__ == "__main__":
    unittest.main()
