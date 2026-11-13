"""mlmv encoder: interface, locality (zero gradient beyond the theoretical radius), template, RF refusal."""

import tempfile
import unittest

import torch

from mlmv.config import load_config
from mlmv.models.encoder import ChannelGroupNorm, Encoder, Template
from mlmv.train.train_levels import run

SMALL = dict(stem_channels=8, trunk_channels=16, trunk_blocks=2, norm_groups=4, dims={"G": 4, "L": 4, "S": 8})


class TestEncoder(unittest.TestCase):
    def test_interface_shapes(self):
        enc = Encoder(**SMALL)
        x = torch.randn(2, 1, 16, 16, 16)
        out = enc(x, "flair")
        self.assertEqual(
            {k: tuple(v.shape) for k, v in out.items()},
            {"G": (2, 4, 8, 8, 8), "L": (2, 4, 8, 8, 8), "S": (2, 8, 8, 8, 8)},
        )

    def test_default_radius_is_within_budget(self):
        cfg = load_config()
        from mlmv.models.encoder import build_encoder

        enc = build_encoder(cfg.model)
        self.assertEqual(enc.theoretical_radius(), 2.5)
        self.assertLessEqual(enc.theoretical_radius(), cfg.model.max_rf_radius)
        self.assertEqual(Encoder(trunk_spatial_convs=8).theoretical_radius(), 9.5)

    def test_gradient_zero_beyond_theoretical_radius(self):
        torch.manual_seed(0)
        for spatial, stem_norm in ((0, "none"), (1, "none"), (2, "channel")):
            enc = Encoder(**{**SMALL, "trunk_spatial_convs": spatial, "stem_norm": stem_norm, "norm_groups": 2})
            for p in enc.parameters():  # random biases/affines so no path is zero by construction
                torch.nn.init.normal_(p, std=0.3)
            s, low, high = enc.theoretical_footprint()
            x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
            u = (8, 7, 9)
            enc.features(x, "t1")[(0, slice(None)) + u].sum().backward()
            nz = x.grad[0, 0].abs() > 0
            inside = torch.zeros_like(nz)
            inside[tuple(slice(s * c + low, s * c + high + 1) for c in u)] = True
            self.assertFalse((nz & ~inside).any(), f"gradient outside the footprint (spatial convs {spatial})")
            # tight: the footprint's outermost faces are reached
            idx = nz.nonzero()
            self.assertEqual(int(idx[:, 0].min()), s * u[0] + low)
            self.assertEqual(int(idx[:, 0].max()), s * u[0] + high)

    def test_channel_norm_is_pointwise(self):
        norm = ChannelGroupNorm(2, 8)
        x = torch.randn(1, 8, 5, 5, 5)
        y = x.clone()
        y[0, :, 2, 2, 2] += 3.0
        diff = (norm(x) - norm(y)).abs().sum(1)[0]
        self.assertGreater(float(diff[2, 2, 2]), 0)
        diff[2, 2, 2] = 0
        self.assertEqual(float(diff.max()), 0.0)

    def test_tiny_groups_rejected(self):
        with self.assertRaises(ValueError):
            ChannelGroupNorm(8, 8)

    def test_template(self):
        t = Template(3, (2, 2, 2), momentum=0.5)
        a, b = torch.ones(4, 3, 2, 2, 2), 3 * torch.ones(4, 3, 2, 2, 2)
        t.update(a, "t1")
        self.assertTrue(torch.allclose(t.t1, torch.ones(3, 2, 2, 2)))
        t.update(b, "t1")
        self.assertTrue(torch.allclose(t.t1, 2 * torch.ones(3, 2, 2, 2)))
        self.assertTrue(torch.allclose(t.subtract(b, "t1"), torch.ones_like(b)))
        self.assertEqual(float(t.flair.abs().sum()), 0.0)  # per modality
        self.assertIn("t1", t.state_dict())

    def test_refuses_to_train_beyond_rf_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(overrides={"model": {"trunk_spatial_convs": 8}, "run": {"out_dir": tmp, "device": "cpu"}})
            with self.assertRaises(SystemExit) as ctx:
                run(cfg, log=lambda s: None)
            self.assertIn("refusing to train", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
