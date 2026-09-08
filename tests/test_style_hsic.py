"""CPU tests of HSIC and the actual VQVAE/training path, without ADNI dependencies.

AST loading omits only unrelated module imports. Model/loss/training implementations
are executed unchanged; fixed masks do not call the omitted MONAI utility module.
"""

import ast
import logging
import math
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from training.style_hsic import style_content_hsic_loss

ROOT = Path(__file__).resolve().parents[1]


def load_model():
    source = ROOT / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(a.name == "utils.utils" for a in node.names))
    ]
    namespace = {"__name__": "hsic_model_test"}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["VQVAE"]


def load_config():
    source = ROOT / "utils/config.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(a.name == "data.datasets" for a in node.names))
    ]
    namespace = {"datasets": types.SimpleNamespace(SyntheticBrainDataset=object, MyCustomDataset=object)}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["parse_args"], namespace["update_args"]


class StyleHSICTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Model = load_model()

    def setUp(self):
        torch.manual_seed(19)

    def test_detects_nonlinear_and_sign_flipped_spatial_leak(self):
        target = torch.randn(192, 1)
        signal = target.square()
        # Every subject has zero GAP style, but its spatial pattern encodes anatomy.
        spatial = torch.cat([signal, -signal], dim=1)[:, None, :, None, None]
        style = torch.cat([spatial, -spatial])
        loss, diag = style_content_hsic_loss({0: style}, target, 2)
        null, _ = style_content_hsic_loss({0: style}, target[torch.randperm(len(target))], 2)
        self.assertGreater(loss.item(), null.item() + 0.15)
        self.assertAlmostEqual(diag["Style/hsic_L0_v0"], diag["Style/hsic_L0_v1"], places=6)
        self.assertTrue(torch.equal(spatial.mean((2, 3, 4)), torch.zeros(192, 1)))

    def test_constant_features_and_targets_have_finite_gradients(self):
        for constant_target in (False, True):
            for constant_style in (False, True):
                target = torch.ones(16, 3) if constant_target else torch.randn(16, 3)
                style = torch.ones(32, 2, 3) if constant_style else torch.randn(32, 2, 3)
                style.requires_grad_()
                loss, _ = style_content_hsic_loss({0: style}, target, 2)
                loss.backward()
                self.assertTrue(torch.isfinite(style.grad).all())
                if constant_style or constant_target:
                    self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_gradients_and_scale_invariance(self):
        target = torch.randn(32, 2, requires_grad=True)
        style = torch.randn(64, 3, 4, requires_grad=True)
        loss, _ = style_content_hsic_loss({0: style}, target, 2)
        scaled, _ = style_content_hsic_loss({0: 0.01 * style + 3}, target, 2)
        self.assertAlmostEqual(loss.item(), scaled.item(), places=5)
        loss.backward()
        self.assertGreater(style.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(style.grad).all())
        self.assertIsNone(target.grad)

    def test_small_batches_and_invalid_inputs(self):
        style = torch.randn(6, 2, requires_grad=True)
        loss, diag = style_content_hsic_loss({0: style}, torch.randn(3, 2), 2)
        self.assertEqual(diag["Style/hsic_skipped_small_batch"], 1)
        loss.backward()
        self.assertEqual(style.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "nonempty decoder style"):
            style_content_hsic_loss({}, torch.randn(4, 2), 2)
        with self.assertRaisesRegex(ValueError, "n_views"):
            style_content_hsic_loss({0: style}, torch.randn(4, 2), 2)

    def model(self, **kwargs):
        return self.Model(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            nb_levels=1,
            embed_dim=8,
            nb_entries=8,
            scaling_rates=[2],
            use_checkpoint=False,
            content_size=6,
            style_size=2,
            content_style_levels=[0],
            mask_mode="fixed",
            inject_style_to_decoder=True,
            style_injection_mode="input",
            norm_type="layer",
            **kwargs,
        )

    def test_model_default_unchanged_and_features_are_prequantization(self):
        for separate in (False, True):
            model = self.model(
                separate_encoders=separate, quantize_style=True, separate_style_codebooks=separate, style_spatial_size=2
            )
            model.eval()
            x = torch.randn(8, 1, 8, 8, 8)
            seen = []
            handle = model.style_codebooks["0"].register_forward_pre_hook(lambda m, a: seen.append(a[0]))
            regular = model(x, n_views=2, pool_only=True)
            self.assertEqual(len(regular), 8)
            seen.clear()
            output, features = model(x, n_views=2, pool_only=True, return_style_features=True)
            handle.remove()
            self.assertTrue(torch.equal(regular[0], output[0]))
            self.assertEqual(features[0].shape, (8, 2, 2, 2, 2))
            expected = features[0][:4] if separate else features[0]
            self.assertTrue(torch.equal(seen[0], expected))
            features[0].retain_grad()
            loss, _ = style_content_hsic_loss(features, torch.randn(4, 2), 2)
            loss.backward()
            self.assertGreater(features[0].grad.norm().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in model.encoders.parameters()))

    def test_skipped_reconstruction_still_supplies_style_without_codebooks(self):
        model = self.model(quantize_style=True, detach_style_injection=True)
        model.eval()
        x = torch.randn(8, 1, 8, 8, 8)
        regular = model(x, n_views=2, pool_only=True, return_recon=False)
        output, features = model(x, n_views=2, pool_only=True, return_recon=False, return_style_features=True)
        self.assertIsNone(output[0])
        self.assertEqual(sum(output[1]).item(), 0)
        self.assertTrue(torch.equal(regular[2][0], output[2][0]))
        self.assertTrue(features[0].requires_grad)
        self.assertEqual(features[0].shape[-3:], (4, 4, 4))

    def test_config_is_opt_in_and_rejects_unsupported_data(self):
        parse, update = load_config()
        self.assertEqual(parse().parse_args([]).scale_style_hsic_loss, 0)
        for flags, message in [
            (["--scale-style-hsic-loss", "-1"], "finite and nonnegative"),
            (["--scale-style-hsic-loss", "nan"], "finite and nonnegative"),
            (["--scale-style-hsic-loss", "1"], "synthetic ground-truth"),
            (["--dataset-name", "synthetic", "--scale-style-hsic-loss", "1"], "inject-style"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                update(parse().parse_args(flags))

    def test_training_weight_and_backward_with_reconstruction_skipped(self):
        source = ROOT / "training/main_multimodal.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_step")
        namespace = dict(
            torch=torch,
            F=F,
            math=math,
            autocast=torch.autocast,
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
            logger=logging.getLogger(__name__),
            style_content_hsic_loss=style_content_hsic_loss,
            NAN_SKIPPED_STEPS=0,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        train_step = namespace["train_step"]
        parse, update = load_config()
        args = update(
            parse().parse_args(
                [
                    "--dataset-name",
                    "synthetic",
                    "--inject-style-to-decoder",
                    "--batch-size",
                    "8",
                    "--scale-contrastive-loss",
                    "0",
                    "--scale-style-hsic-loss",
                    "0",
                ]
            )
        )
        model = self.model().eval()
        data = {
            "image": [torch.randn(8, 1, 8, 8, 8), torch.randn(8, 1, 8, 8, 8)],
            "gt_latents": {"z_content": torch.randn(8, 3)},
        }

        def run(optimizer=None, reconstruct=True):
            return train_step(
                data,
                [model],
                [],
                lambda h, *a, **kw: h.sum() * 0,
                optimizer,
                list(model.parameters()),
                args,
                recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
                force_compute_recon=reconstruct,
            )

        baseline = run()
        self.assertNotIn("Style/hsic", baseline[-1])
        args.scale_style_hsic_loss = 2.0
        penalized = run()
        self.assertAlmostEqual(penalized[0] - baseline[0], penalized[-1]["Style/hsic_weighted"], places=5)
        self.assertAlmostEqual(penalized[-1]["Style/hsic_weighted"], 2 * penalized[-1]["Style/hsic"], places=6)
        self.assertEqual(penalized[1:4], baseline[1:4])
        before = [p.detach().clone() for p in model.encoders.parameters()]
        result = run(torch.optim.SGD(model.parameters(), lr=0.1), reconstruct=False)
        self.assertGreater(result[-1]["Style/hsic"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.encoders.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        del data["gt_latents"]
        with self.assertRaisesRegex(ValueError, "gt_latents"):
            run()
        args.scale_style_hsic_loss = 0
        run()  # Labels are unnecessary when disabled.

    def test_autocast_and_level_averaging(self):
        target = torch.randn(16, 2)
        style = torch.randn(32, 4, 3, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, diag = style_content_hsic_loss({0: style, 1: style.square()}, target, 2)
        expected = sum(diag[f"Style/hsic_L{level}_v{view}"] for level in (0, 1) for view in (0, 1)) / 4
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(style.grad).all())


if __name__ == "__main__":
    unittest.main()
