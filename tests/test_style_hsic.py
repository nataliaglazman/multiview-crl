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
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training.style_hsic import _unbiased_hsic, style_content_hsic_loss, style_independence_loss

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


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

    def test_training_step_style_independence_mode(self):
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
            style_independence_loss=style_independence_loss,
            style_infonce_loss=load_function("training/losses.py", "style_infonce_loss"),
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
                    "--style-contrastive-mode",
                    "independence",
                ]
            )
        )
        model = self.model().eval()
        data = {"image": [torch.randn(8, 1, 8, 8, 8), torch.randn(8, 1, 8, 8, 8)]}

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
        self.assertFalse(any(k.startswith("Style/") for k in baseline[-1]))  # zero weight: the mode is inert
        args.scale_style_contrastive_loss = 1.5
        penalized = run()
        diag = penalized[-1]
        self.assertAlmostEqual(penalized[0] - baseline[0], 1.5 * diag["Style/independence_L0"], places=5)
        self.assertAlmostEqual(
            diag["Style/independence_L0"], max(diag["Style/xview_hsic_L0"], 0) + diag["Style/var_hinge_L0"], places=5
        )
        self.assertNotIn("Style/infonce_L0", diag)
        self.assertEqual(penalized[2:4], baseline[2:4])
        self.assertAlmostEqual(diag["Style/independence_weighted"], 1.5 * diag["Style/independence_L0"], places=5)
        # Earlier encoder pools can vary while the post-bottleneck style is constant.
        # The variance hinge must see that collapse, independently of the pooled path.
        original_forward = model.forward

        def constant_style(*a, **kw):
            normal, features = original_forward(*a, **kw)
            return normal, {level: tensor * 0 + 2 for level, tensor in features.items()}

        with patch.object(model, "forward", side_effect=constant_style):
            collapsed = run()[-1]
        self.assertEqual(collapsed["Style/var_hinge_L0"], 2)
        self.assertEqual(collapsed["Style/independence_L0"], 2)
        args.style_independence_var_weight = 0.25
        weighted = run()[-1]
        self.assertAlmostEqual(
            weighted["Style/independence_L0"],
            max(weighted["Style/xview_hsic_L0"], 0) + 0.25 * weighted["Style/var_hinge_L0"],
            places=5,
        )
        args.style_independence_var_weight = 1.0
        before = [p.detach().clone() for p in model.encoders.parameters()]
        run(torch.optim.SGD(model.parameters(), lr=0.1), reconstruct=False)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.encoders.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        args.style_contrastive_mode = "cosine"
        legacy = run()
        self.assertIn("Style/infonce_L0", legacy[-1])
        self.assertNotIn("Style/xview_hsic_L0", legacy[-1])


class StyleIndependenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def views(shared, batch=128, width=16, nonlinear=False, noise=1.0):
        # True per-view styles are independent; a large per-view offset plays the modality mean.
        anatomy = torch.randn(batch, 1)
        loading = torch.randn(1, width)
        v0 = 10 + noise * torch.randn(batch, width)
        v1 = -10 + noise * torch.randn(batch, width)
        if shared:
            v0 = v0 + 3 * anatomy * loading
            v1 = v1 + 3 * (anatomy.square() if nonlinear else anatomy) * loading
        return v0, v1

    def score(self, v0, v1):
        return style_independence_loss(torch.cat([v0, v1]))[1]["xview_hsic"]

    def test_detects_shared_anatomy_that_the_cosine_loss_misses(self):
        legacy = load_function("training/losses.py", "style_infonce_loss")
        independent = self.views(False)
        shared = self.views(True)
        self.assertLess(abs(self.score(*independent)), 0.05)
        self.assertGreater(self.score(*shared), 0.6)
        self.assertLess(abs(self.score(shared[0], shared[1][torch.randperm(128)])), 0.05)  # subject-shuffled null
        self.assertGreater(self.score(*self.views(True, nonlinear=True)), 0.1)  # a in one view, a^2 in the other
        # Pushing the modality means apart satisfies the legacy loss with the leak intact:
        # it then scores the leaking styles BETTER than independent ones. Dependence is unmoved.
        moved = (shared[0] + 90, shared[1] - 90)
        self.assertLess(legacy(*moved).item(), legacy(*independent).item())
        self.assertAlmostEqual(self.score(*moved), self.score(*shared), places=4)

    def test_high_dimensional_independent_styles_score_near_zero(self):
        # Decoder-bound style has thousands of coordinates per subject, where the biased
        # estimator's diagonal dominates. 128 subjects matches the training batch.
        self.assertLess(abs(self.score(*self.views(False, width=4096, noise=3.0))), 0.05)
        self.assertGreater(self.score(*self.views(True, width=4096, noise=3.0)), 0.5)
        self.assertGreater(self.score(*self.views(True, width=4096, noise=3.0, nonlinear=True)), 0.1)

    def test_dependence_ignores_coordinatewise_offsets_and_scales(self):
        v0, v1 = self.views(True, batch=32)
        _, plain = style_independence_loss(torch.cat([v0, v1]))
        _, moved = style_independence_loss(torch.cat([3 * v0 + 5, -0.2 * v1 - 7]))
        self.assertAlmostEqual(plain["xview_hsic"], moved["xview_hsic"], places=4)

    def test_collapse_is_caught_by_the_hinge_not_the_dependence(self):
        style = torch.ones(16, 3, 2, requires_grad=True)
        loss, diag = style_independence_loss(style)
        loss.backward()
        self.assertAlmostEqual(diag["xview_hsic"], 0.0, places=6)
        self.assertAlmostEqual(diag["var_hinge"], 2.0, places=6)
        self.assertTrue(torch.isfinite(style.grad).all())
        self.assertEqual(diag["xview_hsic_degenerate"], 1)

    def test_small_batches_autocast_and_invalid_inputs(self):
        style = torch.randn(6, 2, requires_grad=True)
        loss, diag = style_independence_loss(style)
        self.assertEqual(diag["xview_hsic_skipped_small_batch"], 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(style.grad).all())
        self.assertGreater(style.grad.abs().sum().item(), 0)  # hinge still trains the actual style
        style = torch.randn(16, 4, 3, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = style_independence_loss(style)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(style.grad).all())
        with self.assertRaisesRegex(ValueError, "even batch"):
            style_independence_loss(torch.randn(7, 2))
        for bad in (torch.tensor(1.0), torch.empty(0, 2), torch.empty(8, 0)):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                style_independence_loss(bad)
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                style_independence_loss(style, variance_weight=weight)

    def test_unbiased_estimator_matches_paper_value_and_gradient(self):
        # Independent expression of Song et al. Eq. 5, without mean subtraction.
        for n in (4, 13, 64):
            x, y = torch.randn(n, 5, dtype=torch.float64), torch.randn(n, 7, dtype=torch.float64)
            k = torch.exp(-torch.cdist(x, x).square() / 10).requires_grad_()
            l = torch.exp(-torch.cdist(y, y).square() / 14).requires_grad_()
            off = 1 - torch.eye(n, dtype=torch.float64)
            a, b = k * off, l * off
            expected = ((a * b).sum() + a.sum() * b.sum() / ((n - 1) * (n - 2)) - 2 * (a @ b).sum() / (n - 2)) / (
                n * (n - 3)
            )
            actual = _unbiased_hsic(k, l)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-9)
            for left, right in zip(
                torch.autograd.grad(actual, (k, l), retain_graph=True), torch.autograd.grad(expected, (k, l))
            ):
                torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-9)
        with self.assertRaisesRegex(ValueError, "at least four"):
            _unbiased_hsic(torch.eye(3), torch.eye(3))

    def test_spatial_variation_counts_but_fixed_spatial_templates_do_not(self):
        amplitude = torch.randn(64, 1) * 3
        amplitude = amplitude / amplitude.std(unbiased=False) * 3
        spatial = torch.cat((amplitude, -amplitude), dim=1)[:, None]
        self.assertEqual(float(spatial.mean(-1).abs().max()), 0)
        _, variable = style_independence_loss(torch.cat((spatial, spatial)))
        self.assertEqual(variable["var_hinge"], 0)
        template = torch.tensor([[-5.0, 5.0]]).expand(128, 1, 2).clone().requires_grad_()
        loss, fixed = style_independence_loss(template)
        self.assertEqual(fixed["var_hinge"], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(template.grad).all())

    def test_one_view_anatomy_leak_is_not_claimed_to_be_identifiable(self):
        anatomy = torch.randn(256, 1)
        # Anatomy is perfectly readable in view 0 while view 1 carries independent style.
        v0 = anatomy.repeat(1, 16)
        v1 = torch.randn(256, 16)
        self.assertLess(abs(self.score(v0, v1)), 0.05)

    def test_config_defaults_to_legacy_and_validates_independence(self):
        parse, update = load_config()
        self.assertEqual(parse().parse_args([]).style_contrastive_mode, "cosine")
        base = ["--dataset-name", "synthetic", "--batch-size", "8", "--style-contrastive-mode", "independence"]
        with self.assertRaisesRegex(ValueError, "inject-style"):
            update(parse().parse_args(base + ["--scale-style-contrastive-loss", "1"]))
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            update(parse().parse_args(base + ["--scale-style-contrastive-loss", "-1"]))
        update(parse().parse_args(base))  # zero weight: nothing required
        update(parse().parse_args(base + ["--scale-style-contrastive-loss", "1", "--inject-style-to-decoder"]))
        with self.assertRaisesRegex(ValueError, "synthetic data"):
            update(
                parse().parse_args(
                    [
                        "--style-contrastive-mode",
                        "independence",
                        "--scale-style-contrastive-loss",
                        "1",
                        "--inject-style-to-decoder",
                    ]
                )
            )
        for weight in ("-1", "nan", "inf"):
            with self.assertRaisesRegex(ValueError, "variance|var-weight"):
                update(
                    parse().parse_args(
                        base
                        + [
                            "--scale-style-contrastive-loss",
                            "1",
                            "--inject-style-to-decoder",
                            "--style-independence-var-weight",
                            weight,
                        ]
                    )
                )


if __name__ == "__main__":
    unittest.main()
