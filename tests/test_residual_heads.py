"""Residual branches keep fixed training preprocessing and optimize joint information."""

import io
import unittest

import torch

from models.residual_heads import ResidualHeads, joint_uniform_mmd


class ResidualHeadsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(23)
        # A common occupied half-volume gives both kept and excluded tail cells.
        self.images = torch.randn(12, 2, 8, 8, 8) + 3
        self.images[:, :, :4] = 0

    def module(self, fitted=True, **options):
        model = ResidualHeads(
            8,
            content_units=2,
            hidden_channels=4,
            head_hidden=8,
            components=2,
            joint_input_units=5,
            joint_dim=5,
            **options,
        )
        if fitted:
            model.fit(self.images)
        return model

    def view_major(self, images=None):
        images = self.images if images is None else images
        return torch.cat([images[:, 0:1], images[:, 1:2]])

    def test_fit_is_training_only_fixed_and_preserves_rng(self):
        model = self.module(fitted=False)
        before = torch.random.get_rng_state().clone()
        model.fit(self.images)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertTrue(model.fitted)
        self.assertEqual(model.fit_subjects.item(), len(self.images))
        self.assertEqual(model.interior.sum().item(), 4)
        self.assertTrue(model.tail_started.all())
        self.assertFalse(model.brain[:, :, :4].any())
        for view in range(2):
            standardized = model._standardize(self.images[:, view : view + 1])
            expected = standardized.mean(0) * model.brain[view]
            torch.testing.assert_close(model.mean[view], expected)
            basis = model.basis[view].flatten(1)
            torch.testing.assert_close(basis @ basis.T, torch.eye(2), rtol=1e-4, atol=1e-5)

        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            model.fit(self.images + 20)
        fixed = {key: model.state_dict()[key].clone() for key in ("mean", "basis", "brain", "residual_sd", "interior")}
        model.train()
        model(self.view_major(self.images * 2))
        for key, value in fixed.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_residual_orthogonality_scale_and_per_image_affine_invariance(self):
        model = self.module().eval()
        residuals = model.residual_images(self.view_major())
        for view, part in enumerate(residuals.chunk(2)):
            matrix = part.flatten(1)
            basis = model.basis[view].flatten(1)
            torch.testing.assert_close(matrix @ basis.T, torch.zeros(len(self.images), 2), atol=2e-5, rtol=0)
            masked = part[:, 0, model.brain[view, 0]]
            self.assertAlmostEqual(masked.square().mean().item(), 1, places=5)
            self.assertEqual(part[:, :, :4].abs().sum().item(), 0)
        gain = torch.linspace(0.5, 2, len(self.images))[:, None, None, None, None]
        bias = torch.linspace(-0.4, 1, len(self.images))[:, None, None, None, None]
        changed = (self.images * gain + bias) * (self.images != 0)
        torch.testing.assert_close(
            model.residual_images(self.view_major(changed)),
            residuals,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_state_restore_and_single_view_eval_are_identical(self):
        model = self.module().train()
        residual = model(self.view_major())
        raw = torch.randn(len(residual), 3)
        model.calibrate_codes(raw, residual)
        model.joint_loss(raw, residual)
        model.eval()
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        restored = self.module(fitted=False).eval()
        restored.load_state_dict(torch.load(saved, weights_only=True))
        states = {key: value.clone() for key, value in restored.state_dict().items()}
        batch = self.view_major()[:4].repeat(2, 1, 1, 1, 1)
        both = restored(batch)
        torch.testing.assert_close(both, model(batch))
        first = restored(batch[:4], n_views=1, view_idx=0)
        second = restored(batch[4:], n_views=1, view_idx=1)
        torch.testing.assert_close(both, torch.cat([first, second]))
        restored.joint_loss(raw, residual.detach())
        for key, value in states.items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_gradients_reach_both_independent_encoders_raw_code_and_projector(self):
        model = self.module().train()
        images = self.view_major().clone().requires_grad_()
        residual = model(images)
        raw = torch.randn(len(images), 3, requires_grad=True)
        loss, terms = model.joint_loss(raw, residual)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(terms["uniformity"].item(), 0)
        loss.backward()
        self.assertIsNone(images.grad)  # input pixels are observations, not optimized parameters
        self.assertGreater(raw.grad.abs().sum().item(), 0)
        for encoder in model.encoders:
            self.assertTrue(all(parameter.grad is not None for parameter in encoder.parameters()))
            self.assertGreater(
                sum(parameter.grad.abs().sum().item() for parameter in encoder.parameters()),
                0,
            )
        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.projector.parameters()))
        self.assertTrue(all(buffer.grad is None for buffer in model.buffers()))
        first_ids = {id(parameter) for parameter in model.encoders[0].parameters()}
        self.assertFalse(first_ids.intersection(id(parameter) for parameter in model.encoders[1].parameters()))

    def test_uniformity_alone_backpropagates_to_both_blocks(self):
        model = self.module().eval()
        raw = torch.randn(16, 3, requires_grad=True)
        residual = torch.randn(16, 2, requires_grad=True)
        model.calibrate_codes(raw.detach(), residual.detach())
        loss, _ = model.joint_loss(raw, residual, raw_weight=0, residual_weight=0)
        loss.backward()
        self.assertGreater(raw.grad.abs().sum().item(), 0)
        self.assertGreater(residual.grad.abs().sum().item(), 0)

    def test_evaluation_uses_population_gauge_not_other_batch_subjects(self):
        model = self.module().eval()
        raw, residual = torch.randn(20, 3), torch.randn(20, 2)
        model.calibrate_codes(raw, residual)
        joint = torch.cat([raw, residual], 1)
        mean, sd = model.code_mean[0], model.code_var[0].sqrt()
        first = model.projector((joint[:1] - mean) / sd)
        companions = torch.cat([joint[:1], 100 * joint[1:]])
        again = model.projector((companions - mean) / sd)[:1]
        torch.testing.assert_close(first, again)
        before = model.code_mean.clone()
        model.train()
        model.joint_loss(raw + 20, residual + 20)
        self.assertGreater((model.code_mean - before).abs().sum().item(), 0)

    def test_joint_kernel_detects_redundancy_with_uniform_marginals(self):
        # Both columns of duplicate have exactly the same uniform marginal as independent.
        u = (torch.arange(512, dtype=torch.float32) + 0.5) / 512
        duplicate = torch.stack([u, u], 1)
        generator = torch.Generator().manual_seed(3)
        independent = torch.stack([u, u[torch.randperm(len(u), generator=generator)]], 1)
        reference = torch.rand(512, 2, generator=generator)
        bad = joint_uniform_mmd(duplicate, reference)
        good = joint_uniform_mmd(independent, reference)
        self.assertGreater(bad.item(), good.item() + 0.02)
        torch.testing.assert_close(
            joint_uniform_mmd(reference, reference),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )

    def test_batch_alignment_removes_detached_gauge_scale_contraction(self):
        raw, residual = torch.randn(64, 3), torch.randn(64, 2)
        legacy = self.module(fitted=False).eval()
        corrected = self.module(fitted=False, alignment_mode="batch").eval()
        legacy.calibrate_codes(raw, residual)
        corrected.calibrate_codes(raw, residual)
        # Detached population normalization gives the shrink gradient seen in
        # the failed run, even at calibrated population statistics.
        scale = torch.tensor(1.0, requires_grad=True)
        loss, _ = legacy.joint_loss(scale * raw, scale * residual, uniform_weight=0)
        old_gradient = torch.autograd.grad(loss, scale)[0]
        torch.testing.assert_close(old_gradient, 2 * loss.detach(), rtol=1e-5, atol=1e-5)
        self.assertGreater(old_gradient.item(), 1)
        values = []
        for value in (1e-3, 0.1, 1.0, 10.0):
            scale = torch.tensor(value, requires_grad=True)
            loss, _ = corrected.joint_loss(scale * raw, scale * residual, uniform_weight=0)
            derivative = torch.autograd.grad(loss, scale)[0]
            # dL/dlog(scale) is comparable across common code scales.
            self.assertLess(abs((derivative * scale).item()), 2e-5)
            values.append(loss.item())
        self.assertLess(max(values) - min(values), 2e-5)

    def test_batch_alignment_epsilon_effect_is_explicit_near_zero(self):
        raw, residual = torch.randn(64, 3), torch.randn(64, 2)
        radial = []
        for epsilon in (1e-14, 1e-12, 1e-8):
            model = self.module(fitted=False, alignment_mode="batch", alignment_eps=epsilon).eval()
            scale = torch.tensor(1e-3, requires_grad=True)
            loss, _ = model.joint_loss(scale * raw, scale * residual, uniform_weight=0)
            derivative = torch.autograd.grad(loss, scale)[0]
            radial.append(abs((derivative * scale).item()))
        self.assertLess(radial[0], 2e-6)
        self.assertLess(radial[1], 2e-5)
        self.assertGreater(radial[2], 1e-3)
        # Epsilon-regularized alignment does not prevent collapse at
        # arbitrarily small variance; exactly constant codes remain stationary.
        model = self.module(fitted=False, alignment_mode="batch").eval()
        zeros = torch.zeros(8, 5, requires_grad=True)
        loss, _ = model.joint_loss(zeros[:, :3], zeros[:, 3:], uniform_weight=0)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, zeros)[0]).all())

    def test_batch_alignment_accepts_positive_affine_view_gauges_and_backpropagates(self):
        model = self.module(alignment_mode="batch").eval()
        first_raw, first_residual = torch.randn(16, 3), torch.randn(16, 2)
        raw = torch.cat([first_raw, 3 * first_raw + 8]).requires_grad_()
        residual = torch.cat([first_residual, 0.4 * first_residual - 2]).requires_grad_()
        _, terms = model.joint_loss(raw, residual)
        self.assertLess(terms["image_alignment"].item(), 1e-10)
        self.assertLess(terms["residual_alignment"].item(), 1e-10)
        model.calibrate_codes(raw.detach(), residual.detach())
        loss, _ = model.joint_loss(raw, residual, raw_weight=0, residual_weight=0)
        loss.backward()
        self.assertGreater(raw.grad.abs().sum().item(), 0)
        self.assertGreater(residual.grad.abs().sum().item(), 0)

    def test_population_default_preserves_legacy_state_and_numerical_loss(self):
        original = self.module().eval()
        explicit = self.module(fitted=False, alignment_mode="population").eval()
        corrected = self.module(fitted=False, alignment_mode="batch").eval()
        explicit.load_state_dict(original.state_dict(), strict=True)
        corrected.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(set(original.state_dict()), set(corrected.state_dict()))
        raw, residual = torch.randn(12, 3), torch.randn(12, 2)
        torch.manual_seed(8)
        expected, first = original.joint_loss(raw, residual)
        torch.manual_seed(8)
        actual, second = explicit.joint_loss(raw, residual)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_full_cohort_diagnostics_are_fixed_and_detect_block_concentration(self):
        model = self.module().train()
        raw, residual = torch.randn(128, 3), torch.randn(128, 2)
        model.calibrate_codes(raw, residual)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        varied = model.diagnostics(raw, residual)
        concentrated = model.diagnostics(raw[:, :1].expand(-1, 3), residual[:, :1].expand(-1, 2))
        shrunk = model.diagnostics(0.01 * raw, 0.01 * residual)
        for key in ("image_raw_variance", "residual_raw_variance"):
            torch.testing.assert_close(shrunk[key], 1e-4 * varied[key], rtol=1e-5, atol=1e-10)
        self.assertGreater(varied["image_effective_rank"].item(), 2.8)
        self.assertGreater(varied["residual_effective_rank"].item(), 1.8)
        self.assertAlmostEqual(concentrated["image_effective_rank"].item(), 1, places=5)
        self.assertAlmostEqual(concentrated["residual_effective_rank"].item(), 1, places=5)
        # Rank and variance should remain distinct even at very small scales.
        _, small_rank = model._moments(1e-7 * raw[:, :1].expand(-1, 3))
        self.assertAlmostEqual(small_rank.item(), 1, places=5)
        for name in (
            "image_variance",
            "residual_variance",
            "projected_effective_rank",
            "code_sd_min",
            "projector_weight_norm",
        ):
            self.assertTrue(torch.isfinite(varied[name]))
            self.assertGreaterEqual(varied[name].item(), 0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "two subjects"):
            self.module(alignment_mode="batch").joint_loss(raw[:2], residual[:2])

    def test_invalid_fit_and_view_layout_are_rejected(self):
        model = self.module(fitted=False)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model(self.view_major())
        with self.assertRaisesRegex(ValueError, "more than"):
            model.fit(self.images[:2])
        with self.assertRaisesRegex(ValueError, "90%"):
            model.fit(torch.zeros_like(self.images))
        model.fit(self.images)
        with self.assertRaisesRegex(ValueError, "even"):
            model(self.view_major()[:3])
        with self.assertRaisesRegex(ValueError, "view_idx"):
            model(self.view_major()[:3], n_views=1)
        with self.assertRaisesRegex(ValueError, "combined code width"):
            model.joint_loss(torch.randn(4, 4), torch.randn(4, 2))


if __name__ == "__main__":
    unittest.main()
