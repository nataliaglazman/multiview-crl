"""The spatial prediction target must be preserved through the content vector."""

import io
import unittest

import numpy as np
import torch

from eval.encoder import burden_recovery
from models import vector_prediction
from models.vector_prediction import (
    QUANTILES,
    VectorPrediction3d,
    _quantiles,
    deviation_quantiles,
    prepare_vector_targets,
    vector_prediction_loss,
)


class VectorPredictionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def fitted(self, channels=3, grid=(2, 2, 2)):
        predictor = VectorPrediction3d(channels, grid, hidden=8)
        images = torch.randn(6, 1, 4, 4, 4)
        predictor.accumulate_template(images, torch.ones_like(images))
        predictor.finalize_template()
        return predictor

    def test_decoders_receive_only_content_and_targets_have_no_gradient(self):
        predictor = self.fitted()
        images = torch.randn(4, 1, 4, 4, 4, requires_grad=True)
        foreground = torch.ones_like(images, requires_grad=True)
        full_code = torch.randn(4, 5, requires_grad=True)
        seen = []
        hooks = [
            d.register_forward_pre_hook(lambda _, args: seen.append(tuple(args[0].shape))) for d in predictor.decoders
        ]
        loss, _ = vector_prediction_loss(predictor, full_code[:, :3], images, foreground)
        loss.backward()
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen, [(2, 3), (2, 3)])
        self.assertIsNone(images.grad)
        self.assertIsNone(foreground.grad)
        self.assertGreater(full_code.grad[:, :3].abs().sum().item(), 0)
        self.assertEqual(full_code.grad[:, 3:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for d in predictor.decoders for p in d.parameters()))
        self.assertTrue(all(buffer.grad is None for buffer in predictor.buffers()))
        with self.assertRaisesRegex(ValueError, "only"):
            predictor(images)

    def test_target_direction_and_templates_are_swapped(self):
        predictor = self.fitted(channels=1)
        images = torch.randn(4, 1, 2, 2, 2)
        foreground = torch.ones_like(images)
        normalized, _ = prepare_vector_targets(images, foreground, predictor.grid)
        with torch.no_grad():
            predictor.template[0].fill_(2)
            predictor.template[1].fill_(-3)
            for decoder, constant in zip(predictor.decoders, (1.5, -0.5)):
                for parameter in decoder.parameters():
                    parameter.zero_()
                decoder[-1].bias.fill_(constant)
        content = torch.zeros(4, 1)
        prediction = predictor(content)
        torch.testing.assert_close(prediction[:2], torch.full_like(prediction[:2], 1.5))
        torch.testing.assert_close(prediction[2:], torch.full_like(prediction[2:], -0.5))
        _, terms = vector_prediction_loss(predictor, content, images, foreground, deviation_strength=0)
        expected_first = (prediction[:2] - (normalized[2:] - predictor.template[1])).square().mean()
        expected_second = (prediction[2:] - (normalized[:2] - predictor.template[0])).square().mean()
        torch.testing.assert_close(terms["t1_to_flair"], expected_first)
        torch.testing.assert_close(terms["flair_to_t1"], expected_second)
        torch.testing.assert_close(terms["mean"], (expected_first + expected_second) / 2)

    def test_foreground_excludes_image_background_and_prediction_errors(self):
        predictor = self.fitted(channels=1)
        images = torch.randn(2, 1, 4, 4, 4)
        foreground = torch.ones_like(images)
        foreground[..., :2, :2, :2] = 0
        modified = images.clone()
        modified[foreground == 0] = 1e4
        before, keep = prepare_vector_targets(images, foreground, predictor.grid)
        after, _ = prepare_vector_targets(modified, foreground, predictor.grid)
        torch.testing.assert_close(before, after)
        for decoder in predictor.decoders:
            with torch.no_grad():
                for parameter in decoder.parameters():
                    parameter.zero_()
                decoder[-1].bias[0] = 1e5
        loss, _ = vector_prediction_loss(predictor, torch.zeros(2, 1), images, foreground)
        loss.backward()
        self.assertFalse(keep[:, :, 0, 0, 0].any())
        for decoder in predictor.decoders:
            self.assertEqual(decoder[-1].bias.grad[0].item(), 0)
        altered_loss, _ = vector_prediction_loss(predictor, torch.zeros(2, 1), modified, foreground)
        torch.testing.assert_close(loss, altered_loss)

    def test_zscore_is_invariant_to_brain_gain_and_bias_at_partial_cells(self):
        images = torch.randn(4, 1, 4, 4, 4)
        foreground = torch.ones_like(images)
        foreground[:, :, 0, :2, :2] = 0  # one kept cell has occupancy 0.5
        gain = torch.tensor([0.7, 1.3, 2.1, 0.4])[:, None, None, None, None]
        bias = torch.tensor([3.0, -0.2, -1.0, 0.8])[:, None, None, None, None]
        changed = (images * gain + bias) * foreground
        first, keep = prepare_vector_targets(images, foreground, (2, 2, 2))
        second, second_keep = prepare_vector_targets(changed, foreground, (2, 2, 2))
        torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(keep, second_keep)
        self.assertFalse(first.requires_grad)
        for row in first:
            self.assertAlmostEqual(row.mean().item(), 0, places=5)
            self.assertAlmostEqual(row.square().mean().item(), 1, places=5)

    def test_streaming_training_templates_and_state_roundtrip(self):
        images = torch.randn(6, 1, 4, 4, 4, requires_grad=True)
        foreground = torch.ones_like(images)
        foreground[0, :, 0, :2, :2] = 0
        whole = VectorPrediction3d(3, (2, 2, 2), hidden=8)
        whole.accumulate_template(images, foreground)
        whole.finalize_template()
        streamed = VectorPrediction3d(3, (2, 2, 2), hidden=8)
        for indices in ([0, 1, 3, 4], [2, 5]):
            streamed.accumulate_template(images[indices], foreground[indices])
        streamed.finalize_template()
        torch.testing.assert_close(streamed.template, whole.template)
        torch.testing.assert_close(streamed.template_counts, whole.template_counts)
        torch.testing.assert_close(streamed.template_subject_count, torch.tensor([3, 3]))
        self.assertEqual(streamed.template_counts[0, 0, 0, 0, 0].item(), 2.5)
        saved = io.BytesIO()
        torch.save(streamed.state_dict(), saved)
        saved.seek(0)
        restored = VectorPrediction3d(3, (2, 2, 2), hidden=8)
        restored.load_state_dict(torch.load(saved, weights_only=True))
        for name, value in streamed.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        before = restored.template.clone()
        vector_prediction_loss(restored, torch.randn(6, 3), images.detach() + 20, foreground)
        torch.testing.assert_close(restored.template, before)
        with self.assertRaisesRegex(ValueError, "already fitted"):
            restored.accumulate_template(images, foreground)
        restored.reset_template()
        self.assertFalse(restored.template_fitted)
        self.assertEqual(restored.template_subject_count.sum().item(), 0)

    def test_deviation_weights_are_bounded_and_each_direction_is_normalized(self):
        predictor = self.fitted(channels=1)
        with torch.no_grad():
            predictor.template[0].fill_(100)
            predictor.template[1].zero_()
            for decoder in predictor.decoders:
                for parameter in decoder.parameters():
                    parameter.zero_()
        images = torch.randn(4, 1, 2, 2, 2)
        content = torch.zeros(4, 1)
        loss, terms = vector_prediction_loss(
            predictor, content, images, torch.ones_like(images), deviation_strength=100
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(terms["weight_min"].item(), 0.25 - 1e-6)
        self.assertLessEqual(terms["weight_max"].item(), 4 + 1e-6)
        # With cap=1 every kept weight is exactly one, independently of targets.
        uniform, uniform_terms = vector_prediction_loss(
            predictor, content, images, torch.ones_like(images), deviation_strength=100, weight_cap=1
        )
        torch.testing.assert_close(uniform, uniform_terms["unweighted"])
        self.assertEqual(uniform_terms["weight_min"].item(), 1)
        self.assertEqual(uniform_terms["weight_max"].item(), 1)

    def test_local_lesion_changes_the_gradient_into_the_global_vector(self):
        predictor = VectorPrediction3d(1, (2, 2, 2), hidden=2)
        images = torch.arange(8).float().reshape(1, 1, 2, 2, 2).repeat(2, 1, 1, 1, 1)
        foreground = torch.ones_like(images)
        predictor.accumulate_template(images, foreground)
        predictor.finalize_template()
        with torch.no_grad():
            for decoder in predictor.decoders:
                for parameter in decoder.parameters():
                    parameter.zero_()
                decoder[0].weight[0, 0] = 1
                decoder[-1].weight[0, 0] = 1
        content = torch.zeros(2, 1, requires_grad=True)
        loss, _ = vector_prediction_loss(predictor, content, images, foreground)
        baseline_gradient = torch.autograd.grad(loss, content)[0]
        lesion_images = images.clone()
        lesion_images[1, 0, 0, 0, 0] += 20  # FLAIR target: only the T1 input vector must change.
        changed, _ = vector_prediction_loss(predictor, content, lesion_images, foreground)
        changed_gradient = torch.autograd.grad(changed, content)[0]
        self.assertGreater((changed_gradient[0] - baseline_gradient[0]).abs().item(), 0.01)
        torch.testing.assert_close(changed_gradient[1], baseline_gradient[1])

    def test_gauge_makes_each_subjects_prediction_independent_of_its_batch(self):
        predictor = self.fitted().train()
        predictor.standardized(torch.randn(8, 3))
        predictor.eval()
        content = torch.randn(8, 3)
        with torch.no_grad():
            whole, part = predictor(content), predictor(content[[0, 1, 4, 5]])
        torch.testing.assert_close(part, whole[[0, 1, 4, 5]], rtol=0, atol=0)

    def test_gauge_spans_anatomy_neighbourhood_batches(self):
        """Tight batches around scattered centres: the population's spread, not one batch's."""
        predictor = VectorPrediction3d(1, (2, 2, 2), hidden=2).train()
        for _ in range(1500):
            predictor.standardized(torch.randn(1) + 0.01 * torch.randn(8, 1))
        self.assertAlmostEqual(predictor.input_variance.sqrt().mean().item(), 1, delta=0.3)
        self.assertLess(predictor.input_mean.abs().max().item(), 0.5)

    def test_gauge_moves_only_in_training_and_its_statistics_carry_no_gradient(self):
        predictor = self.fitted().train()
        content = torch.randn(8, 3, requires_grad=True)
        predictor.standardized(content)
        self.assertFalse(predictor.input_mean.requires_grad)
        predictor.eval()
        mean, variance = predictor.input_mean.clone(), predictor.input_variance.clone()
        output = predictor.standardized(5 * content)
        torch.testing.assert_close(predictor.input_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(predictor.input_variance, variance, rtol=0, atol=0)
        gradient = torch.autograd.grad(output.sum(), content)[0]
        expected = 5 / variance.sqrt().repeat_interleave(4, dim=0)
        torch.testing.assert_close(gradient, expected)

    def test_step_zero_floor_uses_the_batch_and_stores_no_gauge(self):
        predictor = self.fitted().eval()
        content = torch.randn(8, 3)
        output = predictor.standardized(content)
        self.assertFalse(bool(predictor.input_gauge_fitted))
        torch.testing.assert_close(output[:4].mean(0), torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(output[4:].var(0, unbiased=False), torch.ones(3))

    def quantile_predictor(self, channels=3, components=2, subjects=20):
        predictor = VectorPrediction3d(channels, (4, 4, 4), hidden=8, target="quantile", components=components)
        images = torch.randn(2 * subjects, 1, 8, 8, 8)
        foreground = torch.ones_like(images)
        for start in range(0, subjects, 6):
            pick = list(range(start, min(start + 6, subjects)))
            pick += [subjects + i for i in pick]
            predictor.accumulate_template(images[pick], foreground[pick])
        predictor.finalize_template()
        return predictor, images, foreground

    def test_quantiles_follow_numpy_and_the_eval_reads_the_same_points(self):
        values = torch.randn(5, 37).double()
        torch.testing.assert_close(
            _quantiles(values), torch.from_numpy(np.quantile(values.numpy(), QUANTILES, axis=1).T), rtol=0, atol=1e-12
        )
        self.assertEqual(QUANTILES, burden_recovery.QUANTILES)
        self.assertEqual(vector_prediction.INTERIOR, burden_recovery.INTERIOR)

    def test_quantile_target_keeps_how_much_and_drops_where(self):
        predictor, _, _ = self.quantile_predictor()
        predictor.target_components.zero_()
        residual = torch.randn(2, 64)
        moved = residual.clone()
        moved[:, 5], moved[:, 40] = residual[:, 40].clone(), residual[:, 5].clone()
        torch.testing.assert_close(deviation_quantiles(predictor, moved), deviation_quantiles(predictor, residual))
        bigger = residual.clone()
        bigger[0, 5] = 30
        changed = deviation_quantiles(predictor, bigger) - deviation_quantiles(predictor, residual)
        self.assertGreater(changed[0, -1].item(), 5)
        self.assertEqual(changed[1].abs().max().item(), 0)
        self.assertLess(changed[0, QUANTILES.index(0.5)].abs().item(), 1)

    def test_quantile_fit_uses_training_rows_by_view_and_survives_a_state_roundtrip(self):
        predictor, images, foreground = self.quantile_predictor()
        self.assertEqual(predictor._fit_rows, [])
        normalized, keep = prepare_vector_targets(images, foreground, (4, 4, 4))
        residual = (normalized - predictor.template.repeat_interleave(20, 0)) * keep
        for view, rows in enumerate(residual.flatten(1).chunk(2)):
            expected = torch.linalg.svd(rows, full_matrices=False).Vh[:2]
            overlap = predictor.target_components[view] @ expected.T
            torch.testing.assert_close(overlap.abs(), torch.eye(2), atol=1e-3, rtol=0)
        quantiles = deviation_quantiles(predictor, residual.flatten(1)).chunk(2)
        for view, rows in enumerate(quantiles):
            torch.testing.assert_close(predictor.target_mean[view], rows.mean(0))
            torch.testing.assert_close(predictor.target_sd[view], rows.std(0))
        stream = io.BytesIO()
        torch.save(predictor.state_dict(), stream)
        stream.seek(0)
        other = VectorPrediction3d(3, (4, 4, 4), hidden=8, target="quantile", components=2)
        other.load_state_dict(torch.load(stream, weights_only=True))
        content = torch.randn(40, 3)
        first = vector_prediction_loss(predictor.eval(), content, images, foreground)[0]
        torch.testing.assert_close(vector_prediction_loss(other.eval(), content, images, foreground)[0], first)

    def test_quantile_loss_predicts_the_other_view_through_the_content_only(self):
        predictor, images, foreground = self.quantile_predictor()
        predictor.eval()
        content = torch.randn(40, 3, requires_grad=True)
        traced = images.clone().requires_grad_()
        loss, terms = vector_prediction_loss(predictor, content, traced, foreground)
        self.assertIsNone(torch.autograd.grad(loss, traced, retain_graph=True, allow_unused=True)[0])
        self.assertEqual(terms["weight_min"].item(), terms["weight_max"].item())
        torch.testing.assert_close(terms["mean"], (terms["t1_to_flair"] + terms["flair_to_t1"]) / 2)
        before = torch.autograd.grad(loss, content)[0]
        changed = images.clone()
        changed[20, 0, 3, 3, 3] += 50  # FLAIR subject 0: only the T1 input row may notice.
        moved = torch.autograd.grad(vector_prediction_loss(predictor, content, changed, foreground)[0], content)[0]
        self.assertGreater((moved[0] - before[0]).abs().max().item(), 0)
        torch.testing.assert_close(moved[20], before[20], rtol=0, atol=0)

    def test_quantile_fit_refuses_a_grid_without_interior_and_too_few_subjects(self):
        with self.assertRaisesRegex(ValueError, "finer grid"):
            predictor = VectorPrediction3d(3, (4, 4, 4), target="quantile", components=2)
            images = torch.randn(4, 1, 8, 8, 8)
            foreground = torch.zeros_like(images)
            foreground[:, :, ::2] = 1  # Every cell half brain: kept, but none interior.
            predictor.accumulate_template(images, foreground)
            predictor.finalize_template()
        with self.assertRaisesRegex(ValueError, "components need more"):
            self.quantile_predictor(components=2, subjects=2)
        with self.assertRaisesRegex(ValueError, "spatial or quantile"):
            VectorPrediction3d(3, (4, 4, 4), target="histogram")

    def test_invalid_or_empty_targets_fail_clearly(self):
        predictor = VectorPrediction3d(2, (2, 2, 2))
        images = torch.randn(2, 1, 4, 4, 4)
        with self.assertRaisesRegex(ValueError, "nonempty foreground"):
            prepare_vector_targets(images, torch.zeros_like(images), predictor.grid)
        with self.assertRaisesRegex(ValueError, "one image channel"):
            prepare_vector_targets(images.repeat(1, 2, 1, 1, 1), torch.ones_like(images), predictor.grid)
        with self.assertRaisesRegex(ValueError, "fit the input"):
            prepare_vector_targets(images, torch.ones_like(images), (5, 2, 2))
        with self.assertRaisesRegex(ValueError, "without training"):
            predictor.finalize_template()
        with self.assertRaisesRegex(ValueError, "training subjects first"):
            vector_prediction_loss(predictor, torch.zeros(2, 2), images, torch.ones_like(images))


if __name__ == "__main__":
    unittest.main()
