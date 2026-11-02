"""Position-free local-shape targets and their vector-only prediction path."""

import copy
import io
import unittest

import numpy as np
import torch

from models.patch_distribution import (
    PATCH_QUANTILES,
    fixed_patch_projections,
    projected_patch_quantiles,
    sorted_quantiles,
)
from models.vector_prediction import (
    QUANTILES,
    VectorPrediction3d,
    patch_distribution_targets,
    prepare_vector_targets,
    vector_prediction_loss,
)


class PatchDistributionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(23)

    def test_fixed_filters_are_normalized_reproducible_and_preserve_global_rng(self):
        before = torch.get_rng_state().clone()
        filters = fixed_patch_projections(3, 8, 1729)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(filters.shape, (8, 1, 3, 3, 3))
        torch.testing.assert_close(filters.flatten(1).norm(dim=1), torch.ones(8))
        center = torch.zeros(1, 3, 3, 3)
        center[0, 1, 1, 1] = 1
        torch.testing.assert_close(filters[0], center, rtol=0, atol=0)
        torch.testing.assert_close(filters[1:].flatten(1).mean(1), torch.zeros(7), atol=2e-8, rtol=0)
        torch.testing.assert_close(fixed_patch_projections(3, 8, 1729), filters, rtol=0, atol=0)
        self.assertFalse(torch.equal(fixed_patch_projections(3, 8, 1730)[1:], filters[1:]))

    def test_quantiles_ignore_bag_order_and_match_numpy_interpolation(self):
        values = torch.randn(2, 3, 5, 41, generator=self.generator).double()
        permutation = torch.randperm(41, generator=self.generator)
        expected = np.moveaxis(np.quantile(values.numpy(), PATCH_QUANTILES, axis=-1), 0, -1)
        actual = sorted_quantiles(values)
        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=1e-12)
        torch.testing.assert_close(sorted_quantiles(values[..., permutation]), actual, rtol=0, atol=0)
        self.assertEqual(PATCH_QUANTILES[1:-1], QUANTILES)

    def test_permuting_complete_local_patches_preserves_their_projected_distribution(self):
        origins = ((1, 1, 1), (1, 1, 8), (8, 1, 1), (8, 8, 8))
        patches = torch.randn(2, 4, 1, 3, 3, 3, generator=self.generator)
        interior = torch.zeros(1, 1, 12, 12, 12, dtype=torch.bool)
        original = torch.zeros(2, 1, 12, 12, 12)
        permuted = original.clone()
        permutation = [2, 0, 3, 1]
        for index, (x, y, z) in enumerate(origins):
            selection = (..., slice(x, x + 3), slice(y, y + 3), slice(z, z + 3))
            interior[selection] = True
            original[selection] = patches[:, index]
            permuted[selection] = patches[:, permutation[index]]
        filters = fixed_patch_projections(3, 8, 1729)
        expected = projected_patch_quantiles(original, interior, filters, scales=(1,))
        actual = projected_patch_quantiles(permuted, interior, filters, scales=(1,))
        self.assertEqual(actual.shape, (2, 1, 8, 19))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_local_shape_distinguishes_equal_intensity_histograms(self):
        images = torch.zeros(2, 1, 16, 16, 16)
        images[0, 0, 6:10, 6:10, 6:10] = 1
        for x in (3, 6, 9, 12):
            for y in (3, 6, 9, 12):
                for z in (3, 6, 9, 12):
                    images[1, 0, x, y, z] = 1
        torch.testing.assert_close(images[0].flatten().sort().values, images[1].flatten().sort().values)
        interior = torch.ones(1, 1, 16, 16, 16, dtype=torch.bool)
        descriptors = projected_patch_quantiles(images, interior, fixed_patch_projections(3, 12, 1729), (1,))
        # The centre-voxel projection sees exactly the same marginal histogram.
        torch.testing.assert_close(descriptors[0, :, 0], descriptors[1, :, 0], rtol=0, atol=0)
        self.assertGreater((descriptors[0, :, 1:] - descriptors[1, :, 1:]).abs().max().item(), 0.05)

    def test_prevalence_changes_the_distribution_instead_of_normalizing_each_blob(self):
        images = torch.zeros(2, 1, 16, 16, 16)
        images[:, :, 2:5, 2:5, 2:5] = 1
        images[1, :, 6:9, 6:9, 6:9] = 1
        images[1, :, 10:13, 10:13, 10:13] = 1
        descriptors = projected_patch_quantiles(
            images, torch.ones(1, 1, 16, 16, 16, dtype=torch.bool), fixed_patch_projections(3, 1, 1729), (1,)
        )
        tail = PATCH_QUANTILES.index(0.98)
        self.assertEqual(descriptors[0, 0, 0, -1].item(), descriptors[1, 0, 0, -1].item())
        self.assertGreater(descriptors[1, 0, 0, tail].item(), descriptors[0, 0, 0, tail].item())

    def test_endpoints_retain_one_noise_free_abnormal_voxel_below_existing_tail_cutoffs(self):
        images = torch.zeros(2, 1, 32, 32, 32)
        images[1, 0, 16, 16, 16] = 1
        descriptors = projected_patch_quantiles(
            images, torch.ones(1, 1, 32, 32, 32, dtype=torch.bool), fixed_patch_projections(3, 1, 1729), (1,)
        )
        torch.testing.assert_close(descriptors[0], torch.zeros_like(descriptors[0]), rtol=0, atol=0)
        torch.testing.assert_close(
            descriptors[1, ..., 1:-1], torch.zeros_like(descriptors[1, ..., 1:-1]), rtol=0, atol=0
        )
        self.assertEqual(descriptors[1, 0, 0, -1].item(), 1)

    def test_targets_are_detached_and_invalid_foreground_never_leaks_into_patches(self):
        images = torch.zeros(1, 1, 16, 16, 16)
        images[0, 0, 8, 8, 8] = 100
        images.requires_grad_()
        filters = fixed_patch_projections(3, 8, 1729).requires_grad_()
        interior = torch.ones_like(images, dtype=torch.bool)
        valid = interior.clone()
        valid[0, 0, 8, 8, 8] = False
        descriptors = projected_patch_quantiles(images, interior, filters, (1, 2), valid=valid)
        self.assertFalse(descriptors.requires_grad)
        torch.testing.assert_close(descriptors, torch.zeros_like(descriptors), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "No complete interior patch"):
            projected_patch_quantiles(images, torch.zeros_like(interior), filters, (1,))

    def test_explicit_anatomical_masks_retain_zero_t1_lesions_and_exclude_background(self):
        images = torch.zeros(2, 1, 12, 12, 12)
        foreground = torch.zeros_like(images)
        foreground[..., 2:10, 2:10, 2:10] = 1
        images[foreground.bool()] = 2
        images[0, 0, 6, 6, 6] = 0
        images[1, 0, 6, 6, 6] = 4
        target, keep = prepare_vector_targets(images, foreground, (12, 12, 12))
        self.assertTrue(keep[0, 0, 6, 6, 6])
        self.assertLess(target[0, 0, 6, 6, 6].item(), 0)
        filters = fixed_patch_projections(3, 4, 1729)
        descriptors = projected_patch_quantiles(target, foreground[:1].bool(), filters, (1,), valid=keep)
        self.assertLess(descriptors[0, 0, 0, 0].item(), -1)
        self.assertGreater(descriptors[1, 0, 0, -1].item(), 1)
        changed = images.clone()
        changed[~foreground.bool()] = 1000
        changed_target, changed_keep = prepare_vector_targets(changed, foreground, (12, 12, 12))
        changed_descriptors = projected_patch_quantiles(
            changed_target, foreground[:1].bool(), filters, (1,), valid=changed_keep
        )
        torch.testing.assert_close(changed_descriptors, descriptors, rtol=0, atol=0)


class PatchDistributionPredictionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(19)

    def predictor(self):
        return VectorPrediction3d(
            3,
            (8, 8, 8),
            hidden=8,
            target="patch_distribution",
            components=2,
            patch_size=3,
            patch_scales=(1, 2),
            patch_projections=4,
            patch_seed=1729,
        )

    def fitted_predictor(self):
        predictor = self.predictor()
        subjects = 20
        images = torch.randn(2 * subjects, 1, 8, 8, 8, generator=self.generator)
        foreground = torch.ones_like(images)
        for start in range(0, subjects, 6):
            rows = list(range(start, min(start + 6, subjects)))
            rows += [subjects + row for row in rows]
            predictor.accumulate_template(images[rows], foreground[rows])
        predictor.finalize_template()
        return predictor, images, foreground

    def test_train_fitted_population_features_are_frozen_during_validation_and_replay_exactly(self):
        predictor, train_images, foreground = self.fitted_predictor()
        self.assertEqual(predictor.template_subject_count.tolist(), [20, 20])
        self.assertEqual(predictor._fit_rows, [])
        normalized, keep = prepare_vector_targets(train_images, foreground, predictor.grid)
        residual = ((normalized - predictor.template.repeat_interleave(20, 0)) * keep).flatten(1)
        features = patch_distribution_targets(predictor, residual, keep.flatten(1)).reshape(2, 20, -1)
        torch.testing.assert_close(predictor.target_mean, features.mean(1))
        torch.testing.assert_close(predictor.target_sd, features.std(1).clamp_min(1e-6))
        for view, rows in enumerate(residual.chunk(2)):
            basis = torch.linalg.svd(rows, full_matrices=False).Vh[:2]
            torch.testing.assert_close(
                (predictor.target_components[view] @ basis.T).abs(), torch.eye(2), atol=1e-3, rtol=0
            )

        predictor.eval()
        before = copy.deepcopy(predictor.state_dict())
        validation = torch.randn(10, 1, 8, 8, 8, generator=self.generator)
        validation[:, :, 2:5, 2:5, 2:5] += 4
        content = torch.randn(10, 3, generator=self.generator)
        expected, _ = vector_prediction_loss(predictor, content, validation, torch.ones_like(validation))
        self.assertTrue(torch.isfinite(expected))
        for name, value in predictor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "already fitted"):
            predictor.accumulate_template(validation, torch.ones_like(validation))

        stream = io.BytesIO()
        torch.save(predictor.state_dict(), stream)
        stream.seek(0)
        restored = self.predictor().eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        actual, _ = vector_prediction_loss(restored, content, validation, torch.ones_like(validation))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_loss_sends_gradients_to_every_content_unit_and_both_decoders_but_not_targets(self):
        predictor, images, foreground = self.fitted_predictor()
        predictor.eval()
        content = torch.randn(40, 3, generator=self.generator, requires_grad=True)
        images = images.clone().requires_grad_()
        loss, terms = vector_prediction_loss(predictor, content, images, foreground)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(terms["mean"], (terms["t1_to_flair"] + terms["flair_to_t1"]) / 2)
        before = torch.autograd.grad(loss, content, retain_graph=True)[0]
        changed = images.detach().clone()
        changed[20, 0, 3, 3, 3] += 50  # FLAIR subject 0 changes only its T1-input target.
        moved = torch.autograd.grad(vector_prediction_loss(predictor, content, changed, foreground)[0], content)[0]
        self.assertGreater((moved[0] - before[0]).abs().max().item(), 0)
        torch.testing.assert_close(moved[20], before[20], rtol=0, atol=0)
        loss.backward()
        self.assertIsNone(images.grad)
        self.assertTrue(torch.isfinite(content.grad).all())
        self.assertTrue((content.grad.abs().sum(0) > 0).all())
        for decoder in predictor.decoders:
            for parameter in decoder.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_decoding_orders_quantiles_in_the_opposite_views_physical_units(self):
        predictor = self.predictor().eval()
        outputs = predictor.target_mean.shape[1]
        with torch.no_grad():
            predictor.target_mean[0].copy_(torch.linspace(-5, 8, outputs))
            predictor.target_mean[1].copy_(torch.linspace(80, -30, outputs))
            predictor.target_sd[0].copy_(torch.linspace(0.1, 2, outputs))
            predictor.target_sd[1].copy_(torch.linspace(10, 1, outputs))
            for decoder in predictor.decoders:
                for parameter in decoder.parameters():
                    parameter.zero_()
            predictor.decoders[0][-1].bias.copy_(torch.linspace(1, -2, outputs))
            predictor.decoders[1][-1].bias.copy_(torch.linspace(-3, 4, outputs))
        content = torch.randn(4, 3, generator=self.generator)
        decoded = predictor(content)
        for input_view in range(2):
            target_view = 1 - input_view
            mean, sd = predictor.target_mean[target_view], predictor.target_sd[target_view]
            physical = (decoded[input_view * 2 : (input_view + 1) * 2] * sd + mean).reshape(2, -1, 19)
            raw = predictor.decoders[input_view][-1].bias * sd + mean
            expected = raw.reshape(-1, 19).sort(dim=-1).values
            torch.testing.assert_close(physical, expected.expand(2, -1, -1), rtol=1e-6, atol=1e-5)
            self.assertTrue((physical.diff(dim=-1) >= -1e-5).all())

    def test_same_shape_checkpoints_reject_different_patch_scale_order_or_projection_seed(self):
        original = self.predictor()
        state = original.state_dict()
        settings = dict(
            content_channels=3,
            grid=(8, 8, 8),
            hidden=8,
            target="patch_distribution",
            components=2,
            patch_size=3,
            patch_scales=(1, 2),
            patch_projections=4,
            patch_seed=1729,
        )
        for change in (dict(patch_scales=(2, 1)), dict(patch_seed=1730)):
            with self.subTest(change=change):
                changed = VectorPrediction3d(**{**settings, **change})
                self.assertEqual(set(changed.state_dict()), set(state))
                self.assertEqual(
                    {name: value.shape for name, value in changed.state_dict().items()},
                    {name: value.shape for name, value in state.items()},
                )
                with self.assertRaisesRegex(RuntimeError, "descriptor settings do not match"):
                    changed.load_state_dict(state)

    def test_failed_population_fit_does_not_mark_incomplete_targets_as_fitted(self):
        predictor = self.predictor()
        images = torch.randn(40, 1, 8, 8, 8, generator=self.generator)
        foreground = torch.zeros_like(images)
        foreground[..., ::2, :, :] = 1
        predictor.accumulate_template(images, foreground)
        with self.assertRaisesRegex(ValueError, "No complete interior patch"):
            predictor.finalize_template()
        self.assertFalse(bool(predictor.template_fitted))


if __name__ == "__main__":
    unittest.main()
