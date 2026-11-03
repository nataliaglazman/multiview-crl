"""Independent held-out probes split the exported blocks without changing training."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from eval.encoder.residual_head_recovery import _rendered_burden, residual_head_recovery


class Subjects(Dataset):
    def __init__(self, count=80, lesion_target="position", rendered=False):
        generator = np.random.default_rng(13)
        self.values = generator.normal(size=(count, 2)).astype(np.float32)
        self.lesion_target = lesion_target
        if rendered:
            self.values[:, 1] = np.arange(count) % 7 + 1
            self._inner = SimpleNamespace(
                _draw_pseudo_mri=self._draw,
                sample_seed_for=lambda index: index,
                renderer=SimpleNamespace(render_structure=self._render),
                clean_content=True,
            )

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        # A loader or renderer can consume every random stream without changing
        # the optimization stream after this diagnostic returns.
        random.random()
        np.random.random()
        torch.rand(())
        image = torch.from_numpy(self.values[index]).reshape(1, 2, 1, 1)
        return {
            "image": (image, image * 0.8 + 0.2),
            "gt_latents": {"z_content": torch.tensor([*self.values[index], self.values[index, 1]])},
        }

    def _draw(self, seed):
        return {
            "z_content": self.values[seed],
            "z_deformation": None,
            "z_fissure": None,
            "z_lesion": None,
        }

    @staticmethod
    def _render(content, deformation, fissure, device, **kwargs):
        lesion = torch.zeros(2, 2, 2)
        lesion.flatten()[: int(content[1])] = 1
        return torch.zeros_like(lesion), lesion


class SeparateCodes(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(0.2)
        self.scale = nn.Parameter(torch.ones(()))
        self.content_channels = 1
        self.residual_units = 1
        self.residual_heads = nn.Identity()
        self.residual_heads.input_size = 2

    def forward(self, images, **kwargs):
        values = self.dropout(self.norm(images.flatten(1))) * self.scale
        codes = torch.stack((values[:, 0], torch.ones_like(values[:, 0]), values[:, 1]), dim=1)
        return None, None, [codes]

    @staticmethod
    def unit_split(width):
        assert width == 3
        return [0, 2], [1]


class ResidualHeadRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def assert_numpy_state_equal(self, first, second):
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:], second[2:])

    def test_separate_blocks_recover_their_own_factors_and_combined_recovers_both(self):
        result = residual_head_recovery(SeparateCodes(), Subjects(), "cpu", 11)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["folds"], 5)
        self.assertTrue(result["probe"]["evaluation_only"])
        self.assertFalse(result["rendered_burden_available"])
        first, second = result["factor_names"][:2]
        for view in ("t1", "flair"):
            blocks = result["views"][view]
            self.assertEqual(
                [blocks[key]["width"] for key in ("image", "residual", "combined")],
                [1, 1, 2],
            )
            self.assertGreater(blocks["image"]["r2"][first], 0.99)
            self.assertLess(blocks["image"]["r2"][second], 0.2)
            self.assertGreater(blocks["residual"]["r2"][second], 0.99)
            self.assertLess(blocks["residual"]["r2"][first], 0.2)
            self.assertGreater(blocks["combined"]["r2"][first], 0.99)
            self.assertGreater(blocks["combined"]["r2"][second], 0.99)
        # Position-target factor 2 must retain its actual meaning.
        self.assertNotIn("lesion_burden", result["factor_names"])

    def test_preserves_buffers_modes_gradients_and_random_streams_even_on_failure(self):
        model = SeparateCodes().train()
        model.dropout.eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        modes = [module.training for module in model.modules()]
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        residual_head_recovery(model, Subjects(12), "cpu", 5)
        for name, expected in initial.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(
            all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters())
        )
        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_state_equal(np.random.get_state(), numpy_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        with patch(
            "eval.encoder.residual_head_recovery._out_of_fold_r2",
            side_effect=RuntimeError("probe failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                residual_head_recovery(model, Subjects(12), "cpu", 5)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_state_equal(np.random.get_state(), numpy_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        for name, expected in initial.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_actual_rendered_voxels_are_a_separate_target_from_burden_latent(self):
        dataset = Subjects(40, lesion_target="burden", rendered=True)
        np.testing.assert_allclose(_rendered_burden(SeparateCodes(), dataset), dataset.values[:, 1])
        result = residual_head_recovery(SeparateCodes(), dataset, "cpu", 10)
        self.assertIn("lesion_burden", result["factor_names"])
        self.assertIn("rendered_burden_voxels", result["factor_names"])
        self.assertTrue(result["rendered_burden_available"])
        for view in result["views"].values():
            self.assertGreater(view["residual"]["r2"]["rendered_burden_voxels"], 0.99)
        # Position targets may also have a rendered mask, without being renamed.
        dataset.lesion_target = "position"
        position = residual_head_recovery(SeparateCodes(), dataset, "cpu", 10)
        self.assertNotIn("lesion_burden", position["factor_names"])
        self.assertTrue(position["rendered_burden_available"])

    def test_subject_identity_cannot_memorize_random_targets_across_outer_folds(self):
        subjects = Subjects(30)
        model = SeparateCodes()
        model.content_channels = len(subjects)
        # In-sample regression could memorize every label through this identity
        # matrix; an outer held-out subject has no learned identity coefficient.
        codes = np.column_stack((np.eye(len(subjects)), np.zeros(len(subjects))))
        targets = np.random.default_rng(9).normal(size=(len(subjects), 2))
        with patch(
            "eval.protocol.score_checkpoint.encode_blocks",
            return_value={"content": (codes, codes), "z_content": targets},
        ):
            report = residual_head_recovery(model, subjects, "cpu", 7)
        for view in report["views"].values():
            self.assertTrue(all(score < 0.1 for score in view["image"]["r2"].values()))
            self.assertTrue(all(score < 0.1 for score in view["combined"]["r2"].values()))

    def test_tiny_cohorts_disabled_models_and_constant_factors(self):
        model = SeparateCodes()
        model.residual_units = 0
        self.assertIsNone(residual_head_recovery(model, Subjects(), "cpu", 8))
        model.residual_units = 1
        short = residual_head_recovery(model, Subjects(3), "cpu", 2)
        self.assertEqual(short["status"], "insufficient_subjects")
        self.assertEqual(short["views"], {})
        four = Subjects(4)
        four.values[:, 1] = 1
        report = residual_head_recovery(model, four, "cpu", 2)
        self.assertEqual(report["folds"], 4)
        constant = report["factor_names"][1]
        self.assertTrue(
            all(block["r2"][constant] is None for view in report["views"].values() for block in view.values())
        )


if __name__ == "__main__":
    unittest.main()
