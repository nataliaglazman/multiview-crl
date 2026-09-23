"""Complete spheres in final WM, latent control, and legacy data compatibility."""

import argparse
import ast
import unittest
from pathlib import Path

import numpy as np
import torch

from data.datasets import SyntheticBrainDataset
from eval.synthetic_dataset import PseudoMRIRenderer


class PlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def render(self, renderer, z=None, clean=True):
        return renderer.render_structure(
            torch.zeros(9) if z is None else z, torch.zeros(4, 4, 4), torch.zeros(8, 8, 8), "cpu", clean=clean
        )

    def assert_sphere(self, renderer, tissue, load):
        support = load.bool()
        self.assertTrue(support.any())
        self.assertTrue(torch.all(tissue[support] == 2))
        positions = torch.nonzero(support).float()
        index = positions.mean(0)
        torch.testing.assert_close(index, index.round(), atol=1e-5, rtol=0)
        centre = renderer.coords[tuple(index.round().long())]
        expected = (torch.norm(renderer.coords - centre, dim=-1) < renderer.lesion_radius).float()
        torch.testing.assert_close(load, expected, atol=0, rtol=0)
        return int(load.sum())

    def test_full_sphere_no_csf_fissure_or_cortex_overlap(self):
        for identifiable in (False, True):
            r = PseudoMRIRenderer(res=64, identifiable_ventricle=identifiable, lesion_placement="wm_interior")
            counts = []
            for seed in range(24):
                z = torch.randn(9, generator=torch.Generator().manual_seed(seed))
                tissue, load = self.render(r, z)
                counts.append(self.assert_sphere(r, tissue, load))
                again = self.render(r, z)
                torch.testing.assert_close(load, again[1], atol=0, rtol=0)
            self.assertEqual(len(set(counts)), 1)

    def test_each_position_latent_controls_its_coordinate(self):
        r = PseudoMRIRenderer(res=64, lesion_placement="wm_interior")
        for axis in range(3):
            centres = []
            for value in torch.linspace(-2, 2, 11):
                z = torch.zeros(9)
                z[2 + axis] = value
                tissue, load = self.render(r, z)
                self.assert_sphere(r, tissue, load)
                centres.append(float(torch.nonzero(load).float().mean(0)[axis]))
            self.assertGreater(len(set(centres)), 4)
            self.assertTrue(np.all(np.diff(centres) >= 0))

    def test_admissible_quantiles_handle_endpoints_and_cavities(self):
        r = PseudoMRIRenderer(res=32, lesion_radius=0.14, lesion_placement="wm_interior")
        tissue = torch.full((32, 32, 32), 2)
        tissue[14:18] = 1  # disconnected halves separated by CSF
        for u in (-1.0, 0.0, 1.0):
            load = r._sphere_in_white_matter(tissue, torch.tensor([u, u, u]))
            self.assert_sphere(r, tissue, load)
            full_size = (torch.norm(r.coords - r.coords[16, 16, 16], dim=-1) < r.lesion_radius).sum()
            self.assertEqual(int(load.sum()), int(full_size))

    def test_no_fit_and_invalid_configuration_fail_instead_of_shrinking(self):
        r = PseudoMRIRenderer(res=32, lesion_radius=0.4, lesion_placement="wm_interior")
        tissue = torch.zeros(32, 32, 32, dtype=torch.long)
        tissue[10:15, 10:15, 10:15] = 2
        with self.assertRaisesRegex(ValueError, "will not shrink or discard"):
            r._sphere_in_white_matter(tissue, torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "No white matter"):
            r._sphere_in_white_matter(tissue * 0, torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "bounded prior"):
            r._sphere_in_white_matter(tissue, torch.tensor([2.0, 0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "only for sphere"):
            PseudoMRIRenderer(lesion_mode="field", lesion_placement="wm_interior")
        with self.assertRaisesRegex(ValueError, "positive finite"):
            PseudoMRIRenderer(lesion_radius=0, lesion_placement="wm_interior")

    def test_legacy_formula_and_anatomy_unchanged(self):
        old = PseudoMRIRenderer(res=64)
        fixed = PseudoMRIRenderer(res=64, lesion_placement="wm_interior")
        z = torch.zeros(9)
        z[2:5] = torch.tensor([0.2, -0.5, 0.7])
        tissue, load = self.render(old, z)
        centre = torch.tanh(z[2:5]) * (0.38 / np.sqrt(3))
        expected = ((torch.norm(old.coords - centre, dim=-1) < 0.1) & (torch.norm(old.coords, dim=-1) < 0.5)).float()
        torch.testing.assert_close(load, expected, rtol=0, atol=0)
        new_tissue, new_load = self.render(fixed, z)
        torch.testing.assert_close(tissue, new_tissue, rtol=0, atol=0)
        self.assert_sphere(fixed, new_tissue, new_load)
        self.assertTrue(bool((load.bool() & (tissue != 2)).any()))

    def test_dataset_and_training_evaluation_factories_forward_setting(self):
        # Extract only this factory, avoiding unrelated training / MONAI imports.
        file = Path(__file__).resolve().parents[1] / "training/main_conv_synthetic.py"
        tree = ast.parse(file.read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_dataset")
        namespace = {"SyntheticBrainDataset": SyntheticBrainDataset}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), str(file), "exec"), namespace)
        args = argparse.Namespace(
            cache=False,
            res=64,
            synthetic_mode="pseudo_mri",
            seed=42,
            n_content=9,
            n_style=3,
            synthetic_style_scale=1.0,
            synthetic_content_scale=1.0,
            synthetic_n_deformation_grid=4,
            synthetic_n_fissure_grid=8,
            synthetic_hierarchical_content=False,
            synthetic_normalize="fixed_reference",
            synthetic_causal=False,
            synthetic_clean_content=True,
            synthetic_lesion_placement="wm_interior",
        )
        for split in ("train", "val", "test"):
            ds = namespace["make_dataset"](args, split, 2)
            self.assertEqual(ds._inner.renderer.lesion_placement, "wm_interior")
            a, b, lat = ds._inner[0]
            tissue, load = ds._inner.renderer.render_structure(
                lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=True
            )
            self.assert_sphere(ds._inner.renderer, tissue, load)
            replay = ds._inner.render_pseudo_mri(
                lat["z_content"],
                lat["z_deformation"],
                lat["z_fissure"],
                lat["z_style_v1"],
                lat["z_style_v2"],
                ds._inner.sample_seed_for(0),
            )
            torch.testing.assert_close(a, replay[0], rtol=0, atol=0)
            torch.testing.assert_close(b, replay[1], rtol=0, atol=0)
        from eval.run_dci_synthetic import build_synthetic_test_set

        eval_args = argparse.Namespace(
            synthetic_res=64,
            synthetic_n_content=9,
            synthetic_lesion_placement="wm_interior",
            synthetic_clean_content=True,
        )
        ds = build_synthetic_test_set(eval_args, num_samples=2, causal=False, cache=False)
        self.assertEqual(ds._inner.renderer.lesion_placement, "wm_interior")
        file = Path(__file__).resolve().parents[1] / "eval/score_checkpoint.py"
        tree = ast.parse(file.read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_val_dataset")
        exec(compile(ast.Module(body=[factory], type_ignores=[]), str(file), "exec"), namespace)
        scored = namespace["make_val_dataset"](vars(args), 2)
        trained_val = namespace["make_dataset"](args, "val", 2)
        self.assertEqual(scored._inner.renderer.lesion_placement, "wm_interior")
        for a, b in zip(scored._inner[0][:2], trained_val._inner[0][:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        del args.synthetic_lesion_placement
        self.assertEqual(namespace["make_dataset"](args, "val", 2)._inner.renderer.lesion_placement, "legacy")


if __name__ == "__main__":
    unittest.main()
