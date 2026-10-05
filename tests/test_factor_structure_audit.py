"""Known subspaces, real image replay, ensemble covariance and renderer applicability."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from eval.encoder.encoder_target_protocol import dataset, digest
from eval.synthetic import factor_structure_audit as audit
from eval.synthetic import latent_spatial_kernels as kernels
from eval.synthetic.synthetic_dataset import LesionPlacementError
from tests.test_encoder_target_followups import config


class FactorStructureTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_known_confounded_distinct_rank_deficient_and_zero_directions(self):
        maps = np.zeros((9, 64))
        for k, coordinate in zip(audit.ANATOMY, range(5)):
            maps[k, coordinate] = 1
        maps[2, 0], maps[3, 5], maps[4, 0], maps[4, 5], maps[8, 6] = 1, 1, 1, 1, 1
        active, valid = np.ones(9, bool), np.ones(9, bool)
        rows, blocks, cosines, normalized, gram, norms = audit.response_geometry(maps, active, valid)
        np.testing.assert_allclose([rows[k]["residual_energy_vs_anatomy"] for k in (2, 3, 4)], [0, 1, 0.5], atol=1e-12)
        residual = audit.residual_map(4, audit.ANATOMY, normalized, gram, norms, 1e-5)
        np.testing.assert_allclose(residual, maps[3], atol=1e-12)
        lesion = next(r for r in blocks if r["block"] == "lesion_vs_anatomy")
        self.assertEqual((lesion["target_rank"], lesion["reference_rank"], lesion["residual_target_rank"]), (2, 5, 1))
        self.assertFalse(lesion["target_full_rank"])
        self.assertLess(lesion["min_principal_angle_deg"], 1e-5)
        self.assertLess(lesion["min_residual_singular_value"], 1e-7)
        sulcal = next(r for r in blocks if r["block"] == "sulcal_vs_anatomy")
        self.assertAlmostEqual(sulcal["min_principal_angle_deg"], 90)
        maps[8], active[8] = 0, False
        rows, blocks, *_ = audit.response_geometry(maps, active, valid)
        sulcal = next(r for r in blocks if r["block"] == "sulcal_vs_anatomy")
        self.assertEqual(sulcal["target_rank"], 0)
        self.assertTrue(np.isnan(sulcal["min_principal_angle_deg"]))
        self.assertTrue(np.isnan(rows[8]["residual_energy_vs_anatomy"]))
        valid[0] = False
        rows, blocks, *_ = audit.response_geometry(maps, active, valid)
        self.assertTrue(np.isnan(rows[3]["residual_energy_vs_anatomy"]))
        self.assertFalse(blocks[0]["complete"])

    def test_footprints_distinguish_spatial_extent_without_amplitude_bias(self):
        regions = audit.region_indices((8, 8, 8), 4)
        local = np.zeros((8, 8, 8))
        local[1, 1, 1], local[1, 1, 2] = 1, -1
        global_map = np.ones((8, 8, 8))
        a, b = [audit.footprint(x, 0.5, regions, 1e-6) for x in (local, global_map)]
        self.assertEqual((a["gap_survival"], b["gap_survival"]), (0, 1))
        self.assertLess(a["energy90_voxel_fraction"], b["energy90_voxel_fraction"])
        self.assertLess(a["effective_regions"], b["effective_regions"])
        scaled = audit.footprint(local * 100, 0.5, regions, 1e-6)
        self.assertEqual(a["energy90_voxel_fraction"], scaled["energy90_voxel_fraction"])

    def test_restoration_matches_existing_recipe_and_preserves_gp_options(self):
        cfg = config()
        old = dataset(cfg, 64, "test")
        new, _ = audit.restore_dataset(cfg, 64)
        for v in range(2):
            torch.testing.assert_close(old[0]["image"][v], new[0]["image"][v], atol=0, rtol=0)
        gp, _ = audit.restore_dataset(
            config(
                synthetic_field_prior="gp",
                synthetic_field_grid=4,
                synthetic_field_kernels="repeated",
                synthetic_field_lengthscales=[0.5, 1.5],
            ),
            64,
        )
        self.assertEqual(gp._inner.field_prior, "gp")
        self.assertEqual(gp._inner.field_kernels, "repeated")
        self.assertEqual(gp._inner.field_lengthscales, (0.5, 1.5))

    def test_render_replays_native_anatomy_and_only_perturbs_requested_control(self):
        ds, _ = audit.restore_dataset(config(), 64)
        ctx = audit.context(ds, 0)
        original = ctx["lat"]["z_content"].clone()
        first, _ = audit.render(ds, ctx, original)
        second, _ = audit.render(ds, ctx, original)
        np.testing.assert_array_equal(first, second)
        for k in (0, 2, 8):
            z = original.clone()
            z[k] += 0.25
            raw = ds._inner.render_pseudo_mri(
                z,
                ctx["lat"]["z_deformation"],
                ctx["lat"]["z_fissure"],
                ctx["lat"]["z_style_v1"],
                ctx["lat"]["z_style_v2"],
                ctx["seed"],
            )
            expected = np.stack(
                [((x * gain + bias) * raw[2]).numpy()[0] for x, (gain, bias) in zip(raw[:2], ctx["affines"])]
            )
            actual, _ = audit.render(ds, ctx, z)
            np.testing.assert_array_equal(actual, expected)
        torch.testing.assert_close(original, ctx["lat"]["z_content"], atol=0, rtol=0)

    def test_covariance_uses_ensemble_centering_and_preserves_constant_random_fields(self):
        rng = np.random.default_rng(22)
        samples = rng.normal(size=(100, 64)) + np.arange(64)[None] * 10
        np.testing.assert_allclose(kernels.covariance(samples), np.cov(samples, rowvar=False), atol=1e-12)
        # A field constant in space but random between subjects must retain variance.
        constant = np.repeat(rng.normal(size=(100, 1)), 64, axis=1)
        cov, normalized = kernels.lag_profiles(constant, 4)
        self.assertGreater(cov[0, 0], 0.5)
        np.testing.assert_allclose(normalized, 1, atol=1e-12)
        np.testing.assert_allclose(
            kernels.kernel_distance(kernels.covariance(samples), 7 * kernels.covariance(samples)), 0, atol=1e-12
        )

    def test_distinct_spatial_dependencies_produce_different_kernel_shapes(self):
        rng = np.random.default_rng(41)
        n, grid = 600, 4
        short = rng.normal(size=(n, grid**3))
        long = rng.normal(size=(n, 1)) + 0.1 * rng.normal(size=(n, grid**3))
        _, short_profile = kernels.lag_profiles(short, grid)
        _, long_profile = kernels.lag_profiles(long, grid)
        self.assertLess(abs(short_profile[:, 1:].mean()), 0.05)
        self.assertGreater(long_profile[:, 1:].mean(), 0.9)
        self.assertGreater(kernels.kernel_distance(kernels.covariance(short), kernels.covariance(long)), 1)

    def test_full_cli_outputs_exact_niftis_kernels_and_explicit_inactive_paths(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "settings.json").write_text(json.dumps(config()))
            before = digest(run / "settings.json")
            out = root / "out"
            argv = [
                "--run-dir",
                str(run),
                "--out-dir",
                str(out),
                "--num-samples",
                "2",
                "--subject-offset",
                "0",
                "--eps",
                ".1",
                ".5",
                "--kernel-samples",
                "16",
                "--kernel-grid",
                "4",
                "--kernel-bootstrap",
                "12",
                "--nifti-subjects",
                "1",
            ]
            audit.main(argv)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(digest(run / "settings.json"), before)
            components = report["kernels"]["components"]
            for info in components.values():
                self.assertGreater(info["average_raw_variance"], 0)
                self.assertEqual(info["structural_multiplier"], 0)
                self.assertFalse(info["active_structural_path"])
            self.assertFalse(report["kernels"]["applicability"]["theorem_guarantee_established"])
            with np.load(out / "latent_covariances.npz") as bank:
                self.assertEqual(bank["scalar_covariance"].shape, (9, 9))
                self.assertEqual(bank["z_deformation_covariance"].shape, (64, 64))
                np.testing.assert_array_equal(bank["z_deformation_effective_covariance"], 0)
            ds, _ = audit.restore_dataset(config(), 64)
            ctx = audit.context(ds, 0)
            a, b = ctx["lat"]["z_content"].clone(), ctx["lat"]["z_content"].clone()
            a[2] -= 0.5
            b[2] += 0.5
            expected = audit.render(ds, ctx, b)[0][0] - audit.render(ds, ctx, a)[0][0]
            image = nib.load(out / "nifti" / "subject0_t1_eps0.5_lesion_x_delta.nii.gz")
            np.testing.assert_array_equal(image.get_fdata(), expected)
            self.assertEqual(image.header.get_xyzt_units()[0], "unknown")
            self.assertTrue((out / "response_summary.png").is_file())
            self.assertTrue((out / "kernel_diagnostics.png").is_file())
            self.assertFalse(list(out.rglob("*.pt")))
            with self.assertRaises(FileExistsError):
                audit.main(argv)

    def test_invalid_placements_are_counted_not_redrawn_or_treated_as_zero(self):
        ds, _ = audit.restore_dataset(config(), 64)
        original = audit.render
        ctx = audit.context(ds, 0)
        baseline = float(ctx["lat"]["z_content"][0])

        def fail_size(dataset, context, content):
            if float(content[0]) != baseline:
                raise LesionPlacementError("test no room")
            return original(dataset, context, content)

        args = SimpleNamespace(
            region_grid=4,
            nifti_subjects=0,
            subject_offset=0,
            num_samples=1,
            eps=[0.5],
            response_atol=1e-6,
            svd_rtol=1e-5,
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            audit, "render", side_effect=fail_size
        ), contextlib.redirect_stdout(io.StringIO()):
            result = audit.response_audit(ds, args, Path(tmp))
            self.assertEqual(result["n_failed_pairs"], 1)
            with (Path(tmp) / "responses.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            failure = next(r for r in rows if r["factor"] == "brain_size")
            self.assertEqual(failure["status"], "render_failed")
            lesion = next(r for r in rows if r["factor"] == "lesion_x")
            self.assertEqual(lesion["anatomy_reference_complete"], "False")
            self.assertEqual(lesion["residual_energy_vs_anatomy"], "")


if __name__ == "__main__":
    unittest.main()
