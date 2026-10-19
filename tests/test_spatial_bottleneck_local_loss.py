"""Hard spatial errors, foreground exclusion, and checkpoint/evaluator compatibility."""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import torch
from test_spatial_bottleneck import source_model, trainer_argv, write_source_run
from torch.utils.data import DataLoader

from eval.protocol import spatial_bottleneck_eval as factor_eval
from models.spatial_bottleneck import SpatialBottleneck, paired_objective, reconstruction_terms
from training import spatial_bottleneck_experiment as experiment


class LocalReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_top_fraction_is_per_subject_view_and_keeps_partial_edge_patches(self):
        # Three patches with means 1, 5, 9; the final patch contains only one cell.
        # ceil(0.5 * 3) selects two patches in EACH of the four maps.
        error = torch.tensor([1.0, 1.0, 5.0, 5.0, 9.0]).reshape(1, 1, 1, 1, 1, 5)
        amplitudes = torch.arange(1.0, 5.0).reshape(2, 2, 1, 1, 1, 1)
        error = (error * amplitudes).expand(2, 2, 2, 1, 1, 5).clone().requires_grad_()
        terms = reconstruction_terms(error, hard_patch_weight=0.5, hard_patch_fraction=0.5, hard_patch_size=(1, 1, 2))
        self.assertAlmostEqual(terms["reconstruction_mean"].item(), 4.2 * 2.5, places=5)
        self.assertAlmostEqual(terms["reconstruction_hard"].item(), 7 * 2.5, places=5)
        self.assertAlmostEqual(terms["reconstruction"].item(), (4.2 + 0.5 * 7) * 2.5, places=5)
        terms["reconstruction"].backward()
        self.assertGreater(error.grad[0, 0, 0, 0, 0, 4], error.grad[0, 0, 0, 0, 0, 0])
        self.assertGreater(error.grad[0, 0, 0, 0, 0, 2], error.grad[0, 0, 0, 0, 0, 0])

    def test_background_cannot_affect_loss_selection_or_gradients(self):
        error = torch.tensor([1.0, 1.0, 1e9, 1e9, 9.0]).reshape(1, 1, 1, 1, 1, 5).repeat(2, 2, 2, 1, 1, 1)
        error.requires_grad_()
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.5]).reshape(1, 1, 1, 1, 1, 5).repeat(2, 2, 1, 1, 1, 1)
        mask.requires_grad_()
        options = dict(foreground=mask, hard_patch_weight=1.0, hard_patch_fraction=0.5, hard_patch_size=(1, 1, 2))
        terms = reconstruction_terms(error, **options)
        self.assertAlmostEqual(terms["reconstruction_mean"].item(), 2.6, places=6)
        self.assertEqual(terms["reconstruction_hard"].item(), 9.0)
        changed = error.detach().clone()
        changed[..., 2:4] = 0
        torch.testing.assert_close(terms["reconstruction"], reconstruction_terms(changed, **options)["reconstruction"])
        terms["reconstruction"].backward()
        self.assertEqual(error.grad[..., 2:4].abs().sum().item(), 0)
        self.assertIsNone(mask.grad)
        # A patch bigger than the whole map is one partial patch, with no dilution by padding.
        one = reconstruction_terms(error, foreground=mask, hard_patch_weight=1, hard_patch_size=(4, 4, 8))
        torch.testing.assert_close(one["reconstruction_hard"], one["reconstruction_mean"])

    def test_foreground_renormalises_existing_weights_and_rejects_empty_support(self):
        error = torch.tensor([1.0, 9.0, 1000.0]).reshape(1, 1, 1, 1, 1, 3).repeat(2, 2, 2, 1, 1, 1)
        mask = torch.tensor([1.0, 1.0, 0.0]).reshape(1, 1, 1, 1, 1, 3).repeat(2, 2, 1, 1, 1, 1)
        weights = torch.tensor([3.0, 1.0, 1000.0], requires_grad=True)
        result = reconstruction_terms(error, weights, mask)
        self.assertEqual(result["reconstruction"].item(), 3.0)
        torch.testing.assert_close(reconstruction_terms(error)["reconstruction"], error.mean(), rtol=0, atol=0)
        for invalid in (torch.zeros_like(mask), mask * math.nan, mask * 2):
            with self.assertRaises(ValueError):
                reconstruction_terms(error, foreground=invalid)
        with self.assertRaisesRegex(ValueError, "positive mass"):
            reconstruction_terms(error, torch.zeros(3), mask)

    def test_foreground_extraction_preserves_maps_and_uses_each_views_input_mask(self):
        rows = []
        for i in range(3):
            mask1, mask2 = torch.zeros(1, 12, 14, 16), torch.zeros(1, 12, 14, 16)
            mask1[:, : 5 + i] = 1
            mask2[:, :, : 7 + i] = 1
            rows.append(dict(image=[torch.randn_like(mask1), torch.randn_like(mask2)], mask=[mask1, mask2]))
        source = source_model()
        before = {k: v.clone() for k, v in source.state_dict().items()}
        plain, indices = experiment.extract_content_maps(source, DataLoader(rows, batch_size=2), "cpu")
        maps, selected, foreground = experiment.extract_content_maps(
            source, DataLoader(rows, batch_size=2), "cpu", return_foreground=True
        )
        torch.testing.assert_close(maps, plain, rtol=0, atol=0)
        self.assertEqual(indices, selected)
        expected = torch.stack(
            [torch.nn.functional.adaptive_avg_pool3d(torch.stack(row["mask"]), maps.shape[-3:]) for row in rows]
        )
        torch.testing.assert_close(foreground, expected, rtol=0, atol=0)
        self.assertTrue(((foreground > 0) & (foreground < 1)).any())
        for k, v in source.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Map bank needs"):
            experiment.extract_content_maps(
                source,
                DataLoader(rows, batch_size=2),
                "cpu",
                max_bytes=maps.numel() * maps.element_size(),
                return_foreground=True,
            )
        with self.assertRaisesRegex(ValueError, "requires input foreground masks"):
            experiment.extract_content_maps(
                source, DataLoader([dict(image=r["image"]) for r in rows]), "cpu", return_foreground=True
            )

    def test_training_and_validation_share_loss_and_are_batch_independent(self):
        maps = torch.randn(5, 2, 2, 3, 4, 5)
        head = SpatialBottleneck((2, 3, 4, 5), 2, 2, 8)
        head.fit_normalization(maps)
        foreground = torch.ones(5, 2, 1, 3, 4, 5)
        foreground[..., :2] = 0
        foreground[..., 2] = 0.5
        options = dict(hard_patch_weight=0.7, hard_patch_fraction=0.4, hard_patch_size=(2, 3, 2))
        weights = torch.rand(2, 3, 4, 5) + 0.1
        _, expected = paired_objective(head, maps, recon_weights=weights, foreground=foreground, **options)
        a, codes_a, _ = experiment.evaluate(head, maps, 2, "cpu", lambda _: weights, foreground, **options)
        b, codes_b, _ = experiment.evaluate(head, maps, 5, "cpu", lambda _: weights, foreground, **options)
        for name in (
            "reconstruction_mean",
            "reconstruction_hard",
            "reconstruction_objective",
            "foreground_reconstruction",
        ):
            self.assertAlmostEqual(a[name], b[name], places=5)
        self.assertAlmostEqual(a["reconstruction_objective"], expected["reconstruction"].item(), places=5)
        torch.testing.assert_close(codes_a, codes_b)
        args = experiment.parse_args(["--run-dir", "/tmp/s", "--out-dir", "/tmp/o"])
        score = a["reconstruction_objective"] + a["alignment"] + 0.1 * a["variance"]
        self.assertAlmostEqual(experiment.objective_score(a, args), score)
        with self.assertRaisesRegex(ValueError, "must match"):
            experiment.evaluate(head, maps, 2, "cpu", foreground=foreground[:2])

    def test_masked_hard_patch_run_remains_compatible_with_factor_evaluation(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            run = write_source_run(root)
            out = root / "local_loss"
            before = experiment.file_digest(run / "vqvae_model.pt")
            report = experiment.main(
                trainer_argv(run, out)
                + [
                    "--hard-patch-weight",
                    "0.7",
                    "--hard-patch-fraction",
                    "0.25",
                    "--hard-patch-size",
                    "3",
                    "3",
                    "3",
                    "--recon-foreground-only",
                    "--recon-weighting",
                    "deviation",
                ]
            )
            self.assertEqual(experiment.file_digest(run / "vqvae_model.pt"), before)
            self.assertTrue(report["training"]["reconstruction_objective"]["foreground_only"])
            for split in ("validation", "test"):
                metrics = report[split]
                self.assertAlmostEqual(
                    metrics["reconstruction_objective"],
                    metrics["reconstruction_mean"] + 0.7 * metrics["reconstruction_hard"],
                    places=5,
                )
                self.assertTrue(0 < metrics["foreground_occupancy"] < 1)
                self.assertTrue(math.isfinite(metrics["foreground_reconstruction_over_template"]))
            outputs = torch.load(out / "test_outputs.pt", weights_only=True)
            self.assertEqual(outputs["foreground_occupancy"].shape, (4, 2, 1, 8, 8, 8))
            payload = factor_eval.main(
                [
                    "--bottleneck-dirs",
                    str(out),
                    "--out-dir",
                    str(root / "eval"),
                    "--device",
                    "cpu",
                    "--probes",
                    "linear",
                    "--bootstrap",
                    "100",
                    "--patch-grid",
                    "none",
                ]
            )
            self.assertTrue(all(v < 1e-5 for v in payload["bottlenecks"][0]["parity"].values()))

    def test_invalid_local_loss_arguments_fail_early(self):
        for flags in (
            ["--hard-patch-weight", "-1"],
            ["--hard-patch-fraction", "0"],
            ["--hard-patch-fraction", "nan"],
            ["--hard-patch-size", "1", "0", "2"],
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment.parse_args(["--run-dir", "/tmp/s", "--out-dir", "/tmp/o", *flags])


if __name__ == "__main__":
    unittest.main()
