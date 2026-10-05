"""Real audit replay, matched contrast, statistical controls and exact NIfTI images."""

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from eval.encoder import encoder_lesion_contrast as contrast
from eval.encoder import encoder_lesion_intervention as intervention
from eval.encoder.encoder_target_protocol import dataset, digest
from eval.protocol.score_checkpoint import build_model
from tests.test_encoder_target_followups import config


class LesionContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_determinism = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cfg = config()
        run = cls.root / "run"
        run.mkdir()
        (run / "settings.json").write_text(json.dumps(cls.cfg))
        model = build_model(cls.cfg, "cpu")
        for name in ("model.pt", "model_init.pt"):
            torch.save(model.state_dict(), run / name)
        cls.source = cls.root / "movement"
        with contextlib.redirect_stdout(io.StringIO()):
            intervention.main(
                [
                    "--run-dir",
                    str(run),
                    "--out-dir",
                    str(cls.source),
                    "--num-samples",
                    "12",
                    "--subject-offset",
                    "0",
                    "--grids",
                    "1",
                    "2",
                    "--bootstrap",
                    "0",
                    "--device",
                    "cpu",
                ]
            )
        # New tool must work even when no model/settings file remains accessible.
        shutil.rmtree(run)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_determinism[0], warn_only=cls.old_determinism[1])

    def test_complete_replay_needs_no_model_and_exports_exact_nifti_images(self):
        out = self.root / "contrast"
        before = {p.name: digest(p) for p in self.source.iterdir()}
        args = ["--audit-dir", str(self.source), "--out-dir", str(out), "--bootstrap", "12", "--nifti-per-group", "1"]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            intervention, "fit_readouts", side_effect=AssertionError("Must not refit probes")
        ), patch.object(torch.Tensor, "backward", side_effect=AssertionError("No training")):
            contrast.main(args)
        self.assertEqual(before, {p.name: digest(p) for p in self.source.iterdir()})
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["source_audit_unchanged"])
        self.assertTrue(report["image_replay_verified"])
        self.assertFalse(report["probes_refit"])
        self.assertEqual((report["n_subjects"], report["n_pairs"]), (12, 36))
        self.assertEqual(report["n_nifti_examples"], 8)  # 2 subjects * 2 endpoints * 2 modalities
        with (out / "subject_errors.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for arm in ("trained", "initial"):
            subset = [
                r
                for r in rows
                if r["arm"] == arm
                and r["view"] == "t1"
                and r["grid"] == "2"
                and r["stage"] == "backbone"
                and r["probe"] == "ridge"
                and r["condition"] == "observed"
            ]
            self.assertEqual(len(subset), 12)  # not 36 independent axes / 72 independent endpoints
            self.assertEqual(sum(int(r["n_pairs"]) for r in subset), 36)
        for row in report["associations"]:
            self.assertLessEqual(row["n_subjects"], 12)
        for row in report["contrast_groups"]:
            if row["view"] != "t1" or row["grouping"] != "t1_matched":
                continue
            counterpart = next(
                r
                for r in report["contrast_groups"]
                if all(
                    r[k] == row[k] for k in ("arm", "grid", "stage", "probe", "condition", "grouping", "contrast_group")
                )
                and r["view"] == "flair"
            )
            self.assertEqual(row["n_subjects"], counterpart["n_subjects"])
        ds = dataset(self.cfg, 64, "test")
        with (out / "nifti" / "examples.csv").open() as stream:
            examples = list(csv.DictReader(stream))
        self.assertEqual({r["group"] for r in examples}, {"low", "high"})
        for row in examples:
            image = nib.load(out / "nifti" / row["image"])
            sample = contrast.render_pair(ds, int(row["subject_id"]), row["intervention_axis"], 0.5)
            original = sample[row["endpoint"]][contrast.VIEWS.index(row["view"])].numpy()[0]
            np.testing.assert_array_equal(image.get_fdata(), original)
            np.testing.assert_array_equal(image.affine, np.eye(4))
            self.assertEqual(image.header.get_xyzt_units()[0], "unknown")
            self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
            mask = nib.load(out / "nifti" / row["lesion_mask"])
            np.testing.assert_array_equal(mask.get_fdata(), sample["lesions"]["ab".index(row["endpoint"])])
            self.assertEqual(mask.get_data_dtype(), np.dtype("uint8"))
        self.assertFalse(list(out.rglob("*.pt")))
        self.assertFalse(list(out.rglob("*.npy")))
        with self.assertRaises(FileExistsError):
            contrast.main(args)

    def test_renderer_low_gain_t1_is_less_visible_than_high_gain_and_flair(self):
        ds = dataset(config(res=32, synthetic_lesion_radius=0.14), 64, "test")
        renderer = ds._inner.renderer
        _, _, lat = ds._inner[0]
        tissue, lesion = renderer.render_structure(
            lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=True
        )
        scores = {}
        for style, name in ((torch.tensor([-1.0, -1.0, 0.0]), "low"), (torch.tensor([1.0, 1.0, 0.0]), "high")):
            for view in ("T1", "FLAIR"):
                image = renderer.render_modality(tissue, lesion, style, view, 42, "cpu").numpy()[0]
                reference = renderer.render_modality(tissue, torch.zeros_like(lesion), style, view, 42, "cpu").numpy()[
                    0
                ]
                metrics = contrast.contrast_metrics(image, reference, lesion.numpy() > 0, tissue.numpy())
                scores[name, view] = metrics["matched_contrast"]
                (
                    self.assertLess(metrics["signed_matched_contrast"], 0)
                    if view == "T1"
                    else self.assertGreater(metrics["signed_matched_contrast"], 0)
                )
        self.assertGreater(scores["high", "T1"], 5 * scores["low", "T1"])
        self.assertGreater(scores["low", "FLAIR"], 5 * scores["low", "T1"])

    def test_references_preserve_per_sample_and_shared_normalization(self):
        for normalization in ("per_sample", "shared"):
            ds = dataset(config(synthetic_normalize=normalization), 64, "test")
            reference = contrast.render_reference(ds, 0)
            pair = contrast.render_pair(ds, 0, "x", 0.5)
            for end, lesion in zip("ab", pair["lesions"]):
                for v in range(2):
                    metrics = contrast.contrast_metrics(
                        pair[end][v].numpy()[0], reference["images"][v].numpy()[0], lesion, reference["tissue"]
                    )
                    self.assertGreater(metrics["matched_contrast"], 0)
                    self.assertGreater(reference["styles"][v]["normalized_preblur_noise_sigma"], 0)

    def test_subject_association_detects_signal_and_adjusts_a_planted_confound(self):
        rng = np.random.default_rng(23)
        n = 1000
        nuisance = rng.normal(size=n)
        x = nuisance + 0.6 * rng.normal(size=n)
        y = -nuisance + 0.6 * rng.normal(size=n)
        controls = np.column_stack((nuisance, np.ones(n)))
        result = contrast.association(x, y, controls, 0, 42)
        self.assertLess(result["spearman"], -0.6)
        self.assertLess(abs(result["partial_spearman"]), 0.1)
        true_signal = contrast.association(x, -x + 0.1 * rng.normal(size=n), controls, 20, 42)
        self.assertLess(true_signal["partial_spearman"], -0.9)
        self.assertLess(true_signal["partial_spearman_ci95_high"], -0.9)
        self.assertTrue(np.isnan(contrast.rank_correlation(np.ones(10), np.arange(10))))
        self.assertTrue(np.isnan(contrast.rank_correlation(np.arange(7), np.arange(7))))
        groups, _ = contrast.contrast_groups(np.ones(12))
        np.testing.assert_array_equal(groups, "middle")

    def test_mismatched_order_missing_predictions_and_changed_images_are_rejected(self):
        for fault in ("order", "missing", "hash"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=self.root) as tmp:
                source = Path(tmp) / "source"
                shutil.copytree(self.source, source)
                if fault in ("order", "missing"):
                    path = source / "trained_predictions.npz"
                    with np.load(path) as bank:
                        values = {k: bank[k] for k in bank.files}
                    if fault == "order":
                        values["intervention_axis"] = values["intervention_axis"][::-1]
                    else:
                        del values["t1_g2_backbone_ridge_observed"]
                    np.savez_compressed(path, **values)
                    with self.assertRaisesRegex(ValueError, "order|keys"):
                        contrast.load_audit(source)
                else:
                    report = json.loads((source / "report.json").read_text())
                    for arm in report["cohorts"]:
                        report["cohorts"][arm]["moves"]["input_sha256"] = "0" * 64
                    (source / "report.json").write_text(json.dumps(report))
                    out = Path(tmp) / "failed"
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "SHA256"):
                        contrast.main(["--audit-dir", str(source), "--out-dir", str(out), "--nifti-per-group", "0"])
                    self.assertEqual(json.loads((out / "report.json").read_text())["status"], "failed")
                    self.assertFalse((out / "nifti").exists())


if __name__ == "__main__":
    unittest.main()
