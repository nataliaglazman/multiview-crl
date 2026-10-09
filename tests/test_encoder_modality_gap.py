"""Planted modality differences, fold isolation, and a real ADNI checkpoint diagnostic."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.adni import encoder_modality_gap as probe


def sphere(n=180, d=8):
    return probe.normalize_rows(np.random.default_rng(12).normal(size=(n, d)))


class ModalityGapTests(unittest.TestCase):
    def test_tiny_offset_is_detected_then_removed(self):
        shared = sphere()
        a = np.column_stack((shared, np.full(len(shared), 0.01)))
        b = np.column_stack((shared, np.full(len(shared), -0.01)))
        report, _ = probe.diagnose(a, b, np.arange(len(a)))
        self.assertGreater(report["geometry_normalized"]["paired_cosine_mean"], 0.999)
        self.assertGreater(report["probes"]["normalized_linear"]["accuracy"], 0.98)
        self.assertGreater(report["probes"]["normalized_rbf"]["accuracy"], 0.98)
        for kind in probe.PROBES:
            self.assertAlmostEqual(report["probes"][f"mean_centered_{kind}"]["accuracy"], 0.5)
        self.assertAlmostEqual(report["paired_cosine_after_fold_centering"], 1.0)

    def test_nonlinear_probe_detects_difference_with_equal_means(self):
        # Unit-circle axes versus diagonals: equal population means but different shapes.
        # No mean-only or linear test can diagnose this difference reliably.
        a = np.tile([[1, 0], [-1, 0], [0, 1], [0, -1]], (60, 1))
        b = np.tile([[1, 1], [-1, -1], [1, -1], [-1, 1]], (60, 1)) / np.sqrt(2)
        report, _ = probe.diagnose(a, b, np.arange(len(a)))
        self.assertAlmostEqual(report["geometry_normalized"]["centroid_distance"], 0)
        self.assertGreater(report["probes"]["mean_centered_rbf"]["accuracy"], 0.95)
        self.assertLess(report["probes"]["mean_centered_linear"]["accuracy"], 0.6)

    def test_identical_and_collapsed_features_do_not_claim_leakage(self):
        for a in (sphere(), np.zeros((12, 3))):
            report, _ = probe.diagnose(a, a, np.arange(len(a)))
            for value in report["probes"].values():
                self.assertEqual(value["accuracy"], 0.5)
            json.dumps(report, allow_nan=False)
        self.assertIsNone(report["geometry_normalized"]["paired_cosine_mean"])
        self.assertEqual(report["geometry_normalized"]["zero_vectors"], [12, 12])

    def test_repeated_subjects_share_folds_and_preprocessing_uses_training_only(self):
        subjects = np.repeat(np.arange(24), 2)
        a, b = sphere(len(subjects), 3), sphere(len(subjects), 3) * 0.4
        for train, test in probe.subject_folds(subjects, 3, 42):
            self.assertFalse(set(subjects[train]) & set(subjects[test]))
            x_train, x_test, means = probe.fold_features(a, b, train, test, True)
            changed_a, changed_b = a.copy(), b.copy()
            changed_a[test] += 100
            changed_b[test] -= 200
            other_train, other_test, other_means = probe.fold_features(changed_a, changed_b, train, test, True)
            np.testing.assert_array_equal(other_train, x_train)
            np.testing.assert_array_equal(other_means, means)
            np.testing.assert_allclose(other_test[: len(test)] - x_test[: len(test)], 100)
        _, rows = probe.diagnose(a, b, subjects)
        for subject in np.unique(subjects):
            self.assertEqual(len({row["fold"] for row in rows if row["subject"] == str(subject)}), 1)

    def test_invalid_features_fail_instead_of_reporting_chance(self):
        a = sphere(6, 3)
        for b, subjects, folds in ((a[:, :2], range(6), 3), (a, range(5), 3), (a, range(6), 7)):
            with self.assertRaises(ValueError):
                probe.diagnose(a, b, subjects, folds)
        a[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            probe.diagnose(a, a, range(6))

    def test_feature_cli_writes_reproducible_reports_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "input.npz"
            a = sphere(30)
            np.savez_compressed(
                features,
                content_t1=a,
                content_flair=a,
                subjects=np.arange(len(a)).astype(str),
                metadata_json=np.array(json.dumps({"checkpoint_sha256": "example"})),
            )
            argv = ["--features", str(features), "--out-dir", str(root / "report")]
            with contextlib.redirect_stdout(io.StringIO()):
                first = probe.main(argv)
                second = probe.main(["--features", str(root / "report/features.npz"), "--out-dir", str(root / "again")])
            self.assertEqual(first["probes"], second["probes"])
            saved = json.loads((root / "report/report.json").read_text())
            self.assertEqual(saved["source"]["checkpoint_sha256"], "example")
            with (root / "report/predictions.csv").open() as fp:
                self.assertEqual(len(list(csv.DictReader(fp))), 60)
            with self.assertRaises(FileExistsError):
                probe.main(argv)


class CheckpointDiagnosticTests(unittest.TestCase):
    def test_saved_checkpoint_extracts_only_original_validation_subjects(self):
        import torch
        from test_encoder_adni import SHAPE, TINY, fake_adni

        from eval.protocol.score_checkpoint import build_model
        from training import main_conv_synthetic as trainer

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = fake_adni(root)
            args = trainer.parse_args(
                [
                    "--dataset-name",
                    "ADNI_stripped_masks",
                    "--dataroot",
                    str(root),
                    "--labels-path",
                    str(labels),
                    "--spatial-size",
                    *map(str, SHAPE),
                    "--image-spacing",
                    "1",
                    "--val-frac",
                    "0.4",
                    "--test-frac",
                    "0.1",
                    "--no-cache",
                    *TINY,
                    "--conv-readout",
                    "mlp",
                    "--norm-type",
                    "layer",
                ]
            )
            cfg = vars(args)
            run = root / "run"
            run.mkdir()
            (run / "settings.json").write_text(json.dumps(cfg))
            train = trainer.make_real_dataset(args, "train")
            val = trainer.make_real_dataset(args, "val")
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.describe_real_split(args, train, val, run)
            torch.save(build_model(cfg, "cpu").state_dict(), run / "model_best.pt")
            before = hashlib.sha256((run / "model_best.pt").read_bytes()).hexdigest()
            argv = ["--run-dir", str(run), "--device", "cpu", "--batch-size", "5", "--out-dir", str(root / "result")]
            with contextlib.redirect_stdout(io.StringIO()):
                report = probe.main(argv)
            self.assertEqual(report["pairs"], 12)
            self.assertEqual(report["source"]["checkpoint_sha256"], before)
            self.assertEqual(hashlib.sha256((run / "model_best.pt").read_bytes()).hexdigest(), before)
            split = json.loads((run / "split.json").read_text())
            with np.load(root / "result/features.npz", allow_pickle=False) as features:
                self.assertEqual(features["content_t1"].shape, (12, 9))
                self.assertEqual(features["subjects"].tolist(), split["subjects"]["val"])
                self.assertFalse(set(features["subjects"]) & set(split["subjects"]["test"]))
            split["subjects"]["val"] = split["subjects"]["val"][::-1]
            (run / "split.json").write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "validation subjects differ"):
                probe.extract_checkpoint(probe.parse_args(argv))


if __name__ == "__main__":
    unittest.main()
