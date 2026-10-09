"""Demographic decoding, subject isolation, coverage and probe statistics."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from data.adni_demographics import decode_demographics
from eval.adni import encoder_probes


class DemographicTests(unittest.TestCase):
    def test_optional_columns_and_adni_codes(self):
        self.assertEqual(decode_demographics({"Group": "CN"}), {})
        self.assertEqual(decode_demographics({"PTGENDER": 2.0}), {"gender": 2})
        for value, expected in (
            ("1|5", 6),
            ("5|1", 6),
            ("5|5", 5),
            (6, 6),
            (8, 8),
            (9, 9),
        ):
            self.assertEqual(decode_demographics({"PTRACCAT": value}), {"race": expected})
        for missing in (None, "", np.nan, -4, "invalid", "1.5"):
            self.assertEqual(
                decode_demographics({"PTGENDER": missing, "PTRACCAT": missing}),
                {"gender": -1, "race": -1},
            )
        for unknown in ("7", "1|7", "99"):
            self.assertEqual(decode_demographics({"PTRACCAT": unknown}), {"race": -1})


class ProbeTests(unittest.TestCase):
    def representations(self, labels):
        labels = np.asarray(labels)
        features = np.column_stack([np.ones(len(labels)), labels == 0, labels == 1, labels == 2]).astype(float)
        return {
            "content_v0": features,
            "content_v1": features[:, ::-1].copy(),
            "labels": labels,
            "subjects": np.array([f"subject-{i}" for i in range(len(labels))]),
        }

    def test_strong_signal_and_imbalanced_baselines(self):
        reps = self.representations([0] * 60 + [1] * 12 + [2] * 6)
        reps["demographics"] = {
            "gender": reps["labels"].copy(),
            "race": reps["labels"].copy(),
        }
        reps["demographics"]["race"][:3] = -1
        scores = encoder_probes.evaluate_subject_probes(reps)
        for target in ("diagnosis", "gender", "race"):
            for view in (0, 1):
                self.assertEqual(scores[f"content/{target}_probe_acc_v{view}"], 1.0)
                self.assertEqual(scores[f"content/{target}_probe_balanced_acc_v{view}"], 1.0)
        self.assertAlmostEqual(scores["content/diagnosis_probe_chance"], 60 / 78)
        self.assertAlmostEqual(scores["content/diagnosis_probe_balanced_chance"], 1 / 3)
        self.assertEqual(scores["content/race_probe_n"], 75)
        self.assertEqual(scores["content/race_probe_missing"], 3)

    def test_missing_singleton_and_conflicting_subjects_are_reported(self):
        labels = np.array([0, 0, 1, 1, 2, -1, 0, 1])
        subjects = np.array(["a", "b", "c", "d", "singleton", "missing", "conflict", "conflict"])
        keep, folds, stats = encoder_probes._subject_folds(labels, subjects)
        np.testing.assert_array_equal(np.flatnonzero(keep), [0, 1, 2, 3])
        self.assertEqual((stats["missing"], stats["rare"], stats["conflicting"]), (1, 1, 2))
        self.assertEqual((stats["classes"], stats["folds"], stats["class_2_subjects"]), (2, 2, 1))
        for train, test in folds:
            self.assertFalse(set(subjects[keep][train]) & set(subjects[keep][test]))
            self.assertEqual(set(labels[keep][train]), {0, 1})

    def test_repeat_scans_and_pooled_views_cannot_cross_subject_folds(self):
        reps = self.representations(np.repeat([0, 1] * 6, 2))
        reps["subjects"] = np.repeat(np.arange(12), 2)
        # Embed subject ID in the feature array so every score call can audit its folds.
        reps["content_v0"][:, 0] = reps["subjects"]
        reps["content_v1"][:, 0] = reps["subjects"]

        def audit(features, labels, folds):
            for train, test in folds:
                self.assertFalse(set(features[train, 0]) & set(features[test, 0]))
            self.assertEqual(
                sorted(np.concatenate([test for _, test in folds]).tolist()),
                list(range(len(labels))),
            )
            return 0.5, 0.5

        with patch.object(encoder_probes, "_score", side_effect=audit) as score:
            encoder_probes.evaluate_subject_probes(reps)
        self.assertEqual(score.call_count, 3)

    def test_scaler_fits_training_rows_only(self):
        features = np.array([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0], [200.0, 0.0]])
        labels = np.array([0, 1, 0, 1])
        fitted = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                fitted.append(X.copy())
                return super().fit(X, y, sample_weight=sample_weight)

        folds = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ]
        with patch.object(encoder_probes, "StandardScaler", RecordingScaler):
            encoder_probes._score(features, labels, folds)
        self.assertEqual([len(values) for values in fitted], [2, 2])
        np.testing.assert_allclose(fitted[0], np.eye(2))

    def test_unavailable_target_does_not_invent_chance_accuracy(self):
        for labels in ([-1] * 6, [1] * 6, [0, 1, 2]):
            scores = encoder_probes.evaluate_subject_probes(self.representations(labels))
            self.assertEqual(scores["content/diagnosis_probe_folds"], 0)
            self.assertTrue(np.isnan(scores["content/diagnosis_probe_acc_v0"]))


if __name__ == "__main__":
    unittest.main()
