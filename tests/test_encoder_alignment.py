"""Planted alignment, collapse, repeated subjects and parity with offline modality scoring."""

import contextlib
import io
import unittest

import numpy as np

from eval.adni.encoder_alignment import alignment_metrics, print_alignment_report
from eval.adni.encoder_modality_gap import diagnose


def reps(a, b, subjects=None):
    return {
        "content_v0": a,
        "content_v1": b,
        "subjects": np.arange(len(a)).astype(str) if subjects is None else np.asarray(subjects),
    }


class AlignmentTests(unittest.TestCase):
    def test_correct_pairing_beats_permuted_pairs_with_identical_modality_distributions(self):
        a = np.eye(12)
        matched = alignment_metrics(reps(a, a))
        mismatched = alignment_metrics(reps(a, np.roll(a, 6, axis=0)))
        self.assertEqual(matched["val/content_retrieval_top1"], 1.0)
        self.assertEqual(mismatched["val/content_retrieval_top1"], 0.0)
        self.assertEqual(matched["content/alignment_gap"], 1.0)
        self.assertLess(mismatched["content/alignment_gap"], 0)
        self.assertEqual(matched["content/centroid_distance"], 0.0)
        self.assertEqual(mismatched["content/centroid_distance"], 0.0)
        self.assertAlmostEqual(matched["content/normalized_eff_rank_v0"], 11)

    def test_collapsed_features_get_chance_retrieval_even_at_top_five(self):
        for a in (np.ones((12, 4)), np.zeros((12, 4))):
            scores = alignment_metrics(reps(a, a))
            self.assertAlmostEqual(scores["val/content_retrieval_top1"], 1 / 12)
            self.assertAlmostEqual(scores["val/content_retrieval_top5"], 5 / 12)
            self.assertAlmostEqual(scores["val/content_retrieval_mean_rank"], 6.5)
            self.assertEqual(scores["content/modality_normalized_linear_acc"], 0.5)
            self.assertEqual(scores["content/normalized_eff_rank_v0"], 0)
            self.assertEqual(scores["content/normalized_spread_v1"], 0)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                print_alignment_report(scores)
            self.assertIn("chance modality accuracy can reflect collapse", output.getvalue())

    def test_high_cosine_alone_does_not_imply_subject_alignment(self):
        a = np.column_stack([np.full(12, 100.0), np.eye(12)])
        scores = alignment_metrics(reps(a, np.roll(a, 6, axis=0)))
        self.assertGreater(scores["content/cross_view_cosine_mean"], 0.999)
        self.assertLess(scores["content/alignment_gap"], 0)
        self.assertEqual(scores["val/content_retrieval_top1"], 0)

    def test_repeated_visits_are_positives_and_not_unpaired_controls(self):
        a = np.repeat(np.eye(2), 2, axis=0)
        scores = alignment_metrics(reps(a, a, ["a", "a", "b", "b"]))
        self.assertEqual(scores["val/content_retrieval_top1"], 1.0)
        self.assertEqual(scores["val/content_retrieval_chance"], 0.5)
        self.assertEqual(scores["content/alignment_gap"], 1.0)
        self.assertEqual(scores["alignment/probe_folds"], 2)

    def test_modality_offset_and_all_probe_scores_match_offline_diagnostic(self):
        shared = np.random.default_rng(1).normal(size=(60, 8))
        shared /= np.linalg.norm(shared, axis=1, keepdims=True)
        a = np.column_stack([shared, np.full(60, 0.02)])
        b = np.column_stack([shared, np.full(60, -0.02)])
        data = reps(a, b)
        scores = alignment_metrics(data)
        report, _ = diagnose(a, b, data["subjects"])
        for key, value in report["probes"].items():
            self.assertEqual(scores[f"content/modality_{key}_acc"], value["accuracy"])
        self.assertEqual(scores["content/modality_probe_acc"], scores["content/modality_normalized_linear_acc"])
        self.assertGreater(scores["content/modality_probe_acc"], 0.98)
        self.assertEqual(scores["content/modality_mean_centered_rbf_acc"], 0.5)
        self.assertEqual(scores["val/content_retrieval_top1"], 1)
        self.assertGreater(scores["content/centroid_distance_over_within_rms"], 0)
        np.testing.assert_array_equal(a[:, -1], np.full(60, 0.02))

    def test_invalid_features_raise_instead_of_becoming_chance_accuracy(self):
        a = np.eye(12)
        a[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            alignment_metrics(reps(a, a))

    def test_prints_initial_previous_reference_and_actual_content_width(self):
        scores = alignment_metrics(reps(np.eye(12), np.eye(12)))
        floor, previous = dict(scores), dict(scores)
        floor["val/content_retrieval_top1"] = 0.2
        previous["val/content_retrieval_top1"] = 0.7
        with contextlib.redirect_stdout(io.StringIO()) as output:
            print_alignment_report(scores, floor=floor, previous=previous)
        text = output.getvalue()
        self.assertIn("12 content channels", text)
        self.assertIn("Δ previous", text)
        self.assertIn("+0.800", text)
        self.assertIn("+0.300", text)
        self.assertIn("its score does not measure learned invariance", text)


if __name__ == "__main__":
    unittest.main()
