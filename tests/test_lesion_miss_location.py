"""Per-group lesion-centre decoding from a spatial_bottleneck_eval predictions file."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.lesion import lesion_miss_location as miss


def write_predictions(path, truth, predictions):
    """``truth`` (n, 3) in [-1, 1]; ``predictions`` maps (representation, probe) to (n, 2 views, 3)."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "view", "representation", "probe", "factor", "truth", "prediction"])
        for (name, kind), values in predictions.items():
            for subject in range(len(truth)):
                for view in range(2):
                    for axis, factor in enumerate(miss.CENTRES):
                        writer.writerow(
                            [subject, view + 1, name, kind, factor, truth[subject, axis], values[subject, view, axis]]
                        )
                    writer.writerow([subject, view + 1, name, kind, "brain_size", 0.0, 0.0])


class DecodingByGroupTests(unittest.TestCase):
    def test_group_scores_use_all_subject_variance_and_check_subjects(self):
        rng = np.random.default_rng(0)
        n, res = 40, 16
        voxels = rng.integers(2, 14, size=(n, 3)).astype(float)
        truth = voxels * 2 / (res - 1) - 1
        perfect = np.repeat(truth[:, None], 2, axis=1)
        mean = np.broadcast_to(truth.mean(0), (n, 2, 3)).copy()
        mixed = perfect.copy()
        mixed[20:30] = mean[20:30]
        # The audit covers the first 30 subjects: 20 found, 10 lost; the file holds all 40.
        groups = {s: "hit" if s < 20 else "new miss" for s in range(30)}
        audit = {s: tuple(voxels[s]) for s in range(30)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test_predictions.csv"
            predictions = {("perfect", "linear"): perfect, ("mean", "linear"): mean, ("mixed", "rbf"): mixed}
            write_predictions(path, truth, predictions)
            scores, count = miss.decoding_by_group(path, groups, audit, res)
            self.assertEqual(count, n)
            self.assertEqual(set(scores), set(predictions))
            for factor in miss.CENTRES:
                variance = truth[:, miss.CENTRES.index(factor)].var()
                error = np.square(truth[:, miss.CENTRES.index(factor)] - truth[:, miss.CENTRES.index(factor)].mean())
                self.assertEqual(scores[("perfect", "linear")][factor], {"all": 1.0, "hit": 1.0, "new miss": 1.0})
                self.assertAlmostEqual(scores[("mean", "linear")][factor]["all"], 0.0, places=12)
                mixed_score = scores[("mixed", "rbf")][factor]
                self.assertEqual(mixed_score["hit"], 1.0)
                self.assertAlmostEqual(mixed_score["new miss"], 1 - error[20:30].mean() / variance, places=12)
                self.assertAlmostEqual(mixed_score["all"], 1 - error[20:30].sum() / (n * variance), places=12)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                miss.print_decoding(scores, count)
            self.assertIn("mixed", printed.getvalue())

            shifted = dict(audit)
            shifted[3] = (voxels[3, 0] + 1, voxels[3, 1], voxels[3, 2])
            with self.assertRaisesRegex(ValueError, "same subjects"):
                miss.decoding_by_group(path, groups, shifted, res)
            with self.assertRaisesRegex(ValueError, "not in"):
                miss.decoding_by_group(path, {**groups, 99: "hit"}, {**audit, 99: (1.0, 1.0, 1.0)}, res)


if __name__ == "__main__":
    unittest.main()
