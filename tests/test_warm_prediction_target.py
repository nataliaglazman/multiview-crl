"""warm.py --predict-target: the momentum copy's backbone map summarized as its GAP, its octant averages or its
per-channel quantiles over the interior cells."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import warm


class FixedMaps:
    """Stands in for the momentum copy: its encoder returns the given (2B, C, 16, 16, 16) maps."""

    def __init__(self, maps):
        self.maps = maps

    def _encode(self, images, views, view_idx):
        return self.maps


def blob_maps(centre, b=2, channels=3):
    """Two views of ``b`` subjects; channel 0 holds a bright 2-cell cube at ``centre``, the rest is weak noise."""
    torch.manual_seed(0)
    maps = 0.01 * torch.randn(2 * b, channels, 16, 16, 16)
    x, y, z = centre
    maps[:, 0, x : x + 2, y : y + 2, z : z + 2] += 5.0
    return maps


KEEP = torch.zeros(16**3, dtype=torch.bool)
KEEP.view(16, 16, 16)[2:14, 2:14, 2:14] = True  # interior cells


class PredictionTargetTests(unittest.TestCase):
    def test_gap_is_the_existing_target(self):
        maps = blob_maps((4, 4, 4))
        target = warm.prediction_target(FixedMaps(maps), None, "gap", KEEP)
        torch.testing.assert_close(target, warm.target_gap(FixedMaps(maps), None))
        self.assertEqual(tuple(target.shape), (2, 2, 3))

    def test_grid_is_the_octant_averages(self):
        maps = blob_maps((4, 4, 4))
        target = warm.prediction_target(FixedMaps(maps), None, "grid", KEEP)
        self.assertEqual(tuple(target.shape), (2, 2, 3 * 8))
        first_octant = maps[0, 0, :8, :8, :8].mean()
        torch.testing.assert_close(target[0, 0, 0], first_octant)

    def test_quantiles_ignore_where_the_lesion_sits_and_the_grid_does_not(self):
        here, there = blob_maps((4, 4, 4)), blob_maps((10, 9, 4))
        for kind, same in (("quantile", True), ("grid", False)):
            a = warm.prediction_target(FixedMaps(here), None, kind, KEEP)
            b = warm.prediction_target(FixedMaps(there), None, kind, KEEP)
            self.assertEqual(bool(torch.allclose(a[:, :, :8], b[:, :, :8], atol=0.05)), same, kind)
        quantile = warm.prediction_target(FixedMaps(here), None, "quantile", KEEP)
        self.assertEqual(tuple(quantile.shape), (2, 2, 3 * len(warm.PREDICT_QUANTILES)))


if __name__ == "__main__":
    unittest.main()
