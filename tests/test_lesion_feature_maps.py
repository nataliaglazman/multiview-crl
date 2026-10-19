"""Footprint numbers behind the lesion feature-map figures."""

import unittest

import numpy as np

from eval.lesion.lesion_feature_maps import footprint_summary


class FootprintSummaryTests(unittest.TestCase):
    def test_localised_coherent_and_cancelling_footprints(self):
        centre = np.array([4.0, 5.0, 3.0])
        blob = np.zeros((3, 8, 8, 8))
        blob[0, 4, 5, 3] = 2.0
        blob[1, 4, 5, 4] = 1.0
        s = footprint_summary(blob, content=[0, 1], centre=centre)
        np.testing.assert_array_equal(s["peak"], centre)
        self.assertEqual(s["peak_distance_cells"], 0.0)
        self.assertEqual(s["near_fraction"], 1.0)
        self.assertEqual(s["style_to_content"], 0.0)
        self.assertEqual(s["sign_coherence"], 1.0)

        dipole = np.zeros((3, 8, 8, 8))
        dipole[0, 4, 5, 3], dipole[0, 4, 6, 3] = 1.0, -1.0
        dipole[2, 0, 0, 0] = 3.0
        s = footprint_summary(dipole, content=[0, 1], centre=centre)
        self.assertEqual(s["sign_coherence"], 0.0)
        self.assertAlmostEqual(s["style_to_content"], 3 / np.sqrt(2))

        far = np.zeros((3, 8, 8, 8))
        far[1, 0, 0, 0] = 1.0
        s = footprint_summary(far, content=[0, 1], centre=centre)
        self.assertEqual(s["near_fraction"], 0.0)
        self.assertAlmostEqual(s["peak_distance_cells"], np.linalg.norm(centre))


if __name__ == "__main__":
    unittest.main()
