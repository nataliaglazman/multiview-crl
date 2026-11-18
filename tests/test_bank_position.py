"""bank.py --lesion-position fixed: the lesion goes to the voxel nearest FIXED_CENTRE whose full-size sphere fits in the
white matter, whatever placement was drawn."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import bank
import training.main_conv_synthetic as T


def renderer():
    return T.make_dataset(bank.run_args(0.4, causal=True), "test", 4)._inner.renderer


def centre_of(load):
    cells = torch.nonzero(load > 0).double()
    weights = load[load > 0].double()
    return ((cells * weights[:, None]).sum(0) / weights.sum()).numpy()


class FixedPositionTests(unittest.TestCase):
    def setUp(self):
        self.r = renderer()
        bank.set_position(self.r, "fixed")
        self.centre = np.array(bank.FIXED_CENTRE)

    def white(self, lower, upper):
        tissue = torch.zeros((self.r.res,) * 3, dtype=torch.long)
        tissue[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = 2
        return tissue

    def nearest_that_fits(self, tissue):
        """Brute force over the whole volume, independent of bank.fixed_quantiles."""
        spacing = 2.0 / (self.r.res - 1)
        reach = self.r.lesion_radius + spacing / 2
        white = tissue.numpy() == 2
        fits = np.argwhere(distance_transform_edt(np.pad(white, 1), sampling=spacing)[1:-1, 1:-1, 1:-1] > reach + 1e-7)
        return fits[np.argmin(np.linalg.norm(fits - self.centre, axis=1))]

    def test_at_the_fixed_centre_when_it_fits(self):
        tissue = self.white(self.centre - 10, self.centre + 11)
        load = self.r._burden_in_white_matter(tissue, 0.0, torch.rand(1, 3))
        np.testing.assert_allclose(centre_of(load), self.centre, atol=1e-9)

    def test_otherwise_at_the_nearest_voxel_that_fits(self):
        tissue = self.white(self.centre + np.array([6, -3, -10]), self.centre + np.array([26, 18, 11]))
        expected = self.nearest_that_fits(tissue)
        self.assertGreater(np.linalg.norm(expected - self.centre), 5)
        load = self.r._burden_in_white_matter(tissue, 0.5, torch.rand(1, 3))
        np.testing.assert_allclose(centre_of(load), expected, atol=1e-9)

    def test_the_drawn_placement_is_ignored(self):
        tissue = self.white(self.centre - 12, self.centre + 9)
        loads = [
            self.r._burden_in_white_matter(tissue, -0.3, torch.tensor([q])) for q in ([0.1, 0.9, 0.5], [0.8, 0.2, 0.3])
        ]
        torch.testing.assert_close(loads[0], loads[1])

    def test_random_leaves_the_renderer_alone(self):
        r = renderer()
        before = r._burden_in_white_matter
        bank.set_position(r, "random")
        self.assertEqual(r._burden_in_white_matter, before)


if __name__ == "__main__":
    unittest.main()
