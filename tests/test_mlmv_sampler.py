"""mlmv view-pair sampler: S spread over regions, L halves separated, G sets disjoint, determinism."""

import unittest

import numpy as np
import torch

from mlmv.config import load_config
from mlmv.data.geometry import build_geometry, pairwise_min_distance
from mlmv.data.sampler import ViewPairSampler
from mlmv.synth.three_level import voronoi_regions


def make(fraction=1.0, rho=2.5):
    cfg = load_config(overrides={"sampler": {"S": {"positions": 64}, "L": {"fraction": fraction}}})
    labels = np.repeat(np.repeat(np.repeat(voronoi_regions(16, 8, seed=0), 2, 0), 2, 1), 2, 2)
    geo = build_geometry(labels, rho, "chebyshev", 8, 64, log=lambda s: None)
    return ViewPairSampler(geo, cfg.sampler), geo, cfg


class TestSampler(unittest.TestCase):
    def test_s_positions_spread_over_regions(self):
        sampler, geo, _ = make()
        out = sampler.sample(3, torch.Generator().manual_seed(0))
        self.assertEqual(tuple(out["S_idx"].shape), (3, 64))
        labels = geo.labels.ravel()
        self.assertTrue((labels[out["S_idx"].numpy()] == out["S_region"].numpy()).all())
        counts = np.bincount(out["S_region"][0].numpy())[1:]
        self.assertEqual(set(counts.tolist()), {8})  # 64 positions over 8 regions

    def test_l_halves_are_separated_subsets(self):
        for fraction in (1.0, 0.25):
            sampler, geo, cfg = make(fraction)
            out = sampler.sample(2, torch.Generator().manual_seed(1))
            n_l = out["L_region"].shape[1]
            self.assertEqual(n_l, min(cfg.sampler.L.regions, len(geo.eligible)))
            shape = geo.shape
            for b in range(2):
                for k in range(n_l):
                    region = int(out["L_region"][b, k])
                    halves = []
                    for h in range(2):
                        flat = out["L_flat"][out["L_seg"] == (b * n_l + k) * 2 + h].numpy()
                        self.assertGreater(len(flat), 0)
                        coords = np.stack(np.unravel_index(flat, shape), 1)
                        self.assertTrue((geo.eroded[tuple(coords.T)] == region).all())
                        halves.append(coords)
                    self.assertGreater(pairwise_min_distance(halves[0], halves[1], "chebyshev"), 2 * geo.rho)
                    if fraction == 1.0:
                        self.assertGreaterEqual(min(len(h) for h in halves), geo.min_positions_per_half)

    def test_g_sets_disjoint_balanced(self):
        sampler, geo, cfg = make()
        out = sampler.sample(2, torch.Generator().manual_seed(2))
        m = out["G_member"]
        self.assertEqual(m.shape[1], cfg.sampler.G.splits_per_subject)
        self.assertFalse((m[:, :, 0] & m[:, :, 1]).any())
        self.assertTrue((m[:, :, 0] | m[:, :, 1]).all())
        vol = torch.tensor(sampler.volumes)
        share = (m[:, :, 0].double() @ vol) / vol.sum()
        self.assertTrue(((share >= 0.4) & (share <= 0.6)).all())

    def test_deterministic_given_generator(self):
        sampler, _, _ = make()
        a = sampler.sample(2, torch.Generator().manual_seed(5))
        b = sampler.sample(2, torch.Generator().manual_seed(5))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)


if __name__ == "__main__":
    unittest.main()
