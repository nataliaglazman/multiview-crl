"""mlmv geometry: eroded regions and half-splits are more than 2 rho apart; G sets are disjoint and balanced."""

import itertools
import unittest

import numpy as np
import torch

from mlmv.data.geometry import (
    build_geometry,
    draw_region_sets,
    erode_regions,
    half_split,
    majority_downsample,
    pairwise_min_distance,
    random_normals,
)
from mlmv.synth.three_level import voronoi_regions


def voronoi_with_background(size=32, n=8, seed=0):
    labels = np.repeat(np.repeat(np.repeat(voronoi_regions(size // 2, n, seed), 2, 0), 2, 1), 2, 2)
    centre = (np.indices(labels.shape) - (size - 1) / 2) / (size / 2)
    labels[(centre**2).sum(0) > 0.95] = 0  # a "brain" ball with background around it
    return labels


class TestErosion(unittest.TestCase):
    def test_eroded_regions_more_than_2rho_apart(self):
        labels = voronoi_with_background()
        for metric, rho in itertools.product(("chebyshev", "euclidean"), (1.5, 2.5, 3.0)):
            eroded = erode_regions(labels, rho, metric)
            regions = [r for r in np.unique(eroded) if r > 0]
            self.assertGreaterEqual(len(regions), 2, (metric, rho))
            for a, b in itertools.combinations(regions, 2):
                d = pairwise_min_distance(np.argwhere(eroded == a), np.argwhere(eroded == b), metric)
                self.assertGreater(d, 2 * rho, (metric, rho, a, b))

    def test_chebyshev_erosion_separates_receptive_boxes(self):
        # The theoretical receptive field is a box: no shared input needs L-inf separation > 2 rho.
        labels = voronoi_with_background(seed=3)
        eroded = erode_regions(labels, 2.5, "chebyshev")
        regions = [r for r in np.unique(eroded) if r > 0]
        for a, b in itertools.combinations(regions, 2):
            d = pairwise_min_distance(np.argwhere(eroded == a), np.argwhere(eroded == b), "chebyshev")
            self.assertGreater(d, 5.0)

    def test_background_is_not_eroded_against(self):
        labels = np.zeros((20, 20, 20), dtype=np.int64)
        labels[2:18, 2:18, 2:10] = 1
        labels[2:18, 2:18, 10:18] = 2
        eroded = erode_regions(labels, 1.5, "chebyshev")
        self.assertEqual(eroded[2, 9, 3], 1)  # next to background, far from region 2: kept
        self.assertEqual(eroded[9, 9, 9], 0)  # next to region 2: removed

    def test_majority_downsample(self):
        labels = np.zeros((4, 4, 4), dtype=np.int64)
        labels[:2, :2, :2] = 5
        labels[0, 0, 0] = 7
        labels[2:, 2:, 2:] = 3
        out = majority_downsample(labels, 2)
        self.assertEqual(out[0, 0, 0], 5)
        self.assertEqual(out[1, 1, 1], 3)
        self.assertEqual(out[0, 1, 0], 0)


class TestHalfSplit(unittest.TestCase):
    def test_halves_nonempty_and_more_than_2rho_apart(self):
        labels = voronoi_with_background(seed=1)
        for metric, rho in (("chebyshev", 2.5), ("euclidean", 2.5), ("chebyshev", 1.5)):
            geo = build_geometry(
                labels, rho, metric, min_positions_per_half=8, n_normals=64, seed=4, log=lambda s: None
            )
            self.assertTrue(geo.eligible)
            for region in geo.eligible:
                coords = geo.coords(region)
                # eligibility normals: both halves keep >= min positions
                for normal in random_normals(64, seed=4):
                    a, b = half_split(coords, normal, rho, metric)
                    self.assertGreaterEqual(min(a.sum(), b.sum()), 8)
                    self.assertGreater(pairwise_min_distance(coords[a], coords[b], metric), 2 * rho)
                # any normal: whatever the halves hold is more than 2 rho apart
                for normal in random_normals(16, seed=region):
                    a, b = half_split(coords, normal, rho, metric)
                    if a.any() and b.any():
                        self.assertGreater(pairwise_min_distance(coords[a], coords[b], metric), 2 * rho)


class TestRegionSets(unittest.TestCase):
    def test_sets_disjoint_cover_and_ratio(self):
        regions = np.arange(1, 13)
        volumes = np.random.default_rng(0).integers(200, 900, 12)
        gen = torch.Generator().manual_seed(0)
        for _ in range(200):
            a, b = draw_region_sets(regions, volumes, (0.4, 0.6), gen)
            self.assertEqual(set(a) & set(b), set())
            self.assertEqual(set(a) | set(b), set(regions))
            share = volumes[a - 1].sum() / volumes.sum()
            self.assertTrue(0.4 <= share <= 0.6)


class TestBuildGeometry(unittest.TestCase):
    def test_ineligible_regions_are_merged_and_logged(self):
        labels = voronoi_with_background(size=24, n=8, seed=0)
        logs = []
        geo = build_geometry(labels, 1.5, "chebyshev", min_positions_per_half=8, n_normals=64, log=logs.append)
        self.assertTrue(geo.eligible)
        for row in geo.table:
            if row["eligible"]:
                self.assertGreaterEqual(row["worst_half"], 8)
        self.assertEqual(len(logs), len(geo.merges))
        # eroded map only holds eligible regions, and they stay apart
        for a, b in itertools.combinations(geo.eligible, 2):
            self.assertGreater(pairwise_min_distance(geo.coords(a), geo.coords(b), "chebyshev"), 3.0)

    def test_default_synthetic_regions_all_eligible(self):
        labels = np.repeat(np.repeat(np.repeat(voronoi_regions(16, 8, seed=0), 2, 0), 2, 1), 2, 2)
        geo = build_geometry(labels, 2.5, "chebyshev", 8, 64, log=lambda s: None)
        self.assertEqual(len(geo.eligible), 8)


if __name__ == "__main__":
    unittest.main()
