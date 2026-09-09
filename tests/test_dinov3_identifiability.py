"""DINOv3 embedding + identifiability tests: slicing, windowing, probes, PC recovery."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval import dinov3_embed_synthetic as embed
from eval import dinov3_identifiability as score
from eval import run_causal_recovery as recovery


class SlicingTests(unittest.TestCase):
    def test_positions_are_bin_midpoints_and_reject_impossible_requests(self):
        self.assertEqual(embed.slice_positions(64, 1), [32])
        self.assertEqual(embed.slice_positions(64, 3), [10, 32, 53])
        self.assertEqual(embed.slice_positions(4, 4), [0, 1, 2, 3])
        for size, n in ((0, 1), (8, 0), (4, 5)):
            with self.subTest(size=size, n=n):
                with self.assertRaises(ValueError):
                    embed.slice_positions(size, n)

    def test_planes_follow_the_generator_axis_convention_and_slot_order(self):
        volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        planes = list(embed.volume_planes(volume, ["sagittal", "coronal", "axial"], 1))
        # x is left-right, so fixing axis 0 is a sagittal plane; axis 1 coronal; axis 2 axial.
        np.testing.assert_array_equal(planes[0], volume[1])
        np.testing.assert_array_equal(planes[1], volume[:, 1])
        np.testing.assert_array_equal(planes[2], volume[:, :, 2])
        self.assertEqual(
            embed.slot_names(volume.shape, ["sagittal", "axial"], 1),
            ["sagittal:1", "axial:2"],
        )
        self.assertEqual(len(embed.slot_names((8, 8, 8), ["axial", "coronal"], 3)), 6)


class WindowingTests(unittest.TestCase):
    def test_a_dataset_window_keeps_the_intensity_difference_a_per_slice_one_deletes(self):
        dim = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        bright = 2.0 * dim  # the same anatomy under a style gain of 2
        shared = embed.plane_window(np.concatenate([dim.ravel(), bright.ravel()]), (1, 99))
        self.assertLess(embed.window_to_uint8(dim, shared).mean(), embed.window_to_uint8(bright, shared).mean())
        self.assertEqual(
            embed.window_to_uint8(dim, embed.plane_window(dim, (1, 99))).mean(),
            embed.window_to_uint8(bright, embed.plane_window(bright, (1, 99))).mean(),
        )

    def test_degenerate_window_is_black_rather_than_a_divide_by_zero(self):
        out = embed.window_to_uint8(np.ones((4, 4)), (1.0, 1.0))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.sum(), 0)

    def test_window_clips_instead_of_wrapping(self):
        plane = np.array([[-5.0, 0.5, 7.0]])
        out = embed.window_to_uint8(plane, (0.0, 1.0))
        np.testing.assert_array_equal(out, np.array([[0, 127, 255]], dtype=np.uint8))


class TokenLayoutTests(unittest.TestCase):
    def test_prefix_is_inferred_from_the_sequence_not_trusted_from_the_config(self):
        import argparse

        self.assertEqual(embed.token_prefix(261, 256, argparse.Namespace(num_register_tokens=4)), 5)
        # A config that disagrees is reported but does not override the observed layout.
        self.assertEqual(embed.token_prefix(261, 256, argparse.Namespace(num_register_tokens=0)), 5)
        with self.assertRaises(ValueError):
            embed.token_prefix(256, 256, argparse.Namespace())

    def test_pooling_modes_read_the_intended_tokens(self):
        torch = self.skip_without_torch()
        hidden = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
        patches = hidden[:, 3:]
        torch.testing.assert_close(embed.pool_tokens(hidden, 3, (2, 2), "cls", 1), hidden[:, 0])
        # mean must average the 4 PATCH tokens, not the CLS and register tokens with them.
        torch.testing.assert_close(embed.pool_tokens(hidden, 3, (2, 2), "mean", 1), patches.mean(dim=1))
        self.assertEqual(embed.pool_tokens(hidden, 3, (2, 2), "cls_mean", 1).shape, (2, 6))
        torch.testing.assert_close(embed.pool_tokens(hidden, 3, (2, 2), "grid", 2), patches.transpose(1, 2).flatten(1))
        torch.testing.assert_close(embed.pool_tokens(hidden, 3, (2, 2), "grid", 1), patches.mean(dim=1))
        with self.assertRaises(ValueError):
            embed.pool_tokens(hidden, 3, (2, 2), "nonsense", 1)

    def skip_without_torch(self):
        try:
            import torch
        except ImportError:  # the scoring half of this file must stay torch-free
            self.skipTest("torch not installed")
        return torch


class OrientationTests(unittest.TestCase):
    def test_a_chains_cpdag_is_undirected_and_a_colliders_is_not(self):
        chain = recovery.true_cpdag(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.assertEqual(recovery.edge_type(chain, 0, 1), "undirected")
        collider = recovery.true_cpdag(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]))
        self.assertEqual(recovery.edge_type(collider, 0, 2), "forward")
        self.assertEqual(recovery.edge_type(collider, 0, 1), "none")

    def test_orientation_counts_separate_reversed_from_merely_unoriented(self):
        truth = recovery.true_cpdag(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]))
        perfect = recovery.orientation_metrics(truth, truth)
        self.assertTrue(perfect["cpdag_exact_match"])
        self.assertEqual((perfect["correct_directed"], perfect["cpdag_shd"]), (2, 0))

        flipped = truth.copy()
        flipped[0, 2], flipped[2, 0] = 1, -1  # 0 -> 2 becomes 2 -> 0
        reversed_score = recovery.orientation_metrics(flipped, truth)
        self.assertEqual(reversed_score["reversed"], 1)
        self.assertEqual((reversed_score["correct_directed"], reversed_score["cpdag_shd"]), (1, 1))

        unoriented = truth.copy()
        unoriented[2, 0] = -1  # 0 -> 2 becomes 0 - 2
        weak = recovery.orientation_metrics(unoriented, truth)
        self.assertEqual((weak["undirected_in_estimate"], weak["reversed"], weak["cpdag_shd"]), (1, 0, 1))

        missing = truth.copy()
        missing[0, 2] = missing[2, 0] = 0
        gone = recovery.orientation_metrics(missing, truth)
        self.assertEqual((gone["correct_directed"], gone["cpdag_shd"]), (1, 1))

    def test_orientation_is_opt_in_and_does_not_change_the_existing_panel(self):
        rng = np.random.RandomState(41)
        z = rng.randn(600, 3)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        plain = recovery.evaluate_arrays(z, z, truth, alphas=[0.05])
        oriented = recovery.evaluate_arrays(z, z, truth, alphas=[0.05], orientation=True)
        self.assertNotIn("orientation", plain["best"])
        self.assertNotIn("true_cpdag", plain)
        self.assertEqual(plain["best"]["f1"], oriented["best"]["f1"])
        # The chain's equivalence class is undirected, so an exact match claims no arrows.
        found = oriented["best"]["orientation"]
        self.assertTrue(found["cpdag_exact_match"])
        self.assertEqual((found["both_undirected"], found["correct_directed"]), (2, 0))


class IndepTestTests(unittest.TestCase):
    """--indep-test: fisherz sees only linear dependence, kci sees the nonlinear part."""

    @staticmethod
    def _nonlinear_pair(seed=0, n=200):
        """x -> y = x^2, an edge with (exactly) zero sample linear correlation.

        ``x`` is antithetic, so sum(x^3) == 0 by construction and the sample corr(x, x^2)
        is zero up to the noise term rather than up to sampling luck. That makes this a
        deterministic separation between the two tests, not a seed-dependent one.
        """
        rng = np.random.RandomState(seed)
        u = rng.randn(n)
        x = np.concatenate([u, -u])
        y = x**2 + 0.1 * rng.randn(len(x))
        return np.column_stack([x, y]), np.array([[0, 1], [0, 0]], dtype=bool)

    def test_kci_finds_a_purely_nonlinear_edge_that_fisherz_cannot(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                z, adjacency = self._nonlinear_pair(seed)
                self.assertLess(abs(float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])), 0.05)
                linear = recovery.evaluate_arrays(z, z, adjacency, alphas=[0.05], indep_test="fisherz")
                kernel = recovery.evaluate_arrays(z, z, adjacency, alphas=[0.05], indep_test="kci")
                self.assertEqual((linear["best"]["tp"], linear["best"]["fn"]), (0, 1))
                self.assertEqual((kernel["best"]["tp"], kernel["best"]["fn"]), (1, 0))

    def test_both_tests_agree_on_a_linear_chain(self):
        rng = np.random.RandomState(7)
        z = rng.randn(300, 3)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        for test in recovery.INDEP_TESTS:
            with self.subTest(indep_test=test):
                result = recovery.evaluate_arrays(z, z, truth, alphas=[0.05], indep_test=test)
                self.assertTrue(result["best"]["exact_match"])
                self.assertEqual(result["indep_test"], test)

    def test_capping_the_conditioning_set_only_adds_skeleton_edges(self):
        # max_cond_set is what makes kci finish at realistic factor counts; the cost is
        # that pairs needing a larger conditioning set keep their edge.
        rng = np.random.RandomState(5)
        z = rng.randn(400, 4)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        z[:, 3] += 1.3 * z[:, 2]
        truth = np.zeros((4, 4), dtype=bool)
        truth[0, 1] = truth[1, 2] = truth[2, 3] = True
        unbounded = recovery.evaluate_arrays(z, z, truth, alphas=[0.05])
        capped = recovery.evaluate_arrays(z, z, truth, alphas=[0.05], max_cond_set=0)
        self.assertEqual(unbounded["max_cond_set"], None)
        self.assertEqual(capped["max_cond_set"], 0)
        self.assertTrue(unbounded["best"]["exact_match"])
        # With no conditioning allowed, PC cannot separate the chain's non-adjacent pairs.
        self.assertGreater(capped["best"]["fp"], 0)
        self.assertEqual(capped["best"]["fn"], 0)

    def test_default_is_fisherz_and_an_unknown_test_is_rejected(self):
        z, adjacency = self._nonlinear_pair()
        self.assertEqual(recovery.evaluate_arrays(z, z, adjacency, alphas=[0.05])["indep_test"], "fisherz")
        with self.assertRaises(ValueError):
            recovery.evaluate_arrays(z, z, adjacency, alphas=[0.05], indep_test="spearman")


class ReadoutTests(unittest.TestCase):
    """--readout-dim and --holdout-readout, the two knobs on the graph readout."""

    def setUp(self):
        rng = np.random.RandomState(11)
        self.n = 400
        z = rng.randn(self.n, 3)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        self.z = z
        self.truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        self.X = z @ rng.randn(3, 40) + 0.1 * rng.randn(self.n, 40)

    def test_default_width_reproduces_the_original_rule(self):
        # 64, floored at the factor count, capped at N/5 and at the block's own width.
        self.assertEqual(recovery.readout_width(2000, 5000, 9), 64)
        self.assertEqual(recovery.readout_width(2000, 40, 9), 40)
        self.assertEqual(recovery.readout_width(30, 5000, 9), 9)  # N/5 < n_content, floor binds
        self.assertEqual(recovery.readout_width(200, 5000, 9), 40)  # N/5 = 40 < 64

    def test_explicit_width_pins_it_but_cannot_exceed_the_block(self):
        self.assertEqual(recovery.readout_width(2000, 5000, 9, 16), 16)
        self.assertEqual(recovery.readout_width(2000, 5000, 9, 200), 200)
        self.assertEqual(recovery.readout_width(2000, 48, 9, 64), 48)  # a narrow block keeps its width
        self.assertEqual(recovery.readout_width(30, 5000, 9, 64), 30)
        with self.assertRaises(ValueError):
            recovery.readout_width(2000, 5000, 9, 0)

    def test_readout_dim_is_reported_and_changes_only_the_readout(self):
        wide = recovery.evaluate_arrays(self.X, self.z, self.truth, alphas=[0.05])
        thin = recovery.evaluate_arrays(self.X, self.z, self.truth, alphas=[0.05], readout_dim=3)
        self.assertEqual((wide["graph_readout_dim"], thin["graph_readout_dim"]), (40, 3))
        # raw/partial come from the full-width probe, so pinning the readout must not move them.
        self.assertAlmostEqual(wide["raw_r2_mean"], thin["raw_r2_mean"], places=12)
        self.assertAlmostEqual(wide["partial_r2_mean"], thin["partial_r2_mean"], places=12)

    def test_holdout_runs_pc_on_unseen_rows_only(self):
        insample = recovery.evaluate_arrays(self.X, self.z, self.truth, alphas=[0.05])
        held = recovery.evaluate_arrays(self.X, self.z, self.truth, alphas=[0.05], holdout_readout=True)
        self.assertEqual((insample["readout_mode"], held["readout_mode"]), ("in_sample", "holdout"))
        self.assertEqual((insample["graph_samples"], held["graph_samples"]), (self.n, int(0.3 * self.n)))
        # The planted chain survives the split; both should still recover it exactly.
        self.assertTrue(insample["best"]["exact_match"])
        self.assertTrue(held["best"]["exact_match"])

    def test_holdout_correlations_use_the_test_rows_not_all_rows(self):
        held = recovery.evaluate_arrays(self.X, self.z, self.truth, alphas=[0.05], holdout_readout=True)
        for factor in held["factors"]:
            self.assertIsNotNone(factor["decoded_gt_correlation"])
            self.assertGreater(factor["decoded_gt_correlation"], 0.9)

    def test_holdout_refuses_a_test_split_too_small_for_pc(self):
        with self.assertRaises(ValueError):
            recovery.evaluate_arrays(self.X[:60], self.z[:60], self.truth, alphas=[0.05], holdout_readout=True)

    def test_defaults_leave_existing_results_untouched(self):
        before = recovery.evaluate_arrays(self.X, self.z, self.truth, alphas=[0.05])
        after = recovery.evaluate_arrays(
            self.X, self.z, self.truth, alphas=[0.05], readout_dim=None, holdout_readout=False
        )
        self.assertEqual(before["best"], after["best"])
        self.assertEqual(before["factors"], after["factors"])


def _bundle(X, z, adjacency, names, raw=None, path="planted", style=None, style_names=()):
    return dict(
        path=path,
        X=X,
        raw=raw,
        z_content=z,
        z_style=style,
        adjacency=adjacency,
        content_names=list(names),
        style_names=list(style_names),
        meta={},
        view="1",
    )


def _options(**overrides):
    import argparse

    base = dict(
        # A Path here on purpose: the report has to stay JSON-serialisable with the real CLI.
        embeddings=Path("emb.npz"),
        probe_kind="ridge",
        seeds=(0, 1),
        n_splits=5,
        n_null=2,
        null_seed=0,
        probe_dim=score.PROBE_DIM_AUTO,
        alphas=[0.05],
        diagnostic_alpha=0.05,
        orientation=True,
        pc_ceiling=True,
        with_graph=True,
        readout_dim=None,
        holdout_readout=False,
        indep_test="fisherz",
        max_cond_set=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(0)
        cls.n = 300
        z = rng.randn(cls.n, 3)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        cls.z = z
        cls.adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        cls.names = ["a", "b", "c"]
        cls.signal = _bundle(z @ rng.randn(3, 24) + 0.1 * rng.randn(cls.n, 24), z, cls.adjacency, cls.names)
        cls.noise = _bundle(rng.randn(cls.n, 24), z, cls.adjacency, cls.names, path="floor")

    def test_planted_factors_score_above_null_and_above_an_untrained_floor(self):
        rows = score.score_block(
            dict(self.signal, raw=np.random.RandomState(1).randn(self.n, 8)), self.noise, "content", _options()
        )
        for name in self.names:
            self.assertGreater(rows[name]["gap"], 0.8, name)
            self.assertGreater(rows[name]["delta_floor"], 0.7, name)
            self.assertGreater(rows[name]["delta_voxels"], 0.7, name)
        self.assertGreater(rows["_block"]["mcc_mean"], 0.9)

    def test_signal_free_features_score_at_zero_after_the_null_correction(self):
        rows = score.score_block(self.noise, None, "content", _options())
        for name in self.names:
            self.assertLess(abs(rows[name]["gap"]), 0.15, (name, rows[name]))
        self.assertNotIn("floor_gap", rows["a"])

    def test_constant_factors_are_dropped_from_both_the_table_and_the_adjacency(self):
        z = np.column_stack([self.z, np.zeros(self.n)])
        adjacency = np.zeros((4, 4), dtype=bool)
        adjacency[:3, :3] = self.adjacency
        adjacency[3, 0] = True  # an edge on the constant factor, which must go with it
        bundle = _bundle(self.signal["X"], z, adjacency, [*self.names, "constant"])
        rows = score.score_block(bundle, None, "content", _options())
        self.assertNotIn("constant", rows)
        panel = score.score_graphs(bundle, None, _options())["embeddings"]
        self.assertEqual(panel["factor_names"], self.names)
        self.assertEqual(np.asarray(panel["true_dag"]).shape, (3, 3))

    def test_pc_recovers_the_planted_chain_and_reports_the_ground_truth_ceiling(self):
        panels = score.score_graphs(self.signal, self.noise, _options())
        self.assertTrue(panels["embeddings"]["best"]["exact_match"])
        self.assertTrue(panels["truth"]["best"]["exact_match"])
        self.assertIn("floor", panels)
        self.assertTrue(panels["embeddings"]["alpha_sweep"][0]["orientation"]["cpdag_exact_match"])

    def test_parent_only_features_keep_raw_r2_but_lose_the_childs_partial_r2(self):
        rng = np.random.RandomState(3)
        parent_only = _bundle(
            np.column_stack([self.z[:, 0], rng.randn(self.n, 10)]), self.z, self.adjacency, self.names
        )
        child = score.score_graphs(parent_only, None, _options(pc_ceiling=False))["embeddings"]["factors"][1]
        self.assertGreater(child["raw_r2"], 0.4)
        self.assertLess(child["partial_r2"], 0.15)

    def test_no_adjacency_means_no_graph_section_rather_than_a_crash(self):
        bundle = _bundle(self.signal["X"], self.z, None, self.names)
        self.assertEqual(score.score_graphs(bundle, None, _options()), {})

    def test_probe_dim_auto_reduces_only_the_wide_block(self):
        narrow, width = score.reduce_features(np.random.RandomState(0).randn(200, 20), score.PROBE_DIM_AUTO)
        self.assertEqual((narrow.shape[1], width), (20, 20))
        wide, width = score.reduce_features(np.random.RandomState(0).randn(200, 5000), score.PROBE_DIM_AUTO)
        self.assertEqual((wide.shape[1], width), (50, 50))
        fixed, width = score.reduce_features(np.random.RandomState(0).randn(200, 5000), 8)
        self.assertEqual((fixed.shape[1], width), (8, 8))


class RoundTripTests(unittest.TestCase):
    def test_saved_arrays_load_back_into_a_scorable_bundle(self):
        rng = np.random.RandomState(5)
        n = 60
        z = rng.randn(n, 3)
        arrays = dict(
            emb_view1=rng.randn(n, 12).astype(np.float32),
            emb_view2=rng.randn(n, 12).astype(np.float32),
            raw_view1=rng.randn(n, 4).astype(np.float32),
            raw_view2=rng.randn(n, 4).astype(np.float32),
            z_content=z.astype(np.float32),
            z_style_v1=rng.randn(n, 3).astype(np.float32),
            z_style_v2=rng.randn(n, 3).astype(np.float32),
            causal_adj=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32),
            meta=json.dumps({"content_factor_names": ["a", "b", "c"], "style_factor_names": ["gain", "bias", "sig"]}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emb.npz"
            np.savez_compressed(path, **arrays)
            single = score.load_bundle(path, "1")
            self.assertEqual(single["X"].shape, (n, 12))
            self.assertEqual(single["content_names"], ["a", "b", "c"])
            self.assertTrue(single["adjacency"][0, 1])
            both = score.load_bundle(path, "both")
            self.assertEqual((both["X"].shape[1], both["raw"].shape[1]), (24, 8))
            with self.assertRaises(KeyError):
                score.load_bundle(Path(tmp) / "emb.npz", "3")

    def test_report_and_csv_render_from_a_full_score(self):
        rng = np.random.RandomState(7)
        n = 200
        z = rng.randn(n, 3)
        z[:, 1] += 1.3 * z[:, 0]
        z[:, 2] += 1.3 * z[:, 1]
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        bundle = _bundle(
            z @ rng.randn(3, 16) + 0.1 * rng.randn(n, 16),
            z,
            adjacency,
            ["a", "b", "c"],
            raw=rng.randn(n, 6),
            style=rng.randn(n, 2),
            style_names=["gain", "bias"],
        )
        result = score.score(bundle, None, _options())
        text = score.format_report(result)
        for section in ("1. CONTENT FACTORS", "2. STYLE / NUISANCE", "3. GRAPH RECOVERY", "4. VERDICT"):
            self.assertIn(section, text)
        json.dumps(result, allow_nan=False, default=float)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factors.csv"
            score.write_csv(path, result)
            rows = path.read_text().strip().splitlines()
        self.assertEqual(len(rows), 6)  # header + 3 content + 2 style
        self.assertIn("partial_r2", rows[0])


if __name__ == "__main__":
    unittest.main()
