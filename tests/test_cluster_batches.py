"""--stratify-batches without torch: the cluster and neighbourhood batch samplers, their diagnostics and flag rules."""

import ast
import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import data.cluster_batches as cluster_batches
from training import encoder_args

SIZES = (5, 40, 17, 64, 8, 33)  # clusters 0 and 4 are smaller than a batch of 16


def labels_of(sizes, seed=0):
    labels = np.concatenate([np.full(n, c) for c, n in enumerate(sizes)])
    return np.random.default_rng(seed).permutation(labels)


def epochs(sampler, n):
    return [list(sampler) for _ in range(n)]


class SamplerTests(unittest.TestCase):
    def test_a_batch_never_mixes_clusters_and_is_always_full(self):
        labels = labels_of(SIZES)
        sampler = cluster_batches.ClusterBatchSampler(labels, 16, seed=3)
        self.assertEqual(sampler.kept, [1, 2, 3, 5])
        for epoch in epochs(sampler, 3):
            self.assertEqual(len(epoch), len(sampler))
            self.assertEqual(len(sampler), 40 // 16 + 17 // 16 + 64 // 16 + 33 // 16)
            for batch in epoch:
                self.assertEqual(len(batch), 16)
                self.assertEqual(len(set(labels[batch])), 1)
                self.assertTrue(all(isinstance(i, int) for i in batch))
            seen = [i for batch in epoch for i in batch]
            self.assertEqual(len(seen), len(set(seen)))  # no subject twice in an epoch
            self.assertFalse(set(labels[seen]) & {0, 4})  # dropped clusters never appear

    def test_epochs_cycle_through_the_clusters_and_replay_from_the_seed(self):
        labels = labels_of(SIZES)
        sampler = cluster_batches.ClusterBatchSampler(labels, 16, seed=3)
        first, second = epochs(sampler, 2)
        # One pass of the cycle takes a batch from every cluster before any cluster gives a second;
        # later passes keep that order and skip the clusters that have run out.
        cycle = [int(labels[b[0]]) for b in first[:4]]
        self.assertEqual(sorted(cycle), [1, 2, 3, 5])
        left = {c: SIZES[c] // 16 for c in cycle}
        expected = []
        while any(left.values()):
            for c in cycle:
                if left[c]:
                    expected.append(c)
                    left[c] -= 1
        self.assertEqual([int(labels[b[0]]) for b in first], expected)
        self.assertNotEqual(first, second)
        replay = cluster_batches.ClusterBatchSampler(labels, 16, seed=3)
        self.assertEqual(epochs(replay, 2), [first, second])
        self.assertNotEqual(list(cluster_batches.ClusterBatchSampler(labels, 16, seed=4)), first)

    def test_one_cluster_is_the_whole_bank_and_too_small_clusters_refuse(self):
        sampler = cluster_batches.ClusterBatchSampler(np.zeros(70, dtype=int), 16, seed=0)
        self.assertEqual(len(sampler), 4)
        self.assertEqual(len({i for b in sampler for i in b}), 64)
        with self.assertRaises(ValueError):
            cluster_batches.ClusterBatchSampler(labels_of((5, 8)), 16, seed=0)

    def test_kmeans_with_one_cluster_needs_no_fit_and_k_clusters_split_the_features(self):
        features = np.random.default_rng(0).normal(size=(300, 2))
        np.testing.assert_array_equal(cluster_batches.kmeans_labels(features, 1, seed=0), np.zeros(300))
        labels = cluster_batches.kmeans_labels(features, 4, seed=0)
        self.assertEqual(sorted(set(labels.tolist())), [0, 1, 2, 3])
        np.testing.assert_array_equal(labels, cluster_batches.kmeans_labels(features, 4, seed=0))


def pools(n=100, k=20, seed=0):
    return cluster_batches.nearest_neighbours(np.random.default_rng(seed).normal(size=(n, 3)), k)


def first_batches(sampler, count):
    """The first ``count`` batches a training loop draws: ``sampler`` epoch after epoch."""
    drawn = []
    while len(drawn) < count:
        drawn.extend(sampler)
    return drawn[:count]


class NeighbourhoodTests(unittest.TestCase):
    def test_neighbours_are_the_closest_other_subjects_on_standardized_features(self):
        line = np.arange(10.0)[:, None]
        found = cluster_batches.nearest_neighbours(line, 2)
        self.assertEqual(found.shape, (10, 2))
        self.assertEqual(set(found[0].tolist()), {1, 2})
        self.assertEqual(set(found[5].tolist()), {4, 6})
        self.assertEqual(set(found[9].tolist()), {7, 8})
        # Standardized: a column's units do not decide which subjects are near.
        features = np.random.default_rng(0).normal(size=(200, 2))
        np.testing.assert_array_equal(
            cluster_batches.nearest_neighbours(features, 10),
            cluster_batches.nearest_neighbours(features * [1.0, 1000.0], 10),
        )
        # Identical rows tie at distance 0; a subject is still never its own neighbour.
        ties = cluster_batches.nearest_neighbours(np.array([[0.0], [0.0], [0.0], [5.0], [6.0]]), 2)
        for i, row in enumerate(ties):
            self.assertNotIn(i, row.tolist())
            self.assertEqual(len(set(row.tolist())), 2)
        self.assertEqual(set(ties[0].tolist()), {1, 2})
        for bad in (0, 10):
            with self.assertRaises(ValueError):
                cluster_batches.nearest_neighbours(line, bad)
        with self.assertRaises(ValueError):
            cluster_batches.nearest_neighbours(np.array([[0.0], [np.nan], [1.0]]), 1)

    def test_a_batch_is_its_anchor_then_distinct_subjects_from_the_anchor_pool(self):
        neighbours = pools()
        sampler = cluster_batches.NeighbourhoodBatchSampler(neighbours, 8, seed=3)
        self.assertEqual(len(sampler), 100 // 8)
        for epoch in epochs(sampler, 3):
            self.assertEqual(len(epoch), len(sampler))
            for batch in epoch:
                self.assertEqual(len(batch), 8)
                self.assertEqual(len(set(batch)), 8)
                self.assertTrue(all(isinstance(i, int) for i in batch))
                self.assertLessEqual(set(batch[1:]), set(neighbours[batch[0]].tolist()))

    def test_every_subject_anchors_once_per_pass_and_passes_run_across_epochs(self):
        neighbours = pools()
        sampler = cluster_batches.NeighbourhoodBatchSampler(neighbours, 8, seed=3)
        drawn = first_batches(sampler, 300)  # 25 loader epochs of 12 batches
        for start in (0, 100, 200):
            anchors = sorted(batch[0] for batch in drawn[start : start + 100])
            self.assertEqual(anchors, list(range(100)))
        self.assertNotEqual([b[0] for b in drawn[:100]], [b[0] for b in drawn[100:200]])
        replay = cluster_batches.NeighbourhoodBatchSampler(neighbours, 8, seed=3)
        self.assertEqual(first_batches(replay, 300), drawn)
        other = cluster_batches.NeighbourhoodBatchSampler(neighbours, 8, seed=4)
        self.assertNotEqual(first_batches(other, 12), drawn[:12])

    def test_a_pool_of_exactly_b_minus_1_gives_the_whole_pool(self):
        neighbours = pools(k=7)
        for batch in cluster_batches.NeighbourhoodBatchSampler(neighbours, 8, seed=0):
            self.assertEqual(set(batch[1:]), set(neighbours[batch[0]].tolist()))

    def test_subjects_a_cluster_sampler_drops_still_train(self):
        labels = labels_of(SIZES)
        centres = np.random.default_rng(1).normal(scale=10, size=(len(SIZES), 4))
        features = centres[labels] + np.random.default_rng(2).normal(size=(len(labels), 4))
        dropped = set(np.flatnonzero(np.isin(labels, [0, 4])).tolist())
        clustered = {
            i for epoch in epochs(cluster_batches.ClusterBatchSampler(labels, 16, seed=0), 5) for b in epoch for i in b
        }
        self.assertFalse(clustered & dropped)
        neighbours = cluster_batches.nearest_neighbours(features, 30)
        drawn = first_batches(cluster_batches.NeighbourhoodBatchSampler(neighbours, 16, seed=0), len(labels))
        self.assertEqual({i for b in drawn for i in b}, set(range(len(labels))))
        self.assertLessEqual(dropped, {b[0] for b in drawn})

    def test_bad_pools_refuse(self):
        good = pools(n=20, k=5)
        cluster_batches.NeighbourhoodBatchSampler(good, 6, seed=0)
        own = good.copy()
        own[3, 0] = 3
        repeated = good.copy()
        repeated[3, 1] = repeated[3, 0]
        outside = good.copy()
        outside[3, 0] = 20
        for neighbours, batch in (
            (own, 6),
            (repeated, 6),
            (outside, 6),
            (good, 7),  # pool smaller than B-1
            (good, 1),
            (good.astype(float), 6),
            (good[:, None], 6),
        ):
            with self.assertRaises(ValueError):
                cluster_batches.NeighbourhoodBatchSampler(neighbours, batch, seed=0)

    def test_summary_is_the_runs_first_anchor_pass(self):
        rng = np.random.default_rng(0)
        factors = np.column_stack([rng.uniform(-1, 1, size=(1000, 2)), np.ones(1000)])
        names = ["brain_size", "ventricle_size", "lesion_burden"]
        neighbours = cluster_batches.nearest_neighbours(factors[:, [0]], 50)
        summary = cluster_batches.summarize_neighbourhood(neighbours, 16, 7, factors, names, [0, 1], matched=[0])
        drawn = np.array(first_batches(cluster_batches.NeighbourhoodBatchSampler(neighbours, 16, seed=7), 1000))
        preview = summary["preview"]
        self.assertEqual(preview["subject_draw_counts"], np.bincount(drawn.ravel(), minlength=1000).tolist())
        self.assertEqual(preview["anchor_counts"], [1] * 1000)
        self.assertEqual(preview["unique_subjects"], 1000)
        self.assertEqual(sum(preview["subject_draw_counts_first_epoch"]), 16 * (1000 // 16))
        self.assertEqual(
            (summary["batches_per_epoch"], summary["subject_draws_per_epoch"], summary["batches_per_anchor_pass"]),
            (62, 62 * 16, 1000),
        )
        self.assertEqual(summary["matched_factors"], ["brain_size"])
        self.assertEqual(summary["anatomy_factors"], ["brain_size", "ventricle_size"])
        # The matched factor's spread among a batch's negatives collapses; the free one keeps it.
        self.assertLess(summary["sd_ratio"]["brain_size"], 0.1)
        self.assertAlmostEqual(summary["sd_ratio"]["ventricle_size"], 1.0, delta=0.05)
        self.assertIsNone(summary["sd_ratio"]["lesion_burden"])  # constant column
        self.assertAlmostEqual(summary["sd_ratio_matched_mean"], summary["sd_ratio"]["brain_size"])
        json.dumps(summary, allow_nan=False)
        code = cluster_batches.summarize_neighbourhood(neighbours, 16, 7, factors, names, [0, 1])
        self.assertIsNone(code["matched_factors"])
        self.assertIsNone(code["sd_ratio_matched_mean"])


class DiagnosticTests(unittest.TestCase):
    def test_anatomy_leaves_out_every_lesion_latent_under_either_target(self):
        # eval.metrics.dci.content_factor_names(9, target), read from its source: the module imports torch.
        names = dci_content_factor_names()
        burden = [*names[:2], "lesion_burden", "unused_3", "unused_4", *names[5:]]
        for target, listed in (("position", names), ("burden", burden)):
            with self.subTest(target=target):
                self.assertEqual(cluster_batches.anatomy_columns(listed), [0, 1, 5, 6, 7, 8])
                self.assertEqual(tuple(listed[j] for j in [0, 1, 5, 6, 7, 8]), encoder_args.ANATOMY_FACTORS)

    def test_sd_ratio_is_one_for_one_cluster_and_shrinks_only_the_clustered_factor(self):
        rng = np.random.default_rng(0)
        factors = rng.uniform(-1, 1, size=(4000, 2))
        one = cluster_batches.within_cluster_sd_ratio(factors, np.zeros(4000, dtype=int), 32)
        np.testing.assert_allclose(one, 1.0)
        # Ten equal bins of column 0: its spread falls to ~1/10, column 1 keeps all of it.
        labels = np.minimum(((factors[:, 0] + 1) * 5).astype(int), 9)
        ratio = cluster_batches.within_cluster_sd_ratio(factors, labels, 32)
        self.assertAlmostEqual(ratio[0], 0.1, delta=0.01)
        self.assertAlmostEqual(ratio[1], 1.0, delta=0.03)

    def test_summary_counts_what_the_sampler_keeps(self):
        labels = labels_of(SIZES)
        names = ["brain_size", "lesion_burden", "ventricle_size"]
        factors = np.random.default_rng(1).normal(size=(len(labels), 3))
        summary = cluster_batches.summarize(labels, 16, factors, names, [0, 2], k=6)
        sampler = cluster_batches.ClusterBatchSampler(labels, 16, seed=0)
        self.assertEqual(summary["clusters"], 6)
        self.assertEqual(summary["clusters_kept"], len(sampler.kept))
        self.assertEqual(summary["subjects_in_kept_clusters"], 40 + 17 + 64 + 33)
        self.assertEqual(summary["batches_per_epoch"], len(sampler))
        self.assertEqual(summary["subjects_per_epoch"], 16 * len(sampler))
        self.assertEqual(summary["anatomy_factors"], ["brain_size", "ventricle_size"])
        # A code was clustered: no factor was, and the theory line refers to the anatomy.
        self.assertIsNone(summary["clustered_factors"])
        self.assertIsNone(summary["sd_ratio_clustered_mean"])
        self.assertAlmostEqual(summary["theory_k_pow_minus_1_over_d"], 6**-0.5)
        # The oracle clustered one of the two anatomy factors (--stratify-factors brain_size).
        chosen = cluster_batches.summarize(labels, 16, factors, names, [0, 2], k=6, clustered=[0])
        self.assertEqual(chosen["clustered_factors"], ["brain_size"])
        self.assertEqual(chosen["anatomy_factors"], ["brain_size", "ventricle_size"])
        self.assertAlmostEqual(chosen["sd_ratio_clustered_mean"], chosen["sd_ratio"]["brain_size"])
        self.assertAlmostEqual(chosen["theory_k_pow_minus_1_over_d"], 1 / 6)

    def test_within_cluster_probe_self_test(self):
        from eval.encoder import within_cluster_probe

        with contextlib.redirect_stdout(io.StringIO()):
            within_cluster_probe.self_test()


SYNTHETIC = ["--res", "32", "--num-train-samples", "640", "--batch-size", "32"]


def parse(*extra):
    return encoder_args.parse_args([*SYNTHETIC, *extra])


def refused(test, *extra):
    with contextlib.redirect_stderr(io.StringIO()), test.assertRaises(SystemExit):
        parse(*extra)


class FlagTests(unittest.TestCase):
    def test_none_is_the_default_and_leaves_the_shuffled_loader_untouched(self):
        args = parse()
        self.assertEqual((args.stratify_batches, args.stratify_clusters, args.stratify_from), ("none", None, None))

        def never():
            raise AssertionError("a default run must not build a cluster sampler")

        expected = {"batch_size": 32, "shuffle": True, "drop_last": True}
        self.assertEqual(cluster_batches.loader_batching(args, never), expected)
        # Settings saved before the flag existed read as none.
        old = SimpleNamespace(**{k: v for k, v in vars(args).items() if not k.startswith("stratify_")})
        self.assertEqual(cluster_batches.loader_batching(old, never), expected)
        # A stratified run hands the loader its sampler and nothing else about batching.
        sampler = object()
        stratified = parse("--stratify-batches", "oracle", "--stratify-clusters", "10")
        self.assertEqual(cluster_batches.loader_batching(stratified, lambda: sampler), {"batch_sampler": sampler})

    def test_flag_rules(self):
        self.assertEqual(parse("--stratify-batches", "oracle", "--stratify-clusters", "20").stratify_clusters, 20)
        self.assertEqual(
            parse("--stratify-batches", "code", "--stratify-clusters", "1", "--stratify-from", "runs/x").stratify_from,
            "runs/x",
        )
        refused(self, "--stratify-clusters", "10")  # without --stratify-batches
        refused(self, "--stratify-from", "runs/x")
        refused(self, "--stratify-batches", "oracle")  # K is required
        refused(self, "--stratify-batches", "oracle", "--stratify-clusters", "0")
        refused(self, "--stratify-batches", "oracle", "--stratify-clusters", "21")  # 640 // 32 = 20 batches
        refused(self, "--stratify-batches", "oracle", "--stratify-clusters", "10", "--stratify-from", "runs/x")
        refused(self, "--stratify-batches", "code", "--stratify-clusters", "10")
        refused(self, "--stratify-batches", "oracle", "--stratify-clusters", "10", "--synthetic-mode", "categorical")

    def test_neighbourhood_flag_rules(self):
        oracle = ["--stratify-batches", "oracle", "--stratify-sampler", "neighbourhood"]
        args = parse(*oracle, "--stratify-neighbours", "100")
        self.assertEqual(
            (args.stratify_sampler, args.stratify_neighbours, args.stratify_clusters), ("neighbourhood", 100, None)
        )
        self.assertEqual(parse("--stratify-batches", "oracle", "--stratify-clusters", "10").stratify_sampler, "cluster")
        parse(*oracle, "--stratify-neighbours", "31")  # B-1
        parse(*oracle, "--stratify-neighbours", "639")  # N-1
        code = ["--stratify-batches", "code", "--stratify-from", "runs/x", "--stratify-sampler", "neighbourhood"]
        self.assertEqual(parse(*code, "--stratify-neighbours", "64").stratify_from, "runs/x")
        refused(self, *oracle)  # the pool size is required
        refused(self, *oracle, "--stratify-neighbours", "30")  # cannot fill a batch of 32
        refused(self, *oracle, "--stratify-neighbours", "640")  # every other subject is 639
        refused(self, *oracle, "--stratify-neighbours", "100", "--stratify-clusters", "10")
        refused(self, "--stratify-batches", "oracle", "--stratify-clusters", "10", "--stratify-neighbours", "100")
        refused(self, "--stratify-sampler", "neighbourhood")  # without --stratify-batches
        refused(self, "--stratify-neighbours", "100")
        options = {"stratify_batches": "oracle", "stratify_sampler": "neighbourhood", "stratify_neighbours": 100}
        roundtrip = encoder_args.parse_args([*SYNTHETIC, *encoder_args.cli_arguments(options)])
        self.assertEqual((roundtrip.stratify_sampler, roundtrip.stratify_neighbours), ("neighbourhood", 100))

    def test_the_cli_module_needs_only_the_standard_library(self):
        # scripts/launch.py imports it on submit hosts that may have nothing but PyYAML.
        code = "import sys; sys.modules.update(numpy=None, torch=None, sklearn=None); import training.encoder_args"
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)

    def test_stratify_factors_rules(self):
        oracle = ["--stratify-batches", "oracle", "--stratify-clusters", "10"]
        self.assertIsNone(parse(*oracle).stratify_factors)
        chosen = parse(*oracle, "--stratify-factors", "brain_size", "cortical_thickness").stratify_factors
        self.assertEqual(chosen, ["brain_size", "cortical_thickness"])
        refused(self, "--stratify-factors", "brain_size")  # without --stratify-batches
        code = ["--stratify-batches", "code", "--stratify-clusters", "10", "--stratify-from", "runs/x"]
        refused(self, *code, "--stratify-factors", "brain_size")  # code mode clusters a code, not factors
        refused(self, *oracle, "--stratify-factors", "lesion_x")  # lesion latents are never clustered
        refused(self, *oracle, "--stratify-factors", "brain_sise")
        refused(self, *oracle, "--stratify-factors", "brain_size", "brain_size")
        refused(self, *oracle, "--stratify-factors")
        # The launcher writes a YAML list or a --set list back as one option and its items.
        options = {"stratify_batches": "oracle", "stratify_clusters": 10, "stratify_factors": chosen}
        self.assertEqual(
            encoder_args.parse_args([*SYNTHETIC, *encoder_args.cli_arguments(options)]).stratify_factors, chosen
        )


def dci_content_factor_names():
    """``eval.metrics.dci.CONTENT_FACTOR_NAMES`` without importing its module (which imports torch)."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "eval" / "metrics" / "dci.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONTENT_FACTOR_NAMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("CONTENT_FACTOR_NAMES not found in eval/metrics/dci.py")


if __name__ == "__main__":
    unittest.main()
