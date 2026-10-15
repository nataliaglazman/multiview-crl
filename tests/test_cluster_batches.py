"""--stratify-batches without torch: the one-cluster batch sampler, its diagnostics and its flag rules."""

import ast
import contextlib
import io
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
