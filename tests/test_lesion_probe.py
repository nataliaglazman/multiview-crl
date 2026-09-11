"""Numerical recovery, leakage, and spatial extraction controls for lesion probes."""

import argparse
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eval.lesion_probe import block_gram, extract_features, fit_probes, make_dataset, split_subjects


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=2)
        self.threads.__enter__()

    def tearDown(self):
        self.threads.__exit__(None, None, None)

    def test_gram_matches_explicit_joint_features_and_train_only_scaling(self):
        x = np.random.default_rng(1).normal(size=(40, 8))
        train = np.arange(24)
        a, wa = block_gram(x, np.arange(3), train, chunk_size=2)
        b, wb = block_gram(x, np.arange(3, 8), train, chunk_size=2)
        z = StandardScaler().fit(x[train]).transform(x)
        np.testing.assert_allclose(a + b, z @ z.T, atol=1e-10)
        self.assertEqual(wa + wb, 8)
        x[24:] *= 1000
        changed, _ = block_gram(x, np.arange(8), train)
        np.testing.assert_allclose(changed[:24, :24], (a + b)[:24, :24], atol=1e-10)

    def test_joint_nonlinear_recovers_interaction_that_separate_and_linear_miss(self):
        rng = np.random.default_rng(4)
        x = np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (60, 1))
        y = x[:, 0] * x[:, 1]
        x += rng.normal(scale=0.04, size=x.shape)
        targets = np.column_stack([y, 2 * y, -y, y, 2 * y, -y])
        splits = split_subjects(len(x), 0)
        gram, width = block_gram(x, np.arange(2), splits[0])
        rows, _ = fit_probes(gram, width, targets, splits)

        def r2(rows, kind, condition="observed"):
            row = next(
                r for r in rows if r["probe"] == kind and r["condition"] == condition and r["target"] == "physical"
            )
            return np.mean(row["test"]["r2_xyz"])

        self.assertGreater(r2(rows, "rbf"), 0.95)
        self.assertLess(r2(rows, "ridge"), 0.1)
        self.assertLess(r2(rows, "rbf", "shuffled"), 0.2)
        for column in (0, 1):
            gram, width = block_gram(x, np.array([column]), splits[0])
            separate, _ = fit_probes(gram, width, targets, splits)
            self.assertLess(r2(separate, "rbf"), 0.2)

    def test_test_labels_cannot_change_model_selection_or_predictions(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 4))
        y = x @ rng.normal(size=(4, 6))
        splits = split_subjects(len(x), 2)
        gram, width = block_gram(x, np.arange(4), splits[0])
        rows, predictions = fit_probes(gram, width, y, splits)
        altered = y.copy()
        altered[splits[2]] += 1000
        other, other_predictions = fit_probes(gram, width, altered, splits)
        for key in predictions:
            np.testing.assert_allclose(predictions[key], other_predictions[key])
        before = [(r["alpha"], r["gamma"]) for r in rows if r["condition"] == "observed"]
        after = [(r["alpha"], r["gamma"]) for r in other if r["condition"] == "observed"]
        self.assertEqual(before, after)

    def test_subject_splits_are_disjoint(self):
        splits = split_subjects(100, 0)
        self.assertEqual([len(s) for s in splits], [60, 20, 20])
        self.assertEqual(len(np.unique(np.concatenate(splits))), 100)

    def test_dual_ridge_matches_explicit_spatial_regression(self):
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(9)
        x = rng.normal(size=(80, 20))
        y = np.dot(x, rng.normal(size=(20, 6)))
        train, val, test = splits = split_subjects(len(x), 0)
        gram, width = block_gram(x, np.arange(x.shape[1]), train)
        rows, predictions = fit_probes(gram, width, y, splits)
        xs = StandardScaler().fit(x[train])
        ys = StandardScaler().fit(y[train])
        for target, sl in (("physical", slice(0, 3)), ("latent", slice(3, 6))):
            row = next(
                r for r in rows if r["probe"] == "ridge" and r["condition"] == "observed" and r["target"] == target
            )
            model = Ridge(alpha=row["alpha"] * width, fit_intercept=False).fit(
                xs.transform(x[train]), ys.transform(y[train])[:, sl]
            )
            expected = model.predict(xs.transform(x[test])) * ys.scale_[sl] + ys.mean_[sl]
            np.testing.assert_allclose(predictions[("ridge", target)], expected, atol=1e-6)


class ExtractionTests(unittest.TestCase):
    def test_native_spatial_features_and_per_view_masks(self):
        import torch

        torch.set_num_threads(2)

        class ToyEncoder:
            def __call__(self, x, **kw):
                assert kw["return_recon"] is False
                assert kw["pool_only"] is False
                return (
                    None,
                    [],
                    [torch.cat([x, x * 2], dim=1)],
                    None,
                    [],
                    [],
                    {0: (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))},
                )

        args = argparse.Namespace(
            synthetic_res=32,
            synthetic_mode="pseudo_mri",
            synthetic_content_prior="uniform",
            synthetic_content_squash="none",
            synthetic_clean_content=True,
            synthetic_lesion_radius=0.14,
            synthetic_normalize="per_sample",
        )
        ds = make_dataset(args, 4, "iid", "test")
        with tempfile.TemporaryDirectory() as directory:
            arrays, masks, shape, y, ids = extract_features(ToyEncoder(), ds, "cpu", directory, 3)
            self.assertEqual(shape, (2, 32, 32, 32))
            np.testing.assert_array_equal(masks, [[True, False], [False, True]])
            np.testing.assert_array_equal(ids, np.arange(4))
            self.assertEqual(y.shape, (4, 6))
            for view in range(2):
                expected = ds[0]["image"][view].numpy().flatten()
                np.testing.assert_allclose(arrays[view][0, : 32**3], expected)
                np.testing.assert_allclose(arrays[view][0, 32**3 :], expected * 2)
            del arrays


if __name__ == "__main__":
    unittest.main()
