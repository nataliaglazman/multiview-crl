"""Causal probe tests with planted signals and the actual synthetic MRI renderer."""

import ast
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eval import identifiability_report as report
from eval import parent_adjusted as adjusted
from eval.marginal_independence import MarginalShuffledDataset, permute_content_marginals


def synthetic_brain_class():
    # Execute the production class unchanged; omit unrelated ADNI/MONAI imports.
    source = Path(__file__).resolve().parents[1] / "data/datasets.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SyntheticBrainDataset"]
    namespace = {"MultiviewDataset": torch.utils.data.Dataset, "torch": torch, "np": np}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["SyntheticBrainDataset"]


class ParentAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(41)
        cls.parents = rng.uniform(-2, 2, (700, 1))
        cls.noise = rng.randn(700) * 0.4
        cls.y = cls.parents[:, 0] ** 2 + cls.noise
        cls.plans = adjusted.prepare_parent_folds(cls.y, cls.parents, seeds=(0,))

    def test_removes_nonlinear_parent_signal_but_keeps_own_variation(self):
        def score(X):
            features = adjusted.prepare_probe_folds(X, self.plans, 0)
            return adjusted.score_parent_folds(features, self.plans, n_null=1)

        parent = score(self.parents**2)
        own = score(self.noise[:, None])
        self.assertGreater(parent["full_r2_raw"], 0.8)
        self.assertLess(abs(parent["r2_raw"]), 0.08)
        self.assertGreater(own["r2_raw"], 0.8)
        self.assertGreater(parent["parent_r2"], 0.8)

    def test_parent_training_excludes_every_predicted_subject(self):
        fits = []

        class SpyParent:
            def fit(self, X, y):
                self.ids = set(X[:, 0])
                fits.append(self.ids)
                return self

            def predict(self, X):
                # Checks both outer test predictions and inner training predictions.
                assert not self.ids.intersection(X[:, 0])
                return np.zeros(len(X))

        ids = np.arange(60, dtype=float)
        with patch.object(adjusted, "_parent_model", side_effect=lambda seed: SpyParent()):
            plans = adjusted.prepare_parent_folds(ids, ids[:, None], seeds=(0,))
        self.assertEqual(len(fits), 20)
        for fi, fold in enumerate(plans[0][1]):
            for fit in fits[fi * 4 : (fi + 1) * 4]:
                self.assertTrue(fit.issubset(set(fold["train"])))
                self.assertFalse(fit.intersection(fold["test"]))

    def test_pca_and_scaling_fit_training_rows_only(self):
        rng = np.random.RandomState(11)
        X = rng.randn(60, 10)
        X[0] += 100  # A held-out outlier must not influence that fold's transform.
        plans = adjusted.prepare_parent_folds(X[:, 0], np.empty((60, 0)), seeds=(0,))
        features = adjusted.prepare_probe_folds(X, plans, 3)
        for fold, (train, test) in zip(plans[0][1], features[0]):
            pca = PCA(n_components=3, random_state=0).fit(X[fold["train"]])
            scaler = StandardScaler().fit(pca.transform(X[fold["train"]]))
            np.testing.assert_allclose(train, scaler.transform(pca.transform(X[fold["train"]])))
            np.testing.assert_allclose(test, scaler.transform(pca.transform(X[fold["test"]])))

    def test_parentless_targets_unchanged_and_nulls_stay_within_splits(self):
        y = np.arange(60, dtype=float)
        plans = adjusted.prepare_parent_folds(y, np.empty((60, 0)), seeds=(0,))
        fitted_targets = []

        class SpyProbe:
            def fit(self, X, target):
                fitted_targets.append(target.copy())
                return self

            def predict(self, X):
                return np.zeros(len(X))

        features = adjusted.prepare_probe_folds(y[:, None], plans, 0)
        with patch.object(adjusted, "_make_regressor", side_effect=lambda kind, seed: SpyProbe()):
            score = adjusted.score_parent_folds(features, plans, n_null=2)
        self.assertEqual(score["full_r2_raw"], score["r2_raw"])
        for fi, fold in enumerate(plans[0][1]):
            np.testing.assert_array_equal(fold["raw_test"], fold["residual_test"])
            np.testing.assert_array_equal(fold["raw_train"], fold["residual_train"])
            for target in fitted_targets[fi * 4 : (fi + 1) * 4]:
                np.testing.assert_array_equal(np.sort(target), y[fold["train"]])

    def test_blocks_poolings_parallelism_and_twins_share_parent_fits(self):
        rng = np.random.RandomState(5)
        gt = rng.randn(80, 2)
        gt[:, 1] += gt[:, 0] ** 2
        reprs = {
            "gap": {0: (gt[:, :1], gt[:, 1:], None, None, {})},
            "patch": {0: (gt[:, 1:], gt[:, :1], None, None, {})},
        }
        args = (reprs, 0, gt, ["brain_size", "ventricle_size"], np.array([[0, 1], [0, 0]]), (0,), 1)
        cache = {}
        with patch.object(adjusted, "prepare_parent_folds", wraps=adjusted.prepare_parent_folds) as prepare:
            first = report.nonlinear_parent_scores(*args, probe_dim=0, parent_cache=cache)
            self.assertEqual(prepare.call_count, 2)
            second = report.nonlinear_parent_scores(*args, probe_dim=0, n_jobs=2, parent_cache=cache)
            style = report.nonlinear_parent_scores(*args, probe_dim=0, block=report._STYLE, parent_cache=cache)
            self.assertEqual(prepare.call_count, 2)
        for factor in first:
            self.assertEqual(first[factor]["pooling"], "gap")
            for pool in first[factor]["by_pooling"]:
                for metric, value in first[factor]["by_pooling"][pool].items():
                    np.testing.assert_allclose(value, second[factor]["by_pooling"][pool][metric])
        self.assertGreater(style["ventricle_size"]["full_r2_raw"], 0.99)
        self.assertLess(first["ventricle_size"]["full_r2_raw"], 0.3)
        empty = {"gap": {0: (gt, None, None, None, {})}}
        self.assertEqual(report.nonlinear_parent_scores(empty, *args[1:], block=report._STYLE, parent_cache=cache), {})


class MarginalIndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Brain = synthetic_brain_class()

    def test_exact_marginals_reproducibility_and_reduced_dependence(self):
        rng = np.random.RandomState(1)
        parent = rng.randn(2000)
        original = np.column_stack([parent, 2 * parent, -parent, np.ones(2000)])
        shuffled, permutations = permute_content_marginals(original, seed=2)
        np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(original, axis=0))
        np.testing.assert_array_equal(shuffled, original[permutations, np.arange(4)])
        np.testing.assert_array_equal(shuffled, permute_content_marginals(original, seed=2)[0])
        self.assertFalse(np.array_equal(shuffled, permute_content_marginals(original, seed=3)[0]))
        corr = np.corrcoef(shuffled[:, :3], rowvar=False)
        self.assertLess(np.abs(corr[np.triu_indices(3, 1)]).max(), 0.1)

    def test_rerenders_shuffled_labels_keeps_nuisances_seeds_and_normalization(self):
        for prior in ("normal", "uniform"):
            with self.subTest(prior=prior):
                source = self.Brain(
                    mode="test",
                    spatial_size=(12, 12, 12),
                    synthetic_num_samples=8,
                    synthetic_causal=True,
                    synthetic_content_prior=prior,
                    synthetic_lesion_mode="field",
                    synthetic_normalize="fixed_reference",
                )
                # Reference constants must stay attached to the original population.
                source._compute_fixed_reference()
                constants = source._fixed_mean, source._fixed_scale
                original = [source._inner[i][2] for i in range(len(source))]
                shuffled = MarginalShuffledDataset(source, seed=4)
                self.assertIsNone(report._causal_adjacency(shuffled))
                self.assertIsNotNone(report._causal_adjacency(source))
                self.assertTrue(shuffled.evaluation_distribution["source_causal"])
                self.assertIsNone(source._cache)
                rows = []
                changed_image = False
                for i in range(len(shuffled)):
                    item = shuffled[i]
                    self.assertIs(item, shuffled[i])
                    z = item["gt_latents"]
                    self.assertNotIn("causal_adj", z)
                    self.assertNotIn("brain_mask", z)
                    for key in ("z_style_v1", "z_style_v2", "z_deformation", "z_fissure", "z_lesion"):
                        torch.testing.assert_close(z[key], original[i][key], rtol=0, atol=0)
                    raw1, raw2, mask = source._inner.render_pseudo_mri(
                        z["z_content"],
                        z["z_deformation"],
                        z["z_fissure"],
                        z["z_style_v1"],
                        z["z_style_v2"],
                        source._inner.sample_seed_for(i),
                        z_lesion=z["z_lesion"],
                    )
                    expected = source.normalize_views(raw1, raw2, mask, mask)
                    for actual, want in zip(item["image"], expected):
                        torch.testing.assert_close(actual, want, rtol=0, atol=0)
                    torch.testing.assert_close(item["mask"][0], mask, rtol=0, atol=0)
                    changed_image |= not torch.equal(item["image"][0], source[i]["image"][0])
                    rows.append(z["z_content"])
                self.assertTrue(changed_image)
                self.assertEqual(constants, (source._fixed_mean, source._fixed_scale))
                np.testing.assert_array_equal(
                    np.sort(torch.stack(rows).numpy(), axis=0),
                    np.sort(torch.stack([z["z_content"] for z in original]).numpy(), axis=0),
                )
                batch = next(iter(torch.utils.data.DataLoader(shuffled, batch_size=4)))
                self.assertEqual(batch["gt_latents"]["z_content"].shape, (4, 9))
                self.assertEqual(batch["image"][0].shape, (4, 1, 12, 12, 12))

    def test_rejects_invalid_input(self):
        for content in (np.ones((1, 3)), np.ones((3, 0)), np.array([[np.nan], [1]])):
            with self.assertRaises(ValueError):
                permute_content_marginals(content)
        source = self.Brain(synthetic_num_samples=2, synthetic_mode="primitives", synthetic_normalize="fixed_reference")
        with self.assertRaisesRegex(ValueError, "pseudo_mri"):
            MarginalShuffledDataset(source)


class ReportFlagTests(unittest.TestCase):
    def test_score_run_keeps_headlines_and_dci_nulls_when_adjustment_changes(self):
        from eval import run_dci_compare as compare
        from eval import run_dci_synthetic as synthetic

        rng = np.random.RandomState(13)
        gt, style_gt = rng.randn(80, 2), rng.randn(80, 1)
        gt[:, 1] += gt[:, 0] ** 2
        content = np.column_stack([gt, rng.randn(80)])
        style = np.column_stack([style_gt, gt[:, 1]])
        info = dict(content_names=["brain_size", "ventricle_size"], style_names=["bias"])
        levels = {0: (content, style, None, None, info)}
        extractor = types.SimpleNamespace(
            _extract_synthetic_representations=lambda *a, **kw: (levels, gt, style_gt, style_gt)
        )
        dataset = types.SimpleNamespace(scm={"adj": np.array([[0, 1], [0, 0]])})
        common = dict(
            run_dir="unused",
            dataset=dataset,
            poolings=[("gap", "gap")],
            level=0,
            seeds=(0,),
            n_null=1,
            batch_size=8,
            num_workers=0,
            device="cpu",
            probe_dim=1,
            with_dci=True,
            causal="match",
        )

        def fake_dci(*args, **kwargs):
            # The DCI scorer receives its permutation RNG as the eighth argument.
            return {"dci_d": float(args[7].rand())}

        with (
            patch.dict("sys.modules", {"eval.dci": extractor}),
            patch.object(compare, "_resolve_checkpoint", return_value="unused"),
            patch.object(compare, "_score_dci", side_effect=fake_dci),
            patch.object(synthetic, "load_model_from_run_dir", return_value=(object(), None, "cpu")),
        ):
            legacy = report.score_run(**common)
            nonlinear = report.score_run(**common, parent_adjustment="nonlinear")
            self.assertIn("partial", legacy)
            self.assertNotIn("partial", nonlinear)
            self.assertEqual(set(nonlinear["parent_adjusted"]), {"content", "style"})
            self.assertEqual(nonlinear["parent_adjustment_config"]["preprocessing"], "outer_train_only")
            for field in ("per_factor", "mcc", "leakage", "dci"):
                self.assertEqual(
                    json.dumps(legacy[field], sort_keys=True), json.dumps(nonlinear[field], sort_keys=True)
                )
            # No graph must be reported as unavailable, not silently treated as iid.
            common["dataset"] = types.SimpleNamespace()
            missing = report.score_run(**common, parent_adjustment="nonlinear")
            self.assertIn("unavailable", missing["parent_adjustment_status"])
            # Distribution metadata must survive scoring for JSON replay.
            common["causal"] = "shuffled"
            common["dataset"].evaluation_distribution = {"mode": "shuffled", "shuffle_seed": 4}
            shuffled = report.score_run(**common)
            self.assertEqual(shuffled["evaluation_distribution"], common["dataset"].evaluation_distribution)
            self.assertNotIn("partial", shuffled)

    def invoke(self, flags, dataset=None):
        from eval import run_dci_synthetic as synthetic

        with (
            patch("sys.argv", ["report", "--run-dir", "unused", *flags]),
            patch.object(synthetic, "load_run_args", return_value=object()),
            patch.object(synthetic, "build_synthetic_test_set", return_value=dataset) as build,
            patch.object(report, "score_run", return_value={}) as score,
            patch.object(report, "print_report"),
        ):
            report.main()
        return build, score

    def test_defaults_remain_matched_and_legacy(self):
        build, score = self.invoke(["--no-floor"])
        self.assertTrue(build.call_args.kwargs["causal"])
        self.assertNotIn("cache", build.call_args.kwargs)
        self.assertEqual(score.call_args.kwargs["parent_adjustment"], "legacy")
        self.assertIsNone(score.call_args.kwargs["parent_cache"])

    def test_nonlinear_cache_shared_with_floor_runs(self):
        _, score = self.invoke(["--parent-adjustment", "nonlinear", "--floor-seeds", "2"])
        self.assertEqual(score.call_count, 3)
        caches = [call.kwargs["parent_cache"] for call in score.call_args_list]
        self.assertTrue(all(cache is caches[0] for cache in caches))
        self.assertEqual(score.call_args.kwargs["parent_adjustment"], "nonlinear")

    def test_shuffled_uses_matched_source_without_source_image_cache(self):
        with patch("eval.marginal_independence.MarginalShuffledDataset", return_value="shuffled") as wrapper:
            build, score = self.invoke(["--causal", "shuffled", "--shuffle-seed", "12", "--no-floor"], "source")
        self.assertEqual(build.call_args.kwargs, {"causal": True, "cache": False})
        wrapper.assert_called_once_with("source", seed=12)
        self.assertEqual(score.call_args.kwargs["dataset"], "shuffled")
        self.assertEqual(score.call_args.kwargs["causal"], "shuffled")

    def test_rejects_adjustment_of_shuffled_distribution_before_loading(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(["--causal", "shuffled", "--parent-adjustment", "nonlinear"])

    def test_json_replay_retains_nonlinear_and_shuffle_metadata_without_fits(self):
        score = dict(full_r2_raw=0.9, r2_raw=0.7, r2_null=-0.1, r2=0.8, r2_std=0.02, parent_r2=0.6)
        factor = dict(pooling="gap", by_pooling={"gap": score}, **score)
        result = dict(
            name="test",
            n_samples=100,
            level=0,
            poolings="gap",
            probe_dim=0,
            per_factor={},
            mcc={},
            mcc_per_factor_pooling="gap",
            causal="match",
            parent_adjustment="nonlinear",
            parent_adjusted={"content": {"child": factor}},
            n_parents={"child": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "in.json", Path(directory) / "out.json"
            for shuffled in (False, True):
                if shuffled:
                    result.pop("parent_adjusted")
                    result.pop("parent_adjustment")
                    result.update(
                        causal="shuffled",
                        evaluation_distribution=dict(
                            mode="shuffled",
                            shuffle_seed=0,
                            mean_abs_correlation_before=0.8,
                            mean_abs_correlation_after=0.02,
                            permutation_sha256="example",
                        ),
                    )
                source.write_text(json.dumps({"run": result, "floor": None}))
                output = io.StringIO()
                with (
                    patch("sys.argv", ["report", "--from-json", str(source), "--out", str(dest)]),
                    patch.object(report, "score_run", side_effect=AssertionError("must not fit")),
                    patch.object(adjusted, "prepare_parent_folds", side_effect=AssertionError("must not fit")),
                    contextlib.redirect_stdout(output),
                ):
                    report.main()
                self.assertEqual(json.loads(dest.read_text())["run"], result)
                self.assertIn("Marginals are exact" if shuffled else "NONLINEAR PARENT-ADJUSTED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
