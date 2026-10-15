"""Planted causes of modality leakage, the loss mirror, and extraction from a saved ADNI checkpoint."""

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval.adni import encoder_modality_gap as diagnostic
from eval.adni import encoder_modality_gap_causes as causes

D = 20


def basis():
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((D, D)))
    return q


def subjects_codes(n, rng, low_variance=3):
    """Subject variation with the measured total spread. Along basis directions 0-2 subjects vary 5x
    less than typical, along 16-18 3x more (19 carries the shared mean)."""
    q = basis()
    variance = np.ones(D)
    variance[:low_variance] = 0.2
    variance[16:19] = 3.0
    variance *= 0.7955 / variance.sum()
    return rng.standard_normal((n, D)) * np.sqrt(variance) @ q.T, q


def pair(n=289, seed=1, offset=0.38, offset_dirs=slice(0, 3), flair_gain=None):
    """Raw T1/FLAIR codes roughly matching the measured geometry; one cause planted at a time."""
    rng = np.random.default_rng(seed)
    s, q = subjects_codes(n, rng)
    mean = 0.25 * q[:, -1]
    direction = q[:, offset_dirs].sum(1)
    g = offset * direction / np.linalg.norm(direction)
    flair_s = s
    if flair_gain is not None:  # FLAIR reads three subject directions at this gain
        p = q[:, 3:6]
        flair_s = s + (flair_gain - 1) * (s @ p) @ p.T
    noise = np.sqrt(0.106 / D)
    a = mean + s + g / 2 + noise * rng.standard_normal((n, D))
    b = mean + flair_s - g / 2 + noise * rng.standard_normal((n, D))
    return a, b, np.array([f"{i % 30 + 2:03d}_S_{4000 + i:04d}" for i in range(n)])


def low_rank_pair(outside_share, n=289, seed=4, flair_private=0.0):
    """Codes that use 7 of 20 directions (like the measured run); ``outside_share`` of the offset's
    squared length lies in the unused 13. ``flair_private`` adds FLAIR-only variation along one unused direction."""
    rng = np.random.default_rng(seed)
    q = basis()
    variance = np.r_[np.geomspace(1.0, 0.25, 7), np.full(13, 0.004)]
    variance *= 0.79 / variance.sum()
    inside = q[:, :7] @ rng.standard_normal(7)
    outside = q[:, 7:19] @ rng.standard_normal(12)
    g = 0.38 * (
        np.sqrt(1 - outside_share) * inside / np.linalg.norm(inside)
        + np.sqrt(outside_share) * outside / np.linalg.norm(outside)
    )
    s = rng.standard_normal((n, D)) * np.sqrt(variance) @ q.T
    noise = 0.07 * np.sqrt(variance / variance.mean())
    a = 0.25 * q[:, -1] + s + g / 2 + rng.standard_normal((n, D)) * noise @ q.T
    b = 0.25 * q[:, -1] + s - g / 2 + rng.standard_normal((n, D)) * noise @ q.T
    b += flair_private * rng.standard_normal((n, 1)) * q[:, 12]
    return a, b, np.arange(n).astype(str)


def args(**changes):
    values = dict(
        folds=3,
        seed=1729,
        permutations=200,
        probe_permutations=0,
        matched_draws=2,
        loss_batches=40,
        loss_batch_sizes=None,
        tau=None,
        covariates=None,
        subject_column="Subject",
        min_group=5,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def rng():
    return np.random.default_rng(7)


class OffsetTests(unittest.TestCase):
    def test_low_variance_shift_beats_random_directions_and_reads_sideways(self):
        a, b, _ = pair()
        out = causes.offset_geometry(causes.normalize_rows(a), causes.normalize_rows(b), rng())
        self.assertLess(out["radial_share"], 0.1)
        self.assertGreater(out["lda_accuracy"], out["random_direction_accuracy"]["p95"])
        self.assertGreater(out["low_variance_half_share"], out["random_low_variance_half_share"]["p95"])
        self.assertGreater(out["random_direction_percentile"], 0.95)

    def test_high_variance_shift_does_not(self):
        a, b, _ = pair(offset_dirs=slice(16, 19))
        out = causes.offset_geometry(causes.normalize_rows(a), causes.normalize_rows(b), rng())
        self.assertLess(out["lda_accuracy"], out["random_direction_accuracy"]["median"])
        self.assertLess(out["low_variance_half_share"], 0.5)

    def test_concentration_difference_reads_radial(self):
        r = np.random.default_rng(3)
        s, q = subjects_codes(400, r, low_variance=0)
        mean = 0.6 * q[:, -1]
        a = mean + s
        b = mean + 0.4 * s  # FLAIR codes cluster closer to the shared mean
        out = causes.offset_geometry(causes.normalize_rows(a), causes.normalize_rows(b), rng())
        self.assertGreater(out["radial_share"], 0.9)
        self.assertGreater(out["mean_length_flair"], out["mean_length_t1"])

    def test_low_rank_codes_keep_the_linear_tag_in_unused_directions(self):
        a, b, subjects = low_rank_pair(0.1)
        out = causes.offset_geometry(causes.normalize_rows(a), causes.normalize_rows(b), rng())
        split = out["used_subspace"]
        self.assertEqual(split["directions"], 7)
        self.assertTrue(out["random_direction_saturated"])
        self.assertGreater(split["offset_share_inside"], 0.8)
        self.assertLess(split["accuracy_inside"], 0.8)
        self.assertGreater(split["accuracy_outside"], 0.9)
        folds = list(causes.subject_folds(subjects, 3, 1729))
        probes = causes.subspace_probes(a, b, folds)
        self.assertEqual(probes["used_directions_per_fold"], [7, 7, 7])
        self.assertLess(probes["used_linear"], 0.8)
        self.assertGreater(probes["unused_linear"], 0.9)
        a, b, subjects = low_rank_pair(0.0)
        self.assertLess(causes.subspace_probes(a, b, folds)["unused_linear"], 0.6)

    def test_swap_null_bounds_the_centroid_distance_of_identical_modalities(self):
        a, b, _ = pair(offset=0.0)
        out = causes.check_offset(causes.normalize_rows(a), causes.normalize_rows(b), rng(), 200)
        self.assertLess(out["centroid_distance"], 2 * out["centroid_distance_null_p95"])
        a, b, _ = pair()
        out = causes.check_offset(causes.normalize_rows(a), causes.normalize_rows(b), rng(), 200)
        self.assertGreater(out["centroid_distance"], 3 * out["centroid_distance_null_p95"])


def shape(a, b, permutations=300):
    """check_shape on the rows analyse_split gives it: raw offset removed, then L2-normalized."""
    return causes.check_shape(*(causes.normalize_rows(x) for x in causes.remove_raw_offset(a, b)), rng(), permutations)


class ShapeTests(unittest.TestCase):
    def test_flags_directions_read_at_a_different_gain(self):
        out = shape(*pair(offset=0.38, flair_gain=0.4)[:2])
        self.assertGreaterEqual(out["directions_flair_lower"], 3)
        self.assertLessEqual(out["directions_flair_higher"], 1)
        self.assertLess(out["shape_distance_p"], 0.01)

    def test_an_offset_alone_is_not_a_shape_difference(self):
        a, b, _ = pair(offset=0.38)
        out = shape(a, b)
        self.assertGreater(out["shape_distance_p"], 0.05)
        self.assertLessEqual(out["directions_flair_lower"] + out["directions_flair_higher"], 2)
        # Normalizing first would compress each cloud along its own mean direction.
        distorted = causes.check_shape(causes.normalize_rows(a), causes.normalize_rows(b), rng(), 300)
        self.assertLess(distorted["shape_distance_p"], out["shape_distance_p"])

    def test_variance_shares_locate_a_private_direction(self):
        out = shape(*low_rank_pair(0.0, flair_private=0.06)[:2])
        self.assertGreaterEqual(out["directions_flair_higher"], 1)
        top_t1, top_flair = out["t1_variance_share"][-1], out["flair_variance_share"][-1]
        self.assertGreater(top_flair, 3 * top_t1)
        self.assertLess(top_flair, 0.05)  # a small share of FLAIR's variance, but one T1 lacks

    def test_ratios_ignore_any_linear_map_applied_to_both(self):
        a, b, _ = pair(offset=0.0, flair_gain=0.5)
        m = np.random.default_rng(5).standard_normal((D, D))
        # Up to the relative ridge that keeps dead channels invertible.
        np.testing.assert_allclose(causes.variance_ratios(a, b), causes.variance_ratios(a @ m, b @ m), rtol=1e-4)


class LadderTests(unittest.TestCase):
    def folds(self, subjects):
        return list(causes.subject_folds(subjects, 3, 1729))

    def test_first_two_rungs_reproduce_the_diagnostic(self):
        a, b, subjects = pair(flair_gain=0.5)
        accuracy, margin = causes.probe_ladder(a, b, self.folds(subjects), decisions=True)
        report, _ = diagnostic.diagnose(a, b, subjects)
        for key, value in report["probes"].items():
            self.assertEqual(accuracy[key], value["accuracy"])
        self.assertTrue(np.isfinite(margin).all())

    def test_whitening_removes_a_covariance_difference(self):
        # A large constant channel keeps every norm close to 10, so L2 normalization is nearly
        # linear and the planted difference stays second order.
        r = np.random.default_rng(11)
        s = r.standard_normal((300, D))
        flair = s.copy()
        flair[:, :3] *= 0.3
        noise = 0.3
        a = np.column_stack([np.full(300, 10.0), s + noise * r.standard_normal(s.shape)])
        b = np.column_stack([np.full(300, 10.0), flair + noise * r.standard_normal(s.shape)])
        subjects = np.arange(300).astype(str)
        accuracy, _ = causes.probe_ladder(a, b, self.folds(subjects))
        self.assertGreater(accuracy["mean_centered_rbf"], 0.65)
        self.assertLess(accuracy["whitened_rbf"], 0.56)

    def test_raw_offset_removal_undoes_what_normalization_distorts(self):
        # A constant raw offset on codes whose norms vary widely: after L2 normalization the offset
        # shrinks with each subject's norm, so removing one normalized-space mean leaves a residual.
        # That residual is not second order, so whitening keeps it too.
        r = np.random.default_rng(13)
        n, d = 400, 5
        s = np.exp(r.normal(0, 1.0, size=(n, 1))) * r.standard_normal((n, d))
        g = np.zeros(d)
        g[0] = 2.0
        a = s + g / 2 + 0.05 * r.standard_normal((n, d))
        b = s - g / 2 + 0.05 * r.standard_normal((n, d))
        subjects = np.arange(n).astype(str)
        accuracy, _ = causes.probe_ladder(a, b, self.folds(subjects))
        self.assertGreater(accuracy["mean_centered_rbf"], 0.75)
        self.assertGreater(accuracy["whitened_rbf"], 0.75)
        self.assertLess(accuracy["raw_offset_removed_rbf"], 0.56)

    def test_swap_null_and_matched_draws(self):
        a, b, subjects = pair(n=120)
        out, margin = causes.check_ladder(a, b, subjects, 3, 1729, rng(), 5)
        self.assertEqual(len(margin), 240)
        self.assertGreater(out["accuracy"]["normalized_linear"], max(out["null_p95"]["normalized_linear"], 0.8))
        self.assertLessEqual(out["p_value"]["normalized_linear"], 1 / 6)
        matched = causes.check_matched(a, b, subjects, 60, 2, 3, 1729, rng())
        self.assertEqual(matched["subjects"], 60)
        self.assertEqual(
            set(matched["accuracy_mean"]), set(out["accuracy"]) | {"subspace_used_linear", "subspace_unused_linear"}
        )
        self.assertEqual(set(out["subspace"]), {"used_linear", "unused_linear", "used_directions_per_fold"})


class LossTests(unittest.TestCase):
    def test_mirror_matches_training_infonce(self):
        import torch

        if importlib.util.find_spec("lpips") is None:
            # training.losses imports LPIPS at module scope; the InfoNCE path never builds it.
            sys.modules["lpips"] = types.ModuleType("lpips")
            sys.modules["lpips"].LPIPS = None
        from training import losses
        from training import main_conv_synthetic as trainer

        a, b, _ = pair(n=16)
        hz = torch.tensor(np.stack([a, b]), dtype=torch.float64)
        sim, criterion = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
        for cross in (True, False):
            expected = losses.infonce_loss(
                hz,
                sim,
                criterion,
                tau=0.1,
                estimated_content_indices=[list(range(D))],
                subsets=[(0, 1)],
                cross_view_negs_only=cross,
            )
            loss, top1 = causes.infonce(a, b, 0.1, cross)
            self.assertAlmostEqual(loss, float(expected), places=9)
            self.assertAlmostEqual(top1, expected._contrastive_diag["top1_acc"], places=12)
        options = SimpleNamespace(
            content_channels=D,
            standardize_code=True,
            contrastive_loss_type="infonce",
            tau=0.1,
            cross_view_negs_only=True,
        )
        model = SimpleNamespace(project=lambda x: x)
        expected = trainer.contrastive_loss(hz.reshape(32, D), model, options, sim, criterion)
        self.assertAlmostEqual(causes.infonce(a, b, 0.1, True, standardize=True)[0], float(expected), places=9)

    def settings(self, **changes):
        return {**causes.RECIPE, **changes}

    def test_prices_a_planted_offset_and_nothing_else(self):
        a, b, _ = pair()
        out = causes.check_loss(a, b, self.settings(), rng(), 50, [32])["by_batch_size"]["32"]
        self.assertGreater(out["offset_cost"], 0.01)
        self.assertGreater(out["misalignment_cost"], out["offset_cost"])
        a, b, _ = pair(offset=0.0)
        out = causes.check_loss(a, b, self.settings(), rng(), 50, [32])["by_batch_size"]["32"]
        self.assertLess(abs(out["offset_cost"]), 0.1 * out["misalignment_cost"])

    def test_a_tag_in_unused_directions_is_nearly_free(self):
        a, b, _ = low_rank_pair(0.1)
        out = causes.check_loss(a, b, self.settings(), rng(), 50, [32])["by_batch_size"]["32"]
        # The cosine ignores row length, so the normalized baseline is the as-is loss.
        self.assertAlmostEqual(out["normalized"]["loss"], out["as_is"]["loss"], places=9)
        self.assertGreater(out["used_part_cost"], 20 * abs(out["unused_part_cost"]))

    def test_standardized_code_and_other_losses_are_blind_or_skipped(self):
        a, b, _ = pair()
        out = causes.check_loss(a, b, self.settings(standardize_code=True), rng(), 20, [32])
        self.assertAlmostEqual(out["by_batch_size"]["32"]["offset_cost"], 0.0, places=9)
        self.assertIn(
            "skipped", causes.check_loss(a, b, self.settings(contrastive_loss_type="barlow_twins"), rng(), 2, [8])
        )
        self.assertIn("skipped", causes.check_loss(a, b, self.settings(contrastive_proj_dim=8), rng(), 2, [8]))
        out = causes.check_loss(a[:10], b[:10], self.settings(), rng(), 2, [32])
        self.assertIn("skipped", out["by_batch_size"]["32"])


class SubjectTests(unittest.TestCase):
    def test_finds_a_planted_numeric_and_categorical_covariate(self):
        r = np.random.default_rng(2)
        n = 200
        a, b, subjects = pair(n=n, offset=0.0)
        ratio = 1 + 0.05 * r.standard_normal(n)
        protocol = np.where(r.random(n) < 0.5, "2D", "3D")
        direction = basis()[:, 0]
        a = a + 2.0 * (ratio - 1)[:, None] * direction + np.where(protocol == "2D", 0.1, 0.0)[:, None] * direction
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "covariates.csv"
            with path.open("w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["Subject", "FLAIR_protocol", "WMH", "Group"])
                for i, subject in enumerate(subjects):
                    wmh = "" if i == 0 else f"{r.random():.4f}"
                    writer.writerow([subject, protocol[i], wmh, ["CN", "MCI", "AD"][i % 3]])
            columns, matched = causes.read_covariates(path, "Subject", subjects)
        self.assertEqual(matched, n)
        self.assertEqual(columns["WMH"][0], "numeric")
        self.assertTrue(np.isnan(columns["WMH"][1][0]))
        self.assertEqual(columns["FLAIR_protocol"][0], "categorical")
        na, nb = causes.normalize_rows(a), causes.normalize_rows(b)
        measures = causes.subject_measures(na, nb, np.zeros(2 * n))
        covariates = {**columns, **causes.descriptor_covariates({"mask_volume_ratio": ratio})}
        covariates["site"] = ("categorical", causes.site_codes(subjects))
        tests = causes.check_subjects(measures, covariates, 5)
        found = {(t["measure"], t["covariate"]) for t in tests if t["q"] < 0.01}
        self.assertIn(("offset_along_gap", "mask_volume_ratio"), found)
        self.assertIn(("offset_along_gap", "FLAIR_protocol"), found)
        self.assertNotIn(("offset_along_gap", "WMH"), found)
        self.assertNotIn(("offset_along_gap", "site"), found)

    def test_site_codes_need_adni_ids_and_q_values_are_monotone(self):
        self.assertIsNone(causes.site_codes(np.array(["a", "b", "002_S_0001"])))
        self.assertEqual(causes.site_codes(np.array(["002_S_0001", "x"])).tolist(), ["002", None])
        p = np.array([0.01, 0.04, 0.03, 0.5])
        q = causes.benjamini_hochberg(p)
        self.assertTrue(np.all(q >= p))
        self.assertTrue(np.all(np.diff(q[np.argsort(p)]) >= 0))


class FeatureCliTests(unittest.TestCase):
    def write(self, path, a, b, subjects, metadata, **descriptors):
        np.savez_compressed(
            path,
            content_t1=a,
            content_flair=b,
            subjects=subjects,
            metadata_json=np.array(json.dumps(metadata)),
            **{f"desc_{k}": v for k, v in descriptors.items()},
        )

    def test_two_splits_round_trip_and_the_diagnostic_format(self):
        quick = ["--permutations", "50", "--probe-permutations", "3", "--loss-batches", "10", "--matched-draws", "1"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {"resolved_settings": {"tau": 0.2, "batch_size": 16}}
            val = pair(n=90, seed=1)
            train = pair(n=150, seed=2, offset=0.1)
            ratio = np.linspace(0.9, 1.1, 90)
            self.write(root / "val.npz", *val, {**settings, "split": "val"}, mask_volume_ratio=ratio)
            self.write(root / "train.npz", *train, {**settings, "split": "train"})
            argv = ["--features", f"val={root / 'val.npz'}", str(root / "train.npz"), "--out-dir", str(root / "out")]
            with contextlib.redirect_stdout(io.StringIO()):
                first = causes.main(argv + quick)
            self.assertEqual(list(first["splits"]), ["val", "train"])
            self.assertEqual(first["settings"]["tau"], 0.2)
            self.assertIn("cross_view_negs_only", first["settings"]["assumed_from_recipe"])
            self.assertEqual(first["splits"]["val"]["ladder"]["permutations"], 3)
            self.assertEqual(first["splits"]["train"]["ladder"]["permutations"], 0)
            self.assertEqual(first["splits"]["train"]["matched"]["subjects"], 90)
            self.assertIn("16", first["splits"]["val"]["loss"]["by_batch_size"])
            self.assertAlmostEqual(first["splits"]["val"]["descriptors"]["mask_volume_ratio"]["median"], 1.0, places=6)
            self.assertEqual(first["splits"]["train"]["descriptors"], {})
            self.assertIsNotNone(first["comparison"])
            self.assertIn("probe subspace_unused_linear", first["comparison"])
            saved = json.loads((root / "out/report.json").read_text())
            self.assertEqual(saved["splits"]["val"]["ladder"], first["splits"]["val"]["ladder"])
            with (root / "out/subjects_val.csv").open() as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 90)
            self.assertIn("mask_volume_ratio", rows[0])
            replay = ["--features", f"val={root / 'out/features_val.npz'}", f"train={root / 'out/features_train.npz'}"]
            with contextlib.redirect_stdout(io.StringIO()):
                second = causes.main(replay + ["--out-dir", str(root / "again")] + quick)
            self.assertEqual(first["splits"], second["splits"])
            with self.assertRaises(FileExistsError):
                causes.main(argv + quick)
            # encoder_modality_gap's features.npz: no split recorded, no descriptors.
            self.write(root / "features.npz", *val, {"checkpoint_sha256": "x"})
            with contextlib.redirect_stdout(io.StringIO()):
                single = causes.main(["--features", str(root / "features.npz"), "--out-dir", str(root / "old")] + quick)
            self.assertEqual(list(single["splits"]), ["val"])
            self.assertIsNone(single["comparison"])

    def test_invalid_features_fail_instead_of_reporting(self):
        a, b, subjects = pair(n=12)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            bad = a.copy()
            bad[0, 0] = np.nan
            causes.analyse_split({"content_t1": bad, "content_flair": b, "subjects": subjects}, causes.RECIPE, args())
        with self.assertRaises(ValueError):
            causes.analyse_split(
                {"content_t1": a, "content_flair": b[:, :3], "subjects": subjects}, causes.RECIPE, args()
            )


class CheckpointExtractionTests(unittest.TestCase):
    def test_encodes_unaugmented_train_and_val_with_descriptors(self):
        import hashlib

        import torch

        if importlib.util.find_spec("lpips") is None:
            sys.modules["lpips"] = types.ModuleType("lpips")
            sys.modules["lpips"].LPIPS = None
        from test_encoder_adni import SHAPE, TINY, fake_adni

        from eval.protocol.score_checkpoint import build_model
        from training import main_conv_synthetic as trainer

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = fake_adni(root)
            options = trainer.parse_args(
                ["--dataset-name", "ADNI_stripped_masks", "--dataroot", str(root), "--labels-path", str(labels)]
                + ["--spatial-size", *map(str, SHAPE), "--image-spacing", "1", "--val-frac", "0.4", "--test-frac"]
                + ["0.1", "--no-cache", *TINY, "--conv-readout", "mlp", "--norm-type", "layer"]
            )
            cfg = vars(options)
            run = root / "run"
            run.mkdir()
            (run / "settings.json").write_text(json.dumps(cfg))
            train = trainer.make_real_dataset(options, "train")
            val = trainer.make_real_dataset(options, "val")
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.describe_real_split(options, train, val, run)
            torch.save(build_model(cfg, "cpu").state_dict(), run / "model_best.pt")
            before = hashlib.sha256((run / "model_best.pt").read_bytes()).hexdigest()

            augmented = [train[i]["image"][0] for i in range(6)] + [train[i]["image"][0] for i in range(6)]
            self.assertTrue(any(not torch.equal(x, y) for x, y in zip(augmented[:6], augmented[6:])))
            causes.disable_augmentation(train)
            np.testing.assert_array_equal(np.asarray(train[0]["image"][0]), np.asarray(train[0]["image"][0]))
            np.testing.assert_array_equal(
                np.asarray(train[0]["image"][1]),
                np.asarray(
                    val.monai_transform(
                        {
                            "image_t1": train.items[0]["image"],
                            "image_t2": train.items[0]["z_image"],
                            "label": 0,
                            "mask_t1": train.items[0]["mask_image"],
                            "mask_t2": train.items[0]["mask_z_image"],
                        }
                    )["image_t2"]
                ),
            )

            quick = ["--permutations", "20", "--probe-permutations", "2", "--loss-batches", "5", "--matched-draws", "1"]
            argv = ["--run-dir", str(run), "--device", "cpu", "--batch-size", "5"] + quick
            with contextlib.redirect_stdout(io.StringIO()):
                report = causes.main(argv + ["--out-dir", str(root / "first")])
                causes.main(argv + ["--out-dir", str(root / "second")])
            self.assertEqual(report["source"]["checkpoint_sha256"], before)
            self.assertEqual(hashlib.sha256((run / "model_best.pt").read_bytes()).hexdigest(), before)
            split = json.loads((run / "split.json").read_text())
            for name in ("val", "train"):
                with np.load(root / f"first/features_{name}.npz") as first, np.load(
                    root / f"second/features_{name}.npz"
                ) as second:
                    self.assertEqual(first["subjects"].tolist(), split["subjects"][name])
                    self.assertFalse(set(first["subjects"]) & set(split["subjects"]["test"]))
                    self.assertEqual(first["content_t1"].shape, (len(split["subjects"][name]), 9))
                    np.testing.assert_array_equal(first["content_t1"], second["content_t1"])
                    np.testing.assert_allclose(first["desc_mask_dice"], 1.0)  # the fake views share one mask
                    self.assertTrue((first["desc_mask_ml_t1"] > 0).all())
                    self.assertTrue(np.isfinite(first["desc_blur_flair"]).all())
            self.assertEqual(report["splits"]["train"]["matched"]["subjects"], len(split["subjects"]["val"]))


if __name__ == "__main__":
    unittest.main()
