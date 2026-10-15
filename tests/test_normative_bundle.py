"""Normative content fits training images only and exports the shared row-identity contract."""

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval.dino.dinov3_embed_synthetic import save
from eval.dino.dinov3_identifiability import load_bundle
from eval.metrics.bundle_identity import factor_digest
from eval.protocol import export_normative_bundle as normative


class NormativeArrayTests(unittest.TestCase):
    def test_planted_shared_dimensions_spatial_rotation_and_brain_frame_peak(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(normative._self_test(), 0)

    def test_standardizer_keeps_training_statistics_for_new_rows(self):
        fit = np.array([[1.0, 5.0], [3.0, 5.0]])
        transform = normative.standardizer(fit)
        np.testing.assert_array_equal(transform([[4.0, 7.0]]), [[2.0, 2.0]])

    def test_empty_brain_has_neutral_coordinates(self):
        np.testing.assert_array_equal(normative.brain_frame_peak(np.ones((3, 4, 5)), np.zeros((3, 4, 5))), 0)

    def test_invalid_sizes_are_rejected_before_rendering_or_fitting(self):
        invalid = [
            ("--num-samples", "1"),
            ("--batch-size", "0"),
            ("--num-workers", "-1"),
            ("--fit-samples", "2"),
            ("--components", "0"),
            ("--shared-dims", "21"),
            ("--shared-dims", "0"),
            ("--null-permutations", "0"),
        ]
        for args in invalid:
            with self.subTest(args=args), patch.object(normative, "build") as build, contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                try:
                    normative.main(["--run-dir", "unused", "--out", "unused.npz", *args])
                finally:
                    build.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch"), "export integration requires torch")
class NormativeExportTests(unittest.TestCase):
    def setUp(self):
        import torch

        class Images:
            lesion_target = "position"

            def __init__(self, seed, n):
                rng = np.random.default_rng(seed)
                self.z = rng.normal(size=(n, 3)).astype(np.float32)
                self.z[:, 1] += self.z[:, 0] * 0.8
                maps = np.random.default_rng(2).normal(size=(3, 216))
                signal = (self.z @ maps).reshape(n, 1, 6, 6, 6)
                self.images = [
                    torch.tensor(10 + signal + rng.normal(size=signal.shape) * 0.2, dtype=torch.float32)
                    for _ in range(2)
                ]
                self.style = rng.normal(size=(n, 2, 3)).astype(np.float32)
                self.scm = {"adj": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])}

            def __len__(self):
                return len(self.z)

            def __getitem__(self, i):
                return dict(
                    image=[x[i] for x in self.images],
                    gt_latents=dict(z_content=self.z[i], z_style_v1=self.style[i, 0], z_style_v2=self.style[i, 1]),
                )

        self.train, self.test = Images(1, 72), Images(3, 24)
        self.cli = argparse.Namespace(
            run_dir="unused",
            fit_samples=72,
            num_samples=24,
            split="test",
            causal=True,
            distribution="match",
            batch_size=12,
            num_workers=0,
            components=5,
            null_permutations=10,
            seed=0,
            shared_dims=3,
            rotation="spatial_ica",
            block="per_view",
            lesion="peak",
        )
        self.generator = {"synthetic_seed": 3, "synthetic_res": 6}

    def build(self):
        def split(_run, mode, n, causal):
            data = self.train if mode == "train" else self.test
            self.assertEqual(len(data), n)
            return data, self.generator

        with patch.object(normative, "generator_split", side_effect=split) as factory, patch.object(
            normative, "normative_model", wraps=normative.normative_model
        ) as fit:
            result = normative.build(self.cli)
        self.assertEqual(factory.call_args_list[0].args, ("unused", "train", 72, True))
        self.assertEqual(factory.call_args_list[1].args, ("unused", "test", 24, self.cli.causal))
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.args[0].shape, (72, 2, 216))
        np.testing.assert_array_equal(fit.call_args.args[0][:, 0], self.train.images[0].numpy().reshape(72, -1))
        return result

    def test_bundle_round_trip_preserves_rows_views_and_rotated_vs_lesion_columns(self):
        embeddings, latents, meta = self.build()
        self.assertEqual(meta["fit_split"], "train")
        self.assertEqual((meta["mode_subjects"], meta["calibration_subjects"]), (48, 24))
        self.assertEqual(meta["rotated_channels"], [0, 1, 2])
        self.assertEqual(meta["lesion_channels"], [3, 4, 5])
        self.assertEqual(meta["factor_digest"], factor_digest(latents))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normative.npz"
            save(path, embeddings, latents, {}, [], meta)
            for view in (1, 2):
                loaded = load_bundle(path, str(view))
                self.assertEqual(loaded["X"].shape, (24, 6))
                self.assertTrue(np.isfinite(loaded["X"]).all())
                np.testing.assert_array_equal(loaded["z_content"], self.test.z)
                np.testing.assert_array_equal(loaded["adjacency"], self.test.scm["adj"])
                self.assertEqual(loaded["identity"]["factor_digest"], meta["factor_digest"])

    def test_factor_labels_cannot_change_any_exported_channel(self):
        before, _, meta = self.build()
        self.train.z = self.train.z[::-1].copy()
        self.test.z = self.test.z[::-1].copy()
        after, _, renamed = self.build()
        for view in (1, 2):
            np.testing.assert_allclose(before[view], after[view], atol=1e-8)
        self.assertNotEqual(meta["factor_digest"], renamed["factor_digest"])

    def test_unified_anatomy_is_the_mean_of_per_view_channels_and_iid_has_no_graph(self):
        self.cli.lesion = "none"
        per_view, _, _ = self.build()
        self.cli.block, self.cli.causal, self.cli.distribution = "unified", False, "iid"
        self.test.scm = None
        unified, latents, meta = self.build()
        self.assertEqual(set(unified), {1})
        self.assertEqual(unified[1].shape, (24, 3))
        np.testing.assert_allclose(unified[1], (per_view[1] + per_view[2]) / 2, atol=1e-8)
        self.assertNotIn("causal_adj", latents)
        self.assertFalse(meta["causal"])
        self.assertEqual(meta["lesion_channels"], [])

    def test_encoder_generator_uses_the_same_split_seed_and_factors_as_checkpoint_scoring(self):
        from eval.protocol.score_checkpoint import make_dataset

        cfg = dict(
            res=16,
            n_content=9,
            n_style=3,
            seed=42,
            data_seed=17,
            synthetic_causal=True,
            synthetic_causal_graph="random",
            synthetic_clean_content=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "settings.json").write_text(json.dumps(cfg))
            for mode, causal in (("train", True), ("test", True), ("test", False), ("val", True)):
                actual, settings = normative.generator_split(tmp, mode, 4, causal)
                expected = make_dataset(dict(cfg, synthetic_causal=causal), 4, mode=mode)
                self.assertEqual(settings["synthetic_seed"], 17)
                for i in range(4):
                    for key in ("z_content", "z_style_v1", "z_style_v2"):
                        np.testing.assert_array_equal(actual[i]["gt_latents"][key], expected[i]["gt_latents"][key])
                    np.testing.assert_array_equal(actual[i]["image"][0], expected[i]["image"][0])


if __name__ == "__main__":
    unittest.main()
