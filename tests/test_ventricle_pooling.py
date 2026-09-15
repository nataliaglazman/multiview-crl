"""Pooling parity, spatial-information oracle, microbatch masking and held-out isolation."""

import ast
import contextlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sklearn  # noqa: F401 - import compiled dependencies before sys.modules patches
import torch
from threadpoolctl import threadpool_limits

from eval import ventricle_pooling as vp

ROOT = Path(__file__).resolve().parents[1]


def arrays_from_maps(maps, targets, n_fit):
    descriptors, _ = vp.pool_descriptors(maps, torch.ones(64, dtype=torch.bool), (4, 4, 4), (2, 2, 2))
    arrays = {
        f"{view}/{pool}": tensor[v].numpy() for v, view in enumerate(vp.VIEWS) for pool, tensor in descriptors.items()
    }
    arrays.update(
        targets=np.asarray(targets),
        is_fit=np.arange(len(targets)) < n_fit,
        subject_id=np.arange(len(targets)),
        mask_group=np.arange(len(targets)),
    )
    return arrays


def real_model():
    spec = importlib.util.spec_from_file_location("pooling_test_vqvae", ROOT / "models/vqvae.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"utils.utils": types.ModuleType("utils.utils")}):
        spec.loader.exec_module(module)
    return module.VQVAE(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        nb_levels=1,
        embed_dim=4,
        nb_entries=8,
        scaling_rates=[2],
        use_checkpoint=False,
        content_size=3,
        style_size=1,
        mask_mode="fixed",
        separate_encoders=True,
        inject_style_to_decoder=True,
        norm_type="layer",
    ).eval()


class Samples:
    res = 8

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1400)
        y = float(rng.normal())
        x = torch.tensor(rng.normal(size=(1, 8, 8, 8)), dtype=torch.float32)
        mask = torch.zeros_like(x)
        # Every individual microbatch sees only one half, but logical batches see both.
        mask[:, (i % 2) * 4 : (i % 2 + 1) * 4] = 1
        return {
            "image": [x, -0.3 * x + y],
            "mask": [mask, mask.clone()],
            "gt_latents": {"z_content": torch.tensor([0.0, y])},
        }


def settings():
    return types.SimpleNamespace(
        mask_mode="fixed",
        patch_grid=[4, 4, 4],
        vqvae_nb_levels=1,
        batch_size=4,
        patch_foreground_mask=True,
        patch_foreground_thresh=0.05,
        synthetic_res=8,
        patch_contrastive=True,
    )


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stats_match_existing_eval_formula_and_channel_selection(self):
        tree = ast.parse((ROOT / "eval/dci.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pool_and_split_view"]
        scope = {"torch": torch, "np": np}
        exec(compile(tree, "eval/dci.py", "exec"), scope)
        maps = torch.randn(2, 7, 5, 4, 4, 4)
        selected = [1, 3]
        ours, _ = vp.pool_descriptors(
            maps[:, :, selected].flatten(3),
            torch.ones(64, dtype=torch.bool),
            (4, 4, 4),
            (2, 2, 2),
        )
        for v in range(2):
            expected, _ = scope["_pool_and_split_view"](maps[v], selected, "stats", False, True)
            np.testing.assert_allclose(ours["stats"][v].numpy(), expected, atol=1e-6)

    def test_masked_stats_and_regions_use_original_coordinates(self):
        z = torch.arange(64, dtype=torch.float32).reshape(1, 1, 1, 64).expand(2, 3, 2, 64)
        keep = torch.tensor([i % 3 == 0 for i in range(64)])
        pools, counts = vp.pool_descriptors(z, keep, (4, 4, 4), (2, 2, 2))
        expected = torch.cat(
            [
                z[..., keep].mean(-1),
                z[..., keep].std(-1),
                z[..., keep].amax(-1),
                z[..., keep].amin(-1),
            ],
            -1,
        )
        torch.testing.assert_close(pools["stats"], expected)
        self.assertEqual(sum(counts), int(keep.sum()))
        # First 2x2x2 cell has original flat indices 0,1,4,5,16,17,20,21.
        torch.testing.assert_close(pools["regional_mean_std"][..., :2], z[..., [0, 21]].mean(-1))

    def test_empty_and_singleton_regions_are_finite_and_explicit(self):
        z = torch.randn(2, 4, 1, 64)
        keep = torch.zeros(64, dtype=torch.bool)
        keep[[0, 63]] = True
        pools, counts = vp.pool_descriptors(z, keep, (4, 4, 4), (2, 2, 2))
        self.assertEqual(counts, [1, 0, 0, 0, 0, 0, 0, 1])
        region = pools["regional_mean_std"].reshape(2, 4, 8, 2)
        self.assertTrue(torch.isfinite(region).all())
        self.assertEqual(float(region[..., 1].abs().sum()), 0.0)
        self.assertEqual(float(region[:, :, 1:7].abs().sum()), 0.0)
        keep[63] = False
        with self.assertRaisesRegex(ValueError, "at least two"):
            vp.pool_descriptors(z, keep, (4, 4, 4), (2, 2, 2))

    def test_extrema_can_miss_extent_while_mean_std_changes(self):
        z = torch.ones(2, 2, 1, 64)
        z[:, 0, :, :4] = 0
        z[:, 1, :, :16] = 0
        pools, _ = vp.pool_descriptors(z, torch.ones(64, dtype=torch.bool), (4, 4, 4), (2, 2, 2))
        torch.testing.assert_close(pools["stats"][:, 0, 2:], pools["stats"][:, 1, 2:])
        self.assertFalse(torch.allclose(pools["mean_std"][:, 0], pools["mean_std"][:, 1]))

    def test_regions_recover_signed_spatial_signal_invisible_to_global_stats(self):
        # Every amplitude occurs with both signs, in BOTH partitions. Global statistic
        # vectors are identical for +/- pairs, so even a nonlinear readout cannot tell.
        rng = np.random.default_rng(5)
        amplitudes = rng.uniform(0.1, 1.0, 80)
        y = np.column_stack((amplitudes, -amplitudes)).reshape(-1)
        z = torch.tensor(y, dtype=torch.float32)[None, :, None, None].expand(2, -1, 1, 64).clone()
        z[..., 32:] *= -1
        arrays = arrays_from_maps(z, y, 100)
        for p in ("gap", "mean_std", "stats"):
            np.testing.assert_allclose(arrays[f"t1/{p}"][::2], arrays[f"t1/{p}"][1::2], atol=1e-7)
        with threadpool_limits(limits=2):
            rows, _, _ = vp.evaluate(arrays, ["ridge"], 0, 3, 30)
        for row in rows:
            if row["pooling"] == "regional_mean_std":
                self.assertGreater(row["r2"], 0.99)
                self.assertGreater(row["vs_mean_std_delta_ci_low"], 0.8)
            else:
                self.assertLess(row["r2"], 0.02)

    def test_test_targets_never_change_fitted_predictions_or_hyperparameters(self):
        rng = np.random.default_rng(10)
        y = rng.normal(size=60)
        z = torch.tensor(rng.normal(size=(2, 60, 1, 64)), dtype=torch.float32)
        z += torch.tensor(y)[None, :, None, None]
        arrays = arrays_from_maps(z, y, 40)
        with threadpool_limits(limits=2):
            _, before, selected = vp.evaluate(arrays, ["ridge", "rbf"], 7, 2, 10)
            arrays["targets"] = y.copy()
            arrays["targets"][40:] = 100 + y[40:] * 10
            _, after, other = vp.evaluate(arrays, ["ridge", "rbf"], 7, 2, 10)
        self.assertEqual(selected, other)
        for key in before:
            if key != "targets":
                np.testing.assert_array_equal(before[key], after[key])

    def test_paired_intervals_zero_for_identical_predictions(self):
        y = np.linspace(-1, 1, 40)
        result = vp.paired_score(y, y * 0.8, y * 0.8, 0, 100)
        self.assertEqual(result["delta_r2"], 0)
        self.assertEqual(result["delta_ci_low"], 0)
        self.assertEqual(result["delta_ci_high"], 0)

    def test_rejects_cross_partition_mask_groups_and_unsupported_settings(self):
        z = torch.randn(2, 12, 1, 64)
        arrays = arrays_from_maps(z, np.arange(12.0), 8)
        arrays["mask_group"][:] = 0
        with self.assertRaisesRegex(ValueError, "share foreground"):
            vp.validate_arrays(arrays)
        for attr, value in (
            ("mask_mode", "onthefly"),
            ("contrastive_proj_dim", 12),
            ("split_encoder_norm", True),
        ):
            args = settings()
            setattr(args, attr, value)
            with self.assertRaises(ValueError):
                vp.validate_settings(args, 0, (4, 4, 4), (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "divide"):
            vp.region_ids((4, 4, 4), (3, 2, 2))


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_microbatch_invariance_and_parity_with_existing_loss_feature_extractor(
        self,
    ):
        from torch.utils.data import DataLoader, Subset

        from eval.reconstruction_attribution import _content_features, frozen_checkpoint

        model, ds, args = real_model(), Samples(10), settings()
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        with frozen_checkpoint(model):
            a, details = vp.extract_features(model, ds, args, "cpu", (4, 4, 4), (2, 2, 2), 5, 4, 1)
            b, other = vp.extract_features(model, ds, args, "cpu", (4, 4, 4), (2, 2, 2), 5, 4, 4)
            batch = next(iter(DataLoader(Subset(ds, range(4)), batch_size=4)))
            train_hz = _content_features(model, batch, args, "cpu", (4, 4, 4), 0).detach()
        self.assertEqual(details, other)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-6)
        for v, view in enumerate(vp.VIEWS):
            np.testing.assert_allclose(a[f"{view}/gap"][:4], train_hz[v].mean(-1).numpy(), rtol=1e-5, atol=1e-6)
            expected = torch.cat(
                (
                    train_hz[v].mean(-1),
                    train_hz[v].std(-1),
                    train_hz[v].amax(-1),
                    train_hz[v].amin(-1),
                ),
                -1,
            )
            np.testing.assert_allclose(a[f"{view}/stats"][:4], expected.numpy(), rtol=1e-5, atol=1e-6)
        self.assertEqual(len(details["mask_groups"][0]["kept_positions"]), 64)
        self.assertEqual(len(details["mask_groups"][1]["kept_positions"]), 32)
        vp.validate_arrays(a)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)

    def test_end_to_end_checkpoint_unchanged_and_cache_replay_needs_no_model(self):
        model, args = real_model(), settings()
        fake = types.ModuleType("eval.run_dci_synthetic")
        fake.load_run_args = lambda *a: args
        fake.load_model_from_run_dir = lambda *a, **kw: (model, args, "cpu")
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 17}, checkpoint)
            original = checkpoint.read_bytes()
            cli = vp.parser().parse_args(
                [
                    "--run-dir",
                    temp,
                    "--fit-samples",
                    "12",
                    "--test-samples",
                    "8",
                    "--encode-batch",
                    "2",
                    "--probes",
                    "ridge",
                    "--folds",
                    "2",
                    "--bootstrap",
                    "10",
                    "--threads",
                    "2",
                    "--out",
                    str(Path(temp) / "result"),
                ]
            )
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), patch(
                "eval.ventricle_routing.make_dataset",
                side_effect=lambda args, n, *a: Samples(n),
            ), contextlib.redirect_stdout(io.StringIO()):
                report = vp.run(cli)
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(report["metadata"]["checkpoint_step"], 17)
            cache = Path(cli.out) / "features.npz"
            self.assertTrue(cache.exists())
            self.assertTrue((Path(cli.out) / "pooling_scores.csv").exists())
            saved = json.loads((Path(cli.out) / "summary.json").read_text())
            self.assertEqual(saved["results"], report["results"])
            replay = vp.parser().parse_args(
                [
                    "--features",
                    str(cache),
                    "--probes",
                    "ridge",
                    "--folds",
                    "2",
                    "--bootstrap",
                    "10",
                    "--threads",
                    "2",
                    "--out",
                    str(Path(temp) / "replay"),
                ]
            )
            fake.load_model_from_run_dir = lambda *a, **kw: self.fail("Cache mode loaded a model")
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), contextlib.redirect_stdout(io.StringIO()):
                cached_report = vp.run(replay)
            self.assertEqual(report["results"], cached_report["results"])
            with self.assertRaises(FileExistsError):
                vp.run(replay)


if __name__ == "__main__":
    unittest.main()
