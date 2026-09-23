"""Pooling semantics, held-out selection, and real encoder/renderer smoke test."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from eval.pooling_probe import (
    TARGETS,
    block_gram,
    fit_readouts,
    main,
    native_maps,
    paired_delta,
    regional_pool,
    split_subjects,
)


def real_model(**kwargs):
    # Execute the real architecture, omitting unrelated MONAI utility imports.
    source = Path(__file__).resolve().parents[1] / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "utils.utils" for a in n.names))
    ]
    namespace = {"__name__": "pooling_probe_test_model"}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["VQVAE"](
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        nb_levels=1,
        embed_dim=8,
        nb_entries=8,
        scaling_rates=[2],
        use_checkpoint=False,
        content_size=6,
        style_size=2,
        content_style_levels=[0],
        mask_mode="fixed",
        inject_style_to_decoder=True,
        style_injection_mode="input",
        norm_type="layer",
        **kwargs,
    ).eval()


class PoolingTests(unittest.TestCase):
    def test_signed_tails_and_spatial_layout(self):
        x = torch.arange(-32, 32, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
        pooled, info, selection = regional_pool(x, 2, 0.25, True)
        expected = {k: [] for k in pooled}
        for z in (0, 2):
            for y in (0, 2):
                for w in (0, 2):
                    values = sorted(x[0, 0, z : z + 2, y : y + 2, w : w + 2].flatten().tolist())
                    expected["mean"].append(np.mean(values))
                    expected["max"].append(values[-1])
                    expected["upper"].append(np.mean(values[-2:]))
                    expected["lower"].append(np.mean(values[:2]))
        for name in pooled:
            np.testing.assert_allclose(pooled[name].flatten(), expected[name])
        self.assertEqual(info["k"], 2)
        self.assertEqual(selection["upper"].sum(), 16)
        self.assertEqual(selection["upper"][0, 3, 3, 3], 1)
        self.assertEqual(selection["lower"][0, 0, 0, 0], 1)

    def test_small_tail_is_max_and_full_tail_is_mean(self):
        x = torch.randn(2, 3, 16, 16, 16)
        pools, info, _ = regional_pool(x, 8, 0.05)
        self.assertEqual(info["sites_per_region"], 8)
        self.assertEqual(info["k"], 1)
        torch.testing.assert_close(pools["upper"], pools["max"])
        pools, _, _ = regional_pool(x, 8, 1)
        torch.testing.assert_close(pools["upper"], pools["mean"])
        torch.testing.assert_close(pools["lower"], pools["mean"])

    def test_invalid_grids_and_nonfinite_features_fail(self):
        for grid, fraction in ((3, 0.25), (8, 0.25), (2, 0), (2, 1.1)):
            with self.assertRaises(ValueError):
                regional_pool(torch.ones(1, 2, 4, 4, 4), grid, fraction)
        with self.assertRaises(ValueError):
            regional_pool(torch.full((1, 1, 4, 4, 4), float("nan")), 2, 0.25)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=2)
        self.threads.__enter__()
        torch.set_num_threads(2)

    def tearDown(self):
        self.threads.__exit__(None, None, None)

    def test_tails_recover_planted_signal_cancelled_by_mean(self):
        rng = np.random.default_rng(21)
        u = rng.uniform(0.5, 3, 120).astype(np.float32)
        x = torch.zeros(120, 1, 2, 2, 2)
        x[:, 0, 0, 0, 0] = torch.from_numpy(u)
        x[:, 0, 1, 1, 1] = -torch.from_numpy(u)
        pools, _, _ = regional_pool(x, 1, 0.25)
        y = np.column_stack([u * (j + 1) for j in range(len(TARGETS))])
        splits = split_subjects(len(y), 0)
        for stat in ("mean", "upper", "lower"):
            flat = pools[stat].flatten(1).numpy()
            gram, width = block_gram(flat, np.arange(flat.shape[1]), splits[0])
            rows, _ = fit_readouts(gram, width, y, splits, 0)
            score = next(
                r["test_r2"]
                for r in rows
                if r["probe"] == "ridge" and r["condition"] == "observed" and r["target"] == "ventricle_size"
            )
            self.assertLess(score, 0.05) if stat == "mean" else self.assertGreater(score, 0.99)
            null = [r["test_r2"] for r in rows if r["condition"] == "shuffled"]
            self.assertLess(max(null), 0.25)

    def test_test_labels_cannot_change_selection_or_predictions_even_for_null(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(80, 5))
        y = x @ rng.normal(size=(5, len(TARGETS)))
        splits = split_subjects(len(y), 0)
        gram, width = block_gram(x, np.arange(5), splits[0])
        rows, predictions = fit_readouts(gram, width, y, splits, 0)
        y[splits[2]] += 1000
        other, other_predictions = fit_readouts(gram, width, y, splits, 0)
        for a, b in zip(rows, other):
            self.assertEqual((a["alpha"], a["gamma"]), (b["alpha"], b["gamma"]))
        for key in predictions:
            np.testing.assert_allclose(predictions[key], other_predictions[key])

    def test_rbf_recovers_nonlinear_interaction_that_ridge_misses(self):
        rng = np.random.default_rng(8)
        x = np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (40, 1))
        signal = x[:, 0] * x[:, 1]
        x += rng.normal(scale=0.03, size=x.shape)
        y = np.column_stack([signal * (j + 1) for j in range(len(TARGETS))])
        splits = split_subjects(len(y), 0)
        gram, width = block_gram(x, np.arange(2), splits[0])
        rows, _ = fit_readouts(gram, width, y, splits, 0)
        scores = {
            r["probe"]: r["test_r2"] for r in rows if r["condition"] == "observed" and r["target"] == "ventricle_size"
        }
        self.assertGreater(scores["rbf"], 0.95)
        self.assertLess(scores["ridge"], 0.2)

    def test_paired_interval_and_train_only_scaling(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(50, 7))
        splits = split_subjects(50, 2)
        gram, _ = block_gram(x, np.arange(7), splits[0])
        x[splits[2]] += 1000
        changed, _ = block_gram(x, np.arange(7), splits[0])
        idx = np.ix_(splits[0], splits[0])
        np.testing.assert_allclose(gram[idx], changed[idx])
        y = rng.normal(size=(30, 6))
        np.testing.assert_array_equal(paired_delta(y, y, y, 0, 30), np.zeros((2, 6)))
        self.assertTrue((paired_delta(y, y, y * 0, 0, 30) > 0).all())

    def test_native_tap_matches_forward_and_preserves_parameters_and_buffers(self):
        for separate in (False, True):
            for masked in (False, True):
                model = real_model(separate_encoders=separate, latent_mask=masked)
                x = torch.randn(6, 1, 16, 16, 16)
                mask = torch.zeros_like(x)
                mask[..., 4:12, 4:12, 4:12] = 1
                before = {k: v.clone() for k, v in model.state_dict().items()}
                features, masks = native_maps(model, x, mask, 0)
                self.assertEqual(features.shape, (6, 8, 8, 8, 8))
                for k, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[k], rtol=0, atol=0)
                self.assertFalse(model.encoders[0]._forward_hooks)
                if masked:
                    self.assertEqual(features[..., 0, 0, 0].abs().sum(), 0)
                with patch.object(model, "forward", side_effect=RuntimeError("test failure")):
                    with self.assertRaises(RuntimeError):
                        native_maps(model, x, mask, 0)
                self.assertFalse(model.encoders[0]._forward_hooks)

    def test_cli_real_model_renderer_checkpoint_unchanged_and_reports_valid(self):
        args = argparse.Namespace(
            synthetic_mode="pseudo_mri",
            synthetic_res=16,
            synthetic_clean_content=True,
            synthetic_normalize="fixed_reference",
            synthetic_n_content=9,
            synthetic_n_style=3,
        )
        model = real_model()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict()}, checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = Path(tmp) / "audit"
            with patch(
                "eval.run_dci_synthetic.load_model_from_run_dir", return_value=(model, args, "cpu")
            ), contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--run-dir",
                        tmp,
                        "--num-samples",
                        "40",
                        "--batch-size",
                        "7",
                        "--grids",
                        "1",
                        "4",
                        "--examples",
                        "2",
                        "--threads",
                        "2",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["pooling"]["flair_g4"]["k"], 2)
            self.assertEqual(len(summary["scores"]), 2 * 2 * 5 * 2 * 2 * 6)
            predictions = np.load(output / "predictions.npz")
            combined = np.concatenate([predictions[f"{s}_indices"] for s in ("train", "validation", "test")])
            self.assertEqual(len(np.unique(combined)), 40)
            self.assertEqual(predictions["flair_g4_mean_tails_ridge_observed"].shape, (8, 6))
            examples = np.load(output / "examples.npz")
            self.assertEqual(examples["t1_image"].shape, (2, 1, 16, 16, 16))
            self.assertEqual(examples["t1_g4_upper_selection"].shape, (2, 8, 8, 8))
            self.assertFalse(model.encoders[0]._forward_hooks)


if __name__ == "__main__":
    unittest.main()
