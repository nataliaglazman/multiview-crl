"""Controls for style-stage probes, exact decoder swaps, and frozen rendering."""

import argparse
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.style_path_audit import (
    capture_path,
    check_endpoint,
    effective_style,
    encode,
    fit_probe,
    main,
    make_dataset,
    render_sample,
    replay,
    response_metrics,
    swap_batch,
    validate_model,
)


def real_model(**kwargs):
    # Same isolated model load as test_style_hsic: omit unrelated MONAI utils.
    source = Path(__file__).resolve().parents[1] / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "utils.utils" for a in n.names))
    ]
    namespace = {"__name__": "style_audit_test_model"}
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
        style_spatial_size=1,
        **kwargs,
    ).eval()


class ToyDecoder(torch.nn.Module):
    def __init__(self, use_style):
        super().__init__()
        self.use_style = use_style

    def forward(self, content, style):
        return content * 0 + style if self.use_style else content


class ToyModel(torch.nn.Module):
    def __init__(self, use_style):
        super().__init__()
        self.nb_levels = 1
        self.content_style_levels = [0]
        self.inject_style_to_decoder = True
        self.mask_mode = "fixed"
        self.decoders = torch.nn.ModuleList([ToyDecoder(use_style)])
        self._last_style_id_outputs = {}
        self.eval()

    def _bottleneck_style(self, x):
        return x

    def forward(self, x, **kwargs):
        return (self.decoders[0](x, style=self._bottleneck_style(x)),)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_effective_targets_and_noise_sign(self):
        np.testing.assert_allclose(effective_style(torch.tensor([2.0, -2.0, -3.0]), 1), [1.3, -0.1, 0.16])
        np.testing.assert_equal(
            effective_style(torch.tensor([2.0, -2.0, -3.0]), 1), effective_style(torch.tensor([2.0, -2.0, 3.0]), 1)
        )
        self.assertEqual(effective_style(torch.tensor([-1.0, 0.0, 0.0]), 10)[0], 0.05)

    def test_response_identity_zero_and_endpoint_failure(self):
        x = np.arange(20, dtype=float)
        self.assertAlmostEqual(response_metrics(x, x)["gain"], 1)
        self.assertAlmostEqual(response_metrics(x, x)["relative_error"], 0)
        self.assertEqual(response_metrics(x, x * 0)["gain"], 0)
        self.assertTrue(np.isnan(response_metrics(x * 0, x)["gain"]))
        with self.assertRaisesRegex(ValueError, "endpoint replay failed"):
            check_endpoint(torch.ones(10), torch.zeros(10))

    def test_probe_positive_constant_and_test_label_isolation(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(180, 6))
        y = 2 * x[:, 0] - x[:, 2]
        split = np.split(rng.permutation(len(x)), [108, 144])
        m, pred = fit_probe(x, y, split)
        self.assertGreater(m["true_r2"], 0.99)
        self.assertLess(m["shuffled_r2"], 0.2)
        changed_y = y.copy()
        changed_y[split[2]] += 1000
        altered, altered_pred = fit_probe(x, changed_y, split)
        self.assertEqual(m["true_alpha"], altered["true_alpha"])
        np.testing.assert_array_equal(pred["true"], altered_pred["true"])
        constant, _ = fit_probe(np.ones((180, 3)), y, split)
        self.assertEqual(constant["varying_fit_columns"], 0)
        self.assertLessEqual(constant["true_r2"], 0)

    def test_pooling_can_remove_signal_without_collapsed_raw_features(self):
        rng = np.random.default_rng(8)
        y = rng.normal(size=180)
        x = np.stack([y, -y], axis=1)
        split = np.split(rng.permutation(180), [108, 144])
        raw, _ = fit_probe(x, y, split)
        pooled, _ = fit_probe(x.mean(1, keepdims=True), y, split)
        self.assertGreater(raw["true_r2"], 0.99)
        self.assertLessEqual(pooled["true_r2"], 0)

    def test_swaps_identify_style_and_content_only_decoders(self):
        mask = torch.ones(1, 4, 4, 4)
        lo, hi = [mask * 0.2, mask * 0.3], [mask * 0.7, mask * 0.9]
        samples = [(lo, mask, None, {"gain": [lo, hi]}) for _ in range(3)]
        for use_style in (True, False):
            model = ToyModel(use_style)
            validate_model(model)
            rows = swap_batch(model, samples, "cpu", "gain")
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertTrue(row["resolved"])
                self.assertAlmostEqual(row["joint_gain"], 1)
                self.assertAlmostEqual(row["style_mean_gain"], float(use_style))
                self.assertAlmostEqual(row["content_mean_gain"], float(not use_style))
                self.assertAlmostEqual(row["interaction_rms_ratio"], 0, places=6)
                self.assertEqual(row["endpoint_rms"], 0)

    def test_capture_restored_after_error(self):
        model = ToyModel(True)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with capture_path(model):
                raise RuntimeError("deliberate")
        self.assertNotIn("_bottleneck_style", model.__dict__)
        self.assertEqual(len(model.decoders[0]._forward_pre_hooks), 0)

    def test_real_vqvae_boundaries_replay_and_checkpoint_unchanged(self):
        images = [[torch.randn(1, 8, 8, 8) for _ in range(2)] for _ in range(3)]
        masks = [torch.ones(1, 8, 8, 8) for _ in images]
        for quantize in (False, True):
            for separate in (False, True):
                model = real_model(
                    quantize_style=quantize, separate_encoders=separate, separate_style_codebooks=separate
                )
                validate_model(model)
                state = {k: v.clone() for k, v in model.state_dict().items()}
                path = encode(model, images, masks, "cpu")
                self.assertEqual(path["raw"].shape, (6, 2, 4, 4, 4))
                self.assertEqual(path["pooled"].shape, (6, 2, 1, 1, 1))
                torch.testing.assert_close(path["pooled"], path["raw"].mean((2, 3, 4), keepdim=True))
                self.assertEqual(bool(path["ids"]), quantize)
                if not quantize:
                    torch.testing.assert_close(path["pooled"], path["injected"], rtol=0, atol=0)
                y = replay(model, path["content"], path["injected"], path["output"].shape[2:])
                self.assertLess(check_endpoint(path["output"], y), 1e-6)
                # Actual swaps also run with both separate view codebooks and a shared book.
                samples = [(s, m, None, {"bias": [s, [x + 0.1 for x in s]]}) for s, m in zip(images, masks)]
                rows = swap_batch(model, samples, "cpu", "bias")
                self.assertTrue(all(r["resolved"] for r in rows))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_natural_matches_dataset_and_gain_bias_use_frozen_noise_and_affine(self):
        args = argparse.Namespace(
            synthetic_mode="pseudo_mri",
            synthetic_res=16,
            synthetic_clean_content=True,
            synthetic_normalize="per_sample",
            synthetic_n_content=9,
            synthetic_n_style=3,
        )
        ds = make_dataset(args, 4, "iid", "test")
        original, mask, targets, pairs = render_sample(ds, 0, 0.5)
        for a, b in zip(original, ds[0]["image"]):
            torch.testing.assert_close(a, b)
        # Zero-amplitude paired intervention renders identical endpoints, even though
        # each sets the latent to zero rather than the subject's natural value.
        _, _, _, zero = render_sample(ds, 0, 0)
        for pair in zero.values():
            for a, b in zip(*pair):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        again = render_sample(ds, 0, 0.5)[3]
        for factor, (low, high) in pairs.items():
            for v in range(2):
                torch.testing.assert_close(low[v], again[factor][0][v], rtol=0, atol=0)
                self.assertGreater(float((high[v] - low[v])[mask > 0].abs().mean()), 1e-4)
                self.assertEqual(float(high[v][mask == 0].abs().sum()), 0)
                # Per-sample re-normalizing each variant would erase this mean shift.
                self.assertGreater(float(high[v][mask > 0].mean() - low[v][mask > 0].mean()), 0.01)
        self.assertTrue(np.isfinite(targets).all())

    def test_full_cli_writes_reports_with_real_renderer_and_model(self):
        args = argparse.Namespace(
            synthetic_mode="pseudo_mri",
            synthetic_res=16,
            synthetic_clean_content=True,
            synthetic_normalize="fixed_reference",
            synthetic_n_content=9,
            synthetic_n_style=3,
        )
        model = real_model(quantize_style=True, separate_style_codebooks=True)
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "style_path_audit",
                "--run-dir",
                "unused",
                "--num-samples",
                "40",
                "--swap-samples",
                "3",
                "--batch-size",
                "4",
                "--out-dir",
                directory,
            ]
            with patch("sys.argv", argv), patch(
                "eval.run_dci_synthetic.load_model_from_run_dir", return_value=(model, args, "cpu")
            ), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(len(report["probes"]), 24)
            self.assertEqual(len(report["swaps"]), 4)
            for metrics in report["swaps"].values():
                self.assertEqual(metrics["n_resolved"], 3)
            split = report["subject_split"]
            self.assertEqual(len(set(sum(split, []))), 40)
            self.assertEqual(report["code_usage"]["t1"]["assignments_per_subject"], 1)
            saved = np.load(Path(directory) / "probe_predictions.npz")
            self.assertTrue((saved["t1/raw/noise_sigma/truth"] >= 0.01).all())
            self.assertEqual(len((Path(directory) / "swaps.csv").read_text().splitlines()), 13)


if __name__ == "__main__":
    unittest.main()
