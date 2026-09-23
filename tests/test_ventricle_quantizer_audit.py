"""Known affine maps, code-boundary crossings, coupled renders and real-model CLI."""

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

from eval.ventricle_quantizer_audit import (
    audit,
    fixed_affine,
    main,
    make_dataset,
    projection_diagnostics,
    projection_response,
    regions_at,
    render_pair,
    response_metrics,
    state_digest,
    summarize,
    usage_stats,
)


def real_model(**kwargs):
    source = Path(__file__).resolve().parents[1] / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "utils.utils" for a in n.names))
    ]
    namespace = {"__name__": "quantizer_audit_test_model"}
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


def dataset_args():
    return argparse.Namespace(
        synthetic_mode="pseudo_mri",
        synthetic_res=32,
        synthetic_clean_content=True,
        synthetic_normalize="per_sample",
        synthetic_n_content=9,
        synthetic_n_style=3,
        synthetic_identifiable_ventricle=True,
    )


class QuantizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)

    def test_projection_singular_values_rank_and_pseudoinverse(self):
        conv = torch.nn.Conv3d(2, 3, 1)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 0.5], [0.0, 0.0]])[:, :, None, None, None])
            conv.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        info = projection_diagnostics(conv)
        self.assertEqual(info["rank_float32"], 2)
        self.assertAlmostEqual(info["condition_number"], 6.0)
        low, high = torch.randn(2, 2, 2, 2), torch.randn(2, 2, 2, 2)
        with torch.no_grad():
            pl, ph = conv(low[None])[0], conv(high[None])[0]
        region = torch.ones(2, 2, 2, dtype=torch.bool)
        result = projection_response(conv, low, high, pl, ph, region)
        self.assertLess(result["projection_inverse_response_relative_error"], 1e-6)
        with torch.no_grad():
            conv.weight[:, 1].zero_()
            pl, ph = conv(low[None])[0], conv(high[None])[0]
        info = projection_diagnostics(conv)
        self.assertEqual(info["input_nullity_float32"], 1)
        self.assertIsNone(info["condition_number"])
        result = projection_response(conv, low, high, pl, ph, region)
        self.assertGreater(result["projection_inverse_response_relative_error"], 0.1)

    def test_responses_that_stay_within_and_cross_a_code_boundary(self):
        low = torch.tensor([[[[0.1, 0.4]]]])
        high = torch.tensor([[[[0.2, 0.6]]]])
        qlow = torch.zeros_like(low)
        qhigh = torch.tensor([[[[0.0, 1.0]]]])
        ids_low, ids_high = qlow[0].long(), qhigh[0].long()
        region = torch.ones_like(ids_low, dtype=torch.bool)
        result = response_metrics(
            low, high, qlow, qhigh, ids_low, ids_high, low.clone(), qlow.clone(), ids_low.clone(), region
        )
        self.assertEqual(result["changed_code_fraction"], 0.5)
        self.assertEqual(result["replay_changed_code_fraction"], 0)
        self.assertTrue(result["resolved_pre"])
        near = torch.tensor([[[True, False]]])
        same = response_metrics(low, high, qlow, qhigh, ids_low, ids_high, low, qlow, ids_low, near)
        self.assertEqual(same["quant_to_pre_rms_ratio"], 0)
        self.assertEqual(same["quant_response_relative_error"], 1)
        self.assertIsNone(same["quant_response_cosine"])
        noisy = response_metrics(low, high, qlow, qhigh, ids_low, ids_high, low + 0.1, qlow, ids_low, region)
        self.assertFalse(noisy["resolved_pre"])
        self.assertIsNone(noisy["quant_to_pre_rms_ratio"])

    def test_zero_and_empty_responses_have_no_ratios(self):
        x = torch.ones(2, 2, 2, 2)
        ids = torch.zeros(2, 2, 2, dtype=torch.long)
        for occupied in (False, True):
            region = torch.full((2, 2, 2), occupied)
            result = response_metrics(x, x, x, x, ids, ids, x, x, ids, region)
            self.assertFalse(result["resolved_pre"])
            self.assertFalse(result["resolved_quant"])
            if occupied:
                self.assertIsNone(result["quant_to_pre_rms_ratio"])

    def test_spatial_regions_partition_map_and_usage_is_not_dead_code_count(self):
        support = torch.zeros(1, 16, 16, 16)
        support[:, 7, 7, 7] = 1
        regions = regions_at(support, (8, 8, 8), 1)
        self.assertEqual(int(regions["affected_bins"].sum()), 1)
        self.assertEqual(int(regions["neighborhood"].sum()), 27)
        self.assertTrue(torch.equal(regions["neighborhood"] | regions["outside"], regions["all"]))
        self.assertFalse((regions["neighborhood"] & regions["outside"]).any())
        stats = usage_stats(np.array([3, 3, 0]))
        self.assertEqual(stats["active_codes"], 2)
        self.assertAlmostEqual(stats["perplexity"], 2)

    def test_fixed_normalization_and_deterministic_interventions(self):
        ds = make_dataset(dataset_args(), 4, "iid", "test")
        zero = render_pair(ds, 0, 0)
        self.assertEqual(zero["support_voxels"], 0)
        for v in range(2):
            torch.testing.assert_close(zero["low"][v], zero["high"][v], rtol=0, atol=0)
        pair = render_pair(ds, 0, 0.5)
        again = render_pair(ds, 0, 0.5)
        self.assertGreater(pair["changed_tissue_voxels"], 0)
        for v in range(2):
            torch.testing.assert_close(pair["low"][v], again["low"][v], rtol=0, atol=0)
            self.assertEqual(float((pair["high"][v] - pair["low"][v])[~pair["support"]].abs().max()), 0)
        self.assertAlmostEqual(pair["z1_high"] - pair["z1_low"], 1.0)
        raw = torch.arange(8.0).reshape(1, 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "not foreground-affine"):
            fixed_affine(raw, raw.square(), torch.ones_like(raw))

    def test_cli_real_renderer_codebooks_replay_and_checkpoint_unchanged(self):
        model = real_model(
            quantize_style=True, separate_encoders=True, separate_content_codebooks=True, separate_style_codebooks=True
        )
        before_state = state_digest(model)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 123}, checkpoint)
            before_file = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            out = Path(tmp) / "audit"
            with patch(
                "eval.run_dci_synthetic.load_model_from_run_dir", return_value=(model, dataset_args(), "cpu")
            ), contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--run-dir",
                        tmp,
                        "--num-samples",
                        "3",
                        "--batch-size",
                        "2",
                        "--eps",
                        ".25",
                        ".5",
                        "--examples",
                        "1",
                        "--threads",
                        "2",
                        "--output-dir",
                        str(out),
                    ]
                )
            report = json.loads((out / "summary.json").read_text())
            self.assertEqual(report["checkpoint_step"], 123)
            self.assertEqual(len(report["summary"]), 2 * 2 * 2 * 4)
            self.assertTrue(report["model_state_unchanged"])
            self.assertEqual(before_file, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(before_state, state_digest(model))
            maps = np.load(out / "examples.npz")
            self.assertEqual(maps["eps0.25_sample0_flair_content_changed_codes"].shape, (16, 16, 16))
            self.assertEqual(maps["eps0.25_sample0_flair_content_input_delta"].shape, (1, 32, 32, 32))
            import csv

            with (out / "responses.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * 2 * 2 * 2 * 4)
            self.assertTrue(
                all(
                    float(r["replay_changed_code_fraction"]) == 0 for r in rows if r.get("replay_changed_code_fraction")
                )
            )
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_unquantized_style_is_skipped_and_constant_codebook_cannot_respond(self):
        model = real_model()
        with torch.no_grad():
            model.codebooks[0].embed.zero_()
        ds = make_dataset(dataset_args(), 2, "iid", "test")
        rows, _, _ = audit(model, ds, "cpu", [0.5], 2, 1, 0)
        self.assertEqual({r["block"] for r in rows}, {"content"})
        self.assertTrue(all(r.get("changed_code_fraction", 0) == 0 for r in rows))
        self.assertTrue(all(r.get("quant_delta_rms", 0) == 0 for r in rows))
        self.assertTrue(any(r["resolved_pre"] for r in rows))
        self.assertTrue(all(s["n_resolved_quant"] == 0 for s in summarize(rows)))


if __name__ == "__main__":
    unittest.main()
