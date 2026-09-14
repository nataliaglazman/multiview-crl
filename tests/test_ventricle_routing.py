"""Routing oracles, real renderer controls and actual VQVAE swap round trips.

AST loading omits unrelated ADNI imports, as in test_style_hsic.py.
The renderer, dataset normalizers and model implementations run unchanged.
"""

import argparse
import ast
import contextlib
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from eval.ventricle_routing import (
    assess_endpoint_replay,
    audit,
    decode_swaps,
    main,
    make_dataset,
    render_pair,
    score_swaps,
    stable_replay_math,
)

ROOT = Path(__file__).resolve().parents[1]


def load_without_monai(relative):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(a.name in ("utils.utils", "pandas") for a in node.names)
            or isinstance(node, ast.ImportFrom)
            and node.module == "utils.utils"
        )
    ]
    module = types.ModuleType("routing_test_" + source.stem)
    exec(compile(tree, str(source), "exec"), module.__dict__)
    return module


class OracleCodebook(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return x, x.new_zeros(()), x[:, 0] - self.offset


class RoutingOracle(torch.nn.Module):
    """Identity endpoints with a known fraction of the response carried by content."""

    def __init__(self, content_weight):
        super().__init__()
        self.weight = content_weight
        self.nb_levels = 1
        self.inject_style_to_decoder = True
        self.separate_content_codebooks = True
        self.codebooks = torch.nn.ModuleList([OracleCodebook(0)])
        self.codebooks_v1 = torch.nn.ModuleList([OracleCodebook(10)])
        self.eval()

    def forward(self, x, **kwargs):
        assert kwargs["n_views"] == 2
        first = self.codebooks[0](x[: len(x) // 2])
        second = self.codebooks_v1[0](x[len(x) // 2 :])
        code = torch.cat([first[2], second[2]])
        self._last_style_spatials = {0: x.clone()}
        result = (x, [], [torch.cat([x, x], 1)], None, [], [code], {0: torch.tensor([[1, 0]])}, {})
        return result, {0: x.clone()}

    def decode_codes(self, quantized_codes, styles, **kwargs):
        return self.weight * quantized_codes[0] + (1 - self.weight) * styles[0]


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.xa = np.zeros((8,) * 3)
        self.xb = self.xa.copy()
        self.xb[3:5, 3:5, 3:5] = -2
        self.support = self.xb != 0
        self.foreground = np.ones_like(self.support)

    def score(self, content_weight):
        y = {"aa": self.xa, "bb": self.xb, "ba": content_weight * self.xb, "ab": (1 - content_weight) * self.xb}
        return score_swaps(self.xa, self.xb, y, self.support, self.foreground)

    def test_content_style_and_mixed_routes_with_negative_contrast(self):
        for weight in (0, 0.3, 1):
            r = self.score(weight)
            self.assertAlmostEqual(r["content_mean_gain"], weight)
            self.assertAlmostEqual(r["style_mean_gain"], 1 - weight)
            self.assertAlmostEqual(r["joint_gain"], 1)
            self.assertAlmostEqual(r["joint_cosine"], 1)
            self.assertAlmostEqual(r["joint_relative_error"], 0)
            self.assertAlmostEqual(r["interaction_rms_ratio"], 0)

    def test_interaction_and_constant_decoder_are_not_exclusive_routes(self):
        y = {"aa": self.xa, "ba": self.xa, "ab": self.xa, "bb": self.xb}
        r = score_swaps(self.xa, self.xb, y, self.support, self.foreground)
        self.assertEqual(r["content_at_style_a_gain"], 0)
        self.assertEqual(r["content_at_style_b_gain"], 1)
        self.assertEqual(r["style_at_content_a_gain"], 0)
        self.assertEqual(r["style_at_content_b_gain"], 1)
        self.assertEqual(r["interaction_gain"], 1)
        self.assertEqual(r["content_mean_gain"] + r["style_mean_gain"], r["joint_gain"])
        r = score_swaps(self.xa, self.xb, dict.fromkeys(y, self.xa), self.support, self.foreground)
        self.assertEqual(r["joint_gain"], 0)
        self.assertEqual(r["joint_relative_error"], 1)
        self.assertTrue(np.isnan(r["joint_cosine"]))

    def test_invisible_intervention_and_invalid_outputs(self):
        y = dict.fromkeys(("aa", "ba", "ab", "bb"), self.xa)
        r = score_swaps(self.xa, self.xa, y, self.support * False, self.foreground)
        self.assertFalse(r["valid_input"])
        self.assertTrue(np.isnan(r["style_mean_gain"]))
        y["ab"] = self.xa + np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            score_swaps(self.xa, self.xb, y, self.support, self.foreground)


class ReplayValidationTests(unittest.TestCase):
    def test_sparse_cuda_sized_error_passes_only_with_resolved_local_signal(self):
        reference = np.full((2, 32, 32, 32), 0.424446)
        reference[0, 16, 16, 16] = 0
        replay = reference.copy()
        replay[0, 16, 16, 16] += 0.00025034
        roi = np.zeros((32,) * 3, bool)
        roi[14:19, 14:19, 14:19] = True
        delta = roi * 0.1
        self.assertFalse(np.allclose(replay, reference, atol=2e-5, rtol=2e-4))
        resolved = assess_endpoint_replay(replay, reference, delta, roi)
        self.assertTrue(resolved["endpoint_signal_resolved"])
        self.assertLess(resolved["endpoint_error_to_input_ratio"], 0.01)
        tiny_signal = assess_endpoint_replay(replay, reference, delta * 1e-4, roi)
        self.assertFalse(tiny_signal["endpoint_signal_resolved"])
        self.assertGreater(tiny_signal["endpoint_error_to_input_ratio"], 1)
        absent_signal = assess_endpoint_replay(replay, reference, delta * 0, roi)
        self.assertFalse(absent_signal["endpoint_signal_resolved"])

    def test_substantial_endpoint_mismatch_still_raises(self):
        reference = np.ones((2, 8, 8, 8))
        replay = reference.copy()
        replay[1] += 0.1
        with self.assertRaisesRegex(ValueError, "substantial mismatch"):
            assess_endpoint_replay(replay, reference, np.ones((8,) * 3), np.ones((8,) * 3, bool))

    def test_cuda_flags_are_restored_even_on_error(self):
        def flags():
            return (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.enabled,
            )

        before = flags()
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with stable_replay_math():
                self.assertEqual(flags()[:4], (False, False, False, True))
                self.assertEqual(flags()[4], before[4])
                raise RuntimeError("injected failure")
        self.assertEqual(flags(), before)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.dataset_module = load_without_monai("data/datasets.py")
        cls.Model = load_without_monai("models/vqvae.py").VQVAE

    def dataset(self, normalization="per_sample", **kwargs):
        settings = dict(
            synthetic_mode="pseudo_mri",
            synthetic_res=32,
            synthetic_n_content=9,
            synthetic_content_prior="uniform",
            synthetic_content_squash="none",
            synthetic_clean_content=True,
            synthetic_identifiable_ventricle=True,
            synthetic_normalize=normalization,
            synthetic_causal=True,
        )
        settings.update(kwargs)
        with patch.dict("sys.modules", {"data.datasets": self.dataset_module}):
            return make_dataset(argparse.Namespace(**settings), 3, "iid", "test")

    def test_real_rendering_freezes_noise_factors_and_normalization(self):
        for normalization in ("per_sample", "shared", "fixed_reference"):
            ds = self.dataset(normalization)
            self.assertFalse(ds._inner.causal)
            with patch.object(ds._inner, "render_pseudo_mri", wraps=ds._inner.render_pseudo_mri) as render:
                sample = render_pair(ds, 0, eps=0.5)
                low, high = render.call_args_list[-2:]
                delta = high.args[0] - low.args[0]
                torch.testing.assert_close(delta[1], torch.tensor(1.0))
                torch.testing.assert_close(delta[torch.arange(9) != 1], torch.zeros(8))
                for a, b in zip(low.args[1:], high.args[1:]):
                    if isinstance(a, torch.Tensor):
                        self.assertTrue(torch.equal(a, b))
                    else:
                        self.assertEqual(a, b)
            again = render_pair(ds, 0, eps=0.5)
            self.assertGreater(sample["support"].sum(), 0)
            outside = ~maximum_filter(sample["support"], size=3)
            for v in range(2):
                self.assertTrue(torch.equal(sample["a"][v], again["a"][v]))
                diff = (sample["b"][v] - sample["a"][v]).numpy()[0]
                self.assertLess(np.abs(diff[outside]).max(), 2e-6)

    def test_end_to_end_oracles_and_partial_batch(self):
        ds = self.dataset()
        for weight in (0, 0.3, 1):
            rows, summary, panels = audit(RoutingOracle(weight), ds, "cpu", eps=0.5, batch_size=2, examples=1)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(panels), 1)
            for view in ("t1", "flair"):
                self.assertGreater(summary[view]["n_valid_input"], 0)
            for row in rows:
                if row["valid_input"]:
                    self.assertAlmostEqual(row["joint_gain"], 1, places=4)
                    self.assertAlmostEqual(row["content_mean_gain"], weight, places=4)
                    self.assertAlmostEqual(row["style_mean_gain"], 1 - weight, places=4)

    def test_endpoint_guard_and_training_guard(self):
        ds = self.dataset()
        samples = [render_pair(ds, 0, eps=0.5)]
        model = RoutingOracle(0.5)
        decode = model.decode_codes
        with patch.object(model, "decode_codes", side_effect=lambda *a, **kw: decode(*a, **kw) + 1):
            with self.assertRaisesRegex(ValueError, "endpoint"):
                decode_swaps(model, samples, "cpu")
        model.train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            decode_swaps(model, samples, "cpu")

    def test_unresolved_rows_are_retained_but_excluded_from_routing_summary(self):
        class SmallReplayOffset(RoutingOracle):
            def decode_codes(self, *args, **kwargs):
                return super().decode_codes(*args, **kwargs) + 2e-6

        ds = self.dataset()

        def tiny_pair(dataset, index, eps):
            sample = render_pair(dataset, index, eps)
            sample["b"] = [x + sample["mask"] * 1e-6 for x in sample["a"]]
            return sample

        with patch("eval.ventricle_routing.render_pair", side_effect=tiny_pair):
            rows, summary, _ = audit(SmallReplayOffset(1), ds, "cpu", eps=0.5, examples=0)
        self.assertTrue(any(r["valid_input"] for r in rows))
        self.assertFalse(any(r["valid_routing"] for r in rows))
        for view in summary.values():
            self.assertEqual(view["n"], 3)
            self.assertEqual(view["n_valid_routing"], 0)
            self.assertEqual(view["metrics"]["style_mean_gain"]["n_valid"], 0)
            self.assertIsNone(view["metrics"]["style_mean_gain"]["median"])

    def test_decoder_batch_geometry_matches_original_forward(self):
        class BatchDependentOracle(RoutingOracle):
            def forward(self, x, **kwargs):
                out, pre = super().forward(x, **kwargs)
                return (out[0] + len(x) * 0.01, *out[1:]), pre

            def decode_codes(self, quantized_codes, **kwargs):
                return super().decode_codes(quantized_codes, **kwargs) + len(quantized_codes[0]) * 0.01

        ds = self.dataset()
        for weight in (0, 1):
            rows, _, _ = audit(BatchDependentOracle(weight), ds, "cpu", eps=0.5, batch_size=2, examples=0)
            for row in rows:
                if row["valid_input"]:
                    self.assertAlmostEqual(row["joint_gain"], 1, places=5)
                    self.assertAlmostEqual(row["content_mean_gain"], weight, places=5)
                    self.assertAlmostEqual(row["endpoint_replay_max_abs"], 0, places=6)

    def test_replay_uses_actual_ste_values_when_id_lookup_loses_precision(self):
        torch.manual_seed(29)
        model = self.Model(
            hidden_channels=16,
            res_channels=8,
            nb_res_layers=2,
            nb_levels=1,
            embed_dim=16,
            nb_entries=32,
            scaling_rates=[4],
            content_size=12,
            style_size=4,
            content_style_levels=[0],
            mask_mode="fixed",
            inject_style_to_decoder=True,
            style_injection_mode="input",
            quantize_style=True,
            norm_type="layer",
            decoder_norm_type="group",
            final_recon_norm=False,
            use_checkpoint=False,
        ).eval()
        with torch.no_grad():
            model.codebooks[0].conv_in.weight.mul_(10000)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        samples = [
            {
                "a": [torch.randn(1, 32, 32, 32) for _ in range(2)],
                "b": [torch.randn(1, 32, 32, 32) for _ in range(2)],
                "mask": torch.ones(1, 32, 32, 32),
                "support": np.ones((32,) * 3, dtype=bool),
            }
            for _ in range(2)
        ]
        decoded, diagnostics = decode_swaps(model, samples, "cpu")
        exact = torch.from_numpy(np.concatenate([decoded[v][k] for v in range(2) for k in ("aa", "bb")]))[:, None]
        with torch.inference_mode():
            # Legacy API remains available, but lookup alone is not exact forward replay.
            lookup = model.decode_codes(
                model._last_id_outputs[0], styles=dict(model._last_style_spatials), target_spatial_size=(32,) * 3
            )
        self.assertFalse(torch.allclose(lookup, exact, atol=2e-5, rtol=2e-4))
        self.assertGreater(float((lookup - exact).abs().max()), 2e-5)
        for view in diagnostics:
            self.assertLess(float(view["endpoint_replay_max_abs"].max()), 2e-6)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(len(model.codebooks[0]._forward_hooks), 0)
        with self.assertRaisesRegex(ValueError, "every model level"):
            model.decode_codes(quantized_codes={})

    def test_capture_hooks_are_removed_if_forward_fails(self):
        model = RoutingOracle(1)
        samples = [render_pair(self.dataset(), 0)]
        with patch.object(model, "forward", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                decode_swaps(model, samples, "cpu")
        self.assertEqual(len(model.codebooks[0]._forward_hooks), 0)
        self.assertEqual(len(model.codebooks_v1[0]._forward_hooks), 0)

    def test_cli_writes_report_rows_and_panels(self):
        ds = self.dataset()
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "ventricle_routing",
                "--run-dir",
                "oracle",
                "--out-dir",
                tmp,
                "--num-samples",
                "3",
                "--eps",
                "0.5",
                "--examples",
                "1",
            ]
            with (
                patch("sys.argv", argv),
                patch(
                    "eval.run_dci_synthetic.load_model_from_run_dir",
                    return_value=(RoutingOracle(0), argparse.Namespace(), "cpu"),
                ),
                patch("eval.ventricle_routing.make_dataset", return_value=ds),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                main()
            report = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(report["summary"]["t1"]["n"], 3)
            self.assertAlmostEqual(report["summary"]["t1"]["metrics"]["style_mean_gain"]["median"], 1, places=5)
            with (Path(tmp) / "samples.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)
            self.assertTrue((Path(tmp) / "examples.png").read_bytes().startswith(b"\x89PNG"))
            self.assertIn("joint_cosine", output.getvalue())

    def test_real_model_multilevel_and_separate_codebooks_preserve_state(self):
        for separate, quantized, levels in ((False, False, 1), (True, True, 1), (True, True, 2)):
            with self.subTest(separate=separate, quantized=quantized, levels=levels):
                torch.manual_seed(29)
                model = self.Model(
                    hidden_channels=8,
                    res_channels=4,
                    nb_res_layers=1,
                    nb_levels=levels,
                    embed_dim=8,
                    nb_entries=8,
                    scaling_rates=[2] * levels,
                    content_size=6,
                    style_size=2,
                    content_style_levels=list(range(levels)),
                    mask_mode="fixed",
                    inject_style_to_decoder=True,
                    style_injection_mode="input",
                    quantize_style=quantized,
                    separate_encoders=separate,
                    separate_content_codebooks=separate,
                    separate_style_codebooks=separate,
                    norm_type="layer",
                    final_recon_norm=False,
                    use_checkpoint=False,
                ).eval()
                before = {k: v.clone() for k, v in model.state_dict().items()}
                samples = [
                    {
                        "a": [torch.randn(1, 8, 8, 8) for _ in range(2)],
                        "b": [torch.randn(1, 8, 8, 8) for _ in range(2)],
                        "mask": torch.ones(1, 8, 8, 8),
                        "support": np.ones((8,) * 3, dtype=bool),
                    }
                    for _ in range(2)
                ]
                decoded, diagnostics = decode_swaps(model, samples, "cpu")
                for view in range(2):
                    self.assertEqual(decoded[view]["ab"].shape, (2, 8, 8, 8))
                    self.assertIn(f"content_post_L{levels - 1}_delta_rms", diagnostics[view])
                    self.assertIn(f"style_post_L{levels - 1}_delta_rms", diagnostics[view])
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]), key)


if __name__ == "__main__":
    unittest.main()
