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

from eval.ventricle_routing import audit, decode_swaps, main, make_dataset, render_pair, score_swaps

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


class RoutingOracle(torch.nn.Module):
    """Identity endpoints with a known fraction of the response carried by content."""

    def __init__(self, content_weight):
        super().__init__()
        self.weight = content_weight
        self.nb_levels = 1
        self.inject_style_to_decoder = True
        self.separate_content_codebooks = True
        self.codebooks = [types.SimpleNamespace(embed_code=lambda x: x)]
        self.codebooks_v1 = [types.SimpleNamespace(embed_code=lambda x: x + 10)]
        self.eval()

    def forward(self, x, **kwargs):
        assert kwargs["n_views"] == 2
        code = x[:, 0].clone()
        code[len(x) // 2 :] -= 10
        self._last_style_spatials = {0: x.clone()}
        result = (x, [], [torch.cat([x, x], 1)], None, [], [code], {0: torch.tensor([[1, 0]])}, {})
        return result, {0: x.clone()}

    def decode_codes(self, code, styles, content_view_idx, **kwargs):
        content = code[:, None] + 10 * content_view_idx
        return self.weight * content + (1 - self.weight) * styles[0]


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
