"""Lesion routing: planted spatial codes, controlled rendering and real VQVAE replay."""

import argparse
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
from test_ventricle_routing import RoutingOracle, load_without_monai

from eval.lesion_routing import (
    audit,
    main,
    make_dataset,
    render_pair,
    score_pair,
    state_digest,
    summarize,
    verify_checkpoint,
)
from eval.ventricle_routing import decode_swaps


def translated_lesion():
    a, b = torch.zeros(1, 12, 12, 12), torch.zeros(1, 12, 12, 12)
    a[:, 2:4, 5:7, 5:7], b[:, 8:10, 5:7, 5:7] = 1, 1
    lesions = [x[0].numpy() > 0 for x in (a, b)]
    return {
        "index": 0,
        "axis": "x",
        "eps": 0.5,
        "a": [a, -a],
        "b": [b, -b],
        "mask": torch.ones_like(a),
        "support": lesions[0] != lesions[1],
        "lesions": lesions,
        "centroids": [np.argwhere(m).mean(0) for m in lesions],
        "controls": [np.zeros(3), np.ones(3)],
    }


class SpatialCodeTests(unittest.TestCase):
    def test_perfect_reconstruction_and_routing_despite_zero_gap_signal(self):
        sample = translated_lesion()
        for weight in (0, 0.3, 1):
            decoded, diagnostics = decode_swaps(RoutingOracle(weight), [sample], "cpu", measure_gap=True)
            for view in range(2):
                row = score_pair(
                    sample,
                    view,
                    {k: v[0] for k, v in decoded[view].items()},
                    diagnostics[view],
                    0,
                )
                self.assertTrue(row["valid_routing"])
                self.assertAlmostEqual(row["joint_gain"], 1, places=6)
                self.assertAlmostEqual(row["content_mean_gain"], weight, places=6)
                self.assertAlmostEqual(row["style_mean_gain"], 1 - weight, places=6)
                self.assertAlmostEqual(row["joint_energy_in_affected_fraction"], 1)
                self.assertEqual(row["joint_outside_rms"], 0)
                for block in ("content", "style"):
                    self.assertGreater(row[f"{block}_post_L0_delta_rms"], 0)
                    self.assertEqual(row[f"{block}_post_L0_gap_delta_rms"], 0)

    def test_joint_interaction_is_reported_even_with_perfect_endpoints(self):
        class Interaction(RoutingOracle):
            def decode_codes(self, quantized_codes, styles, **kwargs):
                return quantized_codes[0] * styles[0].abs()

        sample = translated_lesion()
        decoded, diagnostics = decode_swaps(Interaction(0.5), [sample], "cpu", measure_gap=True)
        row = score_pair(sample, 0, {k: v[0] for k, v in decoded[0].items()}, diagnostics[0], 0)
        self.assertAlmostEqual(row["joint_gain"], 1)
        self.assertAlmostEqual(row["content_mean_gain"], 0.5)
        self.assertAlmostEqual(row["style_mean_gain"], 0.5)
        self.assertAlmostEqual(row["interaction_rms_ratio"], 1)

    def test_constant_decoder_and_unresolved_or_empty_pairs_do_not_claim_a_route(self):
        class Constant(RoutingOracle):
            def forward(self, x, **kwargs):
                out, style = super().forward(x, **kwargs)
                return (out[0] * 0, *out[1:]), style

            def decode_codes(self, quantized_codes, **kwargs):
                return quantized_codes[0] * 0

        sample = translated_lesion()
        decoded, diagnostics = decode_swaps(Constant(1), [sample], "cpu", measure_gap=True)
        row = score_pair(sample, 0, {k: v[0] for k, v in decoded[0].items()}, diagnostics[0], 0)
        self.assertEqual(row["joint_gain"], 0)
        self.assertEqual(row["joint_relative_error"], 1)
        self.assertTrue(np.isnan(row["joint_cosine"]))
        diagnostics[0]["endpoint_signal_resolved"][0] = False
        unresolved = score_pair(sample, 0, {k: v[0] for k, v in decoded[0].items()}, diagnostics[0], 0)
        group = summarize([unresolved])["x/t1"]
        self.assertEqual(group["resolved"], 0)
        self.assertIsNone(group["metrics"]["joint_gain"]["mean"])
        sample["lesions"][0][:] = False
        diagnostics[0]["endpoint_signal_resolved"][0] = True
        empty = score_pair(sample, 0, {k: v[0] for k, v in decoded[0].items()}, diagnostics[0], 0)
        self.assertFalse(empty["valid_routing"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.dataset_module = load_without_monai("data/datasets.py")
        cls.Model = load_without_monai("models/vqvae.py").VQVAE

    def settings(self, normalization="fixed_reference"):
        return argparse.Namespace(
            synthetic_mode="pseudo_mri",
            synthetic_res=32,
            synthetic_n_content=9,
            synthetic_clean_content=True,
            synthetic_normalize=normalization,
            synthetic_causal=False,
            synthetic_lesion_radius=0.14,
        )

    def dataset(self, normalization="fixed_reference"):
        with patch.dict("sys.modules", {"data.datasets": self.dataset_module}):
            return make_dataset(self.settings(normalization), 3)

    def test_renderer_changes_only_requested_coordinate_and_preserves_noise_and_normalization(
        self,
    ):
        for normalization in ("fixed_reference", "shared", "per_sample"):
            ds = self.dataset(normalization)
            for axis in "xyz":
                with patch.object(ds._inner, "render_pseudo_mri", wraps=ds._inner.render_pseudo_mri) as render:
                    sample = render_pair(ds, 0, axis, 0.5)
                    a, b = render.call_args_list[-2:]
                    expected = torch.zeros(9)
                    expected[2 + "xyz".index(axis)] = 1
                    torch.testing.assert_close(b.args[0] - a.args[0], expected)
                    for first, second in zip(a.args[1:], b.args[1:]):
                        if isinstance(first, torch.Tensor):
                            self.assertTrue(torch.equal(first, second))
                        else:
                            self.assertEqual(first, second)
                repeated = render_pair(ds, 0, axis, 0.5)
                for view in range(2):
                    self.assertTrue(torch.equal(sample["a"][view], repeated["a"][view]))
                    delta = (sample["b"][view] - sample["a"][view]).numpy()[0]
                    self.assertLess(
                        np.abs(delta[~maximum_filter(sample["support"], size=3)]).max(),
                        2e-6,
                    )
                self.assertEqual(sample["a"][0].shape, (1, 32, 32, 32))

    def test_real_dataset_oracle_partial_batches_and_registered_state_guard(self):
        ds = self.dataset()
        rows, summary = audit(RoutingOracle(0.3), ds, "cpu", batch_size=2)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(summary), 6)
        self.assertTrue(any(r["valid_routing"] for r in rows))
        for row in rows:
            if row["valid_routing"]:
                self.assertAlmostEqual(row["joint_gain"], 1, places=5)
                self.assertAlmostEqual(row["content_mean_gain"], 0.3, places=5)

        class Mutating(RoutingOracle):
            def __init__(self):
                super().__init__(1)
                self.register_buffer("counter", torch.zeros((), dtype=torch.long))

            def forward(self, x, **kwargs):
                self.counter.add_(1)
                return super().forward(x, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "changed a registered"):
            audit(Mutating(), ds, "cpu", axes=("x",))

    def model(self, global_style):
        return self.Model(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            nb_levels=1,
            embed_dim=8,
            nb_entries=16,
            scaling_rates=[2],
            content_size=4,
            style_size=4,
            content_style_levels=[0],
            mask_mode="fixed",
            inject_style_to_decoder=True,
            style_injection_mode="input",
            quantize_style=not global_style,
            separate_encoders=True,
            separate_content_codebooks=True,
            separate_style_codebooks=not global_style,
            style_spatial_size=int(global_style),
            final_recon_norm=False,
            use_checkpoint=False,
        ).eval()

    def test_real_vqvae_spatial_and_global_style_exact_replay(self):
        ds = self.dataset()
        for global_style in (False, True):
            model = self.model(global_style)
            before = state_digest(model)
            rows, _ = audit(model, ds, "cpu", axes=("x",), batch_size=2)
            self.assertEqual(before, state_digest(model))
            self.assertTrue(all(r["endpoint_replay_rms"] < 1e-6 for r in rows))
            if global_style:
                for r in rows:
                    self.assertAlmostEqual(r["style_post_L0_delta_rms"], r["style_post_L0_gap_delta_rms"])

    def test_cli_reports_images_nifti_checkpoint_integrity_and_mismatch_rejection(self):
        import nibabel as nib

        model = self.model(False)
        loader = types.ModuleType("eval.run_dci_synthetic")
        loader.load_model_from_run_dir = lambda *a, **kw: (
            model,
            self.settings(),
            "cpu",
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 123}, checkpoint)
            original = checkpoint.read_bytes()
            output = Path(tmp) / "audit"
            argv = [
                "lesion_routing",
                "--run-dir",
                tmp,
                "--num-samples",
                "3",
                "--axes",
                "x",
                "--examples",
                "1",
                "--save-nifti",
                "--out-dir",
                str(output),
            ]
            with patch.dict(
                "sys.modules",
                {
                    "eval.run_dci_synthetic": loader,
                    "data.datasets": self.dataset_module,
                },
            ), patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["checkpoint"]["step"], 123)
            self.assertTrue(report["registered_state_unchanged"])
            self.assertEqual(checkpoint.read_bytes(), original)
            with (output / "responses.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            for view in ("t1", "flair"):
                prefix = output / f"sample0000_x_{view}"
                self.assertGreater(prefix.with_suffix(".png").stat().st_size, 1000)
                volume = nib.load(str(prefix) + "_input_delta.nii.gz")
                self.assertEqual(volume.shape, (32, 32, 32))
                np.testing.assert_equal(volume.affine, np.eye(4))
            with torch.no_grad():
                next(model.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "does not exactly match"):
                verify_checkpoint(model, checkpoint)


if __name__ == "__main__":
    unittest.main()
