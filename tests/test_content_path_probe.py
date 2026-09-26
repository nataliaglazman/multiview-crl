"""Real-forward stage identity, quantizer controls, and end-to-end frozen audit."""

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

from eval.content_path_probe import main, mean_descriptor, stage_maps, state_digest
from eval.pooling_probe import native_maps


def real_model(**kwargs):
    source = Path(__file__).resolve().parents[1] / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "utils.utils" for a in n.names))
    ]
    namespace = {"__name__": "content_path_probe_test_model"}
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


class StageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)
        self.threads = threadpool_limits(limits=2)
        self.threads.__enter__()
        self.x = torch.randn(6, 1, 16, 16, 16)
        self.mask = torch.zeros_like(self.x)
        self.mask[..., 4:12, 4:12, 4:12] = 1

    def tearDown(self):
        self.threads.__exit__(None, None, None)

    def test_stages_match_real_forward_for_shared_and_separate_paths(self):
        for separate in (False, True):
            for quantized_style in (False, True):
                with self.subTest(separate=separate, quantized_style=quantized_style):
                    model = real_model(
                        separate_encoders=separate,
                        separate_content_codebooks=separate,
                        separate_style_codebooks=separate,
                        quantize_style=quantized_style,
                        style_spatial_size=2,
                        latent_mask=True,
                    )
                    before = state_digest(model)
                    stages, partitions = stage_maps(model, self.x, self.mask)
                    reference, _ = native_maps(model, self.x, self.mask, 0)
                    torch.testing.assert_close(stages[("content", "pre_norm")], reference[:, :6])
                    torch.testing.assert_close(stages[("style", "pre_norm")], reference[:, 6:])
                    self.assertEqual(partitions[0].sum(), 6)
                    post = model.content_norms["0"](reference)
                    torch.testing.assert_close(stages[("content", "post_norm")], post[:, :6])
                    torch.testing.assert_close(stages[("style", "post_norm")], post[:, 6:])
                    self.assertEqual(stages[("content", "pre_quant")].shape, (6, 8, 8, 8, 8))
                    self.assertEqual(stages[("content", "decoder_input")].shape[1], 8)  # all embed coordinates
                    for v in range(2):
                        sl = slice(v * 3, (v + 1) * 3)
                        cb = model.codebooks_v1[0] if separate and v else model.codebooks[0]
                        with torch.inference_mode():
                            expected = cb.project(post[sl, :6])
                            quantized = cb.quantize(expected)[0]
                        torch.testing.assert_close(stages[("content", "pre_quant")][sl], expected)
                        torch.testing.assert_close(stages[("content", "decoder_input")][sl], quantized)
                    self.assertEqual(stages[("style", "bottleneck")].shape, (6, 2, 2, 2, 2))
                    self.assertEqual(("style", "pre_quant") in stages, quantized_style)
                    if not quantized_style:
                        torch.testing.assert_close(stages[("style", "decoder_input")], stages[("style", "bottleneck")])
                    self.assertEqual(before, state_digest(model))
                    self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
                    self.assertNotIn("_bottleneck_style", model.__dict__)

    def test_degenerate_codebook_loses_signal_only_after_projection(self):
        model = real_model()
        with torch.no_grad():
            model.codebooks[0].embed.zero_()
        before = state_digest(model)
        stages, _ = stage_maps(model, self.x, self.mask)
        self.assertGreater(float(stages[("content", "pre_quant")].var()), 0.01)
        self.assertEqual(float(stages[("content", "decoder_input")].abs().max()), 0)
        self.assertEqual(before, state_digest(model))

    def test_style_global_bottleneck_does_not_create_fake_spatial_features(self):
        model = real_model(style_spatial_size=1)
        stages, _ = stage_maps(model, self.x, self.mask)
        descriptor, effective = mean_descriptor(stages[("style", "decoder_input")], 8)
        self.assertEqual(effective, 1)
        self.assertEqual(descriptor.shape, (6, 2))
        descriptor, effective = mean_descriptor(stages[("content", "pre_norm")], 8)
        self.assertEqual(effective, 8)
        self.assertEqual(descriptor.shape, (6, 6 * 512))
        with self.assertRaises(ValueError):
            mean_descriptor(stages[("content", "pre_norm")], 3)

    def test_hooks_and_style_wrapper_removed_after_failure(self):
        model = real_model(quantize_style=True)
        before = state_digest(model)
        with patch.object(model.decoders[0], "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                stage_maps(model, self.x, self.mask)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertNotIn("_bottleneck_style", model.__dict__)
        self.assertEqual(before, state_digest(model))

    def test_train_mode_rejected_before_forward(self):
        model = real_model().train()
        before = state_digest(model)
        with self.assertRaisesRegex(ValueError, "eval mode"):
            stage_maps(model, self.x, self.mask)
        self.assertEqual(before, state_digest(model))

    def test_full_cli_real_renderer_shared_splits_valid_reports_and_unchanged_checkpoint(self):
        args = argparse.Namespace(
            synthetic_mode="pseudo_mri",
            synthetic_res=16,
            synthetic_clean_content=True,
            synthetic_normalize="fixed_reference",
            synthetic_n_content=9,
            synthetic_n_style=3,
        )
        model = real_model(
            quantize_style=True, separate_content_codebooks=True, separate_style_codebooks=True, style_spatial_size=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 99}, checkpoint)
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
                        "4",
                        "--threads",
                        "2",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["model_state_unchanged"])
            self.assertEqual(summary["checkpoint_step"], 99)
            self.assertEqual(len(summary["scores"]), 2 * 9 * 2 * 2 * 6)
            self.assertEqual(summary["stage_metadata"]["flair_style_decoder_input_g4"]["effective_grid"], 1)
            saved = np.load(output / "predictions.npz")
            indices = np.concatenate([saved[f"{s}_indices"] for s in ("train", "validation", "test")])
            self.assertEqual(len(np.unique(indices)), 40)
            self.assertEqual(saved["flair_content_decoder_input_g4_ridge_observed"].shape, (8, 6))
            quant = next(
                r
                for r in summary["scores"]
                if r["block"] == "content" and r["stage"] == "decoder_input" and r["condition"] == "observed"
            )
            self.assertEqual(quant["reference_stage"], "pre_quant")
            first = next(r for r in summary["scores"] if r["stage"] == "pre_norm" and r["condition"] == "observed")
            self.assertEqual(first["delta_vs_previous"], 0)


if __name__ == "__main__":
    unittest.main()
