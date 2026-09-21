"""Decoder dependence controls and real VQ-VAE/rendering integration."""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.ventricle_decoder_audit import (
    effects_from_endpoints,
    main,
    make_dataset,
    render_pair,
    score_response,
    state_digest,
    swap_batch,
)


class ToyDecoder(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, content, style):
        if self.mode == "content":
            return content
        if self.mode == "style":
            return style
        if self.mode == "interaction":
            return content * style
        return content * 0


class ToyModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.nb_levels = 1
        self.content_style_levels = [0]
        self.inject_style_to_decoder = True
        self.mask_mode = "fixed"
        self.decoders = torch.nn.ModuleList([ToyDecoder(mode)])
        self._last_style_id_outputs = {}
        self.eval()

    def _bottleneck_style(self, x):
        return x

    def forward(self, x, **kwargs):
        return (self.decoders[0](x, style=self._bottleneck_style(x)),)


def real_model(**kwargs):
    source = Path(__file__).resolve().parents[1] / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "utils.utils" for a in n.names))
    ]
    namespace = {"__name__": "ventricle_decoder_test_model"}
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


class DecoderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def sample(self):
        mask = torch.ones(1, 4, 4, 4)
        return dict(low=[mask * 0.2, mask * 0.3], high=[mask * 0.7, mask * 0.9], mask=mask)

    def test_known_content_style_and_nonresponding_decoders(self):
        sample = self.sample()
        for mode in ("content", "style", "zero"):
            decoded, error, repeat = swap_batch(ToyModel(mode), [sample] * 3, "cpu")
            effects = effects_from_endpoints(**decoded)
            for v in range(2):
                j = v * 3
                row = score_response(
                    sample["high"][v] - sample["low"][v],
                    {k: x[j] for k, x in effects.items()},
                    error[j],
                    repeat[j],
                    sample["mask"].bool(),
                )
                self.assertTrue(row["resolved_input"])
                self.assertAlmostEqual(row["joint_gain"], float(mode != "zero"), places=6)
                self.assertAlmostEqual(row["content_mean_gain"], float(mode == "content"), places=6)
                self.assertAlmostEqual(row["style_mean_gain"], float(mode == "style"), places=6)
                self.assertAlmostEqual(row["interaction_rms"], 0, places=6)
                self.assertEqual(row["endpoint_replay_rms"], 0)
                if mode == "zero":
                    self.assertEqual(row["joint_relative_error"], 1)
                    self.assertIsNone(row["joint_cosine"])

    def test_interaction_contexts_and_additive_attribution_identity(self):
        sample = self.sample()
        decoded, _, _ = swap_batch(ToyModel("interaction"), [sample], "cpu")
        e = effects_from_endpoints(**decoded)
        torch.testing.assert_close(e["content_mean"] + e["style_mean"], e["joint"])
        self.assertGreater(float(e["interaction"].abs().mean()), 0.2)
        self.assertFalse(torch.allclose(e["content_at_low_style"], e["content_at_high_style"]))

    def test_empty_zero_input_and_numerical_error_do_not_claim_routing(self):
        x = torch.ones(1, 4, 4, 4)
        effects = effects_from_endpoints(x * 0, x, x, x * 2)
        for target, error, region in ((x * 0, x * 0, x.bool()), (x, x * 0.2, x.bool()), (x, x * 0, x.bool() & False)):
            row = score_response(target, effects, error, x * 0, region)
            self.assertFalse(row["resolved_input"])
            self.assertIsNone(row["joint_gain"])
            self.assertIsNone(row["joint_relative_error"])
        row = score_response(x * 0, effects, x * 0, x * 0, x.bool())
        self.assertEqual(row["joint_rms"], 2)  # Outside-support spill remains visible.

    def test_bad_endpoint_is_rejected_and_hooks_removed(self):
        model = ToyModel("content")
        with patch("eval.ventricle_decoder_audit.replay", side_effect=lambda m, c, s, spatial: c + 1):
            with self.assertRaisesRegex(ValueError, "endpoint replay failed"):
                swap_batch(model, [self.sample()], "cpu")
        self.assertNotIn("_bottleneck_style", model.__dict__)
        self.assertFalse(model.decoders[0]._forward_pre_hooks)

    def test_real_model_continuous_global_and_quantized_spatial_style(self):
        ds = make_dataset(dataset_args(), 2, "iid", "test")
        sample = render_pair(ds, 0, 0.5)
        self.assertGreater(sample["support_voxels"], 0)
        for quantized, size, separate in ((False, 1, False), (True, 0, True)):
            model = real_model(
                quantize_style=quantized,
                style_spatial_size=size,
                separate_encoders=separate,
                separate_content_codebooks=separate,
                separate_style_codebooks=separate,
            )
            before = state_digest(model)
            decoded, error, repeat = swap_batch(model, [sample], "cpu")
            self.assertEqual(decoded["lh"].shape, (2, 1, 32, 32, 32))
            self.assertLess(float(error.max()), 1e-5)
            self.assertEqual(float(repeat.max()), 0)
            self.assertEqual(state_digest(model), before)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_full_cli_reports_regions_reconstruction_and_leaves_checkpoint_unchanged(self):
        model = real_model(quantize_style=False, style_spatial_size=1)
        before = state_digest(model)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 97001}, checkpoint)
            file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
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
            self.assertEqual(report["checkpoint_step"], 97001)
            self.assertTrue(report["model_state_unchanged"])
            self.assertEqual(len(report["summary"]), 12)
            with (out / "responses.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * 2 * 2 * 3)
            for row in rows:
                if row["region"] == "outside":
                    self.assertEqual(row["joint_gain"], "")
                if row["resolved_input"] == "True":
                    self.assertAlmostEqual(
                        float(row["content_mean_gain"]) + float(row["style_mean_gain"]),
                        float(row["joint_gain"]),
                        places=5,
                    )
            maps = np.load(out / "examples.npz")
            self.assertEqual(maps["eps0.25_sample0_flair_hl"].shape, (1, 32, 32, 32))
            self.assertEqual(len(list(out.glob("*.png"))), 4)
            self.assertEqual(state_digest(model), before)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), file_hash)


if __name__ == "__main__":
    unittest.main()
