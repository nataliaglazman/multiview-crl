"""Lesion donor semantics, known routing oracles, exact replay and frozen checkpoints."""

import argparse
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
import sklearn  # noqa: F401 - initialize compiled dependencies before module patches
import torch
from scipy.ndimage import maximum_filter

from eval import lesion_routing as lr

ROOT = Path(__file__).resolve().parents[1]
# Reuse the existing exact-tensor oracle and dependency-light source loader. No
# tests are executed by importing these fixtures.
spec = importlib.util.spec_from_file_location("lesion_route_fixtures", ROOT / "tests/test_ventricle_routing.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class LesionRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.datasets = fixtures.load_without_monai("data/datasets.py")
        cls.Model = fixtures.load_without_monai("models/vqvae.py").VQVAE

    def settings(self, **overrides):
        args = dict(
            synthetic_mode="pseudo_mri",
            synthetic_res=32,
            synthetic_n_content=9,
            synthetic_content_prior="uniform",
            synthetic_content_squash="none",
            synthetic_clean_content=True,
            synthetic_identifiable_ventricle=True,
            synthetic_normalize="per_sample",
            synthetic_causal=True,
            synthetic_lesion_radius=0.14,
            vqvae_nb_levels=1,
            inject_style_to_decoder=True,
            mask_mode="fixed",
        )
        args.update(overrides)
        return argparse.Namespace(**args)

    def dataset(self, n=3, **overrides):
        with patch.dict("sys.modules", {"data.datasets": self.datasets}):
            return lr.make_dataset(self.settings(**overrides), n, "iid", "test")

    def test_donors_mean_off_then_on_and_freeze_normalization(self):
        for norm in ("per_sample", "shared", "fixed_reference"):
            ds = self.dataset(synthetic_normalize=norm)
            sample = lr.lesion_pair(ds, 0)
            on = ds[0]["image"]
            outside = ~maximum_filter(sample["support"], size=3)
            self.assertGreater(sample["support"].sum(), 0)
            for view in range(2):
                torch.testing.assert_close(sample["b"][view], on[view])
                delta = (sample["b"][view] - sample["a"][view]).numpy()[0]
                self.assertLess(float(np.abs(delta[outside]).max()), 2e-6)
                self.assertGreater(float(np.linalg.norm(delta)), 0)
            repeated = lr.lesion_pair(ds, 0)
            for state in ("a", "b"):
                for view in range(2):
                    torch.testing.assert_close(sample[state][view], repeated[state][view], atol=0, rtol=0)

    def test_content_style_and_mixed_oracles_with_partial_batch(self):
        ds = self.dataset()
        for weight in (0.0, 0.3, 1.0):
            rows, summary, panels = lr.audit(fixtures.RoutingOracle(weight), ds, "cpu", 2, 1, 20)
            self.assertEqual(len(rows), 6)
            self.assertEqual({p["view"] for p in panels}, {"t1", "flair"})
            for row in rows:
                self.assertTrue(row["valid_routing"])
                self.assertAlmostEqual(row["joint_gain"], 1, places=5)
                for style in ("a", "b"):
                    self.assertAlmostEqual(row[f"content_at_style_{style}_gain"], weight, places=5)
                    self.assertAlmostEqual(row[f"style_at_content_{style}_gain"], 1 - weight, places=5)
                self.assertAlmostEqual(row["interaction_rms_ratio"], 0, places=5)
                self.assertAlmostEqual(row["endpoint_replay_rms"], 0, places=6)
            for result in summary.values():
                m = result["metrics"]
                self.assertEqual(result["n_valid_routing"], 3)
                self.assertAlmostEqual(
                    m["content_mean_gain"]["mean"] + m["style_mean_gain"]["mean"], m["joint_gain"]["mean"]
                )

    def test_interaction_is_reported_in_both_donor_contexts(self):
        a = np.zeros((8, 8, 8))
        b = a.copy()
        b[3:5, 3:5, 3:5] = -1  # T1-like negative contrast
        # Lesion response appears ONLY when both codes come from the on donor.
        decoded = {"aa": a, "ba": a, "ab": a, "bb": b}
        effects = lr.response_images(a, b, decoded)
        scores = lr.score_swaps(a, b, decoded, b != 0, np.ones_like(a))
        self.assertEqual(scores["content_at_style_a_gain"], 0)
        self.assertEqual(scores["content_at_style_b_gain"], 1)
        self.assertEqual(scores["style_at_content_a_gain"], 0)
        self.assertEqual(scores["style_at_content_b_gain"], 1)
        self.assertEqual(scores["content_mean_gain"], 0.5)
        self.assertEqual(scores["style_mean_gain"], 0.5)
        self.assertEqual(scores["interaction_rms_ratio"], 1)
        np.testing.assert_array_equal(effects["content_mean"] + effects["style_mean"], effects["joint"])

    def test_empty_intervention_is_retained_but_not_given_a_route(self):
        ds = self.dataset()
        actual = lr.lesion_pair

        def invisible(dataset, index):
            sample = actual(dataset, index)
            sample["b"] = [x.clone() for x in sample["a"]]
            sample["support"][:] = False
            sample["centroid"][:] = np.nan
            return sample

        with patch.object(lr, "lesion_pair", side_effect=invisible):
            rows, summary, _ = lr.audit(fixtures.RoutingOracle(1), ds, "cpu", draws=10)
        self.assertEqual(len(rows), 6)
        for result in summary.values():
            self.assertEqual(result["n_valid_input"], 0)
            self.assertEqual(result["n_valid_routing"], 0)
            self.assertEqual(result["n_empty_lesions"], 3)
            self.assertIsNone(result["metrics"]["content_mean_gain"]["mean"])

    def test_bad_endpoint_is_rejected_and_capture_hooks_are_removed(self):
        ds = self.dataset()
        model = fixtures.RoutingOracle(0.5)
        decode = model.decode_codes
        with patch.object(model, "decode_codes", side_effect=lambda *a, **kw: decode(*a, **kw) + 1):
            with self.assertRaisesRegex(ValueError, "endpoint"):
                lr.audit(model, ds, "cpu", draws=10)
        self.assertFalse(model.codebooks[0]._forward_hooks)
        self.assertFalse(model.codebooks_v1[0]._forward_hooks)

    def test_small_endpoint_error_above_signal_excludes_routing(self):
        ds = self.dataset()
        actual = lr.lesion_pair

        def tiny(dataset, index):
            sample = actual(dataset, index)
            sample["b"] = [x + sample["mask"] * 1e-6 for x in sample["a"]]
            return sample

        class Offset(fixtures.RoutingOracle):
            def decode_codes(self, *a, **kw):
                return super().decode_codes(*a, **kw) + 2e-6

        with patch.object(lr, "lesion_pair", side_effect=tiny):
            _, summary, _ = lr.audit(Offset(1), ds, "cpu", examples=0, draws=10)
        for view in summary.values():
            self.assertEqual(view["n_valid_routing"], 0)
            self.assertIsNone(view["metrics"]["joint_gain"]["median"])

    def test_real_vqvae_and_separate_codebooks_preserve_state(self):
        model = self.Model(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            nb_levels=1,
            embed_dim=4,
            nb_entries=8,
            scaling_rates=[2],
            content_size=3,
            style_size=1,
            content_style_levels=[0],
            mask_mode="fixed",
            inject_style_to_decoder=True,
            style_injection_mode="input",
            separate_encoders=True,
            separate_content_codebooks=True,
            separate_style_codebooks=True,
            quantize_style=True,
            norm_type="layer",
            decoder_norm_type="group",
            final_recon_norm=False,
            use_checkpoint=False,
        )
        model.train()
        model.decoders.eval()
        modes = [m.training for m in model.modules()]
        flags = [p.requires_grad for p in model.parameters()]
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        with lr.frozen_checkpoint(model):
            rows, summary, _ = lr.audit(model, self.dataset(n=2), "cpu", examples=0, draws=10)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
        for row in rows:
            self.assertTrue(row["valid_routing"])
            self.assertAlmostEqual(row["content_mean_gain"] + row["style_mean_gain"], row["joint_gain"], places=6)
        for book in list(model.codebooks) + list(model.codebooks_v1):
            self.assertFalse(book._forward_hooks)

    def test_cli_writes_both_views_and_never_changes_checkpoint(self):
        ds, model, args = self.dataset(), fixtures.RoutingOracle(0.3), self.settings()
        # The state-protection context expects an encoder parameter container.
        # This inert module leaves the oracle's analytically known routing intact.
        model.encoders = torch.nn.Linear(1, 1)
        fake = types.ModuleType("eval.run_dci_synthetic")
        fake.load_run_args = lambda *a: args
        fake.load_model_from_run_dir = lambda *a, **kw: (model, args, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 23001}, checkpoint)
            original = checkpoint.read_bytes()
            out = Path(directory) / "result"
            cli = lr.parser().parse_args(
                [
                    "--run-dir",
                    directory,
                    "--num-samples",
                    "3",
                    "--bootstrap",
                    "10",
                    "--examples",
                    "1",
                    "--out-dir",
                    str(out),
                ]
            )
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), patch.object(
                lr, "make_dataset", return_value=ds
            ), contextlib.redirect_stdout(io.StringIO()):
                report = lr.run(cli)
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(report["checkpoint_step"], 23001)
            saved = json.loads((out / "summary.json").read_text())
            self.assertEqual(saved["donors"], {"a": "lesion absent", "b": "lesion present"})
            self.assertEqual(set(saved["summary"]), {"t1", "flair"})
            self.assertTrue((out / "samples.csv").is_file())
            self.assertEqual((out / "responses.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}):
                with self.assertRaises(FileExistsError):
                    lr.run(cli)

    def test_unsupported_runs_fail_before_decoding(self):
        for override in (
            {"vqvae_nb_levels": 2},
            {"mask_mode": "onthefly"},
            {"inject_style_to_decoder": False},
            {"split_encoder_norm": True},
            {"contrastive_only": True},
            {"synthetic_lesion_mode": "field"},
        ):
            with self.assertRaises(ValueError):
                lr.validate(self.settings(**override))


if __name__ == "__main__":
    unittest.main()
