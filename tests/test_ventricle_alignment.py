"""Scientific controls for paired ventricular responses and BT mismatch losses."""

import argparse
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.lesion.lesion_alignment import bt_terms, extract_stages, foreground_positions, response_rows
from eval.ventricle import ventricle_alignment as va

ROOT = Path(__file__).resolve().parents[1]


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(**overrides):
    values = dict(
        patch_contrastive=True,
        patch_grid=(2, 2, 2),
        patch_grid_per_level=None,
        patch_foreground_mask=True,
        patch_foreground_thresh=0.05,
        patch_center_mode="position",
        bt_patch_stat="fold",
        bt_lambda=0,
        bt_sim_coeff=1,
        bt_std_coeff=0,
        bt_normalize_terms=True,
        bt_sim_normalize=False,
        bt_corr_ema=0,
        bt_patch_weight=0.5,
        bt_gap_weight=0.25,
        bt_gap_lambda=None,
        bt_gap_sim_coeff=None,
        bt_gap_std_coeff=None,
        scale_contrastive_loss=10,
        contrastive_level_weights=[0.4],
        batch_size=4,
        bt_gap_pooling="gap",
        synthetic_mode="pseudo_mri",
        synthetic_res=16,
        synthetic_n_content=9,
        synthetic_content_prior="uniform",
        synthetic_content_squash="none",
        synthetic_clean_content=True,
        synthetic_identifiable_ventricle=True,
        synthetic_normalize="fixed_reference",
        synthetic_causal=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.losses = fixture_module("test_patch_signal_audit").source_losses()
        helpers = fixture_module("test_ventricle_routing")
        cls.datasets = helpers.load_without_monai("data/datasets.py")
        cls.Model = helpers.load_without_monai("models/vqvae.py").VQVAE

    def model(self, **kwargs):
        options = dict(
            hidden_channels=4,
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
            norm_type="layer",
            use_checkpoint=False,
            inject_style_to_decoder=True,
        )
        options.update(kwargs)
        torch.manual_seed(17)
        return self.Model(**options).eval()

    def dataset(self, args, n=6):
        with patch.dict("sys.modules", {"data.datasets": self.datasets}):
            return va.make_dataset(args, n, "match", "test")

    def test_balanced_swap_preserves_every_endpoint_and_subject_identity(self):
        low = torch.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4).float()
        high = low + 1000
        conditions = va.balanced_features(low, high)
        for condition in conditions.values():
            torch.testing.assert_close(condition[0], conditions["matched"][0])
            # Entire feature vectors, not merely their marginal channel values.
            actual = sorted(tuple(row) for row in condition[1].flatten(1).tolist())
            expected = sorted(tuple(row) for row in conditions["matched"][1].flatten(1).tolist())
            self.assertEqual(actual, expected)
        torch.testing.assert_close(conditions["ventricle_mismatched"][1, :5], high[1])
        torch.testing.assert_close(conditions["ventricle_mismatched"][1, 5:], low[1])
        with self.assertRaisesRegex(ValueError, "N>=2"):
            va.balanced_features(low[:, :1], high[:, :1])

    def test_shared_factor_has_positive_mismatch_and_exact_raw_mse_delta(self):
        torch.manual_seed(2)
        low = torch.randn(1, 8, 3, 8).repeat(2, 1, 1, 1)
        high = low + torch.randn(1, 8, 3, 8).repeat(2, 1, 1, 1) * 0.2
        with patch.dict("sys.modules", {"training.losses": self.losses}):
            row = va.pairing_losses(low, high, settings(), "loss_patch", 0)[0]
        expected = float((high[0] - low[0]).square().mean())
        self.assertAlmostEqual(row["ventricle_mismatched_minus_matched_sim_loss"], expected, places=6)
        self.assertGreater(row["ventricle_mismatched_minus_matched_weighted_total"], 0)
        self.assertAlmostEqual(row["matched_arm_scale"], 2)
        self.assertAlmostEqual(row["ventricle_mismatched_minus_matched_var_loss"], 0, places=6)

    def test_absent_factor_is_exact_null_and_whole_subject_control_still_changes(self):
        torch.manual_seed(9)
        low = torch.randn(1, 8, 4, 8).repeat(2, 1, 1, 1)
        with patch.dict("sys.modules", {"training.losses": self.losses}):
            rows = va.pairing_losses(low, low, settings(bt_corr_ema=0.99), "loss_patch", 0)
        for row in rows:
            self.assertEqual(row["ventricle_mismatched_minus_matched_weighted_total"], 0)
            self.assertGreater(row["subject_mismatched_minus_matched_weighted_total"], 0)

    def test_terms_match_shipped_loss_including_global_whitening_and_overrides(self):
        torch.manual_seed(2)
        hz = torch.randn(2, 16, 4)
        args = settings(
            bt_lambda=2,
            bt_gap_lambda=3,
            bt_gap_sim_coeff=0.7,
            bt_gap_std_coeff=0.2,
            bt_sim_whiten=True,
            bt_sim_whiten_eps=0.05,
        )
        with patch.dict("sys.modules", {"training.losses": self.losses}):
            actual = bt_terms(hz, args, "loss_gap", 0)
        expected = self.losses.barlow_twins_loss(
            hz,
            [list(range(4))],
            [(0, 1)],
            lambd=3,
            sim_coeff=0.7,
            std_coeff=0.2,
            sim_whiten=True,
            sim_whiten_eps=0.05,
            normalize_terms=True,
        )
        self.assertAlmostEqual(actual["unweighted_total"], float(expected), places=6)
        self.assertAlmostEqual(actual["weighted_total"], float(expected), places=6)

    def test_ema_conditions_are_independent_and_sim_is_not_attenuated(self):
        torch.manual_seed(19)
        low, high = torch.randn(2, 8, 3, 8), torch.randn(2, 8, 3, 8)
        for patch_stat in ("fold", "per_position"):
            args = settings(bt_corr_ema=0.99, bt_lambda=6, bt_patch_stat=patch_stat)
            with patch.dict("sys.modules", {"training.losses": self.losses}):
                rows = va.pairing_losses(low, high, args, "loss_patch", 0)
                again = va.pairing_losses(low, high, args, "loss_patch", 0)
            self.assertEqual(rows, again)
            self.assertEqual(rows[1]["mode"], "ema_matched_reference")
            for condition in ("matched", "ventricle_mismatched", "subject_mismatched"):
                self.assertEqual(rows[0][f"{condition}_sim_loss"], rows[1][f"{condition}_sim_loss"])
                self.assertAlmostEqual(rows[0][f"{condition}_var_loss"], rows[1][f"{condition}_var_loss"], places=6)
            # Settled matched history plus another matched batch reproduces its correlation.
            self.assertAlmostEqual(rows[0]["matched_on_diag_loss"], rows[1]["matched_on_diag_loss"], places=5)

    def test_zero_response_and_single_subject_never_invent_alignment(self):
        low = torch.ones(2, 1, 3, 8)
        row = response_rows(low, low, [0], "test", "match")[0]
        self.assertTrue(np.isnan(row["delta_cosine"]))
        self.assertTrue(np.isnan(row["delta_wrong_subject_cosine"]))

    def samples(self):
        torch.manual_seed(22)
        result = []
        for idx in range(3):
            mask = torch.zeros(1, 8, 8, 8)
            mask[:, : 4 if idx < 2 else 8, :4, :4] = 1
            result.append(
                dict(
                    low=[torch.randn(1, 8, 8, 8) for _ in range(2)],
                    high=[torch.randn(1, 8, 8, 8) for _ in range(2)],
                    mask=mask,
                    metadata={"index": idx},
                )
            )
        return result

    def test_encoding_chunks_match_one_forward_with_fixed_mask_and_projection(self):
        for separate in (False, True):
            model = self.model(separate_encoders=separate)
            model._contrastive_proj_heads = torch.nn.ModuleDict(
                {"L0": torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))}
            ).eval()
            args, samples = settings(bt_gap_pooling="stats"), self.samples()
            before = va.state_digest(model)
            with patch.dict("sys.modules", {"training.losses": self.losses}):
                chunked, _, keep = va.encode_group(model, samples, args, "cpu", 0, 1)
                single, _, _ = va.encode_group(model, samples, args, "cpu", 0, 3)
            self.assertEqual(int(keep.sum()), 2)
            for endpoint in ("low", "high"):
                for stage in single[endpoint]:
                    torch.testing.assert_close(chunked[endpoint][stage], single[endpoint][stage], atol=2e-6, rtol=2e-5)
                expected = self.losses.stats_pool(chunked[endpoint]["loss_patch"])[0]
                torch.testing.assert_close(chunked[endpoint]["loss_gap"], expected)
            self.assertEqual(va.state_digest(model), before)
            self.assertFalse(model.content_norms["0"]._forward_pre_hooks)

    def test_default_extractor_matches_explicit_batch_union(self):
        args, samples, model = settings(), self.samples(), self.model()
        images, masks = [s["low"] for s in samples], [s["mask"] for s in samples]
        keep = foreground_positions(masks, args, 0)
        default, _, a = extract_stages(model, images, masks, args, "cpu", 0)
        explicit, _, b = extract_stages(model, images, masks, args, "cpu", 0, keep_override=keep)
        self.assertTrue(torch.equal(a, b))
        for stage in default:
            torch.testing.assert_close(default[stage], explicit[stage])

    def test_last_single_subject_gets_responses_but_no_self_shuffle_loss(self):
        args, samples, model = settings(), self.samples(), self.model()
        for sample in samples:
            sample["metadata"].update(valid_input=True, isolated_intervention=True)
        with patch.object(va, "render_sample", side_effect=lambda ds, idx, eps: samples[idx]), patch.dict(
            "sys.modules", {"training.losses": self.losses}
        ):
            report = va.audit(model, range(3), args, "cpu", batch_size=1, loss_batch_size=4)
        self.assertEqual(report["counts"]["valid_input"], 3)
        self.assertEqual(report["counts"]["in_pairing_loss"], 2)
        self.assertEqual(report["counts"]["loss_batches"], 1)
        for row in report["samples"]:
            if row["index"] == 2:
                self.assertTrue(np.isnan(row["delta_wrong_subject_cosine"]))
        for row in report["losses"]:
            self.assertEqual(row["loss_rows"], 4)

    def test_no_isolated_pairs_is_reported_without_fabricated_alignment(self):
        args, samples, model = settings(), self.samples(), self.model()
        for sample in samples:
            sample["metadata"].update(valid_input=False, isolated_intervention=False)
        with patch.object(va, "render_sample", side_effect=lambda ds, idx, eps: samples[idx]), patch.object(
            va, "encode_group", side_effect=AssertionError("Confounded pairs must not be encoded")
        ):
            report = va.audit(model, range(3), args, "cpu", loss_batch_size=4)
        self.assertEqual(report["counts"]["confounded"], 3)
        self.assertEqual(report["summary"], {})
        self.assertEqual(report["losses"], [])

    def test_real_renderer_confounded_lesion_motion_is_excluded(self):
        args = settings(synthetic_res=32, synthetic_lesion_placement="wm_interior")
        ds = self.dataset(args)
        samples = [va.render_sample(ds, i, 0.5) for i in range(len(ds))]
        confounded = [s for s in samples if s["metadata"]["lesion_changed_voxels"]]
        self.assertTrue(confounded)
        for sample in confounded:
            self.assertFalse(sample["metadata"]["valid_input"])
            self.assertEqual(sample["metadata"]["exclusion_reason"], "lesion_geometry_changed")
        isolated = [s for s in samples if s["metadata"]["valid_input"]]
        self.assertTrue(isolated)
        sample = isolated[0]
        again = va.render_sample(ds, sample["metadata"]["index"], 0.5)
        for view in range(2):
            self.assertTrue(torch.equal(sample["low"][view], again["low"][view]))
            self.assertTrue(torch.equal(sample["high"][view], again["high"][view]))

    def test_cli_real_renderer_and_vqvae_write_losses_without_mutating_checkpoint(self):
        args, model = settings(bt_corr_ema=0.99), self.model(separate_encoders=True)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 123}, checkpoint)
            saved_bytes = checkpoint.read_bytes()
            cli = va.parser().parse_args(
                [
                    "--run-dir",
                    tmp,
                    "--num-samples",
                    "7",
                    "--batch-size",
                    "1",
                    "--loss-batch-size",
                    "4",
                    "--eps",
                    "0.5",
                    "--out-dir",
                    str(Path(tmp) / "audit"),
                ]
            )
            output = io.StringIO()
            with patch.dict(
                "sys.modules", {"training.losses": self.losses, "data.datasets": self.datasets}
            ), patch.object(va, "load_model", return_value=(model, args, "cpu")), redirect_stdout(output):
                va.main(cli)
            self.assertEqual(checkpoint.read_bytes(), saved_bytes)
            result = json.loads((Path(tmp) / "audit/summary.json").read_text())
            self.assertTrue(result["registered_state_unchanged"])
            self.assertGreater(result["counts"]["match"]["valid_input"], 0)
            self.assertTrue(result["batch_bt_terms"])
            for row in result["batch_bt_terms"]:
                self.assertEqual(row["loss_rows"], 4)
                self.assertIn("ventricle_mismatched_minus_matched_weighted_total", row)
            self.assertIn("high-cos", output.getvalue())
            self.assertIn("Δventricle", output.getvalue())
            for name in ("samples", "pairing_losses", "interventions", "channels"):
                self.assertTrue((Path(tmp) / "audit" / f"{name}.csv").is_file())

    def test_state_mutation_and_changed_channel_selection_are_rejected(self):
        args, samples, model = settings(), self.samples(), self.model()
        original = va.extract_stages
        counter = 0

        def changed(*a, **kw):
            nonlocal counter
            stages, selected, keep = original(*a, **kw)
            counter += 1
            if counter == 2:
                selected = [s.flip(0) for s in selected]
            return stages, selected, keep

        with patch.object(va, "extract_stages", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "channel selection changed"):
                va.encode_group(model, samples, args, "cpu", 0, 1)
        ds = self.dataset(args, 4)
        original_render = va.render_sample

        def mutate(*a):
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
            return original_render(*a)

        with patch.object(va, "render_sample", side_effect=mutate), patch.dict(
            "sys.modules", {"training.losses": self.losses}
        ):
            with self.assertRaisesRegex(RuntimeError, "parameter or buffer changed"):
                va.audit(model, ds, args, "cpu", eps=0.5, loss_batch_size=4)


if __name__ == "__main__":
    unittest.main()
