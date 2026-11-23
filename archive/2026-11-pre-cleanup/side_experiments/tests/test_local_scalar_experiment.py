"""Geometry, loss isolation, counterfactual evaluation and actual CPU pipelines."""

import contextlib
import csv
import inspect
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from eval.encoder.encoder_target_protocol import digest
from eval.encoder.local_scalar_audit import evaluate_arm
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model
from eval.synthetic.synthetic_dataset import LesionPlacementError
from models.local_scalar_readout import LocalDecoder, LocalReadout, bands, barlow_twins, within_view_decorrelation
from models.scalar_readout import flip_volume
from tests.test_encoder_target_followups import config
from training import local_scalar_data as data
from training import local_scalar_experiment as experiment
from training.local_scalar_objectives import ResidualTarget, tensors, unsupervised_objective
from training.scalar_readout_data import Banks, Extractor

ROOT = Path(__file__).resolve().parents[1]


def random_bank(n=32, views=2):
    rng = np.random.default_rng(17)
    features = rng.normal(size=(n, views, 4, 4, 4, 4)).astype(np.float32)
    support = np.ones((n, views, 1, 4, 4, 4), np.float32)
    return dict(features=features, support=support, global_=rng.normal(size=(n, views, 3)).astype(np.float32))


class LocalScalarTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_native_coordinates_and_brain_frame_follow_blob_and_reflection(self):
        model = LocalReadout(1, 4, 16, heads=2)
        with torch.no_grad():
            model.keypoints.logits.weight.fill_(40)
            model.keypoints.logits.bias.zero_()
        x = torch.zeros(2, 1, 1, 4, 4, 4)
        x[0, 0, 0, 1, 2, 3] = 1
        x[1] = flip_volume(x[0], [-1, 1, -1])
        support = torch.ones_like(x)
        output = model(x, support)
        expected = 2 * (torch.tensor([1.0, 2.0, 3.0]) * 4 + 1.5) / 15 - 1
        torch.testing.assert_close(output["physical"][0, 0], expected.expand(2, 3), atol=1e-5, rtol=0)
        torch.testing.assert_close(output["physical"][1], output["physical"][0] * torch.tensor([-1.0, 1.0, -1.0]))
        self.assertEqual(output["code"].shape, (2, 1, 7))

    def test_decorrelation_ignores_modality_offset_but_detects_duplication(self):
        t = torch.linspace(-1, 1, 32)
        u = t.square() - t.square().mean()
        content = torch.stack((t + 3, t - 3), 1)[..., None].requires_grad_()
        code = torch.stack((u + 3, u - 3), 1)[..., None].requires_grad_()
        self.assertLess(float(within_view_decorrelation(code, content).detach()), 1e-10)
        duplicate = content.detach().clone().requires_grad_()
        loss = within_view_decorrelation(duplicate, content)
        self.assertGreater(float(loss.detach()), 0.99)
        loss.backward()
        self.assertIsNone(content.grad)
        self.assertIsNotNone(duplicate.grad)

    def test_barlow_uses_correlations_not_modality_offsets_and_penalizes_duplicates(self):
        class IdentityProjection:
            def project(self, z):
                return z

        q = torch.randn(64, 3)
        q = q - q.mean(0)
        # An explicitly centred orthogonal basis.
        q, _ = torch.linalg.qr(q)
        paired = torch.stack((q, q), 1)
        loss = barlow_twins(IdentityProjection(), paired, paired)
        shifted = paired + torch.tensor([3.0, -2.0])[None, :, None]
        torch.testing.assert_close(loss, barlow_twins(IdentityProjection(), shifted, shifted), atol=1e-6, rtol=0)
        duplicate = paired[..., :1].expand(-1, -1, 3)
        self.assertGreater(float(barlow_twins(IdentityProjection(), duplicate, duplicate)), float(loss) + 0.01)
        self.assertLessEqual(LocalReadout(4, 4, 16).projector.out_features, 13)

    def test_signed_template_and_decoder_bottleneck(self):
        decoder = LocalDecoder(2, 4, 16, heads=1, views=1)
        with torch.no_grad():
            decoder.blob_values.zero_()
        centre = torch.zeros(2, 1, 1, 3)
        amplitude = torch.tensor([[[1.0]], [[-1.0]]], requires_grad=True)
        output = decoder(centre, amplitude)
        torch.testing.assert_close(output[0], -output[1])
        self.assertEqual(list(inspect.signature(decoder.forward).parameters), ["physical", "amplitude"])
        output[0].square().mean().backward()
        self.assertGreater(float(amplitude.grad.abs().sum()), 0)
        signed = bands(output)["highpass"]
        torch.testing.assert_close(signed[0], -signed[1])

    def test_training_statistics_are_frozen_reloadable_and_ssl_rejects_labels(self):
        bank = random_bank()
        bank["global"] = bank.pop("global_")
        bank.update(
            flip_features=bank["features"][..., ::-1].copy(),
            photo_features=bank["features"] * 1.01,
            flip_support=bank["support"],
            photo_support=bank["support"],
            signs=np.tile([1.0, 1.0, -1.0], (32, 1)).astype(np.float32),
        )
        target = ResidualTarget(bank, channels=2, ridge=1)
        before = state_digest(target)
        restored = ResidualTarget.from_state_dict(target.state_dict())
        batch = tensors(bank, np.arange(8), "cpu")
        torch.testing.assert_close(target.residual(batch), restored.residual(batch))
        args = experiment.parse_args(["--run-dir", "/tmp/source", "--out-dir", "/tmp/out"])
        for arm in ("infonce", "decorrelated", "barlow", "residual"):
            head = LocalReadout(4, 4, 16)
            decoder = LocalDecoder(2, 4, 16)
            loss, _ = unsupervised_objective(head, decoder, target, batch, arm, args)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        self.assertEqual(before, state_digest(target))
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            unsupervised_objective(head, decoder, target, {**batch, "truth": torch.zeros(8, 9)}, "residual", args)
        with self.assertRaisesRegex(ValueError, "interventions"):
            experiment.train_arm(head, decoder, target, bank, {}, "residual", args, "cpu")

    def test_failed_interventions_keep_identity_and_legacy_extractor_handles_brain_branch(self):
        cfg = config()
        model = build_model(cfg, "cpu").eval().requires_grad_(False)
        args = SimpleNamespace(views=["t1"], grid=4)
        real = data.render

        def fail(ds, ctx, z):
            if z[0] != ctx["lat"]["z_content"][0]:
                raise LesionPlacementError("test no room")
            return real(ds, ctx, z)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(
            data, "render", side_effect=fail
        ):
            banks = Banks(tmp)
            try:
                bank, meta = data.interventions(
                    cfg, "test", 1, 0, [0.5], args, data.FrozenExtractor(model, 4, ["t1"], "cpu"), banks, "pairs"
                )
                self.assertEqual(meta["failed_pairs"], 1)
                self.assertEqual(len(bank["rows"]), 9)
                self.assertFalse(bank["rows"][0]["valid"])
            finally:
                banks.close()
        branched = build_model(config(lesion_keypoints=2), "cpu").eval()
        extract = Extractor(branched, SimpleNamespace(view="t1", grid=4), "cpu")
        maps, global_code = extract(torch.ones(2, 1, 16, 16, 16))
        self.assertEqual(global_code.shape, (2, 9))

    def run_tiny(self, root, extra=()):
        run = root / "source"
        run.mkdir()
        cfg = config()
        (run / "settings.json").write_text(json.dumps(cfg))
        torch.save(build_model(cfg, "cpu").state_dict(), run / "model.pt")
        out = root / "result"
        argv = [
            "--run-dir",
            str(run),
            "--out-dir",
            str(out),
            "--cache-dir",
            str(root / "cache"),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--train-samples",
            "8",
            "--probe-samples",
            "8",
            "--test-samples",
            "8",
            "--train-pair-subjects",
            "1",
            "--test-pair-subjects",
            "1",
            "--batch-size",
            "4",
            "--grid",
            "4",
            "--target-channels",
            "2",
            "--eval-eps",
            ".5",
            "--bootstrap",
            "3",
            *extra,
        ]
        before = digest(run / "model.pt")
        experiment.main(argv)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(before, digest(run / "model.pt"))
        self.assertEqual(report["encoder_state_before"], report["encoder_state_after"])
        self.assertTrue(report["residual_target_unchanged"])
        self.assertEqual(list((root / "cache").iterdir()), [])
        with self.assertRaises(FileExistsError):
            experiment.main(argv)
        return out, report

    def test_all_arms_actual_training_evaluation_and_identical_initialization(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out, report = self.run_tiny(Path(tmp))
            hashes = [meta["initial_state_sha256"] for name, meta in report["arms"].items() if name != "free_oracle"]
            self.assertEqual(len(set(hashes)), 1)
            self.assertNotEqual(
                report["cohorts"]["train"]["generator_seed"], report["cohorts"]["fixed_val"]["generator_seed"]
            )
            self.assertTrue(
                set(report["cohorts"]["fixed_test"]["subject_ids"]).isdisjoint(
                    report["cohorts"]["fixed_interventions"]["subject_ids"]
                )
            )
            for arm in experiment.ARMS:
                torch.load(out / f"{arm}.pt", weights_only=True, map_location="cpu")
            evaluation = out / "evaluation_fixed"
            with (evaluation / "recovery.csv").open() as f:
                rows = list(csv.DictReader(f))
            for method in ("shuffled_scalar", "selected_native_head", "joint_ridge", "amplitude_scalar"):
                self.assertTrue(any(r["method"] == method for r in rows))
            with (evaluation / "movement.csv").open() as f:
                self.assertTrue(list(csv.DictReader(f)))
            self.assertTrue((evaluation / "recovery.png").exists())
            with np.load(evaluation / "truth.npz", allow_pickle=False) as truth:
                self.assertEqual(truth["test_targets_t1"].shape, (8, 16))
                self.assertEqual(truth["target_names"][-1], "brain_centroid_z")

    def test_head_selection_uses_validation_even_when_another_head_wins_on_test(self):
        rng = np.random.default_rng(42)
        banks, codes = {}, {}
        for split, count in (("val", 32), ("test", 32), ("pairs", 4)):
            leading = (count, 2) if split == "pairs" else (count,)
            raw = rng.normal(size=(*leading, 9)).astype(np.float32)
            truth = raw.copy()
            truth[..., 2:5] *= 0.2
            banks[split] = dict(
                raw=raw,
                truth=truth,
                support=np.ones((*leading, 1, 1, 4, 4, 4), np.float32),
                contrast=np.linspace(0.1, 0.9, count)[:, None],
            )
            physical = np.zeros((*leading, 1, 2, 3), np.float32)
            physical[..., 0, 0 if split == "val" else 1, :] = truth[..., 2:5]
            amplitude = truth[..., None, 8:9]
            codes[split] = dict(
                physical=physical,
                relative=physical,
                amplitude=amplitude,
                code=np.concatenate((physical.reshape(*leading, 1, 6), amplitude), -1),
            )
        banks["pairs"]["rows"] = [
            dict(subject_id=i, factor_index=2, factor="lesion_x", eps=0.5, valid=True, pair_index=i, zero_image=False)
            for i in range(4)
        ]
        args = SimpleNamespace(views=["t1"], grid=4, resolution=16, seed=42, bootstrap=3)
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_arm("initial", codes, banks, args, Path(tmp))
            calibration = json.loads((Path(tmp) / "initial_calibration.json").read_text())
            self.assertEqual(calibration["t1"]["selected_head"], 0)
            exact = next(r for r in result["localization"] if r["head"] == 1)
            self.assertEqual(exact["mean_error_vox"], 0)
            self.assertFalse(exact["selected_on_validation"])

    def test_single_view_ssl_only_uses_no_training_interventions_and_two_visibility_cohorts(self):
        splits = []
        actual = experiment.interventions

        def record(*args, **kwargs):
            splits.append(args[1])
            return actual(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(
            experiment, "interventions", side_effect=record
        ):
            out, report = self.run_tiny(
                Path(tmp),
                [
                    "--arms",
                    "barlow",
                    "residual",
                    "--views",
                    "flair",
                    "--train-intensity",
                    "styled",
                    "--eval-intensities",
                    "fixed",
                    "styled",
                ],
            )
            self.assertEqual(splits, ["test", "test"])
            self.assertNotIn("train_interventions", report["cohorts"])
            self.assertEqual(report["cohorts"]["train"]["intensity"], "styled")
            self.assertTrue((out / "evaluation_fixed").exists())
            self.assertTrue((out / "evaluation_styled").exists())

    def test_launchers_quote_paths_and_slurm_uses_scratch(self):
        env = dict(
            os.environ,
            ENCODER_PYTHON="/tmp/python with spaces",
            LOCAL_SCALAR_OUTPUT="/tmp/output with spaces",
            ENCODER_REFERENCE_RUN="/tmp/run with spaces",
            LOCAL_SCALAR_TASK_ID="5",
        )
        local = subprocess.check_output(
            [
                "bash",
                str(ROOT / "experiments/generated/local_scalar.local.sh"),
                "/tmp/source with spaces",
                "--dry-run",
                "--views",
                "t1",
            ],
            env=env,
            text=True,
        )
        self.assertIn("/tmp/source with spaces", shlex.split(local))
        cluster = subprocess.check_output(
            ["bash", str(ROOT / "experiments/generated/local_scalar.slurm_bio.sh"), "--dry-run"], env=env, text=True
        )
        command = shlex.split(cluster)
        self.assertIn("/tmp/run with spaces", command)
        self.assertEqual(command[command.index("--train-intensity") + 1], "styled")
        self.assertEqual(command[command.index("--seed") + 1], "242")
        self.assertTrue(command[command.index("--out-dir") + 1].startswith("/scratch/"))


if __name__ == "__main__":
    unittest.main()
