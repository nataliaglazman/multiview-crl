"""Supervised burden must travel through the original global content vector."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from eval.protocol.score_checkpoint import build_model
from training import encoder_burden_control as control


def settings():
    return dict(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=0,
        downscale_factor=4,
        latent_dim=7,
        content_channels=4,
        conv_readout="mlp",
        encoder_head_hidden=8,
        norm_type="layer",
        res=16,
        seed=42,
        batch_size=2,
        num_train_samples=6,
        num_val_samples=4,
        synthetic_lesion_target="burden",
        synthetic_lesion_placement="wm_interior",
        synthetic_lesion_count=1,
        synthetic_normalize="fixed_reference",
        cpu_threads=1,
        train_steps=2,
        log_every=1,
        eval_every=1,
    )


class Subjects(Dataset):
    def __init__(self, count, split):
        self.count, self.split, self.res = count, split, 16
        self._fixed_mean, self._fixed_scale = 2.0, 3.0
        self.base = torch.linspace(-1, 1, 16**3).reshape(1, 16, 16, 16)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "image": (self.base + index * self.base.square(), self.base.flip(-1) - index * self.base.sin()),
            "index": index,
        }


class BurdenControlTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_heads_only_receive_content_and_gradients_reach_both_backbones(self):
        encoder = build_model(settings(), "cpu")
        model = control.BurdenControl(encoder)
        features = torch.randn(4, 7, requires_grad=True)
        with patch.object(encoder, "forward", return_value=(None, None, [features])):
            prediction = model(torch.zeros(4, 1, 16, 16, 16))
            prediction.square().sum().backward()
        self.assertGreater(features.grad[:, :4].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, 4:].abs().sum().item(), 0)
        self.assertTrue(all(head.in_features == 4 for head in model.heads))
        model.zero_grad(set_to_none=True)
        model(torch.randn(4, 1, 16, 16, 16)).square().sum().backward()
        for backbone in (encoder.encoder, encoder.encoder_v1):
            self.assertIsNotNone(backbone)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters()))

    def test_dry_run_preserves_minimum_radius_and_never_builds_or_renders(self):
        cfg = {**settings(), "res": 64, "synthetic_lesion_min_radius_vox": 1.9}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "reference"
            source.mkdir()
            (source / "settings.json").write_text(json.dumps(cfg))
            output = Path(folder) / "new"
            with patch.object(control, "dataset", side_effect=AssertionError("No rendering")), patch.object(
                control, "build_model", side_effect=AssertionError("No model")
            ), contextlib.redirect_stdout(io.StringIO()):
                plan = control.run(
                    control.parse_args(
                        [
                            "--run-dir",
                            str(source),
                            "--out-dir",
                            str(output),
                            "--device",
                            "cuda",
                            "--dry-run",
                        ]
                    )
                )
            self.assertEqual(plan["minimum_radius_voxels"], 1.9)
            self.assertEqual(plan["content_channels"], 4)
            self.assertFalse(output.exists())

    def test_two_steps_validate_then_test_without_mutating_reference(self):
        created = []

        def make_data(cfg, count, split):
            created.append(split)
            if split == "test":
                self.assertEqual(created, ["train", "val", "test"])
            return Subjects(count, split)

        def table(subjects):
            value = np.arange(len(subjects), dtype=np.float32) + 1
            return np.column_stack((value, value / 2))

        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference"
            reference.mkdir()
            original = json.dumps(settings())
            (reference / "settings.json").write_text(original)
            output = Path(folder) / "control"
            with patch.object(control, "dataset", side_effect=make_data), patch.object(
                control, "burden_table", side_effect=table
            ), contextlib.redirect_stdout(io.StringIO()):
                report = control.run(
                    control.parse_args(
                        [
                            "--run-dir",
                            str(reference),
                            "--out-dir",
                            str(output),
                            "--device",
                            "cpu",
                            "--test-samples",
                            "4",
                        ]
                    )
                )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["step"], 2)
            self.assertEqual(len(report["history"]), 2)
            self.assertEqual(set(report["test"]), {"t1", "flair"})
            self.assertEqual((reference / "settings.json").read_text(), original)
            initial = torch.load(output / "model_init.pt", weights_only=True)
            trained = torch.load(output / "model.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(initial[k], trained[k]) for k in initial))
            self.assertFalse(any(k.startswith("heads.") for k in trained))
            self.assertEqual(
                set(torch.load(output / "control.pt", weights_only=True)),
                {"model", "optimizer", "step", "target_mean_sd"},
            )
            self.assertTrue((output / "report.json").is_file())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                control.run(
                    control.parse_args(
                        [
                            "--run-dir",
                            str(reference),
                            "--out-dir",
                            str(output),
                            "--device",
                            "cpu",
                        ]
                    )
                )


if __name__ == "__main__":
    unittest.main()
