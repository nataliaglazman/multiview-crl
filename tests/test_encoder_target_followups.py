"""Actual forwards/optimization, split isolation, and cluster-command checks."""

import contextlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval.encoder import encoder_spatial_target_audit as audit
from eval.encoder import encoder_target_protocol as protocol
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model
from scripts import generate_encoder_followups_slurm as slurm
from training import encoder_target_control as control


def config(**changes):
    return dict(
        dict(
            encoder_architecture="conv",
            conv_readout="mlp",
            encoder_head_hidden=8,
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            downscale_factor=4,
            latent_dim=12,
            content_channels=9,
            no_separate_encoders=False,
            seed=42,
            data_seed=42,
            model_seed=42,
            res=16,
            n_content=9,
            n_style=3,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_mode="pseudo_mri",
            synthetic_lesion_placement="wm_interior",
            synthetic_causal=False,
            num_train_samples=20,
            num_val_samples=20,
            cpu_threads=1,
            deterministic=True,
            deterministic_warn_only=True,
        ),
        **changes,
    )


class EncoderTargetFollowupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    def test_physical_targets_and_heatmap_coordinates(self):
        ds = protocol.dataset(config(), 20, "val")
        lat = ds[0]["gt_latents"]
        target, support = protocol.sample_targets(ds._inner, lat)
        self.assertEqual(target.shape, (14,))
        np.testing.assert_allclose(target[12], 0.06 * torch.tanh(lat["z_content"][8]).item(), rtol=1e-6)
        self.assertAlmostEqual(target[13], abs(target[12]))
        np.testing.assert_allclose(
            target[9:12], (support[..., None] * ds._inner.renderer.coords).sum((0, 1, 2)).numpy() / support.sum().item()
        )
        # A cell at (1,2,3) covers average-pooling voxel centres (2.5,4.5,6.5).
        logits = torch.full((1, 4, 4, 4), -100.0)
        logits[0, 1, 2, 3] = 100
        np.testing.assert_allclose(
            control.heatmap_centroid(logits, 8).numpy()[0], 2 * np.array([2.5, 4.5, 6.5]) / 7 - 1, atol=1e-6
        )
        with self.assertRaisesRegex(ValueError, "data drift"):
            protocol.dataset(config(synthetic_lesion_radius=0.2), 20, "val")

    def test_spatial_capture_matches_actual_global_code_and_preserves_state(self):
        for cfg in (
            config(),
            config(encoder_architecture="resnet18", conv_readout="linear", resnet_norm="group"),
            config(encoder_architecture="resnet18", conv_readout="linear", resnet_output_stride=8),
        ):
            with self.subTest(architecture=cfg["encoder_architecture"], norm=cfg.get("resnet_norm")):
                model = build_model(cfg, "cpu")
                before = state_digest(model)
                x = torch.randn(4, 1, 16, 16, 16)
                values, shape, _ = audit.capture(model, x, [1], True)
                with torch.inference_mode():
                    code = model(x, pool_only=True, n_views=2)[2][0]
                np.testing.assert_allclose(values[1, "projected"], code[:, :9].numpy(), rtol=1e-6)
                self.assertIn((shape[0], "backbone"), values)
                self.assertEqual(values[1, "hidden"].shape, (4, 8))
                self.assertEqual(state_digest(model), before)
                self.assertFalse(model.encoder._forward_hooks)
                self.assertFalse(model.to_encoding[1]._forward_hooks)

    def test_complete_frozen_audit_preserves_source_and_pairs_initial(self):
        cfg = config()
        model = build_model(cfg, "cpu")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "settings.json").write_text(json.dumps(cfg))
            for name in ("model.pt", "model_init.pt"):
                torch.save(model.state_dict(), run / name)
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            out = Path(tmp) / "audit"
            audit.main(
                [
                    "--run-dir",
                    str(run),
                    "--out-dir",
                    str(out),
                    "--test-samples",
                    "10",
                    "--device",
                    "cpu",
                    "--include-native",
                ]
            )
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir()})
            self.assertEqual(set(r["view"] for r in report["probes"]), {"t1", "flair"})
            self.assertEqual(set(r["probe"] for r in report["probes"]), {"ridge", "rbf"})
            self.assertEqual(set(r["condition"] for r in report["probes"]), {"observed", "shuffled"})
            self.assertEqual(set(r["target"] for r in report["probes"]), set(protocol.TARGETS))
            self.assertEqual(
                report["cohorts"]["trained"]["test"]["input_sha256"],
                report["cohorts"]["initial"]["test"]["input_sha256"],
            )
            self.assertEqual(len(report["probe_split"]["fit_validation_ids"]), 15)
            self.assertEqual(len(report["probe_split"]["tune_validation_ids"]), 5)
            with np.load(out / "trained_predictions.npz") as a, np.load(out / "initial_predictions.npz") as b:
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])
            with self.assertRaises(FileExistsError):
                audit.main(["--run-dir", str(run), "--out-dir", str(out), "--device", "cpu"])

    def test_test_labels_never_select_frozen_probe(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(50, 5)).astype("float32")
        y = rng.normal(size=(50, len(protocol.TARGETS))).astype("float32")
        y[:, 0] = x[:, 0] * 2
        banks = {
            "val": ({("t1", 1, "backbone"): x[:40]}, y[:40], {}),
            "test": ({("t1", 1, "backbone"): x[40:]}, y[40:], {}),
        }
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(seed=9)
            first, _ = audit.score_banks(banks, args, "before", Path(tmp))
            banks["test"][1][:] += 100
            second, _ = audit.score_banks(banks, args, "after", Path(tmp))
            self.assertEqual([(r["alpha"], r["gamma"]) for r in first], [(r["alpha"], r["gamma"]) for r in second])
            with np.load(Path(tmp) / "before_predictions.npz") as a, np.load(Path(tmp) / "after_predictions.npz") as b:
                for name in a.files:
                    if name != "truth":
                        np.testing.assert_array_equal(a[name], b[name])

    def test_supervised_training_never_reads_test_and_updates_both_heads(self):
        class NoTest(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("Test accessed during training")
                return super().__getitem__(key)

        rng = np.random.default_rng(7)
        bank = {
            "images": rng.normal(size=(8, 1, 8, 8, 8)).astype("float32"),
            "heatmaps": rng.uniform(size=(8, 4, 4, 4)).astype("float32"),
            "targets": rng.normal(size=(8, 14)).astype("float32"),
        }
        bank["heatmaps"] /= bank["heatmaps"].sum((1, 2, 3), keepdims=True)
        model = control.TargetControl(4, 2)
        old_map = model.lesion.weight.detach().clone()
        old_reg = model.regression[-1].weight.detach().clone()
        args = SimpleNamespace(
            seed=3,
            lr=0.001,
            steps=2,
            batch_size=4,
            shuffle_targets=False,
            lesion_weight=1.0,
            regression_weight=1.0,
            log_every=2,
            eval_every=2,
            resolution=8,
            view="t1",
        )
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            report = {}
            control.train(
                model,
                NoTest(train=bank, val=bank),
                args,
                "cpu",
                control.target_scaler(bank["targets"]),
                Path(tmp),
                report,
            )
            self.assertEqual(report["completed_steps"], 2)
        self.assertFalse(torch.equal(old_map, model.lesion.weight))
        self.assertFalse(torch.equal(old_reg, model.regression[-1].weight))

    def test_complete_supervised_control_uses_fresh_output_and_test_split(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "original"
            run.mkdir()
            source = json.dumps(config())
            (run / "settings.json").write_text(source)
            out = Path(tmp) / "control"
            control.main(
                [
                    "--run-dir",
                    str(run),
                    "--out-dir",
                    str(out),
                    "--view",
                    "flair",
                    "--device",
                    "cpu",
                    "--steps",
                    "2",
                    "--width",
                    "4",
                    "--grid",
                    "2",
                    "--batch-size",
                    "4",
                    "--test-samples",
                    "10",
                ]
            )
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["completed_steps"], 2)
            self.assertEqual(report["selection"], "final_step")
            self.assertEqual(report["target_scaler"]["fit_split"], "train")
            self.assertEqual(report["cohorts"]["test"]["generator_split_seed"], 44)
            self.assertEqual(len(list(run.iterdir())), 1)
            self.assertEqual((run / "settings.json").read_text(), source)
            truth = np.load(out / "data/train/targets.npy")
            mean, std = control.target_scaler(truth)
            np.testing.assert_array_equal(report["target_scaler"]["mean"], mean)
            np.testing.assert_array_equal(report["target_scaler"]["std"], std)
            self.assertTrue((out / "model.pt").is_file())
            self.assertEqual(len(report["test"]["factors"]), 9)

    def test_slurm_arrays_preview_all_tasks_and_validate_indices(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            slurm.main(["--output-dir", tmp])
            base_env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
            base_env.update(ENCODER_REPO="/repo with spaces", ENCODER_PYTHON="/env/bin/python")
            for kind, count, module in (
                ("spatial_probes", 3, "eval.encoder.encoder_spatial_target_audit"),
                ("target_controls", 2, "training.encoder_target_control"),
            ):
                script = Path(tmp) / f"encoder_{kind}_s42.slurm_bio.sh"
                subprocess.run(["bash", "-n", str(script)], check=True)
                self.assertIn(f"#SBATCH --array=0-{count-1}", script.read_text())
                for idx in range(count):
                    preview = subprocess.check_output(
                        ["bash", str(script), "--dry-run"], env={**base_env, "ENCODER_TASK_ID": str(idx)}, text=True
                    )
                    tokens = shlex.split(preview)
                    self.assertEqual(tokens[:3], ["/env/bin/python", "-m", module])
                    self.assertIn("/repo with spaces/", tokens[tokens.index("--run-dir") + 1])
                result = subprocess.run(
                    ["bash", str(script), "--dry-run"], env={**base_env, "ENCODER_TASK_ID": "9"}, capture_output=True
                )
                self.assertEqual(result.returncode, 2)
                result = subprocess.run(["bash", str(script)], env=base_env, capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
