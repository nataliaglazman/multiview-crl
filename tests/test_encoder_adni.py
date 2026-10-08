"""Encoder-only training on real paired volumes: the flag rules, a real loop on a fake ADNI tree, the cache."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from eval.protocol import score_checkpoint
from training import main_conv_synthetic as trainer

GROUPS = ("CN", "MCI", "AD")
SHAPE = (16, 20, 16)
REAL = ["--dataset-name", "ADNI_stripped_masks", "--dataroot", "/data", "--labels-path", "/labels.csv"]
REAL += ["--spatial-size", "96", "112", "96", "--val-frac", "0.2", "--test-frac", "0.1"]
TINY = ["--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--tau", "0.1", "--batch-size", "2", "--train-steps", "2", "--eval-every", "2", "--log-every", "1"]


def fake_adni(root, subjects=30):
    """<root>/ADNI_stripped_masks/<Subject>/{t1,t2} as load_data expects, masks alongside, plus labels.csv.

    Each subject's brain is an ellipsoid with its own size and a ventricle that grows with the
    group, imaged with opposite CSF contrast in the two views; outside the mask is exactly zero.
    """
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij"))
    rows = []
    for i in range(subjects):
        subject, group = f"{i:03d}_S_{i:04d}", GROUPS[i % 3]
        radius = 0.75 + 0.15 * rng.random()
        brain = (np.square(grid / radius).sum(0) <= 1).astype(np.float32)
        ventricle = np.square(grid).sum(0) <= (0.15 + 0.08 * (i % 3) + 0.05 * rng.random()) ** 2
        tissue = brain * (0.8 + 0.1 * rng.standard_normal(SHAPE))
        views = {"t1": np.where(ventricle, 0.1, tissue) * 300, "t2": np.where(ventricle, 1.0, tissue * 0.6) * 300}
        for view, name in (("t1", "T1"), ("t2", "FLAIR")):
            folder = Path(root) / "ADNI_stripped_masks" / subject / view
            folder.mkdir(parents=True)
            nib.save(nib.Nifti1Image((views[view] * brain).astype(np.float32), np.eye(4)), folder / f"{name}.nii.gz")
            nib.save(nib.Nifti1Image(brain, np.eye(4)), folder / f"{name}_brain_mask.nii.gz")
        rows.append(f"{subject},{group}")
    labels = Path(root) / "labels.csv"
    labels.write_text("Subject,Group\n" + "\n".join(rows) + "\n")
    return labels


def run(argv):
    """The real main() on CPU; returns its stdout."""
    with patch.object(sys, "argv", ["trainer", *argv]), contextlib.redirect_stdout(io.StringIO()) as output:
        trainer.main()
    return output.getvalue()


class ParserTests(unittest.TestCase):
    def rejects(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.parse_args(argv)

    def test_real_data_requirements(self):
        args = trainer.parse_args(REAL)
        self.assertEqual((args.dataset_name, args.best_metric, args.spatial_size), (REAL[1], "val_loss", [96, 112, 96]))
        self.assertEqual(trainer.parse_args([]).best_metric, "block_mcc")
        for drop in ("--dataroot", "--labels-path", "--spatial-size", "--val-frac"):
            i = REAL.index(drop)
            width = 4 if drop == "--spatial-size" else 2
            with self.subTest(missing=drop):
                self.rejects(REAL[:i] + REAL[i + width :])
        for bad in (
            ["--val-frac", "0.6", "--test-frac", "0.4"],
            ["--image-spacing", "0"],
            ["--spatial-size", "0", "8", "8"],
        ):
            with self.subTest(bad=bad):
                self.rejects(REAL + bad)

    def test_flags_belong_to_one_kind_of_data(self):
        for flag in (
            ["--res", "64"],
            ["--n-content", "5"],
            ["--num-train-samples", "100"],
            ["--synthetic-causal"],
            ["--synthetic-normalize", "fixed_reference"],
            ["--eval-pooling", "patch"],
            ["--lesion-keypoints", "1"],
            ["--spatial-recovery-eval"],
            ["--best-metric", "block_mcc"],
        ):
            with self.subTest(synthetic_only=flag):
                self.rejects(REAL + flag)
        for flag in (
            ["--val-frac", "0.2"],
            ["--asymmetric-aug"],
            ["--spatial-size", "8", "8", "8"],
            ["--best-metric", "val_loss"],
        ):
            with self.subTest(real_only=flag):
                self.rejects(flag)
        self.assertEqual(trainer.parse_args(REAL + ["--best-metric", "none"]).best_metric, "none")

    def test_grids_must_fit_the_anisotropic_map(self):
        patch_args = ["--patch-loss-weight", "1", "--train-patch-grid"]
        # Conv stride 4 floors 96x112x96 to 24x28x24.
        self.assertEqual(trainer.parse_args(REAL + patch_args + ["24", "28", "24"]).train_patch_grid, [24, 28, 24])
        self.rejects(REAL + patch_args + ["24", "29", "24"])
        # ResNet stride 32 rounds up: 3x4x3.
        resnet = REAL + ["--encoder-architecture", "resnet18"] + patch_args
        self.assertEqual(trainer.parse_args(resnet + ["3", "4", "3"]).train_patch_grid, [3, 4, 3])
        self.rejects(resnet + ["4", "4", "4"])
        small = REAL[: REAL.index("--spatial-size")] + ["--spatial-size", "4", "4", "4", "--val-frac", "0.2"]
        self.rejects(small + ["--global-pool", "attention"])


class RealTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.labels = fake_adni(self.root)
        self.data = ["--dataset-name", "ADNI_stripped_masks", "--dataroot", str(self.root)]
        self.data += ["--labels-path", str(self.labels), "--spatial-size", *map(str, SHAPE), "--image-spacing", "1"]
        self.data += [
            "--val-frac",
            "0.4",
            "--test-frac",
            "0.1",
            "--out-dir",
            str(self.root / "runs"),
            "--device",
            "cpu",
        ]

    def test_trains_selects_on_held_out_loss_and_refuses_synthetic_scoring(self):
        output = run(self.data + TINY + ["--model-id", "adni", "--no-cache", "--require-new-run"])
        run_dir = self.root / "runs" / "adni"
        self.assertIn("input: 16x20x16 volumes at 1 mm -> 4x5x4 backbone map (stride 4)", output)
        self.assertIn("held-out subjects", output)

        split = json.loads((run_dir / "split.json").read_text())["subjects"]
        sets = {mode: set(subjects) for mode, subjects in split.items()}
        self.assertEqual(sum(map(len, sets.values())), 30)
        self.assertEqual(len(set.union(*sets.values())), 30)
        for mode in ("val", "test"):
            groups = {int(subject.split("_")[0]) % 3 for subject in sets[mode]}
            self.assertEqual(groups, {0, 1, 2}, f"{mode} is not stratified")

        floor = json.loads((run_dir / "separation_step0.json").read_text())
        trained = json.loads((run_dir / "separation_step2.json").read_text())
        for key in (
            "val/loss",
            "val/content_retrieval_top1",
            "content/diagnosis_probe_acc_v0",
            "style/modality_probe_acc",
        ):
            self.assertIn(key, trained)
        self.assertAlmostEqual(trained["val/content_retrieval_chance"], 1 / 12)
        self.assertNotEqual(floor["val/loss"], trained["val/loss"])

        best = json.loads((run_dir / "best_checkpoint.json").read_text())
        self.assertEqual((best["metric"], best["step"], best["floor_subtracted"]), ("val_loss", 2, False))
        self.assertAlmostEqual(best["raw"], trained["val/loss"])
        self.assertAlmostEqual(best["value"], -trained["val/loss"])

        cfg = json.loads((run_dir / "settings.json").read_text())
        state = torch.load(run_dir / "model_best.pt", weights_only=True)
        initial = torch.load(run_dir / "model_init.pt", weights_only=True)
        self.assertTrue(any(not torch.equal(state[k], initial[k]) for k in state))
        restored = score_checkpoint.build_model(cfg, "cpu", state)
        self.assertEqual(restored(torch.zeros(2, 1, *SHAPE), pool_only=True, n_views=2)[2][0].shape, (2, 12))
        with self.assertRaisesRegex(ValueError, "real data"):
            score_checkpoint.make_dataset(cfg, 2)

    def test_persistent_cache_masked_patch_training_and_independent_augmentation(self):
        masks = str(self.root / "ADNI_stripped_masks")  # as on the cluster: masks share the image tree
        argv = self.data + TINY + ["--cache-dir", str(self.root / "cache"), "--masks-dir", masks]
        argv += ["--asymmetric-aug", "--patch-loss-weight", "0.5", "--train-patch-grid", "2", "2", "2"]
        argv += ["--patch-foreground-mask", "--no-floor-eval", "--model-id", "cached"]
        output = run(argv)
        self.assertIn("patch foreground mask: first batch keeps", output)
        self.assertNotIn("NOTE: without --asymmetric-aug", output)
        caches = list((self.root / "cache").glob("preprocessed_*"))
        self.assertEqual(len(caches), 1)
        # Train and val are preprocessed; the test subjects are never read.
        self.assertEqual(len(list(caches[0].glob("*.pt"))), 27)
        progress = json.loads((self.root / "runs" / "cached" / "training_progress.json").read_text())
        self.assertEqual((progress["status"], progress["step"]), ("complete", 2))
        terms = progress["last_loss_terms"]
        self.assertAlmostEqual(terms["total"], terms["global"] + terms["patch_weighted"], places=5)

    def test_augmentation_replays_from_its_seed_and_differs_per_worker(self):
        def views(seed):
            args = trainer.parse_args(self.data + ["--no-cache", "--asymmetric-aug", "--data-seed", str(seed)])
            dataset = trainer.make_dataset(args, "train", None)
            return dataset, torch.cat([torch.cat(dataset[i]["image"]).as_subclass(torch.Tensor) for i in range(4)])

        dataset, first = views(3)
        torch.testing.assert_close(views(3)[1], first, rtol=0, atol=0)
        self.assertFalse(torch.equal(views(4)[1], first))
        # Each worker reseeds its copy from torch.initial_seed(), which differs per worker.
        draws = []
        for worker_seed in (11, 12):
            with patch.object(torch.utils.data, "get_worker_info", return_value=SimpleNamespace(dataset=dataset)):
                with patch.object(torch, "initial_seed", return_value=worker_seed):
                    trainer.seed_augmentation_worker(0)
            draws.append(torch.cat(dataset[0]["image"]).as_subclass(torch.Tensor))
        self.assertFalse(torch.equal(*draws))


if __name__ == "__main__":
    unittest.main()
