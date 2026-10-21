"""Scientific controls, trainer parity, and reproducible checkpoint replay for the probe."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval.encoder import code_standardization_probe as probe
from eval.protocol.score_checkpoint import build_model
from training import main_conv_synthetic as trainer


def options(**changes):
    return SimpleNamespace(tau=0.2, cross_view_negs_only=changes.get("cross_view_negs_only", True))


def synthetic_code():
    generator = torch.Generator().manual_seed(391)
    targets = torch.randn(48, 3, generator=generator, dtype=torch.float64)
    targets -= targets.mean(0)
    targets = torch.linalg.qr(targets).Q * np.sqrt(len(targets) - 1)
    base = targets * torch.tensor([0.001, 1.0, 0.2]) + 10
    return torch.stack((base, base.clone())), targets


class StandardizationProbeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_small_shared_factor_gains_relative_variance_and_loss_reliance(self):
        raw, targets = synthetic_code()
        _, rows, channels = probe.analyze_batch(
            raw, targets, ["brain_size", "lesion_burden", "thickness"], ["brain_size"], options(), shuffles=4
        )
        selected = {r["mode"]: r for r in rows if r["association"] == "unique"}
        centered, std = selected["centered"], selected["standardized"]
        self.assertLess(centered["variance_share_t1"], 1e-5)
        self.assertGreater(std["variance_share_t1"], 0.3)
        self.assertGreater(std["excess_delta_loss_recomputed"], centered["excess_delta_loss_recomputed"] + 0.1)
        self.assertGreater(channels[0]["scale_multiplier"], 900)

    def test_centering_and_scaling_are_separate_and_affine_correlation_is_unchanged(self):
        raw, targets = synthetic_code()
        centered = probe.transform(raw, "centered")
        std = probe.transform(raw, "standardized")
        torch.testing.assert_close(std, trainer.standardize_contrastive_code(raw))
        for channel in range(3):
            for code in (raw, centered, std):
                corr = torch.corrcoef(torch.stack((code[0, :, channel], targets[:, channel])))[0, 1]
                self.assertAlmostEqual(float(corr), 1.0, places=10)
        self.assertGreater(float((std - centered).norm()), 1)

    def test_first_order_removal_matches_finite_difference_with_both_negative_sets(self):
        raw, targets = synthetic_code()
        direction, _ = probe.factor_direction(targets, 1, "unique")
        part = probe.component(raw, direction)
        for cross in (False, True):
            args = options(cross_view_negs_only=cross)
            for mode in probe.MODES:
                code = raw.clone().requires_grad_(True)
                loss = probe.code_loss(probe.transform(code, mode), args)
                gradient = torch.autograd.grad(loss, code)[0]
                slope = float(-(gradient * part).sum())
                eps = 1e-5
                forward = probe.code_loss(probe.transform(raw - eps * part, mode), args)
                backward = probe.code_loss(probe.transform(raw + eps * part, mode), args)
                self.assertAlmostEqual(float((forward - backward) / (2 * eps)), slope, places=6)

    def test_partial_association_does_not_credit_a_collinear_factor(self):
        generator = torch.Generator().manual_seed(1)
        a = torch.randn(32, generator=generator, dtype=torch.float64)
        targets = torch.stack((a, 2 * a, torch.ones_like(a)), dim=1)
        direction, _ = probe.factor_direction(targets, 0, "marginal")
        self.assertIsNotNone(direction)
        direction, share = probe.factor_direction(targets, 0, "unique")
        self.assertIsNone(direction)
        self.assertLess(share, 1e-10)
        self.assertIsNone(probe.factor_direction(targets, 2, "marginal")[0])

    def test_unrepresented_orthogonal_factor_has_no_linear_removal_effect(self):
        raw, targets = synthetic_code()
        raw[:, :, 0] = 10
        _, rows, _ = probe.analyze_batch(raw, targets, ["missing", "a", "b"], ["missing"], options(), shuffles=2)
        for row in rows:
            self.assertAlmostEqual(row["delta_loss_recomputed"], 0, places=8)
            self.assertLess(row["variance_share_t1"], 1e-12)

    def test_zero_codes_are_finite_and_nulls_replay(self):
        raw, targets = synthetic_code()
        raw.zero_()
        a = probe.analyze_batch(raw, targets, ["a", "b", "c"], ["a"], options(), shuffles=2)
        b = probe.analyze_batch(raw, targets, ["a", "b", "c"], ["a"], options(), shuffles=2)
        self.assertEqual(a, b)
        json.dumps(a, allow_nan=False)
        self.assertTrue(all(r["delta_loss_recomputed"] == 0 for r in a[1]))

    def test_extract_preserves_batchnorm_buffers_module_modes_and_parameter_gradients(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(4)
                self.project = torch.nn.Linear(3, 5)

            def forward(self, images, **kwargs):
                return None, None, [self.bn(images)]

        model = Encoder().eval()
        model.bn.train()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        states = [m.training for m in model.modules()]
        args = SimpleNamespace(content_channels=3, tau=0.2, cross_view_negs_only=True, contrastive_loss_type="infonce")
        raw, parity = probe.extract_code(model, torch.randn(16, 4), args, "train")
        self.assertEqual(raw.shape, (2, 8, 5))
        self.assertEqual(max(parity.values()), 0)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        self.assertEqual(states, [m.training for m in model.modules()])
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))

    def test_real_checkpoint_cli_replays_recorded_clusters_and_never_writes_source(self):
        argv = [
            "--res",
            "16",
            "--hidden-channels",
            "8",
            "--res-channels",
            "4",
            "--nb-res-layers",
            "1",
            "--latent-dim",
            "12",
            "--conv-readout",
            "mlp",
            "--encoder-head-hidden",
            "8",
            "--num-train-samples",
            "80",
            "--num-val-samples",
            "64",
            "--batch-size",
            "16",
            "--stratify-batches",
            "oracle",
            "--stratify-clusters",
            "2",
            "--standardize-code",
            "--synthetic-lesion-target",
            "burden",
            "--synthetic-lesion-placement",
            "wm_interior",
            "--synthetic-normalize",
            "fixed_reference",
            "--best-metric",
            "none",
            "--device",
            "cpu",
            "--cpu-threads",
            "1",
            "--no-cache",
        ]
        args = trainer.parse_args(argv)
        cfg = vars(args)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / "settings.json").write_text(json.dumps(cfg))
            dataset = trainer.make_dataset(args, "train", args.num_train_samples)
            trainer.stratified_batch_sampler(args, dataset, "cpu", root)
            model = build_model(cfg, "cpu")
            torch.save(model.state_dict(), root / "model_init.pt")
            with torch.no_grad():
                next(model.parameters()).add_(0.001)
            torch.save(model.state_dict(), root / "model.pt")
            original = {p.name: probe.digest(p) for p in root.iterdir()}
            cli = probe.parse_args(
                [
                    "--run-dir",
                    str(root),
                    "--checkpoints",
                    "model_init.pt",
                    "model.pt",
                    "--batches",
                    "2",
                    "--shuffles",
                    "2",
                    "--factors",
                    "brain_size",
                    "lesion_burden",
                    "--device",
                    "cpu",
                ]
            )
            report = probe.run(cli)
            self.assertLess(report["parity_max_absolute_error"], 1e-6)
            self.assertEqual(len(report["batches"]), 2)
            self.assertEqual(len(report["baseline_summary"]), 6)
            labels = np.array(json.loads((root / "stratification.json").read_text())["labels"])
            for batch in report["batches"]:
                self.assertEqual(len(set(labels[batch["indices"]])), 1)
            for name, sha in original.items():
                self.assertEqual(probe.digest(root / name), sha)
            with np.load(root / "standardization_probe" / "codes.npz") as arrays:
                self.assertEqual(arrays["checkpoint_0"].shape[:3], (2, 2, 16))
                self.assertEqual(arrays["targets"].shape, (2, 16, 7))
            json.loads((root / "standardization_probe" / "report.json").read_text())
            with self.assertRaisesRegex(ValueError, "nonempty"):
                probe.run(cli)
            cli.out_dir = root / "repeat"
            repeated = probe.run(cli)
            self.assertEqual(report["factor_summary"], repeated["factor_summary"])
            # A stale cluster assignment file must fail instead of silently refitting.
            saved = json.loads((root / "stratification.json").read_text())
            saved["k"] = 3
            (root / "stratification.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                probe.training_loader(args, dataset, root)

    def test_neighbourhood_run_replays_its_recorded_pools(self):
        argv = [
            *("--res", "16", "--num-train-samples", "80", "--num-val-samples", "64", "--batch-size", "16"),
            *("--stratify-batches", "oracle", "--stratify-sampler", "neighbourhood", "--stratify-neighbours", "30"),
            *("--stratify-balance", "2", "--train-steps", "50"),
            *("--synthetic-normalize", "fixed_reference", "--device", "cpu", "--no-cache"),
        ]
        args = trainer.parse_args(argv)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            dataset = trainer.make_dataset(args, "train", args.num_train_samples)
            sampler = trainer.stratified_batch_sampler(args, dataset, "cpu", root)
            saved = json.loads((root / "stratification.json").read_text())
            self.assertEqual((saved["sampler"], saved["neighbours"], saved["balance"]), ("neighbourhood", 30, 2.0))
            self.assertEqual(saved["preview"]["unique_subjects"], 80)
            self.assertEqual(saved["run"]["batches"], 50)
            loader, labels = probe.training_loader(args, dataset, root)
            self.assertIsNone(labels)
            self.assertEqual(list(loader.batch_sampler), list(sampler))
            pools = np.array(saved["neighbour_indices"])
            for batch in loader.batch_sampler:
                self.assertLessEqual(set(batch[1:]), set(pools[batch[0]].tolist()))
            for key, stale in (("neighbours", 31), ("balance", 0.0)):
                (root / "stratification.json").write_text(json.dumps({**saved, key: stale}))
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    probe.training_loader(args, dataset, root)


if __name__ == "__main__":
    unittest.main()
