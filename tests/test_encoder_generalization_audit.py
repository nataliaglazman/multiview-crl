"""Loss parity, held-out probes, train-only BN updates and complete checkpoint audit."""

import ast
import contextlib
import gc
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval import encoder_generalization_audit as audit
from eval.checkpoint_lesion_analysis import state_digest
from eval.score_checkpoint import build_model, make_dataset, make_val_dataset


def config(**kwargs):
    result = dict(
        encoder_architecture="resnet18",
        encoder_head_hidden=4,
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        downscale_factor=4,
        latent_dim=12,
        content_channels=9,
        no_separate_encoders=True,
        seed=42,
        res=16,
        n_content=9,
        n_style=3,
        synthetic_normalize="fixed_reference",
        synthetic_clean_content=True,
        synthetic_mode="pseudo_mri",
        synthetic_causal=True,
        synthetic_causal_graph="random",
        synthetic_causal_edge_prob=0.4,
        num_train_samples=20,
        num_val_samples=20,
        batch_size=4,
        contrastive_loss_type="infonce",
        tau=0.1,
        cross_view_negs_only=True,
    )
    return {**result, **kwargs}


class UnlabeledImages(torch.utils.data.Dataset):
    """Calibration must work without even having ground-truth factors available."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        shape = (1, 16, 16, 16)
        return {"index": index, "image": [torch.full(shape, index + 2.0), torch.full(shape, 0.3 * index - 1.0)]}


class EncoderGeneralizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def tearDown(self):
        gc.collect()

    def test_loss_matches_real_training_reductions_and_retrieval_ties(self):
        # Read actual loss definitions without importing unrelated LPIPS losses.
        source = Path(__file__).resolve().parents[1] / "training/losses.py"
        tree = ast.parse(source.read_text())
        selected = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in ("infonce_base_loss", "_merge_diags")
        ]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(2, 13, 9))
        for cross_only in (True, False):
            result = audit.retrieval_metrics(a, b, 0.13, cross_only)
            actual = namespace["infonce_base_loss"](
                torch.tensor(np.stack((a, b))),
                list(range(9)),
                nn.CosineSimilarity(dim=-1),
                nn.CrossEntropyLoss(),
                tau=0.13,
                cross_view_negs_only=cross_only,
            )
            self.assertAlmostEqual(result["t1_to_flair"]["training_loss"], actual.item(), places=10)
        perfect = audit.retrieval_metrics(np.eye(12), np.eye(12), 0.1)
        wrong = audit.retrieval_metrics(np.eye(12), np.roll(np.eye(12), 1, axis=0), 0.1)
        collapsed = audit.retrieval_metrics(np.zeros((12, 9)), np.zeros((12, 9)), 0.1)
        for direction in perfect:
            self.assertEqual(perfect[direction]["top1"], 1)
            self.assertEqual(perfect[direction]["mrr"], 1)
            self.assertEqual(wrong[direction]["top1"], 0)
            self.assertAlmostEqual(collapsed[direction]["top1"], 1 / 12)
            self.assertAlmostEqual(collapsed[direction]["training_loss"], 2 * np.log(12))
            self.assertGreater(wrong[direction]["training_loss"], perfect[direction]["training_loss"] + 10)
        self.assertEqual(audit.effective_rank(np.zeros((12, 9))), 0)

    def test_real_global_stages_both_views_shared_and_separate_and_projector(self):
        for shared in (True, False):
            model = build_model(config(no_separate_encoders=shared, contrastive_proj_dim=5), "cpu")
            x = torch.randn(4, 1, 16, 16, 16)
            original = state_digest(model)
            values = audit.capture_batch(model, x)
            with torch.inference_mode():
                backbone = model._encode(x, n_views=2, view_idx=None).mean((2, 3, 4))
                hidden = model.to_encoding[1](model.to_encoding[0](backbone))
                final = model(x, pool_only=True, n_views=2)[2][0][:, :9]
                np.testing.assert_allclose(values["backbone"], backbone, atol=1e-6)
                np.testing.assert_allclose(values["hidden"], hidden, atol=1e-6)
                np.testing.assert_allclose(values["content"], final, atol=1e-6)
                np.testing.assert_allclose(values["loss_space"], model.project(final), atol=1e-6)
                np.testing.assert_allclose(values["content_l2"], torch.nn.functional.normalize(final, dim=1), atol=1e-6)
            self.assertEqual(values["loss_space"].shape, (4, 5))
            self.assertEqual(state_digest(model), original)
            self.assertTrue(all(not module._forward_hooks for module in model.modules()))
            model.train()
            with self.assertRaisesRegex(ValueError, "evaluation mode"):
                audit.capture_batch(model, x)
            del model

    def test_bn_recalibration_changes_only_copy_buffers_and_uses_no_labels(self):
        model = build_model(config(), "cpu")
        model.encoder.bn1.running_mean.fill_(50)
        model.encoder.bn1.running_var.fill_(20)
        original = state_digest(model)
        ds = UnlabeledImages()
        with contextlib.redirect_stdout(io.StringIO()):
            candidate, report = audit.recalibrate_batchnorm(model, ds, "cpu", 2, 7, 11)
        self.assertEqual(report["used_samples"], 6)  # Drop a partial batch; no unequal weighting.
        self.assertEqual(len(set(report["subject_ids"])), 6)
        self.assertEqual(report["subject_split"], "train")
        self.assertEqual(state_digest(model), original)
        self.assertNotEqual(state_digest(candidate), original)
        self.assertEqual(
            audit.tensor_digest(model.named_parameters()), audit.tensor_digest(candidate.named_parameters())
        )
        self.assertTrue(all(not module.training for module in candidate.modules()))
        self.assertEqual(candidate.encoder.bn1.momentum, model.encoder.bn1.momentum)
        means, variances = [], []
        with torch.inference_mode():
            for start in range(0, 6, 2):
                samples = [ds[i]["image"] for i in report["subject_ids"][start : start + 2]]
                x = torch.stack([sample[view] for view in (0, 1) for sample in samples])
                stem = model.encoder.conv1(x)
                means.append(stem.mean((0, 2, 3, 4)))
                variances.append(stem.var((0, 2, 3, 4), unbiased=True))
        torch.testing.assert_close(candidate.encoder.bn1.running_mean, torch.stack(means).mean(0))
        torch.testing.assert_close(candidate.encoder.bn1.running_var, torch.stack(variances).mean(0))
        self.assertTrue(all(layer["batches"] == 3 for layer in report["layers"].values()))

    def test_probes_detect_readout_loss_and_nonlinearity_without_test_leakage(self):
        rng = np.random.default_rng(8)

        def bank(n):
            x = rng.choice([-1.0, 1.0], size=(n, 3)) + rng.normal(scale=0.03, size=(n, 3))
            y = np.column_stack((x[:, 2], x[:, 0] * x[:, 1], np.ones(n)))
            return {
                "ids": np.arange(n),
                "targets": y,
                "features": {
                    (view, stage): values
                    for view in audit.VIEWS
                    for stage, values in (("backbone", x), ("content", x[:, :2]))
                },
            }

        banks = {"val": bank(160), "test": bank(120)}
        with contextlib.redirect_stdout(io.StringIO()):
            rows, predictions, split = audit.probe_rows(banks, "original", ["linear", "interaction", "constant"], 9)

        def score(stage, kind, target, condition="observed"):
            return next(
                row["test_r2"]
                for row in rows
                if (row["view"], row["stage"], row["probe"], row["target"], row["condition"])
                == ("t1", stage, kind, target, condition)
            )

        self.assertGreater(score("backbone", "ridge", "linear"), 0.99)
        self.assertLess(score("content", "ridge", "linear"), 0.15)
        self.assertGreater(score("content", "rbf", "interaction"), 0.95)
        self.assertLess(score("content", "ridge", "interaction"), 0.2)
        self.assertLess(score("content", "rbf", "interaction", "shuffled"), 0.2)
        self.assertTrue(np.isnan(score("content", "ridge", "constant")))
        self.assertFalse(set(split["fit_validation_ids"]) & set(split["tune_validation_ids"]))
        banks["test"]["targets"] += 1000
        with contextlib.redirect_stdout(io.StringIO()):
            changed, next_predictions, _ = audit.probe_rows(banks, "original", ["linear", "interaction", "constant"], 9)
        for old, new in zip(rows, changed):
            self.assertEqual((old["alpha"], old["gamma"]), (new["alpha"], new["gamma"]))
        for name in predictions:
            if name != "test_truth":
                np.testing.assert_array_equal(predictions[name], next_predictions[name])

    def test_splits_share_graph_but_not_subjects_and_keep_legacy_validation(self):
        cfg = config()
        datasets = {split: make_dataset(cfg, 20, split) for split in ("train", "val", "test")}
        legacy = make_val_dataset(cfg, 20)
        for split, ds in datasets.items():
            np.testing.assert_array_equal(ds._inner.scm["adj"], legacy._inner.scm["adj"])
            if split == "val":
                torch.testing.assert_close(ds[0]["image"][0], legacy[0]["image"][0], rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(ds[0]["gt_latents"]["z_content"], legacy[0]["gt_latents"]["z_content"]))

    def test_complete_real_resnet_checkpoint_cli_all_three_tests(self):
        cfg = config()
        model = build_model(cfg, "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "settings.json").write_text(json.dumps(cfg))
            checkpoint = run / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            del model
            with contextlib.redirect_stdout(io.StringIO()):
                audit.main(
                    [
                        "--run-dir",
                        str(run),
                        "--num-samples",
                        "10",
                        "--probe-samples",
                        "20",
                        "--batch-size",
                        "2",
                        "--retrieval-batch-size",
                        "4",
                        "--retrieval-draws",
                        "2",
                        "--bn-samples",
                        "8",
                        "--bn-batch-size",
                        "2",
                        "--no-cuda",
                    ]
                )
            (out,) = run.glob("encoder_audit_*")
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["arms"], ["original", "bn_recalibrated"])
            self.assertTrue(report["original_state_unchanged"])
            self.assertEqual(report["batchnorm"]["used_samples"], 8)
            self.assertEqual(len(report["probes"]), 720)
            self.assertEqual(len(report["retrieval"]), 24)
            self.assertEqual(set(report["cohorts"]["train"]) - set(range(20)), set())
            self.assertEqual(len(report["probe_split"]["fit_validation_ids"]), 15)
            for split in ("train", "val", "test"):
                self.assertEqual(
                    report["state"]["original"][split]["input_sha256"],
                    report["state"]["bn_recalibrated"][split]["input_sha256"],
                )
            self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(report["checkpoint_sha256"], before)
            for arm in report["arms"]:
                with np.load(out / f"{arm}_features.npz") as data:
                    self.assertEqual(data["test__flair__backbone"].shape, (10, 512))
                    self.assertEqual(data["test__t1__hidden"].shape, (10, 4))
                self.assertTrue((out / f"{arm}_predictions.npz").exists())
            self.assertTrue((out / "retrieval.csv").exists())
            self.assertTrue((out / "probes.csv").exists())

    def test_conv_without_bn_is_explicitly_not_applicable(self):
        model = build_model(config(encoder_architecture="conv"), "cpu")
        candidate, info = audit.recalibrate_batchnorm(model, UnlabeledImages(), "cpu", 2, 8, 9)
        self.assertIsNone(candidate)
        self.assertEqual(info["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
