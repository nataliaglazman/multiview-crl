"""InfoNCE math, backbone gradients, train/test separation, and HF checkpoint round trip."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval import dinov3_embed_synthetic as embed
from training import finetune_dino as train
from utils.config import parse_dino_finetune_args

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class InfoNCETests(unittest.TestCase):
    def test_exact_identity_loss_and_pair_permutation(self):
        features = torch.eye(4)
        loss, metrics = train.symmetric_infonce(features, features, temperature=0.5)
        self.assertAlmostEqual(loss.item(), math.log(1 + 3 * math.exp(-2)), places=6)
        self.assertEqual(metrics["accuracy"], 1.0)
        mismatch, _ = train.symmetric_infonce(features, features.roll(1, 0), temperature=0.5)
        self.assertGreater(mismatch.item(), loss.item())

    def test_symmetry_scaling_and_gradients_to_both_views(self):
        torch.manual_seed(1)
        first = torch.randn(5, 8, requires_grad=True)
        second = torch.randn(5, 8, requires_grad=True)
        loss, _ = train.symmetric_infonce(first, second)
        swapped, _ = train.symmetric_infonce(second * 3, first * 2)
        torch.testing.assert_close(loss, swapped)
        loss.backward()
        for features in (first, second):
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertGreater(features.grad.norm().item(), 0)

    def test_no_negatives_and_invalid_temperature_are_rejected(self):
        for size, temperature in ((1, 0.1), (2, 0), (2, -1), (2, float("nan"))):
            with self.subTest(size=size, temperature=temperature), self.assertRaises(ValueError):
                train.symmetric_infonce(torch.ones(size, 3), torch.ones(size, 3), temperature)


@unittest.skipIf(torch is None, "torch not installed")
class DinoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError:
            raise unittest.SkipTest("transformers with DINOv3 support not installed")
        cls.config = DINOv3ViTConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=8,
            num_register_tokens=2,
        )
        cls.model_class = DINOv3ViTModel

    def options(self, output):
        return parse_dino_finetune_args(
            [
                "--output-dir",
                str(output),
                "--num-samples",
                "4",
                "--batch-size",
                "2",
                "--res",
                "16",
                "--image-size",
                "16",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--projection-hidden-dim",
                "16",
                "--projection-dim",
                "8",
                "--window-pilot",
                "2",
            ]
        )

    def test_training_encoding_matches_eval_for_all_pooling_and_aggregation_modes(self):
        model = self.model_class(self.config).eval()
        cli = self.options("unused")
        cli.axes, cli.slices = ["axial", "coronal"], 2
        cli.plane_batch_size = 3  # chunk boundaries cut across subjects
        cli.batch_size, cli.volume_batch, cli.raw_grid = 3, 2, 0
        rng = np.random.RandomState(0)
        dataset = [
            {"image": [torch.tensor(rng.rand(1, 8, 8, 8), dtype=torch.float32) for _ in range(2)], "gt_latents": {}}
            for _ in range(2)
        ]
        volumes = torch.stack([row["image"][0] for row in dataset])
        for pooling in ("cls", "mean", "cls_mean", "grid"):
            cli.token_pool = pooling
            for aggregation in ("mean", "concat"):
                cli.slice_agg = aggregation
                with self.subTest(pooling=pooling, aggregation=aggregation):
                    encoded = train.encode_volumes(
                        volumes,
                        model,
                        self.config,
                        cli,
                        torch.device("cpu"),
                        (0, 1),
                        embed.IMAGENET_MEAN,
                        embed.IMAGENET_STD,
                    )
                    extracted, _, _, _ = embed.extract(
                        dataset,
                        model,
                        torch.device("cpu"),
                        torch.float32,
                        self.config,
                        cli,
                        (0, 1),
                        embed.IMAGENET_MEAN,
                        embed.IMAGENET_STD,
                    )
                    np.testing.assert_allclose(
                        encoded.detach().numpy(), embed.aggregate(extracted[1], aggregation), rtol=1e-5, atol=1e-6
                    )
                    encoded.square().mean().backward()
                    self.assertGreater(model.embeddings.patch_embeddings.weight.grad.norm().item(), 0)
                    model.zero_grad()

    def test_real_generator_training_export_and_held_out_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self.model_class(self.config)
            model.save_pretrained(root / "initial")
            initial = model.embeddings.patch_embeddings.weight.detach().clone()
            output = root / "run"
            args = [
                "--output-dir",
                str(output),
                "--model-id",
                str(root / "initial"),
                "--local-files-only",
                "--num-samples",
                "4",
                "--batch-size",
                "2",
                "--res",
                "16",
                "--image-size",
                "16",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--projection-hidden-dim",
                "16",
                "--projection-dim",
                "8",
                "--image-mean",
                "0.485",
                "0.456",
                "0.406",
                "--image-std",
                "0.229",
                "0.224",
                "0.225",
                "--gradient-checkpointing",
            ]
            self.assertEqual(train.main(args), 0)
            loaded = self.model_class.from_pretrained(output / "encoder", local_files_only=True)
            self.assertFalse(torch.equal(initial, loaded.embeddings.patch_embeddings.weight))
            checkpoint = torch.load(output / "training_state.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(checkpoint["optimizer"]["state"])
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(ValueError):
                train.main(args)

            cli = self.options(root / "unused")
            _, train_inner, _ = embed.build_dataset(cli, split="train")
            _, test_inner, _ = embed.build_dataset(cli)
            self.assertNotEqual(train_inner.sample_seed_for(0), test_inner.sample_seed_for(0))
            torch.testing.assert_close(train_inner.scm, test_inner.scm)
            saved_window = json.loads((output / "preprocessing.json").read_text())["window_bounds"]
            self.assertEqual(len(saved_window), 2)
            with patch.object(
                embed, "estimate_window", side_effect=AssertionError("must reuse the training window")
            ), patch(
                "data.datasets.SyntheticBrainDataset._compute_fixed_reference",
                side_effect=AssertionError("must reuse the training generator normalization"),
            ):
                self.assertEqual(
                    embed.main(
                        [
                            "--model-id",
                            str(output / "encoder"),
                            "--local-files-only",
                            "--run-dir",
                            str(output),
                            "--preprocessing",
                            str(output / "preprocessing.json"),
                            "--out",
                            str(root / "emb.npz"),
                            "--num-samples",
                            "4",
                            "--device",
                            "cpu",
                            "--raw-grid",
                            "0",
                        ]
                    ),
                    0,
                )
            with np.load(root / "emb.npz") as arrays:
                self.assertEqual(arrays["emb_view1"].shape, (4, 32))
                self.assertEqual(arrays["emb_view2"].shape, (4, 32))
                self.assertTrue(np.isfinite(arrays["emb_view1"]).all())


if __name__ == "__main__":
    unittest.main()
