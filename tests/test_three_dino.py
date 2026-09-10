"""3DINO integration; set THREE_DINO_REPO to test the actual upstream architecture."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from eval import dinov3_embed_synthetic as embed
from models import three_dino as three
from training import finetune_dino as train
from utils.config import parse_dino_finetune_args


def options():
    return SimpleNamespace(
        volume_size=32,
        window="per_volume",
        window_pct=[0.05, 99.95],
        token_pool="cls",
        grid_size=2,
        backbone="3dino",
        volume_batch=2,
        num_workers=0,
        views=[1, 2],
        raw_grid=0,
    )


class PreprocessingTests(unittest.TestCase):
    def test_full_volume_shape_range_constant_and_batch_independence(self):
        cli = options()
        x = torch.arange(16**3).float().reshape(1, 1, 16, 16, 16)
        out = three.prepare_volumes(torch.cat([x, x * 3 + 7]), cli, None)
        self.assertEqual(tuple(out.shape), (2, 1, 32, 32, 32))
        self.assertGreaterEqual(out.min().item(), -1)
        self.assertLessEqual(out.max().item(), 1)
        torch.testing.assert_close(out[0], out[1])
        torch.testing.assert_close(out[:1], three.prepare_volumes(x, cli, None))
        self.assertEqual(three.prepare_volumes(torch.ones_like(x), cli, None).abs().sum().item(), 0)
        cli.window = "dataset"
        fixed = three.prepare_volumes(torch.cat([x, x + 1000]), cli, (0, 6000))
        self.assertGreater(fixed[1].mean().item(), fixed[0].mean().item())

    def test_rejects_2d_and_nonfinite_input(self):
        for x in (torch.ones(2, 1, 32, 32), torch.ones(2, 1, 1, 32, 32), torch.full((2, 1, 32, 32, 32), float("nan"))):
            with self.assertRaises(ValueError):
                three.prepare_volumes(x, options(), None)

    def test_cli_backend_defaults_preserve_2d_behavior(self):
        ordinary = parse_dino_finetune_args(["--output-dir", "unused"])
        self.assertEqual(
            (ordinary.token_pool, ordinary.window, ordinary.window_pct), ("cls_mean", "dataset", [1.0, 99.0])
        )
        volumetric = parse_dino_finetune_args(
            ["--output-dir", "unused", "--backbone", "3dino", "--three-dino-repo", "repo"]
        )
        self.assertEqual((volumetric.token_pool, volumetric.window, volumetric.volume_size), ("cls", "per_volume", 112))
        self.assertEqual(volumetric.model_id, "AICONSlab/3DINO-ViT")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_dino_finetune_args(["--output-dir", "unused", "--backbone", "3dino"])


@unittest.skipUnless(os.environ.get("THREE_DINO_REPO"), "Set THREE_DINO_REPO to the official checkout")
class UpstreamIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        three.import_upstream(os.environ["THREE_DINO_REPO"])
        from dinov2.models.vision_transformer import DinoVisionTransformer3d

        cls.model_class = DinoVisionTransformer3d
        torch.set_num_threads(2)

    def tiny(self, spec=None):
        # Actual upstream transformer and Conv3d patches, reduced width/depth for CPU testing.
        return self.model_class(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3, block_chunks=0, init_values=1e-5
        )

    def wrapped(self):
        spec = {"student": {"arch": "vit_large_3d", "patch_size": 16}, "img_size": 32}
        return three.ThreeDINOEncoder(self.tiny(), spec, {"test": True})

    def test_real_3d_tokens_pooling_and_train_eval_agree(self):
        model = self.wrapped().eval()
        cli = options()
        config = SimpleNamespace(patch_size=16)
        torch.manual_seed(14)
        dataset = [{"image": [torch.randn(1, 32, 32, 32) for _ in range(2)], "gt_latents": {}} for _ in range(3)]
        volumes = torch.stack([row["image"][0] for row in dataset])
        for pooling, width in (("cls", 24), ("mean", 24), ("cls_mean", 48), ("grid", 192)):
            cli.token_pool = pooling
            actual = train.encode_volumes(volumes, model, config, cli, "cpu", None, None, None)
            self.assertEqual(tuple(actual.shape), (3, width))
            extracted, _, _, slots = embed.extract(dataset, model, "cpu", torch.float32, config, cli, None, None, None)
            np.testing.assert_allclose(actual.detach().numpy(), extracted[1][:, 0], atol=1e-6, rtol=1e-5)
            self.assertEqual(slots, ["volume"])
        hidden = model(three.prepare_volumes(volumes, cli, None)).last_hidden_state
        expected = hidden[:, 1:].transpose(1, 2).flatten(1)
        torch.testing.assert_close(actual, expected)

    def test_backbone_gradients_checkpointing_and_export_round_trip(self):
        model = self.wrapped().train()
        model.gradient_checkpointing_enable()
        config, cli = SimpleNamespace(patch_size=16), options()
        x = torch.randn(2, 1, 32, 32, 32)
        before = model.backbone.patch_embed.proj.weight.detach().clone()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        first = train.encode_volumes(x, model, config, cli, "cpu", None, None, None)
        second = train.encode_volumes(x + 0.1 * torch.randn_like(x), model, config, cli, "cpu", None, None, None)
        loss, _ = train.symmetric_infonce(first, second)
        loss.backward()
        self.assertGreater(model.backbone.patch_embed.proj.weight.grad.norm().item(), 0)
        optim.step()
        self.assertFalse(torch.equal(before, model.backbone.patch_embed.proj.weight))
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            load_cli = SimpleNamespace(
                three_dino_repo=os.environ["THREE_DINO_REPO"],
                three_dino_weights=tmp,
                random_init=False,
                patch_size=None,
                device="cpu",
                dtype="float32",
            )
            with patch.object(three, "build_backbone", side_effect=self.tiny):
                loaded, _, _, _ = three.load_encoder(load_cli)
            torch.testing.assert_close(model(x).last_hidden_state, loaded(x).last_hidden_state)
            teacher = Path(tmp) / "teacher.pth"
            state = {"module.backbone." + k: v for k, v in model.backbone.state_dict().items()}
            state["module.dino_head.weight"] = torch.zeros(1)
            torch.save({"teacher": state}, teacher)
            three.load_backbone_weights(loaded.backbone, teacher)
            state.pop("module.backbone.cls_token")
            torch.save({"teacher": state}, teacher)
            with self.assertRaises(RuntimeError):
                three.load_backbone_weights(loaded.backbone, teacher)
        load_cli.three_dino_weights = "/missing/teacher.pth"
        with self.assertRaises(FileNotFoundError):
            three.load_encoder(load_cli)

    def test_real_synthetic_training_and_saved_preprocessing_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.wrapped().save_pretrained(root / "initial")
            common = [
                "--backbone",
                "3dino",
                "--three-dino-repo",
                os.environ["THREE_DINO_REPO"],
                "--device",
                "cpu",
                "--num-samples",
                "4",
            ]
            with patch.object(three, "build_backbone", side_effect=self.tiny):
                train.main(
                    [
                        *common,
                        "--three-dino-weights",
                        str(root / "initial"),
                        "--output-dir",
                        str(root / "run"),
                        "--batch-size",
                        "2",
                        "--epochs",
                        "1",
                        "--res",
                        "16",
                        "--volume-size",
                        "32",
                        "--window",
                        "dataset",
                        "--projection-hidden-dim",
                        "16",
                        "--projection-dim",
                        "8",
                        "--gradient-checkpointing",
                    ]
                )
                pre = root / "run/preprocessing.json"
                saved = json.loads(pre.read_text())
                self.assertEqual(saved["backbone"], "3dino")
                with patch.object(embed, "estimate_window", side_effect=AssertionError("Must reuse training window")):
                    embed.main(
                        [
                            *common,
                            "--three-dino-weights",
                            str(root / "run/encoder"),
                            "--run-dir",
                            str(root / "run"),
                            "--preprocessing",
                            str(pre),
                            "--out",
                            str(root / "emb.npz"),
                            "--raw-grid",
                            "0",
                        ]
                    )
            with np.load(root / "emb.npz") as arrays:
                self.assertEqual(arrays["emb_view1"].shape, (4, 24))
                self.assertEqual(arrays["emb_view2"].shape, (4, 24))
                meta = json.loads(str(arrays["meta"]))
                self.assertEqual(meta["backbone"], "3dino")
                self.assertEqual(meta["slots"], ["volume"])
                self.assertEqual(meta["window_values"], saved["window_bounds"])
                self.assertEqual(meta["image_size"], 32)


if __name__ == "__main__":
    unittest.main()
