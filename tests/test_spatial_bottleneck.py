"""Spatial preservation, frozen VQ extraction, and a real checkpoint-to-report run."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.spatial_bottleneck import SpatialBottleneck, paired_objective
from models.vqvae import VQVAE
from training import spatial_bottleneck_experiment as experiment


def source_model(mode="fixed"):
    return VQVAE(
        in_channels=1,
        hidden_channels=4,
        res_channels=2,
        nb_res_layers=1,
        nb_levels=1,
        embed_dim=4,
        nb_entries=8,
        scaling_rates=[2],
        content_size=2,
        style_size=2,
        content_style_levels=[0],
        separate_encoders=True,
        mask_mode=mode,
        norm_type="layer",
        use_checkpoint=False,
    )


class SpatialBottleneckTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_spatial_order_survives_equal_global_means_and_decoder_only_receives_code(self):
        model = SpatialBottleneck((2, 3, 4, 5), latent_dim=3, width=2, hidden=8)
        maps = torch.zeros(2, 2, 3, 4, 5)
        maps[0, 0, 0, 1, 1] = maps[1, 0, 2, 1, 1] = 2
        torch.testing.assert_close(maps[0].mean((1, 2, 3)), maps[1].mean((1, 2, 3)))
        seen = []
        hook = model.decoder.register_forward_pre_hook(lambda _, inputs: seen.append(tuple(inputs[0].shape)))
        code, reconstructed = model(maps)
        hook.remove()
        self.assertEqual(seen, [(2, 3)])
        self.assertEqual(reconstructed.shape, maps.shape)
        self.assertGreater((code[0] - code[1]).abs().max().item(), 1e-6)
        torch.testing.assert_close(model.decode(code[:1].expand(2, -1))[0], model.decode(code[:1].expand(2, -1))[1])
        with self.assertRaisesRegex(ValueError, "only"):
            model.decode(maps)

    def test_training_only_normalization_and_detached_targets(self):
        model = SpatialBottleneck((2, 3, 4, 5), 3, 2, 8)
        maps = torch.randn(6, 2, 2, 3, 4, 5, requires_grad=True)
        model.fit_normalization(maps)
        torch.testing.assert_close(model.template, maps.detach().mean((0, 1)))
        expected = (maps.detach() - model.template).square().mean((0, 1, 3, 4, 5)).sqrt()
        torch.testing.assert_close(model.scale.flatten(), expected)
        template, scale = model.template.clone(), model.scale.clone()
        loss, _ = paired_objective(model, maps)
        loss.backward()
        self.assertIsNone(maps.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.readout.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters()))
        experiment.evaluate(model, maps.detach() + 100, 2, "cpu")
        torch.testing.assert_close(model.template, template)
        torch.testing.assert_close(model.scale, scale)

    def test_modality_offsets_cannot_satisfy_subject_variance(self):
        model = SpatialBottleneck((1, 2, 2, 2), 2, 2, 8)
        maps = torch.zeros(4, 2, 1, 2, 2, 2)
        maps[:, 1] = 20
        _, terms = paired_objective(model, maps)
        self.assertAlmostEqual(terms["variance"].item(), 0.99, places=5)

    def test_actual_vq_maps_remain_continuous_and_source_state_is_frozen(self):
        for mode in ("fixed", "learned"):
            with self.subTest(mode=mode):
                source = source_model(mode)
                if mode == "learned":
                    with torch.no_grad():
                        source.channel_logits["0"].copy_(torch.tensor([3.0, 2.0, 0.0, -1.0]))
                        source.channel_logits_v1["0"].copy_(torch.tensor([-1.0, 0.0, 2.0, 3.0]))
                before = {k: v.clone() for k, v in source.state_dict().items()}
                rows = [{"image": [torch.randn(1, 8, 10, 12), torch.randn(1, 8, 10, 12)]} for _ in range(4)]
                bank, indices = experiment.extract_content_maps(source, DataLoader(rows, batch_size=2), "cpu")
                self.assertEqual(bank.shape, (4, 2, 2, 4, 5, 6))
                self.assertEqual(indices, [[0, 1], [0, 1]] if mode == "fixed" else [[0, 1], [2, 3]])
                self.assertFalse(bank.requires_grad)
                self.assertFalse(source.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in source.parameters()))
                with torch.no_grad():
                    x = torch.stack([rows[i]["image"][v] for v in range(2) for i in range(2)])
                    native = source(x, pool_only=False, return_recon=False, n_views=2)[2][0]
                torch.testing.assert_close(bank[:2, 0], native[:2, indices[0]])
                torch.testing.assert_close(bank[:2, 1], native[2:, indices[1]])
                for name, actual in source.state_dict().items():
                    torch.testing.assert_close(actual, before[name], rtol=0, atol=0)

    def test_unstable_masks_memory_and_masked_input_fail_explicitly(self):
        rows = [{"image": [torch.randn(1, 8, 8, 8)] * 2} for _ in range(4)]
        loader = DataLoader(rows, batch_size=2)
        with self.assertRaisesRegex(ValueError, "batch-dependent"):
            experiment.extract_content_maps(source_model("onthefly"), loader, "cpu")
        with self.assertRaisesRegex(ValueError, "Map bank"):
            experiment.extract_content_maps(source_model(), loader, "cpu", max_bytes=1)
        source = source_model()
        source.latent_mask = True
        with self.assertRaisesRegex(ValueError, "foreground"):
            experiment.extract_content_maps(source, loader, "cpu")

    def test_training_recovers_spatial_variation_hidden_from_gap(self):
        # Two spatial basis patterns have zero global mean. GAP is constant,
        # while a two-value code has enough capacity to reconstruct their mixture.
        basis = torch.zeros(2, 1, 3, 3, 3)
        basis[0, 0, 0, 1, 1], basis[0, 0, 2, 1, 1] = 1, -1
        basis[1, 0, 1, 0, 1], basis[1, 0, 1, 2, 1] = 1, -1
        weights = torch.randn(48, 2)
        maps = torch.einsum("nk,kcdhw->ncdhw", weights, basis)
        pairs = maps[:, None].repeat(1, 2, 1, 1, 1, 1)
        torch.testing.assert_close(pairs.mean((3, 4, 5)), torch.zeros(48, 2, 1))
        args = experiment.parse_args(
            [
                "--run-dir",
                "/tmp/source",
                "--out-dir",
                "/tmp/new-output",
                "--latent-dim",
                "2",
                "--width",
                "4",
                "--hidden",
                "32",
                "--steps",
                "160",
                "--log-every",
                "40",
                "--batch-size",
                "16",
                "--lr",
                "0.005",
                "--variance-weight",
                "0.01",
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model, history = experiment.fit_readout(pairs[:32], pairs[32:40], args, "cpu")
        metrics, _, _ = experiment.evaluate(model, pairs[40:], 4, "cpu")
        self.assertLess(metrics["reconstruction_over_template"], 0.2)
        self.assertGreater(history["best_step"], 0)
        self.assertEqual(metrics["alignment"], 0)

    def test_real_checkpoint_cli_roundtrip_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            run, out = root / "source", root / "bottleneck"
            run.mkdir()
            cfg = dict(
                dataset_name="synthetic",
                vqvae_hidden_channels=4,
                vqvae_res_channels=2,
                vqvae_nb_res_layers=1,
                vqvae_nb_levels=1,
                vqvae_embed_dim=4,
                vqvae_nb_entries=8,
                vqvae_scaling_rates=[2],
                content_indices=[[0, 1]],
                style_indices=[2, 3],
                content_style_levels=[0],
                separate_encoders=True,
                mask_mode="fixed",
                norm_type="layer",
                synthetic_mode="pseudo_mri",
                synthetic_res=16,
                synthetic_seed=42,
                synthetic_normalize="none",
                synthetic_clean_content=True,
                synthetic_n_content=9,
                synthetic_n_style=3,
            )
            (run / "settings.json").write_text(json.dumps(cfg))
            torch.save({"encoders": source_model().state_dict(), "step": 1}, run / "vqvae_model.pt")
            before = experiment.file_digest(run / "vqvae_model.pt")
            argv = [
                "--run-dir",
                str(run),
                "--out-dir",
                str(out),
                "--device",
                "cpu",
                "--latent-dim",
                "2",
                "--width",
                "2",
                "--hidden",
                "8",
                "--steps",
                "2",
                "--log-every",
                "1",
                "--batch-size",
                "2",
                "--train-samples",
                "4",
                "--val-samples",
                "2",
                "--test-samples",
                "2",
            ]
            report = experiment.main(argv)
            self.assertEqual(experiment.file_digest(run / "vqvae_model.pt"), before)
            self.assertEqual(report["label_access"], "none")
            self.assertEqual([report["splits"][s]["sample_seed"] for s in ("train", "val", "test")], [42, 43, 44])
            self.assertEqual(report["source"]["content_indices_per_view"], [[0, 1], [0, 1]])
            self.assertEqual(json.loads((out / "report.json").read_text())["latent_dim"], 2)
            restored = experiment.load_bottleneck(out / "bottleneck.pt")
            saved = torch.load(out / "bottleneck.pt", weights_only=True)
            original = SpatialBottleneck(**saved["model_config"])
            original.load_state_dict(saved["state_dict"])
            sample = torch.randn(2, *restored.map_shape)
            torch.testing.assert_close(restored.encode(sample), original.encode(sample), rtol=0, atol=0)
            torch.testing.assert_close(restored.decode(restored.encode(sample)), original(sample)[1], rtol=0, atol=0)
            outputs = torch.load(out / "test_outputs.pt", weights_only=True)
            self.assertEqual(outputs["codes"].shape, (2, 2, 2))
            self.assertEqual(outputs["gap"].shape, (2, 2, 2))
            with self.assertRaisesRegex(ValueError, "new directory"):
                experiment.main(argv)


if __name__ == "__main__":
    unittest.main()
