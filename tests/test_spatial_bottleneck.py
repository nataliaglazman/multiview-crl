"""Spatial preservation, frozen VQ extraction, and a real checkpoint-to-report run."""

import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.protocol import spatial_bottleneck_eval as factor_eval
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


def write_source_run(root, **overrides):
    """A tiny synthetic VQ-VAE run directory with settings.json and a checkpoint."""
    run = Path(root) / "source"
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
    cfg.update(overrides)
    (run / "settings.json").write_text(json.dumps(cfg))
    torch.save({"encoders": source_model().state_dict(), "step": 1}, run / "vqvae_model.pt")
    return run


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
            run, out = write_source_run(root), root / "bottleneck"
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


def trainer_argv(run, out, latent_dim=2, train_samples=6):
    counts = ["--train-samples", str(train_samples), "--val-samples", "4", "--test-samples", "4"]
    sizes = ["--latent-dim", str(latent_dim), "--width", "2", "--hidden", "8", "--batch-size", "2"]
    return ["--run-dir", str(run), "--out-dir", str(out), "--device", "cpu", "--steps", "3", "--log-every", "1"] + (
        counts + sizes
    )


class FactorEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(0)

    def test_kernel_probe_and_pca_match_explicit_features(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        X = self.rng.normal(size=(40, 60))
        y = X[:, :3] @ np.array([1.0, -2.0, 0.5]) + 0.1 * self.rng.normal(size=40)
        train, val, test = np.arange(24), np.arange(24, 32), np.arange(32, 40)
        K = factor_eval.center_kernel(X @ X.T, train)
        fit = factor_eval.fit_select([({"kernel": "linear"}, K, [3.0])], y[:, None], train, val, test)
        self.assertEqual(fit["choice"][0], {"kernel": "linear", "penalty": 3.0})
        expected = Ridge(alpha=3.0).fit(X[train], y[train]).predict(X[test])
        np.testing.assert_allclose(fit["test_pred"][:, 0], expected, rtol=1e-8, atol=1e-8)

        scores = factor_eval.pca_scores(X @ X.T, train, 4)
        pca = PCA(4).fit(X[train])
        np.testing.assert_allclose(np.abs(scores), np.abs(pca.transform(X)), rtol=1e-6, atol=1e-8)
        residual = np.square(X[test] - pca.inverse_transform(pca.transform(X[test]))).sum()
        total = np.square(X[test] - X[train].mean(0)).sum()
        ratio = factor_eval.pca_reconstruction_ratio(X @ X.T, train, scores, test)
        self.assertAlmostEqual(ratio, residual / total, places=8)
        with self.assertRaisesRegex(ValueError, "fewer than"):
            factor_eval.pca_scores(X @ X.T, train[:5], 8)

    def test_selection_and_fit_never_read_test_labels(self):
        X = self.rng.normal(size=(30, 5))
        Y = np.stack([X[:, 0] ** 2, X[:, 1]], 1)
        train, val, test = np.arange(18), np.arange(18, 24), np.arange(24, 30)
        K = factor_eval.standardized_kernel(X, train)
        scrambled = Y.copy()
        scrambled[test] = self.rng.normal(size=(len(test), 2)) * 100
        for kind in factor_eval.PROBES:
            a = factor_eval.fit_select(factor_eval.probe_kernels(K, train, kind), Y, train, val, test)
            b = factor_eval.fit_select(factor_eval.probe_kernels(K, train, kind), scrambled, train, val, test)
            np.testing.assert_array_equal(a["test_pred"], b["test_pred"])
            self.assertEqual(a["choice"], b["choice"])

    def test_gap_blind_spatial_factor_is_read_from_map_and_pca_not_gap(self):
        # Factor 0 moves mass between two voxels (GAP constant); factor 1 shifts the whole map.
        n = 90
        factors = self.rng.normal(size=(n, 2))
        pattern = torch.zeros(1, 4, 4, 4)
        pattern[0, 0, 1, 1], pattern[0, 3, 2, 2] = 1.0, -1.0
        maps = factors[:, 0, None, None, None, None] * pattern.numpy() + factors[:, 1, None, None, None, None]
        maps = torch.tensor(maps + 0.05 * self.rng.normal(size=maps.shape), dtype=torch.float32)
        pairs = torch.stack((maps, maps + 0.05 * torch.randn_like(maps)), 1)
        rows = (np.arange(60), np.arange(60, 75), np.arange(75, n))
        gram = factor_eval.map_gram([pairs[:, 0], pairs[:, 1]], pairs[:60].mean((0, 1)), torch.ones(1, 1, 1, 1))
        views = [np.arange(n), np.arange(n, 2 * n)]
        gap = pairs.mean((3, 4, 5)).double().numpy()
        scores = factor_eval.pca_scores(gram, np.concatenate([rows[0], n + rows[0]]), 2)
        kernels = {
            "map": [gram[np.ix_(v, v)] for v in views],
            "gap": [factor_eval.standardized_kernel(gap[:, view], rows[0]) for view in (0, 1)],
            "pca2": [factor_eval.standardized_kernel(scores[v], rows[0]) for v in views],
        }
        draws = self.rng.integers(0, 15, size=(200, 15))
        scored = factor_eval.score_representations(kernels, [factors, factors], rows, factor_eval.PROBES, draws)
        for kind in factor_eval.PROBES:
            self.assertLess(scored["gap"][kind]["test_r2"][0], 0.2)
            self.assertGreater(scored["gap"][kind]["test_r2"][1], 0.9)
            for name in ("map", "pca2"):
                self.assertGreater(scored[name][kind]["test_r2"][0], 0.9)
            self.assertEqual(scored["map"][kind]["boot"].shape, (200, 2))
        diff = factor_eval.paired_difference(scored["map"]["linear"], scored["gap"]["linear"], [0, 1])
        self.assertGreater(diff["ci"][0, 0], 0.5)
        self.assertEqual(diff["p_le_0"][0], 0.0)

    def test_checkpoint_to_factor_report_with_parity_guards(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            run = write_source_run(root, synthetic_shared_gain_bias=True, synthetic_lesion_placement="wm_interior")
            first, second = root / "d2", root / "d3"
            experiment.main(trainer_argv(run, first, 2))
            experiment.main(trainer_argv(run, second, 3))
            argv = ["--out-dir", str(root / "eval"), "--device", "cpu", "--bootstrap", "100", "--test-samples", "6"]
            payload = factor_eval.main(["--bottleneck-dirs", str(first), str(second)] + argv)

            names = {"map", "patch2x2x2", "gap", "pca2", "pca3", "z:d2", "z_init:d2", "z:d3", "z_init:d3"}
            self.assertEqual(set(payload["results"]), names)
            factors = [f["name"] for f in payload["factors"]]
            self.assertEqual(factors[:2], ["brain_size", "ventricle_size"])
            self.assertEqual(factors[-3:], ["gain", "bias", "noise_sigma"])
            blocks = {f["name"]: f["block"] for f in payload["factors"]}
            self.assertEqual([n for n in factors if blocks[n] == "lesion_centre"], list(factor_eval.CENTRES))
            self.assertEqual([blocks[n] for n in factors[-3:]], ["style_shared", "style_shared", "style"])
            self.assertTrue(payload["lesion_targets"]["included"])
            self.assertNotIn("lesion_volume", blocks)
            for record in payload["bottlenecks"]:
                self.assertTrue(record["z_init_reproduced"])
                self.assertTrue(all(v < 1e-5 for v in record["parity"].values()), record["parity"])
                self.assertEqual(record["compression"]["n_test"], 6)
                self.assertTrue(0 <= record["compression"]["pca_reconstruction_over_template"] <= 1)

            from eval.encoder.encoder_target_protocol import sample_targets
            from eval.protocol.run_dci_synthetic import load_run_args

            dataset = experiment.build_split_dataset(load_run_args(run), "test", 3)
            tap = factor_eval.LabelTap(DataLoader(dataset, batch_size=2), None, dataset._inner)
            self.assertEqual(len(list(tap)), 2)
            expected = [sample_targets(dataset._inner, dataset[i]["gt_latents"])[0][9:12] for i in range(3)]
            np.testing.assert_allclose(tap.centre_labels(), np.stack(expected))
            self.assertIsNone(factor_eval.centre_generator(object())[0])

            burden_args = Namespace(**{**vars(load_run_args(run)), **factor_eval.BURDEN_OVERRIDES})
            dataset = experiment.build_split_dataset(burden_args, "test", 8)
            tap = factor_eval.LabelTap(DataLoader(dataset, batch_size=4), None, dataset._inner)
            self.assertEqual(len(list(tap)), 2)
            fraction = (np.tanh(tap.labels()[0][:, 2]) + 1) / 2
            self.assertGreater(np.corrcoef(tap.lesion_labels()[:, 0], fraction)[0, 1], 0.95)
            burden = factor_eval.main(
                ["--bottleneck-dirs", str(first), "--out-dir", str(root / "burden"), "--device", "cpu"]
                + ["--bootstrap", "100", "--burden-images"]
            )
            self.assertEqual(burden["generator_overrides"], factor_eval.BURDEN_OVERRIDES)
            burden_blocks = {f["name"]: f["block"] for f in burden["factors"]}
            self.assertEqual(
                (burden_blocks["lesion_burden"], burden_blocks["lesion_volume"]), ("content", "lesion_size")
            )
            self.assertFalse(any(name.startswith("unused") for name in burden_blocks))
            self.assertTrue(all(v < 1e-5 for v in burden["bottlenecks"][0]["parity"].values()))
            halves = burden["lesion_centre_by_size"]["z:d2"]["linear"]
            self.assertEqual(
                {half: set(r2) for half, r2 in halves.items()},
                {h: set(factor_eval.CENTRES) for h in ("larger", "smaller")},
            )
            compared = payload["comparisons"]["d2"]["rbf"]
            self.assertEqual(set(compared), {"map", "patch2x2x2", "gap", "pca2", "z_init:d2"})
            self.assertEqual(set(payload["latent_dim_selection"]), {"linear", "rbf"})
            saved = json.loads((root / "eval" / "factor_recovery.json").read_text())
            self.assertEqual(saved["splits"]["test"], {"n": 6, "n_bottleneck": 4, "sample_seed": 44})
            with (root / "eval" / "comparisons.csv").open() as handle:
                self.assertEqual(len(handle.readlines()), 1 + 2 * 2 * 5 * (len(factors) + 1))

            outputs = torch.load(first / "test_outputs.pt", weights_only=True)
            outputs["codes"] = outputs["codes"].roll(1, 0)
            torch.save(outputs, first / "test_outputs.pt")
            with self.assertRaisesRegex(ValueError, "at least"):
                factor_eval.main(["--bottleneck-dirs", str(first), "--out-dir", str(root / "e"), "--test-samples", "3"])
            with self.assertRaisesRegex(ValueError, "do not reproduce"):
                factor_eval.main(["--bottleneck-dirs", str(first), "--out-dir", str(root / "eval2"), "--device", "cpu"])
            other = root / "other_split"
            experiment.main(trainer_argv(run, other, train_samples=4))
            with self.assertRaisesRegex(ValueError, "different source"):
                factor_eval.main(["--bottleneck-dirs", str(second), str(other), "--out-dir", str(root / "eval3")])


if __name__ == "__main__":
    unittest.main()
