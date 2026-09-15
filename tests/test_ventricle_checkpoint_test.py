"""Numerical loss parity, nonlinear readout oracle, and checkpoint restoration."""

import ast
import contextlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sklearn  # noqa: F401 - Load compiled SciPy dependencies before sys.modules patches.
import torch

from eval import ventricle_checkpoint_test as vt

ROOT = Path(__file__).resolve().parents[1]


def shipped_barlow():
    # Execute the unchanged loss functions without importing unrelated LPIPS/MONAI code.
    tree = ast.parse((ROOT / "training/losses.py").read_text())
    wanted = {"barlow_twins_loss", "_center_patch_features", "_merge_diags"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    scope = {"torch": torch, "F": torch.nn.functional, "contextlib": contextlib}
    exec(compile(tree, str(ROOT / "training/losses.py"), "exec"), scope)
    return scope["barlow_twins_loss"]


def settings(**kwargs):
    result = dict(
        contrastive_loss_type="barlow_twins",
        patch_contrastive=True,
        mask_mode="fixed",
        vqvae_nb_levels=1,
        inject_style_to_decoder=True,
        batch_size=4,
        patch_grid=[2, 2, 2],
        bt_lambda=6,
        bt_gap_lambda=None,
        bt_normalize_terms=True,
        bt_corr_ema=0.99,
        bt_sim_coeff=0.0114,
        bt_gap_sim_coeff=None,
        bt_gap_weight=1,
        bt_patch_weight=1,
        scale_contrastive_loss=100,
        bt_sim_normalize=False,
        bt_patch_stat="fold",
        contrastive_proj_dim=0,
        contrastive_proj_mode="head",
        synthetic_res=8,
    )
    result.update(kwargs)
    return types.SimpleNamespace(**result)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_redundancy_value_and_gradient_match_training_with_settled_ema(self):
        bt = shipped_barlow()
        torch.manual_seed(7)
        for norm in (False, True):
            for decay in (0.0, 0.99):
                hz = torch.randn(2, 9, 4, requires_grad=True)
                reference = vt.gap_correlation(torch.randn(2, 9, 4)).detach()

                def original(lam):
                    state = {(0, 0, 1): {"c": reference.clone(), "t": 10000, "sig": ((4, 4), 4)}}
                    return bt(
                        hz,
                        [list(range(4))],
                        [[0, 1]],
                        lambd=lam,
                        normalize_terms=norm,
                        corr_ema=state,
                        corr_ema_decay=decay,
                    )

                expected = original(1.0) - original(0.0)
                actual = vt.gap_redundancy(hz, norm, decay, reference)
                torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=2e-5, atol=2e-6)
                expected_g = torch.autograd.grad(expected, hz, retain_graph=True)[0]
                actual_g = torch.autograd.grad(actual, hz)[0]
                torch.testing.assert_close(actual_g, expected_g, rtol=3e-5, atol=1e-7)

    def test_mse_value_and_gradient_match_training(self):
        bt = shipped_barlow()
        for normalize in (False, True):
            hz = torch.randn(2, 11, 3, requires_grad=True)
            expected = bt(hz, [list(range(3))], [[0, 1]], sim_coeff=1, sim_normalize=normalize) - bt(
                hz, [list(range(3))], [[0, 1]], sim_coeff=0, sim_normalize=normalize
            )
            actual = vt.similarity_loss(hz, normalize)
            torch.testing.assert_close(actual.reshape_as(expected), expected)
            a = torch.autograd.grad(actual, hz, retain_graph=True)[0]
            b = torch.autograd.grad(expected, hz)[0]
            torch.testing.assert_close(a, b)

    def test_weights_use_inheritance_normalization_is_not_applied_twice(self):
        self.assertEqual(vt.gap_weights(settings()), {"gap_redundancy": 600.0, "gap_mse": 1.1400000000000001})
        w = vt.gap_weights(
            settings(bt_gap_lambda=2, bt_gap_sim_coeff=0.3, contrastive_level_weights=[0.5], bt_gap_weight=0.2)
        )
        self.assertAlmostEqual(w["gap_redundancy"], 20)
        self.assertAlmostEqual(w["gap_mse"], 3)

    def test_nonlinear_probe_recovers_information_missed_by_linear_probe(self):
        from threadpoolctl import threadpool_limits

        rng = np.random.default_rng(8)
        x = rng.uniform(-2, 2, (240, 1))
        y = x[:, 0] ** 2
        with threadpool_limits(limits=2):
            probes, _ = vt.fit_probes({"t1/content/spatial": x[:160]}, y[:160])
            pred = vt.predict_probes(probes, {"t1/content/spatial": x[160:]})
        self.assertLess(vt.r2(y[160:], pred["t1/content/spatial/ridge"]), 0.1)
        self.assertGreater(vt.r2(y[160:], pred["t1/content/spatial/rbf"]), 0.95)

    def test_refitting_does_not_mistake_a_coordinate_change_for_loss(self):
        from threadpoolctl import threadpool_limits

        rng = np.random.default_rng(2)
        x = rng.normal(size=(100, 2))
        y = x[:, 0] + 2 * x[:, 1]
        with threadpool_limits(limits=2):
            probes, _ = vt.fit_probes({"t1/content/gap": x[:70]}, y[:70])
            probes = {k: v for k, v in probes.items() if k.endswith("ridge")}
            fixed = vt.predict_probes(probes, {"t1/content/gap": -x[70:]})
            refit = vt.predict_probes(probes, {"t1/content/gap": -x[70:]}, {"t1/content/gap": -x[:70]}, y[:70])
        self.assertLess(vt.r2(y[70:], next(iter(fixed.values()))), 0)
        self.assertGreater(vt.r2(y[70:], next(iter(refit.values()))), 0.99)

    def test_paired_bootstrap_zero_and_sign(self):
        y = np.arange(20, dtype=float)
        base = y + 2
        zero = vt.paired_delta(y, base, base)
        self.assertEqual(zero["delta_r2"], 0)
        self.assertEqual(zero["delta_r2_ci_low"], 0)
        better = vt.paired_delta(y, base, y)
        self.assertGreater(better["delta_r2_ci_low"], 0)

    def test_routing_comparison_uses_common_valid_subjects(self):
        metrics = ("content_mean_gain", "style_mean_gain", "joint_gain", "joint_relative_error")
        a = [dict(index=i, modality="t1", valid_routing=True, **dict.fromkeys(metrics, i)) for i in range(3)]
        b = [dict(r, valid_routing=r["index"] != 2, **dict.fromkeys(metrics, r["index"] + 0.1)) for r in a]
        result = vt.routing_delta(a, b)
        self.assertEqual(result["t1"]["n_common_valid"], 2)
        self.assertAlmostEqual(result["t1"]["content_mean_gain"]["median_paired_delta"], 0.1)
        self.assertEqual(result["flair"]["n_common_valid"], 0)


def real_model(**kwargs):
    spec = importlib.util.spec_from_file_location("vent_checkpoint_model", ROOT / "models/vqvae.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"utils.utils": types.ModuleType("utils.utils")}):
        spec.loader.exec_module(module)
    config = dict(
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        nb_levels=1,
        embed_dim=4,
        nb_entries=8,
        scaling_rates=[2],
        use_checkpoint=False,
        content_size=3,
        style_size=1,
        mask_mode="fixed",
        inject_style_to_decoder=True,
        norm_type="layer",
    )
    config.update(kwargs)
    return module.VQVAE(**config)


def batch(n=3):
    images = [torch.randn(n, 1, 8, 8, 8) for _ in range(2)]
    return {
        "image": images,
        "mask": [torch.ones_like(x) for x in images],
        "gt_latents": {"z_content": torch.randn(n, 9)},
    }


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_quantized_features_separate_codebooks_and_global_style(self):
        model = real_model(
            separate_encoders=True,
            separate_content_codebooks=True,
            quantize_style=True,
            separate_style_codebooks=True,
            style_spatial_size=1,
        ).eval()
        b = batch()
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        with vt.frozen_checkpoint(model):
            features, y, codes = vt.decoder_features(model, [b], "cpu", 2)
            self.assertEqual(features["t1/content/spatial"].shape, (3, 4 * 8))
            self.assertEqual(features["flair/content/gap"].shape, (3, 4))
            self.assertEqual(features["t1/style/gap"].shape, features["t1/style/spatial"].shape)
            self.assertEqual(codes.shape, (2, 3, 64))
            np.testing.assert_array_equal(y, b["gt_latents"]["z_content"][:, 1].numpy())
            # These must be the actual post-quantization tensors, not encoder features.
            x = torch.cat(b["image"])
            captured = []
            h = model.codebooks[0].register_forward_hook(lambda m, i, o: captured.append(o[0].detach().clone()))
            with torch.no_grad():
                model(x, return_recon=True, n_views=2)
            h.remove()
            expected = captured[0].mean((2, 3, 4)).numpy()
            np.testing.assert_allclose(features["t1/content/gap"], expected, atol=1e-6)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
        self.assertFalse(model.codebooks[0]._forward_hooks)

    def test_model_and_hooks_restore_on_error(self):
        model = real_model().train()
        model.decoders.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with vt.frozen_checkpoint(model):
                with patch.object(model, "forward", side_effect=RuntimeError("failure")):
                    vt.decoder_features(model, [batch()], "cpu", 2)
        self.assertTrue(model.training)
        self.assertFalse(model.decoders.training)
        self.assertFalse(model.codebooks[0]._forward_hooks)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_gradient_directions_do_not_use_labels_and_tiny_step_descends(self):
        model = real_model().eval()
        batches = [batch(4), batch(4)]
        args = settings(patch_foreground_mask=True)
        with vt.frozen_checkpoint(model) as restore:
            params = [p for _, p in vt._encoder_params(model)]
            grads, ref, stats = vt.collect_directions(model, batches, args, "cpu", (2, 2, 2), params)
            for b in batches:
                b.pop("gt_latents")
            other, _, _ = vt.collect_directions(model, batches, args, "cpu", (2, 2, 2), params)
            for key in grads:
                torch.testing.assert_close(grads[key], other[key], rtol=0, atol=0)
                restore()
                self.assertGreater(float(grads[key].norm()), 0)
                vt.temporary_step(params, grads[key], 1e-5)
                after = vt.measure_gap_losses(model, batches, args, "cpu", (2, 2, 2), ref)
                self.assertLess(after[key], stats[key]["unweighted_loss"])

    def test_rejects_incompatible_run_instead_of_silently_changing_the_loss(self):
        for override in (
            dict(contrastive_proj_dim=12),
            dict(vqvae_nb_levels=2),
            dict(mask_mode="onthefly"),
            dict(split_encoder_norm=True),
            dict(bt_gap_weight=0),
        ):
            with self.assertRaises(ValueError):
                vt.validate(settings(**override))

    def test_complete_run_saves_paired_results_and_restores_real_model(self):
        from threadpoolctl import threadpool_limits

        model = real_model().eval()
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        args = settings(bt_gap_lambda=0)  # Exercise the zero-direction skip too.

        class Samples:
            res = 8

            def __init__(self, n, split):
                self.n, self.split = n, split

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                rng = np.random.default_rng(i + (10000 if self.split == "val" else 20000))
                target = float(rng.normal())
                x = torch.tensor(rng.normal(size=(1, 8, 8, 8)), dtype=torch.float32) + target
                return {
                    "image": [x, x * 0.7],
                    "mask": [torch.ones_like(x)] * 2,
                    "gt_latents": {"z_content": torch.tensor([0.0, target])},
                }

        fake = types.ModuleType("eval.run_dci_synthetic")
        fake.load_run_args = lambda *a: args
        fake.load_model_from_run_dir = lambda *a, **kw: (model, args, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            torch.save({"encoders": saved}, checkpoint)
            original_bytes = checkpoint.read_bytes()
            cli = types.SimpleNamespace(
                run_dir=temp,
                checkpoint="model.pt",
                device="cpu",
                seed=0,
                grad_batch_size=4,
                grad_batches=2,
                fit_samples=13,
                test_samples=13,
                routing_samples=0,
                routing_batch=2,
                probe_causal="iid",
                cache_images=False,
                encode_batch=4,
                probe_grid=2,
                eps=0.25,
                out=str(Path(temp) / "result"),
                relative_steps=[1e-5],
            )
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), patch.object(
                vt, "make_dataset", side_effect=lambda args, n, causal, split: Samples(n, split)
            ), threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
                report = vt.run(cli)
            self.assertTrue(report["restoration_verified"])
            self.assertEqual(len(report["trials"]), 1)
            self.assertEqual(report["trials"][0]["direction"], "gap_mse")
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            disk = json.loads((Path(cli.out) / "summary.json").read_text())
            self.assertTrue(disk["restoration_verified"])
            self.assertTrue((Path(cli.out) / "probe_deltas.csv").exists())
            arrays = np.load(Path(cli.out) / "baseline_features.npz")
            self.assertEqual(len(arrays["targets"]), 26)
            self.assertTrue(all(abs(v) < 1e-10 for v in report["null_replay_delta_r2"].values()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
