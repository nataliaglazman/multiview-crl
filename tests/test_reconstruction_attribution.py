"""CPU tests: python -m unittest discover -s tests -p test_reconstruction_attribution.py.

The shipped BT functions are loaded from their AST to avoid unrelated LPIPS/MONAI
dependencies; the numerical and autograd code under test is unchanged.
"""

import ast
import contextlib
import csv
import importlib.util
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eval import reconstruction_attribution as ra


class TinyModel(torch.nn.Module):
    """At w=.5, sim descent shrinks w and worsens reconstruction of both positive views."""

    def __init__(self):
        super().__init__()
        self.encoders = torch.nn.Conv3d(1, 1, 1, bias=False)
        self.decoder = torch.nn.Conv3d(1, 1, 1, bias=False)
        self.register_buffer("codebook_updates", torch.tensor(0))
        self.content_channels_per_level = {0: 1}
        with torch.no_grad():
            self.encoders.weight.fill_(0.5)
            self.decoder.weight.fill_(1.0)

    def forward(self, x, return_recon=True, patch_grid=None, **kwargs):
        if self.training:
            self.codebook_updates.add_(1)
        z = self.encoders(x)
        features = F.adaptive_avg_pool3d(z, patch_grid).flatten(2) if patch_grid else z.mean((2, 3, 4))
        return self.decoder(z) if return_recon else None, [], [features], None, [], [], {}, {}


def sample(i):
    x = torch.full((1, 2, 2, 2), 0.1 + i * 0.01)
    return {"image": [x, x * 2], "mask": [torch.ones_like(x), torch.ones_like(x)]}


def settings(**updates):
    args = dict(
        contrastive_loss_type="barlow_twins",
        patch_contrastive=True,
        bt_sim_coeff=0.4,
        bt_gap_sim_coeff=0.2,
        bt_patch_weight=1.0,
        bt_gap_weight=1.0,
        bt_sim_normalize=False,
        scale_contrastive_loss=10.0,
        scale_recon_loss=2.0,
        batch_size=2,
        patch_grid=(2, 2, 2),
        subsets=[(0, 1)],
        vqvae_nb_levels=1,
    )
    args.update(updates)
    return types.SimpleNamespace(**args)


class ReconstructionAttributionTests(unittest.TestCase):
    def test_similarity_matches_shipped_loss_values_and_gradients(self):
        source = Path(__file__).resolve().parents[1] / "training" / "losses.py"
        tree = ast.parse(source.read_text())
        names = {"_center_patch_features", "_merge_diags", "barlow_twins_loss"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        subset = ast.Module(body=functions, type_ignores=[])
        namespace = {"torch": torch, "F": F, "contextlib": contextlib}
        exec(compile(subset, str(source), "exec"), namespace)
        bt = namespace["barlow_twins_loss"]
        for shape in ((2, 6, 3), (2, 6, 3, 4)):
            for normalize in (False, True):
                for stat in ("fold", "per_position"):
                    with self.subTest(shape=shape, normalize=normalize, stat=stat):
                        torch.manual_seed(5)
                        hz = torch.randn(shape, requires_grad=True)
                        config = dict(
                            estimated_content_indices=[[0, 1, 2]],
                            subsets=[[0, 1]],
                            center_mode="position" if len(shape) == 4 else "none",
                            patch_stat=stat,
                            sim_normalize=normalize,
                            normalize_terms=True,
                        )
                        with_sim = bt(hz, sim_coeff=1.0, **config)
                        without = bt(hz, sim_coeff=0.0, **config)
                        actual = ra.similarity_loss(hz, normalize, stat)
                        self.assertAlmostEqual(float(actual), with_sim._contrastive_diag["sim_loss"], places=6)
                        expected_grad = torch.autograd.grad(with_sim - without, hz, retain_graph=True)[0]
                        actual_grad = torch.autograd.grad(actual, hz)[0]
                        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)

    def test_normalized_denominator_is_detached(self):
        hz = torch.tensor([[[0.0], [1.0], [2.0]], [[1.0], [3.0], [5.0]]], requires_grad=True)
        loss = ra.similarity_loss(hz, normalize=True)
        gradient = torch.autograd.grad(loss, hz)[0]
        # Scaling both inputs has zero forward derivative for a fully differentiable ratio,
        # but the shipped detached denominator gives the numerator's radial derivative 2L.
        self.assertAlmostEqual(float((gradient * hz).sum()), 2 * float(loss), places=5)

    def test_weights_preserve_explicit_zero_and_inheritance(self):
        args = settings(bt_gap_sim_coeff=0.0, contrastive_level_weights=[0.5])
        self.assertEqual(ra.sim_weights(args, 0), {"patch_sim": 2.0, "gap_sim": 0.0})
        args.bt_gap_sim_coeff = None
        self.assertEqual(ra.sim_weights(args, 0), {"patch_sim": 2.0, "gap_sim": 2.0})

    def test_pixel_metric_is_raw_and_foreground_weighted(self):
        x = torch.zeros(2, 1, 1, 1, 2)
        y = torch.tensor([2.0, 99.0, 4.0, 4.0]).reshape_as(x).requires_grad_()
        mask = torch.tensor([1.0, 0.0, 1.0, 1.0]).reshape_as(x)
        sums, counts = ra.pixel_sums(x, y, mask)
        self.assertAlmostEqual(float(sums.sum() / counts.sum()), 10 / 3, places=6)
        gradient = torch.autograd.grad(sums.sum(), y)[0]
        self.assertEqual(float(gradient.flatten()[1]), 0.0)
        self.assertEqual(float(gradient.flatten()[0]), 1.0)  # No saturation dead zone.
        sums, counts = ra.pixel_sums(x, y, mask, clamp=True)
        self.assertEqual(float(sums.sum() / counts.sum()), 1.0)

    def test_state_and_mixed_modes_restored_after_error(self):
        model = TinyModel().train()
        model.decoder.eval()
        model.decoder.weight.requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with ra.frozen_checkpoint(model):
                self.assertFalse(model.training)
                self.assertFalse(model.decoder.weight.requires_grad)
                params = [p for _, p in ra._encoder_params(model)]
                ra.temporary_step(params, torch.ones(1), 0.3)
                model.codebook_updates.add_(9)
                raise RuntimeError("intentional")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertFalse(model.decoder.training)
        self.assertFalse(model.decoder.weight.requires_grad)

    def test_per_view_masks_and_foreground_are_used(self):
        class MaskedModel(torch.nn.Module):
            def forward(self, x, **kwargs):
                features = torch.tensor(
                    [
                        [[2.0, 4.0], [200.0, 400.0]],
                        [[3.0, 5.0], [300.0, 500.0]],
                        [[900.0, 800.0], [6.0, 8.0]],
                        [[700.0, 600.0], [7.0, 9.0]],
                    ]
                )
                masks = (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
                return None, [], [features], None, [], [], {0: masks}, {}

        x = torch.ones(2, 1, 1, 1, 2)
        mask = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2).expand_as(x)
        batch = {"image": [x, x], "mask": [mask, mask]}
        selected = ra._content_features(MaskedModel(), batch, settings(patch_foreground_mask=True), "cpu", (1, 1, 2), 0)
        torch.testing.assert_close(selected, torch.tensor([[[[2.0]], [[3.0]]], [[[6.0]], [[7.0]]]]))

    def test_end_to_end_reports_known_conflict_and_restores_checkpoint(self):
        model = TinyModel()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        fake = types.ModuleType("eval.run_dci_synthetic")
        fake.load_model_from_run_dir = lambda *a, **kw: (model, settings(), torch.device("cpu"))
        fake.build_synthetic_test_set = lambda args, n, **kw: [sample(i) for i in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model.pt").touch()
            cli = types.SimpleNamespace(
                run_dir=directory,
                checkpoints=["model.pt"],
                grad_batch_size=None,
                grad_batches=2,
                grid=None,
                level=0,
                recon_samples=3,
                recon_batch_size=2,
                causal="match",
                recon_clamp=False,
                etas=[0.001, 0.002],
                out=directory,
            )
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), contextlib.redirect_stdout(io.StringIO()):
                ra.run(cli)
            with Path(directory, "model_reconstruction_attribution.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            for row in rows:
                delta = float(row["delta_mae"])
                prediction = float(row["predicted_delta_mae"])
                if row["direction"] == "recon_pixel_control":
                    self.assertLess(delta, 0)
                else:
                    self.assertLess(float(row["cos_recon"]), 0)
                    self.assertGreater(delta, 0)
                    self.assertEqual(float(row["conflict_fraction"]), 1.0)
                self.assertAlmostEqual(delta, prediction, delta=2e-8)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_real_vqvae_forward_gradients_and_codebook_freeze(self):
        # Fixed masks in eval mode never call the unrelated Gumbel utilities.
        source = Path(__file__).resolve().parents[1] / "models" / "vqvae.py"
        spec = importlib.util.spec_from_file_location("attribution_test_vqvae", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"utils.utils": types.ModuleType("utils.utils")}):
            spec.loader.exec_module(module)
        torch.manual_seed(2)
        model = module.VQVAE(
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
            separate_encoders=True,
            inject_style_to_decoder=True,
        )
        batch = {
            "image": [torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8, 8)],
            "mask": [torch.ones(2, 1, 8, 8, 8), torch.ones(2, 1, 8, 8, 8)],
        }
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with ra.frozen_checkpoint(model) as restore:
            params = [p for _, p in ra._encoder_params(model)]
            baseline, grad = ra.reconstruction_metrics(model, [batch], "cpu", params)
            self.assertTrue(torch.isfinite(grad).all())
            grads, _, _ = ra.collect_sim_gradients(
                model, [batch], settings(bt_sim_normalize=True), "cpu", (2, 2, 2), 0, params
            )
            self.assertGreater(float(grads["both_sim"].norm()), 0.0)
            ra.temporary_step(params, grads["both_sim"], 1e-5)
            measured, _ = ra.reconstruction_metrics(model, [batch], "cpu")
            self.assertTrue(torch.isfinite(torch.tensor(measured["mae"])))
            restore()
            restored, _ = ra.reconstruction_metrics(model, [batch], "cpu")
            self.assertEqual(restored, baseline)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
