"""Registered local statistics, projector gradients and real training integration."""

import ast
import importlib.util
import logging
import math
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training.vicreg_local import (
    RegisteredVICReg,
    attach_vicregl_heads,
    registered_level_loss,
    validate_vicregl_args,
    vicreg_position_terms,
)
from utils.config import parse_args, update_args

ROOT = Path(__file__).resolve().parents[1]


class RegisteredVICRegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        spec = importlib.util.spec_from_file_location("hsic_fixtures", ROOT / "tests/test_style_hsic.py")
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        cls.Model = fixtures.load_model()

    def setUp(self):
        torch.manual_seed(10)

    def args(self, *flags):
        args = parse_args().parse_args(
            [
                "--dataset-name",
                "synthetic",
                "--vqvae-nb-levels",
                "1",
                "--vqvae-hidden-channels",
                "8",
                "--vqvae-embed-dim",
                "4",
                "--vqvae-scaling-rates",
                "2",
                "--content-size",
                "6",
                "--mask-mode",
                "fixed",
                "--content-style-levels",
                "0",
                "--inject-style-to-decoder",
                "--patch-contrastive",
                "--patch-grid",
                "2",
                "2",
                "2",
                "--batch-size",
                "4",
                "--contrastive-loss-type",
                "vicregl",
                "--scale-recon-loss",
                "1",
                "--vicregl-local-dim",
                "4",
                "--vicregl-global-dim",
                "4",
                "--vicregl-hidden",
                "8",
                *flags,
            ]
        )
        fake = types.ModuleType("data.datasets")
        fake.SyntheticBrainDataset = object
        with patch.dict("sys.modules", {"data.datasets": fake}):
            return update_args(args)

    def model(self):
        return self.Model(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            nb_levels=1,
            embed_dim=4,
            nb_entries=8,
            scaling_rates=[2],
            content_size=6,
            style_size=2,
            content_style_levels=[0],
            mask_mode="fixed",
            inject_style_to_decoder=True,
            style_injection_mode="input",
            norm_type="layer",
            use_checkpoint=False,
        )

    def test_shared_positional_anatomy_cannot_satisfy_local_variance(self):
        z = torch.randn(1, 32, 4).expand(8, -1, -1)
        terms = vicreg_position_terms(z, z)
        self.assertAlmostEqual(float(terms["var"]), 0.99, places=5)
        self.assertEqual(float(terms["sim"]), 0)
        # Across-subject variance satisfies the criterion even if spatially constant.
        subject = torch.randn(8, 1, 4) * 3
        self.assertLess(
            float(vicreg_position_terms(subject.expand(-1, 32, -1), subject.expand(-1, 32, -1))["var"]), 0.05
        )

    def test_covariances_do_not_cancel_between_positions(self):
        a = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        x = torch.stack((torch.stack((a, a), -1), torch.stack((a, -a), -1)), 1)
        terms = vicreg_position_terms(x, x)
        self.assertAlmostEqual(float(terms["cov"]), (4 / 3) ** 2, places=5)

    def test_offset_alignment_is_not_erased_by_centering(self):
        x = torch.randn(8, 4, 3)
        self.assertAlmostEqual(float(vicreg_position_terms(x, x + 2)["sim"]), 4, places=5)

    def test_masked_statistics_use_only_valid_subjects_and_positions(self):
        x, y = torch.randn(4, 3, 2), torch.randn(4, 3, 2)
        valid = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
        x.requires_grad_()
        actual = vicreg_position_terms(x, y, valid)
        expected = vicreg_position_terms(x[:2, :1], y[:2, :1])
        for key in ("sim", "var", "cov"):
            torch.testing.assert_close(actual[key], expected[key])
        sum(actual[k] for k in ("sim", "var", "cov")).backward()
        self.assertEqual(float(x.grad[~valid].abs().sum()), 0)
        self.assertAlmostEqual(actual["eligible_fraction"], 1 / 3, places=6)

    def test_constant_and_empty_support_have_finite_gradients(self):
        for valid in (None, torch.zeros(4, 3, dtype=torch.bool)):
            x = torch.zeros(4, 3, 2, requires_grad=True)
            terms = vicreg_position_terms(x, x, valid)
            sum(terms[k] for k in ("sim", "var", "cov")).backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_projectors_are_independent_and_global_pools_before_projection(self):
        objective = RegisteredVICReg(3, local_dim=4, global_dim=5, hidden=7)
        x = torch.randn(2, 8, 3, 6, requires_grad=True)
        seen = []
        handle = objective.global_head.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach()))
        loss = objective(x)
        handle.remove()
        torch.testing.assert_close(seen[0], x.mean(-1))
        loss.backward()
        self.assertGreater(float(x.grad.norm()), 0)
        for head in (objective.local_head, objective.global_head):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertTrue(
            set(map(id, objective.local_head.parameters())).isdisjoint(map(id, objective.global_head.parameters()))
        )

    def test_no_head_and_arm_weights_match_manual_terms(self):
        x = torch.randn(2, 8, 3, 6)
        model = RegisteredVICReg(3, no_projectors=True, local_weight=2, global_weight=0.5)
        self.assertEqual(len(list(model.parameters())), 0)
        local = vicreg_position_terms(x[0].permute(0, 2, 1), x[1].permute(0, 2, 1))
        glob = vicreg_position_terms(x[0].mean(-1)[:, None], x[1].mean(-1)[:, None])
        expected = 2 * (25 * local["sim"] + 25 * local["var"] + local["cov"])
        expected += 0.5 * (25 * glob["sim"] + 25 * glob["var"] + glob["cov"])
        torch.testing.assert_close(model(x), expected)

    def test_content_selection_has_no_style_gradient(self):
        x = torch.randn(2, 8, 5, 6, requires_grad=True)
        RegisteredVICReg(3)(x, [[0, 2, 4]], [(0, 1)]).backward()
        self.assertEqual(float(x.grad[:, :, [1, 3]].abs().sum()), 0)

    def test_amp_and_invalid_batch(self):
        model = RegisteredVICReg(3)
        x = torch.randn(2, 8, 3, 6, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(x)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaisesRegex(ValueError, "subjects>=2"):
            model(x[:, :1])

    def test_configuration_and_incompatible_flags(self):
        args = self.args()
        self.assertEqual(args.contrastive_loss_type, "vicregl")
        for key, value in (
            ("patch_contrastive", False),
            ("mask_mode", "learned"),
            ("vqvae_nb_levels", 2),
            ("use_moco", True),
            ("contrastive_proj_dim", 16),
            ("patch_center_mode", "position"),
            ("vicregl_local_weight", float("nan")),
            ("vicreg_std_coeff", 0),
        ):
            invalid = types.SimpleNamespace(**vars(args))
            setattr(invalid, key, value)
            with self.assertRaises(ValueError):
                validate_vicregl_args(invalid)
        self.assertEqual(parse_args().parse_args([]).contrastive_loss_type, "infonce")

    def test_experiment_yaml_resolves_to_valid_training_flags(self):
        from scripts.launch import config_to_cli_args, resolve_config

        config = resolve_config(ROOT / "experiments/synthetic_causal_vicregl.yaml", "local", {})
        args = parse_args().parse_args(config_to_cli_args(config))
        validate_vicregl_args(args)
        self.assertFalse(args.resume_training)
        self.assertEqual(args.patch_grid, [8, 8, 8])
        self.assertEqual(args.content_size, 12)
        config["vicregl_no_projectors"] = True
        self.assertTrue(parse_args().parse_args(config_to_cli_args(config)).vicregl_no_projectors)

    def test_shared_eval_loader_restores_heads_before_loading(self):
        args = self.args()
        args.vqvae_res_channels = 4
        args.vqvae_nb_res_layers = 1
        args.vqvae_nb_entries = 8
        args.norm_type = "layer"
        args.style_injection_mode = "input"
        model = self.model()
        attach_vicregl_heads(model, args)
        path = ROOT / "eval/run_dci_synthetic.py"
        function = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "load_model_from_run_dir"
        )
        namespace = dict(torch=torch, os=os, logger=logging.getLogger(__name__), load_run_args=lambda _: args)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        fake = types.ModuleType("models.vqvae")
        fake.VQVAE = self.Model
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"models.vqvae": fake}):
            torch.save({"encoders": model.state_dict(), "step": 1}, Path(tmp) / "vqvae_model.pt")
            loaded, _, _ = namespace["load_model_from_run_dir"](tmp, device="cpu")
        loaded.load_state_dict(model.state_dict(), strict=True)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[k], value, rtol=0, atol=0)

    def test_actual_train_step_updates_encoder_and_both_heads_without_labels(self):
        path = ROOT / "training/main_multimodal.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_step")
        namespace = dict(
            torch=torch,
            F=F,
            math=math,
            autocast=torch.autocast,
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
            logger=logging.getLogger(__name__),
            registered_level_loss=registered_level_loss,
            NAN_SKIPPED_STEPS=0,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        args = self.args("--patch-foreground-mask")
        model = self.model().eval()
        attach_vicregl_heads(model, args)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = dict(
            image=[torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8)],
            mask=[torch.ones(4, 1, 8, 8, 8), torch.ones(4, 1, 8, 8, 8)],
        )

        def should_not_call(*a, **kw):
            raise AssertionError("Legacy loss dispatcher called")

        result = namespace["train_step"](
            data,
            [model],
            [],
            should_not_call,
            optimizer,
            list(model.parameters()),
            args,
            recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
            force_compute_recon=True,
            patch_loss_func=should_not_call,
        )
        for prefix in ("encoders.", "_vicregl_heads.L0.local_head.", "_vicregl_heads.L0.global_head."):
            self.assertTrue(
                any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith(prefix)),
                prefix,
            )
        self.assertIn("Contrastive/vicregl_local_var_L0", result[-1])
        self.assertTrue(all(math.isfinite(float(v)) for v in result[:4]))
        # Heads participate in strict state round trips and don't alter VQVAE outputs.
        clone = self.model().eval()
        attach_vicregl_heads(clone, args)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "test.pt"
            torch.save(model.state_dict(), checkpoint)
            clone.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        for k, v in clone.state_dict().items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
