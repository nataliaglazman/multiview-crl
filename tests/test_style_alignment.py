"""Pair-construction controls and real VQVAE training tests for style alignment."""

import ast
import importlib.util
import logging
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eval.synthetic.synthetic_dataset import Synthetic3DDisentanglementDataset
from training.style_alignment import within_modality_style_loss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("style_alignment_fixtures", ROOT / "tests/test_style_hsic.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def load_dataset():
    path = ROOT / "data/datasets.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "utils.utils")
        and not (isinstance(node, ast.Import) and any(a.name == "pandas" for a in node.names))
    ]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["SyntheticBrainDataset"]


def config(*extra):
    parse, update = fixtures.load_config()
    return update(
        parse().parse_args(
            [
                "--dataset-name",
                "synthetic",
                "--synthetic-mode",
                "pseudo_mri",
                "--synthetic-normalize",
                "fixed_reference",
                "--inject-style-to-decoder",
                "--mask-mode",
                "fixed",
                "--batch-size",
                "4",
                "--style-contrastive-mode",
                "within_modality",
                *extra,
            ]
        )
    )


class StyleAlignmentLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_same_style_matches_without_aligning_modalities(self):
        t1 = torch.randn(64, 3) * 3 + 20
        flair = torch.randn(64, 3) * 3 - 20
        style = torch.cat((t1, flair)).requires_grad_()
        paired = style.detach().clone().requires_grad_()
        loss, diag = within_modality_style_loss(style, paired)
        self.assertEqual(loss.item(), 0)
        self.assertGreater(diag["alignment_mismatched_mse_v0"], 1)
        self.assertGreater(diag["alignment_mismatched_mse_v1"], 1)
        loss.backward()
        self.assertTrue(torch.isfinite(style.grad).all())
        self.assertTrue(torch.isfinite(paired.grad).all())

    def test_one_view_spatial_anatomy_leak_is_penalized_despite_zero_gap(self):
        style = torch.randn(64, 3, 2) * 3
        anatomy = torch.randn(32, 1, 1)
        leak = anatomy * torch.tensor([1.0, -1.0])
        left, right = style.clone(), style.clone()
        left[:32] += leak
        right[:32] -= leak
        torch.testing.assert_close(left.mean(-1), right.mean(-1))
        left.requires_grad_()
        right.requires_grad_()
        loss, diag = within_modality_style_loss(left, right, variance_weight=0)
        self.assertGreater(diag["alignment_mse_v0"], 0.1)
        self.assertEqual(diag["alignment_mse_v1"], 0)
        loss.backward()
        self.assertGreater(left.grad[:32].norm(), 0)
        self.assertGreater(right.grad[:32].norm(), 0)
        self.assertEqual(left.grad[32:].norm(), 0)

    def test_constant_spatial_templates_fail_variance_control(self):
        style = torch.tensor([-10.0, 10.0]).expand(16, 3, 2).clone().requires_grad_()
        paired = style.detach().clone().requires_grad_()
        loss, diag = within_modality_style_loss(style, paired, variance_weight=2)
        self.assertEqual(diag["alignment_mse"], 0)
        self.assertEqual(diag["alignment_var_hinge"], 1)
        self.assertEqual(loss.item(), 2)
        loss.backward()
        self.assertTrue(torch.isfinite(style.grad).all())
        self.assertTrue(torch.isfinite(paired.grad).all())

    def test_amp_and_invalid_inputs(self):
        a = torch.randn(8, 2, 3, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = within_modality_style_loss(a, b)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        for shape in ((7, 2), (0, 2), (8, 0), ()):
            with self.assertRaises(ValueError):
                within_modality_style_loss(torch.empty(shape), torch.empty(shape))
        with self.assertRaisesRegex(ValueError, "at least two"):
            within_modality_style_loss(a[:2], b[:2])
        with self.assertRaises(ValueError):
            within_modality_style_loss(a, b[:, :1])
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                within_modality_style_loss(a, b, variance_weight=weight)


class StylePairDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Dataset = load_dataset()

    def dataset(self, **kwargs):
        options = dict(
            mode="train",
            spatial_size=(16, 16, 16),
            synthetic_num_samples=6,
            synthetic_clean_content=True,
            synthetic_normalize="fixed_reference",
            synthetic_style_alignment_pairs=True,
            cache=True,
        )
        options.update(kwargs)
        ds = self.Dataset(**options)
        # Same fixed transform for both branches; avoid estimating it in every test.
        ds._fixed_mean, ds._fixed_scale = 0.2, 0.8
        return ds

    def test_actual_pairs_share_acquisition_and_change_anatomy(self):
        ds = self.dataset()
        ds._cache[0] = ds._render(0)
        expected = ds._cache[0][-1]
        with patch.object(ds._inner, "render_pseudo_mri", wraps=ds._inner.render_pseudo_mri) as render:
            batch = ds[0]
        args, kwargs = render.call_args
        self.assertFalse(torch.equal(args[0], expected["z_content"]))
        torch.testing.assert_close(args[3], expected["z_style_v1"])
        torch.testing.assert_close(args[4], expected["z_style_v2"])
        self.assertEqual(args[5], ds._inner.sample_seed_for(0))
        self.assertIsNotNone(kwargs["noise_seed"])
        raw = ds._inner.render_pseudo_mri(*args, **kwargs)
        for view in range(2):
            torch.testing.assert_close(batch["style_pair_image"][view], (raw[view] - 0.2) / 0.8 * raw[2])
            torch.testing.assert_close(batch["style_pair_mask"][view], raw[2])
            self.assertFalse(torch.equal(batch["image"][view], batch["style_pair_image"][view]))
        torch.testing.assert_close(batch["gt_latents"]["z_content"], expected["z_content"])

    def test_pairs_refresh_despite_anchor_cache_and_validation_is_repeatable(self):
        ds = self.dataset()
        first, second = ds[0], ds[0]
        torch.testing.assert_close(first["image"][0], second["image"][0])
        self.assertFalse(torch.equal(first["style_pair_image"][0], second["style_pair_image"][0]))
        val = self.dataset(mode="val")
        with patch.object(val._inner, "_first_fitting", wraps=val._inner._first_fitting) as fitting:
            first, second = val[0], val[0]
        torch.testing.assert_close(first["style_pair_image"][0], second["style_pair_image"][0])
        self.assertTrue(all(0 <= call.args[0] < len(val) for call in fitting.call_args_list))

    def test_disabled_path_preserves_original_images(self):
        ds, plain = self.dataset(), self.dataset(synthetic_style_alignment_pairs=False)
        enabled, disabled = ds[2], plain[2]
        self.assertNotIn("style_pair_image", disabled)
        for a, b in zip(enabled["image"], disabled["image"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_noise_changes_without_changing_bias_field(self):
        inner = Synthetic3DDisentanglementDataset(
            num_samples=2, res=16, n_content=9, clean_content=True, mode="pseudo_mri"
        )
        draw = inner._draw_pseudo_mri(inner.sample_seed_for(0))
        args = [
            draw[k]
            for k in (
                "z_content",
                "z_deformation",
                "z_fissure",
                "z_style_v1",
                "z_style_v2",
            )
        ]
        noise_fields = []
        original = inner.renderer._seeded_noise

        def record(*a, **kw):
            value = original(*a, **kw)
            noise_fields.append(value.clone())
            return value

        with patch.object(inner.renderer, "_seeded_noise", side_effect=record):
            a = inner.render_pseudo_mri(*args, sample_seed=99)
            b = inner.render_pseudo_mri(*args, sample_seed=99, noise_seed=123)
            c = inner.render_pseudo_mri(*args, sample_seed=99)
        for i in range(2):
            torch.testing.assert_close(noise_fields[i], noise_fields[i + 2])
            torch.testing.assert_close(a[i], c[i], rtol=0, atol=0)
            self.assertFalse(torch.equal(a[i], b[i]))

    def test_invalid_pair_datasets_fail_early(self):
        for kwargs in (
            {"synthetic_normalize": "shared"},
            {"synthetic_num_samples": 1},
            {"synthetic_mode": "random"},
        ):
            with self.assertRaises(ValueError):
                self.dataset(**kwargs)


class StyleAlignmentTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Model = fixtures.load_model()
        source = ROOT / "training/main_multimodal.py"
        function = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "train_step"
        )
        ns = dict(
            torch=torch,
            F=F,
            math=math,
            autocast=torch.autocast,
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
            logger=logging.getLogger(__name__),
            within_modality_style_loss=within_modality_style_loss,
            NAN_SKIPPED_STEPS=0,
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            ns,
        )
        cls.train_step = staticmethod(ns["train_step"])

    def test_real_training_weight_gradients_and_encoder_only_pair(self):
        for separate in (False, True):
            torch.manual_seed(3)
            args = config("--scale-contrastive-loss", "0")
            model = self.Model(
                hidden_channels=8,
                res_channels=4,
                nb_res_layers=1,
                nb_levels=1,
                embed_dim=8,
                nb_entries=8,
                scaling_rates=[2],
                use_checkpoint=False,
                content_size=6,
                style_size=2,
                content_style_levels=[0],
                mask_mode="fixed",
                inject_style_to_decoder=True,
                norm_type="layer",
                style_spatial_size=1,
                separate_encoders=separate,
                quantize_style=True,
            ).eval()
            data = {
                "image": [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8)],
                "style_pair_image": [
                    torch.randn(4, 1, 8, 8, 8),
                    torch.randn(4, 1, 8, 8, 8),
                ],
            }  # Deliberately no gt_latents: the loss is label-free.

            def run(optimizer=None, reconstruct=True):
                return self.train_step(
                    data,
                    [model],
                    [],
                    lambda h, *a, **kw: h.sum() * 0,
                    optimizer,
                    list(model.parameters()),
                    args,
                    recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
                    force_compute_recon=reconstruct,
                )

            with patch.object(model, "forward", wraps=model.forward) as forward:
                baseline = run()
                self.assertEqual(forward.call_count, 1)
            args.scale_style_contrastive_loss = 1.7
            with patch.object(model, "forward", wraps=model.forward) as forward:
                result = run()
                self.assertEqual(
                    [c.kwargs["return_recon"] for c in forward.call_args_list],
                    [True, False],
                )
            diag = result[-1]
            self.assertAlmostEqual(
                result[0] - baseline[0],
                diag["Style/within_modality_weighted"],
                places=5,
            )
            self.assertAlmostEqual(
                diag["Style/within_modality_weighted"],
                1.7 * diag["Style/within_modality_L0"],
                places=5,
            )
            self.assertNotIn("Style/infonce_L0", diag)
            self.assertNotIn("Style/xview_hsic_L0", diag)
            self.assertEqual(result[2:4], baseline[2:4])
            before = {k: v.detach().clone() for k, v in model.named_parameters()}
            run(torch.optim.SGD(model.parameters(), lr=0.02), reconstruct=False)
            changed = [k for k, v in model.named_parameters() if not torch.equal(v, before[k])]
            self.assertTrue(any(k.startswith("encoders.") for k in changed))
            if separate:
                self.assertTrue(any(k.startswith("encoders_v1.") for k in changed))
            self.assertTrue(all(torch.isfinite(v).all() for v in model.parameters()))
            del data["style_pair_image"]
            with self.assertRaisesRegex(ValueError, "style_pair_image"):
                run()

    def test_config_validation_and_disabled_defaults(self):
        parse, update = fixtures.load_config()
        default = parse().parse_args([])
        self.assertEqual(default.style_contrastive_mode, "cosine")
        self.assertEqual(default.scale_style_contrastive_loss, 0)
        config("--scale-style-contrastive-loss", "1")
        for flags in (
            ["--synthetic-normalize", "shared"],
            ["--mask-mode", "onthefly"],
            ["--batch-size", "1"],
            ["--style-alignment-var-weight", "nan"],
            ["--style-alignment-var-weight", "-1"],
            ["--synthetic-style-scale", "0"],
        ):
            with self.assertRaises(ValueError):
                config("--scale-style-contrastive-loss", "1", *flags)
        with self.assertRaises(ValueError):
            config("--scale-style-contrastive-loss", "-1")
        # Selecting an inactive mode imposes no new requirements.
        update(parse().parse_args(["--style-contrastive-mode", "within_modality"]))

    def test_generated_pairs_with_latent_masks_train_without_factor_targets(self):
        ds = load_dataset()(
            spatial_size=(16, 16, 16),
            synthetic_num_samples=4,
            synthetic_normalize="fixed_reference",
            synthetic_clean_content=True,
            synthetic_style_alignment_pairs=True,
            synthetic_causal=True,
            synthetic_causal_graph="random",
            synthetic_causal_edge_prob=0.4,
        )
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
        del batch["gt_latents"]
        model = self.Model(
            hidden_channels=8,
            res_channels=4,
            nb_res_layers=1,
            nb_levels=1,
            embed_dim=8,
            nb_entries=8,
            scaling_rates=[2],
            use_checkpoint=False,
            content_size=6,
            style_size=2,
            content_style_levels=[0],
            mask_mode="fixed",
            inject_style_to_decoder=True,
            norm_type="layer",
            style_spatial_size=1,
            latent_mask=True,
            quantize_style=True,
        ).train()
        args = config("--scale-style-contrastive-loss", "0.1", "--scale-contrastive-loss", "0")
        quantizer_calls = []
        handles = [
            layer.register_forward_pre_hook(lambda m, a: quantizer_calls.append(m))
            for layer in list(model.codebooks) + list(model.style_codebooks.values())
        ]
        result = self.train_step(
            batch,
            [model],
            [],
            lambda h, *a, **kw: h.sum() * 0,
            torch.optim.SGD(model.parameters(), lr=0.001),
            list(model.parameters()),
            args,
            recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
            force_compute_recon=True,
        )
        for handle in handles:
            handle.remove()
        self.assertEqual(len(quantizer_calls), 2)  # One content + one style; none for the paired forward.
        self.assertGreater(result[-1]["Style/within_modality_weighted"], 0)
        self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))


if __name__ == "__main__":
    unittest.main()
