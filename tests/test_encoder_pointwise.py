"""The fully local encoder and its pooled objective: reach, bound, rows, loss terms, rules, the recipe and a real run."""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

import eval.encoder.encoder_receptive_field as reach
import eval.encoder.encoder_spatial_target_audit as audit
import eval.encoder.encoder_target_protocol as protocol
import eval.encoder.spatial_recovery_monitor as monitor
import training.losses as losses
from eval.protocol.score_checkpoint import build_model
from models.conv_footprint import conv_footprint, interior_bins, receptive_field_voxels
from models.multiview_encoder import MultiviewConvEncoder
from models.vqvae import Encoder
from training import main_conv_synthetic as trainer

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(
    hidden_channels=8,
    res_channels=4,
    nb_res_layers=0,
    latent_dim=7,
    content_channels=4,
    norm_type="layer",
    activation="gelu",
    content_bound="sigmoid",
)
LOCAL = (
    "--res 32 --hidden-channels 8 --res-channels 4 --nb-res-layers 0 --latent-dim 7 --content-channels 4 "
    "--norm-type layer --activation gelu --content-bound sigmoid --global-loss-weight 0 --pointwise-loss-weight 1 "
    "--train-patch-grid 8 8 8 --pointwise-rows 64 --best-metric none"
).split()


def refused(argv):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            trainer.parse_args(argv)
        except SystemExit:
            return True
    return False


class PaddedConv(nn.Module):
    """A conv with its padding made an explicit constant, so a test can change what padding reads."""

    def __init__(self, conv, value):
        super().__init__()
        self.conv, self.width, self.value = conv, conv.padding[0], value
        conv.padding = (0, 0, 0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.width,) * 6, value=self.value["now"]))


def open_residual_blocks(encoder):
    for name, parameter in encoder.named_parameters():
        if name.endswith("alpha"):
            parameter.data.fill_(1.0)


def cells_reading_padding(encoder, size):
    """(d, h, w) bool: the map cells that move when every conv's padding constant changes."""
    value = {"now": 0.0}
    for parent in list(encoder.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv3d):
                setattr(parent, name, PaddedConv(child, value))
    x = torch.randn(1, 1, size, size, size)
    with torch.no_grad():
        first = encoder(x)
        value["now"] = 1.0
        second = encoder(x)
    return (first - second).abs().amax(1)[0] > 0


class ReachTests(unittest.TestCase):
    def test_interior_positions_are_exactly_the_cells_padding_cannot_reach(self):
        for downscale, blocks in ((4, 0), (4, 1), (4, 2), (2, 1)):
            with self.subTest(downscale=downscale, blocks=blocks):
                config = {**SMALL, "downscale_factor": downscale, "nb_res_layers": blocks}
                model = MultiviewConvEncoder(**config)
                open_residual_blocks(model.encoder)
                interior = model.interior_positions((64,) * 3, (64 // downscale,) * 3)
                self.assertTrue(torch.equal(~cells_reading_padding(model.encoder, 64).flatten(), interior))

    def test_the_numbers_quoted_for_the_current_backbone(self):
        self.assertEqual(conv_footprint(4, 2), (4, -23, 26))
        self.assertEqual((receptive_field_voxels(4, 2), receptive_field_voxels(4, 0)), (50, 18))
        self.assertEqual((len(interior_bins(64, 16, 4, 2)), len(interior_bins(64, 16, 4, 0))), (4, 12))
        self.assertEqual((len(interior_bins(64, 8, 4, 2)), len(interior_bins(64, 8, 4, 0))), (2, 6))

    def test_gradient_support_of_a_cell_is_its_footprint(self):
        for blocks in (0, 2):
            with self.subTest(blocks=blocks):
                encoder = MultiviewConvEncoder(**{**SMALL, "nb_res_layers": blocks}).encoder
                open_residual_blocks(encoder)
                x = torch.randn(1, 1, 64, 64, 64, requires_grad=True)
                # A random projection: channel norm makes a cell's plain channel sum exactly constant.
                cell = encoder(x)[0, :, 8, 8, 8]
                gradient = torch.autograd.grad((cell * torch.randn_like(cell)).sum(), x)[0][0, 0]
                touched = (gradient != 0).nonzero()
                stride, low, high = conv_footprint(4, blocks)
                self.assertEqual(touched.min(0).values.tolist(), [stride * 8 + low] * 3)
                self.assertEqual(touched.max(0).values.tolist(), [stride * 8 + high] * 3)

    def test_a_coarser_bin_is_interior_only_if_every_cell_it_averages_is(self):
        for size, bins, blocks in ((64, 8, 2), (64, 8, 0), (64, 5, 0), (64, 3, 0), (80, 7, 1), (96, 11, 1)):
            with self.subTest(size=size, bins=bins, blocks=blocks):
                cells = size // 4
                kept = set(interior_bins(size, cells, 4, blocks))
                inner = torch.tensor([float(i in kept) for i in range(cells)])
                pooled = F.adaptive_avg_pool1d(inner[None, None], bins)[0, 0]
                expected = (pooled > 1 - 1e-6).nonzero().flatten().tolist()
                self.assertEqual(interior_bins(size, bins, 4, blocks), expected)

    def test_a_voxel_moves_only_the_cells_that_read_it_under_channel_norm(self):
        x = torch.randn(1, 1, 64, 64, 64)
        moved = x.clone()
        moved[0, 0, 30, 33, 28] += 1.0
        stride, low, high = conv_footprint(4, 0)
        reads = [(v - high) / stride <= j <= (v - low) / stride for v, j in zip((30, 33, 28), (0, 0, 0))]
        self.assertEqual(reads, [False, False, False])
        axes = [torch.arange(16) * stride for _ in range(3)]
        spans = [(axis + low <= v) & (v <= axis + high) for axis, v in zip(axes, (30, 33, 28))]
        expected = spans[0][:, None, None] & spans[1][None, :, None] & spans[2][None, None, :]
        for norm in ("layer", "group"):
            with self.subTest(norm=norm), torch.no_grad():
                encoder = MultiviewConvEncoder(**{**SMALL, "norm_type": norm}).encoder
                changed = (encoder(x) - encoder(moved)).abs().amax(1)[0] > 0
                self.assertTrue(torch.equal(changed, expected) if norm == "layer" else bool(changed.all()))

    def test_the_resnet_has_no_computed_reach(self):
        model = MultiviewConvEncoder(latent_dim=7, content_channels=4, encoder_architecture="resnet18")
        with self.assertRaises(ValueError):
            model.interior_positions((64,) * 3, (2,) * 3)


class ActivationTests(unittest.TestCase):
    def test_relu_is_the_default_and_every_choice_draws_the_same_weights(self):
        def kinds(module):
            return {type(part) for part in module.modules()}

        config = {**SMALL, "nb_res_layers": 1, "conv_readout": "mlp"}
        torch.manual_seed(3)
        relu = MultiviewConvEncoder(**{**config, "activation": "relu"})
        self.assertTrue({nn.ReLU, nn.LeakyReLU} <= kinds(relu))
        self.assertIn(nn.ReLU, kinds(Encoder(1, 8, 4, 1, 4)))
        self.assertNotIn(nn.GELU, kinds(Encoder(1, 8, 4, 1, 4)))
        for name, smooth in (("gelu", nn.GELU), ("silu", nn.SiLU)):
            with self.subTest(name=name):
                torch.manual_seed(3)
                model = MultiviewConvEncoder(**{**config, "activation": name})
                self.assertIn(smooth, kinds(model))
                self.assertFalse({nn.ReLU, nn.LeakyReLU} & kinds(model))
                self.assertEqual(list(model.state_dict()), list(relu.state_dict()))
                for key, value in relu.state_dict().items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_it_refuses_what_it_does_not_reach(self):
        for changes in ({"activation": "tanh"}, {"encoder_architecture": "resnet18"}, {"conv_readout": "dense"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MultiviewConvEncoder(**{**SMALL, "content_bound": "none", **changes})


class BoundTests(unittest.TestCase):
    def test_every_output_and_every_direct_reader_sees_the_bounded_code(self):
        x = 30 * torch.randn(8, 1, 32, 32, 32)
        for readout in ("linear", "mlp"):
            with self.subTest(readout=readout):
                model = MultiviewConvEncoder(**SMALL, conv_readout=readout, encoder_head_hidden=8)
                pooled, patches = model.global_and_patch_features(x, (4, 4, 4), n_views=2)
                h = model._encode(x, 2, None)
                if readout == "mlp":
                    # As eval/encoder/regional_probe.py and eval/lesion/checkpoint_lesion_analysis.py read it.
                    direct = model.patch_readout(h.flatten(2).transpose(1, 2)).transpose(1, 2)
                else:
                    direct = model.to_encoding(h).flatten(2)
                outputs = dict(
                    pooled=pooled,
                    patches=patches,
                    direct=direct,
                    unpooled=model(x, n_views=2)[2][0].flatten(2),
                    gridded=model(x, pool_only=True, n_views=2, patch_grid=(4, 4, 4))[2][0],
                    glob=model(x, pool_only=True, n_views=2)[2][0],
                )
                for name, output in outputs.items():
                    self.assertTrue(((output[:, :4] > 0) & (output[:, :4] < 1)).all(), name)
                self.assertTrue(((direct[:, 4:] < 0) | (direct[:, 4:] > 1)).any())

    def test_a_linear_head_reports_the_mean_of_its_cells(self):
        model = MultiviewConvEncoder(**SMALL)
        x = torch.randn(4, 1, 32, 32, 32)
        pooled, patches = model.global_and_patch_features(x, (8, 8, 8), n_views=2)
        torch.testing.assert_close(pooled, patches.mean(2))

    def test_the_bound_draws_no_random_numbers(self):
        def built(bound, readout):
            torch.manual_seed(4)
            return MultiviewConvEncoder(**{**SMALL, "content_bound": bound}, conv_readout=readout)

        mlp_plain, mlp_bound = built("none", "mlp"), built("sigmoid", "mlp")
        for key, value in mlp_plain.state_dict().items():
            self.assertTrue(torch.equal(mlp_bound.state_dict()[key], value), key)
        linear_plain, linear_bound = built("none", "linear"), built("sigmoid", "linear")
        self.assertTrue(torch.equal(linear_plain.to_encoding.weight, linear_bound.to_encoding[0].weight))

    def test_it_refuses_what_a_fixed_shared_head_cannot_cover(self):
        for changes in (
            {"mask_mode": "learned"},
            {"conv_readout": "dense", "activation": "relu"},
            {"global_pool": "moments"},
            {"global_pool": "attention"},
            {"conv_readout": "mlp", "separate_spatial_readout": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MultiviewConvEncoder(**{**SMALL, **changes})


class RowTests(unittest.TestCase):
    def test_rows_stay_inside_the_interior_and_each_subjects_own_brain(self):
        # Subject 0 holds brain in the low half along depth, subject 1 in the high half. Subject 1's FLAIR adds cell 0.
        masks = torch.zeros(4, 1, 16, 16, 16)
        masks[[0, 2], :, :8] = 1
        masks[[1, 3], :, 8:] = 1
        masks[3, :, :8, :8, :8] = 1
        interior = torch.tensor([True, False, False, False, False, False, True, True])
        args = SimpleNamespace(pointwise_rows=100, train_patch_grid=[2, 2, 2], patch_foreground_thresh=0.5)
        subject, position = trainer.pointwise_rows(args, 2, 8, masks, interior)
        self.assertEqual(set(zip(subject.tolist(), position.tolist())), {(0, 0), (1, 0), (1, 6), (1, 7)})

    def test_rows_are_a_uniform_draw_without_replacement_and_replay_from_the_seed(self):
        args = SimpleNamespace(pointwise_rows=50)
        torch.manual_seed(9)
        first = trainer.pointwise_rows(args, 8, 40)
        torch.manual_seed(9)
        second = trainer.pointwise_rows(args, 8, 40)
        self.assertTrue(torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]))
        self.assertEqual(len(set(zip(first[0].tolist(), first[1].tolist()))), 50)
        self.assertGreater(first[0].unique().numel(), 4)
        self.assertGreater(first[1].unique().numel(), 15)

    def test_too_few_rows_are_refused_rather_than_averaged_over_nothing(self):
        with self.assertRaises(ValueError):
            trainer.pointwise_rows(SimpleNamespace(pointwise_rows=8), 2, 4, interior=torch.zeros(4, dtype=torch.bool))

    def test_negative_sets(self):
        subject, position = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
        self.assertTrue(torch.equal(trainer.pointwise_exclusions(subject, position, "pooled"), torch.eye(4).bool()))
        strict = trainer.pointwise_exclusions(subject, position, "strict")
        self.assertEqual((~strict).nonzero().tolist(), [[0, 3], [1, 2], [2, 1], [3, 0]])


class LossTests(unittest.TestCase):
    @staticmethod
    def distance(x, y, kernel):
        if kernel == "torus":
            return 1 - torch.cos(2 * math.pi * (x - y)).mean()
        return 2 * math.pi**2 * (x - y).square().mean()

    def test_terms_match_a_written_out_sum(self):
        torch.manual_seed(0)
        a, b = torch.rand(6, 3), torch.rand(6, 3)
        exclude = torch.eye(6, dtype=torch.bool)
        tau = 0.2
        for kernel in ("torus", "box"):
            for cross in (True, False):
                with self.subTest(kernel=kernel, cross=cross):
                    terms = losses.pointwise_alignment_entropy((a, b), (a, b), exclude, tau, kernel, cross)
                    sums = []
                    for x, y in ((a, b), (b, a)):
                        for i in range(6):
                            logits = [-self.distance(x[i], y[j], kernel) / tau for j in range(6) if j != i]
                            if not cross:
                                logits += [-self.distance(x[i], x[j], kernel) / tau for j in range(6) if j != i]
                            sums.append(torch.logsumexp(torch.stack(logits), 0))
                    torch.testing.assert_close(terms["entropy"], torch.stack(sums).mean(), rtol=1e-5, atol=1e-5)
                    align = 2 * math.pi**2 * (a - b).square().mean() / tau
                    torch.testing.assert_close(terms["align"], align)

    def test_one_tau_is_one_bandwidth_and_only_entropy_wraps(self):
        x = torch.full((1, 3), 0.5)
        near = x + 1e-2
        torus, box = losses._row_distance(x, near, "torus"), losses._row_distance(x, near, "box")
        self.assertAlmostEqual(float(torus / box), 1, places=2)
        low, high = torch.full((1, 3), 0.01), torch.full((1, 3), 0.99)
        self.assertLess(float(losses._row_distance(low, high, "torus")), 0.01)
        self.assertGreater(float(losses._row_distance(low, high, "box")), 15)
        exclude = torch.eye(2, dtype=torch.bool)
        rows = torch.cat([low, torch.full((1, 3), 0.5)])
        pair = (rows, torch.cat([high, torch.full((1, 3), 0.5)]))
        self.assertGreater(float(losses.pointwise_alignment_entropy(pair, pair, exclude, 1.0)["align"]), 8)

    def test_identical_rows_read_zero_alignment_and_the_entropy_ceiling(self):
        rows = torch.full((8, 3), 0.3)
        exclude = torch.eye(8, dtype=torch.bool)
        for kernel in ("torus", "box"):
            terms = losses.pointwise_alignment_entropy((rows, rows), (rows, rows), exclude, 0.1, kernel)
            self.assertAlmostEqual(float(terms["align"]), 0, places=5)
            self.assertAlmostEqual(float(terms["entropy"]), math.log(7), places=4)

    def test_entropy_falls_as_rows_spread_and_reads_its_own_input(self):
        torch.manual_seed(1)
        spread = torch.rand(64, 3)
        tight = 0.5 + 0.02 * (spread - 0.5)
        exclude = torch.eye(64, dtype=torch.bool)
        wide = losses.pointwise_alignment_entropy((tight, tight), (spread, spread), exclude, 0.1)
        narrow = losses.pointwise_alignment_entropy((spread, spread), (tight, tight), exclude, 0.1)
        self.assertLess(float(wide["entropy"]), float(narrow["entropy"]) - 1)
        self.assertEqual((float(wide["align"]), float(narrow["align"])), (0.0, 0.0))

    def test_a_row_without_negatives_is_dropped_and_none_left_is_refused(self):
        rows = torch.rand(3, 2)
        exclude = torch.eye(3, dtype=torch.bool)
        exclude[0] = exclude[:, 0] = True
        kept = losses.pointwise_alignment_entropy((rows, rows), (rows, rows), exclude, 0.1)
        self.assertTrue(torch.isfinite(kept["entropy"]))
        with self.assertRaises(ValueError):
            losses.pointwise_alignment_entropy((rows, rows), (rows, rows), torch.ones(3, 3).bool(), 0.1)
        with self.assertRaises(ValueError):
            losses.pointwise_alignment_entropy((rows, rows), (rows, rows), torch.zeros(3, 3).bool(), 0.1)

    def test_the_wrapped_kernel_fills_the_box_where_the_plain_one_saturates(self):
        def saturation(kernel):
            torch.manual_seed(0)
            view = nn.Linear(3, 9)
            head = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
            optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
            exclude = torch.eye(256, dtype=torch.bool)
            for _ in range(500):
                code = torch.sigmoid(head(view(2 * torch.rand(256, 3) - 1).detach()))
                entropy = losses.pointwise_alignment_entropy((code, code), (code, code), exclude, 1.0, kernel)[
                    "entropy"
                ]
                optimizer.zero_grad()
                entropy.backward()
                optimizer.step()
            with torch.no_grad():
                code = torch.sigmoid(head(view(2 * torch.rand(4000, 3) - 1)))
            return float(((code < 0.01) | (code > 0.99)).float().mean())

        torus, box = saturation("torus"), saturation("box")
        self.assertLess(torus, 0.02)
        self.assertGreater(box, 0.3)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.criteria = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()

    def test_unit_global_weight_is_the_old_objective_exactly(self):
        model = MultiviewConvEncoder(**{**SMALL, "content_bound": "none", "activation": "relu", "norm_type": "group"})
        x = torch.randn(8, 1, 32, 32, 32)
        old = SimpleNamespace(
            content_channels=4, contrastive_loss_type="infonce", tau=0.1, cross_view_negs_only=True, latent_dim=7
        )
        new = SimpleNamespace(**vars(old), global_loss_weight=1.0, pointwise_loss_weight=0.0, interior_only=False)
        self.assertEqual(
            float(trainer.training_objective(model, x, old, *self.criteria)[1]),
            float(trainer.training_objective(model, x, new, *self.criteria)[1]),
        )

    def test_zero_global_weight_leaves_only_the_pointwise_gradient(self):
        args = trainer.parse_args(LOCAL + ["--interior-only"])
        model = trainer.build_encoder(args)
        x = torch.randn(8, 1, 32, 32, 32)

        def gradient(key):
            torch.manual_seed(5)
            model.zero_grad()
            _, total, terms = trainer.training_objective(model, x, args, *self.criteria)
            (total if key == "total" else terms[key]).backward()
            return [p.grad.clone() for p in model.parameters() if p.grad is not None]

        whole, pointwise = gradient("total"), gradient("pointwise")
        self.assertTrue(whole)
        self.assertGreater(sum(float(g.abs().sum()) for g in pointwise), 0)
        for left, right in zip(whole, pointwise):
            self.assertTrue(torch.equal(left, right))

    def test_total_adds_every_weighted_term(self):
        argv = LOCAL + ["--global-loss-weight", "0.25", "--pointwise-loss-weight", "2", "--patch-loss-weight", "0.5"]
        args = trainer.parse_args(argv)
        model = trainer.build_encoder(args)
        _, total, terms = trainer.training_objective(model, torch.randn(8, 1, 32, 32, 32), args, *self.criteria)
        expected = 0.25 * terms["global"] + 0.5 * terms["patch"] + 2 * terms["pointwise"]
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(
            terms["pointwise"], terms["pointwise_align"] + terms["pointwise_entropy"], rtol=1e-5, atol=1e-5
        )

    def test_interior_only_restricts_both_losses_to_positions_that_read_no_padding(self):
        args = trainer.parse_args(LOCAL + ["--patch-loss-weight", "1", "--interior-only"])
        model = trainer.build_encoder(args)
        interior = model.interior_positions((32,) * 3, args.train_patch_grid)
        self.assertEqual(int(interior.sum()), 4**3)
        patches = torch.rand(8, 7, 512, requires_grad=True)
        restricted = trainer.patch_contrastive_loss(patches, model, args, *self.criteria, interior=interior)
        by_hand = trainer.patch_contrastive_loss(patches[..., interior], model, args, *self.criteria)
        torch.testing.assert_close(restricted, by_hand)
        trainer.pointwise_loss(patches, args, interior=interior).backward()
        touched = patches.grad.abs().sum((0, 1)) > 0
        self.assertTrue(touched[interior].any())
        self.assertFalse(touched[~interior].any())

    def test_foreground_and_interior_meet_in_the_patch_loss(self):
        args = trainer.parse_args(LOCAL + ["--patch-loss-weight", "1", "--interior-only", "--patch-foreground-mask"])
        model = trainer.build_encoder(args)
        interior = model.interior_positions((32,) * 3, args.train_patch_grid)
        foreground = torch.zeros(8, 1, 32, 32, 32, dtype=torch.bool)
        foreground[..., :16] = True
        patches = torch.rand(8, 7, 512)
        keep = trainer.foreground_positions(foreground, args.train_patch_grid, 0.05) & interior
        self.assertTrue(0 < int(keep.sum()) < int(interior.sum()))
        torch.testing.assert_close(
            trainer.patch_contrastive_loss(patches, model, args, *self.criteria, foreground, None, interior),
            trainer.patch_contrastive_loss(
                patches[..., keep], model, trainer.parse_args(LOCAL + ["--patch-loss-weight", "1"]), *self.criteria
            ),
        )

    def test_a_subject_independent_template_fills_raw_entropy_but_reads_no_subject(self):
        torch.manual_seed(2)
        template = torch.rand(1, 7, 512).expand(8, -1, -1).clone()
        raw = trainer.parse_args(LOCAL)
        residual = trainer.parse_args(LOCAL + ["--pointwise-entropy-input", "residual"])
        on_raw = trainer.pointwise_loss(template, raw)._pointwise_terms
        on_residual = trainer.pointwise_loss(template, residual)._pointwise_terms
        self.assertLess(on_raw["subject_share"], 1e-6)
        self.assertAlmostEqual(on_raw["align"], 0, places=5)
        self.assertLess(on_raw["entropy"], math.log(63) - 1)
        self.assertAlmostEqual(on_residual["entropy"], math.log(63), places=4)

    def test_view_agreement_and_residual_rank_see_past_the_template(self):
        torch.manual_seed(4)
        shared = torch.rand(400, 5)
        self.assertAlmostEqual(trainer.view_agreement(shared, 3 * shared + 1), 1, places=5)
        self.assertLess(abs(trainer.view_agreement(shared, torch.rand(400, 5))), 0.15)
        template = torch.rand(1, 7, 512).expand(8, -1, -1).clone()
        parts = trainer.pointwise_loss(template, trainer.parse_args(LOCAL))._pointwise_terms
        self.assertGreater(parts["eff_rank"], 2)
        self.assertEqual(parts["residual_rank"], 0)

    def test_subject_share_is_one_for_a_code_with_no_template(self):
        self.assertGreater(trainer.subject_share(torch.rand(32, 4, 100), torch.arange(100)), 0.95)

    def test_strict_negatives_and_within_view_rows_reach_the_loss(self):
        torch.manual_seed(3)
        patches = torch.rand(8, 7, 512)
        base = float(trainer.pointwise_loss(patches, trainer.parse_args(LOCAL)))
        for extra in (["--pointwise-negatives", "strict"], ["--no-cross-view-negs-only"]):
            torch.manual_seed(3)
            self.assertNotAlmostEqual(
                float(trainer.pointwise_loss(patches, trainer.parse_args(LOCAL + extra))), base, 3
            )


class FlagTests(unittest.TestCase):
    def test_the_recipe_flags_pass_and_the_default_run_has_none_of_them(self):
        self.assertFalse(refused(LOCAL + ["--interior-only", "--patch-foreground-mask"]))
        defaults = trainer.parse_args([])
        self.assertEqual(
            (defaults.activation, defaults.content_bound, defaults.global_loss_weight, defaults.pointwise_loss_weight),
            ("relu", "none", 1.0, 0.0),
        )
        self.assertFalse(defaults.interior_only)

    def test_refused_combinations(self):
        base = LOCAL
        no_bound = [flag for flag in LOCAL if flag not in ("--content-bound", "sigmoid")]
        no_loss = "--res 32 --hidden-channels 8 --res-channels 4 --nb-res-layers 0".split()
        for name, argv in {
            "pointwise without a bound": no_bound,
            "rows": base + ["--pointwise-rows", "1"],
            "tau": base + ["--pointwise-tau", "0"],
            "weight": base + ["--pointwise-loss-weight", "-1"],
            "global weight": base + ["--global-loss-weight", "-1"],
            "projector": base + ["--contrastive-proj-dim", "8"],
            "grid": base + ["--train-patch-grid", "9", "9", "9"],
            "no loss at all": no_loss + ["--global-loss-weight", "0"],
            "options without the loss": no_loss + ["--pointwise-kernel", "box"],
            "interior without a local loss": no_loss + ["--interior-only"],
            "foreground without a local loss": no_loss + ["--patch-foreground-mask"],
            "interior on the resnet": base + ["--interior-only", "--encoder-architecture", "resnet18"],
            "receptive field wider than the volume": base + ["--interior-only", "--nb-res-layers", "2"],
            "bound with a learned mask": no_loss + ["--content-bound", "sigmoid", "--mask-mode", "learned"],
            "bound with a dense readout": no_loss + ["--content-bound", "sigmoid", "--conv-readout", "dense"],
            "bound with attention": no_loss + ["--content-bound", "sigmoid", "--global-pool", "attention"],
            "activation on the resnet": no_loss + ["--activation", "gelu", "--encoder-architecture", "resnet18"],
            "activation on a dense head": no_loss + ["--activation", "gelu", "--conv-readout", "dense"],
        }.items():
            with self.subTest(name=name):
                self.assertTrue(refused(argv))

    def test_a_coarse_grid_is_refused_when_no_bin_stays_inside(self):
        self.assertTrue(refused(LOCAL[:-5] + ["--train-patch-grid", "2", "2", "2", "--interior-only", "--res", "32"]))

    def test_spatial_recovery_follows_the_pointwise_grid(self):
        args = trainer.parse_args(
            LOCAL + ["--res", "64", "--spatial-recovery-eval", "--train-patch-grid", "16", "16", "16"]
        )
        self.assertEqual(args.spatial_recovery_grids, [1, 16])


class RecipeTests(unittest.TestCase):
    def test_the_recipe_holds_every_requirement(self):
        import yaml

        recipe = yaml.safe_load((ROOT / "experiments" / "encoder" / "pointwise.yaml").read_text())
        expected = dict(
            norm_type="layer",
            activation="gelu",
            nb_res_layers=0,
            content_bound="sigmoid",
            global_loss_weight=0.0,
            pointwise_loss_weight=1.0,
            train_patch_grid=[16, 16, 16],
            interior_only=True,
        )
        self.assertEqual({key: recipe[key] for key in expected}, expected)
        self.assertNotIn("global_pool", recipe)
        self.assertNotIn("stratify_batches", recipe)

    def test_channel_estimates(self):
        burden = dict(res=64, synthetic_clean_content=True, synthetic_lesion_target="burden", synthetic_lesion_count=1)
        self.assertEqual(
            reach.shared_dimensions(burden, 18), dict(scalars=7, lesion_placement=3, field_sites=0, total=10)
        )
        position = dict(res=64, synthetic_clean_content=True)
        self.assertEqual(reach.shared_dimensions(position, 18)["total"], 9)
        fields = dict(res=64, synthetic_clean_content=False)
        self.assertEqual(reach.shared_dimensions(fields, 18)["field_sites"], 3**3 + 4**3)
        self.assertEqual(reach.shared_dimensions(fields, 50)["field_sites"], 4**3 + 8**3)
        scaled = dict(position, synthetic_content_amp_scale=[1, 0, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(reach.shared_dimensions(scaled, 18)["scalars"], 4)
        self.assertIsNone(reach.shared_dimensions(dict(dataset_name="adni"), 18))


class SpatialRecoveryOnBurdenTests(unittest.TestCase):
    """The recipe's generator: the spatial protocol has to render its lesions and name its columns."""

    config = vars(
        trainer.parse_args(
            "--res 32 --batch-size 4 --num-train-samples 8 --num-val-samples 4 --synthetic-clean-content "
            "--synthetic-lesion-radius 0.25 "
            "--synthetic-lesion-placement wm_interior --synthetic-lesion-target burden --synthetic-lesion-count 1 "
            "--synthetic-normalize fixed_reference".split()
        )
    )

    def rows(self, names):
        return [
            dict(view="t1", grid=8, stage="backbone", probe="ridge", condition=condition, target=name, test_r2=0.5)
            | dict(arm="trained", delta_initial=None)
            for condition in ("observed", "shuffled")
            for name in names
        ]

    def test_targets_render_a_burden_lesion_and_keep_their_column_order(self):
        names = protocol.target_names(self.config)
        self.assertEqual(names[2:5], ("lesion_burden", "unused_3", "unused_4"))
        self.assertEqual(protocol.target_names({}), protocol.TARGETS)
        dataset = protocol.dataset(self.config, 2, "val")
        latents = dataset[0]["gt_latents"]
        target, support = protocol.sample_targets(dataset._inner, latents)
        self.assertEqual(len(target), len(names))
        self.assertGreater(float(support.sum()), 0)
        self.assertEqual(float(target[names.index("lesion_burden")]), float(latents["z_content"][2]))

    def test_summaries_read_the_burden_alone_and_position_runs_read_as_before(self):
        self.assertEqual(protocol.lesion_columns(protocol.target_names(self.config)), ("lesion_burden",))
        self.assertEqual(protocol.lesion_columns(protocol.TARGETS), ("lesion_x", "lesion_y", "lesion_z"))
        rows = self.rows(protocol.target_names(self.config))
        for row in rows:
            row["test_r2"] = 0.9 if row["target"] == "lesion_burden" else 0.0
        self.assertAlmostEqual(audit.focused_summary(rows)[0]["lesion_latent_mean_r2"], 0.9)
        with contextlib.redirect_stdout(io.StringIO()) as table:
            monitor.print_recovery(rows, 0)
        self.assertIn("burden", table.getvalue())
        self.assertIn("+0.900", table.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as table:
            monitor.print_recovery(self.rows(protocol.TARGETS), 0)
        self.assertIn("lesion_xyz", table.getvalue())

    def test_the_probes_get_the_names_they_score(self):
        banks = {
            split: ({("t1", 8, "backbone"): torch.rand(40, 6).numpy()}, torch.rand(40, 14).numpy(), {})
            for split in ("val", "test")
        }
        names = protocol.target_names(self.config)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            rows, _ = audit.score_banks(banks, SimpleNamespace(seed=0), "trained", Path(tmp), names)
        self.assertEqual({row["target"] for row in rows}, set(names))


class RealRunTests(unittest.TestCase):
    def test_a_short_run_records_the_objective_and_rebuilds_from_its_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = LOCAL + ["--interior-only", "--patch-foreground-mask", "--batch-size", "4", "--out-dir", tmp]
            argv += ["--model-id", "smoke", "--require-new-run", "--device", "cpu", "--train-steps", "3"]
            argv += ["--eval-every", "3", "--log-every", "1", "--num-train-samples", "8", "--num-val-samples", "20"]
            argv += ["--no-cache", "--no-floor-eval", "--synthetic-clean-content", "--synthetic-normalize"]
            argv += ["fixed_reference"]
            with patch.object(sys, "argv", ["trainer", *argv]), patch.object(
                trainer, "evaluate", return_value=({}, {})
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                trainer.main()
            log = output.getvalue()
            directory = Path(tmp) / "smoke"
            settings = json.loads((directory / "settings.json").read_text())
            progress = json.loads((directory / "training_progress.json").read_text())
            terms = progress["last_loss_terms"]
            self.assertEqual(progress["status"], "complete")
            self.assertAlmostEqual(terms["total"], terms["pointwise_weighted"], places=4)
            self.assertEqual(
                (progress["activation"], progress["content_bound"], progress["receptive_field_voxels"]),
                ("gelu", "sigmoid", 18),
            )
            for part in trainer.POINTWISE_PARTS:
                self.assertIn(f"pointwise_{part}", terms)
            for phrase in ("receptive field: 18 voxels", "interior only: 64/512", "cell eff_rank", "subject share"):
                self.assertIn(phrase, log)

            state = torch.load(directory / "model.pt", map_location="cpu")
            model = build_model(settings, "cpu", state)
            self.assertEqual((model.content_bound, model.activation, model.normalization), ("sigmoid", "gelu", "layer"))
            with contextlib.redirect_stdout(io.StringIO()) as probe:
                payload = reach.main(
                    ["--run-dir", str(directory), "--subjects", "1", "--cells", "2", "--draws", "1", "--device", "cpu"]
                )
            self.assertEqual(payload["hard"]["width_voxels"], 18)
            self.assertTrue(payload["hard"]["scan"]["ok"])
            self.assertEqual(payload["shared_dimensions"]["hard"]["total"], 9)
            self.assertLess(payload["effective"]["t1"]["as built"]["outside_hard_box"], 1e-6)
            self.assertIn("no operation outside the receptive field", probe.getvalue())


if __name__ == "__main__":
    unittest.main()
