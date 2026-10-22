"""--standardize-patch-code and --patch-similarity: the standardised patch code and the size-aware patch InfoNCE."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from training import encoder_args, losses  # noqa: E402
from training import main_conv_synthetic as trainer  # noqa: E402

SYNTHETIC = ["--res", "32", "--num-train-samples", "640", "--batch-size", "32", "--patch-loss-weight", "1"]
TINY = ["--res", "32", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--content-channels", "9", "--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1"]
TINY += ["--best-metric", "none", "--batch-size", "4", "--lr", "0.05", "--log-every", "1", "--device", "cpu"]
TINY += ["--num-train-samples", "8", "--num-val-samples", "64", "--no-cache", "--require-new-run"]
TINY += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference", "--train-steps", "2"]
TINY += ["--patch-loss-weight", "1", "--train-patch-grid", "4", "4", "4", "--patch-foreground-mask"]
CRITERION = torch.nn.CrossEntropyLoss()


def patch_loss(patches, standardize, similarity="cosine"):
    """The trainer's patch InfoNCE on (2B, units, positions) patches, with an identity head."""
    model = SimpleNamespace(project=lambda z: z, spatial_readout=None)
    args = SimpleNamespace(
        content_channels=patches.shape[1],
        tau=0.1,
        cross_view_negs_only=True,
        patch_loss_type="infonce",
        standardize_patch_code=standardize,
        patch_similarity=similarity,
    )
    return trainer.patch_contrastive_loss(patches, model, args, torch.nn.CosineSimilarity(dim=-1), CRITERION)


def infonce(hz, similarity, cross_view_negs_only=True):
    """losses.patch_infonce_loss on a (views, subjects, units, positions) code, all units content."""
    return losses.patch_infonce_loss(
        hz,
        None,
        CRITERION,
        tau=0.1,
        estimated_content_indices=[list(range(hz.shape[2]))],
        subsets=[(0, 1)],
        cross_view_negs_only=cross_view_negs_only,
        similarity=similarity,
    ).squeeze()


def gradient_per_cell(loss_fn, x):
    """Squared gradient of ``loss_fn`` summed over views and units: (subjects, positions)."""
    leaf = x.clone().requires_grad_()
    (grad,) = torch.autograd.grad(loss_fn(leaf), leaf)
    return grad.square().sum((0, 2))


class StandardizePatchCodeTests(unittest.TestCase):
    def test_centred_at_every_position_and_scaled_per_unit_over_positions(self):
        hz = torch.randn(2, 16, 5, 30, dtype=torch.float64) * torch.arange(1.0, 6.0)[:, None] + 3.0
        out = trainer.standardize_patch_code(hz)
        torch.testing.assert_close(out.mean(1), torch.zeros(2, 5, 30, dtype=torch.float64))
        pooled = out.square().sum((1, 3)) / ((hz.shape[1] - 1) * hz.shape[3])
        torch.testing.assert_close(pooled, torch.ones(2, 5, dtype=torch.float64), rtol=1e-5, atol=1e-5)

    def test_a_flat_position_stays_flat_and_a_rare_unit_is_scaled_up(self):
        hz = torch.randn(2, 16, 2, 20, dtype=torch.float64)
        hz[..., 0] = 7.0  # a position alike in every subject: per-position scaling would divide noise by ~0
        hz[:, :, 0] = 0.0
        hz[:, :3, 0, 5] = 0.2  # unit 0 varies only at one position, in three subjects, and weakly
        out = trainer.standardize_patch_code(hz)
        self.assertEqual(float(out[..., 0].abs().max()), 0.0)
        # Raw, unit 0 is ~10x smaller than unit 1; standardized, its rare deviations stand out instead.
        self.assertGreater(float(out[:, :, 0].abs().max()), 2 * float(out[:, :, 1].abs().max()))

    def test_the_loss_ignores_each_units_scale_and_each_positions_offset(self):
        torch.manual_seed(0)
        patches = torch.randn(16, 6, 10, dtype=torch.float64)  # 8 subjects x 2 views, 6 units, 10 positions
        moved = patches.clone()
        moved[:8, 2] *= 5.0  # one unit rescaled in view 1
        moved[8:] += torch.randn(1, 6, 10, dtype=torch.float64)  # an offset per unit and position in view 2
        for similarity in ("cosine", "dot"):
            with self.subTest(similarity=similarity):
                # Up to the 1e-6 added to each unit's spread.
                torch.testing.assert_close(
                    patch_loss(moved, True, similarity), patch_loss(patches, True, similarity), rtol=1e-5, atol=1e-5
                )
        self.assertGreater(abs(float(patch_loss(moved, False) - patch_loss(patches, False))), 1e-3)


class PatchSimilarityTests(unittest.TestCase):
    def test_dot_is_the_standardised_codes_dot_product_over_units_and_tau(self):
        torch.manual_seed(1)
        patches = torch.randn(12, 5, 7, dtype=torch.float64)  # 6 subjects x 2 views, 5 units, 7 positions
        hz = trainer.standardize_patch_code(torch.stack(patches.chunk(2, dim=0)))
        scores = torch.einsum("bcp,dcp->pbd", hz[0], hz[1]) / (5 * 0.1)
        targets = torch.arange(6).repeat(7)
        expected = F.cross_entropy(scores.reshape(42, 6), targets)
        expected = expected + F.cross_entropy(scores.transpose(1, 2).reshape(42, 6), targets)
        torch.testing.assert_close(patch_loss(patches, True, "dot"), expected)

    def test_dot_equals_cosine_when_every_cell_has_the_typical_size(self):
        torch.manual_seed(2)
        hz = torch.randn(2, 6, 5, 7, dtype=torch.float64)
        hz = hz / hz.norm(dim=2, keepdim=True) * 5**0.5  # every vector as long as one of 5 unit-variance units
        for negatives in (True, False):
            with self.subTest(cross_view_negs_only=negatives):
                torch.testing.assert_close(infonce(hz, "dot", negatives), infonce(hz, "cosine", negatives))
        with self.assertRaises(ValueError):
            infonce(hz, "euclidean")

    def test_a_position_alike_in_every_subject_drops_out_only_under_dot(self):
        torch.manual_seed(3)
        shared = torch.randn(16, 6, 10, dtype=torch.float64)
        patches = torch.cat([shared, shared]) + 0.5 * torch.randn(32, 6, 10, dtype=torch.float64)
        patches[:, :, 0] = 5.0 + 1e-3 * torch.randn(32, 6, dtype=torch.float64)  # alike in every subject
        flat_share = {}
        for similarity in ("cosine", "dot"):
            views = torch.stack(patches.chunk(2))  # the trainer's (2B, units, positions) is views x subjects
            cell = gradient_per_cell(lambda x: patch_loss(x.reshape(32, 6, 10), True, similarity), views)
            flat_share[similarity] = float(cell[:, 0].sum() / cell.sum())
        # One position in ten: the cosine scales its leftover noise back to unit length and pulls on it hardest.
        self.assertGreater(flat_share["cosine"], 0.5)
        self.assertLess(flat_share["dot"], 1e-4)

    def test_a_cell_that_differs_more_from_the_batch_pulls_harder_only_under_dot(self):
        torch.manual_seed(4)
        hz = torch.randn(2, 16, 6, 10, dtype=torch.float64)
        hz[1, 0, :, 0] = -hz[0, 0, :, 0]  # subject 0's two views disagree at position 0
        grown = hz.clone()
        grown[:, 0, :, 0] *= 4.0  # ... and now differ four times as much from the rest of the batch
        share = {}
        for similarity in ("cosine", "dot"):
            for name, x in (("typical", hz), ("grown", grown)):
                cell = gradient_per_cell(lambda leaf: infonce(leaf, similarity), x)
                share[similarity, name] = float(cell[0, 0] / cell.sum())
        # The cosine's pull on a vector falls as 1/size; the dot product's grows with it.
        self.assertLess(share["cosine", "grown"], share["cosine", "typical"] / 8)
        self.assertGreater(share["dot", "grown"], 2 * share["dot", "typical"])


class FlagTests(unittest.TestCase):
    def test_flag_rules(self):
        self.assertFalse(encoder_args.parse_args(SYNTHETIC).standardize_patch_code)
        self.assertTrue(encoder_args.parse_args([*SYNTHETIC, "--standardize-patch-code"]).standardize_patch_code)
        self.assertEqual(encoder_args.parse_args(SYNTHETIC).patch_similarity, "cosine")
        dot = encoder_args.parse_args([*SYNTHETIC, "--standardize-patch-code", "--patch-similarity", "dot"])
        self.assertEqual(dot.patch_similarity, "dot")
        for refused in (
            ["--res", "32", "--num-train-samples", "640", "--batch-size", "32", "--standardize-patch-code"],
            [*SYNTHETIC, "--patch-loss-type", "barlow_twins", "--standardize-patch-code"],
            [*SYNTHETIC, "--mask-mode", "learned", "--standardize-patch-code"],
            [*SYNTHETIC, "--patch-similarity", "dot"],
        ):
            with self.subTest(argv=refused), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    encoder_args.parse_args(refused)
        options = {"patch_loss_weight": 1.0, "standardize_patch_code": True, "patch_similarity": "dot"}
        argv = ["--res", "32", "--num-train-samples", "640", *encoder_args.cli_arguments(options)]
        parsed = encoder_args.parse_args(argv)
        self.assertTrue(parsed.standardize_patch_code)
        self.assertEqual(parsed.patch_similarity, "dot")

    def test_a_run_trains_with_either_similarity(self):
        for similarity in ("cosine", "dot"):
            with self.subTest(similarity=similarity), tempfile.TemporaryDirectory() as tmp:
                argv = ["trainer", "--out-dir", tmp, "--model-id", "std", *TINY, "--standardize-patch-code"]
                argv += ["--patch-similarity", similarity]
                with patch.object(sys, "argv", argv), patch.object(trainer, "evaluate", return_value=({}, {})):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        trainer.main()
                run = Path(tmp) / "std"
                settings = json.loads((run / "settings.json").read_text())
                self.assertTrue(settings["standardize_patch_code"])
                self.assertEqual(settings["patch_similarity"], similarity)
                self.assertIn(f"({similarity} similarity, code centred at each position", output.getvalue())
                self.assertEqual(json.loads((run / "training_progress.json").read_text())["step"], 2)
                state = torch.load(run / "model.pt", weights_only=True)
                self.assertTrue(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()))


if __name__ == "__main__":
    unittest.main()
