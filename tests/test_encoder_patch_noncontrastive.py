"""Negative-free patch losses (--patch-loss-type barlow_twins|vicreg): formulas, flat regions, wiring, training."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eval.protocol.score_checkpoint import build_model
from training import main_conv_synthetic as trainer

SMALL = [
    "--res",
    "16",
    "--hidden-channels",
    "8",
    "--res-channels",
    "4",
    "--nb-res-layers",
    "1",
    "--latent-dim",
    "12",
    "--conv-readout",
    "mlp",
    "--encoder-head-hidden",
    "8",
    "--train-patch-grid",
    "2",
    "2",
    "2",
    "--tau",
    "0.1",
    "--best-metric",
    "none",
]


def refused(argv):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            trainer.parse_args(argv)
        except SystemExit:
            return True
    return False


def position_centred(z):
    """(B, C, P) -> each (channel, position) minus its mean over subjects."""
    return z - z.mean(0, keepdim=True)


class NoncontrastivePatchLossTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.criteria = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())

    def config(self, kind, *extra):
        argv = SMALL + ["--patch-loss-weight", "0.5", "--batch-size", "4", "--patch-loss-type", kind, *extra]
        return vars(trainer.parse_args(argv))

    def test_parser_accepts_each_global_and_refuses_what_would_do_nothing(self):
        self.assertEqual(trainer.parse_args([]).patch_loss_type, "infonce")
        for argv in (
            ["--patch-loss-type", "barlow_twins"],
            ["--patch-loss-weight", "1", "--vicreg-sim-coeff", "1"],
            ["--patch-loss-weight", "1", "--patch-loss-type", "barlow_twins", "--vicreg-cov-coeff", "2"],
            ["--patch-loss-weight", "1", "--patch-loss-type", "vicreg", "--vicreg-std-coeff", "0"],
            ["--patch-loss-weight", "1", "--patch-loss-type", "vicreg", "--vicreg-sim-coeff", "-1"],
            ["--patch-loss-weight", "1", "--patch-loss-type", "vicreg", "--vicreg-cov-coeff", "inf"],
            ["--patch-loss-weight", "1", "--patch-loss-type", "barlow_twins", "--mask-mode", "learned"],
            ["--patch-loss-weight", "1", "--contrastive-loss-type", "barlow_twins"],
        ):
            with self.subTest(argv=argv):
                self.assertTrue(refused(argv))
        for kind in ("barlow_twins", "vicreg"):
            for global_ in ("infonce", "barlow_twins"):
                with self.subTest(kind=kind, global_=global_):
                    argv = ["--patch-loss-weight", "1", "--patch-loss-type", kind, "--contrastive-loss-type", global_]
                    self.assertFalse(refused(argv))

    def test_barlow_twins_patch_is_the_position_centred_pooled_correlation(self):
        cfg = self.config("barlow_twins", "--bt-lambda", "0.3")
        model = build_model(cfg, "cpu")
        features = torch.randn(8, 12, 5, requires_grad=True)
        loss = trainer.patch_contrastive_loss(features, model, SimpleNamespace(**cfg), *self.criteria)
        a, b = (position_centred(z).permute(0, 2, 1).reshape(-1, 9) for z in features[:, :9].chunk(2))
        a, b = ((z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-6) for z in (a, b))
        c = a.T @ b / a.shape[0]
        off = ~torch.eye(9, dtype=torch.bool)
        expected = (c.diagonal() - 1).square().sum() + 0.3 * c[off].square().sum()
        torch.testing.assert_close(loss, expected)
        self.assertAlmostEqual(loss._patch_terms["on_diag"], float((c.diagonal() - 1).square().sum()), places=4)
        loss.backward()
        self.assertGreater(features.grad[:, :9].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, 9:].abs().sum().item(), 0)
        # Anatomy every subject shares is centred away, so it cannot satisfy the alignment.
        atlas = torch.randn(1, 12, 5).expand(8, -1, -1)
        atlas_loss = trainer.patch_contrastive_loss(atlas, model, SimpleNamespace(**cfg), *self.criteria)
        torch.testing.assert_close(atlas_loss, torch.tensor(9.0))

    def test_vicreg_patch_uses_raw_mse_and_pooled_centred_statistics(self):
        cfg = self.config("vicreg", "--vicreg-sim-coeff", "2", "--vicreg-std-coeff", "3", "--vicreg-cov-coeff", "0.5")
        model = build_model(cfg, "cpu")
        features = torch.randn(8, 12, 5, requires_grad=True)
        args = SimpleNamespace(**cfg)
        loss = trainer.patch_contrastive_loss(features, model, args, *self.criteria)
        x, y = features[:, :9].chunk(2)
        var = cov = 0
        off = ~torch.eye(9, dtype=torch.bool)
        for z in (x, y):
            rows = position_centred(z).permute(0, 2, 1).reshape(-1, 9)
            covariance = rows.T @ rows / (5 * (4 - 1))
            var = var + F.relu(1 - torch.sqrt(covariance.diagonal() + 1e-4)).mean() / 2
            cov = cov + covariance[off].square().sum() / 9
        torch.testing.assert_close(loss, 2 * F.mse_loss(x, y) + 3 * var + 0.5 * cov)
        loss.backward()
        self.assertEqual(features.grad[:, 9:].abs().sum().item(), 0)
        # A per-view constant offset leaves the centred statistics alone but not the invariance.
        shifted = features.detach().clone()
        shifted[4:, :9] += 0.7
        terms = trainer.patch_contrastive_loss(shifted, model, args, *self.criteria)._patch_terms
        before = trainer.patch_contrastive_loss(features.detach(), model, args, *self.criteria)._patch_terms
        self.assertAlmostEqual(terms["var"], before["var"], places=5)
        self.assertAlmostEqual(terms["cov"], before["cov"], places=5)
        self.assertGreater(terms["sim"], before["sim"] + 0.4)

    def test_flat_region_is_not_pushed_to_tell_subjects_apart(self):
        """Rich positions carry a shared subject signal; flat ones only delta * (shared nuisance + view noise).

        The push on each flat component, -dL/dlog(scale), is read relative to the push on the rich
        signal. Per-position InfoNCE is scale-free at every position, so the flat nuisance is pushed
        as hard whatever delta is, and harder than the real signal. The pooled losses push it ~delta^2
        and push the view-specific noise down.
        """
        B, P = 16, 4
        signal = torch.randn(B, 9, P)
        rich_noise = 0.3 * torch.randn(2, B, 9, P)
        nuisance = torch.randn(1, B, 9, P).expand(2, -1, -1, -1)
        view_noise = torch.randn(2, B, 9, P)
        style = torch.zeros(2 * B, 3, 2 * P)

        def pushes(kind, delta):
            cfg = self.config(kind)
            model = build_model(cfg, "cpu")
            r, a, n = (torch.ones((), requires_grad=True) for _ in range(3))
            rich = r * signal + rich_noise
            flat = delta * (a * nuisance + n * view_noise)
            content = torch.cat([rich, flat], dim=-1).reshape(2 * B, 9, 2 * P)
            features = torch.cat([content, style], dim=1)
            loss = trainer.patch_contrastive_loss(features, model, SimpleNamespace(**cfg), *self.criteria)
            g_signal, g_nuisance, g_noise = torch.autograd.grad(loss, (r, a, n))
            return (g_nuisance / g_signal).item(), (g_noise / g_signal).item()

        nce_wide, _ = pushes("infonce", 0.1)
        nce_narrow, _ = pushes("infonce", 0.01)
        self.assertGreater(nce_wide, 1)
        self.assertAlmostEqual(nce_narrow / nce_wide, 1, places=3)
        for kind in ("barlow_twins", "vicreg"):
            with self.subTest(kind=kind):
                wide, noise = pushes(kind, 0.1)
                narrow, _ = pushes(kind, 0.01)
                self.assertGreater(wide, 0)
                self.assertLess(wide, 0.05)
                self.assertLess(narrow, wide / 50)
                self.assertLess(noise, 0)

    def test_objective_adds_weighted_patch_and_reaches_the_patch_head(self):
        for kind, parts in (("barlow_twins", {"on_diag", "off_diag"}), ("vicreg", {"sim", "var", "cov", "std"})):
            for separate in (False, True):
                with self.subTest(kind=kind, separate=separate):
                    cfg = self.config(kind, *(["--separate-spatial-readout"] if separate else []))
                    model = build_model(cfg, "cpu").train()
                    _, total, terms = trainer.training_objective(
                        model, torch.randn(8, 1, 16, 16, 16), SimpleNamespace(**cfg), *self.criteria
                    )
                    torch.testing.assert_close(total, terms["global"] + 0.5 * terms["patch"])
                    self.assertEqual(
                        {k[len("patch_") :] for k in terms if k.startswith("patch_")} - {"weighted"}, parts
                    )
                    terms["patch"].backward()
                    head = model.spatial_readout if separate else model.to_encoding
                    for module in (model.encoder, model.encoder_v1, head):
                        grads = [p.grad for p in module.parameters()]
                        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                    if separate:
                        self.assertTrue(all(p.grad is None for p in model.to_encoding.parameters()))

    def test_real_training_records_the_patch_objective_and_its_parts(self):
        for kind, parts in (("barlow_twins", ("on_diag", "off_diag")), ("vicreg", ("sim", "var", "cov", "std"))):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                argv = SMALL + ["--patch-loss-weight", "0.5", "--patch-loss-type", kind, "--batch-size", "2"]
                argv += ["--out-dir", tmp, "--model-id", "smoke", "--require-new-run", "--device", "cpu"]
                argv += ["--train-steps", "2", "--eval-every", "2", "--log-every", "1", "--num-train-samples", "8"]
                argv += ["--num-val-samples", "20", "--no-cache", "--no-floor-eval", "--synthetic-clean-content"]
                argv += ["--synthetic-normalize", "fixed_reference"]
                with patch.object(sys, "argv", ["trainer", *argv]), patch.object(
                    trainer, "evaluate", return_value=({}, {})
                ), contextlib.redirect_stdout(io.StringIO()) as output:
                    trainer.main()
                directory = Path(tmp) / "smoke"
                settings = json.loads((directory / "settings.json").read_text())
                progress = json.loads((directory / "training_progress.json").read_text())
                self.assertEqual(settings["patch_loss_type"], kind)
                self.assertEqual((progress["status"], progress["patch_loss_type"]), ("complete", kind))
                terms = progress["last_loss_terms"]
                self.assertAlmostEqual(terms["total"], terms["global"] + terms["patch_weighted"], places=5)
                for part in parts:
                    self.assertIn(f"patch_{part}", terms)
                    self.assertIn(f"{part} ", output.getvalue())
                self.assertIn(f"patch {trainer.LOSS_NAMES[kind]}", output.getvalue())


if __name__ == "__main__":
    unittest.main()
