"""--standardize-code in the encoder-only trainer: its flag rules and the global InfoNCE it computes."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import torch

import training.losses as losses
from training import main_conv_synthetic as trainer

TINY = ["--res", "16", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1", "--best-metric", "none"]
CRITERIA = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())
NO_HEAD = SimpleNamespace(project=lambda z: z)


class StandardizeCodeTests(unittest.TestCase):
    def test_flag_is_off_by_default_and_refuses_objectives_it_would_undo(self):
        self.assertFalse(trainer.parse_args(TINY).standardize_code)
        self.assertTrue(trainer.parse_args([*TINY, "--standardize-code"]).standardize_code)
        for extra in (["--contrastive-loss-type", "barlow_twins"], ["--mask-mode", "learned"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args([*TINY, "--standardize-code", *extra])

    def test_loss_is_infonce_on_per_view_batch_z_scores(self):
        torch.manual_seed(0)
        pooled = 1 + 0.01 * torch.randn(16, 12)  # two views of 8 subjects whose codes barely differ
        plain_args, args = trainer.parse_args(TINY), trainer.parse_args([*TINY, "--standardize-code"])
        standardized = trainer.contrastive_loss(pooled, NO_HEAD, args, *CRITERIA)
        hz = torch.stack([pooled[:8], pooled[8:]])[..., :9]
        z = (hz - hz.mean(1, keepdim=True)) / (hz.std(1, keepdim=True) + 1e-6)
        expected = losses.infonce_loss(
            z,
            sim_metric=CRITERIA[0],
            criterion=CRITERIA[1],
            tau=0.1,
            estimated_content_indices=[list(range(9))],
            subsets=[(0, 1)],
            cross_view_negs_only=True,
        ).squeeze()
        torch.testing.assert_close(standardized, expected)
        # Codes this alike leave plain InfoNCE at its chance value; standardized, the loss tells them apart.
        chance = trainer.contrastive_loss(torch.ones(16, 12), NO_HEAD, plain_args, *CRITERIA)
        plain = trainer.contrastive_loss(pooled, NO_HEAD, plain_args, *CRITERIA)
        self.assertLess(abs(float(plain - chance)), 0.01)
        self.assertGreater(abs(float(standardized - chance)), 0.1)

    def test_settings_without_the_flag_keep_the_plain_loss(self):
        pooled = torch.randn(16, 12)
        old = SimpleNamespace(
            **{k: v for k, v in vars(trainer.parse_args(TINY)).items() if k not in ("standardize_code", "center_code")}
        )
        torch.testing.assert_close(
            trainer.contrastive_loss(pooled, NO_HEAD, old, *CRITERIA),
            trainer.contrastive_loss(pooled, NO_HEAD, trainer.parse_args(TINY), *CRITERIA),
        )

    def test_center_only_overrides_a_generated_standardized_recipe(self):
        args = trainer.parse_args([*TINY, "--standardize-code", "--no-standardize-code", "--center-code"])
        self.assertTrue(args.center_code)
        self.assertFalse(args.standardize_code)
        self.assertFalse(trainer.parse_args(TINY).center_code)
        for extra in (["--standardize-code"], ["--contrastive-loss-type", "barlow_twins"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args([*TINY, "--center-code", *extra])

    def test_center_only_matches_loss_and_gradients_and_removes_offsets_without_scaling(self):
        torch.manual_seed(143)
        hz = torch.randn(2, 8, 12, dtype=torch.float64) * torch.linspace(0.01, 2, 12)
        pooled = hz.flatten(0, 1).requires_grad_(True)
        args = trainer.parse_args([*TINY, "--center-code"])
        actual = trainer.contrastive_loss(pooled, NO_HEAD, args, *CRITERIA)
        content = pooled.reshape(2, 8, 12)[..., :9]
        expected = losses.infonce_loss(
            content - content.mean(1, keepdim=True),
            sim_metric=CRITERIA[0],
            criterion=CRITERIA[1],
            tau=args.tau,
            cross_view_negs_only=args.cross_view_negs_only,
        ).squeeze()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, pooled, retain_graph=True)[0], torch.autograd.grad(expected, pooled)[0]
        )
        shifted = (hz + torch.randn(2, 1, 12) * 10).flatten(0, 1)
        torch.testing.assert_close(trainer.contrastive_loss(shifted, NO_HEAD, args, *CRITERIA), actual)
        standardized = trainer.contrastive_loss(
            pooled, NO_HEAD, trainer.parse_args([*TINY, "--standardize-code"]), *CRITERIA
        )
        self.assertGreater(abs(float(actual.detach() - standardized.detach())), 0.01)


if __name__ == "__main__":
    unittest.main()
