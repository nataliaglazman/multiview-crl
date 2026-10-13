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
        old = SimpleNamespace(**{k: v for k, v in vars(trainer.parse_args(TINY)).items() if k != "standardize_code"})
        torch.testing.assert_close(
            trainer.contrastive_loss(pooled, NO_HEAD, old, *CRITERIA),
            trainer.contrastive_loss(pooled, NO_HEAD, trainer.parse_args(TINY), *CRITERIA),
        )


if __name__ == "__main__":
    unittest.main()
