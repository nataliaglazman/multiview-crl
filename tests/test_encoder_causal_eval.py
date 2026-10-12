"""--causal-eval: the in-training label-free PC eval of the encoder-only trainer."""

import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import training.encoder_args as encoder_args
from eval.causal import latent_causal_discovery, run_causal_recovery


def parse(*argv):
    return encoder_args.parse_args(list(argv))


def refuses(*argv):
    with contextlib.redirect_stderr(io.StringIO()), unittest.TestCase().assertRaises(SystemExit):
        parse(*argv)
    return True


class CausalEvalArgsTests(unittest.TestCase):
    def test_choices_are_the_eval_modules_own(self):
        self.assertEqual(encoder_args.CAUSAL_REDUCTIONS, latent_causal_discovery.REDUCTIONS)
        self.assertEqual(encoder_args.CAUSAL_INDEP_TESTS, run_causal_recovery.INDEP_TESTS)
        self.assertEqual(tuple(encoder_args.CAUSAL_ALPHAS), run_causal_recovery.DEFAULT_ALPHAS)

    def test_off_by_default_and_accepted_on_a_causal_run(self):
        self.assertFalse(parse().causal_eval)
        args = parse("--synthetic-causal", "--causal-eval", "--causal-eval-grid", "2", "2", "2")
        self.assertEqual(args.causal_eval_grid, [2, 2, 2])
        parse("--synthetic-causal", "--causal-eval", "--causal-eval-grid", "2", "2", "2", "--causal-eval-reduce", "pc1")

    def test_refusals(self):
        self.assertTrue(refuses("--causal-eval"))
        self.assertTrue(refuses("--causal-eval-reduce", "pc1"))
        self.assertTrue(refuses("--synthetic-causal", "--causal-eval", "--causal-eval-reduce", "std"))
        self.assertTrue(refuses("--synthetic-causal", "--causal-eval", "--content-channels", "4"))
        self.assertTrue(refuses("--synthetic-causal", "--causal-eval", "--causal-eval-grid", "9", "9", "9"))
        self.assertTrue(refuses("--synthetic-causal", "--causal-eval", "--causal-eval-alphas", "0"))

    def test_refused_on_real_data(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertTrue(
                refuses(
                    "--dataset-name", "ADNI_stripped", "--dataroot", root, "--labels-path", root,
                    "--spatial-size", "32", "32", "32", "--val-frac", "0.2", "--causal-eval",
                )
            )  # fmt: skip


class CausalEvalScoringTests(unittest.TestCase):
    def test_channels_that_are_the_factors_recover_the_graph_above_the_null(self):
        # A chain SCM rendered straight into the channels, behind a permutation and sign flips:
        # the in-loop scorer must undo the naming and beat the random-naming null.
        rng = np.random.default_rng(0)
        n, d = 2000, 5
        adj = np.zeros((d, d), dtype=bool)
        z = np.zeros((n, d))
        for j in range(d):
            z[:, j] = rng.normal(size=n) + (0.9 * z[:, j - 1] if j else 0.0)
            if j:
                adj[j - 1, j] = True
        perm = rng.permutation(d + 2)
        channels = np.concatenate([z, rng.normal(size=(n, 2))], axis=1)[:, perm] * rng.choice([-1, 1], d + 2)
        cli = SimpleNamespace(reduce="gap", alphas=[0.01], indep_test="fisherz", max_cond_set=None, null_draws=50)
        out = latent_causal_discovery.evaluate_reduced(
            latent_causal_discovery.reduce_channels(channels[:, :, None], "gap"), z, adj, cli
        )
        self.assertEqual(out["best"]["f1"], 1.0)
        self.assertGreater(out["best"]["f1_over_null"], 0.3)


if __name__ == "__main__":
    unittest.main()
