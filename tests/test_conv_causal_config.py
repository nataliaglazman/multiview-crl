"""SCM flags must select the same generative graph in training and evaluation."""

import argparse
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from data.datasets import SyntheticBrainDataset
from eval.score_checkpoint import make_val_dataset
from eval.synthetic_dataset import build_content_scm


def training_functions():
    # Execute the real parser and dataset factory without importing unrelated
    # training/perceptual-loss dependencies.
    source = Path(__file__).resolve().parents[1] / "training/main_conv_synthetic.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("parse_args", "make_dataset")
    ]
    namespace = {"argparse": argparse, "SyntheticBrainDataset": SyntheticBrainDataset}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["parse_args"], namespace["make_dataset"]


class ConvCausalConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.parse, cls.factory = map(staticmethod, training_functions())

    def args(self, *extra):
        return self.parse(["--no-cache", "--res", "16", "--synthetic-normalize", "fixed_reference", *extra])

    def test_random_graph_and_weights_match_across_splits_and_saved_evaluation(self):
        for architecture in ("conv", "resnet18"):
            for probability in (0.0, 0.5, 1.0):
                args = self.args(
                    "--encoder-architecture",
                    architecture,
                    "--synthetic-causal",
                    "--synthetic-causal-graph",
                    "random",
                    "--synthetic-causal-edge-prob",
                    str(probability),
                )
                expected = build_content_scm(9, graph_type="random", edge_prob=probability, seed=args.seed)
                datasets = [self.factory(args, mode, 2) for mode in ("train", "val", "test")]
                cfg = json.loads(json.dumps(vars(args)))  # settings.json round-trip
                scored = make_val_dataset(cfg, 2)
                for ds in [*datasets, scored]:
                    self.assertTrue(ds._inner.causal)
                    np.testing.assert_array_equal(ds._inner.scm["adj"], expected["adj"])
                    for key, weight in expected["weights"].items():
                        torch.testing.assert_close(ds._inner.scm["weights"][key], weight, rtol=0, atol=0)
                if probability == 0:
                    self.assertEqual(int(expected["adj"].sum()), 0)
                elif probability == 1:
                    np.testing.assert_array_equal(expected["adj"], np.triu(np.ones((9, 9), dtype=bool), 1))
                else:
                    self.assertGreater(int(expected["adj"].sum()), 0)
                    self.assertLess(int(expected["adj"].sum()), 36)
                # Matching adjacency alone is insufficient: the same validation
                # index must also produce identical latent values and rendered inputs.
                trained_val, loaded_val = datasets[1][0], scored[0]
                torch.testing.assert_close(
                    trained_val["gt_latents"]["z_content"], loaded_val["gt_latents"]["z_content"], rtol=0, atol=0
                )
                for a, b in zip(trained_val["image"], loaded_val["image"]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_chain_and_full_ignore_edge_probability(self):
        for graph in ("chain", "full"):
            expected = np.eye(9, k=1, dtype=bool) if graph == "chain" else np.triu(np.ones((9, 9), dtype=bool), 1)
            for probability in (0, 1):
                args = self.args(
                    "--synthetic-causal",
                    "--synthetic-causal-graph",
                    graph,
                    "--synthetic-causal-edge-prob",
                    str(probability),
                )
                ds = self.factory(args, "train", 2)
                np.testing.assert_array_equal(ds._inner.scm["adj"], expected)

    def test_legacy_defaults_and_causal_off_remain_compatible(self):
        args = self.args("--synthetic-causal")
        self.assertEqual(args.synthetic_causal_graph, "chain")
        self.assertEqual(args.synthetic_causal_edge_prob, 0.5)
        modern = self.factory(args, "val", 2)
        del args.synthetic_causal_graph, args.synthetic_causal_edge_prob
        legacy = self.factory(args, "val", 2)
        scored = make_val_dataset(vars(args), 2)
        for ds in (legacy, scored):
            np.testing.assert_array_equal(modern._inner.scm["adj"], ds._inner.scm["adj"])
            torch.testing.assert_close(modern._inner[0][2]["z_content"], ds._inner[0][2]["z_content"], rtol=0, atol=0)
        off = self.args("--synthetic-causal-graph", "random", "--synthetic-causal-edge-prob", "1")
        self.assertFalse(self.factory(off, "train", 2)._inner.causal)
        self.assertFalse(make_val_dataset(vars(off), 2)._inner.causal)

    def test_invalid_graph_or_probability_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for probability in ("-0.1", "1.1", "nan", "inf"):
                with self.assertRaises(SystemExit):
                    self.args("--synthetic-causal-edge-prob", probability)
            with self.assertRaises(SystemExit):
                self.args("--synthetic-causal-graph", "unknown")


if __name__ == "__main__":
    unittest.main()
