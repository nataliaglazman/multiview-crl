"""CPU tests that train_step's non-finite loss guards log a warning and skip, rather than raise.

train_step is AST-loaded together with the module's own stdlib imports and top-level
assignments (``logger``, ``NAN_SKIPPED_STEPS``). The other train_step tests inject a
``logger`` into that namespace, which is how guards referencing a name the module never
bound went unnoticed, so nothing the guards read is supplied from outside the module here.
"""

import ast
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


def load_model():
    source = ROOT / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(a.name == "utils.utils" for a in node.names))
    ]
    namespace = {"__name__": "nan_guard_model_test"}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["VQVAE"]


def load_config():
    source = ROOT / "utils/config.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(a.name == "data.datasets" for a in node.names))
    ]
    namespace = {"datasets": types.SimpleNamespace(SyntheticBrainDataset=object, MyCustomDataset=object)}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["parse_args"], namespace["update_args"]


def load_train_step():
    source = ROOT / "training/main_multimodal.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if (isinstance(node, ast.Import) and all(a.name.split(".")[0] in sys.stdlib_module_names for a in node.names))
        or (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "train_step")
    ]
    # Only the third-party names on train_step's default path are supplied; logging, math and
    # the module-level logger come from the source itself.
    namespace = dict(torch=torch, F=F, autocast=torch.autocast, clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


class NonFiniteLossGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Model = load_model()

    def setUp(self):
        torch.manual_seed(11)
        self.module = load_train_step()
        parse, update = load_config()
        self.args = update(
            parse().parse_args(["--dataset-name", "synthetic", "--inject-style-to-decoder", "--batch-size", "4"])
        )
        self.model = self.Model(
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
            style_injection_mode="input",
            norm_type="layer",
        ).eval()
        self.data = {"image": [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8)]}

    def step(self, contrastive_scale=0.0, recon_scale=1.0):
        # Scaling by NaN keeps each loss attached to the graph, as a real divergence would be.
        return self.module["train_step"](
            self.data,
            [self.model],
            [],
            lambda h, *a, **kw: h.sum() * contrastive_scale,
            torch.optim.SGD(self.model.parameters(), lr=0.1),
            list(self.model.parameters()),
            self.args,
            recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean() * recon_scale,
            force_compute_recon=True,
        )

    def test_nonfinite_contrastive_loss_is_zeroed_and_logged(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        with self.assertLogs("multiview_crl", level="WARNING") as logs:
            total, contrastive, recon, vq, *_ = self.step(contrastive_scale=NAN)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("contrastive loss is non-finite — batch SKIPPED", logs.output[0])
        self.assertIn("L0=nan", logs.output[0])
        # Only the contrastive term is dropped; the finite reconstruction still trains.
        self.assertEqual(contrastive, 0.0)
        self.assertAlmostEqual(total, recon + vq, places=5)
        self.assertEqual(self.module["NAN_SKIPPED_STEPS"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, self.model.parameters())))

    def test_nonfinite_total_loss_skips_backward_and_logs(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        with self.assertLogs("multiview_crl", level="WARNING") as logs:
            total, *_ = self.step(recon_scale=NAN)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("NaN/Inf in loss — backward SKIPPED (skip #1)", logs.output[0])
        self.assertEqual(total, 0.0)
        self.assertEqual(self.module["NAN_SKIPPED_STEPS"], 1)
        # Skipped means untouched: no update, and no gradient left over for the next step.
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.model.parameters())))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
