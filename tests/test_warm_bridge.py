"""warm.py --entropy-units (the bridge: InfoNCE on the first N code units, alignment alone on the rest) and
bridge_view.py, which writes those N units out as a run of their own."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import bridge_view
import warm

TAU = 0.1
SIM = torch.nn.CosineSimilarity(dim=-1)


def codes(seed, b=32, units=24):
    """(2b, units) view-major codes whose two views share a common part."""
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(b, units, generator=g)
    return torch.cat([shared + 0.5 * torch.randn(b, units, generator=g) for _ in range(2)]).double()


class BridgeLossTests(unittest.TestCase):
    def test_entropy_on_every_unit_is_the_recipe(self):
        code = codes(0)
        bridge = warm.bridge_infonce(code, 24, 24, TAU, SIM)
        recipe = warm.hard_negative_infonce(code, 24, TAU, 0.0, SIM)
        self.assertAlmostEqual(float(bridge), float(recipe), places=10)

    def test_infonce_on_the_block_alignment_on_the_rest(self):
        code, b, k = codes(1), 32, 10
        hz = warm.T.standardize_contrastive_code(torch.stack([code[:b], code[b:]]))
        block = hz[..., :k] / hz[..., :k].norm(dim=-1, keepdim=True)
        rest = hz[..., k:] / hz[..., k:].norm(dim=-1, keepdim=True)
        expected = 0.0
        for i, j in ((0, 1), (1, 0)):
            logits = block[i] @ block[j].T / TAU
            expected = expected + (torch.logsumexp(logits, 1) - logits.diag()).mean()
            expected = expected + (1 - (rest[i] * rest[j]).sum(-1)).mean() / TAU
        self.assertAlmostEqual(float(warm.bridge_infonce(code, 24, k, TAU, SIM)), float(expected), places=8)

    def test_units_past_the_block_feel_only_the_alignment(self):
        code, b, k = codes(2).requires_grad_(True), 32, 10
        (bridge,) = torch.autograd.grad(warm.bridge_infonce(code, 24, k, TAU, SIM), code)
        hz = warm.T.standardize_contrastive_code(torch.stack([code[:b], code[b:]]))
        alignment = -2 * (SIM(hz[0, :, k:], hz[1, :, k:]) / TAU).mean()
        (aligned,) = torch.autograd.grad(alignment, code)
        (recipe,) = torch.autograd.grad(warm.hard_negative_infonce(code, k, TAU, 0.0, SIM), code)
        torch.testing.assert_close(bridge[:, k:], aligned[:, k:])
        torch.testing.assert_close(bridge[:, :k], recipe[:, :k])

    def test_minimising_it_makes_the_block_agree(self):
        """Stage 46's form had a degenerate minimum (about -15 here): the block's two views disagreed and shrank to a
        small share of each code, a few outliers carrying its batch spread. Free codes that minimise this form keep
        the block's views together and the block a full share of the code."""
        b, k = 32, 10
        torch.manual_seed(0)
        code = torch.randn(2 * b, 24, dtype=torch.double, requires_grad=True)
        opt = torch.optim.Adam([code], lr=0.01)
        for _ in range(1500):
            opt.zero_grad()
            loss = warm.bridge_infonce(code, 24, k, TAU, SIM)
            loss.backward()
            opt.step()
        with torch.no_grad():
            hz = warm.T.standardize_contrastive_code(torch.stack([code[:b], code[b:]]))
            agreement = float(SIM(hz[0, :, :k], hz[1, :, :k]).mean())
            share = float((hz[..., :k].norm(dim=-1) / hz.norm(dim=-1)).median())
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertGreater(agreement, 0.9)
        self.assertGreater(share, 0.3)


class ViewTests(unittest.TestCase):
    def test_the_view_exports_the_first_units(self):
        torch.manual_seed(3)
        keep = np.zeros(8**3, dtype=bool)
        keep[100:300] = True
        whole = warm.TailCode(keep, latent=28, units=24)
        maps = torch.randn(16, 64, 8, 8, 8)
        whole.train()
        whole(maps)  # running statistics, as a trained run has them
        whole.eval()
        part = warm.TailCode(keep, latent=14, units=10)
        part.load_state_dict(bridge_view.first_units(whole.state_dict(), 10, 24))
        part.eval()
        with torch.no_grad():
            a, b = whole(maps), part(maps)
        torch.testing.assert_close(b[:, :10], a[:, :10])
        torch.testing.assert_close(b[:, 10:], a[:, 24:])


if __name__ == "__main__":
    unittest.main()
