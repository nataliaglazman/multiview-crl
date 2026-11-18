"""warm.py --entropy-units (the bridge: InfoNCE's alignment on all code units, its entropy on the first N) and
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

    def test_alignment_on_all_units_entropy_on_the_block(self):
        code, b, k = codes(1), 32, 10
        hz = warm.T.standardize_contrastive_code(torch.stack([code[:b], code[b:]]))
        expected = 0.0
        for i, j in ((0, 1), (1, 0)):
            unit = hz / hz.norm(dim=-1, keepdim=True)
            block = hz[..., :k] / hz[..., :k].norm(dim=-1, keepdim=True)
            pos = (unit[i] * unit[j]).sum(-1) / TAU
            expected = expected + (torch.logsumexp(block[i] @ block[j].T / TAU, 1) - pos).mean()
        self.assertAlmostEqual(float(warm.bridge_infonce(code, 24, k, TAU, SIM)), float(expected), places=8)

    def test_units_past_the_block_feel_only_the_alignment(self):
        code, b, k = codes(2).requires_grad_(True), 32, 10
        (bridge,) = torch.autograd.grad(warm.bridge_infonce(code, 24, k, TAU, SIM), code)
        hz = warm.T.standardize_contrastive_code(torch.stack([code[:b], code[b:]]))
        alignment = -sum((SIM(hz[i], hz[j]) / TAU).mean() for i, j in ((0, 1), (1, 0)))
        (aligned,) = torch.autograd.grad(alignment, code)
        torch.testing.assert_close(bridge[:, k:], aligned[:, k:])
        self.assertGreater(float((bridge[:, :k] - aligned[:, :k]).abs().max()), 1e-6)  # the block feels the entropy


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
