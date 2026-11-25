"""warm.py's --real readouts (gauge_readouts): each must change one thing about the evaluated code and leave the run
untouched. The recomputed gauge must standardize by the bank's own feature statistics, the batch-statistics readout
must equal TailCode's batch gauge applied by hand, and the jitter must repeat exactly from one evaluation to the next.
A fake backbone and bank stand in for the network and ADNI. Run by path with PYTHONPATH as in HANDOFF section 3."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "out/regional_pilot_20261028/shim", ROOT / "out/burden_feasibility_20261102"):
    sys.path.insert(0, str(extra))
import warm  # noqa: E402

CHANNELS, MAP, SIDE = 4, 4, 8  # backbone channels, map side, image side


class FakeBackbone(nn.Module):
    """A fixed conv to a MAP^3 map: (2B, 1, SIDE^3) view-major images -> (2B, CHANNELS, MAP^3)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv3d(1, CHANNELS, 3, padding=1)

    def _encode(self, x, views, _):
        assert views == 2
        return F.adaptive_avg_pool3d(torch.tanh(self.conv(x)), MAP)


class FakeBank:
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.t1 = rng.standard_normal((n, 1, SIDE, SIDE, SIDE)).astype(np.float32)
        self.flair = (-0.7 * self.t1 + 0.3 * rng.standard_normal(self.t1.shape)).astype(np.float32)

    def __len__(self):
        return len(self.t1)

    def images(self, indices, device):
        indices = np.asarray(indices)
        return torch.from_numpy(np.concatenate([self.t1[indices], self.flair[indices]])).to(device)


def trained_pool(model, bank, gauge):
    """A TailCode whose running statistics training has moved away from the start."""
    keep = np.zeros(MAP**3, dtype=bool)
    keep[::2] = True
    torch.manual_seed(1)
    pool = warm.TailCode(keep, channels=CHANNELS, latent=5, gauge=gauge, units=3)
    pool.train()
    with torch.no_grad():
        for start in range(0, len(bank), 6):
            pool(model._encode(bank.images(np.arange(start, min(start + 6, len(bank))), "cpu") * 1.5, 2, None))
    return pool.eval()


def state(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class RecalibrationTests(unittest.TestCase):
    def check(self, gauge):
        model, bank = FakeBackbone(), FakeBank(20, 0)
        pool = trained_pool(model, bank, gauge)
        before = state(pool)
        fresh = warm.recalibrated_pool(model, pool, bank, "cpu", batch=7)
        self.assertTrue(same(before, state(pool)), "the run's own pool changed")
        self.assertFalse(pool.training)
        with torch.no_grad():
            maps = model._encode(bank.images(np.arange(len(bank)), "cpu"), 2, None)
            for view, part in enumerate(maps.chunk(2)):
                pooled = pool.ungauged(part, view, update=False).double()
                g = fresh.gauges[view]
                mean, var = (g.running_mean, g.running_var) if gauge == "batch" else (g.mean, g.var)
                torch.testing.assert_close(mean.double(), pooled.mean(0), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(var.double(), pooled.var(0, unbiased=False), rtol=1e-4, atol=1e-6)
                gauged = fresh.pooled_view(part, view)  # eval mode: the recomputed statistics
                keep = pooled.var(0) > 1e-3  # near-constant features gauge to about 0, not 1 (the gauge's eps)
                torch.testing.assert_close(
                    gauged.mean(0).double(), torch.zeros_like(gauged.mean(0)).double(), atol=1e-4, rtol=0
                )
                torch.testing.assert_close(
                    gauged.var(0, unbiased=False)[keep].double(),
                    torch.ones(int(keep.sum())).double(),
                    atol=1e-2,
                    rtol=0,
                )

    def test_batchnorm_gauge(self):
        self.check("batch")

    def test_running_gauge(self):
        self.check("running")


class BatchGaugeTests(unittest.TestCase):
    def test_matches_batchnorm_by_hand_and_leaves_state(self):
        model, bank = FakeBackbone(), FakeBank(18, 2)
        pool = trained_pool(model, bank, "batch")
        pool.drop = "tails"  # a group-dropout run evaluated mid-step: the readout must not drop anything
        before = state(pool)
        rng = np.random.default_rng(3)
        sets = [rng.choice(len(bank), 6, replace=False) for _ in range(4)]
        got = warm.batch_gauge_top1(model, pool, bank, sets, "cpu")
        self.assertTrue(same(before, state(pool)))
        self.assertFalse(pool.training)
        self.assertEqual(pool.drop, "tails")
        expected = []
        with torch.no_grad():
            for rows in sets:
                maps = model._encode(bank.images(rows, "cpu"), 2, None)
                codes = [
                    pool.head(F.batch_norm(pool.ungauged(part, view, update=False), None, None, training=True))[:, :3]
                    for view, part in enumerate(maps.chunk(2))
                ]
                expected.append(warm.cross_view_top1(*(c.numpy() for c in codes))["mean"])
        self.assertAlmostEqual(got, float(np.mean(expected)), places=12)


class JitterTests(unittest.TestCase):
    def test_same_draws_every_evaluation(self):
        x = FakeBank(4, 4).images(np.arange(4), "cpu")
        first, second = warm.jittered(2, 3, False), warm.jittered(2, 3, False)
        a, b = first(x), second(x)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, x))
        self.assertFalse(torch.equal(first(x), a), "the stream advances within one evaluation")


class ReadoutKeyTests(unittest.TestCase):
    def readouts(self, gauge, shift, degrees):
        model, held, train = FakeBackbone(), FakeBank(12, 5), FakeBank(20, 6)
        pool = trained_pool(model, train, gauge)
        before = state(pool)
        rng = np.random.default_rng(0)
        sets = [rng.choice(len(held), 6, replace=False) for _ in range(3)]
        out = warm.gauge_readouts(model, pool, held, "cpu", sets, train, shift, degrees, False)
        self.assertTrue(same(before, state(pool)))
        return out

    def test_batchnorm_gauge_with_jitter(self):
        out = self.readouts("batch", 2, 3)
        self.assertEqual(list(out), ["batch_clean", "batch_jittered", "running_jittered", "recalibrated"])
        self.assertEqual(set(out["recalibrated"]), {"top1", "top1_32", "modality_probe"})
        self.assertEqual(set(out["recalibrated"]["modality_probe"]), set(warm.PROBES))

    def test_running_gauge_has_no_batch_readouts(self):
        self.assertEqual(list(self.readouts("running", 2, 3)), ["running_jittered", "recalibrated"])

    def test_no_jitter_no_jittered_readouts(self):
        self.assertEqual(list(self.readouts("batch", 0, 0)), ["batch_clean", "recalibrated"])


if __name__ == "__main__":
    unittest.main()
