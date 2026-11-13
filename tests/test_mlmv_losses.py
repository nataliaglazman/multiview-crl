"""mlmv losses: context centring, Barlow Twins identity, InfoNCE negatives, pooling."""

import unittest

import torch

from mlmv.losses.barlow import BarlowLoss, RunningStandardiser, context_centre
from mlmv.losses.infonce import Queue, g_negative_mask, info_nce, l_negative_mask, s_negative_mask
from mlmv.losses.levels import pool_segments, pool_sets


def whitened(n, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, p, generator=g, dtype=torch.float64)
    z = z - z.mean(0)
    cov = z.T @ z / n
    evals, evecs = torch.linalg.eigh(cov)
    return (z @ evecs @ torch.diag(evals.rsqrt()) @ evecs.T).float()


class TestBarlow(unittest.TestCase):
    def test_identical_standardised_inputs_give_identity(self):
        z = whitened(4096, 16)
        loss_fn = BarlowLoss(16, lambda_off=0.005, centring=False)
        c = loss_fn.correlation(z, z)
        self.assertTrue(torch.allclose(c, torch.eye(16), atol=1e-3), float((c - torch.eye(16)).abs().max()))
        loss, info = BarlowLoss(16, centring=False)(z, z)
        self.assertLess(float(loss), 1e-4)
        self.assertAlmostEqual(info["mean_cii"], 1.0, places=3)

    def test_context_constant_feature_contributes_nothing(self):
        torch.manual_seed(0)
        n, p = 512, 8
        context = torch.randint(0, 16, (n,))
        za, zb = torch.randn(n, p), torch.randn(n, p)
        za[:, 0] = zb[:, 0] = 0.0
        base = BarlowLoss(p).correlation(za, zb, context)
        shift = torch.randn(16)[context] * 10  # constant within each context, shared by both views
        za2, zb2 = za.clone(), zb.clone()
        za2[:, 0] += shift
        zb2[:, 0] += shift
        c = BarlowLoss(p).correlation(za2, zb2, context)
        self.assertTrue(torch.allclose(c, base, atol=1e-5))
        self.assertLess(float(c[0].abs().max() + c[:, 0].abs().max()), 1e-6)
        # without centring the same feature dominates C
        off = BarlowLoss(p, centring=False).correlation(za2, zb2, context)
        self.assertGreater(float(off[0, 0]), 0.5)

    def test_context_centre(self):
        z = torch.tensor([[1.0], [3.0], [10.0], [20.0]])
        out = context_centre(z, torch.tensor([0, 0, 1, 1]))
        self.assertTrue(torch.allclose(out[:, 0], torch.tensor([-1.0, 1.0, -5.0, 5.0])))
        self.assertTrue(torch.equal(context_centre(z, None), z))

    def test_running_standardiser_first_step_is_batch(self):
        std = RunningStandardiser(4, momentum=0.99)
        z = torch.randn(256, 4) * 3 + 2
        out = std(z)
        self.assertTrue(torch.allclose(out.mean(0), torch.zeros(4), atol=1e-4))
        self.assertTrue(torch.allclose(out.std(0, unbiased=False), torch.ones(4), atol=1e-3))


class TestInfoNCE(unittest.TestCase):
    def test_s_negatives_exclude_positive_and_near_positions(self):
        torch.manual_seed(0)
        n, rho = 300, 2.5
        subject, region = torch.randint(0, 3, (n,)), torch.randint(0, 4, (n,))
        coords = torch.randint(0, 20, (n, 3))
        for metric in ("chebyshev", "euclidean"):
            mask = s_negative_mask(subject, region, coords, rho, metric)
            self.assertFalse(mask.diagonal().any())
            i, j = mask.nonzero().T
            diff = (coords[i] - coords[j]).abs().float()
            dist = diff.amax(1) if metric == "chebyshev" else diff.pow(2).sum(1).sqrt()
            self.assertTrue((dist > 2 * rho).all())
            self.assertTrue((subject[i] == subject[j]).all() and (region[i] == region[j]).all())
            self.assertTrue(mask.any())

    def test_l_and_g_negatives(self):
        subject = torch.tensor([0, 0, 0, 1, 1])
        region = torch.tensor([1, 2, 1, 1, 2])
        lm = l_negative_mask(subject, region)
        self.assertFalse(lm.diagonal().any())
        self.assertTrue(lm[0, 1] and not lm[0, 2] and not lm[0, 4])
        gm = g_negative_mask(subject)
        self.assertTrue(gm[0, 3] and not gm[0, 1] and not gm.diagonal().any())

    def test_queue_excludes_same_subject(self):
        q = Queue(4, 3)
        q.push(torch.randn(6, 3), torch.tensor([0, 1, 2, 3, 4, 5]))
        z, sid = q.contents()
        self.assertEqual(len(z), 4)
        self.assertEqual(sorted(sid.tolist()), [2, 3, 4, 5])
        za = torch.randn(2, 3)
        loss, info = info_nce(
            za,
            za,
            g_negative_mask(torch.tensor([2, 7])),
            0.1,
            queue=z,
            queue_mask=torch.tensor([2, 7])[:, None] != sid[None],
        )
        self.assertTrue(torch.isfinite(loss))
        # anchor of subject 2: 1 batch negative + 3 queue entries; subject 7: 1 + 4
        self.assertEqual(info["n_neg"], 4.5)

    def test_aligned_views_score_better(self):
        torch.manual_seed(0)
        z = torch.randn(64, 16)
        mask = torch.ones(64, 64, dtype=torch.bool)
        aligned, _ = info_nce(z, z, mask, 0.1)
        shuffled, _ = info_nce(z, z[torch.randperm(64)], mask, 0.1)
        self.assertLess(float(aligned), float(shuffled))


class TestPooling(unittest.TestCase):
    def test_pool_segments_and_sets(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 10)
        flat = torch.tensor([0, 1, 2, 5, 9, 4])
        seg = torch.tensor([0, 0, 1, 1, 2, 3])  # 2 segments per subject
        out = pool_segments(x, flat, seg, 2, 2)
        self.assertTrue(torch.allclose(out[0], x[0, :, [0, 1]].mean(1)))
        self.assertTrue(torch.allclose(out[1], x[0, :, [2, 5]].mean(1)))
        self.assertTrue(torch.allclose(out[2], x[1, :, 9]))
        membership = torch.zeros(10, 3)
        membership[[0, 1], 0] = 1
        membership[[2, 3, 4], 1] = 1
        membership[[7], 2] = 1
        member = torch.zeros(2, 1, 2, 3, dtype=torch.bool)
        member[:, 0, 0, [0, 2]] = True
        member[:, 0, 1, 1] = True
        pooled = pool_sets(x, membership, member)
        self.assertTrue(torch.allclose(pooled[1, 0, 0], x[1, :, [0, 1, 7]].mean(1), atol=1e-6))
        self.assertTrue(torch.allclose(pooled[0, 0, 1], x[0, :, [2, 3, 4]].mean(1), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
