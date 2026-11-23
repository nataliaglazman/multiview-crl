"""warm.py --harmonic-weight: the non-linear redundancy penalty charges bent copies of a unit, and nothing linear."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import warm


def bent(b):
    """A harmonic of b: a function of it with no linear correlation, at unit variance."""
    return (b.square() - 1) / 2**0.5


def codes(kind, n, generator):
    """(2, n, 4) codes, the same structure in both views."""
    b, c, d, e = torch.randn(4, n, generator=generator)
    units = {
        "independent": [b, c, d, e],
        "linear": [b, c, d, 0.8 * b + 0.6 * e],
        "bent": [b, c, d, bent(b)],
        "split": [b + bent(b), b - bent(b), c, d],
    }[kind]
    one = torch.stack(units, 1)
    return torch.stack([one, one + 0.05 * torch.randn(one.shape, generator=generator)])


def trained_penalty(kind, steps=1500, batch=64):
    generator = torch.Generator().manual_seed(0)
    critic = warm.HarmonicCritic(4, 64)
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-3)
    for _ in range(steps):
        penalty, fit = warm.harmonic_term(codes(kind, batch, generator), critic, optimizer)
    with torch.no_grad():
        held = []
        for _ in range(20):
            pair = codes(kind, batch, generator)
            whitened = critic.whiten(pair)
            for view in whitened:
                target, linear, full = critic(view, frozen=True)
                held.append(
                    torch.relu((target - linear).square().mean(0) - (target - full).square().mean(0)).mean().item()
                )
    return sum(held) / len(held), fit


class HarmonicPenaltyTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_critic_starts_as_its_linear_map(self):
        critic = warm.HarmonicCritic(4, 8)
        z = torch.randn(10, 4)
        _, linear, full = critic(z)
        self.assertTrue(torch.equal(linear, full))
        _, frozen_linear, frozen_full = critic(z, frozen=True)
        self.assertTrue(torch.equal(frozen_linear, frozen_full))

    def test_bases_are_orthonormal_and_start_with_the_units(self):
        critic = warm.HarmonicCritic(5, 8, bases=4)
        self.assertTrue(torch.equal(critic.rotations[0], torch.eye(5)))
        for rotation in critic.rotations:
            self.assertTrue(torch.allclose(rotation.T @ rotation, torch.eye(5), atol=1e-5))

    def test_no_coordinate_predicts_itself(self):
        critic = warm.HarmonicCritic(3, 8, bases=3)
        with torch.no_grad():
            critic.linear.fill_(1.0)
            critic.outer.fill_(1.0)
        z = torch.randn(5, 3)
        for m, rotation in enumerate(critic.rotations):
            for k in range(3):
                moved = z + 10.0 * rotation[:, k]  # changes basis m's coordinate k and none of its others
                before, after = critic(z)[2][:, m, k], critic(moved)[2][:, m, k]
                self.assertTrue(torch.allclose(before, after, atol=1e-4), (m, k))

    def test_whitening_leaves_no_linear_correlation(self):
        critic = warm.HarmonicCritic(4, 8, momentum=0.05)
        generator = torch.Generator().manual_seed(1)
        for _ in range(400):
            critic.track(codes("linear", 256, generator))
        whitened = critic.whiten(codes("linear", 4096, generator))
        for view in whitened:
            cov = torch.cov(view.T)
            self.assertLess(float((cov - torch.eye(4)).abs().max()), 0.12)

    def test_penalty_charges_a_bent_copy_and_nothing_linear(self):
        got = {kind: trained_penalty(kind)[0] for kind in ("independent", "linear", "bent", "split")}
        self.assertLess(got["independent"], 0.03, got)
        self.assertLess(got["linear"], 0.03, got)
        self.assertGreater(got["bent"], 0.15, got)
        # Split across two units, the bent copy is linearly predictable from the other unit unless the code is
        # whitened first; whitening is what leaves only the non-linear route.
        self.assertGreater(got["split"], 0.15, got)

    def test_penalty_gradient_reaches_the_code_and_never_the_critic(self):
        generator = torch.Generator().manual_seed(2)
        critic = warm.HarmonicCritic(4, 16)
        optimizer = torch.optim.Adam(critic.parameters(), lr=3e-3)
        for _ in range(200):
            warm.harmonic_term(codes("bent", 64, generator), critic, optimizer)
        pair = codes("bent", 64, generator).requires_grad_(True)
        penalty, _ = warm.harmonic_term(pair, critic, optimizer)
        for parameter in critic.parameters():
            parameter.grad = None
        penalty.backward()
        self.assertGreater(float(pair.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))

    def test_options_reject_what_the_penalty_cannot_use(self):
        base = [
            "warm.py",
            "bank",
            "/nonexistent/stage60_out",
            "cold",
            "2",
            "1",
            "--pool",
            "tail",
            "--patch-weight",
            "0",
        ]
        cases = [
            ["--harmonic-weight", "-1"],
            ["--harmonic-weight", "1", "--harmonic-hidden", "0"],
            ["--harmonic-weight", "1", "--harmonic-lr", "0"],
            ["--harmonic-weight", "1", "--pool", "gap"],
            ["--harmonic-weight", "1", "--gauge", "running"],
            ["--harmonic-weight", "1", "--patch-weight", "1"],
            ["--harmonic-weight", "1", "--contrastive-loss-type", "barlow_twins"],
            ["--harmonic-weight", "1", "--group-dropout", "0.1"],
        ]
        for extra in cases:
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra):
                with self.assertRaisesRegex(ValueError, "--harmonic"):
                    warm.main()
        self.assertFalse(Path("/nonexistent/stage60_out").exists())


if __name__ == "__main__":
    unittest.main()
