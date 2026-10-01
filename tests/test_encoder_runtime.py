"""Strict legacy settings and opt-in CUDA nondeterministic backward fallback."""

import os
import unittest
from unittest.mock import patch

import torch

from utils.encoder_runtime import configure_encoder_runtime


class EncoderRuntimeTests(unittest.TestCase):
    def setUp(self):
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", benchmark)
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", cudnn_deterministic)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def test_legacy_strict_settings_reset_warn_only(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        configure_encoder_runtime({"deterministic": True})
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_warning_mode_keeps_deterministic_algorithms_enabled(self):
        torch.backends.cudnn.benchmark = True
        configure_encoder_runtime({"deterministic": True, "deterministic_warn_only": True})
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_warn_only_without_determinism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires deterministic"):
            configure_encoder_runtime({"deterministic_warn_only": True})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for MaxPool3d backward regression")
    def test_cuda_maxpool_backward_can_run_in_warning_mode(self):
        configure_encoder_runtime({"deterministic": True, "deterministic_warn_only": True})
        x = torch.arange(125, dtype=torch.float32, device="cuda").reshape(1, 1, 5, 5, 5).requires_grad_()
        torch.nn.functional.max_pool3d(x, kernel_size=3, stride=2, padding=1).sum().backward()
        torch.cuda.synchronize()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
