"""Apply saved numerical runtime choices consistently in training and evaluation."""

import os

import torch


def configure_encoder_runtime(settings):
    warn_only = settings.get("deterministic_warn_only", False)
    if warn_only and not settings.get("deterministic", False):
        raise ValueError("deterministic_warn_only requires deterministic=True")
    threads = settings.get("cpu_threads")
    if threads is not None:
        if threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(threads)
    if settings.get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # CUDA MaxPool3d backward has no deterministic implementation. Permit
        # explicit opt-in fallback while keeping legacy strict settings strict.
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
