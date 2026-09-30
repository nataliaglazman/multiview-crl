"""Apply saved numerical runtime choices consistently in training and evaluation."""

import os

import torch


def configure_encoder_runtime(settings):
    threads = settings.get("cpu_threads")
    if threads is not None:
        if threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(threads)
    if settings.get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
