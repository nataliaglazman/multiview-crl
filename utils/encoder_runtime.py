"""Apply saved numerical runtime choices consistently in training and evaluation."""

import os

import torch


def select_encoder_device(requested="auto", no_cuda=False):
    """Resolve ``--device``: ``auto`` keeps the legacy CUDA-else-CPU choice; MPS is opt-in.

    An explicitly requested accelerator that is missing raises rather than silently
    falling back to CPU.
    """
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"Unknown encoder device: {requested}")
    if no_cuda:
        if requested not in ("auto", "cpu"):
            raise ValueError("--no-cuda forces CPU and cannot be combined with --device cuda/mps")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("This PyTorch build has no MPS support; use an Apple-silicon PyTorch environment")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable (needs macOS 12.3+ and GPU access)")
    return requested


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
