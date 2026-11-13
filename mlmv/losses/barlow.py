"""Barlow Twins with context centring and running standardisation.

For one level with paired expander outputs ``za, zb: [N, P]`` and a context id per row:

1. centre each view within its context (subtract the context's mean of that view); skip if context is None;
2. standardise each dimension with running statistics: an EMA (``momentum``, bias-corrected) of each step's
   mean and second moment of the centred embeddings, per view; the statistics are detached;
3. ``C = za~^T zb~ / N`` and ``L = sum_i (1 - C_ii)^2 + lambda_off * sum_{i != j} C_ij^2``.

Centring is what keeps coarser content from solving a finer level's loss: anything constant within a context
is removed exactly before C is formed. ``standardise="batch"`` uses the current batch's statistics instead
(with gradient), as in plain Barlow Twins. ``corr_ema > 0`` replaces C by a running cross-correlation whose
newest term carries the gradient; that also scales both loss terms' gradients by ``1 - corr_ema``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def context_centre(z, context):
    """``z [N, P]`` minus the mean of z over the rows sharing each row's ``context [N]`` id."""
    if context is None:
        return z
    _, inverse = torch.unique(context, return_inverse=True)
    n_ctx = int(inverse.max()) + 1
    sums = torch.zeros(n_ctx, z.shape[1], device=z.device, dtype=z.dtype).index_add_(0, inverse, z)
    counts = torch.bincount(inverse, minlength=n_ctx).to(z.dtype).clamp_min(1)
    return z - (sums / counts[:, None])[inverse]


class RunningStandardiser(nn.Module):
    """Per-dimension standardisation with EMA statistics (bias-corrected, detached). ``[N, P] -> [N, P]``."""

    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer("m1", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def stats(self):
        correction = 1 - self.momentum ** max(int(self.steps), 1)
        mean = self.m1 / correction
        var = (self.m2 / correction - mean**2).clamp_min(self.eps)
        return mean, var

    def forward(self, z):
        if self.training:
            with torch.no_grad():
                zf = z.detach().float()
                self.m1.mul_(self.momentum).add_(zf.mean(0), alpha=1 - self.momentum)
                self.m2.mul_(self.momentum).add_((zf**2).mean(0), alpha=1 - self.momentum)
                self.steps += 1
        mean, var = self.stats()
        return (z - mean) / var.sqrt()


def batch_standardise(z, eps: float = 1e-5):
    mean = z.mean(0, keepdim=True)
    var = z.var(0, unbiased=False, keepdim=True)
    return (z - mean) / (var + eps).sqrt()


class BarlowLoss(nn.Module):
    """``forward(za, zb, context) -> (loss, {"mean_cii": float, "on": float, "off": float})``."""

    def __init__(
        self,
        dim: int,
        lambda_off: float = 0.005,
        centring: bool = True,
        standardise: str = "running",
        momentum: float = 0.99,
        corr_ema: float = 0.0,
    ):
        super().__init__()
        if standardise not in ("running", "batch"):
            raise ValueError(f"standardise must be running or batch, got {standardise!r}")
        self.lambda_off, self.centring, self.standardise, self.corr_ema = lambda_off, centring, standardise, corr_ema
        self.std_a = RunningStandardiser(dim, momentum)
        self.std_b = RunningStandardiser(dim, momentum)
        self.register_buffer("c_ema", torch.zeros(dim, dim))
        self.register_buffer("c_steps", torch.zeros((), dtype=torch.long))

    def correlation(self, za, zb, context=None):
        """Cross-correlation C ``[P, P]`` after centring and standardisation (fp32, autocast off)."""
        device_type = za.device.type if za.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            za, zb = za.float(), zb.float()
            if self.centring:
                za, zb = context_centre(za, context), context_centre(zb, context)
            if self.standardise == "running":
                za, zb = self.std_a(za), self.std_b(zb)
            else:
                za, zb = batch_standardise(za), batch_standardise(zb)
            c = za.T @ zb / za.shape[0]
            if self.corr_ema > 0:
                m, t = self.corr_ema, int(self.c_steps) + 1
                corrected = (m * self.c_ema + (1 - m) * c) / (1 - m**t)  # bias-corrected; newest term has grad
                if self.training:
                    with torch.no_grad():
                        self.c_ema.mul_(m).add_(c.detach(), alpha=1 - m)
                        self.c_steps += 1
                c = corrected
        return c

    def forward(self, za, zb, context=None):
        c = self.correlation(za, zb, context)
        diag = torch.diagonal(c)
        on = ((1 - diag) ** 2).sum()
        off = (c**2).sum() - (diag**2).sum()
        loss = on + self.lambda_off * off
        return loss, {"mean_cii": float(diag.mean().detach()), "on": float(on.detach()), "off": float(off.detach())}
