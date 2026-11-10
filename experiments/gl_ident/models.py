"""Encoders: local field encoder r_k, window encoder g_k, whole-image pooled baseline g_pool.

Each model holds one encoder per view (separate weights). Inputs are centred on the generator's fixed
intensity offset (3) before the first layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.gl_ident.config import RunConfig
from experiments.gl_ident.generator import INTENSITY_OFFSET, conv_cls, pixel_unshuffle


class LocalEncoder(nn.Module):
    """pixel-unshuffle(m) -> 1x1 conv -> n_g x [3x3 conv + LeakyReLU] -> 1x1 conv; one code per coarse site."""

    def __init__(self, dim, m, hidden, n_g, out_dim, padding="circular"):
        super().__init__()
        conv = conv_cls(dim)
        layers = [conv(m**dim, hidden, 1)]
        for _ in range(n_g):
            layers += [conv(hidden, hidden, 3, padding=1, padding_mode=padding), nn.LeakyReLU(0.2)]
        layers.append(conv(hidden, out_dim, 1))
        self.net = nn.Sequential(*layers)
        self.m = m

    def forward(self, x):
        return self.net(pixel_unshuffle(x - INTENSITY_OFFSET, self.m))


class PooledEncoder(nn.Module):
    """pixel-unshuffle(m) -> 1x1 conv -> n_conv x [3x3 conv + LeakyReLU] -> global average pool -> MLP.

    The window encoder g_k (zero padding: a window is a crop) and the whole-image baseline g_pool
    (circular padding) are both this module.
    """

    def __init__(self, dim, m, hidden, n_conv, mlp_hidden, out_dim, padding):
        super().__init__()
        conv = conv_cls(dim)
        layers = [conv(m**dim, hidden, 1)]
        for _ in range(n_conv):
            layers += [conv(hidden, hidden, 3, padding=1, padding_mode=padding), nn.LeakyReLU(0.2)]
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden, out_dim),
        )
        self.m = m

    def forward(self, x):
        h = self.conv(pixel_unshuffle(x - INTENSITY_OFFSET, self.m))
        return self.mlp(h.flatten(2).mean(-1))


def code_dim(cfg: RunConfig) -> int:
    """Dimension the theory asks for (d_hat for local, d_gamma for global/pooled)."""
    g = cfg.generator
    if cfg.model.kind == "local":
        return cfg.model.d_hat or (g.d_c + g.d_gamma)
    return g.d_gamma


def out_dim(cfg: RunConfig) -> int:
    extra = int(cfg.model.sphere_extra_dim and cfg.loss.similarity == "cosine")
    return code_dim(cfg) + extra


class TwoView(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        g, mc = cfg.generator, cfg.model
        d = out_dim(cfg)

        def make():
            if mc.kind == "local":
                return LocalEncoder(g.dim, g.stride, mc.hidden, mc.n_g, d, mc.padding)
            padding = "zeros" if mc.kind == "global" else mc.padding
            return PooledEncoder(g.dim, g.stride, mc.hidden, mc.n_conv, mc.mlp_hidden, d, padding)

        self.enc = nn.ModuleList([make(), make()])
        self.kind = mc.kind

    def forward(self, x, k):
        return self.enc[k](x)


def normalise(z, similarity, dim):
    return F.normalize(z, dim=dim) if similarity == "cosine" else z


@torch.no_grad()
def local_receptive_field(encoder: LocalEncoder, n_sites: int, dim: int) -> int:
    """Largest circular Chebyshev distance (in sites) from an output site to an input block with nonzero gradient."""
    m = encoder.m
    device = next(encoder.parameters()).device
    x = (torch.randn(1, 1, *([n_sites * m] * dim), device=device) + INTENSITY_OFFSET).requires_grad_(True)
    centre = (n_sites // 2,) * dim
    with torch.enable_grad():
        out = encoder(x)
        out[(0, slice(None)) + centre].sum().backward()
    g = x.grad[0, 0].abs()
    blocks = g.reshape(*sum(([n_sites, m] for _ in range(dim)), [])).amax(dim=tuple(range(1, 2 * dim, 2)))
    far = 0
    for idx in (blocks > 0).nonzero().tolist():
        far = max(far, max(min(abs(i - c), n_sites - abs(i - c)) for i, c in zip(idx, centre)))
    return far
