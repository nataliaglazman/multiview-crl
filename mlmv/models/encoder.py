"""Dense 3D encoder: one stem per modality, a shared trunk, one 1^3 head split into the G, L and S blocks.

Nothing here mixes information across the volume. Normalisation is per position (statistics over the channels
of a group at each voxel), so the theoretical receptive field is set by the convolutions alone; GroupNorm's
whole-volume statistics would make every output depend on every input voxel. Only the first
``trunk_spatial_convs`` trunk convs are 3^3; the rest are 1^3, which adds depth without reach.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LEVELS = ("G", "L", "S")
MODALITIES = ("t1", "flair")


class ChannelGroupNorm(nn.Module):
    """Per-position group norm: each voxel's channels are normalised within ``groups`` groups.

    ``[B, C, X, Y, Z] -> [B, C, X, Y, Z]``; statistics never cross voxels or samples.
    """

    MIN_PER_GROUP = 4

    def __init__(self, groups: int, channels: int, eps: float = 1e-5):
        super().__init__()
        if channels % groups:
            raise ValueError(f"{channels} channels do not split into {groups} groups")
        if channels // groups < self.MIN_PER_GROUP:
            # A per-position group of one channel normalises to exactly 0 (only the bias survives); two leave
            # +-1. Too few channels per group erase the signal rather than normalise it.
            raise ValueError(f"{channels // groups} channels per group (< {self.MIN_PER_GROUP}); use fewer groups")
        self.groups, self.eps = groups, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        b, c, *spatial = x.shape
        xg = x.float().reshape(b, self.groups, c // self.groups, *spatial)
        mean = xg.mean(2, keepdim=True)
        var = xg.var(2, unbiased=False, keepdim=True)
        xg = (xg - mean) * torch.rsqrt(var + self.eps)
        shape = (1, c) + (1,) * len(spatial)
        return (xg.reshape(b, c, *spatial) * self.weight.view(shape) + self.bias.view(shape)).to(x.dtype)


def make_norm(kind: str, groups: int, channels: int) -> nn.Module:
    if kind == "channel":
        return ChannelGroupNorm(groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"norm must be 'channel' or 'none' (no whole-volume norms), got {kind!r}")


def conv(cin, cout, kernel, stride=1):
    return nn.Conv3d(cin, cout, kernel, stride=stride, padding=(kernel - 1) // 2 if stride == 1 else 1)


class ResBlock(nn.Module):
    """Pre-activation residual block: ``x + conv_b(act(norm(conv_a(act(norm(x))))))``."""

    def __init__(self, channels, kernels, norm, groups):
        super().__init__()
        self.kernels = tuple(kernels)
        self.n1, self.c1 = make_norm(norm, groups, channels), conv(channels, channels, kernels[0])
        self.n2, self.c2 = make_norm(norm, groups, channels), conv(channels, channels, kernels[1])

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


def footprint(layers):
    """(stride, low, high): output cell j reads input voxels ``stride*j + low .. stride*j + high`` per axis.

    ``layers`` is a list of (kernel, stride, padding) along the deepest path, in forward order.
    """
    stride, low, high = 1, 0, 0
    for kernel, step, padding in layers:
        low -= stride * padding
        high += stride * (kernel - 1 - padding)
        stride *= step
    return stride, low, high


class Encoder(nn.Module):
    """``forward(x [B,1,X,Y,Z], modality) -> {"G": [B,d_G,x,y,z], "L": [B,d_L,x,y,z], "S": [B,d_S,x,y,z]}``.

    Also ``features(x, modality) -> [B, D, x, y, z]`` with D = d_G + d_L + d_S in that channel order.
    """

    def __init__(
        self,
        stem_channels=32,
        trunk_channels=128,
        trunk_blocks=4,
        trunk_spatial_convs=1,
        down_kernel=3,
        stride=2,
        norm="channel",
        norm_groups=8,
        dims=None,
        use_checkpoint=False,
        stem_norm="none",
    ):
        super().__init__()
        if stride != 2:
            raise ValueError("the trunk has exactly one stride-2 downsampling")
        self.dims = dict(dims or {"G": 16, "L": 32, "S": 64})
        self.use_checkpoint = use_checkpoint
        sc, tc = stem_channels, trunk_channels
        # Stems are unnormalised by default: right after the first conv every channel is a linear function of
        # the same 3^3 patch, so a per-position norm there removes the patch's local intensity scale.
        self.stems = nn.ModuleDict(
            {
                m: nn.Sequential(
                    conv(1, sc, 3),
                    make_norm(stem_norm, norm_groups, sc),
                    nn.SiLU(),
                    conv(sc, sc, 3),
                    make_norm(stem_norm, norm_groups, sc),
                    nn.SiLU(),
                )
                for m in MODALITIES
            }
        )
        self.down = nn.Sequential(conv(sc, tc, down_kernel, stride=2), make_norm(norm, norm_groups, tc), nn.SiLU())
        kernels = [3 if i < trunk_spatial_convs else 1 for i in range(2 * trunk_blocks)]
        if trunk_spatial_convs > 2 * trunk_blocks:
            raise ValueError("more spatial convs than trunk convs")
        self.blocks = nn.ModuleList(
            ResBlock(tc, kernels[2 * i : 2 * i + 2], norm, norm_groups) for i in range(trunk_blocks)
        )
        self.out_norm = make_norm(norm, norm_groups, tc)
        self.head = nn.Conv3d(tc, sum(self.dims[k] for k in LEVELS), 1)
        self._layers = (
            [(3, 1, 1), (3, 1, 1), (down_kernel, 2, 1)] + [(k, 1, (k - 1) // 2) for k in kernels] + [(1, 1, 0)]
        )

    @property
    def stride(self) -> int:
        return footprint(self._layers)[0]

    def theoretical_footprint(self):
        return footprint(self._layers)

    def theoretical_radius(self) -> float:
        """Half-width of the input footprint, in feature voxels."""
        s, low, high = footprint(self._layers)
        return (high - low) / (2 * s)

    def features(self, x, modality: str):
        if modality not in self.stems:
            raise KeyError(f"unknown modality {modality!r}")
        h = self.down(self.stems[modality](x))
        for block in self.blocks:
            if self.use_checkpoint and self.training and h.requires_grad:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.head(F.silu(self.out_norm(h)))

    def split(self, feats):
        out, start = {}, 0
        for level in LEVELS:
            out[level] = feats[:, start : start + self.dims[level]]
            start += self.dims[level]
        return out

    def forward(self, x, modality: str):
        return self.split(self.features(x, modality))


class Template(nn.Module):
    """EMA of each modality's feature at each position over training subjects: ``T_k(u)``, ``[D, x, y, z]``.

    ``update`` folds in a batch mean (the first batch initialises it); ``subtract`` returns ``F - T``.
    Buffers, so the template is saved with every checkpoint.
    """

    def __init__(self, channels: int, grid, momentum: float = 0.999, enabled: bool = True):
        super().__init__()
        self.momentum, self.enabled = momentum, enabled
        for m in MODALITIES:
            self.register_buffer(m, torch.zeros(channels, *grid))
            self.register_buffer(f"{m}_count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, feats, modality: str):
        if not self.enabled:
            return
        t, count = getattr(self, modality), getattr(self, f"{modality}_count")
        mean = feats.detach().float().mean(0)
        if count.item() == 0:
            t.copy_(mean)
        else:
            t.mul_(self.momentum).add_(mean, alpha=1 - self.momentum)
        count += 1

    def subtract(self, feats, modality: str):
        if not self.enabled:
            return feats
        return feats - getattr(self, modality).to(feats.dtype)


class MultiLevelModel(nn.Module):
    """Encoder plus template. ``blocks(x, modality, update)`` -> template-subtracted G, L, S blocks."""

    def __init__(self, encoder: Encoder, template: Template):
        super().__init__()
        self.encoder, self.template = encoder, template

    def blocks(self, x, modality: str, update: bool = False):
        feats = self.encoder.features(x, modality)
        if update:
            self.template.update(feats, modality)
        return self.encoder.split(self.template.subtract(feats, modality))


def build_encoder(model_cfg) -> Encoder:
    return Encoder(
        stem_channels=model_cfg.stem_channels,
        trunk_channels=model_cfg.trunk_channels,
        trunk_blocks=model_cfg.trunk_blocks,
        trunk_spatial_convs=model_cfg.trunk_spatial_convs,
        down_kernel=model_cfg.down_kernel,
        stride=model_cfg.stride,
        norm=model_cfg.norm,
        norm_groups=model_cfg.norm_groups,
        dims=model_cfg.dims,
        use_checkpoint=model_cfg.checkpoint,
        stem_norm=model_cfg.stem_norm,
    )
