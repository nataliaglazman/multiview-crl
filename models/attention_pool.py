"""Multi-head attention pooling over the positions of a 3D feature map.

A drop-in replacement for global average pooling (GAP) in the encoder-only readout.
GAP of translation-equivariant features cancels anything that only moves: a
fixed-size lesion deleted in one place and added in another leaves the average
unchanged. It also dilutes a compact signal with the rest of the volume. Attention
pooling weights the positions instead, and its positional encoding lets a head that
locks onto a structure also report where that structure is.

For head m with learned query q_m and backbone features h_p at position p:

    x_p    = h_p + W_pos phi(p)                 phi: fixed Fourier features of p
    a_m(p) = softmax_p(q_m . LayerNorm(x_p))    over all positions
    out_m  = sum_p a_m(p) x_p[channels of m]    each head pools C / heads channels

A learned query scored against a linear key projection reduces to one learned vector
per head, so the keys are the normalized tokens themselves. Values are the raw tokens,
since the readout that follows is already a learned map. The key LayerNorm keeps
attention sharpness tied to |q_m| rather than to the backbone's feature scale.

Both parameters start at zero, so the weights are uniform, the positional term
vanishes and the pool equals GAP up to float rounding. Zero initialization draws no
random numbers: every other weight of a model built from the same seed is unchanged,
and so is its untrained floor. ``num_frequencies=0`` drops the positional encoding,
which leaves attention permutation-invariant over positions, like GAP.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def fourier_positions(spatial_size, num_frequencies, device=None, dtype=None):
    """(N, 6 * num_frequencies) Fourier features of a (D, H, W) grid's cell centres.

    Centres are in [-1, 1], in ``flatten`` order. Each axis gets sin and cos at angular
    frequencies 2^k * pi / 2. The lowest is half a period across the volume, so it is
    monotone along the axis and opposite faces do not wrap onto each other.
    """
    axes = [(torch.arange(n, device=device, dtype=dtype) * 2 + 1) / n - 1 for n in spatial_size]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(spatial_size))
    freqs = (math.pi / 2) * 2.0 ** torch.arange(num_frequencies, device=device, dtype=dtype)
    angles = (coords[:, :, None] * freqs).flatten(1)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class AttentionPool3d(nn.Module):
    """(B, C, D, H, W) -> (B, C): per-head softmax over positions, one channel slice per head."""

    def __init__(self, channels: int, num_heads: int = 4, num_frequencies: int = 4):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError(f"num_heads ({num_heads}) must divide the {channels} pooled channels")
        if num_frequencies < 0:
            raise ValueError(f"num_frequencies must be nonnegative, got {num_frequencies}")
        self.channels = channels
        self.num_heads = num_heads
        self.num_frequencies = num_frequencies
        self.query = nn.Parameter(torch.zeros(num_heads, channels))
        if num_frequencies > 0:
            self.position = nn.Parameter(torch.zeros(channels, 6 * num_frequencies))
        else:
            self.register_parameter("position", None)

    def tokens(self, h: torch.Tensor) -> torch.Tensor:
        """Backbone map -> position-augmented tokens (B, N, C)."""
        x = h.flatten(2).transpose(1, 2)
        if self.position is None:
            return x
        phi = fourier_positions(h.shape[2:], self.num_frequencies, device=h.device, dtype=h.dtype)
        return x + phi @ self.position.T

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head weights (B, heads, N) over the token positions; each row sums to 1."""
        keys = F.layer_norm(x, (self.channels,))
        return torch.einsum("bnc,mc->bmn", keys, self.query).softmax(dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {h.shape[1]}")
        x = self.tokens(h)
        b, n, c = x.shape
        values = x.reshape(b, n, self.num_heads, c // self.num_heads)
        return torch.einsum("bmn,bnmc->bmc", self.attention(x), values).reshape(b, c)

    def maps(self, h: torch.Tensor) -> torch.Tensor:
        """Attention weights on the backbone grid, (B, heads, D, H, W)."""
        return self.attention(self.tokens(h)).reshape(h.shape[0], self.num_heads, *h.shape[2:])
