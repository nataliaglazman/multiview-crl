"""First spatial moments of a 3D feature map, read into the global readout beside GAP.

GAP keeps each channel's mean and nothing about where it sits, so a fixed-size structure that
only moves (a lesion) leaves it unchanged. The first moments keep that:

    m_c^x = (1/N) sum_p x_p h_c(p)        likewise for y and z

with p over the N positions of the backbone map and (x, y, z) each position's coordinates on
that map, scaled to [-1, 1] per axis. They are linear in the map, so a channel that responds
only to a fixed-size blob gives mass * centre / N: the blob's position, read off linearly.

The pool returns the readout's extra input term, W m with m = [m^x, m^y, m^z] (3C numbers).
The encoder adds it to its first readout layer's pre-activation (the hidden layer of the MLP
readout, or the code itself for the linear readout). That is the same as appending 3C zero
columns to that layer. W starts at zero and is built with torch.zeros, which draws no random
numbers: a model built from the same seed has the GAP model's weights, and its readout is the
GAP model's exactly until training moves W.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def axis_coordinates(size: int, like: torch.Tensor) -> torch.Tensor:
    """Position coordinates along one axis of the map, in [-1, 1]; a single position sits at 0."""
    if size == 1:
        return like.new_zeros(1)
    return torch.linspace(-1.0, 1.0, size, device=like.device, dtype=like.dtype)


def spatial_moments(h: torch.Tensor) -> torch.Tensor:
    """Return each channel's coordinate-weighted means as ``(B, 3C)``."""
    if h.ndim != 5:
        raise ValueError(f"Expected a 5D feature map, got {tuple(h.shape)}")
    d, hh, w = h.shape[2:]
    mx = (h.mean((3, 4)) * axis_coordinates(d, h)).mean(-1)
    my = (h.mean((2, 4)) * axis_coordinates(hh, h)).mean(-1)
    mz = (h.mean((2, 3)) * axis_coordinates(w, h)).mean(-1)
    return torch.cat([mx, my, mz], dim=1)


class MomentPool3d(nn.Module):
    def __init__(self, channels: int, out_features: int):
        super().__init__()
        if channels < 1 or out_features < 1:
            raise ValueError(f"channels and out_features must be positive, got {channels} and {out_features}")
        self.channels = channels
        self.weight = nn.Parameter(torch.zeros(out_features, 3 * channels))

    def moments(self, h: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) map -> (B, 3C): every channel's x moments, then y, then z."""
        if h.ndim != 5 or h.shape[1] != self.channels:
            raise ValueError(f"Expected a (B, {self.channels}, D, H, W) map, got {tuple(h.shape)}")
        return spatial_moments(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The readout's extra pre-activation, (B, out_features)."""
        return F.linear(self.moments(h), self.weight)
