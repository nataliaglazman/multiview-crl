"""A dense readout: strided convolutions to a small grid, flattened, then an MLP to the code.

GAP averages the backbone map over every position before the readout. A structure that only
moves leaves that average unchanged, and a small one is diluted by the whole volume. So the
global code cannot be an invertible function of every content latent, whatever the training: in
a supervised control with labels, lesion coordinates stayed at R² 0 through GAP. Flattening keeps
each position's features in inputs of their own, so the readout can compute any function of the
map, position included.

Flattening the conv backbone's 16^3 map at --res 64 directly would give the first linear layer
64 x 4,096 inputs (26M weights at width 100). Strided convolutions first take it to a small grid:

    h (C x 16^3) -> [conv 4^3 stride 2, norm, ReLU] x downsample   -> C x 4^3
                 -> adaptive average to grid^3 (skipped when the map is already grid^3)
                 -> flatten (C x grid^3) -> Linear -> LeakyReLU -> Linear -> code

The convolutions are learned, so they can gather a small site's evidence before the resolution
drops, which averaging into 4^3 bins cannot. The adaptive average makes the input width independent
of the input size (ADNI's 22x27x22 map reaches 5x6x5 and is averaged to 4^3), so a saved model
rebuilds from its settings alone.

It is a global readout only. It has no per-position or per-patch output, and it starts from PyTorch's
random initialization, so its untrained floor is its own, not the GAP model's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseReadout3d(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        out_features: int,
        downsample: int = 2,
        grid: int = 4,
        norm_type: str = "group",
    ):
        super().__init__()
        from models.vqvae import get_group_norm

        if channels < 1 or hidden < 1 or out_features < 1:
            raise ValueError(
                f"channels, hidden and out_features must be positive, got {channels}, {hidden}, {out_features}"
            )
        if downsample < 0 or grid < 1:
            raise ValueError(f"downsample must be nonnegative and grid positive, got {downsample} and {grid}")
        self.channels = channels
        self.downsample = downsample
        self.grid = grid
        layers = []
        for _ in range(downsample):
            layers += [
                nn.Conv3d(channels, channels, 4, stride=2, padding=1),
                get_group_norm(channels, norm_type=norm_type),
                nn.ReLU(inplace=True),
            ]
        self.down = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(channels * grid**3, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, out_features),
        )

    def features(self, h: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) backbone map -> (B, C * grid^3), the MLP's input."""
        if h.ndim != 5 or h.shape[1] != self.channels:
            raise ValueError(f"Expected a (B, {self.channels}, D, H, W) map, got {tuple(h.shape)}")
        z = self.down(h)
        shape = (self.grid,) * 3
        if tuple(z.shape[2:]) != shape:
            if any(size < self.grid for size in z.shape[2:]):
                raise ValueError(
                    f"The dense readout's {self.downsample} downsampling steps leave a {tuple(z.shape[2:])} map, "
                    f"smaller than its {self.grid}^3 grid"
                )
            z = F.adaptive_avg_pool3d(z, shape)
        return z.flatten(1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The global code, (B, out_features)."""
        return self.mlp(self.features(h))
