"""Per-level expanders, used only inside the loss: Linear(d -> h1), BatchNorm, ReLU, ..., Linear(-> h_k)."""

from __future__ import annotations

import torch.nn as nn


class Expander(nn.Module):
    """``[N, d] -> [N, widths[-1]]``. With the default widths [512, 512]: Linear, BatchNorm, ReLU, Linear."""

    def __init__(self, d_in: int, widths=(512, 512)):
        super().__init__()
        layers, d = [], d_in
        for i, w in enumerate(widths):
            layers.append(nn.Linear(d, w))
            if i < len(widths) - 1:
                layers += [nn.BatchNorm1d(w), nn.ReLU()]
            d = w
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
