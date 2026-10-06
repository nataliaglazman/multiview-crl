"""Spatial-softmax keypoints over a 3D feature map: a readout that reports WHERE as scalars.

GAP of translation-equivariant features cancels a structure that only moves, so a lesion's
position never reaches the pooled code. Each keypoint head here scores every position with a
1x1 conv, takes a softmax over positions and returns the expected cell-centre coordinate:

    a_k(p) = softmax_p(w_k . h_p + b_k)
    c_k    = sum_p a_k(p) p                       (x, y, z) per head

A head that fires on a lesion reports the lesion's coordinates, and moving the lesion by d
moves c_k by d. For a template in Gaussian noise with a flat prior this expectation is the
posterior-mean position, i.e. the principled way to turn a place code into scalars.

``frame="brain"`` confines the heads to the brain (weights proportional to exp(logit) times the
brain occupancy) and expresses the coordinates relative to the input brain's own centroid and
per-axis spread (from the input's nonzero support at the feature grid). A head sitting on the
brain boundary, or spread evenly over the brain, then reports the same coordinate whatever the
brain's size, so brain size cannot capture it. The generator also places lesions relative to the
white-matter extent, so this is the frame lesion_x/y/z are defined in. ``frame="grid"`` keeps
absolute cell-centre coordinates over the whole grid.

``norm="layer"`` applies LayerNorm over channels at each voxel before the logits, which removes a
per-voxel gain on the features. It helped against a pure gain and hurt against a geometric factor
in a toy, so it is off by default.

The logits start small and random rather than at zero: with zero weights every subject's
coordinates are exactly the grid centre, the branch's own InfoNCE sees identical codes and its
gradient vanishes.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def cell_centres(spatial_size, device=None, dtype=None):
    """(N, 3) cell-centre coordinates in [-1, 1] of a (D, H, W) grid, in ``flatten`` order."""
    axes = [(torch.arange(n, device=device, dtype=dtype) * 2 + 1) / n - 1 for n in spatial_size]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(spatial_size))


def brain_frame(support, grid, eps=1e-6):
    """Centroid and per-axis spread of each subject's brain occupancy.

    ``support`` is (B, 1, D, H, W) in [0, 1] on the feature grid, ``grid`` the (N, 3) cell centres.
    Returns two (B, 3) tensors. Both scale and shift with the brain, so dividing a coordinate's
    offset from the centroid by the spread is invariant to a global rescaling of the brain.
    """
    weights = support.flatten(1)
    mass = weights.sum(1, keepdim=True).clamp_min(eps)
    centre = weights @ grid / mass
    variance = weights @ grid.square() / mass - centre.square()
    return centre, variance.clamp_min(eps).sqrt()


class KeypointPool3d(nn.Module):
    """(B, C, D, H, W) -> (B, 3K): one expected (x, y, z) per head."""

    def __init__(self, channels: int, num_keypoints: int = 4, norm: str = "none", frame: str = "brain"):
        super().__init__()
        if num_keypoints < 1:
            raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")
        if norm not in ("none", "layer"):
            raise ValueError(f"norm must be none or layer, got {norm!r}")
        if frame not in ("brain", "grid"):
            raise ValueError(f"frame must be brain or grid, got {frame!r}")
        self.channels = channels
        self.num_keypoints = num_keypoints
        self.norm = norm
        self.frame = frame
        self.logits = nn.Conv3d(channels, num_keypoints, 1)
        nn.init.normal_(self.logits.weight, std=0.1 / math.sqrt(channels))
        nn.init.zeros_(self.logits.bias)

    @property
    def out_features(self) -> int:
        return 3 * self.num_keypoints

    def attention(self, h: torch.Tensor, support: torch.Tensor = None) -> torch.Tensor:
        """Per-head weights (B, K, N) over the positions; each row sums to 1.

        With ``support`` (brain occupancy on the feature grid) the weights are proportional to
        exp(logit) times occupancy, so heads only look inside the brain.
        """
        if h.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {h.shape[1]}")
        if self.norm == "layer":
            h = F.layer_norm(h.movedim(1, -1), (self.channels,)).movedim(-1, 1)
        logits = self.logits(h).flatten(2)
        if support is not None:
            logits = logits + support.to(h.dtype).flatten(2).clamp_min(1e-6).log()
        return logits.softmax(dim=-1)

    def forward(self, h: torch.Tensor, support: torch.Tensor = None) -> torch.Tensor:
        grid = cell_centres(h.shape[2:], device=h.device, dtype=h.dtype)
        if self.frame == "grid":
            return (self.attention(h) @ grid).flatten(1)
        if support is None:
            raise ValueError("frame='brain' needs the brain support on the feature grid")
        # Attention is confined to the brain: over the whole grid, a near-uniform head reports
        # the grid centre, whose position relative to the brain is a function of brain size.
        coords = self.attention(h, support) @ grid  # (B, K, 3)
        centre, spread = brain_frame(support.to(h.dtype), grid)
        return ((coords - centre[:, None]) / spread[:, None]).flatten(1)

    def maps(self, h: torch.Tensor, support: torch.Tensor = None) -> torch.Tensor:
        """Keypoint weights on the feature grid, (B, K, D, H, W), as ``forward`` uses them."""
        if self.frame == "brain" and support is None:
            raise ValueError("frame='brain' needs the brain support on the feature grid")
        weights = self.attention(h, support if self.frame == "brain" else None)
        return weights.reshape(h.shape[0], self.num_keypoints, *h.shape[2:])
