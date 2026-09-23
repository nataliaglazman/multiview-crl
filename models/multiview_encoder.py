"""Encoder-only multi-view contrastive models for 3D volumes.

``conv`` (default) keeps the original VQ-VAE backbone and affine readout:
    3D strided conv + GroupNorm + ResidualStack -> 1x1 conv -> GAP.
``resnet18`` adapts the upstream image encoder architecture to volumes:
    3D ResNet-18 -> GAP -> Linear(512, 100) -> LeakyReLU -> Linear(100, latent_dim).
The hidden readout width is configurable. This is a 3D adaptation, not a change
to the training objective or the view-sharing policy. Both have no decoder.

The first ``content_channels`` units are the content block, the rest are style.
``forward`` returns the same 8-tuple as ``VQVAE``/``MultiviewVAE`` so
``eval.dci.compute_dci_synthetic`` scores this model unchanged.
"""

import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.resnet3d import ResNet18Features3d
from utils.helper import HelperModule


class MultiviewConvEncoder(HelperModule):
    """Two-encoder 3D conv model with a fixed content/style channel split. No decoder."""

    def build(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        res_channels: int = 32,
        nb_res_layers: int = 2,
        downscale_factor: int = 4,
        latent_dim: int = 16,
        content_channels: int = 9,
        separate_encoders: bool = True,
        use_checkpoint: bool = False,
        proj_dim: int = 0,
        proj_hidden: int = 256,
        encoder_architecture: str = "conv",
        encoder_head_hidden: int = 100,
    ):
        assert (
            0 < content_channels <= latent_dim
        ), f"content_channels ({content_channels}) must be in (0, latent_dim={latent_dim}]"
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.content_channels = content_channels
        self.separate_encoders = separate_encoders
        self.proj_dim = proj_dim
        self.encoder_architecture = encoder_architecture
        self.encoder_head_hidden = encoder_head_hidden
        if encoder_architecture not in ("conv", "resnet18"):
            raise ValueError(f"Unknown encoder_architecture: {encoder_architecture!r}")
        if encoder_architecture == "resnet18" and encoder_head_hidden <= 0:
            raise ValueError("encoder_head_hidden must be positive")

        # --- View-specific encoders ---
        # Encoder 1 is a deep copy of encoder 0 so both views start in the same
        # feature space (random init puts the two views in incompatible subspaces
        # and flattens the cross-view contrastive landscape — see models/vqvae.py).
        if encoder_architecture == "conv":
            from models.vqvae import Encoder

            self.encoder = Encoder(
                in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor, use_checkpoint
            )
        else:
            self.encoder = ResNet18Features3d(in_channels)
        self.encoder_v1 = copy.deepcopy(self.encoder) if separate_encoders else None

        # --- Projection head to the encoding space ---
        if encoder_architecture == "conv":
            # Preserve the original module names, initialization order and weights.
            self.to_encoding = nn.Conv3d(hidden_channels, latent_dim, 1)
        else:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.to_encoding = nn.Sequential(
                nn.Linear(512, encoder_head_hidden),
                nn.LeakyReLU(),
                nn.Linear(encoder_head_hidden, latent_dim),
            )

        # --- Fixed content/style mask over the latent units ---
        fixed_mask = torch.zeros(1, latent_dim)
        fixed_mask[0, :content_channels] = 1.0
        self.register_buffer("content_mask", fixed_mask)

        # --- Contrastive projection head (SimCLR/MoCo recipe) ---
        # The loss runs on the head's output while probes keep reading the pre-head
        # encoding, so InfoNCE can over-compress its own space toward view-invariance
        # without flattening the units being scored. Without it the aligned space and
        # the probed space are the same vector, and alignment removes content along
        # with view information. Shared across views so both encoders, which do not
        # otherwise share weights, land in one comparison space.
        if proj_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(content_channels, proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_dim),
            )
        else:
            self.projector = None

    def _encode(self, x: torch.FloatTensor, n_views: int, view_idx) -> torch.FloatTensor:
        """Route each view through its own encoder, returning a (B, hidden, d, h, w) map."""
        if n_views == 2 and self.encoder_v1 is not None:
            b = x.shape[0] // 2
            return torch.cat([self.encoder(x[:b]), self.encoder_v1(x[b:])], dim=0)
        enc = self.encoder_v1 if (view_idx == 1 and self.encoder_v1 is not None) else self.encoder
        return enc(x)

    def project(self, content_block: torch.FloatTensor) -> torch.FloatTensor:
        """Map a pooled content block into the loss-facing space.

        Applies to the last dimension, so ``(B, C)`` and ``(n_views, B, C)`` both work.
        Returns the input unchanged when no head is configured, so callers can call it
        unconditionally. ``forward`` deliberately does not call it — eval and the DCI
        probes must read the pre-head encoding.
        """
        return content_block if self.projector is None else self.projector(content_block)

    @staticmethod
    def _patch_pool(feat: torch.FloatTensor, patch_grid) -> torch.FloatTensor:
        """Average-pool a spatial map to a (B, C, P) patch grid."""
        if len(patch_grid) > 0 and isinstance(patch_grid[0], (list, tuple)):
            patch_grid = patch_grid[0]  # single level here
        return F.adaptive_avg_pool3d(feat, tuple(patch_grid)).flatten(2)

    def forward(
        self,
        x: torch.FloatTensor,
        return_recon: bool = True,
        pool_only: bool = False,
        n_views: int = 1,
        subsets=None,
        view_idx=None,
        patch_grid=None,
    ) -> Tuple:
        """Encoder-only forward.

        ``return_recon`` is accepted for signature parity with VQVAE/MultiviewVAE
        but ignored — there is no decoder. Returns the 8-tuple
        ``(reconstruction=None, diffs=[0], encoder_features, content_indices,
        [], [], soft_content_masks, {})`` so the synthetic-DCI eval is unchanged.

        For ResNet, GAP precedes the nonlinear readout. Patch probes apply that
        readout separately AFTER averaging each bin; unpooled probes apply it at
        each spatial position. Averaging these diagnostic outputs does not, in
        general, reproduce the trained global encoding.
        """
        h = self._encode(x, n_views, view_idx)
        if self.encoder_architecture == "resnet18":
            if pool_only and patch_grid is None:
                feat = self.to_encoding(self.avgpool(h).flatten(1))
            else:
                if pool_only:
                    grid = (
                        patch_grid[0]
                        if len(patch_grid) > 0 and isinstance(patch_grid[0], (list, tuple))
                        else patch_grid
                    )
                    if len(grid) != 3 or any(g < 1 or g > size for g, size in zip(grid, h.shape[2:])):
                        raise ValueError(
                            f"ResNet patch grid {grid} must fit its spatial map {tuple(h.shape[2:])}; "
                            "the backbone downsamples by 32 (64^3 inputs give 2^3 maps)."
                        )
                    h = F.adaptive_avg_pool3d(h, tuple(grid))
                feat = self.to_encoding(h.flatten(2).transpose(1, 2)).transpose(1, 2)
                if not pool_only:
                    feat = feat.reshape(h.shape[0], self.latent_dim, *h.shape[2:])
            return (
                None,
                [x.new_zeros(())],
                [feat],
                [list(range(self.content_channels))],
                [],
                [],
                {0: self.content_mask},
                {},
            )

        feat = self.to_encoding(h)  # (B, latent_dim, d, h, w)

        if pool_only:
            if patch_grid is not None:
                pooled = self._patch_pool(feat, patch_grid)
            else:
                pooled = feat.mean(dim=[2, 3, 4])
            encoder_features: List[torch.Tensor] = [pooled]
        else:
            encoder_features = [feat]

        soft_content_masks = {0: self.content_mask}
        estimated_content_indices = [list(range(self.content_channels))]

        return (
            None,
            [x.new_zeros(())],
            encoder_features,
            estimated_content_indices,
            [],
            [],
            soft_content_masks,
            {},
        )
