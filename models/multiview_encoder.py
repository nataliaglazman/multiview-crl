"""Encoder-only multi-view contrastive model (Yao et al. ICLR 2024, image track).

This is the paper's *image-experiment* setup adapted to 3D volumes: per-view conv
encoders trained purely by the content-alignment − entropy contrastive loss, with
**no decoder and no reconstruction** (App D.2/D.3; the paper's ResNet-18 →
LeakyReLU → Linear(encoding_size) head, here in 3D). It is deliberately
``models.vae.MultiviewVAE`` *minus* the decoder/KL, so encoder-only-contrastive vs
VQ-VAE-with-reconstruction is a controlled test of the paper's claim that
reconstruction hurts identifiability — same backbone, only the objective differs.

Pipeline (per view):
    volume (B,1,D,H,W)
      → models.vqvae.Encoder  (3D strided conv + GroupNorm + ResidualStack)
      → (B, hidden, d, h, w)
      → 1x1 conv to the encoding space  → (B, latent_dim, d, h, w)
      → pool: GAP (paper's ResNet→GAP→Linear; a 1x1 conv then GAP == GAP then
        linear) or patch-grid → encoding vector.

The first ``content_channels`` units are the content block, the rest are style.
``forward`` returns the same 8-tuple as ``VQVAE``/``MultiviewVAE`` so
``eval.dci.compute_dci_synthetic`` scores this model unchanged.
"""

import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.vqvae import Encoder
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
    ):
        assert (
            0 < content_channels <= latent_dim
        ), f"content_channels ({content_channels}) must be in (0, latent_dim={latent_dim}]"
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.content_channels = content_channels
        self.separate_encoders = separate_encoders

        # --- View-specific encoders ---
        # Encoder 1 is a deep copy of encoder 0 so both views start in the same
        # feature space (random init puts the two views in incompatible subspaces
        # and flattens the cross-view contrastive landscape — see models/vqvae.py).
        self.encoder = Encoder(
            in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor, use_checkpoint
        )
        self.encoder_v1 = copy.deepcopy(self.encoder) if separate_encoders else None

        # --- Projection head to the encoding space ---
        # 1x1 conv to latent_dim channels. Deterministic (no VAE head): the encoding
        # is the pooled projection, matching the paper's GAP→Linear readout.
        self.to_encoding = nn.Conv3d(hidden_channels, latent_dim, 1)

        # --- Fixed content/style mask over the latent units ---
        fixed_mask = torch.zeros(1, latent_dim)
        fixed_mask[0, :content_channels] = 1.0
        self.register_buffer("content_mask", fixed_mask)

    def _encode(self, x: torch.FloatTensor, n_views: int, view_idx) -> torch.FloatTensor:
        """Route each view through its own encoder, returning a (B, hidden, d, h, w) map."""
        if n_views == 2 and self.encoder_v1 is not None:
            b = x.shape[0] // 2
            return torch.cat([self.encoder(x[:b]), self.encoder_v1(x[b:])], dim=0)
        enc = self.encoder_v1 if (view_idx == 1 and self.encoder_v1 is not None) else self.encoder
        return enc(x)

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
        """
        h = self._encode(x, n_views, view_idx)
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
