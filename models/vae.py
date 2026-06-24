"""Simple multiview VAE (no VQ, no codebooks) for the synthetic identifiability check.

Two view-specific encoders (T1 / T2-like) map paired synthetic volumes to a shared
Gaussian latent grid.  The latent channels are split by a *fixed* mask into a content
block (first ``content_channels``) and a style block (the rest).  Cross-view alignment
of the pooled content block is driven by a contrastive loss; per-view reconstruction is
driven by an L1 loss; a small KL term keeps it variational.

The intended experiment: set ``content_channels`` to the true number of recoverable
shared factors (the synthetic dataset's ``n_content``) and verify that the content block
becomes identifiable — high content->content informativeness, low content->view leakage —
via ``eval.dci.compute_dci_synthetic``.  Sweeping ``content_channels`` should show the
identifiability peak at the correct dimensionality.

The conv building blocks (``Encoder``, ``Decoder``, ``get_group_norm``) are reused from
``models.vqvae`` so this stays a thin wrapper: only the variational head, the fixed
content/style split, and an eval-compatible ``forward`` live here.  ``forward`` returns the
same 8-tuple layout as ``VQVAE.forward`` (reconstruction, per-level "diffs", per-level
features, content indices, decoder outputs, id outputs, soft content masks, style ids) so
the existing synthetic-DCI evaluation accepts this model unchanged.
"""

import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.vqvae import Decoder, Encoder, get_group_norm
from utils.helper import HelperModule


class MultiviewVAE(HelperModule):
    """Two-encoder Gaussian VAE with a fixed content/style channel split."""

    def build(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        res_channels: int = 32,
        nb_res_layers: int = 2,
        downscale_factor: int = 4,
        latent_channels: int = 16,
        content_channels: int = 9,
        beta_kl: float = 1e-4,
        separate_encoders: bool = True,
        use_checkpoint: bool = False,
    ):
        assert (
            0 < content_channels <= latent_channels
        ), f"content_channels ({content_channels}) must be in (0, latent_channels={latent_channels}]"
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.content_channels = content_channels
        self.beta_kl = beta_kl
        self.separate_encoders = separate_encoders

        # --- View-specific encoders ---
        # Encoder 1 is a deep copy of encoder 0 so both start in the same feature
        # space (random init would put the two views in incompatible subspaces and
        # flatten the cross-view contrastive loss landscape — see models/vqvae.py).
        self.encoder = Encoder(
            in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor, use_checkpoint
        )
        if separate_encoders:
            self.encoder_v1 = copy.deepcopy(self.encoder)
        else:
            self.encoder_v1 = None

        # --- Gaussian latent head (shared across views) ---
        # 1x1 conv to (mu, logvar); a second 1x1 conv maps a latent sample back to
        # hidden_channels for the decoder.
        self.to_latent = nn.Conv3d(hidden_channels, 2 * latent_channels, 1)
        self.from_latent = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_channels, 1),
            get_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # --- Decoder (shared across views; plain, no style injection) ---
        # final_norm=False so the output can carry absolute per-sample intensity
        # instead of being instance-normalised to a fixed mean/std.
        self.decoder = Decoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            res_channels=res_channels,
            nb_res_layers=nb_res_layers,
            upscale_factor=downscale_factor,
            use_checkpoint=use_checkpoint,
            style_channels=0,
            final_norm=False,
        )

        # --- Fixed content/style mask ---
        # First ``content_channels`` latent channels are content, the rest style.
        # Registered as a buffer so it moves with .to(device) and is picked up by
        # eval.dci.compute_dci_synthetic (which reads soft_content_masks[0]).
        fixed_mask = torch.zeros(1, latent_channels)
        fixed_mask[0, :content_channels] = 1.0
        self.register_buffer("content_mask", fixed_mask)

    # ------------------------------------------------------------------ #
    def _encode(self, x: torch.FloatTensor, n_views: int, view_idx: int | None) -> torch.FloatTensor:
        """Route each view through its own encoder, returning a (B, hidden, d, h, w) map."""
        if n_views == 2 and self.encoder_v1 is not None:
            b = x.shape[0] // 2
            h0 = self.encoder(x[:b])
            h1 = self.encoder_v1(x[b:])
            return torch.cat([h0, h1], dim=0)
        enc = self.encoder_v1 if (view_idx == 1 and self.encoder_v1 is not None) else self.encoder
        return enc(x)

    @staticmethod
    def _patch_pool(feat: torch.FloatTensor, patch_grid) -> torch.FloatTensor:
        """Average-pool a spatial map to a (B, C, P) patch grid."""
        if len(patch_grid) > 0 and isinstance(patch_grid[0], (list, tuple)):
            patch_grid = patch_grid[0]  # single level here
        pooled = F.adaptive_avg_pool3d(feat, tuple(patch_grid))
        return pooled.flatten(2)

    def forward(
        self,
        x: torch.FloatTensor,
        return_recon: bool = True,
        pool_only: bool = False,
        n_views: int = 1,
        subsets=None,
        view_idx: int | None = None,
        patch_grid=None,
    ) -> Tuple:
        """Forward pass.

        Args:
            x: Input ``(B, C, D, H, W)``.  When ``n_views == 2`` the batch is the
               concatenation ``[view0 | view1]`` and each half is routed through its
               own encoder.
            return_recon: If False, skip the decoder/KL (contrastive-only mode).
            pool_only: If True, return globally pooled ``(B, latent_channels)`` features
               (or ``(B, latent_channels, P)`` patch features when ``patch_grid`` is set)
               instead of the spatial latent map.
            n_views, view_idx, patch_grid, subsets: kept for parity with
               ``VQVAE.forward`` so the synthetic-DCI eval can call this unchanged.

        Returns:
            8-tuple ``(reconstruction, diffs, encoder_features, estimated_content_indices,
            decoder_outputs, id_outputs, soft_content_masks, style_id_outputs)``.
            ``diffs`` holds the per-sample KL (the VAE analogue of the VQ commitment
            loss); ``encoder_features`` is a single-level list.
        """
        in_size = x.shape[2:]
        h = self._encode(x, n_views, view_idx)

        mu, logvar = self.to_latent(h).chunk(2, dim=1)
        logvar = logvar.clamp(-10.0, 10.0)
        # Sample only when training and reconstructing; the representation used for
        # contrastive loss / probing is always the deterministic mean.
        if self.training and return_recon:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        feat = mu

        recon = None
        kl = x.new_zeros(())
        if return_recon:
            recon = self.decoder(self.from_latent(z))
            if recon.shape[2:] != in_size:
                recon = F.interpolate(recon, size=in_size, mode="trilinear", align_corners=False)
            # KL to N(0, I), summed over latent dims, averaged over the batch.
            kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(dim=[1, 2, 3, 4]).mean()

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
            recon,
            [kl],
            encoder_features,
            estimated_content_indices,
            [],
            [],
            soft_content_masks,
            {},
        )
