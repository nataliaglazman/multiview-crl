"""Classic (non-variational) multiview autoencoder with a *dense-vector* bottleneck.

Unlike ``models.vae.MultiviewVAE`` — whose 1x1-conv latent stays a spatial grid, so
"the representation" only exists after a pooling choice (GAP / patch / stats) — this
model flattens the encoder output through a single ``Linear`` to a global latent
*vector* of size ``latent_dim``.  The vector IS the representation: there is nothing
to pool, which removes the pooling degree of freedom entirely.

Everything else mirrors ``MultiviewVAE`` so the swap is controlled — same conv
backbone (``models.vqvae.Encoder``/``Decoder``), same two view-specific encoders,
same fixed content/style channel split, same 8-tuple ``forward`` — so
``eval.dci.compute_dci_synthetic`` scores it unchanged.  The differences from the VAE
are exactly the two things under test:

  * deterministic bottleneck (no ``mu``/``logvar``, no KL) — a *classic* AE;
  * a dense ``Linear`` bottleneck instead of a 1x1 conv, so the latent has no spatial
    extent.  With ``latent_dim`` ~ ``n_content + n_style`` this is a genuine
    information bottleneck matched to the synthetic data's intrinsic dimensionality,
    rather than the VAE's ~512x-redundant spatial latent.

Because the latent is a vector, ``pool_only`` / ``patch_grid`` are no-ops: the model
always returns the ``(B, latent_dim)`` code.  Evaluate it at ``pooling="gap"`` (the
natural — and only sensible — readout for a dense vector).
"""

import copy
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.vqvae import Decoder, Encoder, get_group_norm
from utils.helper import HelperModule


class MultiviewAE(HelperModule):
    """Two-encoder classic AE with a dense-vector bottleneck and fixed content/style split."""

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
        res: int = 32,
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
        # Encoder 1 is a deep copy of encoder 0 so both views start in the same feature
        # space (random init puts the two views in incompatible subspaces and flattens
        # the cross-view contrastive landscape — see models/vqvae.py).
        self.encoder = Encoder(
            in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor, use_checkpoint
        )
        self.encoder_v1 = copy.deepcopy(self.encoder) if separate_encoders else None

        # --- Infer the encoder's spatial output shape once ---
        # The dense bottleneck's Linear layers need a fixed input size, and it must be
        # known before any forward (so a state_dict loads cleanly). A cheap CPU dummy
        # pass is robust to whatever downsampling the Encoder actually applies.
        with torch.no_grad():
            enc_shape = self.encoder(torch.zeros(1, in_channels, res, res, res)).shape[1:]
        self.bottleneck_shape = tuple(int(s) for s in enc_shape)  # (hidden, d, h, w)
        flat_dim = math.prod(self.bottleneck_shape)

        # --- Dense bottleneck (the one defining difference from the VAE) ---
        # flatten -> Linear -> (B, latent_dim). No spatial extent, nothing to pool.
        self.to_latent = nn.Sequential(nn.Flatten(1), nn.Linear(flat_dim, latent_dim))
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.Unflatten(1, self.bottleneck_shape),
            get_group_norm(self.bottleneck_shape[0]),
            nn.ReLU(inplace=True),
        )

        # --- Decoder (shared across views; plain, no style injection) ---
        # final_norm=False so the output can carry absolute per-sample intensity.
        self.decoder = Decoder(
            in_channels=self.bottleneck_shape[0],
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
        # First ``content_channels`` latent units are content, the rest style. Buffer so
        # it moves with .to(device) and is read by eval.dci.compute_dci_synthetic
        # (which reads soft_content_masks[0]).
        fixed_mask = torch.zeros(1, latent_dim)
        fixed_mask[0, :content_channels] = 1.0
        self.register_buffer("content_mask", fixed_mask)

    # ------------------------------------------------------------------ #
    def _encode(self, x: torch.FloatTensor, n_views: int, view_idx: int | None) -> torch.FloatTensor:
        """Route each view through its own encoder, returning a (B, hidden, d, h, w) map."""
        if n_views == 2 and self.encoder_v1 is not None:
            b = x.shape[0] // 2
            return torch.cat([self.encoder(x[:b]), self.encoder_v1(x[b:])], dim=0)
        enc = self.encoder_v1 if (view_idx == 1 and self.encoder_v1 is not None) else self.encoder
        return enc(x)

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
        """Forward pass, returning the same 8-tuple layout as ``VQVAE``/``MultiviewVAE``.

        The latent is a dense vector, so ``pool_only`` / ``patch_grid`` are accepted
        for signature parity but are no-ops — ``encoder_features`` is always the
        ``(B, latent_dim)`` code.  ``diffs`` is ``[0]`` (a classic AE has no KL /
        commitment term) so the tuple stays compatible with the synthetic-DCI eval.
        """
        in_size = x.shape[2:]
        h = self._encode(x, n_views, view_idx)
        z = self.to_latent(h)  # (B, latent_dim) — the representation

        recon = None
        if return_recon:
            recon = self.decoder(self.from_latent(z))
            if recon.shape[2:] != in_size:
                recon = F.interpolate(recon, size=in_size, mode="trilinear", align_corners=False)

        encoder_features: List[torch.Tensor] = [z]
        soft_content_masks = {0: self.content_mask}
        estimated_content_indices = [list(range(self.content_channels))]

        return (
            recon,
            [x.new_zeros(())],
            encoder_features,
            estimated_content_indices,
            [],
            [],
            soft_content_masks,
            {},
        )
