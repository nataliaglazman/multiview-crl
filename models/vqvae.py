import copy
from math import log2
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import utils.utils as utils
from utils.helper import HelperModule


def get_group_norm(channels, target_groups=32):
    """
    Finds the largest divisor of 'channels' that is <= 'target_groups'.
    Ensures num_groups is always valid for nn.GroupNorm.
    """
    # Start at target_groups and work downwards to find the first divisor
    for g in range(target_groups, 0, -1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)

    # Fallback (should theoretically never be reached as 1 divides everything)
    return nn.GroupNorm(1, channels)


class ReZero(HelperModule):
    """3D ReZero residual block with learnable scaling parameter."""

    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            get_group_norm(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            get_group_norm(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(HelperModule):
    """Stack of 3D ReZero residual blocks with optional gradient checkpointing."""

    def build(
        self,
        in_channels: int,
        res_channels: int,
        nb_layers: int,
        use_checkpoint: bool = True,
    ):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) for _ in range(nb_layers)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.use_checkpoint and self.training:
            x = checkpoint(self.stack, x, use_reentrant=False)
        else:
            x = self.stack(x)
        return x


class Encoder(HelperModule):
    """3D Encoder with strided convolutions for downsampling."""

    def build(
        self,
        in_channels: int,
        hidden_channels: int,
        res_channels: int,
        nb_res_layers: int,
        downscale_factor: int,
        use_checkpoint: bool = True,
    ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(c_channel, n_channel, 4, stride=2, padding=1),
                    get_group_norm(n_channel),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv3d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(get_group_norm(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers, use_checkpoint=use_checkpoint))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(HelperModule):
    """3D Decoder with transposed convolutions for upsampling.

    When ``style_channels > 0`` the final Conv3d is held in a separate
    ``self.final_conv`` attribute so that style features can be concatenated
    onto the penultimate feature map before the output projection.  All other
    layers live in ``self.layers`` as before.
    """

    def build(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        res_channels: int,
        nb_res_layers: int,
        upscale_factor: int,
        use_checkpoint: bool = True,
        style_channels: int = 0,
    ):
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        self.style_channels = style_channels

        layers = [nn.Conv3d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(
            ResidualStack(
                hidden_channels,
                res_channels,
                nb_res_layers,
                use_checkpoint=use_checkpoint,
            )
        )
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(c_channel, n_channel, 4, stride=2, padding=1),
                    get_group_norm(n_channel),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, out_channels

        if style_channels > 0:
            # Keep the body (everything up to the final conv) in self.layers;
            # the final conv is stored separately so we can insert style features
            # between them at forward time.
            self.layers = nn.Sequential(*layers)
            # The penultimate feature map has c_channel channels; style is
            # concatenated before projecting to out_channels.
            self.final_conv = nn.Sequential(
                nn.Conv3d(c_channel + style_channels, out_channels, 3, stride=1, padding=1),
                get_group_norm(out_channels),
            )
        else:
            layers.append(nn.Conv3d(c_channel, out_channels, 3, stride=1, padding=1))
            layers.append(get_group_norm(out_channels))
            self.layers = nn.Sequential(*layers)
            self.final_conv = None

    def forward(
        self,
        x: torch.FloatTensor,
        style: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        Args:
            x: Input feature map ``(B, in_channels, D, H, W)``.
            style: Optional style tensor ``(B, style_channels, D', H', W')``.
                   Required when ``self.style_channels > 0``.  Will be
                   trilinearly upsampled to match ``x``'s spatial dims before
                   concatenation.

        Returns:
            Decoded output ``(B, out_channels, D_out, H_out, W_out)``.
        """
        feat = self.layers(x)
        if self.final_conv is not None:
            if style is None:
                raise ValueError("Decoder was built with style_channels > 0 but no style tensor was provided.")
            if style.shape[2:] != feat.shape[2:]:
                style = F.interpolate(style, size=feat.shape[2:], mode="trilinear", align_corners=False)
            feat = torch.cat([feat, style], dim=1)
            feat = self.final_conv(feat)
        return feat


# Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# No reason to reinvent this rather complex mechanism.
#
# Adapted for 3D volumes (brain MRI).
#
# Essentially handles the "discrete" part of the network, and training through EMA rather than
# the third term in the loss function.


class CodeLayer(HelperModule):
    """3D Vector Quantization layer with EMA codebook updates."""

    def build(self, in_channels: int, embed_dim: int, nb_entries: int):
        self.conv_in = nn.Conv3d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    def project(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project input to embed_dim via conv_in. Returns (B, embed_dim, D, H, W)."""
        return self.conv_in(x.float())

    @torch.amp.autocast("cuda", enabled=False)
    def quantize(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        """Quantize a pre-projected (B, embed_dim, D, H, W) tensor."""
        x = x.float().permute(0, 2, 3, 4, 1)
        flatten = x.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 4, 1, 2, 3), diff, embed_ind

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        return self.quantize(self.project(x))

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Upscaler(HelperModule):
    """3D Upscaler for hierarchical code conditioning."""

    def build(
        self,
        embed_dim: int,
        scaling_rates: list[int],
    ):
        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose3d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(get_group_norm(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)


# Main VQ-VAE-2 Module for 3D volumes, capable of supporting arbitrary number of levels.
# Adapted for brain MRI data.
#
# TODO: A lot of this class could do with a refactor. It works, but at what cost?
# TODO: Add discrete code decoding function


class VQVAE(HelperModule):
    def build(
        self,
        in_channels: int = 1,  # 1 for grayscale MRI
        hidden_channels: int = 128,
        res_channels: int = 32,
        nb_res_layers: int = 2,
        nb_levels: int = 3,
        embed_dim: int = 64,
        nb_entries: int = 512,
        scaling_rates: list[int] = [8, 4, 2],
        use_checkpoint: bool = True,  # Gradient checkpointing to save memory
        content_size: int = 0,  # # of content dims in original latent space
        style_size: int = 0,  # # of style dims in original latent space
        inject_style_to_decoder: bool = False,  # Append style latent from encoder-0 to final decoder layer
    ):
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"
        self.nb_levels = nb_levels
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    in_channels,
                    hidden_channels,
                    res_channels,
                    nb_res_layers,
                    scaling_rates[0],
                    use_checkpoint,
                )
            ]
        )
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(
                Encoder(
                    hidden_channels,
                    hidden_channels,
                    res_channels,
                    nb_res_layers,
                    sr,
                    use_checkpoint,
                )
            )

        # Optional content/style separation on the encoder's hidden_channels.
        # The Gumbel mask is applied to raw encoder output at level 0,
        # BEFORE the codebook projection, so it operates in a space where
        # individual channels tend to specialise and are more separable.
        if content_size > 0 and style_size > 0:
            total_size = content_size + style_size
            self.content_channels = max(1, round(content_size / total_size * hidden_channels))
            # Learnable logits over hidden_channels (one per encoder channel).
            self.channel_logits = nn.Parameter(torch.zeros(hidden_channels))
        else:
            self.content_channels = None
            self.channel_logits = None

        # Optional style injection: style channels from the hidden_channels space
        # are upsampled and concatenated onto the penultimate decoder-0 feature map.
        self.inject_style_to_decoder = inject_style_to_decoder and (self.channel_logits is not None)
        if self.inject_style_to_decoder:
            self.style_channels = hidden_channels - self.content_channels
        else:
            self.style_channels = 0

        self.codebooks = nn.ModuleList()
        # Index 0 = finest level (l=0 in forward): has decoder conditioning.
        # When content masking is active, encoder input is content_channels only.
        if self.channel_logits is not None:
            self.codebooks.append(CodeLayer(self.content_channels + embed_dim, embed_dim, nb_entries))
        else:
            self.codebooks.append(CodeLayer(hidden_channels + embed_dim, embed_dim, nb_entries))
        # Indices 1..nb_levels-2: middle levels with decoder conditioning
        for i in range(1, nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels + embed_dim, embed_dim, nb_entries))
        # Index nb_levels-1 = coarsest level: no decoder conditioning
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    embed_dim * nb_levels,
                    hidden_channels,
                    in_channels,
                    res_channels,
                    nb_res_layers,
                    scaling_rates[0],
                    use_checkpoint,
                    style_channels=self.style_channels,
                )
            ]
        )
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(
                Decoder(
                    embed_dim * (nb_levels - 1 - i),
                    hidden_channels,
                    embed_dim,
                    res_channels,
                    nb_res_layers,
                    sr,
                    use_checkpoint,
                )
            )

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            rates = scaling_rates[1 : len(scaling_rates) - i][::-1]  # noqa: E203
            self.upscalers.append(Upscaler(embed_dim, rates))

    def forward(self, x, return_recon=True, pool_only=False, n_views=1, subsets=None):
        """Forward pass through VQ-VAE-2.

        Args:
            x: Input tensor (B, C, D, H, W)
            return_recon: If False, skip decoder for memory efficiency (contrastive-only mode)
            pool_only: If True, return per-level pooled (B, C) vectors instead of spatial maps.
            n_views: Number of views (unused, kept for API compat).
            subsets: View subsets (unused, kept for API compat).

        Returns:
            final_output: Reconstruction (or None if return_recon=False)
            diffs: VQ commitment losses per level
            encoder_features: Per-level encoder features.
                              If pool_only=True:  list of (B, hidden_channels) pooled vectors
                              If pool_only=False: list of (B, C, D, H, W) spatial maps
            estimated_content_indices: Content channel indices from the Gumbel mask
                                       (None if channel_logits not configured)
            decoder_outputs: Decoder features per level (or empty list)
            id_outputs: Codebook indices per level
        """
        encoder_outputs = []  # Spatial (5D) feature maps, consumed by codebook/decoder loop
        encoder_pools = []  # Pooled (B, C) vectors, returned for contrastive loss
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []
        estimated_content_indices = None
        style_spatial = None  # style channels from encoder level-0; used for decoder injection

        # Encoder forward pass
        enc_input = x
        for i, enc in enumerate(self.encoders):
            enc_input = enc(enc_input)
            encoder_outputs.append(enc_input)
            if pool_only:
                encoder_pools.append(enc_input.mean(dim=[2, 3, 4]))

        del x, enc_input

        # --- Content/style mask on level-0 encoder output (hidden_channels) ---
        # Applied BEFORE the codebook projection so the mask operates on raw
        # encoder channels, which tend to specialise and are more separable
        # than the compressed embed_dim projection.
        content_enc_out = None  # content-only spatial map for level-0 codebook
        if self.channel_logits is not None:
            enc_out_l0 = encoder_outputs[0]

            logits = self.channel_logits.unsqueeze(0)  # (1, hidden_channels)
            if self.training:
                soft_mask = utils.topk_gumbel_softmax(
                    k=self.content_channels,
                    logits=logits,
                    tau=1.0,
                    hard=True,
                )
            else:
                hard_mask = torch.zeros_like(logits)
                topk_idx = torch.topk(logits, self.content_channels, dim=1).indices
                hard_mask.scatter_(1, topk_idx, 1.0)
                soft_mask = hard_mask

            content_mask_bool = soft_mask.bool()
            content_idx = torch.where(content_mask_bool)[-1].tolist()
            style_idx = torch.where(~content_mask_bool)[-1].tolist()
            estimated_content_indices = [content_idx]

            # Extract style spatial for decoder injection
            if self.inject_style_to_decoder and return_recon:
                style_spatial = enc_out_l0[:, style_idx, :, :, :]

            # Soft-masked content channels (gradient flows through soft_mask)
            masked_l0 = enc_out_l0 * soft_mask.view(1, -1, 1, 1, 1)
            content_enc_out = masked_l0[:, content_idx, :, :, :]
            del masked_l0

        # --- Codebook + decoder loop (top-down: coarsest → finest) ---
        for l in range(self.nb_levels - 1, -1, -1):
            codebook = self.codebooks[l]

            enc_out = encoder_outputs[l]
            encoder_outputs[l] = None  # free memory

            if l == 0 and content_enc_out is not None:
                # Level 0 with content masking: codebook sees only content channels
                enc_for_codebook = content_enc_out
                del content_enc_out
            else:
                enc_for_codebook = enc_out
            del enc_out

            expected_in = codebook.conv_in.in_channels

            if len(decoder_outputs) and return_recon:
                dec_out = decoder_outputs[-1]
                if dec_out.shape[2:] != enc_for_codebook.shape[2:]:
                    dec_out = F.interpolate(
                        dec_out,
                        size=enc_for_codebook.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                combined = torch.cat([enc_for_codebook, dec_out], dim=1)
                del enc_for_codebook
                code_q, code_d, emb_id = codebook(combined)
                del combined
            else:
                # Pad with zero conditioning channels if needed (e.g. pool_only
                # skips reconstruction so there's no decoder output to concat)
                if expected_in > enc_for_codebook.shape[1]:
                    cond_channels = expected_in - enc_for_codebook.shape[1]
                    zeros = torch.zeros(
                        enc_for_codebook.shape[0],
                        cond_channels,
                        *enc_for_codebook.shape[2:],
                        device=enc_for_codebook.device,
                        dtype=enc_for_codebook.dtype,
                    )
                    enc_for_codebook = torch.cat([enc_for_codebook, zeros], dim=1)
                code_q, code_d, emb_id = codebook(enc_for_codebook)
                del enc_for_codebook

            diffs.append(code_d)
            id_outputs.append(emb_id)

            if return_recon:
                decoder = self.decoders[l]
                upscaled_codes = []
                target_size = code_q.shape[2:]
                for i, c in enumerate(code_outputs):
                    upscaled = self.upscalers[i](c, upscale_counts[i])
                    if upscaled.shape[2:] != target_size:
                        upscaled = F.interpolate(
                            upscaled,
                            size=target_size,
                            mode="trilinear",
                            align_corners=False,
                        )
                    upscaled_codes.append(upscaled)
                code_outputs = upscaled_codes
                upscale_counts = [u + 1 for u in upscale_counts]

                decoder_in = torch.cat([code_q, *code_outputs], axis=1)
                if self.inject_style_to_decoder and style_spatial is not None:
                    decoder_outputs.append(decoder(decoder_in, style=style_spatial))
                else:
                    decoder_outputs.append(decoder(decoder_in))

                code_outputs.append(code_q)
                upscale_counts.append(0)

        if return_recon:
            final_output = decoder_outputs[-1]
        else:
            final_output = None
            decoder_outputs = []

        encoder_features = encoder_pools if pool_only else encoder_outputs

        return (
            final_output,
            diffs,
            encoder_features,
            estimated_content_indices,
            decoder_outputs,
            id_outputs,
        )

    def decode_codes(self, *cs, style=None):
        """Decode from discrete codes back to the input space.

        Args:
            *cs: Per-level codebook indices.
            style: Optional style tensor for decoder-0 when
                   ``inject_style_to_decoder`` is enabled.  If ``None`` and
                   style injection is configured, a zero tensor is used.
        """
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 4, 1, 2, 3)
            target_size = code_q.shape[2:]
            upscaled_codes = []
            for i, c in enumerate(code_outputs):
                upscaled = self.upscalers[i](c, upscale_counts[i])
                if upscaled.shape[2:] != target_size:
                    upscaled = F.interpolate(
                        upscaled,
                        size=target_size,
                        mode="trilinear",
                        align_corners=False,
                    )
                upscaled_codes.append(upscaled)
            code_outputs = upscaled_codes
            upscale_counts = [u + 1 for u in upscale_counts]

            decoder_in = torch.cat([code_q, *code_outputs], axis=1)
            if self.inject_style_to_decoder:
                if style is None:
                    # Provide a zero placeholder so the decoder's final_conv
                    # receives the expected number of channels.
                    dec_feat = decoder.layers(decoder_in)
                    style = torch.zeros(
                        dec_feat.shape[0],
                        self.style_channels,
                        *dec_feat.shape[2:],
                        device=dec_feat.device,
                        dtype=dec_feat.dtype,
                    )
                decoder_outputs.append(decoder(decoder_in, style=style))
            else:
                decoder_outputs.append(decoder(decoder_in))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]


class MoCoEncoder(nn.Module):
    """
    MoCo (Momentum Contrast) wrapper around a VQVAE model.

    Maintains:
    - An *online* VQVAE (updated normally by the optimizer).
    - A *momentum* copy of only the encoder stack (EMA-updated, no gradients).
    - One FIFO circular queue per encoder level filled with past momentum-key
      embeddings.  The queues act as cheap, decoupled negatives for the
      contrastive loss, so the effective negative count is ``queue_size``
      regardless of the per-step batch size.

    Args:
        vqvae_model (VQVAE): The online VQVAE instance (already on device).
        queue_size  (int):   Number of negatives stored per level queue.
        momentum    (float): EMA coefficient for the momentum encoder update.
                             Typical value: 0.999.
        nb_levels   (int):   Number of VQ-VAE encoder levels (must match
                             ``vqvae_model.nb_levels``).
    """

    def __init__(
        self,
        vqvae_model,
        queue_size: int = 4096,
        momentum: float = 0.999,
        nb_levels: int = 3,
    ):
        super().__init__()
        self.online = vqvae_model
        self.momentum = momentum
        self.queue_size = queue_size
        self.nb_levels = nb_levels

        # Unwrap DataParallel if present to access the underlying VQVAE attributes
        raw_vqvae = vqvae_model.module if hasattr(vqvae_model, "module") else vqvae_model

        # ---- Momentum encoder: a deep copy of the encoder stack only --------
        # We do NOT copy codebooks or decoders — keys are pre-quantisation
        # global-average-pooled feature vectors.
        self.momentum_encoders = nn.ModuleList([copy.deepcopy(enc) for enc in raw_vqvae.encoders])
        for enc in self.momentum_encoders:
            for p in enc.parameters():
                p.requires_grad = False

        # ---- Per-level queues -----------------------------------------------
        # All levels pool from hidden_channels (the mask is on encoder output,
        # not on the codebook projection).
        hidden_channels = self._infer_hidden_channels(raw_vqvae)

        for lvl in range(nb_levels):
            q = F.normalize(torch.randn(hidden_channels, queue_size), dim=0)
            self.register_buffer(f"queue_{lvl}", q)

        self.queue_ptrs = [0] * nb_levels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_hidden_channels(vqvae_model: VQVAE) -> int:
        """Return the hidden_channels width by inspecting the encoder stack."""
        enc0 = vqvae_model.encoders[0]
        # The last nn.Conv3d in Encoder.layers outputs hidden_channels
        for m in reversed(list(enc0.layers.modules())):
            if isinstance(m, nn.Conv3d):
                return m.out_channels
        raise RuntimeError("Could not infer hidden_channels from encoder.")

    def _get_queue(self, level: int) -> torch.Tensor:
        return getattr(self, f"queue_{level}")

    # ------------------------------------------------------------------
    # Momentum update (EMA)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _momentum_update(self):
        """EMA-update momentum encoders from the online model."""
        raw_vqvae = self.online.module if hasattr(self.online, "module") else self.online
        for online_enc, mom_enc in zip(raw_vqvae.encoders, self.momentum_encoders):
            for p_online, p_mom in zip(online_enc.parameters(), mom_enc.parameters()):
                p_mom.data.mul_(self.momentum).add_(p_online.data, alpha=1.0 - self.momentum)

    # ------------------------------------------------------------------
    # Key encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_keys(self, x: torch.Tensor):
        """
        Encode ``x`` with the momentum encoder stack and return per-level
        global-average-pooled feature vectors ``(B, hidden_channels)``.
        """
        key_pools = []
        enc_input = x
        for mom_enc in self.momentum_encoders:
            enc_input = mom_enc(enc_input)
            key_pools.append(enc_input.mean(dim=[2, 3, 4]))
        return key_pools

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def enqueue(self, keys: list):
        """
        Write new keys into the per-level circular queues.

        Args:
            keys (list[torch.Tensor]): One tensor per level, shape ``(B, C)``.
                                       Must be detached.
        """
        for lvl, k in enumerate(keys):
            queue = self._get_queue(lvl)  # (C, Q)
            batch = k.shape[0]
            ptr = self.queue_ptrs[lvl]

            # Wrap-around write
            if ptr + batch <= self.queue_size:
                queue[:, ptr : ptr + batch] = k.T  # noqa: E203
            else:
                # Split write across the boundary
                tail = self.queue_size - ptr
                queue[:, ptr:] = k[:tail].T
                queue[:, : batch - tail] = k[tail:].T

            self.queue_ptrs[lvl] = (ptr + batch) % self.queue_size

    # ------------------------------------------------------------------
    # Forward (delegates to online VQVAE)
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        """
        Forward pass through the *online* VQVAE, then EMA-update the momentum
        encoder.  The queue is updated externally (in the training loop) after
        ``encode_keys`` has been called, so that the snapshot passed to the loss
        function is consistent with the keys generated this step.

        Returns the same outputs as ``VQVAE.forward``.
        """
        outputs = self.online(x, **kwargs)
        self._momentum_update()
        return outputs

    # ------------------------------------------------------------------
    # Convenience: expose queue snapshots as a list
    # ------------------------------------------------------------------

    @property
    def queues(self):
        """Return all queues as an ordered list (level 0 first)."""
        return [self._get_queue(lvl) for lvl in range(self.nb_levels)]


if __name__ == "__main__":
    from utils.helper import get_parameter_count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with 3D brain MRI-like input
    nb_levels = 3
    # For 96x112x96 input with scaling_rates=[4, 2, 2]:
    # Level 0: 96->24, Level 1: 24->12, Level 2: 12->6
    net = VQVAE(
        in_channels=1,  # Grayscale MRI
        hidden_channels=64,  # Reduced for memory
        res_channels=32,
        nb_res_layers=2,
        nb_levels=nb_levels,
        embed_dim=32,  # Reduced for memory
        nb_entries=512,
        scaling_rates=[4, 2, 2],
    ).to(device)
    print(f"Number of trainable parameters: {get_parameter_count(net)}")

    # Test with 3D volume (B, C, D, H, W)
    x = torch.randn(1, 1, 96, 112, 96).to(device)
    print(f"Input shape: {x.shape}")

    recon, diffs, enc_out, est_content_idx, dec_out, id_out = net(x)
    print(f"\nReconstruction shape: {recon.shape}")
    print("\nEncoder outputs:")
    print("\n".join(f"  Level {i}: {y.shape}" for i, y in enumerate(enc_out)))
    print("\nDecoder outputs:")
    print("\n".join(f"  Level {i}: {y.shape}" for i, y in enumerate(dec_out)))
    print("\nVQ commitment losses (diffs):")
    print("\n".join(f"  Level {i}: {y:.6f}" for i, y in enumerate(diffs)))
    print("\nCode indices shapes:")
    print("\n".join(f"  Level {i}: {y.shape}" for i, y in enumerate(id_out)))
