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


class SpatialFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on a spatial style tensor.

    Given a content feature map ``feat`` of shape ``(B, C, D, H, W)`` and a
    style tensor ``style`` of shape ``(B, S, D', H', W')``, produce
    per-location scale (gamma) and shift (beta) to modulate the content:

        out = feat * (1 + gamma) + beta

    The style tensor is trilinearly upsampled to match the content spatial
    dimensions before the 1×1×1 projection, so the modulation is spatially
    varying — important for brain MRI where T1/T2 intensity differences are
    region-dependent.
    """

    def __init__(self, style_channels: int, feature_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(style_channels, feature_channels * 2, 1)
        # Initialise so that gamma≈0 and beta≈0, i.e. the modulation starts
        # as a near-identity and the decoder can learn without style initially.
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        feat: torch.FloatTensor,
        style: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if style.shape[2:] != feat.shape[2:]:
            style = F.interpolate(
                style,
                size=feat.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        gamma, beta = self.conv(style).chunk(2, dim=1)
        return feat * (1 + gamma) + beta


class Decoder(HelperModule):
    """3D Decoder with transposed convolutions for upsampling.

    Style injection modes (``style_injection_mode``):

    * ``"concat"`` (default / legacy) — style is concatenated onto the
      penultimate feature map before the final 3×3 conv.  Simple but style
      only influences the very last layer.
    * ``"film"`` — Spatial FiLM (Feature-wise Linear Modulation).  Style
      modulates the feature map via learned scale+shift **after each decoder
      stage** (initial conv, residual stack, and every upsampling step).
      This gives the decoder access to style information at every resolution.
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
        style_injection_mode: str = "concat",
    ):
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        assert style_injection_mode in ("concat", "film"), (
            f"Unknown style_injection_mode: {style_injection_mode!r}. " f"Expected 'concat' or 'film'."
        )
        upscale_steps = int(log2(upscale_factor))
        self.style_channels = style_channels
        self.style_injection_mode = style_injection_mode

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

        # Track the output channel count of each upsample stage for FiLM sizing.
        upsample_out_chs = []
        _c, _n = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            upsample_out_chs.append(_n)
            _c, _n = _n, out_channels

        if style_channels > 0 and style_injection_mode == "film":
            # FiLM mode: store each layer separately so we can apply
            # SpatialFiLM between them.  A FiLM layer is inserted after the
            # initial conv+residual block and after each upsampling step.
            self.layers = nn.ModuleList(layers)
            self.film_layers = nn.ModuleList()
            # After initial conv+res (channels = hidden_channels)
            self.film_layers.append(SpatialFiLM(style_channels, hidden_channels))
            # After each upsample step — must match the actual output channels
            for upsample_ch in upsample_out_chs:
                self.film_layers.append(SpatialFiLM(style_channels, upsample_ch))
            # Final projection (no style concat needed — FiLM already applied)
            self.output_conv = nn.Sequential(
                nn.Conv3d(c_channel, out_channels, 3, stride=1, padding=1),
                get_group_norm(out_channels),
            )
            self.final_conv = None  # not used in FiLM mode

        elif style_channels > 0 and style_injection_mode == "concat":
            # Legacy concat mode: style is concatenated at the end.
            self.layers = nn.Sequential(*layers)
            self.film_layers = None
            self.output_conv = None
            self.final_conv = nn.Sequential(
                nn.Conv3d(c_channel + style_channels, out_channels, 3, stride=1, padding=1),
                get_group_norm(out_channels),
            )
        else:
            # No style injection
            layers.append(nn.Conv3d(c_channel, out_channels, 3, stride=1, padding=1))
            layers.append(get_group_norm(out_channels))
            self.layers = nn.Sequential(*layers)
            self.film_layers = None
            self.output_conv = None
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
                   trilinearly upsampled to match spatial dims as needed.

        Returns:
            Decoded output ``(B, out_channels, D_out, H_out, W_out)``.
        """
        if self.film_layers is not None:
            # FiLM mode: run layers one-by-one with FiLM modulation between.
            # layers[0] = initial conv, layers[1] = residual stack,
            # layers[2:] = upsample stages.
            if style is None:
                raise ValueError("Decoder was built with style (FiLM) but no style tensor was provided.")
            feat = self.layers[0](x)  # initial conv
            feat = self.layers[1](feat)  # residual stack
            feat = self.film_layers[0](feat, style)  # FiLM after res block
            for i, upsample in enumerate(self.layers[2:]):
                feat = upsample(feat)
                feat = self.film_layers[i + 1](feat, style)  # FiLM after each upsample
            feat = self.output_conv(feat)
            return feat

        feat = self.layers(x)
        if self.final_conv is not None:
            # Concat mode
            if style is None:
                raise ValueError("Decoder was built with style (concat) but no style tensor was provided.")
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
    """3D Vector Quantization layer with EMA codebook updates and dead-entry reset."""

    def build(
        self,
        in_channels: int,
        embed_dim: int,
        nb_entries: int,
        reset_threshold: float = 1.0,
        reset_every: int = 100,
        ema_decay: float = 0.999,
    ):
        self.conv_in = nn.Conv3d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = ema_decay
        self.eps = 1e-5

        # Dead-entry reset hyperparams
        self.reset_threshold = reset_threshold  # cluster_size below this → dead
        self.reset_every = reset_every  # check every N forward passes

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("_fwd_count", torch.tensor(0, dtype=torch.long))

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

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            # ── Dead-entry reset ─────────────────────────────────────────
            # Periodically replace codebook entries that have near-zero
            # cluster_size with randomly sampled encoder outputs + noise.
            self._fwd_count += 1
            if self.reset_every > 0 and self._fwd_count % self.reset_every == 0:
                dead = self.cluster_size < self.reset_threshold  # (n_embed,)
                n_dead = dead.sum().item()
                if n_dead > 0:
                    # Sample replacements from the current batch of encoder outputs
                    n_flat = flatten.shape[0]
                    replace_idx = torch.randint(0, n_flat, (n_dead,), device=flatten.device)
                    new_embeds = flatten[replace_idx].T  # (dim, n_dead)
                    # Add small noise to break symmetry
                    new_embeds = new_embeds + torch.randn_like(new_embeds) * 0.01
                    self.embed.data[:, dead] = new_embeds
                    self.embed_avg.data[:, dead] = new_embeds
                    self.cluster_size.data[dead] = self.reset_threshold  # warm-start the EMA count

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
        content_style_levels: list[int] | None = None,  # Levels with learnable Gumbel mask
        content_ratios: list[float] | None = None,  # Per-level content ratio (fraction of hidden_channels)
        separate_encoders: bool = False,  # If True, create a second encoder stack for view 1
        mask_mode: str = "onthefly",  # "learned" or "onthefly"
        quantize_style: bool = False,  # If True, style channels get their own independent codebook per level
        style_embed_dim: int | None = None,  # Embedding dim for style codebooks (defaults to embed_dim)
        style_nb_entries: int | None = None,  # Codebook size for style codebooks (defaults to nb_entries)
        style_injection_mode: str = "concat",  # "concat" (legacy) or "film" (Spatial FiLM at every decoder stage)
        cb_ema_decay: float = 0.999,  # EMA momentum for codebook running averages
        cb_reset_every: int = 100,  # Reset dead codebook entries every N forwards (0 = disable)
        cb_reset_threshold: float = 1.0,  # EMA cluster_size below this → dead
    ):
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"
        self.nb_levels = nb_levels
        self.separate_encoders = separate_encoders
        self._cb_reset_every = cb_reset_every
        self._cb_reset_threshold = cb_reset_threshold

        def _make_encoder_stack():
            stack = nn.ModuleList(
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
            for _i, sr in enumerate(scaling_rates[1:]):
                stack.append(
                    Encoder(
                        hidden_channels,
                        hidden_channels,
                        res_channels,
                        nb_res_layers,
                        sr,
                        use_checkpoint,
                    )
                )
            return stack

        self.encoders = _make_encoder_stack()

        # Second (view-1) encoder stack: deep copy of view-0 so both start
        # in the same feature space.  This is critical for cross-view
        # contrastive learning — random init puts the two encoders in
        # incompatible subspaces where cosine similarity ≈ 0 for everything,
        # creating a flat loss landscape that prevents bootstrapping.
        if separate_encoders:
            self.encoders_v1 = copy.deepcopy(self.encoders)
        else:
            self.encoders_v1 = None

        # --- Per-level content/style separation ---
        # Each level in content_style_levels gets its own learnable Gumbel mask
        # (channel_logits) on the hidden_channels encoder output.
        if content_style_levels is None:
            content_style_levels = [0]
        self.content_style_levels = sorted(set(content_style_levels))

        # Per-level content_channels: each masked level can have a different
        # number of content channels.  Stored as a dict: level → int.
        # Also keep self.content_channels as the value for the first masked
        # level, for backward-compat (notebook, contrastive loss ratio, etc.).
        has_content_style = content_size > 0 and style_size > 0 and len(self.content_style_levels) > 0

        if has_content_style:
            total_size = content_size + style_size
            default_ratio = content_size / total_size

            # Build per-level ratios
            if content_ratios is not None:
                assert len(content_ratios) == len(self.content_style_levels), (
                    f"content_ratios has {len(content_ratios)} entries but "
                    f"content_style_levels has {len(self.content_style_levels)}"
                )
                level_ratios = {lvl: r for lvl, r in zip(self.content_style_levels, content_ratios)}
            else:
                level_ratios = {lvl: default_ratio for lvl in self.content_style_levels}

            self.content_channels_per_level = {
                lvl: max(1, round(r * hidden_channels)) for lvl, r in level_ratios.items()
            }
            # Backward compat: single value from first masked level
            self.content_channels = self.content_channels_per_level[self.content_style_levels[0]]

            # --- Mask logits source ---
            # "learned":  persistent nn.Parameter per level (optionally per view).
            # "onthefly": logits computed from average encoder activations each
            #             forward pass, shared across views.  Matches the original
            #             multiview-crl repo (Yao et al., 2024).
            self.mask_mode = mask_mode

            if mask_mode == "learned":
                self.channel_logits = nn.ParameterDict(
                    {str(lvl): nn.Parameter(torch.zeros(hidden_channels)) for lvl in self.content_style_levels}
                )
                # Per-view content selectors (Def 3.5, Yao et al.):
                # When separate_encoders is active, view-1 gets its own logits.
                if separate_encoders:
                    self.channel_logits_v1 = nn.ParameterDict(
                        {str(lvl): nn.Parameter(torch.zeros(hidden_channels)) for lvl in self.content_style_levels}
                    )
                else:
                    self.channel_logits_v1 = None
            elif mask_mode == "fixed":
                # Fixed channel assignment: first content_channels are content,
                # the rest are style.  No learnable mask parameters — the
                # contrastive loss drives the encoder to put shared information
                # in the content channels and modality-specific information in
                # the style channels.
                self.channel_logits = nn.ParameterDict()  # empty
                self.channel_logits_v1 = None
                # Pre-compute fixed masks per level (registered as buffers so
                # they move with .to(device) and appear in state_dict).
                for lvl in self.content_style_levels:
                    k_lvl = self.content_channels_per_level[lvl]
                    fixed_mask = torch.zeros(1, hidden_channels)
                    fixed_mask[0, :k_lvl] = 1.0
                    self.register_buffer(f"fixed_mask_{lvl}", fixed_mask)
            else:
                # "onthefly" — no learnable mask parameters; logits are derived
                # from encoder outputs at runtime.
                self.channel_logits = nn.ParameterDict()  # empty (no learnable params)
                self.channel_logits_v1 = None
        else:
            self.content_channels = None
            self.content_channels_per_level = {}
            self.channel_logits = nn.ParameterDict()  # empty
            self.mask_mode = mask_mode
            self.channel_logits_v1 = None

        has_any_mask = self.content_channels is not None and len(self.content_channels_per_level) > 0

        # --- Style codebooks (Option A: fully independent, no cross-level conditioning) ---
        self.quantize_style = quantize_style and has_any_mask and inject_style_to_decoder
        if self.quantize_style:
            _style_embed = style_embed_dim if style_embed_dim is not None else embed_dim
            _style_entries = style_nb_entries if style_nb_entries is not None else nb_entries
            self.style_embed_dim = _style_embed
            self.style_codebooks = nn.ModuleDict()
            for lvl in self.content_style_levels:
                sc = hidden_channels - self.content_channels_per_level[lvl]
                self.style_codebooks[str(lvl)] = CodeLayer(
                    sc,
                    _style_embed,
                    _style_entries,
                    reset_threshold=cb_reset_threshold,
                    reset_every=cb_reset_every,
                    ema_decay=cb_ema_decay,
                )
        else:
            self.style_embed_dim = None
            self.style_codebooks = nn.ModuleDict()

        # --- Contrastive projection head ---
        # MoCo v2 / SimCLR-style: projects pooled encoder features into a
        # space where the contrastive loss operates.  Keeps the encoder
        # representation richer by decoupling it from the contrastive
        # objective.  The projection head output is discarded at eval time.
        proj_dim = 128
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_channels, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        # Optional style injection: style channels from each masked level's
        # encoder output are fed into the corresponding decoder.
        # style_channels_per_level: level → int (complement of content_channels).
        self.inject_style_to_decoder = inject_style_to_decoder and has_any_mask
        if self.inject_style_to_decoder:
            self.style_channels_per_level = {
                lvl: hidden_channels - cc for lvl, cc in self.content_channels_per_level.items()
            }
            # Backward compat
            self.style_channels = self.style_channels_per_level.get(self.content_style_levels[0], 0)
        else:
            self.style_channels_per_level = {}
            self.style_channels = 0

        # --- Codebooks ---
        # Each level's codebook input channels depend on whether the level is
        # masked (content_channels for that level) or not (hidden_channels),
        # and whether it has decoder conditioning (+embed_dim) from above.
        self.codebooks = nn.ModuleList()
        for lvl in range(nb_levels):
            if has_any_mask and lvl in self.content_channels_per_level:
                enc_ch = self.content_channels_per_level[lvl]
            else:
                enc_ch = hidden_channels

            if lvl == nb_levels - 1:
                # Coarsest level: no decoder conditioning from above
                self.codebooks.append(
                    CodeLayer(
                        enc_ch,
                        embed_dim,
                        nb_entries,
                        reset_threshold=cb_reset_threshold,
                        reset_every=cb_reset_every,
                        ema_decay=cb_ema_decay,
                    )
                )
            else:
                # Has decoder conditioning from the level above
                self.codebooks.append(
                    CodeLayer(
                        enc_ch + embed_dim,
                        embed_dim,
                        nb_entries,
                        reset_threshold=cb_reset_threshold,
                        reset_every=cb_reset_every,
                        ema_decay=cb_ema_decay,
                    )
                )

        # Decoders: each decoder at a masked level receives its own style_channels.
        self.style_injection_mode = style_injection_mode
        self.decoders = nn.ModuleList()
        for lvl in range(nb_levels):
            if lvl == 0:
                dec_in = embed_dim * nb_levels
                dec_out = in_channels
            else:
                dec_in = embed_dim * (nb_levels - lvl)
                dec_out = embed_dim
            if self.quantize_style and lvl in self.style_channels_per_level:
                sc = self.style_embed_dim
            elif self.inject_style_to_decoder:
                sc = self.style_channels_per_level.get(lvl, 0)
            else:
                sc = 0
            self.decoders.append(
                Decoder(
                    dec_in,
                    hidden_channels,
                    dec_out,
                    res_channels,
                    nb_res_layers,
                    scaling_rates[lvl],
                    use_checkpoint,
                    style_channels=sc,
                    style_injection_mode=style_injection_mode if sc > 0 else "concat",
                )
            )

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            rates = scaling_rates[1 : len(scaling_rates) - i][::-1]  # noqa: E203
            self.upscalers.append(Upscaler(embed_dim, rates))

    def forward(self, x, return_recon=True, pool_only=False, n_views=1, subsets=None, view_idx=None, patch_grid=None):
        """Forward pass through VQ-VAE-2.

        Args:
            x: Input tensor (B, C, D, H, W)
            return_recon: If False, skip decoder for memory efficiency (contrastive-only mode)
            pool_only: If True, return per-level pooled (B, C) vectors instead of spatial maps.
            n_views: Number of views. When 2 and separate_encoders is active, the
                     batch is split in half (view 0 | view 1) and routed through
                     the respective encoder stacks.
            subsets: View subsets (unused, kept for API compat).
            view_idx: When separate_encoders is active and n_views=1, selects which
                      encoder stack to use (0 → self.encoders, 1 → self.encoders_v1).
                      Defaults to 0 if not specified.  Ignored when n_views=2.
            patch_grid: Optional tuple (D, H, W) for patch-level pooling. When set
                        and pool_only=True, features are pooled to this spatial grid
                        instead of globally, returning (B, C, D*H*W) per level.

        Returns:
            final_output: Reconstruction (or None if return_recon=False)
            diffs: VQ commitment losses per level
            encoder_features: Per-level encoder features.
                              If pool_only=True and patch_grid=None:  list of (B, hidden_channels) pooled vectors
                              If pool_only=True and patch_grid set:   list of (B, hidden_channels, n_patches) tensors
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
        # Per-level differentiable Gumbel masks, returned for contrastive loss.
        # Keys are level indices (int); only populated for levels in content_style_levels.
        soft_content_masks = {}
        style_spatials = {}  # level → style channels for decoder injection

        # --- Content/style masks on encoder outputs (hidden_channels) ---
        # Applied BEFORE the codebook projection so the mask operates on raw
        # encoder channels, which tend to specialise and are more separable
        # than the compressed embed_dim projection.
        #
        # Two mask modes:
        #   "onthefly" — logits = avg activation per channel (shared across
        #                views, no learnable params).  Matches the original
        #                multiview-crl repo (Yao et al., 2024).
        #   "learned"  — logits from persistent nn.Parameter channel_logits
        #                (optionally per-view with channel_logits_v1).
        content_enc_outs = {}  # level → content-only spatial map
        has_any_mask = self.content_channels is not None and len(self.content_channels_per_level) > 0

        # Determine if per-view learned masks are active
        # (only for "learned" mode with separate encoders — "fixed" and
        # "onthefly" always use a single shared mask across views)
        use_per_view_masks = (
            has_any_mask
            and getattr(self, "mask_mode", "onthefly") == "learned"
            and self.separate_encoders
            and getattr(self, "channel_logits_v1", None) is not None
            and n_views == 2
        )

        # Helper function to apply mask to encoder output at a specific level
        def apply_mask_to_level(enc_out_lvl, lvl, use_per_view_masks):
            """Apply content/style mask to encoder output and return masked version and metadata.

            Returns:
                Per-view masks as a tuple (mask_v0, mask_v1) for per-view mode,
                or a single mask tensor for shared mode.  Returns None if the
                level is not in content_style_levels.
            """
            nonlocal estimated_content_indices
            k_lvl = self.content_channels_per_level.get(lvl, self.content_channels)

            if use_per_view_masks:
                # --- Per-view learned masks ---
                B = enc_out_lvl.shape[0] // 2
                enc_v0, enc_v1 = enc_out_lvl[:B], enc_out_lvl[B:]

                logits_v0 = self.channel_logits[str(lvl)].unsqueeze(0)
                logits_v1 = self.channel_logits_v1[str(lvl)].unsqueeze(0)

                if self.training:
                    mask_v0 = utils.topk_gumbel_softmax(k=k_lvl, logits=logits_v0, tau=1.0, hard=True)
                    mask_v1 = utils.topk_gumbel_softmax(k=k_lvl, logits=logits_v1, tau=1.0, hard=True)
                else:
                    hard_v0 = torch.zeros_like(logits_v0)
                    hard_v0.scatter_(1, torch.topk(logits_v0, k_lvl, dim=1).indices, 1.0)
                    mask_v0 = hard_v0
                    hard_v1 = torch.zeros_like(logits_v1)
                    hard_v1.scatter_(1, torch.topk(logits_v1, k_lvl, dim=1).indices, 1.0)
                    mask_v1 = hard_v1

                soft_content_masks[lvl] = (mask_v0, mask_v1)
                idx_v0 = torch.where(mask_v0.bool())[-1].tolist()
                idx_v1 = torch.where(mask_v1.bool())[-1].tolist()

                if estimated_content_indices is None:
                    estimated_content_indices = [idx_v0, idx_v1]

                if self.inject_style_to_decoder and return_recon:
                    style_idx_v0 = torch.where(~mask_v0.bool())[-1].tolist()
                    style_idx_v1 = torch.where(~mask_v1.bool())[-1].tolist()
                    style_spatials[lvl] = torch.cat(
                        [enc_v0[:, style_idx_v0, :, :, :], enc_v1[:, style_idx_v1, :, :, :]],
                        dim=0,
                    )

                masked_v0 = enc_v0 * mask_v0.view(1, -1, 1, 1, 1)
                masked_v1 = enc_v1 * mask_v1.view(1, -1, 1, 1, 1)
                content_enc_outs[lvl] = torch.cat(
                    [masked_v0[:, idx_v0, :, :, :], masked_v1[:, idx_v1, :, :, :]],
                    dim=0,
                )
                del masked_v0, masked_v1
                return (mask_v0, mask_v1)

            else:
                # --- Shared mask (fixed / on-the-fly / single learned) ---
                _mode = getattr(self, "mask_mode", "onthefly")

                if _mode == "fixed":
                    # Deterministic: first k channels = content, rest = style.
                    # No Gumbel noise, no learnable params.
                    soft_mask = getattr(self, f"fixed_mask_{lvl}").to(enc_out_lvl.device)
                elif _mode == "onthefly":
                    # On-the-fly: logits = mean activation per channel,
                    # averaged across batch (and all views if concatenated).
                    logits = enc_out_lvl.mean(dim=[0, 2, 3, 4]).unsqueeze(0)  # (1, C)
                else:
                    # Learned: logits from persistent nn.Parameter.
                    # In single-view mode (n_views=1) with per-view masks,
                    # select the correct logits based on view_idx.
                    _has_v1_logits = getattr(self, "channel_logits_v1", None) is not None
                    if self.separate_encoders and _has_v1_logits and view_idx == 1:
                        logits = self.channel_logits_v1[str(lvl)].unsqueeze(0)  # (1, C)
                    else:
                        logits = self.channel_logits[str(lvl)].unsqueeze(0)  # (1, C)

                if _mode != "fixed":
                    if self.training:
                        soft_mask = utils.topk_gumbel_softmax(
                            k=k_lvl,
                            logits=logits,
                            tau=1.0,
                            hard=True,
                        )
                    else:
                        hard_mask = torch.zeros_like(logits)
                        topk_idx = torch.topk(logits, k_lvl, dim=1).indices
                        hard_mask.scatter_(1, topk_idx, 1.0)
                        soft_mask = hard_mask

                soft_content_masks[lvl] = soft_mask
                content_mask_bool = soft_mask.bool()
                content_idx = torch.where(content_mask_bool)[-1].tolist()

                if estimated_content_indices is None:
                    estimated_content_indices = [content_idx]

                if self.inject_style_to_decoder and return_recon:
                    style_idx = torch.where(~content_mask_bool)[-1].tolist()
                    style_spatials[lvl] = enc_out_lvl[:, style_idx, :, :, :]

                masked = enc_out_lvl * soft_mask.view(1, -1, 1, 1, 1)
                content_enc_outs[lvl] = masked[:, content_idx, :, :, :]
                del masked
                return soft_mask

        # Encoder forward pass
        if self.separate_encoders and self.encoders_v1 is not None and n_views == 2:
            # View-specific encoders: split batch, encode through separate stacks,
            # then re-concatenate so the rest of the forward pass is unchanged.
            B = x.shape[0] // 2
            x_v0, x_v1 = x[:B], x[B:]
            del x

            enc_in_v0 = x_v0
            enc_in_v1 = x_v1
            del x_v0, x_v1
            for i, (enc_v0, enc_v1) in enumerate(zip(self.encoders, self.encoders_v1)):
                enc_in_v0 = enc_v0(enc_in_v0)
                enc_in_v1 = enc_v1(enc_in_v1)
                enc_out = torch.cat([enc_in_v0, enc_in_v1], dim=0)

                encoder_outputs.append(enc_out)
                # Pool BEFORE masking so contrastive loss sees full channels
                # and can apply its own differentiable soft mask.
                if pool_only:
                    if patch_grid is not None:
                        # Patch-level pooling: (B, C, D, H, W) → (B, C, Pd*Ph*Pw)
                        pool_v0 = F.adaptive_avg_pool3d(enc_in_v0, patch_grid).flatten(2)
                        pool_v1 = F.adaptive_avg_pool3d(enc_in_v1, patch_grid).flatten(2)
                    else:
                        pool_v0 = enc_in_v0.mean(dim=[2, 3, 4])
                        pool_v1 = enc_in_v1.mean(dim=[2, 3, 4])
                    encoder_pools.append(torch.cat([pool_v0, pool_v1], dim=0))

                # Apply mask and zero out style channels so the next encoder
                # only sees content. Keeps tensor shape at hidden_channels.
                if i in self.content_style_levels:
                    masks = apply_mask_to_level(enc_out, i, use_per_view_masks)
                    if isinstance(masks, tuple):
                        mask_v0, mask_v1 = masks
                        enc_in_v0 = enc_in_v0 * mask_v0.view(1, -1, 1, 1, 1)
                        enc_in_v1 = enc_in_v1 * mask_v1.view(1, -1, 1, 1, 1)
                    else:
                        enc_in_v0 = enc_in_v0 * masks.view(1, -1, 1, 1, 1)
                        enc_in_v1 = enc_in_v1 * masks.view(1, -1, 1, 1, 1)
            del enc_in_v0, enc_in_v1
        else:
            # Single-view path.  When separate_encoders is active, view_idx
            # selects which encoder stack to use (0 or 1).
            enc_stack = self.encoders
            if self.separate_encoders and self.encoders_v1 is not None and view_idx == 1:
                enc_stack = self.encoders_v1

            enc_input = x
            for i, enc in enumerate(enc_stack):
                enc_input = enc(enc_input)

                encoder_outputs.append(enc_input)
                # Pool BEFORE masking (same reason as dual-view path).
                if pool_only:
                    if patch_grid is not None:
                        encoder_pools.append(F.adaptive_avg_pool3d(enc_input, patch_grid).flatten(2))
                    else:
                        encoder_pools.append(enc_input.mean(dim=[2, 3, 4]))

                # Apply mask and zero out style for next encoder.
                if i in self.content_style_levels:
                    masks = apply_mask_to_level(enc_input, i, use_per_view_masks)
                    if masks is not None:
                        if isinstance(masks, tuple):
                            enc_input = enc_input * masks[0].view(1, -1, 1, 1, 1)
                        else:
                            enc_input = enc_input * masks.view(1, -1, 1, 1, 1)
            del x, enc_input

        # --- Codebook + decoder loop (top-down: coarsest → finest) ---
        # When we only need pooled encoder features (contrastive-only), skip the
        # codebook entirely — its commitment loss conflicts with contrastive learning
        # and the zero-padded input produces meaningless gradients.
        skip_codebook = pool_only and not return_recon

        # --- Style quantization (independent codebooks, no cross-level conditioning) ---
        style_id_outputs = {}
        if self.quantize_style and return_recon and not skip_codebook:
            for lvl in self.content_style_levels:
                if lvl in style_spatials:
                    style_cb = self.style_codebooks[str(lvl)]
                    style_q, style_d, style_emb_id = style_cb(style_spatials[lvl])
                    style_spatials[lvl] = style_q  # replace raw with quantized
                    diffs.append(style_d)
                    style_id_outputs[lvl] = style_emb_id
        for l in range(self.nb_levels - 1, -1, -1):
            codebook = self.codebooks[l]

            enc_out = encoder_outputs[l]
            encoder_outputs[l] = None  # free memory

            if l in content_enc_outs:
                # This level has content masking: codebook sees only content channels
                enc_for_codebook = content_enc_outs.pop(l)
            else:
                enc_for_codebook = enc_out
            del enc_out

            if skip_codebook:
                # No codebook forward — avoids commitment loss that conflicts
                # with contrastive learning and saves compute.
                diffs.append(torch.zeros(1, device=enc_for_codebook.device))
                id_outputs.append(None)
                del enc_for_codebook
                continue

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
                if self.inject_style_to_decoder and l in style_spatials:
                    decoder_outputs.append(decoder(decoder_in, style=style_spatials[l]))
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
            soft_content_masks,
            style_id_outputs,
        )

    def decode_codes(self, *cs, style=None, styles=None, style_codes=None):
        """Decode from discrete codes back to the input space.

        Args:
            *cs: Per-level codebook indices.
            style: Optional style tensor for decoder-0 (legacy single-level API).
                   If ``None`` and style injection is configured, zeros are used.
            styles: Optional dict mapping level index → style tensor for that
                    decoder.  Takes precedence over ``style`` when both are given.
                    Pass ``{0: style_l0, 1: style_l1, ...}`` for per-level style.
            style_codes: Optional dict mapping level index → style codebook indices
                         (LongTensor).  Only used when ``quantize_style`` is active.
                         Decoded through the style codebook and passed to the decoder.
                         Takes precedence over ``styles`` for levels that have entries.
        """
        if styles is None:
            styles = {}
        if style_codes is None:
            style_codes = {}
        # Legacy compat: if only `style` is provided, use it for level 0
        if style is not None and 0 not in styles:
            styles[0] = style

        # Decode style codes through style codebooks
        if self.quantize_style:
            for lvl_str, style_cb in self.style_codebooks.items():
                lvl = int(lvl_str)
                if lvl in style_codes:
                    style_q = style_cb.embed_code(style_codes[lvl]).permute(0, 4, 1, 2, 3)
                    styles[lvl] = style_q

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
            if decoder.style_channels > 0:
                lvl_style = styles.get(l, None)
                if lvl_style is None:
                    # Provide a zero placeholder so the decoder's final_conv
                    # receives the expected number of channels.
                    dec_feat = decoder.layers(decoder_in)
                    lvl_style = torch.zeros(
                        dec_feat.shape[0],
                        decoder.style_channels,
                        *dec_feat.shape[2:],
                        device=dec_feat.device,
                        dtype=dec_feat.dtype,
                    )
                decoder_outputs.append(decoder(decoder_in, style=lvl_style))
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

        # Second momentum encoder stack for view-specific encoders
        if getattr(raw_vqvae, "separate_encoders", False) and raw_vqvae.encoders_v1 is not None:
            self.momentum_encoders_v1 = nn.ModuleList([copy.deepcopy(enc) for enc in raw_vqvae.encoders_v1])
            for enc in self.momentum_encoders_v1:
                for p in enc.parameters():
                    p.requires_grad = False
        else:
            self.momentum_encoders_v1 = None

        # ---- Per-level queues -----------------------------------------------
        # All levels pool from hidden_channels (the mask is on encoder output,
        # not on the codebook projection).
        hidden_channels = self._infer_hidden_channels(raw_vqvae)
        self._separate_queues = self.momentum_encoders_v1 is not None

        for lvl in range(nb_levels):
            q = F.normalize(torch.randn(hidden_channels, queue_size), dim=0)
            self.register_buffer(f"queue_{lvl}", q)

        # When using separate encoders, maintain per-view queues so that
        # negatives for view-0 queries come only from view-0 samples (and
        # vice versa).  Without this, the queue mixes features from both
        # encoders, giving trivially easy negatives and preventing
        # cross-view alignment.
        if self._separate_queues:
            for lvl in range(nb_levels):
                q_v1 = F.normalize(torch.randn(hidden_channels, queue_size), dim=0)
                self.register_buffer(f"queue_v1_{lvl}", q_v1)
            self.register_buffer("queue_v1_ptrs", torch.zeros(nb_levels, dtype=torch.long))

        # Registered buffer so queue pointers survive state_dict / .to(device)
        self.register_buffer("queue_ptrs", torch.zeros(nb_levels, dtype=torch.long))

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

    def _get_queue(self, level: int, view: int = 0) -> torch.Tensor:
        if view == 1 and self._separate_queues:
            return getattr(self, f"queue_v1_{level}")
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
        # EMA-update the second encoder stack if present
        if self.momentum_encoders_v1 is not None and raw_vqvae.encoders_v1 is not None:
            for online_enc, mom_enc in zip(raw_vqvae.encoders_v1, self.momentum_encoders_v1):
                for p_online, p_mom in zip(online_enc.parameters(), mom_enc.parameters()):
                    p_mom.data.mul_(self.momentum).add_(p_online.data, alpha=1.0 - self.momentum)

    # ------------------------------------------------------------------
    # Key encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_keys(self, x: torch.Tensor, n_views: int = 1, patch_grid=None):
        """
        Encode ``x`` with the momentum encoder stack and return per-level
        pooled feature vectors.

        When ``patch_grid`` is set (a tuple of 3 ints), uses adaptive average
        pooling to that grid instead of global average pooling, returning
        ``(B, hidden_channels, n_patches)`` per level.

        When ``separate_encoders`` is active and ``n_views == 2``, the first
        half of the batch is routed through the view-0 momentum encoders and
        the second half through the view-1 momentum encoders, then
        re-concatenated — mirroring the online encoder split.
        """

        def _pool(t):
            if patch_grid is not None:
                return F.adaptive_avg_pool3d(t, patch_grid).flatten(2)
            return t.mean(dim=[2, 3, 4])

        if self.momentum_encoders_v1 is not None and n_views == 2:
            B = x.shape[0] // 2
            key_pools = []
            enc_v0, enc_v1 = x[:B], x[B:]
            for mom_enc_v0, mom_enc_v1 in zip(self.momentum_encoders, self.momentum_encoders_v1):
                enc_v0 = mom_enc_v0(enc_v0)
                enc_v1 = mom_enc_v1(enc_v1)
                key_pools.append(torch.cat([_pool(enc_v0), _pool(enc_v1)], dim=0))
            return key_pools
        else:
            key_pools = []
            enc_input = x
            for mom_enc in self.momentum_encoders:
                enc_input = mom_enc(enc_input)
                key_pools.append(_pool(enc_input))
            return key_pools

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def enqueue(self, keys: list, n_views: int = 1):
        """
        Write new keys into the per-level circular queues.

        When ``n_views == 2`` and separate per-view queues are active, the
        first half of each key tensor is written to the view-0 queue and
        the second half to the view-1 queue.  Otherwise all keys go into
        a single shared queue (backward compatible).

        Args:
            keys (list[torch.Tensor]): One tensor per level, shape ``(B, C)``.
                                       Must be detached.
            n_views (int): Number of views concatenated in each key tensor.
        """
        for lvl, k in enumerate(keys):
            if self._separate_queues and n_views == 2:
                B = k.shape[0] // 2
                k_v0, k_v1 = k[:B], k[B:]
                self._enqueue_single(lvl, k_v0, view=0)
                self._enqueue_single(lvl, k_v1, view=1)
            else:
                self._enqueue_single(lvl, k, view=0)

    @torch.no_grad()
    def _enqueue_single(self, lvl: int, k: torch.Tensor, view: int = 0):
        """Write ``k`` into the queue for ``(lvl, view)``."""
        queue = self._get_queue(lvl, view)  # (C, Q)
        ptrs = self.queue_v1_ptrs if (view == 1 and self._separate_queues) else self.queue_ptrs
        batch = k.shape[0]
        ptr = int(ptrs[lvl].item())

        # Wrap-around write
        if ptr + batch <= self.queue_size:
            queue[:, ptr : ptr + batch] = k.T  # noqa: E203
        else:
            tail = self.queue_size - ptr
            queue[:, ptr:] = k[:tail].T
            queue[:, : batch - tail] = k[tail:].T

        ptrs[lvl] = (ptr + batch) % self.queue_size

    # ------------------------------------------------------------------
    # Forward (delegates to online VQVAE)
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        """
        Forward pass through the *online* VQVAE.  The momentum encoder is NOT
        updated here — call ``momentum_update()`` explicitly after the backward
        pass so that the momentum encoder stays consistent with the keys used
        in the contrastive loss this step.

        Returns the same outputs as ``VQVAE.forward``.
        """
        return self.online(x, **kwargs)

    def momentum_update(self):
        """Public wrapper: EMA-update momentum encoders from the online model.

        Should be called once per training step, **after** ``optimizer.step()``.
        """
        self._momentum_update()

    # ------------------------------------------------------------------
    # Convenience: expose queue snapshots as a list
    # ------------------------------------------------------------------

    @property
    def queues(self):
        """Return all view-0 (or shared) queues as an ordered list (level 0 first)."""
        return [self._get_queue(lvl, view=0) for lvl in range(self.nb_levels)]

    @property
    def queues_v1(self):
        """Return all view-1 queues (only available with separate encoders)."""
        if not self._separate_queues:
            return self.queues  # fallback: same as view-0
        return [self._get_queue(lvl, view=1) for lvl in range(self.nb_levels)]


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

    recon, diffs, enc_out, est_content_idx, dec_out, id_out, soft_masks = net(x)
    print(f"\nReconstruction shape: {recon.shape}")
    print("\nEncoder outputs:")
    print("\n".join(f"  Level {i}: {y.shape}" for i, y in enumerate(enc_out)))
    print("\nDecoder outputs:")
    print("\n".join(f"  Level {i}: {y.shape}" for i, y in enumerate(dec_out)))
    print("\nVQ commitment losses (diffs):")
    print("\n".join(f"  Level {i}: {y:.6f}" for i, y in enumerate(diffs)))
    print("\nCode indices shapes:")
    print("\n".join(f"  Level {i}: {y.shape}" for i, y in enumerate(id_out)))
