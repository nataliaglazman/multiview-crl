"""Encoder-only multi-view contrastive models for 3D volumes.

``conv`` (default) keeps the original VQ-VAE backbone and affine readout:
    3D strided conv + GroupNorm + ResidualStack -> 1x1 conv -> GAP.
``resnet18`` adapts the upstream image encoder architecture to volumes:
    3D ResNet-18 -> GAP -> Linear(512, 100) -> LeakyReLU -> Linear(100, latent_dim).
The hidden readout width is configurable. This is a 3D adaptation, not a change
to the training objective or the view-sharing policy. Both have no decoder.
Optional ablations add a pooled MLP to conv, replace ResNet BatchNorm with
GroupNorm, remove late ResNet strides, or replace the global GAP with attention
pooling (``models.attention_pool``). Defaults preserve existing checkpoints.

The first ``content_channels`` units are the content block, the rest are style.
An optional lesion branch (``models.keypoint_pool``) appends ``3 * lesion_keypoints``
keypoint coordinates AFTER the style units of the global code; the content mask marks
them as content, so the DCI probes score them with the content block.
``forward`` returns the same 8-tuple as ``VQVAE``/``MultiviewVAE`` so
``eval.metrics.dci.compute_dci_synthetic`` scores this model unchanged.
"""

import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.attention_pool import AttentionPool3d
from models.keypoint_pool import KeypointPool3d
from models.normative_residual import NormativeResidual
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
        conv_readout: str = "linear",
        resnet_norm: str = "batch",
        resnet_output_stride: int = 32,
        separate_spatial_readout: bool = False,
        global_pool: str = "gap",
        attention_pool_heads: int = 4,
        attention_pool_frequencies: int = 4,
        norm_type: str = "group",
        lesion_keypoints: int = 0,
        lesion_norm: str = "none",
        lesion_frame: str = "brain",
        lesion_proj_dim: int = 8,
        lesion_temperature: float = 1.0,
        lesion_input: str = "features",
        lesion_normative_components: int = 20,
        lesion_input_size: int = 0,
        lesion_head_init: str = "positive",
        lesion_detector: str = "shared",
        lesion_branch_frozen: bool = False,
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
        if conv_readout not in ("linear", "mlp"):
            raise ValueError("conv_readout must be linear or mlp")
        if encoder_architecture == "resnet18" and conv_readout != "linear":
            raise ValueError("conv_readout only applies to the conv architecture")
        if encoder_architecture == "conv" and (resnet_norm != "batch" or resnet_output_stride != 32):
            raise ValueError("resnet_norm and resnet_output_stride only apply to resnet18")
        self.conv_readout = conv_readout
        self.readout_type = "mlp" if encoder_architecture == "resnet18" else conv_readout
        self.separate_spatial_readout = separate_spatial_readout
        if separate_spatial_readout and self.readout_type != "mlp":
            raise ValueError("separate_spatial_readout requires an MLP readout")
        if separate_spatial_readout and proj_dim != 0:
            raise ValueError("separate_spatial_readout currently requires proj_dim=0 (no shared loss projector)")
        # "group": GroupNorm statistics pool over all positions of a sample, so they reach
        # background cells. "layer": per-voxel LayerNorm over channels, no spatial pooling.
        if norm_type not in ("group", "layer"):
            raise ValueError(f"norm_type must be group or layer, got {norm_type!r}")
        if encoder_architecture == "resnet18" and norm_type != "group":
            raise ValueError("norm_type applies to the conv architecture; use resnet_norm for resnet18")
        self.backbone_stride = resnet_output_stride if encoder_architecture == "resnet18" else downscale_factor
        self.normalization = resnet_norm if encoder_architecture == "resnet18" else norm_type
        if self.readout_type == "mlp" and encoder_head_hidden <= 0:
            raise ValueError("encoder_head_hidden must be positive")
        if global_pool not in ("gap", "attention"):
            raise ValueError(f"Unknown global_pool: {global_pool!r}")
        self.global_pool = global_pool
        if lesion_keypoints < 0:
            raise ValueError(f"lesion_keypoints must be nonnegative, got {lesion_keypoints}")
        if lesion_keypoints > 0 and lesion_proj_dim < 1:
            raise ValueError(f"lesion_proj_dim must be positive, got {lesion_proj_dim}")
        self.lesion_units = 3 * lesion_keypoints
        if lesion_input not in ("features", "residual"):
            raise ValueError(f"lesion_input must be features or residual, got {lesion_input!r}")
        if lesion_keypoints > 0 and lesion_input == "residual" and lesion_input_size < 1:
            raise ValueError("lesion_input='residual' needs lesion_input_size (the cubic input resolution)")
        self.lesion_input = lesion_input
        if lesion_detector not in ("shared", "separate", "separate_conv"):
            raise ValueError("lesion_detector must be shared, separate or separate_conv")
        if (lesion_detector != "shared" or lesion_branch_frozen) and (
            lesion_input != "residual" or lesion_keypoints < 1
        ):
            raise ValueError("Separate/frozen lesion detectors require a residual lesion branch")
        self.lesion_detector = lesion_detector
        self.lesion_branch_frozen = lesion_branch_frozen
        if lesion_head_init not in ("positive", "random", "negative"):
            raise ValueError(f"lesion_head_init must be positive, random or negative, got {lesion_head_init!r}")
        if lesion_head_init != "positive" and lesion_input != "residual":
            raise ValueError("lesion_head_init only applies to lesion_input='residual'")
        if lesion_detector == "separate_conv" and lesion_head_init != "positive":
            raise ValueError("lesion_head_init sign controls only apply to simple residual detectors")

        # --- View-specific encoders ---
        # Encoder 1 is a deep copy of encoder 0 so both views start in the same
        # feature space (random init puts the two views in incompatible subspaces
        # and flattens the cross-view contrastive landscape — see models/vqvae.py).
        if encoder_architecture == "conv":
            from models.vqvae import Encoder

            self.encoder = Encoder(
                in_channels,
                hidden_channels,
                res_channels,
                nb_res_layers,
                downscale_factor,
                use_checkpoint,
                norm_type=norm_type,
            )
        else:
            self.encoder = ResNet18Features3d(in_channels, norm=resnet_norm, output_stride=resnet_output_stride)
        self.encoder_v1 = copy.deepcopy(self.encoder) if separate_encoders else None

        # --- Projection head to the encoding space ---
        if self.readout_type == "linear":
            # Preserve the original module names, initialization order and weights.
            self.to_encoding = nn.Conv3d(hidden_channels, latent_dim, 1)
        else:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.to_encoding = nn.Sequential(
                nn.Linear(512 if encoder_architecture == "resnet18" else hidden_channels, encoder_head_hidden),
                nn.LeakyReLU(),
                nn.Linear(encoder_head_hidden, latent_dim),
            )

        # Untie the spatial content mapping without changing the global head or
        # consuming additional random draws. Both heads remain shared across views.
        self.spatial_readout = None
        if separate_spatial_readout:
            self.spatial_readout = copy.deepcopy(self.to_encoding)
            output = self.spatial_readout[-1]
            output.weight = nn.Parameter(output.weight[:content_channels].detach().clone())
            output.bias = nn.Parameter(output.bias[:content_channels].detach().clone())
            output.out_features = content_channels

        # --- Fixed content/style mask over the latent units ---
        # Lesion keypoint coordinates, if any, follow the style units and count as content.
        fixed_mask = torch.zeros(1, latent_dim + self.lesion_units)
        fixed_mask[0, :content_channels] = 1.0
        fixed_mask[0, latent_dim:] = 1.0
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

        # --- Global pooling ---
        # Built last and zero-initialized, so it draws no random numbers: every weight
        # above matches the GAP model from the same seed, and the pool starts as GAP.
        # Shared across views, like the readout it feeds.
        self.attention_pool = None
        if global_pool == "attention":
            self.attention_pool = AttentionPool3d(
                512 if encoder_architecture == "resnet18" else hidden_channels,
                num_heads=attention_pool_heads,
                num_frequencies=attention_pool_frequencies,
            )

        # --- Lesion branch ---
        # Built after everything else: it draws random numbers, so building it last leaves
        # every weight above, and the untrained floor of those units, equal to the model
        # without the branch from the same seed. Shared across views. Its InfoNCE runs on
        # lesion_projector's output while probes read the raw coordinates.
        # With lesion_input="residual" the heads read each view's residual against a low-rank
        # normative model (models.normative_residual) instead of the backbone map: brighter- and
        # darker-than-normal channels, pooled to the feature grid. This reduces dominant anatomy
        # variation, but residual anatomy can still capture a head. Separate detectors can learn
        # distinct T1/FLAIR appearances while reporting coordinates in the same frame.
        self.lesion_pool = None
        self.lesion_pool_v1 = None
        self.lesion_adapter = None
        self.lesion_adapter_v1 = None
        self.lesion_projector = None
        self.normative = None
        if lesion_keypoints > 0:
            if lesion_input == "residual":
                # Buffers only, no random draws; fit_normative() fills them before training.
                self.normative = NormativeResidual(2, lesion_normative_components, (lesion_input_size,) * 3)
            self.lesion_pool = KeypointPool3d(
                2 if lesion_input == "residual" else (512 if encoder_architecture == "resnet18" else hidden_channels),
                num_keypoints=lesion_keypoints,
                norm=lesion_norm,
                frame=lesion_frame,
                temperature=lesion_temperature,
            )
            if lesion_input == "residual" and lesion_head_init != "random":
                with torch.no_grad():
                    # positive: every head starts attending TO anomalies; random signs would point half
                    # away. negative: every head starts avoiding them, the control for whether training
                    # finds the lesion without that head start. Same draws either way, only the signs differ.
                    self.lesion_pool.logits.weight.abs_()
                    if lesion_head_init == "negative":
                        self.lesion_pool.logits.weight.neg_()
            self.lesion_projector = nn.Linear(self.lesion_units, lesion_proj_dim)
            # Additional capacity must not change the global/projector initialisation or
            # the RNG state seen by the data loader. Separate simple heads start identical
            # to the shared control, but receive independent modality-specific gradients.
            if lesion_detector == "separate_conv":
                with torch.random.fork_rng(devices=[]):
                    self.lesion_adapter = nn.Sequential(
                        nn.Conv3d(2, 16, 3, padding=1),
                        nn.GELU(),
                        nn.Conv3d(16, 16, 3, padding=1),
                        nn.GELU(),
                    )
                    self.lesion_pool = KeypointPool3d(
                        16, lesion_keypoints, lesion_norm, lesion_frame, lesion_temperature
                    )
                self.lesion_adapter_v1 = copy.deepcopy(self.lesion_adapter)
            if lesion_detector != "shared":
                self.lesion_pool_v1 = copy.deepcopy(self.lesion_pool)
            if lesion_branch_frozen:
                for part in self.lesion_modules():
                    part.requires_grad_(False)

    def lesion_modules(self):
        """Trainable parts of the branch, excluding the frozen normative buffers."""
        return tuple(
            part
            for part in (
                self.lesion_pool,
                self.lesion_pool_v1,
                self.lesion_adapter,
                self.lesion_adapter_v1,
                self.lesion_projector,
            )
            if part is not None
        )

    def _lesion_readout(self, inputs, support, n_views=1, view_idx=None, maps=False):
        def read(h, mask, view):
            pool = self.lesion_pool_v1 if view == 1 and self.lesion_pool_v1 is not None else self.lesion_pool
            adapter = self.lesion_adapter_v1 if view == 1 else self.lesion_adapter
            if adapter is not None:
                h = adapter(h)
            return pool.maps(h, mask) if maps else pool(h, mask)

        if self.lesion_pool_v1 is None:
            return read(inputs, support, 0)
        if n_views == 2:
            if len(inputs) % 2:
                raise ValueError("Two-view lesion batches must have even length")
            b = len(inputs) // 2
            return torch.cat(
                [
                    read(inputs[:b], None if support is None else support[:b], 0),
                    read(inputs[b:], None if support is None else support[b:], 1),
                ]
            )
        return read(inputs, support, 1 if view_idx == 1 else 0)

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

    @property
    def patch_readout(self):
        """The active spatial head; the legacy shared head when separation is off."""
        return self.to_encoding if self.spatial_readout is None else self.spatial_readout

    def _global_code(
        self, h: torch.FloatTensor, x: torch.FloatTensor = None, n_views: int = 1, view_idx=None
    ) -> torch.FloatTensor:
        """(B, latent_dim [+ lesion_units]) global encoding: pool the backbone map, then apply the global head.

        ``x`` is the input batch that produced ``h`` (view-major when ``n_views`` is 2); only the
        lesion branch reads it, for its brain frame and its residual input.
        """
        if self.attention_pool is None:
            # The original GAP expressions, so existing checkpoints replay bit-for-bit.
            if self.readout_type == "mlp":
                code = self.to_encoding(self.avgpool(h).flatten(1))
            else:
                code = self.to_encoding(h).mean(dim=[2, 3, 4])
        else:
            pooled = self.attention_pool(h)
            if self.readout_type == "mlp":
                code = self.to_encoding(pooled)
            else:
                code = self.to_encoding(pooled[:, :, None, None, None]).flatten(1)
        return self._append_lesion(code, h, x, n_views, view_idx)

    def _append_lesion(
        self, code: torch.FloatTensor, h: torch.FloatTensor, x: torch.FloatTensor, n_views: int = 1, view_idx=None
    ) -> torch.FloatTensor:
        """Concatenate the lesion branch's keypoint coordinates after the global code's units."""
        if self.lesion_pool is None:
            return code
        inputs = self._lesion_inputs(h, x, n_views, view_idx)
        return torch.cat([code, self._lesion_readout(inputs, self._lesion_support(h, x), n_views, view_idx)], dim=1)

    def _lesion_inputs(self, h: torch.FloatTensor, x: torch.FloatTensor, n_views: int = 1, view_idx=None):
        """What the keypoint heads read: the backbone map, or each view's normative residual.

        The residual is split into brighter- and darker-than-normal channels and average-pooled to
        the feature grid. With two views the first half of ``x`` is T1 (view 0) and the second FLAIR.
        """
        if self.lesion_input != "residual":
            return h
        if x is None:
            raise ValueError("The residual lesion input needs the input images")
        if n_views == 2:
            b = x.shape[0] // 2
            z = torch.cat([self.normative(x[:b], 0), self.normative(x[b:], 1)])
        else:
            z = self.normative(x, 1 if view_idx == 1 else 0)
        maps = torch.cat([z.clamp_min(0), (-z).clamp_min(0)], dim=1)
        return F.adaptive_avg_pool3d(maps, tuple(h.shape[2:]))

    def fit_normative(self, fit_images: torch.FloatTensor, calibration_images: torch.FloatTensor) -> None:
        """Fit the residual input's normative model from (N, 2, D, H, W) training images."""
        if self.normative is None:
            raise ValueError("This model's lesion branch does not read a normative residual")
        self.normative.fit(fit_images, calibration_images)

    def _lesion_support(self, h: torch.FloatTensor, x: torch.FloatTensor):
        """Brain occupancy on the feature grid for the brain frame; None for the grid frame.

        From the input's nonzero support: the synthetic and skull-stripped inputs are
        exactly zero outside the brain.
        """
        if self.lesion_pool.frame != "brain":
            return None
        if x is None:
            raise ValueError("The lesion branch's brain frame needs the input images")
        return F.adaptive_avg_pool3d((x != 0).to(h.dtype), tuple(h.shape[2:]))

    def _content_indices(self, width: int) -> List[int]:
        """Content units of an output with ``width`` units: the content block, then any lesion units.

        Integer arithmetic, not the mask, so no forward pays a device sync for it.
        """
        return list(range(min(self.content_channels, width))) + list(range(self.latent_dim, width))

    def project_lesion(self, lesion_block: torch.FloatTensor) -> torch.FloatTensor:
        """Map the lesion coordinates into their own loss-facing space (last dimension).

        Each coordinate is first standardized over every subject and view in the call. An
        untrained head spreads over most of the brain, so its coordinates differ between
        subjects by ~1e-2 or less, every projected code is nearly the same direction, and the
        branch's InfoNCE sits at chance with almost no gradient. Probes read the raw coordinates.
        """
        flat = lesion_block.reshape(-1, lesion_block.shape[-1])
        standardized = (lesion_block - flat.mean(0)) / flat.std(0).clamp_min(1e-6)
        return self.lesion_projector(standardized)

    def lesion_code(self, x: torch.FloatTensor, view_idx=None) -> torch.FloatTensor:
        """(B, lesion_units) keypoint coordinates of a single-view batch, through one view's encoder.

        For training the branch on pairs that the global code never sees, e.g. two augmented
        copies of the FLAIR view, both routed through the FLAIR encoder (``view_idx=1``).
        """
        if self.lesion_pool is None:
            raise ValueError("This model has no lesion branch")
        h = self._encode(x, 1, view_idx)
        return self._lesion_readout(self._lesion_inputs(h, x, 1, view_idx), self._lesion_support(h, x), 1, view_idx)

    def lesion_maps(self, x: torch.FloatTensor, n_views: int = 1, view_idx=None) -> torch.FloatTensor:
        """Lesion keypoint weights, (B, K, d, h, w); each head's map sums to 1."""
        if self.lesion_pool is None:
            raise ValueError("This model has no lesion branch")
        h = self._encode(x, n_views, view_idx)
        return self._lesion_readout(
            self._lesion_inputs(h, x, n_views, view_idx), self._lesion_support(h, x), n_views, view_idx, maps=True
        )

    def attention_maps(self, x: torch.FloatTensor, n_views: int = 1, view_idx=None) -> torch.FloatTensor:
        """Global attention weights, (B, heads, d, h, w); each head's map sums to 1.

        Views are routed as in ``forward``. A GAP model has no maps (its weights are uniform).
        """
        if self.attention_pool is None:
            raise ValueError("This model pools globally with GAP; it has no attention maps")
        return self.attention_pool.maps(self._encode(x, n_views, view_idx))

    def global_and_patch_features(self, x, patch_grid, n_views=2):
        """Two training readouts from ONE backbone pass, before any loss projector.

        Returns (2B, latent_dim) and (2B, spatial_channels, P). The global MLP
        receives globally pooled features (GAP or attention); the active patch MLP
        receives each averaged bin.
        The separate spatial head has content_channels outputs and no style units;
        the legacy shared head has latent_dim outputs.
        Do not average patch MLP outputs to approximate the global readout.
        """
        h = self._encode(x, n_views, view_idx=None)
        grid = tuple(patch_grid)
        if len(grid) != 3 or any(g < 1 or g > size for g, size in zip(grid, h.shape[2:])):
            raise ValueError(f"Training patch grid {grid} must fit spatial map {tuple(h.shape[2:])}")
        if self.readout_type == "mlp":
            pooled = self._global_code(h, x, n_views)
            patches = self.patch_readout(self._patch_pool(h, grid).transpose(1, 2)).transpose(1, 2)
        else:
            features = self.to_encoding(h)
            if self.attention_pool is None:
                pooled = self._append_lesion(features.mean(dim=[2, 3, 4]), h, x, n_views)
            else:
                pooled = self._global_code(h, x, n_views)
            patches = self._patch_pool(features, grid)
        return pooled, patches

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

        The global pool (GAP, or attention pooling) precedes the global head. Patch
        and unpooled outputs use patch_readout, after bin averaging or at each
        native position, respectively. With a separate spatial head these outputs
        contain only content_channels units. Explicit patch_grid=[1,1,1] also uses
        the spatial head and averages; patch_grid=None with pool_only=True always
        returns the global encoding.
        """
        h = self._encode(x, n_views, view_idx)
        if self.readout_type == "mlp":
            if pool_only and patch_grid is None:
                feat = self._global_code(h, x, n_views, view_idx)
            else:
                if pool_only:
                    grid = (
                        patch_grid[0]
                        if len(patch_grid) > 0 and isinstance(patch_grid[0], (list, tuple))
                        else patch_grid
                    )
                    if len(grid) != 3 or any(g < 1 or g > size for g, size in zip(grid, h.shape[2:])):
                        raise ValueError(
                            f"{self.encoder_architecture} patch grid {grid} must fit its spatial map "
                            f"{tuple(h.shape[2:])}; the backbone downsamples by {self.backbone_stride}."
                        )
                    h = F.adaptive_avg_pool3d(h, tuple(grid))
                feat = self.patch_readout(h.flatten(2).transpose(1, 2)).transpose(1, 2)
                if not pool_only:
                    feat = feat.reshape(h.shape[0], feat.shape[1], *h.shape[2:])
            mask = self.content_mask[:, : feat.shape[1]]
            return (
                None,
                [x.new_zeros(())],
                [feat],
                [self._content_indices(feat.shape[1])],
                [],
                [],
                {0: mask},
                {},
            )

        if pool_only and patch_grid is None:
            encoder_features: List[torch.Tensor] = [self._global_code(h, x, n_views, view_idx)]
        else:
            feat = self.to_encoding(h)  # (B, latent_dim, d, h, w)
            encoder_features = [self._patch_pool(feat, patch_grid) if pool_only else feat]

        # Only the global code carries lesion units; patch and unpooled maps have latent_dim.
        mask = self.content_mask[:, : encoder_features[0].shape[1]]
        soft_content_masks = {0: mask}
        estimated_content_indices = [self._content_indices(encoder_features[0].shape[1])]

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
