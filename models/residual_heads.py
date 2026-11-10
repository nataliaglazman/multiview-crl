"""Independent residual features and a joint alignment/uniformity objective.

The image encoder lives outside this module.  A fixed, training-fitted image
transform supplies a separate small encoder in each view; no image-encoder
features or parameters enter that path.  The exported blocks are aligned
separately, while a shared auxiliary projector tests their *joint* distribution.
The finite-sample kernel objective is an experiment, not an identifiability proof.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.vqvae import ChannelLayerNorm3d


def joint_uniform_mmd(samples: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    """Biased, nonnegative multiscale RBF MMD² against joint uniform samples.

    All coordinates enter the same kernel.  Matching each coordinate's marginal
    distribution cannot satisfy this discrepancy when coordinates are redundant.
    Bandwidths scale with sqrt(dimension); the positive mixture is characteristic.
    ``reference`` is useful for deterministic diagnostics and must match device,
    dtype and feature dimension.  Ordinary training draws a fresh reference.
    """
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("samples must be a nonempty (subjects, joint_dim) matrix")
    if reference is None:
        reference = torch.rand_like(samples)
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != samples.shape[1]:
        raise ValueError("reference must have the same joint dimension as samples")
    if reference.device != samples.device or reference.dtype != samples.dtype:
        raise ValueError("reference must share samples' device and dtype")

    # Matrix multiplication avoids cdist backward restrictions on some devices.
    def squared_distances(a, b):
        return (a.square().sum(1)[:, None] + b.square().sum(1)[None] - 2 * a @ b.T).clamp_min(0)

    xx = squared_distances(samples, samples)
    yy = squared_distances(reference, reference)
    xy = squared_distances(samples, reference)
    discrepancy = samples.new_zeros(())
    for bandwidth in (0.05, 0.1, 0.2, 0.5, 1.0):
        denominator = 2 * samples.shape[1] * bandwidth**2
        discrepancy = discrepancy + (
            torch.exp(-xx / denominator).mean()
            + torch.exp(-yy / denominator).mean()
            - 2 * torch.exp(-xy / denominator).mean()
        )
    # Its exact value is nonnegative; roundoff can otherwise produce tiny negatives.
    return (discrepancy / 5).clamp_min(0)


class ResidualHeads(nn.Module):
    """Training-fitted image residuals -> independent encoders -> residual code.

    ``fit`` accepts only the training images supplied by the caller, shaped
    (N, 2, size, size, size).  It fits a separate mean and PCA basis per view,
    a scalar residual SD per view, and the common interior-cell mask.  These
    buffers stay fixed afterwards.  Feature and code gauges are population
    running statistics updated only in training, with detached statistics.

    ``forward`` uses the repository's view-major layout: (T1 batch, FLAIR batch).
    A single-view call requires ``n_views=1, view_idx=0 or 1``.  It does not change
    running statistics in evaluation.  The shared readout has no batch norm.
    """

    THRESHOLDS = (2.0, 3.0, 4.0)

    def __init__(
        self,
        input_size: int,
        content_units: int = 4,
        hidden_channels: int = 16,
        head_hidden: int = 100,
        components: int = 16,
        *,
        joint_input_units: int,
        joint_dim: int,
        frozen_image: bool = True,
        momentum: float = 0.01,
        alignment_mode: str = "population",
        alignment_eps: float = 1e-12,
    ):
        super().__init__()
        if input_size < 4 or input_size % 4:
            raise ValueError("input_size must be a positive multiple of four")
        if min(content_units, hidden_channels, head_hidden, joint_dim) < 1:
            raise ValueError("content, hidden and joint dimensions must be positive")
        if joint_input_units <= content_units:
            raise ValueError("joint_input_units must include nonempty image and residual blocks")
        if components < 0 or not 0 < momentum <= 1:
            raise ValueError("components must be nonnegative and momentum must be in (0, 1]")
        if alignment_mode not in ("population", "batch"):
            raise ValueError("alignment_mode must be 'population' or 'batch'")
        if not 0 < alignment_eps < float("inf"):
            raise ValueError("alignment_eps must be finite and positive")
        self.input_size = int(input_size)
        self.content_units = int(content_units)
        self.hidden_channels = int(hidden_channels)
        self.components = int(components)
        self.joint_input_units = int(joint_input_units)
        self.joint_dim = int(joint_dim)
        self.frozen_image = bool(frozen_image)
        self.momentum = float(momentum)
        # Plain attributes preserve the exact legacy checkpoint buffer layout.
        self.alignment_mode = alignment_mode
        self.alignment_eps = float(alignment_eps)
        self.grid_size = input_size // 4
        cells = self.grid_size**3

        def encoder():
            return nn.Sequential(
                nn.Conv3d(1, hidden_channels, 3, stride=2, padding=1),
                ChannelLayerNorm3d(hidden_channels),
                nn.GELU(),
                nn.Conv3d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
                ChannelLayerNorm3d(hidden_channels),
                nn.GELU(),
            )

        self.encoders = nn.ModuleList([encoder(), encoder()])
        pooled_units = hidden_channels * (1 + 2 * len(self.THRESHOLDS))
        self.head = nn.Sequential(
            nn.Linear(pooled_units, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, content_units),
        )
        self.projector = nn.Sequential(
            nn.Linear(joint_input_units, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, joint_dim),
            nn.Sigmoid(),
        )

        spatial = (input_size,) * 3
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("fit_subjects", torch.zeros((), dtype=torch.long))
        self.register_buffer("brain", torch.zeros(2, 1, *spatial, dtype=torch.bool))
        self.register_buffer("mean", torch.zeros(2, 1, *spatial))
        self.register_buffer("basis", torch.zeros(2, components, *spatial))
        self.register_buffer("residual_sd", torch.ones(2))
        self.register_buffer("interior", torch.zeros(cells, dtype=torch.bool))
        self.register_buffer("tail_mean", torch.zeros(2, hidden_channels, cells))
        self.register_buffer("tail_var", torch.ones(2, hidden_channels, cells))
        self.register_buffer("tail_started", torch.zeros(2, dtype=torch.bool))
        self.register_buffer("code_mean", torch.zeros(2, joint_input_units))
        self.register_buffer("code_var", torch.ones(2, joint_input_units))
        self.register_buffer("code_started", torch.zeros(2, dtype=torch.bool))

    @staticmethod
    def _standardize(images):
        """Per-image brain mean/SD, including normalized background inside the population mask."""
        foreground = images != 0
        count = foreground.flatten(1).sum(1).clamp_min(1).to(images.dtype)
        mean = (images * foreground).flatten(1).sum(1) / count
        centred = images - mean[:, None, None, None, None]
        variance = (centred.square() * foreground).flatten(1).sum(1) / count
        return centred / variance.clamp_min(1e-8).sqrt()[:, None, None, None, None]

    @torch.no_grad()
    def fit(self, images: torch.Tensor) -> None:
        """Fit once on TRAINING images; never accepts labels or test/calibration images.

        PCA is computed on CPU float32 with a private RNG context.  The feature
        gauges are initialized from these same training subjects using the
        current residual encoders, in bounded batches.  No optimizer or RNG
        state is changed; no validation observations enter this operation.
        """
        expected = (2, self.input_size, self.input_size, self.input_size)
        if bool(self.fitted):
            raise RuntimeError("residual preprocessing is already fitted")
        if images.ndim != 5 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"expected training images (N, {', '.join(map(str, expected))})")
        if images.shape[0] <= self.components:
            raise ValueError(f"need more than {self.components} training subjects to fit PCA")
        images = images.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(images).all()):
            raise ValueError("training images must be finite")
        occupancy = (images != 0).float().mean(0)
        brain = occupancy >= 0.5
        interior = F.avg_pool3d(occupancy.amin(0)[None, None], kernel_size=4, stride=4).flatten() >= 0.9
        if not bool(interior.any()):
            raise ValueError("training images have no cells with at least 90% mean brain occupancy")
        if any(int(mask.sum()) <= self.components for mask in brain):
            raise ValueError("too few population-brain voxels for the requested PCA components")

        fitted_mean = torch.zeros_like(self.mean, device="cpu")
        fitted_basis = torch.zeros_like(self.basis, device="cpu")
        fitted_sd = torch.ones_like(self.residual_sd, device="cpu")
        with torch.random.fork_rng(devices=[]):
            # Seed CPU only; torch.manual_seed would also reset accelerator RNGs.
            torch.random.default_generator.manual_seed(0)
            for view in range(2):
                mask = brain[view]
                standardized = self._standardize(images[:, view : view + 1])[:, 0]
                rows = standardized[:, mask]
                mean = rows.mean(0)
                delta = rows - mean
                if self.components:
                    # Oversampling helps the leading directions; store only the requested components.
                    q = min(self.components + 8, rows.shape[0] - 1, rows.shape[1])
                    _, _, vectors = torch.pca_lowrank(delta, q=q, center=False, niter=4)
                    basis = vectors[:, : self.components].T.contiguous()
                    residual = delta - (delta @ basis.T) @ basis
                    fitted_basis[view, :, mask] = basis
                else:
                    residual = delta
                fitted_mean[view, 0, mask] = mean
                fitted_sd[view] = residual.square().mean().sqrt().clamp_min(1e-6)

        self.brain.copy_(brain[:, None])
        self.mean.copy_(fitted_mean)
        self.basis.copy_(fitted_basis)
        self.residual_sd.copy_(fitted_sd)
        self.interior.copy_(interior)
        self.fit_subjects.fill_(len(images))
        self.fitted.fill_(True)
        self._calibrate_tails(images)

    def _check_images(self, images):
        if not bool(self.fitted):
            raise RuntimeError("fit residual preprocessing on training images before forward")
        if images.ndim != 5 or images.shape[1] != 1 or tuple(images.shape[2:]) != (self.input_size,) * 3:
            raise ValueError(f"expected (subjects, 1, {self.input_size}, {self.input_size}, {self.input_size}) images")
        if images.shape[0] == 0:
            raise ValueError("image batch cannot be empty")

    @staticmethod
    def _parts(images, n_views, view_idx):
        if n_views == 2 and view_idx is None:
            if images.shape[0] % 2:
                raise ValueError("paired views need an even, view-major batch")
            return tuple(enumerate(images.chunk(2)))
        if n_views == 1 and view_idx in (0, 1):
            return ((view_idx, images),)
        raise ValueError("use paired n_views=2 or single n_views=1 with view_idx=0 or 1")

    def _residual_view(self, images, view):
        if self.frozen_image:
            images = images.detach()
        centred = (self._standardize(images) - self.mean[view]) * self.brain[view]
        flattened = centred.flatten(1)
        basis = self.basis[view].flatten(1)
        residual = flattened - (flattened @ basis.T) @ basis
        return residual.reshape_as(images) / self.residual_sd[view]

    def residual_images(self, images: torch.Tensor, n_views: int = 2, view_idx: int | None = None) -> torch.Tensor:
        """The fixed image-derived residual; useful for independent audits."""
        self._check_images(images)
        return torch.cat([self._residual_view(part, view) for view, part in self._parts(images, n_views, view_idx)])

    @torch.no_grad()
    def _calibrate_tails(self, images, batch_size=16):
        device = self.mean.device
        for view, encoder in enumerate(self.encoders):
            sums = torch.zeros_like(self.tail_mean[view], device="cpu", dtype=torch.float64)
            squares = torch.zeros_like(sums)
            for start in range(0, len(images), batch_size):
                part = images[start : start + batch_size, view : view + 1].to(device)
                rows = encoder(self._residual_view(part, view)).flatten(2).cpu().double()
                sums += rows.sum(0)
                squares += rows.square().sum(0)
            mean = sums / len(images)
            variance = (squares / len(images) - mean.square()).clamp_min(1e-6)
            self.tail_mean[view].copy_(mean)
            self.tail_var[view].copy_(variance)
            self.tail_started[view] = True

    @torch.no_grad()
    def _update_stats(self, rows, mean, variance, started):
        detached = rows.detach()
        batch_mean = detached.mean(0)
        batch_var = detached.var(0, unbiased=False)
        if not bool(started):
            mean.copy_(batch_mean)
            variance.copy_(batch_var.clamp_min(1e-6))
            started.fill_(True)
        else:
            change = batch_mean - mean
            variance.mul_(1 - self.momentum).add_(batch_var, alpha=self.momentum)
            variance.add_(change.square(), alpha=self.momentum * (1 - self.momentum)).clamp_min_(1e-6)
            mean.add_(change, alpha=self.momentum)

    def forward(self, images: torch.Tensor, n_views: int = 2, view_idx: int | None = None) -> torch.Tensor:
        self._check_images(images)
        outputs = []
        for view, part in self._parts(images, n_views, view_idx):
            feature = self.encoders[view](self._residual_view(part, view))
            rows = feature.flatten(2)
            if self.training:
                self._update_stats(
                    rows,
                    self.tail_mean[view],
                    self.tail_var[view],
                    self.tail_started[view],
                )
            # Cloning guards backward if a later microbatch updates the running buffers.
            mean = self.tail_mean[view].detach().clone()
            sd = self.tail_var[view].detach().clone().clamp_min(1e-6).sqrt()
            z = ((rows - mean) / sd)[:, :, self.interior]
            pooled = [feature.mean((2, 3, 4))]
            for threshold in self.THRESHOLDS:
                positive = F.softplus(5 * (z - threshold)) / 5
                negative = F.softplus(5 * (-z - threshold)) / 5
                pooled.extend(
                    [
                        torch.log1p(100 * positive.mean(2)),
                        torch.log1p(100 * negative.mean(2)),
                    ]
                )
            outputs.append(self.head(torch.cat(pooled, 1)))
        return torch.cat(outputs)

    def _joint_codes(self, raw_content, residual_content):
        if raw_content.ndim != 2 or residual_content.ndim != 2 or len(raw_content) != len(residual_content):
            raise ValueError("image and residual codes must be equally sized matrices")
        if residual_content.shape[1] != self.content_units:
            raise ValueError("residual code width differs from content_units")
        if raw_content.shape[1] + residual_content.shape[1] != self.joint_input_units:
            raise ValueError("combined code width differs from joint_input_units")
        if len(raw_content) == 0 or len(raw_content) % 2:
            raise ValueError("joint loss needs an even nonempty, view-major paired batch")
        return torch.cat([raw_content, residual_content], 1)

    @torch.no_grad()
    def calibrate_codes(self, raw_content: torch.Tensor, residual_content: torch.Tensor) -> None:
        """Initialize code population gauges from paired TRAINING codes before step-0 evaluation.

        This explicit calibration resets the gauge to the supplied population;
        call it once using the fitted training cohort, rather than each batch.
        """
        joined = self._joint_codes(raw_content, residual_content)
        for view, part in enumerate(joined.chunk(2)):
            self.code_mean[view].copy_(part.mean(0))
            self.code_var[view].copy_(part.var(0, unbiased=False).clamp_min(1e-6))
            self.code_started[view] = True

    def joint_loss(
        self,
        raw_content: torch.Tensor,
        residual_content: torch.Tensor,
        *,
        raw_weight: float = 1.0,
        residual_weight: float = 1.0,
        uniform_weight: float = 1.0,
    ):
        """Separate alignment plus joint MMD uniformity, with gradients to both blocks.

        The legacy ``population`` alignment uses detached running gauges.
        ``batch`` alignment differentiates current-batch mean and variance,
        removing the spurious common-scale contraction when variance is large
        relative to ``alignment_eps``. This centered correlation objective does
        not constrain per-view means/scales and is not a full anti-collapse
        guarantee: constant codes are stationary and epsilon matters near zero.

        The projector always uses the population gauges, so its per-subject
        input remains independent of evaluation batch companions. Evaluation
        changes no fitted/running state; the MMD reference remains stochastic.
        """
        if min(raw_weight, residual_weight, uniform_weight) < 0:
            raise ValueError("joint loss weights must be nonnegative")
        joined = self._joint_codes(raw_content, residual_content)
        if self.alignment_mode == "batch" and len(joined) < 4:
            raise ValueError("batch alignment needs at least two subjects per view")
        standardized = []
        for view, part in enumerate(joined.chunk(2)):
            if self.training:
                self._update_stats(
                    part,
                    self.code_mean[view],
                    self.code_var[view],
                    self.code_started[view],
                )
            mean = self.code_mean[view].detach().clone()
            sd = self.code_var[view].detach().clone().clamp_min(1e-6).sqrt()
            standardized.append((part - mean) / sd)
        if self.alignment_mode == "batch":
            alignment = []
            for part in joined.chunk(2):
                centered = part - part.mean(0)
                alignment.append(centered / (centered.square().mean(0) + self.alignment_eps).sqrt())
        else:
            alignment = standardized
        first, second = alignment
        raw_units = self.joint_input_units - self.content_units
        image_alignment = (first[:, :raw_units] - second[:, :raw_units]).square().mean()
        residual_alignment = (first[:, raw_units:] - second[:, raw_units:]).square().mean()
        projected = self.projector(torch.cat(standardized))
        # Estimate each view's joint law separately.  Paired duplicates are not
        # mistaken for independent extra subjects in a pooled reference bank.
        uniformity = sum(joint_uniform_mmd(part) for part in projected.chunk(2)) / 2
        total = raw_weight * image_alignment + residual_weight * residual_alignment + uniform_weight * uniformity
        diagnostics = self._diagnostics(joined, standardized, projected)
        terms = {
            "image_alignment": image_alignment,
            "residual_alignment": residual_alignment,
            "uniformity": uniformity,
            "total": total,
            **diagnostics,
        }
        return total, terms

    @staticmethod
    def _moments(rows, legacy_floor=False):
        centered = rows.detach() - rows.detach().mean(0)
        covariance = centered.T @ centered / len(centered)
        # Participation ratio is a variance-concentration diagnostic, not an
        # estimate of intrinsic manifold dimension or information sufficiency.
        floor = 1e-12 if legacy_floor else torch.finfo(covariance.dtype).tiny
        rank = covariance.trace().square() / covariance.square().sum().clamp_min(floor)
        return centered.square().mean(), rank

    @torch.no_grad()
    def _diagnostics(self, joined, standardized, projected):
        raw_units = self.joint_input_units - self.content_units
        values = torch.cat(standardized)
        joint_variance, joint_rank = self._moments(values, legacy_floor=True)

        def view_mean_moments(parts):
            moments = [self._moments(part) for part in parts]
            return tuple(torch.stack([pair[index] for pair in moments]).mean() for index in (0, 1))

        image_variance, image_rank = view_mean_moments([part[:, :raw_units] for part in standardized])
        residual_variance, residual_rank = view_mean_moments([part[:, raw_units:] for part in standardized])
        _, projected_rank = view_mean_moments(projected.chunk(2))
        raw_parts = joined.chunk(2)
        image_raw_variance = torch.stack(
            [part[:, :raw_units].var(0, unbiased=False).mean() for part in raw_parts]
        ).mean()
        residual_raw_variance = torch.stack(
            [part[:, raw_units:].var(0, unbiased=False).mean() for part in raw_parts]
        ).mean()
        return {
            # Retain existing joint metric definitions for old-run comparison.
            "joint_variance": joint_variance,
            "joint_effective_rank": joint_rank,
            "projected_variance": projected.detach().var(0, unbiased=False).mean(),
            # New block/rank metrics average the per-view values, so different
            # view means or directions cannot masquerade as subject diversity.
            "image_variance": image_variance,
            "residual_variance": residual_variance,
            # These two use the exported units before population standardization.
            "image_raw_variance": image_raw_variance,
            "residual_raw_variance": residual_raw_variance,
            "image_effective_rank": image_rank,
            "residual_effective_rank": residual_rank,
            "projected_effective_rank": projected_rank,
            # Raw exported-code SD reveals scale contraction hidden by a gauge.
            "code_sd_min": torch.stack([part.std(0, unbiased=False).min() for part in joined.chunk(2)]).min(),
            "projector_weight_norm": torch.stack(
                [parameter.detach().square().sum() for parameter in self.projector.parameters() if parameter.ndim > 1]
            )
            .sum()
            .sqrt(),
        }

    @torch.no_grad()
    def diagnostics(self, raw_content: torch.Tensor, residual_content: torch.Tensor):
        """Population-gauged moments on any supplied cohort, without state/RNG changes.

        A full held-out cohort gives held-out moments; passing a training batch
        gives batch diagnostics. Neither fits or updates the population gauge.
        """
        joined = self._joint_codes(raw_content, residual_content)
        standardized = [
            (part - self.code_mean[view]) / self.code_var[view].clamp_min(1e-6).sqrt()
            for view, part in enumerate(joined.chunk(2))
        ]
        projected = self.projector(torch.cat(standardized))
        return self._diagnostics(joined, standardized, projected)
