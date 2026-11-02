"""Predict the other modality's spatial residual from a global content vector.

Targets and population templates use images and brain masks only. The two
decoders receive only the content vectors: there is no spatial-feature bypass.
All batches are view-major (all T1 subjects, then the same FLAIR subjects).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

# target="quantile": where in each subject's sorted deviations the decoder reads, and which cells count.
QUANTILES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
INTERIOR = 0.9


def _grid_shape(grid):
    shape = tuple(grid)
    if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("grid must contain three positive integers")
    return shape


@torch.no_grad()
def _prepared_targets(images, foreground, grid, occupancy_threshold, eps):
    shape = _grid_shape(grid)
    if images.ndim != 5 or images.shape[1] != 1:
        raise ValueError("images must have shape (2B,1,D,H,W); only one image channel is supported")
    if images.shape[0] < 2 or images.shape[0] % 2:
        raise ValueError("images must contain a nonempty even, view-major batch")
    if foreground.shape != images.shape:
        raise ValueError("foreground must have the same shape as images")
    if any(g > size for g, size in zip(shape, images.shape[2:])):
        raise ValueError("target grid must fit the input image")
    if not math.isfinite(occupancy_threshold) or not 0 < occupancy_threshold <= 1:
        raise ValueError("occupancy_threshold must be in (0,1]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    images = images.detach().float()
    foreground = foreground.detach().to(device=images.device, dtype=images.dtype)
    if not torch.isfinite(images).all() or not torch.isfinite(foreground).all():
        raise ValueError("images and foreground must be finite")
    if (foreground < 0).any() or (foreground > 1).any():
        raise ValueError("foreground occupancy must lie in [0,1]")
    occupancy = F.adaptive_avg_pool3d(foreground, shape)
    keep = occupancy >= occupancy_threshold
    counts = keep.sum((1, 2, 3, 4), keepdim=True)
    if (counts == 0).any():
        raise ValueError("Every target image needs a nonempty foreground on the target grid")
    # Foreground cell means retain gain/bias invariance at partially occupied
    # boundaries. Background intensity never enters the target or its z-score.
    pooled = F.adaptive_avg_pool3d(images * foreground, shape) / occupancy.clamp_min(eps)
    mean = (pooled * keep).sum((1, 2, 3, 4), keepdim=True) / counts
    centered = pooled - mean
    variance = (centered.square() * keep).sum((1, 2, 3, 4), keepdim=True) / counts
    normalized = centered / variance.sqrt().clamp_min(eps)
    return normalized.masked_fill(~keep, 0), keep, occupancy


def prepare_vector_targets(images, foreground, grid, occupancy_threshold=0.5, eps=1e-6):
    """Detached per-image z-scores of foreground-pooled cells, and kept cells.

    The z-score mean and population SD use only cells with sufficient brain
    occupancy. A constant foreground becomes zero; empty foreground raises.
    """
    normalized, keep, _ = _prepared_targets(images, foreground, grid, occupancy_threshold, eps)
    return normalized, keep


def _quantiles(values):
    """(N, cells) -> (N, len(QUANTILES)), with numpy's linear interpolation."""
    ordered = values.sort(dim=1).values
    position = values.new_tensor(QUANTILES) * (values.shape[1] - 1)
    low, high = position.floor().long(), position.ceil().long()
    return torch.lerp(ordered[:, low], ordered[:, high], position - low)


def deviation_quantiles(predictor, residual):
    """(2B, cells) view-major template residuals -> (2B, Q): what is abnormal in each image, not where.

    Each view keeps its interior cells, loses its fitted population directions, and is sorted, so the same
    lesion gives the same row wherever it sits while a bigger one moves the row's tails.
    """
    rows = []
    for view, part in enumerate(residual.chunk(2)):
        interior = predictor.target_interior[view].to(part.device)
        basis = predictor.target_components[view].to(part)
        part = part * interior
        rows.append(_quantiles((part - part @ basis.T @ basis)[:, interior]))
    return torch.cat(rows)


class VectorPrediction3d(nn.Module):
    """Two vector-only decoders, indexed by the input modality (T1, FLAIR).

    Each decoder reads its view's units in a running population gauge, one fixed affine map of the
    subject's own vector. Raw units need it: the batch-standardized InfoNCE leaves their scale free, and
    they start with a between-subject SD near 0.008, which leaves the decoder a constant input and the
    encoder a gradient 36,000x below the global term's. Batch statistics would not do: anatomy-neighbourhood
    batches share their anatomy, so subtracting the batch mean removes what the decoder has to predict.

    ``target`` is what they decode. "spatial" is the other view's residual on ``grid``. "quantile" is
    ``QUANTILES`` of that residual after removing ``components`` population directions, standardized over
    the template's training subjects. A spatial target pays nothing for burden, because a lesion can sit
    anywhere (training/ENCODER_VECTOR_PREDICTION.md); this one drops the position.
    """

    def __init__(self, content_channels, grid, hidden=128, gauge_momentum=0.01, target="spatial", components=8):
        super().__init__()
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (content_channels, hidden, components)):
            raise ValueError("content_channels, hidden and components must be positive integers")
        if not 0 < gauge_momentum <= 1:
            raise ValueError("gauge_momentum must be in (0,1]")
        if target not in ("spatial", "quantile"):
            raise ValueError(f"target must be spatial or quantile, got {target!r}")
        self.content_channels = content_channels
        self.grid = _grid_shape(grid)
        self.hidden = hidden
        self.gauge_momentum = gauge_momentum
        self.target = target
        self.components = components
        cells = math.prod(self.grid)
        outputs = cells if target == "spatial" else len(QUANTILES)
        self.decoders = nn.ModuleList(
            nn.Sequential(nn.Linear(content_channels, hidden), nn.GELU(), nn.Linear(hidden, outputs)) for _ in range(2)
        )
        target_shape = (2, 1, *self.grid)
        self.register_buffer("template", torch.zeros(target_shape))
        self.register_buffer("template_sum", torch.zeros(target_shape))
        self.register_buffer("template_counts", torch.zeros(target_shape))
        self.register_buffer("template_subject_count", torch.zeros(2, dtype=torch.long))
        self.register_buffer("template_fitted", torch.tensor(False))
        self.register_buffer("input_mean", torch.zeros(2, content_channels))
        self.register_buffer("input_variance", torch.ones(2, content_channels))
        self.register_buffer("input_gauge_fitted", torch.tensor(False))
        self._fit_rows = []
        if target == "quantile":
            self.register_buffer("target_interior", torch.zeros(2, cells, dtype=torch.bool))
            self.register_buffer("target_components", torch.zeros(2, components, cells))
            self.register_buffer("target_mean", torch.zeros(2, len(QUANTILES)))
            self.register_buffer("target_sd", torch.ones(2, len(QUANTILES)))

    @torch.no_grad()
    def _update_gauge(self, views):
        """Running per-view mean and spread; the spread is about the running mean, so it spans batches."""
        mean, variance = views.mean(1), views.var(1, unbiased=False)
        if bool(self.input_gauge_fitted):
            self.input_mean.lerp_(mean, self.gauge_momentum)
            spread = (views - self.input_mean[:, None]).square().mean(1)
            self.input_variance.lerp_(spread, self.gauge_momentum)
        else:
            self.input_mean.copy_(mean)
            self.input_variance.copy_(variance)
            self.input_gauge_fitted.fill_(True)

    def standardized(self, content):
        """(2B,C) content in the view's running gauge; only training moves it, and stats carry no gradient.

        Before the first training step (the step-0 floor) there is no gauge yet, so the batch's own applies.
        """
        views = torch.stack(content.chunk(2, dim=0))
        if self.training:
            self._update_gauge(views.detach())
        if bool(self.input_gauge_fitted):
            mean, variance = self.input_mean[:, None], self.input_variance[:, None]
        else:
            mean, variance = views.detach().mean(1, True), views.detach().var(1, False, True)
        return ((views - mean) / variance.sqrt().clamp_min(1e-6)).flatten(0, 1)

    def forward(self, content):
        """(2B,C) input-view content vectors -> (2B,1,*grid) other-view predictions."""
        if content.ndim != 2 or content.shape[1] != self.content_channels:
            raise ValueError(f"Decoder inputs must be only (2B,{self.content_channels}) content vectors")
        if content.shape[0] < 2 or content.shape[0] % 2:
            raise ValueError("content must contain a nonempty even, view-major batch")
        first, second = self.standardized(content).chunk(2, dim=0)
        output = torch.cat((self.decoders[0](first), self.decoders[1](second)))
        return output if self.target == "quantile" else output.reshape(-1, 1, *self.grid)

    @torch.no_grad()
    def reset_template(self):
        """Begin a new fit using only the externally supplied training split."""
        self.template.zero_()
        self.template_sum.zero_()
        self.template_counts.zero_()
        self.template_subject_count.zero_()
        self.template_fitted.fill_(False)
        self._fit_rows = []

    @torch.no_grad()
    def accumulate_template(self, images, foreground, occupancy_threshold=0.5):
        """Accumulate training subjects; no encoder, labels or random draws are used."""
        if bool(self.template_fitted):
            raise ValueError("Template is already fitted; call reset_template before a new training fit")
        normalized, keep, occupancy = _prepared_targets(images, foreground, self.grid, occupancy_threshold, 1e-6)
        n = images.shape[0] // 2
        targets = normalized.reshape(2, n, 1, *self.grid)
        weights = (occupancy * keep).reshape(2, n, 1, *self.grid)
        self.template_sum.add_((targets * weights).sum(1).to(self.template_sum))
        self.template_counts.add_(weights.sum(1).to(self.template_counts))
        self.template_subject_count.add_(n)
        if self.target == "quantile":
            self._fit_rows.append((targets.flatten(2).cpu(), keep.reshape(2, n, -1).cpu()))

    @torch.no_grad()
    def finalize_template(self):
        """Finish a training population template; never-observed cells remain zero."""
        if (self.template_subject_count == 0).any():
            raise ValueError("Cannot fit the population template without training subjects in both views")
        self.template.copy_(self.template_sum / self.template_counts.clamp_min(1e-12))
        self.template_fitted.fill_(True)
        if self.target == "quantile":
            self._fit_quantile_target()

    def _fit_quantile_target(self):
        """Interior cells, population directions and quantile scale, from the template's own subjects only."""
        normalized = torch.cat([rows for rows, _ in self._fit_rows], dim=1)
        keep = torch.cat([rows for _, rows in self._fit_rows], dim=1)
        self._fit_rows = []
        residual = (normalized - self.template.flatten(1)[:, None].cpu()) * keep
        occupancy = (self.template_counts.flatten(1) / self.template_subject_count[:, None]).cpu()
        for view in range(2):
            interior = occupancy[view] >= INTERIOR
            if not interior.any():
                raise ValueError(f"No cell of grid {self.grid} is at least {INTERIOR:g} brain; use a finer grid")
            rows = residual[view][:, interior]
            if self.components >= min(rows.shape):
                raise ValueError(f"{self.components} components need more than {tuple(rows.shape)} subjects and cells")
            basis = torch.zeros(self.components, interior.numel())
            basis[:, interior] = torch.linalg.svd(rows, full_matrices=False).Vh[: self.components]
            self.target_components[view].copy_(basis)
            self.target_interior[view].copy_(interior)
        quantiles = deviation_quantiles(self, residual.flatten(0, 1)).unflatten(0, (2, -1))
        self.target_mean.copy_(quantiles.mean(1))
        self.target_sd.copy_(quantiles.std(1).clamp_min(1e-6))


def vector_prediction_loss(
    predictor, content, images, foreground, deviation_strength=1.0, weight_cap=4.0, occupancy_threshold=0.5
):
    """Cross-modal residual MSE through global content vectors only.

    Image targets, fitted templates, foreground and deviation weights are
    detached. Weights are capped before normalization to mean one over each
    target's kept cells. Every subject and direction then contributes equally.
    """
    if not math.isfinite(deviation_strength) or deviation_strength < 0:
        raise ValueError("deviation_strength must be finite and nonnegative")
    if not math.isfinite(weight_cap) or weight_cap < 1:
        raise ValueError("weight_cap must be finite and at least one")
    if not bool(predictor.template_fitted):
        raise ValueError("Fit the vector prediction population template on training subjects first")
    if content.shape[0] != images.shape[0]:
        raise ValueError("content and images must contain the same paired subjects")
    prediction = predictor(content)
    normalized, keep = prepare_vector_targets(images, foreground, predictor.grid, occupancy_threshold)
    n = len(prediction) // 2
    if predictor.target == "quantile":
        template = predictor.template.detach().to(normalized).repeat_interleave(n, dim=0)
        quantiles = deviation_quantiles(predictor, ((normalized - template) * keep).flatten(1))
        mean, sd = predictor.target_mean.repeat_interleave(n, 0), predictor.target_sd.repeat_interleave(n, 0)
        standardized = ((quantiles - mean) / sd).to(prediction)
        error = (prediction - torch.cat((standardized[n:], standardized[:n]))).square().mean(1)
        one = error.new_ones(())
        total = error.mean()
        return total, dict(
            mean=total,
            t1_to_flair=error[:n].mean(),
            flair_to_t1=error[n:].mean(),
            unweighted=total,
            unweighted_t1_to_flair=error[:n].mean(),
            unweighted_flair_to_t1=error[n:].mean(),
            weight_min=one,
            weight_max=one,
            foreground_fraction=predictor.target_interior.float().mean(),
        )
    # Templates are indexed by TARGET view, while decoders by INPUT view.
    template = predictor.template.detach().to(normalized)
    residual = normalized - template.repeat_interleave(n, dim=0)
    target = torch.cat((residual[n:], residual[:n])).to(prediction)
    keep = torch.cat((keep[n:], keep[:n])).to(device=prediction.device)
    with torch.no_grad():
        count = keep.sum((1, 2, 3, 4))
        weights = (1 + deviation_strength * target.abs()).clamp(max=weight_cap)
        mean_weight = (weights * keep).sum((1, 2, 3, 4)) / count
        weights = (weights / mean_weight[:, None, None, None, None]).masked_fill(~keep, 0)
    error = (prediction - target).square()
    weighted = (error * weights).sum((1, 2, 3, 4)) / count
    unweighted = (error * keep).sum((1, 2, 3, 4)) / count
    total = weighted.mean()
    actual_weights = weights[keep]
    terms = dict(
        mean=total,
        t1_to_flair=weighted[:n].mean(),
        flair_to_t1=weighted[n:].mean(),
        unweighted=unweighted.mean(),
        unweighted_t1_to_flair=unweighted[:n].mean(),
        unweighted_flair_to_t1=unweighted[n:].mean(),
        weight_min=actual_weights.min(),
        weight_max=actual_weights.max(),
        foreground_fraction=keep.float().mean(),
    )
    return total, terms
