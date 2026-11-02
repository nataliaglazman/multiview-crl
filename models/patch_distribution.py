"""Position-free distributions of fixed local projections of 3D residual patches.

The projection bank and sampling lattice are independent of the subject. Every
valid patch contributes, including normal tissue, so abnormality prevalence is
retained. No learned feature extractor or lesion-dependent crop selects targets.
"""

import torch
from torch.nn import functional as F

# Endpoints keep a single abnormal patch observable even below the 0.1% tail.
# Equal weight per entry intentionally emphasizes tails; this is not uniform
# quadrature of the exact sliced Wasserstein integral.
PATCH_QUANTILES = (
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.98,
    0.99,
    0.995,
    0.998,
    0.999,
    1.0,
)


def fixed_patch_projections(patch_size, projections, seed):
    """Unit-norm 3D filters from a private CPU RNG; the first reads the centre voxel.

    Remaining directions have zero spatial mean, emphasizing local structure
    rather than duplicating the centre intensity distribution.
    """
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    if isinstance(projections, bool) or not isinstance(projections, int) or projections < 1:
        raise ValueError("patch_projections must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("patch_seed must be an integer in [0,2**63)")
    if patch_size == 1 and projections > 1:
        raise ValueError("A one-voxel patch has only one projection; use patch_projections=1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    filters = torch.randn(projections, patch_size**3, generator=generator)
    if projections > 1:
        filters[1:] -= filters[1:].mean(1, keepdim=True)
    filters[0].zero_()
    filters[0, patch_size**3 // 2] = 1
    filters /= filters.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return filters.reshape(projections, 1, patch_size, patch_size, patch_size)


def sorted_quantiles(values, probabilities=PATCH_QUANTILES):
    """Quantiles of the last (unordered samples) axis, using linear interpolation."""
    if values.shape[-1] == 0:
        raise ValueError("Patch distributions need at least one valid patch")
    ordered = values.sort(dim=-1).values
    probability = values.new_tensor(probabilities)
    if not torch.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Quantile probabilities must be finite and in [0,1]")
    position = probability * (values.shape[-1] - 1)
    low, high = position.floor().long(), position.ceil().long()
    return torch.lerp(ordered[..., low], ordered[..., high], position - low)


def validate_patch_scales(grid, patch_size, scales):
    if not scales or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in scales):
        raise ValueError("patch_scales must contain positive integers")
    if len(set(scales)) != len(scales):
        raise ValueError("patch_scales must be unique")
    if any(min(grid) // scale < patch_size for scale in scales):
        raise ValueError("Each pooled target grid must fit the full patch_size")


@torch.no_grad()
def projected_patch_quantiles(residual, interior, filters, scales=(1, 2), valid=None):
    """(N,1,D,H,W) -> (N,scales,projections,quantiles), with detached targets.

    Each scale first averages nonoverlapping blocks, then applies valid 3D
    convolutions. A patch must lie entirely inside the fixed population interior
    and the subject's anatomical foreground. All qualifying centres contribute;
    no target-dependent top-k selection or abnormality mask changes their mass.
    """
    if residual.ndim != 5 or residual.shape[1] != 1 or residual.shape[0] == 0:
        raise ValueError("residual must have shape (N,1,D,H,W), N > 0")
    if interior.shape != (1, 1, *residual.shape[2:]):
        raise ValueError("interior must have shape (1,1,D,H,W)")
    if filters.ndim != 5 or filters.shape[1] != 1 or len(set(filters.shape[2:])) != 1:
        raise ValueError("filters must have shape (projections,1,k,k,k)")
    patch_size = filters.shape[-1]
    validate_patch_scales(residual.shape[2:], patch_size, scales)
    residual = residual.detach().float()
    filters = filters.detach().to(residual)
    interior = interior.detach().to(device=residual.device, dtype=torch.bool)
    if valid is None:
        valid = torch.ones_like(residual, dtype=torch.bool)
    elif valid.shape != residual.shape:
        raise ValueError("valid foreground must have the same shape as residual")
    support = interior & valid.detach().to(device=residual.device, dtype=torch.bool)
    if not torch.isfinite(residual).all() or not torch.isfinite(filters).all():
        raise ValueError("Residuals and projection filters must be finite")
    rows = []
    for scale in scales:
        values = residual if scale == 1 else F.avg_pool3d(residual, scale, stride=scale)
        fraction = support.float() if scale == 1 else F.avg_pool3d(support.float(), scale, stride=scale)
        keep = F.avg_pool3d((fraction >= 1 - 1e-6).float(), patch_size, stride=1) >= 1 - 1e-6
        projected = F.conv3d(values, filters).flatten(2)
        keep = keep.flatten(1)
        if not keep.any(1).all():
            raise ValueError(
                f"No complete interior patch at scale {scale}; use a finer grid, smaller patches or fewer scales"
            )
        rows.append(torch.stack([sorted_quantiles(row[:, mask]) for row, mask in zip(projected, keep)]))
    return torch.stack(rows, dim=1)
