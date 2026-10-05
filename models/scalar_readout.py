"""Nine-value readouts for the controlled scalar compression experiment.

The geometric readout exports physical centroid coordinates, not the renderer's
WM-quantile lesion controls. Its other six values are unconstrained scalars.
"""

import torch
import torch.nn.functional as F
from torch import nn

SEMANTIC_NAMES = (
    "brain_size",
    "ventricle_size",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_amplitude",
)


def pool_grid(x, grid):
    if any(n < grid or n % grid for n in x.shape[-3:]):
        raise ValueError(f"Grid {grid} must divide and fit map {tuple(x.shape[-3:])}")
    return F.avg_pool3d(x, tuple(n // grid for n in x.shape[-3:]))


def coordinates(grid, resolution):
    # Nominal image-bin centers, in the renderer's [-1,1] index coordinates.
    axis = 2 * ((torch.arange(grid).float() + 0.5) * resolution / grid - 0.5) / (resolution - 1) - 1
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


class ScalarReadout(nn.Module):
    def __init__(self, channels, grid=8, width=16, resolution=64, geometric=True, image_source=False):
        super().__init__()
        self.grid = grid
        self.geometric = geometric
        self.stem = nn.Sequential(
            nn.Conv3d(channels, width, 3, stride=2 if image_source else 1, padding=1),
            nn.SiLU(),
            nn.Conv3d(width, width, 3, stride=2 if image_source else 1, padding=1),
            nn.SiLU(),
        )
        self.register_buffer("coordinates", coordinates(grid, resolution))
        if geometric:
            self.location = nn.Conv3d(width, 1, 1)
            self.weights = nn.Conv3d(width, 6, 1)
            self.values = nn.Conv3d(width, 6, 1)
        else:
            self.free = nn.Sequential(
                nn.Conv3d(width, 4, 1),
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(4 * grid**3, 128),
                nn.SiLU(),
                nn.Linear(128, 9),
            )

    def forward(self, features):
        h = pool_grid(self.stem(features), self.grid)
        if not self.geometric:
            return self.free(h)
        location = self.location(h).flatten(1).softmax(-1)
        center = location @ self.coordinates
        weights = self.weights(h).flatten(2).softmax(-1)
        scalars = (weights * self.values(h).flatten(2)).sum(-1)
        # No subsequent MLP is allowed to remix the geometric coordinates.
        return torch.cat((scalars[:, :2], center, scalars[:, 2:]), 1)


class DescriptorDecoder(nn.Module):
    """Training-only auxiliary head. The ONLY input is the nine scalar values."""

    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(nn.Linear(9, 128), nn.SiLU(), nn.Linear(128, 2 * grid**3))

    def forward(self, code):
        return self.net(code).reshape(-1, 2, self.grid, self.grid, self.grid)


def descriptors(images, grid):
    """Label-free local mean and high-pass energy, calculated before binning.

    Energy is not a signed sulcal-amplitude label. Neither descriptor is claimed
    sufficient for the factors: recovering them remains an experimental question.
    """
    smooth = F.avg_pool3d(F.pad(images, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, stride=1)
    energy = (images - smooth).square()
    return torch.cat((pool_grid(images, grid), pool_grid(energy, grid)), 1)


def flip_volume(image, signs):
    return torch.flip(image, [image.ndim - 3 + k for k, s in enumerate(signs) if s < 0])


def ssl_objective(model, decoder, batch, descriptor_mean, descriptor_std, weights):
    """No target/factor/intervention argument: ordinary images and augmentations only."""
    z = model(batch["features"])
    z_flip = model(batch["flip_features"])
    z_photo = model(batch["photo_features"])
    target = (batch["descriptors"] - descriptor_mean) / descriptor_std
    flipped = (batch["flip_descriptors"] - descriptor_mean) / descriptor_std
    reconstruction = (
        F.mse_loss(decoder(z), target) + F.mse_loss(decoder(z_flip), flipped) + F.mse_loss(decoder(z_photo), target)
    ) / 3
    equivariance = F.mse_loss(z_flip[:, 2:5], z[:, 2:5] * batch["flip_signs"])
    photometric = F.mse_loss(z_photo, z)
    # An anti-collapse heuristic, not proof that each coordinate is a true factor.
    variance = F.relu(0.1 - torch.sqrt(z.var(0, unbiased=False) + 1e-6)).square().mean()
    terms = dict(reconstruction=reconstruction, equivariance=equivariance, photometric=photometric, variance=variance)
    return sum(weights[k] * v for k, v in terms.items()), terms
