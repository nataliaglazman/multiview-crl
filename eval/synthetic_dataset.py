import os

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PseudoMRIRenderer(nn.Module):
    def __init__(self, res=64):
        super().__init__()
        self.res = res
        grid = torch.linspace(-1, 1, res)
        self.register_buffer(
            "coords",
            torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1),
        )

    def _seeded_noise(self, scale, gen, device):
        n = torch.randn(
            1,
            1,
            self.res // scale,
            self.res // scale,
            self.res // scale,
            generator=gen,
            device=device,
        )
        return F.interpolate(n, size=(self.res,) * 3, mode="trilinear", align_corners=False).squeeze(0).squeeze(0)

    def _upsample_field(self, z_field, device):
        """Trilinear-upsample a small (K, K, K) latent grid to volume resolution.

        Deterministic — same latent grid → same field. Used in place of seeded
        random noise so the gyral / fissure pattern becomes a discoverable
        content latent rather than an unrecoverable per-sample seed.
        """
        return (
            F.interpolate(
                z_field.to(device).float()[None, None],
                size=(self.res,) * 3,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    def render_structure(self, z_content, z_deformation, z_fissure, device):
        """Deterministic given (z_content, z_deformation, z_fissure). Shared across views.

        z_deformation: small (K, K, K) grid → trilinear-upsampled into the
            cortical-deformation field (gyral pattern proxy). Replaces the
            previous random `_seeded_noise(scale=8, ...)` call so the gyral
            pattern is a recoverable latent rather than an unrecoverable seed.
        z_fissure: small (K, K, K) grid → drives the longitudinal fissure
            wiggle (was `_seeded_noise(scale=16, ...)`).
        """
        radii_wm = 0.5 + z_content[0].clamp(-1, 1) * 0.1
        radii_gm = radii_wm + 0.15
        ventricle_size = 0.15 + z_content[1].clamp(-1, 1) * 0.05

        dist = torch.norm(self.coords, dim=-1)

        deformation = self._upsample_field(z_deformation, device) * 0.1
        deformed_dist = dist + deformation

        mask_gm = deformed_dist < radii_gm
        mask_wm = deformed_dist < radii_wm

        x_coords = self.coords[..., 0]
        ventricle_split = torch.abs(x_coords) > 0.05
        mask_csf = (deformed_dist < ventricle_size) & ventricle_split

        fissure_noise = self._upsample_field(z_fissure, device) * 0.05
        fissure_mask = (torch.abs(x_coords + fissure_noise) < 0.03) & mask_gm

        tissue_map = torch.zeros_like(dist, dtype=torch.long)
        tissue_map[mask_gm] = 3
        tissue_map[mask_wm] = 2
        tissue_map[mask_csf] = 1
        tissue_map[fissure_mask] = 1

        lesion_xyz = z_content[2:5].clamp(-0.6, 0.6).to(device)
        lesion_mask = (torch.norm(self.coords - lesion_xyz, dim=-1) < 0.1) & mask_wm

        return tissue_map, lesion_mask

    # render_modality consumes 3 style components (gain, bias, noise sigma).
    # Shorter z_style is right-padded with zeros so it never IndexErrors —
    # missing components simply default to "no modulation".
    N_STYLE_COMPONENTS = 3

    def render_modality(self, tissue_map, lesion_mask, z_style, modality, view_seed, device):
        """View-specific rendering. z_style drives gain, bias, and noise sigma."""
        gen = torch.Generator(device=device).manual_seed(int(view_seed))

        if modality == "T1":
            base = torch.tensor([0.0, 0.1, 0.8, 0.5], device=device)
            lesion_int = 0.4
        elif modality == "FLAIR":
            base = torch.tensor([0.0, 0.1, 0.4, 0.8], device=device)
            lesion_int = 1.0
        else:
            raise ValueError(f"Unknown modality {modality}")

        # Right-pad z_style with zeros if the caller supplied fewer components
        # than render_modality consumes. Defensive guard for misconfigured runs.
        if z_style.numel() < self.N_STYLE_COMPONENTS:
            pad = torch.zeros(self.N_STYLE_COMPONENTS - z_style.numel(), device=z_style.device)
            z_style = torch.cat([z_style.flatten(), pad])

        gain = 1.0 + z_style[0].clamp(-1, 1) * 0.3
        bias = z_style[1].clamp(-1, 1) * 0.1
        lut = base * gain + bias
        volume = lut[tissue_map]

        volume = torch.where(lesion_mask, torch.full_like(volume, lesion_int), volume)

        bias_field = 1.0 + self._seeded_noise(scale=4, gen=gen, device=device) * 0.15
        volume = volume * bias_field

        sigma = 0.01 + z_style[2].abs() * 0.05
        real = torch.randn(volume.shape, generator=gen, device=device) * sigma
        imag = torch.randn(volume.shape, generator=gen, device=device) * sigma
        volume = torch.sqrt((volume + real) ** 2 + imag**2)

        volume = F.avg_pool3d(
            volume.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(0)
        return volume


class Primitive3DRenderer(nn.Module):
    def __init__(self, res=32, modality="T1"):
        super().__init__()
        self.res = res
        self.modality = modality
        grid = torch.linspace(-1, 1, res)
        # indexing='ij' ensures correct spatial alignment
        self.register_buffer("coords", torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1))

    def _apply_lut(self, x):
        if self.modality == "T1":
            return torch.pow(x, 1.5)
        elif self.modality == "FLAIR":
            # Fluid suppression simulation
            return 1.0 - torch.exp(-((x - 0.7) ** 2) / 0.1)
        return x

    def forward(self, z_t, z_b_shared, z_b_style):
        # Ensure inputs are [B, D, H, W]
        if z_t.dim() == 3:
            z_t, z_b_shared, z_b_style = z_t.unsqueeze(0), z_b_shared.unsqueeze(0), z_b_style.unsqueeze(0)

        B = z_t.shape[0]
        device = z_t.device
        # Output volume: [B, 1, res, res, res]
        volume = torch.zeros((B, 1, self.res, self.res, self.res), device=device)

        # Iterate through the 4x4x4 latent grid
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # Check existence for the whole batch at once
                    # mask_exists shape: [B, 1, 1, 1, 1]
                    mask_exists = (z_t[:, i, j, k] > 0).float().view(B, 1, 1, 1, 1)

                    if mask_exists.sum() == 0:
                        continue

                    # Define center in range [-0.75, 0.75]
                    center = torch.tensor([(i - 1.5) / 2, (j - 1.5) / 2, (k - 1.5) / 2], device=device)

                    # Calculate sphere distance: dist shape [res, res, res]
                    dist = torch.norm(self.coords - center, dim=-1)

                    # Radius depends on z_t value
                    radius = 0.12 + (z_t[:, i, j, k].float() / 25.0).view(B, 1, 1, 1, 1)

                    # Primitive shape
                    primitive = (dist.unsqueeze(0).unsqueeze(0) < radius).float()

                    # Intensity from shared + style
                    intensity = (z_b_shared[:, i, j, k].float() + z_b_style[:, i, j, k].float()) / 32.0
                    intensity = intensity.view(B, 1, 1, 1, 1)

                    # Add to volume using max pooling (simulates occlusion/additive density)
                    volume = torch.max(volume, primitive * intensity * mask_exists)

        return self._apply_lut(volume)


class Random3DRenderer(nn.Module):
    """
    A fixed, randomly initialized 3D convolutional decoder.
    Simulates a complex physical rendering process.
    """

    def __init__(self, K_t, K_b, output_res=32):
        super().__init__()
        self.emb_t = nn.Embedding(K_t, 16)
        self.emb_b = nn.Embedding(K_b, 16)

        # Starts at 8x8x8, upsamples to 32x32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Freeze weights to maintain a consistent "ground truth" rendering function
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, z_t, z_b):
        # Embed inputs: [B, D, H, W] -> [B, D, H, W, C] -> [B, C, D, H, W]
        e_t = self.emb_t(z_t).permute(0, 4, 1, 2, 3)
        e_b = self.emb_b(z_b).permute(0, 4, 1, 2, 3)

        # Align spatial dimensions (Upsample e_t to 8x8x8 to match e_b)
        e_t_feat = F.interpolate(e_t, size=(8, 8, 8), mode="trilinear", align_corners=False)

        # Combine embeddings and decode
        combined = torch.cat([e_t_feat, e_b], dim=1)  # 16 + 16 = 32 channels
        return self.decoder(combined)


class Synthetic3DDisentanglementDataset(Dataset):
    """
    A drop-in 3D synthetic dataset generator.
    Outputs: (View 1, View 2, Ground Truth Latents Dictionary)
    """

    def __init__(
        self,
        num_samples=1000,
        res=32,
        seed=42,
        mode="primitives",
        n_content=5,
        n_style=3,
        n_deformation_grid=4,
        n_fissure_grid=8,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.res = res
        self.mode = mode
        self.seed = seed
        self.n_content = n_content
        self.n_style = n_style
        # Spatial-content grid sizes for pseudo_mri mode. Trilinear-upsampled
        # to (res, res, res) → drives the deformation / fissure fields.
        # Default 4³ for the gyral pattern (low-frequency, ~16 dof per axis at res=32)
        # and 8³ for the fissure (slightly higher frequency).
        self.n_deformation_grid = n_deformation_grid
        self.n_fissure_grid = n_fissure_grid

        if mode == "pseudo_mri":
            # render_structure indexes z_content[0..4] and render_modality indexes
            # z_style[0..2]. Smaller values silently disable the corresponding
            # anatomical / nuisance factor — warn loudly so it's not a surprise.
            if n_content < 5:
                import warnings

                warnings.warn(
                    f"pseudo_mri renderer consumes z_content[0..4] but n_content={n_content}. "
                    f"Missing components will default to 0 (no anatomical variation on those axes). "
                    f"Pass --synthetic-n-content 5 (or greater) to fully exercise the renderer.",
                    stacklevel=2,
                )
            if n_style < PseudoMRIRenderer.N_STYLE_COMPONENTS:
                import warnings

                warnings.warn(
                    f"pseudo_mri renderer consumes z_style[0..{PseudoMRIRenderer.N_STYLE_COMPONENTS - 1}] "
                    f"but n_style={n_style}. Missing components will default to 0 (no contrast/noise modulation). "
                    f"Pass --synthetic-n-style {PseudoMRIRenderer.N_STYLE_COMPONENTS} to fully exercise it.",
                    stacklevel=2,
                )

        # Hyperparameters from the recipe (used by primitives / random modes)
        self.grid_t = 4  # Top-level spatial grid
        self.grid_m = 8  # Middle-level spatial grid
        self.grid_b = 16  # Bottom-level spatial grid
        self.K_t = 10  # Top-level categories
        self.K_m = 16  # Middle-level categories
        self.K_b = 20  # Bottom-level categories

        if mode == "primitives":
            self.renderer_v1 = Primitive3DRenderer(res=res, modality="T1")
            self.renderer_v2 = Primitive3DRenderer(res=res, modality="FLAIR")
        elif mode == "pseudo_mri":
            self.renderer = PseudoMRIRenderer(res=res)
        else:
            # Fallback to the Conv-based random decoders
            self.renderer_v1 = Random3DRenderer(8, 16, res)
            self.renderer_v2 = Random3DRenderer(8, 16, res)

        torch.manual_seed(seed)

    def _build_renderer(self, seed):
        torch.manual_seed(seed)
        return Random3DRenderer(self.K_t, self.K_b, self.res)

    def __len__(self):
        return self.num_samples

    def _pseudo_mri_item(self, idx):
        sample_seed = self.seed * 1000003 + idx
        sample_gen = torch.Generator().manual_seed(sample_seed)

        z_content = torch.randn(self.n_content, generator=sample_gen)
        # Spatial content latents — drive the gyral pattern and fissure shape
        # via deterministic upsampling. Each subject gets a unique, recoverable
        # cortical pattern instead of an unrecoverable seed.
        z_deformation = torch.randn(
            self.n_deformation_grid,
            self.n_deformation_grid,
            self.n_deformation_grid,
            generator=sample_gen,
        )
        z_fissure = torch.randn(
            self.n_fissure_grid,
            self.n_fissure_grid,
            self.n_fissure_grid,
            generator=sample_gen,
        )
        z_style_v1 = torch.randn(self.n_style, generator=sample_gen)
        z_style_v2 = torch.randn(self.n_style, generator=sample_gen)

        device = torch.device("cpu")
        with torch.no_grad():
            tissue, lesion = self.renderer.render_structure(
                z_content,
                z_deformation,
                z_fissure,
                device=device,
            )
            x_v1 = self.renderer.render_modality(
                tissue,
                lesion,
                z_style_v1,
                "T1",
                view_seed=sample_seed * 2,
                device=device,
            )
            x_v2 = self.renderer.render_modality(
                tissue,
                lesion,
                z_style_v2,
                "FLAIR",
                view_seed=sample_seed * 2 + 1,
                device=device,
            )

        # True foreground mask from the structural render — shared across views
        # since brain anatomy is content. Shape: [1, res, res, res] to match
        # the image tensors.
        brain_mask = (tissue > 0).unsqueeze(0).float()

        latents = {
            "z_content": z_content,
            "z_deformation": z_deformation,
            "z_fissure": z_fissure,
            "z_style_v1": z_style_v1,
            "z_style_v2": z_style_v2,
            "brain_mask": brain_mask,
        }
        return x_v1, x_v2, latents

    def _categorical_item(self, idx):
        # Top-level z_t
        z_t = torch.randint(0, self.K_t, (self.grid_t, self.grid_t, self.grid_t))

        # Middle-level z_m conditioned on z_t
        z_t_upsampled = (
            F.interpolate(
                z_t.float().view(1, 1, self.grid_t, self.grid_t, self.grid_t),
                size=(self.grid_m, self.grid_m, self.grid_m),
                mode="nearest",
            )
            .squeeze()
            .long()
        )

        z_m_base = torch.randint(0, self.K_m // 2, (self.grid_m, self.grid_m, self.grid_m))
        z_m_offset = (z_t_upsampled % 2) * (self.K_m // 2)
        z_m = z_m_base + z_m_offset

        # Bottom-level z_b conditioned on z_m
        z_m_upsampled = (
            F.interpolate(
                z_m.float().view(1, 1, self.grid_m, self.grid_m, self.grid_m),
                size=(self.grid_b, self.grid_b, self.grid_b),
                mode="nearest",
            )
            .squeeze()
            .long()
        )

        z_b_base = torch.randint(0, self.K_b // 2, (self.grid_b, self.grid_b, self.grid_b))
        z_b_offset = (z_m_upsampled % 2) * (self.K_b // 2)
        z_b = z_b_base + z_b_offset

        mid_z = self.grid_b // 2
        z_b_shared = z_b[:, :, :mid_z]

        z_b_style_v1 = torch.randint(0, self.K_b, (self.grid_b, self.grid_b, self.grid_b - mid_z))
        z_b_style_v2 = torch.randint(0, self.K_b, (self.grid_b, self.grid_b, self.grid_b - mid_z))

        z_b_v1 = torch.cat([z_b_shared, z_b_style_v1], dim=2)
        z_b_v2 = torch.cat([z_b_shared, z_b_style_v2], dim=2)

        with torch.no_grad():
            if self.mode == "primitives":
                x_v1 = self.renderer_v1(z_t, z_b_shared, z_b_style_v1).squeeze(0)
                x_v2 = self.renderer_v2(z_t, z_b_shared, z_b_style_v2).squeeze(0)
            else:
                x_v1 = self.renderer_v1(z_t.unsqueeze(0), z_b_v1.unsqueeze(0)).squeeze(0)
                x_v2 = self.renderer_v2(z_t.unsqueeze(0), z_b_v2.unsqueeze(0)).squeeze(0)

        x_v1 = x_v1 + torch.randn_like(x_v1) * 0.01
        x_v2 = x_v2 + torch.randn_like(x_v2) * 0.01

        latents = {
            "z_t": z_t,
            "z_m": z_m,
            "z_b_shared": z_b_shared,
            "z_b_style_v1": z_b_style_v1,
            "z_b_style_v2": z_b_style_v2,
        }
        return x_v1, x_v2, latents

    def __getitem__(self, idx):
        if self.mode == "pseudo_mri":
            return self._pseudo_mri_item(idx)
        return self._categorical_item(idx)


import numpy as np


def view_3d_volume(tensor_3d):
    import plotly.graph_objects as go

    vol = tensor_3d.squeeze().cpu().numpy()
    res = vol.shape[0]

    # Create a 3D coordinate grid
    X, Y, Z = np.mgrid[0:res, 0:res, 0:res]

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=vol.flatten(),
            isomin=vol.min() + 0.1,  # Ignore empty space
            isomax=vol.max(),
            opacity=0.2,  # Transparency
            surface_count=15,  # Number of isosurfaces
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        scene_xaxis_showticklabels=False, scene_yaxis_showticklabels=False, scene_zaxis_showticklabels=False
    )
    fig.show()


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Create an output directory for saving NIfTI files
    out_dir = "pseudo_mri_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize dataset (Now using the new mode)
    dataset = Synthetic3DDisentanglementDataset(num_samples=1000, res=100, seed=42, mode="pseudo_mri")

    # 2. Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # 3. Fetch and save all batches
    sample_idx = 0
    print(f"Starting generation of {dataset.num_samples} synthetic MRI pairs...")
    for i, (view1, view2, gt_latents) in enumerate(dataloader):
        # Iterate over the items in the current batch
        batch_size = view1.shape[0]
        for b in range(batch_size):
            v1_np = view1[b].squeeze().cpu().numpy()
            v2_np = view2[b].squeeze().cpu().numpy()

            # Create NIfTI images with an identity affine matrix
            nifti_v1 = nib.Nifti1Image(v1_np, affine=np.eye(4))
            nifti_v2 = nib.Nifti1Image(v2_np, affine=np.eye(4))

            p1 = os.path.join(out_dir, f"sample_{sample_idx:04d}_T1.nii.gz")
            p2 = os.path.join(out_dir, f"sample_{sample_idx:04d}_FLAIR.nii.gz")
            nib.save(nifti_v1, p1)
            nib.save(nifti_v2, p2)

            sample_idx += 1

        if (i + 1) % 10 == 0:
            print(f"Saved {sample_idx} samples...")

    print(f"Successfully generated and saved all {sample_idx} synthetic MRI pairs to '{out_dir}'.")
