import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def build_content_scm(n_dims, graph_type="chain", edge_prob=0.5, seed=0):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n_dims, n_dims), dtype=bool)

    if graph_type == "chain":
        for i in range(n_dims - 1):
            adj[i, i + 1] = True
    elif graph_type == "full":
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                adj[i, j] = True
    elif graph_type == "random":
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                if rng.rand() < edge_prob:
                    adj[i, j] = True
    else:
        raise ValueError(f"Unknown causal graph type: {graph_type}")

    parents = {}
    weights = {}
    gen = torch.Generator().manual_seed(seed)
    for idx in range(n_dims):
        parent_indices = np.where(adj[:, idx])[0].tolist()
        parents[idx] = parent_indices
        if len(parent_indices) > 0:
            w = torch.randn(len(parent_indices), generator=gen)
            w = w / (w.norm() + 1e-8)
            weights[idx] = w
        else:
            weights[idx] = torch.tensor([])

    return {"adj": adj, "parents": parents, "weights": weights, "n_dims": n_dims}


def sample_content_from_scm(scm, generator, noise_scale=0.4, nonlinearity="leaky_relu"):
    n = scm["n_dims"]
    z = torch.zeros(n)
    noise = torch.randn(n, generator=generator)

    for d in range(n):
        pa = scm["parents"][d]
        if len(pa) == 0:
            z[d] = noise[d]
        else:
            parent_vals = z[pa]
            w = scm["weights"][d]
            signal = (parent_vals * w).sum()
            if nonlinearity == "leaky_relu":
                signal = F.leaky_relu(signal, 0.2)
            elif nonlinearity == "tanh":
                signal = torch.tanh(signal)
            z[d] = signal + noise_scale * noise[d]
    return z


def _gaussian_blur_field(field, sigma):
    """Separable 3D Gaussian blur with periodic ('wrap') padding -> stationary field.

    Convolving white noise with a Gaussian of width ``sigma`` IS the spectral
    construction of a GP sample whose covariance is a squared-exponential kernel of
    that length-scale (exact for a stationary SE kernel up to lattice/boundary
    effects). ``sigma`` is in lattice units.
    """
    if sigma <= 0:
        return field
    from scipy.ndimage import gaussian_filter

    out = gaussian_filter(field.detach().cpu().numpy().astype(np.float64), sigma=float(sigma), mode="wrap")
    return torch.from_numpy(out).to(field.dtype)


def sample_gp_field(grid, lengthscale, generator, *, prior="gp", dof=3.0, tau_seed=0):
    """One stationary GP/TP latent field on a ``grid**3`` lattice, at unit variance.

    The field latents of Halva et al. (AISTATS 2024): each independent component is
    a spatial field drawn from a process with its own covariance kernel. Here the
    kernel is squared-exponential with length-scale ``lengthscale`` (set via the
    Gaussian-blur construction above). ``prior='gp'`` is the Gaussian case;
    ``prior='tp'`` rescales the field by ``1/sqrt(tau)``, ``tau ~ Gamma(dof/2,
    dof/2)`` (Student-t process, their Prop. 1) -> heavier tails. Their Theorem 2:
    GP components are identifiable iff their kernels are DISTINCT, whereas the
    non-Gaussian t-process stays identifiable even with REPEATED kernels -- which
    is exactly the ``field_kernels`` x ``field_prior`` control this enables.

    Amplitude is left to the caller (the renderer's deformation/fissure gains);
    the field is returned unit-variance so ``field_scale=1`` matches the legacy iid
    amplitude.
    """
    white = torch.randn(grid, grid, grid, generator=generator)
    field = _gaussian_blur_field(white, lengthscale)
    field = field / (field.std() + 1e-8)
    if prior == "tp":
        tau = np.random.RandomState(int(tau_seed) % (2**32 - 1)).gamma(dof / 2.0, 2.0 / dof)
        field = field / (float(tau) ** 0.5 + 1e-8)
    return field


# clean-content mode (synthetic_clean_content): factor applied to the unlabeled
# z_deformation / z_fissure fields. 0.0 removes them entirely so the named content
# factors fully determine structural variance. 1.0 (default mode) leaves them at
# full amplitude.
CLEAN_NUISANCE_SCALE = 0.0


class PseudoMRIRenderer(nn.Module):
    def __init__(self, res=64, style_scale=1.0, content_scale=1.0):
        super().__init__()
        self.res = res
        self.style_scale = style_scale
        self.content_scale = content_scale
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

    # Number of z_content components consumed by render_structure.
    # [0] brain size (WM radius), [1] ventricle size, [2:5] lesion position,
    # [5] cortical thickness, [6] temporal-lobe atrophy (hippocampal proxy),
    # [7] left–right asymmetry, [8] sulcal widening.
    N_CONTENT_COMPONENTS = 9

    def render_structure(self, z_content, z_deformation, z_fissure, device, clean=False):
        """Deterministic given (z_content, z_deformation, z_fissure). Shared across views.

        z_content layout (9 components, extras default to 0):
            [0]  brain size      — WM radius ±0.1 around 0.5
            [1]  ventricle size  — CSF cavity ±0.05 around 0.15
            [2:5] lesion xyz     — WM lesion center (direction within the WM)
            [5]  cortical thickness — GM shell width ±0.06 around 0.15
            [6]  temporal atrophy — shrinks a compact bilateral inferior–lateral
                                   temporal region (hippocampal volume proxy)
            [7]  L–R asymmetry   — differential atrophy across hemispheres
            [8]  sulcal widening  — depth of a deterministic gyral corrugation

        z_deformation: small (K, K, K) grid → trilinear-upsampled into a random
            per-sample gyral corrugation. Pure nuisance: zeroed in clean mode.
        z_fissure: small (K, K, K) grid → drives the longitudinal fissure
            wiggle. Pure nuisance: zeroed in clean mode.
        """
        # Right-pad z_content with zeros when the caller supplies fewer dims
        # than the renderer consumes (back-compat with n_content=5 runs).
        if z_content.numel() < self.N_CONTENT_COMPONENTS:
            pad = torch.zeros(self.N_CONTENT_COMPONENTS - z_content.numel())
            z_content = torch.cat([z_content.flatten(), pad])

        # clean-content mode: tanh-squash (monotone → fully recoverable) instead of the
        # hard clamp (which flattens the ~1/3 of N(0,1) values that overflow ±1), and
        # zero out the unlabeled deformation/fissure nuisance so the named factors dominate.
        nuisance = CLEAN_NUISANCE_SCALE if clean else 1.0

        def _sq(z, a=1.0):
            return a * torch.tanh(z) if clean else z.clamp(-a, a)

        radii_wm = 0.5 + _sq(z_content[0]) * 0.1 * self.content_scale
        cortical_thickness = 0.15 + _sq(z_content[5]) * 0.06 * self.content_scale
        radii_gm = radii_wm + cortical_thickness
        ventricle_size = 0.15 + _sq(z_content[1]) * 0.05 * self.content_scale

        dist = torch.norm(self.coords, dim=-1)
        x_coords = self.coords[..., 0]
        y_coords = self.coords[..., 1]
        z_coords = self.coords[..., 2]

        # Nuisance gyral field (z_deformation): a per-sample random corrugation,
        # zeroed in clean-content mode. Pure nuisance — no named factor rides on
        # it (sulcal widening below has its own deterministic channel).
        deformation = self._upsample_field(z_deformation, device) * 0.1 * nuisance
        deformed_dist = dist + deformation

        # Sulcal widening (z_content[8]): a DETERMINISTIC high-frequency gyral
        # corrugation whose depth is set by z_content[8]. Unlike the nuisance
        # field above it does not vanish in clean-content mode, so it stays a
        # recoverable named factor. Sign flips the gyral phase (map stays
        # injective); |z| sets sulcal depth → surface roughness.
        gyral_pattern = torch.sin(12 * x_coords) * torch.sin(12 * y_coords) * torch.sin(12 * z_coords)
        sulcal_amp = _sq(z_content[8]) * 0.06 * self.content_scale
        deformed_dist = deformed_dist + gyral_pattern * sulcal_amp

        # Temporal-lobe atrophy (z_content[6]): shrink the WM/GM boundary inside a
        # compact, BILATERAL region over the (inferior, lateral, mid-A/P) temporal
        # lobes — a hippocampal / medial-temporal volume-loss proxy. A localized
        # Gaussian bump (not a product of half-space sigmoids) keeps the effect
        # off the brain centre and the superior/anterior cortex, so it no longer
        # drags the whole-brain volume down; |x| makes it symmetric across
        # hemispheres so it does not confound the L–R asymmetry factor.
        tw_x = torch.abs(x_coords) - 0.30  # lateral lobes at |x| ≈ 0.30
        tw_y = y_coords - 0.05
        tw_z = z_coords + 0.35  # inferior
        temporal_weight = torch.exp(-(tw_x**2 + tw_y**2 + tw_z**2) / (2 * 0.18**2))
        temporal_shrink = _sq(z_content[6]) * 0.12 * self.content_scale
        deformed_dist = deformed_dist + temporal_weight * temporal_shrink

        # Left–right asymmetry: differential atrophy across hemispheres.
        # Positive z_content[7] → left hemisphere (x < 0) more atrophied.
        lr_weight = torch.tanh(-3 * x_coords)  # smooth L/R gradient, ∈ (−1, 1)
        lr_shift = _sq(z_content[7]) * 0.08 * self.content_scale
        deformed_dist = deformed_dist + lr_weight * lr_shift

        mask_gm = deformed_dist < radii_gm
        mask_wm = deformed_dist < radii_wm

        ventricle_split = torch.abs(x_coords) > 0.05
        mask_csf = (deformed_dist < ventricle_size) & ventricle_split

        fissure_noise = self._upsample_field(z_fissure, device) * 0.05 * nuisance
        fissure_mask = (torch.abs(x_coords + fissure_noise) < 0.03) & mask_gm

        tissue_map = torch.zeros_like(dist, dtype=torch.long)
        tissue_map[mask_gm] = 3
        tissue_map[mask_wm] = 2
        tissue_map[mask_csf] = 1
        tissue_map[fissure_mask] = 1

        # WM lesion (z_content[2:5]): place the 0.1-radius lesion so it lands
        # INSIDE the white matter for (almost) every draw. z_content[2:5] is a
        # direction in the unit cube; scaling by lesion_reach/√3 bounds the centre
        # norm to lesion_reach = radii_wm − margin, so centre+radius stays within
        # WM regardless of direction. (Old code scaled to ±0.6 per axis → centre
        # norm up to ~1.04 ≫ radii_wm≈0.5, so the lesion was absent in ~95% of
        # samples and its 3 position dims were near-dead / unidentifiable.)
        lesion_dir = _sq(z_content[2:5], 1.0).to(device)
        lesion_reach = (radii_wm - 0.12).clamp_min(0.05)
        lesion_xyz = lesion_dir * (lesion_reach / (3**0.5))
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

        gain = (1.0 + z_style[0].clamp(-1, 1) * 0.3 * self.style_scale).clamp_min(0.05)
        bias = z_style[1].clamp(-1, 1) * 0.1 * self.style_scale
        lut = base * gain + bias
        volume = lut[tissue_map]

        volume = torch.where(lesion_mask, torch.full_like(volume, lesion_int), volume)

        bias_field = 1.0 + self._seeded_noise(scale=4, gen=gen, device=device) * 0.15 * self.style_scale
        volume = volume * bias_field

        sigma = 0.01 + z_style[2].abs() * 0.05 * self.style_scale
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
        hierarchical_content=False,
        causal=False,
        causal_graph="chain",
        causal_edge_prob=0.5,
        causal_noise_scale=0.4,
        causal_nonlinearity="leaky_relu",
        clean_content=False,
        style_scale=1.0,
        content_scale=1.0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.res = res
        self.mode = mode
        self.seed = seed
        self.n_content = n_content
        self.n_style = n_style
        self.hierarchical_content = hierarchical_content
        self.causal = causal
        self.causal_noise_scale = causal_noise_scale
        self.causal_nonlinearity = causal_nonlinearity
        self.clean_content = clean_content

        if causal and hierarchical_content:
            raise ValueError("--synthetic-causal and --synthetic-hierarchical-content are mutually exclusive")

        if causal and mode != "pseudo_mri":
            raise ValueError(f"--synthetic-causal requires --synthetic-mode pseudo_mri, got '{mode}'")

        if causal:
            self.scm = build_content_scm(n_content, causal_graph, causal_edge_prob, seed)
        else:
            self.scm = None

        # Spatial-content grid sizes for pseudo_mri mode. Trilinear-upsampled
        # to (res, res, res) → drives the deformation / fissure fields.
        # Default 4³ for the gyral pattern (low-frequency, ~16 dof per axis at res=32)
        # and 8³ for the fissure (slightly higher frequency).
        self.n_deformation_grid = n_deformation_grid
        self.n_fissure_grid = n_fissure_grid

        if mode == "pseudo_mri":
            # render_structure indexes z_content[0..8] and render_modality indexes
            # z_style[0..2]. Smaller values silently disable the corresponding
            # anatomical / nuisance factor — warn loudly so it's not a surprise.
            if n_content < PseudoMRIRenderer.N_CONTENT_COMPONENTS:
                import warnings

                warnings.warn(
                    f"pseudo_mri renderer consumes z_content[0..{PseudoMRIRenderer.N_CONTENT_COMPONENTS - 1}] "
                    f"but n_content={n_content}. Missing components will default to 0 "
                    f"(no anatomical variation on those axes). Pass --synthetic-n-content "
                    f"{PseudoMRIRenderer.N_CONTENT_COMPONENTS} (or greater) to fully exercise the renderer.",
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
            self.renderer = PseudoMRIRenderer(res=res, style_scale=style_scale, content_scale=content_scale)
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

    # Hierarchical content structure: a global atrophy scalar drives the
    # regional content dims via fixed coupling weights + independent residuals.
    # Indices into z_content that each regional factor occupies:
    #   [0] brain size, [1] ventricle, [5] cortical thickness,
    #   [6] temporal atrophy, [8] sulcal widening.
    # [2:5] (lesion xyz) and [7] (L-R asymmetry) are independent of global.
    _HIER_COUPLINGS = {
        0: -1.0,  # global atrophy → smaller brain
        1: -0.8,  # global atrophy → bigger ventricles (anticorrelated)
        5: -0.9,  # global atrophy → thinner cortex
        6: 1.5,  # global atrophy → more temporal atrophy (AD-like)
        8: 0.7,  # global atrophy → wider sulci
    }
    _HIER_RESIDUAL_SCALE = 0.6

    def _sample_hierarchical_content(self, gen):
        """Sample z_content with a shared global-atrophy factor.

        Returns (z_content, z_global, z_residuals) where:
          z_content  = the actual vector fed to the renderer
          z_global   = scalar global atrophy severity
          z_residuals = independent per-dim residuals (same shape as z_content)

        The relationship is:
          z_content[i] = coupling[i] * z_global + residual_scale * residual[i]
        for coupled dims, and z_content[i] = residual[i] for independent dims.
        """
        z_global = torch.randn(1, generator=gen)
        z_residuals = torch.randn(self.n_content, generator=gen)
        z_content = z_residuals.clone() * self._HIER_RESIDUAL_SCALE
        for idx, weight in self._HIER_COUPLINGS.items():
            if idx < self.n_content:
                z_content[idx] = weight * z_global.item() + self._HIER_RESIDUAL_SCALE * z_residuals[idx]
        return z_content, z_global, z_residuals

    def _pseudo_mri_item(self, idx):
        sample_seed = self.seed * 1000003 + idx
        sample_gen = torch.Generator().manual_seed(sample_seed)

        if self.causal:
            z_content = sample_content_from_scm(self.scm, sample_gen, self.causal_noise_scale, self.causal_nonlinearity)
        elif self.hierarchical_content:
            z_content, z_global, z_residuals = self._sample_hierarchical_content(sample_gen)
        else:
            z_content = torch.randn(self.n_content, generator=sample_gen)

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
                clean=self.clean_content,
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

        brain_mask = (tissue > 0).unsqueeze(0).float()

        latents = {
            "z_content": z_content,
            "z_deformation": z_deformation,
            "z_fissure": z_fissure,
            "z_style_v1": z_style_v1,
            "z_style_v2": z_style_v2,
            "brain_mask": brain_mask,
        }
        if self.causal:
            latents["causal_adj"] = torch.from_numpy(self.scm["adj"].astype(np.float32))
        elif self.hierarchical_content:
            latents["z_global_atrophy"] = z_global
            latents["z_content_residuals"] = z_residuals

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
