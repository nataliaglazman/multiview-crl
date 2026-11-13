"""Synthetic three-level benchmark: global g, regional r_R, super local s(u), with known ground truth.

Extends ``experiments/gl_ident/generator.py`` (its seeding helpers and 3D pixel shuffle) from two levels to
three, with site, per-scan style, a weak super-local factor w and lesions.

Latent lattice ``L^3`` (default 16^3); each cell decodes to an ``m^3`` image block (m = image / lattice = 4)
through a fixed random network per modality: a 3^3 conv on the lattice (18 -> hidden), LeakyReLU(0.2), a 1^3
conv (hidden -> m^3), then a pixel shuffle. Two layers, kernel 3, upsampling x m. Unlike a stride-4 transposed
conv with kernel 3, which leaves one-voxel gaps between blocks, every voxel is covered; each block reads its
cell's 3^3 neighbourhood, so the decoder reaches one cell. Weights are seeded, orthogonal and frozen.

Latent field per cell, in channel order: [g (4), r_R(u) (4), s(u) (4), w(u) (1), site embedding (2),
lesion l(u) (1), style sigma_k (2)] = 18 channels. Style enters only its own modality.

Every random number of a subject is drawn from a generator seeded by (split, index), in a fixed order whatever
the variant, so all variants share the same draws (common random numbers):

    g, r noise, s, w, site (uniform), site (Gumbel), style, lesion blobs, eta_r, eta_g

Variants (``synth.variant``):
    exact                 as above.
    noisy                 cell u receives r_R + eta_r(u) and g + eta_g(u); eta has variance noise_tau2.
    one_modality_lesion   lesions enter FLAIR only.
    dependent             r_R ~ N(A g, I) (one A for all regions).
    site_confounded       site ~ softmax(confound_strength * V g).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.gl_ident.generator import derive_seed, pixel_shuffle, seeded

VARIANTS = ("exact", "noisy", "one_modality_lesion", "dependent", "site_confounded")
MODALITIES = ("t1", "flair")


def gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    return torch.exp(-(x**2) / (2 * sigma**2))


def smooth_noise(n: int, channels: int, size: int, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """Stationary unit-variance Gaussian field ``[n, c, size, size, size]``: white noise * normalised Gaussian.

    The noise is drawn on a padded lattice and convolved without padding, so every cell sees a full kernel.
    """
    k1 = gaussian_kernel_1d(sigma)
    r = (len(k1) - 1) // 2
    k = k1[:, None, None] * k1[None, :, None] * k1[None, None, :]
    k = (k / k.square().sum().sqrt()).float()
    white = torch.randn(n * channels, 1, size + 2 * r, size + 2 * r, size + 2 * r, generator=gen)
    return F.conv3d(white, k[None, None]).reshape(n, channels, size, size, size)


def voronoi_regions(lattice: int, n_regions: int, seed: int) -> np.ndarray:
    """``[L, L, L]`` labels 1..n from the Voronoi cells of ``n_regions`` random seeds.

    When n is a perfect cube (8 = 2^3) the seeds are jittered on a k^3 grid (one uniform seed in the central
    half of each block), so cells are random but of similar size; otherwise seeds are drawn uniformly with a
    minimum spacing of 0.6 times the mean cell side.
    """
    rng = np.random.default_rng(seed)
    k = round(n_regions ** (1 / 3))
    if k**3 == n_regions:
        block = lattice / k
        corners = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), -1).reshape(-1, 3) * block
        seeds = corners + block * (0.25 + 0.5 * rng.random((n_regions, 3)))
    else:
        spacing = 0.6 * (lattice**3 / n_regions) ** (1 / 3)
        seeds = []
        for _ in range(100000):
            p = rng.random(3) * lattice
            if all(np.linalg.norm(p - q) >= spacing for q in seeds):
                seeds.append(p)
                if len(seeds) == n_regions:
                    break
        if len(seeds) < n_regions:
            raise RuntimeError(f"could not place {n_regions} spaced seeds in a {lattice}^3 lattice")
        seeds = np.array(seeds)
    centres = np.stack(np.meshgrid(*[np.arange(lattice) + 0.5] * 3, indexing="ij"), -1)
    dist = np.linalg.norm(centres[..., None, :] - seeds, axis=-1)
    return dist.argmin(-1).astype(np.int64) + 1


class ThreeLevelGenerator(nn.Module):
    """Frozen two-modality renderer with ground-truth latents.

    ``latents(split, indices) -> dict`` and ``render(lat) -> (t1, flair)``, each ``[n, 1, I, I, I]``.
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        if cfg.variant not in VARIANTS:
            raise ValueError(f"synth.variant must be one of {VARIANTS}, got {cfg.variant!r}")
        if cfg.image % cfg.lattice:
            raise ValueError("synth.image must be a multiple of synth.lattice")
        self.cfg = cfg
        self.L, self.m = cfg.lattice, cfg.image // cfg.lattice
        d = cfg.dims
        self.d = dict(g=d["g"], r=d["r"], s=d["s"], w=d["w"], site=d["site"], lesion=1, style=d["style"])
        self.channels = sum(self.d.values())
        if self.channels > self.m**3:
            raise ValueError(f"{self.channels} latent channels exceed the {self.m ** 3} voxels of a cell")
        self.regions_np = voronoi_regions(self.L, cfg.n_regions, derive_seed("regions", cfg.weights_seed))
        self.n_regions = int(self.regions_np.max())
        self.register_buffer("regions", torch.from_numpy(self.regions_np))
        gen = torch.Generator().manual_seed(derive_seed("weights", cfg.weights_seed))
        self.register_buffer("site_emb", torch.randn(cfg.n_sites, d["site"], generator=gen))
        a = torch.randn(d["r"], d["g"], generator=gen)
        self.register_buffer("A", cfg.dependent_scale * a / a.norm(dim=1, keepdim=True))
        self.register_buffer("V", torch.randn(cfg.n_sites, d["g"], generator=gen))
        self.decoders = nn.ModuleList()
        for k in range(2):
            with seeded(derive_seed("decoder", cfg.weights_seed, k)):
                net = nn.Sequential(
                    nn.Conv3d(self.channels, cfg.decoder_hidden, 3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(cfg.decoder_hidden, self.m**3, 1),
                )
                for layer in net:
                    if isinstance(layer, nn.Conv3d):
                        nn.init.orthogonal_(layer.weight)
                        nn.init.normal_(layer.bias, std=0.1)
            self.decoders.append(net)
        self.register_buffer("out_mean", torch.zeros(2))
        self.register_buffer("out_std", torch.ones(2))
        self.requires_grad_(False)
        self.to(device)
        self._fit_reference()

    @property
    def device(self):
        return self.site_emb.device

    def feature_labels(self, stride: int) -> np.ndarray:
        """Region map on the encoder's feature grid (``image / stride`` per axis), by repeating lattice cells."""
        grid = self.cfg.image // stride
        if grid % self.L:
            raise ValueError(f"feature grid {grid} is not a multiple of the lattice {self.L}")
        f = grid // self.L
        return np.repeat(np.repeat(np.repeat(self.regions_np, f, 0), f, 1), f, 2)

    # ------------------------------------------------------------------ latents
    def _one(self, split: str, index: int) -> dict:
        cfg, L, d = self.cfg, self.L, self.d
        gen = torch.Generator().manual_seed(derive_seed("subject", split, int(index)))
        g = torch.randn(d["g"], generator=gen)
        r_noise = torch.randn(self.n_regions, d["r"], generator=gen)
        s = smooth_noise(1, d["s"], L, cfg.s_width, gen)[0]
        w = smooth_noise(1, d["w"], L, cfg.s_width, gen)[0] * math.sqrt(cfg.w_variance)
        site_uniform = int(torch.randint(cfg.n_sites, (1,), generator=gen))
        gumbel = -torch.log(-torch.log(torch.rand(cfg.n_sites, generator=gen).clamp(1e-12, 1 - 1e-12)))
        style = torch.randn(2, d["style"], generator=gen)
        lo, hi = cfg.lesion_blobs
        n_blobs = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        centres = torch.rand(hi, 3, generator=gen) * L
        radii = cfg.lesion_radius[0] + torch.rand(hi, generator=gen) * (cfg.lesion_radius[1] - cfg.lesion_radius[0])
        eta_r = smooth_noise(1, d["r"], L, cfg.s_width, gen)[0]
        eta_g = smooth_noise(1, d["g"], L, cfg.s_width, gen)[0]

        r = r_noise + (g @ self.A.cpu().T if cfg.variant == "dependent" else 0)
        if cfg.variant == "site_confounded":
            site = int(torch.argmax(cfg.confound_strength * (self.V.cpu() @ g) + gumbel))
        else:
            site = site_uniform
        cell = torch.stack(torch.meshgrid(*[torch.arange(L) + 0.5] * 3, indexing="ij"), -1)
        lesion = torch.zeros(L, L, L)
        for c, rad in zip(centres[:n_blobs], radii[:n_blobs]):
            lesion = torch.maximum(lesion, ((cell - c).norm(dim=-1) <= rad).float())
        return dict(
            g=g, r=r, s=s, w=w, site=torch.tensor(site), style=style, lesion=lesion[None], eta_r=eta_r, eta_g=eta_g
        )

    def latents(self, split: str, indices) -> dict:
        """Batched ground truth for subjects ``indices`` of ``split``: g [n,4], r [n,R,4], s [n,4,L,L,L],
        w [n,1,L,L,L], lesion [n,1,L,L,L], site [n], site_emb [n,2], style [n,2,2], plus the eta fields."""
        items = [self._one(split, i) for i in indices]
        lat = {k: torch.stack([it[k] for it in items]).to(self.device) for k in items[0]}
        lat["site_emb"] = self.site_emb[lat["site"]]
        return lat

    # ------------------------------------------------------------------ rendering
    def field(self, lat: dict, modality: str, site=None) -> torch.Tensor:
        """Latent field ``[n, 18, L, L, L]`` for one modality; ``site`` overrides the site (counterfactual)."""
        cfg, L = self.cfg, self.L
        n = lat["g"].shape[0]
        expand = lambda v: v[:, :, None, None, None].expand(-1, -1, L, L, L)  # noqa: E731
        g = expand(lat["g"])
        r = lat["r"][:, self.regions - 1].permute(0, 4, 1, 2, 3)  # [n, d_r, L, L, L]
        if cfg.variant == "noisy":
            tau = math.sqrt(cfg.noise_tau2)
            g = g + tau * lat["eta_g"]
            r = r + tau * lat["eta_r"]
        site_emb = self.site_emb[lat["site"] if site is None else site]
        lesion = lat["lesion"]
        if cfg.variant == "one_modality_lesion" and modality == "t1":
            lesion = torch.zeros_like(lesion)
        style = lat["style"][:, MODALITIES.index(modality)]
        parts = [g, r, lat["s"], lat["w"], expand(site_emb), lesion, expand(style)]
        out = torch.cat(parts, 1)
        assert out.shape == (n, self.channels, L, L, L)
        return out

    def _raw(self, lat, modality, site=None):
        k = MODALITIES.index(modality)
        return pixel_shuffle(self.decoders[k](self.field(lat, modality, site)), self.m)

    @torch.no_grad()
    def render(self, lat: dict, site=None):
        """``(t1, flair)``, each ``[n, 1, I, I, I]``, normalised by the fixed reference statistics."""
        return tuple((self._raw(lat, m, site) - self.out_mean[k]) / self.out_std[k] for k, m in enumerate(MODALITIES))

    @torch.no_grad()
    def _fit_reference(self, n: int = 128):
        lat = self.latents("reference", range(n))
        for k, m in enumerate(MODALITIES):
            raw = self._raw(lat, m)
            self.out_mean[k], self.out_std[k] = raw.mean(), raw.std()


class SyntheticSource:
    """The training loop's view of the generator: batches of images, the feature-grid region map, splits."""

    def __init__(self, synth_cfg, stride: int, device="cpu"):
        self.gen = ThreeLevelGenerator(synth_cfg, device=device)
        self.stride = stride
        self.feature_labels = self.gen.feature_labels(stride)
        self.feature_mask = None  # every position is "brain"
        self.sizes = {"train": synth_cfg.n_train, "val": synth_cfg.n_val, "test": synth_cfg.n_test}

    @property
    def image_shape(self):
        return (self.gen.cfg.image,) * 3

    def batch(self, split: str, indices) -> dict:
        lat = self.gen.latents(split, list(indices))
        t1, flair = self.gen.render(lat)
        return dict(t1=t1, flair=flair, latents=lat, subject=torch.as_tensor(list(indices)))
