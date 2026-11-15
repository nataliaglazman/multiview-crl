"""Synthetic three-level benchmark: global g, regional r_R, super local s(u), with known ground truth.

Extends ``experiments/gl_ident/generator.py`` (its seeding helpers and 3D pixel shuffle) from two levels to
three, with site, per-scan style, a weak super-local factor w and lesions.

Latents live on a lattice of cells (default 16^3); each cell covers an ``m^3`` block of voxels (m = image /
lattice = 4). Latent field per cell, in channel order: [g (4), r_R(u) (4), s(u) (4), w(u) (1), site embedding
(2), lesion l(u) (1), style sigma_k (2)] = 18 channels. Style enters only its own modality.

Two renderers turn the field into images (``synth.renderer``):

``lattice`` (default)
    A fixed random network per modality: a 3^3 conv on the lattice (18 -> hidden), LeakyReLU(0.2), a 1^3 conv
    (hidden -> m^3), then a pixel shuffle, so each cell's 64 outputs fill its own block. Injective by design and
    reaching one cell, but the images are a voxel-level checkerboard. Regions are Voronoi cells of the cube.

``brain``
    A fixed procedural brain (``brain_template.py``: tissue classes, 13 regions) is the same in every subject.
    The latent field is upsampled smoothly (trilinear) to the image grid, and a fixed pointwise network per
    modality turns it into one intensity change per tissue class, so the same latents look different in grey and
    white matter. Each voxel is the partial-volume mix of its tissues' T1 or FLAIR intensities, each scaled by
    ``1 + modulation * change``. Lesions are rendered as soft spheres in white matter (dark in T1, bright in
    FLAIR) instead of entering the field. Site adds a fixed smooth bias field per site, shared by both modalities,
    and a per-site Rician noise level; background is zero, as in skull-stripped data. Regions are the template's.

Every random number of a subject is drawn from a generator seeded by (split, index), in a fixed order whatever
the variant or renderer, so variants share the same draws (common random numbers):

    g, r noise, s, w, site (uniform), site (Gumbel), style, lesion blobs, eta_r, eta_g, noise seed

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
from scipy import ndimage

from experiments.gl_ident.generator import derive_seed, pixel_shuffle, seeded
from mlmv.data.geometry import majority_downsample

VARIANTS = ("exact", "noisy", "one_modality_lesion", "dependent", "site_confounded")
RENDERERS = ("lattice", "brain")
MODALITIES = ("t1", "flair")
BRAIN_TISSUES = ("csf", "gm", "deep_gm", "wm")  # template labels 1..4


def as_shape(v) -> tuple:
    """An int (cube) or a 3-list -> a 3-tuple of ints."""
    return (int(v),) * 3 if isinstance(v, (int, np.integer)) else tuple(int(s) for s in v)


def gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    return torch.exp(-(x**2) / (2 * sigma**2))


def smooth_noise(n: int, channels: int, size, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """Stationary unit-variance Gaussian field ``[n, c, *size]``: white noise * normalised Gaussian.

    The noise is drawn on a padded lattice and convolved without padding, so every cell sees a full kernel.
    """
    size = as_shape(size)
    k1 = gaussian_kernel_1d(sigma)
    r = (len(k1) - 1) // 2
    k = k1[:, None, None] * k1[None, :, None] * k1[None, None, :]
    k = (k / k.square().sum().sqrt()).float()
    white = torch.randn(n * channels, 1, *(s + 2 * r for s in size), generator=gen)
    return F.conv3d(white, k[None, None]).reshape(n, channels, *size)


def voronoi_regions(lattice, n_regions: int, seed: int) -> np.ndarray:
    """``[Lx, Ly, Lz]`` labels 1..n from the Voronoi cells of ``n_regions`` random seeds.

    When n is a perfect cube (8 = 2^3) the seeds are jittered on a k^3 grid (one uniform seed in the central
    half of each block), so cells are random but of similar size; otherwise seeds are drawn uniformly with a
    minimum spacing of 0.6 times the mean cell side.
    """
    shape = np.array(as_shape(lattice), dtype=np.float64)
    rng = np.random.default_rng(seed)
    k = round(n_regions ** (1 / 3))
    if k**3 == n_regions:
        block = shape / k
        corners = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), -1).reshape(-1, 3) * block
        seeds = corners + block * (0.25 + 0.5 * rng.random((n_regions, 3)))
    else:
        spacing = 0.6 * (shape.prod() / n_regions) ** (1 / 3)
        seeds = []
        for _ in range(100000):
            p = rng.random(3) * shape
            if all(np.linalg.norm(p - q) >= spacing for q in seeds):
                seeds.append(p)
                if len(seeds) == n_regions:
                    break
        if len(seeds) < n_regions:
            raise RuntimeError(f"could not place {n_regions} spaced seeds in a {tuple(shape)} lattice")
        seeds = np.array(seeds)
    centres = np.stack(np.meshgrid(*[np.arange(int(s)) + 0.5 for s in shape], indexing="ij"), -1)
    dist = np.linalg.norm(centres[..., None, :] - seeds, axis=-1)
    return dist.argmin(-1).astype(np.int64) + 1


def soft_spheres(centres, radii, count: int, shape, edge: float = 0.5) -> torch.Tensor:
    """``[*shape]`` union of soft spheres (voxel units; voxel i has centre i + 0.5); 0 where count is 0."""
    out = torch.zeros(shape, device=centres.device)
    if count == 0:
        return out
    axes = [torch.arange(s, device=centres.device, dtype=torch.float32) + 0.5 for s in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    for c, r in zip(centres[:count], radii[:count]):
        out = torch.maximum(out, torch.sigmoid((r - (grid - c).norm(dim=-1)) / edge))
    return out


class ThreeLevelGenerator(nn.Module):
    """Frozen two-modality renderer with ground-truth latents.

    ``latents(split, indices) -> dict`` and ``render(lat) -> (t1, flair)``, each ``[n, 1, X, Y, Z]``.
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        if cfg.variant not in VARIANTS:
            raise ValueError(f"synth.variant must be one of {VARIANTS}, got {cfg.variant!r}")
        if cfg.renderer not in RENDERERS:
            raise ValueError(f"synth.renderer must be one of {RENDERERS}, got {cfg.renderer!r}")
        self.cfg = cfg
        self.image_shape, self.L = as_shape(cfg.image), as_shape(cfg.lattice)
        factors = {i // l for i, l in zip(self.image_shape, self.L)}
        if any(i % l for i, l in zip(self.image_shape, self.L)) or len(factors) != 1:
            raise ValueError(
                f"synth.image {self.image_shape} must be the same multiple of synth.lattice {self.L} per axis"
            )
        self.m = factors.pop()
        d = cfg.dims
        self.d = dict(g=d["g"], r=d["r"], s=d["s"], w=d["w"], site=d["site"], lesion=1, style=d["style"])
        self.channels = sum(self.d.values())
        gen = torch.Generator().manual_seed(derive_seed("weights", cfg.weights_seed))
        self.register_buffer("site_emb", torch.randn(cfg.n_sites, d["site"], generator=gen))
        a = torch.randn(d["r"], d["g"], generator=gen)
        self.register_buffer("A", cfg.dependent_scale * a / a.norm(dim=1, keepdim=True))
        self.register_buffer("V", torch.randn(cfg.n_sites, d["g"], generator=gen))
        if cfg.renderer == "lattice":
            self._init_lattice(cfg)
        else:
            self._init_brain(cfg)
        self.n_regions = int(self.regions_np.max())
        self.register_buffer("regions", torch.from_numpy(self.regions_np))
        self.requires_grad_(False)
        self.to(device)
        self._fit_reference()

    # ------------------------------------------------------------------ construction
    def _init_lattice(self, cfg):
        if self.channels > self.m**3:
            raise ValueError(f"{self.channels} latent channels exceed the {self.m ** 3} voxels of a cell")
        self.regions_np = voronoi_regions(self.L, cfg.n_regions, derive_seed("regions", cfg.weights_seed))
        self.image_regions_np = self.regions_np.repeat(self.m, 0).repeat(self.m, 1).repeat(self.m, 2)
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

    def _init_brain(self, cfg):
        from mlmv.synth.brain_template import build_brain_template

        b = cfg.brain
        tpl = build_brain_template(
            self.image_shape, b["voxel_mm"], seed=derive_seed("template", cfg.weights_seed) % 2**32
        )
        self.template = tpl
        tissue = tpl["tissue"].astype(np.int64)
        self.image_regions_np = tpl["regions"].astype(np.int64)
        self.regions_np = majority_downsample(self.image_regions_np, self.m)
        # partial-volume tissue fractions [5, X, Y, Z] (background first), summing to 1 at every voxel
        frac = np.stack(
            [ndimage.gaussian_filter((tissue == t).astype(np.float32), b["partial_volume"]) for t in range(5)]
        )
        frac /= frac.sum(0, keepdims=True)
        self.register_buffer("tissue_frac", torch.from_numpy(frac))
        self._wm_cpu = torch.from_numpy(frac[4].copy())  # latents are drawn on the CPU
        wm_cells = F.avg_pool3d(torch.from_numpy(frac[4])[None, None], self.m)[0, 0]
        cand = torch.nonzero(wm_cells >= b["lesion_wm_fraction"]).float()
        if len(cand) == 0:
            raise ValueError("no lattice cell is white matter enough to hold a lesion")
        self.lesion_candidates = (cand + 0.5) * self.m  # voxel units, CPU
        mu = [[b[m][t] for t in (*BRAIN_TISSUES, "lesion")] for m in MODALITIES]
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))  # [2, 5]
        # per-site smooth log bias field, shared by both modalities, unit std inside the brain
        gen = torch.Generator().manual_seed(derive_seed("bias", cfg.weights_seed))
        width = b["bias_width_mm"] / (b["voxel_mm"] * self.m)
        low = smooth_noise(cfg.n_sites, 1, self.L, width, gen)
        bias = F.interpolate(low, size=self.image_shape, mode="trilinear", align_corners=False)[:, 0]
        brain = torch.from_numpy(tissue > 0)
        bias = (bias - bias[:, brain].mean(1)[:, None, None, None]) / bias[:, brain].std(1)[:, None, None, None]
        self.register_buffer("log_bias", b["bias_strength"] * bias)
        self.register_buffer("noise_sd", torch.linspace(b["noise"][0], b["noise"][1], cfg.n_sites))
        self.decoders = nn.ModuleList()
        for k in range(2):
            with seeded(derive_seed("brain-decoder", cfg.weights_seed, k)):
                net = nn.Sequential(
                    nn.Conv3d(self.channels, b["decoder_hidden"], 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(b["decoder_hidden"], len(BRAIN_TISSUES), 1),
                )
                for layer in net:
                    if isinstance(layer, nn.Conv3d):
                        nn.init.orthogonal_(layer.weight)
                        nn.init.normal_(layer.bias, std=0.1)
            self.decoders.append(net)
        self.register_buffer("mod_mean", torch.zeros(2, len(BRAIN_TISSUES)))
        self.register_buffer("mod_std", torch.ones(2, len(BRAIN_TISSUES)))
        self.register_buffer("out_scale", torch.ones(2))

    @property
    def device(self):
        return self.site_emb.device

    @property
    def brain(self) -> bool:
        return self.cfg.renderer == "brain"

    def feature_labels(self, stride: int) -> np.ndarray:
        """Region map on the encoder's feature grid (``image / stride`` per axis), 0 outside the brain."""
        if any(s % stride for s in self.image_shape):
            raise ValueError(f"image {self.image_shape} is not divisible by the encoder stride {stride}")
        return majority_downsample(self.image_regions_np, stride)

    def feature_mask(self, stride: int):
        """Positions the S level may sample and the probes may pool over (None: every position)."""
        return self.feature_labels(stride) > 0 if self.brain else None

    # ------------------------------------------------------------------ latents
    def _one(self, split: str, index: int) -> dict:
        cfg, L, d = self.cfg, self.L, self.d
        gen = torch.Generator().manual_seed(derive_seed("subject", split, int(index)))
        g = torch.randn(d["g"], generator=gen)
        r_noise = torch.randn(self.n_regions_for_draw, d["r"], generator=gen)
        s = smooth_noise(1, d["s"], L, cfg.s_width, gen)[0]
        w = smooth_noise(1, d["w"], L, cfg.s_width, gen)[0] * math.sqrt(cfg.w_variance)
        site_uniform = int(torch.randint(cfg.n_sites, (1,), generator=gen))
        gumbel = -torch.log(-torch.log(torch.rand(cfg.n_sites, generator=gen).clamp(1e-12, 1 - 1e-12)))
        style = torch.randn(2, d["style"], generator=gen)
        lo, hi = cfg.lesion_blobs
        n_blobs = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        u_centres = torch.rand(hi, 3, generator=gen)
        radii = cfg.lesion_radius[0] + torch.rand(hi, generator=gen) * (cfg.lesion_radius[1] - cfg.lesion_radius[0])
        eta_r = smooth_noise(1, d["r"], L, cfg.s_width, gen)[0]
        eta_g = smooth_noise(1, d["g"], L, cfg.s_width, gen)[0]
        noise_seed = int(torch.randint(2**31 - 1, (1,), generator=gen))

        r = r_noise + (g @ self.A.cpu().T if cfg.variant == "dependent" else 0)
        if cfg.variant == "site_confounded":
            site = int(torch.argmax(cfg.confound_strength * (self.V.cpu() @ g) + gumbel))
        else:
            site = site_uniform
        if self.brain:
            # lesion centres on white-matter cells, radii in voxels; ground truth = lesion fraction per cell
            cand = self.lesion_candidates
            idx = (u_centres[:, 0] * len(cand)).long().clamp(max=len(cand) - 1)
            centres = (
                cand[idx] + torch.stack([u_centres[:, 1] - 0.5, u_centres[:, 2] - 0.5, torch.zeros(hi)], 1) * self.m
            )
            radii = radii * self.m
            vol = soft_spheres(centres, radii, n_blobs, self.image_shape, self.cfg.brain["lesion_edge"])
            vol = vol * self._wm_cpu
            lesion = F.avg_pool3d(vol[None, None], self.m)[0]
        else:
            centres = u_centres * torch.tensor(L, dtype=torch.float32)
            cell = torch.stack(torch.meshgrid(*[torch.arange(n) + 0.5 for n in L], indexing="ij"), -1)
            lesion = torch.zeros(L)
            for c, rad in zip(centres[:n_blobs], radii[:n_blobs]):
                lesion = torch.maximum(lesion, ((cell - c).norm(dim=-1) <= rad).float())
            lesion = lesion[None]
        return dict(
            g=g,
            r=r,
            s=s,
            w=w,
            site=torch.tensor(site),
            style=style,
            lesion=lesion,
            eta_r=eta_r,
            eta_g=eta_g,
            lesion_count=torch.tensor(n_blobs),
            lesion_centres=centres,
            lesion_radii=radii,
            noise_seed=torch.tensor(noise_seed),
        )

    @property
    def n_regions_for_draw(self) -> int:
        return int(self.regions_np.max())

    def latents(self, split: str, indices) -> dict:
        """Batched ground truth for subjects ``indices`` of ``split``: g [n,4], r [n,R,4], s [n,4,*L],
        w [n,1,*L], lesion [n,1,*L] (lattice: blob indicator; brain: rendered lesion fraction per cell), site [n],
        site_emb [n,2], style [n,2,2], plus the eta fields, the blob parameters and the noise seed."""
        items = [self._one(split, i) for i in indices]
        lat = {k: torch.stack([it[k] for it in items]).to(self.device) for k in items[0]}
        lat["site_emb"] = self.site_emb[lat["site"]]
        return lat

    # ------------------------------------------------------------------ rendering
    def field(self, lat: dict, modality: str, site=None) -> torch.Tensor:
        """Latent field ``[n, 18, *L]`` for one modality; ``site`` overrides the site (counterfactual).

        Cells outside every region (brain background) get r = 0. The brain renderer draws lesions itself, so
        their channel is zero here.
        """
        cfg, L = self.cfg, self.L
        n = lat["g"].shape[0]
        expand = lambda v: v[:, :, None, None, None].expand(-1, -1, *L)  # noqa: E731
        g = expand(lat["g"])
        inside = (self.regions > 0).to(lat["r"].dtype)
        r = lat["r"][:, (self.regions - 1).clamp_min(0)].permute(0, 4, 1, 2, 3) * inside
        if cfg.variant == "noisy":
            tau = math.sqrt(cfg.noise_tau2)
            g = g + tau * lat["eta_g"]
            r = r + tau * lat["eta_r"]
        site_emb = self.site_emb[lat["site"] if site is None else site]
        lesion = lat["lesion"]
        if self.brain or (cfg.variant == "one_modality_lesion" and modality == "t1"):
            lesion = torch.zeros_like(lesion)
        style = lat["style"][:, MODALITIES.index(modality)]
        out = torch.cat([g, r, lat["s"], lat["w"], expand(site_emb), lesion, expand(style)], 1)
        assert out.shape == (n, self.channels, *L)
        return out

    def _raw(self, lat, modality, site=None):
        k = MODALITIES.index(modality)
        return pixel_shuffle(self.decoders[k](self.field(lat, modality, site)), self.m)

    def _modulation(self, field_one, k: int):
        """``[1, 18, *L]`` -> per-tissue intensity change ``[1, 4, X, Y, Z]`` (reference-standardised)."""
        up = F.interpolate(field_one, size=self.image_shape, mode="trilinear", align_corners=False)
        mod = self.decoders[k](up)
        return (mod - self.mod_mean[k][None, :, None, None, None]) / self.mod_std[k][None, :, None, None, None]

    def _brain_one(self, lat, i: int, modality: str, site=None, noise: bool = True, scale: bool = True):
        """One subject's brain image ``[1, X, Y, Z]``."""
        b, k = self.cfg.brain, MODALITIES.index(modality)
        sub = {key: v[i : i + 1] for key, v in lat.items()}
        mod = self._modulation(self.field(sub, modality, None if site is None else site[i : i + 1]), k)[0]
        frac = self.tissue_frac[1:]  # [4, X, Y, Z]
        mu = self.mu[k]
        base = mu[:4, None, None, None] * (1 + b["modulation"] * mod)
        img = (frac * base).sum(0, keepdim=True)
        if not (self.cfg.variant == "one_modality_lesion" and modality == "t1"):
            les = (
                soft_spheres(
                    sub["lesion_centres"][0],
                    sub["lesion_radii"][0],
                    int(sub["lesion_count"][0]),
                    self.image_shape,
                    b["lesion_edge"],
                )
                * self.tissue_frac[4]
            )
            img = img + les[None] * (mu[4] - mu[3]) * (1 + b["modulation"] * mod[3:4])
        sid = int((lat["site"] if site is None else site)[i])
        img = img * torch.exp(self.log_bias[sid])[None]
        if scale:
            img = img / self.out_scale[k]
        if noise:
            gen = torch.Generator().manual_seed(int(sub["noise_seed"][0]) * 2 + k)
            n1, n2 = torch.randn(2, *self.image_shape, generator=gen).to(img.device)
            sd = self.noise_sd[sid]
            img = torch.sqrt((img + sd * n1) ** 2 + (sd * n2) ** 2)
        return img * (1 - self.tissue_frac[0])

    @torch.no_grad()
    def render(self, lat: dict, site=None):
        """``(t1, flair)``, each ``[n, 1, X, Y, Z]``. Lattice: normalised by fixed reference mean and std.
        Brain: divided by a fixed reference scale (brain intensities roughly in [0, 1]), background 0."""
        if self.brain:
            n = lat["g"].shape[0]
            return tuple(torch.stack([self._brain_one(lat, i, m, site) for i in range(n)]) for m in MODALITIES)
        return tuple((self._raw(lat, m, site) - self.out_mean[k]) / self.out_std[k] for k, m in enumerate(MODALITIES))

    @torch.no_grad()
    def _fit_reference(self, n: int = 128):
        if not self.brain:
            lat = self.latents("reference", range(n))
            for k, m in enumerate(MODALITIES):
                raw = self._raw(lat, m)
                self.out_mean[k], self.out_std[k] = raw.mean(), raw.std()
            return
        lat = self.latents("reference", range(16))
        brain = self.tissue_frac[0] < 0.5
        for k, m in enumerate(MODALITIES):
            mods = torch.stack(
                [
                    self._modulation(self.field({key: v[i : i + 1] for key, v in lat.items()}, m), k)[0][:, brain]
                    for i in range(len(lat["g"]))
                ]
            )  # [n, 4, voxels], with the identity standardisation
            self.mod_mean[k] = mods.mean((0, 2))
            self.mod_std[k] = mods.std((0, 2))
            high = [
                torch.quantile(self._brain_one(lat, i, m, noise=False, scale=False)[0][brain][::7], 0.99)
                for i in range(len(lat["g"]))
            ]
            self.out_scale[k] = torch.stack(high).mean()


class SyntheticSource:
    """The training loop's view of the generator: batches of images, the feature-grid region map, splits."""

    def __init__(self, synth_cfg, stride: int, device="cpu"):
        self.gen = ThreeLevelGenerator(synth_cfg, device=device)
        self.stride = stride
        self.feature_labels = self.gen.feature_labels(stride)
        self.feature_mask = self.gen.feature_mask(stride)
        self.sizes = {"train": synth_cfg.n_train, "val": synth_cfg.n_val, "test": synth_cfg.n_test}

    @property
    def image_shape(self):
        return self.gen.image_shape

    def batch(self, split: str, indices) -> dict:
        lat = self.gen.latents(split, list(indices))
        t1, flair = self.gen.render(lat)
        return dict(t1=t1, flair=flair, latents=lat, subject=torch.as_tensor(list(indices)))
