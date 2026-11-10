"""Synthetic two-view generator with continuous global (gamma), local (c) and style (s_k) latents.

    x_k = Noise_k( Nuisance_nu( F_k(gamma, c, s_k) ) )

Every random number is drawn on the CPU from a seeded ``torch.Generator``; the fields are then smoothed and
rendered on the generator's device. Latents depend only on their seed, so every condition (SNR, lambda_nu)
sees the same latents (common random numbers); noise has its own seed and its own stream.
"""

from __future__ import annotations

import zlib
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.gl_ident.config import GeneratorConfig

INTENSITY_OFFSET = 3.0


def derive_seed(*keys) -> int:
    """Stable integer seed from a mix of ints and strings."""
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return int(np.random.SeedSequence(ints).generate_state(1)[0])


def cpu_generator(*keys) -> torch.Generator:
    return torch.Generator().manual_seed(derive_seed(*keys))


@contextmanager
def seeded(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def conv_cls(dim: int):
    return {2: nn.Conv2d, 3: nn.Conv3d}[dim]


def pixel_shuffle(x: torch.Tensor, m: int) -> torch.Tensor:
    """(B, C*m^d, *N) -> (B, C, *(N*m)); equals F.pixel_shuffle for d = 2."""
    B, Cm, *spatial = x.shape
    d = len(spatial)
    C = Cm // m**d
    x = x.reshape(B, C, *([m] * d), *spatial)
    perm = [0, 1]
    for a in range(d):
        perm += [2 + d + a, 2 + a]
    return x.permute(perm).reshape(B, C, *[n * m for n in spatial])


def pixel_unshuffle(x: torch.Tensor, m: int) -> torch.Tensor:
    """Inverse of :func:`pixel_shuffle`."""
    B, C, *spatial = x.shape
    d = len(spatial)
    shape = [B, C]
    for s in spatial:
        shape += [s // m, m]
    x = x.reshape(shape)
    perm = [0, 1] + [3 + 2 * a for a in range(d)] + [2 + 2 * a for a in range(d)]
    return x.permute(perm).reshape(B, C * m**d, *[s // m for s in spatial])


def laplace(shape, gen: torch.Generator) -> torch.Tensor:
    """Unit-variance Laplace (scale 1/sqrt 2) by inverse CDF."""
    u = torch.rand(shape, generator=gen) - 0.5
    return -(0.5**0.5) * torch.sign(u) * torch.log1p(-2 * u.abs()).clamp_min(-50.0)


def gaussian_kernel_spectrum(n: int, dim: int, ell: float) -> torch.Tensor:
    """FFT of a periodic Gaussian kernel (sigma = ell sites) with unit L2 norm.

    Unit L2 norm makes the smoothed field of i.i.d. unit-variance noise unit-variance.
    """
    i = torch.arange(n, dtype=torch.float64)
    k1 = sum(torch.exp(-((i + j * n) ** 2) / (2 * ell**2)) for j in range(-4, 5))
    k = k1
    for _ in range(dim - 1):
        k = k.unsqueeze(-1) * k1
    k = k / k.square().sum().sqrt()
    return torch.fft.fftn(k).real.float()


def smooth(x: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(-spectrum.ndim, 0))
    return torch.fft.ifftn(torch.fft.fftn(x, dim=dims) * spectrum, dim=dims).real


def upsample_circular(field: torch.Tensor, m: int) -> torch.Tensor:
    """Bi/trilinear upsampling by m that wraps around the edges (site centres at (u + 0.5) m - 0.5)."""
    d = field.ndim - 2
    padded = F.pad(field, (1, 1) * d, mode="circular")
    mode = {2: "bilinear", 3: "trilinear"}[d]
    up = F.interpolate(padded, scale_factor=m, mode=mode, align_corners=False)
    crop = (slice(None), slice(None)) + (slice(m, -m),) * d
    return up[crop]


class Mixer(nn.Module):
    """F_k: frozen random local network, one coarse site -> m^dim pixels."""

    def __init__(self, cfg: GeneratorConfig, seed: int):
        super().__init__()
        conv = conv_cls(cfg.dim)
        in_ch = cfg.d_c + cfg.d_s + cfg.d_gamma
        with seeded(seed):
            self.net = nn.Sequential(
                conv(in_ch, cfg.hidden, 1),
                nn.LeakyReLU(0.2),
                conv(cfg.hidden, cfg.hidden, 3, padding=1, padding_mode=cfg.padding),
                nn.LeakyReLU(0.2),
                conv(cfg.hidden, cfg.stride**cfg.dim, 1),
            )
            for layer in self.net:
                if isinstance(layer, conv):
                    nn.init.orthogonal_(layer.weight)
        self.m = cfg.stride
        self.requires_grad_(False)

    def forward(self, c, s, gamma):
        g = gamma.reshape(*gamma.shape, *([1] * (c.ndim - 2))).expand(-1, -1, *c.shape[2:])
        return pixel_shuffle(self.net(torch.cat([c, s, g], dim=1)), self.m)


class SyntheticGenerator(nn.Module):
    def __init__(self, cfg: GeneratorConfig, device="cpu"):
        super().__init__()
        assert cfg.d_c + cfg.d_s + cfg.d_gamma <= cfg.stride**cfg.dim, "channel budget (G6)"
        assert cfg.rho_f == 1, "the single 3x3 conv in F_k fixes rho_f = 1"
        self.cfg = cfg
        self.mixers = nn.ModuleList([Mixer(cfg, derive_seed("mixer", cfg.weights_seed, k)) for k in (1, 2)])
        gen = cpu_generator("coupling", cfg.weights_seed)
        A = torch.randn(cfg.d_c, cfg.d_gamma, generator=gen)
        B = torch.randn(2, cfg.d_s, cfg.d_c, generator=gen)
        self.register_buffer("A", cfg.a_scale * A / A.norm(dim=1, keepdim=True))
        self.register_buffer("B", cfg.b_scale * B / B.norm(dim=2, keepdim=True))
        n, d = cfg.n_sites, cfg.dim
        for name, ell in (("c", cfg.ell_c), ("s", cfg.ell_s), ("nu", cfg.ell_nu)):
            self.register_buffer(f"spec_{name}", gaussian_kernel_spectrum(n, d, ell))
        self.register_buffer("c_mean", torch.zeros(cfg.d_c))
        self.register_buffer("c_std", torch.ones(cfg.d_c))
        self.register_buffer("out_mean", torch.zeros(2))
        self.register_buffer("out_std", torch.ones(2))
        self.to(device)
        self._fit_reference()

    @property
    def device(self):
        return self.A.device

    def _expand(self, v: torch.Tensor) -> torch.Tensor:
        return v.reshape(*v.shape, *([1] * self.cfg.dim))

    @torch.no_grad()
    def _fit_reference(self):
        """Normalisation constants from a fixed reference batch (stored as buffers)."""
        lat = self.sample_latents(
            self.cfg.reference_size, cpu_generator("reference", self.cfg.reference_seed), normalise=False
        )
        dims = (0,) + tuple(range(2, 2 + self.cfg.dim))
        self.c_mean.copy_(lat["c"].mean(dims))
        self.c_std.copy_(lat["c"].std(dims))
        lat = self.sample_latents(self.cfg.reference_size, cpu_generator("reference", self.cfg.reference_seed))
        for k in (1, 2):
            out = torch.cat([self._raw(chunk, k) for chunk in split_latents(lat, 256)])
            self.out_mean[k - 1] = out.mean()
            self.out_std[k - 1] = out.std()

    @torch.no_grad()
    def sample_latents(self, n: int, gen: torch.Generator, normalise: bool = True) -> dict:
        cfg = self.cfg
        grid = (cfg.n_sites,) * cfg.dim
        # Fixed draw order, whatever the config, so latents are common across conditions.
        gamma = torch.randn(n, cfg.d_gamma, generator=gen)
        eps_c = laplace((n, cfg.d_c, *grid), gen)
        eps_s = [laplace((n, cfg.d_s, *grid), gen) for _ in (1, 2)]
        eps_nu = laplace((n, 1, *grid), gen)
        dev = self.device
        gamma = gamma.to(dev)
        c = smooth(eps_c.to(dev), self.spec_c) + self._expand(gamma @ self.A.T)
        if normalise:
            c = (c - self._expand(self.c_mean)) / self._expand(self.c_std)
        lat = dict(gamma=gamma, c=c, nu=smooth(eps_nu.to(dev), self.spec_nu))
        for k in (1, 2):
            s = smooth(eps_s[k - 1].to(dev), self.spec_s)
            if cfg.style_depends_on_content:
                s = s + torch.einsum("sc,nc...->ns...", self.B[k - 1], c)
            lat[f"s{k}"] = s
        return lat

    def _raw(self, lat: dict, k: int) -> torch.Tensor:
        return self.mixers[k - 1](lat["c"], lat[f"s{k}"], lat["gamma"])

    def clean(self, lat: dict, k: int) -> torch.Tensor:
        """Noise- and nuisance-free intensity of view k, (n, 1, *pixels)."""
        return INTENSITY_OFFSET + (self._raw(lat, k) - self.out_mean[k - 1]) / self.out_std[k - 1]

    @torch.no_grad()
    def observe(self, lat: dict, noise_gen: torch.Generator | None, snr=None, lambda_nu=None, noise=None):
        """Both views (n, 1, *pixels). snr / lambda_nu / noise default to the config."""
        cfg = self.cfg
        snr = cfg.snr if snr is None else snr
        lambda_nu = cfg.lambda_nu if lambda_nu is None else lambda_nu
        noise = cfg.noise if noise is None else noise
        views = [self.clean(lat, k) for k in (1, 2)]
        if lambda_nu:
            gain = 1 + lambda_nu * upsample_circular(lat["nu"], cfg.stride)
            views = [v * gain for v in views]
        sigma = 0.0 if snr is None or np.isinf(snr) else 1.0 / snr
        if sigma > 0:
            # n1 for both views first, then n2, so the Gaussian draw equals the Rician n1 draw.
            n1 = [torch.randn(v.shape, generator=noise_gen).to(v.device) for v in views]
            if noise == "gaussian":
                views = [v + sigma * a for v, a in zip(views, n1)]
            elif noise == "rician":
                n2 = [torch.randn(v.shape, generator=noise_gen).to(v.device) for v in views]
                views = [torch.sqrt((v + sigma * a) ** 2 + (sigma * b) ** 2) for v, a, b in zip(views, n1, n2)]
            else:
                raise ValueError(noise)
        return views[0], views[1]

    def split(self, name: str) -> dict:
        """Fixed val / test latents (deterministic in the split seed)."""
        seed, n = {"val": (self.cfg.val_seed, self.cfg.n_val), "test": (self.cfg.test_seed, self.cfg.n_test)}[name]
        return self.sample_latents(n, cpu_generator("split", name, seed))

    def split_images(self, name: str, lat: dict, chunk: int = 250):
        seed = {"val": self.cfg.val_seed, "test": self.cfg.test_seed}[name]
        gen = cpu_generator("split-noise", name, seed)
        pairs = [self.observe(part, gen) for part in split_latents(lat, chunk)]
        return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


def split_latents(lat: dict, chunk: int):
    n = lat["gamma"].shape[0]
    for start in range(0, n, chunk):
        yield {k: v[start : start + chunk] for k, v in lat.items()}


def index_latents(lat: dict, idx) -> dict:
    return {k: v[idx] for k, v in lat.items()}
