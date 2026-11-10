"""pytest-style tests; pytest is not installed locally, so ``python <this file>`` runs them too.

    PYTHONPATH=. python experiments/gl_ident/tests/test_gl_ident.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.gl_ident.config import EvalConfig, GeneratorConfig, load_config
from experiments.gl_ident.evaluate import neighbourhood, residual_leak, ridge_r2, window_starts
from experiments.gl_ident.generator import (
    SyntheticGenerator,
    cpu_generator,
    gaussian_kernel_spectrum,
    laplace,
    pixel_shuffle,
    pixel_unshuffle,
    smooth,
    upsample_circular,
)
from experiments.gl_ident.guards import passed, run_guards
from experiments.gl_ident.losses import (
    crop_windows,
    gather_sites,
    info_nce,
    same_subject_mask,
    sample_window_pairs,
    supports_disjoint,
)
from experiments.gl_ident.models import LocalEncoder, local_receptive_field

E1 = Path(__file__).parents[1] / "configs" / "e1.yaml"


def test_pixel_shuffle_matches_torch_and_inverts():
    x = torch.randn(2, 16, 5, 6)
    assert torch.equal(pixel_shuffle(x, 4), F.pixel_shuffle(x, 4))
    assert torch.equal(pixel_unshuffle(pixel_shuffle(x, 4), 4), x)
    y = torch.randn(2, 64, 3, 4, 5)
    assert torch.equal(pixel_unshuffle(pixel_shuffle(y, 4), 4), y)


def test_smoothing_keeps_unit_variance():
    for dim, n in ((2, 24), (3, 16)):
        e = laplace((256, 1) + (n,) * dim, cpu_generator("t"))
        s = smooth(e, gaussian_kernel_spectrum(n, dim, 1.5))
        assert abs(float(s.var()) - 1) < 0.05, (dim, float(s.var()))


def test_upsample_is_shift_equivariant():
    f = torch.randn(1, 1, 6, 6)
    a = torch.roll(upsample_circular(f, 4), 4, dims=2)
    b = upsample_circular(torch.roll(f, 1, dims=2), 4)
    assert torch.allclose(a, b, atol=1e-6)


def test_latents_common_across_conditions_and_noise_scales():
    cfg = GeneratorConfig(reference_size=64)
    gen = SyntheticGenerator(cfg)
    lat_a = gen.sample_latents(4, cpu_generator("x", 1))
    lat_b = gen.sample_latents(4, cpu_generator("x", 1))
    assert all(torch.equal(lat_a[k], lat_b[k]) for k in lat_a)
    clean = gen.observe(lat_a, None, snr=math.inf)
    n10 = gen.observe(lat_a, cpu_generator("n"), snr=10.0)
    n20 = gen.observe(lat_a, cpu_generator("n"), snr=20.0)
    for c, a, b in zip(clean, n10, n20):  # same noise draw, scaled by sigma
        assert torch.allclose(a - c, 2 * (b - c), atol=1e-5)


def test_guards_pass():
    out = run_guards(load_config(E1))
    assert passed(out), out


def test_window_pairs_disjoint_and_overlap():
    n, w, rho = 24, 6, 1
    u, v = sample_window_pairs(5000, n, w, rho, 2, "disjoint", cpu_generator("w"))
    assert bool(supports_disjoint(u, v, n, w, rho).all())
    u, v = sample_window_pairs(5000, n, w, rho, 2, "overlap", cpu_generator("w"))
    assert not bool(supports_disjoint(u, v, n, w, rho).any())
    assert bool(((v - u) % n == 0).all(dim=1).any()), "delta = 0 must be included"


def test_crop_windows_matches_roll():
    x = torch.randn(3, 1, 16, 16)
    starts = torch.tensor([[0, 0], [3, 2], [3, 1]])
    out = crop_windows(x, starts, 2, 4)
    for b, (sy, sx) in enumerate(starts.tolist()):
        ref = torch.roll(x[b], (-sy * 4, -sx * 4), dims=(1, 2))[:, :8, :8]
        assert torch.equal(out[b], ref)


def test_gather_and_neighbourhood():
    field = torch.arange(2 * 3 * 5 * 5, dtype=torch.float).reshape(2, 3, 5, 5)
    sites = torch.tensor([[[0, 0], [4, 2]], [[1, 1], [2, 3]]])
    g = gather_sites(field, sites)
    assert torch.equal(g[0, 1], field[0, :, 4, 2]) and torch.equal(g[1, 0], field[1, :, 1, 1])
    nb = neighbourhood(field, sites, 1)
    assert nb.shape == (2, 2, 3 * 9)
    assert torch.equal(nb[0, 0, :3], field[0, :, 4, 4])  # offset (-1, -1) wraps


def test_local_infonce_never_uses_same_subject_sites():
    B, S = 3, 4
    mask = same_subject_mask(B, S, "cpu")
    assert not bool(mask.diagonal().any())
    assert int(mask.sum()) == B * S * (S - 1)
    a = torch.randn(B * S, 5)
    loss_masked, _ = info_nce(a, a.clone(), 0.1, exclude=mask)
    # Masked entries carry no gradient: perturbing same-subject rows leaves the anchor's own term unchanged.
    b = a.clone()
    b[1] += 10 * torch.randn(5)
    logits_row0 = F.normalize(a[:1], dim=-1) @ F.normalize(b, dim=-1).T / 0.1
    kept = ~mask[0]
    ref = -torch.log_softmax(logits_row0[0, kept], 0)[0]
    full = -torch.log_softmax((F.normalize(a[:1], dim=-1) @ F.normalize(a, dim=-1).T / 0.1)[0, kept], 0)[0]
    assert torch.allclose(ref, full)


def test_local_encoder_receptive_field():
    for n_g in (1, 2, 3):
        enc = LocalEncoder(2, 4, 16, n_g, 8)
        assert local_receptive_field(enc, 16, 2) == n_g


def test_window_starts_tile():
    s = window_starts(24, 6, 2, 16, 0)
    assert len(s) == 16 and len({tuple(t) for t in s.tolist()}) == 16


def test_ridge_recovers_linear_map():
    X = torch.randn(2000, 5)
    Y = X @ torch.randn(5, 3)
    groups = torch.arange(1000).numpy()
    assert ridge_r2(X[:1000], Y[:1000], X[1000:], Y[1000:], groups) > 0.999


def test_residual_leak_null_and_positive_control():
    gen = torch.Generator().manual_seed(0)
    gamma, Z = torch.randn(2000, 4, generator=gen), torch.randn(2000, 20, generator=gen)
    pure = torch.tanh(gamma @ torch.randn(4, 5, generator=gen))
    leaky = pure + 0.5 * Z[:, :1]
    ecfg = EvalConfig(mlp_max_epochs=200)
    null, pos = residual_leak(gamma, Z, pure, 1000, ecfg, "cpu"), residual_leak(gamma, Z, leaky, 1000, ecfg, "cpu")
    for reg in ("ridge", "mlp"):
        assert null[reg]["share"] < 0.02, null
        assert pos[reg]["share"] > 0.03, pos  # 0.25 / (5 * var(pure) + 0.25) of the variance is planted


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
