#!/usr/bin/env python
"""How far does one latent cell reach in the decoded volume? — the resolution ceiling on
every per-position identifiability claim.

Two separate arguments in this project make claims about the latent field *per position*,
and both are bounded by the same number:

  * patch-interior (multi-view CRL): at latent patch side ``P`` the claim holds only on
    the patch interior, with an unidentified boundary shell of thickness ``rho``.  The
    usable fraction in 3D is ``((P - 2 rho) / P)**3``.
  * pointwise mixing (spatial nonlinear ICA): the generative model assumes the mixing acts
    at a single index, i.e. ``rho == 0`` exactly.

``rho`` is a property of the decoder, not of the latent grid.  This script measures it:

  0. guard scan — any spatially global operation (attention, adaptive pooling, spatially
     reduced normalization) makes ``rho`` unbounded outright;
  1. theoretical receptive field from the architecture (see ``probe.receptive_field``);
  2. forward Jacobian energy ``E_u[v] = sum_c |d x[v] / d z_q[c,u]|**2`` by Hutchinson JVP;
  3. reverse footprint: the latent cells one output voxel depends on, by VJP;
  4. stationarity across anatomical location and grid boundary;
  5. interior fraction vs candidate patch side;
  6. controls — non-overlapping (patch-embed) upsampling, which must return ``rho ~ 0``
     or the measurement itself is wrong.

Arms.  ``full`` is the decoder as built and is the headline.  ``frozen_norm`` repeats the
measurement with every normalization statistic pinned to its value at the operating point,
which removes the global mean/variance path and leaves the convolutional Jacobian.  The
difference between the two IS the cost of using a spatially reduced normalization; it is
reported, not hidden.

Usage:
    python -m probe.jacobian_spread --run-dir results/synthetic/<run> --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import probe.receptive_field as rfmod

logger = logging.getLogger(__name__)

QUANTILES = (0.50, 0.95, 0.99)
STRATA = ("periventricular", "deep_wm", "cortical_ribbon", "grid_boundary")
INTERIOR_STRATA = ("periventricular", "deep_wm", "cortical_ribbon")


# --------------------------------------------------------------------------------------
# normalization handling
# --------------------------------------------------------------------------------------


class FrozenGroupNorm3d(nn.Module):
    """GroupNorm whose (mu, sigma) are pinned to their value at the operating point.

    At the calibration point this reproduces the original layer exactly, so the decoder's
    output is unchanged.  Its Jacobian, however, drops the ``d mu / d z`` and
    ``d sigma / d z`` terms — precisely the paths that couple every output voxel to every
    input voxel.  Measuring both arms separates "the convolutions reach this far" from
    "the normalization reaches everywhere".
    """

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        self.weight = gn.weight
        self.bias = gn.bias
        self.register_buffer("mu", torch.zeros(1, gn.num_groups, 1))
        self.register_buffer("rstd", torch.ones(1, gn.num_groups, 1))
        self.calibrating = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        xg = x.reshape(b, self.num_groups, -1)
        if self.calibrating:
            mu = xg.mean(-1, keepdim=True)
            var = xg.var(-1, unbiased=False, keepdim=True)
            self.mu = mu[:1].detach().clone()
            self.rstd = (var[:1] + self.eps).rsqrt().detach().clone()
        y = (xg - self.mu) * self.rstd
        y = y.reshape(b, c, *x.shape[2:])
        if self.weight is not None:
            shape = (1, -1) + (1,) * (x.dim() - 2)
            y = y * self.weight.reshape(shape) + self.bias.reshape(shape)
        return y


def _set_calibrating(module: nn.Module, flag: bool) -> None:
    for m in module.modules():
        if isinstance(m, FrozenGroupNorm3d):
            m.calibrating = flag


def freeze_norm_statistics(decoder: nn.Module, run_forward) -> nn.Module:
    """Copy ``decoder``, swap every GroupNorm for a frozen one, calibrate at the operating point."""
    frozen = copy.deepcopy(decoder)
    for name, m in list(frozen.named_modules()):
        if isinstance(m, nn.GroupNorm):
            rfmod._set_submodule(frozen, name, FrozenGroupNorm3d(m))
    _set_calibrating(frozen, True)
    with torch.no_grad():
        run_forward(frozen)
    _set_calibrating(frozen, False)
    return frozen


# --------------------------------------------------------------------------------------
# control architectures
# --------------------------------------------------------------------------------------


def make_control_decoder(decoder: nn.Module, variant: str) -> nn.Module:
    """Locality controls built from the trained decoder with the geometry changed.

    ``nonoverlap``  — transposed convs go from (k=4, s=2, p=1) to (k=s, p=0), so each
                      input cell writes its own non-overlapping block.  Everything else,
                      including the 3x3x3 convolutions, is untouched.
    ``patch_embed`` — additionally every 3x3x3 convolution becomes 1x1x1.  The decoder is
                      then literally a per-cell patch decoder: one latent cell writes
                      exactly its own s**3 output block and nothing else.  This is the
                      arm the measurement is validated against.

    Weights are inherited from the trained model rather than reinitialised — the centre
    tap for the 1x1x1 convs, the non-overlapping core for the transposed convs — so the
    operating point stays as close to the real decoder as the geometry allows.
    """
    if variant not in ("nonoverlap", "patch_embed"):
        raise ValueError(f"unknown control variant {variant!r}")
    ctrl = copy.deepcopy(decoder)

    for name, m in list(ctrl.named_modules()):
        if isinstance(m, nn.ConvTranspose3d):
            k = rfmod._triple(m.kernel_size)
            s = rfmod._triple(m.stride)
            p = rfmod._triple(m.padding)
            if k == s and p == (0, 0, 0):
                continue
            if any(p[a] + s[a] > k[a] for a in range(3)):
                raise ValueError(f"{name}: cannot carve a non-overlapping kernel out of k={k}, s={s}, p={p}")
            new = nn.ConvTranspose3d(m.in_channels, m.out_channels, s, stride=s, padding=0, bias=m.bias is not None)
            with torch.no_grad():
                new.weight.copy_(
                    m.weight[:, :, p[0] : p[0] + s[0], p[1] : p[1] + s[1], p[2] : p[2] + s[2]]  # noqa: E203
                )
                if m.bias is not None:
                    new.bias.copy_(m.bias)
            rfmod._set_submodule(ctrl, name, new)
        elif variant == "patch_embed" and isinstance(m, nn.Conv3d):
            k = rfmod._triple(m.kernel_size)
            if k == (1, 1, 1):
                continue
            new = nn.Conv3d(m.in_channels, m.out_channels, 1, stride=1, padding=0, bias=m.bias is not None)
            with torch.no_grad():
                new.weight.copy_(
                    m.weight[:, :, k[0] // 2 : k[0] // 2 + 1, k[1] // 2 : k[1] // 2 + 1, k[2] // 2 : k[2] // 2 + 1]
                )  # noqa: E203,E501
                if m.bias is not None:
                    new.bias.copy_(m.bias)
            rfmod._set_submodule(ctrl, name, new)
    return ctrl


def disable_inplace(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


# --------------------------------------------------------------------------------------
# Jacobian energy
# --------------------------------------------------------------------------------------


def _jvp(fn, primal: torch.Tensor, tangent: torch.Tensor):
    try:
        return torch.func.jvp(fn, (primal,), (tangent,))
    except (AttributeError, RuntimeError):
        return torch.autograd.functional.jvp(fn, primal, tangent, create_graph=False, strict=False)


def forward_energy(fn, z: torch.Tensor, site, n_draws: int, gen: torch.Generator) -> torch.Tensor:
    """``E_u[v] = sum_c |d x[v] / d z[c,u]|**2`` by Hutchinson JVP with Rademacher tangents.

    The ``n_draws`` tangents are batched along the batch dimension.  Batch elements are
    independent (every normalization here is per sample), so one JVP call yields all draws.
    """
    c = z.shape[1]
    zb = z.expand(n_draws, *z.shape[1:]).contiguous()
    tangent = torch.zeros_like(zb)
    signs = torch.randint(0, 2, (n_draws, c), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    tangent[:, :, site[0], site[1], site[2]] = signs
    _, jv = _jvp(fn, zb, tangent)
    return (jv**2).sum(dim=1).mean(dim=0).detach()


def forward_energy_exact(fn, z: torch.Tensor, site, chunk: int = 16) -> torch.Tensor:
    """The same energy computed exactly, one basis tangent per latent channel.

    Only used to check that the ``n_draws`` estimator has converged; too expensive to be
    the default (it costs ``C_z`` JVPs per site instead of one).
    """
    c = z.shape[1]
    total = None
    for start in range(0, c, chunk):
        stop = min(start + chunk, c)
        n = stop - start
        zb = z.expand(n, *z.shape[1:]).contiguous()
        tangent = torch.zeros_like(zb)
        for i in range(n):
            tangent[i, start + i, site[0], site[1], site[2]] = 1.0
        _, jv = _jvp(fn, zb, tangent)
        part = (jv**2).sum(dim=1).sum(dim=0).detach()
        total = part if total is None else total + part
    return total


def reverse_energy(fn, z: torch.Tensor, voxel, n_draws: int, gen: torch.Generator) -> torch.Tensor:
    """``E_v[u] = sum_c |d x[v] / d z[c,u]|**2`` — the latent footprint of one output voxel.

    A random cotangent supported on the single output voxel ``v`` is pulled back to the
    latent grid.  With a single output channel the Rademacher draw only flips the overall
    sign, so the squared pullback is deterministic and one draw is exact.
    """
    zb = z.expand(n_draws, *z.shape[1:]).contiguous().requires_grad_(True)
    out = fn(zb)
    cot = torch.zeros_like(out)
    signs = torch.randint(0, 2, (n_draws, out.shape[1]), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    cot[:, :, voxel[0], voxel[1], voxel[2]] = signs
    (grad,) = torch.autograd.grad(out, zb, cot, retain_graph=False)
    return (grad**2).sum(dim=1).mean(dim=0).detach()


# --------------------------------------------------------------------------------------
# radial reduction
# --------------------------------------------------------------------------------------


def _chebyshev_radius_map(shape, centre, device) -> torch.Tensor:
    axes = [torch.arange(shape[a], dtype=torch.float32, device=device) - float(centre[a]) for a in range(3)]
    r = axes[0].abs().reshape(-1, 1, 1)
    r = torch.maximum(r, axes[1].abs().reshape(1, -1, 1))
    r = torch.maximum(r, axes[2].abs().reshape(1, 1, -1))
    return r


def radial_quantiles(energy: torch.Tensor, centre, quantiles=QUANTILES) -> dict:
    """Smallest Chebyshev radius enclosing each energy fraction, plus the hard support radius.

    No smoothing, thresholding or clipping: the energy map goes in as measured.
    """
    r = _chebyshev_radius_map(energy.shape, centre, energy.device).reshape(-1)
    e = energy.reshape(-1)
    total = float(e.sum())
    out = {"total_energy": total, "support_voxel_fraction": float((e > 0).float().mean())}
    if total <= 0:
        # A probe can land where the last ReLU is dead; the Jacobian there is exactly
        # zero and carries no radius.  Flagged rather than counted as "perfectly local".
        out.update({f"r_{int(q * 100)}": float("nan") for q in quantiles})
        out["r_support"] = float("nan")
        return out
    order = torch.argsort(r)
    r_sorted = r[order]
    cum = torch.cumsum(e[order].double(), dim=0)
    for q in quantiles:
        idx = int(torch.searchsorted(cum, torch.tensor(q * total, dtype=torch.float64)).item())
        idx = min(idx, r_sorted.numel() - 1)
        out[f"r_{int(q * 100)}"] = float(r_sorted[idx])
    nz = e > 0
    out["r_support"] = float(r[nz].max())
    return out


def per_axis_quantiles(energy: torch.Tensor, centre, quantiles=QUANTILES) -> dict:
    """Per-axis radii from the marginal energy, for anisotropic volumes."""
    out = {}
    total = float(energy.sum())
    for a in range(3):
        marg = energy.sum(dim=tuple(i for i in range(3) if i != a))
        coord = torch.arange(energy.shape[a], dtype=torch.float32, device=energy.device) - float(centre[a])
        order = torch.argsort(coord.abs())
        cum = torch.cumsum(marg[order], dim=0)
        d_sorted = coord.abs()[order]
        for q in quantiles:
            idx = int(torch.searchsorted(cum, torch.tensor(q * total)).item())
            idx = min(idx, d_sorted.numel() - 1)
            out.setdefault(f"r_{int(q * 100)}", []).append(float(d_sorted[idx]))
    return out


def radial_histogram(energy: torch.Tensor, centre, bin_width: float, n_bins: int) -> np.ndarray:
    """Energy summed into Chebyshev-radius bins — the aligned profile averaged over subjects.

    Bins are *centred* on multiples of ``bin_width`` rather than left-aligned.  With an
    integer stride the site centre falls on a half-integer grid, so every realisable
    radius is a multiple of 0.5 and lands exactly on a bin centre: the binned quantiles
    then carry no rounding bias relative to the exact per-site radii.
    """
    r = _chebyshev_radius_map(energy.shape, centre, energy.device).reshape(-1)
    idx = torch.clamp(torch.round(r / bin_width).long(), min=0, max=n_bins - 1)
    hist = torch.zeros(n_bins, dtype=torch.float64)
    hist.scatter_add_(0, idx, energy.reshape(-1).double())
    return hist.numpy()


def quantiles_from_histogram(hist: np.ndarray, bin_width: float, quantiles=QUANTILES) -> dict:
    total = hist.sum()
    out = {"total_energy": float(total)}
    if not np.isfinite(total) or total <= 0:
        out.update({f"r_{int(q * 100)}": float("nan") for q in quantiles})
        out["r_support"] = float("nan")
        return out
    cum = np.cumsum(hist)
    for q in quantiles:
        idx = int(np.searchsorted(cum, q * total))
        idx = min(idx, len(hist) - 1)
        out[f"r_{int(q * 100)}"] = float(idx * bin_width)
    nz = np.nonzero(hist > 0)[0]
    out["r_support"] = float(nz[-1] * bin_width) if len(nz) else 0.0
    return out


# --------------------------------------------------------------------------------------
# site selection
# --------------------------------------------------------------------------------------


def tissue_fractions(tissue: torch.Tensor, latent_shape) -> dict[str, torch.Tensor]:
    """Per-latent-cell occupancy of each synthetic tissue label."""
    labels = {"csf": 1, "wm": 2, "gm": 3}
    out = {}
    for name, lab in labels.items():
        ind = (tissue == lab).float()[None, None]
        out[name] = F.adaptive_avg_pool3d(ind, latent_shape)[0, 0]
    out["brain"] = F.adaptive_avg_pool3d((tissue > 0).float()[None, None], latent_shape)[0, 0]
    return out


def stratify_sites(frac: dict[str, torch.Tensor], latent_shape, n_per_stratum: int, gen: torch.Generator, margin=2):
    """Pick latent sites per anatomical stratum, plus a grid-boundary set measured separately."""
    grid = torch.stack(torch.meshgrid(*[torch.arange(s) for s in latent_shape], indexing="ij"), -1)
    on_boundary = torch.zeros(latent_shape, dtype=torch.bool)
    for a in range(3):
        on_boundary |= grid[..., a] < margin
        on_boundary |= grid[..., a] >= latent_shape[a] - margin

    masks = {
        "periventricular": (frac["csf"] > 0.02) & (frac["brain"] > 0.8) & ~on_boundary,
        "deep_wm": (frac["wm"] > 0.85) & (frac["csf"] <= 0.02) & ~on_boundary,
        "cortical_ribbon": (frac["gm"] > 0.25) & (frac["brain"] < 0.98) & ~on_boundary,
        "grid_boundary": on_boundary & (frac["brain"] > 0.05),
    }
    if not bool(masks["grid_boundary"].any()):
        masks["grid_boundary"] = on_boundary

    chosen = {}
    for name, m in masks.items():
        idx = torch.nonzero(m, as_tuple=False)
        if idx.numel() == 0:
            chosen[name] = []
            continue
        n = min(n_per_stratum, idx.shape[0])
        sel = torch.randperm(idx.shape[0], generator=gen)[:n]
        chosen[name] = [tuple(int(v) for v in idx[i]) for i in sel]
    return chosen


# --------------------------------------------------------------------------------------
# interior fraction
# --------------------------------------------------------------------------------------


def interior_fraction(P: float, rho: float) -> float:
    if not np.isfinite(rho):
        return float("nan")
    return max(0.0, (P - 2.0 * rho) / P) ** 3


def smallest_patch_side(target: float, rho: float) -> float:
    """``P >= 2 rho / (1 - target**(1/3))`` — the closed form for a target interior fraction."""
    denom = 1.0 - target ** (1.0 / 3.0)
    if denom <= 0:
        return float("inf")
    return 2.0 * rho / denom


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def capture_decoder_inputs(model, x, level: int):
    """Run the model and grab exactly the tensors decoder ``level`` was handed."""
    cap = {}
    dec = model.decoders[level]

    def hook(_mod, args, kwargs):
        cap["z"] = args[0].detach()
        style = kwargs.get("style", None)
        if style is None and len(args) > 1:
            style = args[1]
        cap["style"] = None if style is None else style.detach()

    h = dec.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(x, n_views=x.shape[0])
    finally:
        h.remove()
    return cap["z"], cap.get("style")


def build_arm(decoder, variant, norm_mode, z, style):
    """Materialise one measurement arm: geometry variant x normalization treatment."""
    dec = copy.deepcopy(decoder) if variant is None else make_control_decoder(decoder, variant)
    disable_inplace(dec)
    dec.eval()

    def run(d):
        _ = d(z, style=style) if getattr(d, "style_channels", 0) > 0 else d(z)

    if norm_mode == "frozen":
        dec = freeze_norm_statistics(dec, run)
    return dec


def make_fn(dec, style):
    if getattr(dec, "style_channels", 0) > 0:
        return lambda t: dec(t, style=style.expand(t.shape[0], *style.shape[1:]))
    return lambda t: dec(t)


def git_sha(path="."):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0, help="Decoder level to probe (0 = finest).")
    ap.add_argument("--n-subjects", type=int, default=5)
    ap.add_argument("--n-sites-per-stratum", type=int, default=7)
    ap.add_argument("--n-reverse-voxels", type=int, default=12)
    ap.add_argument("--draws", type=int, default=8, help="Rademacher tangents per site (R).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="", help="Filename prefix, for probing more than one checkpoint.")
    ap.add_argument("--brain-width-mm", type=float, default=140.0, help="Assumed brain width for the mm conversion.")
    ap.add_argument("--exact-check", action="store_true", help="Also compute the exact energy on two sites.")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    torch.set_num_threads(cli.threads)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")  # gradient magnitudes are the measurement: float32, no AMP

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    ckpt = cli.checkpoint or os.path.join(cli.run_dir, "vqvae_best.pt")
    model, args, device = load_model_from_run_dir(cli.run_dir, ckpt, device)
    model.eval()
    decoder = model.decoders[cli.level]
    train_mode_layers = [n for n, m in decoder.named_modules() if m.training]
    norm_note = (
        "GroupNorm and the ChannelLayerNorm3d wrapper have no train/eval distinction: they recompute "
        "their statistics on every forward pass, in training and at inference alike.  eval() therefore "
        "changes nothing about them, and the Jacobian measured here is the inference-time one."
    )

    # settings.json carries no `tag`, but save_dir names the run on the cluster it trained
    # on — the only durable identifier when the checkpoint has been copied elsewhere.
    save_dir = getattr(args, "save_dir", None)
    report: dict = {
        "run_tag": getattr(args, "tag", None) or (os.path.basename(str(save_dir).rstrip("/")) if save_dir else None),
        "training_save_dir": save_dir,
        "run_dir": os.path.abspath(cli.run_dir),
        "checkpoint": os.path.abspath(ckpt),
        "checkpoint_sha256": file_sha256(ckpt),
        "repo_git_sha": git_sha(os.path.dirname(os.path.abspath(__file__)) + "/.."),
        "level": cli.level,
        "n_subjects": cli.n_subjects,
        "R": cli.draws,
        "seed": cli.seed,
        "dtype": "float32",
        "amp": False,
        "model_training_mode": len(train_mode_layers) > 0,
        "normalization_mode_note": norm_note,
    }
    ck_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    report["checkpoint_step"] = ck_meta.get("step")
    report["checkpoint_git_sha"] = ck_meta.get("git_sha")
    if report["checkpoint_git_sha"] is None:
        report["checkpoint_git_sha_note"] = (
            "this checkpoint records no git SHA (utils/checkpointing.py does not store one); "
            "checkpoint_sha256 and checkpoint_step identify it instead"
        )
    del ck_meta

    # ---------------- Step 0: guard -------------------------------------------------
    scan = rfmod.scan_for_global_ops(decoder)
    print("\n" + "=" * 88)
    print("STEP 0  guard scan of the decoder module graph")
    print("=" * 88)
    if scan["ok"]:
        print("  PASS — no global or non-local operation found.")
    else:
        print(f"  FAIL — {len(scan['offenders'])} spatially global operation(s) in the decoder:")
        for o in scan["offenders"]:
            print(f"    {o['module']:32s} {o['reason']}")
        print("\n  Every output voxel therefore depends on every latent cell: rho is UNBOUNDED,")
        print("  and per-position identifiability is invalid regardless of patch size.")
        print("  The measurement below continues in two arms so the size of the effect is on")
        print("  record: 'full' (as built) and 'frozen_norm' (statistics pinned, convolutional")
        print("  path only).  Only 'full' describes the decoder that was trained.")
    report["guard"] = scan

    # ---------------- data + operating point ---------------------------------------
    ds = build_synthetic_test_set(args, max(cli.n_subjects, 8), cache=False, causal=True)
    inner = getattr(ds, "_inner", None)
    gen = torch.Generator().manual_seed(cli.seed)

    subjects = []
    for i in range(cli.n_subjects):
        item = ds[i]
        x = torch.stack([item["image"][0], item["image"][1]], 0).to(device)
        z, style = capture_decoder_inputs(model, x, cli.level)
        z = z[:1].contiguous()
        style = None if style is None else style[:1].contiguous()
        tissue = None
        if inner is not None and hasattr(inner, "renderer"):
            lat = item["gt_latents"]
            with torch.no_grad():
                tissue, _ = inner.renderer.render_structure(
                    lat["z_content"],
                    lat["z_deformation"],
                    lat["z_fissure"],
                    device=torch.device("cpu"),
                    clean=getattr(inner, "clean_content", False),
                )
        subjects.append({"z": z, "style": style, "tissue": tissue, "mask": item["mask"][0][0]})

    latent_shape = tuple(subjects[0]["z"].shape[2:])
    with torch.no_grad():
        probe_out = decoder(subjects[0]["z"], style=subjects[0]["style"])
    out_shape = tuple(probe_out.shape[2:])
    stride = [out_shape[a] / latent_shape[a] for a in range(3)]
    n_latent_channels = subjects[0]["z"].shape[1]

    report["latent_grid"] = list(latent_shape)
    report["output_shape"] = list(out_shape)
    report["stride_per_axis"] = stride
    report["latent_channels"] = n_latent_channels
    print(f"\n  latent grid {latent_shape} x {n_latent_channels}ch  ->  output {out_shape}   stride {stride}")

    # Anatomical scale for the mm conversion.  A synthetic volume has no affine, and
    # `image_spacing` is an ADNI-pipeline argument that survives in the config with its
    # default value — using it here would silently invent a physical scale.  For synthetic
    # data the brain's own measured width is the only honest yardstick.
    brain = subjects[0]["mask"] > 0
    nz = torch.nonzero(brain, as_tuple=False)
    brain_extent_vox = float((nz.max(0).values - nz.min(0).values + 1).float().max()) if nz.numel() else float("nan")
    spacing = getattr(args, "image_spacing", None)
    is_synthetic = str(getattr(args, "dataset_name", "")).lower().startswith("synthetic")
    if spacing and not is_synthetic:
        mm_per_voxel = float(spacing)
        mm_source = f"args.image_spacing = {spacing} mm (real acquisition, affine is meaningful)"
    elif brain_extent_vox == brain_extent_vox:
        mm_per_voxel = cli.brain_width_mm / brain_extent_vox
        mm_source = (
            f"synthetic volume has no affine (args.image_spacing={spacing} is an unused ADNI default); "
            f"calibrated by mapping the measured {brain_extent_vox:g}-voxel brain width onto an assumed "
            f"{cli.brain_width_mm:g} mm brain, giving {mm_per_voxel:.3g} mm/voxel"
        )
    else:
        mm_per_voxel, mm_source = None, "no brain mask and no usable affine; mm conversion unavailable"
    report["mm_per_voxel"] = mm_per_voxel
    report["mm_calibration"] = mm_source
    report["brain_extent_voxels"] = brain_extent_vox
    print(f"  mm scale: {mm_source}")
    report["centre_convention"] = {
        "forward_primary": "c(u) = (u + 0.5) * s",
        "forward_secondary": "the decoder's true output centre, slope*u + offset from probe.receptive_field",
        "reverse_primary": "floor(v / s), the latent cell that owns the probed output voxel",
        "reverse_secondary": "v / s - 0.5, the continuous inverse of c(u)",
        "radius": "Chebyshev (L-infinity), since convolutional receptive fields are boxes",
    }

    # ---------------- Step 1: theoretical RF ---------------------------------------
    style0 = subjects[0]["style"]
    traced = rfmod.trace_leaf_execution(decoder, lambda: decoder(subjects[0]["z"], style=style0))
    rf = rfmod.analytic_receptive_field(traced)
    exact = rfmod.exact_support_halfwidth(
        decoder, latent_shape, n_latent_channels, None if style0 is None else tuple(style0.shape), device=device
    )
    print("\n" + "=" * 88)
    print("STEP 1  theoretical receptive field (convolutional path)")
    print("=" * 88)
    print(rfmod.format_rf_report(scan, rf, exact))
    report["theoretical_rf"] = {
        "half_width_voxels": rf.half_width_voxels,
        "extent_voxels": rf.extent_voxels,
        "half_width_cells": rf.half_width_cells,
        "stride_per_axis": rf.stride,
        "centre_map_slope": [a.centre_slope for a in rf.axes],
        "centre_map_offset": [a.centre_offset for a in rf.axes],
        "exact_support_check": exact,
        "accumulation": rf.steps,
    }
    rho_theory_cells = max(rf.half_width_cells)
    rf_half_vox = max(rf.half_width_voxels)

    # The frozen-norm arm is only interpretable if it sits at the same operating point as
    # the decoder it is derived from — same output, different Jacobian.  Verified, not assumed.
    _live = build_arm(decoder, None, "live", subjects[0]["z"], style0)
    _froz = build_arm(decoder, None, "frozen", subjects[0]["z"], style0)
    with torch.no_grad():
        _a, _b = _live(subjects[0]["z"], style=style0), _froz(subjects[0]["z"], style=style0)
    report["frozen_norm_fidelity"] = {
        "max_abs_diff": float((_a - _b).abs().max()),
        "max_rel_diff": float((_a - _b).abs().max() / _a.abs().max().clamp_min(1e-12)),
        "note": "frozen statistics reproduce the decoder output; only the Jacobian differs",
    }
    print(
        f"\n  frozen-norm arm reproduces the decoder output to {report['frozen_norm_fidelity']['max_abs_diff']:.2e} "
        f"absolute ({report['frozen_norm_fidelity']['max_rel_diff']:.2e} relative)"
    )
    del _live, _froz, _a, _b

    # ---------------- arms -----------------------------------------------------------
    arms = [("full", None, "live"), ("frozen_norm", None, "frozen")]
    if not cli.skip_controls:
        arms += [
            ("control_patch_embed", "patch_embed", "frozen"),
            ("control_patch_embed_live_norm", "patch_embed", "live"),
            ("control_nonoverlap", "nonoverlap", "frozen"),
        ]

    bin_width = 0.5
    n_bins = int(max(out_shape) / bin_width) + 2
    n_bins_lat = int(max(latent_shape) / bin_width) + 2

    results: dict = {}
    t_start = time.time()
    for arm_name, variant, norm_mode in arms:
        per_site = []
        hist = {s: np.zeros(n_bins) for s in STRATA}
        hist_an = np.zeros(n_bins)
        hist_rev = np.zeros(n_bins_lat)
        hist_rev_owner = np.zeros(n_bins_lat)
        rev_per_probe = []
        site_curves = []

        for si, subj in enumerate(subjects):
            z, style = subj["z"], subj["style"]
            dec = build_arm(decoder, variant, norm_mode, z, style)
            fn = make_fn(dec, style)

            if subj["tissue"] is not None:
                frac = tissue_fractions(subj["tissue"], latent_shape)
            else:
                m = F.adaptive_avg_pool3d(subj["mask"][None, None].float(), latent_shape)[0, 0]
                frac = {"brain": m, "wm": m, "gm": m * 0, "csf": m * 0}
            sgen = torch.Generator().manual_seed(cli.seed * 977 + si)
            sites = stratify_sites(frac, latent_shape, cli.n_sites_per_stratum, sgen)

            for stratum, site_list in sites.items():
                for site in site_list:
                    centre = [(site[a] + 0.5) * stride[a] for a in range(3)]
                    centre_analytic = [rf.axes[a].centre_slope * site[a] + rf.axes[a].centre_offset for a in range(3)]
                    e = forward_energy(fn, z, site, cli.draws, sgen)
                    q = radial_quantiles(e, centre)
                    qa = radial_quantiles(e, centre_analytic)
                    pax = per_axis_quantiles(e, centre)
                    beyond = float(
                        e.reshape(-1)[
                            (_chebyshev_radius_map(e.shape, centre, e.device).reshape(-1) > rf_half_vox)
                        ].sum()
                    )
                    per_site.append(
                        {
                            "arm": arm_name,
                            "subject": si,
                            "site": list(site),
                            "stratum": stratum,
                            **{k: v for k, v in q.items()},
                            "r_95_analytic_centre": qa["r_95"],
                            "per_axis": pax,
                            "energy_beyond_theoretical_rf": beyond / max(q["total_energy"], 1e-30),
                        }
                    )
                    hist[stratum] += radial_histogram(e, centre, bin_width, n_bins)
                    if stratum in INTERIOR_STRATA:
                        hist_an += radial_histogram(e, centre_analytic, bin_width, n_bins)
                    if len(site_curves) < 400:
                        site_curves.append((arm_name, stratum, radial_histogram(e, centre, bin_width, n_bins)))

            rgen = torch.Generator().manual_seed(cli.seed * 131 + si)
            interior_sites = [s for st in INTERIOR_STRATA for s in sites[st]]
            picks = interior_sites[: cli.n_reverse_voxels]
            for site in picks:
                # Probe the centre of the output block the site owns.  If the last ReLU is
                # dead there the pullback is identically zero — that voxel depends on
                # nothing and would score a spuriously perfect rho — so move to another
                # voxel of the same block instead of recording the degenerate case.
                base = [int(round((site[a] + 0.5) * stride[a] - 0.5)) for a in range(3)]
                lo = [int(site[a] * stride[a]) for a in range(3)]
                offsets = [(0, 0, 0)] + [
                    (i, j, k)
                    for i in range(int(stride[0]))
                    for j in range(int(stride[1]))
                    for k in range(int(stride[2]))
                ]
                e = q = voxel = None
                for n_try, off in enumerate(offsets):
                    voxel = tuple(base[a] if n_try == 0 else lo[a] + off[a] for a in range(3))
                    voxel = tuple(min(max(voxel[a], 0), out_shape[a] - 1) for a in range(3))
                    e = reverse_energy(fn, z, voxel, 1 if probe_out.shape[1] == 1 else cli.draws, rgen)
                    q = radial_quantiles(e, [voxel[a] / stride[a] - 0.5 for a in range(3)])
                    if q["total_energy"] > 0:
                        break
                lat_centre = [voxel[a] / stride[a] - 0.5 for a in range(3)]
                # The patch-interior argument counts contaminated CELLS around the cell
                # that owns the voxel, so floor(v/s) is the convention-free centre; the
                # continuous inverse of c(u) = (u+0.5)s is kept alongside it because it
                # sits half a cell off and inflates every radius by that much.
                owner = [float(int(voxel[a] // stride[a])) for a in range(3)]
                q_owner = radial_quantiles(e, owner)
                rev_per_probe.append(
                    {
                        "arm": arm_name,
                        "subject": si,
                        "site": list(site),
                        "voxel": list(voxel),
                        "n_try": n_try,
                        "r_95_owner_centre": q_owner["r_95"],
                        **q,
                    }
                )
                hist_rev += radial_histogram(e, lat_centre, bin_width, n_bins_lat)
                hist_rev_owner += radial_histogram(e, owner, bin_width, n_bins_lat)

            del dec
        results[arm_name] = {
            "per_site": per_site,
            "hist": hist,
            "hist_an": hist_an,
            "hist_rev": hist_rev,
            "hist_rev_owner": hist_rev_owner,
            "rev": rev_per_probe,
            "curves": site_curves,
        }
        agg = quantiles_from_histogram(sum(hist.values()), bin_width)
        logger.info("arm %-30s r95=%6.2f vox  (%.0fs elapsed)", arm_name, agg["r_95"], time.time() - t_start)

    # ---------------- Step 2/3 summaries --------------------------------------------
    def summarise(arm_name):
        r = results[arm_name]
        interior_hist = sum(r["hist"][s] for s in INTERIOR_STRATA)
        fwd = quantiles_from_histogram(interior_hist, bin_width)
        fwd_an = quantiles_from_histogram(r["hist_an"], bin_width)
        rev = quantiles_from_histogram(r["hist_rev"], bin_width)
        rev_own = quantiles_from_histogram(r["hist_rev_owner"], bin_width)
        interior_all = [p for p in r["per_site"] if p["stratum"] in INTERIOR_STRATA]
        # A site whose Jacobian is identically zero (the last ReLU is dead over its whole
        # footprint) has no radius.  Counted, then excluded — averaging it in as "0" would
        # report the decoder as more local than it is.
        interior = [p for p in interior_all if p["total_energy"] > 0]
        n_dead = len(interior_all) - len(interior)
        n_dead_rev = sum(1 for p in r["rev"] if not p["total_energy"] > 0)
        by_subject = {}
        for p in interior:
            by_subject.setdefault(p["subject"], []).append(p["r_95"])
        subj_means = [float(np.mean(v)) for v in by_subject.values()]
        per_axis_95 = np.array([p["per_axis"]["r_95"] for p in interior]).mean(axis=0).tolist() if interior else []
        out = {
            "rho_50_voxels": fwd["r_50"],
            "rho_95_voxels": fwd["r_95"],
            "rho_99_voxels": fwd["r_99"],
            "rho_support_voxels": fwd["r_support"],
            "rho_50_cells": fwd["r_50"] / max(stride),
            "rho_95_cells": fwd["r_95"] / max(stride),
            "rho_99_cells": fwd["r_99"] / max(stride),
            "rho_95_mm": (fwd["r_95"] * mm_per_voxel) if mm_per_voxel else None,
            "rho_95_voxels_analytic_centre": fwd_an["r_95"],
            "rho_95_cells_analytic_centre": fwd_an["r_95"] / max(stride),
            "per_axis_r95_voxels": per_axis_95,
            "rho_lat_50_cells": rev_own["r_50"],
            "rho_lat_95_cells": rev_own["r_95"],
            "rho_lat_99_cells": rev_own["r_99"],
            "rho_lat_support_cells": rev_own["r_support"],
            "rho_lat_95_cells_prescribed_centre": rev["r_95"],
            "rho_lat_support_cells_prescribed_centre": rev["r_support"],
            "across_subject_mean_r95_voxels": float(np.mean(subj_means)) if subj_means else None,
            "across_subject_std_r95_voxels": float(np.std(subj_means)) if subj_means else None,
            "energy_beyond_theoretical_rf": (
                float(np.mean([p["energy_beyond_theoretical_rf"] for p in interior])) if interior else None
            ),
            "support_voxel_fraction": (
                float(np.mean([p["support_voxel_fraction"] for p in interior])) if interior else None
            ),
            "latent_support_cell_fraction": (
                float(np.mean([p["support_voxel_fraction"] for p in r["rev"] if p["total_energy"] > 0]))
                if any(p["total_energy"] > 0 for p in r["rev"])
                else None
            ),
            "n_sites_total": len(r["per_site"]),
            "n_interior_sites": len(interior),
            "n_dead_forward_sites": n_dead,
            "n_reverse_probes": len(r["rev"]),
            "n_dead_reverse_probes": n_dead_rev,
        }
        if out["across_subject_mean_r95_voxels"]:
            out["across_subject_cv"] = out["across_subject_std_r95_voxels"] / out["across_subject_mean_r95_voxels"]
        by_stratum = {}
        for st in STRATA:
            h = r["hist"][st]
            if h.sum() > 0:
                qq = quantiles_from_histogram(h, bin_width)
                by_stratum[st] = {"r_95_voxels": qq["r_95"], "r_95_cells": qq["r_95"] / max(stride)}
        out["by_location"] = by_stratum
        vals = [
            v["r_95_voxels"] for k, v in by_stratum.items() if k in INTERIOR_STRATA and np.isfinite(v["r_95_voxels"])
        ]
        if len(vals) >= 2 and min(vals) > 0:
            out["stationarity_ratio"] = max(vals) / min(vals)
            out["stationarity_flag"] = bool(out["stationarity_ratio"] > 1.25)
        return out

    report["arms"] = {a: summarise(a) for a, _, _ in arms}
    report["per_site_detail"] = {a: results[a]["per_site"] for a, _, _ in arms}
    report["reverse_detail"] = {a: results[a]["rev"] for a, _, _ in arms}
    report["radial_profiles"] = {
        "bin_width": bin_width,
        "radius_of_bin_i": "i * bin_width",
        "forward_units": "output voxels",
        "reverse_units": "latent cells",
        "by_arm": {
            a: {
                "forward_interior": sum(results[a]["hist"][st] for st in INTERIOR_STRATA).tolist(),
                "forward_by_stratum": {st: results[a]["hist"][st].tolist() for st in STRATA},
                "reverse_owner_centre": results[a]["hist_rev_owner"].tolist(),
            }
            for a, _, _ in arms
        },
    }

    print("\n" + "=" * 88)
    print("STEP 2/3  empirical Jacobian spread")
    print("=" * 88)
    print("  forward radii in output voxels; rho_95 also in latent cells; reverse rho_lat in cells")
    print(
        f"  {'arm':32s} {'rho50':>6s} {'rho95':>6s} {'rho99':>6s} {'supp':>6s} | {'cells':>6s} "
        f"{'lat95':>6s} {'latsup':>6s} | {'>RF':>7s} {'vox>0':>6s} {'cell>0':>6s}"
    )
    for a, _, _ in arms:
        s = report["arms"][a]
        print(
            f"  {a:32s} {s['rho_50_voxels']:6.2f} {s['rho_95_voxels']:6.2f} {s['rho_99_voxels']:6.2f} "
            f"{s['rho_support_voxels']:6.2f} | {s['rho_95_cells']:6.3f} {s['rho_lat_95_cells']:6.3f} "
            f"{s['rho_lat_support_cells']:6.2f} | {(s['energy_beyond_theoretical_rf'] or 0):7.4f} "
            f"{(s['support_voxel_fraction'] or 0):6.3f} {(s['latent_support_cell_fraction'] or 0):6.3f}"
        )
    print("  '>RF' = energy fraction beyond the theoretical convolutional RF; 'vox>0'/'cell>0' =")
    print("  fraction of output voxels / latent cells with strictly non-zero Jacobian energy.")

    # ---------------- Step 4: stationarity ------------------------------------------
    print("\n" + "=" * 88)
    print("STEP 4  spatial stationarity of rho_95 (voxels)")
    print("=" * 88)
    for a, _, _ in arms:
        s = report["arms"][a]
        loc = " ".join(f"{k}={v['r_95_voxels']:.2f}" for k, v in s["by_location"].items())
        flag = s.get("stationarity_flag")
        print(f"  {a:32s} {loc}")
        if flag is not None:
            verdict = "NON-STATIONARY (>1.25)" if flag else "stationary"
            print(f"  {'':32s} interior max/min = {s['stationarity_ratio']:.3f} -> {verdict}")

    # ---------------- Step 5: interior fraction -------------------------------------
    grid_side = max(latent_shape)
    candidate_P = [2, 4, 8, 16, 32, grid_side]
    candidate_P = sorted(set(candidate_P))
    table = {}
    for a, _, _ in arms:
        rho = report["arms"][a]["rho_lat_95_cells"]
        rows = [
            {"P": P, "interior_fraction": interior_fraction(P, rho), "exceeds_grid": P > grid_side} for P in candidate_P
        ]
        need = {}
        for target in (0.5, 0.75, 0.9):
            p_req = smallest_patch_side(target, rho)
            need[str(target)] = {
                "P_required": p_req,
                "exceeds_grid": (p_req > grid_side) or not np.isfinite(p_req),
            }
        table[a] = {"rho_lat_95_cells": rho, "rows": rows, "required_P": need}
    report["interior_fraction"] = {"grid_side": grid_side, "candidate_P": candidate_P, "by_arm": table}

    # ---------------- acceptance checks + verdict -----------------------------------
    full = report["arms"]["full"]
    frozen = report["arms"]["frozen_norm"]
    checks = {
        "empirical_rho95_le_theoretical_rf": {
            "value": full["rho_95_voxels"],
            "bound": rf_half_vox,
            "pass": full["rho_95_voxels"] <= rf_half_vox + 1e-6,
            "note": "with a spatially global normalization in the graph the convolutional RF is not a bound",
        },
        "frozen_norm_rho95_le_theoretical_rf": {
            "value": frozen["rho_95_voxels"],
            "bound": rf_half_vox,
            "pass": frozen["rho_95_voxels"] <= rf_half_vox + 1e-6,
        },
        "across_subject_std_under_20pct": {
            "value": full.get("across_subject_cv"),
            "std": full.get("across_subject_std_r95_voxels"),
            "mean": full.get("across_subject_mean_r95_voxels"),
            "pass": (full.get("across_subject_cv") or 1.0) <= 0.20,
        },
        "forward_reverse_agree": {
            "rho_95_cells": full["rho_95_cells"],
            "rho_lat_95_cells": full["rho_lat_95_cells"],
            "ratio": (full["rho_95_cells"] / full["rho_lat_95_cells"]) if full["rho_lat_95_cells"] > 0 else None,
            "pass": full["rho_lat_95_cells"] > 0 and 1 / 1.5 <= full["rho_95_cells"] / full["rho_lat_95_cells"] <= 1.5,
            "note": (
                "forward and reverse are transposes of the same operator, so they should agree up to the "
                "site/voxel sampling.  A large gap means a bug or a genuinely asymmetric architecture."
            ),
        },
        "closed_form_self_test": {
            "P_for_interior_0.9_at_rho_1": smallest_patch_side(0.9, 1.0),
            "expected_approx": 58.0,
            "pass": abs(smallest_patch_side(0.9, 1.0) - 57.95) < 0.1,
            "note": "tau = 0.1 must give P ~ 58*rho",
        },
    }
    if "control_patch_embed" in report["arms"]:
        c = report["arms"]["control_patch_embed"]
        checks["control_rho95_below_half_cell"] = {
            "value": c["rho_95_cells_analytic_centre"],
            "value_prescribed_centre": c["rho_95_cells"],
            "pass": c["rho_95_cells_analytic_centre"] < 0.5,
            "note": (
                "measured about the decoder's true output centre (slope*u + offset).  The prescribed "
                "c(u) = (u+0.5)*s sits half an output voxel away from it, which adds exactly 0.5 voxels "
                "to every radius and is the whole difference between the two numbers here."
            ),
        }
        checks["control_rho_lat95_below_half_cell"] = {
            "value": c["rho_lat_95_cells"],
            "value_prescribed_centre": c["rho_lat_95_cells_prescribed_centre"],
            "pass": c["rho_lat_95_cells"] < 0.5,
            "note": "centred on the latent cell that owns the probed voxel, floor(v/s)",
        }
    report["acceptance"] = checks
    print("\n" + "=" * 88)
    print("ACCEPTANCE CHECKS")
    print("=" * 88)
    for name, c in checks.items():
        val = c.get("value", c.get("ratio", c.get("P_for_interior_0.9_at_rho_1")))
        val_s = f"{val:.4g}" if isinstance(val, (int, float)) and val is not None else str(val)
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name:38s} {val_s}")

    if cli.exact_check:
        report["estimator_check"] = run_exact_check(decoder, subjects[0], cli, stride, latent_shape)

    verdict = build_verdict(report, rho_theory_cells, grid_side)
    report["verdict"] = verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    for line in verdict["lines"]:
        print("  " + line)

    # ---------------- outputs --------------------------------------------------------
    os.makedirs(cli.out_dir, exist_ok=True)
    pre = cli.prefix
    json_path = os.path.join(cli.out_dir, f"{pre}rho_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    write_interior_fraction_md(os.path.join(cli.out_dir, f"{pre}interior_fraction.md"), report)
    plot_profiles(os.path.join(cli.out_dir, f"{pre}radial_profile.png"), results, arms, bin_width, stride, report)
    print(f"\n  wrote {json_path}")
    print(f"  wrote {os.path.join(cli.out_dir, pre + 'interior_fraction.md')}")
    print(f"  wrote {os.path.join(cli.out_dir, pre + 'radial_profile.png')}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(str(type(o)))


def build_verdict(report, rho_theory_cells, grid_side):
    full = report["arms"]["full"]
    frozen = report["arms"]["frozen_norm"]
    lines = []
    guard_ok = report["guard"]["ok"]

    if not guard_ok:
        n = len(report["guard"]["offenders"])
        lines.append(
            f"Step 0 FAILED: {n} spatially global operation(s) in the decoder "
            f"({report['guard']['offenders'][0]['class']}). rho is unbounded by construction."
        )
    lines.append(
        f"Theoretical convolutional RF: half-width {report['theoretical_rf']['half_width_voxels'][0]:g} voxels "
        f"= {rho_theory_cells:g} latent cells (grid is {grid_side} cells per axis)."
    )
    lines.append(
        f"Measured (as built): rho_95 = {full['rho_95_voxels']:.2f} voxels = {full['rho_95_cells']:.2f} cells; "
        f"reverse rho_lat,95 = {full['rho_lat_95_cells']:.2f} cells; hard support reaches "
        f"{full['rho_support_voxels']:.1f} voxels."
    )
    if full.get("rho_95_mm"):
        lines.append(
            f"In anatomical units: rho_95 = {full['rho_95_mm']:.1f} mm, against a brain "
            f"{report['brain_extent_voxels'] * report['mm_per_voxel']:.0f} mm across "
            f"({report['mm_calibration']})."
        )
    lines.append(
        f"Hard support: one output voxel has non-zero Jacobian energy on "
        f"{(full['latent_support_cell_fraction'] or 0) * 100:.1f}% of latent cells "
        f"(convolutional path alone: {(frozen['latent_support_cell_fraction'] or 0) * 100:.1f}%). "
        f"{(full['energy_beyond_theoretical_rf'] or 0) * 100:.2f}% of forward energy lies beyond the "
        f"convolutional receptive field, i.e. travels through the normalization."
    )
    lines.append(
        f"Normalization-frozen (convolutional path only): rho_95 = {frozen['rho_95_voxels']:.2f} voxels "
        f"= {frozen['rho_95_cells']:.2f} cells; rho_lat,95 = {frozen['rho_lat_95_cells']:.2f} cells."
    )

    tbl = report["interior_fraction"]["by_arm"]["full"]
    at_grid = [r for r in tbl["rows"] if r["P"] == grid_side]
    if at_grid:
        lines.append(
            f"Interior fraction at the largest patch that fits the grid (P = {grid_side}): "
            f"{at_grid[0]['interior_fraction']:.4f}."
        )
    req = tbl["required_P"]["0.5"]
    lines.append(
        f"Interior fraction 0.5 needs P >= {req['P_required']:.1f} cells"
        + (" — larger than the whole latent grid." if req["exceeds_grid"] else ".")
    )

    supported = (not report["interior_fraction"]["by_arm"]["full"]["required_P"]["0.5"]["exceeds_grid"]) and guard_ok
    if supported:
        headline = "Per-position identifiability is available at the patch sizes tabulated above."
    else:
        headline = (
            "The current decoder cannot support per-position identifiability claims at ANY patch "
            "size that fits the latent grid."
        )
    lines.append("=> " + headline)
    if not guard_ok:
        lines.append(
            "   The pointwise-mixing (rho = 0) assumption is violated outright; the patch-interior "
            "argument has no interior left."
        )
    return {"lines": lines, "supported": supported, "guard_ok": guard_ok}


def write_interior_fraction_md(path, report):
    grid = report["interior_fraction"]["grid_side"]
    Ps = report["interior_fraction"]["candidate_P"]
    by_arm = report["interior_fraction"]["by_arm"]
    lines = [
        "# Interior fraction vs latent patch side",
        "",
        f"`interior_fraction(P) = max(0, (P - 2*rho)/P)^3`, with `rho = rho_lat,95` (latent cells).",
        f"Latent grid: **{grid}** cells per axis — any `P` beyond that does not exist in this model.",
        "",
        "| arm | rho_lat,95 (cells) | " + " | ".join(f"P={p}" for p in Ps) + " |",
        "|---|---|" + "---|" * len(Ps),
    ]
    for arm, d in by_arm.items():
        cells = " | ".join(f"{r['interior_fraction']:.4f}" for r in d["rows"])
        lines.append(f"| `{arm}` | {d['rho_lat_95_cells']:.3f} | {cells} |")
    lines += [
        "",
        "## Smallest patch side reaching a target interior fraction",
        "",
        "`P >= 2*rho / (1 - target^(1/3))`.  At target 0.9 this is `P ~ 58*rho`.",
        "",
        "| arm | 0.50 | 0.75 | 0.90 |",
        "|---|---|---|---|",
    ]
    for arm, d in by_arm.items():
        cells = []
        for t in ("0.5", "0.75", "0.9"):
            r = d["required_P"][t]
            mark = " (> grid)" if r["exceeds_grid"] else ""
            cells.append(f"{r['P_required']:.1f}{mark}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines += ["", "Generated by `probe/jacobian_spread.py`.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_profiles(path, results, arms, bin_width, stride, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colours = {a[0]: c for a, c in zip(arms, ["#c1272d", "#0b5394", "#2e7d32", "#8e44ad", "#e08214"])}
    # Several arms land on identical radii, so colour alone hides curves under each other.
    styles = {a[0]: s for a, s in zip(arms, ["-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1))])}
    widths = {a[0]: w for a, w in zip(arms, [2.6, 2.2, 2.6, 1.8, 1.6])}

    rf_half = report["theoretical_rf"]["half_width_voxels"][0]

    ax = axes[0, 0]
    for arm_name, _, _ in arms:
        for _, _, h in results[arm_name]["curves"][:120]:
            if h.sum() <= 0:
                continue
            ax.plot(np.arange(len(h)) * bin_width, np.cumsum(h) / h.sum(), color=colours[arm_name], alpha=0.07, lw=0.7)
        agg = sum(results[arm_name]["hist"][st] for st in INTERIOR_STRATA)
        ax.plot(
            np.arange(len(agg)) * bin_width,
            np.cumsum(agg) / agg.sum(),
            color=colours[arm_name],
            ls=styles[arm_name],
            lw=widths[arm_name],
            label=f"{arm_name}  (r95 {report['arms'][arm_name]['rho_95_voxels']:.0f})",
        )
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.axvline(rf_half, color="k", ls="--", lw=1, label=f"theoretical RF ({rf_half:g})")
    ax.set_xlabel("Chebyshev radius from site centre (output voxels)")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("Forward: energy vs radius (thin = one probed site)")
    ax.set_xlim(0, None)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[0, 1]
    for arm_name, _, _ in arms:
        agg = sum(results[arm_name]["hist"][st] for st in INTERIOR_STRATA)
        r = np.arange(len(agg)) * bin_width
        m = agg > 0
        ax.semilogy(
            r[m], agg[m] / agg.sum(), color=colours[arm_name], ls=styles[arm_name], lw=widths[arm_name], label=arm_name
        )
    ax.axvline(rf_half, color="k", ls="--", lw=1)
    ax.set_xlabel("Chebyshev radius (output voxels)")
    ax.set_ylabel("energy fraction per radius shell")
    ax.set_title(
        "Forward: radial energy density (log)\nenergy right of the dashed line did not come through the convolutions"
    )
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for arm_name, _, _ in arms:
        h = results[arm_name]["hist_rev_owner"]
        if h.sum() <= 0:
            continue
        ax.plot(
            np.arange(len(h)) * bin_width,
            np.cumsum(h) / h.sum(),
            color=colours[arm_name],
            ls=styles[arm_name],
            lw=widths[arm_name],
            label=f"{arm_name}  (rho_lat,95 {report['arms'][arm_name]['rho_lat_95_cells']:.1f})",
        )
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.set_xlabel("Chebyshev radius from output voxel (latent cells)")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("Reverse: latent footprint of one output voxel")
    ax.set_xlim(0, report["latent_grid"][0])
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, 1]
    grid = report["interior_fraction"]["grid_side"]
    Pgrid = np.arange(2, 4 * grid + 1)
    for arm_name, _, _ in arms:
        rho = report["arms"][arm_name]["rho_lat_95_cells"]
        ax.plot(
            Pgrid,
            [interior_fraction(p, rho) for p in Pgrid],
            color=colours[arm_name],
            ls=styles[arm_name],
            lw=widths[arm_name],
            label=arm_name,
        )
    ax.axvline(grid, color="k", ls="--", lw=1, label=f"latent grid = {grid}")
    for t in (0.5, 0.9):
        ax.axhline(t, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("latent patch side P (cells)")
    ax.set_ylabel("interior fraction")
    ax.set_title("Derived resolution ceiling")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="center right")

    fig.suptitle(
        f"Decoder Jacobian spread — {report.get('run_tag') or os.path.basename(report['run_dir'])} "
        f"(step {report.get('checkpoint_step')}, {report['n_subjects']} subjects, R={report['R']})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_exact_check(decoder, subj, cli, stride, latent_shape):
    """Confirm the R-draw Hutchinson estimate agrees with the exact per-channel Jacobian.

    The exact energy costs one JVP per latent channel per site, so it is a spot check, not
    the default.  What matters is that ``r_95`` from ``R`` random tangents lands on the
    exact value: the headline radii come from a profile pooled over subjects and sites, so
    they carry far less estimator noise than the single-site numbers compared here.
    """
    print("\n" + "=" * 88)
    print("ESTIMATOR CHECK  Hutchinson (R draws) vs exact per-channel Jacobian")
    print("=" * 88)
    z, style = subj["z"], subj["style"]
    dec = build_arm(decoder, None, "live", z, style)
    fn = make_fn(dec, style)
    g = torch.Generator().manual_seed(12345)
    c = latent_shape[0] // 2
    rows = []
    for site in [(c, c, c), (c + 3, c, c), (c, c + 3, c - 2), (c - 3, c + 2, c + 3)]:
        centre = [(site[a] + 0.5) * stride[a] for a in range(3)]
        e_hat = forward_energy(fn, z, site, cli.draws, g)
        e_ex = forward_energy_exact(fn, z, site)
        q_hat, q_ex = radial_quantiles(e_hat, centre), radial_quantiles(e_ex, centre)
        row = {
            "site": list(site),
            "r_95_hutchinson": q_hat["r_95"],
            "r_95_exact": q_ex["r_95"],
            "energy_hutchinson": q_hat["total_energy"],
            "energy_exact": q_ex["total_energy"],
            "r_95_rel_diff": abs(q_hat["r_95"] - q_ex["r_95"]) / max(q_ex["r_95"], 1e-9),
        }
        rows.append(row)
        print(
            f"  site {site}: r95 hutchinson {row['r_95_hutchinson']:.2f} vs exact {row['r_95_exact']:.2f} "
            f"(rel diff {row['r_95_rel_diff']:.3f}); energy {row['energy_hutchinson']:.4g} vs {row['energy_exact']:.4g}"
        )
    return {"R": cli.draws, "n_latent_channels": int(z.shape[1]), "rows": rows}


if __name__ == "__main__":
    sys.exit(main())
