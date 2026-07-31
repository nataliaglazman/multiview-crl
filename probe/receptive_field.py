"""Analytic receptive field of a decoder stack, and the guard scan that decides whether
that number means anything at all.

Both per-position identifiability arguments used in this project need a *hard support*
bound on how far one latent cell's influence reaches in the output volume:

  * patch-interior (multi-view CRL) — per-position claims hold only on the interior of a
    latent patch of side ``P``, with an unidentified boundary shell of thickness ``rho``;
  * pointwise mixing (spatial nonlinear ICA) — assumes ``rho == 0`` exactly.

``analytic_receptive_field`` is the textbook accumulation over conv / transposed-conv /
pool / upsample layers.  ``exact_support_halfwidth`` re-derives the same quantity
numerically by pushing a single non-zero latent cell through an all-positive-weight
surrogate of identical geometry, where the support is exactly the reachable set (no sign
cancellation is possible).  The two must agree; disagreement means the accumulation is
wrong, not that the architecture is subtle.

Both numbers describe the CONVOLUTIONAL path only.  A layer that reduces over spatial
dimensions — GroupNorm, InstanceNorm, spatial LayerNorm, adaptive pooling, attention —
couples every output position to every input position, so no finite receptive field
bounds it.  ``scan_for_global_ops`` finds those layers.  If it returns anything, the
receptive field below is not a bound on the decoder and must not be reported as one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Modules that mix information across spatial positions without a bounded kernel.  Each
# entry is (predicate, reason).  A single hit invalidates every per-position claim.
_ATTENTION_NAMES = ("attention", "attn", "nonlocal", "non_local", "transformer")


def _triple(v) -> tuple[int, int, int]:
    if isinstance(v, (tuple, list)):
        return tuple(int(x) for x in v)
    return (int(v), int(v), int(v))


def classify_module(m: nn.Module) -> tuple[str, str]:
    """Return ``(kind, reason)`` for one leaf module.

    kind is one of:
      ``"local"``     — bounded kernel, contributes to the receptive field;
      ``"pointwise"`` — acts at a single position, contributes nothing;
      ``"global"``    — reduces or mixes over spatial dims, receptive field unbounded;
      ``"unknown"``   — not recognised; treated as global, because assuming locality is
                        the failure mode that silently validates a false claim.
    """
    cls = type(m).__name__
    lowered = cls.lower()

    if any(tok in lowered for tok in _ATTENTION_NAMES):
        return "global", f"{cls}: attention/non-local block mixes all positions"
    if isinstance(m, nn.MultiheadAttention):
        return "global", f"{cls}: attention mixes all positions"
    if isinstance(m, nn.Linear):
        return "global", f"{cls}: fully-connected layer (spans whatever axis it is applied to)"
    if isinstance(m, (nn.AdaptiveAvgPool3d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
        return "global", f"{cls}: adaptive pooling reduces over spatial extent"
    if isinstance(m, (nn.AdaptiveMaxPool3d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool1d)):
        return "global", f"{cls}: adaptive pooling reduces over spatial extent"
    if isinstance(m, nn.GroupNorm):
        return "global", (
            f"{cls}(groups={m.num_groups}, channels={m.num_channels}): normalization statistics "
            "are reduced over ALL spatial positions, so every output voxel depends on every input voxel"
        )
    if isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm2d, nn.InstanceNorm1d)):
        return "global", f"{cls}: per-sample statistics reduced over all spatial positions"
    if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.training or not m.track_running_stats:
            return "global", f"{cls}: batch statistics reduced over all spatial positions (not in running-stat mode)"
        return "pointwise", f"{cls}: frozen running statistics, acts as a per-channel affine"
    if isinstance(m, nn.LayerNorm):
        # nn.LayerNorm over the trailing channel axis only (the ChannelLayerNorm3d idiom)
        # is per-voxel and therefore local.  Over more than one axis it is reducing over
        # something spatial.
        if len(m.normalized_shape) == 1:
            return "pointwise", f"{cls}(normalized_shape={tuple(m.normalized_shape)}): per-voxel over channels"
        return "global", f"{cls}(normalized_shape={tuple(m.normalized_shape)}): reduces over more than the channel axis"

    if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        return "local", f"{cls}(kernel={_triple(m.kernel_size)}, stride={_triple(m.stride)})"
    if isinstance(m, (nn.ConvTranspose3d, nn.ConvTranspose2d, nn.ConvTranspose1d)):
        return "local", f"{cls}(kernel={_triple(m.kernel_size)}, stride={_triple(m.stride)})"
    if isinstance(m, (nn.AvgPool3d, nn.MaxPool3d, nn.AvgPool2d, nn.MaxPool2d)):
        return "local", f"{cls}(kernel={_triple(m.kernel_size)}, stride={_triple(m.stride)})"
    if isinstance(m, nn.Upsample):
        return "local", f"{cls}(scale_factor={m.scale_factor}, mode={m.mode})"

    if isinstance(
        m,
        (
            nn.ReLU,
            nn.LeakyReLU,
            nn.GELU,
            nn.SiLU,
            nn.ELU,
            nn.Tanh,
            nn.Sigmoid,
            nn.Identity,
            nn.Dropout,
            nn.Dropout3d,
        ),
    ):
        return "pointwise", cls
    return "unknown", f"{cls}: unrecognised module, cannot certify locality"


def _pointwise_norm_prefixes(root: nn.Module) -> set[str]:
    """Names of wrapper modules whose internals are already known to be per-voxel.

    ``ChannelLayerNorm3d`` permutes channels last and applies ``nn.LayerNorm`` over them.
    Its inner ``nn.LayerNorm`` is classified correctly on its own, but the wrapper is the
    unit worth reporting, so the inner module is suppressed to avoid double-counting.
    """
    out = set()
    for name, m in root.named_modules():
        if type(m).__name__ == "ChannelLayerNorm3d":
            out.add(name)
    return out


def scan_for_global_ops(root: nn.Module) -> dict[str, Any]:
    """Walk ``root`` and report every operation that is not receptive-field-bounded."""
    wrappers = _pointwise_norm_prefixes(root)
    offenders: list[dict[str, str]] = []
    inventory: dict[str, int] = {}

    for name, m in root.named_modules():
        if name == "":
            continue
        if any(name.startswith(w + ".") for w in wrappers):
            continue
        if name in wrappers:
            kind, reason = "pointwise", "ChannelLayerNorm3d: normalizes over channels within each voxel"
        else:
            if len(list(m.children())) > 0:
                continue
            kind, reason = classify_module(m)
        inventory[kind] = inventory.get(kind, 0) + 1
        if kind in ("global", "unknown"):
            offenders.append({"module": name, "class": type(m).__name__, "kind": kind, "reason": reason})

    return {"ok": len(offenders) == 0, "offenders": offenders, "inventory": inventory}


@dataclass
class TracedLayer:
    name: str
    cls: str
    kind: str
    module: nn.Module
    in_shape: tuple
    out_shape: tuple


def trace_leaf_execution(root: nn.Module, run_forward) -> list[TracedLayer]:
    """Record the leaf modules ``root`` actually executes, in order, with their shapes.

    Execution order is recovered from forward hooks rather than assumed from
    ``named_modules()``, so branching (style-injection mode, FiLM) is handled as it
    actually runs.  Residual branches appear as a straight chain: for support purposes
    that is correct, because the skip path is a subset of the convolution path.
    """
    traced: list[TracedLayer] = []
    handles = []

    def make_hook(name, m):
        def hook(_mod, inp, out):
            kind, _ = classify_module(m)
            traced.append(
                TracedLayer(
                    name=name,
                    cls=type(m).__name__,
                    kind=kind,
                    module=m,
                    in_shape=tuple(inp[0].shape) if isinstance(inp, tuple) and torch.is_tensor(inp[0]) else (),
                    out_shape=tuple(out.shape) if torch.is_tensor(out) else (),
                )
            )

        return hook

    for name, m in root.named_modules():
        if name == "" or len(list(m.children())) > 0:
            continue
        handles.append(m.register_forward_hook(make_hook(name, m)))
    try:
        with torch.no_grad():
            run_forward()
    finally:
        for h in handles:
            h.remove()
    return traced


@dataclass
class AxisRF:
    half_width: float
    extent: float
    scale: float
    centre_slope: float
    centre_offset: float


@dataclass
class RFResult:
    axes: list[AxisRF] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)

    @property
    def half_width_voxels(self) -> list[float]:
        return [a.half_width for a in self.axes]

    @property
    def extent_voxels(self) -> list[float]:
        return [a.extent for a in self.axes]

    @property
    def stride(self) -> list[float]:
        return [a.scale for a in self.axes]

    @property
    def half_width_cells(self) -> list[float]:
        return [a.half_width / a.scale for a in self.axes]


def analytic_receptive_field(traced: list[TracedLayer], ndim: int = 3) -> RFResult:
    """Accumulate the receptive field of one latent cell through the traced layer chain.

    Tracked per axis, in units of the CURRENT feature grid:
      ``half``   half-width of the support of a single latent cell's influence,
      ``scale``  grid cells per latent cell so far,
      ``slope``/``offset``  the affine map latent index -> support centre.

    Returned half-widths and extents are in OUTPUT voxels; divide by ``scale`` for latent
    cells.  ``extent = 2 * half + 1`` is the full box side.
    """
    half = [0.0] * ndim
    scale = [1.0] * ndim
    slope = [1.0] * ndim
    offset = [0.0] * ndim
    steps: list[dict] = []

    for layer in traced:
        m = layer.module
        if layer.kind in ("pointwise", "global"):
            continue

        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.AvgPool3d, nn.MaxPool3d, nn.AvgPool2d, nn.MaxPool2d)):
            k = _triple(m.kernel_size)
            s = _triple(m.stride)
            p = _triple(m.padding if not isinstance(m.padding, str) else 0)
            d = _triple(getattr(m, "dilation", 1))
            for a in range(ndim):
                reach = d[a] * (k[a] - 1) / 2.0
                half[a] = (half[a] + reach) / s[a]
                scale[a] = scale[a] / s[a]
                slope[a] = slope[a] / s[a]
                offset[a] = (offset[a] + p[a] - reach) / s[a]
        elif isinstance(m, (nn.ConvTranspose3d, nn.ConvTranspose2d, nn.ConvTranspose1d)):
            k = _triple(m.kernel_size)
            s = _triple(m.stride)
            p = _triple(m.padding)
            d = _triple(getattr(m, "dilation", 1))
            for a in range(ndim):
                reach = d[a] * (k[a] - 1) / 2.0
                half[a] = s[a] * half[a] + reach
                scale[a] = scale[a] * s[a]
                slope[a] = slope[a] * s[a]
                offset[a] = offset[a] * s[a] - p[a] + reach
        elif isinstance(m, nn.Upsample):
            f = _triple(m.scale_factor if m.scale_factor is not None else 1)
            linear = str(m.mode) in ("linear", "bilinear", "trilinear", "bicubic")
            for a in range(ndim):
                # nearest: one input owns exactly f output cells.  linear: the stencil of
                # an output position spans two source samples, so one input reaches f
                # further on each side.
                half[a] = f[a] * half[a] + (f[a] if linear else (f[a] - 1) / 2.0)
                scale[a] = scale[a] * f[a]
                slope[a] = slope[a] * f[a]
                offset[a] = offset[a] * f[a] + (f[a] - 1) / 2.0
        else:
            continue

        steps.append(
            {
                "layer": layer.name,
                "class": layer.cls,
                "half_width": list(half),
                "scale": list(scale),
            }
        )

    res = RFResult(steps=steps)
    for a in range(ndim):
        res.axes.append(
            AxisRF(
                half_width=half[a],
                extent=2 * half[a] + 1,
                scale=scale[a],
                centre_slope=slope[a],
                centre_offset=offset[a],
            )
        )
    return res


def _set_submodule(root: nn.Module, qualified: str, new: nn.Module) -> None:
    parts = qualified.split(".")
    parent = root
    for p in parts[:-1]:
        parent = (
            parent[int(p)] if isinstance(parent, (nn.Sequential, nn.ModuleList)) and p.isdigit() else getattr(parent, p)
        )
    last = parts[-1]
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def ones_surrogate(decoder: nn.Module) -> nn.Module:
    """A geometry-identical copy of ``decoder`` with all-positive weights and no norms.

    Every conv weight is set to 1 and every bias to 0, normalizations and nonlinearities
    become the identity, and ReZero gates are opened (``alpha = 1``).  Feeding a single
    non-zero latent cell then makes the set of non-zero outputs EXACTLY the set of output
    voxels the architecture can reach — every contribution is positive, so nothing can
    cancel.  This is the ground truth the analytic accumulation is checked against.
    """
    sur = copy.deepcopy(decoder)
    for name, m in list(sur.named_modules()):
        if name == "":
            continue
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            with torch.no_grad():
                m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.zero_()
        elif type(m).__name__ == "ChannelLayerNorm3d" or isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm3d)):
            _set_submodule(sur, name, nn.Identity())
        elif isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)):
            _set_submodule(sur, name, nn.Identity())
    for name, p in sur.named_parameters():
        if name.endswith("alpha"):
            with torch.no_grad():
                p.fill_(1.0)
    return sur


def exact_support_halfwidth(
    decoder: nn.Module,
    latent_shape: tuple[int, int, int],
    n_latent_channels: int,
    style_shape: tuple | None,
    site: tuple[int, int, int] | None = None,
    device=None,
) -> dict[str, Any]:
    """Numerically exact hard support of one latent cell, via the all-ones surrogate."""
    device = device or torch.device("cpu")
    sur = ones_surrogate(decoder).to(device).eval()
    site = site or tuple(s // 2 for s in latent_shape)

    z = torch.zeros(1, n_latent_channels, *latent_shape, device=device)
    z[0, :, site[0], site[1], site[2]] = 1.0
    kwargs = {}
    if getattr(decoder, "style_channels", 0) > 0:
        kwargs["style"] = torch.zeros(*style_shape, device=device)
    with torch.no_grad():
        out = sur(z, **kwargs)
    energy = out.abs().sum(dim=(0, 1))
    nz = torch.nonzero(energy > 0, as_tuple=False)
    if nz.numel() == 0:
        return {"reachable": False}
    lo = nz.min(0).values.tolist()
    hi = nz.max(0).values.tolist()
    centre = [(lo[a] + hi[a]) / 2.0 for a in range(3)]
    half = [(hi[a] - lo[a]) / 2.0 for a in range(3)]
    return {
        "reachable": True,
        "site": list(site),
        "bbox_lo": lo,
        "bbox_hi": hi,
        "centre": centre,
        "half_width_voxels": half,
        "extent_voxels": [hi[a] - lo[a] + 1 for a in range(3)],
        "output_shape": list(out.shape[2:]),
    }


def format_rf_report(scan: dict, rf: RFResult, exact: dict | None) -> str:
    lines = ["theoretical receptive field (convolutional path only)"]
    for a, axis in enumerate(rf.axes):
        lines.append(
            f"  axis {a}: half-width {axis.half_width:g} output voxels | extent {axis.extent:g} | "
            f"stride {axis.scale:g} | rho_theory {axis.half_width / axis.scale:g} latent cells"
        )
    if exact and exact.get("reachable"):
        lines.append(f"  exact support half-width (all-ones surrogate): {exact['half_width_voxels']} output voxels")
    if not scan["ok"]:
        lines.append("  NOTE: the decoder contains spatially global operations; the numbers above")
        lines.append("        bound the convolutional path ONLY and are not a bound on the decoder.")
    return "\n".join(lines)
