"""Registered local/global VICReg, inspired by VICRegL (not a reproduction).

Corresponding positions are paired directly. Local variance and covariance use
SUBJECTS at each position; positions are never folded into the sample axis.
No feature nearest-neighbour matching, crops, anatomical labels or EMA are used.
"""

import math
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn


def vicreg_position_terms(x, y, valid=None, eps=1e-4):
    """B,P,D -> scalar terms, averaged over positions with >=2 valid subjects.

    Invariance sees raw embeddings; only covariance/variance subtract subject
    means. Sample variance/covariance use n-1. Each view contributes half the
    variance/covariance penalty. FP32 and sqrt(var+eps) keep AMP/collapse finite.
    """
    if x.ndim != 3 or x.shape != y.shape or min(x.shape) < 1:
        raise ValueError("Expected matching nonempty (subjects, positions, channels) tensors")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    if valid is None:
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if valid.shape != x.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean (subjects, positions) mask")
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        count = valid.sum(0)
        eligible = count >= 2
        if not bool(eligible.any()):
            zero = (x.sum() + y.sum()) * 0
            return dict(sim=zero, var=zero, cov=zero, std=zero, eligible_fraction=0.0)
        x, y = x[:, eligible], y[:, eligible]
        mask = valid[:, eligible, None].float()
        n = count[eligible].float()
        sim = (((x - y).square() * mask).sum(0) / n[:, None]).mean()
        vars_, covs, stds = [], [], []
        d = x.shape[-1]
        off = ~torch.eye(d, device=x.device, dtype=torch.bool)
        for z in (x, y):
            mean = (z * mask).sum(0) / n[:, None]
            centered = (z - mean) * mask
            variance = centered.square().sum(0) / (n - 1)[:, None]
            std = torch.sqrt(variance + eps)
            vars_.append(F.relu(1 - std).mean())
            stds.append(std.mean())
            covariance = torch.einsum("bpc,bpd->pcd", centered, centered) / (n - 1)[:, None, None]
            covs.append(covariance[:, off].square().sum(-1).mean() / d)
        return dict(
            sim=sim,
            var=sum(vars_) / 2,
            cov=sum(covs) / 2,
            std=sum(stds) / 2,
            eligible_fraction=float(eligible.float().mean()),
        )


class RegisteredVICReg(nn.Module):
    def __init__(
        self,
        channels,
        local_dim=16,
        global_dim=16,
        hidden=64,
        local_weight=1.0,
        global_weight=0.25,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        no_projectors=False,
    ):
        super().__init__()
        if min(channels, local_dim, global_dim, hidden) < 1:
            raise ValueError("Feature dimensions must be positive")
        numbers = (local_weight, global_weight, sim_coeff, std_coeff, cov_coeff)
        if any(not math.isfinite(x) or x < 0 for x in numbers) or local_weight + global_weight <= 0:
            raise ValueError("Weights must be finite, nonnegative, with at least one active arm")
        if std_coeff <= 0:
            raise ValueError("VICReg requires a positive variance coefficient")
        self.channels = channels
        self.local_weight, self.global_weight = local_weight, global_weight
        self.coefficients = dict(sim=sim_coeff, var=std_coeff, cov=cov_coeff)

        def head(dim, active):
            if no_projectors or not active:
                return nn.Identity()
            return nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, dim))

        # Shared across views; separate across local/global objectives. No BatchNorm
        # that could satisfy the variance hinge by normalizing the subject axis.
        self.local_head = head(local_dim, local_weight > 0)
        self.global_head = head(global_dim, global_weight > 0)

    def forward(self, hz, estimated_content_indices=None, subsets=None, soft_content_mask=None, patch_valid=None):
        if hz.ndim != 4 or hz.shape[0] < 2 or hz.shape[1] < 2 or hz.shape[-1] < 1:
            raise ValueError("Registered VICReg needs (views>=2, subjects>=2, channels, positions)")
        if patch_valid is None:
            patch_valid = torch.ones((hz.shape[0], hz.shape[1], hz.shape[3]), dtype=torch.bool, device=hz.device)
        if patch_valid.shape != (hz.shape[0], hz.shape[1], hz.shape[3]) or patch_valid.dtype != torch.bool:
            raise ValueError("patch_valid must be boolean (views, subjects, positions)")
        subsets = subsets if subsets is not None else [list(range(hz.shape[0]))]
        indices = (
            estimated_content_indices
            if estimated_content_indices is not None
            else [list(range(hz.shape[2]))] * len(subsets)
        )
        if len(indices) != len(subsets) or not subsets:
            raise ValueError("Need one content-index list per nonempty view subset")
        total, diagnostics, pairs = hz.sum() * 0, {}, 0
        for subset, idx in zip(subsets, indices):
            if len(subset) < 2 or len(set(subset)) != len(subset):
                raise ValueError("Each subset needs at least two distinct views")
            selected = hz[list(subset)]
            if soft_content_mask is not None:
                selected = selected * soft_content_mask.reshape(1, 1, -1, 1)
            z = selected[:, :, idx, :].permute(0, 1, 3, 2)  # V,B,P,C
            if z.shape[-1] != self.channels:
                raise ValueError(f"Head expects {self.channels} content channels, got {z.shape[-1]}")
            valid = patch_valid[list(subset)]
            local = self.local_head(z) if self.local_weight else None
            if self.global_weight:
                # Foreground-weighted GAP BEFORE its independent nonlinear head.
                pooled = (z * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]
                global_ = self.global_head(pooled)
            for a, b in combinations(range(len(subset)), 2):
                pairs += 1
                for arm, weight in (("local", self.local_weight), ("global", self.global_weight)):
                    if weight == 0:
                        continue
                    if arm == "local":
                        terms = vicreg_position_terms(local[a], local[b], valid[a] & valid[b])
                    else:
                        samples = valid[a].any(-1) & valid[b].any(-1)
                        terms = vicreg_position_terms(global_[a, :, None], global_[b, :, None], samples[:, None])
                    arm_loss = sum(self.coefficients[k] * terms[k] for k in self.coefficients)
                    total = total + weight * arm_loss
                    for key, value in dict(terms, loss=arm_loss, weighted_loss=weight * arm_loss).items():
                        name = f"vicregl_{arm}_{key}"
                        diagnostics[name] = diagnostics.get(name, 0) + float(
                            value.detach() if torch.is_tensor(value) else value
                        )
        total = total / pairs
        total._contrastive_diag = {key: value / pairs for key, value in diagnostics.items()}
        return total


def validate_vicregl_args(args):
    if getattr(args, "contrastive_loss_type", None) != "vicregl":
        return
    if not getattr(args, "patch_contrastive", False):
        raise ValueError("vicregl requires --patch-contrastive and registered corresponding grids")
    if getattr(args, "mask_mode", "fixed") != "fixed" or getattr(args, "vqvae_nb_levels", 1) != 1:
        raise ValueError("Initial vicregl implementation requires a fixed channel split and one VQ level")
    if getattr(args, "use_moco", False):
        raise ValueError("vicregl uses no MoCo queue; disable --use-moco")
    if getattr(args, "contrastive_proj_dim", 0) > 0 or getattr(args, "contrastive_proj_mode", "head") != "head":
        raise ValueError("vicregl owns its local/global heads; disable the generic contrastive projector")
    if getattr(args, "patch_center_mode", "none") != "none":
        raise ValueError(
            "vicregl requires patch_center_mode=none; it centers statistics internally, not alignment inputs"
        )
    if getattr(args, "batch_size", 2) < 2:
        raise ValueError("vicregl needs at least two subjects per batch (not per accumulation window)")
    # Validate numerical options without consuming RNG by constructing projectors.
    RegisteredVICReg(1, **vicregl_kwargs(args, force_no_projectors=True))


def vicregl_kwargs(args, force_no_projectors=False):
    return dict(
        local_dim=getattr(args, "vicregl_local_dim", 16),
        global_dim=getattr(args, "vicregl_global_dim", 16),
        hidden=getattr(args, "vicregl_hidden", 64),
        local_weight=getattr(args, "vicregl_local_weight", 1.0),
        global_weight=getattr(args, "vicregl_global_weight", 0.25),
        sim_coeff=getattr(args, "vicreg_sim_coeff", 25.0),
        std_coeff=getattr(args, "vicreg_std_coeff", 25.0),
        cov_coeff=getattr(args, "vicreg_cov_coeff", 1.0),
        no_projectors=force_no_projectors or getattr(args, "vicregl_no_projectors", False),
    )


def attach_vicregl_heads(model, args):
    """Call before optimizer construction/DP wrapping and before checkpoint loading."""
    if getattr(args, "contrastive_loss_type", None) != "vicregl":
        return
    validate_vicregl_args(args)
    channels = model.content_channels_per_level.get(0)
    if channels is None:
        raise ValueError("vicregl requires content/style separation at level 0")
    model._vicregl_heads = nn.ModuleDict({"L0": RegisteredVICReg(channels, **vicregl_kwargs(args))})


def registered_level_loss(model, level, patch_valid):
    from functools import partial

    return partial(model._vicregl_heads[f"L{level}"], patch_valid=patch_valid)
