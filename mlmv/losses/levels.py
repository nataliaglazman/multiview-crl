"""Per-level pairs and losses: S at single positions, L over region halves, G over region sets.

Inputs are the template-subtracted block features of both modalities, ``{"t1": {"G": [B,d,P], ...}, ...}``
with the spatial grid flattened to P, plus one ``ViewPairSampler.sample`` draw.

=====  ==========================================  ==========================  =============================
level  positive pair (view a = T1, view b = FLAIR)  centring context            InfoNCE negatives
=====  ==========================================  ==========================  =============================
S      S_T1(u), S_FL(u)                            same subject and region     same subject and region, > 2 rho
L      mean L_T1 over half A, mean L_FL over B;    same subject                the other regions' halves,
       and the swapped pair                                                    same subject
G      mean G_T1 over set 1, mean G_FL over set 2; none                        other subjects (batch + queue)
       and the swapped pair
=====  ==========================================  ==========================  =============================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlmv.losses.barlow import BarlowLoss
from mlmv.losses.infonce import Queue, g_negative_mask, info_nce, l_negative_mask, s_negative_mask
from mlmv.models.heads import Expander

LEVELS = ("G", "L", "S")


def _swap_pairs(t1, fl):
    """``[B, K, 2, d]`` pooled codes -> rows (T1 h0, FL h1) then (T1 h1, FL h0): ``[B, 2K, d]`` each."""
    return torch.cat([t1[:, :, 0], t1[:, :, 1]], 1), torch.cat([fl[:, :, 1], fl[:, :, 0]], 1)


def _no_autocast(x):
    """Pooling sums tens of thousands of positions; keep it out of bf16."""
    return torch.autocast(device_type=x.device.type if x.device.type in ("cuda", "cpu") else "cpu", enabled=False)


def gather_s(x, idx):
    """``x [B, d, P]``, ``idx [B, n]`` -> ``[B * n, d]``."""
    b, d, _ = x.shape
    return torch.gather(x, 2, idx[:, None, :].expand(-1, d, -1)).permute(0, 2, 1).reshape(-1, d)


def pool_segments(x, flat, seg, n_seg_per_subject, batch_size):
    """Mean of ``x [B, d, P]`` over each segment; segment s belongs to subject ``s // n_seg_per_subject``."""
    b, d, p = x.shape
    with _no_autocast(x):
        rows = x.float().permute(0, 2, 1).reshape(b * p, d)
        subject = seg // n_seg_per_subject
        picked = rows[subject * p + flat]
        n_seg = batch_size * n_seg_per_subject
        sums = torch.zeros(n_seg, d, device=x.device, dtype=picked.dtype).index_add_(0, seg, picked)
        counts = torch.bincount(seg, minlength=n_seg).to(sums.dtype).clamp_min(1)
        return sums / counts[:, None]


def pool_sets(x, membership, member):
    """``x [B, d, P]``; ``membership [P, R]`` eroded regions; ``member [B, K, 2, R]`` -> ``[B, K, 2, d]``."""
    with _no_autocast(x):
        region_sum = x.float() @ membership  # [B, d, R]
        counts = membership.sum(0)  # [R]
        m = member.float()
        return torch.einsum("bkhr,bdr->bkhd", m, region_sum) / (m @ counts)[..., None]


def build_pairs(feats, sample, membership, n_l: int):
    """Per level: ``(za [N, d], zb [N, d], meta)`` with meta holding subject, region, context, coords."""
    dev = feats["t1"]["S"].device
    batch = feats["t1"]["S"].shape[0]
    out = {}

    idx = sample["S_idx"].to(dev)
    n_s = idx.shape[1]
    region = sample["S_region"].to(dev).flatten()
    subject = torch.arange(batch, device=dev).repeat_interleave(n_s)
    out["S"] = (
        gather_s(feats["t1"]["S"], idx),
        gather_s(feats["flair"]["S"], idx),
        dict(
            subject=subject,
            region=region,
            context=subject * (int(region.max()) + 1) + region,
            coords=sample["S_coord"].to(dev).reshape(-1, 3),
        ),
    )

    flat, seg = sample["L_flat"].to(dev), sample["L_seg"].to(dev)
    pooled = {m: pool_segments(feats[m]["L"], flat, seg, 2 * n_l, batch).reshape(batch, n_l, 2, -1) for m in feats}
    za, zb = _swap_pairs(pooled["t1"], pooled["flair"])
    subject = torch.arange(batch, device=dev).repeat_interleave(2 * n_l)
    region = torch.cat([sample["L_region"], sample["L_region"]], 1).to(dev).flatten()
    out["L"] = (
        za.reshape(-1, za.shape[-1]),
        zb.reshape(-1, zb.shape[-1]),
        dict(subject=subject, region=region, context=subject),
    )

    member = sample["G_member"].to(dev)
    pooled = {m: pool_sets(feats[m]["G"], membership, member) for m in feats}
    za, zb = _swap_pairs(pooled["t1"], pooled["flair"])
    n_g = member.shape[1]
    subject = torch.arange(batch, device=dev).repeat_interleave(2 * n_g)
    out["G"] = (za.reshape(-1, za.shape[-1]), zb.reshape(-1, zb.shape[-1]), dict(subject=subject, context=None))
    return out


def variance_floor(z, eps: float = 1e-4):
    return F.relu(1 - torch.sqrt(z.float().var(0) + eps)).mean()


class LevelObjective(nn.Module):
    """Expanders and per-level losses. ``forward(feats, sample, subject_ids) -> (losses dict, logs dict, pairs)``."""

    def __init__(self, cfg, dims: dict, membership, rho: float, metric: str):
        super().__init__()
        self.cfg, self.rho, self.metric = cfg, rho, metric
        width = cfg.model.expander[-1]
        self.expanders = nn.ModuleDict({lvl: Expander(dims[lvl], cfg.model.expander) for lvl in LEVELS})
        self.objective = cfg.loss.objective
        if self.objective == "barlow":
            self.losses = nn.ModuleDict(
                {
                    lvl: BarlowLoss(
                        width,
                        cfg.loss.lambda_off,
                        cfg.loss.centring,
                        cfg.loss.standardise,
                        cfg.loss.buffer.momentum,
                        cfg.loss.corr_ema,
                    )
                    for lvl in LEVELS
                }
            )
        elif self.objective == "infonce":
            self.queue = Queue(cfg.loss.infonce.queue, width)
        else:
            raise ValueError(f"loss.objective must be barlow or infonce, got {self.objective!r}")
        self.register_buffer("membership", torch.as_tensor(membership, dtype=torch.float32))

    def set_membership(self, membership):
        self.membership = torch.as_tensor(membership, dtype=torch.float32, device=self.membership.device)

    def level_loss(self, level, za, zb, meta, subject_ids):
        ea, eb = self.expanders[level](za), self.expanders[level](zb)
        if self.objective == "barlow":
            loss, info = self.losses[level](ea, eb, meta["context"])
        else:
            tau = self.cfg.loss.infonce.temperature
            if level == "S":
                mask = s_negative_mask(meta["subject"], meta["region"], meta["coords"], self.rho, self.metric)
                loss, info = info_nce(ea, eb, mask, tau)
            elif level == "L":
                loss, info = info_nce(ea, eb, l_negative_mask(meta["subject"], meta["region"]), tau)
            else:
                gid = subject_ids.to(ea.device)[meta["subject"]]
                q, q_sub = self.queue.contents()
                loss, info = info_nce(
                    ea, eb, g_negative_mask(gid), tau, queue=q, queue_mask=gid[:, None] != q_sub[None]
                )
                if self.training:
                    self.queue.push(F.normalize(eb.detach().float(), dim=1), gid)
        if self.cfg.model.variance_floor:
            floor = variance_floor(ea) + variance_floor(eb)
            loss = loss + self.cfg.model.variance_floor_weight * floor
            info["variance_floor"] = float(floor.detach())
        return loss, info

    def forward(self, feats, sample, subject_ids):
        n_l = sample["L_region"].shape[1]
        pairs = build_pairs(feats, sample, self.membership, n_l)
        losses, logs = {}, {}
        for level in LEVELS:
            za, zb, meta = pairs[level]
            losses[level], logs[level] = self.level_loss(level, za, zb, meta, subject_ids)
        return losses, logs, pairs
