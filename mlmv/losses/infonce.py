"""InfoNCE with context negatives (ablation): cosine similarity, temperature tau.

``l_n = -log exp(s(za_n, zb_n)/tau) / (exp(s(za_n, zb_n)/tau) + sum_{m in N(n)} exp(s(za_n, zb_m)/tau))``

The negatives ``N(n)`` are given as a boolean mask over the other rows of the batch (and optionally over a
queue of earlier view-b embeddings). The mask builders below encode the spec's table:

* S: same subject and region, more than ``2 * rho`` from u;
* L: the other regions' halves of the same subject;
* G: other subjects, from the batch and the queue.

The positive is never a negative: the diagonal is removed from every mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def s_negative_mask(subject, region, coords, rho: float, metric: str = "chebyshev"):
    """``[N, N]`` bool: same subject and region, separation > 2 rho. ``coords [N, 3]`` in feature voxels."""
    diff = (coords[:, None, :] - coords[None, :, :]).abs().float()
    dist = diff.amax(-1) if metric == "chebyshev" else diff.pow(2).sum(-1).sqrt()
    mask = (subject[:, None] == subject[None]) & (region[:, None] == region[None]) & (dist > 2 * rho)
    return mask & ~torch.eye(len(subject), dtype=torch.bool, device=subject.device)


def l_negative_mask(subject, region):
    """``[N, N]`` bool: same subject, different region."""
    return (subject[:, None] == subject[None]) & (region[:, None] != region[None])


def g_negative_mask(subject):
    """``[N, N]`` bool: different subject."""
    return subject[:, None] != subject[None]


def info_nce(za, zb, neg_mask, temperature: float, queue=None, queue_mask=None):
    """Mean InfoNCE over rows. ``za, zb [N, P]``; ``neg_mask [N, N]``; ``queue [Q, P]``, ``queue_mask [N, Q]``."""
    za, zb = F.normalize(za.float(), dim=1), F.normalize(zb.float(), dim=1)
    logits = za @ zb.T / temperature
    pos = torch.diagonal(logits)
    neg_mask = neg_mask & ~torch.eye(len(za), dtype=torch.bool, device=za.device)
    neg = logits.masked_fill(~neg_mask, float("-inf"))
    parts = [pos[:, None], neg]
    if queue is not None and len(queue):
        q = (za @ F.normalize(queue.float(), dim=1).T / temperature).masked_fill(~queue_mask, float("-inf"))
        parts.append(q)
    loss = torch.logsumexp(torch.cat(parts, 1), dim=1) - pos
    acc = (pos[:, None] >= torch.cat(parts[1:], 1).max(1, keepdim=True).values).float().mean()
    return loss.mean(), {
        "top1": float(acc.detach()),
        "n_neg": float(torch.cat(parts[1:], 1).isfinite().sum(1).float().mean()),
    }


class Queue(nn.Module):
    """FIFO of detached view-b embeddings and their subject ids (for the G negatives)."""

    def __init__(self, size: int, dim: int):
        super().__init__()
        self.size = size
        self.register_buffer("z", torch.zeros(size, dim))
        self.register_buffer("subject", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def contents(self):
        n = int(self.count)
        return self.z[:n], self.subject[:n]

    @torch.no_grad()
    def push(self, z, subject):
        for start in range(0, len(z), self.size):
            chunk, sid = z[start : start + self.size].detach().float(), subject[start : start + self.size]
            idx = (int(self.ptr) + torch.arange(len(chunk), device=chunk.device)) % self.size
            self.z[idx], self.subject[idx] = chunk, sid
            self.ptr.copy_((self.ptr + len(chunk)) % self.size)
            self.count.copy_(torch.clamp(self.count + len(chunk), max=self.size))
