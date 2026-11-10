"""Alignment objectives: local per-site, global distant-window, whole-image pooled."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.gl_ident.config import RunConfig


def info_nce(a, b, tau, similarity="cosine", exclude=None):
    """Symmetric InfoNCE: row i of ``a`` and row i of ``b`` are positives; every other row of the other view is a
    negative unless ``exclude[i, j]`` (whose diagonal must be False)."""
    if similarity == "cosine":
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        logits = a @ b.T / tau
    elif similarity == "l2":
        logits = -torch.cdist(a, b).square() / tau
    else:
        raise ValueError(similarity)
    if exclude is not None:
        logits = logits.masked_fill(exclude, float("-inf"))
    targets = torch.arange(a.shape[0], device=a.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    with torch.no_grad():
        diag = dict(top1=float((logits.argmax(1) == targets).float().mean()), pos=float(logits.diagonal().mean() * tau))
    return loss, diag


def gather_sites(field, sites):
    """field (B, d, *N), sites (B, S, dim) -> (B, S, d)."""
    B, d, *grid = field.shape
    lin = torch.zeros(sites.shape[:2], dtype=torch.long, device=field.device)
    for a, n in enumerate(grid):
        lin = lin * n + sites[..., a]
    flat = field.reshape(B, d, -1)
    return torch.gather(flat, 2, lin.unsqueeze(1).expand(-1, d, -1)).transpose(1, 2)


def same_subject_mask(B, S, device):
    """(B*S, B*S) True where rows are different sites of the same subject (never used as negatives)."""
    subj = torch.arange(B, device=device).repeat_interleave(S)
    same = subj[:, None] == subj[None, :]
    return same & ~torch.eye(B * S, dtype=torch.bool, device=device)


def sample_sites(B, S, n, dim, gen):
    return torch.randint(0, n, (B, S, dim), generator=gen)


def sample_window_pairs(B, n, w, rho_f, dim, mode, gen):
    """Window starts (u, v), (B, dim) each, in coarse sites.

    disjoint: along one random axis the circular offset delta = v - u is in [L, n - L], L = w + 2 rho_f, so the
    latent supports (window + rho_f margin) do not overlap; the other axes are uniform.
    overlap: |delta| < L (circularly) on every axis, delta = 0 included.
    """
    L = w + 2 * rho_f
    u = torch.randint(0, n, (B, dim), generator=gen)
    if mode == "disjoint":
        assert n - L >= L, "grid too small for disjoint supports"
        delta = torch.randint(0, n, (B, dim), generator=gen)
        axis = torch.randint(0, dim, (B,), generator=gen)
        delta[torch.arange(B), axis] = torch.randint(L, n - L + 1, (B,), generator=gen)
    elif mode == "overlap":
        delta = torch.randint(-(L - 1), L, (B, dim), generator=gen) % n
    else:
        raise ValueError(mode)
    return u, (u + delta) % n


def supports_disjoint(u, v, n, w, rho_f):
    """True per row if the circular offset is in [L, n - L] along at least one axis."""
    L = w + 2 * rho_f
    delta = (v - u) % n
    return ((delta >= L) & (delta <= n - L)).any(dim=-1)


def crop_windows(x, starts, size_sites, m):
    """Circular crop of (size_sites * m)^dim pixels starting at site ``starts`` (B, dim)."""
    B, dim = x.shape[0], x.ndim - 2
    P = size_sites * m
    ar = torch.arange(P, device=x.device)
    starts = starts.to(x.device)
    out = x
    for a in range(dim):
        idx = (starts[:, a, None] * m + ar) % x.shape[2 + a]
        view = [B, 1] + [1] * dim
        view[2 + a] = P
        shape = list(out.shape)
        shape[2 + a] = P
        out = torch.gather(out, 2 + a, idx.view(view).expand(shape))
    return out


def _barlow_twins(a, b, lambd):
    import training.losses  # the repo's implementation; imported lazily (it pulls in lpips at module scope)

    return training.losses.barlow_twins_loss(torch.stack([a, b]), lambd=lambd)


def compute_loss(model, cfg: RunConfig, x1, x2, gen: torch.Generator):
    """Loss for one batch of subjects; ``gen`` (CPU) draws the sites / windows."""
    lc, g = cfg.loss, cfg.generator
    B, dev = x1.shape[0], x1.device
    if model.kind == "local":
        z1, z2 = model(x1, 0), model(x2, 1)
        sites = sample_sites(B, lc.sites_per_subject, g.n_sites, g.dim, gen).to(dev)
        a = gather_sites(z1, sites).reshape(B * lc.sites_per_subject, -1)
        b = gather_sites(z2, sites).reshape(B * lc.sites_per_subject, -1)
        if lc.objective == "barlow_twins":
            return _barlow_twins(a, b, lc.bt_lambda), {}
        return info_nce(a, b, lc.tau, lc.similarity, exclude=same_subject_mask(B, lc.sites_per_subject, dev))
    if model.kind == "global":
        w = cfg.model.window
        u, v = sample_window_pairs(B, g.n_sites, w, g.rho_f, g.dim, cfg.model.window_offsets, gen)
        a = model(crop_windows(x1, u, w, g.stride), 0)
        b = model(crop_windows(x2, v, w, g.stride), 1)
    else:
        a, b = model(x1, 0), model(x2, 1)
    if lc.objective == "barlow_twins":
        return _barlow_twins(a, b, lc.bt_lambda), {}
    return info_nce(a, b, lc.tau, lc.similarity)
