"""View-pair sampler: per subject, positions for S, region halves for L and region sets for G.

All draws use an explicit ``torch.Generator``. The geometry is shared by every subject (template space, or the
synthetic generator's fixed region map), so only the random draws differ between subjects.

Returned per step (CPU tensors, B subjects):

* ``S_idx [B, nS]`` flat feature positions, ``S_region [B, nS]`` R(u), ``S_coord [B, nS, 3]``;
* ``L_flat [M]`` flat positions of every kept half, ``L_seg [M]`` its segment ``(b * nL + k) * 2 + h``,
  ``L_region [B, nL]``;
* ``G_member [B, nG, 2, R_e]`` bool: which eligible regions make up set h of split k.
"""

from __future__ import annotations

import numpy as np
import torch

from mlmv.data.geometry import Geometry, draw_region_sets, half_split


class ViewPairSampler:
    def __init__(self, geometry: Geometry, cfg, mask=None, max_redraws: int = 100):
        self.geo, self.cfg, self.max_redraws = geometry, cfg, max_redraws
        shape = geometry.shape
        labels = geometry.labels if mask is None else np.where(np.asarray(mask, bool), geometry.labels, 0)
        self.s_regions = [int(r) for r in np.unique(labels[labels > 0])]
        self.s_flat = [torch.from_numpy(np.flatnonzero(labels == r)) for r in self.s_regions]
        self.eligible = list(geometry.eligible)
        if len(self.eligible) < 2:
            raise ValueError(f"G needs at least two eligible regions, geometry has {self.eligible}")
        self.coords = {r: geometry.coords(r) for r in self.eligible}
        self.flat = {r: np.ravel_multi_index(self.coords[r].T, shape) for r in self.eligible}
        self.volumes = np.array([len(self.coords[r]) for r in self.eligible], dtype=np.float64)
        self.n_l = min(cfg.L.regions, len(self.eligible))
        self.shape = shape
        coord_grid = np.stack(np.unravel_index(np.arange(int(np.prod(shape))), shape), -1)
        self.coord_grid = torch.from_numpy(coord_grid)

    def membership(self) -> torch.Tensor:
        """``[P, R_e]`` float: eroded eligible region membership, for region sums ``F @ M``."""
        m = torch.zeros(int(np.prod(self.shape)), len(self.eligible))
        for j, r in enumerate(self.eligible):
            m[torch.from_numpy(self.flat[r]), j] = 1.0
        return m

    def _positions(self, gen):
        n = self.cfg.S.positions
        quota = [n // len(self.s_regions) + (i < n % len(self.s_regions)) for i in range(len(self.s_regions))]
        idx, reg = [], []
        for region, flat, q in zip(self.s_regions, self.s_flat, quota):
            if q == 0:
                continue
            if len(flat) >= q:
                pick = torch.randperm(len(flat), generator=gen)[:q]
            else:
                pick = torch.randint(len(flat), (q,), generator=gen)
            idx.append(flat[pick])
            reg.append(torch.full((q,), region, dtype=torch.long))
        return torch.cat(idx), torch.cat(reg)

    def split_region(self, region: int, gen):
        """Two index arrays into ``self.coords[region]``: the kept positions of half A and half B."""
        coords = self.coords[region]
        for _ in range(self.max_redraws):
            normal = torch.randn(3, generator=gen, dtype=torch.float64).numpy()
            a, b = half_split(coords, normal, self.geo.rho, self.geo.metric)
            if a.sum() >= self.geo.min_positions_per_half and b.sum() >= self.geo.min_positions_per_half:
                break
        else:
            raise RuntimeError(
                f"region {region}: no half-split with {self.geo.min_positions_per_half} positions a side"
            )
        halves = []
        for half in (np.flatnonzero(a), np.flatnonzero(b)):
            k = max(1, int(round(self.cfg.L.fraction * len(half))))
            pick = torch.randperm(len(half), generator=gen)[:k].numpy()
            halves.append(half[pick])
        return halves

    def sample(self, batch_size: int, gen: torch.Generator) -> dict:
        s_idx, s_reg = zip(*[self._positions(gen) for _ in range(batch_size)])
        s_idx, s_reg = torch.stack(s_idx), torch.stack(s_reg)
        l_flat, l_seg, l_region = [], [], []
        for b in range(batch_size):
            chosen = torch.randperm(len(self.eligible), generator=gen)[: self.n_l].tolist()
            regions = [self.eligible[j] for j in chosen]
            l_region.append(regions)
            for k, region in enumerate(regions):
                for h, half in enumerate(self.split_region(region, gen)):
                    l_flat.append(torch.from_numpy(self.flat[region][half]))
                    l_seg.append(torch.full((len(half),), (b * self.n_l + k) * 2 + h, dtype=torch.long))
        n_g = self.cfg.G.splits_per_subject
        member = torch.zeros(batch_size, n_g, 2, len(self.eligible), dtype=torch.bool)
        index = {r: j for j, r in enumerate(self.eligible)}
        for b in range(batch_size):
            for k in range(n_g):
                set1, set2 = draw_region_sets(self.eligible, self.volumes, self.cfg.G.volume_ratio, gen)
                member[b, k, 0, [index[int(r)] for r in set1]] = True
                member[b, k, 1, [index[int(r)] for r in set2]] = True
        return dict(
            S_idx=s_idx,
            S_region=s_reg,
            S_coord=self.coord_grid[s_idx],
            L_flat=torch.cat(l_flat),
            L_seg=torch.cat(l_seg),
            L_region=torch.tensor(l_region, dtype=torch.long),
            G_member=member,
        )
