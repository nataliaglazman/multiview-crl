"""Feature-grid region geometry: eroded region masks, eligibility, half-splits and G region sets.

Every distance here is in feature voxels. ``rho`` is the encoder's receptive-field radius, so two positions
read shared input only if they are within ``2 * rho`` of each other.

Metric. The theoretical receptive field of a 3D conv stack is a box, not a ball. Two positions whose boxes
overlap can be more than ``2 * rho`` apart in Euclidean distance (along a diagonal), so a Euclidean separation
does not guarantee "no shared input". ``metric="chebyshev"`` (the default) measures every separation in the
L-infinity norm, which does guarantee it, and implies the Euclidean statement as well. ``metric="euclidean"``
is the spec's literal ball erosion, kept for comparison.

Erosion. Unlabelled voxels are first given the label of the nearest region, then each region keeps the voxels
far enough from every *other* region. Background is not eroded against: positions next to the brain surface
share no input with another region, and eroding them would remove the cortex from every lobe at 4 mm.
The keep thresholds are chosen so that any two eroded regions end up more than ``2 * rho`` apart:

* chebyshev: keep x iff d_inf(x, other regions) >= t with t = floor(rho + 1/2) + 1. Along a shortest king-move
  path of length n from x in A to y in B, the first t - 1 steps stay in A and the last t - 1 stay in B, so
  n >= 2t - 1 > 2 rho.
* euclidean: keep x iff d_2(x, other regions) > rho + sqrt(3)/2. The voxel nearest the midpoint of x and y lies
  within sqrt(3)/2 of it and belongs to A or not, which bounds |x - y| > 2 rho either way.

Half-split. A plane through the eroded region's centroid with a random unit normal n; positions with
n.(x - c) / k > rho form half A and < -rho half B, where k = ||n||_1 for chebyshev (Hoelder: |n.(x - y)| <=
||n||_1 ||x - y||_inf) and k = 1 for euclidean. The halves are then more than ``2 * rho`` apart.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

METRICS = ("chebyshev", "euclidean")


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def fill_unlabelled(labels: np.ndarray) -> np.ndarray:
    """Every 0 voxel takes the label of its nearest labelled voxel. ``[x, y, z]`` int -> same."""
    if not (labels > 0).any():
        raise ValueError("label map has no labelled voxel")
    if (labels > 0).all():
        return labels.copy()
    _, idx = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    return labels[tuple(idx)]


def distance_to(mask: np.ndarray, metric: str) -> np.ndarray:
    """Distance of every voxel to the nearest True voxel of ``mask`` (inf if mask is empty)."""
    _check_metric(metric)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    if metric == "chebyshev":
        return ndimage.distance_transform_cdt(~mask, metric="chessboard").astype(np.float64)
    return ndimage.distance_transform_edt(~mask)


def keep_threshold(rho: float, metric: str) -> float:
    _check_metric(metric)
    if metric == "chebyshev":
        return math.floor(rho + 0.5) + 1  # keep iff distance >= this
    return rho + math.sqrt(3) / 2  # keep iff distance > this


def erode_regions(labels: np.ndarray, rho: float, metric: str = "chebyshev") -> np.ndarray:
    """Eroded label map: any two eroded regions are more than ``2 * rho`` apart (see module docstring)."""
    ext = fill_unlabelled(labels)
    t = keep_threshold(rho, metric)
    out = np.zeros_like(labels)
    for region in np.unique(labels[labels > 0]):
        dist = distance_to((ext != region), metric)
        keep = dist >= t if metric == "chebyshev" else dist > t
        out[(labels == region) & keep] = region
    return out


def min_separation(labels: np.ndarray, metric: str = "chebyshev") -> float:
    """Smallest distance between voxels of two different regions of ``labels`` (inf if < 2 regions)."""
    best = math.inf
    regions = np.unique(labels[labels > 0])
    for i, a in enumerate(regions[:-1]):
        dist = distance_to(np.isin(labels, regions[i + 1 :]), metric)
        best = min(best, float(dist[labels == a].min()))
    return best


def pairwise_min_distance(a: np.ndarray, b: np.ndarray, metric: str = "chebyshev") -> float:
    """Smallest distance between two point sets ``[n, 3]`` and ``[m, 3]`` (exact, chunked)."""
    _check_metric(metric)
    best = math.inf
    for start in range(0, len(a), 1024):
        diff = np.abs(a[start : start + 1024, None, :].astype(np.float64) - b[None].astype(np.float64))
        d = diff.max(-1) if metric == "chebyshev" else np.sqrt((diff**2).sum(-1))
        best = min(best, float(d.min()))
    return best


def plane_scale(normal: np.ndarray, metric: str) -> float:
    return float(np.abs(normal).sum()) if metric == "chebyshev" else 1.0


def half_split(coords: np.ndarray, normal: np.ndarray, rho: float, metric: str = "chebyshev"):
    """Boolean masks (half A, half B) over ``coords [n, 3]`` for a plane through their centroid."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    signed = (coords - coords.mean(0)) @ normal / plane_scale(normal, metric)
    return signed > rho, signed < -rho


def random_normals(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def majority_downsample(labels: np.ndarray, factor: int) -> np.ndarray:
    """Majority vote over ``factor^3`` blocks; ties go to the smallest label. Shape must divide."""
    if any(s % factor for s in labels.shape):
        raise ValueError(f"shape {labels.shape} is not divisible by {factor}")
    x, y, z = (s // factor for s in labels.shape)
    blocks = labels.reshape(x, factor, y, factor, z, factor).transpose(0, 2, 4, 1, 3, 5).reshape(-1, factor**3)
    values = np.unique(blocks)  # sorted, so argmax picks the smallest label among ties
    counts = np.stack([(blocks == v).sum(1) for v in values], axis=1)
    return values[counts.argmax(1)].reshape(x, y, z)


@dataclass
class Geometry:
    """Region geometry on the feature grid.

    labels   ``[x, y, z]`` int, training regions after merging (0 = none); defines R(u) for the S level.
    eroded   ``[x, y, z]`` int, eroded training regions; L and G pool only over these.
    eligible region ids whose half-splits keep enough positions on each side.
    """

    labels: np.ndarray
    eroded: np.ndarray
    rho: float
    metric: str
    min_positions_per_half: int
    eligible: list
    table: list = field(default_factory=list)
    merges: list = field(default_factory=list)

    @property
    def shape(self):
        return self.labels.shape

    @property
    def regions(self) -> list:
        return [int(r) for r in np.unique(self.labels[self.labels > 0])]

    def coords(self, region: int, eroded: bool = True) -> np.ndarray:
        grid = self.eroded if eroded else self.labels
        return np.argwhere(grid == region)

    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "region_masks.npz", eroded=self.eroded, labels=self.labels)
        with open(d / "eligibility.csv", "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(self.table[0]) if self.table else ["region"])
            writer.writeheader()
            writer.writerows(self.table)
        (d / "geometry.json").write_text(
            json.dumps(
                dict(
                    rho=self.rho,
                    metric=self.metric,
                    eligible=self.eligible,
                    merges=self.merges,
                    min_positions_per_half=self.min_positions_per_half,
                    keep_threshold=keep_threshold(self.rho, self.metric),
                ),
                indent=2,
            )
        )


def _worst_split(coords: np.ndarray, normals: np.ndarray, rho: float, metric: str) -> int:
    if len(coords) == 0:
        return 0
    centred = coords - coords.mean(0)
    scale = np.abs(normals).sum(1) if metric == "chebyshev" else np.ones(len(normals))
    signed = centred @ normals.T / scale[None]  # [n, k]
    return int(np.minimum((signed > rho).sum(0), (signed < -rho).sum(0)).min())


def _neighbour_counts(labels: np.ndarray, region: int) -> dict:
    """Face-adjacent contact counts between ``region`` and every other region."""
    out: dict = {}
    mask = labels == region
    for axis in range(3):
        for shift in (1, -1):
            other = np.roll(labels, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            other[tuple(edge)] = 0  # no wrap-around
            touching = other[mask]
            for r, c in zip(*np.unique(touching[(touching > 0) & (touching != region)], return_counts=True)):
                out[int(r)] = out.get(int(r), 0) + int(c)
    return out


def build_geometry(
    labels,
    rho: float,
    metric: str = "chebyshev",
    min_positions_per_half: int = 8,
    n_normals: int = 256,
    seed: int = 0,
    log=print,
) -> Geometry:
    """Erode, test eligibility, and merge ineligible regions into their largest-contact neighbour.

    Merging repeats until every remaining region is eligible; a region with no neighbour left is dropped
    (kept in ``labels`` for S, excluded from L and G). Each merge or drop is logged and recorded.
    """
    _check_metric(metric)
    labels = np.asarray(labels).astype(np.int64).copy()
    original_sizes = {int(r): int(c) for r, c in zip(*np.unique(labels[labels > 0], return_counts=True))}
    normals = random_normals(n_normals, seed)
    merges, dropped = [], set()
    while True:
        eroded = erode_regions(labels, rho, metric)
        worst = {
            r: _worst_split(np.argwhere(eroded == r), normals, rho, metric)
            for r in np.unique(labels[labels > 0])
            if r not in dropped
        }
        failing = sorted((int((labels == r).sum()), r) for r, w in worst.items() if w < min_positions_per_half)
        if not failing:
            break
        _, region = failing[0]  # smallest failing region first
        contacts = {r: c for r, c in _neighbour_counts(labels, region).items() if r not in dropped}
        if contacts:
            target = max(contacts, key=contacts.get)
            labels[labels == region] = target
            merges.append(dict(region=int(region), merged_into=int(target)))
            log(f"geometry: region {region} is ineligible (worst half {worst[region]}); merged into {target}")
        else:
            dropped.add(region)
            merges.append(dict(region=int(region), merged_into=None))
            log(f"geometry: region {region} is ineligible and has no neighbour; dropped from L and G")
    merged_into = {m["region"]: m["merged_into"] for m in merges}
    final_eroded = {int(r): int(c) for r, c in zip(*np.unique(eroded[eroded > 0], return_counts=True))}
    final_labels = {int(r): int(c) for r, c in zip(*np.unique(labels[labels > 0], return_counts=True))}
    eligible = sorted(int(r) for r in worst if r not in dropped)
    table = []
    for r in sorted(original_sizes):
        alive = r in final_labels
        table.append(
            dict(
                region=r,
                size_before=original_sizes[r],
                size_merged=final_labels.get(r, 0),
                size_eroded=final_eroded.get(r, 0) if alive else 0,
                worst_half=worst.get(r, 0) if alive else 0,
                eligible=r in eligible,
                merged_into="" if merged_into.get(r) is None else merged_into[r],
            )
        )
    return Geometry(
        labels=labels,
        eroded=np.where(np.isin(eroded, eligible), eroded, 0),
        rho=float(rho),
        metric=metric,
        min_positions_per_half=min_positions_per_half,
        eligible=eligible,
        table=table,
        merges=merges,
    )


def draw_region_sets(regions, volumes, ratio, gen, tries: int = 10000):
    """Random partition of ``regions`` into two non-empty sets whose volume share lies in ``ratio``.

    ``gen`` is a ``torch.Generator``; the draw is redone until the share of set 1 is within ``ratio``.
    """
    import torch

    regions = np.asarray(regions)
    volumes = np.asarray(volumes, dtype=np.float64)
    lo, hi = ratio
    for _ in range(tries):
        side = torch.randint(0, 2, (len(regions),), generator=gen).numpy().astype(bool)
        if side.all() or not side.any():
            continue
        share = volumes[side].sum() / volumes.sum()
        if lo <= share <= hi:
            return regions[side], regions[~side]
    raise RuntimeError(f"no split of {len(regions)} regions into volume shares within {ratio} after {tries} tries")
