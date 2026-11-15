"""A fixed procedural brain: tissue classes and 13 regions on an image grid, built in millimetres.

Used by the ``brain`` renderer of ``three_level.py`` as the anatomy every synthetic subject shares. Nothing here
varies between subjects; all subject variation comes from the generator's latents. A real template (the ADNI
template's SynthSeg map) can replace it later through the same output: tissue labels, region labels, and the
names of both.

Axes: x left (-) to right (+), y posterior (-) to anterior (+), z inferior (-) to superior (+), origin at the
grid centre. Sizes are adult-like at the 2 mm ADNI grid (96 x 112 x 96 voxels = 192 x 224 x 192 mm).

Tissue labels: 0 background, 1 CSF, 2 cortical GM, 3 deep GM, 4 WM.
Regions (per hemisphere, left 1-6, right 7-12): frontal, parietal, temporal, occipital, deep grey (nuclei,
ventricles and the white matter between them), cerebellum; 13 brainstem. The hippocampus belongs to the temporal
region and the ventricles to the deep region, as the spec merges them for ADNI.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

TISSUES = ("background", "csf", "gm", "deep_gm", "wm")
LOBES = ("frontal", "parietal", "temporal", "occipital", "deep", "cerebellum")
REGION_NAMES = {i + 1 + 6 * h: f"{'LR'[h]} {lobe}" for h in range(2) for i, lobe in enumerate(LOBES)}
REGION_NAMES[13] = "brainstem"


def _ellipsoid(x, y, z, centre, semi):
    (cx, cy, cz), (a, b, c) = centre, semi
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 + ((z - cz) / c) ** 2 <= 1


def _smooth_field(shape, sigma_vox, seed):
    """Unit-variance smooth random field (Gaussian-filtered white noise)."""
    f = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma_vox, mode="reflect")
    return (f - f.mean()) / f.std()


def build_brain_template(shape=(96, 112, 96), voxel_mm: float = 2.0, seed: int = 0) -> dict:
    """``{"tissue": int8 [X,Y,Z], "regions": int8 [X,Y,Z], "tissue_names", "region_names", "voxel_mm"}``."""
    shape = tuple(int(s) for s in shape)
    axes = [(np.arange(n) - (n - 1) / 2) * voxel_mm for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    ax = np.abs(x)

    # Cerebrum: a superellipsoid, narrower towards the frontal pole, with a flat base that drops under the
    # temporal lobes.
    a = 68 * (1 - 0.12 * np.clip(y / 84, 0, 1)) * (1 - 0.06 * np.clip(-y / 84, 0, 1))
    shell = (ax / a) ** 2.4 + (np.abs(y) / 84) ** 2.2 + (np.abs(z - 12) / 58) ** 2.2
    base = -24 - 18 * np.exp(-(((ax - 44) / 18) ** 2)) * np.exp(-(((y + 2) / 30) ** 2))
    envelope = (shell <= 1) & (z >= base)

    # Clefts cut out of the cerebrum before the cortex is laid down, so grey matter lines them:
    # the interhemispheric fissure (stopping at the corpus callosum) and the lateral (Sylvian) fissure.
    wiggle = 1.5 * np.sin(y / 9.0) * np.cos(z / 13.0)
    fissure = envelope & (np.abs(x - wiggle) < 1.8) & ((z > 20) | (y > 32) | (y < -38) | (z < -6))
    sylvian_z = -2 + 0.25 * y
    sylvian = envelope & (np.abs(z - sylvian_z) < 1.6) & (ax > 34) & (y > -32) & (y < 30)
    cerebrum = envelope & ~fissure & ~sylvian

    # Folded cortex: sulci are the zero sheets of a fixed smooth field near the surface.
    depth = ndimage.distance_transform_edt(cerebrum) * voxel_mm
    folds = _smooth_field(shape, 5.0 / voxel_mm, seed)
    sulcus = cerebrum & (np.abs(folds) < 0.16) & (depth < 14)
    cortex = cerebrum & ~sulcus & ((depth < 4.0) | ((np.abs(folds) < 0.6) & (depth < 18)))

    tissue = np.zeros(shape, np.int8)
    tissue[cerebrum] = 4
    tissue[cortex] = 2
    tissue[envelope & (fissure | sylvian | sulcus)] = 1

    # Deep grey nuclei, hippocampi and ventricles (both sides).
    deep_gm = np.zeros(shape, bool)
    ventricles = np.zeros(shape, bool)
    hippocampus = np.zeros(shape, bool)
    for s in (-1, 1):
        deep_gm |= _ellipsoid(x, y, z, (s * 10, -12, 2), (8, 13, 8))  # thalamus
        deep_gm |= _ellipsoid(x, y, z, (s * 14, 14, 12), (5, 14, 7))  # caudate
        deep_gm |= _ellipsoid(x, y, z, (s * 24, 4, 0), (5, 15, 9))  # putamen
        hippocampus |= _ellipsoid(x, y, z, (s * 27, -15, -14), (5, 16, 5))
        ventricles |= _ellipsoid(x, y, z, (s * 10, -2, 14), (5.5, 30, 6))  # body
        ventricles |= _ellipsoid(x, y, z, (s * 12, 22, 8), (6, 10, 8))  # frontal horn
        ventricles |= _ellipsoid(x, y, z, (s * 16, -32, 8), (5, 10, 6))  # occipital horn
        ventricles |= _ellipsoid(x, y, z, (s * 30, -5, -12), (4, 16, 4))  # temporal horn
    ventricles |= (ax < 2) & (y > -18) & (y < 8) & (z > -6) & (z < 10)  # third ventricle
    tissue[deep_gm & envelope] = 3
    tissue[hippocampus & envelope] = 2
    tissue[ventricles & envelope] = 1

    # Cerebellum under the occipital lobes (folia as stripes), separated from the cerebrum by a CSF tentorium,
    # and the brainstem below the diencephalon.
    cb = _ellipsoid(x, y, z, (0, -58, -40), (50, 30, 22)) & ~envelope
    cb_depth = ndimage.distance_transform_edt(cb) * voxel_mm
    folia = np.sin(2 * np.pi * (0.8 * z + 0.3 * y) / 7.0)
    tentorium = cb & (ndimage.distance_transform_edt(~envelope) * voxel_mm < 2.5)
    stem = ((x / 11) ** 2 + ((y + 22) / 13) ** 2 <= 1) & (z > -82) & (z < -14) & ~cb
    cb_tissue = np.where((cb_depth < 4) | ((folia > 0.3) & (cb_depth < 14)), 2, 4)
    tissue[cb] = cb_tissue[cb]
    tissue[tentorium] = 1
    tissue[stem & ~envelope] = 4

    brain = tissue > 0

    # Regions: positional lobes per hemisphere, laid over every brain voxel (CSF included).
    deep_zone = ((x / 28) ** 2 + ((y + 4) / 36) ** 2 + ((z - 4) / 20) ** 2 <= 1) | deep_gm | ventricles
    deep_zone &= ~hippocampus
    temporal = (z < sylvian_z) & (y > -46) & (y < 30) & (ax > 20) & ~deep_zone
    occipital = (y < -46) & ~temporal
    frontal = (y > 8 + 0.15 * z) & ~temporal & ~deep_zone
    lobe = np.full(shape, 1, np.int8)  # parietal by default
    lobe[frontal] = 0
    lobe[temporal] = 2
    lobe[occipital] = 3
    lobe[deep_zone] = 4
    hemi = (x > 0).astype(np.int8)
    regions = np.zeros(shape, np.int8)
    in_cerebrum = brain & envelope
    regions[in_cerebrum] = (1 + lobe + 6 * hemi)[in_cerebrum]
    regions[brain & cb] = (6 + 6 * hemi)[brain & cb]
    regions[brain & stem & ~envelope & ~cb] = 13
    regions[brain & (regions == 0)] = 13  # stray voxels (tentorium edges) join the brainstem
    return dict(tissue=tissue, regions=regions, tissue_names=TISSUES, region_names=REGION_NAMES, voxel_mm=voxel_mm)
