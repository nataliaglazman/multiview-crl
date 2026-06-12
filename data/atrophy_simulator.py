"""Atrophy simulation on FreeSurfer or MUSE segmentation maps.

Erodes AD-relevant structures (hippocampus, amygdala, entorhinal cortex),
dilates compensating structures (ventricles), optionally injects WMH, and
smooths label boundaries for clean SynthSeg synthesis.

Supports both FreeSurfer aseg and MUSE (CBICA/NiChart) label conventions.
Pass label_set="freesurfer" or label_set="muse" to select; defaults to
auto-detection from the segmentation volume.
"""

import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
)
from scipy.ndimage import label as cc_label

# ── Label constants ───────────────────────────────────────────────────────────

FS = dict(
    CSF=24,
    LH_LATERAL_VENT=4,
    RH_LATERAL_VENT=43,
    LH_HIPPO=17,
    RH_HIPPO=53,
    LH_AMYGDALA=18,
    RH_AMYGDALA=54,
    LH_ENTORHINAL=1006,
    RH_ENTORHINAL=2006,
    LH_PARAHIPPO=1016,
    RH_PARAHIPPO=2016,
    LH_WM=2,
    RH_WM=41,
    BRAIN_STEM=16,
)

MUSE = dict(
    CSF=46,
    LH_LATERAL_VENT=52,
    RH_LATERAL_VENT=51,
    LH_HIPPO=48,
    RH_HIPPO=47,
    LH_AMYGDALA=32,
    RH_AMYGDALA=31,
    LH_ENTORHINAL=117,
    RH_ENTORHINAL=116,
    LH_PARAHIPPO=171,
    RH_PARAHIPPO=170,
    LH_WM=(82, 84, 86, 88),
    RH_WM=(81, 83, 85, 87),
    BRAIN_STEM=35,
)

LABEL_SETS = {"freesurfer": FS, "muse": MUSE}


def detect_label_set(seg):
    """Heuristic: MUSE uses 116/117 for entorhinal (absent in FS aseg);
    FS uses 17 for left hippocampus (MUSE uses 48)."""
    unique = set(np.unique(seg))
    if {116, 117} & unique:
        return "muse"
    if 17 in unique:
        return "freesurfer"
    return "freesurfer"


def get_labels(label_set):
    if label_set not in LABEL_SETS:
        raise ValueError(f"Unknown label_set '{label_set}', choose from {list(LABEL_SETS)}")
    return LABEL_SETS[label_set]


# ── MUSE → FreeSurfer remapping (for SynthSeg synthesis) ─────────────────────

MUSE_TO_FS = {
    0: 0,  # background
    4: 14,  # 3rd ventricle
    11: 15,  # 4th ventricle
    23: 58,  # right accumbens
    30: 26,  # left accumbens
    31: 54,  # right amygdala
    32: 18,  # left amygdala
    35: 16,  # brain stem
    36: 50,  # right caudate
    37: 11,  # left caudate
    38: 47,  # right cerebellum exterior → FS right cerebellum cortex
    39: 8,  # left cerebellum exterior → FS left cerebellum cortex
    40: 46,  # right cerebellum WM
    41: 7,  # left cerebellum WM
    42: 42,  # right cerebral exterior → FS right cerebral cortex
    43: 3,  # left cerebral exterior → FS left cerebral cortex
    46: 24,  # CSF
    47: 53,  # right hippocampus
    48: 17,  # left hippocampus
    49: 44,  # right inf lat ventricle
    50: 5,  # left inf lat ventricle
    51: 43,  # right lateral ventricle
    52: 4,  # left lateral ventricle
    55: 52,  # right pallidum
    56: 13,  # left pallidum
    57: 51,  # right putamen
    58: 12,  # left putamen
    59: 49,  # right thalamus
    60: 10,  # left thalamus
    61: 60,  # right ventral DC
    62: 28,  # left ventral DC
    63: 24,  # right vessel → CSF
    64: 24,  # left vessel → CSF
    69: 0,  # optic chiasm → background
    71: 47,  # cerebellar vermal lobules I-V → right cerebellum cortex
    72: 47,  # cerebellar vermal lobules VI-VII
    73: 47,  # cerebellar vermal lobules VIII-X
    75: 28,  # left basal forebrain → left ventral DC
    76: 60,  # right basal forebrain → right ventral DC
    # White matter: lobar → hemispheric
    81: 41,  # frontal WM right
    82: 2,  # frontal WM left
    83: 41,  # occipital WM right
    84: 2,  # occipital WM left
    85: 41,  # parietal WM right
    86: 2,  # parietal WM left
    87: 41,  # temporal WM right
    88: 2,  # temporal WM left
    89: 41,  # fornix right → right WM
    90: 2,  # fornix left → left WM
    91: 41,  # anterior limb internal capsule right
    92: 2,  # anterior limb internal capsule left
    93: 41,  # posterior limb internal capsule right
    94: 2,  # posterior limb internal capsule left
    95: 2,  # corpus callosum → left WM
}

# All MUSE cortical ROIs (100-207) → FS cortex labels by hemisphere.
# Even MUSE indices = right, odd = left.
for _muse_id in range(100, 208):
    if _muse_id not in MUSE_TO_FS:
        MUSE_TO_FS[_muse_id] = 42 if _muse_id % 2 == 0 else 3


def remap_muse_to_freesurfer(seg):
    """Convert a MUSE-labelled segmentation volume to FreeSurfer aseg labels."""
    out = np.zeros_like(seg)
    for muse_id, fs_id in MUSE_TO_FS.items():
        out[seg == muse_id] = fs_id
    return out


# ── Bilateral pairing (for intensity-consistent synthesis) ────────────────────

# Bilateral subcortical pairs: (right, left) in MUSE convention.
_MUSE_BILATERAL_PAIRS = [
    (23, 30),  # accumbens
    (31, 32),  # amygdala
    (36, 37),  # caudate
    (38, 39),  # cerebellum exterior
    (40, 41),  # cerebellum WM
    (42, 43),  # cerebral exterior
    (47, 48),  # hippocampus
    (49, 50),  # inf lat ventricle
    (51, 52),  # lateral ventricle
    (55, 56),  # pallidum
    (57, 58),  # putamen
    (59, 60),  # thalamus
    (61, 62),  # ventral DC
    (63, 64),  # vessel
    (75, 76),  # basal forebrain
    (81, 82),  # frontal WM
    (83, 84),  # occipital WM
    (85, 86),  # parietal WM
    (87, 88),  # temporal WM
    (89, 90),  # fornix
    (91, 92),  # anterior internal capsule
    (93, 94),  # posterior internal capsule
]
# Cortical ROIs: even = right, odd = left, consecutive pairs
_MUSE_BILATERAL_PAIRS += [(i, i + 1) for i in range(100, 208, 2)]

_FS_BILATERAL_PAIRS = [
    (4, 43),  # lateral ventricle
    (5, 44),  # inf lat ventricle
    (2, 41),  # cerebral WM
    (3, 42),  # cerebral cortex
    (7, 46),  # cerebellum WM
    (8, 47),  # cerebellum cortex
    (10, 49),  # thalamus
    (11, 50),  # caudate
    (12, 51),  # putamen
    (13, 52),  # pallidum
    (17, 53),  # hippocampus
    (18, 54),  # amygdala
    (26, 58),  # accumbens
    (28, 60),  # ventral DC
]


def get_bilateral_map(label_set="muse"):
    """Return {label: canonical_label} so bilateral pairs share one key.

    Both labels in a pair map to the same canonical (the lower ID).
    Unpaired / midline labels map to themselves.
    """
    pairs = _MUSE_BILATERAL_PAIRS if label_set == "muse" else _FS_BILATERAL_PAIRS
    bmap = {}
    for a, b in pairs:
        canonical = min(a, b)
        bmap[a] = canonical
        bmap[b] = canonical
    return bmap


# ── Tissue class grouping (for realistic intensity ranges) ───────────────────

MUSE_TISSUE_CLASSES = {
    "csf": [4, 11, 46, 49, 50, 51, 52],
    "cortical_gm": [42, 43] + list(range(100, 208)),
    "subcortical_gm": [23, 30, 31, 32, 36, 37, 47, 48, 55, 56, 57, 58, 59, 60, 61, 62, 75, 76],
    "wm": [40, 41, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    "cerebellum_gm": [38, 39, 71, 72, 73],
    "brainstem": [35],
    "vessel": [63, 64],
    "other": [69],
}

FS_TISSUE_CLASSES = {
    "csf": [4, 5, 14, 15, 24, 43, 44],
    "cortical_gm": [3, 42],
    "subcortical_gm": [10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60],
    "wm": [2, 7, 41, 46],
    "cerebellum_gm": [8, 47],
    "brainstem": [16],
}

# Intensity ranges per tissue class — (min_mean, max_mean).
# Used when contrast=None (fully random, SynthSeg-style domain randomization).
TISSUE_INTENSITY_RANGES = {
    "csf": (10, 80),
    "cortical_gm": (90, 170),
    "subcortical_gm": (100, 180),
    "wm": (140, 240),
    "cerebellum_gm": (90, 170),
    "brainstem": (110, 190),
    "vessel": (20, 90),
    "other": (50, 150),
}

# ── Contrast-specific priors (mean, std_of_mean) ─────────────────────────────
# Each call samples tissue_mean ~ N(prior_mean, prior_std), so images vary
# across subjects while maintaining the characteristic contrast ordering.

CONTRAST_PRIORS = {
    # T1-weighted: WM bright, GM intermediate, CSF dark
    "t1": {
        "csf": (25, 8),
        "cortical_gm": (105, 12),
        "subcortical_gm": (115, 10),
        "wm": (175, 15),
        "cerebellum_gm": (100, 10),
        "brainstem": (140, 10),
        "vessel": (30, 10),
        "other": (90, 15),
    },
    # T2-weighted: CSF bright, GM intermediate, WM dark
    "t2": {
        "csf": (210, 15),
        "cortical_gm": (120, 12),
        "subcortical_gm": (110, 10),
        "wm": (75, 12),
        "cerebellum_gm": (115, 10),
        "brainstem": (90, 10),
        "vessel": (200, 15),
        "other": (100, 15),
    },
    # FLAIR: CSF suppressed (dark), GM and WM both bright with subtle contrast
    "flair": {
        "csf": (15, 6),
        "cortical_gm": (145, 10),
        "subcortical_gm": (130, 10),
        "wm": (170, 12),
        "cerebellum_gm": (140, 10),
        "brainstem": (150, 10),
        "vessel": (20, 8),
        "other": (135, 12),
    },
}


def get_tissue_classes(label_set="muse"):
    return MUSE_TISSUE_CLASSES if label_set == "muse" else FS_TISSUE_CLASSES


# ── Structure configs ─────────────────────────────────────────────────────────


@dataclass
class StructureConfig:
    label: int
    backfill_label: int
    compensate_label: int
    max_erosion_iters: int = 5
    compensate: bool = True


def _build_configs(L):
    return [
        StructureConfig(L["LH_HIPPO"], L["CSF"], L["LH_LATERAL_VENT"], max_erosion_iters=5),
        StructureConfig(L["RH_HIPPO"], L["CSF"], L["RH_LATERAL_VENT"], max_erosion_iters=5),
        StructureConfig(L["LH_AMYGDALA"], L["CSF"], L["LH_LATERAL_VENT"], max_erosion_iters=4),
        StructureConfig(L["RH_AMYGDALA"], L["CSF"], L["RH_LATERAL_VENT"], max_erosion_iters=4),
        StructureConfig(L["LH_ENTORHINAL"], L["CSF"], L["LH_LATERAL_VENT"], max_erosion_iters=3),
        StructureConfig(L["RH_ENTORHINAL"], L["CSF"], L["RH_LATERAL_VENT"], max_erosion_iters=3),
    ]


DEFAULT_CONFIGS = _build_configs(FS)
MUSE_CONFIGS = _build_configs(MUSE)


def get_configs(label_set):
    if label_set == "muse":
        return MUSE_CONFIGS
    return DEFAULT_CONFIGS


def get_wm_labels(label_set):
    """Return the WM label(s) as a tuple, suitable for inject_wmh."""
    L = get_labels(label_set)
    wm = L["LH_WM"], L["RH_WM"]
    # MUSE WM entries are tuples of lobar labels; flatten
    flat = []
    for w in wm:
        if isinstance(w, tuple):
            flat.extend(w)
        else:
            flat.append(w)
    return tuple(flat)


def get_vent_labels(label_set):
    L = get_labels(label_set)
    return (L["LH_LATERAL_VENT"], L["RH_LATERAL_VENT"])


# ── Core erosion ──────────────────────────────────────────────────────────────


def _largest_cc(mask):
    labeled, n = cc_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


def erode_structure(label_map, cfg, alpha, connectivity=1):
    if alpha == 0:
        return label_map.copy()

    n_iters = max(1, round(alpha * cfg.max_erosion_iters))
    struct = generate_binary_structure(3, connectivity)

    seg = label_map.copy()
    target = seg == cfg.label

    eroded = binary_erosion(target, structure=struct, iterations=n_iters)
    if eroded.any():
        eroded = _largest_cc(eroded)

    lost = target & ~eroded
    n_lost = int(lost.sum())

    seg[target] = cfg.backfill_label
    seg[eroded] = cfg.label

    if cfg.compensate and n_lost > 0:
        compensate_mask = seg == cfg.compensate_label
        candidate = seg == cfg.backfill_label
        dilated = binary_dilation(compensate_mask, structure=struct, iterations=1, mask=candidate)
        new_comp = dilated & ~compensate_mask
        total_added = int(new_comp.sum())
        max_iters = n_iters + 2
        it = 0
        while total_added < n_lost and it < max_iters:
            compensate_mask = dilated
            dilated = binary_dilation(compensate_mask, structure=struct, iterations=1, mask=candidate)
            new_comp = dilated & ~compensate_mask
            total_added += int(new_comp.sum())
            it += 1
        seg[dilated] = cfg.compensate_label

    return seg


# ── WMH injection ─────────────────────────────────────────────────────────────


def _grow_wmh_blob(center, wm_mask, rng, n_steps, struct, keep_prob=0.65):
    """Grow one contiguous, WM-confined lesion blob from a seed voxel.

    Stochastic morphological dilation: each step dilates the current blob by one
    voxel (6-connectivity), then keeps the new front voxels that fall inside WM
    with probability `keep_prob`, giving an organic (non-spherical) boundary.
    Operates on a small window around `center` for speed.

    Returns (blob_sub, lo): the bool sub-volume and its origin in the full array.
    """
    shape = wm_mask.shape
    pad = n_steps + 1
    lo = [max(0, int(center[d]) - pad) for d in range(3)]
    hi = [min(shape[d], int(center[d]) + pad + 1) for d in range(3)]
    sub_wm = wm_mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    blob = np.zeros_like(sub_wm)
    blob[int(center[0]) - lo[0], int(center[1]) - lo[1], int(center[2]) - lo[2]] = True
    for _ in range(n_steps):
        front = binary_dilation(blob, structure=struct) & sub_wm & ~blob
        if not front.any():
            break
        keep = front & (rng.random(front.shape) < keep_prob)
        blob |= keep
    return blob, lo


def inject_wmh(
    label_map,
    wmh_label=77,
    wm_labels=None,
    vent_labels=None,
    fraction=0.05,
    seed=42,
    label_set="freesurfer",
    max_radius=3,
):
    """Inject contiguous, periventricular white-matter-hyperintensity lesions.

    Unlike a per-voxel speckle, lesions are grown as smooth blobs:

      1. A subject-fixed propensity field (depends only on `seed` and volume
         shape — *not* the atrophied geometry) is multiplied by a periventricular
         weight 1/(dist_to_ventricle + 1). Seeding from this field means the same
         `seed` places lesions in the same anatomical locations regardless of the
         atrophy level, so a subject's WMH pattern is stable across alphas.
      2. Seeds are taken in descending score order (highest-propensity
         periventricular WM first); each uncovered seed is grown into a
         contiguous, WM-confined blob by `_grow_wmh_blob`.
      3. Blobs accumulate until the lesion volume reaches `fraction` of total WM.

    Parameters
    ----------
    fraction   : target lesion volume as a fraction of total white matter.
    seed       : RNG seed — pass a per-subject value for cross-alpha stability.
    max_radius : cap on per-lesion growth steps (≈ blob radius in voxels). Larger
                 → fewer, bigger, more confluent lesions.
    """
    seg = label_map.copy()
    rng = np.random.default_rng(seed)

    if wm_labels is None:
        wm_labels = get_wm_labels(label_set)
    if vent_labels is None:
        vent_labels = get_vent_labels(label_set)

    wm_mask = np.isin(seg, list(wm_labels))
    vent_mask = np.isin(seg, list(vent_labels))
    n_wm = int(wm_mask.sum())
    if n_wm == 0:
        return seg

    target_voxels = int(n_wm * fraction)
    if target_voxels < 1:
        return seg

    # Periventricular weight × subject-fixed propensity → per-voxel seed score.
    dist_to_vent = distance_transform_edt(~vent_mask)
    periv = 1.0 / (dist_to_vent + 1.0)
    propensity = gaussian_filter(rng.random(seg.shape, dtype=np.float32), sigma=2.0)
    score = (propensity * periv).astype(np.float32)
    score[~wm_mask] = -1.0  # only seed inside white matter

    struct = generate_binary_structure(3, 1)  # 6-connectivity
    wmh_mask = np.zeros(seg.shape, dtype=bool)

    # Walk candidate seeds from highest score; grow a blob at each uncovered one
    # until the target lesion volume is reached.
    order = np.argsort(score, axis=None)[::-1]
    placed = 0
    for flat in order:
        if placed >= target_voxels or score.flat[flat] <= 0:
            break  # target reached, or exhausted WM seed candidates
        center = np.unravel_index(int(flat), seg.shape)
        if wmh_mask[center]:
            continue  # already inside a previous blob
        n_steps = int(rng.integers(1, max_radius + 1))
        blob, lo = _grow_wmh_blob(center, wm_mask, rng, n_steps, struct)
        window = wmh_mask[lo[0] : lo[0] + blob.shape[0], lo[1] : lo[1] + blob.shape[1], lo[2] : lo[2] + blob.shape[2]]
        placed += int((blob & ~window).sum())
        window |= blob

    seg[wmh_mask] = wmh_label
    return seg


# ── Label smoothing ───────────────────────────────────────────────────────────


def smooth_label_boundaries(label_map, sigma=0.6, labels_to_smooth=None):
    unique = np.unique(label_map) if labels_to_smooth is None else np.array(labels_to_smooth)

    score = np.zeros_like(label_map, dtype=np.float32)
    best = label_map.copy()

    for lbl in unique:
        channel = gaussian_filter((label_map == lbl).astype(np.float32), sigma=sigma)
        mask = channel > score
        score[mask] = channel[mask]
        best[mask] = lbl

    return best


# ── High-level API ────────────────────────────────────────────────────────────


def simulate_atrophy(
    seg_path,
    alpha,
    configs=None,
    label_set=None,
    wmh_fraction=0.0,
    wmh_label=77,
    wmh_seed=42,
    wmh_max_radius=3,
    smooth_sigma=0.6,
    out_path=None,
):
    """Full pipeline: load → erode structures → inject WMH → smooth → save.

    Parameters
    ----------
    seg_path   : path to segmentation (FreeSurfer aseg or MUSE)
    alpha      : global severity in [0, 1]
    configs    : per-structure erosion configs; None = auto from label_set
    label_set  : 'freesurfer', 'muse', or None (auto-detect)
    wmh_seed   : RNG seed for WMH placement; pass a per-subject value (constant
                 across alphas) to keep a subject's lesion pattern stable.
    wmh_max_radius : per-lesion blob radius cap (voxels) for inject_wmh.
    """
    img = nib.load(seg_path)
    seg = np.asarray(img.dataobj).astype(np.int32)

    if label_set is None:
        label_set = detect_label_set(seg)
    if configs is None:
        configs = get_configs(label_set)

    for cfg in configs:
        if (seg == cfg.label).any():
            seg = erode_structure(seg, cfg, alpha)

    if wmh_fraction > 0:
        seg = inject_wmh(
            seg,
            wmh_label=wmh_label,
            fraction=wmh_fraction,
            label_set=label_set,
            seed=wmh_seed,
            max_radius=wmh_max_radius,
        )

    if smooth_sigma > 0:
        seg = smooth_label_boundaries(seg, sigma=smooth_sigma)

    out_img = nib.Nifti1Image(seg, img.affine, img.header)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(out_img, out_path)
    return out_img


def generate_atrophy_spectrum(seg_path, alphas, out_dir, subject_id, **kwargs):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for alpha in alphas:
        out = os.path.join(out_dir, f"{subject_id}_alpha{alpha:.2f}_seg.nii.gz")
        simulate_atrophy(seg_path, alpha=alpha, out_path=out, **kwargs)
        paths.append(out)
    return paths
