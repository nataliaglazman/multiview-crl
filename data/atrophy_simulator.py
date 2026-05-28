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


def inject_wmh(
    label_map, wmh_label=77, wm_labels=None, vent_labels=None, fraction=0.05, seed=42, label_set="freesurfer"
):
    seg = label_map.copy()
    rng = np.random.default_rng(seed)

    if wm_labels is None:
        wm_labels = get_wm_labels(label_set)
    if vent_labels is None:
        vent_labels = get_vent_labels(label_set)

    wm_mask = np.isin(seg, list(wm_labels))
    vent_mask = np.isin(seg, list(vent_labels))

    dist_to_vent = distance_transform_edt(~vent_mask)
    wm_coords = np.argwhere(wm_mask)

    if len(wm_coords) == 0:
        return seg

    dists = dist_to_vent[wm_mask]
    weights = 1.0 / (dists + 1.0)
    weights /= weights.sum()

    n_lesion = max(1, int(wm_mask.sum() * fraction))
    chosen_idx = rng.choice(len(wm_coords), size=n_lesion, replace=False, p=weights)
    chosen = wm_coords[chosen_idx]
    seg[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = wmh_label
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
        seg = inject_wmh(seg, wmh_label=wmh_label, fraction=wmh_fraction, label_set=label_set)

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
