import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from data.datasets import SyntheticBrainDataset
from eval.identifiability_metrics import cv_probe_r2
from eval.lesion_reconstruction import locate
from eval.lesion_visibility import locate_within

torch.set_num_threads(4)
n = 128
ds = SyntheticBrainDataset(
    mode="val",
    spatial_size=(64,) * 3,
    cache=False,
    synthetic_mode="pseudo_mri",
    synthetic_seed=42,
    synthetic_num_samples=n,
    synthetic_n_content=9,
    synthetic_n_style=3,
    synthetic_clean_content=True,
    synthetic_normalize="fixed_reference",
    synthetic_causal=False,
    synthetic_hierarchical_content=False,
)
inner = ds._inner
rows = []
targets = []
centres = []
with torch.inference_mode():
    for i in range(n):
        a, b, lat = inner[i]
        tissue, load = inner.renderer.render_structure(
            lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=True
        )
        support = load.bool()
        mask = lat["brain_mask"]
        coords = torch.nonzero(support).float()
        centre = coords.mean(0).numpy()
        centres.append(centre)
        targets.append(lat["z_content"][2:5].numpy())
        on = ds.normalize_views(a, b, mask, mask)
        offraw = [
            inner.renderer.render_modality(
                tissue, torch.zeros_like(load), lat[f"z_style_v{v+1}"], mod, inner.sample_seed_for(i) * 2 + v, "cpu"
            )
            for v, mod in enumerate(("T1", "FLAIR"))
        ]
        off = ds.normalize_views(*offraw, mask, mask)
        row = dict(
            index=i,
            lesion_voxels=int(support.sum()),
            brain_fraction=float(support.sum() / mask.sum()),
            csf_overlap=float((tissue[support] == 1).float().mean()),
            wm_overlap=float((tissue[support] == 2).float().mean()),
        )
        for v, view in enumerate(("t1", "flair")):
            image = on[v][0].numpy()
            delta = (on[v] - off[v])[0].numpy()
            fg = mask[0].numpy().astype(bool)
            pol = -1 if v == 0 else 1
            blind = locate(image * fg * pol, 3.15)
            wm = locate_within(image * fg * pol, 3.15, tissue.numpy() == 2)
            oracle = locate(delta * fg, 3.15, response=True)
            row[view] = dict(
                input_response_rms=float(np.sqrt((delta[fg] ** 2).mean())),
                local_abs_response=float(np.abs(delta[support.numpy()]).mean()),
                blind_error=float(np.linalg.norm(blind - centre)),
                wm_error=float(np.linalg.norm(wm - centre)),
                oracle_error=float(np.linalg.norm(oracle - centre)),
            )
        rows.append(row)
        if (i + 1) % 32 == 0:
            print(f"rendered {i+1}/{n}", flush=True)
summary = dict(
    n=n,
    seed=42,
    split="val",
    res=64,
    radius_vox=3.15,
    feature_map_side=16,
    lesion_voxels_median=float(np.median([r["lesion_voxels"] for r in rows])),
    lesion_voxels_min=min(r["lesion_voxels"] for r in rows),
    brain_fraction_median=float(np.median([r["brain_fraction"] for r in rows])),
    csf_overlap_median=float(np.median([r["csf_overlap"] for r in rows])),
    subjects_with_csf_overlap=sum(r["csf_overlap"] > 0 for r in rows),
    views={},
)
for view in ("t1", "flair"):
    s = {}
    for mode in ("blind", "wm", "oracle"):
        err = np.array([r[view][mode + "_error"] for r in rows])
        s[mode + "_median_error_vox"] = float(np.nanmedian(err))
        s[mode + "_within_radius_fraction"] = float((err <= 3.15).mean())
    s["local_abs_response_median"] = float(np.median([r[view]["local_abs_response"] for r in rows]))
    summary["views"][view] = s
centres = np.asarray(centres)
targets = np.asarray(targets)
summary["latent_from_true_centroid_cv_ridge_r2"] = [
    cv_probe_r2(centres, targets[:, j], seeds=(0,))["mean"] for j in range(3)
]
report = dict(
    summary=summary,
    rows=rows,
    notes=[
        "Generator-only diagnostic; no trained checkpoint was loaded.",
        "Natural images use fixed-reference normalization matching the specified configuration.",
        "WM-restricted and counterfactual detectors use oracle information and are diagnostic controls, not learned models.",
        "This uses the local renderer implementation; remote code revisions may differ.",
    ],
)
out = Path(__file__).resolve().with_name("visibility.json")
out.write_text(json.dumps(report, indent=2))
print(json.dumps(summary, indent=2))
print(out)
