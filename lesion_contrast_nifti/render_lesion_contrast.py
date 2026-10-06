"""Render encoder-only synthetic subjects with --synthetic-lesion-intensity fixed vs styled.

Writes NIfTIs (encoder-input normalisation, fixed_reference) for
  * natural/   subjects at the T1-gain quantiles, both modes, T1+FLAIR, lesion mask, tissue
  * sweep/     one subject swept over gain z in {-1,-.5,0,.5,1} (bias 0, noise 0) as a 4D volume
plus contrast.csv (signed lesion contrast vs a lesion-free twin) and overview.png.
"""

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.encoder.encoder_lesion_contrast import contrast_metrics  # noqa: E402
from eval.protocol.score_checkpoint import make_dataset  # noqa: E402

OUT = Path(__file__).resolve().parent
N_SCAN = 120
SWEEP_Z = [-1.0, -0.5, 0.0, 0.5, 1.0]

cfg = dict(json.loads((ROOT / "experiments/encoder_comparison.json").read_text())["shared"])
cfg["data_seed"] = cfg["seed"] = 42
datasets = {m: make_dataset(dict(cfg, synthetic_lesion_intensity=m), N_SCAN, mode="test") for m in ("fixed", "styled")}
for ds in datasets.values():
    ds._compute_fixed_reference()
inner = datasets["fixed"]._inner
r_fixed, r_styled = datasets["fixed"]._inner.renderer, datasets["styled"]._inner.renderer
renderers = {"fixed": r_fixed, "styled": r_styled}
VIEWS = {"T1": ("z_style_v1", 0), "FLAIR": ("z_style_v2", 1)}
affine = np.diag([2.0, 2.0, 2.0, 1.0])


def save(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine), str(path))


def norm(ds, x, brain):
    return ((x - ds._fixed_mean) / ds._fixed_scale * brain).numpy()[0]


def gain_of(z):
    return float((1.0 + z.clamp(-1, 1) * 0.3).clamp_min(0.05))


def render(mode, tissue, load, z_style, view, seed):
    _, k = VIEWS[view]
    return renderers[mode].render_modality(tissue, load, z_style, view, view_seed=seed * 2 + k, device="cpu")


# Structure (identical in both modes) + latents for every scanned subject.
subjects = []
for idx in range(N_SCAN):
    seed, d, _ = inner._first_fitting(
        idx,
        lambda s, d: inner.render_pseudo_mri(
            d["z_content"],
            d["z_deformation"],
            d["z_fissure"],
            d["z_style_v1"],
            d["z_style_v2"],
            s,
            z_lesion=d["z_lesion"],
        ),
    )
    tissue, lesion = r_fixed.render_structure(d["z_content"], d["z_deformation"], d["z_fissure"], "cpu", clean=True)
    subjects.append(dict(idx=idx, seed=seed, d=d, tissue=tissue, lesion=lesion))

rows = []
for s in subjects:
    mask, tis = s["lesion"].numpy() > 0, s["tissue"].numpy()
    for view, (key, _) in VIEWS.items():
        z = s["d"][key]
        for mode in ("fixed", "styled"):
            img = render(mode, s["tissue"], s["lesion"], z, view, s["seed"]).numpy()[0]
            ref = render(mode, s["tissue"], torch.zeros_like(s["lesion"]), z, view, s["seed"]).numpy()[0]
            m = contrast_metrics(img, ref, mask, tis)
            rows.append(
                dict(
                    subject=s["idx"],
                    view=view,
                    mode=mode,
                    gain=gain_of(z[0]),
                    bias=float(z[1].clamp(-1, 1) * 0.1),
                    signed_contrast=m["signed_matched_contrast"],
                    lesion_mean=m["lesion_mean"],
                    wm_mean=m["reference_wm_mean"],
                )
            )

with open(OUT / "contrast.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# Natural subjects at T1-gain quantiles (min, 25, 50, 75, max).
t1_gain = np.array([gain_of(s["d"]["z_style_v1"][0]) for s in subjects])
order = np.argsort(t1_gain)
picks = [int(order[int(round(q * (N_SCAN - 1)))]) for q in (0, 0.25, 0.5, 0.75, 1.0)]
for rank, i in enumerate(picks):
    s = subjects[i]
    brain = (s["tissue"] > 0).unsqueeze(0).float()
    stem = OUT / "natural" / f"q{rank}_subj{s['idx']:03d}_T1gain{t1_gain[i]:.2f}"
    save(s["lesion"].numpy(), stem / "lesion_mask.nii.gz")
    save(s["tissue"].numpy(), stem / "tissue.nii.gz")
    for view, (key, _) in VIEWS.items():
        for mode, ds in datasets.items():
            x = render(mode, s["tissue"], s["lesion"], s["d"][key], view, s["seed"])
            save(norm(ds, x, brain), stem / f"{view}_{mode}.nii.gz")

# Controlled gain sweep on the median-gain subject: bias 0, noise 0.
s = subjects[picks[2]]
brain = (s["tissue"] > 0).unsqueeze(0).float()
save(s["lesion"].numpy(), OUT / "sweep" / "lesion_mask.nii.gz")
sweep = {}
for view in VIEWS:
    for mode, ds in datasets.items():
        vols = [
            norm(ds, render(mode, s["tissue"], s["lesion"], torch.tensor([g, 0.0, 0.0]), view, s["seed"]), brain)
            for g in SWEEP_Z
        ]
        sweep[view, mode] = vols
        save(np.stack(vols, -1), OUT / "sweep" / f"{view}_{mode}_gain_0.7-0.85-1.0-1.15-1.3.nii.gz")

# Overview figure: lesion-centred axial slice, sweep columns, rows = view x mode.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mask = s["lesion"].numpy() > 0
cz = int(np.round(np.argwhere(mask).mean(0)[2]))
fig, axes = plt.subplots(
    4, len(SWEEP_Z) + 1, figsize=(15, 10.5), gridspec_kw=dict(width_ratios=[1] * len(SWEEP_Z) + [1.3])
)
for r, (view, mode) in enumerate([("T1", "fixed"), ("T1", "styled"), ("FLAIR", "fixed"), ("FLAIR", "styled")]):
    vols = sweep[view, mode]
    lo, hi = min(v[mask].min() for v in vols) - 1.0, max(v.max() for v in vols)
    for c, (g, v) in enumerate(zip(SWEEP_Z, vols)):
        ax = axes[r, c]
        ax.imshow(v[:, :, cz].T, cmap="gray", origin="lower", vmin=-1.6, vmax=1.6)
        ax.contour(mask[:, :, cz].T, levels=[0.5], colors="r", linewidths=0.6, origin="lower")
        ax.set_xticks([]), ax.set_yticks([])
        if r == 0:
            ax.set_title(f"gain {1 + 0.3 * g:.2f}")
        if c == 0:
            ax.set_ylabel(f"{view} / {mode}", fontsize=12)
    sub = [x for x in rows if x["view"] == view and x["mode"] == mode]
    ax = axes[r, -1]
    ax.scatter(
        [x["gain"] for x in sub],
        [x["signed_contrast"] for x in sub],
        s=8,
        c=[x["bias"] for x in sub],
        cmap="coolwarm",
        vmin=-0.1,
        vmax=0.1,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("gain (natural draws, colour = bias)")
    ax.set_ylabel("lesion − WM (raw)")
fig.suptitle("Lesion contrast across style gain: fixed vs styled lesion intensity (encoder-only recipe, test split)")
fig.tight_layout()
fig.savefig(OUT / "overview.png", dpi=120)

for view in VIEWS:
    for mode in ("fixed", "styled"):
        c = np.array([x["signed_contrast"] for x in rows if x["view"] == view and x["mode"] == mode])
        print(
            f"{view:5s} {mode:6s} contrast  min {c.min():+.3f}  median {np.median(c):+.3f}  max {c.max():+.3f}  "
            f"|max|/|min| {np.abs(c).max() / np.abs(c).min():.1f}"
        )
print("picked natural subjects:", [(subjects[i]["idx"], round(float(t1_gain[i]), 2)) for i in picks])
