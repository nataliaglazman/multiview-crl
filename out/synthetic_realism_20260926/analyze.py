"""Read-only renderer audit; run from the repository root. No model is loaded.

Uses two saved local configurations, not an assumed copy of a remote experiment.
Raw intensities have arbitrary units. Counterfactual response is NOT an SNR/CNR
or a measure of lesion detectability: it uses the known lesion mask and a paired
lesion-removed image with exactly the same noise and bias field.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.datasets import SyntheticBrainDataset

OUT = Path(__file__).resolve().parent
N = 64
torch.set_num_threads(2)


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return (
        {
            "n": len(a),
            "p10": float(np.quantile(a, 0.1)),
            "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)),
        }
        if len(a)
        else {"n": 0}
    )


def make_dataset(config):
    params = inspect.signature(SyntheticBrainDataset.__init__).parameters
    kw = {k: v for k, v in config.items() if k.startswith("synthetic_") and k in params}
    for src, dst in [
        ("n_content", "synthetic_n_content"),
        ("n_style", "synthetic_n_style"),
        ("seed", "synthetic_seed"),
    ]:
        if src in config and dst not in kw:
            kw[dst] = config[src]
    kw.update(synthetic_num_samples=N, synthetic_num_samples_per_mode=None)
    res = config.get("synthetic_res", config.get("res", 64))
    spatial = config.get("spatial_size") or [res] * 3
    return SyntheticBrainDataset(mode="test", cache=False, spatial_size=spatial, **kw), kw


def montage(name, examples, rows):
    # Systematically select the 10th, 50th and 90th percentiles of the *measured*
    # T1 lesion response. These examples are not a random sample or a detection test.
    order = np.argsort([r["t1_lesion_response_raw"] for r in rows])
    chosen = [int(order[round(q * (N - 1))]) for q in [0.1, 0.5, 0.9]]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
    for rownum, idx in enumerate(chosen):
        volumes, lesion = examples[idx]
        center = np.argwhere(lesion).mean(axis=0)
        z = int(round(center[2]))
        cx, cy = center[:2]
        xs = slice(max(0, int(cx) - 8), min(64, int(cx) + 9))
        ys = slice(max(0, int(cy) - 8), min(64, int(cy) + 9))
        for j, (view, vol) in enumerate(zip(["T1", "FLAIR"], volumes)):
            ax = axes[rownum, j]
            ax.imshow(vol[:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=1.15)
            ax.add_patch(plt.Circle((cx, cy), 4.5, fill=False, color="#f8bf40", lw=1))
            ax.set_title(f"{view} · #{idx} · slice {z}", fontsize=10)
            zoom = axes[rownum, j + 2]
            zoom.imshow(vol[xs, ys, z].T, origin="lower", cmap="gray", vmin=0, vmax=1.15)
            zoom.set_title(f"{view} lesion zoom", fontsize=10)
        for ax in axes[rownum]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[rownum, 0].set_ylabel(["Low T1 response", "Median T1 response", "High T1 response"][rownum])
    fig.suptitle(
        f"{name}: synthetic inputs, not reconstructions\n"
        "Same plane in both views; fixed display window 0–1.15 raw units; circle marks lesion",
        fontsize=13,
    )
    fig.savefig(OUT / f"{name}_examples.png", dpi=150)
    plt.close(fig)
    return chosen


def audit(name, config_path):
    config = json.loads(config_path.read_text())
    ds, kw = make_dataset(config)
    inner, renderer = ds._inner, ds._inner.renderer
    rows, examples = [], []
    cube = np.ones((3, 3, 3), bool)
    with torch.inference_mode():
        for idx in range(N):
            t1, flair, lat = inner[idx]
            tissue, load = renderer.render_structure(
                lat["z_content"],
                lat["z_deformation"],
                lat["z_fissure"],
                "cpu",
                clean=inner.clean_content,
                z_lesion=lat.get("z_lesion"),
            )
            labels, lesion = tissue.numpy(), load.numpy() > 0.5
            assert lesion.any()
            seed = inner.sample_seed_for(idx)
            row = {
                "index": idx,
                "accepted_seed": seed,
                "lesion_voxels": int(lesion.sum()),
                "lesion_fraction_brain": float(lesion.sum() / (labels > 0).sum()),
                "lesion_fraction_outside_final_wm": float((lesion & (labels != 2)).sum() / lesion.sum()),
            }
            excluded = binary_dilation(lesion, structure=cube)
            masks = {label: binary_erosion(labels == label, structure=cube) & ~excluded for label in [1, 2, 3]}
            for view_num, (view, image, style_key) in enumerate(
                [("t1", t1, "z_style_v1"), ("flair", flair, "z_style_v2")]
            ):
                style = lat[style_key]
                if idx == 0:
                    replay = renderer.render_modality(tissue, load, style, view.upper(), seed * 2 + view_num, "cpu")
                    assert torch.equal(replay, image), "View seed or rendering path does not reproduce the dataset"
                image = image.squeeze().numpy()
                off = (
                    renderer.render_modality(
                        tissue, torch.zeros_like(load), style, view.upper(), seed * 2 + view_num, "cpu"
                    )
                    .squeeze()
                    .numpy()
                )
                delta = np.abs(image - off)
                row[f"{view}_lesion_response_raw"] = float(delta[lesion].mean())
                gain = max(0.05, 1 + float(style[0].clamp(-1, 1)) * 0.3 * renderer.style_scale)
                bias = float(style[1].clamp(-1, 1)) * 0.1 * renderer.style_scale
                wm, li = (0.8, 0.4) if view == "t1" else (0.4, 1.0)
                row[f"{view}_gain"] = gain
                row[f"{view}_bias"] = bias
                row[f"{view}_nominal_wm_lesion_contrast"] = abs(wm * gain + bias - li)
                for label, label_name in [(1, "csf"), (2, "wm"), (3, "gm")]:
                    roi = masks[label]
                    row[f"{view}_{label_name}_interior_raw"] = float(np.median(image[roi])) if roi.any() else None
                csf, wm_val = row[f"{view}_csf_interior_raw"], row[f"{view}_wm_interior_raw"]
                row[f"{view}_wm_csf_difference_raw"] = wm_val - csf if csf is not None and wm_val is not None else None
            # Display skull-stripped raw images: no centering/rescaling; outside brain black.
            examples.append(([v.squeeze().numpy() * (labels > 0) for v in [t1, flair]], lesion))
            rows.append(row)
    # Exercise the actual fixed-reference normalization, whose statistics use 64
    # subjects from this dataset split. Save constants; never compare raw units to
    # normalized model losses without accounting for this shared affine scaling.
    ds._compute_fixed_reference()
    for row in rows:
        for view in ["t1", "flair"]:
            row[f"{view}_lesion_response_normalized"] = row[f"{view}_lesion_response_raw"] / ds._fixed_scale
    stats = {
        key: describe([r[key] for r in rows if r[key] is not None])
        for key in rows[0]
        if key not in ["index", "accepted_seed"]
    }
    summary = {
        "name": name,
        "source_config": str(config_path),
        "dataset_arguments": kw,
        "source_settings": config,
        "n": N,
        "split": "test",
        "resolution": ds.res,
        "effective_lesion_placement": renderer.lesion_placement,
        "effective_identifiable_ventricle": renderer.identifiable_ventricle,
        "fixed_mean": ds._fixed_mean,
        "fixed_scale": ds._fixed_scale,
        "statistics": stats,
        "displayed_subjects": montage(name, examples, rows),
        "rows": rows,
    }
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        name,
        json.dumps(
            {k: v for k, v in stats.items() if "response" in k or "outside_final" in k or "wm_csf" in k}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    provenance = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "eval/synthetic_dataset.py", ROOT / "data/datasets.py"]
    }
    (OUT / "source_hashes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    audit("wm_interior_noncausal", ROOT / "settings.json")
    audit("vqvae_identifiable_ventricle", ROOT / "synthetic-clean-content-causal-sp-s-1/settings.json")
