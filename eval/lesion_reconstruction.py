"""Quick lesion-location audit; no probe fitting or checkpoint updates.

    python -m eval.lesion_reconstruction --run-dir /path/to/run --causal iid

Reports an image-only FLAIR blob detector on inputs and reconstructions, plus a
paired intervention: remove only the lesion, hold anatomy/style/noise and the
original normalization affine fixed, and reconstruct again. Localize the absolute
reconstruction difference without using the target location. Compare with the
actual rendered lesion-mask centroid, in voxel units, and shuffled-subject targets.

The intervention tests localized sensitivity of the JOINT decoder input, not
content/style separation. Lesion-free images are counterfactual and may be outside
the training distribution. Weak intervention scores alone do not prove information
loss. The direct detector is only interpretable when it works on the inputs.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

logger = logging.getLogger(__name__)


def locate(volume, radius_vox, response=False):
    """Fixed detector, searching the whole image; never sees a GT mask/position.

    For an intervention, find the largest smooth absolute response. For a FLAIR
    image, use a bright-blob difference of Gaussians. No-response cases stay missing.
    """
    a = np.asarray(volume, dtype=np.float64).squeeze()
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 3-D volume")
    sigma = max(0.5, radius_vox / 2)
    score = gaussian_filter(np.abs(a) if response else a, sigma)
    if not response:
        score -= gaussian_filter(a, max(1.0, radius_vox * 1.5))
    peak = float(score.max())
    if peak <= 1e-8:
        return np.full(3, np.nan)
    return np.array(np.unravel_index(score.argmax(), score.shape), dtype=float)


def location_metrics(truth, pred, seed=0):
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    valid = np.isfinite(truth).all(1) & np.isfinite(pred).all(1)
    result = {
        "n": len(truth),
        "n_valid": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
    }
    if not valid.any():
        return result
    t, p = truth[valid], pred[valid]
    error = np.linalg.norm(t - p, axis=1)
    ss = ((t - t.mean(0)) ** 2).sum(0)
    r2 = np.divide(((t - p) ** 2).sum(0), ss, out=np.full(3, np.nan), where=ss > 1e-12)
    rng = np.random.default_rng(seed)
    shuffled = [np.median(np.linalg.norm(t - p[rng.permutation(len(p))], axis=1)) for _ in range(100)]
    result.update(
        median_error_vox=float(np.median(error)),
        p90_error_vox=float(np.quantile(error, 0.9)),
        r2_xyz=(1 - r2).tolist(),
        within_2_vox_fraction_all=float(np.sum(error <= 2) / len(truth)),
        shuffled_median_error_vox=float(np.mean(shuffled)),
    )
    return result


def make_dataset(args, n, causal, split):
    from data.datasets import SyntheticBrainDataset

    accepted = inspect.signature(SyntheticBrainDataset.__init__).parameters
    kw = {k: v for k, v in vars(args).items() if k.startswith("synthetic_") and k in accepted}
    if kw.get("synthetic_mode", "pseudo_mri") != "pseudo_mri":
        raise ValueError("This diagnostic requires synthetic_mode=pseudo_mri")
    if kw.get("synthetic_lesion_mode", "sphere") != "sphere":
        raise ValueError("Field lesions have no lesion_x/y/z position target; use sphere mode")
    if kw.get("synthetic_n_content", 9) < 5:
        raise ValueError("Need at least five content dimensions for all three lesion coordinates")
    if causal == "iid":
        kw.update(synthetic_causal=False, synthetic_hierarchical_content=False)
    kw.update(synthetic_num_samples=n, synthetic_num_samples_per_mode=None)
    res = getattr(args, "synthetic_res", 64)
    return SyntheticBrainDataset(
        mode=split,
        spatial_size=getattr(args, "spatial_size", None) or (res,) * 3,
        cache=False,
        **kw,
    )


def render_pair(ds, idx):
    """Lesion on/off pair with exactly the same view seeds and normalization."""
    import torch

    inner, renderer = ds._inner, ds._inner.renderer
    raw1, raw2, lat = inner[idx]
    tissue, lesion = renderer.render_structure(
        lat["z_content"],
        lat["z_deformation"],
        lat["z_fissure"],
        "cpu",
        clean=inner.clean_content,
    )
    mask = lat["brain_mask"]
    raw = [raw1, raw2]
    on = list(ds.normalize_views(raw1, raw2, mask, mask))
    off = []
    for view, modality in enumerate(("T1", "FLAIR")):
        blank = renderer.render_modality(
            tissue,
            torch.zeros_like(lesion),
            lat[f"z_style_v{view + 1}"],
            modality,
            inner.sample_seed_for(idx) * 2 + view,
            "cpu",
        )
        # Every supported normalizer is affine within foreground. Freeze the ON
        # transform so removing a lesion cannot create a global normalization cue.
        x, y = raw[view][mask > 0].double(), on[view][mask > 0].double()
        xc = x - x.mean()
        if float(xc.square().sum()) < 1e-12:
            raise ValueError("Degenerate input: cannot recover its normalization affine")
        gain = (xc * (y - y.mean())).sum() / xc.square().sum()
        bias = y.mean() - gain * x.mean()
        off.append((blank * gain.float() + bias.float()) * mask)
    support = lesion.numpy() > 0
    truth = np.argwhere(support).mean(0) if support.any() else np.full(3, np.nan)
    return on, off, mask, support, truth, lat["z_content"].numpy()[2:5]


def reconstruct(model, images, masks, device):
    import torch

    # View-major batches are required by separate encoders and per-view codebooks.
    x = torch.cat([torch.stack([s[v] for s in images]) for v in range(2)]).to(device)
    m = torch.cat([torch.stack(masks)] * 2).to(device)
    with torch.inference_mode():
        y = model(x, return_recon=True, pool_only=True, n_views=2, subsets=[(0, 1)], mask=m)[0]
    if y is None or y.shape != x.shape or not bool(torch.isfinite(y).all()):
        raise ValueError("Missing, wrong-shape, or non-finite reconstruction")
    # Match the reconstruction loss/display foreground; unconstrained decoder
    # background must not win the image-only location search. Do not clamp values.
    return (y * m).detach().cpu().numpy().reshape(2, len(images), *y.shape[1:])[:, :, 0]


def audit(model, ds, device, batch_size=4, examples=6):
    rows, panels = [], []
    radius_vox = ds._inner.renderer.lesion_radius * (ds._inner.res - 1) / 2
    for start in range(0, len(ds), batch_size):
        samples = [render_pair(ds, i) for i in range(start, min(start + batch_size, len(ds)))]
        masks = [s[2] for s in samples]
        on_y = reconstruct(model, [s[0] for s in samples], masks, device)
        off_y = reconstruct(model, [s[1] for s in samples], masks, device)
        for b, (on, off, mask, support, truth, latent) in enumerate(samples):
            row = {"index": start + b, "lesion_voxels": int(support.sum())}
            for axis in range(3):
                row[f"truth_{axis}"] = truth[axis]
                row[f"latent_{axis}"] = latent[axis]
            estimates = {
                "input_flair_blob": locate(on[1].numpy(), radius_vox),
                "recon_flair_blob": locate(on_y[1, b], radius_vox),
            }
            for v, name in enumerate(("t1", "flair")):
                dx = (on[v] - off[v]).numpy().squeeze()
                dy = (on_y[v, b] - off_y[v, b]) * mask.numpy().squeeze()
                estimates[f"input_{name}_response"] = locate(dx, radius_vox, response=True)
                estimates[f"recon_{name}_response"] = locate(dy, radius_vox, response=True)
                # Blur expands lesion support by one voxel in the renderer.
                roi = maximum_filter(support, size=3)
                target, effect = dx[roi], dy[roi]
                energy = float(np.square(target).sum())
                row[f"{name}_response_gain"] = float(np.dot(target, effect) / energy) if energy > 1e-12 else np.nan
                row[f"{name}_response_rms_ratio"] = (
                    float(np.linalg.norm(effect) / np.sqrt(energy)) if energy > 1e-12 else np.nan
                )
                row[f"{name}_response_energy_in_lesion_roi"] = float(
                    np.square(effect).sum() / max(np.square(dy).sum(), 1e-12)
                )
            for name, xyz in estimates.items():
                for axis in range(3):
                    row[f"{name}_{axis}"] = xyz[axis]
            rows.append(row)
            if len(panels) < examples and np.isfinite(truth).all():
                z = int(round(truth[2]))
                panels.append(
                    (
                        start + b,
                        truth,
                        z,
                        [
                            on[1].numpy().squeeze()[:, :, z],
                            on_y[1, b, :, :, z],
                            (on[1] - off[1]).numpy().squeeze()[:, :, z],
                            (on_y[1, b] - off_y[1, b])[:, :, z],
                        ],
                    )
                )
        logger.info("Scored %d/%d subjects", len(rows), len(ds))
    truth = [[r[f"truth_{a}"] for a in range(3)] for r in rows]
    summary = {
        name: location_metrics(truth, [[r[f"{name}_{a}"] for a in range(3)] for r in rows]) for name in estimates
    }
    summary["response"] = {
        key: float(np.nanmedian([r[key] for r in rows]))
        for key in rows[0]
        if "response_gain" in key or "rms_ratio" in key or "energy_in" in key
    }
    summary["empty_lesions"] = sum(r["lesion_voxels"] == 0 for r in rows)
    return rows, summary, panels


def save_panels(panels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 4, figsize=(12, 3 * len(panels)), squeeze=False)
    titles = [
        "Input FLAIR",
        "Reconstruction FLAIR",
        "Input: lesion on − off",
        "Recon: lesion on − off",
    ]
    for row, (idx, truth, z, slices) in enumerate(panels):
        lo, hi = np.min(slices[0]), np.max(slices[0])
        scale = max(float(np.max(np.abs(slices[2]))), 1e-8)
        for col, a in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(
                a.T,
                origin="lower",
                cmap="gray" if col < 2 else "coolwarm",
                vmin=lo if col < 2 else -scale,
                vmax=hi if col < 2 else scale,
            )
            ax.plot(truth[0], truth[1], "+", color="lime", markersize=9)
            ax.set_title(titles[col] if row == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"sample {idx}, z={z}")
    fig.suptitle("Green cross: rendered lesion centroid; paired panels share intensity scales")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--causal", choices=["iid", "match"], default="iid")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Subjects per forward (two views each)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=2, help="CPU rendering/inference threads")
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--out-dir", default=None)
    cli = p.parse_args()
    if cli.num_samples < 2 or cli.batch_size < 1 or cli.examples < 0 or cli.cpu_threads < 1:
        p.error("Need >=2 samples, positive batch size/threads, and nonnegative examples")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir

    torch.set_num_threads(cli.cpu_threads)
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device=cli.device, seed=0)
    ds = make_dataset(args, cli.num_samples, cli.causal, cli.split)
    rows, summary, panels = audit(model, ds, device, cli.batch_size, cli.examples)
    directory = Path(cli.out_dir or Path(cli.run_dir) / f"lesion_reconstruction_{cli.causal}")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": vars(cli),
        "run_settings": vars(args),
        "metrics": summary,
        "target": "Rendered lesion-mask centroid in voxel indices, not z_content",
        "counterfactual": "Lesion removed; original normalization affine frozen",
        "limitations": "Blob detection needs input calibration. On/off tests joint decoder sensitivity; off images may be OOD.",
    }
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    with (directory / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_panels(panels, directory / "examples.png")
    print("\nLocation errors in voxels; R² is against physical mask centroid (x/y/z).")
    for name, m in summary.items():
        if not isinstance(m, dict) or "n_valid" not in m:
            continue
        if not m["n_valid"]:
            print(f"  {name:24s} no valid location response")
            continue
        r2 = "/".join(f"{v:.2f}" for v in m["r2_xyz"])
        print(
            f"  {name:24s} median={m['median_error_vox']:.2f}  shuffled={m['shuffled_median_error_vox']:.2f}"
            f"  R²={r2}  valid={m['n_valid']}/{m['n']}"
        )
    print("\nResponse gain (projection onto input lesion effect; identity=1, no response=0):")
    for v in ("t1", "flair"):
        print(f"  {v}: {summary['response'][v + '_response_gain']:.3f}")
    print("\nRead input detector scores first. Good recon localization AND appreciable response gain")
    print("support retained lesion-location information in joint decoder inputs. Weak results are")
    print("inconclusive if input localization fails or the lesion-removal intervention is OOD.")
    print(f"Saved {directory}")


if __name__ == "__main__":
    main()
