"""Where does a lesion sit in a VQ-VAE's feature maps? Figures of the frozen encoder's features.

    python -m eval.lesion.lesion_feature_maps --run-dir results/synthetic/RUN --subjects 0 1 2 3

Each subject is rendered twice: as generated, and with only the lesion removed, holding anatomy,
style, noise and the normalisation affine (``eval.lesion.lesion_reconstruction.render_pair``).
Both pass through the frozen encoder; the difference of the level ``--level`` feature maps is the
lesion's footprint in feature space. One figure per subject, on the slice through the lesion
centre (green cross, mapped onto the coarser feature grid):

* the FLAIR and T1 inputs, and the lesion's effect on each image (lesion on minus off);
* per view, the size of the footprint over the content channels, on that slice and as the
  maximum over all slices, with the style channels' share in the title;
* the channels the lesion changes most in ``--view``: their maps with the lesion, and the change.

``summary.csv`` gives, per subject and view, where the footprint peaks, how much of it lies within
1.5 feature cells of the lesion centre, the style/content footprint ratio, and its sign coherence:
summed over the grid, how much of each content channel's change survives (|sum of change| over
sum of |change|; 1 = one sign everywhere, so averaging keeps it; 0 = its parts cancel). Coherence
does not depend on how few cells the lesion covers. No training, probes or checkpoint writes.
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

VIEWS = ("T1", "FLAIR")


def encode_pair(model, on, off, mask, device, level):
    """Features (condition [on, off], view, C, d, h, w) and each view's content channel indices."""
    import torch

    if getattr(model, "mask_mode", None) == "onthefly":
        raise ValueError("On-the-fly content masks are batch-dependent; use a fixed or learned-mask checkpoint")
    # View-major batch: [on, off] for view 1, then for view 2.
    x = torch.cat([torch.stack([on[v], off[v]]) for v in range(2)]).to(device)
    m = torch.cat([torch.stack([mask, mask])] * 2).to(device)
    with torch.inference_mode():
        out = model(x, return_recon=False, pool_only=False, n_views=2, mask=m)
    if not 0 <= level < len(out[2]):
        raise ValueError(f"Level {level} is not present in this checkpoint")
    features = out[2][level].float().cpu().numpy()
    features = features.reshape(2, 2, *features.shape[1:]).swapaxes(0, 1)
    selection = out[6].get(level) if isinstance(out[6], dict) else None
    channels = features.shape[2]
    if selection is None:
        return features, [list(range(channels))] * 2
    selections = selection if isinstance(selection, tuple) else (selection, selection)
    return features, [torch.where(s.reshape(-1).bool())[0].cpu().tolist() for s in selections]


def footprint_summary(delta, content, centre, near=1.5):
    """Per-view numbers for one subject's feature change ``delta`` (C, d, h, w)."""
    style = [c for c in range(len(delta)) if c not in content]
    energy = np.sqrt(np.square(delta[content]).sum(0))
    peak = np.array(np.unravel_index(energy.argmax(), energy.shape), dtype=float)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in energy.shape], indexing="ij"), -1)
    close = np.linalg.norm(grid - centre, axis=-1) <= near
    total = float(np.square(energy).sum())
    style_energy = float(np.square(delta[style]).sum()) if style else 0.0
    unsigned = float(np.abs(delta[content]).sum())
    signed = float(np.abs(delta[content].sum((1, 2, 3))).sum())
    return dict(
        energy=energy,
        peak=peak,
        peak_distance_cells=float(np.linalg.norm(peak - centre)),
        near_fraction=float(np.square(energy[close]).sum() / total) if total > 0 else np.nan,
        style_to_content=float(np.sqrt(style_energy / total)) if total > 0 else np.nan,
        sign_coherence=signed / unsigned if unsigned > 0 else np.nan,
    )


def plot_subject(path, index, images, effect, features, delta, content, summaries, truth, centre, view, top):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, zf = int(round(truth[2])), int(np.clip(round(centre[2]), 0, delta.shape[-1] - 1))
    v = VIEWS.index(view)
    strength = np.square(delta[v]).sum((1, 2, 3))
    channels = list(np.argsort(strength)[::-1][:top])
    columns = max(4, top)
    fig, axes = plt.subplots(4, columns, figsize=(3 * columns, 12), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_visible(False)

    def show(ax, a, title, cmap, vmin, vmax, mark):
        ax.set_visible(True)
        ax.imshow(a.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot(mark[0], mark[1], "+", color="lime", markersize=10, markeredgewidth=1.5)
        ax.set_title(title, fontsize=9)

    for col, (name, a) in enumerate(zip(("FLAIR", "T1"), (images[1], images[0]))):
        show(axes[0, col], a[:, :, z], f"{name} input", "gray", a.min(), a.max(), truth)
    for col, (name, a) in enumerate(zip(("FLAIR", "T1"), (effect[1], effect[0])), start=2):
        scale = max(float(np.abs(a).max()), 1e-8)
        show(axes[0, col], a[:, :, z], f"{name}: lesion on − off", "coolwarm", -scale, scale, truth)
    for k, name in enumerate(("T1", "FLAIR")):
        s = summaries[VIEWS.index(name)]
        top_value = max(float(s["energy"].max()), 1e-8)
        near = f"{s['near_fraction']:.0%} within 1.5 cells"
        show(
            axes[1, 2 * k],
            s["energy"][:, :, zf],
            f"{name} content footprint\nslice z={zf}; {near}",
            "magma",
            0,
            top_value,
            centre,
        )
        show(
            axes[1, 2 * k + 1],
            s["energy"].max(2),
            f"{name} footprint, max over slices\nstyle/content {s['style_to_content']:.2f}, "
            f"sign coherence {s['sign_coherence']:.2f}",
            "magma",
            0,
            top_value,
            centre,
        )
    for col, c in enumerate(channels):
        block = "content" if c in content[v] else "style"
        a = features[0, v, c]
        show(axes[2, col], a[:, :, zf], f"{view} ch {c} ({block}), with lesion", "viridis", a.min(), a.max(), centre)
        d = delta[v, c]
        scale = max(float(np.abs(d).max()), 1e-8)
        show(axes[3, col], d[:, :, zf], f"{view} ch {c}: change", "coolwarm", -scale, scale, centre)
    fig.suptitle(
        f"Subject {index}: lesion centre {np.round(truth, 1).tolist()} in voxels, "
        f"{np.round(centre, 1).tolist()} on the feature grid\n"
        f"image slice z={z}, feature slice z={zf}; green cross = lesion centre",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3], help="Indices in the split")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--causal", choices=["iid", "match"], default="iid")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Split size to build (images under fixed_reference depend on it below 64; the audits use 64)",
    )
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--view", choices=["flair", "t1"], default="flair", help="View for the per-channel rows")
    parser.add_argument("--top-channels", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if min(args.subjects) < 0 or args.top_channels < 1 or args.level < 0:
        parser.error("Subjects and the level must be nonnegative, and --top-channels positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import torch

    from eval.lesion.lesion_reconstruction import make_dataset, render_pair
    from eval.protocol.run_dci_synthetic import load_model_from_run_dir

    torch.set_num_threads(args.cpu_threads)
    model, run_args, device = load_model_from_run_dir(args.run_dir, args.checkpoint, device=args.device, seed=0)
    model.eval()
    ds = make_dataset(run_args, max(args.num_samples, max(args.subjects) + 1), args.causal, args.split)
    out_dir = args.out_dir or Path(args.run_dir) / "lesion_feature_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    view = "FLAIR" if args.view == "flair" else "T1"

    rows = []
    for index in args.subjects:
        on, off, mask, support, truth, _ = render_pair(ds, index)
        if not np.isfinite(truth).all():
            print(f"Subject {index}: no rendered lesion; skipped", flush=True)
            continue
        features, content = encode_pair(model, on, off, mask, device, args.level)
        delta = features[0] - features[1]
        factor = np.array(on[0].shape[-3:]) / np.array(delta.shape[-3:])
        centre = (truth + 0.5) / factor - 0.5
        summaries = [footprint_summary(delta[v], content[v], centre) for v in range(2)]
        images = [on[v].numpy().squeeze() for v in range(2)]
        effect = [(on[v] - off[v]).numpy().squeeze() for v in range(2)]
        path = out_dir / f"subject_{index:04d}.png"
        plot_subject(
            path, index, images, effect, features, delta, content, summaries, truth, centre, view, args.top_channels
        )
        for v, s in enumerate(summaries):
            rows.append(
                dict(
                    subject=index,
                    view=VIEWS[v],
                    lesion_voxels=int(support.sum()),
                    **{f"centre_{a}": truth[k] for k, a in enumerate("xyz")},
                    **{f"centre_feature_{a}": centre[k] for k, a in enumerate("xyz")},
                    **{f"peak_feature_{a}": s["peak"][k] for k, a in enumerate("xyz")},
                    peak_distance_cells=s["peak_distance_cells"],
                    near_fraction=s["near_fraction"],
                    style_to_content=s["style_to_content"],
                    sign_coherence=s["sign_coherence"],
                )
            )
        print(f"Saved {path}", flush=True)
    if not rows:
        raise SystemExit("No subject had a rendered lesion")
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"\n{'subject':>7} {'view':>5} {'peak→centre (cells)':>20} {'within 1.5 cells':>17} {'style/content':>14} "
        f"{'sign coherence':>15}"
    )
    for r in rows:
        print(
            f"{r['subject']:>7} {r['view']:>5} {r['peak_distance_cells']:>20.2f} {r['near_fraction']:>17.2f} "
            f"{r['style_to_content']:>14.3f} {r['sign_coherence']:>15.3f}"
        )
    print(f"\nSaved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
