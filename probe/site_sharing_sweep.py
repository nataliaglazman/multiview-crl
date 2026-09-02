#!/usr/bin/env python
"""When during training did the decoder stop being locally invertible?

``probe.site_sharing`` answers "is this decoder site-shared and locally invertible" for one
checkpoint.  Two checkpoints of the same run gave 7.7 and 3.6 effective rank, and 0.675 and
0.411 captured energy, which raises a question a single checkpoint cannot settle: is the
local decoding map *collapsing over training*, or do runs simply differ?

That distinction matters because a collapse would be a mechanism for the identifiability
decay recorded in ``paper/2026-08-10-mcc-decay-investigation.md`` — MCC falling from its
early peak over the course of training — rather than one more symptom of it.  A decoder
whose local response loses rank while spreading further would produce exactly that curve.

This runs the same measurement across a run's checkpoints and prints the trend.

What is held fixed
------------------
The subjects, their images, and the measured sites are chosen ONCE and reused at every
checkpoint, so a difference in the table is a difference in the decoder and not in what it
was asked about.  Only ``z`` and ``style`` are re-captured per checkpoint, because they are
the encoder's output and must move with it.  Architecture is checked for drift: a run whose
latent grid changed mid-sweep is not one trajectory and the sweep refuses it.

Checkpoints
-----------
VQ-VAE training writes only ``vqvae_model.pt`` (rolling) and ``vqvae_best.pt``, so a
finished run leaves no trajectory to sweep.  Train with ``--checkpoint-keep-every N`` to
keep ``vqvae_model_<step>.pt`` copies; this script discovers them automatically.  Without
them the only comparison available is across runs, which confounds training time with
whatever else differs between them.

Usage:
    python -m probe.site_sharing_sweep --run-dir results/synthetic/<run>
    python -m probe.site_sharing_sweep --run-dir <run> --checkpoints a.pt,b.pt --arms full
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np

from probe.site_sharing import (
    _json_default,
    block_slices,
    file_sha256,
    git_sha,
    make_linear_decoder,
    measure_arm,
    reduce_arm,
    select_sites,
)

logger = logging.getLogger(__name__)


def discover_checkpoints(run_dir: str, spec: str | None) -> list[str]:
    """Explicit list, glob, or every versioned copy in the run directory, in step order."""
    if spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        out = []
        for p in parts:
            hits = sorted(glob.glob(p))
            out.extend(hits if hits else [p])
    else:
        out = sorted(glob.glob(os.path.join(run_dir, "vqvae_model_*.pt")))
        if not out:
            raise SystemExit(
                f"No versioned checkpoints in {run_dir}. VQ-VAE training keeps only "
                "vqvae_model.pt and vqvae_best.pt, so a finished run has no trajectory to "
                "sweep. Retrain with --checkpoint-keep-every N, or pass --checkpoints "
                "explicitly to compare whatever states you do have."
            )
    missing = [p for p in out if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Checkpoint(s) not found: {missing}")
    return out


def checkpoint_step(path) -> int | None:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("step")
    except Exception:
        return None


def build_fixture(cli, args, device, n_subjects):
    """Images, masks and tissue for the subjects — everything that does NOT depend on weights."""
    import torch

    from eval.run_dci_synthetic import build_synthetic_test_set

    ds = build_synthetic_test_set(args, max(n_subjects, 8), cache=True, causal=True)
    inner = getattr(ds, "_inner", None)
    fixture = []
    for i in range(n_subjects):
        item = ds[i]
        tissue = None
        if inner is not None and hasattr(inner, "renderer"):
            lat = item["gt_latents"]
            with torch.no_grad():
                tissue, _ = inner.renderer.render_structure(
                    lat["z_content"],
                    lat["z_deformation"],
                    lat["z_fissure"],
                    device=torch.device("cpu"),
                    clean=getattr(inner, "clean_content", False),
                )
        fixture.append(
            {
                "x": torch.stack([item["image"][0], item["image"][1]], 0).to(device),
                "mask": item["mask"][0][0],
                "tissue": tissue,
            }
        )
    return fixture


def capture_states(model, fixture, level):
    """Re-read z and style at this checkpoint; the images behind them are unchanged."""
    from probe.jacobian_spread import capture_decoder_inputs

    out = []
    for f in fixture:
        z, style = capture_decoder_inputs(model, f["x"], level)
        out.append(
            {
                "z": z[:1].contiguous(),
                "style": None if style is None else style[:1].contiguous(),
                "tissue": f["tissue"],
                "mask": f["mask"],
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", default=None, help="Comma-separated paths or globs. Default: vqvae_model_*.pt.")
    ap.add_argument(
        "--arms",
        default="full,linear",
        help="Arms to run. frozen_norm is dropped by default: it has "
        "tracked 'full' closely on every run so far and doubles the cost.",
    )
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--sites", choices=("foreground", "strata", "all"), default="foreground")
    ap.add_argument(
        "--max-sites", type=int, default=16, help="Lower than the single-run default: cost is per checkpoint."
    )
    ap.add_argument("--fg-thresh", type=float, default=0.2)
    ap.add_argument("--block-dilation", type=int, default=2)
    ap.add_argument("--profile-dilations", default="0,1,2,4")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--dead-thresh", type=float, default=1e-3)
    ap.add_argument("--margin-tol", type=float, default=0.05)
    ap.add_argument("--spectrum-energy", type=float, default=0.99)
    ap.add_argument("--n-refs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch

    from eval.run_dci_synthetic import load_model_from_run_dir
    from probe.jacobian_spread import build_arm, make_fn

    torch.set_num_threads(cli.threads)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    ckpts = discover_checkpoints(cli.run_dir, cli.checkpoints)
    arms = [a.strip() for a in cli.arms.split(",") if a.strip()]
    profile_dilations = sorted({int(v) for v in cli.profile_dilations.split(",") if v.strip()})
    print(f"\n  sweeping {len(ckpts)} checkpoint(s) x {len(arms)} arm(s): {', '.join(arms)}")

    fixture = sites = strata = None
    latent_shape = out_shape = n_channels = None
    rows = []

    for ci, ckpt in enumerate(ckpts):
        model, args, _ = load_model_from_run_dir(cli.run_dir, ckpt, device)
        model.eval()
        decoder = model.decoders[cli.level]

        if not str(getattr(args, "dataset_name", "")).lower().startswith("synthetic"):
            raise SystemExit(f"Only the synthetic loader is wired up; this run trained on {args.dataset_name!r}.")

        if fixture is None:
            fixture = build_fixture(cli, args, device, cli.n_subjects)

        subjects = capture_states(model, fixture, cli.level)
        ls = tuple(subjects[0]["z"].shape[2:])
        nc = subjects[0]["z"].shape[1]
        with torch.no_grad():
            os_ = tuple(make_fn(decoder, subjects[0]["style"])(subjects[0]["z"]).shape[2:])

        if latent_shape is None:
            import torch.nn.functional as F

            latent_shape, out_shape, n_channels = ls, os_, nc
            brain = F.adaptive_avg_pool3d((subjects[0]["mask"] > 0).float()[None, None], latent_shape)[0, 0]
            sites, strata, _ = select_sites(
                brain,
                latent_shape,
                cli.sites,
                cli.max_sites,
                cli.fg_thresh,
                torch.Generator().manual_seed(cli.seed),
                subjects[0]["tissue"],
                out_shape,
                cli.block_dilation,
            )
            win = tuple(s.stop - s.start for s in block_slices(sites[0], latent_shape, out_shape, cli.block_dilation))
            print(f"  latent {latent_shape} x {nc}ch -> {os_};  window {win};  {len(sites)} site(s) held fixed\n")
        elif (ls, os_, nc) != (latent_shape, out_shape, n_channels):
            # Comparing across a shape change would put an architecture difference in a column
            # labelled by training step.
            raise SystemExit(
                f"{ckpt} has latent {ls} x {nc}ch -> {os_}, but the sweep started from "
                f"{latent_shape} x {n_channels}ch -> {out_shape}. These are not one trajectory."
            )

        step = checkpoint_step(ckpt)
        row = {"checkpoint": os.path.abspath(ckpt), "step": step, "sha256": file_sha256(ckpt), "arms": {}}
        for arm in arms:
            per_subject = []
            for sub in subjects:
                dec = (
                    make_linear_decoder(decoder)
                    if arm == "linear"
                    else build_arm(decoder, None, "frozen" if arm == "frozen_norm" else "live", sub["z"], sub["style"])
                )
                per_subject.append(
                    measure_arm(
                        make_fn(dec, sub["style"]),
                        sub["z"],
                        sites,
                        latent_shape,
                        out_shape,
                        n_channels,
                        cli.chunk,
                        cli.block_dilation,
                        profile_dilations,
                    )
                )
            row["arms"][arm] = reduce_arm(
                per_subject, sites, cli.dead_thresh, cli.n_refs, cli.margin_tol, cli.spectrum_energy
            )
        rows.append(row)
        _print_row(row, cli, arms, ci + 1, len(ckpts))

    print_trend(rows, cli, arms)

    os.makedirs(cli.out_dir, exist_ok=True)
    stem = os.path.join(cli.out_dir, f"{cli.prefix}site_sharing_sweep")
    with open(stem + ".json", "w") as f:
        json.dump(
            {
                "run_dir": os.path.abspath(cli.run_dir),
                "repo_git_sha": git_sha(os.path.dirname(os.path.abspath(__file__)) + "/.."),
                "latent_grid": list(latent_shape),
                "output_shape": list(out_shape),
                "latent_channels": n_channels,
                "sites": [list(s) for s in sites],
                "site_strata": {str(list(k)): v for k, v in strata.items()},
                "cli": vars(cli),
                "checkpoints": rows,
            },
            f,
            indent=2,
            default=_json_default,
        )
    print(f"\n  wrote {stem}.json")
    if cli.plot:
        plot_trend(stem + ".png", rows, arms, cli)
        print(f"  wrote {stem}.png")


def _print_row(row, cli, arms, i, n):
    d = str(cli.block_dilation)
    a = row["arms"][arms[0]]
    print(
        f"  [{i}/{n}] step {row['step']}:  rank {a['effective_rank']:.1f} "
        f"(rv {a['effective_rank_rv']:.1f})  energy@+{d} {a['energy_profile'][d]:.3f}  "
        f"homog {a['homogeneity']['ratio']:.2f}x  bind {a['binding']['primary']['identity_frac_median']:.3f}"
    )


def print_trend(rows, cli, arms):
    """The table the sweep exists for: every column against training step."""
    d = str(cli.block_dilation)
    print("\n" + "=" * 100)
    print("TREND ACROSS CHECKPOINTS")
    print("=" * 100)
    hdr = f"  {'step':>8} {'rank':>7} {'rank_rv':>8} {'E@+0':>7} {f'E@+{d}':>7} {'homog':>7} {'bind':>7} {'m|cos|':>7}"
    if "linear" in arms:
        hdr += f" {'lin_rank':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        a = r["arms"][arms[0]]
        line = (
            f"  {str(r['step']):>8} {a['effective_rank']:>7.1f} {a['effective_rank_rv']:>8.1f} "
            f"{a['energy_profile'].get('0', float('nan')):>7.3f} {a['energy_profile'][d]:>7.3f} "
            f"{a['homogeneity']['ratio']:>7.2f} {a['binding']['primary']['identity_frac_median']:>7.3f} "
            f"{a['binding']['primary']['matched_cos_median']:>7.3f}"
        )
        if "linear" in r["arms"]:
            line += f" {r['arms']['linear']['effective_rank']:>9.1f}"
        print(line)

    steps = [r["step"] for r in rows]
    if len(rows) >= 2 and all(s is not None for s in steps):
        first, last = rows[0]["arms"][arms[0]], rows[-1]["arms"][arms[0]]
        dr = last["effective_rank"] - first["effective_rank"]
        de = last["energy_profile"][d] - first["energy_profile"][d]
        print(f"\n  step {steps[0]} -> {steps[-1]}:  rank {dr:+.1f},  energy@+{d} {de:+.3f}")
        # A rank that falls while the response spreads is the collapse the sweep was built to
        # look for; the opposite pattern rules it out and sends the question back to seeds.
        if dr < -0.5 and de < 0:
            print(
                "  The local map loses rank AND delocalises as training proceeds. That is a\n"
                "  candidate mechanism for the identifiability decay in the mcc-decay note —\n"
                "  worth checking against the MCC curve for the same run, at the same steps."
            )
        elif abs(dr) <= 0.5:
            print("  Rank is flat across training, so the between-checkpoint difference was not training time.")
        else:
            print("  Rank does not fall over training here; the earlier difference is not a collapse.")
    elif any(s is None for s in steps):
        print("\n  Some checkpoints record no step, so the ordering above is by filename, not training time.")


def plot_trend(path, rows, arms, cli):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = str(cli.block_dilation)
    steps = [r["step"] if r["step"] is not None else i for i, r in enumerate(rows)]
    a0 = arms[0]
    series = [
        ("effective rank", [r["arms"][a0]["effective_rank"] for r in rows]),
        (f"captured energy @+{d}", [r["arms"][a0]["energy_profile"][d] for r in rows]),
        ("homogeneity ratio", [r["arms"][a0]["homogeneity"]["ratio"] for r in rows]),
        ("binding identity", [r["arms"][a0]["binding"]["primary"]["identity_frac_median"] for r in rows]),
    ]
    fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 3.2))
    for ax, (title, ys) in zip(np.atleast_1d(axes), series):
        ax.plot(steps, ys, "o-")
        if title == "effective rank" and "linear" in arms:
            ax.plot(steps, [r["arms"]["linear"]["effective_rank"] for r in rows], "s--", label="linear", alpha=0.6)
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
