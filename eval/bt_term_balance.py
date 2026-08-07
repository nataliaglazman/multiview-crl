#!/usr/bin/env python
"""Measure all four Barlow Twins terms on a checkpoint, and size the coefficients from them.

``bt_lambda``, ``bt_sim_coeff`` and ``bt_std_coeff`` all weight terms whose RAW magnitudes
differ by orders of magnitude and depend on the feature scale, the channel width and the
batch size. Guessing them has gone wrong twice on this project already: the inherited
``bt_lambda`` of 0.005 was ~200x weaker at d=44 than the recipe it came from, and a
``bt_sim_coeff`` of 1 put the MSE term far above every other term in the loss.

Reading them off a training curve is awkward because the terms move while the model moves.
This evaluates them at a FIXED checkpoint, at the TRAINING batch size, using the shipped
``barlow_twins_loss`` itself — so the numbers cannot drift from what training computes.

Why batch size matters
----------------------
``off_diag`` estimates a d x d matrix from the rows available, so it carries a sampling
floor of ``d(d-1)/rows`` that is pure noise:

    patch fold   rows = B*P    ~65k    floor ~0.03   negligible
    GAP          rows = B      128     floor ~14.8   dominant

So the same lambda means very different things at the two poolings, which is why they carry
separate coefficients. The floor is printed alongside the measured value; what matters for
``off_diag`` is the EXCESS over it, not the raw number.

What it suggests
----------------
Coefficients that put each term at a chosen share of the total contrastive loss, plus the
value that balances ``sim`` against reconstruction — because the two compete directly and
``scale_contrastive_loss`` multiplies all of them together.

Also reported: ``sim / (2 * feat_std^2)`` against ``1 - corr``. For zero-mean equal-variance
views those are equal, so any excess is the per-view constant OFFSET — the component BT's
standardised correlation is structurally blind to, and the reason the MSE term exists.

Usage
-----
    python -m eval.bt_term_balance --run-dir results/synthetic/RUN --checkpoint-name vqvae_model.pt
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _as_views(content_v1, content_v2, n_patches):
    """(N, P*C) patch-major -> torch (2, N, C, P). n_patches=1 gives (2, N, C)."""
    a, b = np.asarray(content_v1), np.asarray(content_v2)
    if n_patches > 1:
        C = a.shape[1] // n_patches
        a = a.reshape(len(a), n_patches, C).transpose(0, 2, 1)
        b = b.reshape(len(b), n_patches, C).transpose(0, 2, 1)
    return torch.from_numpy(np.stack([a, b]).astype(np.float32))


def measure(hz, batch_size, draws, center_mode, patch_stat, seed=0):
    """Average the shipped loss's own diagnostics over random batches of the TRAINING size."""
    from training.losses import barlow_twins_loss

    rng = np.random.RandomState(seed)
    n = hz.shape[1]
    b = min(batch_size, n)
    keys = ("on_diag_loss", "off_diag_loss", "sim_loss", "var_loss", "feat_std_mean", "pos_sim_mean")
    acc = {k: [] for k in keys}
    d = hz.shape[2]
    for _ in range(draws):
        idx = rng.choice(n, size=b, replace=False)
        sub = hz[:, idx]
        # sim_coeff/std_coeff = 1 so the diagnostics report the terms UNWEIGHTED.
        loss = barlow_twins_loss(
            sub,
            estimated_content_indices=[list(range(d))],
            subsets=[[0, 1]],
            lambd=1.0,
            center_mode=center_mode if sub.ndim == 4 else "none",
            patch_stat=patch_stat,
            sim_coeff=1.0,
            std_coeff=1.0,
        )
        diag = getattr(loss, "_contrastive_diag", None) or {}
        for k in keys:
            if k in diag:
                acc[k].append(float(diag[k]))
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    rows = b * (hz.shape[3] if hz.ndim == 4 and patch_stat != "per_position" else 1)
    out["floor"] = d * (d - 1) / max(rows, 1)
    out["d"] = d
    out["rows"] = rows
    return out


def _fmt(v, nd=4):
    return "   -   " if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser(description="Size bt_lambda / bt_sim_coeff / bt_std_coeff from measurement.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--batch-size", type=int, default=0, help="Training batch size (0 = read from settings).")
    p.add_argument("--draws", type=int, default=16, help="Random batches to average each term over.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    cli = p.parse_args()

    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    args_ = load_run_args(cli.run_dir)
    B = cli.batch_size or int(getattr(args_, "batch_size", 128) or 128)
    grid = getattr(args_, "patch_grid", None)
    center_mode = getattr(args_, "patch_center_mode", "none") or "none"
    patch_stat = getattr(args_, "bt_patch_stat", "fold") or "fold"

    dataset = build_synthetic_test_set(args_, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    results = {}
    poolings = [("gap", "gap")]
    if grid:
        poolings.insert(0, ("patch", tuple(grid)))
    for key, value in poolings:
        ld, _gt, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.encode_batch, cli.num_workers, pooling=value
        )
        if cli.level not in ld:
            continue
        c1, c2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
        if c1 is None or c2 is None or c1.shape[1] == 0:
            continue
        npatch = int(np.prod(value)) if key == "patch" else 1
        hz = _as_views(c1, c2, npatch)
        results[key] = measure(hz, B, cli.draws, center_mode, patch_stat)

    # Reconstruction scale, for sizing the contrastive terms against what they compete with.
    recon_mse = float("nan")
    try:
        from torch.utils.data import DataLoader, Subset

        loader = DataLoader(Subset(dataset, range(min(64, len(dataset)))), batch_size=8)
        errs = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"]
                x = torch.cat(imgs, dim=0).to(device)
                out = model(x, n_views=len(imgs))
                rec = out[0]
                if rec is not None:
                    errs.append(float(((rec.float() - x.float()) ** 2).mean()))
        recon_mse = float(np.mean(errs)) if errs else float("nan")
    except Exception as e:  # noqa: BLE001 - diagnostic only, never worth failing the run
        logger.warning("recon probe skipped: %s", e)
    del model

    print("\n" + "=" * 88)
    print(f"  BARLOW TWINS TERM MAGNITUDES  (checkpoint batch size B={B}, unweighted)")
    print("=" * 88)
    hdr = (
        f"  {'pooling':<8}{'d':>4}{'rows':>8}{'on_diag':>11}{'off_diag':>11}{'floor':>9}{'sim':>12}{'var':>8}{'std':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, r in results.items():
        print(
            f"  {k:<8}{r['d']:>4}{r['rows']:>8}{_fmt(r['on_diag_loss']):>11}{_fmt(r['off_diag_loss']):>11}"
            f"{_fmt(r['floor'], 2):>9}{_fmt(r['sim_loss'], 3):>12}{_fmt(r['var_loss'], 3):>8}"
            f"{_fmt(r['feat_std_mean'], 3):>8}"
        )
    print(f"\n  reconstruction MSE (for scale): {_fmt(recon_mse, 5)}")

    for k, r in results.items():
        print("\n" + "=" * 88)
        print(f"  {k.upper()}")
        print("=" * 88)
        excess = r["off_diag_loss"] - r["floor"]
        print(
            f"  off_diag {r['off_diag_loss']:.2f} = floor {r['floor']:.2f} + real {excess:.2f}"
            f"   ({100 * r['floor'] / max(r['off_diag_loss'], 1e-9):.0f}% noise)"
        )
        if excess < 0.25 * r["floor"]:
            print("    [!] the redundancy term is mostly fitting sampling noise here.")
            print("        Levers: bigger batch (floor halves per doubling) or a narrower content block.")

        s, sd, corr = r["sim_loss"], r["feat_std_mean"], r["pos_sim_mean"]
        if np.isfinite(s) and np.isfinite(sd) and sd > 0:
            norm = s / (2 * sd**2)
            print(f"\n  sim / (2*std^2) = {norm:.3f}   vs   1 - corr = {1 - corr:.3f}")
            if norm > 3 * max(1 - corr, 1e-6):
                print(f"    => the OFFSET dominates the disagreement by ~{norm / max(1 - corr, 1e-6):.0f}x.")
                print("       This is exactly what the MSE term exists to remove, and exactly what")
                print("       BT's standardised correlation cannot see.")
            else:
                print("    => little residual offset; the views already sit in the same region.")

        print("\n  suggested bt_sim_coeff:")
        for tag, target in (
            ("match off_diag (real part)", max(excess, 1e-9)),
            ("match on_diag", max(r["on_diag_loss"], 1e-9)),
            ("match reconstruction MSE", recon_mse),
        ):
            if np.isfinite(target) and np.isfinite(s) and s > 0:
                print(f"    {tag:<28} {target / s:.3e}")
        print("  bt_std_coeff: 1.0 is safe at any scale — the hinge is bounded by 2.0 and goes")
        print("  dormant once feat_std > 1 (currently {:.3f}).".format(sd if np.isfinite(sd) else float("nan")))


if __name__ == "__main__":
    main()
