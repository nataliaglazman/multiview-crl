"""Pick `--bt-lambda` BEFORE training a Barlow Twins run.

Barlow Twins is `on_diag + lambda * off_diag`, where `on_diag` sums **d** terms and
`off_diag` sums **d^2 - d**. The balance between alignment and entropy regularisation
therefore scales with the embedding width, and lambda has to be rescaled roughly as 1/d
to hold it fixed. Zbontar et al. use d=8192 with lambda=0.005, which puts
`lambda * (off terms) / (on terms)` at ~41. The repo default of 0.005 at d=44 puts it at
0.21 — about 200x weaker regularisation than the recipe it was copied from, and the
entropy term is exactly what prevents dimensional collapse.

Yao et al. (ICLR 2024) use lambda=1 for their image experiments, which at small encoding
widths reproduces Zbontar's balance. They also describe their variant as "BarlowTwins with
cosine similarity" while this repo implements the standard per-dimension batch-norm
formulation, so the transfer is not automatic — a different normalisation changes the
typical magnitude of the cross-correlation entries and hence the effective lambda.

This measures the balance on REAL features instead of assuming it. Crucially it does NOT
need a Barlow-Twins-trained model: the loss is a function of embeddings, so any checkpoint
(or an untrained model) gives a usable reading of the balance a BT run would START from —
which is what a fixed hyper-parameter can be set against.

Three normalisations are reported because "cosine similarity" is ambiguous in the paper:
  batchnorm   per-dimension centre+scale across the batch (this repo; Zbontar)
  l2_sample   per-sample L2 normalisation, then the same cross-correlation
  l2_dim      per-dimension L2 normalisation WITHOUT centring (cosine between the two
              views' batch-vectors for each dimension pair)

The `on_diag` arm is computed by calling the shipped `barlow_twins_loss` itself, so the
batchnorm numbers cannot drift from what training actually optimises.

Usage:
  python -m eval.bt_lambda_calibration --run-dir results/synthetic/<any-run>
  python -m eval.bt_lambda_calibration                      # untrained, config defaults
"""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger(__name__)

# Zbontar et al. (2021): d=8192, lambda=0.0051 -> this is the balance BT was tuned at.
ZBONTAR_D, ZBONTAR_LAMBDA = 8192, 0.0051


def count_balance(d, lambd):
    """lambda * (number of off-diagonal terms) / (number of on-diagonal terms)."""
    return lambd * (d * d - d) / max(d, 1)


def cross_corr(z_i, z_j, mode):
    """Cross-correlation matrix under one of the three normalisations."""
    if mode == "batchnorm":
        z_i = (z_i - z_i.mean(0)) / (z_i.std(0, unbiased=False) + 1e-6)
        z_j = (z_j - z_j.mean(0)) / (z_j.std(0, unbiased=False) + 1e-6)
        return (z_i.T @ z_j) / z_i.shape[0]
    if mode == "l2_sample":
        z_i = z_i / (z_i.norm(dim=1, keepdim=True) + 1e-8)
        z_j = z_j / (z_j.norm(dim=1, keepdim=True) + 1e-8)
        return (z_i.T @ z_j) / z_i.shape[0]
    if mode == "l2_dim":
        z_i = z_i / (z_i.norm(dim=0, keepdim=True) + 1e-8)
        z_j = z_j / (z_j.norm(dim=0, keepdim=True) + 1e-8)
        return z_i.T @ z_j
    raise ValueError(mode)


def terms(c):
    on = (c.diagonal() - 1).pow(2).sum().item()
    off = (c.pow(2).sum() - c.diagonal().pow(2).sum()).item()
    return on, off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="Any checkpoint; it need not be BT-trained.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--res", type=int, default=64, help="Only used without --run-dir.")
    ap.add_argument("--patch-grid", type=int, default=8)
    ap.add_argument("--content-size", type=int, default=44, help="Only used without --run-dir.")
    ap.add_argument("--target-balance", type=float, default=None, help="Default: Zbontar's balance.")
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from eval.norm_audit import _default_args, _get_builder, _stub_optional_deps

    _stub_optional_deps()
    from training.losses import barlow_twins_loss

    target = cli.target_balance or count_balance(ZBONTAR_D, ZBONTAR_LAMBDA)
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- features ---------------------------------------------------------------
    if cli.run_dir:
        from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

        model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
        ds = build_synthetic_test_set(args, cli.num_samples, causal=True)
        content_size = int(getattr(args, "content_size", cli.content_size))
        src = f"{cli.run_dir} (loss type: {getattr(args, 'contrastive_loss_type', '?')})"
    else:
        import types as _types

        from eval.run_dci_synthetic import build_synthetic_test_set

        a = _default_args(norm_type="layer", decoder_norm_type="group")
        a.content_indices = [list(range(cli.content_size))]
        model = _get_builder()(a).to(device)
        # Must be a REAL paired dataset: feeding the same volume as both views makes the
        # cross-correlation diagonal exactly 1, so on_diag collapses to 0 and every
        # lambda estimate degenerates. The two views must differ by style, as in training.
        ds = build_synthetic_test_set(
            _types.SimpleNamespace(
                synthetic_res=cli.res,
                spatial_size=None,
                synthetic_mode="pseudo_mri",
                synthetic_seed=42,
                synthetic_n_content=9,
                synthetic_n_style=3,
                synthetic_normalize="fixed_reference",
                synthetic_causal=False,
            ),
            cli.num_samples,
            causal=False,
        )
        content_size = cli.content_size
        src = "UNTRAINED model at config defaults"
    model.eval()

    g = cli.patch_grid
    grid = (g, g, g)
    feats = []
    if True:
        from torch.utils.data import DataLoader

        for batch in DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0):
            x = torch.cat(batch["image"], 0).to(device)
            with torch.no_grad():
                out = model(x, pool_only=True, n_views=2, patch_grid=grid)
                f = (out[2] if isinstance(out, tuple) else out)[0]
            feats.append(f.reshape(2, -1, *f.shape[1:]).cpu())
    hz = torch.cat(feats, dim=1).float()  # (2, N, C, P)
    C = hz.shape[2]
    d = min(content_size, C)
    print(f"\nfeatures from: {src}")
    print(f"patch grid {list(grid)} | channels {C} (content d={d}) | samples {hz.shape[1]}")

    # ---- the shipped loss, so the batchnorm arm cannot drift from training --------
    bt = barlow_twins_loss(hz[:, :, :d, :], lambd=1.0)
    diag = getattr(bt, "_contrastive_diag", {}) or {}
    on_real, off_real = diag.get("on_diag_loss"), diag.get("off_diag_loss")

    # ---- the three normalisations, patch-folded exactly as BT sees them -----------
    zf = hz.permute(0, 1, 3, 2).reshape(2, -1, C)[:, :, :d]
    print(f"folded batch for the correlation: {zf.shape[1]:,} rows x {d} dims\n")

    hdr = f"{'normalisation':<14}{'on_diag':>12}{'off_diag':>12}{'off/on':>10}{'lambda for target':>20}"
    print(hdr + "\n" + "-" * len(hdr))
    lam = {}
    for mode in ("batchnorm", "l2_sample", "l2_dim"):
        on, off = terms(cross_corr(zf[0], zf[1], mode))
        lam[mode] = target * on / max(off, 1e-12)
        print(f"{mode:<14}{on:>12.3f}{off:>12.3f}{off / max(on, 1e-12):>10.2f}{lam[mode]:>20.4f}")

    if on_real is not None:
        print(f"\nshipped barlow_twins_loss (batchnorm): on_diag={on_real:.3f}  off_diag={off_real:.3f}")
        print(f"  -> cross-check against the batchnorm row above; they should match closely.")

    print(f"\ntarget balance = {target:.1f}  (Zbontar d={ZBONTAR_D}, lambda={ZBONTAR_LAMBDA})")
    print("\ncount-only scaling, ignoring the magnitude of the correlations:")
    for dd in (d, 128, 512, ZBONTAR_D):
        print(f"  d={dd:<6} lambda for target = {target / max(dd - 1, 1):.5f}")
    print(
        f"\nrepo default --bt-lambda 0.005 at d={d} gives balance {count_balance(d, 0.005):.2f} "
        f"({count_balance(d, 0.005) / target:.3f}x the target)"
    )
    print(
        f"paper's lambda=1 at d={d} gives balance {count_balance(d, 1.0):.1f} "
        f"({count_balance(d, 1.0) / target:.2f}x the target)"
    )

    print("\nread: if the three 'lambda for target' values agree, the normalisation choice does")
    print("not matter and the count scaling is the whole story. If l2_* differ a lot from")
    print("batchnorm, the paper's lambda=1 does NOT transfer to this repo's formulation and")
    print("the batchnorm column is the one to use — that is what training optimises.")
    print("\nCAVEAT: these are the features a BT run would START from, not converge to. lambda is")
    print("a fixed hyper-parameter, so the initial balance is what you can actually set against.")


if __name__ == "__main__":
    main()
