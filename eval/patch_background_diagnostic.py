"""Is background actually harming the patch-contrastive objective on this run?

Stratifies every patch position by how much brain it covers, then asks three
falsifiable questions on a frozen checkpoint:

1. **Loss share** — do background/near-empty positions carry a disproportionate
   share of the InfoNCE loss?  A position whose features are identical across
   the batch yields a uniform softmax, i.e. CE = log(R), the *maximum*.  If
   background is the problem, its loss share far exceeds its position share.

2. **Gradient share** — loss magnitude is not learning signal.  This takes the
   real per-position gradient w.r.t. the pre-normalisation features, which is
   what actually updates the encoder.  This is the decisive number.

3. **Coverage leak** — under average pooling a partially-covered patch is
   ``frac * mean_fg + (1 - frac) * v_bg``, so its descriptor moves with ``frac``,
   which is local brain extent, i.e. head size.  Correlating each position's
   leading batch-direction with its per-sample coverage tests that mechanism
   directly.  High |r| on mixed positions means masked pooling is needed;
   dropping positions cannot fix it.

Strata (by coverage across the batch, mirroring ``--patch-foreground-thresh``):
  dead        never >= thresh in any sample      — dropped by the current rule
  near-empty  passes ``any`` but mean <= thresh  — ADMITTED by the current rule
  mixed       mean coverage in (thresh, 0.95)
  pure        mean coverage >= 0.95

Usage:
    python -m eval.patch_background_diagnostic /path/to/run_dir
    python -m eval.patch_background_diagnostic /path/to/run_dir --level 0 --grid 8 8 8
"""

import argparse
import logging
import os

import torch
import torch.nn.functional as F

from eval.tau_diagnostic import build_score_rows
from training.losses import _center_patch_features

logger = logging.getLogger(__name__)

STRATA = ("dead", "near-empty", "mixed", "pure")


def extract_patches_with_coverage(model, dataset, args, device, batch_size, grid, level):
    """(hz, frac) for every patch position — no foreground drop, coverage retained.

    hz is (n_views, B, C', P) pre-normalisation content features; frac is
    (n_views*B, P), the brain-mask fraction of each position per sample.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    imgs = batch["image"]
    n_views = len(imgs)
    x = torch.cat(imgs, dim=0).to(device)

    masks = batch.get("mask")
    if masks is None:
        raise SystemExit("dataset yields no 'mask' key; coverage stratification needs brain masks")
    m = torch.cat(masks, 0).to(device).float()

    model.eval()
    with torch.no_grad():
        enc_out = model(x, pool_only=True, n_views=n_views, patch_grid=grid)
        feats = enc_out[2] if isinstance(enc_out, tuple) else [enc_out]
        soft_masks = enc_out[6] if isinstance(enc_out, tuple) and len(enc_out) > 6 else {}

        feat = feats[level]
        if feat.dim() == 2:
            raise SystemExit(f"level {level} is GAP-pooled (no patch axis); this diagnostic needs patch pooling")
        hz = feat.reshape(n_views, -1, *feat.shape[1:])  # (n_views, B, C, P)

        if isinstance(soft_masks, dict) and level in soft_masks:
            mask = soft_masks[level]
            mask = mask[0] if isinstance(mask, tuple) else mask
            content_idx = torch.where(mask.bool())[-1]
        else:
            content_idx = torch.as_tensor(args.content_indices[0], device=hz.device)
        hz = hz[:, :, content_idx, :]

        frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)  # (n_views*B, P)

    return hz, frac


def stratify(frac, thresh):
    """Boolean (P,) masks per stratum, mirroring the training keep-rule."""
    keep_any = (frac >= thresh).any(dim=0)
    cov = frac.mean(dim=0)
    dead = ~keep_any
    near = keep_any & (cov <= thresh)
    pure = keep_any & (cov >= 0.95)
    mixed = keep_any & (cov > thresh) & (cov < 0.95)
    return {"dead": dead, "near-empty": near, "mixed": mixed, "pure": pure}


def per_position_stats(hz, tau, cross_view_negs_only, center_mode="none"):
    """Per-position loss, top-1 and gradient norm. Returns dict of (P,) tensors.

    ``center_mode`` must match the run's ``--patch-center-mode``: training
    centres *before* L2 normalisation (``losses.py``: _patch_infonce_base), and
    on registered volumes the uncentred cosine is dominated by the population
    anatomy shared at that position, which compresses every similarity toward 1.
    Scoring uncentred features from a centred run understates both the gap and
    its spread.  The gradient is still taken w.r.t. the pre-centring features,
    which is what reaches the encoder.
    """
    hz = hz.detach().clone().requires_grad_(True)
    hz_n = F.normalize(_center_patch_features(hz, center_mode), dim=2, eps=1e-6)
    logits, targets, _, _ = build_score_rows(hz_n, tau, cross_view_negs_only)

    P, R, _ = logits.shape
    flat = logits.reshape(P * R, R)
    tgt = targets.repeat(P)

    ce = F.cross_entropy(flat, tgt, reduction="none").view(P, R)
    loss_p = ce.mean(dim=1)
    top1_p = (flat.argmax(dim=1) == tgt).float().view(P, R).mean(dim=1)

    with torch.no_grad():
        probs = flat.softmax(dim=1)
        p_pos = probs.gather(1, tgt[:, None]).squeeze(1).view(P, R).mean(dim=1)

    loss_p.sum().backward()
    grad_p = hz.grad.norm(dim=2).sum(dim=(0, 1))  # (P,) summed over views+batch

    return {
        "loss": loss_p.detach(),
        "top1": top1_p.detach(),
        "p_pos": p_pos,
        "grad": grad_p.detach(),
        "chance": 1.0 / R,
        "max_ce": torch.log(torch.tensor(float(R))).item(),
    }


def dispersion(hz, sel):
    """Across-batch spread of the descriptors at these positions, relative to their magnitude.

    A position whose features are *identical* across the batch yields a uniform
    softmax whose gradient cancels exactly, so it is inert despite carrying the
    maximum loss.  Gradient share climbs steeply once dispersion departs from 0
    (~0.01 is already ~13% of the gradient in a controlled test), and neither CE
    nor top-1 moves while that happens.  This number places the run on that axis.
    """
    if int(sel.sum()) == 0:
        return float("nan")
    d = hz[..., sel].reshape(-1, hz.shape[2], int(sel.sum()))
    mu = d.mean(0)
    sd = d.std(0)
    return float((sd.norm(dim=0) / mu.norm(dim=0).clamp_min(1e-12)).mean())


def coverage_leak(hz, frac, sel):
    """|corr| between each position's leading batch-direction and its coverage.

    Under mixture pooling the descriptor moves along one direction as coverage
    varies, so a single principal direction is the right 1-df test — no
    regression, nothing to overfit.
    """
    if int(sel.sum()) == 0:
        return float("nan"), float("nan")
    n_views, B = hz.shape[0], hz.shape[1]
    h = hz[..., sel].reshape(n_views * B, hz.shape[2], -1)  # (N, C, p)
    f = frac[:, sel]  # (N, p)
    rs = []
    for j in range(h.shape[-1]):
        d = h[:, :, j]
        d = d - d.mean(0, keepdim=True)
        cov = f[:, j] - f[:, j].mean()
        if float(cov.std()) < 1e-6:
            continue
        # leading batch-direction of the descriptor cloud
        u = torch.pca_lowrank(d, q=1)[2][:, 0]
        s = d @ u
        if float(s.std()) < 1e-12:
            continue
        r = float((s * cov).mean() / (s.std(unbiased=False) * cov.std(unbiased=False)))
        rs.append(abs(r))
    if not rs:
        return float("nan"), float(f.std(dim=0).mean())
    return sum(rs) / len(rs), float(f.std(dim=0).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--grid", type=int, nargs=3, default=None, help="Override the run's patch grid (e.g. --grid 8 8 8).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument(
        "--center-mode",
        default=None,
        choices=["none", "position", "double"],
        help="Override the run's --patch-center-mode. Must match training to score the geometry "
        "the objective actually optimised.",
    )
    p.add_argument("--device", default=None)
    cli = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = cli.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(cli.run_dir, ckpt)
    model, args, device = load_model_from_run_dir(cli.run_dir, ckpt, device)

    grid = tuple(cli.grid) if cli.grid else tuple(args.patch_grid)
    batch_size = cli.batch_size or getattr(args, "batch_size", 32)
    tau = cli.tau if cli.tau is not None else float(getattr(args, "tau", 0.1))
    thresh = getattr(args, "patch_foreground_thresh", 0.05)
    cross_only = bool(getattr(args, "cross_view_negs_only", False))

    dataset = build_synthetic_test_set(args, cli.num_samples or batch_size)
    hz, frac = extract_patches_with_coverage(model, dataset, args, device, batch_size, grid, cli.level)
    n_views, B, C, P = hz.shape

    center_mode = cli.center_mode or getattr(args, "patch_center_mode", "none")
    st = per_position_stats(hz, tau, cross_only, center_mode=center_mode)
    strata = stratify(frac, thresh)

    print(f"\nrun: {cli.run_dir}")
    print(f"level {cli.level} | grid {list(grid)} | P={P} | B={B} | content channels={C} | tau={tau:g}")
    print(f"fg-mask active in training: {bool(getattr(args, 'patch_foreground_mask', False))} | thresh={thresh}")
    print(f"center-mode (matched to run): {center_mode}")
    print(f"chance top-1 = {st['chance']:.3f} | max CE = log(R) = {st['max_ce']:.3f}\n")

    tot_loss = float(st["loss"].sum())
    tot_grad = float(st["grad"].sum())

    hdr = f"{'stratum':<12}{'n':>6}{'pos%':>8}{'loss%':>8}{'GRAD%':>8}{'top1':>8}{'p_pos':>8}{'CE':>8}{'disp':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in STRATA:
        sel = strata[name]
        n = int(sel.sum())
        if n == 0:
            print(f"{name:<12}{0:>6}" + f"{'-':>8}" * 7)
            continue
        print(
            f"{name:<12}{n:>6}{n / P:>8.1%}"
            f"{float(st['loss'][sel].sum()) / max(tot_loss, 1e-12):>8.1%}"
            f"{float(st['grad'][sel].sum()) / max(tot_grad, 1e-12):>8.1%}"
            f"{float(st['top1'][sel].mean()):>8.3f}{float(st['p_pos'][sel].mean()):>8.3f}"
            f"{float(st['loss'][sel].mean()):>8.3f}{dispersion(hz, sel):>8.4f}"
        )
    print("\nnote: CE and top1 are near-blind to this failure — in a controlled sweep they stay pinned")
    print("      at max-CE/chance while background gradient share moves 0% -> 80%. Read GRAD%.")

    print("\ncoverage leak (|corr| between leading batch-direction and per-sample coverage):")
    for name in ("near-empty", "mixed", "pure"):
        r, sd = coverage_leak(hz, frac, strata[name])
        note = "  (coverage ~constant; r not meaningful)" if sd == sd and sd < 0.02 else ""
        print(f"  {name:<12} |r| = {r:.3f}   (mean per-position coverage sd = {sd:.3f}){note}")

    # ---- verdict ----
    junk = strata["near-empty"]
    junk_pos = int(junk.sum()) / P
    junk_grad = float(st["grad"][junk].sum()) / max(tot_grad, 1e-12) if int(junk.sum()) else 0.0
    mixed_r, mixed_sd = coverage_leak(hz, frac, strata["mixed"])

    print("\nverdict:")
    if int(junk.sum()) and junk_grad > 1.5 * junk_pos:
        print(
            f"  near-empty positions are {junk_pos:.1%} of positions but {junk_grad:.1%} of the gradient "
            f"({junk_grad/max(junk_pos,1e-12):.1f}x over-weighted) -> the `any` keep-rule is admitting junk; "
            "switch main_multimodal.py:326 to .all(dim=0)."
        )
    elif int(junk.sum()):
        print(
            f"  near-empty positions carry {junk_grad:.1%} of the gradient for {junk_pos:.1%} of positions "
            "-> not over-weighted; tightening the keep-rule will not buy much."
        )
    if mixed_r == mixed_r and mixed_r > 0.5:
        print(
            f"  mixed positions track their own coverage (|r| = {mixed_r:.2f}) -> descriptors encode local brain "
            "extent (head size). Position-dropping cannot fix this; masked pooling can."
        )
    elif mixed_r == mixed_r:
        print(f"  mixed positions do not track coverage (|r| = {mixed_r:.2f}) -> no head-size leak via mixture.")
    if float(st["top1"][strata["pure"]].mean() if int(strata["pure"].sum()) else 0) < 2 * st["chance"]:
        print("  pure-foreground positions are themselves near chance -> the bottleneck is NOT background.")


if __name__ == "__main__":
    main()
