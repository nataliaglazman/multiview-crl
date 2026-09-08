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

``--bt-corr-ema`` moves that floor, so this runs the EMA too: one correlation state carried
across draws, preceded by ``1/(1-m)`` warm-up draws (its time constant) that are evaluated
but not recorded, since averaging an unconverged EMA re-inflates the very floor it removes.
The reported ``off_diag`` is then the averaged matrix training optimises and ``inst`` is the
single-batch value every pre-EMA number was quoted at.  Note also that the EMA scales
on_diag's and off_diag's GRADIENT by ``(1-m)`` while ``sim``, the variance hinge and
reconstruction are untouched: at m=0.99 the correlation half of Barlow Twins pulls 100x
weaker than its share of the loss magnitude suggests, so read the shares below as
magnitudes, not as influence.

What it suggests
----------------
Coefficients that put each term at a chosen share of the total contrastive loss, plus the
value that balances ``sim`` against reconstruction — because the two compete directly and
``scale_contrastive_loss`` multiplies all of them together.  The terms are measured as RAW
sums, so the BT-term suggestions are re-expressed in the run's own parameterisation when it
sets ``--bt-normalize-terms`` (which divides ``on_diag`` by d and ``off_diag`` by d(d-1));
quoting them raw is off by 40x and 1560x at this project's width.  ``sim`` and the
reconstruction probe are means under both, so they carry over unchanged.  The GAP block
suggests ``bt_gap_sim_coeff``, since that arm carries its own coefficient.

"match reconstruction" equates the two UNWEIGHTED terms; the "scale-aware" line beside it
applies ``--scale-recon-loss``, ``--scale-contrastive-loss`` and the arm weight, and is the
one that answers how big ``sim`` is next to reconstruction in the loss actually optimised.

Also reported: ``sim`` split into its per-view constant OFFSET component and the genuine
cross-view misalignment.  The offset is the component BT's standardised correlation is
structurally blind to, and the whole reason the MSE term exists.  It comes from the loss's
own ``sim_offset`` diagnostic — ``(mu_i - mu_j)^2`` measured on the same uncentered tensor
``sim`` runs on, normalised the same way — rather than from the tempting
``sim - (1 - pos_sim_mean)``: ``pos_sim_mean`` is the correlation of the CENTERED features,
and under ``center_mode="position"`` the two differ by the entire shared-anatomy positional
pattern, which over-subtracts badly enough to report a dominant offset as none.

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


def ema_warmup_steps(corr_ema_decay):
    """Draws to burn before recording, when the correlation EMA is on.

    The EMA's value at step t averages the last ~t batches, so an unwarmed run reports the
    high-variance early state and the sampling floor it exists to remove.  1/(1-m) is the
    EMA's own time constant: 100 steps at m=0.99, which is also the point the docstring in
    ``losses.py`` quotes its measured floor reduction at.  Zero when the EMA is off.
    """
    m = float(corr_ema_decay or 0.0)
    # round, not ceil: 1/(1-0.9) evaluates to 10.000000000000002 in binary floating point,
    # and ceil would turn every round decay into one step more than its time constant.
    return 0 if m <= 0 else max(1, int(round(1.0 / (1.0 - m))))


def measure(hz, batch_size, draws, center_mode, patch_stat, sim_normalize=False, seed=0, corr_ema_decay=0.0, warmup=0):
    """Average the shipped loss's own diagnostics over random batches of the TRAINING size.

    ``corr_ema_decay`` mirrors ``--bt-corr-ema``.  It must be threaded through, and one
    ``corr_ema`` dict must persist across draws, or this measures a quantity training does
    not optimise: the EMA averages the cross-correlation over steps, which cuts off_diag's
    ``d(d-1)/rows`` sampling floor by ``(1-m)/(1+m)`` — 12.19 to 0.061 at d=40, B=128,
    m=0.99.  A per-draw dict would reset every step and reproduce the un-EMA'd value
    exactly, which is what ``off_diag_inst`` already reports.

    ``warmup`` draws run before recording starts, sharing that same dict, because the EMA
    needs its time constant to converge and averaging the warm-up in re-inflates the floor
    it removes.  They cost one BT evaluation each on cached features — no forward pass.
    """
    from training.losses import barlow_twins_loss

    rng = np.random.RandomState(seed)
    n = hz.shape[1]
    b = min(batch_size, n)
    keys = (
        "on_diag_loss",
        "off_diag_loss",
        "off_diag_inst",
        "sim_loss",
        "sim_offset",
        "var_loss",
        "feat_std_mean",
        "pos_sim_mean",
    )
    acc = {k: [] for k in keys}
    d = hz.shape[2]
    m = float(corr_ema_decay or 0.0)
    # One dict for every draw: this is the state that makes the EMA an EMA.
    corr_ema = {} if m > 0 else None
    for _step in range(warmup + draws):
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
            # MUST match the run. barlow_twins_loss defaults sim_normalize=False, so
            # omitting it measures the RAW MSE while a --bt-sim-normalize run optimises the
            # variance-normalised one — and every suggestion below is `target / sim`, so it
            # comes out wrong by a factor of 2*feat_std^2. That is 5674x at patch pooling on
            # this project's runs, which turns a correct 0.35 into a useless 6e-5.
            sim_normalize=sim_normalize,
            sim_coeff=1.0,
            std_coeff=1.0,
            corr_ema=corr_ema,
            corr_ema_decay=m,
        )
        if _step < warmup:
            continue
        diag = getattr(loss, "_contrastive_diag", None) or {}
        for k in keys:
            if k in diag:
                acc[k].append(float(diag[k]))
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    rows = b * (hz.shape[3] if hz.ndim == 4 and patch_stat != "per_position" else 1)
    raw_floor = d * (d - 1) / max(rows, 1)
    # The floor the EMA'd off_diag actually carries. Reported as `floor` because that is the
    # term `off_diag` is differenced against below, and off_diag is now the EMA'd one.
    out["floor"] = raw_floor * ((1.0 - m) / (1.0 + m)) if m > 0 else raw_floor
    out["floor_inst"] = raw_floor
    out["corr_ema_decay"] = m
    out["warmup"] = warmup
    out["d"] = d
    out["rows"] = rows
    return out


def foreground_keep(dataset, grid, thresh, batch_size=8, device=None):
    """Patch positions training keeps, as a (P,) bool tensor — or None if none can be built.

    Mirrors ``main_multimodal.py:335`` and ``eval.gradient_attribution``: pool the brain mask
    to the patch grid and keep a position if ANY sample has at least ``thresh`` brain there.
    Without it every BT term here is computed over background positions that training never
    sees, and background is near-constant across subjects — which inflates off_diag (channels
    correlate through the shared background) and deflates the across-subject feature std.
    Measured against this project's own training scalars, that gap was 23x on gap off_diag
    and 121x on the gap variance hinge, while the reconstruction probe — which runs the real
    forward pass and its brain mask — matched to 1%.

    One deviation, unavoidable and benign: training evaluates ``.any()`` over a 128-sample
    batch, this evaluates it over the whole eval set, so it keeps a superset. On registered
    volumes the foreground set barely moves between batches, so the two agree to a handful of
    boundary positions.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    keep = None
    for batch in DataLoader(dataset, batch_size=batch_size):
        m = batch.get("mask")
        if m is None:
            return None
        m = m[0] if isinstance(m, (list, tuple)) else m
        m = torch.as_tensor(m).float()
        if m.ndim == 4:  # (B, D, H, W) -> (B, 1, D, H, W)
            m = m.unsqueeze(1)
        if device is not None:
            m = m.to(device)
        frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)  # (B, P)
        hit = (frac >= thresh).any(dim=0)
        keep = hit if keep is None else (keep | hit)
    return keep


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
    p.add_argument(
        "--ema-warmup",
        type=int,
        default=-1,
        help="Draws to burn before recording, so the --bt-corr-ema correlation average has "
        "converged. -1 (default) uses the EMA's own time constant 1/(1-m), which is 100 at "
        "m=0.99; 0 disables the warm-up and reports the unconverged EMA. Ignored when the run "
        "has no correlation EMA.",
    )
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
    sim_normalize = bool(getattr(args_, "bt_sim_normalize", False))
    # The terms below are measured as RAW SUMS (the loss's own diagnostics, with
    # normalize_terms off). A run with --bt-normalize-terms optimises on_diag/d and
    # off_diag/(d(d-1)) instead, so a suggestion of the form `target / sim` has to be
    # expressed in the same parameterisation or it lands off by d or d(d-1) -- 40x and
    # 1560x at this project's width. sim and recon are means either way and carry over
    # unchanged. Read here rather than assumed, so the printout matches the run.
    normalize_terms = bool(getattr(args_, "bt_normalize_terms", False))
    scale_c = float(getattr(args_, "scale_contrastive_loss", 1.0) or 1.0)
    scale_r = float(getattr(args_, "scale_recon_loss", 1.0) or 1.0)
    corr_ema_decay = float(getattr(args_, "bt_corr_ema", 0.0) or 0.0)
    warmup = ema_warmup_steps(corr_ema_decay) if cli.ema_warmup < 0 else cli.ema_warmup
    if corr_ema_decay > 0:
        need = ema_warmup_steps(corr_ema_decay)
        logger.info(
            "Correlation EMA m=%.4g: warming up %d draws before recording %d (time constant %d).",
            corr_ema_decay,
            warmup,
            cli.draws,
            need,
        )
        if warmup < need:
            logger.warning(
                "--ema-warmup %d is below the EMA time constant 1/(1-m)=%d. off_diag will be "
                "reported before the average has converged, which OVERSTATES it: the sampling "
                "floor the EMA exists to remove is still largely present. Raise it, or read "
                "off_diag_inst instead and treat it as the pre-EMA number it is.",
                warmup,
                need,
            )

    dataset = build_synthetic_test_set(args_, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    def _views(pooling, npatch):
        ld, _gt, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.encode_batch, cli.num_workers, pooling=pooling
        )
        if cli.level not in ld:
            return None
        c1, c2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
        if c1 is None or c2 is None or c1.shape[1] == 0:
            return None
        return _as_views(c1, c2, npatch)

    def _measure(hz):
        return measure(
            hz, B, cli.draws, center_mode, patch_stat, sim_normalize, corr_ema_decay=corr_ema_decay, warmup=warmup
        )

    results = {}
    if grid:
        # ONE extraction, and both arms derive from it — which is also what training does.
        # The GAP term is `z_rec_tuple.mean(-1)` over the ALREADY foreground-filtered patch
        # tensor (main_multimodal.py), not an independent whole-volume average pool. Pooling
        # the whole volume instead folds every background position into the subject vector,
        # and background is the same in every subject, so it shrinks the across-subject
        # variance the GAP hinge and correlation are computed from.
        hz = _views(tuple(grid), int(np.prod(grid)))
        if hz is not None:
            keep = None
            if bool(getattr(args_, "patch_foreground_mask", False)):
                thr = float(getattr(args_, "patch_foreground_thresh", 0.05))
                keep = foreground_keep(dataset, tuple(grid), thr, cli.encode_batch, hz.device)
                if keep is None:
                    logger.warning(
                        "patch_foreground_mask is set but the dataset yielded no 'mask' key; "
                        "measuring over ALL patch positions, which is NOT what training does."
                    )
                elif not bool(keep.any()):
                    logger.warning("foreground mask kept no positions; falling back to all of them.")
                    keep = None
            if keep is not None:
                n_kept, n_all = int(keep.sum()), int(keep.numel())
                logger.info("Foreground patches: keeping %d/%d positions (thresh %.3g).", n_kept, n_all, thr)
                hz = hz[..., keep.to(hz.device)]
            results["patch"] = _measure(hz)
            results["gap"] = _measure(hz.mean(-1))
    else:
        hz = _views("gap", 1)
        if hz is not None:
            results["gap"] = _measure(hz)

    # Reconstruction scale, for sizing the contrastive terms against what they compete with.
    #
    # Must match `Loss/Reconstruction` in TensorBoard, which is NOT a plain volume MSE. It is
    # masked L1 over BRAIN VOXELS ONLY, on predictions clamped to [-1, 1]:
    #   diff = (x - y).abs() * mask ;  loss = diff.sum() / mask.sum()
    # A whole-volume unclamped MSE reads ~6x higher (0.31 vs 0.05 measured), because the
    # background dominates the average and the decoder has no output activation to bound it.
    # Getting this wrong makes every "match reconstruction" suggestion below wrong too.
    recon = float("nan")
    try:
        from torch.utils.data import DataLoader, Subset

        loader = DataLoader(Subset(dataset, range(min(64, len(dataset)))), batch_size=8)
        errs = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"]
                x = torch.cat(imgs, dim=0).to(device).float()
                out = model(x, n_views=len(imgs))
                rec = out[0]
                if rec is None:
                    continue
                y = rec.float().clamp(-1.0, 1.0)
                m = batch.get("mask")
                if m is not None:
                    m = m[0] if isinstance(m, (list, tuple)) else m
                    m = torch.as_tensor(m).float().to(device)
                    m = torch.cat([m] * len(imgs), dim=0)
                    errs.append(float(((x - y).abs() * m).sum() / m.sum().clamp_min(1.0)))
                else:
                    errs.append(float((x - y).abs().mean()))
        recon = float(np.mean(errs)) if errs else float("nan")
    except Exception as e:  # noqa: BLE001 - diagnostic only, never worth failing the run
        logger.warning("recon probe skipped: %s", e)
    del model

    ema_on = corr_ema_decay > 0
    print("\n" + "=" * 88)
    print(f"  BARLOW TWINS TERM MAGNITUDES  (checkpoint batch size B={B}, unweighted)")
    print("=" * 88)
    hdr = (
        f"  {'pooling':<8}{'d':>4}{'rows':>8}{'on_diag':>11}{'off_diag':>11}{'inst':>11}"
        f"{'floor':>9}{'sim':>10}{'offset':>10}{'var':>8}{'std':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, r in results.items():
        print(
            f"  {k:<8}{r['d']:>4}{r['rows']:>8}{_fmt(r['on_diag_loss']):>11}{_fmt(r['off_diag_loss']):>11}"
            f"{_fmt(r['off_diag_inst']):>11}{_fmt(r['floor'], 2):>9}{_fmt(r['sim_loss'], 3):>10}"
            f"{_fmt(r.get('sim_offset'), 3):>10}{_fmt(r['var_loss'], 3):>8}{_fmt(r['feat_std_mean'], 3):>8}"
        )
    if ema_on:
        print(
            f"\n  correlation EMA m={corr_ema_decay:g} is ON, so on_diag/off_diag are the AVERAGED"
            f" matrix\n  training optimises ({warmup} warm-up draws + {cli.draws} recorded)."
            f" 'inst' is the single-batch\n  value every pre-EMA number was quoted at. The floor"
            f" column is likewise the EMA'd one,\n  d(d-1)/rows x (1-m)/(1+m) ="
            f" {results[next(iter(results))]['floor_inst']:.2f} x {(1 - corr_ema_decay) / (1 + corr_ema_decay):.4f}"
            " for the first row."
        )
        print(
            "  Gradient note: the EMA scales on_diag's and off_diag's gradient by (1-m), while sim,"
            "\n  the variance hinge and reconstruction are untouched — so the correlation half of BT"
            f"\n  pulls ~{1 / (1 - corr_ema_decay):.0f}x weaker than these magnitudes suggest"
            f" (effective lambda = bt_lambda x {1 - corr_ema_decay:g})."
        )
    print(f"\n  reconstruction (masked L1, matches Loss/Reconstruction): {_fmt(recon, 5)}")

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

        s, off = r["sim_loss"], r.get("sim_offset", float("nan"))
        if np.isfinite(s) and s > 0 and np.isfinite(off):
            # sim_offset is the (mu_i - mu_j)^2 component of sim_loss, measured exactly by the
            # loss on the same uncentered tensor sim runs on. It replaces the old
            # `sim vs 1 - pos_sim_mean` comparison, which mixed tensors: pos_sim_mean is the
            # correlation of the CENTERED features while sim runs on the uncentered ones, and
            # under center_mode="position" the difference is the entire shared-anatomy
            # positional pattern. That over-subtracted and reported the patch arm's offset as
            # none when it was essentially all of sim.
            frac = 100.0 * off / s
            print(f"\n  sim {s:.4f} = offset {off:.4f} + misalignment {s - off:.4f}   ({frac:.0f}% offset)")
            if frac > 50:
                print("    => the per-view constant OFFSET dominates. This is exactly what the MSE")
                print("       term exists to remove and exactly what BT's standardised correlation")
                print("       cannot see, so the weight here is doing work no other term can.")
            elif frac > 10:
                print("    => a real offset, but most of sim is genuine cross-view misalignment,")
                print("       which on_diag also penalises (as (1-rho)^2 rather than (1-rho)).")
            else:
                print("    => little residual offset; the views already sit in the same region.")
                print("       Weight here is near-redundant with on_diag, differing mainly in that")
                print("       sim keeps a gradient as rho -> 1 where on_diag's vanishes.")

        # Express the BT-term targets in the parameterisation the run trains in. sim and
        # recon are means under both, so only the two summed terms move.
        dd = r["d"]
        on_t = r["on_diag_loss"] / (dd if normalize_terms else 1.0)
        off_t = excess / (dd * (dd - 1) if normalize_terms else 1.0)
        flag = "bt_gap_sim_coeff" if k == "gap" else "bt_sim_coeff"
        print(f"\n  suggested {flag}   (--bt-normalize-terms={normalize_terms}):")
        for tag, target in (
            ("match off_diag (real part)", max(off_t, 1e-9)),
            ("match on_diag", max(on_t, 1e-9)),
            ("match reconstruction", recon),
        ):
            if np.isfinite(target) and np.isfinite(s) and s > 0:
                print(f"    {tag:<28} {target / s:.3e}")
        # "match reconstruction" above equates the two UNWEIGHTED terms, but they reach the
        # optimiser through different scales, and sim is additionally multiplied by its arm
        # weight. This is the line that answers "how big is sim next to recon in the loss
        # that is actually optimised", which is the question the coefficient is really for.
        if np.isfinite(recon) and np.isfinite(s) and s > 0 and scale_c > 0:
            arm_w = float(
                getattr(args_, "bt_gap_weight", 1.0) if k == "gap" else getattr(args_, "bt_patch_weight", 1.0)
            )
            if arm_w > 0:
                print(
                    f"    {'match recon, scale-aware':<28} {scale_r * recon / (scale_c * arm_w * s):.3e}"
                    f"   (scale_recon {scale_r:g} / scale_contrastive {scale_c:g}, arm weight {arm_w:g})"
                )
        # Arm-aware, because "safe" and "advisable" differ here. The hinge is bounded by 2.0
        # and dormant above feat_std 1, so it cannot blow up at any coefficient -- but on the
        # GAP arm it measures std over SUBJECT rows, where the across-subject component is a
        # fraction of what BT normalises by, so a large coefficient asks for a big per-channel
        # rescale of the encoder output. That inflation survives the content normalisation and
        # steps reconstruction to a higher plateau (see --bt-gap-std-coeff in utils/config.py).
        sd = r["feat_std_mean"]
        std_flag = "bt_gap_std_coeff" if k == "gap" else "bt_std_coeff"
        print(f"  {std_flag}: the hinge is bounded by 2.0 and dormant once feat_std > 1")
        print(f"  (currently {_fmt(sd, 3).strip()}, var_loss {_fmt(r['var_loss'], 3).strip()}).", end=" ")
        if k == "gap":
            print("Keep this LOW (0.1-0.5): on subject rows a")
            print("  large value buys a per-channel rescale that costs reconstruction.")
        else:
            print("1.0 is safe at any scale here.")


if __name__ == "__main__":
    main()
