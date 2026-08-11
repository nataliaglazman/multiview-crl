#!/usr/bin/env python
"""Is the patch-MCC decay quantization coarsening?

The hypothesis, after crowding and spectral depth both died under test.  `CodeLayer.quantize`
adds `diff = (quantize.detach() - x).pow(2).mean()` (models/vqvae.py:503), a commitment term
that pulls the CONTINUOUS encoder output toward its assigned centroid.  Over 87k steps that is
a steady clustering pressure, and it acts on exactly the tensor every probe in this project
reads.  If the encoder outputs collapse onto a finite set of centroids, then distinctions
finer than the inter-centroid spacing stop being recoverable — weakest factors first.

It is the only surviving account that fits every measurement on the reference run
(`...mse-real-lambda-01-normalized`, 2001 -> 89001):

    observation                                  quantization coarsening predicts
    ---------------------------------------------------------------------------
    weak factors lose most (rho +0.72 on          yes — a factor varying by less than the
      initial strength; the ONLY predictor)         code spacing becomes unreadable
    n@95% 32 -> 11 with cond@95% FLAT             yes — clustering onto centroids lowers
                                                    rank without ill-conditioning
    forward map IMPROVES (labeled share           yes — a regular, clustered code is easier
      0.468 -> 0.608)                               to predict from the factors
    info_all only -2.9%                           yes — resolution lost, not information
    contrastive terms at a fixed point            yes — VQ is entirely on the recon side
    recon<->vq cosine +0.346 -> -0.532            yes — they move into conflict over exactly
                                                    this window
    sign flip at 2k (recon helps, then hurts)     plausible — codebook spreads to cover the
                                                    data early, only clusters once converged

The three tests
---------------
  1. UTILISATION.  Codebook perplexity, active entries, and how concentrated the code
     assignment is.  Falling utilisation means fewer effective centroids, i.e. coarser
     resolution at fixed cardinality.

  2. RESOLUTION.  RMS distance from encoder output to its assigned centroid, RELATIVE to the
     feature scale.  This is the direct measure of clustering and the one number the
     hypothesis lives or dies on.  Absolute quantization error is uninterpretable here because
     `feat_std` runs 0.29 -> 66.9 over the same window — only the ratio is comparable.

  3. READOUT.  block-MCC on pre-VQ (`z_e`) vs post-VQ (`z_q`) features, patch-pooled.  If the
     two CONVERGE, the encoder outputs have collapsed onto the centroids: that is the
     hypothesis stated as a consequence rather than as a cause.

Reading the result
------------------
    relative quantization error FALLS  +  MCC(z_e) and MCC(z_q) converge
        CONFIRMED. Fix in this order: the VQ double-count (BaselineLoss returns
        quantization_losses at losses.py:1611 and main_multimodal adds it again, so the
        effective weight is doubled AND scaled by scale_recon_loss), then bound feat_std
        (the variance hinge is one-sided and went dormant), then sweep vq_commitment_weight
        down, then try a continuous bottleneck.

    relative error FLAT
        Clustering is not happening and the hypothesis is dead. Do not fall back on
        utilisation alone — a codebook can lose entries without the encoder outputs moving.

    perplexity falls but relative error is flat
        The codebook coarsened while the encoder ignored it. That points at the codebook
        hyperparameters (nb_entries, cb_ema_decay, cb_reset_every), not at the commitment
        pressure.

Two forward-path facts this had to work around
----------------------------------------------
1. `skip_codebook = pool_only and not return_recon` (vqvae.py:1391).  Every existing eval
   path calls `pool_only=True, return_recon=False`, so the codebook NEVER RUNS and every
   number in the investigation so far is pre-VQ.  This script calls
   `pool_only=False, return_recon=False`, which runs the codebook without the decoder.
2. `quantize = x + (quantize - x).detach()` (vqvae.py:504) is straight-through for gradients
   only — the returned VALUE is the centroid, so a forward hook gives real `z_q`.

Scope limit: with `nb_levels > 1`, levels below the top are quantized from
`[enc, decoder_output]` during training but from `[enc, zeros]` here, because
`return_recon=False` produces no decoder output.  The top level is unaffected (it is
quantized before any decoder runs), so on a 1-level run this is exact.  A warning is printed
otherwise.

Usage
-----
    python -m eval.vq_resolution --run-dir results/synthetic/RUN \\
        --checkpoints vqvae_best.pt vqvae_model.pt --num-samples 600
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval.identifiability_metrics import block_mcc

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


def _install_vq_hooks(inner, level):
    """Capture ``z_e`` (post-projection, pre-quantization) and ``z_q`` at one level.

    ``conv_in`` gives z_e because ``CodeLayer.forward`` is ``quantize(project(x))`` and
    ``project`` IS ``conv_in``.  The layer's own output tuple gives ``(z_q, diff, ind)``.
    Both are captured as LISTS: with ``separate_content_codebooks`` the two views are
    quantized in separate calls, so a single-slot capture would silently keep only view 1.
    """
    cap = {"z_e": [], "z_q": [], "diff": [], "ind": []}
    handles = []
    for cb in _codebooks_for(inner, level):
        handles.append(cb.conv_in.register_forward_hook(lambda m, i, o: cap["z_e"].append(o.detach())))

        def out_hook(m, i, o, _cap=cap):
            q, d, ind = o
            _cap["z_q"].append(q.detach())
            _cap["diff"].append(float(d.detach()))
            _cap["ind"].append(ind.detach())

        handles.append(cb.register_forward_hook(out_hook))
    return cap, handles


def _codebooks_for(inner, level):
    """Every content codebook that can fire at ``level`` (both views when split)."""
    out = [inner.codebooks[level]]
    v1 = getattr(inner, "codebooks_v1", None)
    if v1 is not None:
        out.append(v1[level])
    return out


def _patch_pool(t, grid):
    """(B, C, D, H, W) -> (B, P*C) patch-major, matching ``eval.dci._pool_and_split_view``."""
    return F.adaptive_avg_pool3d(t, grid).flatten(2).permute(0, 2, 1).flatten(1)


@torch.no_grad()
def collect(model, dataset, device, level, grid, batch_size, num_workers, max_samples):
    """One pass: patch-pooled z_e/z_q for view 1, plus streaming VQ statistics.

    Statistics are accumulated rather than stored — the full-resolution z_e for 600 samples
    at embed_dim 48 is ~1 GB, and none of the three tests needs it retained.
    """
    from torch.utils.data import DataLoader

    inner = model.module if hasattr(model, "module") else model
    if inner.nb_levels > 1 and level < inner.nb_levels - 1:
        logger.warning(
            "level %d is below the top of a %d-level model: training quantizes it from "
            "[enc, decoder_output] but this runs without a decoder, so z_q differs.",
            level,
            inner.nb_levels,
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    n_embed = int(_codebooks_for(inner, level)[0].n_embed)
    counts = np.zeros(n_embed, dtype=np.int64)
    sse = 0.0  # sum of squared quantization error
    s1 = 0.0  # sum of z_e
    s2 = 0.0  # sum of z_e^2
    nel = 0
    Ze, Zq, GT = [], [], []
    seen = 0

    for batch in loader:
        imgs = batch["image"]
        x = torch.cat(imgs, dim=0).to(device)
        cap, handles = _install_vq_hooks(inner, level)
        try:
            # pool_only=False keeps skip_codebook False, so the codebook actually runs;
            # return_recon=False skips the decoder.
            model(x, pool_only=False, return_recon=False, n_views=len(imgs), subsets=[(0, 1)])
        finally:
            for h in handles:
                h.remove()
        if not cap["z_e"]:
            raise RuntimeError(f"no codebook forward captured at level {level} — is the model quantizing?")

        z_e = torch.cat(cap["z_e"], dim=0).float()
        z_q = torch.cat(cap["z_q"], dim=0).float()

        # Streaming stats over EVERYTHING the codebook saw (both views): utilisation and
        # quantization error are properties of the codebook's operating condition.
        d = (z_q - z_e).pow(2)
        sse += float(d.sum())
        s1 += float(z_e.sum())
        s2 += float(z_e.pow(2).sum())
        nel += z_e.numel()
        for ind in cap["ind"]:
            counts += np.bincount(ind.reshape(-1).cpu().numpy(), minlength=n_embed)

        # View 1 only for the readout test, matching every other probe in this project.
        B = z_e.shape[0] // len(imgs)
        Ze.append(_patch_pool(z_e[:B], grid).cpu())
        Zq.append(_patch_pool(z_q[:B], grid).cpu())
        GT.append(batch["gt_latents"]["z_content"].cpu().numpy())

        seen += B
        del cap, z_e, z_q, d
        if max_samples and seen >= max_samples:
            break

    mean = s1 / max(nel, 1)
    var = max(s2 / max(nel, 1) - mean * mean, 0.0)
    return {
        "counts": counts,
        "rms_qerr": float(np.sqrt(sse / max(nel, 1))),
        "feat_std": float(np.sqrt(var)),
        "z_e": torch.cat(Ze)[:max_samples].numpy(),
        "z_q": torch.cat(Zq)[:max_samples].numpy(),
        "gt": np.concatenate(GT, axis=0)[:max_samples],
    }


# --------------------------------------------------------------------------- #
# The three tests
# --------------------------------------------------------------------------- #


def test_utilisation(counts):
    """Perplexity and assignment concentration. Perplexity RATIO is the comparable number."""
    total = counts.sum()
    p = counts / max(total, 1)
    nz = p[p > 0]
    perp = float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0
    order = np.sort(p)[::-1]
    return {
        "n_embed": int(len(counts)),
        "active": int((counts > 0).sum()),
        "perplexity": perp,
        "perp_ratio": perp / max(len(counts), 1),
        "top10_share": float(order[:10].sum()),
    }


def test_resolution(rms_qerr, feat_std):
    """Quantization error RELATIVE to the feature scale.

    The absolute error is uninterpretable across checkpoints because feat_std runs 0.29 ->
    66.9; only ``rel`` is comparable, and it is the number the hypothesis lives on.
    """
    rel = rms_qerr / max(feat_std, 1e-12)
    return {"rms_qerr": rms_qerr, "feat_std": feat_std, "rel_qerr": rel}


def test_readout(z_e, z_q, gt, seeds, n_splits):
    """block-MCC before vs after quantization. Converging = the outputs sit on the centroids."""
    a = block_mcc(z_e, gt, seeds=seeds, n_splits=n_splits)
    b = block_mcc(z_q, gt, seeds=seeds, n_splits=n_splits)
    return {"mcc_pre": a["mean"], "mcc_post": b["mean"], "gap": a["mean"] - b["mean"]}


def score_checkpoint(model, dataset, device, cli):
    c = collect(model, dataset, device, cli.level, tuple(cli.grid), cli.batch_size, cli.num_workers, cli.num_samples)
    return {
        "util": test_utilisation(c["counts"]),
        "res": test_resolution(c["rms_qerr"], c["feat_std"]),
        "read": test_readout(c["z_e"], c["z_q"], c["gt"], tuple(cli.seeds), cli.n_splits),
        "n": int(len(c["gt"])),
    }


def _report(label, r):
    u, s, k = r["util"], r["res"], r["read"]
    print(f"\n  {label}   (N={r['n']})")
    print(
        f"    1 UTILISATION   active {u['active']}/{u['n_embed']}   perplexity {u['perplexity']:.1f}"
        f"   ratio {u['perp_ratio']:.4f}   top-10 codes hold {u['top10_share']:.1%}"
    )
    print(
        f"    2 RESOLUTION    rms_qerr {s['rms_qerr']:.4g}   feat_std {s['feat_std']:.4g}"
        f"   RELATIVE {s['rel_qerr']:.4f}"
    )
    print(f"    3 READOUT       mcc(z_e) {k['mcc_pre']:.4f}   mcc(z_q) {k['mcc_post']:.4f}" f"   gap {k['gap']:+.4f}")


def _verdict(a, b, early, late):
    print(f"\n{'=' * 78}\nVERDICT — {early} → {late}\n{'=' * 78}")
    d_rel = b["res"]["rel_qerr"] - a["res"]["rel_qerr"]
    d_perp = b["util"]["perp_ratio"] - a["util"]["perp_ratio"]
    d_gap = abs(b["read"]["gap"]) - abs(a["read"]["gap"])
    print(f"  relative quantization error  {a['res']['rel_qerr']:.4f} → {b['res']['rel_qerr']:.4f}  ({d_rel:+.4f})")
    print(
        f"  perplexity ratio             {a['util']['perp_ratio']:.4f} → {b['util']['perp_ratio']:.4f}"
        f"  ({d_perp:+.4f})"
    )
    print(
        f"  |mcc(z_e) − mcc(z_q)|        {abs(a['read']['gap']):.4f} → {abs(b['read']['gap']):.4f}" f"  ({d_gap:+.4f})"
    )

    clustering = d_rel < -0.02 * max(a["res"]["rel_qerr"], 1e-9) and d_rel < -0.005
    converging = d_gap < -0.002
    # Falling quantization error is NOT evidence of coarsening on its own. A codebook that
    # goes from badly under-used to fully used has denser centroids, so the error falls while
    # resolution IMPROVES — the opposite of the hypothesis. Coarsening requires utilisation
    # to fall or hold; check that before reading the error at all.
    if d_perp > 0.05:
        print("\n  => HYPOTHESIS REFUTED, and in the informative direction. Quantization got FINER")
        print(f"     (rel error {a['res']['rel_qerr']:.4f} → {b['res']['rel_qerr']:.4f}) because the codebook")
        print(f"     went from under-used to well-used (perplexity ratio {a['util']['perp_ratio']:.3f} →")
        print(f"     {b['util']['perp_ratio']:.3f}, active {a['util']['active']} → {b['util']['active']}).")
        print("     The discretization is LESS lossy at the end, so it cannot be what the readout lost.")
        if b["read"]["mcc_post"] < a["read"]["mcc_post"]:
            print(f"     Note mcc(z_q) ALSO fell ({a['read']['mcc_post']:.4f} → {b['read']['mcc_post']:.4f}) despite")
            print("     more codes, better used. So the codes carry less factor information while being")
            print("     more numerous: this is not a resolution problem, it is a question of WHAT is")
            print("     being encoded. Do not reach for a codebook hyperparameter.")
    elif clustering and converging:
        print("\n  => CONFIRMED. The encoder outputs collapsed onto the centroids and the pre/post-VQ")
        print("     readouts converged with them. Fix order: (1) the VQ double-count —")
        print("     BaselineLoss returns quantization_losses (losses.py:1611) AND main_multimodal")
        print("     adds it again, so the weight is doubled and scaled by scale_recon_loss;")
        print("     (2) bound feat_std, the variance hinge is one-sided and went dormant;")
        print("     (3) sweep vq_commitment_weight down; (4) try a continuous bottleneck.")
    elif clustering:
        print("\n  => PARTIAL. Clustering is happening but the readouts did not converge, so the")
        print("     coarsening is not (yet) what the probe is losing. Report both, claim neither.")
    elif d_perp < -0.05:
        print("\n  => CODEBOOK COARSENED but the encoder outputs did not move onto it. That points at")
        print("     nb_entries / cb_ema_decay / cb_reset_every, NOT at the commitment pressure.")
    else:
        print("\n  => NOT CONFIRMED. Relative quantization error is flat: the encoder outputs are no")
        print("     closer to the centroids than they were. Quantization coarsening is dead as an")
        print("     explanation, and no other candidate is currently standing.")
    print("\n  Reminder: this is n=1. Nothing here separates cause from correlate — only a run with")
    print("  the VQ pressure changed can do that.")


def main():
    p = argparse.ArgumentParser(description="Is the patch-MCC decay quantization coarsening?")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoints", nargs="+", default=["vqvae_best.pt", "vqvae_model.pt"], help="EARLY FIRST")
    p.add_argument("--num-samples", type=int, default=600)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--grid", type=int, nargs=3, default=None, help="Default: the run's --patch-grid")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--cache-dataset", action="store_true")
    cli = p.parse_args()

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref = load_run_args(cli.run_dir)
    if cli.grid is None:
        cli.grid = list(getattr(ref, "patch_grid", None) or [8, 8, 8])
    dataset = build_synthetic_test_set(
        ref, num_samples=cli.num_samples, cache=cli.cache_dataset, causal=(cli.causal == "match")
    )

    print("\n" + "=" * 78)
    print("  IS THE PATCH-MCC DECAY QUANTIZATION COARSENING?")
    print(f"  level {cli.level}, patch grid {tuple(cli.grid)}, causal={cli.causal}")
    print("=" * 78)

    res = {}
    for name in cli.checkpoints:
        ckpt = os.path.join(cli.run_dir, name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt, None)
        model.to(device)
        res[name] = score_checkpoint(model, dataset, device, cli)
        _report(name, res[name])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(cli.checkpoints) >= 2:
        _verdict(res[cli.checkpoints[0]], res[cli.checkpoints[-1]], cli.checkpoints[0], cli.checkpoints[-1])


if __name__ == "__main__":
    main()
