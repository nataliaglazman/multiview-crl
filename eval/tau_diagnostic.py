"""Post-hoc InfoNCE temperature diagnostic on a frozen checkpoint.

Answers "is tau wrong for this task?" without retraining.  Loads a run's
checkpoint, extracts the content embeddings exactly as the training loop does
(same pooling, same content-channel selection, same L2 normalisation, same
negative-set convention), then recomputes the InfoNCE readouts over a grid of
temperatures on those *frozen* embeddings.

The point of the frozen sweep: tau only matters if the model's similarity
geometry actually sits near a regime boundary.  If ``p_pos`` and ``n_eff``
barely move across tau in [0.02, 1.0], tau is not the binding constraint and a
tau sweep over full training runs is wasted compute.

What a frozen sweep can and cannot show: temperature rescales logits, and
argmax is scale-invariant, so **top-1 accuracy is constant down the column by
construction** — a frozen sweep says nothing about accuracy.  What it does show
is the *gradient regime* the temperature selects (``p_pos``, ``n_eff``), which
is what a retrained model would actually respond to.

Readouts per tau (all dimensionless, all comparable across runs):

* ``loss``  — mirrors the training objective.
* ``top1``  — constant by the argument above; printed only so the run's own tau
  row can be cross-checked against the logged ``Contrastive/top1_acc``.
* ``p_pos``  — mean softmax probability on the positive.  ->1 means the task is
  solved and the gradient has vanished (tau too small, or task too easy).
  ~1/R is chance.
* ``n_eff``  — effective number of negatives receiving gradient, exp(H(w)) over
  the renormalised negative weights.  ~1 means the loss is a hard max over the
  single hardest negative (tau too small: aggressive hard-negative mining, and
  on registered volumes the hardest negative is often a *false* negative).
  ~R-1 means uniform weighting (tau too large: no discriminative shaping).
* ``gap``  — effective logit gap (pos_sim_mean - neg_sim_mean) / tau, in nats.

With ``--false-negative-check`` (synthetic only) it also asks whether the
hardest negatives are GT-closer in content than random negatives, which is the
failure mode a too-small tau amplifies.

Usage:
    python -m eval.tau_diagnostic /path/to/run_dir
    python -m eval.tau_diagnostic /path/to/run_dir --taus 0.02 0.05 0.1 0.3
    python -m eval.tau_diagnostic --selftest        # verify the readouts, no checkpoint
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Effective-gap band (nats) used only to annotate rows.  Below the low end the
# softmax is near-uniform; above the high end it is saturated.
GAP_LO, GAP_HI = 1.0, 5.0
# p_pos above this means the positive already owns the row: no gradient left.
P_POS_SATURATED = 0.95
DEFAULT_TAUS = (0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 1.0)


def build_score_rows(hz_n, tau, cross_view_negs_only):
    """Per-patch logit rows for one view pair, mirroring ``training.losses``.

    Args:
        hz_n: (2, B, C, P) content features, already L2-normalised over C.
        tau: Temperature.
        cross_view_negs_only: Match the run's negative-set convention.  False
            (the default recipe) builds the (P, 2B, 2B) matrix with same-view
            negatives included and the self-similarity diagonal masked, so each
            row has 2B-2 competitors.  True builds (P, B, B): B-1 competitors.

    Returns:
        (logits, targets, pos_raw, neg_raw) — logits (P, R, R) are tau-scaled;
        pos_raw/neg_raw are the *raw* cross-view cosines (tau undone), matching
        what ``pos_sim_mean`` / ``neg_sim_mean`` report in training.
    """
    B = hz_n.shape[1]
    sim_cross = torch.einsum("bcp,dcp->pbd", hz_n[0], hz_n[1]) / tau  # (P, B, B)
    eye = torch.eye(B, dtype=torch.bool, device=hz_n.device)
    offdiag = ~eye

    pos_raw = torch.diagonal(sim_cross, dim1=-2, dim2=-1).reshape(-1) * tau
    neg_raw = sim_cross[:, offdiag].reshape(-1) * tau

    if cross_view_negs_only:
        targets = torch.arange(B, device=hz_n.device)
        return sim_cross, targets, pos_raw, neg_raw

    sim_00 = torch.einsum("bcp,dcp->pbd", hz_n[0], hz_n[0]) / tau
    sim_11 = torch.einsum("bcp,dcp->pbd", hz_n[1], hz_n[1]) / tau
    sim_00[:, eye] = float("-inf")
    sim_11[:, eye] = float("-inf")

    row0 = torch.cat([sim_cross, sim_00], dim=-1)  # (P, B, 2B)
    row1 = torch.cat([sim_11, sim_cross.transpose(-1, -2)], dim=-1)  # (P, B, 2B)
    logits = torch.cat([row0, row1], dim=-2)  # (P, 2B, 2B)
    targets = torch.arange(2 * B, device=hz_n.device)
    return logits, targets, pos_raw, neg_raw


def _row_stats(logits, targets):
    """Loss, top-1, p_pos and effective-negative-count over (P, R, R) logit rows."""
    P, R, _ = logits.shape
    flat = logits.reshape(P * R, R)
    tgt = targets.repeat(P)

    loss = F.cross_entropy(flat, tgt, reduction="sum")
    correct = (flat.argmax(dim=1) == tgt).sum()

    probs = flat.softmax(dim=1)
    p_pos = probs.gather(1, tgt[:, None]).squeeze(1)  # (P*R,)

    # Weights among negatives only: drop the positive column, renormalise.
    # Masked (-inf) same-view self columns contribute exactly 0 and drop out.
    neg_probs = probs.scatter(1, tgt[:, None], 0.0)
    neg_probs = neg_probs / neg_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(neg_probs * neg_probs.clamp_min(1e-12).log()).sum(dim=1)
    n_eff = entropy.exp()

    return {
        "loss_sum": loss.item(),
        "correct": correct.item(),
        "rows": P * R,
        "p_pos_sum": p_pos.sum().item(),
        "n_eff_sum": n_eff.sum().item(),
    }


def tau_report(hz_n, taus, cross_view_negs_only=False, patch_chunk=64):
    """Sweep temperatures over frozen embeddings.  Returns a list of row dicts.

    ``patch_chunk`` bounds peak memory: the default recipe materialises a
    (chunk, 2B, 2B) float matrix per step, so B=128 with chunk=64 is ~17M
    floats rather than the ~134M a full P=512 pass would take.
    """
    n_views, B, _, P = hz_n.shape
    if n_views < 2:
        raise ValueError(f"need 2 views, got {n_views}")
    hz_n = hz_n[:2]

    rows = []
    for tau in taus:
        acc = {"loss_sum": 0.0, "correct": 0, "rows": 0, "p_pos_sum": 0.0, "n_eff_sum": 0.0}
        pos_all, neg_all = [], []
        for start in range(0, P, patch_chunk):
            chunk = hz_n[..., start : start + patch_chunk]
            logits, targets, pos_raw, neg_raw = build_score_rows(chunk, tau, cross_view_negs_only)
            for k, v in _row_stats(logits, targets).items():
                acc[k] += v
            pos_all.append(pos_raw)
            neg_all.append(neg_raw)

        pos_all = torch.cat(pos_all)
        neg_all = torch.cat(neg_all)
        pos_mean, neg_mean = pos_all.mean().item(), neg_all.mean().item()
        n_rows = acc["rows"]
        R = B if cross_view_negs_only else 2 * B

        rows.append(
            {
                "tau": tau,
                "loss": acc["loss_sum"] / n_rows,
                "top1": acc["correct"] / n_rows,
                "p_pos": acc["p_pos_sum"] / n_rows,
                "n_eff": acc["n_eff_sum"] / n_rows,
                "n_negatives": R - 1 if cross_view_negs_only else R - 2,
                "pos_sim_mean": pos_mean,
                "neg_sim_mean": neg_mean,
                "neg_sim_std": neg_all.std().item(),
                "gap_nats": (pos_mean - neg_mean) / tau,
                "spread_nats": neg_all.std().item() / tau,
            }
        )
    return rows


def verdict(row):
    """One-word read on a single tau row."""
    if row["p_pos"] >= P_POS_SATURATED or row["gap_nats"] > GAP_HI:
        return "SATURATED (tau too small)"
    if row["gap_nats"] < GAP_LO:
        return "WEAK (tau too large)"
    if row["n_eff"] < 2.0:
        return "HARD-MAX (one negative dominates)"
    return "ok"


def print_report(rows, tau_run=None):
    n_neg = rows[0]["n_negatives"]
    print(f"\n{'tau':>7}  {'loss':>7}  {'top1':>7}  {'p_pos':>7}  {'n_eff':>8}  {'gap':>7}  verdict")
    print(f"{'':>7}  {'':>7}  {'(const)':>7}  {'':>7}  {'/' + str(n_neg):>8}  {'nats':>7}")
    print("-" * 78)
    for r in rows:
        marker = " <- run" if tau_run is not None and abs(r["tau"] - tau_run) < 1e-9 else ""
        print(
            f"{r['tau']:>7.3f}  {r['loss']:>7.3f}  {r['top1']:>7.3f}  {r['p_pos']:>7.3f}  "
            f"{r['n_eff']:>8.1f}  {r['gap_nats']:>7.2f}  {verdict(r)}{marker}"
        )

    # top1 is argmax-based, hence scale-invariant: on frozen embeddings it is
    # constant down the column by construction.  It is printed only to
    # cross-check the run's own tau row against the logged Contrastive/top1_acc.
    # Everything the frozen sweep can actually tell you lives in p_pos / n_eff:
    # those describe the *gradient regime* the temperature puts you in.
    raw_gap = rows[0]["pos_sim_mean"] - rows[0]["neg_sim_mean"]
    neg_std = rows[0]["neg_sim_std"]
    snr = raw_gap / max(neg_std, 1e-12)
    print(f"\nraw cosine gap (tau-independent): {raw_gap:+.4f}   neg sim std: {neg_std:.4f}   gap/std: {snr:+.2f}")

    if snr < 0.25:
        print("  -> positives barely separate from negatives at ANY temperature.")
        print("     No tau fixes this: the limitation is the representation (or false negatives), not the scaling.")
        return

    print(
        f"tau landing the effective gap in [{GAP_LO:g}, {GAP_HI:g}] nats: "
        f"[{raw_gap / GAP_HI:.3f}, {raw_gap / GAP_LO:.3f}]"
    )

    p_span = max(r["p_pos"] for r in rows) - min(r["p_pos"] for r in rows)
    lo_eff, hi_eff = min(r["n_eff"] for r in rows), max(r["n_eff"] for r in rows)
    print(f"p_pos span across the swept range: {p_span:.3f}   n_eff range: {lo_eff:.1f} -> {hi_eff:.1f} (of {n_neg})")
    if p_span < 0.1:
        print("  -> the gradient regime hardly moves with tau; tau is NOT the binding constraint.")
    else:
        print("  -> tau materially changes which pairs get gradient; worth a training sweep.")


def false_negative_check(hz_n, gt_content, k=5, cross_view_negs_only=False, patch_chunk=64):
    """Are the hardest negatives GT-closer in content than random negatives?

    Ranking is tau-independent, so this runs once.  Returns a dict with the mean
    GT content distance to the top-k hardest cross-view negatives, the mean over
    all negatives, and their ratio.  Ratio well below 1 means the loss is
    spending its largest gradients pushing apart samples that genuinely share
    content — the pathology a small tau amplifies.
    """
    B, P = hz_n.shape[1], hz_n.shape[3]
    z = torch.as_tensor(gt_content, dtype=torch.float32, device=hz_n.device)
    z = (z - z.mean(0)) / z.std(0).clamp_min(1e-8)
    gt_dist = torch.cdist(z, z)  # (B, B)
    offdiag = ~torch.eye(B, dtype=torch.bool, device=hz_n.device)
    baseline = gt_dist[offdiag].mean().item()

    hard_sum, hard_n = 0.0, 0
    for start in range(0, P, patch_chunk):
        chunk = hz_n[..., start : start + patch_chunk]
        sim = torch.einsum("bcp,dcp->pbd", chunk[0], chunk[1])  # (p, B, B)
        sim = sim.masked_fill(~offdiag, float("-inf"))
        hardest = sim.topk(k, dim=-1).indices  # (p, B, k)
        rows = torch.arange(B, device=hz_n.device)[None, :, None].expand_as(hardest)
        hard_sum += gt_dist[rows.reshape(-1), hardest.reshape(-1)].sum().item()
        hard_n += hardest.numel()

    hard_mean = hard_sum / max(hard_n, 1)
    return {"hardest_gt_dist": hard_mean, "random_gt_dist": baseline, "ratio": hard_mean / max(baseline, 1e-12), "k": k}


def extract_content_patches(model, dataset, args, device, batch_size, pooling, level, foreground_mask=True):
    """One frozen batch of L2-normalised content features, shaped (n_views, B, C', P).

    Mirrors the training forward: ``pool_only=True`` with the run's patch grid,
    content channels taken from that level's Gumbel mask, foreground patch
    positions dropped when the run trained with ``--patch-foreground-mask``,
    then L2 normalisation over the channel axis (``losses.py``: hz_n).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    imgs = batch["image"]
    n_views = len(imgs)
    x = torch.cat(imgs, dim=0).to(device)

    model.eval()
    with torch.no_grad():
        if pooling is None:
            enc_out = model(x, pool_only=True, n_views=n_views)
        else:
            enc_out = model(x, pool_only=True, n_views=n_views, patch_grid=pooling)

        feats = enc_out[2] if isinstance(enc_out, tuple) else [enc_out]
        soft_masks = enc_out[6] if isinstance(enc_out, tuple) and len(enc_out) > 6 else {}

        feat = feats[level]
        if feat.dim() == 2:  # GAP: (n_views*B, C) -> single patch position
            feat = feat.unsqueeze(-1)
        hz = feat.reshape(n_views, -1, *feat.shape[1:])  # (n_views, B, C, P)

        if isinstance(soft_masks, dict) and level in soft_masks:
            mask = soft_masks[level]
            mask = mask[0] if isinstance(mask, tuple) else mask
            content_idx = torch.where(mask.bool())[-1]
        else:
            content_idx = torch.as_tensor(args.content_indices[0], device=hz.device)
        hz = hz[:, :, content_idx, :]

        if pooling is not None and foreground_mask and getattr(args, "patch_foreground_mask", False):
            masks = batch.get("mask")
            if masks is not None:
                m = torch.cat(masks, 0).to(device).float()
                thr = getattr(args, "patch_foreground_thresh", 0.05)
                frac = F.adaptive_avg_pool3d(m, tuple(pooling)).flatten(1)
                keep = (frac >= thr).any(dim=0)
                if bool(keep.any()):
                    hz = hz[..., keep]
                    logger.info("foreground mask: kept %d/%d patch positions", int(keep.sum()), keep.numel())
            else:
                logger.warning(
                    "run used --patch-foreground-mask but the batch has no 'mask' key; scoring all positions"
                )

        hz_n = F.normalize(hz, dim=2, eps=1e-6)

    gt = batch.get("gt_latents")
    gt_content = gt["z_content"].numpy() if gt is not None and "z_content" in gt else None
    return hz_n, gt_content


def selftest(device="cpu"):
    """Run the readouts on embeddings with a known signal level.

    Three regimes with the same tau grid: near-duplicate views (easy), a modest
    shared signal, and pure noise.  Confirms p_pos/n_eff/gap move the way the
    docstring claims before you trust them on a checkpoint.
    """
    torch.manual_seed(0)
    B, C, P = 64, 44, 8
    for name, signal in (
        ("easy (views nearly identical)", 0.95),
        ("moderate shared signal", 0.35),
        ("no shared signal", 0.0),
    ):
        base = torch.randn(B, C, P, device=device)
        noise0, noise1 = torch.randn(B, C, P, device=device), torch.randn(B, C, P, device=device)
        v0 = signal * base + (1 - signal) * noise0
        v1 = signal * base + (1 - signal) * noise1
        hz_n = F.normalize(torch.stack([v0, v1]), dim=2, eps=1e-6)
        print(f"\n=== selftest: {name} ===")
        print_report(tau_report(hz_n, DEFAULT_TAUS, cross_view_negs_only=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", help="Run directory containing settings.json and the checkpoint.")
    p.add_argument(
        "--checkpoint", default=None, help="Checkpoint filename or path (default: the run's vqvae_model.pt)."
    )
    p.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Default: the run's own batch_size. Changing it changes the number of negatives, which is part of what tau is tuned against.",
    )
    p.add_argument("--level", type=int, default=0)
    p.add_argument(
        "--num-samples", type=int, default=None, help="Synthetic test-set size (only the first batch is used)."
    )
    p.add_argument(
        "--pooling",
        choices=("run", "patch", "gap"),
        default="run",
        help="'run' follows the checkpoint's patch_contrastive/patch_grid settings.",
    )
    p.add_argument("--patch-chunk", type=int, default=64)
    p.add_argument(
        "--no-foreground-mask",
        action="store_true",
        help="Score every patch position even if the run trained with foreground masking.",
    )
    p.add_argument(
        "--false-negative-check",
        action="store_true",
        help="Compare GT content distance of the hardest negatives against random ones (synthetic runs only).",
    )
    p.add_argument("--fn-k", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--selftest", action="store_true", help="Run on synthetic embeddings with known signal; no checkpoint needed."
    )
    cli = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if cli.selftest:
        selftest()
        return
    if not cli.run_dir:
        p.error("run_dir is required (or pass --selftest)")

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = cli.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(cli.run_dir, ckpt)
    model, args, device = load_model_from_run_dir(cli.run_dir, ckpt, device)

    if cli.pooling == "gap":
        pooling = None
    elif cli.pooling == "patch":
        pooling = tuple(args.patch_grid)
    else:
        pooling = (
            tuple(args.patch_grid)
            if getattr(args, "patch_contrastive", False) and getattr(args, "patch_grid", None)
            else None
        )

    batch_size = cli.batch_size or getattr(args, "batch_size", 32)
    cross_only = bool(getattr(args, "cross_view_negs_only", False))
    tau_run = float(getattr(args, "tau", float("nan")))

    dataset = build_synthetic_test_set(args, cli.num_samples or batch_size)
    hz_n, gt_content = extract_content_patches(
        model, dataset, args, device, batch_size, pooling, cli.level, foreground_mask=not cli.no_foreground_mask
    )

    n_views, B, C, P = hz_n.shape
    print(f"\nrun: {cli.run_dir}")
    print(f"level {cli.level} | {n_views} views | B={B} | content channels={C} | patches={P}")
    print(
        f"pooling={'gap' if pooling is None else f'patch{list(pooling)}'} | "
        f"negatives/anchor={(B - 1) if cross_only else (2 * B - 2)} | trained tau={tau_run:g}"
    )

    taus = sorted(set(cli.taus) | ({tau_run} if np.isfinite(tau_run) else set()))
    print_report(tau_report(hz_n, taus, cross_view_negs_only=cross_only, patch_chunk=cli.patch_chunk), tau_run=tau_run)

    if cli.false_negative_check:
        if gt_content is None:
            logger.warning("--false-negative-check needs GT latents; skipping (non-synthetic dataset?)")
        else:
            fn = false_negative_check(hz_n, gt_content, k=cli.fn_k, patch_chunk=cli.patch_chunk)
            print(f"\nfalse-negative check (top-{fn['k']} hardest cross-view negatives):")
            print(f"  GT content distance, hardest negatives : {fn['hardest_gt_dist']:.3f}")
            print(f"  GT content distance, random negatives  : {fn['random_gt_dist']:.3f}")
            print(f"  ratio                                  : {fn['ratio']:.3f}")
            if fn["ratio"] < 0.9:
                print("  -> hardest negatives share content: small tau is penalising true content matches.")
            else:
                print("  -> hardest negatives are not GT-closer than random; no false-negative pressure.")


if __name__ == "__main__":
    main()
