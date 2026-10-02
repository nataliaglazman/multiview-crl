#!/usr/bin/env python
"""Gradient attribution using the current training objective, without optimizer updates.

Default --target balance reports weighted per-component encoder gradients, conflicts,
module norms and total-gradient parity, plus each term's first-order effect on the i.i.d.
decodability of --decode-factors (default ventricle_size). --target decode adds temporary
unit-direction steps that re-measure that decoding, with matched random controls; --target
mcc also re-measures patch block-MCC. --target reconstruction retains the separate
similarity-only pixel-MAE experiment. See eval/gradients/GRADIENT_ATTRIBUTION.md.

Usage
-----
    python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN
    python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --target decode
    python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --decode-factors ventricle_size brain_size
    python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --target mcc
    python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --target reconstruction
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

# torch is imported lazily inside the measurement path so the reporting helpers below
# stay unit-testable on plain numpy (same convention as eval.protocol.run_dci_compare).

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ENCODER_MODULES = ("encoders", "encoders_v1", "content_norms", "content_projections")


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def linearity_check(etas, deltas):
    """R^2 of dMCC against eta through the origin.

    A finite difference only means "the derivative along this direction" while the response
    is linear in the step. Near 1 means the eta sweep is in that regime; well below means
    the steps are too large and the per-unit numbers should not be quoted.
    """
    e, d = np.asarray(etas, dtype=float), np.asarray(deltas, dtype=float)
    ok = np.isfinite(e) & np.isfinite(d)
    e, d = e[ok], d[ok]
    if len(e) < 2 or not np.any(e):
        return float("nan")
    slope = float(np.dot(e, d) / np.dot(e, e))
    ss_res = float(((d - slope * e) ** 2).sum())
    ss_tot = float((d**2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _encoder_params(model):
    """The parameters that shape the representation, unwrapped from any DataParallel/MoCo.

    Must match --freeze-encoder's target set: these are exactly the parameters
    ``enc_out[2]`` depends on, so a step that leaves them untouched cannot move block-MCC.
    """
    model = getattr(model, "module", model)
    model = getattr(model, "online", model)
    named = []
    for mod_name in ENCODER_MODULES:
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        for pn, p in mod.named_parameters():
            named.append((f"{mod_name}.{pn}", p))
    if not named:
        raise RuntimeError(
            f"No encoder parameters found on {type(model).__name__}. If the model is wrapped "
            "differently, extend the unwrapping in _encoder_params."
        )
    return named


def _flat(grads):
    import torch

    return torch.cat([g.reshape(-1) for g in grads]).detach()


def snr_decomposition(G):
    """Descriptive mean/noise estimates across batches.

    The signed signal estimate can be negative at finite sample size. This does not
    prove that a term is harmless, that its optimum is unattainable, or that training
    follows a random walk. Interpretation assumes comparable independent batches.
    """
    G = np.asarray(G, dtype=np.float64)
    b = G.shape[0]
    if b < 2:
        return {"n_batches": b}
    gbar = G.mean(0)
    naive = float((gbar**2).sum())
    # E||g - gbar||^2 underestimates tr(Cov) by (B-1)/B; correct it.
    tr_cov = float(((G - gbar) ** 2).sum(1).mean()) * b / (b - 1)
    sig2 = naive - tr_cov / b
    per_batch = float(np.sqrt((G**2).sum(1).mean()))
    return {
        "n_batches": b,
        "naive_norm": float(np.sqrt(naive)),
        "signal_norm": float(np.sign(sig2) * np.sqrt(abs(sig2))),
        "signal_sq": sig2,
        "noise_norm": float(np.sqrt(tr_cov)),
        "per_batch_norm": per_batch,
        # Batches needed for the averaged gradient to be majority signal. Infinite when the
        # expected gradient is not distinguishable from zero.
        "batches_for_snr1": (tr_cov / sig2) if sig2 > 0 else float("inf"),
    }


def _mcc_now(model, dataset, device, grid, level, batch_size, gt_cache, seeds, n_splits):
    from eval.diagnostics.patch_mcc_decay import extract_patch_block
    from eval.metrics.identifiability_metrics import block_mcc

    content, gt, _cov = extract_patch_block(model, dataset, device, grid, level, batch_size, 0)
    if gt_cache is None:
        gt_cache = gt
    x = content.reshape(content.shape[0], -1)
    return block_mcc(x, gt_cache, seeds=seeds, n_splits=n_splits)["mean"], gt_cache


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", choices=("balance", "decode", "mcc", "reconstruction"), default="balance")
    ap.add_argument(
        "--decode-factors",
        nargs="*",
        default=["ventricle_size"],
        help="Content factors whose i.i.d. decodability each term is tested against (GAP-assigned factors "
        "only). Pass the flag with no names to skip decoding.",
    )
    ap.add_argument(
        "--decode-samples",
        type=int,
        default=512,
        help="i.i.d. test subjects for decoding, rendered once and cached (~2 MB each at 64^3).",
    )
    ap.add_argument("--checkpoints", nargs="+", default=["vqvae_model.pt"])
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, nargs=3)
    ap.add_argument(
        "--grad-batches",
        type=int,
        default=16,
        help="Batches per gradient estimate. The contrastive gradient's batch-to-batch sd has been "
        "measured at ~69%% of its mean, so a handful of batches cannot resolve its expectation from zero.",
    )
    ap.add_argument(
        "--grad-batch-size", type=int, help="Default: saved training batch size. Smaller batches change BT statistics."
    )
    ap.add_argument(
        "--ema-mode",
        choices=("reference", "instantaneous"),
        default="reference",
        help="reference: a frozen checkpoint estimate of the correlation EMA, NOT its training history. "
        "instantaneous: no EMA. (A cold EMA is instantaneous exactly, by its bias correction.)",
    )
    ap.add_argument(
        "--ema-reference-batches",
        type=int,
        help="Default: the EMA's effective window (1+m)/(1-m), 199 at m=0.99, so the reference is as "
        "noisy as training's own EMA. Fewer batches inflate the off-diagonal gradient; the report "
        "prints the resulting noise next to the training EMA's.",
    )
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers rendering synthetic subjects.")
    ap.add_argument("--mcc-samples", type=int, default=600)
    ap.add_argument(
        "--eval-batch", "--mcc-batch", dest="mcc_batch", type=int, default=8, help="Subjects per evaluation forward."
    )
    ap.add_argument("--etas", type=float, nargs="+")
    ap.add_argument("--recon-samples", type=int, default=64)
    ap.add_argument("--recon-batch-size", type=int, default=4)
    ap.add_argument("--recon-clamp", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--precondition", action="store_true", help="Deprecated: rejected without a verified optimizer-name mapping."
    )
    ap.add_argument(
        "--snr", action="store_true", help="Report batch SNR statistics without a step sweep (included in balance)."
    )
    ap.add_argument("--random-controls", type=int, default=1)
    ap.add_argument("--causal", choices=("match", "iid"), default="match")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int, default=4)
    cli = ap.parse_args(argv)
    if cli.precondition:
        ap.error(
            "--precondition cannot safely map legacy optimizer state to named encoder parameters; raw gradients only."
        )
    counts = (cli.grad_batches, cli.threads, cli.recon_samples, cli.recon_batch_size, cli.mcc_samples, cli.mcc_batch)
    if cli.ema_reference_batches is not None:
        counts += (cli.ema_reference_batches,)
    if min(counts) < 1 or min(cli.random_controls, cli.workers) < 0 or cli.n_splits < 2:
        ap.error("Counts must be positive; random controls and workers nonnegative; n-splits >=2.")
    if cli.grad_batch_size is not None and cli.grad_batch_size < 2:
        ap.error("--grad-batch-size must be at least two subjects.")
    if cli.grid is not None and min(cli.grid) < 1:
        ap.error("Grid entries must be positive.")
    if cli.target == "decode" and not cli.decode_factors:
        ap.error("--target decode needs at least one --decode-factors name.")
    if cli.decode_factors and cli.decode_samples < 4 * cli.n_splits:
        ap.error(f"--decode-samples must be at least {4 * cli.n_splits} for {cli.n_splits}-fold CV.")
    cli.etas = (
        cli.etas
        if cli.etas is not None
        else ([1e-5, 3e-5, 1e-4] if cli.target == "reconstruction" else [0.05, 0.2, 0.8])
    )
    if any(not np.isfinite(e) or e <= 0 for e in cli.etas):
        ap.error("Etas must be finite and positive.")
    if cli.target == "reconstruction":
        if cli.snr:
            ap.error("--snr is not supported by the isolated reconstruction experiment.")
        from eval.gradients.reconstruction_attribution import run
    else:
        if cli.level != 0:
            ap.error("Full loss attribution currently supports single-level models at level 0.")
        from eval.gradients.loss_gradient_audit import run
    run(cli)


if __name__ == "__main__":
    main()
