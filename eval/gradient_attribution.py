#!/usr/bin/env python
"""Which loss is degrading patch-MCC, measured directly rather than inferred.

The decay of ``selection/mcc_by_pool/patch`` after its early peak was traced to the
recon phase by ELIMINATION: every Barlow Twins term was >=93% converged before the peak,
so the contrastive objective was at a fixed point while the metric fell. That is an
argument from timing. This measures the attribution instead.

Two questions, two methods:

  1. HOW BIG is each force, and do they conflict?  Compute the gradient of each loss term
     with respect to the ENCODER parameters only (the ones ``--freeze-encoder`` pins, which
     are exactly the ones ``enc_out[2]`` depends on), then report norms and pairwise
     cosines. A negative cosine between recon and contrastive means they actively pull the
     encoder in opposing directions; near-zero means they are simply unrelated.

  2. WHAT DOES A STEP COST?  Norms and cosines say nothing about a metric that is not
     differentiable (block-MCC ends in a Hungarian match over a cross-validated ridge).
     So take an actual step along each normalised gradient and re-measure the real metric:

         theta' = theta - eta * g / ||g||        ->  block-MCC(theta') - block-MCC(theta)

     This is the honest bridge between "which gradient is larger" and "which gradient
     costs identifiability", because it uses the metric itself, not a differentiable proxy.

Read the sign convention carefully: a NEGATIVE dMCC means stepping along that loss's
descent direction REDUCES block-MCC, i.e. that loss is degrading identifiability.

``--metric rank`` re-measures the effective rank of the GAP-pooled content block instead —
the same ``content_rank`` ``run_dci_compare`` prints — which is what to use when the
representation has collapsed rather than merely lost identifiability.

Barlow Twins is reported as its four ADDITIVE terms (``bt_on``/``bt_off``/``bt_sim``/
``bt_var``, and the ``btgap_*`` companion when ``--bt-gap-weight`` is on), not as one
"contrastive" force, because they pull in different directions and a lumped gradient cannot
say which one moved a metric. See ``_bt_split`` for how the four are recovered exactly from
the shipped loss.

Do NOT predict the SIGN of a term's rank excess from its global optimum. It is tempting to
argue that only off_diag can build rank, since d copies of a single unit-variance signal
minimise on_diag, sim and var simultaneously — but that describes where a term would end up
running alone forever, not the direction it pulls from the current parameters. Measured on
this project at two checkpoints of one run, the signs of ``btgap_on`` and ``btgap_off`` came
out OPPOSITE to that argument at rank 10.3 (+0.267 / -0.405, both at linearity R^2 = 1.000).
Effective rank is the entropy of the singular-value spectrum, so it rewards BALANCE, and
which term balances the spectrum from here is an empirical question. Read the signs off the
table; do not assume them.

Caveats, none of which the numbers announce on their own
--------------------------------------------------------
* Training uses AdamW, which does NOT follow the raw gradient — it rescales per-parameter
  by running second-moment estimates and applies decoupled weight decay. A raw-gradient
  step is therefore an idealisation of what training actually did. Pass
  ``--precondition`` to divide by sqrt(v)+eps from the checkpoint's optimizer state, which
  is much closer to the real update direction.
* Gradients are batch-dependent. ``--grad-batches`` averages over several draws; the
  reported spread across draws is how much of any cosine is noise.
* A finite-difference step is local. Too small and it is swamped by probe noise (block-MCC
  has a per-seed sd around 0.001); too large and it leaves the regime where a single step
  means anything. The eta sweep exists so the reading can be checked for linearity — if
  dMCC is not roughly proportional to eta, the step is too big.
* This attributes the drift at ONE point on the trajectory. The balance of forces changes
  over training, so run it at more than one checkpoint before generalising.

Usage
-----
    python -m eval.gradient_attribution --run-dir results/synthetic/RUN \\
        --checkpoints vqvae_best.pt vqvae_model.pt

    # closer to the real update direction
    python -m eval.gradient_attribution --run-dir results/synthetic/RUN --precondition

    # which term is collapsing the content block
    python -m eval.gradient_attribution --run-dir results/synthetic/RUN \\
        --checkpoints vqvae_model.pt --metric rank --precondition
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

# torch is imported lazily inside the measurement path so the reporting helpers below
# stay unit-testable on plain numpy (same convention as eval.run_dci_compare).

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ENCODER_MODULES = ("encoders", "encoders_v1", "content_norms", "content_projections")

# Display order for the loss terms. Reconstruction first (it is the reference force), then the
# Barlow Twins pieces grouped by pooling, so on_diag sits next to the off_diag it trades against.
_TERM_ORDER = (
    "recon",
    "vq",
    "bt_on",
    "bt_off",
    "bt_sim",
    "bt_var",
    "btgap_on",
    "btgap_off",
    "btgap_sim",
    "btgap_var",
)


def _ordered_keys(d):
    return [k for k in _TERM_ORDER if k in d] + [k for k in d if k not in _TERM_ORDER]


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


def _bt_split(hz, c_idx, out, prefix, weight, ema_factor, *, lambd, sim_coeff, std_coeff, on_coeff=1.0, **kw):
    """Split one Barlow Twins term into its four ADDITIVE pieces, each as its own loss.

    A single lumped "contrastive" gradient cannot answer which part of the objective moves a
    metric, because the four terms want different things: ``on_diag`` and ``sim`` align the
    views, ``var`` fixes per-channel scale, and only ``off_diag`` decorrelates channels. Their
    gradients are measurably opposed — cos(on_diag, off_diag) runs -0.27 to -0.83 across
    checkpoints — so attributing a metric change to "contrastive" says nothing about which of
    them caused it. Which one HELPS a given metric is empirical; see the module docstring.

    ``barlow_twins_loss`` returns ``on_diag + lambd*off_diag + sim_coeff*sim + std_coeff*var``
    as one scalar and ``on_diag`` carries no coefficient of its own, so no choice of
    coefficients isolates the other three. But the loss is a SUM and autograd is linear, so
    four calls with different coefficients recover each piece exactly by subtraction. Done
    this way rather than by reimplementing the terms here so the numbers cannot drift from
    what training computes — the same reason eval.bt_term_balance calls the shipped loss.

    ``ema_factor`` models --bt-corr-ema on the two CORRELATION terms only. With momentum m the
    EMA'd matrix is ``m*c_prev.detach() + (1-m)*c``, so only the current batch carries gradient
    and ``dc/dtheta`` is scaled by ``(1-m)`` once the bias correction has converged; ``sim`` and
    ``var`` read the raw features and are not attenuated. This reproduces the SIGNAL component
    exactly and overstates the noise component, because training's multiplier is the low-noise
    ``c_ema`` where this uses the single-batch ``c`` — read --snr for how large that gap is.
    """
    from training.losses import barlow_twins_loss

    def _call(lam, sim, std):
        return barlow_twins_loss(
            hz, estimated_content_indices=[c_idx], subsets=[[0, 1]], lambd=lam, sim_coeff=sim, std_coeff=std, **kw
        )

    # ``on_coeff`` is applied to the extracted term rather than passed into ``_call``: the three
    # subtractions below need the SAME on_diag in both operands for it to cancel, so the calls
    # stay at the default 1.0 and the weight goes on here.
    base = _call(0.0, 0.0, 0.0)  # on_diag alone: every other coefficient is zero
    out[f"{prefix}on"] = (weight * ema_factor * on_coeff) * base
    if lambd:
        out[f"{prefix}off"] = (weight * ema_factor) * (_call(lambd, 0.0, 0.0) - base)
    if sim_coeff:
        out[f"{prefix}sim"] = weight * (_call(0.0, sim_coeff, 0.0) - base)
    if std_coeff:
        out[f"{prefix}var"] = weight * (_call(0.0, 0.0, std_coeff) - base)


def _losses_at(model, batch, args_, device, grid, level):
    """Recon (+VQ) and contrastive losses for one batch, with the graph intact.

    Computed in SEPARATE forward passes rather than decomposed from one joint pass: the
    contrastive path wants ``pool_only=True`` with a patch grid, the reconstruction path
    wants the full decode. Two passes keeps each loss exactly what training computed for
    it, at the cost of one extra encoder forward.
    """
    import torch
    import torch.nn.functional as F

    from training.losses import BaselineLoss

    imgs, msks = batch["image"], batch["mask"]
    n_views = len(imgs)
    x = torch.cat(imgs, 0).to(device).float()
    m = torch.cat(msks, 0).to(device).float()
    if m.dim() == 4:
        m = m.unsqueeze(1)

    out = {}
    # Under --contrastive-only the decoder and codebook were NEVER TRAINED: that flag skips
    # the whole quantization + decoding path. Computing a reconstruction loss through those
    # random weights produces a large, meaningless gradient that looks like a real force in
    # the table. Skip them rather than report a number that invites the wrong comparison.
    _con_only = bool(getattr(args_, "contrastive_only", False))

    # --- contrastive (Barlow Twins on the content channels, patch-pooled) ---
    enc = model(x, pool_only=True, n_views=n_views, patch_grid=tuple(grid))
    feat = enc[2][level]  # (n_views*B, C, P)
    b = feat.shape[0] // n_views
    hz = torch.stack([feat[:b], feat[b : 2 * b]], dim=0)  # (2, B, C, P)

    # Same foreground rule as training (main_multimodal.py:335): keep a position if ANY
    # sample has >= thresh brain there. Dropping it here would measure a different loss.
    if getattr(args_, "patch_foreground_mask", False):
        thr = float(getattr(args_, "patch_foreground_thresh", 0.05))
        with torch.no_grad():
            frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)
            keep = (frac >= thr).any(dim=0)
            if not bool(keep.any()):
                keep = torch.ones_like(keep)
        hz = hz[..., keep]

    c_idx = list(range(int(getattr(args_, "content_dim", hz.shape[2]))))
    _scale = float(getattr(args_, "scale_contrastive_loss", 1.0))
    _sim_c = float(getattr(args_, "bt_sim_coeff", 0.0))
    _norm_terms = bool(getattr(args_, "bt_normalize_terms", False))
    _sim_norm = bool(getattr(args_, "bt_sim_normalize", False))
    # See _bt_split: (1-m) is what training's optimizer actually receives on the two
    # correlation terms once --bt-corr-ema is on. Omitting it would report those gradients
    # at up to 100x the size the run was trained with.
    _ema_f = 1.0 - float(getattr(args_, "bt_corr_ema", 0.0) or 0.0)

    _bt_split(
        hz,
        c_idx,
        out,
        "bt_",
        _scale * float(getattr(args_, "bt_patch_weight", 1.0)),
        _ema_f,
        lambd=float(getattr(args_, "bt_lambda", 1.0)),
        on_coeff=float(getattr(args_, "bt_on_coeff", 1.0)),
        sim_coeff=_sim_c,
        std_coeff=float(getattr(args_, "bt_std_coeff", 0.0)),
        center_mode=getattr(args_, "patch_center_mode", "none") or "none",
        patch_stat=getattr(args_, "bt_patch_stat", "fold") or "fold",
        sim_normalize=_sim_norm,
        normalize_terms=_norm_terms,
    )

    # The GAP companion (--bt-gap-weight), reproduced because it is the only term whose rows
    # are SUBJECTS: the patch fold's cross-covariance is Cov_subject + Cov_interaction and the
    # interaction dominates on registered volumes, so the patch terms barely constrain subject
    # identity. Effective rank is measured on GAP-pooled features, so leaving this out would
    # omit precisely the term acting on the quantity being attributed. Its lambda and variance
    # hinge carry their own coefficients (--bt-gap-lambda / --bt-gap-std-coeff), which default
    # to the patch ones; center_mode and patch_stat do not apply — there is no position axis.
    _gap_w = float(getattr(args_, "bt_gap_weight", 0.0) or 0.0)
    if _gap_w > 0:
        _gap_lam = getattr(args_, "bt_gap_lambda", None)
        _gap_std = getattr(args_, "bt_gap_std_coeff", None)
        _gap_on = getattr(args_, "bt_gap_on_coeff", None)
        _bt_split(
            hz.mean(-1),  # (2, B, C) — one row per subject
            c_idx,
            out,
            "btgap_",
            _scale * _gap_w,
            _ema_f,
            lambd=float(getattr(args_, "bt_lambda", 1.0) if _gap_lam is None else _gap_lam),
            on_coeff=float(getattr(args_, "bt_on_coeff", 1.0) if _gap_on is None else _gap_on),
            sim_coeff=_sim_c,
            std_coeff=float(getattr(args_, "bt_std_coeff", 0.0) if _gap_std is None else _gap_std),
            sim_normalize=_sim_norm,
            normalize_terms=_norm_terms,
        )

    # --- reconstruction (BaselineLoss, exactly as training scores it) ---
    if _con_only:
        return out
    dec = model(x, return_recon=True, pool_only=False, n_views=n_views, subsets=[(0, 1)])
    recon, diffs = dec[0], dec[1]
    if recon.shape[2:] != x.shape[2:]:
        recon = F.interpolate(recon, size=x.shape[2:], mode="trilinear", align_corners=False)
    rl = BaselineLoss().to(device)
    # NOTE: BaselineLoss ADDS the quantization losses into its own return value
    # (losses.py:1611), and main_multimodal then adds sum(diffs)*vq_commitment_weight on
    # top. So VQ is counted twice in training, at two different weights. Reproduced here
    # rather than corrected, because the point is to attribute what actually ran.
    out["recon"] = rl({"reconstruction": [recon], "quantization_losses": diffs, "mask": m}, x) * float(
        getattr(args_, "scale_recon_loss", 1.0)
    )
    out["vq"] = sum(diffs) * float(getattr(args_, "vq_commitment_weight", 0.25))
    return out


def collect_gradients(model, dataset, args_, device, grid, level, n_batches, batch_size, precond=None):
    """Per-loss gradients w.r.t. the encoder parameters, averaged over batches."""
    import torch
    from torch.utils.data import DataLoader, Subset

    named = _encoder_params(model)
    params = [p for _n, p in named]
    for p in params:
        p.requires_grad_(True)

    loader = DataLoader(
        Subset(dataset, range(min(n_batches * batch_size, len(dataset)))), batch_size=batch_size, shuffle=False
    )
    acc, per_batch, nb = {}, {}, 0
    for batch in loader:
        losses = _losses_at(model, batch, args_, device, grid, level)
        for key, loss in losses.items():
            if not torch.is_tensor(loss) or not loss.requires_grad:
                continue
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            g = [torch.zeros_like(p) if gi is None else gi for gi, p in zip(g, params)]
            flat = _flat(g)
            acc[key] = flat if key not in acc else acc[key] + flat
            per_batch.setdefault(key, []).append(flat.norm().item())
        nb += 1
        del losses
    if nb == 0:
        raise RuntimeError("No batches were processed — is the dataset empty?")
    grads = {k: (v / nb) for k, v in acc.items()}

    if precond is not None:
        # AdamW's realised direction is g / (sqrt(v) + eps), not g. Applying the
        # checkpoint's second-moment estimate makes the finite-difference step below a much
        # better model of the update training actually took.
        for k in grads:
            grads[k] = grads[k] / (precond.sqrt() + 1e-8)
    return named, params, grads, per_batch


def snr_decomposition(G):
    """Split a stack of per-batch gradients into expected-gradient and per-batch noise.

    ``G`` is ``(B, D)``: one flattened gradient per batch. For independent batches

        E||mean(G)||^2 = ||E[g]||^2 + tr(Cov)/B

    so the naive norm of the averaged gradient OVERSTATES the true expected gradient by
    exactly the noise term, and does so more the fewer batches you average. Subtracting it
    gives an unbiased estimate of ||E[g]||^2, which can come out NEGATIVE when the true
    expected gradient is zero — that is the signature, not a bug, so it is reported rather
    than clipped.

    Why this decides the question: a loss whose target is attainable converges to a genuine
    descent direction and keeps a non-zero ||E[g]||. A loss chasing an unattainable target
    (an off-diagonal that cannot go below its d(d-1)/rows sampling floor) ends up with
    E[g] ~ 0 and large per-batch variance — it never stops moving the weights, but the
    motion is a random walk rather than optimisation.
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


def collect_per_batch_gradients(model, dataset, args_, device, grid, level, n_batches, batch_size):
    """One flattened encoder gradient per batch, for the SNR decomposition."""
    import torch
    from torch.utils.data import DataLoader, Subset

    named = _encoder_params(model)
    params = [p for _n, p in named]
    for p in params:
        p.requires_grad_(True)

    loader = DataLoader(
        Subset(dataset, range(min(n_batches * batch_size, len(dataset)))), batch_size=batch_size, shuffle=False
    )
    out = {}
    for batch in loader:
        losses = _losses_at(model, batch, args_, device, grid, level)
        for key, loss in losses.items():
            if not torch.is_tensor(loss) or not loss.requires_grad:
                continue
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            g = [torch.zeros_like(p) if gi is None else gi for gi, p in zip(g, params)]
            out.setdefault(key, []).append(_flat(g).float().cpu().numpy())
        del losses
    return {k: np.stack(v) for k, v in out.items()}


def _mcc_now(model, dataset, device, grid, level, batch_size, gt_cache, seeds, n_splits):
    from eval.identifiability_metrics import block_mcc
    from eval.patch_mcc_decay import extract_patch_block

    content, gt, _cov = extract_patch_block(model, dataset, device, grid, level, batch_size, 0)
    if gt_cache is None:
        gt_cache = gt
    x = content.reshape(content.shape[0], -1)
    return block_mcc(x, gt_cache, seeds=seeds, n_splits=n_splits)["mean"], gt_cache


def _rank_now(model, dataset, device, grid, level, batch_size, gt_cache, seeds, n_splits):
    """Effective rank of the GAP-pooled content block — the same number run_dci_compare reports.

    Deliberately measured at GAP, not at patch: ``score_reprs`` scores ``content_rank`` at the
    gap pooling so it reads as "effective channels", and the patch fold mixes subjects with
    positions, whose variance is ~200x larger on registered volumes and would swamp the
    across-subject spectrum this is asking about.

    Cheaper than the block-MCC readout it parallels — no probe, no cross-validation, no
    Hungarian match, just one SVD of an (N, C) matrix — so the eta sweep costs little beyond
    the forward passes. It is also DETERMINISTIC given the features, so unlike block-MCC there
    is no per-seed band to clear; the floor on a real reading is set by sampling N subjects,
    which the eta sweep holds fixed by reusing one frozen test set.

    ``gt_cache``/``seeds``/``n_splits`` are unused; kept so this is interchangeable with
    ``_mcc_now`` in the sweep.
    """
    from eval.patch_mcc_decay import extract_patch_block
    from eval.run_dci_compare import _effective_rank

    content, gt, _cov = extract_patch_block(model, dataset, device, grid, level, batch_size, 0)
    # extract_patch_block returns (N, P, C), position-major — so GAP is a mean over axis 1.
    return _effective_rank(content.mean(axis=1)), gt if gt_cache is None else gt_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=["vqvae_best.pt", "vqvae_model.pt"])
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, nargs=3, default=None)
    ap.add_argument(
        "--grad-batches",
        type=int,
        default=16,
        help="Batches to average each gradient over. The contrastive gradient's batch-to-batch "
        "sd has been measured at ~69%% of its mean, so a handful of batches does not resolve "
        "its expectation from zero — which is exactly the quantity convergence claims is small.",
    )
    ap.add_argument("--grad-batch-size", type=int, default=8)
    ap.add_argument("--mcc-samples", type=int, default=600, help="Volumes for each block-MCC re-measurement")
    ap.add_argument("--mcc-batch", type=int, default=16)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.2, 0.8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--precondition", action="store_true", help="Divide by sqrt(v) from the optimizer state")
    ap.add_argument(
        "--snr",
        action="store_true",
        help="Run the gradient signal-to-noise decomposition instead of the dMCC sweep. "
        "Decides whether a loss has a genuine expected descent direction or is only chasing "
        "sampling noise. The latter never stops moving the weights but is a random walk, "
        "which the matched-random control in the dMCC table subtracts out by construction "
        "-- so a loss can drive a decay and still show zero excess there.",
    )
    ap.add_argument(
        "--random-controls",
        type=int,
        default=1,
        help="Matched-random directions per loss (0 disables). Without at least one the dMCC "
        "table cannot separate the loss from generic perturbation sensitivity, so this "
        "defaults ON despite roughly doubling runtime.",
    )
    ap.add_argument("--causal", choices=("match", "iid"), default="match")
    ap.add_argument(
        "--metric",
        choices=("mcc", "rank"),
        default="mcc",
        help="What the finite-difference step re-measures. 'mcc' (default) is block-MCC, unchanged. "
        "'rank' is the effective rank of the GAP-pooled content block — the same content_rank "
        "run_dci_compare prints — for attributing a rank change to a specific term. Read the signs "
        "off the table rather than predicting them from each term's global optimum: measured on "
        "this project, on_diag BUILT rank and off_diag destroyed it at one checkpoint, which is the "
        "opposite of what that argument gives. See the module docstring.",
    )
    cli = ap.parse_args()

    import os

    import torch

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in cli.checkpoints:
        path = os.path.join(cli.run_dir, name)
        if not os.path.exists(path):
            logger.warning("missing checkpoint %s — skipping", path)
            continue
        print(f"\n{'=' * 78}\nGRADIENT ATTRIBUTION — {name}\n{'=' * 78}")
        model, args_, _ = load_model_from_run_dir(cli.run_dir, path, device)
        model.to(device).train()
        grid = cli.grid or list(getattr(args_, "patch_grid", [8, 8, 8]))
        dataset = build_synthetic_test_set(args_, cli.mcc_samples, cache=False, causal=cli.causal == "match")

        precond = None
        if cli.precondition:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            osd = ck.get("optimizer_state_dict")
            if not osd:
                logger.warning("--precondition: no optimizer_state_dict in the checkpoint; using raw gradients")
            else:
                logger.info("--precondition: using exp_avg_sq from the checkpoint's AdamW state")
                # Rebuilt in the same order _encoder_params returns, which is the order the
                # flat vectors below use; a mismatch here would silently scale the wrong
                # coordinates, so it is length-checked against the flat gradient.
                named_tmp = _encoder_params(model)
                want = sum(p.numel() for _n, p in named_tmp)
                vs = [s["exp_avg_sq"].reshape(-1) for s in osd["state"].values() if "exp_avg_sq" in s]
                cat = torch.cat(vs) if vs else torch.empty(0)
                if cat.numel() != want:
                    logger.warning(
                        "--precondition: optimizer state covers %d params but the encoder has %d; "
                        "the parameter ordering does not line up, so falling back to raw gradients.",
                        cat.numel(),
                        want,
                    )
                else:
                    precond = cat.to(device)

        if cli.snr:
            per_b = collect_per_batch_gradients(
                model, dataset, args_, device, grid, cli.level, cli.grad_batches, cli.grad_batch_size
            )
            print(f"\n  gradient signal/noise over {cli.grad_batches} batches of {cli.grad_batch_size}")
            print(
                f"  {'loss':<14}{'per-batch':>12}{'mean of B':>12}{'EXPECTED':>12}"
                f"{'noise':>12}{'signal share':>14}{'batches->SNR1':>15}"
            )
            print("  " + "-" * 91)
            for k in _ordered_keys(per_b):
                d = snr_decomposition(per_b[k])
                if "naive_norm" not in d:
                    continue
                share = (d["signal_norm"] / d["per_batch_norm"]) if d["per_batch_norm"] > 0 else float("nan")
                need = d["batches_for_snr1"]
                need_s = "inf" if not np.isfinite(need) else f"{need:.0f}"
                print(
                    f"  {k:<14}{d['per_batch_norm']:>12.4e}{d['naive_norm']:>12.4e}"
                    f"{d['signal_norm']:>12.4e}{d['noise_norm']:>12.4e}{share:>14.3f}{need_s:>15}"
                )
            print("\n    EXPECTED is bias-corrected: ||mean||^2 - tr(Cov)/B. At or below zero means the")
            print("    loss has NO detectable descent direction left, so every step it takes is")
            print("    sampling noise and it random-walks the encoder indefinitely instead of")
            print("    converging. A loss with an attainable target keeps EXPECTED clearly positive.")
            del model, per_b
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        named, params, grads, per_batch = collect_gradients(
            model, dataset, args_, device, grid, cli.level, cli.grad_batches, cli.grad_batch_size, precond
        )
        n_enc = sum(p.numel() for p in params)
        print(f"\n  encoder parameters: {n_enc:,} across {len(named)} tensors")
        print(f"  gradients averaged over {cli.grad_batches} batches of {cli.grad_batch_size}")

        keys = _ordered_keys(grads)
        print(f"\n  {'loss':<14}{'||grad||':>14}{'batch-to-batch sd':>20}")
        print("  " + "-" * 48)
        for k in keys:
            sd = float(np.std(per_batch[k])) if len(per_batch.get(k, [])) > 1 else float("nan")
            print(f"  {k:<14}{grads[k].norm().item():>14.4e}{sd:>20.4e}")

        np_g = {k: grads[k].detach().float().cpu().numpy() for k in keys}
        # Printed as a matrix rather than a list of pairs: splitting Barlow Twins into its four
        # terms takes the pair count from 3 to 45, which is unreadable one per line.
        print(f"\n  pairwise cosine (negative = the two losses pull the encoder apart):")
        _short = {k: k.replace("btgap_", "g_").replace("bt_", "") for k in keys}
        print(f"    {'':<12}" + "".join(f"{_short[k]:>9}" for k in keys))
        for a in keys:
            row = "".join("        ." if a == b else f"{cosine(np_g[a], np_g[b]):>+9.3f}" for b in keys)
            print(f"    {a:<12}{row}")

        _measure = _rank_now if cli.metric == "rank" else _mcc_now
        _mname = "content eff. rank (GAP)" if cli.metric == "rank" else "block-MCC"
        _dname = "d(eff. rank)" if cli.metric == "rank" else "dMCC"
        _degrades = "collapses the representation" if cli.metric == "rank" else "degrades identifiability"

        base, gt_cache = _measure(
            model, dataset, device, grid, cli.level, cli.mcc_batch, None, tuple(cli.seeds), cli.n_splits
        )
        print(f"\n  {_mname} at this checkpoint: {base:.4f}")
        print(f"\n  {_dname} after one step of size eta along -g/||g||  (negative = {_degrades})")
        backup = [p.detach().clone() for p in params]

        def _sweep(direction):
            out = []
            for eta in cli.etas:
                off = 0
                with torch.no_grad():
                    for p, b0 in zip(params, backup):
                        n = p.numel()
                        p.copy_(b0 - eta * direction[off : off + n].view_as(p))
                        off += n
                val, _ = _measure(
                    model, dataset, device, grid, cli.level, cli.mcc_batch, gt_cache, tuple(cli.seeds), cli.n_splits
                )
                out.append(val - base)
            with torch.no_grad():
                for p, b0 in zip(params, backup):
                    p.copy_(b0)
            return out

        def _matched_random(g, seed):
            """A random direction with the SAME per-tensor energy profile as ``g``.

            The control this table is meaningless without. Every direction is unit-norm, so
            each row perturbs the weights by the same amount — and block-MCC degrades under
            almost ANY perturbation of sufficient size. Without a baseline you cannot tell
            "this loss degrades identifiability" from "0.8 of parameter noise degrades
            identifiability". Matching per-tensor norms rather than drawing a flat Gaussian
            keeps the comparison honest: the control spreads its energy across layers the
            same way the real gradient does, so what is left is the DIRECTION within that
            profile, which is the only part attributable to the loss.
            """
            gen = torch.Generator(device="cpu").manual_seed(seed)
            parts, off = [], 0
            for p in params:
                n = p.numel()
                blk = g[off : off + n]
                r = torch.randn(n, generator=gen).to(g.device)
                r = r / max(r.norm().item(), 1e-12) * blk.norm()
                parts.append(r)
                off += n
            r = torch.cat(parts)
            return r / max(r.norm().item(), 1e-12)

        print(f"  {'direction':<20}" + "".join(f"{'eta=' + str(e):>12}" for e in cli.etas) + f"{'linearity R2':>15}")
        print("  " + "-" * (20 + 12 * len(cli.etas) + 15))
        for k in keys:
            deltas = _sweep(grads[k] / max(grads[k].norm().item(), 1e-12))
            r2 = linearity_check(cli.etas, deltas)
            flag = "   <- non-linear" if np.isfinite(r2) and r2 < 0.9 else ""
            print(f"  {k:<20}" + "".join(f"{d:>+12.4f}" for d in deltas) + f"{r2:>15.3f}{flag}")
            if cli.random_controls > 0:
                ctrl = np.mean(
                    [_sweep(_matched_random(grads[k], 1000 + 7 * i)) for i in range(cli.random_controls)], axis=0
                )
                print(f"  {'  random (matched)':<20}" + "".join(f"{d:>+12.4f}" for d in ctrl) + "        [control]")
                excess = [d - c for d, c in zip(deltas, ctrl)]
                print(f"  {'  EXCESS over random':<20}" + "".join(f"{d:>+12.4f}" for d in excess) + "   <- attribution")

        print("\n  Reading it: the EXCESS row is the attribution — how much worse this loss's")
        print("  direction is than a random direction with the same per-layer energy. The raw")
        print(f"  {_dname} row conflates that with generic perturbation sensitivity, which is large.")
        if cli.metric == "rank":
            print("  Effective rank is deterministic given the features, so there is no per-seed band")
            print("  to clear — but it IS sampled: re-run with a different --mcc-samples to see how")
            print("  much of a small excess is the finite test set. A term with a NEGATIVE excess is")
            print("  collapsing the representation beyond what its step size alone would do.")
            print("  These signs are LOCAL to this checkpoint and have been observed to flip between")
            print("  checkpoints of one run — replicate at 2-3 before acting on one. Do not predict")
            print("  them from each term's global optimum: that argument says only off_diag can build")
            print("  rank, and it came out backwards here at R^2 = 1.000.")
        else:
            print("  Compare the excess against block-MCC's per-seed sd (~0.001 at N=1500); inside")
            print("  that band there is no attribution, whatever the raw row says.")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
