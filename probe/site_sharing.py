#!/usr/bin/env python
"""Is the decoding mechanism the *same* at every latent position? — the A1/A4 audit.

The field-valued SCM assumes two things about position that the identifiability results
in its Section 5 rest on entirely:

  * **A1 (site-shared mechanism)** — ``f_j`` does not depend on ``u``.  This is what ties
    the meaning of channel ``j`` at one position to its meaning at another, and it is what
    lifts a per-position identifiability result to a statement about the *field*: one
    global element of the residual gauge group instead of an independent one at each ``u``.
  * **A4 (site-shared decoding)** — ``g_k`` is applied by a weight-shared architecture, so
    the map from latent values to block output does not depend on ``u``.

Both are properties of a *trained* decoder, not of its architecture, and the model's own
Limitations section says so: "the homogeneity of decoder-Jacobian spectra across Lambda is
a direct test and bounds the region over which the results of Section 5 may be claimed."
This script is that test, plus the binding measurement it makes possible.

Two numbers, one Jacobian
-------------------------
At latent site ``u`` we take the block Jacobian

    ``J_u = d x_k|B(u) / d z(u)``      shape ``(|B(u)| * C_out, d)``

by exact JVPs (one basis tangent per latent channel), with the style code held fixed --
so this is the derivative with respect to ``z`` alone, matching ``x_k = g_k(z, s_k, gamma)``.

**1. Spectral homogeneity.**  Split the log singular-value spectrum of ``J_u`` into

    ``scale = mean(log s)``           overall magnitude
    ``shape = log s - scale``         conditioning / anisotropy, invariant to ``J -> cJ``

*Scale* varies with local anatomy and is expected to: a background block decodes to almost
nothing.  **Shape is the site-sharing test** -- under A1/A4 the mechanism's conditioning is
a property of the weights, not of where they are applied.  Reporting the two separately is
what keeps this from being a brightness map.

**2. Binding.**  Match the columns of ``J_u`` to the columns of a reference site's ``J_u0``
by Hungarian assignment on ``|cos|``.  Under A1 the answer is the *identity* permutation --
channel ``j`` is produced by the same filters everywhere, so it must mean the same thing
everywhere.  The fraction of channels that match themselves is the binding score.  This is
the direct empirical counterpart of the "consistent cross-position binding" claim, and it
is exactly what a probe on flattened patch features cannot see: flattening lets an
unconstrained readout absorb a position-dependent permutation before scoring.

Why this is not trivially 1.0
-----------------------------
The decoder's *weights* are shared by construction.  What varies across ``u`` is the
*state*: nonlinearities, and normalization layers whose statistics span the volume.  So the
measurement reads the **effective, state-dependent** mechanism.  Three arms:

    ``full``        the decoder as trained; the headline.
    ``linear``      every nonlinearity and normalization replaced by identity, leaving a
                    pure convolution stack.  That is exactly translation-equivariant, so
                    binding **must** come back ~1.0.  This is the positive control, and the
                    only one that is exact by construction.
    ``frozen_norm`` normalization statistics pinned at the operating point.  **Diagnostic,
                    not a control** — freezing removes the spatial-statistic path but leaves
                    every nonlinearity, and ReLU gating alone makes ``J_u`` position-
                    dependent, so this arm is not expected to reach 1.0.  Its gap from
                    ``full`` is the normalization's cost, as ``probe.jacobian_spread``
                    reports it for rho.  It is a silent no-op on a decoder built with
                    ``norm_type='layer'`` (ChannelLayerNorm3d is not an ``nn.GroupNorm``);
                    that case is detected and reported rather than left to look like a
                    control that passed.

What must hold before the numbers mean anything
-----------------------------------------------
Three conditions gate interpretation, and the verdict refuses to report a headline until
all three pass — announcing "the labelling drifts with position" off an instrument that
cannot reproduce itself would be the worst failure this script could have.

  1. **The window caught the response.**  ``energy_profile`` gives the share of
     ``|dx/dz(u)|^2`` inside the window at each dilation.  A low value at dilation 0 means
     ``J_u`` is a peripheral tail rather than the mechanism — which is itself an A4
     centre-dominance failure worth reporting — and ``--block-dilation`` widens it.
  2. **The assignment was decisive.**  ``d`` columns spanning an effectively much smaller
     space make every permutation score alike, so the identity fraction reads near chance
     whether or not anything drifted.  ``effective_rank`` and the assignment ``margin``
     detect this; ``identity_frac_confident`` restricts to decisive columns.
  3. **The instrument reproduces itself.**  ``same_site_null`` matches one site to itself
     across subjects.  It is the *ceiling* for the across-site score, which cannot be read
     past it.

Self-calibration
----------------
Neither statistic is compared against an invented threshold.  The same site is measured
across several subjects, which gives a *within-site* spread — the measurement's own noise
floor — and the across-site spread is reported as a multiple of it, matched for the sqrt(n)
the subject-mean already bought.  A homogeneity ratio of ~1 means site-sharing holds to
measurement precision; the ratio is the finding, not the raw deviation.  ``true_site_sd``
reports the same thing with the noise subtracted, in nats.

Scope
-----
This audit gates the identifiability metrics; it does not replace them.  ``rho`` and the A4
centre-dominance ratio come from ``probe.jacobian_spread`` and should be read first: if the
guard scan there fails, per-position claims are void regardless of what this script reports,
and the validity map below describes a decoder whose blocks already overlap.

Usage:
    python -m probe.site_sharing --run-dir results/synthetic/<run> --checkpoint <ckpt>
    python -m probe.site_sharing --run-dir <run> --sites strata --max-sites 64 --plot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment

# torch is imported inside the functions that need it, so the statistics below stay
# importable — and therefore unit-testable against synthetic Jacobians — in an environment
# with no torch build.  See tests/test_site_sharing.py.

logger = logging.getLogger(__name__)

# "linear" is the positive control and the only one that is exact by construction; see
# make_linear_decoder.  "frozen_norm" is diagnostic — the cost of the normalization — and
# is NOT a control: pinning the statistics leaves every nonlinearity in place, so its
# Jacobian still varies with position.
ARMS = ("full", "frozen_norm", "linear")


def make_linear_decoder(decoder):
    """A copy of ``decoder`` with every nonlinearity and normalization replaced by identity.

    What remains is a pure (transposed-)convolution stack, which is exactly
    translation-equivariant, so ``J_u`` is the same matrix at every interior site and both
    statistics have a known answer: binding 1.0, homogeneity ratio ~1.  Anything else is a
    defect in the measurement rather than a property of the model, which is what a positive
    control is for.

    This replaces an earlier and wrong assumption that pinning normalization statistics was
    sufficient: freezing removes the *spatial-statistic* path only, and ReLU gating alone
    makes the effective linear map position-dependent.
    """
    import copy

    import torch.nn as nn

    import probe.receptive_field as rfmod

    kill = (
        nn.ReLU,
        nn.LeakyReLU,
        nn.GELU,
        nn.SiLU,
        nn.ELU,
        nn.Tanh,
        nn.Sigmoid,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm3d,
        nn.BatchNorm3d,
    )
    lin = copy.deepcopy(decoder)
    # Deepest-first, so replacing a wrapper (ChannelLayerNorm3d) does not invalidate the
    # path to a child that was also queued for replacement.
    for name, m in sorted(lin.named_modules(), key=lambda kv: -kv[0].count(".")):
        if name and isinstance(m, kill):
            rfmod._set_submodule(lin, name, nn.Identity())
    lin.eval()
    return lin


# --------------------------------------------------------------------------------------
# spectrum: scale / shape decomposition   (pure numpy — unit-testable without a model)
# --------------------------------------------------------------------------------------


def energy_rank(J: np.ndarray, frac: float = 0.99) -> int:
    """How many singular values it takes to hold ``frac`` of ``J``'s squared energy."""
    s = np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)
    p = s**2
    tot = p.sum()
    if tot <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(p) / tot, frac) + 1)


def scale_shape(J: np.ndarray, keep: int | None = None, eps: float = 1e-30) -> tuple[float, np.ndarray]:
    """Split ``J``'s log singular spectrum into overall scale and zero-mean shape.

    ``shape`` is invariant to ``J -> cJ`` for any ``c > 0``, which is the point: local
    anatomy changes how *strongly* a block decodes, and only a change in the *conditioning*
    is evidence against a site-shared mechanism.

    ``keep`` truncates to the leading ``keep`` singular values, and on a rank-deficient
    ``J`` it is what makes the statistic mean anything.  The trailing values of a rank-r
    map with r << d sit on the float32 noise floor, log pushes their *relative* wobble to
    several nats, and an untruncated shape vector then measures numerical noise rather
    than the mechanism — inflating the deviation without any site-sharing failure.
    """
    s = np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)
    if keep is not None:
        s = s[: max(int(keep), 1)]
    logs = np.log(np.maximum(s, eps))
    scale = float(logs.mean())
    return scale, logs - scale


def shape_deviation(shapes: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """RMS log-deviation of each row of ``shapes`` from ``reference`` (default: the median).

    Units are nats per singular value, so the number is comparable across models and grids.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    ref = np.median(shapes, axis=0) if reference is None else np.asarray(reference, dtype=np.float64)
    return np.sqrt(((shapes - ref) ** 2).mean(axis=1))


# --------------------------------------------------------------------------------------
# binding: cross-position channel matching   (pure numpy)
# --------------------------------------------------------------------------------------


def live_channels(col_norms: np.ndarray, dead_thresh: float = 1e-3) -> np.ndarray:
    """Boolean mask of channels that carry signal, from ``(n_meas, d)`` column norms.

    A channel is live if its *median* relative column norm clears ``dead_thresh``.  Dead
    channels have an arbitrary direction, so leaving them in would let assignment noise
    dominate the binding score — and a VQ-VAE with a content mask reliably has some.
    """
    col_norms = np.asarray(col_norms, dtype=np.float64)
    peak = col_norms.max(axis=1, keepdims=True)
    rel = col_norms / np.maximum(peak, 1e-300)
    return np.median(rel, axis=0) >= dead_thresh


def match_columns(Ja: np.ndarray, Jb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian match ``Ja``'s columns to ``Jb``'s on ``|cos|``.

    Returns ``(assignment, matched_cos, margin)`` where ``assignment[j]`` is the column of
    ``Jb`` that column ``j`` of ``Ja`` was matched to.  Absolute cosine, because the theory's
    residual gauge admits a coordinate-wise bijection, and a sign flip is one.

    ``margin[j]`` is how much the chosen match beats the best alternative for that column.
    Without it a low identity fraction is ambiguous: columns spanning a space much smaller
    than ``d`` make every permutation score alike, so the assignment is arbitrary and reads
    near chance whether or not the labelling actually drifted.  A near-zero margin means the
    identity fraction is not measuring anything.
    """
    A = _unit_columns(Ja)
    B = _unit_columns(Jb)
    sim = np.abs(A.T @ B)
    row, col = linear_sum_assignment(-sim)
    assignment = np.empty(A.shape[1], dtype=int)
    assignment[row] = col

    chosen = sim[np.arange(sim.shape[0]), assignment]
    masked = sim.copy()
    masked[np.arange(sim.shape[0]), assignment] = -np.inf
    return assignment, chosen, chosen - masked.max(axis=1)


def effective_rank(J: np.ndarray) -> float:
    """Entropy effective rank of ``J``'s singular spectrum — the matching's real difficulty.

    ``d`` columns living in an effectively ``r``-dimensional space cannot be told apart by
    any assignment when ``r << d``; this is the number to read next to a low margin.
    """
    s = np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)
    p = s**2
    tot = p.sum()
    if tot <= 0:
        return 0.0
    p = p / tot
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def _unit_columns(J: np.ndarray) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    n = np.linalg.norm(J, axis=0, keepdims=True)
    return J / np.maximum(n, 1e-300)


def binding_stats(assignment: np.ndarray, matched_cos: np.ndarray, margin: np.ndarray, margin_tol: float) -> dict:
    """Identity agreement for one (site, subject) match against the reference site."""
    d = len(assignment)
    identity = assignment == np.arange(d)
    confident = margin >= margin_tol
    return {
        "identity_frac": float(identity.mean()),
        "matched_cos": float(np.mean(matched_cos)),
        "matched_cos_identity": float(np.mean(matched_cos[identity])) if identity.any() else float("nan"),
        "margin": float(np.median(margin)),
        "confident_frac": float(confident.mean()),
        # Identity among only the columns whose match was decisive.  When `confident_frac`
        # is low this is the only binding number worth reading.
        "identity_frac_confident": float(identity[confident].mean()) if confident.any() else float("nan"),
        "chance": 1.0 / max(d, 1),
        "identity": identity,
    }


# --------------------------------------------------------------------------------------
# Jacobian extraction
# --------------------------------------------------------------------------------------


def block_slices(site, latent_shape, out_shape, dilation: int = 0) -> tuple[slice, slice, slice]:
    """The output block ``B(u)`` for latent site ``u``, from the decoder's stride.

    ``dilation`` widens the window by that many *latent cells* on each side.  A window
    tighter than the decoder's reach captures only the periphery of the response, and the
    Jacobian columns it yields are then dominated by whatever weak, position-idiosyncratic
    signal lands there — which reads as a binding failure that is really a windowing one.
    Read ``energy_profile`` before choosing a value.
    """
    sl = []
    for a in range(3):
        stride = out_shape[a] / latent_shape[a]
        pad = dilation * stride
        # floor on both edges, so that at dilation 0 the blocks tile Omega exactly even when
        # the stride is fractional (ceil on the upper edge overlaps adjacent blocks by one).
        lo = int(np.floor(site[a] * stride - pad))
        hi = max(int(np.floor((site[a] + 1) * stride + pad)), lo + 1)
        sl.append(slice(max(lo, 0), min(hi, out_shape[a])))
    return tuple(sl)


def block_jacobian(fn, z, site, block, n_channels: int, profile_dilations: dict, chunk: int = 16):
    """``d x|B(u) / d z(u)`` by exact JVPs, plus the response energy captured at each window.

    One basis tangent per latent channel, batched ``chunk`` at a time along the batch axis
    (every normalization here is per sample, so batch elements stay independent).

    ``profile_dilations`` maps a dilation to its slice tuple; the returned profile gives the
    share of ``|d x / d z(u)|^2`` inside each.  It is *not* the A4 ratio — A4 bounds what
    leaks *into* ``B(u)`` from other sites — but it is the same phenomenon from the other
    side, and a low value at dilation 0 means the block is far smaller than the decoder's
    reach, so every column below is a peripheral tail rather than the mechanism.
    """
    import torch

    from probe.jacobian_spread import _jvp

    cols, total = [], 0.0
    energy = {d: 0.0 for d in profile_dilations}
    rows = {d: [] for d in profile_dilations}
    for start in range(0, n_channels, chunk):
        stop = min(start + chunk, n_channels)
        n = stop - start
        zb = z.expand(n, *z.shape[1:]).contiguous()
        tangent = torch.zeros_like(zb)
        for i in range(n):
            tangent[i, start + i, site[0], site[1], site[2]] = 1.0
        _, jv = _jvp(fn, zb, tangent)
        jv = jv.detach()
        total += float((jv**2).sum())
        blk = jv[:, :, block[0], block[1], block[2]]
        cols.append(blk.reshape(n, -1).cpu().numpy().T)
        # The whole response is already in hand before slicing, so both curves cost nothing
        # beyond a few reductions. The energy curve says whether the window was wide enough;
        # the RANK curve says whether widening it would even help — a local map that stays
        # rank-deficient as the window grows is rank-deficient as a matter of architecture,
        # and no window makes z(u) recoverable from the response.
        for d, sl in profile_dilations.items():
            part = jv[:, :, sl[0], sl[1], sl[2]]
            energy[d] += float((part**2).sum())
            rows[d].append(part.reshape(n, -1).cpu().numpy())
    J = np.concatenate(cols, axis=1)
    prof = {d: (e / total if total > 0 else float("nan")) for d, e in energy.items()}
    rank_prof = {}
    for d, blocks in rows.items():
        Jt = np.concatenate(blocks, axis=0).astype(np.float64)  # (channels, window)
        rank_prof[d] = effective_rank_from_gram(Jt @ Jt.T)
    return J, prof, rank_prof


def effective_rank_from_gram(G: np.ndarray) -> float:
    """Entropy effective rank from a ``d x d`` Gram, avoiding an SVD of the tall Jacobian."""
    w = np.linalg.eigvalsh(np.asarray(G, dtype=np.float64))
    w = np.clip(w, 0.0, None)
    tot = w.sum()
    if tot <= 0:
        return 0.0
    p = w / tot
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def select_sites(brain_frac, latent_shape, mode: str, max_sites: int, thresh: float, gen, tissue):
    """Latent sites to measure: anatomical strata, brain foreground, or the whole grid."""
    import torch

    from probe.jacobian_spread import stratify_sites, tissue_fractions

    if mode == "strata":
        if tissue is None:
            raise SystemExit("--sites strata needs synthetic tissue labels; use --sites foreground.")
        chosen = stratify_sites(tissue_fractions(tissue, latent_shape), latent_shape, max_sites // 4 + 1, gen)
        # The strata are not disjoint — periventricular and cortical_ribbon can both claim a
        # site with csf > 0.02 and gm > 0.25 — and a duplicated site would be measured twice
        # and counted twice in every median below.  First stratum to claim it wins.
        sites, strata = [], {}
        for name, lst in chosen.items():
            for s in lst:
                if s not in strata:
                    strata[s] = name
                    sites.append(s)
        return sites, strata

    if mode == "all":
        keep = torch.ones(latent_shape, dtype=torch.bool)
    else:
        keep = brain_frac >= thresh
        if not bool(keep.any()):
            raise SystemExit(f"No latent site has brain fraction >= {thresh}; lower --fg-thresh.")
    idx = torch.nonzero(keep, as_tuple=False)
    if idx.shape[0] > max_sites:
        idx = idx[torch.randperm(idx.shape[0], generator=gen)[:max_sites]]
    return [tuple(int(v) for v in row) for row in idx], {}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def measure_arm(fn, z, sites, latent_shape, out_shape, n_channels, chunk, dilation, profile_dilations):
    """Every site's block Jacobian for one subject, one arm."""
    out = {}
    for site in sites:
        blk = block_slices(site, latent_shape, out_shape, dilation)
        prof_slices = {d: block_slices(site, latent_shape, out_shape, d) for d in profile_dilations}
        J, prof, rank_prof = block_jacobian(fn, z, site, blk, n_channels, prof_slices, chunk=chunk)
        out[site] = {"J": J, "energy_profile": prof, "rank_profile": rank_prof}
    return out


def reduce_arm(
    per_subject: list[dict],
    sites,
    dead_thresh: float,
    n_refs: int,
    margin_tol: float = 0.05,
    spectrum_energy: float = 0.99,
) -> dict:
    """Turn raw Jacobians into the two headline statistics, with their own noise floors."""
    n_subj = len(per_subject)
    if n_subj < 2:
        raise SystemExit(
            "--n-subjects must be >= 2: both statistics are reported against a noise floor measured across subjects."
        )

    norms = np.stack([[np.linalg.norm(per_subject[s][u]["J"], axis=0) for u in sites] for s in range(n_subj)])
    live = live_channels(norms.reshape(-1, norms.shape[-1]), dead_thresh)
    n_live = int(live.sum())
    if n_live < 2:
        raise SystemExit(f"Only {n_live} live channel(s) at --dead-thresh {dead_thresh}; nothing to match.")

    # A window holding fewer rows than there are live channels cannot carry d independent
    # directions, so the spectrum is short and the matching is degenerate before it starts.
    n_rows = per_subject[0][sites[0]]["J"].shape[0]
    if n_rows < n_live:
        raise SystemExit(
            f"The window holds {n_rows} values but there are {n_live} live channels, so J_u has rank "
            f"<= {n_rows} and neither statistic is defined. Widen it with --block-dilation."
        )

    # --- spectra ------------------------------------------------------------------
    # Truncate to the leading directions that actually carry the response, and use ONE
    # length everywhere so the shape vectors are comparable. Past that point the singular
    # values sit on the float32 noise floor, where log turns a numerical wobble into whole
    # nats — which is a deviation with no mechanism behind it.
    keep = min(energy_rank(per_subject[si][u]["J"][:, live], spectrum_energy) for si in range(n_subj) for u in sites)
    keep = max(int(keep), 1)

    scales = np.zeros((n_subj, len(sites)))
    shapes = np.zeros((n_subj, len(sites), keep))
    for si in range(n_subj):
        for ui, u in enumerate(sites):
            scales[si, ui], shapes[si, ui] = scale_shape(per_subject[si][u]["J"][:, live], keep=keep)

    # Across-site spread of the subject-mean shape, against the within-site spread across
    # subjects.  The ratio is the finding: the raw deviation has no scale of its own.
    #
    # The two must be put on the same footing before dividing.  ``dev_across`` is measured
    # on subject-*means*, whose noise is sigma/sqrt(n_subj); ``dev_within`` estimates the
    # single-subject sigma, and estimates it low because it centres on the same n_subj
    # samples it spreads around (hence the Bessel factor).  Matching them makes ratio ~= 1
    # the null rather than an arbitrary small number.
    mean_shape = shapes.mean(axis=0)
    dev_across = shape_deviation(mean_shape)
    dev_within = np.array(
        [shape_deviation(shapes[:, ui, :], shapes[:, ui, :].mean(axis=0)).mean() for ui in range(len(sites))]
    )
    sigma = float(np.median(dev_within)) * np.sqrt(n_subj / (n_subj - 1))
    floor = sigma / np.sqrt(n_subj)
    across = float(np.median(dev_across))
    if floor > 0:
        ratio = across / floor
    else:
        ratio = 1.0 if across == 0 else float("inf")

    # --- binding ------------------------------------------------------------------
    energy = norms.mean(axis=0).sum(axis=1)
    refs = [sites[i] for i in np.argsort(-energy)[: max(n_refs, 1)]]

    per_ref = []
    for ref in refs:
        ident = np.zeros((n_subj, len(sites)))
        mcos = np.zeros((n_subj, len(sites)))
        marg = np.zeros((n_subj, len(sites)))
        conf = np.zeros((n_subj, len(sites)))
        ident_conf = np.full((n_subj, len(sites)), np.nan)
        chan_hits = np.zeros(n_live)
        for si in range(n_subj):
            Jr = per_subject[si][ref]["J"][:, live]
            for ui, u in enumerate(sites):
                st = binding_stats(*match_columns(per_subject[si][u]["J"][:, live], Jr), margin_tol)
                ident[si, ui] = st["identity_frac"]
                mcos[si, ui] = st["matched_cos"]
                marg[si, ui] = st["margin"]
                conf[si, ui] = st["confident_frac"]
                ident_conf[si, ui] = st["identity_frac_confident"]
                chan_hits += st["identity"]
        per_ref.append(
            {
                "reference_site": list(ref),
                "identity_frac_median": float(np.median(ident.mean(axis=0))),
                "identity_frac_mean": float(ident.mean()),
                "matched_cos_median": float(np.median(mcos.mean(axis=0))),
                "margin_median": float(np.median(marg)),
                "confident_frac_median": float(np.median(conf)),
                "identity_frac_confident": (
                    float(np.nanmedian(ident_conf)) if np.isfinite(ident_conf).any() else float("nan")
                ),
                "per_site_identity": ident.mean(axis=0),
                "channel_stability": chan_hits / (n_subj * len(sites)),
            }
        )

    # Same site, different subjects: the binding measurement's own noise floor.  Site
    # identity is held fixed, so anything below 1.0 here is measurement noise, not drift —
    # and it is the CEILING for the across-site score, which cannot be read past it.
    same_site = []
    for u in sites:
        for si in range(1, n_subj):
            st = binding_stats(
                *match_columns(per_subject[si][u]["J"][:, live], per_subject[0][u]["J"][:, live]), margin_tol
            )
            same_site.append(st["identity_frac"])

    eff_rank = float(np.median([effective_rank(per_subject[s][u]["J"][:, live]) for s in range(n_subj) for u in sites]))
    profile_keys = sorted(per_subject[0][sites[0]]["energy_profile"])
    return {
        "n_live_channels": n_live,
        "live_mask": live,
        "dead_channels": [int(i) for i in np.nonzero(~live)[0]],
        # d columns in an effectively r-dimensional space cannot be told apart when r << d.
        "effective_rank": eff_rank,
        "effective_rank_ratio": eff_rank / n_live,
        "spectrum_keep": keep,
        "rank_profile": {
            str(d): float(np.median([per_subject[s][u]["rank_profile"][d] for s in range(n_subj) for u in sites]))
            for d in profile_keys
        },
        "homogeneity": {
            "dev_across_median": across,
            "dev_within_sigma": sigma,
            "dev_within_median": float(floor),
            "ratio": float(ratio),
            # The noise-free part of the across-site spread: what is left of dev_across once
            # the sigma/sqrt(n) already in the subject-means is taken out.
            "true_site_sd": float(np.sqrt(max(across**2 - floor**2, 0.0))),
            "per_site_dev": dev_across,
            "scale_log_range": [float(scales.mean(axis=0).min()), float(scales.mean(axis=0).max())],
        },
        "binding": {
            "primary": per_ref[0],
            "reference_sensitivity": [r["identity_frac_median"] for r in per_ref],
            "chance": 1.0 / n_live,
            "same_site_null": float(np.median(same_site)) if same_site else float("nan"),
        },
        "energy_profile": {
            str(d): float(np.median([per_subject[s][u]["energy_profile"][d] for s in range(n_subj) for u in sites]))
            for d in profile_keys
        },
    }


def to_map(values, sites, latent_shape) -> np.ndarray:
    m = np.full(latent_shape, np.nan)
    for v, u in zip(values, sites):
        m[u] = v
    return m


def git_sha(path="."):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return str(o)


def build_verdict(report: dict, tol: float, bind_tol: float, energy_tol: float = 0.5) -> dict:
    """What may be claimed, and where.

    The gates are ordered by what invalidates what.  A window that captures little of the
    response, a degenerate assignment, or a broken positive control each make the two
    headline numbers unreadable, so they are checked before the numbers are interpreted at
    all — reporting "the labelling drifts" off an instrument that cannot reproduce itself
    would be the worst failure this script could have.
    """
    full = report["arms"]["full"]
    lines, ok, blocked = [], True, False

    # --- gate 1: did the window capture the response? ---------------------------------
    captured = full.get("energy_profile", {}).get("0", float("nan"))
    if np.isfinite(captured) and captured < energy_tol:
        blocked = True
        prof = ", ".join(f"+{d}:{v:.2f}" for d, v in sorted(full["energy_profile"].items(), key=lambda kv: int(kv[0])))
        lines.append(
            f"WINDOW TOO TIGHT: only {captured:.1%} of |dx/dz(u)|^2 lands in B(u), so J_u is a "
            f"peripheral tail of the response and neither statistic is about the mechanism. "
            f"This is itself an A4 centre-dominance failure worth reporting — confirm rho with "
            f"probe.jacobian_spread. Energy by dilation: {prof}. Re-run at a dilation reaching "
            f">= {energy_tol:.0%}."
        )

    # --- gate 2: is the assignment decisive? ------------------------------------------
    prim = full["binding"]["primary"]
    rank_ratio = full.get("effective_rank_ratio", float("nan"))
    if np.isfinite(rank_ratio) and rank_ratio < 0.5:
        blocked = True
        lines.append(
            f"DEGENERATE MATCHING: J_u columns span an effectively {full['effective_rank']:.1f}-dimensional "
            f"space against {full['n_live_channels']} channels ({rank_ratio:.0%}), median assignment margin "
            f"{prim['margin_median']:.3f}. Every permutation scores alike, so the identity fraction reads near "
            "chance whether or not the labelling drifted."
        )
        # A rank that does not recover as the window grows is not a windowing problem: the
        # local decoding map is rank-deficient, z(u) is not recoverable from the response
        # around u at ANY window, and per-position identifiability fails outright.
        rp = full.get("rank_profile", {})
        if rp:
            widest = max(rp, key=lambda k: int(k))
            best = max(rp.values())
            lines.append(
                "  rank by dilation: "
                + "  ".join(f"+{d}:{v:.1f}" for d, v in sorted(rp.items(), key=lambda kv: int(kv[0])))
            )
            if best < 0.5 * full["n_live_channels"]:
                lines.append(
                    f"  It does not recover by dilation +{widest} (best {best:.1f}/{full['n_live_channels']}), so "
                    "this is the LOCAL DECODING MAP being rank-deficient, not a window too small. z(u) is not "
                    "recoverable from the response around u at any window, and per-position identifiability "
                    "fails for this decoder regardless of Lambda."
                )
            else:
                lines.append(
                    f"  It recovers to {best:.1f} by dilation +{widest} — a windowing artefact. Re-run at that dilation."
                )

    # --- gate 3: does the measurement reproduce itself? --------------------------------
    null = full["binding"]["same_site_null"]
    if np.isfinite(null) and null < 0.9:
        blocked = True
        lines.append(
            f"NOISY INSTRUMENT: the same site across subjects matches itself only {null:.3f} of the time. "
            "That is the ceiling for the across-site score, so binding cannot be read past it."
        )

    # --- gate 4: the positive control ---------------------------------------------------
    ctrl = report["arms"].get("linear")
    if ctrl is not None:
        c_bind = ctrl["binding"]["primary"]["identity_frac_median"]
        if c_bind < 0.95:
            blocked = True
            lines.append(
                f"CONTROL FAILED: the linearized decoder binds at {c_bind:.3f}, and it is exactly "
                "translation-equivariant, so the answer must be ~1.0. The measurement is wrong, not the model."
            )
        else:
            lines.append(f"control ok: linearized decoder binds at {c_bind:.3f}, as it must.")

    frozen = report["arms"].get("frozen_norm")
    if frozen is not None:
        if report.get("freeze_was_noop"):
            lines.append(
                "frozen_norm is a NO-OP for this run: the decoder has no nn.GroupNorm to pin (a "
                "norm_type='layer' decoder uses ChannelLayerNorm3d). Its numbers repeat 'full' and say nothing."
            )
        else:
            lines.append(
                f"frozen_norm (diagnostic, not a control): binding {frozen['binding']['primary']['identity_frac_median']:.3f}, "
                f"ratio {frozen['homogeneity']['ratio']:.2f}x — the gap from 'full' is the normalization's cost."
            )

    if blocked:
        lines.append("=> No conclusion about A1 from this run. Fix the gates above and re-run.")
        return {"ok": False, "blocked": True, "lines": lines}

    # --- the numbers, only once the gates pass -----------------------------------------
    ratio = full["homogeneity"]["ratio"]
    bind = prim["identity_frac_median"]
    lines.append(f"full: homogeneity ratio {ratio:.2f}x the measurement floor; binding {bind:.3f}")

    if ratio <= tol:
        lines.append(
            f"A1/A4 spectra are homogeneous to within {tol:.1f}x measurement noise "
            f"(true across-site sd {full['homogeneity']['true_site_sd']:.4f} nats) — "
            "Section 5 may be claimed on the validity region."
        )
    else:
        lines.append(
            f"Spectra are NOT homogeneous ({ratio:.2f}x > {tol:.1f}x; true across-site sd "
            f"{full['homogeneity']['true_site_sd']:.4f} nats). Site-sharing degrades across Lambda; "
            "restrict every downstream identifiability number to the validity map."
        )
        ok = False

    if bind >= bind_tol:
        lines.append(
            f"Channel semantics bind across positions ({bind:.3f} >= {bind_tol:.2f}); one global gauge element."
        )
    else:
        lines.append(
            f"Binding is {bind:.3f} < {bind_tol:.2f}: the channel labelling drifts with position. "
            "A per-position gauge is NOT a global one — the field-level claim does not follow from "
            "the per-position results, and a probe on flattened patch features would hide this."
        )
        ok = False

    return {"ok": ok, "blocked": False, "lines": lines}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0, help="Decoder level to probe (0 = finest).")
    ap.add_argument("--n-subjects", type=int, default=4, help="Subjects; >1 is required for the noise floors.")
    ap.add_argument("--sites", choices=("foreground", "strata", "all"), default="foreground")
    ap.add_argument("--max-sites", type=int, default=48)
    ap.add_argument("--fg-thresh", type=float, default=0.2, help="Min brain fraction for --sites foreground.")
    ap.add_argument("--chunk", type=int, default=16, help="Basis tangents per JVP call; lower this if memory-bound.")
    ap.add_argument(
        "--dead-thresh", type=float, default=1e-3, help="Min median relative column norm for a live channel."
    )
    ap.add_argument("--n-refs", type=int, default=2, help="Reference sites for the binding sensitivity check.")
    ap.add_argument(
        "--tol", type=float, default=3.0, help="Homogeneity ratio allowed, in multiples of the noise floor."
    )
    ap.add_argument("--bind-tol", type=float, default=0.9, help="Binding identity fraction required.")
    ap.add_argument(
        "--block-dilation",
        type=int,
        default=0,
        help="Widen the measurement window by this many latent cells per side. Raise it until "
        "the energy profile shows the window capturing most of the response.",
    )
    ap.add_argument(
        "--profile-dilations",
        default="0,1,2,4",
        help="Dilations to report captured response energy for (free — same JVPs).",
    )
    ap.add_argument(
        "--margin-tol",
        type=float,
        default=0.05,
        help="Min |cos| lead over the runner-up for a match to count as decisive.",
    )
    ap.add_argument(
        "--spectrum-energy",
        type=float,
        default=0.99,
        help="Truncate the log spectrum to the leading directions holding this much squared "
        "energy. Past them the singular values are float32 noise, and log turns that into nats.",
    )
    ap.add_argument("--skip-control", action="store_true", help="Skip the control arms (not recommended).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    import torch.nn.functional as F

    import probe.receptive_field as rfmod
    from probe.jacobian_spread import build_arm, capture_decoder_inputs, make_fn

    torch.set_num_threads(cli.threads)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")  # gradient magnitudes are the measurement: float32, no AMP

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    ckpt = cli.checkpoint or os.path.join(cli.run_dir, "vqvae_best.pt")
    model, args, device = load_model_from_run_dir(cli.run_dir, ckpt, device)
    model.eval()
    decoder = model.decoders[cli.level]

    save_dir = getattr(args, "save_dir", None)
    report: dict = {
        "run_tag": getattr(args, "tag", None) or (os.path.basename(str(save_dir).rstrip("/")) if save_dir else None),
        "run_dir": os.path.abspath(cli.run_dir),
        "checkpoint": os.path.abspath(ckpt),
        "checkpoint_sha256": file_sha256(ckpt),
        "repo_git_sha": git_sha(os.path.dirname(os.path.abspath(__file__)) + "/.."),
        "level": cli.level,
        "n_subjects": cli.n_subjects,
        "seed": cli.seed,
        "dtype": "float32",
        "amp": False,
        "cli": vars(cli),
    }

    print("\n" + "=" * 88)
    print("STEP 0  guard scan of the decoder module graph")
    print("=" * 88)
    scan = rfmod.scan_for_global_ops(decoder)
    report["guard"] = scan
    if scan["ok"]:
        print("  PASS — no global or non-local operation found.")
    else:
        print(f"  WARN — {len(scan['offenders'])} spatially global operation(s):")
        for o in scan["offenders"]:
            print(f"    {o['module']:32s} {o['reason']}")
        print("\n  B(u) blocks are then not disjoint in influence, and A4's centre-dominance")
        print("  clause must be checked with probe.jacobian_spread before these numbers are read.")
        print("  The frozen_norm arm below isolates the convolutional path.")

    # ---------------- data + operating point ---------------------------------------
    # build_synthetic_test_set renders from `args` whatever the run was trained on, so an
    # ADNI run would be measured on synthetic volumes and report a plausible-looking number
    # for a decoder never operated at that point.  Refuse instead.  Nothing below needs
    # ground-truth factors, so the ADNI path is a dataset swap and not a redesign.
    if not str(getattr(args, "dataset_name", "")).lower().startswith("synthetic"):
        raise SystemExit(
            f"This run trained on dataset_name={getattr(args, 'dataset_name', None)!r}, but only the "
            "synthetic loader is wired up here — measuring it on synthetic volumes would put the "
            "decoder at an operating point it never saw. Real-data support needs a loader that "
            "yields {'image': (view1, view2), 'mask': ...} for this run's dataset."
        )
    ds = build_synthetic_test_set(args, max(cli.n_subjects, 8), cache=False, causal=True)
    inner = getattr(ds, "_inner", None)
    gen = torch.Generator().manual_seed(cli.seed)

    subjects = []
    for i in range(cli.n_subjects):
        item = ds[i]
        x = torch.stack([item["image"][0], item["image"][1]], 0).to(device)
        z, style = capture_decoder_inputs(model, x, cli.level)
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
        subjects.append(
            {
                "z": z[:1].contiguous(),
                "style": None if style is None else style[:1].contiguous(),
                "tissue": tissue,
                "mask": item["mask"][0][0],
            }
        )

    latent_shape = tuple(subjects[0]["z"].shape[2:])
    n_channels = subjects[0]["z"].shape[1]
    # Probe the output shape through the same call convention the measurement uses, so a
    # style-less decoder cannot work here and fail inside the JVP loop.
    with torch.no_grad():
        out_shape = tuple(make_fn(decoder, subjects[0]["style"])(subjects[0]["z"]).shape[2:])

    brain_frac = F.adaptive_avg_pool3d((subjects[0]["mask"] > 0).float()[None, None], latent_shape)[0, 0]
    sites, strata = select_sites(
        brain_frac, latent_shape, cli.sites, cli.max_sites, cli.fg_thresh, gen, subjects[0]["tissue"]
    )

    report.update(
        {
            "latent_grid": list(latent_shape),
            "output_shape": list(out_shape),
            "latent_channels": n_channels,
            "n_sites": len(sites),
            "sites": [list(s) for s in sites],
            "site_strata": {str(list(k)): v for k, v in strata.items()},
        }
    )
    print(f"\n  latent grid {latent_shape} x {n_channels}ch  ->  output {out_shape}")
    print(f"  measuring {len(sites)} site(s) x {cli.n_subjects} subject(s) x {n_channels} JVP tangents per arm")

    # ---------------- measure -------------------------------------------------------
    import torch.nn as nn

    # freeze_norm_statistics only swaps nn.GroupNorm. A norm_type='layer' decoder is built
    # from ChannelLayerNorm3d, which is not one, so the arm silently repeats 'full' — the
    # same shape of no-op as the DataParallel --freeze-encoder bug. Detect and say so.
    n_gn = sum(1 for m in decoder.modules() if isinstance(m, nn.GroupNorm))
    report["decoder_groupnorm_count"] = n_gn
    report["freeze_was_noop"] = n_gn == 0

    profile_dilations = sorted({int(v) for v in cli.profile_dilations.split(",") if v.strip() != ""})
    arms = ARMS[:1] if cli.skip_control else ARMS
    report["arms"] = {}
    for arm in arms:
        print("\n" + "=" * 88)
        print(f"ARM  {arm}" + ("   (NO-OP: decoder has no nn.GroupNorm)" if arm == "frozen_norm" and n_gn == 0 else ""))
        print("=" * 88)
        per_subject = []
        for si, sub in enumerate(subjects):
            if arm == "linear":
                dec = make_linear_decoder(decoder)
            else:
                dec = build_arm(decoder, None, "frozen" if arm == "frozen_norm" else "live", sub["z"], sub["style"])
            fn = make_fn(dec, sub["style"])
            per_subject.append(
                measure_arm(
                    fn,
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
            print(f"  subject {si + 1}/{len(subjects)} done")
        red = reduce_arm(per_subject, sites, cli.dead_thresh, cli.n_refs, cli.margin_tol, cli.spectrum_energy)

        h, b = red["homogeneity"], red["binding"]
        p = b["primary"]
        print(f"\n  live channels           {red['n_live_channels']}/{n_channels}")
        print(
            f"  rank at the window      {red['effective_rank']:.1f}/{red['n_live_channels']}  "
            f"({red['effective_rank_ratio']:.0%} — below ~50% the matching is degenerate)"
        )
        print(
            "  captured energy         "
            + "  ".join(f"+{d}:{v:.3f}" for d, v in sorted(red["energy_profile"].items(), key=lambda kv: int(kv[0])))
        )
        print(
            "  effective rank          "
            + "  ".join(f"+{d}:{v:.1f}" for d, v in sorted(red["rank_profile"].items(), key=lambda kv: int(kv[0])))
            + f"   /{red['n_live_channels']} channels"
        )
        print(
            f"                          (by dilation; measured at +{cli.block_dilation}. A rank that stays "
            "flat as the window grows is architectural, not a windowing artefact.)"
        )
        print(f"  spectrum truncated to   {red['spectrum_keep']} of {red['n_live_channels']} singular values")
        print("\n  SPECTRAL HOMOGENEITY (shape of the log spectrum; scale excluded by construction)")
        print(f"    across-site deviation {h['dev_across_median']:.4f} nats  (subject-mean shape vs the median site)")
        print(f"    per-subject sigma     {h['dev_within_sigma']:.4f} nats  (same site, across subjects)")
        print(f"    floor for the means   {h['dev_within_median']:.4f} nats  (sigma / sqrt(n_subjects))")
        print(f"    ratio                 {h['ratio']:.2f}x   (1.0 = homogeneous to measurement precision)")
        print(f"    true across-site sd   {h['true_site_sd']:.4f} nats  (noise removed)")
        print("\n  BINDING (Hungarian match of J_u columns to the reference site; identity expected)")
        print(f"    identity fraction     {p['identity_frac_median']:.3f}   (chance {b['chance']:.3f})")
        print(f"    matched |cos|         {p['matched_cos_median']:.3f}")
        print(
            f"    assignment margin     {p['margin_median']:.3f}   ({p['confident_frac_median']:.0%} of columns decisive)"
        )
        print(f"    identity (confident)  {p['identity_frac_confident']:.3f}   (among decisive columns only)")
        print(f"    same-site null        {b['same_site_null']:.3f}   (ceiling for the above; 1.0 is perfect)")
        print(f"    reference sensitivity {['%.3f' % v for v in b['reference_sensitivity']]}")

        red["per_site_dev_map"] = to_map(h["per_site_dev"], sites, latent_shape)
        red["per_site_identity_map"] = to_map(p["per_site_identity"], sites, latent_shape)
        report["arms"][arm] = red

    # ---------------- validity map + verdict ----------------------------------------
    full = report["arms"]["full"]
    floor = full["homogeneity"]["dev_within_median"]
    dev_map = full["per_site_dev_map"]
    id_map = full["per_site_identity_map"]
    valid = (dev_map <= cli.tol * floor) & (id_map >= cli.bind_tol)
    n_measured = int(np.isfinite(dev_map).sum())
    report["validity"] = {
        "n_sites_valid": int(valid.sum()),
        "n_sites_measured": n_measured,
        "frac_valid": float(valid.sum() / n_measured) if n_measured else float("nan"),
        "tol_multiplier": cli.tol,
        "bind_tol": cli.bind_tol,
    }

    verdict = build_verdict(report, cli.tol, cli.bind_tol)
    report["verdict"] = verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    for line in verdict["lines"]:
        print(f"  {line}")
    print(
        f"\n  validity region: {report['validity']['n_sites_valid']}/{n_measured} measured sites "
        f"({report['validity']['frac_valid']:.1%}). Report identifiability metrics on these sites only."
    )

    # ---------------- write ---------------------------------------------------------
    os.makedirs(cli.out_dir, exist_ok=True)
    stem = os.path.join(cli.out_dir, f"{cli.prefix}site_sharing")
    map_keys = ("per_site_dev_map", "per_site_identity_map")
    maps = {f"{a}_{k}": report["arms"][a][k] for a in report["arms"] for k in map_keys}
    np.savez(stem + "_maps.npz", validity=valid, **maps)

    if cli.plot:
        plot_maps(stem + ".png", report, valid, latent_shape)

    # The maps are full latent grids carrying NaN at unmeasured sites.  json writes those as
    # a bare `NaN`, which is not valid JSON and trips strict parsers — and the arrays are
    # already in the .npz, verbatim.  Keep the JSON to the scalars and per-site vectors.
    for a in report["arms"]:
        for k in map_keys:
            report["arms"][a].pop(k, None)
    with open(stem + ".json", "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    print(f"\n  wrote {stem}.json and {stem}_maps.npz" + (f" and {stem}.png" if cli.plot else ""))


def plot_maps(path, report, valid, latent_shape):
    """Mid-axial slice of each map, one row per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(report["arms"])
    fig, axes = plt.subplots(len(arms), 3, figsize=(12, 4 * len(arms)), squeeze=False)
    k = latent_shape[2] // 2
    for r, arm in enumerate(arms):
        a = report["arms"][arm]
        panels = [
            ("shape deviation (nats)", a["per_site_dev_map"][:, :, k], "magma"),
            ("binding identity frac", a["per_site_identity_map"][:, :, k], "viridis"),
            ("valid sites", valid[:, :, k].astype(float), "Greens"),
        ]
        for c, (title, data, cmap) in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(np.asarray(data, dtype=float).T, origin="lower", cmap=cmap)
            ax.set_title(f"{arm} — {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
