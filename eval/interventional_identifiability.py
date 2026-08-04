#!/usr/bin/env python
"""Interventional (Jacobian) identifiability on the synthetic pseudo-MRI benchmark.

Every identifiability number in this project so far is **observational**: draw
samples, encode them, fit a probe from the representation to the ground-truth
factors.  That protocol has two structural problems on this benchmark, and both
disappear under intervention.

1. **Factor dependence.**  Runs trained with ``--synthetic-causal`` draw
   ``z_content`` from an SCM, so the factors are correlated (under a random graph
   ``brain_size`` and ``ventricle_size`` reach |r| ~ 0.8).  No observational
   metric — block-MCC, DCI, per-factor R² — can then distinguish "recovered
   ``z_1``" from "recovered ``z_2``, which is correlated with ``z_1``".
   Overwriting ``z_j`` instead of conditioning on it breaks the dependence by
   construction.
2. **Pooling is a gauge choice.**  The representation is a field
   ``(C, D, H, W)``; the named ground truth is a vector of global scalars.  Every
   probe therefore has to pick a pooling, and (see ``eval/content_rank_pca.py``)
   the pooling decides the result.  A response *difference* is a field compared
   against a field, so nothing has to be pooled to define the metric — pooling
   enters only as a reported decomposition.

Protocol
--------
For each target latent ``j`` and each of ``N`` base samples, re-render the pair
with everything held fixed — the other content dims, the nuisance
deformation/fissure fields, both style vectors, and the rendering seed — except
``z_j``, which is set to ``z_j ± eps``.  Encode both and take

    ``Delta_j[i] = f(x(z_j + eps)) - f(x(z_j - eps))``      (native latent field)

For small ``eps`` this is ``2*eps*J_h e_j`` to first order, so:

    ``{Delta_j}`` linearly independent  <=>  ``J_h`` full rank
                                        <=>  ``h`` locally invertible

which *is* the statement of local block identifiability, measured rather than
proxied by an MCC.  The metrics below all read off that response set.

What is reported
----------------
* ``snr_content`` / ``snr_style`` — response magnitude in each block over a
  matched null (see below).  ``snr ~ 1`` means the encoder does not respond to
  the factor at all; no amount of probing can recover it.  For a *style* target,
  ``snr_content`` **is** the leakage measure — probe-free, and immune to the
  capacity confound that makes ``run_dci_compare``'s leakage numbers
  baseline-sensitive.
* ``consistency`` — leave-one-out cosine between each sample's response and the
  mean direction.  The mean direction is only a meaningful summary when this is
  high; low values mean ``J_h`` varies strongly across the support (so read the
  per-sample identification accuracy instead).
* ``rank_signal`` — the honest answer to "how many factors are separately
  encoded": singular values of the ``J x dim`` response matrix that exceed a
  sign-flip null threshold.  This is the rank that matters for identifiability,
  and unlike an observational ``eff_rank`` it is *causally* estimated and has a
  calibrated floor.  ``rank_signal < J`` falsifies block identifiability.
* ``max_alias`` — largest ``|cos|`` between a factor's response direction and any
  other factor's.  A probe-free, fit-free DCI matrix.
* ``id_acc`` — nearest-centroid cross-validated accuracy at identifying *which*
  factor was intervened on from a single response.  Sample-level identifiability,
  robust to a varying Jacobian.  Chance is ``1/J``.
* ``gap_frac`` — fraction of the response energy carried by the position mean.
  Writing ``z[n,p,c] = s[n,c] + r[n,p,c]``, GAP recovers ``s`` exactly and
  position-centred patch features recover ``r`` exactly, so this says — measured,
  not assumed — whether a factor's response is global or local.  Compare it
  against ``run_dci_compare.FACTOR_POOLING``, which asserts the same split a
  priori.
* ``rho_loc`` / ``in_mass`` — locality.  The ground-truth influence field of a
  factor is just its *image-space* response ``mean_i |x_b - x_a|``, which needs no
  hard-coded knowledge of the renderer and is defined for every target including
  style.  ``rho_loc`` is Spearman between that field and the latent response
  magnitude, both on the latent grid; ``in_mass`` is the fraction of latent
  response energy inside the image-space support, against the uniform baseline
  printed beside it.
* **nuisance contamination** — ``z_deformation`` / ``z_fissure`` are pure nuisance,
  so their response directions are a reference for "structure the objective
  should NOT have kept".  Reported as response magnitude and as the largest
  ``|cos|`` between any content direction and a nuisance direction.

Nulls
-----
Two, and they answer different questions.

* **Render-noise null** — re-render the *same* latents with a different view seed.
  The only stochastic part of ``render_modality`` is its noise, so this is the
  encoder's response when nothing changed.  It sets the floor for ``snr`` and,
  via sign-flipped resampling of the same responses, the singular-value threshold
  for ``rank_signal``.
* **Nuisance null** — intervene on the deformation/fissure fields.  A stronger,
  more interesting floor: a content response that is not clearly above the
  *nuisance* response is not evidence of content encoding.

Caveats
-------
* ``eps`` must be small enough that the first-order reading holds and large enough
  to clear the noise null.  ``--linearity-check`` re-measures response norms at
  ``eps/2``; the ratio should be ~2.  A ratio well below 2 means saturation
  (a floor/ceiling effect in the renderer, as documented for the ventricle) and
  the first-order interpretation does not apply to that factor.
* The normalization mode matters and is not cosmetic.  Under ``per_sample``
  z-scoring the normalizer partially cancels interventions on global intensity,
  so a low ``gain`` response is a property of the *pipeline*, not only the
  encoder.  The mode is read from the run's ``settings.json`` and printed.
* Match the generator vintage to the training commit.  Checkpoints from before
  ``7ac56a3`` never saw the current ``render_structure``; pass ``--old-generator``
  for those, exactly as in ``run_dci_compare``.

Usage
-----
    python -m eval.interventional_identifiability \\
        --run-dirs runs/contrast runs/recon \\
        --baseline runs/recon \\
        --num-samples 128 --eps 0.5 --level 0 \\
        --out interventional_out

The torch-dependent parts (render + encoder forward) are lazily imported so the
metric layer stays testable on plain numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Nuisance targets are prefixed so they never get mistaken for named factors in
# the rank / aliasing computations, which are defined over content only.
NUISANCE_PREFIX = "~"


# --------------------------------------------------------------------------- #
# Response construction (torch — lazily imported)
# --------------------------------------------------------------------------- #


def _resolve_blocks(model_out, level, all_content):
    """Content / style channel indices for ``level`` from the run's own Gumbel mask."""
    import torch

    from eval.dci import _parse_content_indices

    feat = model_out[2][level]
    n_channels = feat.shape[1]
    soft_masks = model_out[6] if len(model_out) > 6 else {}
    content_idx = None
    if isinstance(soft_masks, dict) and level in soft_masks:
        mask = soft_masks[level]
        mask = mask[0] if isinstance(mask, tuple) else mask
        content_idx = _parse_content_indices(torch.where(mask.bool())[-1])
    if all_content or not content_idx:
        content_idx = list(range(n_channels))
    style_idx = sorted(set(range(n_channels)) - set(content_idx))
    return content_idx, style_idx


def _encode_pair(model, x1, x2, level, device, all_content, field_pool=None):
    """Encode a paired batch to native latent fields.

    Returns ``(content_v1, style_v1, content_v2, style_v2, grid)`` with each array
    shaped ``(B, C_block, P)`` and ``grid`` the ``(D, H, W)`` of the latent map.
    Both views go through one ``n_views=2`` call so ``--separate-encoders`` routing
    matches training.
    """
    import torch
    import torch.nn.functional as F

    x = torch.cat([x1, x2], dim=0).to(device)
    with torch.no_grad():
        out = model(x, return_recon=False, pool_only=False, n_views=2)
    feat = out[2][level]  # (2B, C, D, H, W)
    if feat.dim() != 5:
        raise RuntimeError(f"level {level} is not a spatial map (got shape {tuple(feat.shape)})")
    if field_pool is not None:
        feat = F.adaptive_avg_pool3d(feat, tuple(field_pool))
    grid = tuple(feat.shape[2:])
    content_idx, style_idx = _resolve_blocks(out, level, all_content)

    b = feat.shape[0] // 2
    flat = feat.reshape(feat.shape[0], feat.shape[1], -1).float().cpu().numpy()
    c1 = flat[:b, content_idx, :]
    c2 = flat[b:, content_idx, :]
    s1 = flat[:b, style_idx, :] if style_idx else None
    s2 = flat[b:, style_idx, :] if style_idx else None
    return c1, s1, c2, s2, grid


def _perturb(latents, block, index, delta, direction=None):
    """Copy ``latents`` with one target shifted by ``delta``. Everything else identical."""
    out = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in latents.items()}
    if block == "content":
        out["z_content"][index] = out["z_content"][index] + delta
    elif block == "style":
        # View-1 style only: view 2 must stay fixed, otherwise a "style" response
        # is confounded with a change to the other view's rendering.
        out["z_style_v1"][index] = out["z_style_v1"][index] + delta
    elif block in ("z_deformation", "z_fissure"):
        out[block] = out[block] + delta * direction
    else:
        raise ValueError(f"unknown block {block!r}")
    return out


def _render_batch(dataset, base, targets_delta):
    """Render + normalize one condition for every base sample.

    ``targets_delta`` is ``(block, index, delta, direction, seed_offset)``.  Returns
    stacked ``(x1, x2)`` tensors and the mean absolute image-space difference is
    computed by the caller from two such calls.
    """
    import torch

    inner = dataset._inner
    xs1, xs2 = [], []
    block, index, delta, direction, seed_offset = targets_delta
    for latents, seed in base:
        lat = latents if delta == 0.0 and direction is None else _perturb(latents, block, index, delta, direction)
        x1, x2, mask = inner.render_pseudo_mri(
            lat["z_content"],
            lat["z_deformation"],
            lat["z_fissure"],
            lat["z_style_v1"],
            lat["z_style_v2"],
            seed + seed_offset,
        )
        x1, x2 = dataset.normalize_views(x1, x2, mask, mask.clone())
        xs1.append(x1)
        xs2.append(x2)
    return torch.stack(xs1), torch.stack(xs2)


def collect_responses(
    model,
    dataset,
    base,
    targets,
    level,
    device,
    eps,
    all_content=False,
    batch_size=16,
    field_pool=None,
    null_seed_offset=7919,
):
    """Render, encode and difference every intervention.

    Returns ``responses`` (``name -> dict`` of ``content``/``style``/``image`` arrays),
    plus ``null`` under the same keys and the latent ``grid``.
    """
    import torch

    def _encode_all(x1, x2):
        c1, s1, _c2, _s2, grid = [], [], None, None, None
        for k in range(0, len(x1), batch_size):
            a, b, _c, _s, g = _encode_pair(
                model, x1[k : k + batch_size], x2[k : k + batch_size], level, device, all_content, field_pool
            )
            c1.append(a)
            if b is not None:
                s1.append(b)
            grid = g
        return np.concatenate(c1), (np.concatenate(s1) if s1 else None), grid

    model.eval()
    out, grid = {}, None

    conditions = [(name, block, idx, direction) for name, block, idx, direction in targets]
    # The render-noise null is condition zero: same latents, different view seed.
    conditions = [("__null__", None, None, None)] + conditions

    for name, block, idx, direction in conditions:
        t0 = time.perf_counter()
        if name == "__null__":
            x1a, x2a = _render_batch(dataset, base, (None, None, 0.0, None, 0))
            x1b, x2b = _render_batch(dataset, base, (None, None, 0.0, None, null_seed_offset))
        else:
            x1a, x2a = _render_batch(dataset, base, (block, idx, -eps, direction, 0))
            x1b, x2b = _render_batch(dataset, base, (block, idx, +eps, direction, 0))

        ca, sa, grid = _encode_all(x1a, x2a)
        cb, sb, _ = _encode_all(x1b, x2b)

        out[name] = {
            "content": cb - ca,
            "style": (sb - sa) if (sa is not None and sb is not None) else None,
            # Image-space response: the factor's ground-truth influence field, with
            # no hard-coded knowledge of the renderer.
            "image": (x1b - x1a).abs().mean(dim=0).float().numpy(),
        }
        logger.info("  %-18s rendered+encoded in %.1fs", name, time.perf_counter() - t0)
        del x1a, x2a, x1b, x2b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    null = out.pop("__null__")
    return out, null, grid


# --------------------------------------------------------------------------- #
# Metrics (numpy only)
# --------------------------------------------------------------------------- #


def _fro(a):
    """Per-sample Frobenius norm of a (N, C, P) response stack."""
    return np.sqrt((a.astype(np.float64) ** 2).sum(axis=(1, 2)))


def _flat(a):
    return a.reshape(len(a), -1).astype(np.float64)


def loo_consistency(d):
    """Leave-one-out cosine between each response and the mean of the others.

    Self-inclusion would inflate this at small N, which is exactly the regime the
    evaluator runs in.
    """
    x = _flat(d)
    n = len(x)
    if n < 3:
        return float("nan")
    total = x.sum(0)
    cos = []
    for i in range(n):
        other = (total - x[i]) / (n - 1)
        na, nb = np.linalg.norm(x[i]), np.linalg.norm(other)
        if na < 1e-12 or nb < 1e-12:
            continue
        cos.append(float(x[i] @ other / (na * nb)))
    return float(np.mean(cos)) if cos else float("nan")


def gap_fraction(d):
    """Share of response energy in the position mean (the subject term ``s``).

    ``z[n,p,c] = s[n,c] + r[n,p,c]``: GAP recovers ``s`` exactly and the
    position-centred residual is ``r``, and the two are orthogonal, so this is an
    exact energy split rather than a heuristic.
    """
    x = d.astype(np.float64)
    p = x.shape[2]
    s = x.mean(axis=2, keepdims=True)
    e_s = float((s**2).sum() * p)
    e_all = float((x**2).sum())
    return e_s / e_all if e_all > 0 else float("nan")


def response_matrix(responses, names, key="content"):
    """Rows = mean response direction per factor, unnormalized.

    Left unnormalized on purpose: the sign-flip null below has to compare against
    rows on the same scale, and unit-normalizing would turn a set of pure-noise
    rows into near-orthogonal unit vectors (apparent full rank).
    """
    rows = [responses[n][key].astype(np.float64).mean(0).ravel() for n in names]
    return np.vstack(rows)


def _null_rows(null_response, j, rng):
    """``j`` rows of the form ``mean_i(+-1 * Delta_null[i])``.

    Same sample-size scaling as a real response row, under the hypothesis that the
    mean response is zero — so singular values and cosines computed on these are
    directly comparable to the real ones.
    """
    x = _flat(null_response)
    signs = rng.choice([-1.0, 1.0], size=(j, len(x)))
    return (signs @ x) / len(x)


def signal_rank(r_matrix, null_response, n_rep=64, quantile=0.95, seed=0):
    """Number of singular values of ``r_matrix`` above a sign-flip null threshold.

    Returns ``(rank, singular_values, threshold)``.
    """
    sv = np.linalg.svd(r_matrix, compute_uv=False)
    rng = np.random.RandomState(seed)
    tops = [np.linalg.svd(_null_rows(null_response, len(r_matrix), rng), compute_uv=False)[0] for _ in range(n_rep)]
    thr = float(np.quantile(tops, quantile))
    return int((sv > thr).sum()), sv, thr


def alias_matrix(r_matrix):
    """``|cos|`` between every pair of response directions — a fit-free DCI matrix."""
    norms = np.linalg.norm(r_matrix, axis=1, keepdims=True)
    unit = r_matrix / np.clip(norms, 1e-12, None)
    return np.abs(unit @ unit.T)


def alias_floor(null_response, j, n_rep=32, quantile=0.95, seed=1):
    """Largest ``|cos|`` two *unrelated* directions reach at this sample size.

    ``max_alias`` is attenuated by estimation noise — two identical factors read
    well below 1.0 once each mean direction carries an independent noise component
    — so the raw number is only interpretable against this floor.
    """
    rng = np.random.RandomState(seed)
    tops = []
    for _ in range(n_rep):
        a = alias_matrix(_null_rows(null_response, j, rng))
        np.fill_diagonal(a, 0.0)
        tops.append(float(a.max()))
    return float(np.quantile(tops, quantile))


def identification_accuracy(responses, names, key="content", n_folds=5, seed=0):
    """CV nearest-centroid accuracy at naming the intervened factor from one response.

    Cosine nearest-centroid rather than a fitted classifier: in ``C*P`` dimensions
    with ``N`` samples per class any fitted probe is capacity-bound, and the
    question here is whether the responses are *separated*, not whether a probe can
    be made to separate them.
    """
    x = np.stack([_flat(responses[n][key]) for n in names])  # (J, N, dim)
    j, n, _ = x.shape
    x = x / np.clip(np.linalg.norm(x, axis=2, keepdims=True), 1e-12, None)
    rng = np.random.RandomState(seed)
    folds = rng.permutation(n) % n_folds
    correct = 0
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if not tr.any() or not te.any():
            continue
        cent = x[:, tr, :].mean(1)
        cent = cent / np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12, None)
        for cls in range(j):
            pred = (x[cls, te, :] @ cent.T).argmax(1)
            correct += int((pred == cls).sum())
    return correct / float(j * n), 1.0 / j


def locality(d_content, image_response, grid, support_frac=0.9):
    """Compare the latent response map against the factor's image-space support.

    Returns ``(spearman_rho, in_mass, uniform_baseline)``.  ``uniform_baseline`` is
    the fraction of latent positions the support occupies, i.e. what ``in_mass``
    would be if the response were spread evenly.
    """
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr

    a = np.sqrt((d_content.astype(np.float64) ** 2).sum(1)).mean(0)  # (P,)
    w = torch.as_tensor(image_response).float()
    if w.dim() == 3:
        w = w[None]
    w = F.adaptive_avg_pool3d(w[None], tuple(grid))[0, 0].reshape(-1).numpy().astype(np.float64)

    if a.sum() <= 0 or w.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(a, w).statistic)

    order = np.argsort(-w)
    cum = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cum, support_frac) + 1)
    support = order[:k]
    return rho, float(a[support].sum() / a.sum()), float(k / len(w))


def score_run(responses, null, grid, content_names, style_names, nuisance_names, support_frac=0.9):
    """All metrics for one model. Pure numpy — no torch, no model."""
    null_c = _fro(null["content"]).mean()
    null_s = _fro(null["style"]).mean() if null["style"] is not None else float("nan")

    per_factor = {}
    for name, d in responses.items():
        rc = _fro(d["content"]).mean()
        rs = _fro(d["style"]).mean() if d["style"] is not None else float("nan")
        rho, in_mass, uniform = locality(d["content"], d["image"], grid, support_frac)
        per_factor[name] = {
            "resp_content": float(rc),
            "resp_style": float(rs),
            "snr_content": float(rc / null_c) if null_c > 0 else float("nan"),
            "snr_style": float(rs / null_s) if null_s and null_s > 0 else float("nan"),
            "consistency": loo_consistency(d["content"]),
            "gap_frac": gap_fraction(d["content"]),
            "rho_loc": rho,
            "in_mass": in_mass,
            "in_mass_uniform": uniform,
        }

    present = [n for n in content_names if n in responses]
    # With one channel per normalization group the encoder output has an exactly
    # zero per-channel spatial mean, so the subject term s is identically zero and
    # the GAP/patch decomposition is degenerate — every GAP-pooled probe at this
    # level is reading float noise rather than a weak signal.  Detect it here so
    # a column of 0.000 is not mistaken for a measurement.
    gap_void = (
        all(not np.isfinite(m["gap_frac"]) or m["gap_frac"] < 1e-6 for m in per_factor.values())
        and gap_fraction(null["content"]) < 1e-6
    )

    out = {
        "per_factor": per_factor,
        "null_resp_content": float(null_c),
        "null_resp_style": float(null_s),
        "gap_void": bool(gap_void),
        "content_names": present,
        "style_names": [n for n in style_names if n in responses],
        "nuisance_names": [n for n in nuisance_names if n in responses],
    }

    if len(present) >= 2:
        r = response_matrix(responses, present)
        rank, sv, thr = signal_rank(r, null["content"])
        alias = alias_matrix(r)
        np.fill_diagonal(alias, 0.0)
        acc, chance = identification_accuracy(responses, present)
        out.update(
            {
                "rank_signal": rank,
                "rank_target": len(present),
                "singular_values": [float(v) for v in sv],
                "sv_null_threshold": thr,
                "cond_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
                "alias_matrix": alias.tolist(),
                "alias_floor": alias_floor(null["content"], len(present)),
                "id_acc": float(acc),
                "id_chance": float(chance),
            }
        )
        for k, name in enumerate(present):
            per_factor[name]["max_alias"] = float(alias[k].max())
            per_factor[name]["alias_with"] = present[int(alias[k].argmax())]

        nz = [n for n in nuisance_names if n in responses]
        if nz:
            rn = response_matrix(responses, nz)
            contam = alias_matrix(np.vstack([r, rn]))[: len(present), len(present) :]
            out["nuisance_contamination"] = float(contam.max())
            for k, name in enumerate(present):
                per_factor[name]["nuisance_cos"] = float(contam[k].max())
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  n/a"
    return f"{v:.{nd}f}"


def print_report(rows, baseline_name=None):
    for row in rows:
        name = row["name"]
        print("\n" + "=" * 100)
        print(f"  INTERVENTIONAL IDENTIFIABILITY — {name}")
        print("=" * 100)
        print(
            f"  eps={row['eps']}  N={row['num_samples']}  level={row['level']}  "
            f"normalize={row['normalize']}  causal_base={row['causal_base']}  generator={row['generator']}"
        )
        print(
            f"  null (render-noise) response: content {_f(row['null_resp_content'])}  style {_f(row['null_resp_style'])}"
        )

        print("\n  PER-FACTOR RESPONSE")
        print("  snr>1 = responds above render noise | gap_frac: 1.0 = purely global, 0.0 = purely local")
        print("  '~' on max_alias = at or below the null floor, i.e. not distinguishable from unrelated")
        has_lin = any("linearity_ratio" in m for m in row["per_factor"].values())
        hdr = (
            f"  {'target':<18}{'snr_c':>8}{'snr_s':>8}{'consist':>9}{'gap_frac':>10}"
            f"{'rho_loc':>9}{'in_mass':>9}{'(unif)':>8}{'max_alias':>11}{'nuis_cos':>10}"
            + (f"{'lin(~2)':>9}" if has_lin else "")
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        floor = row.get("alias_floor")
        for group in ("content_names", "style_names", "nuisance_names"):
            for fn in row[group]:
                m = row["per_factor"][fn]
                # A max_alias at or below the null floor carries no information: two
                # unrelated directions reach it at this N. Mark it rather than let the
                # number be read as "well separated".
                al = m.get("max_alias")
                al_s = _f(al)
                if al is not None and floor is not None and np.isfinite(al) and al <= floor:
                    al_s = al_s.strip() + "~"
                print(
                    f"  {fn:<18}{_f(m['snr_content'], 2):>8}{_f(m['snr_style'], 2):>8}"
                    f"{_f(m['consistency']):>9}{_f(m['gap_frac']):>10}{_f(m['rho_loc']):>9}"
                    f"{_f(m['in_mass']):>9}{_f(m['in_mass_uniform']):>8}"
                    f"{al_s:>11}{_f(m.get('nuisance_cos')):>10}"
                    + (f"{_f(m.get('linearity_ratio'), 2):>9}" if has_lin else "")
                )
            if row[group]:
                print()

        if row.get("gap_void"):
            print(
                "  ! gap_frac is identically 0 for every target AND for the null: this level's\n"
                "    per-channel spatial mean is architecturally zero (one channel per normalization\n"
                "    group), so the subject term s does not exist here. The gap/patch split is\n"
                "    degenerate, and any GAP-pooled probe at this level is reading float noise\n"
                "    rather than a weak signal. Check --vqvae-hidden-channels against get_group_norm.\n"
            )

        if "rank_signal" in row:
            print("  CAUSAL RANK — how many content factors are separately encoded")
            print(
                f"    rank_signal {row['rank_signal']} / {row['rank_target']}   "
                f"(singular values above the sign-flip null threshold {_f(row['sv_null_threshold'], 4)})"
            )
            print("    singular values: " + "  ".join(_f(v, 3) for v in row["singular_values"]))
            print(f"    condition number {_f(row['cond_number'], 1)}")
            print(
                f"    alias floor {_f(row['alias_floor'])} — max_alias below this is indistinguishable "
                "from two unrelated directions at this N"
            )
            if row["rank_signal"] < row["rank_target"]:
                print(
                    f"    => FALSIFIED at this level: {row['rank_target'] - row['rank_signal']} factor "
                    "direction(s) are not separable from the null, so no invertible h exists."
                )
            else:
                print(
                    "    => necessary condition MET. Rank alone proves nothing further — read id_acc "
                    "and max_alias for whether the directions are usefully distinct."
                )
            print(
                f"\n    intervention id_acc {_f(row['id_acc'])} (chance {_f(row['id_chance'])})  "
                f"— naming the intervened factor from ONE response"
            )
            if "nuisance_contamination" in row:
                print(
                    f"    worst content/nuisance alignment {_f(row['nuisance_contamination'])} "
                    "— high means a content direction is partly the nuisance field"
                )

    if baseline_name and len(rows) > 1:
        base = next((r for r in rows if r["name"] == baseline_name), None)
        others = [r for r in rows if r["name"] != baseline_name]
        if base is None:
            return
        print("\n" + "=" * 100)
        print(f"  HEAD-TO-HEAD vs {baseline_name}")
        print("=" * 100)
        for r in others:
            print(f"\n  {r['name']} vs {baseline_name}")
            for label, key, better in (
                ("rank_signal", "rank_signal", "higher"),
                ("id_acc", "id_acc", "higher"),
                ("nuisance contamination", "nuisance_contamination", "lower"),
            ):
                mv, bv = r.get(key), base.get(key)
                if mv is None or bv is None:
                    continue
                delta = mv - bv
                win = (delta > 0) if better == "higher" else (delta < 0)
                mark = "WIN " if abs(delta) > 1e-9 and win else ("LOSS" if abs(delta) > 1e-9 else "TIE ")
                print(f"    {mark}  {label:<24}{_f(mv)}  vs  {_f(bv)}   (delta {delta:+.3f}, {better} is better)")
            print("    per-factor snr_content (model / baseline):")
            for fn in r["content_names"]:
                if fn not in base["per_factor"]:
                    continue
                mv = r["per_factor"][fn]["snr_content"]
                bv = base["per_factor"][fn]["snr_content"]
                print(f"      {fn:<20}{_f(mv, 2):>8}  {_f(bv, 2):>8}   {mv - bv:+.2f}")


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "interventional.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    flat = []
    for row in rows:
        for fn, m in row["per_factor"].items():
            flat.append(
                {
                    "model": row["name"],
                    "target": fn,
                    "eps": row["eps"],
                    "level": row["level"],
                    "num_samples": row["num_samples"],
                    "rank_signal": row.get("rank_signal"),
                    "rank_target": row.get("rank_target"),
                    "id_acc": row.get("id_acc"),
                    **m,
                }
            )
    if flat:
        keys = sorted({k for r in flat for k in r})
        with open(os.path.join(out_dir, "interventional.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
    logger.info("wrote %s/interventional.{json,csv}", out_dir)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def build_targets(n_content, n_style, grid_def, grid_fis, include_nuisance, seed=0):
    """``(name, block, index, direction)`` for every intervention target."""
    import torch

    from eval.dci import CONTENT_FACTOR_NAMES, STYLE_FACTOR_NAMES

    content = [(CONTENT_FACTOR_NAMES[i], "content", i, None) for i in range(min(n_content, len(CONTENT_FACTOR_NAMES)))]
    style = [(STYLE_FACTOR_NAMES[i], "style", i, None) for i in range(min(n_style, len(STYLE_FACTOR_NAMES)))]
    nuisance = []
    if include_nuisance:
        g = torch.Generator().manual_seed(seed)
        for block, k in (("z_deformation", grid_def), ("z_fissure", grid_fis)):
            d = torch.randn(k, k, k, generator=g)
            nuisance.append((NUISANCE_PREFIX + block, block, None, d / d.norm()))
    names = ([c[0] for c in content], [s[0] for s in style], [n[0] for n in nuisance])
    return content + style + nuisance, names


def main():
    p = argparse.ArgumentParser(description="Interventional (Jacobian) identifiability on the synthetic benchmark.")
    p.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to compare (settings.json each).")
    p.add_argument("--names", nargs="*", default=None, help="Labels (default: basename of each run-dir).")
    p.add_argument(
        "--baseline", default=None, help="Run-dir to anchor the head-to-head (e.g. the 0-contrastive model)."
    )
    p.add_argument(
        "--baseline-per-block",
        action="store_true",
        help="Score the baseline on its own (arbitrary) content/style split. Default: one all-content "
        "block, matching run_dci_compare, so its content-side numbers are comparable.",
    )
    p.add_argument("--checkpoint-name", default="vqvae_model.pt", help="Checkpoint filename in every run-dir.")
    p.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Base samples per intervention. Cost is 2*N renders per target; 128 is ample for a "
        "direction estimate, and the sign-flip null scales with it.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Intervention size in latent units (the factors are ~N(0,1)). Small enough for the "
        "first-order reading, large enough to clear the render-noise null.",
    )
    p.add_argument("--level", type=int, default=0, help="Encoder level to evaluate.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--field-pool",
        nargs=3,
        type=int,
        default=None,
        metavar=("D", "H", "W"),
        help="Average-pool the latent field to this grid before differencing. Only for memory relief "
        "on large level-0 grids — it coarsens the locality metrics, so leave it off when reporting.",
    )
    p.add_argument(
        "--iid-base",
        action="store_true",
        help="Draw base samples i.i.d. even when the run was trained with an SCM. Default matches "
        "training, so h is probed where it was fitted; the intervention breaks the dependence either way.",
    )
    p.add_argument("--no-nuisance", action="store_true", help="Skip the z_deformation / z_fissure targets.")
    p.add_argument(
        "--linearity-check",
        action="store_true",
        help="Also measure response norms at eps/2. The ratio should be ~2; well below means the "
        "factor is saturating and the first-order reading does not apply to it.",
    )
    p.add_argument("--support-frac", type=float, default=0.9, help="Image-response mass defining a factor's support.")
    p.add_argument(
        "--old-generator",
        action="store_true",
        help="Render with the PRE-7ac56a3 render_structure, for checkpoints trained before 2026-07-05.",
    )
    p.add_argument("--out", default="interventional_out", help="Output directory.")
    cli = p.parse_args()

    import torch

    from eval.run_dci_compare import _resolve_checkpoint
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    specs = list(cli.run_dirs)
    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in specs]
    if len(names) != len(specs):
        p.error("--names must have one entry per --run-dirs")
    baseline_name = None
    if cli.baseline:
        if cli.baseline in specs:
            baseline_name = names[specs.index(cli.baseline)]
        else:
            baseline_name = os.path.basename(os.path.normpath(cli.baseline))
            specs = [cli.baseline] + specs
            names = [baseline_name] + names

    ref_args = load_run_args(specs[0])
    if getattr(ref_args, "synthetic_mode", "pseudo_mri") != "pseudo_mri":
        p.error("interventional evaluation requires --synthetic-mode pseudo_mri (explicit, re-renderable latents)")

    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=(False if cli.iid_base else True))
    if cli.old_generator:
        from eval.legacy_renderer import FIXED_FACTORS, use_legacy_renderer

        use_legacy_renderer(dataset)
        logger.warning(
            "--old-generator: %s are rendered the OLD way; those rows are not comparable "
            "with current-generator runs.",
            ", ".join(FIXED_FACTORS),
        )

    # Base latents, drawn once and shared by every model and every target so the
    # only thing that differs between conditions is the intervened coordinate.
    base = []
    for i in range(cli.num_samples):
        item = dataset[i]
        base.append((item["gt_latents"], dataset._inner.sample_seed_for(i)))

    targets, (content_names, style_names, nuisance_names) = build_targets(
        getattr(ref_args, "synthetic_n_content", 9),
        getattr(ref_args, "synthetic_n_style", 3),
        getattr(ref_args, "synthetic_n_deformation_grid", 4),
        getattr(ref_args, "synthetic_n_fissure_grid", 8),
        include_nuisance=not cli.no_nuisance,
    )
    logger.info(
        "%d targets x %d samples x 2 conditions = %d renders per model (+%d for the null)",
        len(targets),
        cli.num_samples,
        2 * len(targets) * cli.num_samples,
        2 * cli.num_samples,
    )

    rows = []
    for name, run_dir in zip(names, specs):
        logger.info("=== %s (%s) ===", name, run_dir)
        all_content = (name == baseline_name) and not cli.baseline_per_block
        try:
            model, run_args, device = load_model_from_run_dir(
                run_dir, _resolve_checkpoint(run_dir, cli.checkpoint_name)
            )
            responses, null, grid = collect_responses(
                model,
                dataset,
                base,
                targets,
                cli.level,
                device,
                cli.eps,
                all_content=all_content,
                batch_size=cli.batch_size,
                field_pool=cli.field_pool,
            )
            row = score_run(
                responses, null, grid, content_names, style_names, nuisance_names, support_frac=cli.support_frac
            )

            if cli.linearity_check:
                half, _null_h, _g = collect_responses(
                    model,
                    dataset,
                    base,
                    targets,
                    cli.level,
                    device,
                    cli.eps / 2.0,
                    all_content=all_content,
                    batch_size=cli.batch_size,
                    field_pool=cli.field_pool,
                )
                for fn, d in half.items():
                    r_half = _fro(d["content"]).mean()
                    full = row["per_factor"][fn]["resp_content"]
                    row["per_factor"][fn]["linearity_ratio"] = float(full / r_half) if r_half > 0 else float("nan")

            row.update(
                {
                    "name": name,
                    "run_dir": run_dir,
                    "checkpoint": cli.checkpoint_name,
                    "eps": cli.eps,
                    "level": cli.level,
                    "num_samples": cli.num_samples,
                    "grid": list(grid),
                    "baseline_all_content": all_content,
                    "normalize": getattr(run_args, "synthetic_normalize", "per_sample"),
                    "causal_base": (not cli.iid_base) and bool(getattr(run_args, "synthetic_causal", False)),
                    "generator": "legacy_pre_7ac56a3" if cli.old_generator else "current",
                }
            )
            rows.append(row)
        except Exception as e:  # keep one bad run from killing the comparison
            logger.error("Skipping %s (%s): %s", name, run_dir, e)
        finally:
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        logger.error("No models evaluated successfully.")
        return

    print_report(rows, baseline_name)
    write_outputs(rows, cli.out)


if __name__ == "__main__":
    main()
