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
The decoder's *weights* are shared by construction, so a purely convolutional decoder would
score 1.0 on both measurements with nothing learned.  What varies across ``u`` is the
*state*: nonlinearities and — decisively here — normalization layers whose statistics are
computed over the whole volume.  So the measurement reads the **effective, state-dependent**
mechanism, and the ``frozen_norm`` arm is the positive control that isolates it:

    ``full``        the decoder as trained; the headline.
    ``frozen_norm`` normalization statistics pinned at the operating point.  Site-sharing
                    is near-exact here by construction, so this arm **must** score ~1.0.
                    If it does not, the measurement is broken and nothing else here counts.

The gap between the arms is the cost of the normalization, on the same footing that
``probe.jacobian_spread`` reports it for rho.

Self-calibration
----------------
Neither statistic is compared against an invented threshold.  The same site is measured
across several subjects, which gives a *within-site* spread — the measurement's own noise
floor — and the across-site spread is reported as a multiple of it.  A homogeneity ratio of
~1 means site-sharing holds to measurement precision; the ratio is the finding, not the raw
deviation.

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

ARMS = ("full", "frozen_norm")


# --------------------------------------------------------------------------------------
# spectrum: scale / shape decomposition   (pure numpy — unit-testable without a model)
# --------------------------------------------------------------------------------------


def scale_shape(J: np.ndarray, eps: float = 1e-30) -> tuple[float, np.ndarray]:
    """Split ``J``'s log singular spectrum into overall scale and zero-mean shape.

    ``shape`` is invariant to ``J -> cJ`` for any ``c > 0``, which is the point: local
    anatomy changes how *strongly* a block decodes, and only a change in the *conditioning*
    is evidence against a site-shared mechanism.
    """
    s = np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)
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


def match_columns(Ja: np.ndarray, Jb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match ``Ja``'s columns to ``Jb``'s on ``|cos|``.

    Returns ``(assignment, matched_cos)`` where ``assignment[j]`` is the column of ``Jb``
    that column ``j`` of ``Ja`` was matched to.  Absolute cosine, because the theory's
    residual gauge admits a coordinate-wise bijection, and a sign flip is one.
    """
    A = _unit_columns(Ja)
    B = _unit_columns(Jb)
    sim = np.abs(A.T @ B)
    row, col = linear_sum_assignment(-sim)
    assignment = np.empty(A.shape[1], dtype=int)
    assignment[row] = col
    return assignment, sim[row, col]


def _unit_columns(J: np.ndarray) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    n = np.linalg.norm(J, axis=0, keepdims=True)
    return J / np.maximum(n, 1e-300)


def binding_stats(assignment: np.ndarray, matched_cos: np.ndarray) -> dict:
    """Identity agreement for one (site, subject) match against the reference site."""
    d = len(assignment)
    identity = assignment == np.arange(d)
    return {
        "identity_frac": float(identity.mean()),
        "matched_cos": float(np.mean(matched_cos)),
        "matched_cos_identity": float(np.mean(matched_cos[identity])) if identity.any() else float("nan"),
        "chance": 1.0 / max(d, 1),
        "identity": identity,
    }


# --------------------------------------------------------------------------------------
# Jacobian extraction
# --------------------------------------------------------------------------------------


def block_slices(site, latent_shape, out_shape) -> tuple[slice, slice, slice]:
    """The output block ``B(u)`` for latent site ``u``, from the decoder's stride."""
    sl = []
    for a in range(3):
        stride = out_shape[a] / latent_shape[a]
        lo = int(np.floor(site[a] * stride))
        hi = max(int(np.floor((site[a] + 1) * stride)), lo + 1)
        sl.append(slice(lo, min(hi, out_shape[a])))
    return tuple(sl)


def block_jacobian(fn, z: torch.Tensor, site, block, n_channels: int, chunk: int = 16):
    """``d x|B(u) / d z(u)`` by exact JVPs, plus the fraction of response energy inside B(u).

    One basis tangent per latent channel, batched ``chunk`` at a time along the batch axis
    (every normalization here is per sample, so batch elements stay independent).

    ``block_energy_fraction`` is free from the same JVPs: the share of ``|d x / d z(u)|^2``
    that lands inside ``B(u)``.  It is *not* the A4 ratio — A4 bounds what leaks *into*
    ``B(u)`` from other sites — but it is the same phenomenon seen from the other side, and
    a low value means the block is smaller than the decoder's reach.
    """
    import torch

    from probe.jacobian_spread import _jvp

    cols, inside, total = [], 0.0, 0.0
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
        inside += float((blk**2).sum())
        cols.append(blk.reshape(n, -1).cpu().numpy().T)
    J = np.concatenate(cols, axis=1)
    return J, (inside / total if total > 0 else float("nan"))


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


def measure_arm(fn, z, sites, latent_shape, out_shape, n_channels, chunk):
    """Every site's block Jacobian for one subject, one arm."""
    out = {}
    for site in sites:
        blk = block_slices(site, latent_shape, out_shape)
        J, frac = block_jacobian(fn, z, site, blk, n_channels, chunk=chunk)
        out[site] = {"J": J, "block_energy_fraction": frac}
    return out


def reduce_arm(per_subject: list[dict], sites, dead_thresh: float, n_refs: int) -> dict:
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

    # --- spectra ------------------------------------------------------------------
    scales = np.zeros((n_subj, len(sites)))
    shapes = np.zeros((n_subj, len(sites), n_live))
    for si in range(n_subj):
        for ui, u in enumerate(sites):
            scales[si, ui], shapes[si, ui] = scale_shape(per_subject[si][u]["J"][:, live])

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
        chan_hits = np.zeros(n_live)
        for si in range(n_subj):
            Jr = per_subject[si][ref]["J"][:, live]
            for ui, u in enumerate(sites):
                a, c = match_columns(per_subject[si][u]["J"][:, live], Jr)
                st = binding_stats(a, c)
                ident[si, ui] = st["identity_frac"]
                mcos[si, ui] = st["matched_cos"]
                chan_hits += st["identity"]
        per_ref.append(
            {
                "reference_site": list(ref),
                "identity_frac_median": float(np.median(ident.mean(axis=0))),
                "identity_frac_mean": float(ident.mean()),
                "matched_cos_median": float(np.median(mcos.mean(axis=0))),
                "per_site_identity": ident.mean(axis=0),
                "channel_stability": chan_hits / (n_subj * len(sites)),
            }
        )

    # Same site, different subjects: the binding measurement's own noise floor.  Site
    # identity is held fixed, so anything below 1.0 here is measurement noise, not drift.
    same_site = []
    if n_subj > 1:
        for ui, u in enumerate(sites):
            for si in range(1, n_subj):
                a, c = match_columns(per_subject[si][u]["J"][:, live], per_subject[0][u]["J"][:, live])
                same_site.append(binding_stats(a, c)["identity_frac"])

    return {
        "n_live_channels": n_live,
        "live_mask": live,
        "dead_channels": [int(i) for i in np.nonzero(~live)[0]],
        "homogeneity": {
            "dev_across_median": across,
            "dev_within_sigma": sigma,
            "dev_within_median": float(floor),
            "ratio": float(ratio),
            "per_site_dev": dev_across,
            "scale_log_range": [float(scales.mean(axis=0).min()), float(scales.mean(axis=0).max())],
        },
        "binding": {
            "primary": per_ref[0],
            "reference_sensitivity": [r["identity_frac_median"] for r in per_ref],
            "chance": 1.0 / n_live,
            "same_site_null": float(np.median(same_site)) if same_site else float("nan"),
        },
        "block_energy_fraction": float(
            np.median([per_subject[s][u]["block_energy_fraction"] for s in range(n_subj) for u in sites])
        ),
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


def build_verdict(report: dict, tol: float, bind_tol: float) -> dict:
    """What may be claimed, and where."""
    full = report["arms"]["full"]
    ctrl = report["arms"].get("frozen_norm")
    lines, ok = [], True

    if ctrl is not None:
        c_bind = ctrl["binding"]["primary"]["identity_frac_median"]
        c_ratio = ctrl["homogeneity"]["ratio"]
        if c_bind < 0.95:
            ok = False
            lines.append(
                f"CONTROL FAILED: frozen_norm binding is {c_bind:.3f}, expected ~1.0. "
                "Site-sharing is near-exact once normalization statistics are pinned, so a "
                "low value here means the measurement is wrong, not the model. Nothing below counts."
            )
        else:
            lines.append(f"control ok: frozen_norm binding {c_bind:.3f}, homogeneity ratio {c_ratio:.2f}")

    ratio = full["homogeneity"]["ratio"]
    bind = full["binding"]["primary"]["identity_frac_median"]
    lines.append(f"full: homogeneity ratio {ratio:.2f}x the measurement floor; binding {bind:.3f}")

    if ratio <= tol:
        lines.append(
            f"A1/A4 spectra are homogeneous to within {tol:.1f}x measurement noise — Section 5 may be claimed on the measured region."
        )
    else:
        lines.append(
            f"Spectra are NOT homogeneous ({ratio:.2f}x > {tol:.1f}x). Site-sharing degrades across "
            "Lambda; restrict every downstream identifiability number to the validity map."
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

    return {"ok": ok, "lines": lines}


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
    ap.add_argument("--skip-control", action="store_true", help="Skip the frozen_norm arm (not recommended).")
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
    arms = ARMS[:1] if cli.skip_control else ARMS
    report["arms"] = {}
    for arm in arms:
        print("\n" + "=" * 88)
        print(f"ARM  {arm}")
        print("=" * 88)
        per_subject = []
        for si, sub in enumerate(subjects):
            dec = build_arm(decoder, None, "frozen" if arm == "frozen_norm" else "live", sub["z"], sub["style"])
            fn = make_fn(dec, sub["style"])
            per_subject.append(measure_arm(fn, sub["z"], sites, latent_shape, out_shape, n_channels, cli.chunk))
            print(f"  subject {si + 1}/{len(subjects)} done")
        red = reduce_arm(per_subject, sites, cli.dead_thresh, cli.n_refs)

        h, b = red["homogeneity"], red["binding"]
        print(f"\n  live channels           {red['n_live_channels']}/{n_channels}")
        print(f"  block energy fraction   {red['block_energy_fraction']:.3f}  (share of |dx/dz(u)|^2 inside B(u))")
        print("\n  SPECTRAL HOMOGENEITY (shape of the log spectrum; scale excluded by construction)")
        print(f"    across-site deviation {h['dev_across_median']:.4f} nats  (subject-mean shape vs the median site)")
        print(f"    per-subject sigma     {h['dev_within_sigma']:.4f} nats  (same site, across subjects)")
        print(f"    floor for the means   {h['dev_within_median']:.4f} nats  (sigma / sqrt(n_subjects))")
        print(f"    ratio                 {h['ratio']:.2f}x   (1.0 = homogeneous to measurement precision)")
        print("\n  BINDING (Hungarian match of J_u columns to the reference site; identity expected)")
        print(f"    identity fraction     {b['primary']['identity_frac_median']:.3f}   (chance {b['chance']:.3f})")
        print(f"    matched |cos|         {b['primary']['matched_cos_median']:.3f}")
        print(f"    same-site null        {b['same_site_null']:.3f}   (measurement floor; 1.0 is perfect)")
        print(f"    reference sensitivity {['%.3f' % v for v in b['reference_sensitivity']]}")

        red["per_site_dev_map"] = to_map(h["per_site_dev"], sites, latent_shape)
        red["per_site_identity_map"] = to_map(b["primary"]["per_site_identity"], sites, latent_shape)
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
