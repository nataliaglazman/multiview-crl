"""Is the entropy term's uniformity achieved PER POSITION, or only after pooling?

The entropy term exists for exactly one purpose: Yao et al. (ICLR 2024) Lemma C.6
establishes invertibility of the encoder r_k by the chain

    t_k . r_k uniform on the |S_k|-cube  ->  (Zimmermann Prop. 5)  t_k . r_k bijective
                                         ->  r_k injective.

Prop. 5 only yields bijectivity between manifolds of EQUAL dimension, which is why
Defn 3.6 makes t_k dimension-preserving.  ``split_infonce_loss`` currently pools the
projected features over positions before the entropy term
(``training/losses.py`` ``hz_entropy.mean(-1)``), so the map it constrains is
R^(d x P) -> R^d.  That is dimension-REDUCING, Prop. 5 does not apply to it, and
uniformity is therefore only ever enforced on the pooled vector.

Under patch-as-view -- view (k, p) observing the latents in position p's receptive
field -- what Lemma C.6 actually needs is uniformity of the per-position map.  This
script measures whether that already holds as a side effect of the pooled objective.
It needs no retraining: run it on an existing checkpoint.

If per-position uniformity is already close to the pooled value, the pooling gap is
formal and a per-position entropy term buys nothing measurable.  If it is much worse,
that is the motivating number for changing the loss.

Note this reads the FULL representation, not the content block: eq. (3.3) applies no
content selector to the entropy term (Lemma C.5), and restricting uniformity to
content would leave the style channels free to be lossy and break the chain.

Three readouts, each computed per position AND on the pooled vector so they are
directly comparable:

  eff_rank   participation ratio of the feature covariance, normalised by d.
             The strongest of the three: rank deficiency at a position is a direct
             witness that t.r cannot be injective onto a d-dim cube there.
  unif       Wang-Isola uniformity  log E[exp(-2||u-v||^2)]  on L2-normalised
             features (lower = more uniform).  Reported against a same-(N, d)
             uniform-on-sphere reference, without which the raw value means nothing.
  ks         max-over-dimensions KS distance from uniform on the tanh output mapped
             to [0,1].  This is Defn 3.6's cube read literally; `unif` is the sphere
             surrogate the loss actually optimises (it L2-normalises first, dropping
             the radial dimension).  Reported together because they can disagree.

             READ `ks` ONLY IN ONE DIRECTION.  The head ends BatchNorm(affine=False) ->
             Tanh, and tanh approximates the probability integral transform of a
             unit-variance gaussian: tanh(N(0,1)) already sits at KS ~ 0.10 from
             uniform with no training at all.  Near-uniform marginals are therefore
             largely an artefact of the architecture, and a LOW ks is not evidence the
             entropy term achieved anything.  A HIGH ks is still informative -- it means
             saturation or bimodality (saturated tanh reads ~0.35).  The verdict keys on
             eff_rank and unif for this reason.

Positions are stratified by brain coverage: a dead position's features are near
constant, so it fails uniformity trivially and uninterestingly.  Trust the verdict on
the trained strata.

Usage:
  python -m eval.entropy_uniformity --run-dir results/synthetic/<run> --num-samples 800
  python -m eval.entropy_uniformity --run-dir <run> --level 0 --grid 8 8 8
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.patch_background_diagnostic import stratify

logger = logging.getLogger(__name__)

STRATA = ("dead", "near-empty", "mixed", "pure")
TRAINED_STRATA = ("near-empty", "mixed", "pure")


def build_entropy_head(args, model, level):
    """Rebuild the ``--contrastive-proj-mode entropy`` head for one level.

    Mirrors the construction in ``training/main_multimodal.py`` exactly -- the width
    is ``vqvae_hidden_channels`` (|S_k|, the FULL view-specific latent width), not the
    content width, because Defn 3.6 sizes t_k by |S_k| and eq. (3.3) applies no
    content selector to the entropy term.
    """
    cc = model.content_channels_per_level.get(level)
    full = getattr(args, "vqvae_hidden_channels", None) or cc
    hidden = getattr(args, "contrastive_proj_hidden", 256)
    return nn.Sequential(
        nn.Linear(full, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, full),
        nn.BatchNorm1d(full, affine=False),
        nn.Tanh(),
    )


def load_entropy_head(run_dir, checkpoint, args, model, level, device):
    """Load the trained t_k, or return None when the run trained no projection.

    ``load_model_from_run_dir`` builds only the VQVAE, so the head's parameters land
    in the checkpoint's *unexpected* keys and are silently discarded.  Measuring
    uniformity through a freshly initialised head would be meaningless, so the weights
    are pulled from the checkpoint here and loaded strictly.
    """
    ckpt_path = checkpoint or os.path.join(run_dir, "vqvae_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("encoders", ckpt)
    prefix = f"_contrastive_proj_heads.L{level}."
    head_sd = {}
    for k, v in sd.items():
        kk = k.removeprefix("module.")
        if kk.startswith(prefix):
            head_sd[kk.removeprefix(prefix)] = v
    if not head_sd:
        return None
    head = build_entropy_head(args, model, level)
    head.load_state_dict(head_sd, strict=True)
    return head.to(device).eval()


def apply_head(head, hz):
    """Apply t_k per position.  ``hz`` is (V, N, C, P) -> (V, N, C, P).

    Mirrors ``_project_contrastive_content``: the channel axis moves last and every
    leading axis is folded into the batch, because BatchNorm1d would otherwise read
    (V, N, C) as (N=V, C=N, L=C) and normalise the wrong axis entirely.
    """
    x = hz.permute(0, 1, 3, 2)
    shape = x.shape
    out = head(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)
    return out.permute(0, 1, 3, 2).contiguous()


def extract(model, dataset, device, level, grid, batch_size):
    """Full per-position representations and brain coverage.

    Returns
      hz    (V, N, C, P)  full encoder features -- NOT sliced to content channels
      frac  (V*N, P)      brain fraction per position (mirrors training's keep input)
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hz_b, frac_b = [], []
    model.eval()
    for batch in loader:
        imgs = batch["image"]
        n_views = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        masks = batch.get("mask")
        if masks is None:
            raise SystemExit("dataset yields no 'mask' key; coverage stratification needs brain masks")
        m = torch.cat(masks, 0).to(device).float()
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=n_views, patch_grid=grid)
            feats = out[2] if isinstance(out, tuple) else [out]
            feat = feats[level]
            if feat.dim() == 2:
                raise SystemExit(f"level {level} is GAP-pooled (no patch axis); this diagnostic needs patch pooling")
            hz = feat.reshape(n_views, -1, *feat.shape[1:])  # (V, B, C, P)
            frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)  # (V*B, P)
        hz_b.append(hz.float().cpu())
        frac_b.append(frac.cpu().numpy())

    hz = torch.cat(hz_b, dim=1)  # (V, N, C, P)
    V, N = hz.shape[0], hz.shape[1]
    frac = np.concatenate([f.reshape(V, -1, f.shape[-1]) for f in frac_b], axis=1).reshape(V * N, -1)
    return hz, frac


def eff_rank(X):
    """Participation ratio of the covariance of (N, d), normalised to (0, 1].

    A bijection onto a d-dim cube requires full rank, so a value well below 1 is a
    direct witness of non-injectivity rather than merely a distributional mismatch.

    Computed post-tanh, on t.r itself.  The head's BatchNorm rescales each channel,
    which shifts this soft rank, but per-channel scaling is diagonal and cannot
    manufacture independence: genuinely collapsed directions stay collapsed.
    """
    Xc = X - X.mean(0, keepdim=True)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    s1, s2 = ev.sum(), (ev**2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1 / s2) / X.shape[1])


def wang_isola(X):
    """log E[exp(-2||u-v||^2)] over distinct pairs, u = X / ||X||.  Lower = more uniform."""
    u = F.normalize(X.double(), dim=-1, eps=1e-12)
    sim = u @ u.T
    # ||u - v||^2 = 2 - 2 u.v  =>  exp(-2||u-v||^2) = exp(-4 + 4 u.v)
    logits = (-4.0 + 4.0 * sim).flatten()
    n = X.shape[0]
    keep = ~torch.eye(n, dtype=torch.bool).flatten()
    return float(torch.logsumexp(logits[keep], dim=0) - np.log(keep.sum().item()))


def ks_uniform(X):
    """Max-over-dimensions KS distance from Uniform[0,1] on the tanh output.

    Defn 3.6 asks for uniformity on the cube; the tanh codomain (-1,1) is mapped by
    (x+1)/2, an affine reparameterisation that leaves the max-entropy argument intact.
    """
    U = ((X.double() + 1.0) / 2.0).clamp(0.0, 1.0)
    n = U.shape[0]
    Us, _ = torch.sort(U, dim=0)
    grid = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1) / n
    d_plus = (grid - Us).max(dim=0).values
    d_minus = (Us - (grid - 1.0 / n)).max(dim=0).values
    return float(torch.maximum(d_plus, d_minus).max())


def sphere_reference(n, d, seed=0):
    """Wang-Isola value for n genuinely uniform points on S^(d-1)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    return wang_isola(X)


def summarise(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "p10", "p90", "min", "max")}
    return {
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, nargs=3, default=None, help="Override the run's patch grid.")
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cap", type=int, default=512, help="Samples used for the pairwise uniformity term.")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--raw-if-no-head",
        action="store_true",
        help="If the run trained no t_k, measure the raw representation instead of exiting.",
    )
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)

    grid = tuple(cli.grid) if cli.grid else tuple(getattr(args, "patch_grid", (8, 8, 8)))
    thresh = getattr(args, "patch_foreground_thresh", 0.05)

    head = load_entropy_head(cli.run_dir, cli.checkpoint, args, model, cli.level, device)
    proj_mode = getattr(args, "contrastive_proj_mode", "head")
    if head is None:
        msg = (
            f"no trained t_k in this checkpoint (contrastive_proj_mode={proj_mode!r}). "
            "Uniformity is a property of t.r, so without t there is no entropy-term claim to check."
        )
        if not cli.raw_if_no_head:
            raise SystemExit(f"ERROR: {msg}\nPass --raw-if-no-head to measure the raw representation instead.")
        logger.warning("%s Measuring the RAW representation -- this is a weaker, different reading.", msg)
    elif proj_mode != "entropy":
        logger.warning(
            "checkpoint carries a t_k but contrastive_proj_mode=%r, not 'entropy'. "
            "The head may not have been trained by the entropy term in this run.",
            proj_mode,
        )

    ds = build_synthetic_test_set(args, cli.num_samples)
    hz, frac = extract(model, ds, device, cli.level, grid, cli.batch_size)
    if head is not None:
        with torch.no_grad():
            hz = apply_head(head, hz.to(device)).cpu()
    V, N, C, P = hz.shape

    strata = {k: np.asarray(v.cpu().numpy()) for k, v in stratify(torch.as_tensor(frac), thresh).items()}
    n_pair = min(cli.cap, N)
    ref = sphere_reference(n_pair, C)

    print(f"\nrun: {cli.run_dir}")
    print(f"level {cli.level} | grid {list(grid)} -> P={P} | N={N} | width d={C} | views={V}")
    print(f"t_k: {'trained head loaded' if head is not None else 'NONE (raw representation)'} | fg-thresh={thresh}")
    print(f"positions per stratum: " + "  ".join(f"{s}={int(strata[s].sum())}" for s in STRATA))
    print(f"uniform-on-sphere reference (n={n_pair}, d={C}): unif = {ref:.4f}")
    print("eff_rank in (0,1]: 1.0 = full rank.  unif: lower = more uniform, compare to reference.")
    print("ks: 0 = exactly uniform per dimension. LOW ks is uninformative here (BN->tanh gives ~0.10")
    print("    untrained); only HIGH ks means anything, and it means saturation. Judge on eff_rank/unif.")

    for v in range(V):
        X = hz[v]  # (N, C, P)
        pooled = X.mean(-1)  # what the loss actually constrains
        # ks maps (-1,1) -> [0,1] and only means anything on a tanh-bounded t output;
        # on raw features the clamp would silently saturate and report nonsense.
        p_rank, p_unif = eff_rank(pooled), wang_isola(pooled[:n_pair])
        p_ks = ks_uniform(pooled) if head is not None else float("nan")

        print(f"\n=== view {v} ===")
        print(f"{'POOLED (what the loss constrains)':<34}eff_rank={p_rank:>6.3f}  unif={p_unif:>8.4f}  ks={p_ks:>6.3f}")
        hdr = f"{'stratum':<12}{'n':>5}" + f"{'eff_rank':>26}{'unif':>26}{'ks':>20}"
        print(hdr)
        print(f"{'':<12}{'':>5}" + f"{'mean [p10, p90]':>26}{'mean [p10, p90]':>26}{'mean [max]':>20}")
        print("-" * len(hdr))

        res = {}
        for s in STRATA:
            sel = np.where(strata[s])[0]
            if len(sel) == 0:
                res[s] = None
                print(f"{s:<12}{0:>5}")
                continue
            ranks, unifs, kss = [], [], []
            for p in sel:
                Xp = X[:, :, p]
                ranks.append(eff_rank(Xp))
                unifs.append(wang_isola(Xp[:n_pair]))
                kss.append(ks_uniform(Xp) if head is not None else float("nan"))
            r_s, u_s, k_s = summarise(ranks), summarise(unifs), summarise(kss)
            res[s] = {"n": len(sel), "rank": r_s, "unif": u_s, "ks": k_s}
            print(
                f"{s:<12}{len(sel):>5}"
                f"{r_s['mean']:>10.3f} [{r_s['p10']:>5.3f}, {r_s['p90']:>5.3f}]"
                f"{u_s['mean']:>10.4f} [{u_s['p10']:>5.2f}, {u_s['p90']:>5.2f}]"
                f"{k_s['mean']:>10.3f} [{k_s['max']:>5.3f}]"
            )

        print("verdict:")
        fg = [res[s] for s in TRAINED_STRATA if res[s] is not None]
        if not fg:
            print("  no trained-stratum positions; nothing to judge.")
            continue
        w = sum(r["n"] for r in fg)
        rank_fg = sum(r["rank"]["mean"] * r["n"] for r in fg) / w
        unif_fg = sum(r["unif"]["mean"] * r["n"] for r in fg) / w
        # Distance from the uniform reference is the comparable quantity: the raw
        # Wang-Isola value shifts with n and d, the gap to reference does not.
        gap_pos, gap_pooled = unif_fg - ref, p_unif - ref
        print(f"  foreground per-position: eff_rank={rank_fg:.3f}  unif-gap={gap_pos:+.4f}")
        print(f"  pooled:                  eff_rank={p_rank:.3f}  unif-gap={gap_pooled:+.4f}")

        if rank_fg < 0.5:
            print(
                f"  RANK DEFICIENT -- per-position eff_rank={rank_fg:.2f} means t.r spans well under half of "
                f"its {C} dimensions at a typical position. Prop. 5 cannot give a bijection there, so the "
                f"entropy term is NOT delivering per-position injectivity. Per-position entropy is motivated."
            )
        elif gap_pos > 2 * max(gap_pooled, 1e-6) and gap_pos > 0.05:
            print(
                f"  POOLING GAP IS REAL -- per-position uniformity ({gap_pos:+.3f} from reference) is much "
                f"worse than pooled ({gap_pooled:+.3f}). The pooled objective is not achieving per-position "
                f"uniformity as a side effect. Per-position entropy is motivated."
            )
        elif rank_fg > 0.8 and gap_pos < 0.05:
            print(
                f"  GAP IS FORMAL -- per-position eff_rank={rank_fg:.2f} and uniformity is already close to "
                f"reference ({gap_pos:+.3f}). The pooled objective achieves per-position uniformity anyway, "
                f"so switching the loss would close a theoretical hole with little measurable effect."
            )
        else:
            print(
                f"  INTERMEDIATE -- eff_rank={rank_fg:.2f}, per-position unif-gap {gap_pos:+.3f} vs pooled "
                f"{gap_pooled:+.3f}. Partial per-position uniformity; the change is defensible on theory but "
                f"the expected empirical effect is small. Check the per-stratum rows for where it concentrates."
            )


if __name__ == "__main__":
    main()
