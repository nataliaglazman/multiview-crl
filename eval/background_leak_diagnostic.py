"""Why does a factor's per-position R^2 map light up in the BACKGROUND?

Two mechanisms produce a background hotspot, and they call for opposite fixes:

  (A) GLOBAL-SUMMARY LEAKAGE. The background feature is a (noise-free) copy of a
      whole-image aggregate that happens to encode the factor -- e.g. ventricle
      size shifts the global CSF fraction, hence the mean intensity, hence the
      per-channel statistics that survive into every position. Backgrounds carry
      NO local information; they are just the cleanest readout of the global
      summary (no local anatomy competing as probe noise). This is the mechanism
      behind groupnorm-caps-gap-pooled-mcc, now asked per-position.

  (B) LOCAL SIGNAL. The background position genuinely carries factor information,
      via a receptive field reaching the structure, boundary/padding effects, or
      (for brain_size) the brain/background boundary itself moving.

Two discriminators, both validated on constructed features with known structure
(a gap-residual test was tried first and DISCARDED: the brain is ~17% of the box,
so the global mean is dominated by background and "global summary" and "background
readout" become the same vector -- the test could not tell A from B).

  1. DISTANCE SHELLS. Background R^2 in shells of increasing distance from the
     brain. FLAT across shells => a global statistic present at every position
     equally (A). DECAYING => a signal that reaches out from the brain, i.e.
     receptive-field or boundary leak (B).

  2. core | bg. In-brain core readout after residualising out the background
     readout. ~0 => even the central hotspot is just the global readout, nothing
     is genuinely localised. HIGH => there is real local in-brain encoding on top
     of whatever the background is doing.

The common real case for a central factor: shells FLAT (background is a global
readout) AND core|bg HIGH (the centre genuinely encodes it because the structure
is there). Then the background hotspot is an artefact and the centre is real.

Also reports brain_size (the boundary factor) and corr(factor, brain_size),
because a random causal graph can make a central factor inherit brain_size's
boundary hotspot.

Usage:
  python -m eval.background_leak_diagnostic --run-dir results/synthetic/<run> \
      --factor ventricle_size --num-samples 800
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2

logger = logging.getLogger(__name__)


def extract_spatial(model, dataset, args, device, level, grid, batch_size):
    """Native-resolution feature map + brain fraction per position.

    Returns X (N, C, P) view-0 features, frac (N, P) brain fraction, and a dict
    of ground-truth latents (N, d).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    Xs, fr, lat = [], [], {"z_content": []}
    model.eval()
    for batch in loader:
        imgs = batch["image"]
        n_views = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        masks = batch.get("mask")
        if masks is None:
            raise SystemExit("dataset yields no 'mask' key; background stratification needs brain masks")
        m = torch.cat(masks, 0).to(device).float()
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=n_views, patch_grid=grid)
            feats = out[2] if isinstance(out, tuple) else [out]
            f = feats[level]
            if f.dim() == 2:
                f = f.unsqueeze(-1)
            hz = f.reshape(n_views, -1, *f.shape[1:])  # (V,B,C,P)
            frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)  # (V*B, P)
            B = hz.shape[1]
        Xs.append(hz[0].float().cpu().numpy())  # view 0 only: (B, C, P)
        fr.append(frac[:B].cpu().numpy())
        gt = batch.get("gt_latents") or {}
        if "z_content" in gt:
            v = gt["z_content"]
            lat["z_content"].append(np.asarray(v).reshape(v.shape[0], -1))
    X = np.concatenate(Xs, 0)
    frac = np.concatenate(fr, 0)
    Z = np.concatenate(lat["z_content"], 0) if lat["z_content"] else None
    return X, frac, Z


def readout(X, sel):
    """Mean feature over the selected positions -> (N, C)."""
    if sel.sum() == 0:
        return None
    return X[:, :, sel].mean(axis=2)


def resid(a, b):
    """Residual of a (N,C) after linearly regressing out b (N,K)."""
    return a - LinearRegression().fit(b, a).predict(b)


def probe(F_, y, kind="ridge"):
    if F_ is None:
        return float("nan")
    return cv_probe_r2(F_, y, kind=kind)["mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--factor", default="ventricle_size", help=f"one of {CONTENT_FACTOR_NAMES}")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=None, help="feature grid per axis; default = native level resolution")
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--core-thr", type=float, default=0.9)
    ap.add_argument("--bg-thr", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)

    # Native feature resolution: probe encoder_outputs at a grid = its own spatial size.
    grid = (cli.grid,) * 3 if cli.grid else tuple(getattr(args, "patch_grid", (16, 16, 16)))
    ds = build_synthetic_test_set(args, cli.num_samples)
    X, frac, Z = extract_spatial(model, ds, args, device, cli.level, grid, cli.batch_size)
    if Z is None:
        raise SystemExit("no z_content in gt_latents")

    fi = CONTENT_FACTOR_NAMES.index(cli.factor)
    bsi = CONTENT_FACTOR_NAMES.index("brain_size")
    y = Z[:, fi]
    y_bs = Z[:, bsi]

    cov = frac.mean(axis=0)  # per-position mean coverage
    core = cov >= cli.core_thr
    bg = cov <= cli.bg_thr
    N, C, P = X.shape
    print(f"\nrun: {cli.run_dir}")
    print(f"factor: {cli.factor} | level {cli.level} | grid {list(grid)} -> P={P} | N={N} | C={C}")
    print(f"positions: core(>= {cli.core_thr})={int(core.sum())}  background(<= {cli.bg_thr})={int(bg.sum())}\n")

    f_core = readout(X, core)
    f_bg = readout(X, bg)

    r_core = probe(f_core, y)
    r_bg = probe(f_bg, y)
    r_core_given_bg = probe(resid(f_core, f_bg), y) if (f_core is not None and f_bg is not None) else float("nan")

    print(f"{'readout':<26}{cli.factor + ' R2':>18}")
    print("-" * 44)
    print(f"{'core (in-brain)':<26}{r_core:>18.3f}")
    print(f"{'background':<26}{r_bg:>18.3f}")
    print(f"{'core | background removed':<26}{r_core_given_bg:>18.3f}")

    # Distance-to-brain shells: bin background positions by distance to nearest
    # brain position, then probe each shell. Flat -> global; decaying -> RF/boundary.
    brain_pos = np.argwhere((cov >= 0.5).reshape(grid))  # (n_brain, 3) grid coords
    all_pos = np.argwhere(np.ones(grid, bool))  # (P, 3), row order matches flatten
    if len(brain_pos):
        try:
            from scipy.spatial import cKDTree

            dist = cKDTree(brain_pos).query(all_pos)[0]
        except ImportError:  # brute force: P x n_brain is small
            d2 = ((all_pos[:, None, :] - brain_pos[None, :, :]) ** 2).sum(-1)
            dist = np.sqrt(d2.min(axis=1)).astype(float)
    else:
        dist = np.full(P, np.nan)
    bg_d = dist[bg]
    edges = np.quantile(bg_d, [0.0, 1 / 3, 2 / 3, 1.0]) if len(bg_d) else [0, 1, 2, 3]
    print("\ndistance-to-brain shells (background only):")
    print(f"  {'shell':<12}{'dist range':>14}{'n':>7}{'R2':>9}")
    shell_r2 = []
    for i in range(3):
        m = bg.copy()
        d_lo, d_hi = edges[i], edges[i + 1]
        keep = (dist >= d_lo) & (dist <= d_hi if i == 2 else dist < d_hi)
        m &= keep
        rr = probe(readout(X, m), y)
        shell_r2.append(rr)
        print(
            f"  {'near->far'[:1] if False else f'shell {i}':<12}{f'[{d_lo:.1f},{d_hi:.1f})':>14}{int(m.sum()):>7}{rr:>9.3f}"
        )

    # brain_size confound
    r_bg_bs = probe(f_bg, y_bs)
    corr = float(np.corrcoef(y, y_bs)[0, 1])
    r_bg_given_bs = probe(resid(f_bg, y_bs[:, None]), y) if f_bg is not None else float("nan")
    print(f"\nbrain_size confound:")
    print(f"  background -> brain_size R2        : {r_bg_bs:.3f}")
    print(f"  corr({cli.factor}, brain_size)     : {corr:+.3f}")
    print(f"  background -> {cli.factor} | brain_size removed : {r_bg_given_bs:.3f}")

    # verdict
    valid = [s for s in shell_r2 if s == s]
    decays = len(valid) >= 2 and valid[-1] < 0.5 * max(valid[0], 1e-9) and max(valid) > 0.1
    print("\nverdict:")
    if r_bg < 0.1:
        print(f"  background R2 is low ({r_bg:.2f}); no strong hotspot to explain at this grid/level.")
    elif decays:
        print(
            f"  (B) RECEPTIVE-FIELD / BOUNDARY LEAK. Background R2 decays with distance from the brain "
            f"({'->'.join(f'{s:.2f}' for s in shell_r2)}) -> the signal reaches out from the brain, it is not a "
            "flat global readout. Smaller receptive field (lower scaling_rate / shallower level 0) is the lever."
        )
    else:
        print(
            f"  (A) GLOBAL-STATISTIC LEAKAGE. Background R2 is flat across distance shells "
            f"({'->'.join(f'{s:.2f}' for s in shell_r2)}) -> every background position reads the same whole-image "
            "statistic, not local information. The background hotspot is an artefact of a global summary "
            f"(intensity/GroupNorm stats) that {cli.factor} shifts."
        )
        if r_core_given_bg > 0.1:
            print(
                f"  The CENTRE hotspot is separately GENUINE: core|bg = {r_core_given_bg:.2f} -> real local in-brain "
                f"encoding of {cli.factor}, on top of the global background artefact."
            )
        else:
            print(
                f"  And core|bg = {r_core_given_bg:.2f}: even the centre carries no local info beyond the global "
                "readout -- the whole map is the global statistic, nothing is localised."
            )
    if abs(corr) > 0.3 and r_bg_given_bs < 0.5 * r_bg:
        print(
            f"  MEDIATED BY brain_size: corr={corr:+.2f}, and removing brain_size collapses the background readout "
            f"({r_bg_given_bs:.2f} vs {r_bg:.2f}). The hotspot is largely brain_size's boundary signal, inherited "
            "via the causal graph."
        )


if __name__ == "__main__":
    main()
