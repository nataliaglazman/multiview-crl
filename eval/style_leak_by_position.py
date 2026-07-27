"""Does the CONTENT representation leak per-view STYLE, and at which coverage stratum?

Settles the near-empty question: when a masked patch run scores near-perfect top-1
at near-empty / brain-edge positions (positions whose content is near-identical
across subjects), is that separation genuine edge anatomy, or has the content
subspace absorbed the per-view style (gain / bias / noise_sigma)?

Mechanism this exploits: gain/bias/noise_sigma are GLOBAL per-sample intensity
transforms, so the background is their cleanest readout -- there is no anatomy
competing as probe noise (the same argument as background_leak_diagnostic, asked of
style instead of content). A content descriptor that preserves background intensity
statistics therefore leaks gain/bias trivially. So the decisive comparison, per
stratum, is: content-rep -> style R^2 (the leak) vs content-rep -> content R^2
(genuine signal), both judged against the content<->style confound floor.

Readouts per (view, stratum):
  style R^2    content-rep -> that view's z_style (gain/bias/noise)   = THE LEAK
  content R^2  content-rep -> z_content                              = genuine signal
  ceiling      raw patch intensity at those positions -> z_style     = style available there
  floor        GT z_content -> z_style (per view, stratum-independent) = confound baseline
Per-position max/p90 within a stratum guard the ~10x region-averaging dilution that
makes edge-concentrated leak read ~0 in a stratum mean (see the caveat in
background_leak_diagnostic / recovered-factors-are-global-scalars).

Reads dead too, but dead is dropped from the loss in a masked run (main_multimodal
foreground keep-rule), so trust the verdict on near-empty / mixed / pure.

Usage:
  python -m eval.style_leak_by_position --run-dir results/synthetic/<run> --num-samples 800
  python -m eval.style_leak_by_position --run-dir <run> --level 0 --grid 8 8 8 --no-per-position
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F

from eval.dci import STYLE_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2
from eval.patch_background_diagnostic import stratify

logger = logging.getLogger(__name__)

STRATA = ("dead", "near-empty", "mixed", "pure")
TRAINED_STRATA = ("near-empty", "mixed", "pure")  # dead is dropped by the keep-rule


def extract(model, dataset, args, device, level, grid, batch_size):
    """Content-channel patches, raw patch intensity, coverage, and GT latents.

    Mirrors patch_background_diagnostic.extract_patches_with_coverage (same content
    selection via the level's Gumbel mask, no foreground drop) but accumulates over
    the loader and also returns the per-view style / content ground truth.

    Returns
      hz    (V, N, C', P)  content-channel features
      raw   (V, N, P)      mean input intensity per patch position
      frac  (V*N, P)       brain fraction per position (mirrors training's keep input)
      zc    (N, n_c)       shared content factors
      zs    (V, N, n_s)    per-view style factors (view 0 <- z_style_v1, ...)
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hz_b, raw_b, frac_b, zc_b = [], [], [], []
    zs_b = None
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
            soft_masks = out[6] if isinstance(out, tuple) and len(out) > 6 else {}
            feat = feats[level]
            if feat.dim() == 2:
                raise SystemExit(f"level {level} is GAP-pooled (no patch axis); this diagnostic needs patch pooling")
            hz = feat.reshape(n_views, -1, *feat.shape[1:])  # (V, B, C, P)
            if isinstance(soft_masks, dict) and level in soft_masks:
                mk = soft_masks[level]
                mk = mk[0] if isinstance(mk, tuple) else mk
                content_idx = torch.where(mk.bool())[-1]
            else:
                content_idx = torch.as_tensor(args.content_indices[0], device=hz.device)
            hz = hz[:, :, content_idx, :]  # (V, B, C', P)
            frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)  # (V*B, P)
            raw = F.adaptive_avg_pool3d(x, tuple(grid)).flatten(1)  # (V*B, P)

        B = hz.shape[1]
        hz_b.append(hz.float().cpu().numpy())
        raw_b.append(raw.reshape(n_views, B, -1).float().cpu().numpy())
        frac_b.append(frac.cpu().numpy())
        gt = batch["gt_latents"]
        zc_b.append(np.asarray(gt["z_content"]).reshape(gt["z_content"].shape[0], -1))
        views_style = [gt[k] for k in ("z_style_v1", "z_style_v2")[:n_views]]
        vs = [np.asarray(s).reshape(s.shape[0], -1) for s in views_style]
        zs_b = [[] for _ in vs] if zs_b is None else zs_b
        for i, s in enumerate(vs):
            zs_b[i].append(s)

    hz = np.concatenate(hz_b, axis=1)  # (V, N, C', P)
    raw = np.concatenate(raw_b, axis=1)  # (V, N, P)
    # frac was cat'd per batch as (V*B, P); recover (V*N, P) so stratify's any/mean
    # over the sample axis mirrors training's (frac >= thresh).any(dim=0) keep-rule.
    V, N = hz.shape[0], hz.shape[1]
    frac = np.concatenate([f.reshape(V, -1, f.shape[-1]) for f in frac_b], axis=1).reshape(V * N, -1)
    zc = np.concatenate(zc_b, axis=0)  # (N, n_c)
    zs = np.stack([np.concatenate(col, axis=0) for col in zs_b], axis=0)  # (V, N, n_s)
    return hz, raw, frac, zc, zs


def readout(X, sel):
    """Mean feature over selected positions -> (N, C). None if the stratum is empty."""
    if int(sel.sum()) == 0:
        return None
    return X[:, :, sel].mean(axis=2)


def r2(X, y):
    return float("nan") if X is None else cv_probe_r2(X, y)["mean"]


def per_position_style(Xv, y_style, sel, cap, rng):
    """Distribution of per-position style R^2 within a stratum (dilution guard).

    Probes each position's descriptor -> style separately (light CV), so an
    edge-concentrated leak that a stratum mean would dilute still shows up in p90/max.
    """
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return {"mean": float("nan"), "p90": float("nan"), "max": float("nan"), "n": 0}
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    scores = [cv_probe_r2(Xv[:, :, p], y_style, n_splits=3, seeds=(0,))["mean"] for p in idx]
    a = np.array(scores, dtype=float)
    return {
        "mean": float(np.nanmean(a)),
        "p90": float(np.nanpercentile(a, 90)),
        "max": float(np.nanmax(a)),
        "n": len(idx),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument(
        "--grid", type=int, nargs=3, default=None, help="Override the run's patch grid (e.g. --grid 8 8 8)."
    )
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-per-position", action="store_true", help="Skip the per-position dilution guard (faster).")
    ap.add_argument(
        "--pp-cap", type=int, default=160, help="Max positions probed per stratum in the per-position pass."
    )
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)

    grid = tuple(cli.grid) if cli.grid else tuple(getattr(args, "patch_grid", (8, 8, 8)))
    thresh = getattr(args, "patch_foreground_thresh", 0.05)
    ds = build_synthetic_test_set(args, cli.num_samples)
    hz, raw, frac, zc, zs = extract(model, ds, args, device, cli.level, grid, cli.batch_size)
    V, N, C, P = hz.shape
    n_s = zs.shape[2]
    style_names = STYLE_FACTOR_NAMES[:n_s]

    strata = {k: v.cpu().numpy() for k, v in stratify(torch.as_tensor(frac), thresh).items()}
    rng = np.random.default_rng(0)

    print(f"\nrun: {cli.run_dir}")
    print(f"level {cli.level} | grid {list(grid)} -> P={P} | N={N} | content channels={C} | views={V}")
    print(f"style factors: {style_names} | fg-thresh={thresh}")
    print(f"positions per stratum: " + "  ".join(f"{s}={int(strata[s].sum())}" for s in STRATA))

    for v in range(V):
        y_style = zs[v]  # (N, n_s)
        floor = r2(zc, y_style)  # content<->style confound, stratum-independent
        print(f"\n=== view {v} (style target: z_style_v{v + 1}) ===")
        print(f"confound floor  GT z_content -> style  R2 = {floor:.3f}   (leak is only real ABOVE this)")
        hdr = f"{'stratum':<12}{'n':>5}{'style R2':>10}" + "".join(f"{nm[:6]:>8}" for nm in style_names)
        hdr += f"{'content':>9}{'ceiling':>9}"
        if not cli.no_per_position:
            hdr += f"{'pp_mean':>9}{'pp_p90':>8}{'pp_max':>8}"
        print(hdr)
        print("-" * len(hdr))
        res = {}
        for s in STRATA:
            sel = strata[s]
            R = readout(hz[v], sel)
            pp = None if cli.no_per_position else per_position_style(hz[v], y_style, sel, cli.pp_cap, rng)
            res[s] = {
                "n": int(sel.sum()),
                "style": r2(R, y_style),
                "per_fac": [r2(R, y_style[:, fi]) for fi in range(n_s)],
                "content": r2(R, zc),
                "ceil": float("nan") if int(sel.sum()) == 0 else r2(raw[v][:, sel], y_style),
                "pp": pp,
            }
            row = f"{s:<12}{res[s]['n']:>5}{res[s]['style']:>10.3f}" + "".join(f"{x:>8.3f}" for x in res[s]["per_fac"])
            row += f"{res[s]['content']:>9.3f}{res[s]['ceil']:>9.3f}"
            if pp is not None:
                row += f"{pp['mean']:>9.3f}{pp['p90']:>8.3f}{pp['max']:>8.3f}"
            print(row)

        # verdict on the trained strata only (dead is dropped from the loss in a masked
        # run).  Key on style - floor: style decodable above the confound IS leak, whether
        # or not genuine content also coexists.  pp_max escalates an edge-concentrated leak
        # the stratum mean would dilute.
        print("verdict:")
        base = max(floor, 0.0)
        for s in TRAINED_STRATA:
            r = res[s]
            if r["n"] == 0:
                continue
            pp_max = r["pp"]["max"] if r["pp"] is not None else float("nan")
            leak_mean = max(r["style"], 0.0) - base
            leak_edge = (max(pp_max, 0.0) - base) if pp_max == pp_max else float("-inf")
            also = f" (genuine content R2={r['content']:.2f} coexists)" if r["content"] > 0.1 else ""
            if leak_mean > 0.1:
                print(
                    f"  {s:<11}: STYLE LEAK -- style R2={r['style']:.2f} vs floor {floor:.2f}. The content "
                    f"channels carry this view's gain/bias here, so top-1 is partly style, not anatomy.{also}"
                )
            elif leak_edge > 0.2:
                print(
                    f"  {s:<11}: STYLE LEAK (edge) -- stratum mean style R2={r['style']:.2f} is near floor "
                    f"{floor:.2f}, but per-position max {pp_max:.2f} shows the leak is concentrated at a few "
                    f"positions the mean dilutes.{also}"
                )
            elif r["content"] > 0.1:
                print(
                    f"  {s:<11}: GENUINE CONTENT -- style R2={r['style']:.2f} ~ floor {floor:.2f} (no leak); "
                    f"content R2={r['content']:.2f}. The top-1 here is anatomy."
                )
            else:
                print(
                    f"  {s:<11}: UNINFORMATIVE -- style R2={r['style']:.2f} ~ floor {floor:.2f} and content "
                    f"R2={r['content']:.2f} both low. Little decodable either way."
                )


if __name__ == "__main__":
    main()
