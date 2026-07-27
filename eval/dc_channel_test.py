"""Is the shortcut carried by a spatially-uniform "DC" channel?

Established so far: ventricle_size is recovered via a global composition shortcut
(data-side hist R² 0.35 == model recovery 0.35), it sits in the model's channel
MEANS (GAP 0.31), and every position -- including background -- reads it with a
flat distance profile. Those together IMPLY the carrier is a channel that is
roughly constant across space and whose level shifts with the factor. This script
tests that implication directly.

Decomposition of the level-0 feature map F (N, C, P), centred across samples:

    DC[i,c]      = mean_p F[i,c,p]              the uniform (per-sample) level
    RES[i,c,p]   = F[i,c,p] - DC[i,c]           the spatial pattern

Two readouts:

  1. **Probe split.** Predict the factor from DC alone vs from RES alone.
     DC high + RES ~0  -> the factor is a uniform level shift (a DC channel).
     RES high          -> the factor is carried by spatial structure.
     lr_asymmetry is the built-in contrast: a left-right differential CANNOT be a
     uniform shift, so it must show RES signal if the split works at all.

  2. **DC fraction per channel.**  var_i(DC[i,c]) / mean_p var_i(F[i,c,p]).
     1.0 = the channel's entire sample-to-sample variation is a uniform shift;
     0.0 = it varies spatially with no change in level. Reported for the channels
     that actually carry the factor (ranked by |corr(DC[:,c], factor)|), against
     the all-channel median as a baseline.

Usage:
  python -m eval.dc_channel_test --run-dir results/synthetic/<run> --num-samples 800
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2

logger = logging.getLogger(__name__)
FACTORS = ("ventricle_size", "brain_size", "lr_asymmetry")


def pca(X, d=128):
    X = np.asarray(X, np.float64)
    if X.shape[1] <= d:
        return X
    return PCA(n_components=d, random_state=0).fit_transform(StandardScaler().fit_transform(X))


def dc_fraction(F):
    """Per-channel share of sample-to-sample variance that is a UNIFORM level shift.

    var_i(mean_p F) / mean_p var_i(F).  ~1 = spatially uniform channel (DC),
    ~0 = spatial pattern change with no level shift.
    """
    dc = F.mean(axis=2)  # (N, C)
    num = dc.var(axis=0)  # (C,)
    den = F.var(axis=0).mean(axis=1)  # (C,)
    return num / np.maximum(den, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=5, help="How many carrier channels to report.")
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from torch.utils.data import DataLoader

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    model.eval()

    grid = (cli.grid,) * 3
    ds = build_synthetic_test_set(args, cli.num_samples)
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)

    feats, cov_acc, Z = [], [], []
    for batch in loader:
        imgs = batch["image"]
        nv = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=nv, patch_grid=grid)
            f = (out[2] if isinstance(out, tuple) else out)[cli.level]
            f = f.reshape(nv, -1, *f.shape[1:])[0]  # view 0: (B, C, P)
        feats.append(f.cpu().numpy().astype(np.float32))
        m = batch.get("mask")
        if m is not None:
            mm = torch.cat(m, 0).to(device).float()
            cov_acc.append(Fn.adaptive_avg_pool3d(mm, grid).flatten(1)[: f.shape[0]].cpu().numpy())
        v = batch["gt_latents"]["z_content"]
        Z.append(np.asarray(v).reshape(v.shape[0], -1))

    F = np.concatenate(feats, 0)  # (N, C, P)
    Z = np.concatenate(Z, 0)
    cov = np.concatenate(cov_acc, 0).mean(axis=0) if cov_acc else None
    N, C, P = F.shape

    Fc = F - F.mean(axis=0, keepdims=True)  # centre across samples
    DC = Fc.mean(axis=2)  # (N, C)
    RES = Fc - DC[:, :, None]  # (N, C, P)
    dcf = dc_fraction(F)

    print(f"\nrun: {cli.run_dir} | level {cli.level} | grid {list(grid)} | N={N} C={C} P={P}")
    print(
        f"DC fraction across all channels: median {np.median(dcf):.3f}  "
        f"[p10 {np.percentile(dcf,10):.3f}, p90 {np.percentile(dcf,90):.3f}]"
    )

    print("\n1) probe split — which component carries the factor:")
    hdr = f"{'factor':<16}{'DC only':>10}{'RES only':>10}{'full':>10}"
    print(hdr + "\n" + "-" * len(hdr))
    res_pca = pca(RES.reshape(N, -1))
    full_pca = pca(Fc.reshape(N, -1))
    rows = {}
    for fname in FACTORS:
        fi = CONTENT_FACTOR_NAMES.index(fname)
        y = Z[:, fi]
        r_dc = cv_probe_r2(DC, y, kind="ridge")["mean"]
        r_res = cv_probe_r2(res_pca, y, kind="ridge")["mean"]
        r_full = cv_probe_r2(full_pca, y, kind="ridge")["mean"]
        rows[fname] = (r_dc, r_res)
        print(f"{fname:<16}{r_dc:>10.3f}{r_res:>10.3f}{r_full:>10.3f}")

    print("\n2) DC fraction of the channels that carry each factor:")
    hdr = f"{'factor':<16}{'top channels (|corr| w/ factor)':<34}{'their DC frac':>14}"
    print(hdr + "\n" + "-" * len(hdr))
    for fname in FACTORS:
        fi = CONTENT_FACTOR_NAMES.index(fname)
        y = Z[:, fi]
        yc = (y - y.mean()) / (y.std() + 1e-12)
        corr = np.abs((DC - DC.mean(0)) / (DC.std(0) + 1e-12) * yc[:, None]).mean(axis=0)
        top = np.argsort(-corr)[: cli.top_k]
        chans = ", ".join(f"c{c}({corr[c]:.2f})" for c in top)
        print(f"{fname:<16}{chans:<34}{np.mean(dcf[top]):>14.3f}")

    print("\nverdict:")
    v_dc, v_res = rows["ventricle_size"]
    l_dc, l_res = rows["lr_asymmetry"]
    if v_res < 0.3 * max(v_dc, 1e-9):
        print(f"  ventricle_size is a DC (spatially uniform) signal: DC {v_dc:.2f} vs spatial residual {v_res:.2f}.")
        print("  -> every position, background included, reads the same level shift. Confirms the flat shells.")
    else:
        print(f"  ventricle_size retains spatial signal (RES {v_res:.2f} vs DC {v_dc:.2f}) -> NOT a pure DC channel.")
    if l_res > 0.3 * max(l_dc, 1e-9):
        print(f"  lr_asymmetry (control) does carry spatial signal: RES {l_res:.2f} -> the split is working.")
    else:
        print(f"  WARNING: lr_asymmetry RES is also low ({l_res:.2f}). A left-right differential cannot be a uniform")
        print("  shift, so this suggests the RES probe is underpowered — do not read the ventricle result as proof.")


if __name__ == "__main__":
    main()
