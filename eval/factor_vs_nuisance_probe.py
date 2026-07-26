"""Does the representation encode the LABELLED factors, or the nuisance shared content?

The multiview identifiability guarantee (Yao et al. ICLR 2024, Defn 2.3) is
block-identification *up to a smooth bijection*: the shared block is recovered as
a set, entangled.  Their numerical experiments probe a 2-dim content block and
read R^2 ~ 1.0.  This generator's shared block is ~585-dim (z_content 9 +
z_deformation + z_fissure) and only the 9 z_content dims are ever probed, so a
low R^2 on those is consistent with the block being perfectly identified.

This script asks which part of the shared block the representation actually
carries.  Same probe for every target (the kernel-ridge probe Yao evaluate with,
via eval.identifiability_metrics), so the numbers are directly comparable.

Dimensionality is the confound: mean R^2 over 512 fissure dims from a 44-dim
representation is near 0 by rank alone, not by absence of information.  So every
group is ALSO scored on its top-9 principal components -- equal footing with
z_content's 9 dims.  Read that column.

z_style_v1 is the negative control: it is not shared across views and should
read ~0 (Yao's Figure 2 reports exactly that for independent style).

**Ceiling.** Measured through this exact pipeline on an ORACLE representation
(per-patch tissue-class + lesion fractions at 8^3, which is what z_content
modulates), N=800:

    PCA dim     8     16     32     64    128    256    499
    kernel   0.470  0.428  0.426  0.456  0.488  0.411  0.188
    ridge    0.503  0.511  0.517  0.567  0.633  0.624  0.411

Two consequences. z_content is NOT recoverable at R^2 1.0 even from perfect local
anatomy -- the ceiling is ~0.63, so compare model numbers against that, not 1.0.
And the probe degenerates above ~256 PCs, so a low reading at high --patch-pca-dim
measures the probe, not the model. Oracle z_deformation for reference: 0.280
(ridge, 256 PCs) -- well below z_content, so content-over-nuisance ordering is
expected even from a perfect representation.

Usage:
    python -m eval.factor_vs_nuisance_probe results/synthetic/<run> --rep gap
    python -m eval.factor_vs_nuisance_probe results/synthetic/<run> --rep patch --kind ridge
"""

import argparse
import logging
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eval.identifiability_metrics import cv_probe_r2_multi

logger = logging.getLogger(__name__)

TARGETS = ("z_content", "z_deformation", "z_fissure", "z_style_v1")


def extract(model, dataset, args, device, rep, level, batch_size, patch_pca_dim):
    """(reps (N,D), latents dict of (N,d)) for the whole dataset."""
    from torch.utils.data import DataLoader

    grid = None
    if rep == "patch":
        grid = tuple(getattr(args, "patch_grid", (8, 8, 8)))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    reps, lats = [], {k: [] for k in TARGETS}
    model.eval()
    for batch in loader:
        imgs = batch["image"]
        n_views = len(imgs)
        x = torch.cat(imgs, dim=0).to(device)
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=n_views, patch_grid=grid)
            feats = out[2] if isinstance(out, tuple) else [out]
            soft = out[6] if isinstance(out, tuple) and len(out) > 6 else {}
            f = feats[level]
            if f.dim() == 2:
                f = f.unsqueeze(-1)
            hz = f.reshape(n_views, -1, *f.shape[1:])  # (V,B,C,P)
            if isinstance(soft, dict) and level in soft:
                m = soft[level]
                m = m[0] if isinstance(m, tuple) else m
                cidx = torch.where(m.bool())[-1]
            else:
                cidx = torch.as_tensor(args.content_indices[0], device=hz.device)
            hz = hz[:, :, cidx, :]
            # View 0 only: probing a single view's representation, as Yao do (g_k(x_k)).
            r = hz[0]  # (B,C,P)
            r = r.mean(-1) if rep == "gap" else r.flatten(1)
        reps.append(r.float().cpu().numpy())
        gt = batch.get("gt_latents") or {}
        for k in TARGETS:
            if k in gt:
                v = gt[k]
                lats[k].append(np.asarray(v).reshape(v.shape[0], -1))

    X = np.concatenate(reps, 0)
    Y = {k: np.concatenate(v, 0) for k, v in lats.items() if v}

    if rep == "patch" and X.shape[1] > patch_pca_dim:
        logger.info("patch rep %d dims -> PCA to %d", X.shape[1], patch_pca_dim)
        X = PCA(n_components=patch_pca_dim, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    return X, Y


def top_pcs(Y, k, seed=0):
    k = min(k, Y.shape[1], Y.shape[0])
    return PCA(n_components=k, random_state=seed).fit_transform(StandardScaler().fit_transform(Y))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rep", choices=["gap", "patch"], default="gap")
    p.add_argument("--level", type=int, default=0)
    p.add_argument(
        "--kind",
        choices=["ridge", "kernel", "mlp"],
        default="ridge",
        help="Yao use kernel ridge, but RBF distances concentrate in high dimension: on the "
        "oracle control below, ridge beat kernel at EVERY PCA dim (0.633 vs 0.488 at 128). "
        "Use kernel only at low --patch-pca-dim.",
    )
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--patch-pca-dim",
        type=int,
        default=128,
        help="Keep this LOW. Measured on an oracle representation that provably contains "
        "z_content, probe R^2 peaks at ~128 PCs (ridge 0.633) and collapses by 499 (0.411 ridge, "
        "0.188 kernel) as the probe degenerates. A large value zeroes the probe regardless of "
        "what the representation holds, so it cannot be raised for 'rank' reasons.",
    )
    p.add_argument("--n-pc", type=int, default=9, help="Equal-dimensionality control: top-k PCs per group.")
    p.add_argument("--device", default=None)
    cli = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = cli.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(cli.run_dir, ckpt)
    model, args, device = load_model_from_run_dir(cli.run_dir, ckpt, device)

    dataset = build_synthetic_test_set(args, cli.num_samples)
    X, Y = extract(model, dataset, args, device, cli.rep, cli.level, cli.batch_size, cli.patch_pca_dim)

    print(f"\nrun: {cli.run_dir}")
    print(f"representation: {cli.rep} (level {cli.level}) -> {X.shape[1]} dims | N={X.shape[0]} | probe={cli.kind}")
    print(f"equal-dimensionality control: top-{cli.n_pc} PCs per group\n")

    hdr = f"{'target group':<18}{'dims':>7}{'mean R2':>11}{'top-' + str(cli.n_pc) + 'PC R2':>13}"
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for k in TARGETS:
        if k not in Y:
            print(f"{k:<18}{'-':>7}{'(absent)':>11}{'-':>13}")
            continue
        yk = Y[k]
        full = cv_probe_r2_multi(X, yk, kind=cli.kind)["mean"]
        pcs = cv_probe_r2_multi(X, top_pcs(yk, cli.n_pc), kind=cli.kind)["mean"]
        res[k] = (float(np.mean(full)), float(np.mean(pcs)))
        print(f"{k:<18}{yk.shape[1]:>7}{res[k][0]:>11.3f}{res[k][1]:>13.3f}")

    if "z_content" in Y:
        from eval.dci import CONTENT_FACTOR_NAMES

        per = cv_probe_r2_multi(X, Y["z_content"], kind=cli.kind)
        print("\nper-factor R2 (the mean above hides this):")
        order = np.argsort(-per["mean"])
        for j in order:
            nm = CONTENT_FACTOR_NAMES[j] if j < len(CONTENT_FACTOR_NAMES) else f"z_content[{j}]"
            print(f"  {nm:<20}{per['mean'][j]:>8.3f}  +/-{per['std'][j]:.3f}")

    print("\nverdict:")
    if "z_content" in res and ("z_deformation" in res or "z_fissure" in res):
        c = res["z_content"][1]
        n = max(res.get("z_deformation", (0, 0))[1], res.get("z_fissure", (0, 0))[1])
        if n > c + 0.1:
            print(
                f"  nuisance shared content is better recovered than the labelled factors "
                f"({n:.3f} vs {c:.3f} on matched dimensionality) -> the block IS identified, but the "
                "9 probed factors are a small subspace of it. Test with synthetic_clean_content: true."
            )
        elif c > n + 0.1:
            print(
                f"  labelled factors are better recovered ({c:.3f} vs {n:.3f}) -> nuisance-shortcut hypothesis REJECTED."
            )
        else:
            print(
                f"  factors and nuisance are recovered comparably ({c:.3f} vs {n:.3f}) -> not decisive; neither dominates."
            )
    if "z_style_v1" in res:
        s = res["z_style_v1"][1]
        print(
            f"  style control reads {s:.3f}"
            + ("  (expected ~0; sanity OK)" if s < 0.2 else "  (HIGH - style is leaking into content)")
        )


if __name__ == "__main__":
    main()
