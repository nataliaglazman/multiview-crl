#!/usr/bin/env python
"""Does the content block encode the UNLABELED shared content (z_deformation / z_fissure)?

``run_dci_compare`` / ``dci.py`` / ``content_rank_pca`` all score ``z_content`` only —
the 9 named factors.  But ``render_structure`` also builds the shared geometry from
``z_deformation`` (4³=64) and ``z_fissure`` (8³=512), which are per-subject, shared
across views, and never probed.  That is ~585 dims of genuine view-invariant content
InfoNCE can (and cheaply does) discriminate on.

This scores the same content/style blocks against those fields, next to the named
factors, with the usual permutation null floor.  Read it as: if a contrastive model
beats the recon baseline on the FIELDS while tying on ``z_content``, it is extracting
shared content and the 9-factor probe was measuring the wrong 2%.

Usage:
    python -m eval.probe_fields --run-dirs results/baseline results/contrastive \
        --names baseline contrastive --num-samples 500 --poolings gap,4x4x4
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

from eval.dci import _extract_synthetic_representations
from eval.identifiability_metrics import cv_probe_r2
from eval.run_dci_compare import _CONTENT, _STYLE, parse_poolings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def collect_fields(dataset):
    """Per-sample z_deformation / z_fissure, in dataset order (extraction uses shuffle=False)."""
    out = {}
    for i in range(len(dataset)):
        lat = dataset[i]["gt_latents"]
        for key in ("z_deformation", "z_fissure"):
            if key in lat:
                out.setdefault(key, []).append(np.asarray(lat[key]).ravel())
    return {k: np.stack(v) for k, v in out.items()}


def _null(X, Y, n_null, seeds, rng):
    vals = [cv_probe_r2(X, Y[rng.permutation(len(Y))], seeds=seeds)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(description="Probe the unlabeled shared fields from the content block.")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--checkpoint-name", default="vqvae_best.pt")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--poolings", default="gap,4x4x4", help="Fields are spatial — patch should beat gap.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--n-null", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--probe-dim", type=int, default=64, help="PCA-reduce blocks to this width (0 = off).")
    p.add_argument("--out", default="field_probe_out")
    cli = p.parse_args()

    import torch
    from sklearn.decomposition import PCA

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in cli.run_dirs]
    if len(names) != len(cli.run_dirs):
        p.error("--names must have one entry per --run-dirs")
    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))
    rng = np.random.default_rng(0)

    ref_args = load_run_args(cli.run_dirs[0])
    dataset = build_synthetic_test_set(ref_args, cli.num_samples)
    fields = collect_fields(dataset)
    if not fields:
        raise SystemExit("No z_deformation / z_fissure in gt_latents — is this a pseudo_mri synthetic run?")
    logger.info("Fields found: %s", {k: v.shape for k, v in fields.items()})
    if all(np.allclose(v.std(axis=0), 0) for v in fields.values()):
        logger.warning("Fields are constant across samples — this run trained with clean_content, nothing to probe.")

    rows = []
    for name, run_dir in zip(names, cli.run_dirs):
        ckpt = os.path.join(run_dir, cli.checkpoint_name)
        model, _a, device = load_model_from_run_dir(run_dir, ckpt if os.path.exists(ckpt) else None, None)
        for pkey, pval in poolings:
            level_data, gt_content, _s1, _s2 = _extract_synthetic_representations(
                model, dataset, device, cli.batch_size, cli.num_workers, pooling=pval
            )
            if cli.level not in level_data:
                logger.warning("  %s: level %d absent under %s", name, cli.level, pkey)
                continue
            blocks = {"content": level_data[cli.level][_CONTENT], "style": level_data[cli.level][_STYLE]}
            targets = {"z_content": gt_content, **fields}
            for bname, X in blocks.items():
                if X is None or X.shape[1] == 0:
                    continue
                Xp = X
                if cli.probe_dim and Xp.shape[1] > cli.probe_dim:
                    Xp = PCA(n_components=cli.probe_dim, random_state=0).fit_transform(Xp)
                for tname, Y in targets.items():
                    real = cv_probe_r2(Xp, Y, seeds=seeds)["mean"]
                    nul = _null(Xp, Y, cli.n_null, seeds, rng)
                    rows.append(
                        {
                            "model": name,
                            "pooling": pkey,
                            "block": bname,
                            "target": tname,
                            "dims": Y.shape[1],
                            "r2": real,
                            "null": nul,
                            "gap": real - nul,
                        }
                    )
                    logger.info(
                        "  %-14s %-6s %-8s %-14s r2 %6.3f  null %6.3f  gap %6.3f",
                        name,
                        pkey,
                        bname,
                        tname,
                        real,
                        nul,
                        real - nul,
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print("  UNLABELED SHARED CONTENT  (null-subtracted R²; z_content = the 9 you already score)")
    print("=" * 92)
    hdr = f"  {'model':<16s}{'pooling':<8s}{'block':<9s}" + "".join(
        f"{t:>16s}" for t in ("z_content", "z_deformation", "z_fissure")
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in names:
        for pkey, _ in poolings:
            for bname in ("content", "style"):
                sel = {
                    r["target"]: r for r in rows if r["model"] == name and r["pooling"] == pkey and r["block"] == bname
                }
                if not sel:
                    continue
                line = f"  {name:<16s}{pkey:<8s}{bname:<9s}"
                for t in ("z_content", "z_deformation", "z_fissure"):
                    line += f"{sel[t]['gap']:>16.3f}" if t in sel else f"{'—':>16s}"
                print(line)

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "field_probe.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
