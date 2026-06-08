#!/usr/bin/env python
"""Final DCI comparison across models that share architecture / channel allocation.

This is the *direct model-comparison* protocol distilled from the methodology
discussion.  It is for ranking models that differ in the objective (loss type,
contrastive on/off, seed) but **not** in content/style channel counts — if those
differ, the null floor alone is not enough and you must equalize capacity first.

Protocol
--------
* **One frozen synthetic test set**, built once and shared across every model
  (same seed + N), so differences are about the model, not the data draw.
* **Several poolings** (default gap, stats, 2x2x2).  Each ground-truth factor is
  scored under the pooling that can physically expose it (global→gap, scale→stats,
  spatial→patch); the mapping is fixed up-front (``FACTOR_POOLING``) so there is no
  max-over-poolings selection bias.
* **Cross-validated ridge probes** with multiple seeds → mean ± std error bars
  (from ``eval.identifiability_metrics``), not a single-split GBT.
* **Null floor via label permutation** → every informativeness number is reported
  as a GAP (real − null).  The GAP cancels each model's channel-count/shape
  advantage, which is what makes models directly comparable.
* **Optional 0-contrastive baseline anchor** → Δ (model − baseline) isolates what
  the objective *earned* on top of the architecture.
* **Ranks on the theory-aligned headline metrics** — content→style leakage,
  block-MCC, view-invariance — never the capacity-bound content→content diagonal.

Usage
-----
    python -m eval.run_dci_compare \
        --run-dirs runs/infonce runs/moco runs/barlow \
        --baseline runs/no_contrastive \
        --num-samples 2000 --level 0 --out dci_compare_out

The torch-dependent parts (model load + encoder forward) are lazily imported so
the scoring/ranking logic stays unit-testable on plain numpy (see ``__main__``).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os

import numpy as np

from eval.identifiability_metrics import block_mcc, cv_probe_r2, view_invariance

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Each factor is scored under the pooling that can physically expose it.  Fixed
# in advance so the headline number is not a max over noisy pooling estimates.
FACTOR_POOLING = {
    # global morphometry — present in the channel mean
    "brain_size": "gap",
    "ventricle_size": "gap",
    "cortical_thickness": "gap",
    "temporal_atrophy": "gap",
    "lr_asymmetry": "gap",
    "sulcal_widening": "gap",
    # localized — needs spatial layout
    "lesion_x": "patch",
    "lesion_y": "patch",
    "lesion_z": "patch",
    # intensity / scale / noise — needs variance statistics
    "gain": "stats",
    "bias": "gap",
    "noise_sigma": "stats",
}

# Index of the content / style array inside a level_data tuple from
# eval.dci._extract_synthetic_representations: (content, style, content_v2, style_v2, info)
_CONTENT, _STYLE = 0, 1


# --------------------------------------------------------------------------- #
# Pooling parsing
# --------------------------------------------------------------------------- #


def parse_poolings(spec):
    """``"gap,stats,2x2x2"`` → ``[("gap","gap"), ("stats","stats"), ("patch",(2,2,2))]``.

    Returns a list of ``(key, value)`` where ``key`` is the bucket used by
    ``FACTOR_POOLING`` and ``value`` is what ``_extract_synthetic_representations``
    expects (``"gap"``/``"stats"`` or a ``(D,H,W)`` tuple).
    """
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in ("gap", "stats"):
            out.append((tok, tok))
        elif "x" in tok:
            out.append(("patch", tuple(int(x) for x in tok.split("x"))))
        else:
            raise ValueError(f"unrecognized pooling token: {tok!r} (use gap, stats, or e.g. 2x2x2)")
    return out


def _resolve_key(want, avail):
    """Map a desired pooling bucket to one that was actually computed."""
    if want in avail:
        return want
    for fb in ("stats", "gap", "patch"):
        if fb in avail:
            return fb
    return next(iter(avail))


# --------------------------------------------------------------------------- #
# Null floors (label permutation) for the CV probes
# --------------------------------------------------------------------------- #


def _null_cv_r2(X, y, n_null, seeds, rng):
    vals = [cv_probe_r2(X, y[rng.permutation(len(y))], seeds=seeds)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


def _null_block_mcc(X, Z, n_null, seeds, rng):
    vals = [block_mcc(X, Z[rng.permutation(len(Z))], seeds=seeds)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Scoring (pure numpy — no torch, unit-testable)
# --------------------------------------------------------------------------- #


def _block_array(reprs, key, level, block_idx):
    ld = reprs.get(key)
    if ld is None or level not in ld:
        return None
    return ld[level][block_idx]


def _score_block(reprs, level, block_idx, factors, names, avail, n_null, seeds, rng):
    """Per-factor GAP (real − null) test-R² for one repr→factor block.

    ``block_idx`` selects the content (0) or style (1) array; ``factors``/``names``
    select which ground-truth columns to predict.  Each factor uses its assigned
    pooling.  Returns ``{"mean_gap", "per_factor"}``.
    """
    per = {}
    for j, name in enumerate(names):
        key = _resolve_key(FACTOR_POOLING.get(name, "stats"), avail)
        X = _block_array(reprs, key, level, block_idx)
        if X is None or X.shape[1] == 0:
            per[name] = {
                "gap": float("nan"),
                "real": float("nan"),
                "null": float("nan"),
                "std": float("nan"),
                "pooling": key,
            }
            continue
        r = cv_probe_r2(X, factors[:, j], seeds=seeds)
        nl = _null_cv_r2(X, factors[:, j], n_null, seeds, rng)
        per[name] = {"gap": r["mean"] - nl, "real": r["mean"], "null": nl, "std": r["std"], "pooling": key}
    gaps = [v["gap"] for v in per.values() if np.isfinite(v["gap"])]
    return {"mean_gap": float(np.mean(gaps)) if gaps else float("nan"), "per_factor": per}


def score_reprs(reprs, gt_content, gt_style, info, level, n_null=3, seeds=(0, 1, 2)):
    """Turn extracted representations into one comparison row (torch-free).

    ``reprs`` maps a pooling key to a ``level_data`` dict (the output of
    ``_extract_synthetic_representations``).  ``info`` is that level's factor_info.
    """
    avail = set(reprs.keys())
    rng = np.random.RandomState(0)
    cnames, snames = info["content_names"], info["style_names"]
    has_split = info["has_split"]

    cc = _score_block(reprs, level, _CONTENT, gt_content, cnames, avail, n_null, seeds, rng)
    cs = _score_block(reprs, level, _CONTENT, gt_style, snames, avail, n_null, seeds, rng)
    ss = _score_block(reprs, level, _STYLE, gt_style, snames, avail, n_null, seeds, rng) if has_split else None
    sc = _score_block(reprs, level, _STYLE, gt_content, cnames, avail, n_null, seeds, rng) if has_split else None

    # Block-MCC + view-invariance under the richest available pooling.
    mkey = _resolve_key("stats", avail)
    Cc = _block_array(reprs, mkey, level, _CONTENT)
    Sc = _block_array(reprs, mkey, level, _STYLE)
    mcc_cc = block_mcc(Cc, gt_content, seeds=seeds)["mean"] if Cc is not None and Cc.shape[1] else float("nan")
    mcc_cc_null = (
        _null_block_mcc(Cc, gt_content, n_null, seeds, rng) if Cc is not None and Cc.shape[1] else float("nan")
    )
    mcc_cs = block_mcc(Cc, gt_style, seeds=seeds)["mean"] if Cc is not None and Cc.shape[1] else float("nan")
    mcc_cs_null = _null_block_mcc(Cc, gt_style, n_null, seeds, rng) if Cc is not None and Cc.shape[1] else float("nan")

    Cc2 = reprs[mkey][level][2] if mkey in reprs and level in reprs[mkey] else None
    Sc2 = reprs[mkey][level][3] if mkey in reprs and level in reprs[mkey] else None
    if Cc is not None and Cc2 is not None:
        s1 = Sc if Sc is not None and Sc.shape[1] else Cc[:, :0]
        s2 = Sc2 if Sc2 is not None and Sc2.shape[1] else Cc2[:, :0]
        vi = view_invariance(Cc, Cc2, s1, s2, seeds=seeds)
        content_view, style_view, chance = vi["content_acc"], vi.get("style_acc", float("nan")), vi["chance"]
    else:
        content_view = style_view = chance = float("nan")

    sep = mcc_cc - mcc_cs if np.isfinite(mcc_cc) and np.isfinite(mcc_cs) else float("nan")

    return {
        "n_content_channels": info["n_content_channels"],
        "n_style_channels": info["n_style_channels"],
        "leak_c2s": cs["mean_gap"],  # headline ↓  (content must not encode style)
        "info_c2c": cc["mean_gap"],  # diagnostic  (capacity, not the claim)
        "suff_s2s": ss["mean_gap"] if ss else float("nan"),  # style sufficiency ↑
        "leak_s2c": sc["mean_gap"] if sc else float("nan"),  # content leaking into style ↓
        "mcc_cc": mcc_cc,
        "mcc_cc_null": mcc_cc_null,
        "mcc_cs": mcc_cs,
        "mcc_cs_null": mcc_cs_null,
        "content_view": content_view,
        "style_view": style_view,
        "view_chance": chance,
        "separation": sep,  # headline ↑  mcc_cc − mcc_cs
        "detail": {"content2content": cc, "content2style": cs, "style2style": ss, "style2content": sc},
    }


# --------------------------------------------------------------------------- #
# Per-model driver (torch — lazily imported)
# --------------------------------------------------------------------------- #


def evaluate_model(
    name, run_dir, dataset, poolings, level, n_null, seeds, batch_size, num_workers, device, checkpoint=None
):
    """Load a model, extract representations under each pooling, return a row."""
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_synthetic import load_model_from_run_dir

    logger.info("=== evaluating %s (%s) ===", name, run_dir)
    model, _args, device = load_model_from_run_dir(run_dir, checkpoint, device)

    reprs, gt_content, gt_style, info = {}, None, None, None
    for key, value in poolings:
        level_data, gc, gsv1, _gsv2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value
        )
        reprs[key] = level_data
        if gt_content is None:
            gt_content, gt_style = gc, gsv1
        if info is None and level in level_data:
            info = level_data[level][4]

    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if info is None:
        raise RuntimeError(f"level {level} not found in encoder outputs for {name}")

    row = score_reprs(reprs, gt_content, gt_style, info, level, n_null=n_null, seeds=seeds)
    row["name"] = name
    row["run_dir"] = run_dir
    return row


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_HEADLINE_COLS = ["separation", "leak_c2s", "mcc_cc", "mcc_cs", "content_view", "style_view", "suff_s2s"]


def _fmt(v, nd=4):
    return "  nan  " if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def print_table(rows, baseline_name=None):
    """Ranked stdout table.  Ranks by separation↑ (tie-break leakage↓)."""
    base = next((r for r in rows if r["name"] == baseline_name), None)
    ranked = sorted(rows, key=lambda r: (-(r["separation"] if np.isfinite(r["separation"]) else -9), r["leak_c2s"]))

    print("\n" + "=" * 110)
    print("DCI MODEL COMPARISON   (GAP = real − null; ↑ better unless marked ↓)")
    print("=" * 110)
    print(
        f"{'model':16s} {'chan c/s':>9s} {'SEP↑':>8s} {'leak c→s↓':>10s} "
        f"{'mcc c→c↑':>9s} {'mcc c→s↓':>9s} {'c→view≈.5':>10s} {'s→view↑':>8s} {'suff s→s↑':>10s}"
    )
    print("-" * 110)
    for r in ranked:
        tag = "  *base" if r["name"] == baseline_name else ""
        chans = f"{r['n_content_channels']}/{r['n_style_channels']}"
        print(
            f"{r['name'][:16]:16s} {chans:>9s} {_fmt(r['separation']):>8s} {_fmt(r['leak_c2s']):>10s} "
            f"{_fmt(r['mcc_cc']):>9s} {_fmt(r['mcc_cs']):>9s} {_fmt(r['content_view']):>10s} "
            f"{_fmt(r['style_view']):>8s} {_fmt(r['suff_s2s']):>10s}{tag}"
        )

    if base is not None:
        print("-" * 110)
        print(f"Δ vs baseline '{baseline_name}'  (what the objective earned on top of the architecture)")
        print(f"{'model':16s} {'Δsep':>8s} {'Δleak c→s':>10s} {'Δmcc c→c':>9s} {'Δc→view':>9s}")
        for r in ranked:
            if r["name"] == baseline_name:
                continue
            print(
                f"{r['name'][:16]:16s} {_fmt(r['separation'] - base['separation']):>8s} "
                f"{_fmt(r['leak_c2s'] - base['leak_c2s']):>10s} {_fmt(r['mcc_cc'] - base['mcc_cc']):>9s} "
                f"{_fmt(r['content_view'] - base['content_view']):>9s}"
            )
    print("=" * 110)
    print(
        "SEP = mcc(content→content) − mcc(content→style)  ·  leak c→s near 0 = content is style-invariant\n"
        "info_c2c is capacity-bound (high even at 0 contrastive) — not shown; see JSON. Rank on SEP / leak / view.\n"
    )


def write_outputs(rows, out_dir, baseline_name=None):
    os.makedirs(out_dir, exist_ok=True)
    flat_cols = [
        "name",
        "run_dir",
        "n_content_channels",
        "n_style_channels",
        "separation",
        "leak_c2s",
        "info_c2c",
        "suff_s2s",
        "leak_s2c",
        "mcc_cc",
        "mcc_cc_null",
        "mcc_cs",
        "mcc_cs_null",
        "content_view",
        "style_view",
        "view_chance",
    ]
    csv_path = os.path.join(out_dir, "dci_compare.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in flat_cols})

    json_path = os.path.join(out_dir, "dci_compare.json")
    with open(json_path, "w") as f:
        json.dump({"baseline": baseline_name, "models": rows}, f, indent=2, default=float)
    logger.info("Wrote %s and %s", csv_path, json_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Final DCI comparison across models (shared architecture).")
    p.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to compare (settings.json each).")
    p.add_argument("--names", nargs="*", default=None, help="Labels (default: basename of each run-dir).")
    p.add_argument("--baseline", default=None, help="Run-dir to anchor Δ (e.g. the 0-contrastive model).")
    p.add_argument("--num-samples", type=int, default=2000, help="Frozen test-set size, shared across models.")
    p.add_argument("--poolings", default="gap,stats,2x2x2", help="Comma list: gap, stats, and/or DxHxW (e.g. 2x2x2).")
    p.add_argument("--level", type=int, default=0, help="Encoder level to compare on.")
    p.add_argument("--seeds", default="0,1,2", help="Probe CV seeds.")
    p.add_argument("--n-null", type=int, default=3, help="Permutations for the null floor.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="dci_compare_out", help="Output directory.")
    cli = p.parse_args()

    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))

    # Assemble the model list; the baseline (if separate) is evaluated too.
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

    # One frozen test set, built from the first run's settings, reused for all.
    import torch

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_args = load_run_args(specs[0])
    dataset = build_synthetic_test_set(ref_args, cli.num_samples)
    logger.info("Frozen test set: %d samples, shared across %d model(s).", cli.num_samples, len(specs))

    rows = []
    for name, run_dir in zip(names, specs):
        try:
            rows.append(
                evaluate_model(
                    name,
                    run_dir,
                    dataset,
                    poolings,
                    cli.level,
                    cli.n_null,
                    seeds,
                    cli.batch_size,
                    cli.num_workers,
                    device,
                )
            )
        except Exception as e:
            logger.error("Skipping %s (%s): %s", name, run_dir, e)

    if not rows:
        logger.error("No models evaluated successfully.")
        return
    print_table(rows, baseline_name)
    write_outputs(rows, cli.out, baseline_name)


if __name__ == "__main__":
    main()
