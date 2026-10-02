#!/usr/bin/env python
"""Read saved DCI comparison CSVs and turn them into plain-language conclusions.

Operates on the files written by ``run_dci_compare.write_outputs`` — no encoder,
dataset, or GPU needed, so it is cheap to re-run on a downloaded CSV:

  dci_compare.csv             one row per model (the headline metrics)
  dci_compare_per_latent.csv  one row per model x block x factor x pooling

It mirrors how ``run_dci_compare.print_table`` reasons about a run, so the verdicts
agree with the tool's own ranking instead of being generic:

  separation = mcc(content->content) - mcc(content->style)   headline, up
  leak_c2s   = content->style GAP test-R2 (already null-subtracted)  down, ~0 = clean
  mcc_cc vs mcc_cc_null   content recovered above its shape's null floor
  mcc_cs vs mcc_cs_null   content does NOT identify style (margin ~0 good)
  content_view vs view_chance   ~chance = content is view-invariant (the claim)
  style_view  vs view_chance    well above chance = style captures the modality
  suff_s2s    style block is sufficient for the style factors, up

Thresholds below are heuristic display cutoffs only; ranking and the null/chance
margins are what is principled.  Run with ``-h`` for options.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Heuristic display cutoffs (see module docstring — not part of the metric).
LEAK_CLEAN = 0.05  # leak_c2s below this -> content clean of style
LEAK_BAD = 0.15  # leak_c2s above this -> substantial style leakage
VIEW_TOL = 0.05  # content_view within this of chance -> view-invariant
VIEW_BAD = 0.15  # content_view this far above chance -> view-contaminated
MCC_MARGIN = 0.10  # mcc above its null by this -> real (not shape artifact)
SUFF_MIN = 0.10  # suff_s2s above this -> style block carries the style factors
SEP_GOOD = 0.20  # separation above this -> strong content/style split

OK, WARN, BAD, NA = "[ok ]", "[warn]", "[BAD]", "[ -- ]"


_OLD_TO_NEW = {
    "name": "model",
    "n_content_channels": "content_channels",
    "n_style_channels": "style_channels",
    "disentanglement": "overall_score",
    "info_all": "informativeness",
    "leak_c2s": "content_to_style_leak",
    "content_view": "content_view_acc",
    "style_view": "style_view_acc",
    "suff_s2s": "style_sufficiency",
    "leak_s2c": "style_to_content_leak",
    "mcc_cc": "mcc_content_to_content",
    "mcc_cs": "mcc_content_to_style",
    "mcc_cc_null": "mcc_content_to_content_null",
    "mcc_cs_null": "mcc_content_to_style_null",
}


def _compat_rename(df):
    """Rename old-format CSV columns so analysis works on files from either version."""
    return df.rename(columns={old: new for old, new in _OLD_TO_NEW.items() if old in df.columns})


def _f(df_row, col):
    """Float value of a cell, or NaN when the column is absent/empty."""
    if col not in df_row or pd.isna(df_row[col]):
        return float("nan")
    return float(df_row[col])


def _flag(ok, warn):
    """Map two booleans to a verdict marker (ok wins, then warn, else bad)."""
    return OK if ok else (WARN if warn else BAD)


def _verdicts(r):
    """Per-axis (marker, message) verdicts for one model row ``r`` (a Series)."""
    sep = _f(r, "separation")
    leak = _f(r, "content_to_style_leak")
    mcc_cc, mcc_cc_null = _f(r, "mcc_content_to_content"), _f(r, "mcc_content_to_content_null")
    mcc_cs, mcc_cs_null = _f(r, "mcc_content_to_style"), _f(r, "mcc_content_to_style_null")
    cview, schance = _f(r, "content_view_acc"), _f(r, "view_chance")
    sview = _f(r, "style_view_acc")
    suff = _f(r, "style_sufficiency")

    out = []

    # 1. Is content actually recovered (above the null floor for its shape)?
    if np.isfinite(mcc_cc):
        margin = mcc_cc - mcc_cc_null if np.isfinite(mcc_cc_null) else float("nan")
        m = OK if margin >= MCC_MARGIN else (WARN if margin > 0 else BAD)
        out.append((m, f"content identifiable : mcc_cc={mcc_cc:.3f} " f"(null {mcc_cc_null:.3f}, +{margin:.3f})"))
    else:
        out.append((NA, "content identifiable : n/a"))

    # 2. Does content stay clean of style?  (leak_c2s is already null-subtracted)
    if np.isfinite(leak):
        m = _flag(leak < LEAK_CLEAN, leak < LEAK_BAD)
        cs_margin = mcc_cs - mcc_cs_null if np.isfinite(mcc_cs) and np.isfinite(mcc_cs_null) else float("nan")
        extra = f"; mcc_cs +{cs_margin:.3f} over null" if np.isfinite(cs_margin) else ""
        out.append((m, f"content clean of style: leak_c2s={leak:.3f} (~0 good){extra}"))
    else:
        out.append((NA, "content clean of style: n/a"))

    # 2b. Does the style block stay clean of content?  (the over-allocation check —
    # a big style block can soak up anatomy even when content excludes style.)
    leak_sc = _f(r, "style_to_content_leak")
    if np.isfinite(leak_sc):
        m = _flag(leak_sc < LEAK_CLEAN, leak_sc < LEAK_BAD)
        out.append((m, f"style clean of content: leak_s2c={leak_sc:.3f} " f"(~0 good; high = style absorbs anatomy)"))
    else:
        out.append((NA, "style clean of content: n/a"))

    # 3. Is content view-invariant?  content_view should sit at chance.
    if np.isfinite(cview) and np.isfinite(schance):
        excess = cview - schance
        m = _flag(excess <= VIEW_TOL, excess < VIEW_BAD)
        out.append((m, f"content view-invariant: c->view={cview:.3f} vs chance {schance:.3f} " f"(+{excess:.3f})"))
    else:
        out.append((NA, "content view-invariant: n/a"))

    # 4. Is the style block sufficient / does it carry the modality?
    if np.isfinite(suff) or np.isfinite(sview):
        sv_ok = np.isfinite(sview) and np.isfinite(schance) and (sview - schance) > VIEW_BAD
        suff_ok = np.isfinite(suff) and suff > SUFF_MIN
        m = _flag(suff_ok and (sv_ok or not np.isfinite(sview)), suff_ok or sv_ok)
        sv = f", s->view={sview:.3f}" if np.isfinite(sview) else ""
        out.append((m, f"style sufficient      : suff_s2s={suff:.3f}{sv}"))
    else:
        out.append((NA, "style sufficient      : n/a"))

    # 5. Headline separation.
    if np.isfinite(sep):
        m = _flag(sep >= SEP_GOOD, sep > 0)
        out.append((m, f"SEPARATION (headline) : {sep:.3f}  = mcc_cc - mcc_cs"))
    else:
        out.append((NA, "SEPARATION (headline) : n/a"))

    return out


def _rank(df):
    """Rank like print_table: separation desc, tie-break content_to_style_leak asc."""
    sep = df["separation"].fillna(-9.0)
    leak = df["content_to_style_leak"].fillna(9.0)
    return df.assign(_sep=sep, _leak=leak).sort_values(["_sep", "_leak"], ascending=[False, True])


def _per_latent_breakdown(pl_path, model):
    """For one model, list its best-recovered and worst-leaking factors.

    Uses only each factor's assigned (headline) pooling; ``gap`` is real - null R2,
    so high gap means "well predicted from this block".  That is desirable for the
    matched blocks (content->content, style->style) and a red flag for the leakage
    blocks (content->style, style->content).
    """
    pl = pd.read_csv(pl_path, comment="#")
    if "is_assigned" in pl.columns:
        pl = pl.rename(columns={"is_assigned": "is_headline_pooling", "gap": "gap_r2"})
    pl = pl[pl["model"] == model]
    if pl.empty:
        return
    pl = pl[pl["predicted_from"].isin(["content", "style"])].copy()
    pl["block"] = pl["predicted_from"].astype(str) + "->" + pl["factor_type"].astype(str)
    assigned = pl["is_headline_pooling"].astype(str).str.lower().isin(["true", "1"])
    pl = pl[assigned]
    if pl.empty:
        return

    print(f"\n  per-factor breakdown for top model '{model}' (assigned pooling only)")
    desirable = {"content->content": "recovered", "style->style": "recovered"}
    leaky = {"content->style": "LEAK", "style->content": "LEAK"}
    for block, df_b in pl.groupby("block"):
        df_b = df_b.sort_values("gap_r2", ascending=False)
        kind = desirable.get(block) or leaky.get(block) or "?"
        cells = ", ".join(f"{row.factor}={row.gap_r2:+.2f}" for row in df_b.itertuples())
        print(f"    {block:18s} [{kind:9s}] {cells}")
    if any(b in leaky for b in pl["block"].unique()):
        worst = pl[pl["block"].isin(leaky)].sort_values("gap_r2", ascending=False).head(1)
        if not worst.empty and worst.iloc[0]["gap_r2"] > LEAK_CLEAN:
            w = worst.iloc[0]
            print(f"    -> biggest leak: {w['factor']} via {w['block']} (gap {w['gap_r2']:+.2f})")


def _resolve_paths(path):
    """Accept a dir, the headline CSV, or the per-latent CSV; return (headline, per_latent)."""
    if os.path.isdir(path):
        head = os.path.join(path, "dci_compare.csv")
        per = os.path.join(path, "dci_compare_per_latent.csv")
    else:
        d = os.path.dirname(path) or "."
        head = path if path.endswith("dci_compare.csv") else os.path.join(d, "dci_compare.csv")
        per = os.path.join(d, "dci_compare_per_latent.csv")
    return head, (per if os.path.exists(per) else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=".", help="dci_compare.csv, or a directory containing it (default: cwd)")
    ap.add_argument("--baseline", default=None, help="model name to show deltas against")
    ap.add_argument(
        "--per-latent", default=None, help="explicit per-latent CSV path (else auto-discovered beside the headline CSV)"
    )
    args = ap.parse_args(argv)

    head_path, per_path = _resolve_paths(args.path)
    if args.per_latent:
        if not os.path.exists(args.per_latent):
            ap.error(f"--per-latent file not found: {args.per_latent}")
        per_path = args.per_latent
    if not os.path.exists(head_path):
        ap.error(f"no dci_compare.csv at {head_path}")

    df = _compat_rename(pd.read_csv(head_path, comment="#"))
    if df.empty:
        print("dci_compare.csv has no rows.")
        return 0
    ranked = _rank(df)

    print("=" * 78)
    print(f"DCI CONCLUSIONS  ({len(df)} model(s) from {head_path})")
    print("ranked by separation (desc), tie-break content_to_style_leak (asc)")
    print("=" * 78)

    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        chans = (
            f"{int(_f(r,'content_channels'))}/{int(_f(r,'style_channels'))}"
            if np.isfinite(_f(r, "content_channels"))
            else "?/?"
        )
        print(f"\n#{i}  {r['model']}   (content/style channels {chans})")
        for marker, msg in _verdicts(r):
            print(f"    {marker} {msg}")

    # Overall takeaway: the winner + whether anything passes the separation bar.
    top = ranked.iloc[0]
    strong = ranked[ranked["separation"].fillna(-9) >= SEP_GOOD]["model"].tolist()
    print("\n" + "-" * 78)
    print(
        f"VERDICT: best split is '{top['model']}' (separation {_f(top,'separation'):.3f}, "
        f"content_to_style_leak {_f(top,'content_to_style_leak'):.3f})."
    )
    if strong:
        print(f"         clears the strong bar (>= {SEP_GOOD}): {', '.join(strong)}.")
    else:
        print(f"         NOTHING clears the strong-separation bar (>= {SEP_GOOD}) — split is weak.")

    if args.baseline and args.baseline in set(df["model"]):
        base = df[df["model"] == args.baseline].iloc[0]
        print(f"\n  deltas vs baseline '{args.baseline}' (what the objective earned over the architecture):")
        for _, r in ranked.iterrows():
            if r["model"] == args.baseline:
                continue
            dsep = _f(r, "separation") - _f(base, "separation")
            dleak = _f(r, "content_to_style_leak") - _f(base, "content_to_style_leak")
            print(f"    {r['model']:20s} dSEP {dsep:+.3f}   dLEAK {dleak:+.3f}")

    if per_path:
        _per_latent_breakdown(per_path, top["model"])
    else:
        print("\n  (no dci_compare_per_latent.csv found — skipping per-factor breakdown)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
