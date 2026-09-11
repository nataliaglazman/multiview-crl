#!/usr/bin/env python
"""What --patch-center-mode keeps and what it deletes, per content channel.

    python -m eval.patch_variance_split --run-dir results/synthetic/RUN --causal iid

A patch feature z[s, p] for subject s at position p splits into three orthogonal parts:

    z[s, p] = mean + a[p] + b[s] + e[s, p]

    a[p]     what is typical AT THAT POSITION across subjects -- shared anatomy
    b[s]     what is typical FOR THAT SUBJECT across positions -- a spatially flat code
    e[s, p]  this subject, specifically here -- the subject x position interaction

The three centering modes each keep a different suffix of that sum:

    none       mean + a + b + e
    position   b + e          (removes the across-subject mean at each position)
    double     e              (also removes each subject's mean over positions)

So switching to ``double`` deletes ``b`` from the objective. That is the documented
objection to it -- global factors live in ``b`` -- and this script measures the cost
channel by channel instead of guessing it: a channel whose variance is nearly all ``b``
contributes almost nothing after ``double``, and one that is nearly all ``e`` is unchanged.

It also reports each component's CROSS-VIEW correlation, which is what the Barlow Twins
on_diag/sim terms actually see. The interaction's correlation is the pos_sim to expect
under ``double``, so the drop is predicted here rather than discovered after a training run.

Extraction reuses ``eval.bt_term_balance`` -- same grid, foreground mask, content channels
and pooling the run trains with -- so these numbers refer to the real objective. The
decomposition is pure numpy and exact (the three sums of squares add to the total, asserted
at runtime); ``--self-test`` runs it on planted components with no torch and no checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# A channel keeping less than this share of its variance as interaction is reported as
# effectively deleted by `double`. Not a threshold the analysis depends on -- the shares
# are printed -- only the wording of the count in the verdict.
ANNIHILATED_BELOW = 0.05


def components(z):
    """Return (a[p], b[s], e[s,p], grand) for one channel/view, z of shape (S, P)."""
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    grand = z.mean()
    a = z.mean(axis=0) - grand
    b = z.mean(axis=1) - grand
    e = z - grand - a[None, :] - b[:, None]
    return a, b, e, grand


def split_variance(z):
    """Sums of squares for the three parts. They add to the total exactly."""
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    subjects, positions = z.shape
    a, b, e, grand = components(z)
    ss = {
        "position": float(subjects * np.square(a).sum()),
        "subject": float(positions * np.square(b).sum()),
        "interaction": float(np.square(e).sum()),
    }
    total = float(np.square(z - grand).sum())
    if total > 1e-12 and abs(sum(ss.values()) - total) / total > 1e-6:
        raise AssertionError(f"Variance split does not close: {sum(ss.values())} vs {total}")
    ss["total"] = total
    return ss


def correlate(x, y):
    """Pearson correlation over all entries, or None when either side is constant."""
    import numpy as np

    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    xs, ys = x - x.mean(), y - y.mean()
    den = float(np.sqrt(np.square(xs).sum() * np.square(ys).sum()))
    return float((xs * ys).sum() / den) if den > 1e-20 else None


def analyse(hz):
    """Per-channel variance split and per-component cross-view correlation.

    hz is (2, S, C, P): two views, subjects, content channels, patch positions.
    """
    import numpy as np

    hz = np.asarray(hz, dtype=np.float64)
    if hz.ndim != 4 or hz.shape[0] != 2:
        raise ValueError(f"Expected (2, S, C, P), got {hz.shape}")
    if hz.shape[3] < 2:
        raise ValueError("Need at least two patch positions; with P=1 there is no interaction")
    if not np.isfinite(hz).all():
        raise ValueError("Non-finite patch features")
    channels = []
    for channel in range(hz.shape[2]):
        per_view = [split_variance(hz[view, :, channel, :]) for view in range(2)]
        total = sum(v["total"] for v in per_view)
        shares = {
            part: (sum(v[part] for v in per_view) / total if total > 1e-20 else None)
            for part in ("position", "subject", "interaction")
        }
        parts = [components(hz[view, :, channel, :]) for view in range(2)]
        channels.append(
            {
                "channel": channel,
                "shares": shares,
                "variance": total,
                "cross_view": {
                    name: correlate(parts[0][i], parts[1][i])
                    for i, name in enumerate(("position", "subject", "interaction"))
                },
            }
        )
    # Pooled over channels, weighted by variance -- the objective sees all channels at once,
    # so an unweighted mean would let a near-dead channel count like a live one.
    weights = np.array([c["variance"] for c in channels], dtype=float)
    pooled_shares = {}
    for part in ("position", "subject", "interaction"):
        values = np.array([c["shares"][part] or 0.0 for c in channels], dtype=float)
        pooled_shares[part] = float((values * weights).sum() / weights.sum()) if weights.sum() > 0 else None
    pooled_cross = {
        name: correlate(
            np.concatenate([components(hz[0, :, c, :])[i].ravel() for c in range(hz.shape[2])]),
            np.concatenate([components(hz[1, :, c, :])[i].ravel() for c in range(hz.shape[2])]),
        )
        for i, name in enumerate(("position", "subject", "interaction"))
    }
    return {
        "n_subjects": int(hz.shape[1]),
        "n_channels": int(hz.shape[2]),
        "n_positions": int(hz.shape[3]),
        "channels": channels,
        "pooled_shares": pooled_shares,
        "pooled_cross_view": pooled_cross,
    }


def fmt_pct(v):
    return "  n/a" if v is None else f"{100 * v:5.1f}%"


def fmt_r(v):
    return " n/a " if v is None else f"{v:+.3f}"


def print_report(result, center_mode=None):
    print("\n" + "=" * 92)
    print("PATCH VARIANCE SPLIT   z[s,p] = mean + a[p] + b[s] + e[s,p]")
    print(
        f"  {result['n_subjects']} subjects x {result['n_channels']} content channels"
        f" x {result['n_positions']} patch positions"
        + (f"   (run trains with center_mode={center_mode})" if center_mode else "")
    )
    print("=" * 92)
    print("\n  a[p] = anatomy (removed by 'position')   b[s] = flat per-subject code (also removed")
    print("  by 'double')   e[s,p] = subject x position interaction (all that 'double' leaves)")

    print("\n  SHARE OF VARIANCE, per channel                    CROSS-VIEW CORRELATION")
    print("  chan      a[p]     b[s]   e[s,p]   var share      a[p]     b[s]   e[s,p]   under double")
    order = sorted(result["channels"], key=lambda c: -c["variance"])
    total_var = sum(c["variance"] for c in result["channels"]) or 1.0
    for c in order:
        interaction = c["shares"]["interaction"]
        note = "DELETED" if interaction is not None and interaction < ANNIHILATED_BELOW else ""
        print(
            f"  {c['channel']:4d}  {fmt_pct(c['shares']['position'])}  {fmt_pct(c['shares']['subject'])}"
            f"  {fmt_pct(c['shares']['interaction'])}     {fmt_pct(c['variance'] / total_var)}"
            f"     {fmt_r(c['cross_view']['position'])}   {fmt_r(c['cross_view']['subject'])}"
            f"   {fmt_r(c['cross_view']['interaction'])}   {note}"
        )
    pooled = result["pooled_shares"]
    cross = result["pooled_cross_view"]
    print(
        f"  all   {fmt_pct(pooled['position'])}  {fmt_pct(pooled['subject'])}"
        f"  {fmt_pct(pooled['interaction'])}     100.0%"
        f"     {fmt_r(cross['position'])}   {fmt_r(cross['subject'])}   {fmt_r(cross['interaction'])}"
    )

    dead = [c["channel"] for c in result["channels"] if (c["shares"]["interaction"] or 0) < ANNIHILATED_BELOW]
    print(
        f"\n  Under 'double' the objective keeps {fmt_pct(pooled['interaction']).strip()} of the current patch"
        f" variance, and {len(dead)}/{result['n_channels']} channels"
    )
    print(f"  fall below {100 * ANNIHILATED_BELOW:.0f}% of their own{f': {dead}' if dead else ''}.")
    if cross["interaction"] is not None:
        print(
            f"\n  Expect pos_sim to fall to about {cross['interaction']:+.3f} under 'double': that is the"
            " cross-view correlation"
        )
        print("  of the part that survives, which is what BT would then be scored on.")
        if cross["subject"] is not None:
            print(f"  It currently sits near the b[s] correlation of {cross['subject']:+.3f}, which 'double' deletes.")
    print("\n  Read it as a trade: a large b[s] share means 'double' costs you real signal (global")
    print("  factors live there), and a large e[s,p] share means it costs little. Neither is free.")


# main()'s imports are deferred so --self-test needs no torch, which also means a rename in
# one of those modules would not surface until a run had already loaded a checkpoint on the
# cluster. This checks the names exist by parsing, no import required.
REQUIRED_IMPORTS = {
    "eval/bt_term_balance.py": ("_as_views", "foreground_keep"),
    "eval/dci.py": ("_extract_synthetic_representations",),
    "eval/run_dci_compare.py": ("_CONTENT", "_CONTENT_V2"),
    "eval/run_dci_synthetic.py": ("build_synthetic_test_set", "load_model_from_run_dir", "load_run_args"),
}


def module_bindings(path):
    """Every name a module binds at top level: defs, classes, assignments, imports."""
    import ast

    names = set()
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            stack = list(targets)
            while stack:
                target = stack.pop()
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    # _CONTENT, _STYLE = 0, 1 binds both, and missing this is what let a
                    # wrong import module reach the cluster.
                    stack.extend(target.elts)
    return names


def _check_imports(root=None):
    root = Path(root or Path(__file__).resolve().parent.parent)
    results = []
    for path, names in REQUIRED_IMPORTS.items():
        full = root / path
        if not full.exists():
            results.extend((f"{path}::{n}", False) for n in names)
            continue
        bound = module_bindings(full)
        results.extend((f"{path}::{n}", n in bound) for n in names)
    return results


def _self_test():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"self-test needs numpy ({exc})")

    import_checks = _check_imports()
    print("self-test (deferred imports resolve)")
    for label, ok in import_checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print()

    rng = np.random.default_rng(0)
    subjects, positions = 60, 24

    # A raw (S, P) Gaussian is NOT a pure interaction -- it carries its own position and
    # subject marginals -- so each planted component is projected onto its own subspace
    # first and scaled to contribute exactly 1 to the sum of squares. Then a mixture's
    # shares are its squared coefficients, because the three subspaces are orthogonal.
    def plant_position():
        a = rng.normal(size=positions)
        a -= a.mean()
        return a / np.sqrt(subjects * np.square(a).sum())

    def plant_subject():
        b = rng.normal(size=subjects)
        b -= b.mean()
        return b / np.sqrt(positions * np.square(b).sum())

    def plant_interaction():
        e = rng.normal(size=(subjects, positions))
        e = e - e.mean(axis=0)[None, :] - e.mean(axis=1)[:, None] + e.mean()
        return e / np.sqrt(np.square(e).sum())

    a, b, e, other = plant_position(), plant_subject(), plant_interaction(), plant_interaction()
    # Four channels: pure anatomy, pure flat-subject code, pure interaction, 50/30/20 mixture.
    hz = np.zeros((2, subjects, 4, positions))
    for view in range(2):
        hz[view, :, 0, :] = a[None, :] * 10.0
        hz[view, :, 1, :] = b[:, None] * 10.0
        hz[view, :, 2, :] = e * 10.0
        hz[view, :, 3, :] = (a[None, :] * np.sqrt(0.50) + b[:, None] * np.sqrt(0.30) + e * np.sqrt(0.20)) * 30.0
    # Decorrelate view 1's interaction only, so the components are read independently.
    # Equal-weight mix of two orthonormal interactions correlates at 1/sqrt(2) ~ 0.707.
    hz[1, :, 2, :] = (e + other) / np.sqrt(2.0) * 10.0
    got = analyse(hz)
    ch = {c["channel"]: c for c in got["channels"]}

    def share(channel, part):
        return ch[channel]["shares"][part]

    checks = [
        ("pure a[p] reads 100% position", abs(share(0, "position") - 1.0) < 1e-6),
        ("pure b[s] reads 100% subject", abs(share(1, "subject") - 1.0) < 1e-6),
        ("pure e reads 100% interaction", abs(share(2, "interaction") - 1.0) < 1e-6),
        ("pure a[p] would be DELETED by double", share(0, "interaction") < ANNIHILATED_BELOW),
        ("pure b[s] would be DELETED by double", share(1, "interaction") < ANNIHILATED_BELOW),
        ("pure e survives double", share(2, "interaction") > 0.99),
        ("mixture recovers ~50% position", abs(share(3, "position") - 0.5) < 0.02),
        ("mixture recovers ~30% subject", abs(share(3, "subject") - 0.3) < 0.02),
        ("mixture recovers ~20% interaction", abs(share(3, "interaction") - 0.2) < 0.02),
        ("shares sum to 1 per channel", all(abs(sum(c["shares"].values()) - 1.0) < 1e-9 for c in got["channels"])),
        ("identical views correlate at +1 on b[s]", abs(ch[1]["cross_view"]["subject"] - 1.0) < 1e-6),
        ("decorrelated interaction reads ~0.7", 0.6 < ch[2]["cross_view"]["interaction"] < 0.8),
        ("pooled shares sum to 1", abs(sum(got["pooled_shares"].values()) - 1.0) < 1e-9),
    ]
    print("self-test (variance split)")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print_report(got, center_mode="position")
    if not all(ok for _, ok in checks + import_checks):
        raise SystemExit("self-test failed")
    print("\nall checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir")
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--causal", choices=("match", "iid"), default="iid")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default=None, help="Also write the per-channel numbers as JSON + CSV")
    p.add_argument("--self-test", action="store_true")
    cli = p.parse_args()
    if cli.self_test:
        _self_test()
        return
    if not cli.run_dir:
        p.error("Need --run-dir, or --self-test")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import os

    import numpy as np

    # Same sources bt_term_balance imports these from, so the extraction is byte-for-byte
    # the path that script measures the BT terms on.
    from eval.bt_term_balance import _as_views, foreground_keep
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    args_ = load_run_args(cli.run_dir)
    grid = getattr(args_, "patch_grid", None)
    if not grid:
        raise SystemExit("This run has no --patch-grid; there are no patch positions to split")
    center_mode = getattr(args_, "patch_center_mode", "none") or "none"
    dataset = build_synthetic_test_set(args_, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)
    levels, _gt, _s1, _s2 = _extract_synthetic_representations(
        model, dataset, device, cli.encode_batch, cli.num_workers, pooling=tuple(grid)
    )
    if cli.level not in levels:
        raise SystemExit(f"Level {cli.level} not in extracted representations: {sorted(levels)}")
    c1, c2 = levels[cli.level][_CONTENT], levels[cli.level][_CONTENT_V2]
    if c1 is None or c2 is None or c1.shape[1] == 0:
        raise SystemExit("No content channels at this level")
    hz = _as_views(c1, c2, int(np.prod(grid)))
    if bool(getattr(args_, "patch_foreground_mask", False)):
        thresh = float(getattr(args_, "patch_foreground_thresh", 0.05))
        keep = foreground_keep(dataset, tuple(grid), thresh, cli.encode_batch, hz.device)
        if keep is not None and bool(keep.any()):
            logger.info("Foreground patches: keeping %d/%d positions.", int(keep.sum()), int(keep.numel()))
            hz = hz[..., keep.to(hz.device)]
        else:
            logger.warning("Foreground mask kept nothing; measuring over all positions, unlike training.")
    result = analyse(hz.cpu().numpy())
    result["center_mode"] = center_mode
    result["patch_grid"] = list(grid)
    print_report(result, center_mode)
    if cli.out:
        out = Path(cli.out)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with out.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "channel",
                    "share_position",
                    "share_subject",
                    "share_interaction",
                    "variance_share",
                    "cross_view_position",
                    "cross_view_subject",
                    "cross_view_interaction",
                ]
            )
            total = sum(c["variance"] for c in result["channels"]) or 1.0
            for c in result["channels"]:
                writer.writerow(
                    [
                        c["channel"],
                        c["shares"]["position"],
                        c["shares"]["subject"],
                        c["shares"]["interaction"],
                        c["variance"] / total,
                        c["cross_view"]["position"],
                        c["cross_view"]["subject"],
                        c["cross_view"]["interaction"],
                    ]
                )
        print(f"\nWrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
