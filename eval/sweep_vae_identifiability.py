"""Sweep identifiability across a set of trained MultiviewVAE runs.

Runs ``eval.analyze_vae_identifiability.run_analysis`` on every matching run dir
(default ``results/vae_synth_c*``) and prints one table, sorted by
``content_channels``, so the identifiability-vs-dimensionality curve is visible at
a glance.  Also writes a CSV.

The expected pattern when ``content_channels`` is swept around the true
``n_content``:
  * content->content block-MCC / informativeness rise, then plateau at ~n_content.
  * content->view/acc stays ~0.5 up to n_content, then climbs (over-allocation
    leakage) — so the "elbow" is the largest content size still at chance.

Usage:
    python -m eval.sweep_vae_identifiability
    python -m eval.sweep_vae_identifiability --results-glob 'results/vae_synth_c*' --out results/vae_sweep.csv
    python -m eval.sweep_vae_identifiability results/vae_synth_c6 results/vae_synth_c9
"""

import argparse
import csv
import glob
import os

import numpy as np

from eval.analyze_vae_identifiability import run_analysis

# (column header, metric key in the flat metrics dict)
COLUMNS = [
    ("cc_block_mcc", "content->content/block_mcc"),
    ("cc_info_ridge", "content->content/informativeness_ridge"),
    ("cs_block_mcc(leak)", "content->style/block_mcc"),
    ("content->view_acc", "content->view/acc"),
    ("ss_block_mcc", "style->style/block_mcc"),
    ("cc_per_chan_mcc", "per_channel_mcc/content->content"),
]


def _fmt(v):
    return "  nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="Explicit run dirs (overrides --results-glob if given)")
    p.add_argument("--results-glob", default="results/vae_synth_c*", help="Glob for run dirs")
    p.add_argument("--num-test-samples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--null-permute", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--out", default="results/vae_identifiability_sweep.csv")
    args = p.parse_args()

    device = "cpu" if args.no_cuda else None  # None → run_analysis auto-selects cuda if available

    run_dirs = args.run_dirs or sorted(glob.glob(args.results_glob))
    run_dirs = [d for d in run_dirs if os.path.exists(os.path.join(d, "model.pt"))]
    if not run_dirs:
        raise SystemExit(f"No run dirs with model.pt found (glob={args.results_glob!r}, explicit={args.run_dirs})")

    rows = []
    for run_dir in run_dirs:
        print(f"[sweep] analysing {run_dir} ...", flush=True)
        try:
            cfg, metrics = run_analysis(
                run_dir,
                num_test_samples=args.num_test_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                null_permute=args.null_permute,
                device=device,
            )
        except Exception as e:
            print(f"  [warn] skipped {run_dir}: {e}", flush=True)
            continue
        row = {
            "run": os.path.basename(run_dir.rstrip("/")),
            "content_channels": cfg.get("content_channels"),
            "latent_channels": cfg.get("latent_channels"),
            "n_content": cfg.get("n_content"),
        }
        for header, key in COLUMNS:
            row[header] = metrics.get(key, float("nan"))
        rows.append(row)

    rows.sort(key=lambda r: (r["content_channels"] is None, r["content_channels"]))

    headers = ["run", "content_channels", "latent_channels", "n_content"] + [c for c, _ in COLUMNS]
    widths = {h: max(len(h), 16 if h == "run" else 8) for h in headers}
    print("\n=== identifiability vs content dimensionality ===")
    print("  ".join(f"{h:>{widths[h]}}" for h in headers))
    for r in rows:
        line = []
        for h in headers:
            v = r[h]
            cell = _fmt(v) if isinstance(v, float) else str(v)
            line.append(f"{cell:>{widths[h]}}")
        print("  ".join(line))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV -> {args.out}")
    print(
        "\nRead: cc_block_mcc/cc_info_ridge should plateau at content_channels≈n_content;\n"
        "content->view_acc should stay ~0.5 until you over-allocate (then it climbs = leakage)."
    )


if __name__ == "__main__":
    main()
