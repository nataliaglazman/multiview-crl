#!/usr/bin/env python
"""Score several feature bundles through one protocol and put them in one table.

    python -m eval.compare_bundles \\
        --bundles vq_all=results/bundles/vq_all_gap.npz \\
                  vq_content=results/bundles/vq_content_gap.npz \\
                  dino=results/3dino/pretrained.npz \\
        --floors  vq_all=results/bundles/vq_all_gap_floor.npz \\
                  dino=results/3dino/random_init.npz \\
        --equal-width --out results/matched/compare.json

Every bundle is scored by ``eval.dinov3_identifiability.score`` with ONE options object,
so the CV splits, the permutation nulls, the PCA rule, the probe and the floor handling
are not merely specified the same -- they are the same objects doing the same work.  The
splits depend on row count alone and the nulls are redrawn from a fixed seed per bundle,
so bundle *k*'s fold *i* holds the same rows as bundle *j*'s fold *i* and both are
differenced against the same permutation.

Three things this checks that a pair of separate runs cannot:

1. **Row identity.**  Bundles are compared only when their factor digests agree
   (``eval/bundle_identity.py``).  Equal ``--num-samples`` is not evidence of that: two
   draws of one generator at different seeds give 2000 rows each and share none.
2. **Feature width.**  ``auto`` PCA picks a width per block, so two bundles can arrive at
   the table with different probe capacity and part of the difference between them is that.
   The effective width is printed per bundle, and ``--equal-width`` pins every bundle to
   the narrowest block's width so the comparison is at matched capacity.
3. **Floor pairing.**  A floor belongs to its own architecture.  Each bundle's Δfloor is
   computed against the floor given for THAT bundle; one architecture's floor is never
   subtracted from another's, which would not be a baseline correction at all.

Read ``gap`` (real minus permutation null) across bundles, and ``Δfloor`` only where both
bundles have their own floor.  The columns are defined in
``eval/COMPARING_3DINO_VQVAE.md``; ``--self-test`` runs the whole path on planted numpy
arrays with no bundle files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from eval import dinov3_identifiability as scorer
from eval.bundle_identity import compare as compare_identity
from eval.bundle_identity import read_identity
from eval.run_causal_recovery import INDEP_TESTS
from eval.run_dci_compare import PROBE_DIM_AUTO

logger = logging.getLogger(__name__)


def parse_named(values, flag):
    """``label=path`` pairs -> an ordered dict, with the duplicate-label case rejected."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"{flag} takes label=path, got {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"{flag} entry {item!r} has an empty label")
        if label in out:
            raise argparse.ArgumentTypeError(f"{flag} label {label!r} appears twice")
        out[label] = Path(path)
    return out


def check_alignment(bundles, strict=True):
    """Verify every bundle describes the same evaluation rows. Returns the problems found."""
    # A bundle loaded from disk carries the identity taken from the file's own arrays.
    # One assembled in memory (the self-test, callers holding arrays) has no file to read,
    # so fall back to digesting the content factors it does have -- a weaker check, since
    # it cannot see a style-factor or adjacency difference, but not a silent one.
    records = {
        label: bundle.get("identity") or read_identity({}, {"z_content": bundle["z_content"]}, len(bundle["X"]))
        for label, bundle in bundles.items()
    }
    problems = compare_identity(records)
    if problems and strict:
        raise SystemExit(
            "Refusing to compare bundles that are not row-aligned:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRebuild them from one --run-dir with one --num-samples, or pass "
            "--allow-row-mismatch to score them anyway (the table will not be a comparison)."
        )
    for line in problems:
        logger.warning("%s", line)
    return records, problems


def common_width(bundles, floors):
    """The narrowest feature count across everything that will be probed.

    A floor is included because it is probed too: pinning the models to a width its own
    floor cannot reach would compare a reduced model against an unreduced baseline.
    """
    widths = [b["X"].shape[1] for b in bundles.values()]
    widths += [f["X"].shape[1] for f in floors.values() if f is not None]
    return int(min(widths))


def score_all(bundles, floors, options):
    """``{label: report}``, each produced by the identical scoring call."""
    results = {}
    for label, bundle in bundles.items():
        floor = floors.get(label)
        if floor is not None and floor["X"].shape[0] != bundle["X"].shape[0]:
            raise SystemExit(f"floor for {label!r} has {len(floor['X'])} rows, bundle has {len(bundle['X'])}")
        logger.info("Scoring %s (%d x %d)", label, *bundle["X"].shape)
        results[label] = scorer.score(bundle, floor, options)
    return results


def factor_rows(results, block):
    """``[(factor, {label: row})]`` over the factors every bundle scored, in bundle order."""
    per_label = {label: result.get(block, {}) for label, result in results.items()}
    names = [n for n in next(iter(per_label.values()), {}) if n != "_block"]
    shared = [n for n in names if all(n in rows for rows in per_label.values())]
    for name in names:
        if name not in shared:
            logger.warning("Factor %r is missing from some bundles and is left out of the table", name)
    return [(name, {label: rows[name] for label, rows in per_label.items()}) for name in shared]


def format_block(results, block, column):
    """One side-by-side table: factors down, bundles across, ``column`` in the cells."""
    rows = factor_rows(results, block)
    if not rows:
        return ""
    labels = list(results)
    if not any(column in cells[label] for _n, cells in rows for label in labels):
        return ""
    headers = ["factor", *labels]
    if len(labels) == 2:
        headers.append(f"{labels[1]}−{labels[0]}")
    body = []
    for name, cells in rows:
        line = [name] + [scorer._fmt(cells[label].get(column)) for label in labels]
        if len(labels) == 2:
            a, b = cells[labels[0]].get(column), cells[labels[1]].get(column)
            line.append(scorer._fmt(None if a is None or b is None else b - a))
        body.append(line)
    means = []
    for label in labels:
        vals = [cells[label].get(column) for _n, cells in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        means.append(scorer._fmt(float(np.mean(vals)) if vals else None))
    footer = ["mean", *means]
    if len(labels) == 2:
        footer.append("")
    title = {"gap": "real − permutation null", "delta_floor": "gap − own untrained floor"}[column]
    return f"{block.upper()} · {title}\n{scorer._table(headers, [*body, footer])}\n"


def format_widths(results):
    lines = []
    for label, result in results.items():
        block = result.get("content", {}).get("_block", {})
        lines.append(
            f"  {label:<14} {result['num_features']:>7} features"
            f" -> {block.get('probe_features', '?'):>5} probed"
            f"   floor: {'yes' if result.get('floor_path') else 'no'}"
        )
    widths = {r.get("content", {}).get("_block", {}).get("probe_features") for r in results.values()}
    note = ""
    if len({w for w in widths if w is not None}) > 1:
        note = (
            "\n  NOTE: the bundles were probed at different widths, so part of any difference\n"
            "        below is probe capacity rather than representation. Pass --equal-width.\n"
        )
    return "FEATURE WIDTH\n" + "\n".join(lines) + "\n" + note


def format_report(results, problems):
    parts = [
        "=" * 92,
        "MATCHED BUNDLE COMPARISON",
        "=" * 92,
        "",
        format_widths(results),
    ]
    if problems:
        parts += ["ROW ALIGNMENT", *(f"  !! {p}" for p in problems), ""]
    for block in ("content", "style"):
        for column in ("gap", "delta_floor"):
            table = format_block(results, block, column)
            if table:
                parts.append(table)
    parts.append(
        "Read across a row for the same factor under different representations. 'gap' is\n"
        "comparable everywhere; 'Δfloor' only where each bundle has its own floor.\n"
    )
    return "\n".join(parts)


def write_csv(path, results):
    columns = [
        "bundle",
        "block",
        "factor",
        "real",
        "null",
        "gap",
        "std",
        "mcc",
        "floor_gap",
        "delta_floor",
        "voxels_gap",
        "delta_voxels",
        "probe_features",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for label, result in results.items():
            for block in ("content", "style"):
                rows = result.get(block, {})
                width = rows.get("_block", {}).get("probe_features")
                for name, row in rows.items():
                    if name == "_block":
                        continue
                    writer.writerow({**row, "bundle": label, "block": block, "factor": name, "probe_features": width})


def _self_test():
    """Two representations of one planted factor set, scored through the real path."""
    rng = np.random.RandomState(0)
    n, k = 240, 3
    z = rng.randn(n, k)
    z[:, 1] += 1.3 * z[:, 0]
    z[:, 2] += 1.3 * z[:, 1]
    names = ["a", "b", "c"]
    strong = z @ rng.randn(k, 32) + 0.1 * rng.randn(n, 32)
    weak = z @ rng.randn(k, 32) + 3.0 * rng.randn(n, 32)

    def make(X):
        return dict(
            path="<self-test>",
            X=X,
            raw=None,
            z_content=z,
            z_style=None,
            adjacency=None,
            content_names=names,
            style_names=[],
            meta={},
            view="1",
        )

    bundles = {"strong": make(strong), "weak": make(weak)}
    options = argparse.Namespace(
        probe_kind="ridge", seeds=(0, 1), n_splits=3, n_null=2, null_seed=0, probe_dim=0, with_graph=False
    )
    records, problems = check_alignment(bundles, strict=True)
    assert not problems, problems
    assert len({r["factor_digest"] for r in records.values()}) == 1, "identical factors must digest alike"
    results = score_all(bundles, {}, options)
    print(format_report(results, problems))
    gaps = {label: result["content"]["_block"]["mean_gap"] for label, result in results.items()}
    assert gaps["strong"] > gaps["weak"], gaps
    assert gaps["strong"] > 0.5, gaps

    mismatched = {"strong": bundles["strong"], "shifted": make(weak)}
    mismatched["shifted"]["z_content"] = z + 1.0
    assert check_alignment(mismatched, strict=False)[1], "a different factor draw must be reported"
    print("SELF-TEST OK: matched scoring separates the two, and a factor mismatch is caught.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundles", nargs="+", metavar="LABEL=PATH", help="Two or more bundles to compare")
    parser.add_argument(
        "--floors",
        nargs="+",
        metavar="LABEL=PATH",
        default=[],
        help="Untrained twin per bundle label; each is only ever subtracted from its own bundle",
    )
    parser.add_argument("--view", default="1", choices=["1", "2", "both"])
    parser.add_argument("--out", type=Path, help="Write the full report as JSON (and .txt alongside)")
    parser.add_argument("--csv", type=Path, help="Write the per-factor rows as CSV")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run on planted numpy data and exit")
    parser.add_argument(
        "--allow-row-mismatch",
        dest="strict_rows",
        action="store_false",
        help="Score bundles whose rows differ. The output is then not a comparison.",
    )

    probe = parser.add_argument_group("probes (one setting, applied to every bundle)")
    probe.add_argument("--probe-kind", default="ridge", choices=["ridge", "kernel", "mlp"])
    probe.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    probe.add_argument("--n-splits", type=int, default=5)
    probe.add_argument("--n-null", type=int, default=3)
    probe.add_argument("--null-seed", type=int, default=0)
    probe.add_argument(
        "--probe-dim", default=PROBE_DIM_AUTO, help="PCA width for every bundle: 'auto', an integer, or 0 for none"
    )
    probe.add_argument(
        "--equal-width",
        action="store_true",
        help="Override --probe-dim with the narrowest block's width, so every bundle " "is probed at the same capacity",
    )

    graph = parser.add_argument_group("graph")
    graph.add_argument("--with-graph", action="store_true", help="Also run the PC panel per bundle (slow)")
    graph.add_argument("--alphas", type=float, nargs="+", default=list(scorer.DEFAULT_ALPHAS))
    graph.add_argument("--diagnostic-alpha", type=float, default=0.05)
    graph.add_argument("--no-orientation", dest="orientation", action="store_false")
    graph.add_argument("--no-pc-ceiling", dest="pc_ceiling", action="store_false")
    graph.add_argument("--indep-test", default="fisherz", choices=list(INDEP_TESTS))
    graph.add_argument("--max-cond-set", type=int)
    graph.add_argument("--readout-dim", type=int)
    graph.add_argument("--holdout-readout", action="store_true")

    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if cli.self_test:
        _self_test()
        return 0
    if not cli.bundles or len(cli.bundles) < 2:
        parser.error("--bundles needs at least two label=path entries (or pass --self-test)")
    if cli.probe_dim != PROBE_DIM_AUTO:
        try:
            cli.probe_dim = int(cli.probe_dim)
        except ValueError:
            parser.error(f"--probe-dim must be an integer or {PROBE_DIM_AUTO!r}")
    if cli.n_null < 0 or cli.n_splits < 2 or not cli.seeds:
        parser.error("Require --n-null >= 0, --n-splits >= 2 and at least one seed")
    cli.seeds = tuple(cli.seeds)

    bundle_paths = parse_named(cli.bundles, "--bundles")
    floor_paths = parse_named(cli.floors, "--floors")
    unknown = set(floor_paths) - set(bundle_paths)
    if unknown:
        parser.error(f"--floors labels not in --bundles: {', '.join(sorted(unknown))}")

    bundles = {label: scorer.load_bundle(path, cli.view) for label, path in bundle_paths.items()}
    floors = {label: scorer.load_bundle(path, cli.view) for label, path in floor_paths.items()}
    _records, problems = check_alignment(bundles, strict=cli.strict_rows)
    if cli.equal_width:
        cli.probe_dim = common_width(bundles, floors)
        logger.info("--equal-width: probing every bundle at %d features", cli.probe_dim)

    results = score_all(bundles, floors, cli)
    report = format_report(results, problems)
    payload = {
        "bundles": {label: str(path) for label, path in bundle_paths.items()},
        "floors": {label: str(path) for label, path in floor_paths.items()},
        "row_alignment": problems or "verified",
        "options": scorer.options_record(cli),
        "results": results,
    }
    saved = []
    if cli.out:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        cli.out.with_suffix(".txt").write_text(report)
        saved += [cli.out.resolve(), cli.out.with_suffix(".txt").resolve()]
    if cli.csv:
        cli.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(cli.csv, results)
        saved.append(cli.csv.resolve())
    if not cli.quiet:
        print(report)
    for path in saved:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
