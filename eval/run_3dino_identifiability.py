#!/usr/bin/env python
"""Extract and evaluate 3DINO embeddings using the same scorer as 2D DINO.

    python -m eval.run_3dino_identifiability --three-dino-repo ../3DINO \
        --three-dino-weights /path/to/teacher.pth --run-dir results/synthetic/RUN \
        --output-dir results/3dino_evaluation --with-floor

    python -m eval.run_3dino_identifiability --embeddings embeddings.npz \
        --floor random_init.npz --output-dir results/3dino_rescored
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_graph_support(max_cond_set):
    """Older causal-learn releases silently swallow max_k in **kwargs."""
    if max_cond_set is not None:
        import inspect

        from causallearn.search.ConstraintBased.PC import pc

        if "max_k" not in inspect.signature(pc).parameters:
            raise ValueError(
                "Installed causal-learn does not support --max-cond-set. "
                "Upgrade with: python -m pip install 'causal-learn>=0.1.4.8'"
            )


def validate_bundle(path, views):
    """Fail early on a 2D, malformed, or nonfinite embedding artifact."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("backbone") != "3dino":
            raise ValueError(f"{path} is not a 3DINO embedding artifact (meta.backbone must be '3dino')")
        z = data["z_content"]
        if z.ndim != 2 or len(z) < 20 or not np.isfinite(z).all():
            raise ValueError(f"{path}: require at least 20 rows of finite content labels")
        for view in set(v for option in views for v in (["1", "2"] if option == "both" else [option])):
            X = data[f"emb_view{view}"]
            if X.ndim != 2 or len(X) != len(z) or X.shape[1] == 0 or not np.isfinite(X).all():
                raise ValueError(f"{path}: invalid or misaligned view {view} embeddings")
        return meta


def validate_floor(embeddings, floor, views):
    """A floor must change weights only, not samples, modalities, or preprocessing."""
    import numpy as np

    main_meta, floor_meta = (validate_bundle(path, views) for path in (embeddings, floor))
    if not floor_meta.get("random_init"):
        raise ValueError("The supplied --floor is not marked as random initialization")
    keys = (
        "model_id",
        "architecture",
        "volume_size",
        "patch_size",
        "token_pool",
        "grid_size",
        "image_mean",
        "image_std",
        "window",
        "window_pct",
        "window_values",
        "synthetic_fixed_reference",
        "generator",
        "raw_grid",
        "dtype",
    )
    for key in keys:
        if key not in main_meta or key not in floor_meta:
            raise ValueError(f"Floor comparison cannot verify missing metadata: {key}")
        if main_meta[key] != floor_meta[key]:
            raise ValueError(f"Floor preprocessing/model mismatch: {key}")
    revisions = [(meta.get("model_provenance") or {}).get("revision") for meta in (main_meta, floor_meta)]
    if revisions[0] != revisions[1]:
        raise ValueError("Floor and embeddings use different upstream model revisions")
    with np.load(embeddings, allow_pickle=False) as first, np.load(floor, allow_pickle=False) as second:
        for key in (
            "z_content",
            "z_style_v1",
            "z_style_v2",
            "causal_adj",
            "raw_view1",
            "raw_view2",
        ):
            if (key in first) != (key in second):
                raise ValueError(f"Floor artifact is missing a matching {key} array")
            if key in first and (
                first[key].shape != second[key].shape or not np.allclose(first[key], second[key], rtol=1e-6, atol=1e-7)
            ):
                raise ValueError(f"Floor uses different or reordered {key} values")
        for view in set(v for option in views for v in (["1", "2"] if option == "both" else [option])):
            if first[f"emb_view{view}"].shape != second[f"emb_view{view}"].shape:
                raise ValueError(f"Floor and trained embeddings have different widths for view {view}")


def extraction_args(cli, output, random_init=False):
    args = [
        "--backbone",
        "3dino",
        "--three-dino-repo",
        str(cli.three_dino_repo),
        "--three-dino-weights",
        str(cli.three_dino_weights),
        "--out",
        str(output),
        "--num-samples",
        str(cli.num_samples),
        "--volume-batch",
        str(cli.volume_batch),
        "--volume-size",
        str(cli.volume_size),
        "--token-pool",
        cli.token_pool,
        "--grid-size",
        str(cli.grid_size),
        "--window",
        cli.window,
        "--window-pct",
        *map(str, cli.window_pct),
        "--raw-grid",
        str(cli.raw_grid),
        "--num-workers",
        str(cli.num_workers),
        "--dtype",
        cli.dtype,
        "--views",
        "1",
        "2",
    ]
    for key in ("run_dir", "preprocessing", "device"):
        if getattr(cli, key) is not None:
            args.extend(["--" + key.replace("_", "-"), str(getattr(cli, key))])
    if cli.no_cache:
        args.append("--no-cache")
    if random_init:
        args.extend(["--random-init", "--model-seed", str(cli.floor_seed)])
    return args


def scoring_args(cli, embeddings, floor, view):
    base = cli.output_dir / f"report_view{view}"
    args = [
        "--embeddings",
        str(embeddings),
        "--view",
        view,
        "--out",
        str(base.with_suffix(".json")),
        "--csv",
        str(cli.output_dir / f"factors_view{view}.csv"),
        "--probe-kind",
        cli.probe_kind,
        "--probe-dim",
        cli.probe_dim,
        "--n-splits",
        str(cli.n_splits),
        "--n-null",
        str(cli.n_null),
        "--null-seed",
        str(cli.null_seed),
        "--seeds",
        *map(str, cli.seeds),
        "--alphas",
        *map(str, cli.alphas),
        "--diagnostic-alpha",
        str(cli.diagnostic_alpha),
        "--indep-test",
        cli.indep_test,
    ]
    if floor:
        args.extend(["--floor", str(floor)])
    for key in ("readout_dim", "max_cond_set"):
        if getattr(cli, key) is not None:
            args.extend(["--" + key.replace("_", "-"), str(getattr(cli, key))])
    for key in ("holdout_readout", "no_graph", "no_orientation"):
        if getattr(cli, key):
            args.append("--" + key.replace("_", "-"))
    return args


def run_stage(name, module, args, output_dir, manifest):
    """Isolated processes release encoder/GPU memory before the next stage."""
    command = [sys.executable, "-m", module, *args]
    entry = dict(
        name=name,
        command=command,
        log=str(output_dir / f"{name}.log"),
        status="running",
    )
    manifest["stages"].append(entry)
    manifest_path = output_dir / "pipeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[{name}] {shlex.join(command)}", flush=True)
    with Path(entry["log"]).open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    entry.update(status="complete" if code == 0 else "failed", returncode=code)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {entry['log']}. Completed artifacts are retained.")


def write_summary(output_dir, views):
    rows = []
    for view in views:
        report_path = output_dir / f"report_view{view}.json"
        report = json.loads(report_path.read_text())
        block = report["content"].get("_block", {})
        graph = report["graph"].get("embeddings", {})
        best = graph.get("best") or {}
        rows.append(
            dict(
                view=view,
                run_dir=report["embeddings_meta"].get("run_dir"),
                embeddings=report["embeddings_path"],
                num_samples=report["num_samples"],
                num_features=report["num_features"],
                content_gap=block.get("mean_gap"),
                content_mcc=block.get("mcc_mean"),
                floor_gap=block.get("floor_mean_gap"),
                voxel_gap=block.get("voxels_mean_gap"),
                f1=best.get("f1"),
                precision=best.get("precision"),
                recall=best.get("recall"),
                shd=best.get("skeleton_shd"),
                alpha=best.get("alpha"),
                partial_r2=graph.get("partial_r2_mean"),
                report=str(report_path),
                graph_status=("ok" if best else "unavailable" if report["graph"] else "not_evaluated"),
            )
        )
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = [
        "View",
        "Content gap",
        "MCC",
        "F1",
        "Precision",
        "Recall",
        "SHD",
        "Partial R²",
        "Graph",
    ]
    table = [headers]
    for row in rows:
        metrics = [
            row[key]
            for key in (
                "content_gap",
                "content_mcc",
                "f1",
                "precision",
                "recall",
                "shd",
                "partial_r2",
            )
        ]
        table.append(
            [
                row["view"],
                *["—" if value is None or not math.isfinite(value) else f"{value:.3f}" for value in metrics],
                row["graph_status"],
            ]
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    text = "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in table)
    text += "\nContent gap = cross-validated R² minus permutation null; graph R² uses its own probe protocol.\n"
    text += "Graph F1/precision/recall/SHD use best-F1 alpha selected against truth; — = unavailable.\n"
    (output_dir / "summary.txt").write_text(text)
    print("\n" + text)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--three-dino-weights",
        type=Path,
        help="Pretrained .pth or fine-tuned encoder directory",
    )
    source.add_argument("--embeddings", type=Path, help="Existing 3DINO NPZ; skip all model inference")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for this evaluation",
    )
    parser.add_argument("--three-dino-repo", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Synthetic generator settings; auto-detected for fine-tuned encoders",
    )
    parser.add_argument(
        "--preprocessing",
        type=Path,
        help="Saved preprocessing; auto-detected for fine-tuned encoders",
    )
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--with-floor",
        action="store_true",
        help="Also extract a matched random-initialization baseline",
    )
    baseline.add_argument("--floor", type=Path, help="Use an existing matched random-initialization NPZ")
    parser.add_argument("--floor-seed", type=int, default=0)
    parser.add_argument("--eval-views", nargs="+", choices=["1", "2", "both"], default=["1", "2"])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--volume-size", type=int, default=112)
    parser.add_argument("--volume-batch", type=int, default=2)
    parser.add_argument("--token-pool", choices=["cls", "mean", "cls_mean", "grid"], default="cls")
    parser.add_argument("--grid-size", type=int, default=2)
    parser.add_argument("--window", choices=["dataset", "per_volume"], default="per_volume")
    parser.add_argument("--window-pct", type=float, nargs=2, default=[0.05, 99.95])
    parser.add_argument("--raw-grid", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--probe-kind", choices=["ridge", "kernel", "mlp"], default="ridge")
    parser.add_argument("--probe-dim", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=3)
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-orientation", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2])
    parser.add_argument("--diagnostic-alpha", type=float, default=0.05)
    parser.add_argument("--indep-test", choices=["fisherz", "kci"], default="fisherz")
    parser.add_argument("--max-cond-set", type=int)
    parser.add_argument("--readout-dim", type=int)
    parser.add_argument("--holdout-readout", action="store_true")
    cli = parser.parse_args(argv)
    for key, value in vars(cli).items():
        if isinstance(value, Path):
            setattr(cli, key, value.expanduser().resolve())
    cli.eval_views = list(dict.fromkeys(cli.eval_views))
    if cli.n_splits < 2 or cli.n_null < 1 or cli.num_samples < max(20, 2 * cli.n_splits):
        parser.error("Require n-splits >= 2, n-null >= 1, num-samples >= max(20, 2*n-splits)")
    if cli.probe_dim != "auto" and (not cli.probe_dim.isdigit()):
        parser.error("--probe-dim must be 'auto' or a nonnegative integer")
    if any(not 0 < a < 1 for a in [*cli.alphas, cli.diagnostic_alpha]):
        parser.error("PC alphas must be between 0 and 1")
    if cli.max_cond_set is not None and cli.max_cond_set < 0:
        parser.error("--max-cond-set must be nonnegative")
    if cli.readout_dim is not None and cli.readout_dim < 1:
        parser.error("--readout-dim must be positive")
    if not cli.no_graph:
        validate_graph_support(cli.max_cond_set)
    if cli.volume_size < 16 or cli.volume_size % 16 or cli.volume_batch < 1 or cli.grid_size < 1:
        parser.error("Require volume-size a positive multiple of 16 and positive volume-batch/grid-size")
    if cli.raw_grid < 0 or cli.num_workers < 0 or not 0 <= cli.window_pct[0] < cli.window_pct[1] <= 100:
        parser.error("Invalid raw-grid, num-workers or window percentiles")
    if cli.embeddings and cli.with_floor:
        parser.error("--with-floor needs --three-dino-weights; with existing embeddings use --floor")
    if cli.three_dino_weights:
        if not cli.three_dino_weights.exists() or not cli.three_dino_repo:
            parser.error("Extraction requires an existing --three-dino-weights and --three-dino-repo")
        if not (cli.three_dino_repo / "dinov2/models/vision_transformer.py").is_file():
            parser.error("--three-dino-repo must point to the official 3DINO checkout")
        if cli.three_dino_weights.is_dir():
            parent = cli.three_dino_weights.parent
            if cli.run_dir is None and (parent / "settings.json").is_file():
                cli.run_dir = parent
            if cli.preprocessing is None and (parent / "preprocessing.json").is_file():
                cli.preprocessing = parent / "preprocessing.json"
    if cli.preprocessing and json.loads(cli.preprocessing.read_text()).get("backbone") != "3dino":
        parser.error("--preprocessing must be from a 3DINO run")
    if cli.run_dir and not (cli.run_dir / "settings.json").is_file():
        parser.error("--run-dir must contain settings.json")
    if cli.output_dir.exists() and (not cli.output_dir.is_dir() or any(cli.output_dir.iterdir())):
        parser.error("--output-dir must be new or empty; rescore saved NPZs into a new directory")
    embeddings = cli.embeddings or cli.output_dir / "embeddings.npz"
    floor = cli.floor or (cli.output_dir / "random_init.npz" if cli.with_floor else None)
    if cli.embeddings:
        validate_bundle(embeddings, cli.eval_views)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        stages=[],
        status="running",
    )
    try:
        if cli.embeddings is None:
            run_stage(
                "extract",
                "eval.dinov3_embed_synthetic",
                extraction_args(cli, embeddings),
                cli.output_dir,
                manifest,
            )
        meta = validate_bundle(embeddings, cli.eval_views)
        if meta.get("num_samples", 0) < 2 * cli.n_splits:
            raise ValueError("The embedding artifact has too few rows for the requested CV splits")
        if cli.with_floor:
            run_stage(
                "extract_floor",
                "eval.dinov3_embed_synthetic",
                extraction_args(cli, floor, True),
                cli.output_dir,
                manifest,
            )
        if floor:
            validate_floor(embeddings, floor, cli.eval_views)
        for view in cli.eval_views:
            run_stage(
                f"score_view{view}",
                "eval.dinov3_identifiability",
                scoring_args(cli, embeddings, floor, view),
                cli.output_dir,
                manifest,
            )
        rows = write_summary(cli.output_dir, cli.eval_views)
        manifest["status"] = "partial" if any(row["graph_status"] == "unavailable" for row in rows) else "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (cli.output_dir / "pipeline.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Evaluation outputs: {cli.output_dir}")
    return int(manifest["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
