#!/usr/bin/env python
"""Read-only diagnosis of a comparison manifest mismatch; no torch dependency.

Kept separate from compare_encoders.py so installing this diagnostic does not
change any of the Python sources that existing comparison manifests freeze.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_encoders as comparison  # noqa: E402


def differences(saved, current, prefix=""):
    """Describe added, removed and changed JSON values using dotted key paths."""
    if isinstance(saved, dict) and isinstance(current, dict):
        rows = []
        for key in sorted(saved.keys() | current.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in saved:
                rows.append(f"{path}: added {json.dumps(current[key])}")
            elif key not in current:
                rows.append(f"{path}: removed (saved {json.dumps(saved[key])})")
            else:
                rows.extend(differences(saved[key], current[key], path))
        return rows
    if saved != current:
        return [f"{prefix}: saved={json.dumps(saved)}, current={json.dumps(current)}"]
    return []


def diagnose(output, config, only_seed=None):
    output = output.resolve()
    manifest = comparison.read_json(output / "comparison_manifest.json")
    saved = manifest["config"]
    seeds = saved["seeds"] if only_seed is None else [only_seed]
    if any(seed not in saved["seeds"] for seed in seeds):
        raise ValueError("--only-seed must occur in the SAVED manifest configuration")
    report = {
        "output_dir": str(output),
        "schema_version": manifest.get("schema_version"),
        "configuration_differences": differences(saved, config),
        "source_differences": differences(manifest["sources"], comparison.source_hashes()),
        "runs": [],
    }
    # Check runs against their original recipe, not the possibly edited current one.
    for seed in seeds:
        for architecture in comparison.ARCHITECTURES:
            options = comparison.run_options(saved, output, seed, architecture)
            run = Path(options["out_dir"]) / options["model_id"]
            row = {"run": str(run), "model.pt": (run / "model.pt").is_file()}
            try:
                progress = comparison.read_json(run / "training_progress.json")
                row.update(status=progress.get("status"), step=progress.get("step"))
                comparison.validate_run(run, options)
                row["saved_recipe_and_checkpoint_receipt"] = "match"
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["saved_recipe_and_checkpoint_receipt"] = str(error)
            report["runs"].append(row)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep the raw string until validation: Path("") silently becomes the cwd.
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/encoder_comparison.json")
    parser.add_argument("--only-seed", type=int)
    args = parser.parse_args(argv)
    if not args.output_dir.strip():
        parser.error("--output-dir is empty; set OUT to the directory containing comparison_manifest.json")
    report = diagnose(Path(args.output_dir), comparison.read_json(args.config), args.only_seed)
    print(json.dumps(report, indent=2))
    print("\nRead-only check: no files changed and no training/evaluation started.")
    print("Checkpoint receipt matches do not certify which source code ran during training.")
    if report["source_differences"] or report["configuration_differences"]:
        print("Preserve the original manifest. Restore its code/configuration to use the verified comparison workflow.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
