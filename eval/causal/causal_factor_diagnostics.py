"""Factor attribution reports for run_causal_recovery (also replays old JSON)."""

import csv
import json
import math
from pathlib import Path

# z_content layout documented by Synthetic3DDisentanglementDataset.render_structure.
FACTOR_NAMES = (
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
)


def factor_name(dim):
    return FACTOR_NAMES[dim] if dim < len(FACTOR_NAMES) else f"d{dim}"


def graph_at_alpha(result, alpha):
    return next(
        (row for row in result.get("alpha_sweep", []) if math.isclose(row["alpha"], alpha) and "adjacency" in row), None
    )


def comparison_issues(reference, current):
    """Block deltas across known distribution/protocol changes; flag missing metadata."""
    changed, missing = [], []
    for key in ("true_dag", "num_samples", "level", "pooling", "causal_settings"):
        if key not in reference or key not in current:
            missing.append(key)
        elif json.dumps(reference[key], sort_keys=True) != json.dumps(current[key], sort_keys=True):
            changed.append(key)
    if [f["dim"] for f in reference["factors"]] != [f["dim"] for f in current["factors"]]:
        changed.append("factor_dimensions")
    return changed, missing


def build_factor_rows(results, reference_run=None, alpha=0.05):
    usable = [r for r in results if r.get("factors")]
    if not usable:
        return [], None
    reference = usable[0]
    if reference_run:
        exact = [r for r in usable if r["run_dir"] == reference_run]
        matches = exact or [r for r in usable if Path(r["run_dir"]).name == reference_run]
        if len(matches) != 1:
            raise ValueError("--reference-run must match exactly one evaluated directory path or basename")
        reference = matches[0]
    ref_factors = {f["dim"]: f for f in reference["factors"]}
    ref_graph = graph_at_alpha(reference, alpha)
    rows = []
    for order, result in enumerate(results):
        if not result.get("factors"):
            continue
        changed, missing = comparison_issues(reference, result)
        comparable = not changed
        graph = graph_at_alpha(result, alpha)
        truth = result.get("true_skeleton")
        drops = (
            {f["dim"]: max(0.0, ref_factors[f["dim"]]["partial_r2"] - f["partial_r2"]) for f in result["factors"]}
            if comparable
            else {}
        )
        total_drop = sum(drops.values())
        for factor in result["factors"]:
            d = factor["dim"]
            row = dict(
                run_order=order,
                directory_name=Path(result["run_dir"]).name,
                run_dir=result["run_dir"],
                reference_run=reference["run_dir"],
                dim=d,
                factor=factor.get("name", factor_name(d)),
                parents=factor["parents"],
                raw_r2=factor["raw_r2"],
                partial_r2=factor["partial_r2"],
                comparison_status=(
                    "incompatible: " + ", ".join(changed)
                    if changed
                    else "unverified: missing " + ", ".join(missing)
                    if missing
                    else "matched"
                ),
                diagnostic_alpha=alpha,
            )
            for key in ("readout_test_r2", "decoded_variance_ratio", "decoded_gt_correlation"):
                row[key] = factor.get(key)
            if comparable:
                ref = ref_factors[d]
                row.update(
                    delta_raw_r2=factor["raw_r2"] - ref["raw_r2"],
                    delta_partial_r2=factor["partial_r2"] - ref["partial_r2"],
                    mean_partial_delta_contribution=(factor["partial_r2"] - ref["partial_r2"]) / len(ref_factors),
                    partial_drop_share=drops[d] / total_drop if total_drop else 0.0,
                )
            if graph is not None and truth is not None:
                est = graph["adjacency"]
                tp = sum(bool(est[d][j]) and bool(truth[d][j]) for j in range(len(truth)) if j != d)
                fp = [j for j in range(len(truth)) if j != d and est[d][j] and not truth[d][j]]
                fn = [j for j in range(len(truth)) if j != d and truth[d][j] and not est[d][j]]
                p = tp / (tp + len(fp)) if tp + len(fp) else 0.0
                r = tp / (tp + len(fn)) if tp + len(fn) else 0.0
                row.update(
                    incident_precision=p,
                    incident_recall=r,
                    incident_f1=2 * p * r / (p + r) if p + r else 0.0,
                    incident_shd=len(fp) + len(fn),
                    false_neighbors=fp,
                    missing_neighbors=fn,
                )
                if comparable and ref_graph is not None:
                    ref_est = ref_graph["adjacency"]
                    ref_errors = sum(bool(ref_est[d][j]) != bool(truth[d][j]) for j in range(len(truth)) if j != d)
                    row["delta_incident_shd"] = row["incident_shd"] - ref_errors
            rescue = result.get("factor_rescue", {})
            if math.isclose(rescue.get("alpha", -1), alpha):
                repair = next((r for r in rescue.get("factors", []) if r["dim"] == d), {})
                row["rescue_f1_gain"] = repair.get("f1_gain")
                row["rescue_shd_reduction"] = repair.get("shd_reduction")
            rows.append(row)
    return rows, reference["run_dir"]


def format_factor_report(rows, reference, alpha):
    if not rows:
        return "No per-factor scores available.\n"
    lines = [
        f"Factor diagnostics — reference: {reference}",
        "Deltas are current minus reference; negative R² deltas mean degradation.",
        f"Graph diagnostics use fixed alpha={alpha:g}; unavailable metrics appear as —.",
        "Input order is preserved; directory names are not interpreted as training steps.",
    ]
    for run_dir in dict.fromkeys(row["run_dir"] for row in rows):
        run_rows = [r for r in rows if r["run_dir"] == run_dir]
        run_rows.sort(key=lambda r: r.get("delta_partial_r2", 0.0))
        lines.extend([f"\n{run_dir}", f"  Comparison: {run_rows[0]['comparison_status']}"])
        table = [
            ["Factor", "Raw R²", "ΔRaw", "Partial R²", "ΔPartial", "Drop %", "PC R²", "SHD", "ΔSHD", "Rescue ΔSHD"]
        ]
        for row in run_rows:
            values = [row["factor"]]
            for key in (
                "raw_r2",
                "delta_raw_r2",
                "partial_r2",
                "delta_partial_r2",
                "partial_drop_share",
                "readout_test_r2",
                "incident_shd",
                "delta_incident_shd",
                "rescue_shd_reduction",
            ):
                value = row.get(key)
                if value is None:
                    values.append("—")
                elif key == "partial_drop_share":
                    values.append(f"{100*value:.1f}")
                elif "shd" in key:
                    values.append(str(value))
                else:
                    values.append(f"{value:+.3f}" if key.startswith("delta_") else f"{value:.3f}")
            table.append(values)
        widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
        lines.extend(
            "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths))) for r in table
        )
    lines.extend(
        [
            "\nDrop % = share of summed decreases in partial R² (improving factors are excluded).",
            "PC R² = held-out decoding after train-only PCA/scaling at the PC readout dimension.",
            "Incident SHD counts errors touching a factor; each graph error touches TWO factors.",
            "Positive rescue ΔSHD means replacing that decoded factor with truth reduced global SHD.",
            "Rescue is an oracle sensitivity check, not proof of the cause of training collapse.",
            "Raw and partial R² both falling supports loss of linearly decodable factor information.",
            "Stable raw R² with falling partial R² suggests loss beyond linearly fitted parent effects.",
            "Stable feature-probe R² with falling PC R² suggests sensitivity to PCA/readout compression.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_factor_reports(results, output_dir, reference_run=None, alpha=0.05):
    rows, reference = build_factor_rows(results, reference_run, alpha)
    columns = [
        "run_order",
        "directory_name",
        "run_dir",
        "reference_run",
        "dim",
        "factor",
        "parents",
        "comparison_status",
        "raw_r2",
        "delta_raw_r2",
        "partial_r2",
        "delta_partial_r2",
        "mean_partial_delta_contribution",
        "partial_drop_share",
        "readout_test_r2",
        "decoded_variance_ratio",
        "decoded_gt_correlation",
        "diagnostic_alpha",
        "incident_f1",
        "incident_precision",
        "incident_recall",
        "incident_shd",
        "delta_incident_shd",
        "false_neighbors",
        "missing_neighbors",
        "rescue_f1_gain",
        "rescue_shd_reduction",
    ]
    with (output_dir / "causal_recovery_factors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = format_factor_report(rows, reference, alpha)
    (output_dir / "causal_recovery_factors.txt").write_text(report)
    return report
