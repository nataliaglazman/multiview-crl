"""Verify and interpret the supervised controls; never refit or change source results."""

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/encoder-followups-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/encoder-followups-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze import DEST, SOURCE, r2, table


def main():
    source = SOURCE.parent / "supervised"
    frozen_report = json.loads((SOURCE / "conv_mlp_s42/report.json").read_text())
    reports, predictions, errors, frozen_errors, derived = {}, {}, {}, {}, {}
    factor_rows, metric_rows, history_rows = [], [], []
    with np.load(SOURCE / "conv_mlp_s42/trained_predictions.npz") as data:
        frozen = {key: data[key].copy() for key in data.files}
    truth = frozen["truth"]
    for view in ("t1", "flair"):
        directory = source / view
        report = json.loads((directory / "report.json").read_text())
        reports[view] = report
        assert report["status"] == "complete"
        assert report["completed_steps"] == report["arguments"]["steps"] == 2000
        assert report["selection"] == "final_step"
        assert not report["arguments"]["shuffle_targets"]
        assert report["source_sha256"] == frozen_report["source_sha256"]
        for split in ("val", "test"):
            assert (
                report["cohorts"][split]["input_sha256"] == frozen_report["cohorts"]["trained"][split]["input_sha256"]
            )
        with np.load(directory / "test_predictions.npz") as data:
            predictions[view] = {key: data[key].copy() for key in data.files}
        saved = predictions[view]
        assert np.array_equal(saved["truth"], truth)
        training_targets = np.load(directory / "data/train/targets.npy", mmap_mode="r")
        predicted = {
            **{name: saved["regression"][:, j] for j, name in enumerate(report["regression_targets"])},
            **{f"centroid_{axis}": saved["centroid"][:, j] for j, axis in enumerate("xyz")},
            "sulcal_magnitude": np.abs(saved["regression"][:, -1]),
        }
        for row in report["test"]["factors"]:
            j = report["target_names"].index(row["target"])
            y, p = truth[:, j], predicted[row["target"]]
            assert abs(float(r2(y, p)) - row["r2"]) < 1e-6
            assert abs(float(np.sqrt(np.mean((y - p) ** 2))) - row["rmse"]) < 1e-6
            baseline = np.full_like(y, training_targets[:, j].mean())
            assert abs(float(r2(y, baseline)) - row["train_mean_baseline_r2"]) < 1e-6
        errors[view] = np.linalg.norm(saved["centroid"] - truth[:, 9:12], axis=1) * 31.5
        frozen_errors[view] = (
            np.linalg.norm(frozen[f"{view}_g16_backbone_ridge_observed"][:, 9:12] - truth[:, 9:12], axis=1) * 31.5
        )
        err = errors[view]
        assert abs(float(np.median(err)) - report["test"]["median_centroid_error_vox"]) < 1e-6
        assert abs(float(np.mean(err)) - report["test"]["mean_centroid_error_vox"]) < 1e-6
        assert float(np.mean(err <= 3.15)) == report["test"]["within_one_lesion_radius"]
        sign = float(np.mean(np.sign(saved["regression"][:, -1]) == np.sign(truth[:, 12])))
        assert sign == report["test"]["sulcal_sign_accuracy"]
        derived[view] = {
            "centroid_mean_r2": float(np.mean(r2(truth[:, 9:12], saved["centroid"]))),
            "raw_lesion_mean_r2": float(np.mean(r2(truth[:, 2:5], saved["regression"][:, :3]))),
            "error_percentiles_vox": {str(q): float(np.percentile(err, q)) for q in (50, 75, 90, 95, 99, 100)},
            "within_threshold": {str(t): float(np.mean(err <= t)) for t in (0.5, 1, 2, 3.15)},
            "outside_one_radius_count": int(np.sum(err > 3.15)),
            "largest_error_subjects": [
                {"index": int(i), "error_vox": float(err[i])} for i in np.argsort(err)[-5:][::-1]
            ],
        }
        for stage in report["history"]:
            validation = stage["validation"]
            amplitude = next(row["r2"] for row in validation["factors"] if row["target"] == "sulcal_amplitude")
            history_rows.append(
                [
                    view.upper(),
                    stage["step"],
                    f"{validation['median_centroid_error_vox']:.3f}",
                    f"{100 * validation['within_one_lesion_radius']:.2f}%",
                    f"{amplitude:.3f}",
                ]
            )

    assert reports["t1"]["cohorts"] == reports["flair"]["cohorts"]
    for a, b in zip(reports["t1"]["test"]["factors"], reports["flair"]["test"]["factors"]):
        assert a["target"] == b["target"]
        factor_rows.append([a["target"], f"{a['r2']:.4f}", f"{b['r2']:.4f}"])
    for view in ("t1", "flair"):
        d = derived[view]
        for kind, err, value in [
            (
                "Frozen Conv native ridge",
                frozen_errors[view],
                float(np.mean(r2(truth[:, 9:12], frozen[f"{view}_g16_backbone_ridge_observed"][:, 9:12]))),
            ),
            ("Supervised heatmap", errors[view], d["centroid_mean_r2"]),
        ]:
            metric_rows.append(
                [
                    view.upper(),
                    kind,
                    f"{value:.4f}",
                    f"{np.median(err):.3f}",
                    f"{np.mean(err):.3f}",
                    f"{100 * np.mean(err <= 3.15):.2f}%",
                ]
            )

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors = {"t1": "#a64b21", "flair": "#007e87"}
    for view in ("t1", "flair"):
        for err, style, name in [
            (errors[view], "-", "supervised heatmap"),
            (frozen_errors[view], "--", "frozen Conv ridge"),
        ]:
            axes[0].semilogx(
                np.sort(err),
                np.arange(1, len(err) + 1) / len(err),
                style,
                color=colors[view],
                linewidth=2,
                label=f"{view.upper()} — {name}",
            )
        axes[1].scatter(
            truth[:, 12],
            predictions[view]["regression"][:, -1],
            s=12,
            alpha=0.45,
            color=colors[view],
            label=view.upper(),
        )
    axes[0].axvline(3.15, color="0.4", linewidth=1, linestyle=":")
    axes[0].text(2.65, 0.50, "One lesion radius: 3.15 voxels", color="0.3", fontsize=10, rotation=90, va="center")
    axes[0].set(
        xlabel="3D centroid error (voxels; log scale)",
        ylabel="Fraction of test subjects at or below error",
        xlim=(0.025, 35),
        ylim=(0, 1.025),
        title="Lesion localization: all 400 test subjects",
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=2, fontsize=9, frameon=False)
    limit = 1.08 * max(
        float(np.abs(truth[:, 12]).max()), *(float(np.abs(p["regression"][:, -1]).max()) for p in predictions.values())
    )
    axes[1].plot([-limit, limit], [-limit, limit], color="0.4", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="True signed sulcal amplitude",
        ylabel="Predicted signed sulcal amplitude",
        title="Supervised sulcal recovery",
        xlim=(-limit, limit),
        ylim=(-limit, limit),
    )
    axes[1].legend()
    fig.savefig(DEST / "supervised_controls.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Supervised heatmap controls: both views completed",
        "",
        "Both targets are learnable from these synthetic images under explicit supervision. Lesion localization is excellent on FLAIR and strong on most T1 subjects, with a failure tail on T1. Sulcal amplitude is recovered very well in both views. This substantially narrows the interpretation of the earlier frozen-probe failures.",
        "",
        "![Held-out localization and sulcal recovery](supervised_controls.png)",
        "",
        "## Protocol and verification",
        "",
        "Source: `results/encoder_followups_local/20261002_160002/supervised/{t1,flair}`. Both reports are complete: 2,000 updates, batch 8, MPS/PyTorch 2.6.0, seed 42, a fresh 310,634-parameter network for each view, final-step checkpoint selection. Training uses 2,000 labeled subjects; validation and independent test each have 400 subjects. The inputs are full single-view images; lesion masks are training targets, not model inputs. Coordinates come from the expected position of a predicted spatial softmax heatmap; the other targets come from a separate spatial regression head.",
        "",
        "All saved test R², RMSE, training-mean baseline R², centroid error metrics, and sulcal sign accuracies were independently recomputed from predictions. Test targets are identical to the frozen audits, validation/test input hashes match, and evaluation source hashes agree. No models were retrained and original outputs were not modified during this analysis.",
        "",
        "## Lesion localization compared with the frozen Conv probe",
        "",
        table(
            ["View", "Method", "Mean centroid R²", "Median error (vox)", "Mean error (vox)", "Within 3.15 vox"],
            metric_rows,
        ),
        "",
        "FLAIR has median error 0.217 voxels; 398/400 subjects are within one voxel and all 400 are within one lesion radius. The maximum error is 2.378 voxels. T1 has median error 0.284 voxels and 361/400 within one voxel, but 25/400 fall outside one lesion radius. Its 95th-percentile error is 5.558 voxels and maximum is 18.416 voxels. Report this tail rather than relying on the median alone. These are physical-centroid scores, not Dice/IoU segmentation scores; this audit has not measured predicted mask overlap or heatmap calibration.",
        "",
        "## Original latent factors and physical targets",
        "",
        table(["Target", "T1 test R²", "FLAIR test R²"], factor_rows),
        "",
        "Physical localization and recovery of original lesion generator coordinates are different endpoints. The original lesion z latent remains harder (T1 0.598; FLAIR 0.686), even though physical z centroid R² is 0.957/0.999. This is consistent with anatomy-dependent, discretized placement and/or a harder regression task. These results do not establish an irreducible ceiling on latent recovery.",
        "",
        "Signed sulcal amplitude R² is 0.978 (T1) and 0.983 (FLAIR); original sulcal latent R² is 0.915/0.938. Sulcal sign accuracy is 97.75%/97.00%. Magnitude, obtained by taking the absolute predicted signed amplitude, has R² 0.905/0.923. Thus success includes both signed structure and magnitude, rather than magnitude alone.",
        "",
        "## Learning trajectory",
        "",
        table(
            ["View", "Step", "Validation median error (vox)", "Within one radius", "Signed amplitude R²"], history_rows
        ),
        "",
        "The validation trajectories generally improve through the fixed final step, with test performance broadly consistent with final validation. T1's final logged minibatch loss is higher than the preceding snapshot, but its validation localization and raw sulcal latent score improve; one minibatch loss is not evidence that training diverged. The checkpoints were selected by prespecified final step, not by test performance.",
        "",
        "## What the combined experiments establish",
        "",
        "1. A learner can infer physical lesion location and sulcal factors from these images on held-out subjects. The earlier low probe scores cannot be explained simply by the targets being invisible in this synthetic dataset.",
        "2. Sulcal decoding already works well from frozen Conv spatial features. Global averaging and the learned global content readout are important practical bottlenecks in the audited encoders.",
        "3. Lesion localization becomes much stronger with a high-resolution, target-supervised network. The contrastive representation/probe pipeline is limiting practical recovery, but this comparison does not isolate which component is responsible.",
        "4. This is supervised recoverability evidence, not a demonstration that the contrastive encoders identify the factors, disentangle their coordinates, or recover a causal model. Strong target decoding under supervision does not establish uniqueness of an unsupervised latent representation.",
        "",
        "## Comparison limits and next experiment",
        "",
        "The controls change architecture, spatial resolution, loss, probe class, and label budget together: 2,000 supervised training subjects versus 300 labeled fitting subjects for the frozen ridge/RBF probes. They establish observability under this learning protocol, not an architecture ranking or proof that the frozen features contain no lesion information. No shuffled-label supervised training control was run. Results use one training seed and one synthetic test cohort; they do not establish performance on real MRI.",
        "",
        "The most informative next experiment is to use a spatial localization head on each original encoder with a matched labeled training cohort and comparable decoder capacity: first freeze the encoder, then allow fine-tuning. A successful frozen head would implicate probe/readout limitations; a gain only after fine-tuning would show that adapting the representation helps under that decoder and training budget. For a practical supervised model, the successful high-resolution heatmap branch is the strongest starting point for lesions, with a spatial sulcal head alongside it. If retaining an unsupervised objective is essential, use these controls as benchmarks and test a spatial learning objective separately. Inspect the 25 T1 localization failures and confirm chosen changes on another seed and fresh test cohort.",
        "",
        "Reproduce: `python reports/encoder_followups_local_20261002_160002/analyze_controls.py`.",
    ]
    (DEST / "supervised_analysis.md").write_text("\n".join(lines) + "\n")
    (DEST / "supervised_derived_metrics.json").write_text(json.dumps(derived, indent=2) + "\n")
    print(f"Verified both supervised controls; wrote {DEST / 'supervised_analysis.md'}")


if __name__ == "__main__":
    main()
