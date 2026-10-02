"""Summarize completed frozen audits without modifying reports or fitting new probes."""

import json
from pathlib import Path

import numpy as np

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
SOURCE = ROOT / "results/encoder_followups_local/20261002_160002/spatial"
MODELS = {
    "conv_mlp_s42": "Conv + MLP",
    "resnet_groupnorm_s42": "ResNet GroupNorm",
    "resnet_stride8_s42": "ResNet stride 8",
}


def r2(truth, prediction):
    return 1 - ((truth - prediction) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    reports = {name: json.loads((SOURCE / name / "report.json").read_text()) for name in MODELS}
    controls_present = all((SOURCE.parent / "supervised" / view / "report.json").exists() for view in ("t1", "flair"))
    first = next(iter(reports.values()))
    sulcal, lesions, initial, anatomy, derived = [], [], [], [], []
    truth_reference = None
    verified = 0
    for name, report in reports.items():
        assert report["status"] == "complete" and report["source_checkpoints_unchanged"]
        assert report["source_sha256"] == first["source_sha256"]
        assert report["probe_split"] == first["probe_split"]
        for arm in ("trained", "initial"):
            for split in ("val", "test"):
                assert report["cohorts"][arm][split]["input_sha256"] == first["cohorts"][arm][split]["input_sha256"]
        native = report["cohorts"]["trained"]["test"]["native_shape"][0]
        label = MODELS[name]
        rows = report["probes"]

        def score(view, grid, stage, target, arm="trained", condition="observed", probe="ridge"):
            return next(
                row["test_r2"]
                for row in rows
                if (row["arm"], row["view"], row["grid"], row["stage"], row["target"], row["condition"], row["probe"])
                == (arm, view, grid, stage, target, condition, probe)
            )

        for view in ("t1", "flair"):
            stages = [(native, "backbone"), (1, "backbone"), (1, "hidden"), (1, "projected")]
            sulcal.append([label, view.upper(), *[f"{score(view, g, s, 'sulcal_amplitude'):.3f}" for g, s in stages]])
            centroid = lambda g, s: np.mean([score(view, g, s, f"centroid_{axis}") for axis in "xyz"])
            latent = np.mean([score(view, native, "backbone", f"lesion_{axis}") for axis in "xyz"])
            lesions.append(
                [
                    label,
                    view.upper(),
                    f"{centroid(native, 'backbone'):.3f}",
                    f"{centroid(1, 'backbone'):.3f}",
                    f"{centroid(1, 'projected'):.3f}",
                    f"{latent:.3f}",
                ]
            )
            initial.append(
                [
                    label,
                    view.upper(),
                    *[
                        f"{score(view, native, 'backbone', 'sulcal_amplitude', arm=arm, condition=cond):.3f}"
                        for arm, cond in [("initial", "observed"), ("trained", "observed"), ("trained", "shuffled")]
                    ],
                ]
            )
        for target in ("brain_size", "ventricle_size", "cortical_thickness", "temporal_atrophy", "lr_asymmetry"):
            anatomy.append(
                [
                    label,
                    target,
                    f"{score('t1', 1, 'backbone', target):.3f}",
                    f"{score('t1', 1, 'projected', target):.3f}",
                ]
            )

        for arm in ("trained", "initial"):
            with np.load(SOURCE / name / f"{arm}_predictions.npz") as predictions:
                truth = predictions["truth"]
                if truth_reference is None:
                    truth_reference = truth.copy()
                assert np.array_equal(truth_reference, truth)
                for row in rows:
                    if row["arm"] != arm:
                        continue
                    key = f"{row['view']}_g{row['grid']}_{row['stage']}_{row['probe']}_{row['condition']}"
                    j = report["target_names"].index(row["target"])
                    assert abs(float(r2(truth[:, j], predictions[key][:, j])) - row["test_r2"]) < 1e-5
                    verified += 1
                for view in ("t1", "flair"):
                    predicted = predictions[f"{view}_g{native}_backbone_ridge_observed"]
                    error = np.linalg.norm(predicted[:, 9:12] - truth[:, 9:12], axis=1) * 31.5
                    derived.append(
                        {
                            "model": name,
                            "arm": arm,
                            "view": view,
                            "median_centroid_error_vox": float(np.median(error)),
                            "within_one_lesion_radius": float(np.mean(error <= 3.15)),
                            "direct_magnitude_r2": float(r2(truth[:, 13], predicted[:, 13])),
                            "absolute_signed_prediction_magnitude_r2": float(
                                r2(truth[:, 13], np.abs(predicted[:, 12]))
                            ),
                        }
                    )

    lines = [
        "# Local frozen encoder follow-ups: interpretation",
        "",
        "Source: `results/encoder_followups_local/20261002_160002/spatial/`. All three audits completed on MPS with PyTorch 2.6.0.",
        "",
        (
            "Both supervised controls are now available; see [the supervised analysis](supervised_analysis.md) for their results and the combined interpretation."
            if controls_present
            else "No supervised-control report was found under this timestamp at analysis time."
        ),
        "",
        f"Verified {verified:,} per-factor test R² values against saved predictions. Models and initial/trained arms share identical input hashes, target arrays, probe split, and evaluation source hashes. Checkpoints were recorded as unchanged. The protocol fits on 300 original validation subjects, tunes on 100, and tests on 400 separate subjects. Both views, saved initial weights, and shuffled-label probes are present.",
        "",
        "## Main finding",
        "",
        "Sulcal amplitude is highly recoverable from the Conv spatial representation and substantially recoverable from the ResNet backbones, but poorly recovered from the final nine-dimensional content vectors. Lesion localization remains weak: Conv FLAIR contains some usable spatial signal; trained ResNet features give near-baseline results with the tested probes. These are statements about finite-probe recoverability, not proof of identifiability or complete information loss.",
        "",
        "## Where sulcal recovery drops",
        "",
        "Held-out R² for the renderer's signed sulcal amplitude, using ridge throughout (no per-cell selection between ridge and RBF). GAP is global average pooling. Native backbone grids are 16³ / 2³ / 8³ for Conv / GroupNorm ResNet / stride-8 ResNet. Hidden is the actual global MLP hidden output; final content is the actual nine-dimensional output.",
        "",
        table(["Model", "View", "Native backbone", "GAP backbone", "MLP hidden", "Final content"], sulcal),
        "",
        "Conv loses accessibility primarily at spatial averaging: native R² ≈0.98 becomes ≈0 or negative at GAP. Stride-8 ResNet shows a similar drop (≈0.83 to ≈0.02). GroupNorm ResNet retains a strong pooled backbone signal (≈0.69–0.73) but its global MLP/content output suppresses accessibility (≈0.03). These comparisons change feature dimension and fitted probes; a lower score does not establish an irreversible mathematical loss.",
        "",
        "The GroupNorm T1 native ridge result is a failure of this fitted readout on held-out data, not evidence that its map has no sulcal signal: the matched native RBF probe reaches 0.725. One test subject (index 298) contributes 40% of native ridge amplitude squared error; its predicted amplitude is 0.559 despite truth 0.0114. The underlying amplitude range is approximately ±0.06. Keep that result visible; investigate conditioning/out-of-distribution features before claiming the native map is worse than GAP.",
        "",
        "The original raw sulcal latent is also recovered by native Conv ridge (T1 0.892; FLAIR 0.901). The higher amplitude scores partly reflect predicting the bounded rendered quantity rather than the unsquashed latent. This is not merely a magnitude/sign ambiguity.",
        "",
        "## What training added",
        "",
        table(
            ["Model", "View", "Initial native amplitude", "Trained native amplitude", "Shuffled trained native"],
            initial,
        ),
        "",
        "Initial Conv features already recover sulcal amplitude extremely well (≈0.95); training adds a modest improvement. This establishes strong observability through random spatial features, not that contrastive learning discovered a uniquely identifiable factor. Stride-8 native sulcal recovery is lower after training. Sulcal signal also appears in spatial channels labelled style, so the channel names alone do not establish separation of factors.",
        "",
        "## Lesion recovery",
        "",
        "Centroid columns average the three physical-coordinate R² values. Raw lesion columns average the original three generator-latent R² values. They are different targets because placement depends on anatomy and rasterization.",
        "",
        table(
            ["Model", "View", "Native centroid", "GAP centroid", "Final content centroid", "Native raw lesion"], lesions
        ),
        "",
        "Conv FLAIR native centroid R² is 0.300: x/y/z = 0.457/0.353/0.090. Original lesion-latent x/y/z = 0.282/0.309/0.013. Recovery is partial and weakest along z. Its native ridge median physical error is 7.36 voxels, and 10.75% of subjects are within one 3.15-voxel lesion radius. T1 is weaker (centroid mean R² 0.100, median error 9.41 voxels). These are not yet accurate lesion localizers.",
        "",
        "The trained ResNets remain near baseline even when spatial position is retained; increasing final map resolution alone did not rescue lesions under this recipe. For stride-8 FLAIR, the same spatial projected ridge probe changes from initial centroid R² 0.357 to trained −0.013 (raw lesion 0.210 to −0.015). This suggests training made lesion location much less accessible to that readout. Spatial projected features apply the MLP independently per bin; they are diagnostic representations, not the globally pooled vector used during training.",
        "",
        "Conv training effects depend on the readout: its native FLAIR centroid ridge improves from 0.022 to 0.300, but its 2³ backbone centroid ridge changes from 0.336 to 0.291. Do not claim all lesion representations improved. Centroid prediction can also exploit anatomy-dependent placement; a direct localization control remains useful.",
        "",
        "## Broad anatomy still present before the ResNet readout",
        "",
        "T1 ridge test R², using the same global backbone and final content stages:",
        "",
        table(["Model", "Factor", "GAP backbone", "Final content"], anatomy),
        "",
        "The poor final ResNet content scores do not imply that its entire backbone failed. For example, asymmetry is recovered at 0.927/0.961 before the GroupNorm/stride-8 heads but only 0.117/0.166 afterward. Conv final content retains its strong broad-anatomy scores.",
        "",
        "## Additional magnitude diagnostic (post hoc, no refitting)",
        "",
        "Each original probe predicts magnitude separately from signed amplitude. Failure of that direct magnitude readout does not prove magnitude is absent if signed amplitude can be predicted. Taking the absolute value of saved native Conv ridge amplitude predictions gives magnitude R² 0.907 (T1) and 0.901 (FLAIR), compared with 0.603/0.572 for the separately fitted magnitude ridge. These derived scores use the already fitted signed predictor, with no test-label-based tuning; treat this as an exploratory diagnostic.",
        "",
        "## Next decisions supported by these results",
        "",
        "1. For sulcal decoding, start with the already successful frozen Conv spatial features and ridge probe. A usable endpoint exists without encoder retraining. To change the representation itself, test a spatially aware content readout or explicit sulcal objective; merely widening the final global vector does not address the observed pooling failure.",
        (
            "2. The supervised controls have now completed; consult [their analysis](supervised_analysis.md). They strongly improve lesion localization on both views. To isolate the remaining bottleneck, compare supervised spatial heads on frozen versus fine-tuned original encoders with a matched labeled cohort."
            if controls_present
            else "2. Complete the planned supervised controls on both views, especially FLAIR for lesions. Their heatmap supervision and higher-resolution features test whether a target-directed learner can localize lesions from the same inputs. No outcome for this experiment can be inferred from the frozen audits alone."
        ),
        "3. If the supervised lesion control succeeds, test early/high-resolution encoder features and a lesion-aware objective before another generic stride comparison. If it fails, inspect optimization, rendering contrast and target definition before concluding non-observability.",
        "4. Confirm selected findings with additional training seeds and a fresh evaluation cohort. These are exploratory comparisons across many readouts, with only one model seed. Conditional test-set uncertainty would not replace training-seed uncertainty.",
        "",
        "Native backbone feature counts are 262,144 (Conv), 4,096 (GroupNorm ResNet), and 262,144 (stride-8 ResNet); pooled backbone counts are 64/512/512, and final content has 9 coordinates. Equal native dimensionality for Conv and stride-8 does not equalize their spatial layout, normalization, or other architecture choices. The two ResNet variants also differ in normalization, so these runs do not isolate stride as a causal variable.",
        "",
        "Reproduce this summary from repository root: `python reports/encoder_followups_local_20261002_160002/analyze.py`. Only report JSON and saved prediction NPZ files are read; original artifacts and encoder weights are never modified.",
    ]
    (DEST / "analysis.md").write_text("\n".join(lines) + "\n")
    (DEST / "derived_metrics.json").write_text(
        json.dumps({"verified_test_scores": verified, "native_ridge_diagnostics": derived}, indent=2) + "\n"
    )
    print(f"Verified {verified} scores; wrote {DEST / 'analysis.md'}")


if __name__ == "__main__":
    main()
