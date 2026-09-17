"""Controlled FLAIR/T1 renderer audit, no checkpoint or training changes.

This uses the supplied YAML's identifiable/clean 64-cubed renderer preset with
default unit content/style scales. Other inherited run settings are unknown.
IID N(0,1) latents deliberately isolate factor recoverability from SCM shortcuts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eval.synthetic_dataset import PseudoMRIRenderer


def summarize(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return (
        {
            "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
            "n": len(a),
        }
        if len(a)
        else {"n": 0}
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interventions", type=int, default=32)
    ap.add_argument("--probe-samples", type=int, default=512)
    ap.add_argument("--out", default="results/ventricle_diagnosis/flair_signal_audit.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(20260915)
    renderer = PseudoMRIRenderer(res=64, identifiable_ventricle=True)
    zero_def, zero_fissure = torch.zeros(4, 4, 4), torch.zeros(8, 8, 8)
    modalities = ("T1", "FLAIR")
    rows = []
    with torch.no_grad():
        for i in range(args.interventions):
            content = torch.randn(9, generator=rng)
            style = torch.randn(3, generator=rng)
            for factor, dim in (("brain", 0), ("ventricle", 1), ("cortex", 5)):
                states = []
                for delta in (-0.25, 0.25):
                    changed = content.clone()
                    changed[dim] += delta
                    states.append(renderer.render_structure(changed, zero_def, zero_fissure, "cpu", clean=True))
                (a, la), (b, lb) = states
                support = (a != b).numpy()
                roi = torch.from_numpy(maximum_filter(support, size=3)) & ((a > 0) | (b > 0))
                if not roi.any():
                    continue
                for condition, zstyle in (("zero_style", torch.zeros(3)), ("matched_random_style", style)):
                    signals = []
                    row = {
                        "sample": i,
                        "factor": factor,
                        "condition": condition,
                        "changed_voxels": int(support.sum()),
                        "roi_voxels": int(roi.sum()),
                    }
                    for modality in modalities:
                        # Same style AND noise across modalities isolates the effect of their LUTs.
                        low = renderer.render_modality(a, la, zstyle, modality, 10000 + i, "cpu")[0]
                        high = renderer.render_modality(b, lb, zstyle, modality, 10000 + i, "cpu")[0]
                        repeat = renderer.render_modality(a, la, zstyle, modality, 20000 + i, "cpu")[0]
                        signal = high - low
                        noise = repeat - low
                        signals.append(signal.flatten())
                        sn = float(signal[roi].norm())
                        nn = float(noise[roi].norm())
                        row[f"{modality}_signal_norm"] = sn
                        row[f"{modality}_repeat_difference_norm"] = nn
                        # This is repeat-acquisition variability, including bias-field changes,
                        # not a calibrated SNR or an information-theoretic recoverability bound.
                        row[f"{modality}_signal_to_repeat_difference"] = sn / max(nn, 1e-12)
                    row["flair_to_t1_signal"] = row["FLAIR_signal_norm"] / max(row["T1_signal_norm"], 1e-12)
                    row["signal_cosine"] = float(F.cosine_similarity(*signals, dim=0))
                    rows.append(row)
        print("Renderer interventions complete", flush=True)
        features, targets = {}, []
        for i in range(args.probe_samples):
            content = torch.randn(9, generator=rng)
            tissue, lesion = renderer.render_structure(content, zero_def, zero_fissure, "cpu", clean=True)
            targets.append(float(content[1]))
            for modality in modalities:
                style = torch.randn(3, generator=rng)  # independent styles across views, as in training
                x = renderer.render_modality(
                    tissue, lesion, style, modality, 30000 + 2 * i + (modality == "FLAIR"), "cpu"
                )
                x = x * (tissue > 0)[None]
                central = x[:, 22:42, 22:42, 22:42]  # fixed cube; no per-subject ventricular mask
                blocks = {
                    "whole_mean": x.mean().reshape(1),
                    "central_mean": central.mean().reshape(1),
                    "central_spatial": F.adaptive_avg_pool3d(central[None], 4).flatten(),
                }
                for block, value in blocks.items():
                    features.setdefault(f"{modality}/{block}", []).append(value.numpy())
            if (i + 1) % 128 == 0:
                print(f"Rendered {i + 1}/{args.probe_samples} input-probe subjects", flush=True)
    nfit = args.probe_samples * 3 // 4
    y = np.asarray(targets)
    scores = {}
    with threadpool_limits(limits=2):
        for key, values in features.items():
            x = np.asarray(values)
            for kind, estimator, grid in (
                ("ridge", Ridge(), {"regressor__ridge__alpha": [0.01, 1, 100]}),
                (
                    "rbf",
                    KernelRidge(kernel="rbf"),
                    {
                        "regressor__kernelridge__alpha": [0.01, 1],
                        "regressor__kernelridge__gamma": [0.1 / x.shape[1], 1 / x.shape[1], 10 / x.shape[1]],
                    },
                ),
            ):
                probe = TransformedTargetRegressor(
                    regressor=make_pipeline(StandardScaler(), estimator), transformer=StandardScaler()
                )
                search = GridSearchCV(probe, grid, cv=KFold(3, shuffle=True, random_state=0), scoring="r2")
                search.fit(x[:nfit], y[:nfit])
                scores[f"{key}/{kind}"] = float(r2_score(y[nfit:], search.predict(x[nfit:])))
    summary = {}
    for condition in ("zero_style", "matched_random_style"):
        for factor in ("brain", "ventricle", "cortex"):
            selected = [r for r in rows if r["condition"] == condition and r["factor"] == factor]
            summary[f"{condition}/{factor}"] = {
                key: summarize([r[key] for r in selected])
                for key in (
                    "changed_voxels",
                    "flair_to_t1_signal",
                    "signal_cosine",
                    "T1_signal_to_repeat_difference",
                    "FLAIR_signal_to_repeat_difference",
                )
            }
    output = {
        "protocol": {
            "res": 64,
            "identifiable_ventricle": True,
            "clean": True,
            "content_scale": 1,
            "style_scale": 1,
            "latent_distribution": "IID normal",
            "seed": 20260915,
            "intervention_half_width": 0.25,
            "intervention_subjects": args.interventions,
            "probe_fit": nfit,
            "probe_test": len(y) - nfit,
            "input_probes": "raw rendered inputs with background masked; no checkpoint or fixed_reference affine",
            "limits": "Controlled preset, not a reproduction of unspecified inherited run settings."
            " Repeat-noise statistic includes bias fields. Zero style still has sigma=.01 noise."
            " Input readouts are diagnostics using fixed central coordinates; no model was trained.",
        },
        "interventions": summary,
        "input_probe_r2": scores,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"interventions": summary, "input_probe_r2": scores}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
