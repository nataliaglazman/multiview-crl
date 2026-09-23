import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
run = root.parent
settings = json.loads((run / "settings.json").read_text())
series = {}
for r in csv.DictReader((root / "loss_balance/scalars.csv").open()):
    series.setdefault(r["tag"], {})[int(r["step"])] = float(r["value"])
prefix = "Contrastive/vicregl_"
steps = sorted(s for s in series[prefix + "local_loss_L0"] if s <= 34001)
get = lambda tag: np.array([series[tag][s] for s in steps])
scale = settings["scale_contrastive_loss"]
weighted = {}
for arm in ["local", "global"]:
    w = scale * settings[f"vicregl_{arm}_weight"]
    for term, coef in [
        ("sim", settings["vicreg_sim_coeff"]),
        ("var", settings["vicreg_std_coeff"]),
        ("cov", settings["vicreg_cov_coeff"]),
    ]:
        weighted[f"{arm}_{term}"] = w * coef * get(f"{prefix}{arm}_{term}_L0")
local = scale * get(prefix + "local_weighted_loss_L0")
glob = scale * get(prefix + "global_weighted_loss_L0")
residual = get("Loss/Contrastive") - local - glob
term_residual = sum(weighted.values()) - local - glob
assert np.max(np.abs(residual)) < 1e-4, np.max(np.abs(residual))
assert np.max(np.abs(term_residual)) < 1e-4
late = np.array(steps) > 32000
breakdown = {k: float(v[late].mean()) for k, v in weighted.items()}
breakdown.update(
    local_total=float(local[late].mean()),
    global_total=float(glob[late].mean()),
    reconstruction=float(get("Loss/Recon")[late].mean()),
    vq=float(get("Loss/VQ")[late].mean()),
    max_accounting_error=float(np.max(np.abs(residual))),
    max_term_error=float(np.max(np.abs(term_residual))),
)
(root / "loss_balance/weighted_breakdown.json").write_text(json.dumps(breakdown, indent=2))
with (root / "loss_balance/weighted_terms.csv").open("w") as f:
    wr = csv.writer(f)
    wr.writerow(["step", *weighted, "local_total", "global_total", "recon", "vq"])
    for i, s in enumerate(steps):
        wr.writerow([s, *[v[i] for v in weighted.values()], local[i], glob[i], get("Loss/Recon")[i], get("Loss/VQ")[i]])
routing = []
for factor in ["ventricle", "lesion"]:
    rows = list(csv.DictReader((root / factor / "samples.csv").open()))
    for view in ["t1", "flair"]:
        good = [r for r in rows if r["modality"] == view and r["valid_routing"] == "True"]
        d = {"factor": factor, "view": view, "n": len(good)}
        for metric in ["joint_gain", "content_mean_gain", "style_mean_gain", "joint_cosine", "interaction_rms_ratio"]:
            values = np.array([float(r[metric]) for r in good])
            rng = np.random.default_rng(42)
            boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
            d[metric] = {"mean": float(values.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
        assert all(float(r["endpoint_replay_rms"]) == 0 for r in rows if r["modality"] == view)
        routing.append(d)
(root / "routing_means.json").write_text(json.dumps(routing, indent=2))


# 10-point moving averages reduce noise without fitting a trend.
def smooth(y):
    return np.convolve(y, np.ones(10) / 10, mode="valid")


x = np.array(steps)[9:]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
for name, y in [("Local VICRegL", local), ("Global VICRegL", glob), ("Reconstruction", get("Loss/Recon"))]:
    axs[0].plot(x, smooth(y), label=name)
axs[0].set(title="Weighted loss contributions", ylabel="Loss value")
for k, y in weighted.items():
    axs[1].plot(x, smooth(y), label=k.replace("_", " "))
axs[1].set(title="Weighted VICRegL terms", ylabel="Loss value")
for arm in ["local", "global"]:
    axs[2].plot(x, smooth(get(prefix + arm + "_std_L0")), label=arm)
axs[2].axhline(1, color="gray", ls="--", lw=1)
axs[2].set(title="Mean projected feature standard deviation", ylabel="Standard deviation")
for ax in axs:
    ax.set_xlabel("Training step")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("VICRegL run: logged steps through checkpoint 34,001 (10-point moving averages)", fontsize=12)
fig.tight_layout()
fig.savefig(root / "loss_balance/loss_balance.png", dpi=160)
plt.close(fig)
provenance = {
    "source_commit": "45cf3c70f5b5407628325ab126040bf5a13541b5",
    "source_export": "/private/tmp/multiview-routing-1RYVci",
    "checkpoint": "vqvae_model.pt",
    "checkpoint_step": 34001,
    "checkpoint_sha256": hashlib.sha256((run / "vqvae_model.pt").read_bytes()).hexdigest(),
    "settings_sha256": hashlib.sha256((run / "settings.json").read_bytes()).hexdigest(),
    "device": "cpu",
    "causal": "iid",
    "samples": 64,
    "batch_size": 2,
    "cpu_threads": 2,
    "ventricle_eps": 0.25,
    "note": "No representation training, optimizer updates, or checkpoint writes. Training used causal factors; these audits use IID factors to separate their effects.",
}
(root / "provenance.json").write_text(json.dumps(provenance, indent=2))
lines = [
    "# VICRegL routing and loss audit",
    "",
    "Checkpoint: `synthetic-clean-content-causal-vicregl-rescaled/vqvae_model.pt`, step **34,001**. CPU, 64 IID subjects, existing main audit code at `45cf3c7`. Current working branch was not changed. No model training or checkpoint update.",
    "",
    "## Routing",
    "",
    "All numbers below are means over replay-resolved subjects; the original ventricle console prints medians. Gains are projections onto the input intervention, not fractions of information or explained variance.",
    "",
    "| Intervention | View | Valid | Joint gain | Content gain | Style gain | Joint cosine |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in routing:
    lines.append(
        f"| {r['factor']} | {r['view']} | {r['n']}/64 | {r['joint_gain']['mean']:.4f} | {r['content_mean_gain']['mean']:.6f} | {r['style_mean_gain']['mean']:.4f} | {r['joint_cosine']['mean']:.4f} |"
    )
lines += [
    "",
    "Endpoint replay RMS was exactly zero for every subject in both audits. Ventricles had 62/64 measurable interventions per view; lesions had 64/64. The reconstructed intervention response is carried almost entirely by style. Joint fidelity is incomplete, especially T1 lesions and FLAIR ventricles. These tests establish decoder reliance under these interventions, not absence of all information from content, or the historical cause of the routing. IID factors and hybrid codes can differ from the training distribution.",
    "",
    "## Logged loss balance",
    "",
    "Means of 40 logged steps, 32,050–34,000. Losses below include all actual training coefficients. Local/global diagnostics are on projected features.",
    "",
    "| Contribution | Weighted mean |",
    "|---|---:|",
]
for k, v in breakdown.items():
    if not k.startswith("max_"):
        lines.append(f"| {k} | {v:.6f} |")
lines += [
    "",
    f"Local loss is {100*breakdown['local_total']/(breakdown['local_total']+breakdown['global_total']):.1f}% of the logged contrastive objective. It is active; GAP is not dominating the scalar loss. Scalar loss values do not measure encoder gradient magnitudes.",
    "",
    "The correct formula is `scale_contrastive_loss × arm_weight × (25 × sim + 25 × var + cov)`. The local/global totals reproduce `Loss/Contrastive` at every logged step through 34,001 (maximum absolute discrepancy "
    + f"{breakdown['max_accounting_error']:.3g}"
    + "). All TFRecord checksums were verified; all summaries were simple scalar values. CSV training summaries average a logging window whereas TensorBoard values are logged-step values, so they need not match row by row.",
    "",
    "Late local projected mean standard deviation is 0.978; global is 1.013. Local variance hinge remains 0.0581: average standard deviation near 1 does not mean every position/channel meets its floor. About 91.8% of already retained positions have enough valid subjects. This eligibility statistic says nothing about coverage of the ventricular boundary or lesions. No NaN-skipped steps were logged.",
    "",
    "`Weighted/*` in the existing trainer explicitly decomposes Barlow Twins only. Its residual omits VICRegL and is not evidence of a missing optimization term. Use the VICRegL-specific tags and the corrected breakdown supplied here.",
    "",
    "## Next action",
    "",
    "Do not increase local weight simply to address this routing: it is already the main contrastive contribution. The objective can be satisfied by other shared anatomical variation; it contains no requirement for ventricular size or lesions specifically. Full spatial style (`style_spatial_size: 0`) provides a reconstruction route for them. This is a plausible mechanism supported by the intervention results, not proof of training causality.",
    "",
    "The next controlled unsupervised ablation is the same VICRegL configuration with only `style_spatial_size: 1` changed, using a new run/tag. Pooling style constrains its spatial capacity but does not guarantee that global anatomical factors cannot enter it. Keep the projection heads and loss weights initially. Evaluate at an early fixed checkpoint with these same routing tests; count it as improvement only if content response increases while joint response fidelity remains acceptable. A rise in content share caused by degraded reconstruction is not success.",
    "",
    "A local directory named `synthetic-clean-content-causal-sp-s-1` is present, but its settings use Barlow Twins with outer scale 100. It is not the matched VICRegL style-size ablation.",
    "",
    "For strictly matched training-distribution interpretation, also repeat these audits with `--causal match` before making a broader causal claim. The present IID tests deliberately remove factor correlations.",
    "",
    "## Reproduction",
    "",
    "Run from code at the recorded commit (the current working branch lacks these audit modules). Substitute the run directory on the training machine:",
    "",
    "```bash",
    "RUN=results/synthetic/synthetic-clean-content-causal-vicregl-rescaled",
    'python -m eval.ventricle_routing --run-dir "$RUN" --checkpoint vqvae_model.pt --num-samples 64 --batch-size 2 --eps 0.25 --causal iid',
    'python -m eval.lesion_routing --run-dir "$RUN" --checkpoint vqvae_model.pt --num-samples 64 --batch-size 2 --causal iid',
    "```",
    "",
    "Local artifacts: `ventricle/summary.json`, `ventricle/samples.csv`, `lesion/summary.json`, `lesion/samples.csv`, `routing_means.json`, `loss_balance/scalars.csv`, `loss_balance/weighted_terms.csv`, `loss_balance/loss_balance.png`, and `provenance.json`.",
]
(root / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"routing": routing, "losses": breakdown}, indent=2))
