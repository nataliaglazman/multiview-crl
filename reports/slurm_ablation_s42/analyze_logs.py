"""Extract the three supplied SLURM stdout logs and plot their printed metrics.

Run from any directory with a Python environment containing matplotlib/numpy.
Values retain the precision printed in stdout; no checkpoints or jobs are changed.
"""

import csv
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/encoder-ablation-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
FILES = {
    "Conv + MLP": "encoder-ablation-bio-conv-mlp-s42-37697225.out",
    "ResNet + GroupNorm": "encoder-ablation-bio-resnet-groupnorm-s42-37697226.out",
    "ResNet stride 8": "encoder-ablation-bio-resnet-stride8-s42-37697227.out",
}
COLORS = ["#087e8b", "#b95f16", "#7658a5"]
FACTOR_COLUMNS = ["ridge_r2", "block_mcc", "block_mcc_cv_sd", "channel_mcc", "initial_ridge_r2", "printed_delta"]


def parse(path):
    text = path.read_text()
    config = dict(re.findall(r"^\t(\w+): (.*)$", text, re.M))
    history = [
        {"step": int(step), "loss": float(loss), "rank": float(rank)}
        for step, loss, rank in re.findall(
            r"^step\s+(\d+) \| contrastive ([\d.]+) \| content eff_rank ([\d.]+)/9", text, re.M
        )
    ]
    evaluations = []
    parts = re.split(r"\[eval\] synthetic DCI @ step (\d+) \.\.\.", text)
    for i in range(1, len(parts), 2):
        step, chunk = int(parts[i]), parts[i + 1]
        metrics = {k: float(v) for k, v in re.findall(r"^\s+(content->\S+)\s+(-?\d+\.\d+)\s*$", chunk, re.M)}
        blocks = {}
        for block in ("content", "style"):
            section = chunk.split(f"--- per-factor recovery: content→{block} ---")[1].split("---")[0]
            rows = {}
            for line in section.splitlines():
                tokens = line.split()
                if len(tokens) not in (5, 7) or not re.fullmatch(r"\w+", tokens[0]):
                    continue
                try:
                    numbers = [float(x) for x in tokens[1:]]
                except ValueError:
                    continue
                rows[tokens[0]] = dict(zip(FACTOR_COLUMNS, numbers))
            assert len(rows) == (9 if block == "content" else 3), (path, step, block)
            blocks[block] = rows
        evaluations.append({"step": step, "metrics": metrics, "factors": blocks})
    assert [r["step"] for r in history] == list(range(100, 10001, 100))
    assert [r["step"] for r in evaluations] == list(range(0, 10001, 2000))
    assert "done. checkpoints + DCI logs in " in text
    for row in evaluations:
        values = [v["ridge_r2"] for v in row["factors"]["content"].values()]
        assert abs(np.mean(values) - row["metrics"]["content->content/informativeness_ridge"]) < 0.0011
        if row["step"]:
            for block, factors in row["factors"].items():
                for name, values in factors.items():
                    assert values["initial_ridge_r2"] == evaluations[0]["factors"][block][name]["ridge_r2"]
                    # Subtraction of values rounded to three places can differ from
                    # the independently rounded delta by 0.001.
                    assert abs(values["ridge_r2"] - values["initial_ridge_r2"] - values["printed_delta"]) < 0.0011
    return {
        "source": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime": text.splitlines()[:3],
        "config": config,
        "completion": text.splitlines()[-1],
        "training": history,
        "evaluations": evaluations,
    }


def main():
    runs = {label: parse(ROOT / filename) for label, filename in FILES.items()}
    keys = set.union(*(set(run["config"]) for run in runs.values()))
    differences = {
        key: {label: run["config"].get(key) for label, run in runs.items()}
        for key in sorted(keys)
        if len({run["config"].get(key) for run in runs.values()}) > 1
    }
    assert set(differences) == {
        "model_id",
        "encoder_architecture",
        "conv_readout",
        "resnet_norm",
        "resnet_output_stride",
    }
    (DEST / "parsed_logs.json").write_text(
        json.dumps({"config_differences": differences, "runs": runs}, indent=2) + "\n"
    )
    with (DEST / "factor_recovery.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "step", "block", "factor", *FACTOR_COLUMNS])
        writer.writeheader()
        for label, run in runs.items():
            for row in run["evaluations"]:
                for block, factors in row["factors"].items():
                    for factor, values in factors.items():
                        writer.writerow(dict(experiment=label, step=row["step"], block=block, factor=factor, **values))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for (label, run), color in zip(runs.items(), COLORS):
        x = [r["step"] for r in run["training"]]
        axes[0].plot(x, [r["loss"] for r in run["training"]], color=color, label=label)
        axes[1].plot(x, [r["rank"] for r in run["training"]], color=color)
        rows = run["evaluations"]
        axes[2].plot(
            [r["step"] for r in rows],
            [r["metrics"]["content->content/informativeness_ridge"] for r in rows],
            "o-",
            color=color,
        )
    axes[0].set(title="Training loss falls in every run", ylabel="InfoNCE loss (log scale)", yscale="log")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    axes[1].set(title="Representations use multiple dimensions", ylabel="Content effective rank", ylim=(0, 9.5))
    axes[1].axhline(9, color="gray", linestyle=":", linewidth=1)
    axes[2].set(title="Measured factor recovery diverges", ylabel="Mean validation ridge R²", ylim=(-0.01, 0.51))
    for ax in axes:
        ax.set(xlabel="Optimizer step", xlim=(0, 10000), xticks=[0, 2000, 4000, 6000, 8000, 10000])
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(alpha=0.18)
    fig.suptitle("SLURM ablations · seed 42 · 400 validation subjects", fontsize=15)
    fig.savefig(DEST / "training_and_recovery.png", dpi=170)
    plt.close(fig)
    factors = list(next(iter(runs.values()))["evaluations"][-1]["factors"]["content"])
    matrix = np.array(
        [
            [run["evaluations"][-1]["factors"]["content"][factor]["ridge_r2"] for run in runs.values()]
            for factor in factors
        ]
    )
    fig, ax = plt.subplots(figsize=(8.7, 6.2), layout="constrained")
    plot = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    ax.set(
        xticks=range(3),
        xticklabels=list(runs),
        yticks=range(9),
        yticklabels=[s.replace("_", " ") for s in factors],
        title="Final factor recovery · primary-view content vector · step 10,000",
    )
    ax.tick_params(axis="x", labelsize=10)
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(
            column, row, f"{value:.3f}", ha="center", va="center", color="white" if abs(value) > 0.65 else "#222222"
        )
    fig.colorbar(plot, ax=ax, label="Validation ridge R²", shrink=0.8)
    fig.savefig(DEST / "final_factor_recovery.png", dpi=170)
    plt.close(fig)
    for label, run in runs.items():
        first, last = run["evaluations"][0], run["evaluations"][-1]
        key = "content->content/informativeness_ridge"
        print(
            f"{label}: mean ridge {first['metrics'][key]:.3f} -> {last['metrics'][key]:.3f}; {len(run['training'])} training records; {len(run['evaluations'])} evaluations"
        )
    print(f"Saved verified extraction and plots to {DEST}")


if __name__ == "__main__":
    main()
