"""Matched, label-free training pilot on the pre-rendered clean burden bank.

Run from the repository root with the research Python runtime. Bank targets are
used only for held-out ridge probes. Cached images are float16, loaded as float32.
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from torch.utils.data import Dataset

from eval.encoder.residual_head_recovery import _out_of_fold_r2
from scripts import launch
from training import main_conv_synthetic as T

NAMES = [
    "brain_size",
    "ventricle_size",
    "lesion_burden",
    "unused_3",
    "unused_4",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
    "rendered_burden_voxels",
]


class Bank(Dataset):
    def __init__(self, root, split):
        self.views = [np.load(root / f"{split}_{key}.npy", mmap_mode="r") for key in ("t1_00", "flair")]
        self.targets = np.column_stack((np.load(root / f"{split}_zc.npy"), np.load(root / f"{split}_vox.npy")))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {"image": tuple(torch.from_numpy(np.array(view[index], dtype=np.float32))[None] for view in self.views)}

    def images(self, rows, device):
        return torch.from_numpy(
            np.concatenate([np.asarray(view[rows], dtype=np.float32)[:, None] for view in self.views])
        ).to(device)


def rank(rows):
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / len(rows)
    return float(np.trace(covariance) ** 2 / max(np.square(covariance).sum(), np.finfo(np.float32).tiny))


def evaluate(model, bank, device, batch):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    views = [[], []]
    with torch.no_grad():
        for start in range(0, len(bank), batch):
            images = bank.images(np.arange(start, min(start + batch, len(bank))), device)
            pooled = model(images, pool_only=True, n_views=2)[2][0]
            if model.residual_heads is None:
                codes = pooled[:, : model.content_channels]
            else:
                codes = torch.cat(model.split_residual_codes(pooled), 1)
            for rows, part in zip(views, codes.chunk(2)):
                rows.append(part.cpu().numpy())
    codes = [np.concatenate(rows) for rows in views]
    result = {"views": {}, "moments": {}}
    for view, name in enumerate(("t1", "flair")):
        rows = codes[view]
        blocks = {"image": rows[:, : model.content_channels]}
        if model.residual_heads is not None:
            blocks.update(residual=rows[:, model.content_channels :], combined=rows)
        result["views"][name] = {key: _out_of_fold_r2(values, bank.targets, NAMES, 5) for key, values in blocks.items()}
        result["moments"][name] = {
            key: {"rank": rank(values), "variance": float(values.var(0).mean()), "min_sd": float(values.std(0).min())}
            for key, values in blocks.items()
        }
    if model.residual_heads is not None:
        joined = torch.from_numpy(np.concatenate(codes)).to(device)
        d = model.residual_heads.diagnostics(joined[:, : model.content_channels], joined[:, model.content_channels :])
        result["population_gauge_diagnostics"] = {key: float(value) for key, value in d.items()}
    for module, flag in modes:
        module.training = flag
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=Path("out/burden_feasibility_20261102/bank"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train, val = Bank(options.bank, "train"), Bank(options.bank, "val")
    config = launch.resolve_config(
        Path("experiments/encoder/conv_mlp_burden_residual_heads_auxiliary.yaml"), "local", {}
    )
    config.update(batch_size=options.batch, device=options.device, train_steps=options.steps)
    args = T.parse_args(launch.encoder_cli_args(config))
    arms = {}
    for name in ("image_only", "corrected_auxiliary"):
        arm_config = copy.deepcopy(config)
        if name == "image_only":
            arm_config = {key: value for key, value in arm_config.items() if not key.startswith("residual_head_")}
        arm_args = T.parse_args(launch.encoder_cli_args(arm_config))
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = T.build_encoder(arm_args).to(options.device)
        directory = options.out / name
        directory.mkdir()
        (directory / "settings.json").write_text(json.dumps(vars(arm_args), indent=2))
        print(f"initializing {name}", flush=True)
        fitted = T.fit_residual_heads(model, train, arm_args, options.device)
        if fitted:
            (directory / "residual_heads_fit.json").write_text(json.dumps(fitted, indent=2))
        torch.save(model.state_dict(), directory / "model_init.pt")
        arms[name] = (model, arm_args, torch.optim.AdamW(model.parameters(), lr=arm_args.lr))
    # The appended module must leave the initial image function identical.
    for (key, parameter), (other_key, other) in zip(
        arms["image_only"][0].named_parameters(), arms["corrected_auxiliary"][0].named_parameters()
    ):
        assert key == other_key
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)
    (options.out / "pilot_settings.json").write_text(
        json.dumps(
            {
                "bank": str(options.bank.resolve()),
                "train_subjects": len(train),
                "val_subjects": len(val),
                "steps": options.steps,
                "batch": options.batch,
                "device": options.device,
                "seed": 42,
                "sampling": "same uniform subjects without replacement within each batch",
                "cached_input_dtype": str(train.views[0].dtype),
                "training_labels": False,
                "probe": "5-fold held-out StandardScaler + RidgeCV; repeated validation monitoring",
            },
            indent=2,
        )
    )
    sim, criterion = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    batches = np.random.default_rng(42)
    evaluations = {}
    start = time.monotonic()
    with (options.out / "steps.jsonl").open("w") as log:
        for step in range(options.steps + 1):
            if step == 0 or step % 250 == 0 or step == options.steps:
                evaluations[str(step)] = {}
                for name, (model, arm_args, optimizer) in arms.items():
                    report = evaluate(model, val, options.device, options.batch)
                    evaluations[str(step)][name] = report
                    torch.save(model.state_dict(), options.out / name / f"model_step{step}.pt")
                    summary = {view: report["views"][view]["image"] for view in ("t1", "flair")}
                    print(f"eval {step} {name}: {json.dumps(summary)}", flush=True)
                (options.out / "evaluations.json").write_text(json.dumps(evaluations, indent=2, allow_nan=False))
            if step == options.steps:
                break
            indices = batches.choice(len(train), options.batch, replace=False)
            images = train.images(indices, options.device)
            for name, (model, arm_args, optimizer) in arms.items():
                model.train()
                optimizer.zero_grad(set_to_none=True)
                _, loss, terms = T.training_objective(model, images, arm_args, sim, criterion)
                row = {
                    "step": step + 1,
                    "arm": name,
                    "total": float(loss.detach()),
                    **{key: float(value.detach()) for key, value in terms.items()},
                }
                if model.residual_heads is not None and (step + 1) % 100 == 0:
                    row["gradient_rms"] = T.residual_head_gradient_report(model, terms, arm_args)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss at step {step + 1} {name}")
                loss.backward()
                T.clip_encoder_gradients(model, arm_args.grad_clip)
                optimizer.step()
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if (step + 1) % 50 == 0:
                    log.flush()
                    print(
                        f"step {step + 1} {name} total={row['total']:.4f} image={row['global']:.4f} "
                        f"aux={row.get('residual_heads_auxiliary_weighted', 0):.4f} "
                        f"elapsed={time.monotonic()-start:.1f}s",
                        flush=True,
                    )
    for name, (model, _, _) in arms.items():
        torch.save(model.state_dict(), options.out / name / "model.pt")
    print(f"complete in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
