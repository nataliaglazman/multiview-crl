"""Frozen spatial probes during encoder training; labels never enter optimization."""

import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from eval.encoder import encoder_spatial_target_audit as audit
from eval.encoder.encoder_target_protocol import dataset, provenance, save_csv, save_report
from eval.lesion.checkpoint_lesion_analysis import state_digest


@contextmanager
def preserve_training_state(model, device):
    """Dataset construction resets RNGs; probe evaluation must not change training."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_state = torch.mps.get_rng_state() if torch.device(device).type == "mps" else None
    try:
        model.eval()
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        for module, training in modes:
            module.training = training


def add_initial_comparison(rows, initial):
    fields = ("view", "grid", "stage", "probe", "condition", "target")
    lookup = {tuple(row[key] for key in fields): row["test_r2"] for row in initial or []}
    for row in rows:
        floor = lookup.get(tuple(row[key] for key in fields))
        row["initial_test_r2"] = floor
        row["delta_initial"] = row["test_r2"] - floor if floor is not None else None


def print_recovery(rows, step, writer=None):
    """Compact local-target table; CSV/JSON retain every factor and both controls."""
    groups = {}
    for row in rows:
        key = (row["view"], row["grid"], row["stage"], row["probe"])
        groups.setdefault(key, {}).setdefault(row["condition"], {})[row["target"]] = row
        if writer is not None:
            tag = "/".join(map(str, (*key, row["condition"], row["target"])))
            writer.add_scalar(f"spatial_recovery/{tag}/r2", row["test_r2"], step)
            if row["delta_initial"] is not None:
                writer.add_scalar(f"spatial_recovery/{tag}/delta_initial", row["delta_initial"], step)
    print("    --- spatial recovery: held-out probe R² (not an identifiability guarantee) ---", flush=True)
    print(
        "    view  grid stage       probe  lesion_xyz centroid_xyz sulcal_z signed_amp  Δamp_init amp_null", flush=True
    )
    for (view, grid, stage, probe), conditions in groups.items():
        observed = conditions["observed"]
        lesion = np.mean([observed[f"lesion_{axis}"]["test_r2"] for axis in "xyz"])
        centroid = np.mean([observed[f"centroid_{axis}"]["test_r2"] for axis in "xyz"])
        delta = observed["sulcal_amplitude"]["delta_initial"]
        delta_text = f"{delta:+.3f}" if delta is not None else "n/a"
        print(
            f"    {view:5s} {grid:4d} {stage:11s} {probe:5s} {lesion:+10.3f} {centroid:+12.3f} "
            f"{observed['sulcal_widening']['test_r2']:+8.3f} "
            f"{observed['sulcal_amplitude']['test_r2']:+10.3f} {delta_text:>10s} "
            f"{conditions['shuffled']['sulcal_amplitude']['test_r2']:+8.3f}",
            flush=True,
        )


def evaluate_spatial_recovery(model, cfg, device, save_dir, step, initial=None, writer=None):
    """Fit/tune on validation subjects, score separately; keep only small reports.

    This cohort is monitored repeatedly during training and is diagnostic, not an
    untouched final test set. It never selects an encoder checkpoint or changes
    gradients. An offline final audit may use the same protocol on a new cohort.
    """
    directory = Path(save_dir) / "spatial_recovery" / f"step_{step:08d}"
    args = SimpleNamespace(
        batch_size=cfg["spatial_recovery_batch_size"],
        grids=cfg["spatial_recovery_grids"],
        include_native=cfg["spatial_recovery_native"],
        stages=("backbone", "projected"),
        seed=cfg["spatial_recovery_seed"],
    )
    directory.mkdir(parents=True, exist_ok=False)
    arm = "initial" if step == 0 else "trained"
    report = provenance(cfg, args, device)
    report.update(
        step=step,
        arm=arm,
        protocol="75/25 validation fit/tune; separate diagnostic cohort; fixed splits; ridge and RBF with shuffled controls",
        evaluation_only=True,
        feature_banks_retained=False,
    )
    save_report(directory, report)
    before, started = state_digest(model), time.perf_counter()
    print(f"  [eval] spatial recovery @ step {step}; grids={args.grids}; native={args.include_native}", flush=True)
    try:
        with preserve_training_state(model, device), tempfile.TemporaryDirectory(
            prefix=".features-", dir=directory
        ) as temporary:
            banks = {}
            try:
                for split, count in (("val", cfg["num_val_samples"]), ("test", cfg["spatial_recovery_test_samples"])):
                    ds = dataset(cfg, count, split)
                    banks[split] = audit.extract(model, ds, args, device, Path(temporary) / split)
                cohorts = {split: bank[2] for split, bank in banks.items()}
                if initial is not None:
                    for split in cohorts:
                        if cohorts[split]["input_sha256"] != initial["cohorts"][split]["input_sha256"]:
                            raise ValueError("Spatial recovery cohorts changed since initialization")
                with threadpool_limits(limits=1):
                    rows, splits = audit.score_banks(banks, args, arm, directory)
            finally:
                # Close memmaps before TemporaryDirectory removes the large feature banks.
                for bank in banks.values():
                    for array in bank[0].values():
                        array._mmap.close()
        if state_digest(model) != before:
            raise RuntimeError("Spatial recovery changed encoder parameters or buffers")
        add_initial_comparison(rows, rows if step == 0 else initial["probes"] if initial is not None else None)
        report.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            model_state_sha256=before,
            encoder_unchanged=True,
            cohorts=cohorts,
            probe_split=splits,
            probes=rows,
            summary=audit.focused_summary(rows),
        )
        save_csv(directory / "probes.csv", rows)
        save_csv(directory / "summary.csv", report["summary"])
        save_report(directory, report)
        print_recovery(rows, step, writer)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(directory, report)
        raise
