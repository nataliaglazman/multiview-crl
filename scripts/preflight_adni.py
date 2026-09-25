#!/usr/bin/env python3
"""Check that a real-data (ADNI) run will actually find its data, before submitting it.

Everything here is read-only and CPU-only. It resolves the *same* config
``scripts/launch.py`` would submit, then walks the same path-resolution code
``MyCustomDataset`` uses at startup, so a pass means training will see the
subjects reported here — not a re-implementation that can drift from it.

Usage:

    python scripts/preflight_adni.py experiments/adni_real.yaml --cluster slurm
    python scripts/preflight_adni.py experiments/adni_real.yaml --cluster slurm --sample 8
    python scripts/preflight_adni.py experiments/adni_real.yaml --cluster runai \\
        --set val_frac=0.2 image_spacing=2.0

Run it on the cluster (inside the training env — it imports MONAI through
``utils.utils``), on the login node. ``--sample N`` additionally opens N subjects
with nibabel and reports geometry and intensity statistics; that is the check
that catches a dataset whose T1 and T2 are not on the same grid.

Exit status is 0 when the run can proceed, 1 when something would make it fail
or make its numbers meaningless.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.launch import parse_cli_overrides, resolve_config  # noqa: E402

REAL_DATASETS = ("adni", "ADNI_registered", "ADNI_stripped", "ADNI_stripped_masks", "custom")

_status = {"fail": 0, "warn": 0}


def ok(msg):
    print(f"  \033[32mok\033[0m    {msg}")


def warn(msg):
    _status["warn"] += 1
    print(f"  \033[33mWARN\033[0m  {msg}")


def fail(msg):
    _status["fail"] += 1
    print(f"  \033[31mFAIL\033[0m  {msg}")


def info(msg):
    print(f"        {msg}")


def section(title):
    print(f"\n\033[1m{title}\033[0m")


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------


def check_config(config):
    section("1. Config")
    dataset_name = config.get("dataset_name")
    if dataset_name == "synthetic":
        fail("dataset_name is 'synthetic' — this config does not use real data at all.")
        info("Point the experiment at ADNI (e.g. dataset_name: ADNI_stripped_masks).")
        return None
    if dataset_name not in REAL_DATASETS:
        fail(f"dataset_name={dataset_name!r} is not a real-data dataset (expected one of {REAL_DATASETS}).")
        return None
    ok(f"dataset_name = {dataset_name}")

    dataroot = config.get("dataroot")
    labels_path = config.get("labels_path")
    if not dataroot:
        fail("dataroot is unset.")
    if not labels_path:
        fail("labels_path is unset — MyCustomDataset raises on this immediately.")
        info("It comes from experiments/cluster/<cluster>.yaml; did you pass --cluster?")
    if _status["fail"]:
        return None

    datapath = os.path.join(dataroot, dataset_name)
    info(f"datapath   = {datapath}")
    info(f"labels     = {labels_path}")
    info(f"masks_dir  = {config.get('masks_dir')}")
    info(f"cache_dir  = {config.get('cache_dir')}")
    return datapath


def check_paths(config, datapath):
    section("2. Paths on disk")
    all_present = True

    if os.path.isdir(datapath):
        n_entries = len(os.listdir(datapath))
        ok(f"image root exists ({n_entries} entries)")
    else:
        fail(f"image root does not exist: {datapath}")
        all_present = False

    labels_path = config["labels_path"]
    if os.path.isfile(labels_path):
        ok("labels CSV exists")
    else:
        fail(f"labels CSV not found: {labels_path}")
        all_present = False

    masks_dir = config.get("masks_dir")
    if masks_dir:
        if os.path.isdir(masks_dir):
            ok("masks_dir exists")
        else:
            fail(f"masks_dir set but missing: {masks_dir}")
            all_present = False
    else:
        warn("masks_dir unset — masks are expected alongside the images, else derived by thresholding.")

    cache_dir = config.get("cache_dir")
    if cache_dir:
        if os.path.isdir(cache_dir):
            ok(f"cache_dir exists, writable={os.access(cache_dir, os.W_OK)}")
            if not os.access(cache_dir, os.W_OK):
                fail("cache_dir is not writable — caching will fail mid-run.")
        else:
            parent = os.path.dirname(cache_dir.rstrip("/")) or "/"
            if os.path.isdir(parent) and os.access(parent, os.W_OK):
                ok(f"cache_dir does not exist yet but {parent} is writable (it will be created)")
            else:
                fail(f"cache_dir {cache_dir} cannot be created (parent not writable)")
    elif config.get("cache_dataset"):
        warn("cache_dataset is on but cache_dir is unset — volumes are cached in RAM only, every run.")

    return all_present


def check_csv(config):
    section("3. Labels CSV")
    import pandas as pd

    df = pd.read_csv(config["labels_path"])
    ok(f"read {len(df)} rows, columns: {list(df.columns)}")

    for col in ("Subject", "Group"):
        if col not in df.columns:
            fail(f"required column {col!r} is missing — load_data indexes it directly.")
            return None

    n_subjects = df["Subject"].nunique()
    if n_subjects != len(df):
        info(f"{n_subjects} unique subjects across {len(df)} rows (some subjects have several rows)")
    else:
        ok(f"{n_subjects} unique subjects, one row each")

    counts = df["Group"].value_counts().sort_index()
    info("group counts: " + ", ".join(f"{g}={c}" for g, c in counts.items()))
    if len(counts) < 2:
        warn("only one diagnosis group — the content/diagnosis probe will be uninformative.")

    multi = df.groupby("Subject")["Group"].nunique()
    inconsistent = multi[multi > 1]
    if len(inconsistent):
        warn(f"{len(inconsistent)} subject(s) carry more than one Group label: {list(inconsistent.index[:5])}")
        info("The split buckets each subject by its first label; scans are not dropped.")

    return df


def check_items(config, datapath):
    section("4. Path resolution (same code the dataset runs)")

    # load_data logs a warning per missing file; the counts below are the summary.
    logging.disable(logging.WARNING)
    try:
        from data.datasets import MyCustomDataset
    except ImportError as e:
        logging.disable(logging.NOTSET)
        fail(f"cannot import the dataset ({e}).")
        info("Run this inside the training environment (conda activate multiview-env).")
        return None
    try:
        dataset = MyCustomDataset(
            data_dir=datapath,
            mode="train",
            spacing=config.get("image_spacing", 2.0),
            crop_margin=config.get("crop_margin", 0),
            spatial_size=config.get("spatial_size"),
            cache=False,  # constructor only — no volume is read here
            cache_dir=None,
            labels_path=config["labels_path"],
            masks_dir=config.get("masks_dir"),
            asymmetric_aug=config.get("asymmetric_aug", False),
            shared_brain_mask=config.get("shared_brain_mask", False),
            val_frac=config.get("val_frac", 0.0),
            test_frac=config.get("test_frac", 0.0),
            split_seed=config.get("split_seed", 0),
        )
    except Exception as e:
        fail(f"MyCustomDataset failed to construct: {type(e).__name__}: {e}")
        return None
    finally:
        logging.disable(logging.NOTSET)

    all_items = dataset._all_items
    if not all_items:
        fail("no subject resolved to BOTH a T1 and a T2 — nothing to train on.")
        info(f"Expected layout: {datapath}/<Subject>/t1/*.nii.gz and <Subject>/t2/*FLAIR*.nii.gz")
        info("Check that Subject values in the CSV match the directory names exactly.")
        return None

    ok(f"{len(all_items)} scan pairs resolved (T1 + T2 both present)")

    with_masks = sum(1 for it in all_items if "mask_image" in it and "mask_z_image" in it)
    if with_masks == len(all_items):
        ok(f"brain masks found for all {with_masks} pairs (mask pipeline active)")
    elif with_masks == 0:
        warn("no brain masks found — masks will be derived by thresholding (image > 0).")
        info("If you meant to use precomputed masks, check masks_dir and the *_brain_mask.nii.gz suffix.")
    else:
        warn(f"masks found for only {with_masks}/{len(all_items)} pairs.")
        info("Mixed coverage disables the disk-mask pipeline entirely: ALL samples fall back to")
        info("thresholding, so the run silently ignores the masks you do have.")

    return dataset


def check_split(config, dataset):
    section("5. Split")
    from data.splits import split_summary, subject_level_split

    all_items = dataset._all_items
    val_frac = config.get("val_frac", 0.0)
    test_frac = config.get("test_frac", 0.0)

    split = subject_level_split(all_items, val_frac=val_frac, test_frac=test_frac, seed=config.get("split_seed", 0))
    label_names = {v: k for k, v in dataset.label_map.items()}

    if val_frac == 0.0 and test_frac == 0.0:
        fail("no split configured: train, val and test are the same subjects.")
        info("Every validation number this run reports would be a training number, and")
        info("best-checkpoint selection would pick on the training set. Set val_frac.")
        info("Suggested: --set val_frac=0.2 test_frac=0.1")
    else:
        ok(f"subject-level split, seed={config.get('split_seed', 0)}")
        for line in split_summary(all_items, split, label_names=label_names):
            print("      " + line)

        n_val = len(split["val"])
        if n_val < 2 * config.get("batch_size", 4):
            warn(f"val has {n_val} scans, under two batches at batch_size={config.get('batch_size')}.")
            info("Probe accuracies on a val set this small are very noisy.")
        n_train = len(split["train"])
        if n_train < 50:
            warn(f"train has only {n_train} scans.")

    return split


def check_volumes(config, dataset, n_sample):
    section(f"6. Volume geometry ({n_sample} sampled subjects)")
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        warn(f"skipped: {e}")
        return

    items = dataset._all_items
    step = max(1, len(items) // n_sample)
    sampled = items[::step][:n_sample]

    shapes, spacings, mismatched, problems = set(), set(), [], []
    for it in sampled:
        try:
            t1 = nib.load(it["image"])
            t2 = nib.load(it["z_image"])
        except Exception as e:
            problems.append(f"{it['subject']}: unreadable ({type(e).__name__}: {e})")
            continue

        shapes.add(tuple(int(s) for s in t1.shape[:3]))
        spacings.add(tuple(round(float(z), 3) for z in t1.header.get_zooms()[:3]))
        if t1.shape[:3] != t2.shape[:3]:
            mismatched.append(f"{it['subject']}: T1 {t1.shape[:3]} vs T2 {t2.shape[:3]}")

        data = np.asanyarray(t1.dataobj, dtype="float32")
        if not np.isfinite(data).all():
            problems.append(f"{it['subject']}: T1 contains NaN/Inf")
        nonzero = data[data > 0]
        if nonzero.size == 0:
            problems.append(f"{it['subject']}: T1 is all zeros")

    if problems:
        for p in problems[:5]:
            fail(p)
    else:
        ok("all sampled volumes readable, finite and non-empty")

    info(f"native shapes:  {sorted(shapes)}")
    info(f"native spacing: {sorted(spacings)}")

    if mismatched:
        fail(f"{len(mismatched)} sampled subject(s) have T1/T2 on different grids:")
        for m in mismatched[:5]:
            info(m)
        info("The two views are treated as a registered pair; resampling to a common")
        info("spatial_size will not align them. Re-register before training.")
    else:
        ok("T1 and T2 share a grid in every sampled subject")

    if len(spacings) > 1:
        warn("mixed native voxel spacing across subjects — Spacingd resamples to a common grid,")
        info(f"but check that image_spacing={config.get('image_spacing')} is not upsampling most of them.")

    # A last sanity check on intensity scale: the reconstruction loss clamps to
    # [-1, 1] after NormalizeIntensityd, so wildly different raw ranges per
    # modality are worth seeing before they show up as a flat recon loss.
    if sampled:
        it = sampled[0]
        for key, name in (("image", "T1"), ("z_image", "T2")):
            try:
                d = np.asanyarray(nib.load(it[key]).dataobj, dtype="float32")
            except Exception:
                continue
            fg = d[d > 0]
            if fg.size:
                info(f"{name} foreground intensity: min={fg.min():.3g} max={fg.max():.3g} mean={fg.mean():.3g}")


def check_cache(config, dataset):
    section("7. Preprocessing cache")
    spatial_size = config.get("spatial_size")
    if spatial_size is None:
        spacing = config.get("image_spacing", 2.0)
        spatial_size = {1.0: (182, 218, 182), 2.0: (91, 109, 91)}.get(spacing)
        if spatial_size is None:
            spatial_size = tuple(int(s / spacing) for s in (182, 218, 182))
        info(f"spatial_size unset — derived {tuple(spatial_size)} from image_spacing={spacing}")

    voxels = 1
    for s in spatial_size:
        voxels *= int(s)
    # Cached per subject: image_t1, image_t2, mask_t1, mask_t2, all float32.
    per_sample = voxels * 4 * 4
    total = per_sample * len(dataset._all_items)
    info(f"cached volume shape: {tuple(spatial_size)} ({voxels:,} voxels)")
    info(f"per subject: {human_bytes(per_sample)} (2 views + 2 masks, float32)")
    info(f"full dataset: {human_bytes(total)} across {len(dataset._all_items)} subjects")

    cache_dir = config.get("cache_dir")
    if not config.get("cache_dataset"):
        warn("cache_dataset is off — every epoch re-reads and re-resamples NIfTIs from disk.")
        return
    if not cache_dir:
        warn(f"RAM-only cache: expect ~{human_bytes(total)} of host memory held by the training process.")
        return

    fingerprint = dataset._cache_fingerprint()
    cache_root = Path(cache_dir) / f"preprocessed_{fingerprint}"
    if cache_root.is_dir():
        present = len(list(cache_root.glob("*.pt")))
        ok(f"cache exists: {cache_root} ({present}/{len(dataset._all_items)} samples)")
        if present < len(dataset._all_items):
            info("Missing entries are filled in on startup; the run resumes a partial cache.")
    else:
        info(f"cache will be built at: {cache_root}")
        info("First run pays the preprocessing cost once; later runs with the same")
        info("spacing/spatial_size/subject list reuse it.")

    free = None
    try:
        st = os.statvfs(cache_dir if os.path.isdir(cache_dir) else os.path.dirname(cache_dir.rstrip("/")) or "/")
        free = st.f_bavail * st.f_frsize
    except OSError:
        pass
    if free is not None:
        if free < total:
            fail(f"only {human_bytes(free)} free where the cache goes, needs ~{human_bytes(total)}")
        else:
            ok(f"{human_bytes(free)} free on the cache filesystem")


def check_training_settings(config):
    section("8. Training settings")
    bs = config.get("batch_size")
    spatial_size = config.get("spatial_size")
    if spatial_size:
        voxels = 1
        for s in spatial_size:
            voxels *= int(s)
        # Rough activation-memory proxy, only useful for spotting a batch size
        # carried over from 64^3 synthetic runs.
        if bs and voxels * bs > 64**3 * 32:
            warn(f"batch_size={bs} at {tuple(spatial_size)} is a lot of voxels per step.")
            info("Synthetic runs at 64^3 tolerate large batches; real volumes are ~15x bigger.")
            info("If it OOMs, drop batch_size before anything else.")

    if config.get("eval_dci"):
        warn("eval_dci is on, but real data has no ground-truth latents — DCI/R^2 against")
        info("factors is synthetic-only. It is skipped at runtime; the real-data readouts are")
        info("the content/style separation metrics (separation_score, content/modality_invariance,")
        info("style/subject_invariance, content/diagnosis_probe_acc).")

    if config.get("scale_style_hsic_loss", 0.0) > 0:
        fail("scale_style_hsic_loss > 0 requires synthetic ground-truth factors; this run will refuse to start.")

    if not config.get("cache_dataset"):
        info("cache_dataset off: startup is fast but each step re-reads NIfTIs.")

    val_every = config.get("val_every", 0)
    if config.get("val_frac", 0.0) > 0 and not val_every:
        info("val_every is unset — separation metrics still run on the schedule in the training loop.")

    ok(f"batch_size={bs}, train_steps={config.get('train_steps')}, lr={config.get('lr')}")


class _StubDataset:
    """Stands in for MyCustomDataset in --self-test, without torch or MONAI."""

    def __init__(self, items, label_map, fingerprint="deadbeefdeadbeef"):
        self._all_items = items
        self.label_map = label_map
        self._fingerprint = fingerprint

    def _cache_fingerprint(self):
        return self._fingerprint


def _self_test():
    """Exercise the checks that do not need the training stack, on synthetic inputs.

    Covers the split, cache-size and settings checks — the parts that do
    arithmetic or make a judgement. The config, path, CSV and volume checks are
    thin wrappers over the filesystem, pandas and nibabel and are not stubbed.
    """
    items, label_map = [], {"CN": 0, "MCI": 1, "AD": 2}
    for s in range(120):
        items.append({"subject": f"S{s:03d}", "label": s % 3, "image": "t1.nii.gz", "z_image": "t2.nii.gz"})
    dataset = _StubDataset(items, label_map)

    print("\033[1mself-test: split reporting\033[0m")
    check_split({"val_frac": 0.2, "test_frac": 0.1, "split_seed": 0, "batch_size": 4}, dataset)
    expected_fail = _status["fail"]
    assert expected_fail == 0, "a valid split should not report a failure"

    print("\n\033[1mself-test: unsplit config must be flagged\033[0m")
    check_split({"val_frac": 0.0, "test_frac": 0.0, "batch_size": 4}, dataset)
    assert _status["fail"] == 1, "an unsplit run must be a blocking failure"
    _status["fail"] = 0

    print("\n\033[1mself-test: tiny val split warns\033[0m")
    before = _status["warn"]
    check_split({"val_frac": 0.02, "test_frac": 0.0, "split_seed": 0, "batch_size": 64}, dataset)
    assert _status["warn"] > before, "a val split under two batches must warn"

    print("\n\033[1mself-test: cache sizing\033[0m")
    check_cache({"spatial_size": [96, 112, 96], "cache_dataset": True, "cache_dir": None}, dataset)
    # 96*112*96 voxels * 4 tensors * 4 bytes = 16.5 MB/subject, ~1.9 GB for 120.
    check_cache({"image_spacing": 2.0, "cache_dataset": False, "cache_dir": None}, dataset)

    print("\n\033[1mself-test: settings\033[0m")
    _status["warn"] = 0
    check_training_settings({"batch_size": 128, "spatial_size": [96, 112, 96], "eval_dci": True, "lr": 1e-3})
    assert _status["warn"] >= 2, "a 128 batch at full resolution and eval_dci should both warn"
    _status["fail"] = 0
    check_training_settings({"batch_size": 16, "scale_style_hsic_loss": 1.0, "lr": 1e-3})
    assert _status["fail"] == 1, "style HSIC on real data must be a blocking failure"

    print("\n\033[32mself-test passed\033[0m")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Preflight a real-data (ADNI) run: resolve its config and verify the data is there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    if "--self-test" in sys.argv:
        return _self_test()
    parser.add_argument("experiment", type=Path, help="Path to the experiment YAML")
    parser.add_argument("--cluster", default="local", help="Cluster name (matches experiments/cluster/<name>.yaml)")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="Config overrides, e.g. val_frac=0.2")
    parser.add_argument("--sample", type=int, default=0, help="Open N subjects with nibabel and check geometry")
    parser.add_argument("--self-test", action="store_true", help="Check this script's own logic; needs no data")
    args = parser.parse_args()

    if not args.experiment.exists():
        print(f"Error: experiment not found: {args.experiment}", file=sys.stderr)
        return 2

    config = resolve_config(args.experiment, args.cluster, parse_cli_overrides(args.overrides))

    print(f"\033[1mPreflight: {args.experiment} (cluster={args.cluster})\033[0m")

    datapath = check_config(config)
    if datapath is None:
        return report()

    if not check_paths(config, datapath):
        return report()

    if check_csv(config) is None:
        return report()

    dataset = check_items(config, datapath)
    if dataset is None:
        return report()

    check_split(config, dataset)

    if args.sample > 0:
        check_volumes(config, dataset, args.sample)
    else:
        section("6. Volume geometry")
        info("skipped — pass --sample N to open N subjects and check shapes/spacing/intensities")

    check_cache(config, dataset)
    check_training_settings(config)
    return report()


def report():
    print()
    if _status["fail"]:
        print(f"\033[31m{_status['fail']} blocking problem(s), {_status['warn']} warning(s).\033[0m")
        return 1
    if _status["warn"]:
        print(f"\033[33mNo blocking problems, {_status['warn']} warning(s) — review before submitting.\033[0m")
        return 0
    print("\033[32mAll checks passed. Ready to submit.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
