"""Shared targets and provenance for encoder-only lesion/sulcal follow-ups."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval.lesion.checkpoint_lesion_analysis import json_safe
from eval.metrics.dci import CONTENT_FACTOR_NAMES
from eval.protocol.score_checkpoint import make_dataset
from eval.synthetic.synthetic_dataset import SULCAL_CLEFT_HALF_WIDTH

VIEWS = ("t1", "flair")
TARGETS = (*CONTENT_FACTOR_NAMES, "centroid_x", "centroid_y", "centroid_z", "sulcal_amplitude", "sulcal_magnitude")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_report(directory, report):
    path = Path(directory) / "report.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def provenance(cfg, args, device):
    # This is the evaluation/control source snapshot, not a claim about the code
    # that originally trained the checkpoint.
    root = Path(__file__).resolve().parents[2]
    paths = [p for folder in ("eval", "models", "data", "utils", "training") for p in (root / folder).rglob("*.py")]
    return {
        "status": "running",
        "settings": cfg,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "target_names": list(TARGETS),
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in sorted(paths)},
        "scope": "Synthetic encoder-only follow-up; finite-probe success/failure does not prove identifiability.",
    }


def dataset(cfg, count, split):
    if cfg.get("synthetic_mode", "pseudo_mri") != "pseudo_mri" or cfg["n_content"] != 9:
        raise ValueError("This protocol requires the nine-factor pseudo_mri encoder-only recipe")
    # The historical encoder-only factory supports the recipe used in these runs.
    # Reject newer non-default renderer options it cannot restore faithfully.
    unsupported_defaults = {
        "synthetic_lesion_mode": "sphere",
        "synthetic_content_prior": "normal",
        "synthetic_content_squash": "auto",
        "synthetic_content_amp_scale": None,
        "synthetic_cortex_parameterization": "additive",
        "synthetic_center_local_deformations": False,
    }
    for key, default in unsupported_defaults.items():
        if key in cfg and cfg[key] != default:
            raise ValueError(f"The encoder-only dataset factory cannot restore non-default {key}; refusing data drift")
    if count < 1:
        raise ValueError("Dataset size must be positive")
    return make_dataset(cfg, count, mode=split)


def sample_targets(inner, latents):
    """Original nine controls, physical centroid, and the sulcal factor's physical size.

    The size is the signed corrugation amplitude and its magnitude, or under
    sulcal_mode="atrophy" the cleft half-width, which is positive, so both columns hold it.
    No target is inferred from a model or used to pick a crop. Lesion support is
    returned for the explicitly supervised control's loss only.
    """
    r = inner.renderer
    if inner.mode != "pseudo_mri" or r.lesion_mode != "sphere":
        raise ValueError("Need pseudo_mri sphere lesions")
    z = latents["z_content"].detach().cpu()
    if z.shape != (9,):
        raise ValueError("Need exactly nine content factors")
    _, support = r.render_structure(z, latents["z_deformation"], latents["z_fissure"], "cpu", clean=inner.clean_content)
    mass = support.sum()
    if not torch.isfinite(support).all() or mass <= 0:
        raise ValueError("Empty/non-finite lesion; refusing to silently change the evaluated cohort")
    centroid = (support[..., None] * r.coords).sum((0, 1, 2)) / mass
    squash = r.content_squash
    if squash == "auto":
        squash = "tanh" if inner.clean_content else "clamp"
    value = z[8].tanh() if squash == "tanh" else z[8].clamp(-1, 1) if squash == "clamp" else z[8]
    if squash not in ("tanh", "clamp", "none"):
        raise ValueError(f"Unknown content squash: {squash}")
    amp_scale = 1.0 if r.content_amp_scale is None else r.content_amp_scale[8]
    if r.sulcal_mode == "atrophy":
        mid, half = SULCAL_CLEFT_HALF_WIDTH
        amplitude = (mid + value * (half * r.content_scale * amp_scale)).clamp_min(0.0)
    else:
        amplitude = value * (0.06 * r.content_scale * amp_scale)
    target = torch.cat((z, centroid, amplitude.reshape(1), amplitude.abs().reshape(1))).numpy()
    if not np.isfinite(target).all():
        raise ValueError("Non-finite targets")
    return target, support


def dataset_metadata(ds, input_hash, ids):
    return {
        "ids": list(map(int, ids)),
        "generator_split_seed": int(ds._inner.seed),
        "input_sha256": input_hash,
        "fixed_mean": ds._fixed_mean,
        "fixed_scale": ds._fixed_scale,
        "resolution": ds.res,
    }
