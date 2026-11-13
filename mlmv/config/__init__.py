"""Run configuration: ``defaults.yaml`` loaded into nested dataclasses.

``defaults.yaml`` is the only place a default lives; the dataclasses below carry types, not values. A run
is ``defaults.yaml``, then each run YAML in order, then ``--set a.b=value`` overrides (values parsed as YAML).
Unknown keys are errors, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULTS = Path(__file__).with_name("defaults.yaml")


@dataclass
class RunSection:
    name: str
    out_dir: str
    device: str
    amp: str
    log_every: int
    eval_every: int
    ckpt_every: int
    source: str


@dataclass
class DataConfig:
    spacing_mm: float
    grid: list
    intensity_clip_pct: list
    metadata: str
    splits: str


@dataclass
class GeometryConfig:
    atlas: str
    region_map: str
    rho: typing.Any  # "auto" or a number of feature voxels
    min_positions_per_half: int
    metric: str
    eligibility_normals: int


@dataclass
class ModelConfig:
    stem_channels: int
    trunk_channels: int
    trunk_blocks: int
    trunk_spatial_convs: int
    down_kernel: int
    stride: int
    norm: str
    norm_groups: int
    stem_norm: str
    dims: dict
    max_rf_radius: float
    template_subtraction: bool
    template_momentum: float
    expander: list
    variance_floor: bool
    variance_floor_weight: float
    checkpoint: bool


@dataclass
class SamplerS:
    positions: int


@dataclass
class SamplerL:
    regions: int
    fraction: float


@dataclass
class SamplerG:
    volume_ratio: list
    splits_per_subject: int


@dataclass
class SamplerConfig:
    batch_subjects: int
    S: SamplerS
    L: SamplerL
    G: SamplerG


@dataclass
class InfoNCEConfig:
    temperature: float
    queue: int


@dataclass
class BufferConfig:
    size: int
    momentum: float


@dataclass
class LossConfig:
    objective: str
    centring: bool
    standardise: str
    lambda_off: float
    corr_ema: float
    weights: dict
    balance: str
    infonce: InfoNCEConfig
    buffer: BufferConfig


@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    warmup_steps: int
    steps: int
    grad_clip: float
    rho_measure_step: int
    rho_positions: int
    rho_mass: float


@dataclass
class Phase3aConfig:
    enabled: bool
    pairs_per_batch: int
    lambda_x: float
    min_pair_dice: float


@dataclass
class FlowConfig:
    layers: int
    hidden: int
    site_embedding: int
    condition_on_g: bool
    lambda_id: float
    kernel_scales: list
    strata: dict
    min_samples_per_cell: int
    cache_positions_per_region: int
    buffer_per_cell: int
    lr: float
    steps: int


@dataclass
class SynthConfig:
    variant: str
    regions: str
    n_regions: int
    lattice: int
    image: int
    dims: dict
    n_sites: int
    decoder_hidden: int
    s_width: float
    w_variance: float
    lesion_blobs: list
    lesion_radius: list
    noise_tau2: float
    dependent_scale: float
    confound_strength: float
    n_train: int
    n_val: int
    n_test: int
    weights_seed: int


@dataclass
class Thresholds:
    g_r2_min: float
    r_r2_min: float
    s_r2_min: float
    leak_r2_max: float
    g_cka_min: float
    pair_ratio_min: float
    seed_r2_min: float
    eff_rank_frac_min: float


@dataclass
class EvalConfig:
    probes: list
    cv_folds: int
    ridge_alpha: float
    mlp_hidden: list
    mlp_epochs: int
    positions_per_subject: int
    max_subjects: int
    permutations: int
    fdr_q: float
    thresholds: Thresholds


@dataclass
class Config:
    seed: int
    run: RunSection
    data: DataConfig
    geometry: GeometryConfig
    model: ModelConfig
    sampler: SamplerConfig
    loss: LossConfig
    optim: OptimConfig
    phase3a: Phase3aConfig
    flow: FlowConfig
    synth: SynthConfig
    eval: EvalConfig

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, values: dict, where: str):
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = set(values) - set(names)
    missing = set(names) - set(values)
    if unknown or missing:
        raise KeyError(f"{where or 'config'}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    kwargs = {}
    for name in names:
        hint, value = hints[name], values[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise TypeError(f"{where}{name} must be a mapping")
            value = _build(hint, value, f"{where}{name}.")
        kwargs[name] = value
    return cls(**kwargs)


def _merge(base: dict, update: dict, where: str = "") -> None:
    for key, value in update.items():
        if key not in base:
            raise KeyError(f"unknown config key {where}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict) and key not in ("dims", "weights", "strata"):
            _merge(base[key], value, f"{where}{key}.")
        elif isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value


def parse_overrides(items) -> dict:
    """``["a.b=1", "c=[1, 2]"]`` -> nested dict, values parsed as YAML."""
    out: dict = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not key=value")
        *parents, leaf = key.strip().split(".")
        node = out
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def load_config(*paths, overrides=None) -> Config:
    """``defaults.yaml``, then each YAML in ``paths``, then ``overrides`` (a nested dict or a list of a.b=v)."""
    raw = yaml.safe_load(DEFAULTS.read_text())
    for path in paths:
        _merge(raw, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        _merge(raw, overrides if isinstance(overrides, dict) else parse_overrides(overrides))
    return _build(Config, copy.deepcopy(raw), "")


def git_sha(path=".") -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "mlmv"], cwd=path, capture_output=True, text=True)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_config(cfg: Config, path) -> None:
    """Resolved config plus the commit it ran at."""
    out = cfg.to_dict()
    out["_commit"] = git_sha(Path(__file__).resolve().parents[2])
    Path(path).write_text(yaml.safe_dump(out, sort_keys=False))
