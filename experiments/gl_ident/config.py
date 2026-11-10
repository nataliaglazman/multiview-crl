"""Run configuration: nested dataclasses, loadable from YAML.

A YAML file holds any subset of the fields below, nested by section::

    generator: {snr: .inf, lambda_nu: 0.0}
    model: {kind: local, n_g: 2}
    train: {steps: 30000}

Fields not given keep their defaults. ``snr: .inf`` (or ``null``) means no noise.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GeneratorConfig:
    dim: int = 2
    image_size: int = 96
    stride: int = 4  # m
    d_gamma: int = 4
    d_c: int = 4
    d_s: int = 3
    hidden: int = 32
    rho_f: int = 1  # fixed by the single 3x3 conv; recorded, asserted, not a knob
    ell_c: float = 1.5
    ell_s: float = 2.0
    ell_nu: float = 6.0
    a_scale: float = 0.5
    b_scale: float = 0.5
    style_depends_on_content: bool = False
    snr: float = math.inf
    noise: str = "gaussian"  # gaussian | rician
    lambda_nu: float = 0.0
    padding: str = "circular"  # circular | zeros (E7)
    weights_seed: int = 0  # F_k, A, B_k: fixed across every run
    reference_seed: int = 1
    reference_size: int = 4096
    val_seed: int = 2
    test_seed: int = 3
    n_val: int = 2000
    n_test: int = 2000

    @property
    def n_sites(self) -> int:
        assert self.image_size % self.stride == 0
        return self.image_size // self.stride

    @property
    def sigma(self) -> float:
        return 0.0 if self.snr is None or math.isinf(self.snr) else 1.0 / self.snr


@dataclass
class ModelConfig:
    kind: str = "local"  # local | global | pooled
    hidden: int = 128
    n_g: int = 2  # local encoder 3x3 layers; rho_g = n_g
    d_hat: int | None = None  # local code size; None = d_c + d_gamma
    n_conv: int = 3  # window / pooled encoder conv layers
    mlp_hidden: int = 128
    window: int = 6  # w, coarse sites
    window_offsets: str = "disjoint"  # disjoint | overlap (E4)
    # Cosine similarity reads only the direction of the code, which has one degree of freedom fewer than
    # its length. True adds one output unit so the direction still has d_hat (or d_gamma) dimensions.
    sphere_extra_dim: bool = True
    padding: str = "circular"


@dataclass
class LossConfig:
    objective: str = "infonce"  # infonce | barlow_twins
    similarity: str = "cosine"  # cosine | l2
    tau: float = 0.1
    sites_per_subject: int = 16
    bt_lambda: float = 0.005


@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 30000
    batch_size: int = 64
    lr: float = 1e-3
    eval_every: int = 1000
    device: str = "auto"
    log_every: int = 100


@dataclass
class EvalConfig:
    n_fit: int = 1000
    sites_per_subject: int = 64
    n_windows: int = 16
    r_max: int | None = None  # None = R + 2
    seed: int = 7
    mlp_hidden: int = 128
    mlp_max_epochs: int = 300
    mlp_patience: int = 10


@dataclass
class RunConfig:
    name: str = "run"
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(obj, values: dict):
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"{type(obj).__name__} has no field {key!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _merge(current, value)
        else:
            if key == "snr" and value is None:
                value = math.inf
            setattr(obj, key, value)
    return obj


def load_config(*paths, overrides: dict | None = None) -> RunConfig:
    """Defaults, then each YAML in order, then ``overrides`` ({"train.seed": 1, ...})."""
    cfg = RunConfig()
    for path in paths:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for dotted, value in (overrides or {}).items():
        *parents, leaf = dotted.split(".")
        target = cfg
        for parent in parents:
            target = getattr(target, parent)
        _merge(target, {leaf: value})
    return cfg


def save_config(cfg: RunConfig, path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
