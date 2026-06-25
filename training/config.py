"""Experiment configuration (3D Cartesian + polar sensor + POD only)."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SensorConfig:
    """Sensor placement in polar (r, theta) coordinates on the Cartesian grid.

    Each entry of `positions` is a (r, theta_deg) pair: r is normalized in
    [0, 1] (fraction of wafer radius R), theta is in degrees measured CCW from
    +x. The sensor index lookup (core/sensors.py) maps each (r, theta) to the
    nearest (ix, iy) on the canonical grid at use time. `n` must equal
    len(positions) -- it is duplicated so the rest of the pipeline can size
    tensors without unpacking positions.
    """
    n: int = 4
    strategy: str = "custom"
    positions: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class PODConfig:
    k: int = 8


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "bitcn"
    channels: int = 64
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    dropout: float = 0.05
    causal: bool = False


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr_init: float = 1e-3
    lr_final: float = 1e-5
    lr_schedule: str = "cosine"
    weight_decay: float = 0.0
    val_frac: float = 0.10
    sensor_noise: float = 0.0
    grad_clip: float = 1.0
    weight_scheme: str = "sigma"
    print_every: int = 25
    checkpoint_every: int = 20
    value_scale: float = 1.0e6
    report_anim_samples: int = 3
    report_fps: int = 5


@dataclass(frozen=True)
class DataConfig:
    """Canonical 3D grid: Cartesian (Nx, Ny) spatial, Nt time."""
    n_sim: int = 500
    nx: int = 128
    ny: int = 128
    nt: int = 150
    x_end: float = 1.0    # normalized half-width (wafer radius R)
    y_end: float = 1.0
    t_end: float = 1.0
    train_frac: float = 0.80
    seed: int = 7
    npz_dir: str | None = None
    limit: int | None = None
    oversample_tmax_above: float | None = None
    oversample_factor: int = 1
    oversample_source_substring: str | None = None
    oversample_source_factor: int = 1
    oversample_name_prefix: str | None = None
    oversample_name_factor: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pod: PODConfig = field(default_factory=PODConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: tuple[int, ...] = (7, 17, 27)
    output_dir: str = "outputs"
    basis_cache_dir: str = "outputs/basis_cache"
    tag: str = ""


def config_to_dict(cfg: ExperimentConfig) -> dict:
    d = asdict(cfg)
    def _conv(node):
        if isinstance(node, dict):
            return {k: _conv(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [_conv(v) for v in node]
        return node
    return _conv(d)


def config_to_yaml(cfg: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w") as fp:
        yaml.dump(config_to_dict(cfg), fp, default_flow_style=False,
                  sort_keys=False)


_SUBCONFIG_TYPES: dict[str, type] = {}


def _register_subconfigs() -> None:
    if _SUBCONFIG_TYPES:
        return
    import typing
    hints = typing.get_type_hints(ExperimentConfig)
    for f in fields(ExperimentConfig):
        tp = hints.get(f.name, f.type)
        if isinstance(tp, type) and hasattr(tp, "__dataclass_fields__"):
            _SUBCONFIG_TYPES[f.name] = tp


def _build_sub(cls, d: dict):
    _register_subconfigs()
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        sub = _SUBCONFIG_TYPES.get(f.name)
        if isinstance(val, dict) and sub is not None:
            val = _build_sub(sub, val)
        elif isinstance(val, list):
            # Re-tuple lists; nested lists (e.g. sensor positions [[r,theta], ...])
            # become tuples of tuples so the frozen dataclass accepts them.
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        kwargs[f.name] = val
    return cls(**kwargs)


def config_from_yaml(path: str | Path) -> ExperimentConfig:
    with open(path) as fp:
        d = yaml.safe_load(fp)
    return _build_sub(ExperimentConfig, d)


def config_with_overrides(cfg: ExperimentConfig,
                          overrides: dict[str, Any]) -> ExperimentConfig:
    d = config_to_dict(cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        node = d
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return _build_sub(ExperimentConfig, d)
