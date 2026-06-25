"""Model registry."""

from __future__ import annotations
from typing import Type

import torch.nn as nn

_REGISTRY: dict[str, Type[nn.Module]] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def create_model(name: str, **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model: {name!r}, available: {list_models()}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
