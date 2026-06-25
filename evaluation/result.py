"""ResultSet container for evaluation outputs."""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ResultSet:
    config: dict
    global_stats: dict
    per_regime: dict[str, dict]
    per_mode: dict[str, dict]
    truncation_floor: dict
    gap_to_floor: float
    n_params: int
    per_seed_medians: list[float]

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self._to_json(), fp, indent=2)

    def _to_json(self) -> dict:
        d = asdict(self)
        def _conv(obj):
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _conv(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_conv(v) for v in obj]
            return obj
        return _conv(d)

    @classmethod
    def load_json(cls, path: str | Path) -> "ResultSet":
        with open(path) as fp:
            return cls(**json.load(fp))
