"""Ingest a tool-exported sensor-history CSV into the raw dict the 3D real-data
adapter consumes (`{time, w_<label>}`).

Ported from the 2D `wafer_app/ingest/csv_ingest.py` marker parser (robust to
any subset of sensors, header height varies, values located by column-A marker
text -- not fixed rows). The tool CSV has two blocks:

  metadata header   Detect time[ms], Wafer gap[nm], Zero position[nm], ...
  Sampling Data[nm] Time[ms] in col A, one column per sensor label, nm values

The model-ready quantity is w = (sampling_nm - zero_nm) in METRES, time in
SECONDS. Unlike the 2D GUI path this does NOT attach layout geometry -- the 3D
adapter's config (configs/real_exp_n6.yaml) owns the (r, theta) mapping, so all
we emit here is time + one w_<label> series per present sensor.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MISSING = {"", "-", "--", "nan", "n/a", "na"}
NM_TO_M = 1.0e-9
MS_TO_S = 1.0e-3


@dataclass
class ParsedCsv:
    time_ms: np.ndarray                       # (T,)
    data_nm: dict[str, np.ndarray]            # label -> (T,), NaN where missing
    zero_nm: dict[str, float] = field(default_factory=dict)
    detect_ms: dict[str, float] = field(default_factory=dict)
    sampling_start_s: float | None = None
    sampling_end_s: float | None = None
    source: str = ""


def _num(s: str) -> float:
    s = (s or "").strip()
    if s.lower() in _MISSING:
        return float("nan")
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return float("nan")


def _clock_to_s(s: str) -> float | None:
    """'mm:ss.s' or 'hh:mm:ss.s' -> seconds."""
    s = (s or "").strip()
    if not s or ":" not in s:
        return None
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return None
    sec = 0.0
    for p in parts:
        sec = sec * 60.0 + p
    return sec


def _find_marker(rows: list[list[str]], text: str) -> int | None:
    for i, r in enumerate(rows):
        if r and r[0].strip() == text:
            return i
    return None


def _header_labels(header: list[str]) -> dict[str, int]:
    """Sensor label -> column index, for labels in columns 1.. (col A is the
    block's own label / the Time header)."""
    out: dict[str, int] = {}
    for j in range(1, len(header)):
        lab = header[j].strip()
        if lab:
            out[lab] = j
    return out


def _read_keyed_row(rows, marker_idx) -> dict[str, float]:
    """Block <marker> / <label header> / <values> -> {label: value}."""
    if marker_idx is None or marker_idx + 2 >= len(rows):
        return {}
    labels = _header_labels(rows[marker_idx + 1])
    vals = rows[marker_idx + 2]
    return {lab: (_num(vals[j]) if j < len(vals) else float("nan"))
            for lab, j in labels.items()}


def read_csv(path: str | Path) -> ParsedCsv:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p}")
    with open(p, newline="") as fp:
        rows = list(csv.reader(fp))
    if not rows:
        raise ValueError(f"{p}: empty CSV")

    detect = _read_keyed_row(rows, _find_marker(rows, "Detect time[ms]"))
    zero = _read_keyed_row(rows, _find_marker(rows, "Zero position[nm]"))

    start_i = _find_marker(rows, "Sampling Start time")
    end_i = _find_marker(rows, "Sampling End time")
    start_s = (_clock_to_s(rows[start_i][1]) if start_i is not None
               and len(rows[start_i]) > 1 else None)
    end_s = (_clock_to_s(rows[end_i][1]) if end_i is not None
             and len(rows[end_i]) > 1 else None)

    data_i = _find_marker(rows, "Sampling Data[nm]")
    if data_i is None or data_i + 1 >= len(rows):
        raise ValueError(f"{p}: 'Sampling Data[nm]' block not found")
    labels = _header_labels(rows[data_i + 1])           # col A = Time[ms]
    times: list[float] = []
    cols: dict[str, list[float]] = {lab: [] for lab in labels}
    for r in rows[data_i + 2:]:
        t0 = _num(r[0]) if r else float("nan")
        if t0 != t0:                                    # blank/non-numeric -> end
            break
        times.append(t0)
        for lab, j in labels.items():
            cols[lab].append(_num(r[j]) if j < len(r) else float("nan"))
    if not times:
        raise ValueError(f"{p}: no sampling rows under 'Sampling Data[nm]'")

    return ParsedCsv(
        time_ms=np.asarray(times, dtype=float),
        data_nm={lab: np.asarray(v, dtype=float) for lab, v in cols.items()},
        zero_nm={k: v for k, v in zero.items()},
        detect_ms={k: v for k, v in detect.items() if v == v},
        sampling_start_s=start_s, sampling_end_s=end_s, source=str(p.name))


def csv_to_raw_dict(path: str | Path, time_key: str = "time"
                    ) -> dict[str, np.ndarray]:
    """Tool CSV -> the adapter's raw dict: `{time_key: t (s), w_<label>: (m)}`.

    A sensor is emitted only if its column has ANY finite sample (absent
    sensors read all-NaN in a run). w = (sampling_nm - zero_nm) * 1e-9; a NaN
    zero is treated as 0. The keys are `w_<label>` (e.g. w_XM), matching the
    channel keys in configs/real_exp_n6.yaml."""
    parsed = read_csv(path)
    out: dict[str, np.ndarray] = {time_key: parsed.time_ms * MS_TO_S}
    for label, raw in parsed.data_nm.items():
        if not np.any(np.isfinite(raw)):
            continue
        zero = parsed.zero_nm.get(label, 0.0)
        if zero != zero:                                # NaN zero -> 0
            zero = 0.0
        out[f"w_{label}"] = (raw - zero) * NM_TO_M
    return out
