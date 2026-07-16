"""Adapter: real-experiment sensor NPZ -> model-ready (y, t) for the 3D model.

A real run's NPZ holds a ``time`` axis (s) and one displacement series per
sensor (m), each at a fixed (radius, azimuth). The 3D model is
NON-axisymmetric: unlike the 2D pipeline -- which AVERAGES all same-radius
sensors into one radial channel -- here each (r, theta) sensor is a DISTINCT
input channel. The six real sensors (X/Y/D directions x M/E rings) fold, by the
wafer's mirror symmetry about the x and y axes, onto the model's six
quarter-disk positions (theta in {0, 45, 90} x two radii), matching one-to-one.

Inference needs those series
  (a) truncated to the bonding event [t_start, t_cutoff] -- the static
      post-bonding tail would otherwise distort the per-run time normalization
      s = (t - t0) / (t_end - t0) applied downstream; and
  (b) stacked into an (n, T) matrix whose ROW ORDER matches the trained
      model's input-channel order (bundle["sensor_rtheta"]).

Channels are matched to the model's positions BY (r, theta), never by column
order, so a mislabelled or reordered config can never silently feed a sensor
into the wrong channel. Units are assumed SI already (metres, seconds). Set
sign=-1.0 to flip a downward-negative convention mismatch; zero_baseline to
subtract each channel's first kept sample.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def fold_theta(theta_deg: float) -> float:
    """Fold any azimuth into the quarter window [0, 90] by the wafer's mirror
    symmetry about the x and y axes (the Klein four-group of reflections).

    X = 180 -> 0, Y = 90 -> 90, D = -45 -> 45: exactly the ABCDEF fold. Writing
    the physical azimuth or the folded angle in the config both work.
    """
    t = float(theta_deg) % 180.0            # mirror about origin (180 rotation)
    return 180.0 - t if t > 90.0 else t     # mirror about the y axis


@dataclass(frozen=True)
class Channel:
    """One real sensor: a single NPZ array at a fixed (r, theta).

    NO multi-key averaging -- the 3D model keeps every (r, theta) distinct
    (that is the whole point of going non-axisymmetric).
    """
    key: str
    r_phys: float       # radial position, metres
    theta_deg: float    # azimuth, folded into [0, 90]


@dataclass(frozen=True)
class RealExperimentConfig:
    R: float                          # wafer radius, metres (normalizes r)
    channels: tuple[Channel, ...]     # every sensor present in the experiment
    t_cutoff: float                   # s; drop the post-bonding static tail
    t_start: float = 0.0              # s; drop any pre-bonding lead-in
    sign: float = 1.0                 # multiply displacement (use -1.0 to flip)
    zero_baseline: bool = False       # subtract each channel's first kept sample
    time_key: str = "time"            # name of the time array in the NPZ

    def norm_position(self, ch: Channel) -> float:
        """Normalized radial position in [0, 1] (= r_phys / R)."""
        return ch.r_phys / self.R


def real_config_from_yaml(path: str | Path) -> RealExperimentConfig:
    """Load a 3D real-eval config. Each channel needs key, r_phys, theta_deg;
    theta_deg is folded into [0, 90] on load."""
    with open(path) as fp:
        d = yaml.safe_load(fp)
    if "channels" not in d or not d["channels"]:
        raise ValueError(f"{path}: config must declare at least one channel")
    channels = tuple(
        Channel(key=str(c["key"]), r_phys=float(c["r_phys"]),
                theta_deg=fold_theta(float(c["theta_deg"])))
        for c in d["channels"])
    return RealExperimentConfig(
        R=float(d["R"]), channels=channels,
        t_cutoff=float(d["t_cutoff"]), t_start=float(d.get("t_start", 0.0)),
        sign=float(d.get("sign", 1.0)),
        zero_baseline=bool(d.get("zero_baseline", False)),
        time_key=str(d.get("time_key", "time")))


def load_real_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read a real-experiment NPZ.

    Numeric arrays are returned as float64; non-numeric extras (string labels,
    units, metadata) are kept untouched so they never block loading -- the
    adapter only ever reads the time key and the channel keys. A greedy
    float-cast of EVERY key is what would crash on an unrelated metadata field.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"real-experiment NPZ not found: {p}")
    out: dict[str, np.ndarray] = {}
    with np.load(p, allow_pickle=True) as z:
        for k in z.files:
            a = np.asarray(z[k])
            out[k] = (a.astype(np.float64)
                      if np.issubdtype(a.dtype, np.number) else a)
    return out


def _validate(raw: dict[str, np.ndarray], cfg: RealExperimentConfig) -> None:
    """Fail fast on malformed input at this system boundary."""
    if cfg.time_key not in raw:
        raise KeyError(
            f"time array {cfg.time_key!r} missing; have {sorted(raw)}")
    t = raw[cfg.time_key]
    if t.ndim != 1:
        raise ValueError(f"time must be 1-D, got shape {t.shape}")
    if t.size < 2:
        raise ValueError(f"time has {t.size} samples; need >= 2")
    if not np.all(np.isfinite(t)):
        raise ValueError("time contains non-finite values")
    # tolerate repeated / sub-step-jittered timestamps (deduped in assemble);
    # only a backward jump larger than a normal step is a real ordering error.
    dt = np.diff(t)
    forward = dt[dt > 0]
    typ_dt = float(np.median(forward)) if forward.size else 0.0
    if np.any(dt < -typ_dt):
        raise ValueError("time jumps backward by more than one timestep; "
                         "the series order is suspect")
    for ch in cfg.channels:
        if ch.key not in raw:
            raise KeyError(
                f"channel array {ch.key!r} missing; have {sorted(raw)}")
        w = raw[ch.key]
        if w.shape != t.shape:
            raise ValueError(
                f"channel {ch.key!r} shape {w.shape} != time {t.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"channel {ch.key!r} contains non-finite values")
    if not (t[0] <= cfg.t_start < cfg.t_cutoff <= t[-1]):
        raise ValueError(
            "need time[0] <= t_start < t_cutoff <= time[-1]; got "
            f"time=[{t[0]:.4g}, {t[-1]:.4g}], t_start={cfg.t_start}, "
            f"t_cutoff={cfg.t_cutoff}")


def _match_channel(r_norm: float, theta_deg: float,
                   cfg: RealExperimentConfig, r_tol: float,
                   theta_tol: float) -> Channel:
    """The channel whose (normalized r, folded theta) matches, within tol.

    Matching is per-axis nearest-within-tolerance: |r_norm - r| <= r_tol and
    |theta - theta| <= theta_tol. Errors if nothing matches, or if more than
    one channel matches (ambiguous placement). Radius tolerance absorbs the
    rounding of bundle positions (e.g. 0.127/0.15 = 0.84667 stored as 0.847).
    """
    cands = [(i, ch) for i, ch in enumerate(cfg.channels)
             if abs(cfg.norm_position(ch) - r_norm) <= r_tol
             and abs(ch.theta_deg - theta_deg) <= theta_tol]
    if not cands:
        have = [(round(cfg.norm_position(c), 3), c.theta_deg, c.key)
                for c in cfg.channels]
        raise ValueError(
            f"no channel within tol of (r={r_norm:.4f}, theta={theta_deg:.1f}); "
            f"channels at {have}")
    if len(cands) > 1:
        raise ValueError(
            f"(r={r_norm:.4f}, theta={theta_deg:.1f}) is ambiguous: matches "
            f"{[c.key for _, c in cands]} within tol")
    return cands[0][1]


def assemble_inputs(raw: dict[str, np.ndarray], sensor_rtheta,
                    cfg: RealExperimentConfig, *, r_tol: float = 1e-2,
                    theta_tol: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Build (y, t) for a model whose input channels sit at ``sensor_rtheta``.

    ``sensor_rtheta``: (n, 2) array of (r_norm, theta_deg), in the trained
    model's channel order (i.e. bundle["sensor_rtheta"]). May request a SUBSET
    of the configured channels, as long as each requested position matches
    exactly one channel.

    Returns y (n, T) in metres and t (T,) in seconds, both truncated to
    [t_start, t_cutoff]; row i of y is the series at sensor_rtheta[i]. NO
    averaging -- each row is exactly one sensor.
    """
    _validate(raw, cfg)
    sensor_rtheta = np.asarray(sensor_rtheta, dtype=np.float64).reshape(-1, 2)
    t_full = raw[cfg.time_key]
    mask = (t_full >= cfg.t_start) & (t_full <= cfg.t_cutoff)
    # dedup repeated / sub-step-jittered timestamps so t is strictly increasing
    # (np.unique sorts + keeps the first occurrence); channel rows follow the
    # same indices.
    t, keep = np.unique(t_full[mask], return_index=True)
    if t.size < 2:
        raise ValueError(
            f"only {t.size} distinct samples in [{cfg.t_start}, {cfg.t_cutoff}];"
            " widen the window or check the time units")
    rows = []
    for r_norm, theta_deg in sensor_rtheta:
        ch = _match_channel(float(r_norm), float(theta_deg), cfg,
                            r_tol, theta_tol)
        w = cfg.sign * raw[ch.key][mask][keep]
        if cfg.zero_baseline:
            w = w - w[0]
        rows.append(w)
    return np.stack(rows), t
