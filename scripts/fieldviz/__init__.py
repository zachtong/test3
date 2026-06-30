"""Shared rendering helpers for 3D wafer-bonding visualisations.

Centralises the geometry conventions (D2 mirror to full disk, circular
wafer mask), colour scales (RdBu_r symmetric for signed displacement),
provenance footer, and the bonded-mask computation that several viz
scripts need to share. Ported from the 2D wafer_app's
render/views.py::compute_bonded_mask so the bonded-front semantics
match across the 2D and 3D pipelines.

Conventions baked in here:
  - Quarter-disk canonical grid lives in x, y in [0, 1]; mirror across
    BOTH axes to reach the full disk (NOT C4 rotation -- the data has
    D2 dihedral symmetry).
  - Off-disk cells outside the unit circle are masked NaN before
    rendering so they show as the matplotlib 'bad' colour (transparent
    by default) -- this is a display convention only; the loader
    keeps them zero in the training data.
  - Displacement is signed and has a meaningful zero (the unbonded
    state); colour scale uses RdBu_r centred on 0 with vmax =
    abs(field).max() so the cmap stays interpretable across runs.
"""

from __future__ import annotations
from typing import Iterable
import datetime as _dt
import hashlib as _hashlib
import json
import os
from pathlib import Path

import numpy as np


from scripts.fieldviz.palette import (                       # noqa: E402
    WAFER_CMAP, SENSOR_PALETTE, SENSOR_MARKER_COLOR,
    wafer_cmap_to_plotly, sensor_color,
)


__all__ = [
    "mirror_d2", "render_full_disk",
    "shared_diverging_cmap", "wafer_value_range",
    "provenance_footer", "compute_bonded_mask",
    "front_radius_per_t", "FRONT_BONDED_FRAC_DEFAULT",
    "WAFER_CMAP", "SENSOR_PALETTE", "SENSOR_MARKER_COLOR",
    "wafer_cmap_to_plotly", "sensor_color",
]


# Default gap threshold matches the 2D wafer_app default (1 um).
GAP_THRESHOLD_UM_DEFAULT = 1.0
WELL_ABOVE_FACTOR_DEFAULT = 10.0
# When deriving a per-t scalar front from a 3D bonded mask, the front
# radius is the radius at which the bonded fraction in the unbonded
# annulus first drops below this value. Robust against speckle.
FRONT_BONDED_FRAC_DEFAULT = 0.5


def mirror_d2(quarter: np.ndarray) -> np.ndarray:
    """Reflect a quarter-disk field (first quadrant) across BOTH axes.

    Input (Nx, Ny[, ...]) sampled on x in [0, x_end], y in [0, y_end].
    Output (2*Nx-1, 2*Ny-1[, ...]) sampled on x in [-x_end, x_end],
    y in [-y_end, y_end]. The shared axis row / column is NOT duplicated
    so adjacent reflections meet seamlessly.

    Works for any trailing dims (e.g. (Nx, Ny, Nt) field stacks);
    leading two axes are the spatial ones that get mirrored.
    """
    if quarter.ndim < 2:
        raise ValueError(f"need at least 2D input, got shape {quarter.shape}")
    # mirror along y first (axis=1), excluding the shared y=0 row.
    half_xy = np.concatenate(
        [quarter[:, :0:-1], quarter], axis=1)
    # then along x (axis=0), excluding the shared x=0 column.
    full = np.concatenate(
        [half_xy[:0:-1, :], half_xy], axis=0)
    return full


def _full_disk_axes(x_canon: np.ndarray, y_canon: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Mirror-extend the canonical axes for use with imshow extent."""
    x_full = np.concatenate([-x_canon[:0:-1], x_canon])
    y_full = np.concatenate([-y_canon[:0:-1], y_canon])
    return x_full, y_full


def render_full_disk(ax, field: np.ndarray, x_canon: np.ndarray,
                     y_canon: np.ndarray, *,
                     cmap=None, vmin: float | None = None,
                     vmax: float | None = None,
                     mirror: bool = True, mask_off_disk: bool = True,
                     r_end: float = 1.0,
                     sensor_xy: np.ndarray | None = None,
                     sensor_kwargs: dict | None = None):
    """Render one (Nx, Ny) frame onto `ax` with shared display conventions.

    Mirrors quarter -> full disk (if mirror=True), masks cells outside
    the unit circle to NaN so they render transparent (if
    mask_off_disk=True), and optionally overlays sensor positions
    (assumed already in -x_end..x_end coordinates when mirror=True; in
    0..x_end when mirror=False).

    Returns the matplotlib image handle.
    """
    if field.ndim != 2:
        raise ValueError(f"field must be (Nx, Ny), got {field.shape}")
    if mirror:
        F = mirror_d2(field)
        x_axis, y_axis = _full_disk_axes(x_canon, y_canon)
    else:
        F = field.copy()
        x_axis, y_axis = x_canon, y_canon
    if mask_off_disk:
        X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
        off = (X * X + Y * Y) > r_end * r_end
        F = F.astype(np.float64, copy=True)
        F[off] = np.nan
    ext = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    if cmap is None:
        cmap = WAFER_CMAP
    # imshow expects (rows, cols) = (y, x); transpose so x runs horiz.
    im = ax.imshow(F.T, origin="lower", aspect="equal", extent=ext,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    if sensor_xy is not None:
        skw = dict(s=42, marker="x", c=SENSOR_MARKER_COLOR,
                   linewidths=1.8, zorder=5)
        if sensor_kwargs:
            skw.update(sensor_kwargs)
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], **skw)
    return im


def shared_diverging_cmap(field: np.ndarray,
                          symmetric: bool = True,
                          pct_lo: float = 1.0,
                          pct_hi: float = 99.0
                          ) -> tuple[float, float]:
    """Compute vmin / vmax for a diverging colourmap.

    With symmetric=True (the default for signed displacement) vmin = -V,
    vmax = +V, where V is the larger of |percentile(pct_lo)| and
    |percentile(pct_hi)|. Percentile clipping prevents a single off-disk
    outlier from owning the scale; 1-99 is conservative.

    Kept for backward compatibility; new viz should prefer
    `wafer_value_range` paired with `WAFER_CMAP`.
    """
    finite = np.asarray(field)[np.isfinite(field)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(finite, pct_lo))
    hi = float(np.percentile(finite, pct_hi))
    if symmetric:
        v = max(abs(lo), abs(hi))
        return -v, v
    return lo, hi


def wafer_value_range(field: np.ndarray,
                      pct_lo: float = 1.0, pct_hi: float = 99.0,
                      clip_positive_to_zero: bool = True
                      ) -> tuple[float, float]:
    """vmin / vmax for the TEL WAFER_CMAP (sequential purple-to-yellow).

    Wafer-bonding displacement is mostly negative (upper wafer descends
    from rest toward the lower wafer), so the natural range is
    asymmetric. Returns (vmin, vmax) with:

      - vmin = 1st percentile (most-negative tail, clipped to ignore
        a stray off-disk outlier)
      - vmax = 99th percentile; if `clip_positive_to_zero` (default),
        cap vmax at 0 so the colour bar's "0" end stays anchored at
        the rest state (yellow) regardless of small positive numerical
        noise

    Pair with `cmap=WAFER_CMAP`: vmin -> purple (deepest descent),
    vmax -> yellow (rest / unbonded).
    """
    finite = np.asarray(field)[np.isfinite(field)]
    if finite.size == 0:
        return -1.0, 0.0
    vmin = float(np.percentile(finite, pct_lo))
    vmax = float(np.percentile(finite, pct_hi))
    if clip_positive_to_zero and vmax > 0:
        vmax = 0.0
    if vmax <= vmin:
        # degenerate: all values equal. Give a unit range to avoid
        # matplotlib divide-by-zero.
        vmax = vmin + 1.0
    return vmin, vmax


def provenance_footer(fig, *, sim_id: str | None = None,
                      tag: str | None = None,
                      basis_cache_file: str | Path | None = None,
                      results_file: str | Path | None = None,
                      extras: dict | None = None) -> None:
    """Stamp a one-line provenance footer at the bottom of the figure.

    Always includes UTC date + script basename. Optional fields:
    sim_id (basename), tag (training tag), short hashes of
    basis_cache_file and results_file (when present on disk), and any
    extra key=value pairs the caller wants pinned.

    Made MANDATORY across must-have viz so every screenshot in a
    notebook / talk is traceable to the run it came from.
    """
    import sys
    parts = []
    parts.append(_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"))
    parts.append(Path(sys.argv[0]).name if sys.argv else "?")
    if tag:
        parts.append(f"tag={tag}")
    if sim_id:
        parts.append(f"sim={sim_id}")
    if basis_cache_file is not None:
        h = _short_hash(basis_cache_file)
        if h is not None:
            parts.append(f"basis={h}")
    if results_file is not None:
        h = _short_hash(results_file)
        if h is not None:
            parts.append(f"results={h}")
    if extras:
        parts.extend(f"{k}={v}" for k, v in extras.items())
    text = " | ".join(parts)
    fig.text(0.99, 0.005, text, ha="right", va="bottom",
             fontsize=7, color="0.4", family="monospace")


def _short_hash(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        st = p.stat()
        raw = f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()
        return _hashlib.sha256(raw).hexdigest()[:8]
    except OSError:
        return None


def compute_bonded_mask(w_m: np.ndarray, *,
                        gap_threshold_um: float = GAP_THRESHOLD_UM_DEFAULT,
                        well_above_factor: float = WELL_ABOVE_FACTOR_DEFAULT
                        ) -> np.ndarray:
    """Cumulative bonded mask for a 3D upper-wafer displacement field.

    Direct port of the 2D wafer_app/render/views.py::compute_bonded_mask
    extended along the spatial axes from (Nr, Nt) to (Nx, Ny, Nt). The
    lower wafer is approximated by the upper wafer's FINAL frame: this
    rests on the assumption that the sim ends in full bonding (every
    sim in the firehorse2 dataset behaves this way). Time is the last
    axis.

    A cell (x, y, t) is bonded iff all THREE rules hold:

      (1) current gap is below threshold:
            gap_um(x, y, t) = (w(x, y, t) - w(x, y, -1)) * 1e6
            gap_um < gap_threshold_um

      (2) it was previously well-above threshold (i.e. it really did
          start unbonded and shrunk into bonded; rule out cells whose
          gap was always tiny due to numerical noise at the wafer
          edge):
            cumulative-max over t of [gap_um > 10 * gap_threshold_um]

      (3) once bonded, stays bonded -- contact does not break:
            cumulative-max over t of (1) AND (2)

    Returns (Nx, Ny, Nt) bool.
    """
    if w_m.ndim != 3:
        raise ValueError(f"w_m must be (Nx, Ny, Nt), got {w_m.shape}")
    w = np.asarray(w_m, dtype=np.float64)
    gap_um = (w - w[..., -1:]) * 1e6
    below = gap_um < float(gap_threshold_um)
    well_above_ever = np.maximum.accumulate(
        gap_um > float(gap_threshold_um) * float(well_above_factor),
        axis=-1)
    instant_valid = below & well_above_ever
    return np.maximum.accumulate(instant_valid, axis=-1)


def front_radius_per_t(bonded_mask: np.ndarray,
                       x_canon: np.ndarray, y_canon: np.ndarray,
                       *, bonded_frac_threshold: float =
                       FRONT_BONDED_FRAC_DEFAULT) -> np.ndarray:
    """Derive a scalar front radius per timestep from the 3D bonded mask.

    For each t, project the bonded mask onto a radial coordinate (bin
    cells by r), compute the bonded fraction inside each annulus, and
    return the smallest r where the bonded fraction first drops below
    `bonded_frac_threshold`. This is the analogue of the 2D
    `front_radius_gap_mm` but averaged over theta so a single scalar
    summarises the (possibly D2-asymmetric) 3D front.

    Returns (Nt,) float -- NaN where the front does not exist
    (fully unbonded or fully bonded that frame).
    """
    if bonded_mask.ndim != 3:
        raise ValueError(f"bonded_mask must be (Nx, Ny, Nt), got "
                         f"{bonded_mask.shape}")
    X, Y = np.meshgrid(x_canon, y_canon, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    in_disk = R <= 1.0
    # Bin by radius -- 64 annular bins is enough resolution to find the
    # front to within 1/64 of the wafer radius.
    n_bins = 64
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(R.ravel(), edges) - 1, 0, n_bins - 1)
    counts_per_bin = np.bincount(bin_idx[in_disk.ravel()], minlength=n_bins)
    Nt = bonded_mask.shape[-1]
    out = np.full(Nt, np.nan, dtype=np.float64)
    for t in range(Nt):
        b = bonded_mask[..., t].ravel() & in_disk.ravel()
        bonded_per_bin = np.bincount(bin_idx[in_disk.ravel()],
                                     weights=b[in_disk.ravel()].astype(float),
                                     minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(counts_per_bin > 0,
                            bonded_per_bin / np.maximum(counts_per_bin, 1),
                            0.0)
        # Front = smallest r where bonded fraction drops below threshold.
        below = frac < bonded_frac_threshold
        if not below.any():
            # fully bonded at all radii
            continue
        if below.all():
            # fully unbonded
            continue
        i = int(np.argmax(below))   # first True
        out[t] = float(centres[i])
    return out
