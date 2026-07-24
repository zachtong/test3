"""Bonding-front detection for a reconstructed 3D displacement field.

Ported from the 2D GUI (wafer_app/render/views.py: `compute_bonded_mask` and
`front_radius_gap_mm`) and generalized off the axisymmetric radius onto the full
(x, y) grid.

The idea, unchanged from 2D: track the GAP between the upper wafer and its final
(bonded) position,

    gap(x, y, t) = w(x, y, t) - w(x, y, t_final)     (>= 0 while still above),

and call a cell "bonded" once the gap has shrunk below `gap_threshold_m` AFTER
having been clearly open (> `well_above_factor` x threshold) at some earlier
frame -- the hysteresis guard that stops noise near t=0 (already at the final
value without ever opening) from registering as bonded. Bonding is cumulative:
once bonded, always bonded, so the front never retreats.

The bonding FRONT at frame j is the outer boundary of the bonded region. For an
axisymmetric field this is a circle; for a lopsided field it is a deformed
closed curve. `front_radii` returns, per azimuth, the radius of that boundary
(the OUTERMOST bonded/unbonded transition along the ray -- the same rule the 2D
`front_radius_gap_mm` used), so the caller can draw it as a red ring in both the
top-down and 3D views.
"""
from __future__ import annotations

import numpy as np

GAP_THRESHOLD_M = 1.0e-7          # 0.1 um -- the 2D GUI default gap threshold
WELL_ABOVE_FACTOR = 10.0          # gap must once exceed 10x threshold (open)


def bonded_mask(w: np.ndarray, gap_threshold_m: float = GAP_THRESHOLD_M,
                well_above_factor: float = WELL_ABOVE_FACTOR) -> np.ndarray:
    """Cumulative bonded mask, same shape as `w` (..., Nt), True where the upper
    wafer is considered bonded to the lower (at its final position).

    `w` is the (downward, negative) displacement in metres; the last time index
    is taken as the final/bonded state. The three 2D conditions are preserved:
    (1) the gap is below threshold, (2) the gap was clearly open earlier, and
    (3) the bonded region only grows (never retreats)."""
    w = np.asarray(w, dtype=float)
    gap = w - w[..., -1:]                       # >= 0 while still above final
    below = gap < float(gap_threshold_m)
    well_above = np.maximum.accumulate(
        gap > float(gap_threshold_m) * float(well_above_factor), axis=-1)
    valid = below & well_above
    return np.maximum.accumulate(valid, axis=-1)


def _nearest_idx(coord: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Nearest index into a regular ascending grid `coord` for each query `q`."""
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return np.zeros(np.shape(q), dtype=int)
    step = (coord[-1] - coord[0]) / (coord.size - 1)
    i = np.round((np.asarray(q, dtype=float) - coord[0]) / step)
    return np.clip(i, 0, coord.size - 1).astype(int)


def front_radii(mask_j: np.ndarray, xf: np.ndarray, yf: np.ndarray,
                thetas: np.ndarray, n_r: int = 200,
                r_max: float = 1.0) -> np.ndarray:
    """Front radius per azimuth at one frame.

    `mask_j` is the bonded mask (Mx, My) on the regular grid (`xf`, `yf`, in
    canonical x/R, y/R units). For each angle in `thetas` a ray is marched
    outward and the OUTERMOST bonded<->unbonded transition (restricted to inside
    the unit disk) is taken as the front radius. NaN where the ray is fully open
    or fully bonded (no visible front). Returns an array shaped like `thetas`."""
    m = np.asarray(mask_j, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    rs = np.linspace(0.0, float(r_max), int(n_r))
    out = np.full(thetas.shape, np.nan)
    for k, th in enumerate(thetas):
        px, py = rs * np.cos(th), rs * np.sin(th)
        inside = (px * px + py * py) <= 1.0
        ii = np.where(inside)[0]
        if ii.size < 2:
            continue
        bonded = m[_nearest_idx(xf, px[ii]), _nearest_idx(yf, py[ii])] >= 0.5
        if not bonded.any() or bonded.all():        # fully open / fully bonded
            continue
        cross = np.where(np.diff(bonded.astype(np.int8)) != 0)[0]
        if not cross.size:
            continue
        i = int(cross[-1])                           # outermost transition
        out[k] = 0.5 * (rs[ii][i] + rs[ii][i + 1])
    return out


def front_xy(mask_j: np.ndarray, xf: np.ndarray, yf: np.ndarray,
             thetas: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Front ring as (x, y) polylines in canonical units; NaN gaps break the
    line where no front exists at that azimuth."""
    r = front_radii(mask_j, xf, yf, thetas, **kw)
    return r * np.cos(thetas), r * np.sin(thetas)


def sample_nearest(field2d: np.ndarray, xf: np.ndarray, yf: np.ndarray,
                   px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of a (Mx, My) field at query points (px, py).
    NaN queries pass through as NaN (so a broken front ring stays broken)."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    good = np.isfinite(px) & np.isfinite(py)
    out = np.full(px.shape, np.nan)
    if good.any():
        out[good] = np.asarray(field2d, dtype=float)[
            _nearest_idx(xf, px[good]), _nearest_idx(yf, py[good])]
    return out
