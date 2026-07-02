"""Shared matplotlib 3D rendering for wafer-bonding viz.

One function `render_3d_frame` paints one timestep of upper-wafer
displacement onto a `mpl_toolkits.mplot3d.Axes3D`. Optionally also
paints the flat lower wafer (the substrate the upper wafer descends
onto) as a translucent reference plane so the bonding gap is visually
obvious. Used by viz_3d_gif (animation) and viz_3d_strip (static
snapshots) to keep their look and physics consistent.

Conventions baked in here:
  - quarter-disk input -> full disk via D2 mirror (same as topdown
    and interactive)
  - cells outside the unit circle masked to NaN so they render
    transparent
  - color: WAFER_CMAP (TEL palette); same physical scale as the
    other displacement viz
  - camera default: elev=28, azim=-60 -- a 3/4 isometric that shows
    both surface curvature and the bonding-front extent
  - box aspect z is squished (0.40) so the wafer-sized x/y stays
    larger than the micrometer-scale z; otherwise the disk degenerates
    to a thin sliver
"""
from __future__ import annotations
import numpy as np

from scripts.fieldviz import (mirror_d2, WAFER_CMAP,
                               SENSOR_MARKER_COLOR, SENSOR_PALETTE)


DEFAULT_ELEV = 28.0
DEFAULT_AZIM = -60.0
_BOX_ASPECT_Z = 0.40
_LOWER_COLOR = "0.55"   # neutral gray, distinct from WAFER_CMAP
_FRONT_COLOR = SENSOR_PALETTE[6]   # orange, same as the 2D viz


def estimate_lower_z(field: np.ndarray, pct: float = 5.0) -> float:
    """Estimate the lower wafer's constant z-position from a full sim.

    The lower wafer is the substrate that does not move; the upper
    wafer's `u_z` is measured from its rest position (z=0). When a
    cell fully bonds, the upper wafer's `u_z` equals the negative of
    the initial gap at that cell. So a robust estimate of the lower
    wafer's z-position is a low percentile of the final-frame field
    (5th by default -- gets the bonded plateau, not edge noise).
    """
    final = field[..., -1].astype(np.float64)
    finite = final[np.isfinite(final)]
    if finite.size == 0:
        return -1.0e-5
    return float(np.percentile(finite, pct))


def render_3d_frame(ax, field_frame: np.ndarray,
                    x_canon: np.ndarray, y_canon: np.ndarray,
                    vmin: float, vmax: float, *,
                    value_scale: float = 1.0e6,
                    cmap=None,
                    show_lower: bool = False,
                    lower_z: float | None = None,
                    sensor_xy: np.ndarray | None = None,
                    elev: float = DEFAULT_ELEV,
                    azim: float = DEFAULT_AZIM,
                    rcount: int = 64, ccount: int = 64,
                    upper_alpha: float = 0.95,
                    lower_alpha: float = 0.35) -> dict:
    """Render one (Nx, Ny) frame onto a 3D axes; return handles.

    field_frame:  quarter-disk u_z (in meters) at one timestep
    vmin / vmax:  per-sim displacement range (meters, NOT scaled)
    value_scale:  multiplied through to get display units; default 1e6
                  -> micrometers. The z-axis label reflects this.

    Returns a dict {'upper', 'lower', 'sensors'} of mpl artists for
    callers that want to mutate them (e.g. an animation loop).
    """
    if cmap is None:
        cmap = WAFER_CMAP
    F_full = mirror_d2(field_frame.astype(np.float64))
    x_full = np.concatenate([-x_canon[:0:-1], x_canon])
    y_full = np.concatenate([-y_canon[:0:-1], y_canon])
    X, Y = np.meshgrid(x_full, y_full, indexing="ij")
    off_disk = (X * X + Y * Y) > 1.0

    Z_upper = F_full * value_scale
    Z_upper[off_disk] = np.nan
    upper = ax.plot_surface(
        X, Y, Z_upper,
        cmap=cmap,
        vmin=vmin * value_scale,
        vmax=vmax * value_scale,
        rcount=rcount, ccount=ccount,
        linewidth=0, antialiased=True, alpha=upper_alpha)

    lower = None
    if show_lower and lower_z is not None:
        Z_lower = np.full_like(X, lower_z * value_scale)
        Z_lower[off_disk] = np.nan
        lower = ax.plot_surface(
            X, Y, Z_lower, color=_LOWER_COLOR, alpha=lower_alpha,
            rcount=20, ccount=20, linewidth=0, antialiased=False)

    sensors = None
    if sensor_xy is not None and len(sensor_xy):
        # Render sensors as floating markers at the rest plane (z=0
        # in display units) so they stay visible regardless of how
        # deep the surface descends. The wafer rim is at z=0 by
        # convention (upper wafer's rest position).
        sz = np.zeros(len(sensor_xy))
        sensors = ax.scatter(
            sensor_xy[:, 0], sensor_xy[:, 1], sz,
            s=60, marker="x", c=SENSOR_MARKER_COLOR,
            linewidth=2.0, depthshade=False, zorder=10)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(f"u_z (x{value_scale:g})")
    ax.set_box_aspect((1, 1, _BOX_ASPECT_Z))

    return {"upper": upper, "lower": lower, "sensors": sensors}
