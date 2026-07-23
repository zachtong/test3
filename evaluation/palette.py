"""Central color palette for wafer visualizations (ported from the 2D
wafer_app so the 3D field views share the same look).

  WAFER_CMAP     -- 5 base colors interpolated to the height colormap used for
                    the wafer displacement in BOTH the top-down and 3D views;
                    low z (deepest descent) -> purple, high z (near 0) -> yellow.
  SENSOR_PALETTE -- brand-style colors for sensor markers; sensor_color(i)
                    darkens per cycle so any count stays unique.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


def _rgb(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


SENSOR_PALETTE: tuple[tuple[float, float, float], ...] = (
    _rgb(0, 169, 224),         # cyan
    _rgb(120, 190, 32),        # green
    _rgb(128, 49, 167),        # purple
    _rgb(238, 220, 0),         # yellow
    _rgb(218, 24, 132),        # magenta
    _rgb(0, 178, 169),         # turquoise
    _rgb(225, 106, 19),        # orange
)

# purple -> turquoise -> cyan -> green -> yellow; deepest descent -> purple
_CMAP_BASES = (
    _rgb(128, 49, 167),        # purple
    _rgb(0, 178, 169),         # turquoise
    _rgb(0, 169, 224),         # cyan
    _rgb(120, 190, 32),        # green
    _rgb(238, 220, 0),         # yellow
)

WAFER_CMAP = LinearSegmentedColormap.from_list("wafer_app", _CMAP_BASES, N=256)


def sensor_color(index: int) -> tuple[float, float, float]:
    """Unique color per sensor index; palette repeats but darkens each cycle."""
    base = SENSOR_PALETTE[index % len(SENSOR_PALETTE)]
    factor = max(0.25, 1.0 - 0.35 * (index // len(SENSOR_PALETTE)))
    return tuple(c * factor for c in base)
