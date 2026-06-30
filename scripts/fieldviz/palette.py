"""TEL brand palette ported 1:1 from the 2D wafer_app GUI.

Two coordinated palettes (identical to
wafer_app/render/palette.py in the 2D project so the 2D and 3D
projects stay visually consistent across talks and papers):

  WAFER_CMAP -- sequential colormap built from five base colors,
    purple -> turquoise -> cyan -> green -> yellow. Use for the upper-
    wafer displacement field: low displacement (most descended, i.e.
    most negative) maps to purple, near-zero (rest / unbonded) maps
    to yellow. Despite the field being signed, the practical range is
    one-sided (displacement is mostly negative as bonding proceeds)
    so a sequential cmap reads cleaner than a diverging one.

  SENSOR_PALETTE -- 7 brand colors for sensor traces and overlay
    markers. Index 0 = cyan, 4 = magenta (the latter is the safe
    choice for sensor markers on WAFER_CMAP since it never appears in
    the cmap itself and therefore stands out at every value).
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


def _rgb(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


SENSOR_PALETTE: tuple[tuple[float, float, float], ...] = (
    _rgb(0, 169, 224),        # 0 cyan      #00A9E0
    _rgb(120, 190, 32),       # 1 green     #78BE20
    _rgb(128, 49, 167),       # 2 purple    #8031A7
    _rgb(238, 220, 0),        # 3 yellow    #EEDC00
    _rgb(218, 24, 132),       # 4 magenta   #DA1884  <-- sensor marker default
    _rgb(0, 178, 169),        # 5 turquoise #00B2A9
    _rgb(225, 106, 19),       # 6 orange    #E16A13
)

_WAFER_CMAP_BASES = (
    _rgb(128, 49, 167),       # purple    deepest descent
    _rgb(0, 178, 169),        # turquoise
    _rgb(0, 169, 224),        # cyan
    _rgb(120, 190, 32),       # green
    _rgb(238, 220, 0),        # yellow    near zero (rest)
)

WAFER_CMAP = LinearSegmentedColormap.from_list(
    "wafer_app", _WAFER_CMAP_BASES, N=256)

# Default colour for sensor markers (magenta -- not in WAFER_CMAP so
# it stays visible against any value).
SENSOR_MARKER_COLOR = SENSOR_PALETTE[4]


def wafer_cmap_to_plotly() -> list[list]:
    """Convert WAFER_CMAP to the Plotly colorscale format.

    Plotly colorscales are a list of [position, 'rgb(R,G,B)'] pairs;
    matplotlib LinearSegmentedColormap exposes 256 anchor points. We
    subsample evenly to 16 anchors which is enough for visual fidelity
    and keeps the HTML file lean.
    """
    n = 16
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, _ = WAFER_CMAP(t)
        out.append([float(t),
                    f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"])
    return out


def sensor_color(index: int) -> tuple[float, float, float]:
    """Cyclic colour from SENSOR_PALETTE; darkens after one full cycle
    so traces stay distinguishable past 7 sensors."""
    base = SENSOR_PALETTE[index % len(SENSOR_PALETTE)]
    cycle = index // len(SENSOR_PALETTE)
    factor = max(0.25, 1.0 - 0.35 * cycle)
    return tuple(c * factor for c in base)
