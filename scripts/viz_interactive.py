"""Interactive HTML view of one 3D wafer-bonding sim (Plotly).

Writes a standalone HTML file. Open with any browser (Firefox / Chrome
double-click); no server required. Inside the page the operator can:

  - rotate the 3D surface (drag)
  - zoom (scroll)
  - hover any cell to see its (x, y, t-idx, u_z) value
  - advance time via the slider
  - toggle sensors / bonded-region overlay via the legend

The page is one HTML file with embedded JS + the field data; on a 128x128
grid with Nt frames it lands around 10-25 MB depending on the frame
cap. Slows down for very large frame counts -- the --max-frames flag
exists precisely to keep the file size and browser responsiveness in
check.

The display is full-disk: the quarter is mirrored via D2 first so the
surface looks like a physical wafer.

    python scripts/viz_interactive.py --sim /path/to/raw.npz \\
        --out viz/sim_interactive.html
"""

from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.loader import load_dataset                         # noqa: E402
from core.sensors import SensorConfig, place_sensors         # noqa: E402
from scripts.fieldviz import (mirror_d2, compute_bonded_mask,  # noqa: E402
                               front_radius_per_t,
                               wafer_value_range,
                               wafer_cmap_to_plotly)
from scripts.fieldviz.render3d import estimate_lower_z         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", required=True,
                    help="path to a single raw 3D NPZ")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--gap-threshold-um", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=60,
                    help="cap on time-slider frames to keep HTML size "
                    "manageable; evenly subsampled from Nt (default 60)")
    ap.add_argument("--value-scale", type=float, default=1.0e6,
                    help="multiply displacement for display (default 1e6 "
                    "= meters -> micrometers)")
    ap.add_argument("--sensors", default="3-edge",
                    help="'3-edge' (lab rig) or 'r:th,r:th,...'")
    ap.add_argument("--show-lower", action="store_true",
                    help="add a translucent gray reference plane for "
                    "the lower wafer at z = p5(final-frame u_z). "
                    "Toggle via the legend in the rendered HTML.")
    ap.add_argument("--lower-z", type=float, default=None,
                    help="override the auto-estimated lower wafer z "
                    "(in meters; only used with --show-lower)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    try:
        import plotly.graph_objects as go     # noqa: F401
        import plotly.io as pio               # noqa: F401
    except ImportError:
        print("ERROR: plotly is not installed. Install with:",
              file=sys.stderr)
        print("    pip install 'plotly>=5.20'", file=sys.stderr)
        return 1
    import plotly.graph_objects as go

    sim_path = Path(args.sim).expanduser().resolve()
    if not sim_path.is_file():
        print(f"NPZ not found: {sim_path}", file=sys.stderr)
        return 2

    print(f"loading {sim_path.name} ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / sim_path.name
        shutil.copy(sim_path, staged)
        x_canon, y_canon, sims = load_dataset(
            Path(td), nx=args.nx, ny=args.ny, nt=args.nt,
            cache=False, workers=1,
            drop_first_steps=args.drop_first_steps)
    if not sims:
        print("loader rejected this sim via preflight", file=sys.stderr)
        return 1
    sim = sims[0]
    nx, ny, nt = sim.f.shape
    print(f"  loaded {sim.f.shape}", flush=True)

    if args.sensors == "3-edge":
        positions = ((0.95, 0.0), (0.95, 45.0), (0.95, 90.0))
    else:
        positions = tuple(
            tuple(float(x) for x in p.split(":"))
            for p in args.sensors.split(","))
    scfg = SensorConfig(n=len(positions), strategy="custom",
                        positions=positions)
    sensor_xy = place_sensors(scfg)

    # Bonded mask (used for front radius display in the title and as
    # optional contour data; plotly Surface can't natively overlay a
    # 2D contour at a 3D height, so we emit just the front radius in
    # the title text).
    bonded = compute_bonded_mask(
        sim.f.astype(np.float64),
        gap_threshold_um=args.gap_threshold_um)
    front_r = front_radius_per_t(bonded, x_canon, y_canon)

    # Frame subsample
    if nt > args.max_frames:
        frame_idx = np.linspace(0, nt - 1, args.max_frames).astype(int)
    else:
        frame_idx = np.arange(nt)
    print(f"  rendering {len(frame_idx)} of {nt} frames", flush=True)

    # Mirror quarter -> full disk for display
    x_full = np.concatenate([-x_canon[:0:-1], x_canon])
    y_full = np.concatenate([-y_canon[:0:-1], y_canon])
    X_full, Y_full = np.meshgrid(x_full, y_full, indexing="ij")
    R_full = np.sqrt(X_full ** 2 + Y_full ** 2)
    in_disk = R_full <= 1.0

    # Build per-frame z-data (full disk, mask off-disk to NaN so
    # plotly draws nothing there).
    z_frames = []
    for t in frame_idx:
        full = mirror_d2(sim.f[..., t]) * args.value_scale
        full = full.astype(np.float64)
        full[~in_disk] = np.nan
        z_frames.append(full)
    # Asymmetric color range matching the WAFER_CMAP sequential
    # palette (purple=most negative, yellow=zero/rest).
    finite = np.concatenate([zf[np.isfinite(zf)].ravel() for zf in z_frames])
    vmin, vmax = wafer_value_range(finite)
    print(f"  color range [{vmin:.3g}, {vmax:.3g}] (1-99 pct, "
          f"clipped to <= 0 at top)", flush=True)
    wafer_scale = wafer_cmap_to_plotly()

    # Initial trace = first frame
    surface = go.Surface(
        x=X_full, y=Y_full, z=z_frames[0],
        cmin=vmin, cmax=vmax, colorscale=wafer_scale,
        colorbar=dict(title=f"u_z * {args.value_scale:g}", thickness=15),
        name="upper wafer displacement",
        hovertemplate=("x=%{x:.3f}<br>y=%{y:.3f}<br>"
                       "u_z*scale=%{z:.3g}<extra></extra>"),
    )
    # Sensor markers at z = surface value at that (x, y) approximately
    sensor_z0 = []
    for sx, sy in sensor_xy:
        # nearest cell in full grid
        ix = int(np.argmin(np.abs(X_full[:, 0] - sx)))
        iy = int(np.argmin(np.abs(Y_full[0, :] - sy)))
        sensor_z0.append(float(z_frames[0][ix, iy]) if np.isfinite(
            z_frames[0][ix, iy]) else 0.0)
    sensor_trace = go.Scatter3d(
        x=sensor_xy[:, 0], y=sensor_xy[:, 1], z=sensor_z0,
        mode="markers+text",
        marker=dict(size=6, color="red", symbol="x"),
        text=[f"({r:.2g}, {th:g} deg)" for r, th in positions],
        textposition="top center", name="sensors")

    # Animation frames (sensors don't move; just the surface)
    frames = []
    for i, t in enumerate(frame_idx):
        sz = []
        for sx, sy in sensor_xy:
            ix = int(np.argmin(np.abs(X_full[:, 0] - sx)))
            iy = int(np.argmin(np.abs(Y_full[0, :] - sy)))
            sz.append(float(z_frames[i][ix, iy]) if np.isfinite(
                z_frames[i][ix, iy]) else 0.0)
        frames.append(go.Frame(
            data=[
                go.Surface(z=z_frames[i], x=X_full, y=Y_full,
                           cmin=vmin, cmax=vmax, colorscale=wafer_scale,
                           showscale=False),
                go.Scatter3d(x=sensor_xy[:, 0], y=sensor_xy[:, 1], z=sz,
                             mode="markers+text",
                             marker=dict(size=6, color="red", symbol="x"),
                             text=[f"({r:.2g}, {th:g} deg)"
                                   for r, th in positions],
                             textposition="top center"),
            ],
            name=str(int(t)),
            layout=go.Layout(title_text=_title(int(t), nt,
                                               front_r[int(t)],
                                               sim_path.name,
                                               args.gap_threshold_um))))

    # Optional lower-wafer reference plane. Static across frames --
    # we keep it OUT of frames[i].data so it persists during playback.
    fig_data = [surface, sensor_trace]
    if args.show_lower:
        lower_z = (args.lower_z if args.lower_z is not None
                   else estimate_lower_z(sim.f.astype(np.float64)))
        Z_lower = np.full_like(X_full,
                                lower_z * args.value_scale,
                                dtype=np.float64)
        Z_lower[~in_disk] = np.nan
        lower_trace = go.Surface(
            x=X_full, y=Y_full, z=Z_lower,
            surfacecolor=np.zeros_like(Z_lower),
            colorscale=[[0, "rgb(140, 140, 140)"],
                         [1, "rgb(140, 140, 140)"]],
            showscale=False, opacity=0.30,
            name="lower wafer (reference)",
            hovertemplate="lower wafer (z constant)<extra></extra>",
            showlegend=True)
        fig_data.append(lower_trace)
        print(f"  lower wafer plane at z = "
              f"{lower_z * args.value_scale:.3g} "
              f"(units of value_scale)", flush=True)
    fig = go.Figure(data=fig_data, frames=frames)
    fig.update_layout(
        title_text=_title(int(frame_idx[0]), nt,
                          front_r[int(frame_idx[0])],
                          sim_path.name, args.gap_threshold_um),
        scene=dict(
            xaxis_title="x (normalized, r=1 at wafer edge)",
            yaxis_title="y (normalized)",
            zaxis_title=f"u_z * {args.value_scale:g}",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
            xaxis=dict(range=[-1.05, 1.05]),
            yaxis=dict(range=[-1.05, 1.05]),
            zaxis=dict(range=[vmin * 1.1, max(vmax * 1.1, abs(vmin) * 0.1)]),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05, x=0.05,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, currentvalue=dict(prefix="t-idx: "),
            steps=[dict(method="animate", label=str(int(t)),
                        args=[[str(int(t))],
                              dict(frame=dict(duration=0, redraw=True),
                                   mode="immediate")])
                   for t in frame_idx])],
        margin=dict(l=10, r=10, t=60, b=10),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn",
                   full_html=True, auto_play=False)
    print(f"wrote {args.out}", flush=True)
    print(f"open with: xdg-open {args.out}   (Linux)", flush=True)
    return 0


def _title(t_idx, nt, front_r, sim_name, gap_um):
    fr = (f"front_r={front_r:.2f}"
          if np.isfinite(front_r) else "front_r=--")
    return (f"{sim_name}  |  t-idx {t_idx}/{nt - 1}  |  "
            f"{fr}  |  gap_thresh={gap_um:g} um")


if __name__ == "__main__":
    sys.exit(main())
