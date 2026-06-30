"""3D animated GIF of the wafer-bonding process.

mpl_toolkits.mplot3d surface that morphs through time. Upper wafer
descends from rest (z=0, yellow) toward the bonded plateau (deep
purple) as the front sweeps. Optionally draws a translucent flat
lower wafer reference plane so the gap is visually obvious.

Full disk via D2 mirror, per-sim locked colour scale (so amplitude is
physically comparable across frames), default isometric camera
(elev=28, azim=-60). Output is a GIF -- portable, embeds in PPT /
talks without needing ffmpeg.

    python scripts/viz_3d_gif.py --sim /path/to/raw.npz \\
        --out viz/sim_3d.gif

    # add the flat lower-wafer reference plane
    ... --show-lower

    # camera angle: top-down-ish (elev=80) for a "front sweeping" look
    ... --elev 80 --azim -90
"""
from __future__ import annotations
import argparse
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.loader import load_dataset                            # noqa: E402
from core.sensors import SensorConfig, place_sensors            # noqa: E402
from core.simulation import Simulation                           # noqa: E402
from scripts.fieldviz import (wafer_value_range,                # noqa: E402
                               provenance_footer)
from scripts.fieldviz.render3d import (render_3d_frame,         # noqa: E402
                                        estimate_lower_z,
                                        DEFAULT_ELEV, DEFAULT_AZIM)


def render_3d_gif(sim: Simulation, x_canon: np.ndarray,
                  y_canon: np.ndarray,
                  sensor_xy: np.ndarray,
                  out_path: Path | str, *,
                  show_lower: bool = False,
                  lower_z: float | None = None,
                  value_scale: float = 1.0e6,
                  fps: int = 18,
                  max_frames: int = 60,
                  elev: float = DEFAULT_ELEV,
                  azim: float = DEFAULT_AZIM,
                  sim_id: str | None = None,
                  tag: str | None = None,
                  drop_first_steps: int | None = None) -> Path:
    """Render a 3D-surface animation. Pure rendering -- caller supplies
    the already-loaded sim. Used by scripts/viz_all.py so the loader
    pass is shared across multiple per-sim viz."""
    nx, ny, nt = sim.f.shape
    vmin, vmax = wafer_value_range(sim.f)
    if show_lower and lower_z is None:
        lower_z = estimate_lower_z(sim.f.astype(np.float64))

    if nt > max_frames:
        frame_idx = np.linspace(0, nt - 1, max_frames).astype(int)
    else:
        frame_idx = np.arange(nt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d')

    fig = plt.figure(figsize=(7.5, 6.2), constrained_layout=False)
    ax = fig.add_subplot(111, projection="3d")

    def _draw(t_idx: int) -> dict:
        ax.cla()
        # Lock z-axis range to the per-sim physical extent so the
        # vertical scale does NOT auto-rescale per frame (defeats the
        # purpose of seeing the upper wafer descend through it).
        zmin = vmin * value_scale
        # A small headroom above 0 keeps the rest plane visible.
        zmax = max(0.0, vmax * value_scale) + 0.05 * abs(zmin)
        if show_lower and lower_z is not None:
            zmin = min(zmin, lower_z * value_scale)
        handles = render_3d_frame(
            ax, sim.f[..., t_idx], x_canon, y_canon, vmin, vmax,
            value_scale=value_scale,
            show_lower=show_lower, lower_z=lower_z,
            sensor_xy=sensor_xy,
            elev=elev, azim=azim)
        ax.set_zlim(zmin, zmax)
        ax.set_title(
            f"{sim_id or 'sim'}  |  t-idx {t_idx}/{nt - 1}  |  "
            f"3D bonding viz" + ("  (+ lower)" if show_lower else ""),
            fontsize=10)
        return handles

    _draw(int(frame_idx[0]))
    # Shared colorbar (built once from the initial frame's mappable).
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin * value_scale,
                           vmax=vmax * value_scale),
        cmap=None)
    # cmap is set inside render_3d_frame; we re-fetch via the upper
    # surface's cmap reference for the colorbar. Simpler: import WAFER_CMAP.
    from scripts.fieldviz import WAFER_CMAP
    sm.set_cmap(WAFER_CMAP)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.10,
                  label=f"u_z (x{value_scale:g})")

    def update(i):
        _draw(int(frame_idx[i]))
        return []

    print(f"rendering {len(frame_idx)} 3D frames at {fps} fps -> "
          f"{out_path}", flush=True)
    anim = FuncAnimation(fig, update, frames=len(frame_idx),
                         interval=1000 // fps, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      extras={"drop": drop_first_steps,
                              "elev": f"{elev:.0f}",
                              "azim": f"{azim:.0f}",
                              "lower": "y" if show_lower else "n"})
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)
    return Path(out_path)


def _load_one_canonical(raw_path: Path, nx: int, ny: int, nt: int,
                        drop_first_steps: int):
    if not raw_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {raw_path}")
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / raw_path.name
        shutil.copy(raw_path, staged)
        x_canon, y_canon, sims = load_dataset(
            Path(td), nx=nx, ny=ny, nt=nt,
            cache=False, workers=1,
            drop_first_steps=drop_first_steps)
    if not sims:
        raise RuntimeError(f"loader rejected {raw_path.name}")
    return x_canon, y_canon, sims[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", required=True,
                    help="path to a single raw 3D NPZ")
    ap.add_argument("--out", required=True, help="output GIF path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    ap.add_argument("--show-lower", action="store_true",
                    help="draw a translucent flat reference plane for "
                    "the lower wafer at z = p5(final-frame u_z)")
    ap.add_argument("--lower-z", type=float, default=None,
                    help="override the auto-estimated lower wafer z "
                    "(in metres; only used with --show-lower)")
    ap.add_argument("--elev", type=float, default=DEFAULT_ELEV,
                    help=f"camera elevation in deg (default "
                    f"{DEFAULT_ELEV:.0f})")
    ap.add_argument("--azim", type=float, default=DEFAULT_AZIM,
                    help=f"camera azimuth in deg (default "
                    f"{DEFAULT_AZIM:.0f})")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--max-frames", type=int, default=60,
                    help="cap on rendered frames (each 3D frame is "
                    "~0.3-1 s render). Even subsample from Nt.")
    ap.add_argument("--sensors", default="3-edge")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    raw_path = Path(args.sim).expanduser().resolve()
    x_canon, y_canon, sim = _load_one_canonical(
        raw_path, args.nx, args.ny, args.nt, args.drop_first_steps)
    sim_id = raw_path.stem
    print(f"loaded {sim_id}: canonical {sim.f.shape}", flush=True)

    if args.sensors == "3-edge":
        positions = ((1.0, 0.0), (1.0, 45.0), (1.0, 90.0))
    else:
        positions = tuple(
            tuple(float(x) for x in p.split(":"))
            for p in args.sensors.split(","))
    scfg = SensorConfig(n=len(positions), strategy="custom",
                        positions=positions)
    sensor_xy = place_sensors(scfg)

    render_3d_gif(
        sim, x_canon, y_canon, sensor_xy, args.out,
        show_lower=args.show_lower, lower_z=args.lower_z,
        value_scale=args.value_scale,
        fps=args.fps, max_frames=args.max_frames,
        elev=args.elev, azim=args.azim,
        sim_id=sim_id, tag=args.tag,
        drop_first_steps=args.drop_first_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
