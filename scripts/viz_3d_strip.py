"""3-snapshot 3D strip of the wafer-bonding process (paper figure).

Renders a single PNG with three 3D-surface panels side-by-side at
canonical timesteps t=0 / t=mid / t=final. All three share the same
camera angle, colour scale, lower-wafer plane, and box aspect, so the
operator can read the front sweep, the descent depth, and the bonded-
plateau formation at a glance.

Default look:
  - Full disk via D2 mirror
  - Per-sim locked WAFER_CMAP scale (cross-frame amplitude comparable)
  - Isometric camera (elev=28, azim=-60)
  - Lower wafer plane OFF by default (it dominates the figure if on);
    add --show-lower for talks where the gap needs to be obvious

    python scripts/viz_3d_strip.py --sim /path/to/raw.npz \\
        --out viz/sim_3d_strip.png

    # paper figure with lower-wafer reference
    ... --show-lower --out viz/sim_3d_strip_dual.png
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
from scripts.fieldviz import (wafer_value_range, WAFER_CMAP,    # noqa: E402
                               provenance_footer)
from scripts.fieldviz.render3d import (render_3d_frame,         # noqa: E402
                                        estimate_lower_z,
                                        DEFAULT_ELEV, DEFAULT_AZIM)


def render_3d_strip(sim: Simulation, x_canon: np.ndarray,
                    y_canon: np.ndarray,
                    sensor_xy: np.ndarray,
                    out_path: Path | str, *,
                    show_lower: bool = False,
                    lower_z: float | None = None,
                    value_scale: float = 1.0e6,
                    elev: float = DEFAULT_ELEV,
                    azim: float = DEFAULT_AZIM,
                    n_panels: int = 3,
                    sim_id: str | None = None,
                    tag: str | None = None,
                    drop_first_steps: int | None = None) -> Path:
    """Render the static 3-snapshot strip. Pure rendering -- caller
    supplies the loaded sim."""
    nx, ny, nt = sim.f.shape
    if n_panels < 2:
        raise ValueError(f"n_panels must be >= 2, got {n_panels}")
    vmin, vmax = wafer_value_range(sim.f)
    if show_lower and lower_z is None:
        lower_z = estimate_lower_z(sim.f.astype(np.float64))

    snapshot_idx = np.linspace(0, nt - 1, n_panels).astype(int)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(4.0 * n_panels, 4.5),
                      constrained_layout=False)
    zmin = vmin * value_scale
    zmax = max(0.0, vmax * value_scale) + 0.05 * abs(zmin)
    if show_lower and lower_z is not None:
        zmin = min(zmin, lower_z * value_scale)

    for i, t_idx in enumerate(snapshot_idx):
        ax = fig.add_subplot(1, n_panels, i + 1, projection="3d")
        render_3d_frame(
            ax, sim.f[..., t_idx], x_canon, y_canon, vmin, vmax,
            value_scale=value_scale,
            show_lower=show_lower, lower_z=lower_z,
            sensor_xy=sensor_xy,
            elev=elev, azim=azim)
        ax.set_zlim(zmin, zmax)
        ax.set_title(f"t-idx {int(t_idx)}/{nt - 1}", fontsize=10)

    fig.suptitle(
        f"{sim_id or 'sim'}  |  3D wafer bonding -- "
        f"{n_panels} snapshots  |  per-sim cmap",
        fontsize=11)

    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin * value_scale,
                           vmax=vmax * value_scale),
        cmap=WAFER_CMAP)
    sm.set_array([])
    # Shared colorbar at the bottom; raised so its label clears the
    # provenance footer (which sits at y~0.005).
    cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.025])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                  label=f"u_z (x{value_scale:g})")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.22,
                          wspace=0.05)
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      extras={"drop": drop_first_steps,
                              "elev": f"{elev:.0f}",
                              "azim": f"{azim:.0f}",
                              "lower": "y" if show_lower else "n",
                              "n": n_panels})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
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
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    ap.add_argument("--show-lower", action="store_true",
                    help="draw a translucent flat reference plane for "
                    "the lower wafer at z = p5(final-frame u_z); "
                    "useful for talks where the gap needs to be "
                    "physically obvious")
    ap.add_argument("--lower-z", type=float, default=None,
                    help="override the auto-estimated lower wafer z "
                    "(in metres; only used with --show-lower)")
    ap.add_argument("--n-panels", type=int, default=3,
                    help="number of time snapshots (default 3)")
    ap.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    ap.add_argument("--azim", type=float, default=DEFAULT_AZIM)
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

    render_3d_strip(
        sim, x_canon, y_canon, sensor_xy, args.out,
        show_lower=args.show_lower, lower_z=args.lower_z,
        value_scale=args.value_scale,
        n_panels=args.n_panels,
        elev=args.elev, azim=args.azim,
        sim_id=sim_id, tag=args.tag,
        drop_first_steps=args.drop_first_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
