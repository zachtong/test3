"""Radial-slice kymograph trio at theta = 0 / 45 / 90 deg.

This IS the 3D kymograph answer. Each panel is a 2D kymograph
identical in shape to the 2D project's hero figure: x-axis = canonical
time, y-axis = radius r in [0, 1], color = upper-wafer displacement.
Three panels stacked vertically, sharing a single symmetric RdBu_r
color scale across the sim. Sensor radii (if they lie on one of the
plotted angles) get a horizontal tick on the corresponding panel.
Bonded front radius is overlaid as a thin curve on every panel.

Why this layout: under D2 symmetry the three rays carry independent
information about cos(m * theta) content up to m = 4 (theta = 45 deg
is a node for m = 2, 6, 10; theta = 0 / 90 are extrema). Together they
give the operator a sense of azimuthal anisotropy that a single
kymograph (or an azimuthal average) hides.

Quarter-only: each ray lives in the first quadrant by construction.
No mirror.

    python scripts/viz_radial_kymograph.py --sim /path/to/raw_sim.npz \\
        --out viz/kymo_trio.png
    python scripts/viz_radial_kymograph.py --sim /path/to/raw_sim.npz \\
        --angles 0,30,60,90 --out viz/kymo_quad.png
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

from data.loader import load_dataset                         # noqa: E402
from core.simulation import Simulation                        # noqa: E402
from scripts.fieldviz import (wafer_value_range,             # noqa: E402
                               provenance_footer,
                               compute_bonded_mask,
                               front_radius_per_t,
                               WAFER_CMAP, SENSOR_PALETTE)


# Bonded-front line color. Orange (SENSOR_PALETTE[6] = #E16A13) is
# distinct from every value in WAFER_CMAP so the front stays visible
# from yellow (near-zero) all the way to purple (deepest descent).
_FRONT_COLOR = SENSOR_PALETTE[6]


def _sample_radial_kymograph(f: np.ndarray, x_canon: np.ndarray,
                             y_canon: np.ndarray, theta_deg: float,
                             n_r: int = 128,
                             r_max: float = 1.0,
                             r_disk: float = 1.0) -> np.ndarray:
    """Bilinear-sample f along the radial ray at theta_deg, mask-aware.

    Returns (n_r, Nt) -- value of f at r in [0, r_max] (n_r samples)
    and every canonical time step.

    Mask-aware detail: the loader zeroes canonical cells with
    x^2 + y^2 > r_disk^2 (physical off-disk). A raw bilinear
    stencil at a query near the arc has 4 corners on the Cartesian
    grid; at oblique angles (worst case theta=45 deg) one or more
    of those corners fall at r > r_disk and hold value 0, dragging
    the sampled value toward 0 and producing a false kink. The fix
    here: zero the weight of any off-disk corner and renormalize
    surviving weights so they sum to 1. When all 4 corners are
    off-disk the sample stays 0.

    theta=0 / theta=90 rays never hit this because their stencil
    slides along an axis and does not cross the arc; only oblique
    angles show the artifact, which is exactly what we observed
    with the raw bilinear implementation.
    """
    nx, ny, nt = f.shape
    rs = np.linspace(0.0, r_max, n_r)
    t = np.deg2rad(theta_deg)
    xs = rs * np.cos(t)
    ys = rs * np.sin(t)
    # Convert (xs, ys) in canonical [0, 1] to grid index space.
    ix = np.interp(xs, x_canon, np.arange(nx))   # fractional indices
    iy = np.interp(ys, y_canon, np.arange(ny))
    # Vectorised bilinear sample across all (n_r) rays, all (Nt) times.
    ix0 = np.clip(np.floor(ix).astype(int), 0, nx - 2)
    iy0 = np.clip(np.floor(iy).astype(int), 0, ny - 2)
    dx = ix - ix0
    dy = iy - iy0
    # f shape (Nx, Ny, Nt); gather four corners along (ix, iy) at once
    a00 = f[ix0, iy0, :]                                     # (n_r, Nt)
    a10 = f[ix0 + 1, iy0, :]
    a01 = f[ix0, iy0 + 1, :]
    a11 = f[ix0 + 1, iy0 + 1, :]
    # Raw bilinear weights (per-query, broadcast to Nt).
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy
    # Disk mask per stencil corner. Corners at r > r_disk hold value
    # 0 (loader-zeroed) and must not enter the interpolation.
    r_disk_sq = r_disk * r_disk
    m00 = (x_canon[ix0] ** 2 + y_canon[iy0] ** 2) <= r_disk_sq
    m10 = (x_canon[ix0 + 1] ** 2 + y_canon[iy0] ** 2) <= r_disk_sq
    m01 = (x_canon[ix0] ** 2 + y_canon[iy0 + 1] ** 2) <= r_disk_sq
    m11 = (x_canon[ix0 + 1] ** 2 + y_canon[iy0 + 1] ** 2) <= r_disk_sq
    w00 = w00 * m00.astype(np.float64)
    w10 = w10 * m10.astype(np.float64)
    w01 = w01 * m01.astype(np.float64)
    w11 = w11 * m11.astype(np.float64)
    w_sum = w00 + w10 + w01 + w11                            # (n_r,)
    all_off = w_sum <= 0
    w_sum_safe = np.where(all_off, 1.0, w_sum)
    result = (w00[:, None] * a00 + w10[:, None] * a10
              + w01[:, None] * a01 + w11[:, None] * a11
              ) / w_sum_safe[:, None]
    if all_off.any():
        result[all_off, :] = 0.0
    return result


def render_radial_kymograph(sim: Simulation, x_canon: np.ndarray,
                            y_canon: np.ndarray,
                            out_path: Path | str, *,
                            angles: list[float] | None = None,
                            gap_threshold_um: float = 1.0,
                            value_scale: float = 1.0e6,
                            sensor_rs: list[float] | None = None,
                            sensor_thetas: list[float] | None = None,
                            sim_id: str | None = None,
                            tag: str | None = None,
                            drop_first_steps: int | None = None) -> Path:
    """Render the radial-kymograph trio in-process.

    Pure rendering -- caller supplies the loaded sim. Used by
    scripts/viz_all.py to share one loader pass across multiple viz
    of the same sim.
    """
    if angles is None:
        # Include 22.5 and 67.5 -- these fall BETWEEN the physical
        # sensor angles (0, 45, 90) so the reconstruction is
        # sampled at rays no sensor sits on. Without them the
        # kymograph looks flatteringly good because every ray
        # coincides with a sensor position. 5 panels stack a bit
        # taller but the honesty is worth it.
        angles = [0.0, 22.5, 45.0, 67.5, 90.0]
    if sensor_rs is None:
        sensor_rs = [1.0, 1.0, 1.0]
    if sensor_thetas is None:
        sensor_thetas = [0.0, 45.0, 90.0]
    nx, ny, nt = sim.f.shape
    n_r = nx

    kymos = [_sample_radial_kymograph(
        sim.f.astype(np.float64), x_canon, y_canon, th, n_r=n_r)
        for th in angles]
    scaled = [k * value_scale for k in kymos]
    all_scaled = np.concatenate([k.ravel() for k in scaled])
    vmin, vmax = wafer_value_range(all_scaled)

    bonded = compute_bonded_mask(sim.f.astype(np.float64),
                                  gap_threshold_um=gap_threshold_um)
    front_r = front_radius_per_t(bonded, x_canon, y_canon)

    sensor_by_th: dict = {}
    for r, t in zip(sensor_rs, sensor_thetas):
        sensor_by_th.setdefault(round(float(t), 3), []).append(float(r))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(angles), 1,
                             figsize=(11, 2.7 * len(angles)),
                             sharex=True, constrained_layout=True)
    if len(angles) == 1:
        axes = [axes]
    t_axis = np.linspace(0.0, 1.0, nt)
    r_axis = np.linspace(0.0, 1.0, n_r)
    im = None
    for ax, th, ky in zip(axes, angles, scaled):
        im = ax.imshow(ky, origin="lower", aspect="auto",
                       extent=[t_axis[0], t_axis[-1],
                               r_axis[0], r_axis[-1]],
                       vmin=vmin, vmax=vmax, cmap=WAFER_CMAP,
                       interpolation="nearest")
        ax.set_ylabel("r (normalized)")
        ax.set_title(f"theta = {th:g} deg", fontsize=10)
        ax.plot(t_axis, front_r, color=_FRONT_COLOR, lw=1.4,
                label="bonded front (3D mean)")
        for r in sensor_by_th.get(round(float(th), 3), []):
            ax.axhline(r, color="black", lw=0.7, ls="--", alpha=0.6)
            ax.text(t_axis[-1], r, f" sensor r={r:.2g}",
                    va="center", fontsize=7, color="black")
        if ax is axes[0]:
            ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("normalized time")
    fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                 label=f"u_z * {value_scale:g}")
    fig.suptitle(f"{sim_id or 'sim'}  |  radial-slice kymograph trio  |  "
                 f"per-sim shared color scale", fontsize=11)
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      extras={"drop": drop_first_steps,
                              "gap_um": gap_threshold_um,
                              "angles": ",".join(str(a) for a in angles)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)
    return Path(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", required=True,
                    help="path to a single raw 3D NPZ")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--angles", default="0,22.5,45,67.5,90",
                    help="comma-separated theta values in deg "
                    "(default: 0,22.5,45,67.5,90 -- sensor angles "
                    "0/45/90 plus midway rays 22.5/67.5 where NO "
                    "sensor sits, so the sampling is honest)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--gap-threshold-um", type=float, default=1.0)
    ap.add_argument("--value-scale", type=float, default=1.0e6,
                    help="multiply displacement by this for display "
                    "(default 1e6 = meters -> micrometers)")
    ap.add_argument("--sensor-radii", default="1.0,1.0,1.0",
                    help="comma list of r values; if a sensor's theta "
                    "matches one of --angles, its r is marked on that "
                    "panel. Default 1.0,1.0,1.0 matches lab rig.")
    ap.add_argument("--sensor-thetas", default="0,45,90",
                    help="comma list of theta values for the sensor "
                    "radii (paired with --sensor-radii)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    sim_path = Path(args.sim).expanduser().resolve()
    if not sim_path.is_file():
        print(f"NPZ not found: {sim_path}", file=sys.stderr)
        return 2

    print(f"loading {sim_path.name} via loader (drop_first_steps="
          f"{args.drop_first_steps}) ...", flush=True)
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
    print(f"  loaded {sim.f.shape}", flush=True)

    render_radial_kymograph(
        sim, x_canon, y_canon, args.out,
        angles=angles,
        gap_threshold_um=args.gap_threshold_um,
        value_scale=args.value_scale,
        sensor_rs=[float(r) for r in args.sensor_radii.split(",")],
        sensor_thetas=[float(t) for t in args.sensor_thetas.split(",")],
        sim_id=sim_path.stem, tag=args.tag,
        drop_first_steps=args.drop_first_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
