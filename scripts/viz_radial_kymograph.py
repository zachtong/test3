"""Radial-slice kymograph trio at theta = 0 / 45 / 90 deg.

This IS the 3D kymograph answer. Each panel is a 2D kymograph
identical in shape to the 2D project's hero figure: x-axis = canonical
time, y-axis = radius r in [0, 1], colour = upper-wafer displacement.
Three panels stacked vertically, sharing a single symmetric RdBu_r
colour scale across the sim. Sensor radii (if they lie on one of the
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
from scripts.fieldviz import (shared_diverging_cmap,         # noqa: E402
                               provenance_footer,
                               compute_bonded_mask,
                               front_radius_per_t)


def _sample_radial_kymograph(f: np.ndarray, x_canon: np.ndarray,
                             y_canon: np.ndarray, theta_deg: float,
                             n_r: int = 128) -> np.ndarray:
    """Bilinear-sample f along the radial ray at theta_deg.

    Returns (n_r, Nt) -- value of f at r in [0, 1] (n_r samples) and
    every canonical time step. Off-canonical-grid points (the ray may
    not align with grid columns) are interpolated; r values outside the
    canonical extent are clipped to the boundary value via np.interp's
    edge clamp on each axis.
    """
    nx, ny, nt = f.shape
    rs = np.linspace(0.0, 1.0, n_r)
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
    w00 = (1 - dx)[:, None] * (1 - dy)[:, None]
    w10 = dx[:, None] * (1 - dy)[:, None]
    w01 = (1 - dx)[:, None] * dy[:, None]
    w11 = dx[:, None] * dy[:, None]
    return w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", required=True,
                    help="path to a single raw 3D NPZ")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--angles", default="0,45,90",
                    help="comma-separated theta values in deg "
                    "(default: 0,45,90 matching the lab rig)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--gap-threshold-um", type=float, default=1.0)
    ap.add_argument("--value-scale", type=float, default=1.0e6,
                    help="multiply displacement by this for display "
                    "(default 1e6 = metres -> micrometres)")
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
    nx, ny, nt = sim.f.shape
    print(f"  loaded {sim.f.shape}", flush=True)

    # Sample one kymograph per angle
    kymos = []
    n_r = nx
    for th in angles:
        ky = _sample_radial_kymograph(
            sim.f.astype(np.float64), x_canon, y_canon, th, n_r=n_r)
        kymos.append(ky)

    # Shared per-sim symmetric colour scale across all kymos.
    scaled = [k * args.value_scale for k in kymos]
    all_finite = np.concatenate([k[np.isfinite(k)].ravel() for k in scaled])
    if all_finite.size == 0:
        v = 1.0
    else:
        v = float(max(abs(np.percentile(all_finite, 1)),
                      abs(np.percentile(all_finite, 99))))
    if v == 0:
        v = 1.0

    # Bonded front (full 3D), then radius per t for overlay.
    bonded = compute_bonded_mask(
        sim.f.astype(np.float64),
        gap_threshold_um=args.gap_threshold_um)
    front_r = front_radius_per_t(bonded, x_canon, y_canon)

    # Sensor markers: bucket by theta.
    sensor_rs = [float(r) for r in args.sensor_radii.split(",")]
    sensor_ths = [float(t) for t in args.sensor_thetas.split(",")]
    sensor_by_th: dict = {}
    for r, t in zip(sensor_rs, sensor_ths):
        sensor_by_th.setdefault(round(t, 3), []).append(r)

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(angles), 1, figsize=(11, 2.7 * len(angles)),
                             sharex=True, constrained_layout=True)
    if len(angles) == 1:
        axes = [axes]
    t_axis = np.linspace(0.0, 1.0, nt)
    r_axis = np.linspace(0.0, 1.0, n_r)
    for ax, th, ky in zip(axes, angles, scaled):
        im = ax.imshow(ky, origin="lower", aspect="auto",
                       extent=[t_axis[0], t_axis[-1],
                               r_axis[0], r_axis[-1]],
                       vmin=-v, vmax=v, cmap="RdBu_r",
                       interpolation="nearest")
        ax.set_ylabel("r (normalised)")
        ax.set_title(f"theta = {th:g} deg", fontsize=10)
        # Bonding front overlay (the SAME scalar front for every panel
        # -- a property of the global 3D mask, by construction it is
        # azimuthally averaged. Different angles may see the front at
        # slightly different radii in principle, but the scalar overlay
        # is the consistent project-wide measure.)
        ax.plot(t_axis, front_r, color="lime", lw=1.2,
                label="bonded front (3D mean)")
        # Sensor radius markers for this theta.
        for r in sensor_by_th.get(round(th, 3), []):
            ax.axhline(r, color="black", lw=0.7, ls="--", alpha=0.6)
            ax.text(t_axis[-1], r, f" sensor r={r:.2g}",
                    va="center", fontsize=7, color="black")
        if ax is axes[0]:
            ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("normalised time")
    fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                 label=f"u_z * {args.value_scale:g}")
    fig.suptitle(f"{sim_path.stem}  |  radial-slice kymograph trio  |  "
                 f"K=8 displacement (per-sim shared colour scale)",
                 fontsize=11)
    provenance_footer(fig, sim_id=sim_path.name, tag=args.tag,
                      extras={"drop": args.drop_first_steps,
                              "gap_um": args.gap_threshold_um,
                              "angles": args.angles})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
