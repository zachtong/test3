"""Native COMSOL vs loader canonical along a radial ray.

For ONE sim, overlays:
  A. Raw NPZ native points  (x, y, u_z) at the final time, filtered
     to a small angular band around theta = 0 / 45 / 90 deg.
  B. Loader's canonical u_z (r, t=final) sampled along the same rays.

If A is smooth in r near 1 but B has a kink, the kink is introduced
by the loader (Delaunay hull under-shoot + nearest-fill from
anomalous rim natives). If A itself has the kink, the artifact is
in the COMSOL setup (rim BC, mesh sparsity, physics).

    python scripts/diagnose_rim_kink.py \\
        --sim /data/.../ST_3D_big_firehorse_01129.npz \\
        --out viz/rim_kink_diagnose.png
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

from data.loader import (load_dataset,                        # noqa: E402
                          _DISK_MASK_R_END, _WAFER_RADIUS_M)
from scripts.viz_radial_kymograph import _sample_radial_kymograph  # noqa: E402
from scripts.fieldviz import provenance_footer, SENSOR_PALETTE     # noqa: E402


def _load_raw_native_at_final_time(sim_path: Path
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy_normalized, u_z) for the LAST step's LAST time.

    xy in canonical [0, 1] units (native meters / _WAFER_RADIUS_M).
    u_z in meters (same units the loader produces).
    """
    with np.load(sim_path, allow_pickle=True) as z:
        n_steps = int(z["num_wafer_steps"])
        last = n_steps - 1
        prefix = f"step_{last:04d}"
        coords = np.asarray(z[f"{prefix}_coordinates_upper"])  # (3, N)
        disp = np.asarray(
            z[f"{prefix}_displacement_z_corrected_upper"])     # (Ti, N)
        xy = coords[:2, :].T / _WAFER_RADIUS_M                 # (N, 2)
        u_z_final = disp[-1].astype(np.float64)                # (N,)
    return xy, u_z_final


def _filter_near_ray(xy: np.ndarray, theta_deg: float,
                       tol_deg: float, r_lo: float, r_hi: float
                       ) -> np.ndarray:
    """Boolean mask: True where a native point sits within +/-tol_deg
    of the ray at theta_deg AND with r in [r_lo, r_hi]."""
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    th = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    return ((np.abs(th - theta_deg) < tol_deg)
             & (r >= r_lo) & (r <= r_hi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", required=True,
                    help="path to a raw 3D NPZ")
    ap.add_argument("--out", required=True,
                    help="output PNG path")
    ap.add_argument("--angle-tol", type=float, default=3.0,
                    help="angular band width (+/- deg) around each "
                    "ray to include native points (default 3)")
    ap.add_argument("--r-min", type=float, default=0.80,
                    help="lower r for the plot (default 0.80)")
    ap.add_argument("--r-max", type=float, default=1.02,
                    help="upper r for the plot (default 1.02 -- shows "
                    "a sliver past r=1 to make the mask boundary "
                    "visually obvious)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--value-scale", type=float, default=1.0e6,
                    help="multiply u_z by this for display (default "
                    "1e6 = metres -> micrometres)")
    args = ap.parse_args()

    sim_path = Path(args.sim).expanduser().resolve()
    if not sim_path.is_file():
        print(f"NPZ not found: {sim_path}", file=sys.stderr)
        return 2

    print(f"[A] reading raw native cloud from {sim_path.name} ...",
          flush=True)
    xy_native, u_native = _load_raw_native_at_final_time(sim_path)
    print(f"  {xy_native.shape[0]} native points at final step",
          flush=True)

    print(f"[B] running loader (tempdir + symlink) ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / sim_path.name
        shutil.copy(sim_path, staged)
        x_canon, y_canon, sims = load_dataset(
            Path(td), nx=args.nx, ny=args.ny, nt=args.nt,
            cache=False, workers=1,
            drop_first_steps=args.drop_first_steps)
    if not sims:
        print("loader rejected the sim", file=sys.stderr)
        return 1
    f_canon = sims[0].f
    print(f"  canonical shape {f_canon.shape}", flush=True)

    # Sample canonical along each ray at final t. Use r_max=1.0 so
    # the sliver PAST the loader mask (r > 0.99) is also visible.
    n_r_dense = 1024
    canon_finals = {}
    for th in (0.0, 45.0, 90.0):
        km = _sample_radial_kymograph(
            f_canon.astype(np.float64), x_canon, y_canon, th,
            n_r=n_r_dense, r_max=1.0)
        canon_finals[th] = km[:, -1]
    r_axis_canon = np.linspace(0.0, 1.0, n_r_dense)

    # Filter native points into 3 angular bands
    bands = {}
    for th in (0.0, 45.0, 90.0):
        mask = _filter_near_ray(xy_native, th, args.angle_tol,
                                  args.r_min, args.r_max)
        r_pts = np.sqrt(xy_native[mask, 0] ** 2
                         + xy_native[mask, 1] ** 2)
        u_pts = u_native[mask]
        # Sort by r for cleaner scatter visual
        order = np.argsort(r_pts)
        bands[th] = (r_pts[order], u_pts[order])

    # Render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = args.value_scale
    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                              sharex=True, constrained_layout=True)

    # y-limits: use the union of native + canonical values in window
    all_u = []
    for th in (0.0, 45.0, 90.0):
        r_pts, u_pts = bands[th]
        all_u.append(u_pts)
        mask = (r_axis_canon >= args.r_min) & (r_axis_canon
                                                 <= args.r_max)
        all_u.append(canon_finals[th][mask])
    all_u_arr = np.concatenate(all_u) * scale
    finite = all_u_arr[np.isfinite(all_u_arr)]
    ymin = float(np.percentile(finite, 1)) if finite.size else -20
    ymax = float(np.percentile(finite, 99)) if finite.size else 5
    pad = 0.05 * (ymax - ymin + 1e-9)

    for i, th in enumerate((0.0, 45.0, 90.0)):
        ax = axes[i]
        # Native scatter
        r_pts, u_pts = bands[th]
        ax.scatter(r_pts, u_pts * scale, s=8, alpha=0.45,
                    color="0.30",
                    label=f"raw NPZ native "
                          f"(within +/-{args.angle_tol:g} deg)")
        # Canonical line
        canon_u = canon_finals[th] * scale
        show = (r_axis_canon >= args.r_min) & (r_axis_canon
                                                 <= args.r_max)
        ax.plot(r_axis_canon[show], canon_u[show], lw=1.8,
                 color=SENSOR_PALETTE[6],
                 label="loader canonical (sampled along ray)")
        # Vertical marker at rim mask boundary
        ax.axvline(_DISK_MASK_R_END, color="red", ls=":", lw=1,
                     alpha=0.6,
                     label=f"loader rim mask r={_DISK_MASK_R_END:g}")
        ax.axvline(1.0, color="0.5", ls=":", lw=1, alpha=0.4,
                     label="physical wafer edge r=1")
        ax.set_ylabel(f"u_z * {scale:g}")
        ax.set_title(f"theta = {th:g} deg", fontsize=10)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("r (normalized)")

    fig.suptitle(
        f"{sim_path.name}  |  final-frame u_z along radial rays  |  "
        f"raw NPZ vs loader canonical", fontsize=11)
    provenance_footer(fig, sim_id=sim_path.stem,
                        extras={"tol_deg": args.angle_tol,
                                "n_native": int(xy_native.shape[0])})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
