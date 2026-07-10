"""QR-DEIM optimal sensor placement from a POD basis.

Given the fitted POD basis Phi (Nx*Ny, K), pick the n in-disk grid
locations that best observe the modal subspace, via QR column
pivoting -- the standard DEIM / discrete-empirical-interpolation
sensor-placement construction. NO search, NO training: one QR of
the (K, n_disk) matrix picks the locations directly.

Purpose: compare the theoretically-optimal 6-sensor placement (a
sensor may sit anywhere on the quarter disk) against the fixed
hardware catalogue ABCDEF, to answer "is our buildable placement
near-optimal?". Because sensor placement is a second-order effect
here (K dominates), the expected outcome is that ABCDEF lands
close to the QR-DEIM optimum -- which is itself the paper-worthy
result.

Caveat (n < K): with 6 sensors and 12 modes the observation is
underdetermined -- 6 point measurements cannot instantaneously
invert 12 modal coefficients. The BiTCN compensates with temporal
dynamics, which QR-DEIM does not model. So the picked locations
are a principled, fast CANDIDATE (greedy-optimal for the leading
subspace), not a provable optimum for the temporal-inference
pipeline. Still far better motivated than random search, and the
right thing to benchmark ABCDEF against.

    python scripts/qr_deim_sensors.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --n 6 --K 12 \\
        --out viz/qr_deim_n6_k12.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid, disk_mask       # noqa: E402

# The fixed hardware catalogue, for the overlay comparison.
_ABCDEF = [
    ("A", 0.52, 0.0), ("B", 0.52, 45.0), ("C", 0.52, 90.0),
    ("D", 0.847, 0.0), ("E", 0.847, 45.0), ("F", 0.847, 90.0),
]


def qr_deim_pick(Phi_disk: np.ndarray, n: int) -> np.ndarray:
    """Return the first n QR-pivot column indices (into the rows of
    Phi_disk) -- the DEIM sensor locations. Phi_disk is (n_disk, K)."""
    # Pivot the LOCATIONS (columns of Phi_disk.T), so QR of the
    # (K, n_disk) matrix; the pivot order ranks locations by how
    # much each adds to spanning the modal column space.
    _, _, piv = scipy.linalg.qr(Phi_disk.T, pivoting=True,
                                mode="economic")
    return piv[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True,
                    help="path to a basis_cache pod3d_*.npz "
                    "(holds Phi, sigma, spatial_shape)")
    ap.add_argument("--n", type=int, default=6,
                    help="number of sensors to place (default: 6)")
    ap.add_argument("--K", type=int, default=None,
                    help="number of POD modes to observe; default = "
                    "all modes stored in the basis file")
    ap.add_argument("--weight-sigma", action="store_true",
                    help="scale each mode column by its singular "
                    "value before QR, biasing placement toward "
                    "observing high-energy modes (sensible when "
                    "n < K). Default: unweighted (textbook DEIM).")
    ap.add_argument("--r-min", type=float, default=0.2,
                    help="drop candidate cells with r < r-min "
                    "(sensors cannot sit too near the center; "
                    "default: 0.2)")
    ap.add_argument("--r-max", type=float, default=0.98,
                    help="drop candidate cells with r > r-max "
                    "(sensors cannot sit at the very rim; "
                    "default: 0.98)")
    ap.add_argument("--out", default=None,
                    help="optional PNG overlay of picks vs ABCDEF")
    ap.add_argument("--positions-json", default=None,
                    help="optional path to write the picked "
                    "positions as a JSON [[r,theta],...] list")
    args = ap.parse_args()

    basis_path = Path(args.basis)
    if not basis_path.is_file():
        print(f"basis not found: {basis_path}", file=sys.stderr)
        return 2
    with np.load(basis_path) as z:
        Phi = z["Phi"]                              # (Nx*Ny, k_cache)
        sigma = z["sigma"]
        nx, ny = (int(d) for d in z["spatial_shape"])
    k_avail = Phi.shape[1]
    K = args.K if args.K is not None else k_avail
    if K > k_avail:
        print(f"requested K={K} > basis k_cache={k_avail}; "
              f"clamping to {k_avail}", file=sys.stderr)
        K = k_avail
    Phi = Phi[:, :K]
    sigma = sigma[:K]

    # Candidate locations: in-disk AND within the hardware-feasible
    # radial band [r-min, r-max]. Off-disk rows are ~0 and the very
    # center / very rim are not mountable sensor sites.
    x, y = canonical_grid(nx, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r_grid = np.sqrt(X * X + Y * Y)
    mask2d = disk_mask(nx, ny)                       # (Nx, Ny) bool
    mask2d = (mask2d
              & (r_grid >= args.r_min)
              & (r_grid <= args.r_max))
    mask_flat = mask2d.ravel()
    disk_idx = np.where(mask_flat)[0]                # (n_cand,)
    if disk_idx.size < args.n:
        print(f"ERROR: only {disk_idx.size} candidate cell(s) in "
              f"r in [{args.r_min}, {args.r_max}]; need >= {args.n}. "
              f"Widen the band.", file=sys.stderr)
        return 1
    Phi_disk = Phi[disk_idx, :]                      # (n_cand, K)
    if args.weight_sigma:
        Phi_disk = Phi_disk * sigma[None, :]

    if args.n > K:
        print(f"note: n={args.n} > K={K}; QR gives at most K "
              f"distinct pivots, extra picks may be arbitrary",
              file=sys.stderr)

    local = qr_deim_pick(Phi_disk, args.n)           # into disk_idx
    chosen_flat = disk_idx[local]
    ix = chosen_flat // ny
    iy = chosen_flat % ny
    xs = x[ix]
    ys = y[iy]
    r = np.sqrt(xs ** 2 + ys ** 2)
    theta = np.degrees(np.arctan2(ys, xs))

    positions = [[round(float(r_i), 4), round(float(th_i), 2)]
                 for r_i, th_i in zip(r, theta)]

    print(f"QR-DEIM picked {args.n} sensor(s) from {len(disk_idx)} "
          f"candidates in r in [{args.r_min}, {args.r_max}] "
          f"(K={K}, "
          f"{'sigma-weighted' if args.weight_sigma else 'unweighted'}):")
    print(f"  {'#':>2}  {'r':>7}  {'theta':>7}  {'(x, y)':>16}")
    for i, (r_i, th_i, x_i, y_i) in enumerate(
            zip(r, theta, xs, ys)):
        print(f"  {i + 1:>2}  {r_i:7.4f}  {th_i:7.2f}  "
              f"({x_i:6.3f}, {y_i:6.3f})")

    pos_json = json.dumps(positions)
    print(f"\nPositions JSON (paste into --sensors.positions):")
    print(f"  {pos_json}")
    print(f"\nTrain this placement at K={K}:")
    print(f"  python scripts/train.py --config configs/default.yaml \\")
    print(f"      --data.npz_dir <DATASET> --data.workers 64 "
          f"--pod.workers 64 \\")
    print(f"      --pod.k {K} --sensors.n {args.n} "
          f"--sensors.strategy custom \\")
    print(f"      --sensors.positions '{pos_json}' \\")
    print(f"      --tag qrdeim_n{args.n}_k{K}")

    if args.positions_json:
        Path(args.positions_json).parent.mkdir(
            parents=True, exist_ok=True)
        Path(args.positions_json).write_text(pos_json)
        print(f"\nwrote positions -> {args.positions_json}")

    if args.out:
        _render_overlay(r, theta, xs, ys, args.n, K,
                        args.weight_sigma, args.r_min, args.r_max,
                        Path(args.out))
        print(f"wrote overlay -> {args.out}")
    return 0


def _render_overlay(r, theta, xs, ys, n, K, weighted,
                    r_min, r_max, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           constrained_layout=True)
    th_arc = np.linspace(0, 90, 200)
    ax.plot(np.cos(np.deg2rad(th_arc)), np.sin(np.deg2rad(th_arc)),
            color="0.35", lw=2.0)
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for r_ring in (0.52, 0.847):
        ax.plot(r_ring * np.cos(np.deg2rad(th_arc)),
                r_ring * np.sin(np.deg2rad(th_arc)),
                color="0.75", lw=0.8, ls="--")
    # Feasible radial band [r_min, r_max]: shade the annulus edges.
    for rb in (r_min, r_max):
        ax.plot(rb * np.cos(np.deg2rad(th_arc)),
                rb * np.sin(np.deg2rad(th_arc)),
                color="#2a9d8f", lw=1.2, ls="-", alpha=0.7)
    ax.fill_between([0, 1.1], 0, 0, color="#2a9d8f", alpha=0.0,
                    label=f"feasible band r in [{r_min:g}, {r_max:g}]")
    # ABCDEF hardware catalogue (gray squares)
    for letter, rr, tt in _ABCDEF:
        hx, hy = rr * np.cos(np.deg2rad(tt)), rr * np.sin(np.deg2rad(tt))
        ax.scatter([hx], [hy], s=150, marker="s",
                   facecolor="none", edgecolor="0.4", linewidth=1.6,
                   zorder=4)
        ax.annotate(letter, (hx, hy), xytext=(6, 6),
                    textcoords="offset points", fontsize=10,
                    color="0.4")
    # QR-DEIM picks (red circles)
    ax.scatter(xs, ys, s=200, marker="o", color="#e63946",
               edgecolor="black", linewidth=1.2, zorder=5,
               label="QR-DEIM optimal")
    for i, (x_i, y_i) in enumerate(zip(xs, ys)):
        ax.annotate(str(i + 1), (x_i, y_i), xytext=(8, -12),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color="#e63946")
    ax.scatter([], [], s=150, marker="s", facecolor="none",
               edgecolor="0.4", label="ABCDEF hardware")
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"QR-DEIM optimal {n} sensors (K={K}"
                 f"{', sigma-weighted' if weighted else ''}) "
                 f"vs ABCDEF")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
