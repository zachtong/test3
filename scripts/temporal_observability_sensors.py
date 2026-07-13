"""Temporal-observability optimal sensor placement (observability
Gramian), the dynamics-aware successor to QR-DEIM.

Why QR-DEIM is not enough here (n < K, temporal reconstruction):
QR-DEIM maximizes INSTANTANEOUS spatial observability -- it picks
locations where the POD modes are most distinctive in a single
snapshot. For our pipeline the BiTCN reconstructs the modal state
from sensor TIME SERIES, using the bonding front's propagation.
QR-DEIM ignores that entirely; empirically it clustered 5 of 6
sensors on the outer rim (angularly distinctive modes) and lost to
the fixed ABCDEF hardware, whose two-ring radial spread observes
the front at different times.

Note: a naive "reconstruct a(t) from the sensor time series"
objective collapses back to (sigma-weighted) QR-DEIM, because a
linear point measurement y_i(t) = phi_i . a(t) carries the
location only through its spatial signature phi_i; the shared
trajectory a(t) provides no extra per-location temporal info under
a static (time-local) reconstruction. To genuinely use time we
must use the DYNAMICS.

Method: fit a linear dynamics model M to the POD coefficient
trajectories (DMD), then place sensors to maximize the finite-
horizon discrete observability Gramian

    W_o(S) = sum_{tau=0}^{T-1} (M^tau)^T C_S^T C_S M^tau

where C_S stacks the selected locations' spatial signatures (rows
of Phi). W_o is ADDITIVE over sensors, so each candidate has a
per-sensor Gramian W_i (K x K); greedily add the location that
maximizes log det(W + W_i) (D-optimal observability, submodular).
When M = I (no dynamics) W_i = T * phi_i phi_i^T and the greedy
reduces to QR-DEIM -- so this is a strict, dynamics-aware
generalization.

Inputs come from artifacts you already have:
  --basis  a basis_cache pod3d_*.npz  (Phi, spatial_shape)
  --traj   a traj_cache traj_*.npz    (a_train_val: modal coeffs)

    python scripts/temporal_observability_sensors.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --traj  outputs/basis_cache/traj_<key>.npz \\
        --n 6 --K 12 --out viz/tempobs_n6_k12.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid, disk_mask       # noqa: E402

_ABCDEF = [
    ("A", 0.52, 0.0), ("B", 0.52, 45.0), ("C", 0.52, 90.0),
    ("D", 0.847, 0.0), ("E", 0.847, 45.0), ("F", 0.847, 90.0),
]


def fit_dmd(a: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Least-squares linear dynamics M (K x K) with a[:, t+1] ~ M a[:, t],
    fit over every sim and every consecutive time pair.

    a: (n_sim, K, Nt) modal coefficient trajectories.
    """
    n_sim, K, Nt = a.shape
    # Stack consecutive pairs across all sims: columns are samples.
    x = a[:, :, :-1].transpose(1, 0, 2).reshape(K, -1)   # (K, n_sim*(Nt-1))
    y = a[:, :, 1:].transpose(1, 0, 2).reshape(K, -1)     # (K, n_sim*(Nt-1))
    # M = Y X^+  via least squares (stable pseudo-inverse).
    M = y @ np.linalg.pinv(x, rcond=rcond)
    return M


def per_sensor_gramians(Phi_c: np.ndarray, M: np.ndarray,
                        horizon: int, discount: float
                        ) -> np.ndarray:
    """For each candidate location, the finite-horizon observability
    Gramian W_i (K x K). Propagates each location's spatial signature
    backward through the dynamics and accumulates the outer products.

    Phi_c: (n_cand, K) candidate spatial signatures (rows of Phi).
    M:     (K, K) dynamics.
    Returns W: (n_cand, K, K).
    """
    n_cand, K = Phi_c.shape
    Mt = M.T
    W = np.zeros((n_cand, K, K), dtype=np.float64)
    u = Phi_c.astype(np.float64).copy()                  # (n_cand, K) = tau 0
    w = 1.0
    for tau in range(horizon):
        # accumulate w * outer(u_i, u_i) for every candidate at once
        W += w * (u[:, :, None] * u[:, None, :])
        u = u @ Mt                                       # advance: (M^T)^tau
        w *= discount
    return W


def greedy_logdet(W: np.ndarray, n: int, ridge: float
                  ) -> list[int]:
    """Greedily pick n candidate indices maximizing log det of the
    accumulated Gramian (D-optimal observability). W: (n_cand, K, K).

    Nested by construction: the first m picks are the greedy m-set.
    """
    n_cand, K, _ = W.shape
    acc = ridge * np.eye(K)
    chosen: list[int] = []
    remaining = set(range(n_cand))
    for _ in range(n):
        best_i, best_val = -1, -np.inf
        # sorted() so tie-breaks are deterministic / reproducible
        for i in sorted(remaining):
            sign, val = np.linalg.slogdet(acc + W[i])
            if sign > 0 and val > best_val:
                best_val, best_i = val, i
        if best_i < 0:
            break
        chosen.append(best_i)
        remaining.discard(best_i)
        acc = acc + W[best_i]
    return chosen


def _candidate_mask(nx, ny, r_min, r_max):
    x, y = canonical_grid(nx, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r_grid = np.sqrt(X * X + Y * Y)
    mask = disk_mask(nx, ny) & (r_grid >= r_min) & (r_grid <= r_max)
    return x, y, mask


def _a_from_npz_dir(Phi, npz_dir, nx, ny, nt, drop_first_steps,
                    limit):
    """Recompute modal coefficient trajectories a = Phi^T F from a
    SUBSAMPLE of the dataset. Robust alternative to hunting for a
    matching traj cache: the user gives the basis and the dataset,
    and consistency is guaranteed because a is projected onto THIS
    Phi. Loads only `limit` sims (not the full 93 GB F)."""
    from data.loader import load_dataset
    _x, _y, sims = load_dataset(
        npz_dir, nx=nx, ny=ny, nt=nt, limit=limit,
        drop_first_steps=drop_first_steps)
    if not sims:
        raise ValueError(f"no sims loaded from {npz_dir}")
    nspace = Phi.shape[0]
    K = Phi.shape[1]
    a = np.empty((len(sims), K, nt), dtype=np.float64)
    for i, s in enumerate(sims):
        f = np.asarray(s.f, dtype=np.float64).reshape(nspace, -1)
        a[i] = Phi.T @ f                                 # (K, Nt)
    return a


def temporal_observability_positions(
        basis_path, n, traj_path=None, npz_dir=None,
        K=None, r_min=0.2, r_max=0.98, horizon=None, discount=1.0,
        ridge=1e-9, rcond=1e-6, nt=300, drop_first_steps=1,
        limit=400) -> dict:
    """Full pipeline. Modal trajectories a come from either a traj
    cache (traj_path) or are recomputed from a dataset subsample
    (npz_dir). Returns rank-ordered positions + metadata; nested."""
    with np.load(basis_path) as z:
        Phi = z["Phi"]
        nx, ny = (int(d) for d in z["spatial_shape"])
    if traj_path is not None:
        with np.load(traj_path, allow_pickle=False) as z:
            a = z["a_train_val"].astype(np.float64)      # (n_sim, k, Nt)
    elif npz_dir is not None:
        a = _a_from_npz_dir(Phi, npz_dir, nx, ny, nt,
                            drop_first_steps, limit)
    else:
        raise ValueError("provide traj_path or npz_dir")
    k_avail = min(Phi.shape[1], a.shape[1])
    K = k_avail if K is None else min(K, k_avail)
    Phi = Phi[:, :K]
    a = a[:, :K, :]
    n_sim, _, Nt = a.shape
    horizon = Nt if horizon is None else min(horizon, Nt)

    x, y, mask = _candidate_mask(nx, ny, r_min, r_max)
    disk_idx = np.where(mask.ravel())[0]
    if disk_idx.size < n:
        raise ValueError(
            f"only {disk_idx.size} candidates in "
            f"r in [{r_min}, {r_max}]; need >= {n}")
    Phi_c = Phi[disk_idx, :]                              # (n_cand, K)

    M = fit_dmd(a, rcond=rcond)
    W = per_sensor_gramians(Phi_c, M, horizon, discount)
    local = greedy_logdet(W, n, ridge)

    chosen_flat = disk_idx[np.asarray(local)]
    ix = chosen_flat // ny
    iy = chosen_flat % ny
    xs = x[ix]
    ys = y[iy]
    r = np.sqrt(xs ** 2 + ys ** 2)
    theta = np.degrees(np.arctan2(ys, xs))
    positions = [[round(float(ri), 4), round(float(ti), 2)]
                 for ri, ti in zip(r, theta)]
    # spectral radius of M (stability diagnostic)
    eig = np.abs(np.linalg.eigvals(M))
    return dict(positions=positions, r=r, theta=theta, xs=xs, ys=ys,
                n_candidates=int(disk_idx.size), K=int(K),
                Nt=int(Nt), horizon=int(horizon),
                n_sim=int(n_sim), spectral_radius=float(eig.max()),
                r_min=r_min, r_max=r_max)


def _render_overlay(xs, ys, n, K, r_min, r_max, out_path: Path,
                    qr_positions=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 6.8),
                           constrained_layout=True)
    th = np.linspace(0, 90, 200)
    ax.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)),
            color="0.35", lw=2.0)
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (r_min, r_max):
        ax.plot(rb * np.cos(np.deg2rad(th)),
                rb * np.sin(np.deg2rad(th)),
                color="#2a9d8f", lw=1.0, ls="-", alpha=0.6)
    for letter, rr, tt in _ABCDEF:
        hx, hy = rr * np.cos(np.deg2rad(tt)), rr * np.sin(np.deg2rad(tt))
        ax.scatter([hx], [hy], s=150, marker="s", facecolor="none",
                   edgecolor="0.4", linewidth=1.6, zorder=4)
        ax.annotate(letter, (hx, hy), xytext=(6, 6),
                    textcoords="offset points", fontsize=10,
                    color="0.4")
    if qr_positions is not None:
        for p in qr_positions:
            qx = p[0] * np.cos(np.deg2rad(p[1]))
            qy = p[0] * np.sin(np.deg2rad(p[1]))
            ax.scatter([qx], [qy], s=120, marker="^", color="0.55",
                       edgecolor="black", linewidth=0.8, zorder=4)
        ax.scatter([], [], marker="^", color="0.55",
                   edgecolor="black", label="QR-DEIM")
    ax.scatter(xs, ys, s=210, marker="o", color="#1d3557",
               edgecolor="black", linewidth=1.2, zorder=6,
               label="temporal-observability")
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        ax.annotate(str(i + 1), (xi, yi), xytext=(8, -12),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color="#1d3557")
    ax.scatter([], [], marker="s", facecolor="none",
               edgecolor="0.4", label="ABCDEF hardware")
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"Temporal-observability optimal {n} sensors "
                 f"(K={K}) vs ABCDEF")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True,
                    help="basis_cache pod3d_*.npz (Phi)")
    ap.add_argument("--traj", default=None,
                    help="traj_cache traj_*.npz (a_train_val). "
                    "Optional -- if the basis/traj filenames are "
                    "hard to pair (different hash keys), use "
                    "--npz-dir instead to recompute a from the "
                    "dataset directly.")
    ap.add_argument("--npz-dir", default=None,
                    help="dataset dir; recompute modal trajectories "
                    "a = Phi^T F from a subsample (--limit sims). "
                    "Guarantees basis/a consistency. Use the SAME "
                    "grid as training (--nt, --drop-first-steps).")
    ap.add_argument("--limit", type=int, default=400,
                    help="sims to load when using --npz-dir "
                    "(default 400; the placement is a statistical "
                    "property, does not need all sims)")
    ap.add_argument("--nt", type=int, default=300,
                    help="canonical timesteps (must match training; "
                    "default 300)")
    ap.add_argument("--drop-first-steps", type=int, default=1,
                    help="loader trim (must match training; "
                    "default 1)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--K", type=int, default=None,
                    help="POD modes; default = all in the basis")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--horizon", type=int, default=None,
                    help="observability window length in canonical "
                    "timesteps; default = full Nt")
    ap.add_argument("--discount", type=float, default=1.0,
                    help="per-step decay in the Gramian sum "
                    "(<1 down-weights late times; use if the fitted "
                    "dynamics are unstable). Default 1.0")
    ap.add_argument("--ridge", type=float, default=1e-9,
                    help="Gramian regularization for log det")
    ap.add_argument("--out", default=None,
                    help="optional overlay PNG vs ABCDEF")
    ap.add_argument("--qr-positions", default=None,
                    help="optional JSON [[r,theta],...] of the "
                    "QR-DEIM picks, to overlay for comparison")
    ap.add_argument("--positions-json", default=None,
                    help="optional path to write picked positions")
    args = ap.parse_args()

    if not Path(args.basis).is_file():
        print(f"not found: {args.basis}", file=sys.stderr)
        return 2
    if not args.traj and not args.npz_dir:
        print("provide --traj OR --npz-dir", file=sys.stderr)
        return 2
    if args.traj and not Path(args.traj).is_file():
        print(f"not found: {args.traj}", file=sys.stderr)
        return 2
    if args.npz_dir and not Path(args.npz_dir).is_dir():
        print(f"not a directory: {args.npz_dir}", file=sys.stderr)
        return 2
    try:
        res = temporal_observability_positions(
            args.basis, args.n, traj_path=args.traj,
            npz_dir=args.npz_dir, K=args.K,
            r_min=args.r_min, r_max=args.r_max,
            horizon=args.horizon, discount=args.discount,
            ridge=args.ridge, nt=args.nt,
            drop_first_steps=args.drop_first_steps,
            limit=args.limit)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    positions = res["positions"]
    K = res["K"]
    print(f"Temporal-observability picked {args.n} sensor(s) from "
          f"{res['n_candidates']} candidates in "
          f"r in [{args.r_min}, {args.r_max}] "
          f"(K={K}, Nt={res['Nt']}, horizon={res['horizon']}, "
          f"{res['n_sim']} train sims).")
    sr = res["spectral_radius"]
    print(f"  fitted dynamics spectral radius = {sr:.4f}"
          + ("  (STABLE)" if sr <= 1.0 + 1e-6 else
             "  (UNSTABLE -- consider --discount 0.98 or --horizon)"))
    print(f"  Ordered by greedy log-det gain: rank 1 = single most "
          f"observable location; nested (first N = best N-set).")
    print(f"  {'rank':>4}  {'r':>7}  {'theta':>7}")
    for i, (ri, ti) in enumerate(zip(res["r"], res["theta"])):
        print(f"  {i + 1:>4}  {ri:7.4f}  {ti:7.2f}")

    pos_json = json.dumps(positions)
    print(f"\nPositions JSON:\n  {pos_json}")
    print(f"\nTrain this placement at K={K}:")
    print(f"  python scripts/train.py --config configs/default.yaml \\")
    print(f"      --data.npz_dir <DATASET> --data.workers 64 "
          f"--pod.workers 64 \\")
    print(f"      --pod.k {K} --sensors.n {args.n} "
          f"--sensors.strategy custom \\")
    print(f"      --sensors.positions '{pos_json}' \\")
    print(f"      --tag tempobs_n{args.n}_k{K}")

    if args.positions_json:
        Path(args.positions_json).parent.mkdir(parents=True,
                                                exist_ok=True)
        Path(args.positions_json).write_text(pos_json)
        print(f"\nwrote positions -> {args.positions_json}")

    if args.out:
        qr = None
        if args.qr_positions and Path(args.qr_positions).is_file():
            qr = json.loads(Path(args.qr_positions).read_text())
        _render_overlay(res["xs"], res["ys"], args.n, K,
                        args.r_min, args.r_max, Path(args.out), qr)
        print(f"wrote overlay -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
