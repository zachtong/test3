"""Multi-start differentiable placement: the practical way to search for the
GLOBAL-optimal sensor layout instead of trusting one hand-picked init.

The placement objective is non-convex and low-dimensional (n sensors x (r,
theta)); different inits fall into different local optima, so no single init
reliably finds the best. This loads the data ONCE, then gradient-optimizes
from --restarts random inits (cartesian / isotropic), ranks them by the
in-loop validation loss, and:
  - prints the terminal-loss distribution (a tight cluster of best values =>
    you have likely found the global basin; still spreading => raise
    --restarts);
  - writes the TOP-K candidate layouts as JSON so you can retrain them
    properly (scripts/train.py / run_placement_multistart.py) and confirm the
    winner on held-out test -- the cheap search here only RANKS layouts;
  - plots the sorted terminal losses and the top-K layouts on the quarter disk.

    python run_diffplace_multistart.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --restarts 40 --epochs 400 --top-k 5 \\
        --out viz/diffplace/multistart.png
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.train_differentiable_placement import (        # noqa: E402
    _load_phi_and_a, _optimize)

_NPZ_DEFAULT = "/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/"


def _round_pos(pos) -> list:
    return [[round(float(r), 4), round(float(t), 2)] for r, t in pos]


def _render(runs, top_k, out_path, r_min, r_max, n, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.array([r["best_val"] for r in runs])
    order = np.argsort(vals)
    fig, (axl, axp) = plt.subplots(
        1, 2, figsize=(13, 5.6),
        gridspec_kw=dict(width_ratios=[1.0, 1.0]),
        constrained_layout=True)

    # --- left: sorted terminal loss over restarts ---
    rank = np.arange(1, len(vals) + 1)
    axl.plot(rank, vals[order], "-o", ms=4, color="#3d5a80")
    kk = min(top_k, len(vals))
    axl.plot(rank[:kk], vals[order][:kk], "o", ms=9, color="#e63946",
             label=f"top-{kk}")
    axl.set_yscale("log")
    axl.set_xlabel("restart (sorted best -> worst)")
    axl.set_ylabel("in-loop val loss (best over epochs)")
    spread = float(vals.max() - vals.min())
    rel = spread / max(float(vals.mean()), 1e-30)
    axl.set_title(f"{len(vals)} restarts | best {vals.min():.4e} | "
                  f"spread {rel * 100:.0f}% of mean")
    axl.legend(fontsize=9)
    axl.grid(alpha=0.3, which="both")

    # --- right: top-K layouts on the quarter disk ---
    th = np.linspace(0, 90, 200)
    axp.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)), color="0.35", lw=2)
    axp.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    axp.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (r_min, r_max):
        axp.plot(rb * np.cos(np.deg2rad(th)), rb * np.sin(np.deg2rad(th)),
                 color="#2a9d8f", lw=1, alpha=0.6)
    cmap = plt.cm.autumn(np.linspace(0, 0.75, kk))
    for j in range(kk):
        p = np.asarray(runs[order[j]]["best_pos"])
        x = p[:, 0] * np.cos(np.deg2rad(p[:, 1]))
        y = p[:, 0] * np.sin(np.deg2rad(p[:, 1]))
        axp.scatter(x, y, s=110 - 12 * j, color=cmap[j], edgecolor="black",
                    linewidth=0.6, alpha=0.9, zorder=5 - j // 3,
                    label=f"#{j + 1}  {vals[order][j]:.3e}")
    axp.set_xlim(-0.08, 1.15)
    axp.set_ylim(-0.08, 1.15)
    axp.set_aspect("equal")
    axp.set_xlabel("x / R")
    axp.set_ylabel("y / R")
    axp.set_title(f"top-{kk} layouts (n={n}, K={K})")
    axp.legend(fontsize=8, loc="upper right")
    axp.grid(alpha=0.25)

    fig.suptitle("Multi-start placement search", fontweight="bold")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--npz-dir", default=_NPZ_DEFAULT)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--no-random-subsample", action="store_true",
                    help="disable random subsampling of the --limit sims "
                    "(on by default, needed for the merged/ordered dataset)")
    ap.add_argument("--restarts", type=int, default=40,
                    help="number of random inits to optimize (default 40)")
    ap.add_argument("--epochs", type=int, default=400,
                    help="epochs per restart (search phase; retrain finalists "
                    "fully afterwards). Default 400.")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--pos-lr", type=float, default=2e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many best layouts to export + plot (default 5)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pos-dir", default="viz/diffplace/multistart",
                    help="dir for the top-K layout JSONs + summary")
    ap.add_argument("--out", default="viz/diffplace/multistart.png")
    args = ap.parse_args()

    if not Path(args.basis).is_file():
        print(f"basis not found: {args.basis}", file=sys.stderr)
        return 2

    print(f"loading data once (basis + {'traj' if args.traj else 'npz-dir'}) "
          f"...", flush=True)
    Phi, a_np, (nx, ny) = _load_phi_and_a(
        args.basis, args.traj, args.npz_dir, args.K, args.nt,
        args.drop_first_steps, args.limit,
        random_subsample=not args.no_random_subsample, seed=args.base_seed)
    K = Phi.shape[1]
    print(f"loaded Phi {Phi.shape}, a {a_np.shape}; running {args.restarts} "
          f"restarts x {args.epochs} epochs ...", flush=True)

    runs = []
    t0 = time.time()
    for i in range(args.restarts):
        res = _optimize(
            Phi, a_np, nx, ny, n=args.n, init="random", param="cartesian",
            r_min=args.r_min, r_max=args.r_max, epochs=args.epochs,
            lr=args.lr, pos_lr=args.pos_lr, val_frac=args.val_frac,
            seed=args.base_seed + i, device=args.device, verbose=False)
        runs.append(res)
        rate = (i + 1) / max(time.time() - t0, 1e-9)
        print(f"  restart {i + 1:3d}/{args.restarts}  "
              f"best_val {res['best_val']:.4e}  "
              f"ETA {(args.restarts - i - 1) / max(rate, 1e-9):.0f}s",
              flush=True)

    vals = np.array([r["best_val"] for r in runs])
    order = np.argsort(vals)
    kk = min(args.top_k, len(runs))

    print("\n===== multi-start summary =====")
    print(f"  restarts={len(runs)}  best={vals.min():.4e}  "
          f"median={np.median(vals):.4e}  worst={vals.max():.4e}")
    rel = (vals.max() - vals.min()) / max(vals.mean(), 1e-30)
    print(f"  terminal-loss spread = {rel * 100:.0f}% of mean")
    if rel < 0.10:
        print("  -> tight cluster: the restarts largely agree, you have "
              "likely found the global basin.")
    else:
        print("  -> wide spread: multiple basins of differing quality; the "
              "top-K below are the candidates. Raise --restarts if the best "
              "is still improving.")
    print(f"\n  top-{kk} by in-loop val loss:")
    for j in range(kk):
        r = runs[order[j]]
        print(f"    #{j + 1}  val {r['best_val']:.4e}  "
              f"pos {json.dumps(_round_pos(r['best_pos']))}")

    # export top-K layouts + a summary
    pos_dir = Path(args.pos_dir)
    pos_dir.mkdir(parents=True, exist_ok=True)
    for j in range(kk):
        (pos_dir / f"top{j + 1}.json").write_text(
            json.dumps(_round_pos(runs[order[j]]["best_pos"])))
    (pos_dir / "summary.json").write_text(json.dumps(dict(
        restarts=len(runs), K=int(K), n=int(args.n),
        best_val=float(vals.min()), median_val=float(np.median(vals)),
        rel_spread=float(rel),
        top=[dict(rank=j + 1, val=float(vals[order[j]]),
                  positions=_round_pos(runs[order[j]]["best_pos"]))
             for j in range(kk)]), indent=2))
    print(f"\n  wrote top-{kk} layouts to {pos_dir}/top*.json + summary.json")

    _render(runs, args.top_k, Path(args.out), args.r_min, args.r_max,
            args.n, K)
    print(f"  wrote {args.out}")
    print(f"\nNext: retrain the top layouts properly and confirm on test, e.g.\n"
          f"  python scripts/train.py --config configs/default.yaml \\\n"
          f"    --data.npz_dir {args.npz_dir} --pod.k {K} --sensors.n {args.n} "
          f"--sensors.strategy custom \\\n"
          f"    --sensors.positions \"$(cat {pos_dir}/top1.json)\" "
          f"--tag m_multistart_top1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
