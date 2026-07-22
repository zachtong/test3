"""Multi-start differentiable placement: the practical way to search for the
GLOBAL-optimal sensor layout instead of trusting one hand-picked init.

The placement objective is non-convex and lives in a 2*n-dimensional space
(n sensors x (r, theta)) with a big permutation symmetry (the sensors are
interchangeable). Different inits fall into different local optima, so no
single init reliably finds the best. This searches it well by:

  - SPACE-FILLING starts: a Latin-hypercube spread of the 2n-dim init box (far
    better coverage than i.i.d. random when the restart count is small relative
    to the dimension), plus a few physically-motivated structured seeds
    (outer-uniform, diag-45, ABCDEF, ring x angle variants);
  - ANTI-COLLAPSE: min-sep rejection at init + a hinge repulsion penalty during
    optimization, so no layout collapses its sensors together;
  - PERMUTATION-INVARIANT dedup: two restarts that reach the same optimum up to
    relabeling count as ONE, so the exported top-K are genuinely distinct
    configurations.

The data is loaded ONCE and the in-memory Phi/a is reused across all restarts
(no repeated disk/cache reads). The search only RANKS layouts; the top-K are
then retrained properly (optionally, single-seed) and compared.

    python run_diffplace_multistart.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz --restarts 100
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.train_differentiable_placement import (        # noqa: E402
    _load_phi_and_a, _optimize)

PY = sys.executable


def _round_pos(pos) -> list:
    return [[round(float(r), 4), round(float(t), 2)] for r, t in pos]


# --- init-layout generation -----------------------------------------

def _lhs(n_samples, d, rng):
    """Latin-hypercube samples in [0,1]^d: each axis stratified into n_samples
    bins, one sample per bin -- far better space-filling than i.i.d. random
    when n_samples is small relative to d."""
    out = np.empty((n_samples, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n_samples) + rng.random(n_samples)) \
            / n_samples
    return out


def _lhs_layouts(n_layouts, n, r_min, r_max, rng):
    if n_layouts <= 0:
        return []
    U = _lhs(n_layouts, 2 * n, rng)
    outs = []
    for row in U:
        r = r_min + row[0::2] * (r_max - r_min)
        th = row[1::2] * 90.0
        outs.append(np.stack([r, th], axis=1))
    return outs


def _structured_seeds(n, r_min, r_max):
    """A few physically-motivated layouts to seed the search alongside the
    space-filling starts."""
    seeds = [np.stack([np.full(n, r_max), np.linspace(0, 90, n)], axis=1),
             np.stack([np.linspace(r_min, r_max, n), np.full(n, 45.0)], axis=1)]
    if n == 6:
        for ri, ro in [(0.52, 0.847), (0.4, 0.9), (0.3, 0.8)]:
            for a in [(0, 45, 90), (15, 45, 75)]:
                seeds.append(np.array(
                    [[ri, a[0]], [ri, a[1]], [ri, a[2]],
                     [ro, a[0]], [ro, a[1]], [ro, a[2]]], dtype=float))
    return seeds


# --- permutation-invariant dedup ------------------------------------

def _same_layout(a, b, r_tol, th_tol):
    """True if a and b are the same UNORDERED point set within tolerance
    (greedy bijection). Sensors are interchangeable, so two restarts that
    reach the same optimum up to relabeling count as one layout."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        return False
    used = np.zeros(len(b), dtype=bool)
    for pa in a:
        best_j, best_d = -1, 1e18
        for j in range(len(b)):
            if used[j]:
                continue
            dr, dth = abs(pa[0] - b[j, 0]), abs(pa[1] - b[j, 1])
            if dr <= r_tol and dth <= th_tol and (dr / r_tol + dth / th_tol) \
                    < best_d:
                best_d, best_j = dr / r_tol + dth / th_tol, j
        if best_j < 0:
            return False
        used[best_j] = True
    return True


def _dedup_order(runs, order, r_tol, th_tol):
    """Indices from `order` (best-val first), keeping only layouts distinct
    from every already-kept one."""
    kept = []
    for i in order:
        if any(_same_layout(runs[i]["best_pos"], runs[k]["best_pos"],
                            r_tol, th_tol) for k in kept):
            continue
        kept.append(i)
    return kept


# --- rendering ------------------------------------------------------

def _render(runs, distinct, top_k, out_path, r_min, r_max, n, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.array([r["best_val"] for r in runs])
    val_sorted = np.sort(vals)
    kk = min(top_k, len(distinct))
    fig, (axl, axp) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   constrained_layout=True)

    # --- left: sorted terminal loss over restarts, distinct top-K marked ---
    axl.plot(np.arange(1, len(vals) + 1), val_sorted, "-o", ms=4,
             color="#3d5a80")
    dvals = [vals[distinct[j]] for j in range(kk)]
    dranks = [int(np.searchsorted(val_sorted, v) + 1) for v in dvals]
    axl.plot(dranks, dvals, "o", ms=9, color="#e63946",
             label=f"top-{kk} distinct")
    axl.set_yscale("log")
    axl.set_xlabel("restart (sorted best -> worst)")
    axl.set_ylabel("in-loop val loss (best over epochs)")
    rel = float(vals.max() - vals.min()) / max(float(vals.mean()), 1e-30)
    axl.set_title(f"{len(vals)} restarts | {len(distinct)} distinct optima | "
                  f"best {vals.min():.3e} | spread {rel * 100:.0f}%")
    axl.legend(fontsize=9)
    axl.grid(alpha=0.3, which="both")

    # --- right: top-K distinct layouts on the quarter disk ---
    th = np.linspace(0, 90, 200)
    axp.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)), color="0.35", lw=2)
    axp.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    axp.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (r_min, r_max):
        axp.plot(rb * np.cos(np.deg2rad(th)), rb * np.sin(np.deg2rad(th)),
                 color="#2a9d8f", lw=1, alpha=0.6)
    cmap = plt.cm.autumn(np.linspace(0, 0.75, kk))
    for j in range(kk):
        p = np.asarray(runs[distinct[j]]["best_pos"])
        x = p[:, 0] * np.cos(np.deg2rad(p[:, 1]))
        y = p[:, 0] * np.sin(np.deg2rad(p[:, 1]))
        axp.scatter(x, y, s=110 - 12 * j, color=cmap[j], edgecolor="black",
                    linewidth=0.6, alpha=0.9, zorder=6 - j,
                    label=f"#{j + 1}  {vals[distinct[j]]:.3e}")
    axp.set_xlim(-0.08, 1.15)
    axp.set_ylim(-0.08, 1.15)
    axp.set_aspect("equal")
    axp.set_xlabel("x / R")
    axp.set_ylabel("y / R")
    axp.set_title(f"top-{kk} distinct layouts (n={n}, K={K})")
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
    ap.add_argument("--npz-dir", default=None,
                    help="dataset dir (no default; pass it, or use --traj)")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--no-random-subsample", action="store_true",
                    help="disable random subsampling of the --limit sims")
    ap.add_argument("--restarts", type=int, default=100,
                    help="total number of inits to optimize (default 100)")
    ap.add_argument("--sampling", default="lhs", choices=["lhs", "random"],
                    help="space-filling Latin-hypercube (default) or i.i.d. "
                    "random init sampling")
    ap.add_argument("--n-seeds", type=int, default=8,
                    help="how many structured seeds to include before the "
                    "space-filling fill (default 8; capped at what exists)")
    ap.add_argument("--epochs", type=int, default=400,
                    help="epochs per restart (search phase). Default 400.")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--min-sep", type=float, default=0.1,
                    help="minimum pairwise sensor spacing (normalized); "
                    "anti-collapse. 0 disables.")
    ap.add_argument("--rep-coef", type=float, default=50.0)
    ap.add_argument("--pos-lr", type=float, default=2e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many DISTINCT best layouts to export/plot/train")
    ap.add_argument("--no-dedup", action="store_true",
                    help="do not merge permutation-equivalent layouts")
    ap.add_argument("--dedup-r-tol", type=float, default=0.06)
    ap.add_argument("--dedup-th-tol", type=float, default=10.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pos-dir", default="viz/diffplace/multistart")
    ap.add_argument("--out", default="viz/diffplace/multistart.png")
    ap.add_argument("--train-top-k", type=int, default=5,
                    help="after the search, train the top-K DISTINCT layouts "
                    "with a SINGLE seed and compare. 0 = search only.")
    ap.add_argument("--train-seed", type=int, default=7)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pod-workers", type=int, default=64)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if not Path(args.basis).is_file():
        print(f"basis not found: {args.basis}", file=sys.stderr)
        return 2

    print(f"loading data once ...", flush=True)
    Phi, a_np, (nx, ny) = _load_phi_and_a(
        args.basis, args.traj, args.npz_dir, args.K, args.nt,
        args.drop_first_steps, args.limit,
        random_subsample=not args.no_random_subsample, seed=args.base_seed)
    K = Phi.shape[1]

    # build the init layouts: structured seeds first, then space-filling
    rng = np.random.default_rng(args.base_seed)
    seeds = _structured_seeds(args.n, args.r_min, args.r_max)
    seeds = seeds[:max(0, min(args.n_seeds, args.restarts))]
    n_fill = max(0, args.restarts - len(seeds))
    if args.sampling == "lhs":
        fill = _lhs_layouts(n_fill, args.n, args.r_min, args.r_max, rng)
    else:
        fill = [None] * n_fill              # i.i.d. random inside _optimize
    layouts = list(seeds) + list(fill)
    print(f"loaded Phi {Phi.shape}, a {a_np.shape}; {len(seeds)} structured "
          f"seeds + {n_fill} {args.sampling} starts x {args.epochs} epochs ...",
          flush=True)

    runs = []
    t0 = time.time()
    for i, lay in enumerate(layouts):
        res = _optimize(
            Phi, a_np, nx, ny, n=args.n, init="random", init_positions=lay,
            param="cartesian", r_min=args.r_min, r_max=args.r_max,
            epochs=args.epochs, lr=args.lr, pos_lr=args.pos_lr,
            val_frac=args.val_frac, seed=args.base_seed + i,
            device=args.device, verbose=False, min_sep=args.min_sep,
            rep_coef=args.rep_coef)
        runs.append(res)
        rate = (i + 1) / max(time.time() - t0, 1e-9)
        print(f"  restart {i + 1:3d}/{len(layouts)}  "
              f"best_val {res['best_val']:.4e}  "
              f"ETA {(len(layouts) - i - 1) / max(rate, 1e-9):.0f}s", flush=True)

    vals = np.array([r["best_val"] for r in runs])
    order = list(np.argsort(vals))
    distinct = order if args.no_dedup else _dedup_order(
        runs, order, args.dedup_r_tol, args.dedup_th_tol)
    kk = min(args.top_k, len(distinct))

    print("\n===== multi-start summary =====")
    print(f"  restarts={len(runs)}  distinct optima={len(distinct)}  "
          f"best={vals.min():.4e}  median={np.median(vals):.4e}")
    rel = (vals.max() - vals.min()) / max(vals.mean(), 1e-30)
    print(f"  terminal-loss spread = {rel * 100:.0f}% of mean")
    if rel < 0.10 and len(distinct) <= max(3, len(runs) // 10):
        print("  -> few distinct optima + tight losses: likely at the global "
              "basin; top-1 is a strong candidate.")
    else:
        print("  -> multiple distinct basins; the top-K below are candidates. "
              "Raise --restarts if the best is still improving.")
    print(f"\n  top-{kk} DISTINCT by in-loop val loss:")
    for j in range(kk):
        r = runs[distinct[j]]
        print(f"    #{j + 1}  val {r['best_val']:.4e}  "
              f"pos {json.dumps(_round_pos(r['best_pos']))}")

    pos_dir = Path(args.pos_dir)
    pos_dir.mkdir(parents=True, exist_ok=True)
    for j in range(kk):
        (pos_dir / f"top{j + 1}.json").write_text(
            json.dumps(_round_pos(runs[distinct[j]]["best_pos"])))
    (pos_dir / "summary.json").write_text(json.dumps(dict(
        restarts=len(runs), distinct=len(distinct), K=int(K), n=int(args.n),
        best_val=float(vals.min()), rel_spread=float(rel),
        top=[dict(rank=j + 1, val=float(vals[distinct[j]]),
                  positions=_round_pos(runs[distinct[j]]["best_pos"]))
             for j in range(kk)]), indent=2))
    print(f"\n  wrote top-{kk} layouts to {pos_dir}/top*.json + summary.json")

    _render(runs, distinct, args.top_k, Path(args.out), args.r_min,
            args.r_max, args.n, K)
    print(f"  wrote {args.out}")

    tk = min(args.train_top_k, kk)
    if tk <= 0:
        print(f"\nNext: retrain a top layout, e.g.\n  python scripts/train.py "
              f"--config {args.config} --data.npz_dir {args.npz_dir} "
              f"--pod.k {K} --sensors.n {args.n} --sensors.strategy custom \\\n"
              f"    --sensors.positions \"$(cat {pos_dir}/top1.json)\" "
              f"--seeds \"[{args.train_seed}]\" --tag m_multistart_top1")
        return 0

    print(f"\n===== training top-{tk} (single seed {args.train_seed}) =====",
          flush=True)
    trained = []
    for j in range(1, tk + 1):
        tag = f"m_multistart_top{j}"
        rj = Path(args.outputs) / tag / "results.json"
        if args.skip_existing and rj.is_file():
            print(f"[skip] {tag}: results.json exists")
            trained.append((tag, j))
            continue
        pos = json.loads((pos_dir / f"top{j}.json").read_text())
        cmd = [PY, "scripts/train.py", "--config", args.config,
               "--data.npz_dir", args.npz_dir,
               "--pod.workers", str(args.pod_workers),
               "--pod.k", str(K), "--sensors.n", str(args.n),
               "--sensors.strategy", "custom",
               "--sensors.positions", json.dumps(pos),
               "--seeds", json.dumps([args.train_seed]), "--tag", tag]
        print(f"\n[train] {tag}  pos={json.dumps(pos)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            trained.append((tag, j))
        except subprocess.CalledProcessError as e:
            print(f"  train FAILED for {tag}: {e}", file=sys.stderr)

    ok = [(t, j) for t, j in trained
          if (Path(args.outputs) / t / "results.json").is_file()]
    if len(ok) >= 2:
        cmp_out = str(Path(args.out).with_name("multistart_compare.png"))
        cmd = [PY, "scripts/compare_placements.py",
               "--tags", *[t for t, _ in ok],
               "--labels", *[f"top{j}" for _, j in ok],
               "--top-n", "20", "--outputs", args.outputs,
               "--out", cmp_out,
               "--out-json", str(Path(cmp_out).with_suffix(".json"))]
        print(f"\n[compare] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False)
    print("\nNote: these top-K models use ONE seed (fast ranking). Retrain the "
          "final winner with the full seed set for the reported number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
