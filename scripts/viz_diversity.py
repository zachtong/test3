"""Cross-sim STD visualization: how diverse is a folder of NPZ sims?

Computes std across simulations of the canonical (Nx, Ny, Nt)
displacement field via Welford's one-pass mean-variance algorithm
(memory cost = 2 buffer of size (Nx, Ny, Nt) ~80 MB at 128x128x300
float64; independent of the simulation count).

Two views of the result, one figure:

  TOP ROW (3 panels): std kymographs at theta = 0, 45, 90 deg.
    For each angle, sample std(x, y, t) along that radial ray, then
    plot std vs (r, t). Viridis (unsigned). Same x and y axes as the
    canonical viz_radial_kymograph so the operator can A/B the
    diversity image against any one sim's hero kymograph.

  BOTTOM ROW (3 panels): top-down snapshots of std(x, y, t) at
    t = 0.1, 0.5, 0.9 of the normalized trajectory. Full disk
    (D2-mirrored). Viridis. Tells the operator WHERE in space the
    sample-to-sample variance is concentrated and how it migrates
    over time.

The bottom row's three panels use a SHARED viridis color scale so
the eye can compare "is the dataset's variance front-dominated /
edge-dominated / center-dominated, and does that change over time?"

Use --limit N to subsample the folder while iterating.

    python scripts/viz_diversity.py --npz-dir /path/to/3d_npz \\
        --out viz/diversity.png

    # quick preview from a subset
    python scripts/viz_diversity.py --npz-dir /path/to/3d_npz \\
        --limit 200 --out viz/diversity_n200.png
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.loader import load_dataset                         # noqa: E402
from scripts.fieldviz import (mirror_d2, render_full_disk,    # noqa: E402
                               provenance_footer)


# Cache-file prefix; matches the loader's `_` convention so viz_all's
# _pick_sims globbing skips it. Version integer bumps whenever the
# schema (which arrays are stored) changes so old caches auto-invalidate.
_STATS_CACHE_PREFIX = "_diversity_stats_"
_STATS_CACHE_VERSION = 1


def _stats_cache_key(folder: Path, nx: int, ny: int, nt: int,
                     drop_first_steps: int, limit: int | None) -> str:
    """8-char hash uniquely identifying a (folder, grid, drop, limit)
    diversity-stats build. Folder is the ABSOLUTE resolved path, so
    moving the data invalidates the cache automatically."""
    key = json.dumps({
        "version": _STATS_CACHE_VERSION,
        "folder": str(folder.resolve()),
        "nx": nx, "ny": ny, "nt": nt,
        "drop_first_steps": int(drop_first_steps),
        "limit": None if limit is None else int(limit),
    }, sort_keys=True).encode()
    return hashlib.sha256(key).hexdigest()[:8]


def _stats_cache_path(folder: Path, nx: int, ny: int, nt: int,
                      drop_first_steps: int,
                      limit: int | None) -> Path:
    h = _stats_cache_key(folder, nx, ny, nt, drop_first_steps, limit)
    return folder / f"{_STATS_CACHE_PREFIX}{h}.npz"


def _load_stats_cache(path: Path):
    """Return (mean, var, n_eff, x_canon, y_canon) on hit, None on
    miss (including corruption). Version tag is verified explicitly."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d.get("version", -1)) != _STATS_CACHE_VERSION:
                return None
            return (d["mean"].astype(np.float64),
                    d["var"].astype(np.float64),
                    int(d["n_eff"]),
                    d["x_canon"].astype(np.float64),
                    d["y_canon"].astype(np.float64))
    except (OSError, ValueError, KeyError) as e:
        print(f"  diversity cache read failed ({type(e).__name__}: {e}) "
              f"-- rebuilding", flush=True)
        return None


def _save_stats_cache(path: Path, mean: np.ndarray, var: np.ndarray,
                      n_eff: int, x_canon: np.ndarray,
                      y_canon: np.ndarray) -> None:
    """Atomic write via .tmp -> rename so a crash mid-save leaves
    the previous cache intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp,
             version=np.int32(_STATS_CACHE_VERSION),
             mean=mean.astype(np.float32),
             var=var.astype(np.float32),
             n_eff=np.int32(n_eff),
             x_canon=x_canon.astype(np.float64),
             y_canon=y_canon.astype(np.float64))
    tmp.replace(path)


def _welford_mean_var(sims, verbose: bool = True):
    """One-pass mean + variance over the simulation axis.

    Returns (mean_xyt, var_xyt, count). Each sim's f is (Nx, Ny, Nt)
    float32. Accumulators are float64 so summing 5000+ sims of small
    displacements stays well-conditioned.
    """
    if not sims:
        raise ValueError("no sims")
    nx, ny, nt = sims[0].f.shape
    mean = np.zeros((nx, ny, nt), dtype=np.float64)
    M2 = np.zeros((nx, ny, nt), dtype=np.float64)
    n = 0
    t0 = time.time()
    last = [0.0]
    for i, s in enumerate(sims, 1):
        n += 1
        x = s.f.astype(np.float64)
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        now = time.time()
        if verbose and (now - last[0] >= 5.0 or i == len(sims)):
            rate = i / max(now - t0, 1e-9)
            print(f"  Welford: {i}/{len(sims)} sims  ({rate:.1f}/s)",
                  flush=True)
            last[0] = now
    var = M2 / max(n - 1, 1)
    return mean, var, n


def _sample_radial_kymograph(f: np.ndarray, x_canon: np.ndarray,
                             y_canon: np.ndarray, theta_deg: float,
                             n_r: int = 128) -> np.ndarray:
    """Bilinear-sample f(x, y, t) along the radial ray at theta_deg.

    Identical to the helper in viz_radial_kymograph.py; kept locally
    to avoid a cross-script import."""
    nx, ny, nt = f.shape
    rs = np.linspace(0.0, 1.0, n_r)
    t = np.deg2rad(theta_deg)
    xs = rs * np.cos(t)
    ys = rs * np.sin(t)
    ix = np.interp(xs, x_canon, np.arange(nx))
    iy = np.interp(ys, y_canon, np.arange(ny))
    ix0 = np.clip(np.floor(ix).astype(int), 0, nx - 2)
    iy0 = np.clip(np.floor(iy).astype(int), 0, ny - 2)
    dx = ix - ix0
    dy = iy - iy0
    a00 = f[ix0, iy0, :]
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
    ap.add_argument("--npz-dir", required=True,
                    help="folder of converted 3D NPZ files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample to first N sims for a quick preview "
                    "(default: use all)")
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--angles", default="0,45,90")
    ap.add_argument("--snapshot-times", default="0.1,0.5,0.9",
                    help="comma list of normalized t in [0, 1] for "
                    "the bottom-row top-down snapshots")
    ap.add_argument("--value-scale", type=float, default=1.0e6,
                    help="multiply std for display (1e6 m -> um)")
    ap.add_argument("--workers", type=int, default=None,
                    help="loader workers (defaults to loader auto)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any pre-computed diversity-stats cache "
                    "and always redo the full 93 GB load + Welford. "
                    "Use this when you suspect the cache is stale.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild AND overwrite the diversity-stats "
                    "cache even on a valid hit. Implies --no-cache.")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    folder = Path(args.npz_dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"not a directory: {folder}", file=sys.stderr)
        return 2
    angles = [float(a) for a in args.angles.split(",")]
    snap_times = [float(t) for t in args.snapshot_times.split(",")]

    # Try the diversity-stats cache first -- it holds the (mean, var,
    # n_eff, x_canon, y_canon) tensors for this (folder, grid, drop,
    # limit) combination. A hit skips both the 93 GB loader read AND
    # the Welford pass. See _stats_cache_key for what identifies a
    # cache; moving the folder, changing nx/ny/nt/drop/limit all
    # invalidate. If you know the cache is stale, pass --no-cache
    # (or --force to also overwrite the file after the rebuild).
    cache_path = _stats_cache_path(folder, args.nx, args.ny, args.nt,
                                     args.drop_first_steps, args.limit)
    hit = (None if (args.no_cache or args.force)
           else _load_stats_cache(cache_path))
    if hit is not None:
        mean, var, n_eff, x_canon, y_canon = hit
        print(f"diversity cache HIT -> {cache_path.name}  "
              f"(n_eff={n_eff}, shape={mean.shape})", flush=True)
    else:
        if cache_path.exists() and args.no_cache:
            print(f"  --no-cache: skipping {cache_path.name}", flush=True)
        print(f"loading sims from {folder} (limit={args.limit}, "
              f"drop_first_steps={args.drop_first_steps}) ...",
              flush=True)
        t_load = time.time()
        x_canon, y_canon, sims = load_dataset(
            folder, nx=args.nx, ny=args.ny, nt=args.nt,
            cache=True, limit=args.limit,
            workers=args.workers,
            drop_first_steps=args.drop_first_steps)
        print(f"  loaded {len(sims)} sims  shape per sim "
              f"{sims[0].f.shape}  in {time.time() - t_load:.1f}s",
              flush=True)
        print(f"computing Welford mean+var over {len(sims)} sims ...",
              flush=True)
        mean, var, n_eff = _welford_mean_var(sims)
        try:
            _save_stats_cache(cache_path, mean, var, n_eff,
                              x_canon, y_canon)
            print(f"  wrote diversity cache -> {cache_path.name} "
                  f"(~{(cache_path.stat().st_size >> 20)} MB)",
                  flush=True)
        except OSError as e:
            print(f"  WARN: could not write diversity cache "
                  f"({type(e).__name__}: {e}); continuing anyway",
                  flush=True)

    std = np.sqrt(var) * args.value_scale
    nx, ny, nt = std.shape
    print(f"  std field shape {std.shape}  "
          f"median {float(np.median(std)):.3g}  "
          f"p99 {float(np.percentile(std, 99)):.3g}", flush=True)

    # Sample radial kymographs from the std field.
    kymos = []
    for th in angles:
        ky = _sample_radial_kymograph(std, x_canon, y_canon, th, n_r=nx)
        kymos.append(ky)
    # Snapshot t-indices
    snap_idx = [int(round(t * (nt - 1))) for t in snap_times]
    snap_idx = [max(0, min(nt - 1, k)) for k in snap_idx]

    # Shared vmax across all panels (per-dataset). Use 99th percentile
    # of the full std volume so a single outlier cell does not own the
    # scale. vmin is 0 (std is non-negative).
    vmax = float(np.percentile(std, 99))
    if vmax == 0:
        vmax = 1.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, max(len(angles), len(snap_idx)),
                             figsize=(4.0 * max(len(angles),
                                                len(snap_idx)),
                                       8.0),
                             constrained_layout=True)

    t_axis = np.linspace(0.0, 1.0, nt)
    r_axis = np.linspace(0.0, 1.0, nx)

    # TOP ROW: std kymographs at each angle
    for c, (th, ky) in enumerate(zip(angles, kymos)):
        ax = axes[0, c]
        im = ax.imshow(ky, origin="lower", aspect="auto",
                       extent=[t_axis[0], t_axis[-1],
                               r_axis[0], r_axis[-1]],
                       vmin=0, vmax=vmax, cmap="viridis",
                       interpolation="nearest")
        ax.set_title(f"std kymograph @ theta={th:g} deg", fontsize=10)
        ax.set_xlabel("normalized time")
        if c == 0:
            ax.set_ylabel("r (normalized)")
    # Single colorbar for the top row
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.85,
                 location="right",
                 label=f"std across sims  (u_z * {args.value_scale:g})")

    # BOTTOM ROW: top-down std snapshots
    for c, (tn, ti) in enumerate(zip(snap_times, snap_idx)):
        ax = axes[1, c]
        slice_ = std[..., ti]
        im_b = render_full_disk(ax, slice_, x_canon, y_canon,
                                cmap="viridis", vmin=0, vmax=vmax,
                                mirror=True, mask_off_disk=True)
        ax.set_title(f"std(x, y)  t={tn:.2g}  (t-idx {ti}/{nt - 1})",
                     fontsize=10)
        ax.set_xlabel("x")
        if c == 0:
            ax.set_ylabel("y")
    # hide any unused cols in either row
    for r in (0, 1):
        for c in range(max(len(angles), len(snap_idx))):
            if (r == 0 and c >= len(angles)) or (r == 1 and c >= len(snap_idx)):
                axes[r, c].set_visible(False)

    fig.colorbar(im_b, ax=axes[1, :].tolist(), shrink=0.85,
                 location="right",
                 label=f"std across sims  (u_z * {args.value_scale:g})")

    fig.suptitle(f"cross-sim diversity  |  folder: {folder.name}  |  "
                 f"n={n_eff} sims  |  drop_first_steps="
                 f"{args.drop_first_steps}", fontsize=11)
    provenance_footer(fig, tag=args.tag,
                      extras={"n": n_eff, "drop": args.drop_first_steps,
                              "limit": args.limit or "all"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
