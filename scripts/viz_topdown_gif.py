"""Two-panel top-down GIF: raw native NPZ alongside canonical loader output.

Debug workhorse. The point is to QA the loader / converter on EVERY new
sim by comparing the canonical (Nx, Ny, Nt) field rendered as a heatmap
against the raw native point cloud at the same time step. Catches:

  - Coordinate normalisation bugs (raw on a smaller circle than the
    canonical grid expects -- the bug we hit before with the metres
    vs normalised mistake).
  - drop_first_steps misconfiguration (raw includes the pre-contact
    equilibration step; we render it WITHOUT dropping so the operator
    can see exactly what was discarded vs kept).
  - Sensor placement errors (sensors are overlaid on the canonical
    panel; their (x, y) should land somewhere reasonable).
  - Bonded-front behaviour (a contour at gap_threshold is drawn on the
    canonical panel; should sweep inward over time matching the raw
    displacement evolution).

Per-sim global vmin/vmax (stable across the animation; one sim's red
is comparable to its own blue). Title prints t + front radius. GIF
encoded via matplotlib PillowWriter.

    python scripts/viz_topdown_gif.py --sim /path/to/raw_sim.npz \\
        --out viz/sim_topdown.gif

    # explicit raw + canonical inputs (raw .npz + an already-loaded
    # canonical npz with arrays x_canon, y_canon, f):
    python scripts/viz_topdown_gif.py --raw /path/to/raw.npz \\
        --canonical /path/to/canonical_dump.npz --out viz/x.gif

By default --sim points at a raw NPZ; the loader is invoked
in-process to produce the canonical side. Slow if no loader cache
exists for the directory.
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

from data.loader import load_dataset, preflight_npz           # noqa: E402
from core.grid import canonical_grid                          # noqa: E402
from core.sensors import SensorConfig, place_sensors          # noqa: E402
from scripts.fieldviz import (mirror_d2, render_full_disk,    # noqa: E402
                               shared_diverging_cmap,
                               provenance_footer,
                               compute_bonded_mask,
                               front_radius_per_t)


def _load_one_canonical(raw_path: Path, nx: int, ny: int, nt: int,
                        drop_first_steps: int):
    """Stage a single NPZ into a temp dir + load it via the real loader.

    Forces cache=False so we never write a cache into the user's tree
    (the user runs this on real folders and we don't want surprise
    files appearing)."""
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
        raise RuntimeError(f"loader rejected {raw_path.name} via preflight")
    return x_canon, y_canon, sims[0]


def _native_scatter_frame(coords_xy, values, ax, vmin, vmax,
                          title: str, R: float = 0.15):
    """Scatter raw native data: each (x, y) coloured by its value.

    coords_xy is (N, 2) in metres (raw NPZ convention); values is
    (N,). We render in physical units so the operator can confirm the
    physical scale (metres) directly.
    """
    sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=values, s=2,
                    cmap="RdBu_r", vmin=vmin, vmax=vmax,
                    edgecolors="none")
    # Outline the wafer quadrant in physical units.
    import matplotlib.patches as mpatches
    arc = mpatches.Arc((0, 0), 2 * R, 2 * R, theta1=0, theta2=90,
                       color="k", lw=0.8)
    ax.add_patch(arc)
    ax.plot([0, R], [0, 0], "k-", lw=0.8)
    ax.plot([0, 0], [0, R], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.01 * R, 1.05 * R)
    ax.set_ylim(-0.01 * R, 1.05 * R)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    return sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", help="path to a single raw NPZ; loader is "
                     "invoked in-process for the canonical side")
    src.add_argument("--raw", help="path to raw NPZ (use with --canonical)")
    ap.add_argument("--canonical", help="path to canonical npz with "
                    "arrays x_canon, y_canon, f (only with --raw)")
    ap.add_argument("--out", required=True, help="output GIF path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=0,
                    help="default 0 here (UNLIKE training) -- we want "
                    "the operator to see what was dropped; pass 1 only "
                    "if you specifically want the trimmed view")
    ap.add_argument("--gap-threshold-um", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=120,
                    help="cap frames to keep GIF size manageable; "
                    "evenly-spaced subsample if Nt > max-frames")
    ap.add_argument("--sensors", default="3-edge",
                    help="'3-edge' = lab rig (X/+D/Y) or comma list "
                    "'r1:th1,r2:th2,...' in normalised + deg")
    ap.add_argument("--tag", default=None,
                    help="optional tag string for the provenance footer")
    args = ap.parse_args()

    # --- load canonical + (optionally) raw coords ---
    if args.sim:
        raw_path = Path(args.sim).expanduser().resolve()
        x_canon, y_canon, sim = _load_one_canonical(
            raw_path, args.nx, args.ny, args.nt, args.drop_first_steps)
        # also load native step_0000 coords for the raw side
        with np.load(raw_path, allow_pickle=True) as z:
            coords_native = z["step_0000_coordinates_upper"][:2, :].T   # (N, 2)
            disp_native = np.asarray(
                z["step_0000_displacement_z_corrected_upper"])
        sim_id = raw_path.stem
    else:
        if not args.canonical:
            print("--raw requires --canonical", file=sys.stderr)
            return 2
        raw_path = Path(args.raw).expanduser().resolve()
        canon_path = Path(args.canonical).expanduser().resolve()
        with np.load(canon_path, allow_pickle=True) as z:
            x_canon = np.asarray(z["x_canon"])
            y_canon = np.asarray(z["y_canon"])
            f = np.asarray(z["f"])
        from core.simulation import Simulation
        sim = Simulation(f=f.astype(np.float32), params={})
        with np.load(raw_path, allow_pickle=True) as z:
            coords_native = z["step_0000_coordinates_upper"][:2, :].T
            disp_native = np.asarray(
                z["step_0000_displacement_z_corrected_upper"])
        sim_id = raw_path.stem

    nx, ny, nt = sim.f.shape
    print(f"loaded {sim_id}: canonical {sim.f.shape}  "
          f"native step_0000 {disp_native.shape}", flush=True)

    # --- sensor positions ---
    if args.sensors == "3-edge":
        positions = ((1.0, 0.0), (1.0, 45.0), (1.0, 90.0))
    else:
        positions = tuple(
            tuple(float(x) for x in p.split(":"))
            for p in args.sensors.split(","))
    scfg = SensorConfig(n=len(positions), strategy="custom",
                        positions=positions)
    sensor_xy = place_sensors(scfg)   # canonical [0,1] coords
    # Mirror to full-disk coords for display (the canonical render
    # mirrors the field; sensors should land in the right quadrant).
    # Sensors are already in Q1 so mirror leaves them in place; just
    # use as-is.

    # --- bonded mask + front radius for overlay ---
    bonded = compute_bonded_mask(
        sim.f.astype(np.float64),
        gap_threshold_um=args.gap_threshold_um)
    front_r = front_radius_per_t(bonded, x_canon, y_canon)

    # --- per-sim global vmin/vmax (one colour scale across the GIF) ---
    vmin, vmax = shared_diverging_cmap(sim.f, symmetric=True)

    # --- frame subsampling ---
    if nt > args.max_frames:
        frame_idx = np.linspace(0, nt - 1, args.max_frames).astype(int)
    else:
        frame_idx = np.arange(nt)

    # --- bonded contour: do it on the FULL disk via mirror so the
    # contour reads as a closed curve ---
    bonded_mirrored = mirror_d2(bonded.astype(np.float32))
    x_full, y_full = (np.concatenate([-x_canon[:0:-1], x_canon]),
                      np.concatenate([-y_canon[:0:-1], y_canon]))

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                             constrained_layout=True)
    # Native side colour scale: in physical meters too, same per-sim limits
    nvmin = float(disp_native.min())
    nvmax = float(disp_native.max())
    nlim = max(abs(nvmin), abs(nvmax))
    sc_native = _native_scatter_frame(
        coords_native, disp_native[0], axes[0], -nlim, nlim,
        title=f"raw native (step_0000)  t-idx 0", R=0.15)
    cbar_n = fig.colorbar(sc_native, ax=axes[0], shrink=0.85,
                          label="displacement (m)")

    im_canon = render_full_disk(
        axes[1], sim.f[..., frame_idx[0]], x_canon, y_canon,
        vmin=vmin, vmax=vmax, mirror=True, mask_off_disk=True,
        sensor_xy=sensor_xy)
    axes[1].set_xlabel("x (normalised)")
    axes[1].set_ylabel("y (normalised)")
    axes[1].set_title(f"canonical loader output  t-idx {frame_idx[0]}")
    cbar_c = fig.colorbar(im_canon, ax=axes[1], shrink=0.85,
                          label="u_z (m)")

    # Initial bonded contour
    contour_handles = []

    def _redraw_contour(t_idx):
        for h in contour_handles:
            try:
                for c in h.collections:
                    c.remove()
            except Exception:
                pass
        contour_handles.clear()
        if bonded_mirrored[..., t_idx].any():
            cs = axes[1].contour(
                x_full, y_full, bonded_mirrored[..., t_idx].T,
                levels=[0.5], colors="lime", linewidths=1.5)
            contour_handles.append(cs)

    _redraw_contour(frame_idx[0])
    fig.suptitle(f"{sim_id}  |  drop_first_steps={args.drop_first_steps}  "
                 f"|  per-sim shared colour scale",
                 fontsize=10)

    n_native_t = disp_native.shape[0]

    def update(i):
        t_idx = int(frame_idx[i])
        # canonical
        F = sim.f[..., t_idx]
        F_full = mirror_d2(F)
        X, Y = np.meshgrid(x_full, y_full, indexing="ij")
        F_full = F_full.astype(np.float64, copy=True)
        F_full[(X * X + Y * Y) > 1.0] = np.nan
        im_canon.set_data(F_full.T)
        front_text = (f"front_r={front_r[t_idx]:.2f}"
                      if np.isfinite(front_r[t_idx]) else "front_r=--")
        axes[1].set_title(
            f"canonical  t-idx {t_idx}/{nt - 1}  {front_text}")
        _redraw_contour(t_idx)
        # native: pick the closest native time index from step_0000
        # by index ratio (we don't have aligned time bases between
        # native step_0000 and the canonical [0, 1] axis here -- the
        # canonical t-idx already includes ALL steps post-trim, so we
        # just visualise step_0000 in step-local time, capped)
        n_idx = min(int(round(t_idx / (nt - 1) * (n_native_t - 1))),
                    n_native_t - 1)
        sc_native.set_array(disp_native[n_idx])
        axes[0].set_title(
            f"raw native (step_0000)  step-local t-idx {n_idx}")
        return [im_canon, sc_native]

    print(f"rendering {len(frame_idx)} frames at {args.fps} fps "
          f"-> {args.out}", flush=True)
    anim = FuncAnimation(fig, update, frames=len(frame_idx),
                         interval=1000 // args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    provenance_footer(fig, sim_id=sim_id, tag=args.tag,
                      extras={"drop": args.drop_first_steps,
                              "gap_um": args.gap_threshold_um})
    anim.save(args.out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
