"""Top-down GIF of the canonical wafer-bonding field over time.

Default: a single full-disk heatmap that morphs through the
trajectory, with sensors overlaid and a bonded-region contour drawn
in orange (distinct from WAFER_CMAP's purple end). Per-frame colour
normalisation by default so the within-frame bonded-region shape
stays visible at every time step -- the curved lower-wafer profile
emerges as the front sweeps. Pass --norm-mode per-sim to lock the
cmap range across the animation when cross-frame amplitude
comparison matters.

The optional --include-raw flag adds a left panel that scatters the
raw native NPZ's first-step displacement (debugging only; the first
step is the converter's pre-contact equilibration that drop_first_steps
discards, so it is mostly near-zero).

    python scripts/viz_topdown_gif.py --sim /path/to/raw_sim.npz \\
        --out viz/sim_topdown.gif

    # add the raw debug panel
    ... --include-raw

    # lock the colour scale across the whole animation
    ... --norm-mode per-sim

In-process API (used by scripts/viz_all.py): import
`render_topdown_gif(sim, x_canon, y_canon, sensor_xy, out_path, ...)`
to avoid the per-call subprocess + loader cost.
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

from data.loader import load_dataset                          # noqa: E402
from core.sensors import SensorConfig, place_sensors          # noqa: E402
from core.simulation import Simulation                         # noqa: E402
from scripts.fieldviz import (mirror_d2, render_full_disk,    # noqa: E402
                               wafer_value_range,
                               provenance_footer,
                               compute_bonded_mask,
                               front_radius_per_t,
                               WAFER_CMAP, SENSOR_PALETTE)


# Bonded-front contour colour. Orange (SENSOR_PALETTE[6] = #E16A13) is
# distinct from every value in WAFER_CMAP (which spans purple ->
# turquoise -> cyan -> green -> yellow), so the contour stays visible
# both on near-zero (yellow) cells and on deep-bonded (purple) cells.
_FRONT_COLOR = SENSOR_PALETTE[6]


def _load_one_canonical(raw_path: Path, nx: int, ny: int, nt: int,
                        drop_first_steps: int):
    """Stage one NPZ into a tempdir + load via the real loader.

    cache=False so this never writes a multi-GB cache into the user's
    NPZ folder by accident.
    """
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


def render_topdown_gif(sim: Simulation, x_canon: np.ndarray,
                       y_canon: np.ndarray,
                       sensor_xy: np.ndarray,
                       out_path: Path | str, *,
                       gap_threshold_um: float = 1.0,
                       norm_mode: str = "per-frame",
                       fps: int = 24,
                       max_frames: int = 120,
                       sim_id: str | None = None,
                       tag: str | None = None,
                       drop_first_steps: int | None = None,
                       include_raw: bool = False,
                       raw_coords: np.ndarray | None = None,
                       raw_disp: np.ndarray | None = None) -> Path:
    """Render the top-down animation. Pure rendering -- caller supplies
    the already-loaded sim and sensor positions, so viz_all.py can
    invoke this without paying the loader cost again per sim.

    Returns the output Path on success.
    """
    nx, ny, nt = sim.f.shape
    bonded = compute_bonded_mask(sim.f.astype(np.float64),
                                  gap_threshold_um=gap_threshold_um)
    front_r = front_radius_per_t(bonded, x_canon, y_canon)

    # Per-sim limits used as fallback / for the colorbar legend
    sim_vmin, sim_vmax = wafer_value_range(sim.f)

    # Frame subsample
    if nt > max_frames:
        frame_idx = np.linspace(0, nt - 1, max_frames).astype(int)
    else:
        frame_idx = np.arange(nt)

    # Bonded mask mirrored to full disk for the contour overlay.
    bonded_mirrored = mirror_d2(bonded.astype(np.float32))
    x_full = np.concatenate([-x_canon[:0:-1], x_canon])
    y_full = np.concatenate([-y_canon[:0:-1], y_canon])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if include_raw and raw_coords is not None and raw_disp is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                                 constrained_layout=True)
        ax_canon = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(6.4, 5.6),
                                constrained_layout=True)
        axes = [None, ax]                            # unify indexing
        ax_canon = ax

    # --- canonical side ---
    if norm_mode == "per-sim":
        vmin0, vmax0 = sim_vmin, sim_vmax
    else:
        vmin0, vmax0 = wafer_value_range(sim.f[..., frame_idx[0]])
    im_canon = render_full_disk(
        ax_canon, sim.f[..., frame_idx[0]], x_canon, y_canon,
        vmin=vmin0, vmax=vmax0, mirror=True, mask_off_disk=True,
        sensor_xy=sensor_xy)
    ax_canon.set_xlabel("x (normalised)")
    ax_canon.set_ylabel("y (normalised)")
    ax_canon.set_title(f"canonical  t-idx {int(frame_idx[0])}")
    cbar_c = fig.colorbar(im_canon, ax=ax_canon, shrink=0.85,
                          label="u_z (m)")

    # initial bonded-region contour
    contour_handles: list = []

    def _redraw_contour(t_idx: int) -> None:
        for h in contour_handles:
            try:
                for c in h.collections:
                    c.remove()
            except Exception:
                pass
        contour_handles.clear()
        if bonded_mirrored[..., t_idx].any():
            cs = ax_canon.contour(
                x_full, y_full, bonded_mirrored[..., t_idx].T,
                levels=[0.5], colors=[_FRONT_COLOR],
                linewidths=1.7)
            contour_handles.append(cs)

    _redraw_contour(int(frame_idx[0]))

    # --- optional raw side ---
    sc_native = None
    if include_raw and raw_coords is not None and raw_disp is not None:
        ax_raw = axes[0]
        nvmin = float(raw_disp.min())
        nvmax = min(float(raw_disp.max()), 0.0)
        if nvmax <= nvmin:
            nvmax = nvmin + 1.0
        import matplotlib.patches as mpatches
        R = 0.15
        sc_native = ax_raw.scatter(
            raw_coords[:, 0], raw_coords[:, 1], c=raw_disp[0], s=2,
            cmap=WAFER_CMAP, vmin=nvmin, vmax=nvmax, edgecolors="none")
        arc = mpatches.Arc((0, 0), 2 * R, 2 * R, theta1=0, theta2=90,
                            color="k", lw=0.8)
        ax_raw.add_patch(arc)
        ax_raw.plot([0, R], [0, 0], "k-", lw=0.8)
        ax_raw.plot([0, 0], [0, R], "k-", lw=0.8)
        ax_raw.set_aspect("equal")
        ax_raw.set_xlim(-0.01 * R, 1.05 * R)
        ax_raw.set_ylim(-0.01 * R, 1.05 * R)
        ax_raw.set_xlabel("x (m)  -- raw NPZ, step_0000")
        ax_raw.set_ylabel("y (m)")
        ax_raw.set_title(
            "raw native (step_0000 = pre-contact, mostly ~0)")
        fig.colorbar(sc_native, ax=ax_raw, shrink=0.85,
                      label="displacement (m)")

    fig.suptitle(
        f"{sim_id or 'sim'}  |  norm_mode={norm_mode}  |  "
        f"drop_first_steps={drop_first_steps}", fontsize=10)

    n_native_t = raw_disp.shape[0] if (include_raw and raw_disp is not None) else 0

    def update(i):
        t_idx = int(frame_idx[i])
        F = sim.f[..., t_idx]
        if norm_mode == "per-frame":
            vmin_t, vmax_t = wafer_value_range(F)
            im_canon.set_clim(vmin=vmin_t, vmax=vmax_t)
        F_full = mirror_d2(F)
        X, Y = np.meshgrid(x_full, y_full, indexing="ij")
        F_full = F_full.astype(np.float64, copy=True)
        F_full[(X * X + Y * Y) > 1.0] = np.nan
        im_canon.set_data(F_full.T)
        front_text = (f"front_r={front_r[t_idx]:.2f}"
                      if np.isfinite(front_r[t_idx]) else "front_r=--")
        ax_canon.set_title(
            f"canonical  t-idx {t_idx}/{nt - 1}  {front_text}")
        _redraw_contour(t_idx)
        artists = [im_canon]
        if sc_native is not None and n_native_t > 0:
            n_idx = min(int(round(t_idx / max(nt - 1, 1) * (n_native_t - 1))),
                        n_native_t - 1)
            sc_native.set_array(raw_disp[n_idx])
            axes[0].set_title(
                f"raw native (step_0000)  step-local t-idx {n_idx}")
            artists.append(sc_native)
        return artists

    print(f"rendering {len(frame_idx)} frames at {fps} fps -> {out_path}",
          flush=True)
    anim = FuncAnimation(fig, update, frames=len(frame_idx),
                         interval=1000 // fps, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      extras={"drop": drop_first_steps,
                              "gap_um": gap_threshold_um,
                              "norm": norm_mode})
    anim.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)
    return Path(out_path)


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
                    help="default 0 here (UNLIKE training) so the "
                    "pre-contact equilibration step is visible. Pass 1 "
                    "to match training's trim.")
    ap.add_argument("--gap-threshold-um", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--norm-mode",
                    choices=("per-frame", "per-sim"),
                    default="per-frame",
                    help="per-frame (default): each frame uses its own "
                    "vmin/vmax so the curved bonded shape stays visible "
                    "at every time. per-sim: lock the cmap range across "
                    "the animation -- use when cross-frame amplitude "
                    "comparison matters.")
    ap.add_argument("--include-raw", action="store_true",
                    help="add a second panel that scatters the raw NPZ's "
                    "step_0000 displacement (debug only; step_0000 is "
                    "the pre-contact equilibration step that is mostly "
                    "near-zero, hence off by default)")
    ap.add_argument("--sensors", default="3-edge",
                    help="'3-edge' = lab rig (X/+D/Y) or comma list "
                    "'r1:th1,r2:th2,...' in normalised + deg")
    ap.add_argument("--tag", default=None,
                    help="optional tag string for the provenance footer")
    args = ap.parse_args()

    # --- load canonical + (optionally) raw coords ---
    raw_coords = raw_disp = None
    if args.sim:
        raw_path = Path(args.sim).expanduser().resolve()
        x_canon, y_canon, sim = _load_one_canonical(
            raw_path, args.nx, args.ny, args.nt, args.drop_first_steps)
        if args.include_raw:
            with np.load(raw_path, allow_pickle=True) as z:
                raw_coords = z["step_0000_coordinates_upper"][:2, :].T
                raw_disp = np.asarray(
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
        sim = Simulation(f=f.astype(np.float32), params={})
        if args.include_raw:
            with np.load(raw_path, allow_pickle=True) as z:
                raw_coords = z["step_0000_coordinates_upper"][:2, :].T
                raw_disp = np.asarray(
                    z["step_0000_displacement_z_corrected_upper"])
        sim_id = raw_path.stem

    nx, ny, nt = sim.f.shape
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

    render_topdown_gif(
        sim, x_canon, y_canon, sensor_xy, args.out,
        gap_threshold_um=args.gap_threshold_um,
        norm_mode=args.norm_mode,
        fps=args.fps, max_frames=args.max_frames,
        sim_id=sim_id, tag=args.tag,
        drop_first_steps=args.drop_first_steps,
        include_raw=args.include_raw,
        raw_coords=raw_coords, raw_disp=raw_disp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
