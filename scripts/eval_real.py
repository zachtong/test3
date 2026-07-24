"""Minimal real-data evaluation for the 3D model.

Feed a real experiment through a self-contained bundle (scripts/bundle.py) and
reconstruct the full quarter-disk field. The input is either the tool-exported
CSV (`.csv`, parsed like the 2D GUI) or a pre-built SI NPZ (`.npz`). There is
NO ground truth, so this:

  1. ingests the six directional sensors (CSV -> w_<label> in metres, or NPZ)
     and assembles them into the model's ABCDEF channels (fold + match by
     (r, theta), no averaging), then reconstructs w(x, y, t);
  2. runs CHECKPOINT 1 -- the physical sanity gate (units, sign, magnitude,
     positions) that catches nm/um/m and sign-convention errors on the first
     real run;
  3. visualizes the raw traces + truncation window (so you can pick t_start /
     t_cutoff) and the reconstructed field, mirrored to the full disk.

    python scripts/eval_real.py \\
        --bundle bundles/abcdef_k12.pt \\
        --real run01.csv \\
        --config configs/real_exp_n6.yaml \\
        --t-cutoff 8.0 --out-dir viz/real_eval/run01
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.reconstruct import load_bundle, reconstruct_field   # noqa: E402
from data.real_experiment import (                               # noqa: E402
    Channel, RealExperimentConfig, real_config_from_yaml,
    load_real_npz, assemble_inputs, fold_theta)
from data.csv_ingest import csv_to_raw_dict                      # noqa: E402

_UM = 1.0e6                                    # metres -> microns for display


def _default_config() -> RealExperimentConfig:
    """The six directional channels with physical azimuths (folded on use)."""
    chans = (Channel("w_XM", 0.078, fold_theta(180)),
             Channel("w_DM", 0.078, fold_theta(-45)),
             Channel("w_YM", 0.078, fold_theta(90)),
             Channel("w_XE", 0.127, fold_theta(180)),
             Channel("w_DE", 0.127, fold_theta(-45)),
             Channel("w_YE", 0.127, fold_theta(90)))
    return RealExperimentConfig(R=0.15, channels=chans, t_cutoff=13.0)


def _load_raw(path: str) -> dict:
    """Real input: a tool CSV (.csv) or a pre-built SI NPZ (.npz) -- both give
    `{time, w_<label>}`."""
    return (csv_to_raw_dict(path) if str(path).lower().endswith(".csv")
            else load_real_npz(path))


def _apply_window(cfg, raw, t_start, t_cutoff):
    """Apply --t-start / --t-cutoff overrides, then clamp the window into the
    data's actual time range so a stale config value never fails validation on
    the first run. Returns the (possibly adjusted) config."""
    t = np.asarray(raw[cfg.time_key], dtype=float)
    lo, hi = float(t.min()), float(t.max())
    ts = cfg.t_start if t_start is None else float(t_start)
    tc = cfg.t_cutoff if t_cutoff is None else float(t_cutoff)
    ts_c, tc_c = max(ts, lo), min(tc, hi)
    if not (ts_c < tc_c):
        ts_c, tc_c = lo, hi
    if (ts_c, tc_c) != (ts, tc):
        print(f"  window [{ts:g}, {tc:g}] clamped to data range "
              f"[{lo:.4g}, {hi:.4g}] -> [{ts_c:.4g}, {tc_c:.4g}]; set "
              f"--t-start/--t-cutoff off real_inputs.png to pick the bonding "
              f"event", flush=True)
    return dataclasses.replace(cfg, t_start=ts_c, t_cutoff=tc_c)


def checkpoint1(bundle, y, w, cfg) -> list:
    """Physical sanity checks. Each is (name, severity, ok, detail);
    'critical' must pass, 'advisory' is a heads-up (sign / units)."""
    out = []
    rt = np.asarray(bundle["sensor_rtheta"], dtype=float)
    out.append(("sensor r in [0,1]", "critical",
                bool((rt[:, 0] >= 0).all() and (rt[:, 0] <= 1).all()),
                f"r={np.round(rt[:, 0], 3).tolist()}"))
    out.append(("sensor theta in [0,90]", "critical",
                bool((rt[:, 1] >= -1e-6).all() and (rt[:, 1] <= 90 + 1e-6).all()),
                f"theta={np.round(rt[:, 1], 1).tolist()}"))
    out.append(("R = 0.15 m", "advisory", abs(cfg.R - 0.15) < 1e-9,
                f"R={cfg.R}"))
    out.append(("inputs finite", "critical", bool(np.isfinite(y).all()),
                f"y shape {y.shape}"))
    ylo, yhi = float(y.min()) * _UM, float(y.max()) * _UM
    out.append(("inputs ~micron scale", "advisory",
                1e-3 < max(abs(ylo), abs(yhi)) < 1e3,
                f"y in [{ylo:.3f}, {yhi:.3f}] um -- if ~1e4 the input is in nm "
                f"not m; if ~1e-3 it may be double-scaled"))
    frac_neg = float((y < 0).mean())
    out.append(("inputs mostly downward (negative)", "advisory",
                frac_neg > 0.5,
                f"{frac_neg * 100:.0f}% negative -- if positive, set sign=-1.0 "
                f"in the config"))
    out.append(("field finite", "critical", bool(np.isfinite(w).all()),
                f"w shape {w.shape}"))
    wmax = float(np.abs(w).max())
    out.append(("field ~micron scale", "advisory", 1e-9 < wmax < 1e-3,
                f"max|w|={wmax:.2e} m = {wmax * _UM:.3f} um"))
    return out


def _print_checkpoint(checks) -> bool:
    print("\n===== Checkpoint 1 (physical sanity) =====")
    all_critical_ok = True
    for name, sev, ok, detail in checks:
        mark = "OK  " if ok else ("FAIL" if sev == "critical" else "WARN")
        if sev == "critical" and not ok:
            all_critical_ok = False
        print(f"  [{mark}] {name:<32} {detail}")
    print(f"\n  -> {'PASS' if all_critical_ok else 'CRITICAL CHECK FAILED'}"
          f"  (advisory WARN lines are for you to verify, not hard errors)")
    return all_critical_ok


def _mirror_full(w, x, y):
    """Mirror the quarter-disk field (Nx, Ny, Nt) into the full disk via the
    x- and y-axis symmetry the model assumes."""
    nx, ny = w.shape[0], w.shape[1]
    ixq = np.abs(np.arange(2 * nx - 1) - (nx - 1))
    iyq = np.abs(np.arange(2 * ny - 1) - (ny - 1))
    wf = w[ixq][:, iyq]                                    # (2nx-1, 2ny-1, Nt)
    xf = np.concatenate([-x[:0:-1], x])
    yf = np.concatenate([-y[:0:-1], y])
    return wf, xf, yf


def _viz_raw_overview(raw, cfg, out_path):
    """All present sensor traces over the FULL record, with the [t_start,
    t_cutoff] window shaded -- the aid for picking t_start / t_cutoff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(raw[cfg.time_key], dtype=float)
    keys = sorted(k for k in raw if k != cfg.time_key and str(k).startswith("w_"))
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.axvspan(cfg.t_start, cfg.t_cutoff, color="0.8", alpha=0.45, zorder=0,
               label=f"window [{cfg.t_start:.3g}, {cfg.t_cutoff:.3g}] s")
    for k in keys:
        ax.plot(t, np.asarray(raw[k], dtype=float) * _UM, lw=1.3, label=k)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sensor u_z (um)")
    ax.set_title("raw sensor traces + truncation window "
                 "(set --t-start/--t-cutoff from here)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _frame_idx(nt, max_frames):
    if nt <= max_frames:
        return np.arange(nt)
    return np.unique(np.linspace(0, nt - 1, max_frames).astype(int))


def _disp_scale(w, inq):
    """Data-driven z range for the (downward) displacement, in microns. NOT
    symmetric about 0: z_low is the deepest descent (robust min), z_high is
    anchored at 0 (undisplaced) unless a real positive rebound (gas-bulge)
    pushes it above. The WAFER_CMAP then maps z_low -> purple, z_high -> yellow.
    Returns (z_low, z_high)."""
    v = (w[inq, :].ravel() * _UM)
    v = v[np.isfinite(v)]
    if not v.size:
        return -1.0, 0.0
    z_low = float(np.percentile(v, 0.5))
    z_high = max(float(np.percentile(v, 99.5)), 0.0)
    if not (z_high > z_low):
        z_high = z_low + 1e-9
    return z_low, z_high


def _peak_t(w, inq) -> int:
    mag = np.array([np.abs(w[:, :, k][inq]).mean() for k in range(w.shape[2])])
    return int(np.argmax(mag))


def _render_topdown(w, x, y, sensor_xy, out_dir, z_low, z_high, fps,
                    max_frames, draw_front=True):
    """Top-down full-disk displacement animation (WAFER_CMAP, fixed z range
    across all frames). When `draw_front`, a red ring marks the bonding front
    (the outer edge of the bonded region) per frame, as in the 2D GUI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    from evaluation.palette import WAFER_CMAP
    from evaluation.bonding_front import bonded_mask, front_xy

    wf, xf, yf = _mirror_full(w, x, y)
    nt = w.shape[2]
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    outside = (Xf * Xf + Yf * Yf) > 1.0
    ext = [xf[0], xf[-1], yf[0], yf[-1]]
    norm = Normalize(vmin=z_low, vmax=z_high)
    fmask = bonded_mask(wf) if draw_front else None
    thetas = np.linspace(0, 2 * np.pi, 181)

    def slab(ti):
        s = (wf[:, :, ti] * _UM).copy()
        s[outside] = np.nan
        return s.T                                        # imshow wants [y, x]

    frames = _frame_idx(nt, max_frames)
    fig, ax = plt.subplots(figsize=(6.6, 6.2), constrained_layout=True)
    im = ax.imshow(slab(int(frames[0])), origin="lower", extent=ext,
                   cmap=WAFER_CMAP, norm=norm, interpolation="nearest")
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ), np.sin(circ), color="0.3", lw=1.4)
    front_line, = ax.plot([], [], color="red", lw=2.2, zorder=6,
                          label="bonding front")
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=60, facecolor="none",
               edgecolor="k", linewidth=1.3, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="u_z (um)")

    def _upd(fi):
        ti = int(frames[fi])
        im.set_data(slab(ti))
        if fmask is not None:
            fx, fy = front_xy(fmask[:, :, ti], xf, yf, thetas)
            front_line.set_data(fx, fy)
        ax.set_title(f"top-down u_z (um)   t={ti / (nt - 1):.2f}   "
                     f"({ti}/{nt - 1})")
        return [im, front_line]

    gif = Path(out_dir) / "real_field_topdown.gif"
    print(f"rendering {len(frames)}-frame top-down GIF -> {gif}", flush=True)
    FuncAnimation(fig, _upd, frames=len(frames), interval=1000 // fps,
                  blit=False).save(str(gif), writer=PillowWriter(fps=fps),
                                   dpi=100)
    plt.close(fig)


def _render_3d(w, x, y, sensor_xy, sensor_ij, out_dir, z_low, z_high, ts, fps,
               max_frames, elev=22, azim=-60, mesh=64, draw_front=True):
    """3D animation: the upper wafer descends over time onto the LOWER wafer
    (the peak/final-descent field, a fixed floor -- the 'peak snapshot'), on
    WAFER_CMAP with a fixed z range. Sensor markers are posts on the upper
    surface. When `draw_front`, a red ring rides the descending upper surface at
    the bonding front (the outer edge of the bonded region). Mirrors the quarter
    to the full disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D                # noqa: F401
    from evaluation.palette import WAFER_CMAP, sensor_color
    from evaluation.bonding_front import (bonded_mask, front_xy,
                                          sample_nearest)

    wf, xf, yf = _mirror_full(w, x, y)
    nt = w.shape[2]
    m = wf.shape[0]
    xi = np.arange(0, m, max(1, m // mesh))
    wf_d = wf[np.ix_(xi, xi)]                              # (md, md, nt)
    Xd, Yd = np.meshgrid(xf[xi], yf[xi], indexing="ij")
    outside = (Xd * Xd + Yd * Yd) > 1.0
    fmask = bonded_mask(wf) if draw_front else None
    thetas = np.linspace(0, 2 * np.pi, 181)

    def surf(ti):
        Z = (wf_d[:, :, ti] * _UM).copy()
        Z[outside] = np.nan
        return Z

    lower = surf(ts)                                       # peak = lower wafer
    frames = _frame_idx(nt, max_frames)
    post = max(0.04 * abs(z_low), 0.5)
    sxy = np.asarray(sensor_xy, float)
    sij = np.asarray(sensor_ij, int)

    fig = plt.figure(figsize=(7.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")

    def _draw(fi):
        ti = int(frames[fi])
        ax.clear()
        ax.plot_surface(Xd, Yd, lower, cmap=WAFER_CMAP, vmin=z_low, vmax=z_high,
                        alpha=0.5, linewidth=0, antialiased=False,
                        rstride=1, cstride=1)
        ax.plot_surface(Xd, Yd, surf(ti), cmap=WAFER_CMAP, vmin=z_low,
                        vmax=z_high, alpha=0.95, linewidth=0,
                        antialiased=False, rstride=1, cstride=1)
        if fmask is not None:
            fx, fy = front_xy(fmask[:, :, ti], xf, yf, thetas)
            fz = sample_nearest(wf[:, :, ti], xf, yf, fx, fy) * _UM
            ax.plot(fx, fy, fz, color="red", lw=2.6, zorder=12)
        for k in range(len(sxy)):
            zc = float(w[sij[k, 0], sij[k, 1], ti]) * _UM
            ax.plot([sxy[k, 0]] * 2, [sxy[k, 1]] * 2, [zc, zc + post],
                    color="0.25", lw=0.9, zorder=10)
            ax.plot([sxy[k, 0]], [sxy[k, 1]], [zc + post], "o",
                    color=sensor_color(k), markersize=7, markeredgecolor="k",
                    markeredgewidth=0.8, zorder=11)
        ax.set_zlim(z_low, z_high)
        try:
            ax.set_box_aspect((1.0, 1.0, 0.45), zoom=1.05)
        except TypeError:
            ax.set_box_aspect((1.0, 1.0, 0.45))
        except Exception:                                  # noqa: BLE001
            pass
        ax.view_init(elev=elev, azim=azim)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.fill = False
            a.set_tick_params(labelsize=7, colors="0.45")
        ax.set_xlabel("x / R")
        ax.set_ylabel("y / R")
        ax.set_zlabel("u_z (um)")
        ax.set_title(f"upper wafer descending onto lower (peak)   "
                     f"t={ti / (nt - 1):.2f}", fontsize=10)
        return []

    gif = Path(out_dir) / "real_field_3d.gif"
    print(f"rendering {len(frames)}-frame 3D GIF -> {gif}", flush=True)
    FuncAnimation(fig, _draw, frames=len(frames), interval=1000 // fps,
                  blit=False).save(str(gif), writer=PillowWriter(fps=fps),
                                   dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True, help="self-contained .pt")
    ap.add_argument("--real", required=True,
                    help="real input: tool CSV (.csv) or SI NPZ (.npz)")
    ap.add_argument("--config", default=None,
                    help="real-exp channel config YAML (default: the six "
                    "standard XM/DM/YM/XE/DE/YE channels)")
    ap.add_argument("--t-start", type=float, default=None,
                    help="override the window start (s)")
    ap.add_argument("--t-cutoff", type=float, default=None,
                    help="override the window cutoff (s) -- set it to where the "
                    "traces flatten at bonding completion")
    ap.add_argument("--out-dir", default="viz/real_eval")
    ap.add_argument("--no-anim", action="store_true",
                    help="skip the field animations (top-down + 3D GIFs)")
    ap.add_argument("--anim-fps", type=int, default=12)
    ap.add_argument("--anim-frames", type=int, default=40,
                    help="max frames per field GIF (default 40)")
    ap.add_argument("--elev", type=float, default=22.0,
                    help="3D view elevation angle (deg)")
    ap.add_argument("--azim", type=float, default=-60.0,
                    help="3D view azimuth angle (deg)")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    cfg = (real_config_from_yaml(args.config) if args.config
           else _default_config())
    raw = _load_raw(args.real)
    t = np.asarray(raw[cfg.time_key], dtype=float)
    n_ch = len([k for k in raw if str(k).startswith("w_")])
    kind = "CSV" if str(args.real).lower().endswith(".csv") else "NPZ"
    print(f"loaded {kind}: {n_ch} sensor channels, time "
          f"[{t.min():.4g}, {t.max():.4g}] s, {t.size} samples")
    cfg = _apply_window(cfg, raw, args.t_start, args.t_cutoff)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _viz_raw_overview(raw, cfg, out_dir / "real_inputs.png")   # window-pick aid

    rtheta = np.asarray(bundle["sensor_rtheta"], dtype=float)
    y, t_win = assemble_inputs(raw, rtheta, cfg)
    w = reconstruct_field(bundle, y, t_raw=t_win)
    print(f"assembled inputs y {y.shape}, reconstructed field w {w.shape}")

    checks = checkpoint1(bundle, y, w, cfg)
    ok = _print_checkpoint(checks)

    x_canon = np.asarray(bundle["x_canon"])
    y_canon = np.asarray(bundle["y_canon"])
    sensor_xy = np.asarray(bundle["sensor_xy"], dtype=float)
    sensor_ij = np.asarray(bundle["sensor_ij"], dtype=int)
    Xq, Yq = np.meshgrid(x_canon, y_canon, indexing="ij")
    inq = Xq * Xq + Yq * Yq <= 1.0
    z_low, z_high = _disp_scale(w, inq)         # fixed, data-driven, negative
    ts = _peak_t(w, inq)
    made = []
    if not args.no_anim:
        _render_topdown(w, x_canon, y_canon, sensor_xy, out_dir, z_low, z_high,
                        args.anim_fps, args.anim_frames)
        made.append("real_field_topdown.gif")
        _render_3d(w, x_canon, y_canon, sensor_xy, sensor_ij, out_dir, z_low,
                   z_high, ts, args.anim_fps, args.anim_frames,
                   elev=args.elev, azim=args.azim)
        made.append("real_field_3d.gif")

    np.savez(out_dir / "field.npz", w=w, x=x_canon, y=y_canon,
             t=np.linspace(0.0, 1.0, int(bundle["nt"])))
    (out_dir / "summary.json").write_text(json.dumps(dict(
        bundle=str(args.bundle), real=str(args.real), input_kind=kind,
        time_range_s=[float(t.min()), float(t.max())],
        window_s=[float(cfg.t_start), float(cfg.t_cutoff)],
        y_shape=list(y.shape), w_shape=list(w.shape), peak_t=ts,
        field_range_um=[z_low, z_high],
        checkpoint1=[dict(name=n, severity=s, ok=bool(o), detail=d)
                     for n, s, o, d in checks],
        checkpoint1_pass=bool(ok)), indent=2))
    print(f"\nwrote {', '.join(['real_inputs.png'] + made)}, field.npz, "
          f"summary.json to {out_dir}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
