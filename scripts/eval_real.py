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
    return np.linspace(0, nt - 1, max_frames).astype(int)


def _disp_norm(vlo, vhi):
    """A DATA-DRIVEN color norm for the (downward) displacement -- never forced
    symmetric about 0. If the field is essentially one-signed negative, the
    colorbar spans [vlo, 0] (sequential); if there is a real positive rebound
    (gas-bulge), a diverging norm keeps 0 = neutral with ASYMMETRIC extents so
    the small rebound shows without wasting half the bar."""
    import matplotlib.colors as mcolors
    if vlo < 0 and vhi > 0.02 * abs(vlo):
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vlo, vmax=vhi), "coolwarm"
    hi = min(vhi, 0.0)
    if not (hi > vlo):
        hi = vlo + 1e-12
    return mcolors.Normalize(vmin=vlo, vmax=hi), "Blues_r"


def _draw_disk(ax, sensor_xy):
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ), np.sin(circ), color="0.3", lw=1.5)
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=70, marker="o",
               facecolor="none", edgecolor="black", linewidth=1.4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")


def _render_field(w, x, y, sensor_xy, out_dir, *, anim=True, fps=12,
                  max_frames=60):
    """Top-down full-disk displacement: a static snapshot at the peak time and
    (optionally) a GIF over time, both on ONE fixed, data-driven color scale
    (so the animation shows the real descent, not per-frame renormalization)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wf, xf, yf = _mirror_full(w, x, y)                     # (Xf, Yf, nt)
    nt = w.shape[2]
    Xq, Yq = np.meshgrid(x, y, indexing="ij")
    inq = Xq * Xq + Yq * Yq <= 1.0
    finite = (w[inq, :].ravel() * _UM)
    finite = finite[np.isfinite(finite)]
    vlo = float(np.percentile(finite, 1)) if finite.size else -1.0
    vhi = float(np.percentile(finite, 99)) if finite.size else 0.0
    norm, cmap = _disp_norm(vlo, vhi)

    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    outside = (Xf * Xf + Yf * Yf) > 1.0
    ext = [xf[0], xf[-1], yf[0], yf[-1]]

    def slab(ti):
        s = (wf[:, :, ti] * _UM).copy()
        s[outside] = np.nan
        return s.T                                        # imshow wants [y, x]

    mag_t = np.array([np.abs(w[:, :, k][inq]).mean() for k in range(nt)])
    ts = int(np.argmax(mag_t))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    im = ax.imshow(slab(ts), origin="lower", extent=ext, cmap=cmap, norm=norm,
                   interpolation="nearest")
    _draw_disk(ax, sensor_xy)
    ax.set_title(f"reconstructed u_z (um), full disk, peak t*={ts}/{nt - 1}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="u_z (um)")
    fig.savefig(str(out_dir / "real_field.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    if anim:
        from matplotlib.animation import FuncAnimation, PillowWriter
        frames = _frame_idx(nt, max_frames)
        fig, ax = plt.subplots(figsize=(6.6, 6.4), constrained_layout=True)
        im = ax.imshow(slab(int(frames[0])), origin="lower", extent=ext,
                       cmap=cmap, norm=norm, interpolation="nearest")
        _draw_disk(ax, sensor_xy)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="u_z (um)")

        def _upd(fi):
            ti = int(frames[fi])
            im.set_data(slab(ti))
            ax.set_title(f"reconstructed u_z (um)   t={ti / (nt - 1):.2f}   "
                         f"({ti}/{nt - 1})")
            return [im]

        gif = out_dir / "real_field_anim.gif"
        print(f"rendering {len(frames)}-frame field GIF -> {gif}", flush=True)
        FuncAnimation(fig, _upd, frames=len(frames), interval=1000 // fps,
                      blit=False).save(str(gif), writer=PillowWriter(fps=fps),
                                       dpi=100)
        plt.close(fig)
    return ts, vlo, vhi


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
                    help="skip the field-over-time GIF (snapshot only)")
    ap.add_argument("--anim-fps", type=int, default=12)
    ap.add_argument("--anim-frames", type=int, default=60,
                    help="max frames in the field GIF (default 60)")
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
    ts, vlo, vhi = _render_field(
        w, x_canon, y_canon, sensor_xy, out_dir, anim=not args.no_anim,
        fps=args.anim_fps, max_frames=args.anim_frames)
    np.savez(out_dir / "field.npz", w=w, x=x_canon, y=y_canon,
             t=np.linspace(0.0, 1.0, int(bundle["nt"])))
    (out_dir / "summary.json").write_text(json.dumps(dict(
        bundle=str(args.bundle), real=str(args.real), input_kind=kind,
        time_range_s=[float(t.min()), float(t.max())],
        window_s=[float(cfg.t_start), float(cfg.t_cutoff)],
        y_shape=list(y.shape), w_shape=list(w.shape), peak_t=ts,
        field_range_um=[vlo, vhi],
        checkpoint1=[dict(name=n, severity=s, ok=bool(o), detail=d)
                     for n, s, o, d in checks],
        checkpoint1_pass=bool(ok)), indent=2))
    figs = "real_inputs.png, real_field.png"
    figs += ", real_field_anim.gif" if not args.no_anim else ""
    print(f"\nwrote {figs}, field.npz, summary.json to {out_dir}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
