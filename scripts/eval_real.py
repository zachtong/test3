"""Minimal real-data evaluation for the 3D model.

Feed a real experimental sensor NPZ through a self-contained bundle
(scripts/bundle.py) and reconstruct the full quarter-disk field. There is NO
ground truth, so this does three things:

  1. assemble the six directional sensors into the model's ABCDEF channels
     (fold + match by (r, theta), no averaging) and reconstruct w(x, y, t);
  2. run CHECKPOINT 1 -- the physical sanity gate (units, sign, magnitude,
     positions) that catches nm/um/m and sign-convention errors on the first
     real run, exactly the class of bug the 2D port hit;
  3. visualize the reconstructed field, mirrored from the quarter to the full
     disk, plus the assembled sensor inputs.

    python scripts/eval_real.py \\
        --bundle bundles/abcdef_k12.pt \\
        --real run01.npz \\
        --config configs/real_exp_n6.yaml \\
        --out-dir viz/real_eval/run01
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

from scripts.reconstruct import load_bundle, reconstruct_field   # noqa: E402
from data.real_experiment import (                               # noqa: E402
    Channel, RealExperimentConfig, real_config_from_yaml,
    load_real_npz, assemble_inputs, fold_theta)

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


def checkpoint1(bundle, y, w, cfg) -> list:
    """Physical sanity checks. Each is (name, severity, ok, detail);
    severity 'critical' must pass, 'advisory' is a heads-up (sign / units)."""
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
    x- and y-axis symmetry the model assumes. Returns (w_full, x_full,
    y_full)."""
    nx, ny = w.shape[0], w.shape[1]
    ixq = np.abs(np.arange(2 * nx - 1) - (nx - 1))
    iyq = np.abs(np.arange(2 * ny - 1) - (ny - 1))
    wf = w[ixq][:, iyq]                                    # (2nx-1, 2ny-1, Nt)
    xf = np.concatenate([-x[:0:-1], x])
    yf = np.concatenate([-y[:0:-1], y])
    return wf, xf, yf


def _viz_inputs(y, t, rtheta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for i in range(y.shape[0]):
        ax.plot(t, y[i] * _UM, lw=1.6,
                label=f"r={rtheta[i, 0]:.2f}, th={rtheta[i, 1]:.0f}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sensor u_z (um)")
    ax.set_title("assembled sensor inputs (6 channels, ABCDEF order)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _viz_field(w, x, y, sensor_xy, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wf, xf, yf = _mirror_full(w, x, y)
    X, Y = np.meshgrid(xf, yf, indexing="ij")
    disk = X * X + Y * Y <= 1.0
    # peak time = when the in-disk field is largest in magnitude
    inq = (np.asarray(np.meshgrid(x, y, indexing="ij"))[0] ** 2
           + np.asarray(np.meshgrid(x, y, indexing="ij"))[1] ** 2) <= 1.0
    mag_t = np.array([np.abs(w[:, :, k][inq]).mean() for k in range(w.shape[2])])
    ts = int(np.argmax(mag_t))
    slab = (wf[:, :, ts] * _UM).copy()
    slab[~disk] = np.nan
    finite = slab[np.isfinite(slab)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    pcm = ax.pcolormesh(X, Y, slab, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                        shading="auto")
    ax.plot(np.cos(np.linspace(0, 2 * np.pi, 200)),
            np.sin(np.linspace(0, 2 * np.pi, 200)), color="0.3", lw=1.5)
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=70, marker="o",
               facecolor="none", edgecolor="black", linewidth=1.4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"reconstructed u_z (um), full disk, t*={ts}/{w.shape[2] - 1}")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="u_z (um)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True, help="self-contained .pt")
    ap.add_argument("--real", required=True, help="real sensor NPZ")
    ap.add_argument("--config", default=None,
                    help="real-exp channel config YAML (default: the six "
                    "standard XM/DM/YM/XE/DE/YE channels)")
    ap.add_argument("--out-dir", default="viz/real_eval")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    cfg = (real_config_from_yaml(args.config) if args.config
           else _default_config())
    raw = load_real_npz(args.real)
    rtheta = np.asarray(bundle["sensor_rtheta"], dtype=float)

    y, t = assemble_inputs(raw, rtheta, cfg)
    w = reconstruct_field(bundle, y, t_raw=t)
    print(f"assembled inputs y {y.shape}, reconstructed field w {w.shape}")

    checks = checkpoint1(bundle, y, w, cfg)
    ok = _print_checkpoint(checks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_canon = np.asarray(bundle["x_canon"])
    y_canon = np.asarray(bundle["y_canon"])
    sensor_xy = np.asarray(bundle["sensor_xy"], dtype=float)
    _viz_inputs(y, t, rtheta, out_dir / "real_inputs.png")
    ts = _viz_field(w, x_canon, y_canon, sensor_xy, out_dir / "real_field.png")
    np.savez(out_dir / "field.npz", w=w, x=x_canon, y=y_canon,
             t=np.linspace(0.0, 1.0, int(bundle["nt"])))
    (out_dir / "summary.json").write_text(json.dumps(dict(
        bundle=str(args.bundle), real=str(args.real),
        y_shape=list(y.shape), w_shape=list(w.shape), peak_t=ts,
        checkpoint1=[dict(name=n, severity=s, ok=bool(o), detail=d)
                     for n, s, o, d in checks],
        checkpoint1_pass=bool(ok)), indent=2))
    print(f"\nwrote real_inputs.png, real_field.png, field.npz, summary.json "
          f"to {out_dir}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
