"""Leave-one-out sensor self-consistency on REAL data (no ground truth), with
an optional truncation-window sweep.

Each bundle uses a SUBSET of the six ABCDEF sensors. For every bundle: assemble
its real inputs, reconstruct the full quarter-disk field, then read the field
at each LEFT-OUT sensor's location and compare that PREDICTION to the sensor's
actual MEASURED trace (rel-L2). Pure real -- no simulation. Low error at the
held-out sensors means the reconstruction is self-consistent with measurements
it never saw.

With --sweep-t-start / --sweep-t-cutoff (each LO HI STEP), the window is swept
over that grid and the one MINIMIZING the median held-out rel-L2 is chosen --
the same objective the 2D GUI used to auto-pick the bonding window (it should
land where the sensor traces flatten). The final figure/table are produced at
the best window.

    # fixed window
    python scripts/eval_real_loo.py --bundles bundles/*_n5_*.pt \\
        --real run01.csv --config configs/real_exp_n6.yaml --t-cutoff 8

    # sweep the cutoff 6..12 s (step 0.5) and pick the best
    python scripts/eval_real_loo.py --bundles bundles/*_n5_*.pt \\
        --real run01.csv --config configs/real_exp_n6.yaml \\
        --sweep-t-cutoff 6 12 0.5
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import polar_to_xy, xy_to_indices              # noqa: E402
from core.pod_basis import PODBasis                           # noqa: E402
from data.real_experiment import (assemble_inputs,            # noqa: E402
                                  real_config_from_yaml)
from training.normalization import (NormStats, apply_norm,    # noqa: E402
                                    invert_norm)
from scripts.reconstruct import (load_bundle, _build_models,  # noqa: E402
                                 _resample)
from scripts.eval_real import (_load_raw, _apply_window,      # noqa: E402
                               _default_config, _disp_scale, _peak_t,
                               _render_topdown, _render_3d)

_UM = 1.0e6
_ABCDEF = [(0.52, 0.0, "A"), (0.52, 45.0, "B"), (0.52, 90.0, "C"),
           (0.847, 0.0, "D"), (0.847, 45.0, "E"), (0.847, 90.0, "F")]


def _leftout(rtheta, r_tol=0.02, th_tol=5.0):
    used = np.asarray(rtheta, dtype=float).reshape(-1, 2)
    out = []
    for r, th, lab in _ABCDEF:
        present = any(abs(u[0] - r) <= r_tol and abs(u[1] - th) <= th_tol
                      for u in used)
        if not present:
            out.append((r, th, lab))
    return out


def _rel_l2(pred, meas):
    return float(np.linalg.norm(pred - meas)) / max(
        float(np.linalg.norm(meas)), 1e-30)


def _load(path):
    """Load a bundle ONCE and pre-build its models + basis, so a window sweep
    reuses them instead of rebuilding per window."""
    b = load_bundle(path)
    basis = PODBasis(np.asarray(b["Phi"]), np.asarray(b["sigma"]),
                     tuple(int(s) for s in b["spatial_shape"]))
    return dict(b=b, tag=Path(path).stem, models=_build_models(b), basis=basis)


def _recon(L, y, t_raw):
    b = L["b"]
    nt = int(b["nt"])
    y = np.asarray(y, dtype=float)
    if t_raw is not None and y.shape[1] != nt:
        y = _resample(y, t_raw, nt)
    yn = apply_norm(y[None], NormStats(b["y_mean"], b["y_std"]))
    x = torch.tensor(yn, dtype=torch.float32)
    with torch.no_grad():
        preds = [m(x).cpu().numpy() for m in L["models"]]
    Y = invert_norm(np.mean(np.stack(preds), axis=0),
                    NormStats(b["target_mean"], b["target_std"]))[0]
    return L["basis"].reconstruct(Y)


def _one_bundle(L, raw, cfg):
    """Per held-out sensor: predicted (from the field) vs measured trace."""
    b = L["b"]
    nt = int(b["nt"])
    x_c, y_c = np.asarray(b["x_canon"]), np.asarray(b["y_canon"])
    lo = _leftout(b["sensor_rtheta"])
    if not lo:
        return []
    y_in, t_in = assemble_inputs(raw, b["sensor_rtheta"], cfg)
    w = _recon(L, y_in, t_in)
    t_norm = np.linspace(0.0, 1.0, nt)
    recs = []
    for r, th, lab in lo:
        ix, iy = xy_to_indices(*polar_to_xy(r, th), x_c, y_c)
        pred = np.asarray(w[ix, iy, :], dtype=float)
        ym, tm = assemble_inputs(raw, [[r, th]], cfg)
        meas = _resample(ym, tm, nt)[0]
        recs.append(dict(tag=L["tag"], label=lab, rel_l2=_rel_l2(pred, meas),
                         pred=pred, meas=meas, t=t_norm))
    return recs


def _all_records(bundles, raw, cfg):
    recs = []
    for L in bundles:
        recs.extend(_one_bundle(L, raw, cfg))
    return recs


def _detect_end_of_bond(t, traces, rel_thresh=0.02, tail_frac=0.1):
    """End of bonding = end of the FINAL plateau. Each sensor SETTLES to a
    final value (median of its last `tail_frac` of samples). Read backward: the
    LATEST time any sensor is still away from its settled value by more than
    `rel_thresh` of its total travel is where bonding ends. This is the
    noise-robust form of 'the slope stops being flat' -- an intermediate hold
    plateau sits AT a different value (still far from final), so it never fools
    it, and plateau noise (tiny vs the descent) never trips the threshold."""
    t = np.asarray(t, dtype=float)
    n = t.size
    if n < 3:
        return float(t[-1]) if n else 0.0
    tail = max(3, int(tail_frac * n))
    t_end = float(t[0])
    for tr in traces:
        tr = np.asarray(tr, dtype=float)
        final = float(np.median(tr[-tail:]))
        dev = np.abs(tr - final)
        span = float(dev.max())
        if span <= 0:
            continue
        moving = dev > rel_thresh * span         # still away from settled value
        if moving.any():
            t_end = max(t_end, float(t[int(np.max(np.where(moving)[0]))]))
    return t_end


def _grid(spec, fixed):
    if spec is None:
        return [float(fixed)]
    lo, hi, step = spec
    if step <= 0:
        return [float(lo)]
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    return [round(lo + k * step, 6) for k in range(max(1, n))]


def _sweep(bundles, raw, cfg, tstarts, tcutoffs, data_lo, data_hi):
    """median held-out rel-L2 for every valid (t_start, t_cutoff); returns
    (M, best) where M is (len(tstarts), len(tcutoffs)) with NaN for invalid."""
    M = np.full((len(tstarts), len(tcutoffs)), np.nan)
    best = (np.inf, None)
    for i, ts in enumerate(tstarts):
        for j, tc in enumerate(tcutoffs):
            if not (data_lo <= ts < tc <= data_hi):
                continue
            cfg_w = dataclasses.replace(cfg, t_start=float(ts),
                                        t_cutoff=float(tc))
            try:
                recs = _all_records(bundles, raw, cfg_w)
            except (ValueError, KeyError):
                continue
            if not recs:
                continue
            med = float(np.median([r["rel_l2"] for r in recs]))
            M[i, j] = med
            if med < best[0]:
                best = (med, (float(ts), float(tc)))
    return M, best


def _render_sweep(tstarts, tcutoffs, M, best, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    if len(tstarts) > 1 and len(tcutoffs) > 1:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r",
                       extent=[tcutoffs[0], tcutoffs[-1],
                               tstarts[0], tstarts[-1]])
        ax.set_xlabel("t_cutoff (s)")
        ax.set_ylabel("t_start (s)")
        fig.colorbar(im, ax=ax, label="median held-out rel-L2")
        if best[1]:
            ax.plot(best[1][1], best[1][0], "*", color="#e63946", ms=18,
                    markeredgecolor="k")
    else:
        if len(tcutoffs) > 1:
            xs, ys, xl = tcutoffs, M[0, :], "t_cutoff (s)"
        else:
            xs, ys, xl = tstarts, M[:, 0], "t_start (s)"
        ax.plot(xs, ys, "-o", color="#3d5a80")
        if best[1]:
            bx = best[1][1] if len(tcutoffs) > 1 else best[1][0]
            ax.plot(bx, best[0], "*", color="#e63946", ms=18,
                    markeredgecolor="k", label=f"best {best[0]:.3f}")
            ax.legend()
        ax.set_xlabel(xl)
        ax.set_ylabel("median held-out rel-L2")
        ax.grid(alpha=0.3)
    ax.set_title("LOO window sweep (lower = more self-consistent)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _in_used(r, th, used, r_tol=0.02, th_tol=5.0):
    return any(abs(u[0] - r) <= r_tol and abs(u[1] - th) <= th_tol
               for u in used)


def _render_per_model(bundles, records, out_dir):
    """One figure per bundle: LEFT the ABCDEF layout with the 5 used sensors
    SOLID and the held-out one(s) HOLLOW; RIGHT that bundle's held-out
    measured-vs-predicted LOO curve(s)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_tag = {}
    for r in records:
        by_tag.setdefault(r["tag"], []).append(r)
    arc = np.linspace(0, 90, 120)
    written = []
    for L in bundles:
        tag = L["tag"]
        recs = by_tag.get(tag, [])
        if not recs:
            continue
        used = np.asarray(L["b"]["sensor_rtheta"], dtype=float).reshape(-1, 2)
        fig, (axl, axr) = plt.subplots(
            1, 2, figsize=(10.4, 4.5), constrained_layout=True,
            gridspec_kw=dict(width_ratios=[1.0, 1.2]))
        # left: quarter-disk layout
        axl.plot(np.cos(np.deg2rad(arc)), np.sin(np.deg2rad(arc)),
                 color="0.35", lw=2)
        axl.plot([0, 1.05], [0, 0], color="0.7", lw=1)
        axl.plot([0, 0], [0, 1.05], color="0.7", lw=1)
        for r, th, lab in _ABCDEF:
            x, y = r * np.cos(np.deg2rad(th)), r * np.sin(np.deg2rad(th))
            if _in_used(r, th, used):
                axl.scatter(x, y, s=170, marker="o", color="#3d5a80",
                            edgecolor="k", linewidth=1.0, zorder=5)
            else:                                          # held out -> hollow
                axl.scatter(x, y, s=210, marker="o", facecolor="none",
                            edgecolor="#e63946", linewidth=2.6, zorder=6)
            axl.annotate(lab, (x, y), xytext=(7, 6),
                         textcoords="offset points", fontsize=10,
                         fontweight="bold", color="0.2")
        axl.set_aspect("equal")
        axl.set_xlim(-0.08, 1.15)
        axl.set_ylim(-0.08, 1.15)
        axl.set_xlabel("x / R")
        axl.set_ylabel("y / R")
        axl.set_title(f"{tag}\nsolid = used (5),  hollow = held out",
                      fontsize=10)
        axl.grid(alpha=0.25)
        # right: LOO curve(s) for this bundle
        for r in recs:
            axr.plot(r["t"], r["meas"] * _UM, "-", color="black", lw=1.8,
                     label=f"measured ({r['label']})")
            axr.plot(r["t"], r["pred"] * _UM, "--", color="#e63946", lw=1.8,
                     label=f"predicted ({r['label']})")
        axr.set_title("held-out: measured vs predicted   "
                      + ",  ".join(f"{r['label']} relL2={r['rel_l2']:.3f}"
                                   for r in recs), fontsize=9)
        axr.set_xlabel("normalized time")
        axr.set_ylabel("u_z (um)")
        axr.legend(fontsize=8)
        axr.grid(alpha=0.3)
        p = Path(out_dir) / f"model_{tag}.png"
        fig.savefig(str(p), dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p.name)
    return written


def _render(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(records)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             constrained_layout=True, squeeze=False)
    for i, r in enumerate(records):
        ax = axes[i // ncol][i % ncol]
        ax.plot(r["t"], r["meas"] * _UM, "-", color="black", lw=1.8,
                label="measured")
        ax.plot(r["t"], r["pred"] * _UM, "--", color="#e63946", lw=1.8,
                label="predicted (held out)")
        ax.set_title(f"{r['tag']}  |  out {r['label']}  "
                     f"relL2={r['rel_l2']:.3f}", fontsize=9)
        ax.set_xlabel("normalized time")
        ax.set_ylabel("u_z (um)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Leave-one-out sensor self-consistency (real data, no GT)",
                 fontweight="bold")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _pick_field_bundle(bundles, records, field_bundle):
    """Which bundle reconstructs the deployment field for the animations.

    Prefer an explicit --field-bundle (typically the all-six ABCDEF n6 bundle,
    i.e. the real deployment config). Otherwise fall back to the loaded bundle
    with the LOWEST median held-out rel-L2 -- the most self-consistent one, so
    the animation shows the reconstruction we trust most. Returns (L, note)."""
    if field_bundle is not None:
        return _load(field_bundle), f"--field-bundle {Path(field_bundle).stem}"
    by_tag = {}
    for r in records:
        by_tag.setdefault(r["tag"], []).append(r["rel_l2"])
    med = {t: float(np.median(v)) for t, v in by_tag.items()}
    if not med:
        return bundles[0], f"first bundle {bundles[0]['tag']}"
    best_tag = min(med, key=med.get)
    L = next((b for b in bundles if b["tag"] == best_tag), bundles[0])
    return L, f"best LOO bundle {best_tag} (median rel-L2 {med[best_tag]:.3f})"


def _render_field_anims(L, raw, cfg_use, out_dir, args):
    """Reconstruct the full field from bundle L at the chosen window and render
    the top-down + 3D bonding animations (WAFER_CMAP, red bonding front) into
    the per-run folder -- the same views the single-run eval_real produces."""
    b = L["b"]
    y_in, t_in = assemble_inputs(raw, b["sensor_rtheta"], cfg_use)
    w = _recon(L, y_in, t_in)
    x_c, y_c = np.asarray(b["x_canon"]), np.asarray(b["y_canon"])
    sxy = np.asarray(b["sensor_xy"], dtype=float)
    sij = np.asarray(b["sensor_ij"], dtype=int)
    Xq, Yq = np.meshgrid(x_c, y_c, indexing="ij")
    inq = Xq * Xq + Yq * Yq <= 1.0
    z_low, z_high = _disp_scale(w, inq)
    ts = _peak_t(w, inq)
    _render_topdown(w, x_c, y_c, sxy, out_dir, z_low, z_high,
                    args.anim_fps, args.anim_frames,
                    front_r_max=args.front_r_max)
    _render_3d(w, x_c, y_c, sxy, sij, out_dir, z_low, z_high, ts,
               args.anim_fps, args.anim_frames, elev=args.elev, azim=args.azim,
               front_r_max=args.front_r_max)
    return ["real_field_topdown.gif", "real_field_3d.gif"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundles", nargs="+", required=True,
                    help="subset bundles (e.g. bundles/*_n5_*.pt)")
    ap.add_argument("--real", required=True, help="real CSV or NPZ")
    ap.add_argument("--config", default=None,
                    help="channel config with ALL SIX sensors")
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-cutoff", type=float, default=None)
    ap.add_argument("--sweep-t-start", nargs=3, type=float, default=None,
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--sweep-t-cutoff", nargs=3, type=float, default=None,
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--auto-cutoff", action="store_true",
                    help="auto-detect the end of bonding (latest time any "
                    "sensor is still moving = end of the final plateau) and "
                    "sweep t_cutoff +/- halfwidth around it. With this on, "
                    "t_start defaults to sweeping 0..0.5 step 0.1 unless "
                    "--sweep-t-start is given.")
    ap.add_argument("--auto-cutoff-halfwidth", type=float, default=1.5,
                    help="+/- range around the detected end (s, default 1.5)")
    ap.add_argument("--auto-cutoff-step", type=float, default=0.2,
                    help="t_cutoff step in auto mode (s, default 0.2)")
    ap.add_argument("--out-dir", default="viz/real_loo")
    # deployment-field animations (top-down + 3D, with the bonding front)
    ap.add_argument("--field-bundle", default=None,
                    help="bundle used to reconstruct the field for the "
                    "animations (default: the most self-consistent LOO bundle; "
                    "pass the all-six n6 ABCDEF bundle for the true deployment "
                    "field)")
    ap.add_argument("--no-anim", action="store_true",
                    help="skip the top-down + 3D field animations")
    ap.add_argument("--anim-fps", type=int, default=12)
    ap.add_argument("--anim-frames", type=int, default=40,
                    help="max frames per field GIF (default 40)")
    ap.add_argument("--elev", type=float, default=22.0,
                    help="3D view elevation angle (deg)")
    ap.add_argument("--azim", type=float, default=-60.0,
                    help="3D view azimuth angle (deg)")
    ap.add_argument("--front-r-max", type=float, default=0.95,
                    help="cap the bonding-front search radius (<=1.0) to drop "
                    "the noisy, unsupported edge shell of the reconstruction")
    args = ap.parse_args()

    cfg = (real_config_from_yaml(args.config) if args.config
           else _default_config())
    raw = _load_raw(args.real)
    t = np.asarray(raw[cfg.time_key], dtype=float)
    data_lo, data_hi = float(t.min()), float(t.max())
    base = _apply_window(cfg, raw, args.t_start, args.t_cutoff)

    print(f"loading {len(args.bundles)} bundle(s) ...", flush=True)
    bundles = [_load(p) for p in args.bundles]

    out_dir = Path(args.out_dir) / Path(args.real).stem   # per-run subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    swept = None
    cfg_use = base
    t_end = None
    if (args.sweep_t_start is not None or args.sweep_t_cutoff is not None
            or args.auto_cutoff):
        if args.auto_cutoff:
            traces = [np.asarray(raw[ch.key], dtype=float)
                      for ch in cfg.channels if ch.key in raw]
            t_end = _detect_end_of_bond(t, traces)
            hw, st = args.auto_cutoff_halfwidth, args.auto_cutoff_step
            tcutoffs = _grid([t_end - hw, t_end + hw, st], base.t_cutoff)
            tstarts = _grid(args.sweep_t_start
                            if args.sweep_t_start is not None
                            else [0.0, 0.5, 0.1], base.t_start)
            print(f"auto end-of-bond at t={t_end:.2f}s -> t_cutoff grid "
                  f"[{t_end - hw:.2f}, {t_end + hw:.2f}] step {st}; "
                  f"t_start grid {tstarts[0]:g}..{tstarts[-1]:g}", flush=True)
        else:
            tstarts = _grid(args.sweep_t_start, base.t_start)
            tcutoffs = _grid(args.sweep_t_cutoff, base.t_cutoff)
        print(f"sweeping {len(tstarts)}x{len(tcutoffs)} windows ...", flush=True)
        M, best = _sweep(bundles, raw, cfg, tstarts, tcutoffs,
                         data_lo, data_hi)
        if best[1] is None:
            print("no valid window in the sweep range", file=sys.stderr)
            return 1
        bts, btc = best[1]
        print(f"best window: t_start={bts:g}, t_cutoff={btc:g}  "
              f"(median held-out rel-L2 {best[0]:.4f})")
        _render_sweep(tstarts, tcutoffs, M, best, out_dir / "loo_sweep.png")
        cfg_use = dataclasses.replace(cfg, t_start=float(bts),
                                      t_cutoff=float(btc))
        swept = dict(best_window_s=[bts, btc], best_median_rel_l2=best[0],
                     t_starts=tstarts, t_cutoffs=tcutoffs,
                     auto_end_of_bond_s=(None if t_end is None
                                         else float(t_end)))

    records = []
    for L in bundles:
        try:
            records.extend(_one_bundle(L, raw, cfg_use))
        except (ValueError, KeyError, OSError) as e:
            print(f"skip {L['tag']}: {type(e).__name__}: {e}", file=sys.stderr)
    if not records:
        print("no leave-one-out comparisons produced", file=sys.stderr)
        return 1
    records.sort(key=lambda r: r["rel_l2"])

    print("\n===== leave-one-out (predicted vs measured at the held-out "
          f"sensor)  window=[{cfg_use.t_start:g}, {cfg_use.t_cutoff:g}]s =====")
    print(f"  {'held-out':<9} {'rel-L2':>8}   bundle")
    print("  " + "-" * 60)
    for r in records:
        print(f"  {r['label']:<9} {r['rel_l2']:>8.4f}   {r['tag']}")
    rels = np.array([r["rel_l2"] for r in records])
    print(f"\n  median rel-L2 {np.median(rels):.4f}   "
          f"max {rels.max():.4f} (worst held-out sensor)")

    _render(records, out_dir / "loo.png")
    per_model = _render_per_model(bundles, records, out_dir)

    anims = []
    if not args.no_anim:
        L_field, note = _pick_field_bundle(bundles, records, args.field_bundle)
        print(f"\nrendering deployment-field animations from {note} ...",
              flush=True)
        try:
            anims = _render_field_anims(L_field, raw, cfg_use, out_dir, args)
        except (ValueError, KeyError, OSError) as e:
            print(f"  skipped animations: {type(e).__name__}: {e}",
                  file=sys.stderr)

    (out_dir / "summary.json").write_text(json.dumps(dict(
        real=str(args.real),
        window_s=[float(cfg_use.t_start), float(cfg_use.t_cutoff)],
        sweep=swept,
        median_rel_l2=float(np.median(rels)), max_rel_l2=float(rels.max()),
        comparisons=[dict(held_out=r["label"], rel_l2=r["rel_l2"],
                          bundle=r["tag"]) for r in records]), indent=2))
    outs = ["loo.png", "summary.json"] + [f"{len(per_model)}x model_*.png"]
    if swept:
        outs.insert(1, "loo_sweep.png")
    outs += anims
    print(f"\nwrote {', '.join(outs)} to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
