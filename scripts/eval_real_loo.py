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
                               _default_config)

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
    ap.add_argument("--out-dir", default="viz/real_loo")
    args = ap.parse_args()

    cfg = (real_config_from_yaml(args.config) if args.config
           else _default_config())
    raw = _load_raw(args.real)
    t = np.asarray(raw[cfg.time_key], dtype=float)
    data_lo, data_hi = float(t.min()), float(t.max())
    base = _apply_window(cfg, raw, args.t_start, args.t_cutoff)

    print(f"loading {len(args.bundles)} bundle(s) ...", flush=True)
    bundles = [_load(p) for p in args.bundles]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    swept = None
    cfg_use = base
    if args.sweep_t_start is not None or args.sweep_t_cutoff is not None:
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
                     t_starts=tstarts, t_cutoffs=tcutoffs)

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
    (out_dir / "summary.json").write_text(json.dumps(dict(
        real=str(args.real),
        window_s=[float(cfg_use.t_start), float(cfg_use.t_cutoff)],
        sweep=swept,
        median_rel_l2=float(np.median(rels)), max_rel_l2=float(rels.max()),
        comparisons=[dict(held_out=r["label"], rel_l2=r["rel_l2"],
                          bundle=r["tag"]) for r in records]), indent=2))
    outs = "loo.png, summary.json" + (
        ", loo_sweep.png" if swept else "")
    print(f"\nwrote {outs} to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
