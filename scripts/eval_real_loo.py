"""Leave-one-out sensor self-consistency on REAL data (no ground truth).

Each bundle uses a SUBSET of the six ABCDEF sensors. For every bundle: assemble
its real inputs, reconstruct the full quarter-disk field, then read the field
at each LEFT-OUT sensor's location and compare that PREDICTION to the sensor's
actual MEASURED trace. Pure real -- no simulation enters. Low error at the
held-out sensors means the reconstruction is self-consistent with measurements
it never saw. n=5 bundles hold out one sensor each; smaller n hold out more.

    python scripts/eval_real_loo.py \\
        --bundles bundles/merged_sweep_k12_n5_*.pt \\
        --real run01.csv --config configs/real_exp_n6.yaml \\
        --t-cutoff 8 --out-dir viz/real_loo/run01
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import polar_to_xy, xy_to_indices              # noqa: E402
from data.real_experiment import (assemble_inputs,            # noqa: E402
                                  real_config_from_yaml)
from scripts.reconstruct import (load_bundle, reconstruct_field,  # noqa: E402
                                 _resample)
from scripts.eval_real import _load_raw, _apply_window, _default_config  # noqa: E402

_UM = 1.0e6
# the six ABCDEF positions (r_norm, theta_deg) + labels
_ABCDEF = [(0.52, 0.0, "A"), (0.52, 45.0, "B"), (0.52, 90.0, "C"),
           (0.847, 0.0, "D"), (0.847, 45.0, "E"), (0.847, 90.0, "F")]


def _leftout(rtheta, r_tol=0.02, th_tol=5.0):
    """ABCDEF positions NOT covered by this bundle's sensor set."""
    used = np.asarray(rtheta, dtype=float).reshape(-1, 2)
    out = []
    for r, th, lab in _ABCDEF:
        present = any(abs(u[0] - r) <= r_tol and abs(u[1] - th) <= th_tol
                      for u in used)
        if not present:
            out.append((r, th, lab))
    return out


def _rel_l2(pred, meas):
    d = float(np.linalg.norm(pred - meas))
    return d / max(float(np.linalg.norm(meas)), 1e-30)


def _one_bundle(path, raw, cfg):
    """Returns a list of dicts, one per held-out sensor: label, rel_l2,
    pred (nt), meas (nt), t (nt normalized)."""
    b = load_bundle(path)
    nt = int(b["nt"])
    x_c, y_c = np.asarray(b["x_canon"]), np.asarray(b["y_canon"])
    lo = _leftout(b["sensor_rtheta"])
    if not lo:
        return b, []                                   # n=6: nothing held out
    y_in, t_in = assemble_inputs(raw, b["sensor_rtheta"], cfg)
    w = reconstruct_field(b, y_in, t_raw=t_in)         # (Nx, Ny, Nt)
    t_norm = np.linspace(0.0, 1.0, nt)
    recs = []
    for r, th, lab in lo:
        ix, iy = xy_to_indices(*polar_to_xy(r, th), x_c, y_c)
        pred = np.asarray(w[ix, iy, :], dtype=float)
        ym, tm = assemble_inputs(raw, [[r, th]], cfg)  # measured, windowed
        meas = _resample(ym, tm, nt)[0]
        recs.append(dict(label=lab, rel_l2=_rel_l2(pred, meas),
                         pred=pred, meas=meas, t=t_norm))
    return b, recs


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
                    help="the subset bundles (e.g. bundles/*_n5_*.pt)")
    ap.add_argument("--real", required=True, help="real CSV or NPZ")
    ap.add_argument("--config", default=None,
                    help="channel config with ALL SIX sensors (needed to read "
                    "the held-out sensor's measured trace)")
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-cutoff", type=float, default=None)
    ap.add_argument("--out-dir", default="viz/real_loo")
    args = ap.parse_args()

    cfg = (real_config_from_yaml(args.config) if args.config
           else _default_config())
    raw = _load_raw(args.real)
    cfg = _apply_window(cfg, raw, args.t_start, args.t_cutoff)

    records = []
    for path in args.bundles:
        tag = Path(path).stem
        try:
            _b, recs = _one_bundle(path, raw, cfg)
        except (KeyError, ValueError, OSError) as e:
            print(f"skip {tag}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if not recs:
            print(f"skip {tag}: no held-out sensor (n=6?)", file=sys.stderr)
            continue
        for r in recs:
            r["tag"] = tag
            r["bundle"] = str(path)
        records.extend(recs)

    if not records:
        print("no leave-one-out comparisons produced", file=sys.stderr)
        return 1

    records.sort(key=lambda r: r["rel_l2"])
    print("\n===== leave-one-out (predicted vs measured at the held-out "
          "sensor) =====")
    print(f"  {'held-out':<9} {'rel-L2':>8}   bundle")
    print("  " + "-" * 60)
    for r in records:
        print(f"  {r['label']:<9} {r['rel_l2']:>8.4f}   {r['tag']}")
    rels = np.array([r["rel_l2"] for r in records])
    print(f"\n  median rel-L2 {np.median(rels):.4f}   "
          f"max {rels.max():.4f} (worst held-out sensor)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _render(records, out_dir / "loo.png")
    (out_dir / "summary.json").write_text(json.dumps(dict(
        real=str(args.real),
        window_s=[float(cfg.t_start), float(cfg.t_cutoff)],
        median_rel_l2=float(np.median(rels)), max_rel_l2=float(rels.max()),
        comparisons=[dict(held_out=r["label"], rel_l2=r["rel_l2"],
                          bundle=r["tag"]) for r in records]), indent=2))
    print(f"\nwrote loo.png + summary.json to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
