"""Per-mode predicted-vs-true scatter, one panel per mode.

For each POD coefficient a_k, plot one point per test simulation:

  x = per-sim median rel-L2 error of THAT mode's prediction
      (= per_sim_per_mode_errs[a_k][i])
  y = per-sim overall field rel-L2 (per_sim_field_errs[i])

What this answers: does one mode's misprediction dominate the field
error? If a_6 points cluster high-x AND high-y vs a_1 points clumping
near low-x, the per-mode anomaly really IS driving the worst-case
field reconstructions and the next experiment should target that mode
(e.g. add a sensor that observes mode 6's spatial structure).

Per-panel colour points by per-sim rel-L2 (viridis), so each panel is
its own miniature scatter of "this mode's mispredictions vs the rest
of the field error". y=x diagonal drawn for reference; if points sit
ABOVE diagonal, this mode error explains less than the rest; BELOW,
this mode error explains more.

Input: results.json with per_sim_field_errs + per_sim_per_mode_errs
(written by the updated scorer.py; available from any 3D training run
post the fieldviz-suite commit). Requires K = number of modes recorded
in per_sim_per_mode_errs.

    python scripts/viz_ak_scatter.py \\
        --results outputs/<tag>/results.json \\
        --out viz/ak_scatter.png
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

from scripts.fieldviz import provenance_footer               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    with open(results_path) as fp:
        r = json.load(fp)
    field_errs = np.asarray(r.get("per_sim_field_errs", []), dtype=float)
    per_mode = r.get("per_sim_per_mode_errs", {})
    if field_errs.size == 0 or not per_mode:
        print("ERROR: results.json missing per_sim_field_errs or "
              "per_sim_per_mode_errs. Re-run training to populate.",
              file=sys.stderr)
        return 1
    # Sort modes by k (a_1, a_2, ...)
    mode_keys = sorted(per_mode.keys(),
                       key=lambda s: int(s.split("_")[1]))
    K = len(mode_keys)

    n_cols = (K + 1) // 2
    n_rows = 2 if K > 1 else 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.4 * n_rows),
                             constrained_layout=True, squeeze=False)
    # Global axis limits: log-scaled
    err_lo = max(min(field_errs.min(),
                     min(np.asarray(v).min() for v in per_mode.values())),
                 1e-6)
    err_hi = max(field_errs.max(),
                 max(np.asarray(v).max() for v in per_mode.values())) * 1.4

    for k_idx, key in enumerate(mode_keys):
        r_idx = k_idx // n_cols
        c_idx = k_idx % n_cols
        ax = axes[r_idx, c_idx]
        a_err = np.asarray(per_mode[key], dtype=float)
        # Diagonal + 2x / 5x guides
        ax.plot([err_lo, err_hi], [err_lo, err_hi],
                color="0.4", ls="--", lw=1, label="y = x")
        ax.plot([err_lo, err_hi], [err_lo * 0.5, err_hi * 0.5],
                color="0.8", ls=":", lw=0.8)
        ax.plot([err_lo, err_hi], [err_lo * 2.0, err_hi * 2.0],
                color="0.8", ls=":", lw=0.8)
        # Colour by per-sim field rel-L2 (viridis as agreed: unsigned)
        sc = ax.scatter(a_err, field_errs, c=field_errs,
                        cmap="viridis", s=18, alpha=0.85,
                        edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(err_lo, err_hi)
        ax.set_ylim(err_lo, err_hi)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{key}  (median: "
                     f"{float(np.median(a_err)):.3f})", fontsize=10)
        if r_idx == n_rows - 1:
            ax.set_xlabel("per-sim rel-L2 of THIS mode")
        if c_idx == 0:
            ax.set_ylabel("per-sim FIELD rel-L2")
        if k_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # hide unused
    for k_idx in range(K, n_rows * n_cols):
        axes[k_idx // n_cols, k_idx % n_cols].set_visible(False)

    fig.suptitle(f"per-mode error vs field error  |  n={field_errs.size} "
                 f"test sims  |  K={K}", fontsize=12)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85,
                        label="field rel-L2 (color)")
    provenance_footer(fig, tag=args.tag, results_file=results_path,
                      extras={"K": K})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
