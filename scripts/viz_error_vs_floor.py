"""Per-sim error vs POD truncation floor scatter, with marginal histograms.

The single most informative ML diagnostic for a sparse-sensing project.
Each point is one test sim:

  x = truncation_floor_i (the K-mode oracle reconstruction error, i.e.
      the LOWEST relative L2 the model could ever reach for sim i with
      this K, this train+val set)
  y = field_err_i        (the model's actual relative L2 on sim i)

Three regions of the scatter tell different stories:

  - close to y = x diagonal -> model has fully exploited the basis
  - well above diagonal     -> model is the bottleneck (regression /
                                sensor / data)
  - well below diagonal     -> impossible (you found extra structure
                                the K-mode basis cannot describe?
                                check for a bug)

The colour-by-max|a_6| variant directly tests the per-mode-error
hypothesis: if a_6 mispredictions dominate per-sim error, those points
will be the bright ones.

Marginal histograms on top + right show the distributions individually
so the operator can read median + tail at a glance.

    python scripts/viz_error_vs_floor.py \\
        --results outputs/<tag>/results.json --out viz/err_vs_floor.png
    python scripts/viz_error_vs_floor.py \\
        --results outputs/<tag>/results.json --color-by a_6 \\
        --out viz/err_vs_floor_a6.png
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
    ap.add_argument("--results", required=True,
                    help="path to outputs/<tag>/results.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--color-by", default="none",
                    help="'none' (single colour), 'a_<k>' (colour by "
                    "that mode's predicted-vs-true rel-L2 per sim), or "
                    "'all_modes_max' (colour by max across all modes)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    with open(results_path) as fp:
        r = json.load(fp)

    field_errs = np.asarray(r.get("per_sim_field_errs", []), dtype=float)
    floor_errs = np.asarray(r.get("per_sim_floor_errs", []), dtype=float)
    if field_errs.size == 0 or floor_errs.size == 0:
        print("ERROR: results.json has no per_sim_field_errs / "
              "per_sim_floor_errs. Re-run training with the new "
              "scorer.py that records them.", file=sys.stderr)
        return 1
    if field_errs.shape != floor_errs.shape:
        print(f"ERROR: shape mismatch field {field_errs.shape} vs "
              f"floor {floor_errs.shape}", file=sys.stderr)
        return 1

    n = field_errs.size
    median_field = float(np.median(field_errs))
    median_floor = float(np.median(floor_errs))
    gap = median_field / max(median_floor, 1e-12)
    print(f"loaded {n} sims  med field {median_field:.4f}  "
          f"med floor {median_floor:.4f}  gap {gap:.2f}x")

    # --- colour mapping ---
    color_vals = None
    color_label = None
    if args.color_by != "none":
        pm = r.get("per_sim_per_mode_errs", {})
        if args.color_by.startswith("a_"):
            if args.color_by not in pm:
                print(f"--color-by {args.color_by} not in "
                      f"per_sim_per_mode_errs (have {list(pm)})",
                      file=sys.stderr)
                return 1
            color_vals = np.asarray(pm[args.color_by], dtype=float)
            color_label = f"{args.color_by} rel-L2"
        elif args.color_by == "all_modes_max":
            if not pm:
                print("no per_sim_per_mode_errs in results.json",
                      file=sys.stderr)
                return 1
            stacked = np.stack(
                [np.asarray(v, dtype=float) for v in pm.values()],
                axis=0)
            color_vals = stacked.max(axis=0)
            color_label = "max over modes (rel-L2)"
        else:
            print(f"--color-by must be 'none', 'a_<k>', or "
                  f"'all_modes_max' (got {args.color_by})",
                  file=sys.stderr)
            return 1
        assert color_vals.shape == field_errs.shape

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(8, 8), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05,
                           left=0.10, right=0.95, top=0.94, bottom=0.10)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    if color_vals is not None:
        sc = ax_main.scatter(floor_errs, field_errs, c=color_vals,
                             cmap="viridis", s=18, alpha=0.85,
                             edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax_right, shrink=0.9,
                            location="right", pad=0.15,
                            label=color_label)
    else:
        ax_main.scatter(floor_errs, field_errs, s=18, alpha=0.7,
                        c="C0", edgecolors="none")

    # Diagonal y = x
    lo = min(field_errs.min(), floor_errs.min()) * 0.5
    hi = max(field_errs.max(), floor_errs.max()) * 1.5
    lo = max(lo, 1e-6)
    ax_main.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1,
                 label="y = x (perfect)")
    # y = 2x and y = 5x reference lines
    ax_main.plot([lo, hi], [2 * lo, 2 * hi], color="0.7", ls=":", lw=0.8,
                 label="y = 2x")
    ax_main.plot([lo, hi], [5 * lo, 5 * hi], color="0.85", ls=":", lw=0.8,
                 label="y = 5x")

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("truncation floor (per-sim rel-L2)")
    ax_main.set_ylabel("model error (per-sim rel-L2)")
    ax_main.grid(alpha=0.3, which="both")
    ax_main.legend(loc="lower right", fontsize=8)

    # Marginals: log-spaced bins
    nb = 40
    bins = np.geomspace(lo, hi, nb)
    ax_top.hist(floor_errs, bins=bins, color="0.6", edgecolor="black",
                lw=0.3)
    ax_top.axvline(median_floor, color="C3", lw=1.2,
                   label=f"med {median_floor:.4f}")
    ax_top.set_xscale("log")
    ax_top.set_xlim(lo, hi)
    ax_top.set_ylabel("count")
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.grid(alpha=0.3)

    ax_right.hist(field_errs, bins=bins, color="0.6", edgecolor="black",
                  lw=0.3, orientation="horizontal")
    ax_right.axhline(median_field, color="C3", lw=1.2,
                     label=f"med {median_field:.4f}")
    ax_right.set_yscale("log")
    ax_right.set_ylim(lo, hi)
    ax_right.set_xlabel("count")
    ax_right.tick_params(labelleft=False)
    ax_right.legend(fontsize=8, loc="upper right")
    ax_right.grid(alpha=0.3)

    fig.suptitle(
        f"per-sim error vs floor   |   n={n}   gap_to_floor={gap:.2f}x",
        fontsize=11)
    provenance_footer(fig, sim_id=None, tag=args.tag,
                      results_file=results_path,
                      extras={"color_by": args.color_by})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
