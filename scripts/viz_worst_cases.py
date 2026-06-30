"""Visualise the top-N worst test sims for failure-mode triage.

Per worst sim: one PNG containing
  TOP ROW (3 full-disk heatmaps at t*):
    - GT field
    - Model-predicted field
    - |GT - prediction| (absolute error)
  BOTTOM ROW (K subplots):
    - a_k(t) predicted vs true line plot for each of the K POD modes,
      so the operator can see which mode mispredictions are driving
      the field error.

t* is the time index where |GT - prediction| peaks; this is the most
informative single snapshot to render. Files are named with the
per-sim rel-L2 + sim basename so they sort naturally:
  worst_<rel_l2>__<sim_basename>.png

Heavyweight: this script LOADS the dataset, rebuilds the split,
re-fits / cache-hits the POD basis, loads the 3 seed checkpoints,
runs inference. Cost ~1-2 minutes after caches are warm; ~5-30
minutes cold.

    python scripts/viz_worst_cases.py --tag firehorse2_n3_full \\
        --topn 5 --out viz/worst/
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

from scripts.fieldviz import (render_full_disk, provenance_footer,  # noqa: E402
                               WAFER_CMAP, SENSOR_MARKER_COLOR,
                               wafer_value_range)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True,
                    help="training run tag (i.e. outputs/<tag>/...)")
    ap.add_argument("--topn", type=int, default=5,
                    help="number of worst sims to render")
    ap.add_argument("--out", required=True,
                    help="output directory (one PNG per sim)")
    ap.add_argument("--output-dir", default="outputs",
                    help="where to find outputs/<tag>/ (default: outputs)")
    ap.add_argument("--data-dir-override", default=None,
                    help="override --data.npz_dir (e.g. when running on "
                    "a different machine than training)")
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    args = ap.parse_args()

    # Locate per-sim errors via results.json and pick top-N worst by rel-L2.
    results_path = (Path(args.output_dir) / args.tag /
                    "results.json").expanduser().resolve()
    if not results_path.is_file():
        print(f"results.json not found at {results_path}", file=sys.stderr)
        return 2
    with open(results_path) as fp:
        r = json.load(fp)
    field_errs = np.asarray(r.get("per_sim_field_errs", []), dtype=float)
    basenames = r.get("per_sim_basenames", [])
    if field_errs.size == 0 or not basenames:
        print("ERROR: results.json missing per_sim_field_errs or "
              "per_sim_basenames. Re-run training with the new scorer.py.",
              file=sys.stderr)
        return 1
    # Worst-N indices INTO the test split.
    worst_indices = np.argsort(field_errs)[-args.topn:][::-1]
    print(f"top-{args.topn} worst test sims by rel-L2:")
    for j in worst_indices:
        print(f"  test_idx={int(j):4d}  rel_l2={field_errs[j]:.4f}  "
              f"{basenames[j]}")

    # Predict + reconstruct
    from evaluation.run_predict import predict_run_fields
    overrides = {}
    if args.data_dir_override:
        overrides["data.npz_dir"] = args.data_dir_override
    out = predict_run_fields(args.tag, idx=worst_indices.tolist(),
                             output_dir=args.output_dir,
                             overrides=overrides, verbose=True)
    x_canon = out["x_canon"]; y_canon = out["y_canon"]
    K = out["K"]
    t = out["t"]
    sensor_xy = out["sensor_xy"]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for slot, test_idx in enumerate(out["idx"]):
        gt = out["w_true"][slot] * args.value_scale
        pr = out["w_pred"][slot] * args.value_scale
        err = np.abs(gt - pr)
        a_pred = out["a_pred"][slot]                # (K, Nt)
        a_true = out["a_true"][slot]
        bname = out["basenames"][slot]
        rel_l2 = float(field_errs[test_idx])

        # t* = time of max abs error (integrate over space).
        err_integrated = err.sum(axis=(0, 1))
        t_star = int(np.argmax(err_integrated))

        # Shared sequential colour scale for GT + pred (so the visual
        # comparison is valid). Error panel gets its own viridis range.
        vmin, vmax = wafer_value_range(
            np.stack([gt[..., t_star], pr[..., t_star]]))
        err_max = float(np.percentile(err[..., t_star], 99))
        if err_max <= 0:
            err_max = max(float(err.max()), 1e-12)

        n_cols = max(K, 3)
        fig, axes = plt.subplots(2, n_cols,
                                 figsize=(3.5 * n_cols, 7.2),
                                 constrained_layout=True)

        # TOP ROW: 3 full-disk heatmaps at t*
        for c, (panel_data, title, cmap, panel_vmin, panel_vmax) in enumerate([
                (gt[..., t_star], "GT", WAFER_CMAP, vmin, vmax),
                (pr[..., t_star], "predicted", WAFER_CMAP, vmin, vmax),
                (err[..., t_star], "abs error", "viridis", 0, err_max),
        ]):
            ax = axes[0, c]
            render_full_disk(ax, panel_data, x_canon, y_canon,
                             cmap=cmap, vmin=panel_vmin, vmax=panel_vmax,
                             mirror=True, mask_off_disk=True,
                             sensor_xy=sensor_xy)
            ax.set_title(f"{title}  t-idx {t_star}/{gt.shape[-1] - 1}",
                         fontsize=10)
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        # hide top-row spares
        for c in range(3, n_cols):
            axes[0, c].set_visible(False)

        # BOTTOM ROW: K subplots of a_k(t) predicted vs true
        for k_idx in range(K):
            c = k_idx
            ax = axes[1, c]
            ax.plot(t, a_true[k_idx], color="0.3", lw=1.2, label="true")
            ax.plot(t, a_pred[k_idx], color=SENSOR_MARKER_COLOR,
                    lw=1.2, ls="--", label="predicted")
            err_k = float(np.linalg.norm(a_pred[k_idx] - a_true[k_idx])
                          / max(np.linalg.norm(a_true[k_idx]), 1e-12))
            ax.set_title(f"a_{k_idx + 1}  rel-L2={err_k:.3f}",
                         fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlabel("normalised t")
            if c == 0:
                ax.legend(fontsize=7, loc="best")
        for c in range(K, n_cols):
            axes[1, c].set_visible(False)

        fig.suptitle(
            f"{args.tag}  |  test_idx={int(test_idx)}  |  "
            f"rel-L2={rel_l2:.4f}  |  {bname}", fontsize=12)
        provenance_footer(fig, sim_id=bname, tag=args.tag,
                          results_file=results_path,
                          extras={"test_idx": int(test_idx),
                                  "rel_l2": f"{rel_l2:.4f}",
                                  "t_star": t_star})

        # Filename: 0001_relL20.0823_<basename>.png so ls sorts by
        # worst-first; the 4-digit slot prefix avoids ties when two
        # sims share a basename root.
        stem = Path(bname).stem if bname else f"test{int(test_idx)}"
        fname = f"{slot:04d}_relL2{rel_l2:.4f}_{stem}.png"
        outp = out_dir / fname
        fig.savefig(outp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {outp}", flush=True)

    print(f"\nall {len(out['idx'])} worst-case figures -> {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
