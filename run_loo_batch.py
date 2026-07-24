"""Batch leave-one-out over every CSV in a folder.

Runs scripts/eval_real_loo.py on each *.csv under --csv-dir (each writes into
its own <out-dir>/<csv-stem>/ subfolder, so runs never collide), then writes a
cross-file summary: a table + a bar chart of each run's median / worst held-out
rel-L2 and its chosen window, so you can see at a glance which runs
reconstruct self-consistently and which are suspect.

    python run_loo_batch.py --csv-dir /data/real_runs \\
        --bundles bundles/merged_sweep_k12_n5_*.pt \\
        --config configs/real_exp_n6.yaml --auto-cutoff \\
        --out-dir viz/real_loo
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-dir", required=True,
                    help="folder of real CSVs to process")
    ap.add_argument("--glob", default="*.csv",
                    help="which files to pick up (default '*.csv')")
    ap.add_argument("--bundles", nargs="+", required=True,
                    help="the subset bundles (e.g. bundles/*_n5_*.pt)")
    ap.add_argument("--config", default=None,
                    help="channel config with all six sensors")
    ap.add_argument("--out-dir", default="viz/real_loo")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a CSV whose <out-dir>/<stem>/summary.json exists")
    # window controls -- forwarded to eval_real_loo (same for every CSV)
    ap.add_argument("--auto-cutoff", action="store_true")
    ap.add_argument("--auto-cutoff-halfwidth", type=float, default=None)
    ap.add_argument("--auto-cutoff-step", type=float, default=None)
    ap.add_argument("--sweep-t-start", nargs=3, type=float, default=None)
    ap.add_argument("--sweep-t-cutoff", nargs=3, type=float, default=None)
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-cutoff", type=float, default=None)
    # deployment-field animations -- forwarded to eval_real_loo per CSV
    ap.add_argument("--field-bundle", default=None,
                    help="bundle for the field animations (default: best LOO "
                    "bundle; pass the n6 ABCDEF bundle for the deployment field)")
    ap.add_argument("--no-anim", action="store_true",
                    help="skip the field animations for every run")
    ap.add_argument("--anim-fps", type=int, default=None)
    ap.add_argument("--anim-frames", type=int, default=None)
    ap.add_argument("--front-r-max", type=float, default=None,
                    help="cap the bonding-front search radius (<=1.0)")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    csvs = sorted(csv_dir.glob(args.glob))
    if not csvs:
        print(f"no files matching {args.glob!r} in {csv_dir}", file=sys.stderr)
        return 1
    out_root = Path(args.out_dir)
    print(f"processing {len(csvs)} CSV(s) from {csv_dir}/")

    def _fwd():
        f = []
        if args.config:
            f += ["--config", args.config]
        if args.auto_cutoff:
            f += ["--auto-cutoff"]
        if args.auto_cutoff_halfwidth is not None:
            f += ["--auto-cutoff-halfwidth", str(args.auto_cutoff_halfwidth)]
        if args.auto_cutoff_step is not None:
            f += ["--auto-cutoff-step", str(args.auto_cutoff_step)]
        if args.sweep_t_start is not None:
            f += ["--sweep-t-start", *map(str, args.sweep_t_start)]
        if args.sweep_t_cutoff is not None:
            f += ["--sweep-t-cutoff", *map(str, args.sweep_t_cutoff)]
        if args.t_start is not None:
            f += ["--t-start", str(args.t_start)]
        if args.t_cutoff is not None:
            f += ["--t-cutoff", str(args.t_cutoff)]
        if args.field_bundle:
            f += ["--field-bundle", args.field_bundle]
        if args.no_anim:
            f += ["--no-anim"]
        if args.anim_fps is not None:
            f += ["--anim-fps", str(args.anim_fps)]
        if args.anim_frames is not None:
            f += ["--anim-frames", str(args.anim_frames)]
        if args.front_r_max is not None:
            f += ["--front-r-max", str(args.front_r_max)]
        return f

    rows, status = [], {}
    for csv in csvs:
        summ = out_root / csv.stem / "summary.json"
        if args.skip_existing and summ.is_file():
            print(f"[skip] {csv.name}: {summ} exists")
        else:
            cmd = [PY, "scripts/eval_real_loo.py", "--bundles", *args.bundles,
                   "--real", str(csv), "--out-dir", str(out_root)] + _fwd()
            print(f"\n[loo] {csv.name}", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"  FAILED for {csv.name}", file=sys.stderr)
                status[csv.name] = "FAILED"
                continue
        if summ.is_file():
            d = json.loads(summ.read_text())
            rows.append(dict(
                name=csv.stem,
                median=d.get("median_rel_l2"), worst=d.get("max_rel_l2"),
                window=d.get("window_s"),
                end=(d.get("sweep") or {}).get("auto_end_of_bond_s")))
            status[csv.name] = "done"

    if not rows:
        print("no summaries produced", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["median"] is None, r["median"]))
    print("\n===== batch LOO summary "
          f"({len(rows)} runs) =====")
    print(f"  {'run':<22} {'median':>8} {'worst':>8}  {'window (s)':<16} "
          f"{'end':>6}")
    print("  " + "-" * 70)
    for r in rows:
        win = (f"[{r['window'][0]:.2f},{r['window'][1]:.2f}]"
               if r["window"] else "--")
        end = f"{r['end']:.2f}" if r["end"] is not None else "--"
        print(f"  {r['name']:<22} {r['median']:>8.4f} {r['worst']:>8.4f}  "
              f"{win:<16} {end:>6}")

    _render_batch(rows, out_root / "batch_loo_summary.png")
    (out_root / "batch_loo_summary.json").write_text(json.dumps(dict(
        csv_dir=str(csv_dir), n_runs=len(rows), runs=rows,
        status=status), indent=2))
    print(f"\nwrote batch_loo_summary.png + .json to {out_root}/")
    return 0


def _render_batch(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = [r["name"] for r in rows]
    med = np.array([r["median"] for r in rows], dtype=float)
    worst = np.array([r["worst"] for r in rows], dtype=float)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2),
                           constrained_layout=True)
    ax.barh(y - 0.2, med, 0.4, color="#3d5a80", label="median held-out rel-L2")
    ax.barh(y + 0.2, worst, 0.4, color="#e63946", label="worst held-out rel-L2")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("held-out rel-L2 (lower = more self-consistent)")
    ax.set_title("Batch leave-one-out self-consistency across runs")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
