"""One-shot placement report: run the diff-vs-ABCDEF comparison and
build the HTML report + per-figure PDFs, auto-discovering the study
figures under viz/ so you do not have to spell out every path.

You give the two tags (optimized + ABCDEF baseline); this:
  1. runs diagnose_worst_cases --out-json on them,
  2. globs viz/ for the known study figures (k-vs-sensor,
     diminishing returns, sensor importance, QR-DEIM overlay,
     differentiable placement, POD mode atlas),
  3. calls make_placement_report.py with whatever it found.

Anything not found is skipped (the report still builds). Override
any auto-discovered path with the matching --fig-* flag.

    python run_placement_report.py \\
        --diff-tag merged_diffplace_k12_n6_verify \\
        --abcdef-tag merged_sweep_k12_n6_ABCDEF
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def _first(patterns, viz):
    """First existing file matching any of the glob patterns under
    viz/ (searched recursively). Returns str path or None."""
    for pat in patterns:
        hits = sorted(Path(viz).rglob(pat))
        if hits:
            return str(hits[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diff-tag", required=True,
                    help="the optimized-placement model tag")
    ap.add_argument("--abcdef-tag", required=True,
                    help="the fixed ABCDEF baseline tag (same "
                    "dataset + K)")
    ap.add_argument("--viz", default="viz",
                    help="root to search for figures (default: viz)")
    ap.add_argument("--outputs", default="outputs",
                    help="dir with <tag>/results.json (default: "
                    "outputs)")
    ap.add_argument("--out-dir", default="viz/placement_report")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--title",
                    default="Sensor Placement and Project Status")
    # explicit overrides for any auto-discovery miss
    ap.add_argument("--fig-k-vs-sensor", default=None)
    ap.add_argument("--fig-diminishing", default=None)
    ap.add_argument("--fig-importance", default=None)
    ap.add_argument("--fig-qrdeim", default=None)
    ap.add_argument("--fig-diffplace", default=None)
    ap.add_argument("--fig-atlas", default=None,
                    help="POD mode atlas figure (added as an extra)")
    args = ap.parse_args()

    viz = args.viz
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- step 1: comparison JSON ---
    cmp_json = out_dir / "diff_vs_abcdef.json"
    print(f"[1/3] diagnose_worst_cases -> {cmp_json}", flush=True)
    r = subprocess.run(
        [PY, "scripts/diagnose_worst_cases.py",
         "--tags", args.diff_tag, args.abcdef_tag,
         "--top-n", str(args.top_n),
         "--outputs", args.outputs,
         "--out-json", str(cmp_json)])
    if r.returncode != 0 or not cmp_json.is_file():
        print("comparison step failed; aborting", file=sys.stderr)
        return 1

    # --- step 2: auto-discover figures ---
    print(f"[2/3] discovering figures under {viz}/", flush=True)
    found = {
        "k_vs_sensor": args.fig_k_vs_sensor or _first(
            ["*k_vs_sensor*.png", "*k_vs_sensor_box*.png"], viz),
        "diminishing": args.fig_diminishing or _first(
            ["diminishing_returns.png"], viz),
        "importance": args.fig_importance or _first(
            ["sensor_importance.png"], viz),
        "qrdeim": args.fig_qrdeim or _first(
            ["*qrdeim*.png", "*qr_deim*.png"], viz),
        "diffplace": args.fig_diffplace or _first(
            ["*diffplace*.png", "*differentiable*.png"], viz),
        "atlas": args.fig_atlas or _first(
            ["*mode_atlas*.png", "*pod_mode_atlas*.png"], viz),
    }
    for k, v in found.items():
        print(f"    {k:12s} {'-> ' + v if v else '(not found)'}",
              flush=True)

    # --- step 3: build report ---
    print(f"[3/3] building report -> {out_dir}", flush=True)
    cmd = [PY, "scripts/make_placement_report.py",
           "--out-dir", str(out_dir),
           "--title", args.title,
           "--k", str(args.k),
           "--compare-json", str(cmp_json)]
    for flag, key in [("--fig-k-vs-sensor", "k_vs_sensor"),
                      ("--fig-diminishing", "diminishing"),
                      ("--fig-importance", "importance"),
                      ("--fig-qrdeim", "qrdeim"),
                      ("--fig-diffplace", "diffplace")]:
        if found[key]:
            cmd += [flag, found[key]]
    if found["atlas"]:
        cmd += ["--extra", f"POD mode atlas (K={args.k}):"
                f"{found['atlas']}"]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("report build failed", file=sys.stderr)
        return 1
    print(f"\nDone. Open {out_dir / 'report.html'}; "
          f"standalone PDFs in {out_dir / 'pdf'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
