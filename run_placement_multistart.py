"""One-shot driver for the multi-start placement study: train the four models
(outer-uniform init + its optimized layout, diag-45 init + its optimized
layout) and, optionally, compare all six against ABCDEF + optimized-ABCDEF.

It builds the two INIT position sets to match the differentiable-placement
presets, reads the two OPTIMIZED sets straight from their saved diffplace
history (best_pos), writes each as a JSON file for the record, and trains one
model per set via scripts/train.py. Positions are passed to the train
subprocess as a single argv element, so there is NO shell quoting to mangle
the [ ] brackets.

Prereq: you have already run the two diffplace optimizations and kept their
--save-history npz files, e.g.
    ... --init uniform-outer --param cartesian --save-history uniform_cart_hist.npz
    ... --init diag45        --param cartesian --save-history diag45_cart_hist.npz

    python run_placement_multistart.py \\
        --npz-dir /data/merged_dataset \\
        --uniform-hist viz/diffplace/uniform_cart_hist.npz \\
        --diag45-hist  viz/diffplace/diag45_cart_hist.npz \\
        --compare --k-gap 0.006
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable


def _round_pos(pos) -> list:
    return [[round(float(r), 4), round(float(t), 2)] for r, t in pos]


def _init_uniform(n, r_max):
    return [[round(float(r_max), 4), round(float(t), 2)]
            for t in np.linspace(0.0, 90.0, n)]


def _init_diag45(n, r_min, r_max):
    return [[round(float(r), 4), 45.0]
            for r in np.linspace(r_min, r_max, n)]


def _optimized_from_hist(hist_path):
    """best_pos (n, 2) from a diffplace --save-history npz, or None if the
    file is missing / malformed."""
    p = Path(hist_path)
    if not p.is_file():
        print(f"  MISSING history: {p} -- skipping its optimized model. "
              f"Run the diffplace optimization with --save-history first.",
              file=sys.stderr)
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            return _round_pos(z["best_pos"])
    except (OSError, KeyError, ValueError) as e:
        print(f"  BAD history {p}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _train(tag, positions, args) -> bool:
    cmd = [PY, "scripts/train.py",
           "--config", args.config,
           "--data.npz_dir", args.npz_dir,
           "--pod.workers", str(args.pod_workers),
           "--pod.k", str(args.pod_k),
           "--sensors.n", str(args.n),
           "--sensors.strategy", "custom",
           "--sensors.positions", json.dumps(positions),
           "--tag", tag]
    print(f"\n[train] {tag}\n        positions={json.dumps(positions)}",
          flush=True)
    if args.dry_run:
        print(f"        (dry-run) {' '.join(cmd)}")
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  train FAILED for {tag}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", required=True, help="merged dataset dir")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pod-k", type=int, default=12)
    ap.add_argument("--pod-workers", type=int, default=64)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--uniform-hist",
                    default="viz/diffplace/uniform_cart_hist.npz")
    ap.add_argument("--diag45-hist",
                    default="viz/diffplace/diag45_cart_hist.npz")
    ap.add_argument("--pos-dir", default="viz/diffplace",
                    help="where the per-model positions JSON is written")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--tag-uniform-init", default="m_initUniform")
    ap.add_argument("--tag-uniform-opt", default="m_optUniform")
    ap.add_argument("--tag-diag45-init", default="m_initDiag45")
    ap.add_argument("--tag-diag45-opt", default="m_optDiag45")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a tag whose outputs/<tag>/results.json exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run without training")
    # optional comparison at the end
    ap.add_argument("--compare", action="store_true",
                    help="run scripts/compare_placements.py over all six")
    ap.add_argument("--abcdef-tag", default="merged_sweep_k12_n6_ABCDEF")
    ap.add_argument("--opt-abcdef-tag",
                    default="merged_diffplace_k12_n6_verify")
    ap.add_argument("--k-gap", type=float, default=None,
                    help="K=8->K=12 worst-N drop (fraction) for the flatness "
                    "verdict, e.g. 0.006")
    ap.add_argument("--compare-out", default="viz/placement_compare.png")
    args = ap.parse_args()

    pos_dir = Path(args.pos_dir)
    pos_dir.mkdir(parents=True, exist_ok=True)

    # (tag, positions, label) for the four models; opt ones may be None
    opt_uniform = _optimized_from_hist(args.uniform_hist)
    opt_diag45 = _optimized_from_hist(args.diag45_hist)
    plan = [
        (args.tag_uniform_init, _init_uniform(args.n, args.r_max),
         "init-uniform"),
        (args.tag_uniform_opt, opt_uniform, "opt-uniform"),
        (args.tag_diag45_init, _init_diag45(args.n, args.r_min, args.r_max),
         "init-45"),
        (args.tag_diag45_opt, opt_diag45, "opt-45"),
    ]

    results = {}
    for tag, pos, label in plan:
        if pos is None:
            results[tag] = "skipped (no positions)"
            continue
        # record the positions used
        (pos_dir / f"{tag}.json").write_text(json.dumps(pos))
        rj = Path(args.outputs) / tag / "results.json"
        if args.skip_existing and rj.is_file():
            print(f"[skip] {tag}: {rj} exists")
            results[tag] = "skipped (exists)"
            continue
        results[tag] = "trained" if _train(tag, pos, args) else "FAILED"

    print("\n===== training summary =====")
    for tag, status in results.items():
        print(f"  {tag:<16} {status}")

    if args.compare and not args.dry_run:
        trained = [t for t, s in results.items()
                   if s in ("trained", "skipped (exists)")]
        tags = trained + [args.abcdef_tag, args.opt_abcdef_tag]
        # map tag -> label for the four; ABCDEF pair gets fixed labels
        lab_of = {p[0]: p[2] for p in plan}
        lab_of[args.abcdef_tag] = "ABCDEF"
        lab_of[args.opt_abcdef_tag] = "opt-ABCDEF"
        labels = [lab_of.get(t, t) for t in tags]
        cmd = [PY, "scripts/compare_placements.py",
               "--tags", *tags, "--labels", *labels,
               "--top-n", "20", "--outputs", args.outputs,
               "--out", args.compare_out,
               "--out-json", str(Path(args.compare_out).with_suffix(".json"))]
        if args.k_gap is not None:
            cmd += ["--k-gap", str(args.k_gap)]
        print(f"\n[compare] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
