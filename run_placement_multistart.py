"""One-shot driver for the multi-start placement study: make sure all six
models exist and compare them. Three init families -- outer-uniform, diag-45,
and ABCDEF -- each contributes an INIT model and its cartesian-OPTIMIZED model.

For each family it builds/derives the position set, trains whatever is missing
via scripts/train.py (positions passed as a single subprocess argv element, so
no shell quoting can mangle the [ ] brackets), and finally runs
scripts/compare_placements.py over every model that has a results.json.

  init sets   : outer-uniform + diag-45 built to match the diffplace presets;
                ABCDEF init reuses the existing trained sweep model.
  optimized   : read straight from each diffplace --save-history npz (best_pos).

Already-trained tags are skipped with --skip-existing, so a second run only
trains the ones still missing (typically just m_optABCDEF_cart).

    python run_placement_multistart.py --compare --skip-existing
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
_NPZ_DEFAULT = "/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/"


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
        print(f"  MISSING history: {p} -- skipping its optimized model. Run "
              f"the diffplace optimization with --save-history first.",
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


def _build_plan(args):
    """(tag, label, kind, src). kind: 'init' src=positions list; 'opt'
    src=hist npz path; 'existing' src=None (already trained, compare only)."""
    return [
        (args.tag_uniform_init, "init-uniform", "init",
         _init_uniform(args.n, args.r_max)),
        (args.tag_uniform_opt, "opt-uniform", "opt", args.uniform_hist),
        (args.tag_diag45_init, "init-45", "init",
         _init_diag45(args.n, args.r_min, args.r_max)),
        (args.tag_diag45_opt, "opt-45", "opt", args.diag45_hist),
        (args.abcdef_tag, "ABCDEF", "existing", None),
        (args.tag_abcdef_opt, "opt-ABCDEF", "opt", args.abcdef_hist),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", default=_NPZ_DEFAULT,
                    help="merged dataset dir (default: firehorse_1_and_2)")
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
    ap.add_argument("--abcdef-hist",
                    default="viz/diffplace/abcdef_cart_hist.npz")
    ap.add_argument("--pos-dir", default="viz/diffplace",
                    help="where the per-model positions JSON is written")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--tag-uniform-init", default="m_initUniform")
    ap.add_argument("--tag-uniform-opt", default="m_optUniform")
    ap.add_argument("--tag-diag45-init", default="m_initDiag45")
    ap.add_argument("--tag-diag45-opt", default="m_optDiag45")
    ap.add_argument("--abcdef-tag", default="merged_sweep_k12_n6_ABCDEF",
                    help="existing trained ABCDEF-init model (compare only)")
    ap.add_argument("--tag-abcdef-opt", default="m_optABCDEF_cart")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a tag whose outputs/<tag>/results.json exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run without training")
    ap.add_argument("--compare", action="store_true",
                    help="run scripts/compare_placements.py over all six")
    ap.add_argument("--k-gap", type=float, default=None,
                    help="K=8->K=12 worst-N drop (fraction) for the flatness "
                    "verdict, e.g. 0.006")
    ap.add_argument("--compare-out", default="viz/placement_compare.png")
    args = ap.parse_args()

    pos_dir = Path(args.pos_dir)
    pos_dir.mkdir(parents=True, exist_ok=True)
    outputs = Path(args.outputs)

    status = {}
    for tag, label, kind, src in _build_plan(args):
        rj = outputs / tag / "results.json"
        if kind == "existing":
            status[tag] = "existing" if rj.is_file() else "MISSING results.json"
            continue
        positions = src if kind == "init" else _optimized_from_hist(src)
        if positions is None:
            status[tag] = "skipped (no history)"
            continue
        (pos_dir / f"{tag}.json").write_text(json.dumps(positions))
        if args.skip_existing and rj.is_file():
            print(f"[skip] {tag}: {rj} exists")
            status[tag] = "skipped (exists)"
            continue
        status[tag] = "trained" if _train(tag, positions, args) else "FAILED"

    print("\n===== model status =====")
    for tag, st in status.items():
        print(f"  {tag:<28} {st}")

    if args.compare and not args.dry_run:
        plan = _build_plan(args)
        tags, labels = [], []
        for tag, label, _kind, _src in plan:
            if (outputs / tag / "results.json").is_file():
                tags.append(tag)
                labels.append(label)
        if len(tags) < 2:
            print("\nnot enough trained models to compare", file=sys.stderr)
            return 1
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
