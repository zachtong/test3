"""Sensor-subset sweep. Runs train + viz_test_cases for every non-
trivial subset of the 6 physically-realizable sensor locations.

Physical sensor positions (fixed by lab hardware):
    A = (r=0.52,  theta=0 deg)     inner-X
    B = (r=0.52,  theta=45 deg)    inner-D
    C = (r=0.52,  theta=90 deg)    inner-Y
    D = (r=0.847, theta=0 deg)     outer-X
    E = (r=0.847, theta=45 deg)    outer-D
    F = (r=0.847, theta=90 deg)    outer-Y

Sweep = every C(6, n) subset for n in {2, 3, 4, 5, 6} -- 57 configs
total. Priority order (highest information value first, in case the
job runs long and needs to be interrupted):
    n=6 (1)  -> baseline upper bound
    n=3 (20) -> minimal-set analysis, densest info
    n=5 (6)  -> drop-one sensitivity
    n=4 (15) -> mid-range breadth
    n=2 (15) -> lower-bound diagnostic, most disposable

Every config is tagged `sweep_n{N}_{code}` where code is the sorted
subset ID (e.g. `sweep_n3_ADE`), and its outputs land in:
    outputs/sweep_n{N}_{code}/results.json + checkpoints
    viz/sweep_n{N}_{code}/all_picks/*

After the run finishes (or gets Ctrl-C'd), aggregate with
    python scripts/summarize_sweep.py
which reads every outputs/sweep_*/results.json and writes a
sorted CSV + markdown report to viz/sweep_summary.{csv,md}.

Run:
    python run_sweep.py
    # or detach:
    nohup python run_sweep.py > sweep.log 2>&1 &
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

PY = sys.executable

# Default when --npz-dir is not supplied. Change or override via CLI
# when running on a different dataset.
DEFAULT_NPZ_DIR = "/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/"

# The six physical sensor locations. Ordered so subset codes read
# left-to-right as (inner-then-outer, 0-then-45-then-90).
POSITIONS = [
    ("A", 0.52,  0.0),
    ("B", 0.52,  45.0),
    ("C", 0.52,  90.0),
    ("D", 0.847, 0.0),
    ("E", 0.847, 45.0),
    ("F", 0.847, 90.0),
]

# Priority order for the sweep. n=6 first (upper bound), then n=3
# (most info per config), descending sizes, n=2 last.
PRIORITY_N = [6, 3, 5, 4, 2]


def _coords_lookup() -> dict:
    return {p[0]: (p[1], p[2]) for p in POSITIONS}


def build_configs(tag_prefix: str) -> list[dict]:
    """Enumerate every subset in the priority order defined above."""
    coords = _coords_lookup()
    ids = [p[0] for p in POSITIONS]
    out = []
    for n in PRIORITY_N:
        for combo in combinations(ids, n):
            code = "".join(combo)
            positions = [list(coords[c]) for c in combo]
            out.append({
                "n": n,
                "code": code,
                "positions": positions,
                "tag": f"{tag_prefix}_n{n}_{code}",
            })
    return out


def run_train(cfg: dict, npz_dir: str) -> bool:
    positions_json = json.dumps(cfg["positions"])
    cmd = [
        PY, "scripts/train.py",
        "--config", "configs/default.yaml",
        "--data.npz_dir", npz_dir,
        "--data.workers", "64",
        "--pod.workers", "64",
        "--sensors.n", str(cfg["n"]),
        "--sensors.strategy", "custom",
        "--sensors.positions", positions_json,
        "--seeds", "[7]",
        "--tag", cfg["tag"],
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL train] {cfg['tag']} exit={e.returncode}",
              flush=True)
        return False


def run_viz(cfg: dict, pick: str, topn: int, layout: str,
            radial_max_frames: int, radial_dpi: int,
            radial_fps: int) -> bool:
    cmd = [
        PY, "scripts/viz_test_cases.py",
        "--tag", cfg["tag"],
        "--out", f"viz/{cfg['tag']}/all_picks/",
        "--pick", pick,
        "--topn", str(topn),
        "--layout", layout,
        "--show-lower",
        "--radial-max-frames", str(radial_max_frames),
        "--radial-dpi", str(radial_dpi),
        "--radial-fps", str(radial_fps),
    ]
    print(f"[viz] {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL viz] {cfg['tag']} exit={e.returncode}",
              flush=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR,
                    help=f"path to the dataset NPZ folder "
                    f"(default: {DEFAULT_NPZ_DIR})")
    ap.add_argument("--tag-prefix", default="sweep",
                    help="prefix for every config's tag; final tags "
                    "look like <prefix>_n{N}_{code}. Use a distinct "
                    "prefix per dataset so outputs/ + viz/ do not "
                    "collide (e.g. --tag-prefix smalltest_sweep).")
    ap.add_argument("--viz-pick",
                    default="worst,best,median,random",
                    help="comma list passed to viz_test_cases "
                    "--pick (default: worst,best,median,random)")
    ap.add_argument("--viz-topn", type=int, default=10,
                    help="how many sims per pick strategy the viz "
                    "step should render (default: 10)")
    ap.add_argument("--viz-layout",
                    default="snapshot,kymo,radial_anim,"
                    "interactive_compare",
                    help="comma list passed to viz_test_cases "
                    "--layout. Remove 'interactive_compare' to "
                    "skip the slow plotly HTML step; remove "
                    "'radial_anim' to skip the slow GIF step.")
    ap.add_argument("--viz-radial-max-frames", type=int, default=60,
                    help="frame count for radial_anim GIF. Default "
                    "60 matches original quality. 30 halves render "
                    "time with barely-noticeable smoothness loss "
                    "at fps=18.")
    ap.add_argument("--viz-radial-dpi", type=int, default=100,
                    help="dpi for radial_anim GIF (default: 100). "
                    "Lower to ~80 for further render + file-size "
                    "savings; text stays readable.")
    ap.add_argument("--viz-radial-fps", type=int, default=18,
                    help="fps for radial_anim GIF (default: 18)")
    ap.add_argument("--skip-viz", action="store_true",
                    help="skip the per-config viz_test_cases step "
                    "entirely. results.json is still written by "
                    "train.py so summarize_sweep.py + "
                    "analyze_sweep.py work as normal -- they only "
                    "need results.json, not the pick images. Use "
                    "this to get the sweep summary as fast as "
                    "possible; render viz later for the winners "
                    "via run_reviz_sweep.py.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip configs where outputs/<tag>/"
                    "results.json already exists. Use to resume "
                    "a sweep after killing/tweaking mid-run "
                    "without re-training the completed configs. "
                    "Note: viz on the already-done configs stays "
                    "with the ORIGINAL viz args (they were run "
                    "before the tweak); use run_reviz_sweep.py "
                    "later if you need them re-rendered.")
    args = ap.parse_args()

    configs = build_configs(args.tag_prefix)
    total = len(configs)
    print(f"Sweep: {total} configs on {args.npz_dir}, single seed=7",
          flush=True)
    print(f"  viz-pick={args.viz_pick} viz-topn={args.viz_topn}",
          flush=True)
    print(f"  viz-layout={args.viz_layout}", flush=True)
    print(f"  skip-existing={args.skip_existing}", flush=True)
    print(f"Priority-ordered: "
          f"{[cfg['tag'] for cfg in configs[:5]]} ... "
          f"{[cfg['tag'] for cfg in configs[-3:]]}", flush=True)

    t_start = time.time()
    train_ok = viz_ok = failed = skipped = 0
    for i, cfg in enumerate(configs):
        elapsed_hr = (time.time() - t_start) / 3600
        print(f"\n===== [{i + 1}/{total}] {cfg['tag']}  "
              f"(elapsed {elapsed_hr:.2f}h) =====", flush=True)
        if args.skip_existing:
            results_json = Path(f"outputs/{cfg['tag']}/results.json")
            if results_json.is_file():
                print(f"  SKIP: {results_json} exists", flush=True)
                skipped += 1
                continue
        if run_train(cfg, args.npz_dir):
            train_ok += 1
            if args.skip_viz:
                print(f"  [--skip-viz] skipping viz_test_cases",
                      flush=True)
            elif run_viz(cfg, args.viz_pick, args.viz_topn,
                          args.viz_layout,
                          args.viz_radial_max_frames,
                          args.viz_radial_dpi,
                          args.viz_radial_fps):
                viz_ok += 1
            else:
                failed += 1
        else:
            failed += 1
    elapsed_hr = (time.time() - t_start) / 3600
    print(f"\nsweep done in {elapsed_hr:.2f}h -- "
          f"train_ok={train_ok}/{total}  viz_ok={viz_ok}/{total}  "
          f"failed={failed}  skipped={skipped}", flush=True)
    print("Aggregate with: python scripts/summarize_sweep.py",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
