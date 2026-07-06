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


def run_viz(cfg: dict) -> bool:
    cmd = [
        PY, "scripts/viz_test_cases.py",
        "--tag", cfg["tag"],
        "--out", f"viz/{cfg['tag']}/all_picks/",
        "--pick", "worst,best,median,random",
        "--topn", "10",
        "--layout",
        "snapshot,kymo,radial_anim,interactive_compare",
        "--show-lower",
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
    args = ap.parse_args()

    configs = build_configs(args.tag_prefix)
    total = len(configs)
    print(f"Sweep: {total} configs on {args.npz_dir}, single seed=7,"
          f" viz --topn 10 per config", flush=True)
    print(f"Priority-ordered: "
          f"{[cfg['tag'] for cfg in configs[:5]]} ... "
          f"{[cfg['tag'] for cfg in configs[-3:]]}", flush=True)

    t_start = time.time()
    train_ok = viz_ok = failed = 0
    for i, cfg in enumerate(configs):
        elapsed_hr = (time.time() - t_start) / 3600
        print(f"\n===== [{i + 1}/{total}] {cfg['tag']}  "
              f"(elapsed {elapsed_hr:.2f}h) =====", flush=True)
        if run_train(cfg, args.npz_dir):
            train_ok += 1
            if run_viz(cfg):
                viz_ok += 1
            else:
                failed += 1
        else:
            failed += 1
    elapsed_hr = (time.time() - t_start) / 3600
    print(f"\nsweep done in {elapsed_hr:.2f}h -- "
          f"train_ok={train_ok}/{total}  viz_ok={viz_ok}/{total}  "
          f"failed={failed}", flush=True)
    print("Aggregate with: python scripts/summarize_sweep.py",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
