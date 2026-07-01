"""One-off overnight retrain + viz pipeline.

Edit the constants in the CONFIG block below to change any flag for
any of the three steps (train / viz_all / viz_test_cases). Everything
downstream picks the value up from the constants; no CLI arguments
are read.

Run:
    python run_overnight.py

Detach for actual overnight use:
    nohup python run_overnight.py > overnight.log 2>&1 &
    tail -f overnight.log

Fail-fast: if any step exits non-zero the pipeline stops with the
same exit code. Timestamped banner before each step so the morning
log-read shows exactly where wall time went.
"""
# ================== CONFIG ==================

# Common
NPZ_DIR = "/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/"
TAG = "firehorse1_and_2_clean"
OUT_ROOT = f"viz/{TAG}"

# --- step 1: scripts/train.py ---
TRAIN_CONFIG = "configs/default.yaml"
TRAIN_DATA_WORKERS = 32
TRAIN_POD_WORKERS = 32
# Any additional --key value pairs to pass to train.py (rare; e.g.
# ["--pod.k", "6"] to sweep K, or ["--model.channels", "128"] for a
# bigger model). Set to [] for defaults.
TRAIN_EXTRA_ARGS: list = []

# --- step 2: scripts/viz_all.py ---
VIZ_ALL_SHOW_LOWER = True          # translucent lower-wafer plane on 3D viz
VIZ_ALL_TOPN_WORST = 5
VIZ_ALL_N_SAMPLES = 3              # per-sim viz count
VIZ_ALL_EXTRA_ARGS: list = []

# --- step 3: scripts/viz_test_cases.py ---
PICKS = ["worst", "best", "median", "random"]
TOPN = 5
LAYOUTS = ["kymo", "radial_anim"]
SEED = 0                           # for --pick random determinism
VIZ_TEST_EXTRA_ARGS: list = []

# =============================================

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _banner(idx, total, title: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 66}")
    print(f"  {now}   [{idx}/{total}]  {title}")
    print(f"{'=' * 66}", flush=True)


def _run(cmd: list) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n[FAIL] exit={r.returncode}", file=sys.stderr)
        sys.exit(r.returncode)


def main() -> int:
    t0 = time.time()
    py = sys.executable

    _banner(0, 3, f"pipeline start  tag={TAG!r}  npz_dir={NPZ_DIR!r}")
    print(f"out root: {OUT_ROOT}", flush=True)
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    # ---- 1. train ----
    _banner(1, 3, "TRAIN  (expect ~6-8h cold, ~30-60 min if caches HIT)")
    _run([py, "scripts/train.py",
          "--config", TRAIN_CONFIG,
          "--data.npz_dir", NPZ_DIR,
          "--data.workers", str(TRAIN_DATA_WORKERS),
          "--pod.workers", str(TRAIN_POD_WORKERS),
          "--tag", TAG,
          *TRAIN_EXTRA_ARGS])

    # ---- 2. viz_all ----
    _banner(2, 3, "VIZ_ALL  (diversity + per-sim + POD + ML + worst)")
    cmd = [py, "scripts/viz_all.py",
           "--npz-dir", NPZ_DIR,
           "--tag", TAG,
           "--out", OUT_ROOT,
           "--topn-worst", str(VIZ_ALL_TOPN_WORST),
           "--n-samples", str(VIZ_ALL_N_SAMPLES)]
    if VIZ_ALL_SHOW_LOWER:
        cmd.append("--show-lower")
    cmd.extend(VIZ_ALL_EXTRA_ARGS)
    _run(cmd)

    # ---- 3. viz_test_cases ----
    _banner(3, 3, "VIZ_TEST_CASES  (batched; 4 picks x 2 layouts)")
    _run([py, "scripts/viz_test_cases.py",
          "--tag", TAG,
          "--pick", ",".join(PICKS),
          "--topn", str(TOPN),
          "--layout", ",".join(LAYOUTS),
          "--seed", str(SEED),
          "--out", f"{OUT_ROOT}/picks/",
          *VIZ_TEST_EXTRA_ARGS])

    wall = time.time() - t0
    h = int(wall // 3600); m = int((wall % 3600) // 60)
    _banner("done", 3, f"DONE  (wall {h}h{m}m)")
    print(f"tag:       {TAG}")
    print(f"viz root:  {OUT_ROOT}")
    print(f"key files to check first:")
    print(f"  outputs/{TAG}/results.json")
    print(f"  {OUT_ROOT}/ml/err_vs_floor.png")
    print(f"  {OUT_ROOT}/pod/spectrum.png")
    print(f"  {OUT_ROOT}/picks/worst/*_radial.gif")
    print(f"  {OUT_ROOT}/picks/best/*_radial.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
