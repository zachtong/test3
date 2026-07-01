#!/usr/bin/env bash
# Overnight retrain + full viz pipeline in one command.
#
# Runs:
#   1. scripts/train.py               (~6-8h cold, cache rebuild)
#   2. scripts/viz_all.py             (~30 min: diversity + per-sim + POD + ML + worst)
#   3. scripts/viz_test_cases.py     (~10 min: 4 picks x 2 layouts, batched)
#
# Positional args:
#   $1  TAG      training run tag       (default: firehorse1_and_2_clean)
#   $2  NPZ_DIR  data folder            (default: firehorse1_and_2 path)
#
# Usage:
#   bash scripts/overnight_retrain.sh
#   bash scripts/overnight_retrain.sh mytag /path/to/other_data
#
#   # Detach from tty (recommended for overnight):
#   nohup bash scripts/overnight_retrain.sh > overnight.log 2>&1 &
#   tail -f overnight.log
#
# Fails fast: if any step exits non-zero the whole pipeline stops.
# Times each step so you can see where time went in the morning.

set -euo pipefail

TAG="${1:-firehorse1_and_2_clean}"
NPZ_DIR="${2:-/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/}"
OUT_ROOT="viz/${TAG}"

# ---- pre-flight ---------------------------------------------------
if [ ! -d "$NPZ_DIR" ]; then
    echo "ERROR: NPZ_DIR does not exist: $NPZ_DIR" >&2
    exit 2
fi
mkdir -p "$OUT_ROOT"

# ---- timers -------------------------------------------------------
_wall_start=$(date +%s)
_step() {
    _now=$(date '+%Y-%m-%d %H:%M:%S')
    _elapsed=$(( $(date +%s) - _wall_start ))
    echo ""
    echo "===================================================="
    echo "  $_now  (+${_elapsed}s)"
    echo "  $1"
    echo "===================================================="
}

_step "[0/3] pipeline started; tag=$TAG npz_dir=$NPZ_DIR"

# ---- 1. train -----------------------------------------------------
_step "[1/3] TRAIN  (expect ~6-8h cold, ~30-60 min if caches HIT)"
python scripts/train.py --config configs/default.yaml \
    --data.npz_dir "$NPZ_DIR" \
    --data.workers 32 --pod.workers 32 \
    --tag "$TAG"

# ---- 2. viz_all ---------------------------------------------------
_step "[2/3] VIZ_ALL  (diversity + per-sim + POD + ML + worst-5)"
python scripts/viz_all.py \
    --npz-dir "$NPZ_DIR" \
    --tag "$TAG" \
    --out "$OUT_ROOT" \
    --show-lower \
    --topn-worst 5

# ---- 3. viz_test_cases -------------------------------------------
_step "[3/3] VIZ_TEST_CASES  (4 picks x 2 layouts, batched inference)"
python scripts/viz_test_cases.py \
    --tag "$TAG" \
    --pick worst,best,median,random \
    --topn 5 \
    --layout kymo,radial_anim \
    --out "${OUT_ROOT}/picks/"

# ---- done ---------------------------------------------------------
_wall_end=$(date +%s)
_wall_total=$(( _wall_end - _wall_start ))
_h=$(( _wall_total / 3600 ))
_m=$(( (_wall_total % 3600) / 60 ))
echo ""
echo "===================================================="
echo "  DONE  (wall: ${_h}h${_m}m)"
echo "  tag:       $TAG"
echo "  viz root:  $OUT_ROOT"
echo "  key files to check first:"
echo "    outputs/$TAG/results.json"
echo "    $OUT_ROOT/ml/err_vs_floor.png"
echo "    $OUT_ROOT/pod/spectrum.png"
echo "    $OUT_ROOT/picks/worst/*_radial.gif"
echo "    $OUT_ROOT/picks/best/*_radial.gif"
echo "===================================================="
