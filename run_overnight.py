"""Overnight retrain + viz. Values hardcoded per Zach's spec.

Run:
    python run_overnight.py
    # or detach:
    nohup python run_overnight.py > overnight.log 2>&1 &
"""
import subprocess
import sys

PY = sys.executable

# --- 1. train ---
subprocess.run([
    PY, "scripts/train.py",
    "--config", "configs/default.yaml",
    "--data.npz_dir", "/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/",
    "--data.workers", "64",
    "--pod.workers", "64",
    "--tag", "firehorse1_and_2_clean",
], check=True)

# --- 2. viz_all ---
# Skip the interactive HTML: it needs WebGL to view (broken on this
# Linux) and gif_3d + radial_anim + kymo cover the same info.
subprocess.run([
    PY, "scripts/viz_all.py",
    "--npz-dir", "/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/",
    "--out", "viz/firehorse1_and_2_clean",
    "--tag", "firehorse1_and_2_clean",
    "--show-lower",
    "--exclude", "interactive",
], check=True)

# --- 3. viz_test_cases ---
subprocess.run([
    PY, "scripts/viz_test_cases.py",
    "--tag", "firehorse1_and_2_clean",
    "--out", "viz/firehorse1_and_2_clean/all_picks/",
    "--pick", "worst,best,median,random",
    "--topn", "10",
    "--layout", "snapshot,kymo,radial_anim",
], check=True)
