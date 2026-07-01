"""One-off overnight retrain + viz pipeline (CLI-flag configured).

Chains scripts/train.py -> scripts/viz_all.py -> scripts/viz_test_cases.py.
Every knob is a CLI flag with a sensible default; --help prints the
full list. Fail-fast: any step's non-zero exit stops the pipeline
with the same exit code.

Basic usage:
    python run_overnight.py \\
        --npz-dir /path/to/data/ \\
        --tag firehorse1_and_2_clean

Detach for actual overnight run:
    nohup python run_overnight.py <flags> > overnight.log 2>&1 &
    tail -f overnight.log

Anything not exposed as a top-level flag can go through the
--train-extra / --viz-all-extra / --viz-test-extra strings, which
are shell-split and appended to the respective subprocess command
lines. Example:
    --train-extra "--pod.k 6 --model.channels 128"
"""
from __future__ import annotations
import argparse
import shlex
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Common
    ap.add_argument("--npz-dir", required=True,
                    help="folder of converted 3D NPZ files")
    ap.add_argument("--tag", required=True,
                    help="training-run tag (outputs/<tag>/, viz/<tag>/)")
    ap.add_argument("--out-root", default=None,
                    help="viz output root (default: viz/<tag>)")

    # Step 1: train
    g1 = ap.add_argument_group("step 1: train")
    g1.add_argument("--train-config", default="configs/default.yaml")
    g1.add_argument("--data-workers", type=int, default=32)
    g1.add_argument("--pod-workers", type=int, default=32)
    g1.add_argument("--train-extra", default="",
                    help="extra --key value pairs for train.py, "
                    "shell-split (e.g. --train-extra \"--pod.k 6\")")

    # Step 2: viz_all
    g2 = ap.add_argument_group("step 2: viz_all")
    g2.add_argument("--show-lower", action="store_true", default=True,
                    help="translucent lower-wafer plane on 3D viz "
                    "(default: on)")
    g2.add_argument("--no-show-lower", dest="show_lower",
                    action="store_false")
    g2.add_argument("--topn-worst", type=int, default=5)
    g2.add_argument("--n-samples", type=int, default=3,
                    help="per-sim viz count")
    g2.add_argument("--viz-all-extra", default="",
                    help="extra flags for viz_all.py")

    # Step 3: viz_test_cases
    g3 = ap.add_argument_group("step 3: viz_test_cases")
    g3.add_argument("--picks", default="worst,best,median,random",
                    help="comma list of pick modes")
    g3.add_argument("--topn", type=int, default=5,
                    help="per-pick sim count")
    g3.add_argument("--layouts", default="kymo,radial_anim",
                    help="comma list of layouts")
    g3.add_argument("--seed", type=int, default=0,
                    help="rng seed for --pick random")
    g3.add_argument("--viz-test-extra", default="",
                    help="extra flags for viz_test_cases.py")

    # Which steps to run (defaults: all)
    ap.add_argument("--skip", default="",
                    help="comma list of steps to skip: "
                    "'train', 'viz_all', 'viz_test'")

    args = ap.parse_args()

    out_root = args.out_root or f"viz/{args.tag}"
    py = sys.executable
    skips = {s.strip() for s in args.skip.split(",") if s.strip()}
    Path(out_root).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _banner(0, 3,
            f"pipeline start  tag={args.tag!r}  npz_dir={args.npz_dir!r}")
    print(f"out root: {out_root}")
    if skips:
        print(f"skipping: {sorted(skips)}")

    # ---- 1. train ----
    if "train" not in skips:
        _banner(1, 3, "TRAIN  (expect ~6-8h cold, ~30-60 min if caches HIT)")
        _run([py, "scripts/train.py",
              "--config", args.train_config,
              "--data.npz_dir", args.npz_dir,
              "--data.workers", str(args.data_workers),
              "--pod.workers", str(args.pod_workers),
              "--tag", args.tag,
              *shlex.split(args.train_extra)])

    # ---- 2. viz_all ----
    if "viz_all" not in skips:
        _banner(2, 3, "VIZ_ALL  (diversity + per-sim + POD + ML + worst)")
        cmd = [py, "scripts/viz_all.py",
               "--npz-dir", args.npz_dir,
               "--tag", args.tag,
               "--out", out_root,
               "--topn-worst", str(args.topn_worst),
               "--n-samples", str(args.n_samples)]
        if args.show_lower:
            cmd.append("--show-lower")
        cmd.extend(shlex.split(args.viz_all_extra))
        _run(cmd)

    # ---- 3. viz_test_cases ----
    if "viz_test" not in skips:
        _banner(3, 3, "VIZ_TEST_CASES  (batched; picks x layouts)")
        _run([py, "scripts/viz_test_cases.py",
              "--tag", args.tag,
              "--pick", args.picks,
              "--topn", str(args.topn),
              "--layout", args.layouts,
              "--seed", str(args.seed),
              "--out", f"{out_root}/picks/",
              *shlex.split(args.viz_test_extra)])

    wall = time.time() - t0
    h = int(wall // 3600); m = int((wall % 3600) // 60)
    _banner("done", 3, f"DONE  (wall {h}h{m}m)")
    print(f"tag:       {args.tag}")
    print(f"viz root:  {out_root}")
    print(f"key files to check first:")
    print(f"  outputs/{args.tag}/results.json")
    print(f"  {out_root}/ml/err_vs_floor.png")
    print(f"  {out_root}/pod/spectrum.png")
    print(f"  {out_root}/picks/worst/*_radial.gif")
    print(f"  {out_root}/picks/best/*_radial.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
