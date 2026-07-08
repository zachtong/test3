"""K sweep: train the SAME sensor config at multiple POD K values.

Motivated by the per-sim diagnostic (scripts/diagnose_worst_cases.py):
when the worst cases show gap_to_floor ~ 1, the model is already
at the K=8 POD floor for those sims and sensor placement cannot
improve them. The path forward is a larger K basis -- more POD
modes lets f_perp shrink for the hard cases.

This script reads the sensor config (n, strategy, positions) from
an existing trained tag's results.json, then re-trains at each
requested K on the same data + sensor set. Output tags are
<tag-prefix>_k{K} so results are cleanly separated.

After the run, feed all tags into diagnose_worst_cases.py --tags
to see which K value moves the worst-case floor down and by how
much.

Run:
    python run_k_sweep.py \\
        --source-tag <shortname>_sweep_n5_ABDEF \\
        --npz-dir /data/... \\
        --tag-prefix <shortname>_ksweep \\
        --k-values 8,12,16,24

Notes:
  - The basis cache stores modes up to its saved k_cache. Training
    at K=24 refits the basis at k_cache=24 (or configured max),
    which then satisfies all K in [8, 12, 16, 24] via slicing --
    no per-K basis refit needed once the largest K has run.
  - Recommend running the LARGEST K FIRST for that reason.
  - Skips viz by default; use scripts/diagnose_worst_cases.py
    after training to compare per-sim gaps across K values.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def _load_sensor_config(source_tag: str, outputs: Path) -> dict:
    res = outputs / source_tag / "results.json"
    if not res.is_file():
        raise SystemExit(f"source results.json not found: {res}")
    d = json.loads(res.read_text())
    sens = d.get("config", {}).get("sensors", {})
    if not sens:
        raise SystemExit(
            f"no sensors block in {res}; is this a trained tag?")
    return sens


def run_train(cfg: dict, k: int, npz_dir: str, tag: str,
              seeds: str) -> bool:
    positions_json = json.dumps(cfg["positions"])
    cmd = [
        PY, "scripts/train.py",
        "--config", "configs/default.yaml",
        "--data.npz_dir", npz_dir,
        "--data.workers", "64",
        "--pod.workers", "64",
        "--pod.k", str(k),
        "--sensors.n", str(cfg["n"]),
        "--sensors.strategy",
        str(cfg.get("strategy", "custom")),
        "--sensors.positions", positions_json,
        "--seeds", seeds,
        "--tag", tag,
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] K={k} exit={e.returncode}", flush=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-tag", required=True,
                    help="tag of an existing trained model; its "
                    "sensor config is reused for every K value")
    ap.add_argument("--npz-dir", required=True,
                    help="path to the dataset NPZ folder "
                    "(must match the sweep the source-tag came "
                    "from -- otherwise the basis cache misses "
                    "and refits)")
    ap.add_argument("--tag-prefix", required=True,
                    help="prefix for the new tags "
                    "(e.g. smalltest_ksweep -> smalltest_ksweep_k12)")
    ap.add_argument("--k-values", default="8,12,16,24",
                    help="comma-separated K values to train "
                    "(default: 8,12,16,24). Largest first is "
                    "recommended so the basis cache is fit at "
                    "max k_cache once and sliced for smaller K.")
    ap.add_argument("--outputs", default="outputs",
                    help="dir containing tag folders (default: "
                    "outputs)")
    ap.add_argument("--seeds", default="[7]",
                    help="JSON list of seeds to train (default: "
                    "[7] -- matches sensor sweep style)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip K values whose results.json "
                    "already exists")
    args = ap.parse_args()

    sens = _load_sensor_config(args.source_tag, Path(args.outputs))
    print(f"Reusing sensor config from {args.source_tag}:")
    print(f"  n = {sens.get('n')}")
    print(f"  strategy = {sens.get('strategy', 'custom')}")
    print(f"  positions = {sens.get('positions')}")

    ks = [int(k.strip()) for k in args.k_values.split(",")
          if k.strip()]
    # Descending so the largest K refits the basis first; smaller
    # K trains then slice the same basis file.
    ks_ordered = sorted(ks, reverse=True)
    print(f"\nK values to train (largest first): {ks_ordered}")
    print(f"seeds = {args.seeds}")
    print(f"skip-existing = {args.skip_existing}\n")

    t_start = time.time()
    trained = failed = skipped = 0
    for k in ks_ordered:
        tag = f"{args.tag_prefix}_k{k}"
        print(f"\n===== K={k} tag={tag} "
              f"(elapsed {(time.time() - t_start) / 60:.1f} min) =====")
        if args.skip_existing:
            done = (Path(args.outputs) / tag / "results.json"
                    ).is_file()
            if done:
                print(f"  SKIP: results.json already exists")
                skipped += 1
                continue
        if run_train(sens, k, args.npz_dir, tag, args.seeds):
            trained += 1
        else:
            failed += 1

    total_min = (time.time() - t_start) / 60
    print(f"\nk-sweep done in {total_min:.1f} min "
          f"({total_min / 60:.2f} h): trained={trained} "
          f"failed={failed} skipped={skipped}")
    tags_str = " ".join(
        f"{args.tag_prefix}_k{k}" for k in sorted(ks))
    print(f"\nCompare with:")
    print(f"  python scripts/diagnose_worst_cases.py "
          f"--tags {tags_str} --top-n 20")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
