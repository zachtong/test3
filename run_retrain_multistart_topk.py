"""Retrain the multistart top-K layouts with the FULL seed ensemble.

The search phase trained m_multistart_top{j} with a single seed for cheap
ranking, but the six init/opt models are 3-seed ensembles -- comparing them
is seed-count-unfair. This deletes each single-seed tag and retrains it with
the config's full seed set (default [7, 17, 27]) so the comparison is
apples-to-apples. Positions are read from the search's exported
<pos-dir>/top{j}.json; nothing is re-searched.

    python run_retrain_multistart_topk.py --npz-dir /path/to/dataset
    # then: python run_compare_all.py
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", required=True,
                    help="dataset dir (REQUIRED; same one the six models "
                    "were trained on)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="how many top layouts to retrain (default 3)")
    ap.add_argument("--pos-dir", default="viz/diffplace/multistart",
                    help="dir holding the search's top{j}.json exports")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pod-k", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pod-workers", type=int, default=64)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--seeds", default=None,
                    help="JSON seed list to pass to train.py (default: omit "
                    "-> config default [7, 17, 27])")
    ap.add_argument("--no-clean", action="store_true",
                    help="do NOT delete the existing (single-seed) tag dirs "
                    "first. Without cleaning, train.py reuses any completed "
                    "seed checkpoints it finds.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen without deleting/training")
    args = ap.parse_args()

    pos_dir = Path(args.pos_dir)
    outputs = Path(args.outputs)
    status = {}
    for j in range(1, args.top_k + 1):
        tag = f"m_multistart_top{j}"
        pf = pos_dir / f"top{j}.json"
        if not pf.is_file():
            print(f"[skip] {tag}: no {pf}", file=sys.stderr)
            status[tag] = "skipped (no positions)"
            continue
        pos = json.loads(pf.read_text())

        tag_dir = outputs / tag
        if not args.no_clean and tag_dir.exists():
            if args.dry_run:
                print(f"[clean] (dry-run) would delete {tag_dir}")
            else:
                shutil.rmtree(tag_dir)
                print(f"[clean] deleted {tag_dir}")

        cmd = [PY, "scripts/train.py", "--config", args.config,
               "--data.npz_dir", args.npz_dir,
               "--pod.workers", str(args.pod_workers),
               "--pod.k", str(args.pod_k), "--sensors.n", str(args.n),
               "--sensors.strategy", "custom",
               "--sensors.positions", json.dumps(pos),
               "--tag", tag]
        if args.seeds:
            cmd += ["--seeds", args.seeds]
        print(f"\n[train] {tag}  pos={json.dumps(pos)}", flush=True)
        if args.dry_run:
            print(f"        (dry-run) {' '.join(cmd)}")
            status[tag] = "dry-run"
            continue
        try:
            subprocess.run(cmd, check=True)
            status[tag] = "trained (full seeds)"
        except subprocess.CalledProcessError as e:
            print(f"  train FAILED for {tag}: {e}", file=sys.stderr)
            status[tag] = "FAILED"

    print("\n===== retrain summary =====")
    for tag, st in status.items():
        print(f"  {tag:<22} {st}")
    print("\nNext: python run_compare_all.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
