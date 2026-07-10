"""Train the QR-DEIM optimal placement at each sensor count n, to
map the diminishing-returns curve under OPTIMAL placement (vs the
earlier brute-force n-sweep that mixed good and bad placements per
n).

QR-DEIM pivots are nested, so ONE QR call gives the ranked location
list; the first n form the n-optimal set. This driver computes that
list once (at n_max), then trains the prefix for each n at a FIXED
K so the only variable across configs is sensor count + placement.

Tags: <prefix>_n{n}_k{K} (e.g. qrdeim_n4_k12). Compare afterwards
against each other AND your fixed hardware ABCDEF (same K) via:

    python scripts/diagnose_worst_cases.py \\
        --tags qrdeim_n2_k12 ... qrdeim_n10_k12 <ABCDEF_k12_tag> \\
        --top-n 20

    python scripts/plot_qr_deim_nsweep.py \\
        --prefix qrdeim --K 12 --abcdef-tag <ABCDEF_k12_tag> \\
        --out viz/qrdeim_nsweep.png

Run:
    python run_qr_deim_nsweep.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --npz-dir /data/merged_dataset \\
        --n-min 2 --n-max 10 --K 12 \\
        --tag-prefix qrdeim --skip-existing
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.qr_deim_sensors import qr_deim_positions   # noqa: E402

PY = sys.executable


def run_train(positions, n: int, K: int, npz_dir: str,
              tag: str) -> bool:
    positions_json = json.dumps(positions)
    cmd = [
        PY, "scripts/train.py",
        "--config", "configs/default.yaml",
        "--data.npz_dir", npz_dir,
        "--data.workers", "64",
        "--pod.workers", "64",
        "--pod.k", str(K),
        "--sensors.n", str(n),
        "--sensors.strategy", "custom",
        "--sensors.positions", positions_json,
        "--seeds", "[7]",
        "--tag", tag,
    ]
    print(f"[train] n={n} K={K} tag={tag}", flush=True)
    print(f"  positions={positions_json}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {tag} exit={e.returncode}", flush=True)
        return False


def run_viz(tag: str, topn: int, layout: str,
            radial_max_frames: int, radial_dpi: int) -> bool:
    cmd = [
        PY, "scripts/viz_test_cases.py",
        "--tag", tag,
        "--out", f"viz/{tag}/all_picks/",
        "--pick", "worst,best,median,random",
        "--topn", str(topn),
        "--layout", layout,
        "--show-lower",
        "--radial-max-frames", str(radial_max_frames),
        "--radial-dpi", str(radial_dpi),
    ]
    print(f"[viz] {tag}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL viz] {tag} exit={e.returncode}", flush=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True,
                    help="basis_cache pod3d_*.npz for the dataset "
                    "(same dataset the sweep will train on)")
    ap.add_argument("--npz-dir", required=True,
                    help="dataset NPZ dir (must match the basis)")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--K", type=int, default=12,
                    help="fixed POD mode count for every config "
                    "(default: 12)")
    ap.add_argument("--tag-prefix", default="qrdeim",
                    help="tag prefix; tags are <prefix>_n{n}_k{K}")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--weight-sigma", action="store_true",
                    help="sigma-weight the QR (bias to high-energy "
                    "modes; sensible when n < K)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip n whose outputs/<tag>/results.json "
                    "already exists")
    ap.add_argument("--skip-viz", action="store_true",
                    help="train only, no per-config viz")
    ap.add_argument("--viz-topn", type=int, default=5)
    ap.add_argument("--viz-layout", default="snapshot,radial_anim")
    ap.add_argument("--viz-radial-max-frames", type=int, default=30)
    ap.add_argument("--viz-radial-dpi", type=int, default=80)
    args = ap.parse_args()

    basis_path = Path(args.basis)
    if not basis_path.is_file():
        print(f"basis not found: {basis_path}", file=sys.stderr)
        return 2

    # One QR call at n_max; nested prefixes give every n.
    try:
        res = qr_deim_positions(
            basis_path, args.n_max, K=args.K,
            r_min=args.r_min, r_max=args.r_max,
            weight_sigma=args.weight_sigma)
    except ValueError as e:
        print(f"ERROR computing QR-DEIM: {e}", file=sys.stderr)
        return 1
    full_positions = res["positions"]        # rank-ordered, len n_max
    K = res["K"]
    ns = list(range(args.n_min, args.n_max + 1))
    print(f"QR-DEIM n-sweep: n={args.n_min}..{args.n_max}, K={K}, "
          f"{res['n_candidates']} candidates in "
          f"r in [{args.r_min}, {args.r_max}]"
          f"{' (sigma-weighted)' if args.weight_sigma else ''}",
          flush=True)
    print(f"Full ranked placement (first n = n-optimal set):",
          flush=True)
    for i, p in enumerate(full_positions):
        print(f"  rank {i + 1:>2}: r={p[0]:.4f} theta={p[1]:.2f}",
              flush=True)

    t_start = time.time()
    trained = failed = skipped = 0
    for n in ns:
        tag = f"{args.tag_prefix}_n{n}_k{K}"
        positions = full_positions[:n]
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n===== n={n} ({tag})  elapsed {elapsed_min:.1f} min "
              f"=====", flush=True)
        if args.skip_existing and Path(
                f"outputs/{tag}/results.json").is_file():
            print(f"  SKIP: results.json exists", flush=True)
            skipped += 1
            continue
        if run_train(positions, n, K, args.npz_dir, tag):
            trained += 1
            if not args.skip_viz:
                run_viz(tag, args.viz_topn, args.viz_layout,
                        args.viz_radial_max_frames,
                        args.viz_radial_dpi)
        else:
            failed += 1

    total_min = (time.time() - t_start) / 60
    print(f"\nqr-deim n-sweep done in {total_min:.1f} min: "
          f"trained={trained} failed={failed} skipped={skipped}",
          flush=True)
    tags = " ".join(f"{args.tag_prefix}_n{n}_k{K}" for n in ns)
    print(f"\nCompare with:")
    print(f"  python scripts/diagnose_worst_cases.py --tags "
          f"{tags} <ABCDEF_k{K}_tag> --top-n 20", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
