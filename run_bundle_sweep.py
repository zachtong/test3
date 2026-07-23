"""Batch-bundle trained sweep models into self-contained .pt files.

Every tag under --outputs/<tag>/ that has a results.json (filtered by --prefix
and/or --contains) is packaged via scripts/bundle.py, all sharing ONE
--basis-file: the sweep models share the dataset + grid, so a single
pod3d_*.npz whose k_cache >= their K serves them all. Output:
--out-dir/<tag>.pt per model.

    # bundle every ABCDEF-sweep model (all subsets n=2..6) at K=12
    python run_bundle_sweep.py \\
        --basis-file outputs/basis_cache/pod3d_<key>.npz \\
        --prefix merged_sweep_k12

    # only the full 6-sensor one
    python run_bundle_sweep.py --basis-file ... --prefix merged_sweep_k12 \\
        --contains n6
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def _discover(outputs: Path, prefix: str, contains: str) -> list[str]:
    tags = []
    for rj in sorted(outputs.glob("*/results.json")):
        tag = rj.parent.name
        if prefix and not tag.startswith(prefix):
            continue
        if contains and contains not in tag:
            continue
        tags.append(tag)
    return tags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis-file", required=True,
                    help="shared pod3d_*.npz (k_cache >= the models' K)")
    ap.add_argument("--prefix", default="merged_sweep",
                    help="only tags starting with this (default 'merged_sweep')")
    ap.add_argument("--contains", default="",
                    help="further restrict to tags containing this substring "
                    "(e.g. 'n6', 'ABCDEF')")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out-dir", default="bundles")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a tag whose <out-dir>/<tag>.pt already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the tags that would be bundled, do nothing")
    args = ap.parse_args()

    if not args.dry_run and not Path(args.basis_file).is_file():
        print(f"basis file not found: {args.basis_file}", file=sys.stderr)
        return 2

    outputs = Path(args.outputs)
    out_dir = Path(args.out_dir)
    tags = _discover(outputs, args.prefix, args.contains)
    if not tags:
        print(f"no tags under {outputs}/ match prefix={args.prefix!r} "
              f"contains={args.contains!r}", file=sys.stderr)
        return 1
    print(f"matched {len(tags)} tag(s): {', '.join(tags)}")
    if args.dry_run:
        for t in tags:
            print(f"  (dry-run) would bundle {t} -> {out_dir / (t + '.pt')}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for tag in tags:
        pt = out_dir / f"{tag}.pt"
        if args.skip_existing and pt.is_file():
            print(f"[skip] {tag}: {pt} exists")
            status[tag] = "skipped (exists)"
            continue
        cmd = [PY, "scripts/bundle.py", "--tag", tag,
               "--basis-file", args.basis_file, "--out", str(pt),
               "--output-dir", args.outputs]
        print(f"\n[bundle] {tag}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            status[tag] = "bundled"
        except subprocess.CalledProcessError as e:
            print(f"  bundle FAILED for {tag}: {e}", file=sys.stderr)
            status[tag] = "FAILED"

    print("\n===== bundle summary =====")
    for tag, st in status.items():
        print(f"  {tag:<32} {st}")
    n_ok = sum(1 for s in status.values() if s in ("bundled", "skipped (exists)"))
    print(f"\n{n_ok}/{len(tags)} available in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
