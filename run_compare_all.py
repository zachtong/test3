"""Compare the six init/opt placement models plus the multistart top-K in one
table -- tags and labels are built in, so no long argument lists to retype.

Includes every model that has an outputs/<tag>/results.json; missing ones are
listed and skipped. Emits the usual three figures via compare_placements:
<out>, <out stem>_configs.png (one panel per layout), <out stem>_metrics.png
(median / p95 / worst-N bars).

    python run_compare_all.py
    python run_compare_all.py --top-k 5 --k-gap 0.006
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable

_SIX = [
    ("m_initUniform", "init-uniform"),
    ("m_optUniform", "opt-uniform"),
    ("m_initDiag45", "init-45"),
    ("m_optDiag45", "opt-45"),
    ("m_initABCDEF", "init-ABCDEF"),
    ("m_optABCDEF_cart", "opt-ABCDEF"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, default=3,
                    help="how many m_multistart_top{j} to include (default 3)")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--k-gap", type=float, default=None,
                    help="optional K=8->12 worst-N drop (fraction) for the "
                    "flatness verdict")
    ap.add_argument("--out", default="viz/placement_compare_all.png")
    ap.add_argument("--extra-tag", action="append", default=[],
                    metavar="TAG:LABEL",
                    help="additional 'tag:label' to include; repeatable")
    args = ap.parse_args()

    wanted = list(_SIX)
    wanted += [(f"m_multistart_top{j}", f"ms-top{j}")
               for j in range(1, args.top_k + 1)]
    for spec in args.extra_tag:
        if ":" not in spec:
            print(f"bad --extra-tag (need tag:label): {spec}", file=sys.stderr)
            return 2
        tag, label = spec.split(":", 1)
        wanted.append((tag.strip(), label.strip()))

    outputs = Path(args.outputs)
    tags, labels, missing = [], [], []
    for tag, label in wanted:
        if (outputs / tag / "results.json").is_file():
            tags.append(tag)
            labels.append(label)
        else:
            missing.append(tag)
    if missing:
        print("missing (skipped): " + ", ".join(missing))
    if len(tags) < 2:
        print("fewer than 2 trained models found; nothing to compare",
              file=sys.stderr)
        return 1

    cmd = [PY, "scripts/compare_placements.py",
           "--tags", *tags, "--labels", *labels,
           "--top-n", str(args.top_n), "--outputs", args.outputs,
           "--out", args.out,
           "--out-json", str(Path(args.out).with_suffix(".json"))]
    if args.k_gap is not None:
        cmd += ["--k-gap", str(args.k_gap)]
    print(f"[compare] {len(tags)} models: {', '.join(labels)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
