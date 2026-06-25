"""Compare ResultSet JSONs and produce plots.

Usage:
    python scripts/visualize.py outputs/exp1/results.json outputs/exp2/results.json
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from evaluation.result import ResultSet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", type=str, default="outputs/plots")
    args = parser.parse_args()

    results = [(Path(p).parent.name, ResultSet.load_json(p))
               for p in args.results]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return

    labels = [lab for lab, _ in results]
    meds = [r.global_stats["median"] for _, r in results]
    p95s = [r.global_stats["p95"] for _, r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, title in zip(axes, [meds, p95s], ["median", "p95"]):
        ax.bar(labels, vals)
        ax.set_ylabel(f"field rel-L2 ({title})")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130, bbox_inches="tight")
    print(f"saved {out / 'comparison.png'}")

    print(f"\n{'label':>20s}  {'med':>8s}  {'p95':>8s}  {'gap':>6s}  {'params':>10s}")
    for label, r in results:
        g = r.global_stats
        print(f"{label:>20s}  {g['median']:8.4f}  {g['p95']:8.4f}  "
              f"{r.gap_to_floor:5.2f}x  {r.n_params:>10,}")


if __name__ == "__main__":
    main()
