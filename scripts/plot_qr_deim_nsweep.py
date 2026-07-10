"""Headline figure for the QR-DEIM n-sweep: reconstruction error vs
sensor count under OPTIMAL placement, with the fixed hardware
ABCDEF marked as a reference point.

Reads outputs/<prefix>_n{n}_k{K}/results.json for each n and
optionally the ABCDEF config's results.json. Plots median (and p95)
field error vs n, annotates where the curve flattens (adding
sensors stops paying off = the K modes are fully observed), and
drops a marker at ABCDEF's (n=6, error) so you can read off how
far the buildable hardware sits from the optimal-placement curve.

    python scripts/plot_qr_deim_nsweep.py \\
        --prefix qrdeim --K 12 \\
        --abcdef-tag merged_sweep_k12_n6_ABCDEF \\
        --out viz/qrdeim_nsweep.png
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(tag_dir: Path) -> dict | None:
    res = tag_dir / "results.json"
    if not res.is_file():
        return None
    try:
        d = json.loads(res.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    gs = d.get("global_stats", {}) or {}
    return dict(median=gs.get("median"), p95=gs.get("p95"),
                gap=d.get("gap_to_floor"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="qrdeim",
                    help="tag prefix used by run_qr_deim_nsweep")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--abcdef-tag", default=None,
                    help="tag of the fixed-hardware ABCDEF config at "
                    "the same K, to mark as a reference point")
    ap.add_argument("--value-scale", type=float, default=100.0,
                    help="multiply error for display (default 100 "
                    "= percent relative L2)")
    ap.add_argument("--out", default="viz/qrdeim_nsweep.png")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    scale = args.value_scale
    unit = "%" if abs(scale - 100.0) < 1e-9 else f"x{scale:g}"

    # Discover all <prefix>_n{n}_k{K} configs.
    pat = re.compile(rf"^{re.escape(args.prefix)}_n(\d+)_k{args.K}$")
    rows = []
    for d in sorted(outputs.glob(f"{args.prefix}_n*_k{args.K}")):
        m = pat.match(d.name)
        if not m:
            continue
        r = _load(d)
        if r is None or r["median"] is None:
            continue
        rows.append(dict(n=int(m.group(1)), **r))
    if not rows:
        print(f"no {args.prefix}_n*_k{args.K} results in {outputs}",
              file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["n"])
    ns = np.array([r["n"] for r in rows])
    med = np.array([r["median"] for r in rows]) * scale
    p95 = np.array([(r["p95"] if r["p95"] is not None else np.nan)
                    for r in rows]) * scale

    abcdef = None
    if args.abcdef_tag:
        abcdef = _load(outputs / args.abcdef_tag)

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.plot(ns, med, "o-", color="#3d5a80", lw=2.2, markersize=8,
            label="median (QR-DEIM optimal)", zorder=5)
    finite95 = np.isfinite(p95)
    if finite95.any():
        ax.plot(ns[finite95], p95[finite95], "s--",
                color="#e9c46a", lw=1.6, markersize=6,
                label="p95 (QR-DEIM optimal)", zorder=4)

    # Mark where the median curve flattens: first n whose next-step
    # relative improvement drops below 5%.
    knee = None
    for i in range(len(med) - 1):
        if med[i] > 0 and (med[i] - med[i + 1]) / med[i] < 0.05:
            knee = ns[i]
            break
    if knee is not None:
        ax.axvline(knee, color="#3d5a80", ls=":", lw=1, alpha=0.5)
        ax.annotate(f"knee ~n={knee}\n(more sensors stop paying off)",
                    (knee, med[list(ns).index(knee)]),
                    xytext=(12, 24), textcoords="offset points",
                    fontsize=9, color="#3d5a80")

    if abcdef and abcdef["median"] is not None:
        av = abcdef["median"] * scale
        ax.scatter([6], [av], s=220, marker="*", color="#e63946",
                   edgecolor="black", linewidth=1.0, zorder=6,
                   label=f"ABCDEF hardware (n=6): {av:.3f}{unit}")
        # how far above the optimal n=6 point?
        opt6 = None
        if 6 in set(ns):
            opt6 = med[list(ns).index(6)]
        if opt6 is not None:
            gap_pct = (av - opt6) / opt6 * 100 if opt6 else float("nan")
            ax.annotate(
                f"ABCDEF is {gap_pct:+.1f}% vs optimal n=6",
                (6, av), xytext=(10, -28),
                textcoords="offset points", fontsize=9,
                color="#e63946")

    ax.set_xlabel("number of sensors n", fontsize=11)
    ax.set_ylabel(f"field error ({unit})", fontsize=11)
    ax.set_title(f"Reconstruction error vs sensor count "
                 f"(QR-DEIM optimal placement, K={args.K})",
                 fontsize=12)
    ax.set_xticks(ns)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for r in rows:
        print(f"  n={r['n']:>2}: median={r['median'] * scale:.4f}{unit}"
              f"  gap_to_floor={r['gap']:.3f}"
              if r["gap"] is not None else
              f"  n={r['n']:>2}: median={r['median'] * scale:.4f}{unit}")
    if abcdef and abcdef["median"] is not None:
        print(f"  ABCDEF (n=6): median="
              f"{abcdef['median'] * scale:.4f}{unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
