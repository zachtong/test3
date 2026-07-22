"""Redraw the multi-start top-K sensor layouts from a saved summary.json,
with clearly distinguishable colors AND markers, plus a horizontal
one-panel-per-config figure so the layouts never overlap ambiguously.

Works off the search's summary.json alone -- no re-run, no waiting for the
top-K training to finish.

    python scripts/plot_multistart_configs.py \\
        --summary viz/diffplace/multistart/summary.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# high-contrast qualitative colors + distinct marker shapes
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]


def _colors(k):
    import matplotlib.pyplot as plt
    return [plt.cm.tab10(i % 10) for i in range(k)]


def _load_configs(summary_path):
    d = json.loads(Path(summary_path).read_text())
    top = d.get("top", [])
    configs = [dict(rank=t.get("rank", i + 1), val=t.get("val"),
                    pos=np.asarray(t["positions"], dtype=float).reshape(-1, 2))
               for i, t in enumerate(top)]
    return configs, d.get("n"), d.get("K")


def _quarter(ax, r_min, r_max):
    th = np.linspace(0, 90, 200)
    ax.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)), color="0.35", lw=2)
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (r_min, r_max):
        ax.plot(rb * np.cos(np.deg2rad(th)), rb * np.sin(np.deg2rad(th)),
                color="#2a9d8f", lw=1, alpha=0.6)
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def _xy(pos):
    return (pos[:, 0] * np.cos(np.deg2rad(pos[:, 1])),
            pos[:, 0] * np.sin(np.deg2rad(pos[:, 1])))


def plot_overlay(configs, r_min, r_max, out_path, n=None, K=None):
    """All top-K layouts on ONE quarter disk, each a distinct color+marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = _colors(len(configs))
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    _quarter(ax, r_min, r_max)
    for j, c in enumerate(configs):
        x, y = _xy(c["pos"])
        ax.scatter(x, y, s=150, marker=_MARKERS[j % len(_MARKERS)],
                   facecolor=cols[j], edgecolor="black", linewidth=0.8,
                   alpha=0.9, zorder=5,
                   label=f"#{c['rank']}  val={c['val']:.3e}"
                   if c["val"] is not None else f"#{c['rank']}")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"multi-start top-{len(configs)} layouts"
                 + (f"  (n={n}, K={K})" if n else ""))
    ax.legend(fontsize=9, loc="upper right")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_row(configs, r_min, r_max, out_path, n=None, K=None):
    """One quarter-disk panel per config, laid out in a horizontal row, so
    each layout is read cleanly on its own."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = len(configs)
    cols = _colors(m)
    fig, axes = plt.subplots(1, m, figsize=(3.4 * m, 3.7),
                             constrained_layout=True)
    if m == 1:
        axes = [axes]
    for j, (ax, c) in enumerate(zip(axes, configs)):
        _quarter(ax, r_min, r_max)
        x, y = _xy(c["pos"])
        ax.scatter(x, y, s=130, marker=_MARKERS[j % len(_MARKERS)],
                   facecolor=cols[j], edgecolor="black", linewidth=0.8,
                   zorder=5)
        for xi, yi in zip(x, y):
            ax.plot([0, xi], [0, yi], color="0.85", lw=0.8, zorder=1)
        ttl = f"#{c['rank']}"
        if c["val"] is not None:
            ttl += f"\nval={c['val']:.3e}"
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"multi-start top-{m} configurations"
                 + (f"  (n={n}, K={K})" if n else ""), fontweight="bold")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary",
                    default="viz/diffplace/multistart/summary.json",
                    help="the search's summary.json")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--out-overlay", default=None,
                    help="default: <summary dir>/top_overlay.png")
    ap.add_argument("--out-row", default=None,
                    help="default: <summary dir>/top_row.png")
    args = ap.parse_args()

    sp = Path(args.summary)
    if not sp.is_file():
        print(f"summary not found: {sp}", file=sys.stderr)
        return 2
    configs, n, K = _load_configs(sp)
    if not configs:
        print("no configs in summary", file=sys.stderr)
        return 1
    r_min = args.r_min
    r_max = args.r_max
    overlay = Path(args.out_overlay) if args.out_overlay \
        else sp.parent / "top_overlay.png"
    row = Path(args.out_row) if args.out_row else sp.parent / "top_row.png"
    plot_overlay(configs, r_min, r_max, overlay, n, K)
    plot_row(configs, r_min, r_max, row, n, K)
    print(f"wrote {overlay}\nwrote {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
