"""The headline figure: K dominates, sensor placement is secondary.

Loads two (or more) sensor sweeps run at different K on the same
dataset and shows, per K, the distribution of median field error
across ALL sensor configs. The visual argument:

  - each K's box is TIGHT (configs within a K cluster together ->
    sensor placement barely matters once coverage constraints met)
  - the boxes are FAR APART (K=8 vs K=12 gap dwarfs the within-K
    spread -> K is the first-order lever)

Left panel: box + strip of per-config median field err at each K.
Right panel: same data as within-K spread vs cross-K gap, annotated
with the ratio so the "K dominates" claim is quantified on-figure.

Each sweep is identified by a tag prefix; a config's median field
err comes from outputs/<tag>/results.json. Configs are matched
across K by sensor code so the strip lines can connect the same
config across K (shows every config improves, none regress).

    python scripts/plot_k_vs_sensor.py \\
        --sweeps smalltest_sweep:8 smalltest_sweep_k12:12 \\
        --out viz/k_vs_sensor.png
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


def _code_from_tag(tag: str, prefix: str) -> str | None:
    if not tag.startswith(prefix):
        return None
    m = re.fullmatch(r"_n\d+_([A-Z]+)", tag[len(prefix):])
    return m.group(1) if m else None


def _load_sweep(outputs: Path, prefix: str,
                metric: str) -> dict[str, float]:
    """Return {code: metric_value} for every readable config."""
    out: dict[str, float] = {}
    for tag_dir in sorted(outputs.glob(f"{prefix}*")):
        res = tag_dir / "results.json"
        if not res.is_file():
            continue
        code = _code_from_tag(tag_dir.name, prefix)
        if code is None:
            continue
        try:
            d = json.loads(res.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        v = (d.get("global_stats", {}) or {}).get(metric)
        if v is not None:
            out[code] = float(v)
    return out


def _parse_sweeps(specs: list[str]) -> list[tuple[str, str, int]]:
    """Parse 'prefix:label' specs into (prefix, label, sort_key).

    label may be any string (e.g. '8', '12', 'new', 'merged'). If
    it is a pure integer it is shown as 'K=<label>' and sorts
    numerically; otherwise it is shown verbatim and sorts by input
    order. Mixed numeric/string labels keep input order."""
    parsed = []
    all_numeric = True
    for order, s in enumerate(specs):
        if ":" not in s:
            raise SystemExit(
                f"bad --sweeps entry '{s}', expected prefix:label")
        prefix, label = s.rsplit(":", 1)
        is_num = label.lstrip("-").isdigit()
        all_numeric = all_numeric and is_num
        parsed.append([prefix, label, order, is_num])
    if all_numeric:
        parsed.sort(key=lambda t: int(t[1]))
    out = []
    for prefix, label, order, is_num in parsed:
        disp = f"K={label}" if is_num else label
        out.append((prefix, disp, order))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweeps", nargs="+", required=True,
                    help="one or more prefix:label specs. label may "
                    "be a K value (shown 'K=8') or any string "
                    "(shown verbatim), e.g. smalltest_sweep:8 "
                    "smalltest_sweep_k12:12  OR  "
                    "new_sweep_k12:new merged_sweep_k12:merged")
    ap.add_argument("--outputs", default="outputs",
                    help="dir with <tag>/results.json (default: "
                    "outputs)")
    ap.add_argument("--metric", default="median",
                    choices=["median", "p95"],
                    help="which field-error metric (default: "
                    "median)")
    ap.add_argument("--out", default="viz/k_vs_sensor.png",
                    help="output PNG path")
    ap.add_argument("--value-scale", type=float, default=100.0,
                    help="multiply field err for display; default "
                    "100 shows relative L2 as a percentage")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    sweeps = _parse_sweeps(args.sweeps)
    data: list[dict] = []
    for prefix, label, _order in sweeps:
        codes = _load_sweep(outputs, prefix, args.metric)
        if not codes:
            print(f"WARN: no configs for prefix '{prefix}'",
                  file=sys.stderr)
        data.append(dict(prefix=prefix, label=label, codes=codes))
    data = [d for d in data if d["codes"]]
    if len(data) < 1:
        print("no sweeps with data", file=sys.stderr)
        return 1

    scale = args.value_scale
    labels = [d["label"] for d in data]
    unit = "%" if abs(scale - 100.0) < 1e-9 else f"x{scale:g}"

    fig, (ax_box, ax_txt) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw=dict(width_ratios=[2.0, 1.0]),
        constrained_layout=True)

    # ---- Left: box + strip, one column per K ----
    box_data = [np.array(sorted(d["codes"].values())) * scale
                for d in data]
    positions = list(range(len(data)))
    bp = ax_box.boxplot(box_data, positions=positions, widths=0.5,
                         patch_artist=True,
                         medianprops=dict(color="#1a2b4a",
                                          linewidth=1.6),
                         boxprops=dict(facecolor="#dbe6f2",
                                       edgecolor="#3d5a80"),
                         whiskerprops=dict(color="#3d5a80"),
                         capprops=dict(color="#3d5a80"),
                         showfliers=False)
    rng = np.random.default_rng(7)
    for i, d in enumerate(data):
        vals = np.array(list(d["codes"].values())) * scale
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax_box.scatter(np.full(len(vals), i) + jitter, vals,
                       s=26, alpha=0.55, color="#3d5a80",
                       edgecolors="none", zorder=3)

    # Connect the SAME config (by code) across adjacent K columns
    # so every faint line shows one config's K-improvement.
    for i in range(len(data) - 1):
        left, right = data[i], data[i + 1]
        shared = set(left["codes"]) & set(right["codes"])
        for code in shared:
            ax_box.plot(
                [i, i + 1],
                [left["codes"][code] * scale,
                 right["codes"][code] * scale],
                color="0.6", lw=0.4, alpha=0.35, zorder=1)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(labels, fontsize=12)
    ax_box.set_ylabel(f"median field error ({unit})", fontsize=11)
    ax_box.set_title("Field error per sensor config, grouped by "
                     "condition", fontsize=12)
    ax_box.grid(axis="y", alpha=0.3)
    ax_box.set_ylim(bottom=0)

    # ---- Right: quantify within-group spread vs cross-group gap ----
    ax_txt.axis("off")
    lines = ["Within-group spread vs cross-group gap\n"]
    medians_per_k = []
    for d in data:
        vals = np.array(list(d["codes"].values())) * scale
        spread = float(vals.max() - vals.min())
        med = float(np.median(vals))
        medians_per_k.append(med)
        lines.append(
            f"{d['label']}:  median={med:.3f}{unit}   "
            f"config-spread={spread:.3f}{unit}   "
            f"(n={len(vals)} configs)")
    lines.append("")
    # cross-group gaps between consecutive conditions
    for i in range(len(data) - 1):
        gap = abs(medians_per_k[i + 1] - medians_per_k[i])
        pct = (gap / medians_per_k[i] * 100
               if medians_per_k[i] else float("nan"))
        sa = (np.array(list(data[i]["codes"].values())) * scale)
        sb = (np.array(list(data[i + 1]["codes"].values())) * scale)
        avg_spread = float(((sa.max() - sa.min())
                            + (sb.max() - sb.min())) / 2)
        ratio = gap / avg_spread if avg_spread > 0 else float("inf")
        lines.append(
            f"{data[i]['label']} vs {data[i + 1]['label']}:")
        lines.append(f"   cross-group gap = {gap:.3f}{unit} "
                     f"({pct:+.0f}%)")
        lines.append(f"   avg within-group spread = "
                     f"{avg_spread:.3f}{unit}")
        lines.append(f"   gap / spread = {ratio:.1f}x")
        lines.append("")
    if len(data) >= 2:
        lines.append("Interpretation:")
        lines.append("K moves the error far more than any")
        lines.append("sensor choice within a fixed K. Sensor")
        lines.append("placement is a second-order effect once")
        lines.append("angle + radius coverage is satisfied.")

    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=10,
                transform=ax_txt.transAxes)

    fig.suptitle("K dominates; sensor placement is secondary",
                 fontsize=14, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    for d in data:
        vals = np.array(list(d["codes"].values())) * scale
        print(f"  {d['label']}: {len(vals)} configs, "
              f"median={np.median(vals):.4f}{unit}, "
              f"spread={vals.max() - vals.min():.4f}{unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
