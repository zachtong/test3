"""Compare two sensor sweeps run at different K on the SAME dataset.

Given two tag prefixes (e.g. a K=8 sweep and a K=12 sweep), match
configs by their sensor code (the ABCDEF subset), keep only the
configs present in BOTH sweeps, and report:

  - per-config median field err at each K, delta, and pct change
  - whether the top-1 / top-N ranking changed
  - Spearman rank correlation of the two rankings (is the sensor
    ordering K-invariant?)
  - the best config under each K

This answers the question the K-sweep raised: at K=8 the good
configs saturated the POD floor so their ranking was compressed
and noisy; at K=12 sensor placement matters more. Does the sensor
ranking survive the K change (robust conclusion) or shift (higher
K rewards different placements)?

Match key is the sensor code, so the two sweeps must share the
same 6-position catalogue / naming. Configs unique to one sweep
(e.g. n=2 tier only run at K=8) are listed but excluded from the
paired comparison.

    python scripts/compare_k_sweeps.py \\
        --prefix-a smalltest_sweep \\
        --prefix-b smalltest_sweep_k12 \\
        --label-a K8 --label-b K12 \\
        --out-dir viz/k8_vs_k12
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


def _code_from_tag(tag: str, prefix: str) -> str | None:
    """Extract the sensor code from <prefix>_n{N}_{code}.

    The remainder after the prefix must match _n{N}_{code} EXACTLY.
    This is critical when one prefix is a string-prefix of another
    (e.g. 'smalltest_sweep' vs 'smalltest_sweep_k12'): the glob
    'smalltest_sweep*' would otherwise pull in the K=12 configs,
    whose remainder '_k12_n3_ABC' does not start with '_n' and is
    correctly rejected by the anchored fullmatch."""
    if not tag.startswith(prefix):
        return None
    rest = tag[len(prefix):]
    m = re.fullmatch(r"_n\d+_([A-Z]+)", rest)
    return m.group(1) if m else None


def _load_sweep(outputs: Path, prefix: str) -> dict[str, dict]:
    """Return {code: {tag, n, median, p95, gap_to_floor}} for every
    config whose results.json is readable."""
    out: dict[str, dict] = {}
    for tag_dir in sorted(outputs.glob(f"{prefix}*")):
        res = tag_dir / "results.json"
        if not res.is_file():
            continue
        code = _code_from_tag(tag_dir.name, prefix)
        if code is None:
            continue
        try:
            d = json.loads(res.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {tag_dir.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        gs = d.get("global_stats", {}) or {}
        sens = d.get("config", {}).get("sensors", {}) or {}
        out[code] = dict(
            tag=tag_dir.name,
            n=int(sens.get("n") or len(sens.get("positions") or [])),
            median=gs.get("median"),
            p95=gs.get("p95"),
            gap_to_floor=d.get("gap_to_floor"))
    return out


def _spearman(rank_a: list[float], rank_b: list[float]) -> float:
    """Spearman rho via Pearson on ranks. No scipy dependency."""
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if a.size < 2:
        return float("nan")
    a = (a - a.mean())
    b = (b - b.mean())
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def _ranks(values: list[float]) -> list[float]:
    """Ascending competition ranks (1 = smallest). Ties get average
    rank. Lower field err = better = rank 1."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ties
    vals = np.asarray(values, dtype=float)
    for v in np.unique(vals):
        mask = vals == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix-a", required=True,
                    help="tag prefix of sweep A (e.g. the K=8 sweep)")
    ap.add_argument("--prefix-b", required=True,
                    help="tag prefix of sweep B (e.g. the K=12 sweep)")
    ap.add_argument("--label-a", default="A",
                    help="short label for sweep A (default: A)")
    ap.add_argument("--label-b", default="B",
                    help="short label for sweep B (default: B)")
    ap.add_argument("--outputs", default="outputs",
                    help="dir containing <tag>/results.json "
                    "(default: outputs)")
    ap.add_argument("--out-dir", default="viz/k_compare",
                    help="where to write the CSV (default: "
                    "viz/k_compare)")
    ap.add_argument("--top-n", type=int, default=10,
                    help="how many top configs to show in the "
                    "ranking table (default: 10)")
    ap.add_argument("--metric", default="median",
                    choices=["median", "p95"],
                    help="which field-error metric to rank on "
                    "(default: median)")
    ap.add_argument("--out-plot", default=None,
                    help="optional PNG: paired scatter (A vs B with "
                    "y=x diagonal) + slope chart. Points below the "
                    "diagonal = B improved.")
    ap.add_argument("--value-scale", type=float, default=100.0,
                    help="multiply the metric for display "
                    "(default 100 = percent relative L2)")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    a = _load_sweep(outputs, args.prefix_a)
    b = _load_sweep(outputs, args.prefix_b)
    if not a:
        print(f"no configs found for prefix-a '{args.prefix_a}'",
              file=sys.stderr)
        return 1
    if not b:
        print(f"no configs found for prefix-b '{args.prefix_b}'",
              file=sys.stderr)
        return 1

    la, lb = args.label_a, args.label_b
    codes_a = set(a.keys())
    codes_b = set(b.keys())
    shared = sorted(codes_a & codes_b)
    only_a = sorted(codes_a - codes_b)
    only_b = sorted(codes_b - codes_a)

    print(f"Sweep {la} ({args.prefix_a}): {len(a)} configs")
    print(f"Sweep {lb} ({args.prefix_b}): {len(b)} configs")
    print(f"Shared configs (in both): {len(shared)}")
    if only_a:
        print(f"  only in {la}: {len(only_a)} "
              f"({', '.join(only_a[:8])}"
              f"{' ...' if len(only_a) > 8 else ''})")
    if only_b:
        print(f"  only in {lb}: {len(only_b)} "
              f"({', '.join(only_b[:8])}"
              f"{' ...' if len(only_b) > 8 else ''})")

    if not shared:
        print("no shared configs to compare", file=sys.stderr)
        return 1

    m = args.metric
    rows = []
    for code in shared:
        va = a[code].get(m)
        vb = b[code].get(m)
        if va is None or vb is None:
            continue
        va, vb = float(va), float(vb)
        rows.append(dict(
            code=code, n=a[code]["n"],
            a_val=va, b_val=vb,
            delta=vb - va,
            pct=(vb - va) / va * 100 if va else float("nan"),
            a_gap=a[code].get("gap_to_floor"),
            b_gap=b[code].get("gap_to_floor")))

    # Rankings (1 = best = lowest metric) within the shared set.
    a_vals = [r["a_val"] for r in rows]
    b_vals = [r["b_val"] for r in rows]
    a_ranks = _ranks(a_vals)
    b_ranks = _ranks(b_vals)
    for r, ra, rb in zip(rows, a_ranks, b_ranks):
        r["a_rank"] = ra
        r["b_rank"] = rb

    rho = _spearman(a_ranks, b_ranks)

    # Sort display by sweep B metric (the newer K), best first.
    rows_by_b = sorted(rows, key=lambda r: r["b_val"])
    rows_by_a = sorted(rows, key=lambda r: r["a_val"])

    print(f"\n===== Ranking comparison (metric: {m}) =====")
    print(f"Spearman rank correlation ({la} vs {lb}): "
          f"{rho:.3f}")
    if not np.isnan(rho):
        if rho >= 0.9:
            print("  -> ranking is essentially K-INVARIANT; the "
                  "sensor conclusions hold across K.")
        elif rho >= 0.6:
            print("  -> ranking is broadly stable but with some "
                  "reshuffling at higher K.")
        else:
            print("  -> ranking CHANGED substantially with K; "
                  "higher K rewards different sensor placements.")

    best_a = rows_by_a[0]
    best_b = rows_by_b[0]
    print(f"\nBest under {la}: {best_a['code']} "
          f"(n={best_a['n']}, {m}={best_a['a_val']:.4f})")
    print(f"Best under {lb}: {best_b['code']} "
          f"(n={best_b['n']}, {m}={best_b['b_val']:.4f})")
    if best_a["code"] == best_b["code"]:
        print("  -> SAME best config under both K.")
    else:
        # Where does A's winner land under B, and vice versa?
        a_win_b_rank = next(r["b_rank"] for r in rows
                            if r["code"] == best_a["code"])
        b_win_a_rank = next(r["a_rank"] for r in rows
                            if r["code"] == best_b["code"])
        print(f"  -> winner changed. {la}'s winner "
              f"{best_a['code']} ranks #{a_win_b_rank:.0f} under "
              f"{lb}; {lb}'s winner {best_b['code']} ranks "
              f"#{b_win_a_rank:.0f} under {la}.")

    print(f"\n===== Top {args.top_n} by {lb} {m} "
          f"(paired with {la}) =====")
    hdr = (f"  {'rank_' + lb:>8}  {'code':<8}  {'n':>2}  "
           f"{la + '_' + m:>10}  {lb + '_' + m:>10}  "
           f"{'delta':>9}  {'pct':>7}  {'rank_' + la:>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, r in enumerate(rows_by_b[:args.top_n]):
        print(f"  {i + 1:>8d}  {r['code']:<8}  {r['n']:>2d}  "
              f"{r['a_val']:>10.4f}  {r['b_val']:>10.4f}  "
              f"{r['delta']:>+9.4f}  {r['pct']:>+6.1f}%  "
              f"{r['a_rank']:>8.0f}")

    # Aggregate improvement
    deltas = np.array([r["delta"] for r in rows])
    pcts = np.array([r["pct"] for r in rows if np.isfinite(r["pct"])])
    print(f"\n===== Aggregate over {len(rows)} shared configs =====")
    print(f"  {m} improved ({lb} < {la}) in "
          f"{int((deltas < 0).sum())}/{len(rows)} configs")
    print(f"  mean delta: {deltas.mean():+.4f}  "
          f"(mean pct: {pcts.mean():+.1f}%)")
    print(f"  median delta: {np.median(deltas):+.4f}")

    # Write CSV
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"compare_{la}_vs_{lb}_{m}.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["code", "n",
                    f"{la}_{m}", f"{lb}_{m}", "delta", "pct",
                    f"{la}_rank", f"{lb}_rank",
                    f"{la}_gap", f"{lb}_gap"])
        for r in rows_by_b:
            w.writerow([r["code"], r["n"],
                        f"{r['a_val']:.6f}", f"{r['b_val']:.6f}",
                        f"{r['delta']:.6f}",
                        f"{r['pct']:.2f}",
                        f"{r['a_rank']:.0f}", f"{r['b_rank']:.0f}",
                        ("" if r["a_gap"] is None
                         else f"{r['a_gap']:.4f}"),
                        ("" if r["b_gap"] is None
                         else f"{r['b_gap']:.4f}")])
    print(f"\nwrote {csv_path}")

    if args.out_plot:
        _render_pair_plot(rows, la, lb, m, rho, args.value_scale,
                          Path(args.out_plot))
        print(f"wrote {args.out_plot}")
    return 0


def _render_pair_plot(rows, la, lb, m, rho, scale, out_path):
    """Left: scatter of A vs B per config with the y=x diagonal
    (points below the line = B improved). Right: slope chart, one
    line per config from A to B (every config's change at a glance).
    Colored by sensor count n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = "%" if abs(scale - 100.0) < 1e-9 else f"x{scale:g}"
    av = np.array([r["a_val"] for r in rows]) * scale
    bv = np.array([r["b_val"] for r in rows]) * scale
    ns = np.array([r["n"] for r in rows])
    n_uniq = sorted(set(ns.tolist()))
    cmap = plt.get_cmap("viridis")
    color_of = {n: cmap(i / max(len(n_uniq) - 1, 1))
                for i, n in enumerate(n_uniq)}
    colors = [color_of[n] for n in ns]

    fig, (ax_s, ax_sl) = plt.subplots(
        1, 2, figsize=(12.5, 5.6), constrained_layout=True)

    # --- scatter with y=x ---
    lo = 0.0
    hi = float(max(av.max(), bv.max())) * 1.08
    ax_s.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1,
              label="y = x (no change)")
    ax_s.scatter(av, bv, c=colors, s=60, edgecolor="black",
                 linewidth=0.5, zorder=4)
    ax_s.fill_between([lo, hi], [lo, hi], hi, color="#e63946",
                     alpha=0.06)
    ax_s.fill_between([lo, hi], lo, [lo, hi], color="#2a9d8f",
                     alpha=0.06)
    ax_s.text(hi * 0.97, hi * 0.5, f"{lb} better\n(below line)",
              ha="right", va="center", fontsize=9, color="#2a9d8f")
    ax_s.text(hi * 0.35, hi * 0.97, f"{la} better\n(above line)",
              ha="center", va="top", fontsize=9, color="#e63946")
    ax_s.set_xlabel(f"{la} {m} ({unit})")
    ax_s.set_ylabel(f"{lb} {m} ({unit})")
    ax_s.set_xlim(lo, hi)
    ax_s.set_ylim(lo, hi)
    ax_s.set_aspect("equal")
    ax_s.set_title(f"Paired per-config: {la} vs {lb}  "
                   f"(Spearman rho={rho:.3f})")
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="lower right", fontsize=8)

    # --- slope chart ---
    for a_i, b_i, n_i in zip(av, bv, ns):
        ax_sl.plot([0, 1], [a_i, b_i], "-", color=color_of[n_i],
                   alpha=0.55, lw=1.0)
        ax_sl.scatter([0, 1], [a_i, b_i], color=color_of[n_i],
                      s=22, zorder=4)
    ax_sl.set_xticks([0, 1])
    ax_sl.set_xticklabels([la, lb])
    ax_sl.set_ylabel(f"{m} ({unit})")
    ax_sl.set_xlim(-0.2, 1.2)
    ax_sl.set_ylim(bottom=0)
    ax_sl.set_title("Per-config change")
    ax_sl.grid(axis="y", alpha=0.3)
    # legend for n colors
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=color_of[n], markersize=8,
                      label=f"n={n}") for n in n_uniq]
    ax_sl.legend(handles=handles, loc="upper right", fontsize=8,
                 title="sensors")

    fig.suptitle(f"{la} vs {lb}: reconstruction error per config",
                 fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
