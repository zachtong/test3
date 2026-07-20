"""Compare several trained sensor-placement configs head to head, to test
whether the placement objective is FLAT (many different sensor positions ->
nearly equal reconstruction error = no unique optimum, placement is a
second-order effect) or has real structure (some placements clearly win).

This is the read-out for the multi-start placement study: initialize
differentiable placement from several very different layouts (ABCDEF, outer-
uniform, all-at-45-deg, ...), retrain a model at each initial AND each
optimized layout, then run this script over all of them. If the positions
scatter while the worst-case field error barely moves, the landscape is flat.

Reads outputs/<tag>/results.json for each tag: the median / p95 field error,
the per-sim field errors (for the worst-N mean = the decision metric), and the
sensor (r, theta) the run was trained at. Produces:
  - a console table (median, p95, worst-N mean field error) per config;
  - the cross-config SPREAD of each metric, absolute and relative, plus a
    flatness verdict -- and, if --k-gap is given (e.g. the K=8 -> K=12 worst-N
    drop you already measured), whether the placement spread is small compared
    to that first-order lever;
  - a 2-panel figure: LEFT the configs' sensor positions overlaid on the
    quarter disk (they scatter), RIGHT their worst-N field-error bars (nearly
    equal). That picture IS the flatness argument.

    python scripts/compare_placements.py \\
        --tags m_initUniform m_optUniform m_initDiag45 m_optDiag45 \\
               merged_sweep_k12_n6_ABCDEF merged_diffplace_k12_n6_verify \\
        --labels init-uniform opt-uniform init-45 opt-45 ABCDEF opt-ABCDEF \\
        --top-n 20 --k-gap 0.006 --out viz/placement_compare.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _colors(n):
    base = ["#3d5a80", "#e63946", "#2a9d8f", "#e9c46a", "#8338ec",
            "#fb5607", "#457b9d", "#606c38", "#bc6c25", "#264653"]
    return [base[i % len(base)] for i in range(n)]


def _load(tag: str, outputs: Path) -> dict | None:
    res = outputs / tag / "results.json"
    if not res.is_file():
        print(f"skip {tag}: no results.json at {res}", file=sys.stderr)
        return None
    try:
        d = json.loads(res.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"skip {tag}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    field = d.get("per_sim_field_errs", [])
    if not field:
        print(f"skip {tag}: no per_sim_field_errs", file=sys.stderr)
        return None
    cfg = d.get("config", {})
    sensors = cfg.get("sensors", {})
    pos = sensors.get("positions", [])
    gs = d.get("global_stats", {})
    return dict(
        tag=tag,
        field=np.asarray(field, dtype=float),
        median=gs.get("median"), p95=gs.get("p95"),
        positions=np.asarray(pos, dtype=float).reshape(-1, 2)
        if len(pos) else np.zeros((0, 2)),
        K=cfg.get("pod", {}).get("k"))


def _wn(field: np.ndarray, top_n: int) -> float:
    """Mean of the top-N worst (highest) per-sim field errors."""
    return float(np.mean(np.sort(field)[::-1][:top_n]))


def _table(records, top_n, k_gap):
    print(f"\n===== placement comparison (worst-{top_n} = mean of the "
          f"{top_n} highest-field-error test sims) =====\n")
    hdr = (f"  {'label':<16} {'tag':<32} {'median':>9} {'p95':>9} "
           f"{'wN_field':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in records:
        med = f"{r['median'] * 100:.3f}%" if r['median'] is not None else "--"
        p95 = f"{r['p95'] * 100:.3f}%" if r['p95'] is not None else "--"
        print(f"  {r['label']:<16} {r['tag']:<32} {med:>9} {p95:>9} "
              f"{r['wn'] * 100:8.3f}%")

    wn = np.array([r["wn"] for r in records])
    lo, hi = float(wn.min()), float(wn.max())
    spread = hi - lo
    rel = spread / max(float(wn.mean()), 1e-30)
    best = records[int(np.argmin(wn))]
    print(f"\n  worst-N field error: min {lo * 100:.3f}%  max {hi * 100:.3f}%  "
          f"spread {spread * 100:.3f}%  ({rel * 100:.1f}% of the mean)")
    print(f"  best config: {best['label']} ({best['tag']})")
    print(f"\n  Interpretation:")
    if k_gap is not None:
        ratio = spread / max(k_gap, 1e-30)
        print(f"    placement spread {spread * 100:.3f}%  vs  K-gap "
              f"{k_gap * 100:.3f}%  ->  {ratio:.2f}x")
        if ratio < 0.25:
            print("    Placement spread is SMALL vs the K lever: moving the "
                  "sensors all over the disk barely changes worst-case error "
                  "compared to changing K. The placement landscape is flat / "
                  "second-order -- there is no sharp global optimum, and any "
                  "well-spread layout in this family is essentially as good.")
        else:
            print("    Placement spread is COMPARABLE to the K lever: sensor "
                  "position genuinely matters here; prefer the best config.")
    else:
        if rel < 0.05:
            print("    The worst-case error is nearly identical across very "
                  "different layouts (spread < 5% of the mean). Strong sign of "
                  "a FLAT placement landscape: different inits settle at "
                  "different positions with the same quality -> no unique "
                  "optimum, placement is second-order. (Pass --k-gap to scale "
                  "this against the K lever.)")
        else:
            print("    The layouts differ in worst-case error by "
                  f"{rel * 100:.0f}% of the mean -- placement has real "
                  "structure here; the best config is not arbitrary.")


def _render(records, out_path, top_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = _colors(len(records))
    fig, (axp, axb) = plt.subplots(
        1, 2, figsize=(14, 6.2),
        gridspec_kw=dict(width_ratios=[1.0, 1.15]),
        constrained_layout=True)

    # --- LEFT: positions on the quarter disk ---
    th = np.linspace(0, 90, 200)
    axp.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)), color="0.35", lw=2)
    axp.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    axp.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for r, c in zip(records, cols):
        p = r["positions"]
        if not len(p):
            continue
        x = p[:, 0] * np.cos(np.deg2rad(p[:, 1]))
        y = p[:, 0] * np.sin(np.deg2rad(p[:, 1]))
        axp.scatter(x, y, s=90, color=c, edgecolor="black", linewidth=0.6,
                    alpha=0.85, zorder=4, label=r["label"])
    axp.set_xlim(-0.08, 1.15)
    axp.set_ylim(-0.08, 1.15)
    axp.set_aspect("equal")
    axp.set_xlabel("x / R")
    axp.set_ylabel("y / R")
    axp.set_title("sensor positions per config (do they scatter?)")
    axp.legend(loc="upper right", fontsize=8)
    axp.grid(alpha=0.25)

    # --- RIGHT: worst-N field error bars ---
    labels = [r["label"] for r in records]
    wn = np.array([r["wn"] for r in records]) * 100.0
    yy = np.arange(len(records))
    axb.barh(yy, wn, color=cols, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(wn):
        axb.text(v, yy[i], f" {v:.3f}%", va="center", fontsize=9)
    axb.set_yticks(yy)
    axb.set_yticklabels(labels, fontsize=9)
    axb.invert_yaxis()
    axb.set_xlabel(f"worst-{top_n} mean field error (%)")
    spread = wn.max() - wn.min()
    rel = spread / max(wn.mean(), 1e-30)
    axb.axvline(wn.min(), color="0.5", ls="--", lw=1)
    axb.axvline(wn.max(), color="0.5", ls="--", lw=1)
    axb.set_title(f"worst-case error (spread {spread:.3f}%, "
                  f"{rel * 100:.1f}% of mean = "
                  f"{'FLAT' if rel < 0.05 else 'structured'})")
    axb.grid(axis="x", alpha=0.3)

    fig.suptitle("Placement comparison: positions scatter (left) vs "
                 "reconstruction quality (right)", fontweight="bold")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", required=True,
                    help="trained tags to compare (outputs/<tag>/results.json)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="short display label per tag (same order); "
                    "defaults to the tags themselves")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--top-n", type=int, default=20,
                    help="how many worst sims define worst-N (default 20)")
    ap.add_argument("--k-gap", type=float, default=None,
                    help="reference first-order lever to scale the placement "
                    "spread against, e.g. the K=8->K=12 worst-N field-error "
                    "drop as a fraction (0.006 = 0.6%%). If placement spread "
                    "<< k-gap, placement is second-order / the landscape is "
                    "flat.")
    ap.add_argument("--out", default=None, help="comparison figure path")
    ap.add_argument("--out-json", default=None,
                    help="also write the numeric summary to JSON")
    args = ap.parse_args()

    labels = args.labels or args.tags
    if len(labels) != len(args.tags):
        print("--labels must match --tags in count", file=sys.stderr)
        return 2

    outputs = Path(args.outputs)
    records = []
    for tag, lab in zip(args.tags, labels):
        r = _load(tag, outputs)
        if r is None:
            continue
        r["label"] = lab
        r["wn"] = _wn(r["field"], args.top_n)
        records.append(r)
    if len(records) < 2:
        print("need >= 2 loadable tags to compare", file=sys.stderr)
        return 1

    _table(records, args.top_n, args.k_gap)
    if args.out:
        _render(records, Path(args.out), args.top_n)
        print(f"\nwrote {args.out}")
    if args.out_json:
        wn = np.array([r["wn"] for r in records])
        summary = dict(
            top_n=args.top_n,
            configs=[dict(label=r["label"], tag=r["tag"],
                          median=r["median"], p95=r["p95"], wn_field=r["wn"],
                          positions=r["positions"].tolist())
                     for r in records],
            wn_spread=float(wn.max() - wn.min()),
            wn_rel_spread=float((wn.max() - wn.min()) / max(wn.mean(), 1e-30)),
            best=records[int(np.argmin(wn))]["tag"])
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
