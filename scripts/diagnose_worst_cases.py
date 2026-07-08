"""Diagnose whether the worst test sims are model-failures (fixable
by better sensors / bigger model) or physical-floor cases (limited
by K=8 POD truncation, unfixable without more modes).

For each test sim of a given config, compute:
    per_sim_gap = field_err / floor_err
where field_err is model's relative L2 and floor_err is the K=8
truncation floor (both from results.json). The ratio says how
close to physical floor the model got on that specific sim.

Ranks the top-N worst sims by field_err, prints their per-sim
gap. If most worst sims have gap ~ 1, model is at floor everywhere
-- adding sensors won't help, need larger K. If gap >> 1 on some
worst sims, model has real room to improve on those cases.

Also groups worst sims by which POD modes carry their difficulty
(highest per-mode error), so you can see whether hard cases share
a common physical pattern.

    python scripts/diagnose_worst_cases.py \\
        --tag <shortname>_sweep_n5_ABDEF \\
        --top-n 20

Compare across multiple configs at once to see whether "hard sims"
are the SAME sims regardless of sensor set (intrinsic difficulty)
or CHANGE with sensor set (sensor undercoverage):

    python scripts/diagnose_worst_cases.py \\
        --tags <shortname>_sweep_n5_ABDEF <shortname>_sweep_n6_ABCDEF \\
        --top-n 10
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load(tag: str, outputs: Path) -> dict | None:
    res = outputs / tag / "results.json"
    if not res.is_file():
        print(f"skip {tag}: no results.json at {res}",
              file=sys.stderr)
        return None
    try:
        d = json.loads(res.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"skip {tag}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
    field = d.get("per_sim_field_errs", [])
    floor = d.get("per_sim_floor_errs", [])
    names = d.get("per_sim_basenames", [])
    per_mode = d.get("per_sim_per_mode_errs", {})
    if not field or not floor or not names:
        print(f"skip {tag}: missing per-sim arrays",
              file=sys.stderr)
        return None
    return dict(tag=tag, field=np.asarray(field, dtype=float),
                 floor=np.asarray(floor, dtype=float),
                 names=names, per_mode=per_mode,
                 median=d.get("global_stats", {}).get("median"),
                 p95=d.get("global_stats", {}).get("p95"),
                 gap_to_floor=d.get("gap_to_floor"))


def _analyze_one(data: dict, top_n: int) -> None:
    tag = data["tag"]
    field = data["field"]
    floor = data["floor"]
    gap = field / np.maximum(floor, 1e-24)
    n = len(field)

    print(f"\n===== {tag} =====")
    print(f"  n_test={n}  median={data['median']:.4f}  "
          f"p95={data['p95']:.4f}  "
          f"gap_to_floor(median)={data['gap_to_floor']:.3f}")

    order = np.argsort(-field)                    # worst first
    hard_gap = gap[order][:top_n]
    hard_field = field[order][:top_n]
    hard_floor = floor[order][:top_n]
    hard_names = [data["names"][i] for i in order[:top_n]]

    print(f"\n  Worst {top_n} by field_err:")
    print(f"  {'rank':>4}  {'field':>8}  {'floor':>8}  "
          f"{'gap':>6}  basename")
    for r, (fe, fl, g, nm) in enumerate(zip(
            hard_field, hard_floor, hard_gap, hard_names)):
        print(f"  {r + 1:>4d}  {fe:8.4f}  {fl:8.4f}  "
              f"{g:6.2f}  {nm}")

    print(f"\n  Worst-{top_n} per-sim gap statistics:")
    print(f"    min={hard_gap.min():.2f}  "
          f"med={np.median(hard_gap):.2f}  "
          f"max={hard_gap.max():.2f}")

    at_floor = int((hard_gap < 1.3).sum())
    far_from_floor = int((hard_gap > 2.0).sum())
    print(f"    at-floor (gap<1.3): {at_floor}/{top_n}")
    print(f"    far-from-floor (gap>2.0): "
          f"{far_from_floor}/{top_n}")

    print(f"\n  Interpretation for {tag}:")
    if at_floor >= top_n * 0.7:
        print(f"    Most worst-cases are AT their per-sim floor. "
              f"K=8 POD basis cannot represent the physics of "
              f"these sims. Adding more sensors will NOT help. "
              f"Only path forward is to increase K (or accept "
              f"the tail as physical-floor territory).")
    elif far_from_floor >= top_n * 0.5:
        print(f"    Model is far from floor on {far_from_floor} "
              f"worst-cases -- real room to improve via sensor "
              f"placement or model capacity. Worth exploring.")
    else:
        print(f"    Mixed: some at floor, some not. Sensor "
              f"placement might help the far-from-floor subset "
              f"but tail-of-tail is likely physical.")


def _tag_sort_key(tag: str):
    """Sort tags by trailing _k<int> if present (for K sweeps), so a
    comparison table reads low-K to high-K regardless of CLI order.
    Falls back to the tag string."""
    parts = tag.rsplit("_k", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (0, int(parts[1]))
    return (1, tag)


def _compare_metrics_table(datasets: list[dict], top_n: int) -> None:
    """The decision table: absolute field_err vs floor across configs.

    gap is a RATIO (field/floor) and can look better just because
    the floor dropped. What actually matters physically is the
    absolute field_err. This table shows both so you can tell a
    real improvement (field_err down) from a floor-relabeling
    artifact (field_err flat, gap up because floor fell)."""
    print(f"\n\n===== Absolute metrics across configs "
          f"(the real decision table) =====")
    print(f"  worst-{top_n} = mean over the {top_n} highest-"
          f"field_err sims of each config\n")
    ordered = sorted(datasets, key=lambda d: _tag_sort_key(d["tag"]))
    header = (f"  {'tag':<28}  {'med_field':>9}  {'p95_field':>9}  "
              f"{'wN_field':>9}  {'wN_floor':>9}  {'wN_gap':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    best_wn = min(
        float(np.mean(np.sort(d["field"])[::-1][:top_n]))
        for d in ordered)
    for d in ordered:
        field = d["field"]
        floor = d["floor"]
        order = np.argsort(-field)[:top_n]
        wn_field = float(np.mean(field[order]))
        wn_floor = float(np.mean(floor[order]))
        wn_gap = wn_field / max(wn_floor, 1e-24)
        marker = "  <== best worst-case" if (
            abs(wn_field - best_wn) < 1e-12) else ""
        print(f"  {d['tag']:<28}  {d['median']:9.4f}  "
              f"{d['p95']:9.4f}  {wn_field:9.4f}  "
              f"{wn_floor:9.4f}  {wn_gap:7.2f}{marker}")
    print(f"\n  Read this way:")
    print(f"    - wN_field DROPS as K rises  -> higher K genuinely "
          f"improves worst-case reconstruction; keep going.")
    print(f"    - wN_field FLAT while wN_gap RISES -> the extra "
          f"modes are unobservable by the sensors; error is just "
          f"relabeled floor->model. Stop at the K where wN_field "
          f"bottoms out.")
    print(f"    - wN_field RISES at high K -> overshoot; the model "
          f"is actively hurt by modes it cannot predict.")


def _compare_hard_sims(datasets: list[dict], top_n: int) -> None:
    print(f"\n\n===== Cross-config: are the SAME sims hard? =====")
    per_tag_hard = {}
    for d in datasets:
        order = np.argsort(-d["field"])
        per_tag_hard[d["tag"]] = set(d["names"][i]
                                     for i in order[:top_n])
    # Union + overlap counts
    all_names = set()
    for s in per_tag_hard.values():
        all_names |= s
    intersect = set(next(iter(per_tag_hard.values())))
    for s in per_tag_hard.values():
        intersect &= s

    print(f"  union of worst-{top_n} across "
          f"{len(datasets)} configs: {len(all_names)} unique sims")
    print(f"  intersection (hard in EVERY config): "
          f"{len(intersect)} sim(s)")
    if intersect:
        print(f"    -> {sorted(intersect)[:10]}"
              + (" ..." if len(intersect) > 10 else ""))
    overlap_frac = (len(intersect) / top_n if top_n > 0 else 0)
    if overlap_frac >= 0.7:
        print(f"\n  Interpretation: {overlap_frac:.0%} of worst "
              f"sims are shared across configs. Difficulty is "
              f"INTRINSIC to those sims, not sensor-driven. "
              f"Sensor sweep will not help them.")
    elif overlap_frac >= 0.3:
        print(f"\n  Interpretation: partial overlap ({overlap_frac:.0%}). "
              f"Some sims are intrinsically hard; others are "
              f"sensor-dependent. Sensor sweep could help a subset.")
    else:
        print(f"\n  Interpretation: little overlap ({overlap_frac:.0%}). "
              f"Different sensor sets fail on different sims -- "
              f"sensor placement genuinely matters for the tail.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", help="single tag to analyze")
    ap.add_argument("--tags", nargs="*",
                    help="multiple tags to cross-compare")
    ap.add_argument("--outputs", default="outputs",
                    help="dir containing <tag>/results.json "
                    "(default: outputs)")
    ap.add_argument("--top-n", type=int, default=20,
                    help="how many worst sims to inspect "
                    "(default: 20)")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    tags = []
    if args.tag:
        tags.append(args.tag)
    if args.tags:
        tags.extend(args.tags)
    if not tags:
        print("provide --tag or --tags", file=sys.stderr)
        return 2

    datasets = []
    for t in tags:
        d = _load(t, outputs)
        if d is not None:
            datasets.append(d)
    if not datasets:
        return 1

    for d in datasets:
        _analyze_one(d, args.top_n)

    if len(datasets) >= 2:
        _compare_metrics_table(datasets, args.top_n)
        _compare_hard_sims(datasets, args.top_n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
