"""Compact terminal dump of inspect_gt_quality's summary.json.

Prints the aggregate + the top-N most-suspicious sims (default 1)
in a format that fits on a phone screen and is easy to transcribe
verbatim. Ranks sims by descending temporal max_rise (the most
important diagnostic).

    python scripts/dump_gt_summary.py \\
        viz/gt_quality_firehorse/summary.json

    # more sims
    python scripts/dump_gt_summary.py <path> --n 3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary_json",
                    help="path to summary.json from inspect_gt_quality")
    ap.add_argument("--n", type=int, default=1,
                    help="number of sims to detail (default 1). "
                    "Sorted by descending temporal max_rise -- worst "
                    "offender first.")
    args = ap.parse_args()

    p = Path(args.summary_json)
    r = json.loads(p.read_text())

    a = r["aggregate"]
    print(f"AGG rise={a['max_temporal_rise_across_all']:.2e} "
          f"kink45={a['max_kink_45_across_all']:.3f} "
          f"back={a['sims_with_any_raw_backward']}/{r['n_checked']} "
          f"PWF={a['n_pass']}/{a['n_warn']}/{a['n_fail']}")

    per = sorted(r["per_sim"],
                  key=lambda s: -s["temporal"]["max_rise"])[:args.n]
    for i, s in enumerate(per):
        print(f"\n[{i}] {s['basename']} {s['verdict']}")
        rt = s["raw_treal"]
        print(f"  bak n={rt['n_backward']} "
              f"max_s={rt['max_backward']:.2e} "
              f"/dt={rt.get('max_backward_over_median_dt', 0):.2f} "
              f"med_dt={rt.get('median_dt', 0):.2e}")
        tp = s["temporal"]
        print(f"  rise max={tp['max_rise']:.2e} "
              f"cells={tp['n_cells_with_rise']} "
              f"frac={tp['frac_cells_with_rise']:.4f}")
        # One-liner across all 3 angles
        rk = s["radial_kink"]
        parts = []
        for th_name in ("theta=0", "theta=45", "theta=90"):
            v = rk[th_name]
            parts.append(f"{th_name.split('=')[1]:>2}:rk={v['rel_kink']:.3f}"
                          f"{'T' if v['edge_less_descended'] else 'F'}")
        print(f"  " + " | ".join(parts))
        # θ=45 detailed u_z values
        v45 = rk["theta=45"]["values_at_r"]
        print(f"  th45 u@[.85,.9,.95,.99]="
              f"[{v45['0.85']:.2e},{v45['0.9']:.2e},"
              f"{v45['0.95']:.2e},{v45['0.99']:.2e}]")
        print(f"  dead={s['dead_cells']['frac_dead']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
