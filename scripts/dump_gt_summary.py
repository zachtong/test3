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
                    help="number of sims to detail (default 1).")
    ap.add_argument("--sort", default="rise",
                    choices=("rise", "rise_rel", "kink45", "backward"),
                    help="which metric to sort worst-first by "
                    "(default 'rise'). 'rise_rel' is rise as a "
                    "fraction of the sim's peak descent (more "
                    "meaningful than raw rise for cross-sim "
                    "comparison). 'kink45' finds the sim whose 45-deg "
                    "final-frame edge kink is largest.")
    args = ap.parse_args()

    p = Path(args.summary_json)
    r = json.loads(p.read_text())

    a = r["aggregate"]
    rise_rel = a.get("max_rise_rel_across_all", 0.0)
    print(f"AGG rise={a['max_temporal_rise_across_all']:.2e}m "
          f"({100 * rise_rel:.1f}%) "
          f"kink45={a['max_kink_45_across_all']:.3f} "
          f"back={a['sims_with_any_raw_backward']}/{r['n_checked']} "
          f"PWF={a['n_pass']}/{a['n_warn']}/{a['n_fail']}")

    key_fn = {
        "rise": lambda s: -s["temporal"]["max_rise"],
        "rise_rel": lambda s: -s["temporal"].get("max_rise_rel", 0.0),
        "kink45": lambda s: -s["radial_kink"]["theta=45"]["rel_kink"],
        "backward": lambda s: -s["raw_treal"].get(
            "max_backward_over_median_dt", 0),
    }[args.sort]
    per = sorted(r["per_sim"], key=key_fn)[:args.n]
    for i, s in enumerate(per):
        print(f"\n[{i}] {s['basename']} {s['verdict']}")
        rt = s["raw_treal"]
        print(f"  bak n={rt['n_backward']} "
              f"max_s={rt['max_backward']:.2e} "
              f"/dt={rt.get('max_backward_over_median_dt', 0):.2f} "
              f"med_dt={rt.get('median_dt', 0):.2e}")
        tp = s["temporal"]
        peak = tp.get("peak_descent", 0)
        print(f"  rise max={tp['max_rise']:.2e} "
              f"({100 * tp.get('max_rise_rel', 0):.1f}%) "
              f"p99={tp.get('p99_rise', 0):.2e} "
              f"peakDesc={peak:.2e}")
        print(f"  rise cells={tp['n_cells_with_rise']} "
              f"frac={tp['frac_cells_with_rise']:.4f}")
        # One-liner across all 3 angles.
        # rk = rise_from_min / peak_descent  (0 = no kink)
        # rMin = r where u_z is deepest;  rKink = r where u_z re-rises to
        # after that (should be near 1.0 for an edge artifact)
        rk = s["radial_kink"]
        parts = []
        for th_name in ("theta=0", "theta=45", "theta=90"):
            v = rk[th_name]
            parts.append(
                f"{th_name.split('=')[1]:>2}:rk={v['rel_kink']:.2f}"
                f"@rMin={v.get('r_of_min', 0):.2f}"
                f",rKink={v.get('r_of_kink', 0):.3f}")
        print(f"  " + " | ".join(parts))
        # theta=45 fine-grained u_z values at 4 outer r's so the
        # curve shape is visible in text.
        v45 = rk["theta=45"]["values_at_r"]
        print(f"  th45 u@.95={v45.get('0.95', 0):.2e} "
              f"u@.99={v45.get('0.99', 0):.2e} "
              f"u@.995={v45.get('0.995', 0):.2e} "
              f"u@.999={v45.get('0.999', 0):.2e}")
        print(f"  dead={s['dead_cells']['frac_dead']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
