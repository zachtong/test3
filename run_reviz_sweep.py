"""Re-render kymo + radial_anim for existing sweep configs with the
new default angle set (0, 22.5, 45, 67.5, 90 deg).

Uses the SAME --pick / --topn as run_sweep.py so the prediction
cache (viz_test_cases's _test_cache_*.npz) HITs -- no 93 GB F
reload, no inference, just matplotlib re-render. snapshot and
interactive_compare layouts are angle-independent and NOT
re-rendered here.

Modes:
    python run_reviz_sweep.py               # all 57 configs (~14h)
    python run_reviz_sweep.py --top-k 10    # top-10 by median err (~2.5h)
    python run_reviz_sweep.py --tags sweep_n5_ABDEF sweep_n4_BDEF
                                            # explicit tag list

The --top-k mode requires viz/sweep_summary.csv to already exist
(produced by scripts/summarize_sweep.py). If it does not, defaults
to all configs.

Existing kymo / radial_anim files are overwritten in-place.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def _discover_all_sweep_dirs(outputs: Path,
                              prefix: str) -> list[str]:
    """Every outputs/<prefix>* that contains a checkpoint."""
    tags = []
    for d in sorted(outputs.glob(f"{prefix}*")):
        # A trained config has a checkpoints subdir (see
        # training.checkpoint.save_checkpoint layout).
        if not (d / "checkpoints").is_dir():
            # Fallback: any config that at least produced results.json
            if not (d / "results.json").is_file():
                continue
        tags.append(d.name)
    return tags


def _top_k_from_summary(csv_path: Path, k: int) -> list[str] | None:
    if not csv_path.is_file():
        return None
    rows: list[tuple[float, str]] = []
    with open(csv_path) as fp:
        for row in csv.DictReader(fp):
            v = row.get("median_field_err")
            tag = row.get("tag")
            if v is None or not tag:
                continue
            try:
                rows.append((float(v), tag))
            except ValueError:
                continue
    if not rows:
        return None
    rows.sort()
    return [t for _, t in rows[:k]]


def run_viz(tag: str, out_root: Path, pick: str, topn: int,
            layout: str) -> tuple[bool, float]:
    """viz_test_cases has a legacy single-selection behaviour: with
    multiple picks it puts each in its own subdir under --out, but
    with a SINGLE pick it dumps files directly into --out (no
    subdir). To keep the on-disk layout consistent regardless of
    how many picks the user passed, append the pick name to --out
    when the pick string has no comma."""
    picks = [p.strip() for p in pick.split(",") if p.strip()]
    base_out = out_root / tag / "all_picks"
    if len(picks) == 1:
        out_path = base_out / picks[0]
    else:
        out_path = base_out
    t0 = time.time()
    cmd = [
        PY, "scripts/viz_test_cases.py",
        "--tag", tag,
        "--out", str(out_path) + "/",
        "--pick", pick,
        "--topn", str(topn),
        "--layout", layout,
        "--show-lower",
    ]
    print(f"[reviz] {tag}", flush=True)
    ok = True
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAIL exit={e.returncode}", flush=True)
        ok = False
    dt = time.time() - t0
    return ok, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", default="outputs",
                    help="root dir with sweep_*/checkpoints "
                    "(default: outputs)")
    ap.add_argument("--prefix", default="sweep_",
                    help="tag prefix to match (default: sweep_)")
    ap.add_argument("--out-root", default="viz",
                    help="root dir under which per-tag folders live "
                    "(default: viz)")
    ap.add_argument("--summary-csv", default="viz/sweep_summary.csv",
                    help="path to summarize_sweep.py output; needed "
                    "when --top-k is used")
    ap.add_argument("--top-k", type=int, default=None,
                    help="only re-render the top-K configs by "
                    "median field error (needs --summary-csv)")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="explicit tag list (overrides --top-k and "
                    "discovery)")
    ap.add_argument("--pick", default="worst,best,median,random",
                    help="comma list of pick strategies passed to "
                    "viz_test_cases --pick. Default matches what "
                    "run_sweep.py used (worst,best,median,random) "
                    "so viz_test_cases' prediction cache HITs -- "
                    "fast render. Changing this string forces a "
                    "cache MISS -> ~30 min extra per config for "
                    "predict_run_fields to rerun.")
    ap.add_argument("--topn", type=int, default=10,
                    help="how many sims per pick strategy. Default "
                    "10 matches run_sweep.py; same cache-miss "
                    "caveat as --pick.")
    ap.add_argument("--layout", default="kymo,radial_anim",
                    help="comma list of viz_test_cases layouts. "
                    "Default 'kymo,radial_anim' -- the only two "
                    "layouts that sample along angles, so the "
                    "only ones affected by the 5-angle default. "
                    "Add 'snapshot' or 'interactive_compare' if "
                    "you also want to re-render those (they are "
                    "angle-independent so this is redundant unless "
                    "you deleted their files).")
    args = ap.parse_args()

    if args.tags:
        tags = list(args.tags)
        source = "explicit --tags"
    elif args.top_k is not None:
        top = _top_k_from_summary(Path(args.summary_csv),
                                    args.top_k)
        if top is None:
            print(f"summary CSV not found at {args.summary_csv}; "
                  "falling back to all configs", file=sys.stderr)
            tags = _discover_all_sweep_dirs(Path(args.outputs),
                                              args.prefix)
            source = "all (summary CSV missing)"
        else:
            tags = top
            source = f"top-{args.top_k} from {args.summary_csv}"
    else:
        tags = _discover_all_sweep_dirs(Path(args.outputs),
                                          args.prefix)
        source = "all sweep_* dirs"

    if not tags:
        print("no configs to re-render", file=sys.stderr)
        return 1

    # Prediction cache hits only when pick + topn match run_sweep.py.
    cache_matched = (args.pick == "worst,best,median,random"
                      and args.topn == 10)
    est_min = 15 if cache_matched else 45
    if not cache_matched:
        print(f"WARNING: --pick={args.pick} or --topn={args.topn} "
              f"differ from run_sweep.py defaults "
              f"(worst,best,median,random / 10). viz_test_cases "
              f"prediction cache will MISS, forcing "
              f"predict_run_fields to re-run per config "
              f"(~+30 min each). Consider running with the "
              f"defaults if you want to save that time.",
              flush=True)
    print(f"Re-rendering {len(tags)} config(s) [{source}]. "
          f"Estimate: ~{len(tags) * est_min} min "
          f"({len(tags) * est_min / 60:.1f} h) at "
          f"{est_min} min/config.", flush=True)
    out_root = Path(args.out_root)
    t_start = time.time()
    ok_count = 0
    fail_count = 0
    dts: list[float] = []
    for i, tag in enumerate(tags, 1):
        elapsed_min = (time.time() - t_start) / 60
        avg = (sum(dts) / len(dts) / 60) if dts else est_min
        eta_min = avg * (len(tags) - i + 1)
        print(f"\n===== [{i}/{len(tags)}] {tag}   "
              f"elapsed {elapsed_min:.1f} min, "
              f"ETA {eta_min:.1f} min =====", flush=True)
        ok, dt = run_viz(tag, out_root, args.pick, args.topn,
                          args.layout)
        dts.append(dt)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
        print(f"  {tag} done in {dt / 60:.1f} min", flush=True)
    total_min = (time.time() - t_start) / 60
    print(f"\nreviz done in {total_min:.1f} min "
          f"({total_min / 60:.2f} h) -- "
          f"ok={ok_count}/{len(tags)}  fail={fail_count}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
