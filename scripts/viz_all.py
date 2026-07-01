"""Run the whole viz suite on one NPZ folder (and optionally one trained run).

Convenience wrapper around the individual viz_*.py scripts so a single
command produces all the figures you typically want for a new dataset
or a freshly trained run. Each underlying viz is invoked as a
subprocess, so the produced commands match the manual recipes in
docs/VIZ_QUICK_USE.md exactly; you can copy any printed command and
re-run it by hand for debugging.

Two modes:

  --npz-dir <folder>                       data-only (no training)
      Produces:
        - viz/diversity.png (cross-sim std over the folder)
        - viz/per_sim/<basename>/topdown.gif
        - viz/per_sim/<basename>/kymo.png
        - viz/per_sim/<basename>/interactive.html  (if --include includes it)
      for the first --n-samples sims that pass preflight.

  --npz-dir <folder> --tag <tag>           full suite
      Adds:
        - viz/pod/spectrum.png
        - viz/pod/mode_atlas.png
        - viz/ml/err_vs_floor.png         (per_sim arrays required)
        - viz/ml/err_vs_floor_a6.png
        - viz/ml/ak_scatter.png
        - viz/worst/<rank>_<sim>.png      (top-N worst test sims)

Sensible defaults; override anything with the underlying viz's CLI
flags via --extra-<viz>-args "..." (rare).

    python scripts/viz_all.py --npz-dir /path/to/3d_npz \\
        --out viz/firehorse2

    python scripts/viz_all.py --npz-dir /path/to/3d_npz \\
        --tag firehorse2_n3_full --out viz/firehorse2_full

    # Skip the heavy interactive HTML
    python scripts/viz_all.py --npz-dir /path/to/3d_npz \\
        --include diversity,topdown,kymo --out viz/quick

    # Pick worst-by-error sims (requires --tag results.json)
    python scripts/viz_all.py --npz-dir /path/to/3d_npz \\
        --tag <tag> --select byerr --n-samples 5 --out viz/<...>

Re-running is cheap: viz outputs are not regenerated if the target
file already exists (use --force to overwrite). Total wall time on a
warm cache:
  data-only mode (3 sims):        ~30 s + diversity (~2-10 min on 5500)
  full suite, 3 sims:             ~5 min after caches warm
  full suite, 5 worst sims:       ~10 min after caches warm
"""

from __future__ import annotations
import argparse
import json
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


VIZ_NAMES = ("diversity", "topdown", "kymo", "interactive",
             "gif_3d", "strip_3d",
             "pod_spectrum", "pod_mode_atlas",
             "err_vs_floor", "ak_scatter", "worst")
# Names that need --tag (training-run dependent)
TAG_REQUIRED = {"pod_spectrum", "pod_mode_atlas",
                "err_vs_floor", "ak_scatter", "worst"}


def _run(cmd: list[str], skip_existing_target: Path | None,
         force: bool, log: list) -> bool:
    """Run one viz subprocess; record outcome. Returns True on success
    (or skip), False on failure."""
    if (skip_existing_target is not None and not force
            and skip_existing_target.exists()):
        log.append((skip_existing_target, "skipped", 0.0,
                    "target exists, --force to overwrite"))
        return True
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=False)
    dt = time.time() - t0
    ok = (r.returncode == 0)
    log.append((skip_existing_target, "ok" if ok else "FAIL", dt,
                f"exit={r.returncode}"))
    return ok


def _pick_sims(folder: Path, n: int, select: str,
               tag: str | None, output_dir: str) -> list[Path]:
    """Pick `n` NPZ files from folder by the chosen selection rule.

    'first': sorted-by-filename first N
    'random': uniform random N (seed 0 so re-runs are stable)
    'byerr': worst-error N from outputs/<tag>/results.json (needs
             per_sim_field_errs); fallback to 'first' if results.json
             missing
    """
    from data.loader import preflight_npz
    all_files = sorted(p for p in folder.glob("*.npz")
                       if not p.name.startswith("_"))
    # filter to preflight-passing only -- saves the operator a noisy
    # subprocess failure on a known-bad NPZ
    good = []
    for p in all_files:
        ok, _ = preflight_npz(p)
        if ok:
            good.append(p)
        if select == "first" and len(good) >= n:
            break
    if not good:
        return []

    if select == "first":
        return good[:n]
    if select == "random":
        rng = np.random.default_rng(0)
        idx = rng.choice(len(good), size=min(n, len(good)),
                         replace=False)
        return [good[i] for i in sorted(idx)]
    if select == "byerr":
        if not tag:
            print("WARN: --select byerr without --tag; falling back "
                  "to 'first'", file=sys.stderr)
            return good[:n]
        rs = Path(output_dir) / tag / "results.json"
        if not rs.is_file():
            print(f"WARN: {rs} not found; falling back to 'first'",
                  file=sys.stderr)
            return good[:n]
        with open(rs) as fp:
            r = json.load(fp)
        errs = r.get("per_sim_field_errs", [])
        bnames = r.get("per_sim_basenames", [])
        if not errs or not bnames:
            print("WARN: results.json missing per_sim arrays; "
                  "falling back to 'first'", file=sys.stderr)
            return good[:n]
        # Sort by descending error, pick top-n basenames that exist
        order = np.argsort(errs)[::-1]
        out = []
        good_by_name = {p.name: p for p in good}
        for i in order:
            if bnames[i] in good_by_name:
                out.append(good_by_name[bnames[i]])
                if len(out) >= n:
                    break
        return out
    raise ValueError(f"unknown --select {select!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", required=True,
                    help="folder of converted 3D NPZ files")
    ap.add_argument("--out", required=True,
                    help="output viz directory (created if missing)")
    ap.add_argument("--tag", default=None,
                    help="optional training-run tag; if given, adds "
                    "POD + ML + worst-case viz")
    ap.add_argument("--output-dir", default="outputs",
                    help="root for training outputs (default: outputs)")
    ap.add_argument("--basis-cache-dir", default=None,
                    help="override default basis_cache_dir lookup")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="number of sims to render per-sim viz on "
                    "(default 3)")
    ap.add_argument("--select", choices=("first", "random", "byerr"),
                    default="first",
                    help="how to pick the per-sim samples (default: "
                    "'first'; 'byerr' uses --tag's results.json)")
    ap.add_argument("--include", default=None,
                    help="comma list of viz names to include; default "
                    "= all applicable. Choices: " + ", ".join(VIZ_NAMES))
    ap.add_argument("--exclude", default=None,
                    help="comma list of viz names to skip")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output files; default "
                    "behavior is to skip already-rendered figures so "
                    "re-runs are cheap")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--diversity-limit", type=int, default=None,
                    help="cap on sims used for diversity viz; full "
                    "folder by default")
    ap.add_argument("--rebuild-diversity-cache", action="store_true",
                    help="force viz_diversity to redo the 93 GB load "
                    "+ Welford pass and overwrite its stats cache. "
                    "The stats cache holds the (mean, var, n_eff) "
                    "tensors keyed on (folder, grid, drop, limit); a "
                    "hit skips the giant read. Use this only when you "
                    "suspect the cache is stale (folder changed, npz "
                    "files added/removed without a rename).")
    ap.add_argument("--rebuild-worst-cache", action="store_true",
                    help="force viz_worst_cases to re-run "
                    "predict_run_fields (93 GB load + inference) and "
                    "overwrite its cache. The cache invalidates "
                    "automatically on retrain via a checkpoint "
                    "fingerprint; use this flag only when you know "
                    "the cache is wrong for some other reason.")
    ap.add_argument("--show-lower", action="store_true",
                    help="draw the flat lower-wafer reference plane on "
                    "all 3D viz (gif_3d, strip_3d, interactive). "
                    "Default off because the plane dominates the "
                    "figure; turn on for talks where the gap matters.")
    ap.add_argument("--workers", type=int, default=None,
                    help="loader worker count for folder-level viz "
                    "(diversity). None lets the loader auto-pick "
                    "min(host_cores - 2, 32); override e.g. on a fat "
                    "node or a shared box. Per-sim viz already runs "
                    "workers=1 in-process so this flag does not affect "
                    "them.")
    ap.add_argument("--topn-worst", type=int, default=5,
                    help="top-N for viz_worst_cases.py")
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    args = ap.parse_args()

    folder = Path(args.npz_dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"--npz-dir not a directory: {folder}", file=sys.stderr)
        return 2
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve viz set
    enabled = set(VIZ_NAMES)
    if args.include:
        wanted = set(s.strip() for s in args.include.split(","))
        unknown = wanted - enabled
        if unknown:
            print(f"--include unknown viz: {unknown}; valid: {VIZ_NAMES}",
                  file=sys.stderr)
            return 2
        enabled = wanted
    if args.exclude:
        enabled -= set(s.strip() for s in args.exclude.split(","))
    if not args.tag:
        enabled -= TAG_REQUIRED
    # interactive is heavy; only run when explicitly included via
    # --include or always by default? Default: include for n_samples
    # but warn.
    print(f"viz suite: enabled = {sorted(enabled)}", flush=True)
    print(f"out dir: {out_root}", flush=True)

    # Common args to thread into single-sim subprocesses
    common_grid = ["--nx", str(args.nx), "--ny", str(args.ny),
                   "--nt", str(args.nt),
                   "--drop-first-steps", str(args.drop_first_steps)]
    py = sys.executable

    log: list = []

    # === folder-level: diversity ===
    if "diversity" in enabled:
        target = out_root / "diversity.png"
        cmd = [py, str(_root / "scripts" / "viz_diversity.py"),
               "--npz-dir", str(folder),
               "--out", str(target),
               *common_grid]
        if args.diversity_limit:
            cmd += ["--limit", str(args.diversity_limit)]
        if args.workers is not None:
            cmd += ["--workers", str(args.workers)]
        if args.rebuild_diversity_cache:
            cmd += ["--force"]
        if args.tag:
            cmd += ["--tag", args.tag]
        _run(cmd, target, args.force, log)

    # === per-sim picks ===
    # In-process loop: load each picked sim ONCE via load_dataset
    # (single-file tempdir, cache=False), then call the render
    # functions of viz_topdown_gif and viz_radial_kymograph directly.
    # Saves ~80 s per sim per extra viz vs spawning subprocesses each
    # of which would re-do the Delaunay + interpolation. Interactive
    # HTML stays as a subprocess because plotly is an optional dep
    # we lazy-import only when needed.
    sims_picked = []
    _per_sim_inproc = {"topdown", "kymo", "gif_3d", "strip_3d"}
    if any(v in enabled for v in ("topdown", "kymo", "interactive",
                                    "gif_3d", "strip_3d")):
        sims_picked = _pick_sims(folder, args.n_samples, args.select,
                                 args.tag, args.output_dir)
        if not sims_picked:
            print("WARN: 0 preflight-passing NPZ for per-sim viz",
                  file=sys.stderr)
        # In-process imports -- only needed if at least one in-process
        # viz is enabled. Lazy-imported so a user who only wants
        # interactive HTML doesn't pay the matplotlib startup cost.
        if _per_sim_inproc & enabled:
            from data.loader import load_dataset
            from core.sensors import SensorConfig, place_sensors
            from scripts.viz_topdown_gif import render_topdown_gif
            from scripts.viz_radial_kymograph import render_radial_kymograph
            from scripts.viz_3d_gif import render_3d_gif
            from scripts.viz_3d_strip import render_3d_strip
            import shutil as _shutil
            import tempfile as _tempfile
            # Sensor positions (lab rig default)
            positions = ((1.0, 0.0), (1.0, 45.0), (1.0, 90.0))
            scfg = SensorConfig(n=3, strategy="custom",
                                positions=positions)
            sensor_xy = place_sensors(scfg)
        for sim_path in sims_picked:
            sim_stem = sim_path.stem
            sim_dir = out_root / "per_sim" / sim_stem
            sim_dir.mkdir(parents=True, exist_ok=True)

            # Figure out which of {topdown, kymo, gif_3d, strip_3d}
            # need rendering for THIS sim (skip-existing per-target).
            tgt_topdown = sim_dir / "topdown.gif"
            tgt_kymo = sim_dir / "kymo.png"
            tgt_gif3d = sim_dir / "wafer_3d.gif"
            tgt_strip3d = sim_dir / "wafer_3d_strip.png"
            need_topdown = ("topdown" in enabled
                            and (args.force or not tgt_topdown.exists()))
            need_kymo = ("kymo" in enabled
                         and (args.force or not tgt_kymo.exists()))
            need_gif3d = ("gif_3d" in enabled
                          and (args.force or not tgt_gif3d.exists()))
            need_strip3d = ("strip_3d" in enabled
                            and (args.force or not tgt_strip3d.exists()))

            # Skip-existing accounting (logged even when no work done)
            if "topdown" in enabled and not need_topdown:
                log.append((tgt_topdown, "skipped", 0.0,
                            "target exists, --force to overwrite"))
            if "kymo" in enabled and not need_kymo:
                log.append((tgt_kymo, "skipped", 0.0,
                            "target exists, --force to overwrite"))
            if "gif_3d" in enabled and not need_gif3d:
                log.append((tgt_gif3d, "skipped", 0.0,
                            "target exists, --force to overwrite"))
            if "strip_3d" in enabled and not need_strip3d:
                log.append((tgt_strip3d, "skipped", 0.0,
                            "target exists, --force to overwrite"))

            # Load the sim only if at least one in-process viz is wanted.
            if need_topdown or need_kymo or need_gif3d or need_strip3d:
                print(f"\n[in-process] loading {sim_path.name} ...",
                      flush=True)
                t_load = time.time()
                with _tempfile.TemporaryDirectory() as td:
                    staged = Path(td) / sim_path.name
                    _shutil.copy(sim_path, staged)
                    x_canon, y_canon, sims_loaded = load_dataset(
                        Path(td), nx=args.nx, ny=args.ny, nt=args.nt,
                        cache=False, workers=1,
                        drop_first_steps=args.drop_first_steps)
                if not sims_loaded:
                    log.append((sim_dir, "FAIL", time.time() - t_load,
                                f"loader rejected {sim_path.name}"))
                    continue
                sim = sims_loaded[0]
                print(f"[in-process] loaded {sim.f.shape} in "
                      f"{time.time() - t_load:.1f}s", flush=True)

                if need_topdown:
                    t0 = time.time()
                    try:
                        render_topdown_gif(
                            sim, x_canon, y_canon, sensor_xy, tgt_topdown,
                            sim_id=sim_stem, tag=args.tag,
                            drop_first_steps=args.drop_first_steps)
                        log.append((tgt_topdown, "ok",
                                    time.time() - t0, "in-process"))
                    except Exception as e:                  # noqa: BLE001
                        log.append((tgt_topdown, "FAIL",
                                    time.time() - t0, f"{type(e).__name__}: {e}"))
                if need_kymo:
                    t0 = time.time()
                    try:
                        render_radial_kymograph(
                            sim, x_canon, y_canon, tgt_kymo,
                            value_scale=args.value_scale,
                            sim_id=sim_stem, tag=args.tag,
                            drop_first_steps=args.drop_first_steps)
                        log.append((tgt_kymo, "ok",
                                    time.time() - t0, "in-process"))
                    except Exception as e:                  # noqa: BLE001
                        log.append((tgt_kymo, "FAIL",
                                    time.time() - t0, f"{type(e).__name__}: {e}"))
                if need_gif3d:
                    t0 = time.time()
                    try:
                        render_3d_gif(
                            sim, x_canon, y_canon, sensor_xy, tgt_gif3d,
                            show_lower=args.show_lower,
                            value_scale=args.value_scale,
                            sim_id=sim_stem, tag=args.tag,
                            drop_first_steps=args.drop_first_steps)
                        log.append((tgt_gif3d, "ok",
                                    time.time() - t0, "in-process"))
                    except Exception as e:                  # noqa: BLE001
                        log.append((tgt_gif3d, "FAIL",
                                    time.time() - t0,
                                    f"{type(e).__name__}: {e}"))
                if need_strip3d:
                    t0 = time.time()
                    try:
                        render_3d_strip(
                            sim, x_canon, y_canon, sensor_xy, tgt_strip3d,
                            show_lower=args.show_lower,
                            value_scale=args.value_scale,
                            sim_id=sim_stem, tag=args.tag,
                            drop_first_steps=args.drop_first_steps)
                        log.append((tgt_strip3d, "ok",
                                    time.time() - t0, "in-process"))
                    except Exception as e:                  # noqa: BLE001
                        log.append((tgt_strip3d, "FAIL",
                                    time.time() - t0,
                                    f"{type(e).__name__}: {e}"))

            # Interactive HTML stays a subprocess (plotly is optional)
            if "interactive" in enabled:
                target = sim_dir / "interactive.html"
                cmd = [py, str(_root / "scripts" / "viz_interactive.py"),
                       "--sim", str(sim_path),
                       "--out", str(target),
                       *common_grid,
                       *(["--tag", args.tag] if args.tag else []),
                       *(["--show-lower"] if args.show_lower else []),
                       "--value-scale", str(args.value_scale)]
                _run(cmd, target, args.force, log)

    # === POD viz (needs basis cache) ===
    if {"pod_spectrum", "pod_mode_atlas"} & enabled:
        # Locate basis file. If user passed --basis-cache-dir use it;
        # else look under outputs/basis_cache/. Pick the most recent
        # pod3d_*.npz file (heuristic; if you want a specific one,
        # invoke viz_pod_spectrum directly).
        bcdir = Path(args.basis_cache_dir or
                     Path(args.output_dir) / "basis_cache")
        candidates = sorted(bcdir.glob("pod3d_*.npz"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True)
        if not candidates:
            print(f"WARN: no pod3d_*.npz in {bcdir}; skipping POD viz",
                  file=sys.stderr)
        else:
            basis = candidates[0]
            print(f"POD basis -> {basis.name}", flush=True)
            tag_args = (["--tag", args.tag] if args.tag else [])
            if "pod_spectrum" in enabled:
                target = out_root / "pod" / "spectrum.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                cmd = [py, str(_root / "scripts" / "viz_pod_spectrum.py"),
                       "--basis", str(basis),
                       "--out", str(target),
                       *tag_args]
                _run(cmd, target, args.force, log)
            if "pod_mode_atlas" in enabled:
                target = out_root / "pod" / "mode_atlas.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                cmd = [py, str(_root / "scripts" / "viz_pod_mode_atlas.py"),
                       "--basis", str(basis),
                       "--out", str(target),
                       *tag_args]
                _run(cmd, target, args.force, log)

    # === ML diagnostic viz (needs results.json with per_sim arrays) ===
    if {"err_vs_floor", "ak_scatter"} & enabled and args.tag:
        rs = Path(args.output_dir) / args.tag / "results.json"
        if not rs.is_file():
            print(f"WARN: {rs} not found; skipping ML viz",
                  file=sys.stderr)
        else:
            ml_dir = out_root / "ml"
            ml_dir.mkdir(parents=True, exist_ok=True)
            tag_args = ["--tag", args.tag]
            if "err_vs_floor" in enabled:
                target = ml_dir / "err_vs_floor.png"
                cmd = [py, str(_root / "scripts" / "viz_error_vs_floor.py"),
                       "--results", str(rs),
                       "--out", str(target),
                       *tag_args]
                _run(cmd, target, args.force, log)
                # plus the a_6-colored variant (cheap, useful)
                target = ml_dir / "err_vs_floor_a6.png"
                cmd = [py, str(_root / "scripts" / "viz_error_vs_floor.py"),
                       "--results", str(rs),
                       "--color-by", "a_6",
                       "--out", str(target),
                       *tag_args]
                _run(cmd, target, args.force, log)
            if "ak_scatter" in enabled:
                target = ml_dir / "ak_scatter.png"
                cmd = [py, str(_root / "scripts" / "viz_ak_scatter.py"),
                       "--results", str(rs),
                       "--out", str(target),
                       *tag_args]
                _run(cmd, target, args.force, log)

    # === Worst-cases ===
    if "worst" in enabled and args.tag:
        worst_dir = out_root / "worst"
        # Special: worst writes many files; skip-existing checks if the
        # directory has any files already.
        existing = list(worst_dir.glob("*.png")) if worst_dir.is_dir() else []
        if existing and not args.force:
            log.append((worst_dir, "skipped", 0.0,
                        f"{len(existing)} files already; --force to redo"))
        else:
            worst_dir.mkdir(parents=True, exist_ok=True)
            cmd = [py, str(_root / "scripts" / "viz_worst_cases.py"),
                   "--tag", args.tag,
                   "--topn", str(args.topn_worst),
                   "--out", str(worst_dir),
                   "--output-dir", args.output_dir]
            if args.npz_dir:
                cmd += ["--data-dir-override", str(folder)]
            if args.rebuild_worst_cache:
                cmd += ["--force"]
            _run(cmd, worst_dir / "DUMMY", args.force, log)
            # ^ DUMMY target ensures _run never sees an existing file;
            # we already gated above

    # === summary ===
    print("\n" + "=" * 72)
    print(f"VIZ SUITE SUMMARY  (out: {out_root})")
    print("=" * 72)
    print(f"  {'status':>8}  {'wall':>7}  target / note")
    total = 0.0
    n_ok = n_skip = n_fail = 0
    for target, status, dt, note in log:
        marker = ({"ok": "OK", "skipped": "skip",
                   "FAIL": "FAIL"}).get(status, status)
        total += dt
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skip += 1
        elif status == "FAIL":
            n_fail += 1
        target_str = (str(target.relative_to(out_root.parent))
                      if target and out_root.parent in target.parents
                      else (str(target) if target else "-"))
        print(f"  {marker:>8}  {dt:6.1f}s  {target_str}  {note}")
    print(f"\n  total wall: {total:.1f}s  "
          f"({n_ok} ok, {n_skip} skipped, {n_fail} failed)")
    if n_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
