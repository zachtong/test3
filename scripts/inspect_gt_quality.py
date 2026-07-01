"""Pre-training GT trustworthiness diagnostic.

Samples N sims from an NPZ folder, loads them through the CURRENT
loader (rebound-fix + nearest-fill applied), and runs four checks:

  A. Raw tReal monotonicity -- for each sim's raw sample_tReal array,
     count backward jumps and report the largest by magnitude. Post-
     fix the LOADER clamps these, but this reveals how dirty the
     upstream COMSOL export was. Sims with big raw backwards are
     the ones whose predictions previously showed rebound-like
     artifacts.

  B. Loaded temporal monotonicity -- for the loaded canonical field
     f, compute the largest positive diff along the time axis at any
     in-disk cell. Physical wafer bonding is monotonic descent, so
     any positive diff after the fix indicates residual artifact.

  C. Radial kink at r near 1 -- along each of theta = 0/45/90 deg,
     sample u_z at r in {0.85, 0.90, 0.95, 0.99} at the final time
     step and compute the relative jump abs(u_z(r=0.99) -
     u_z(r=0.95)) / max(abs(u_z(r=0.95)), tiny). A large relative
     jump with the edge cell being LESS descended than the interior
     is the 'upward kink' the operator reported at 45 deg.

  D. Dead-cell fraction -- fraction of in-disk cells whose loaded
     f is exactly zero across all time. With the nearest-fill patch
     applied this should be zero; a non-zero value indicates the
     canonicalization is still leaving unfilled cells.

Output:
  <out>/summary.json      per-sim + aggregate stats
  <out>/gt_quality.png    6-panel diagnostic figure

Runtime: ~O(N x 30s) with default 32 loader workers -- 20 sims takes
~2-3 min after the loader cache warms for those particular sims.
Since we pass files via a per-run tempdir (symlinks), the loader
cache never persists between invocations.

    python scripts/inspect_gt_quality.py --npz-dir <folder> \\
        --n-check 20 --out viz/gt_quality/

    # deterministic sample (same 20 sims across re-runs)
    python scripts/inspect_gt_quality.py --npz-dir <folder> \\
        --n-check 20 --seed 0 --out viz/gt_quality/

    # inspect specific sims
    python scripts/inspect_gt_quality.py --npz-dir <folder> \\
        --sim run_00473.npz,run_01102.npz --out viz/gt_quality/
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.loader import (load_dataset,                          # noqa: E402
                          _DISK_MASK_R_END)
from scripts.viz_radial_kymograph import _sample_radial_kymograph  # noqa: E402
from scripts.fieldviz import (provenance_footer,                # noqa: E402
                               WAFER_CMAP, SENSOR_MARKER_COLOR)


# Thresholds. WARN = suspicious; FAIL = definitely a data problem
# (retraining on this sim would learn artifact).
_TREAL_BACKWARD_WARN = 1     # any raw backward is worth noting
# Rise is measured as a fraction of the sim's own peak descent, not
# an absolute metres value -- otherwise sims with 15 um descent get
# scored the same as sims with 150 um descent for the same absolute
# rise. FAIL = rise > 5% of peak descent (clearly artifactual);
# WARN = rise > 0.5% (noticeable but might be numerical noise).
_TEMPORAL_RISE_REL_WARN = 0.005
_TEMPORAL_RISE_REL_FAIL = 0.05
# _RISE_COUNT_THRESHOLD is what we use for n_cells_with_rise (a per-
# cell counter). Set to 1% of peak descent -- below this it is float32
# noise, not signal. Kept relative to the sim so all sims agree on
# what counts as 'has rise'.
_RISE_COUNT_REL_THRESHOLD = 0.01
_KINK_REL_WARN = 0.30        # 30% relative jump at edge
_KINK_REL_FAIL = 0.60        # 60% = a clear discontinuity
_DEAD_CELL_WARN = 0.02       # 2% of in-disk cells with all-zero traces
_DEAD_CELL_FAIL = 0.05


def _raw_treal_stats(npz_path: Path) -> dict:
    """Read sample_tReal from raw NPZ and report backward-jump stats."""
    with np.load(npz_path, allow_pickle=True) as z:
        treal = np.asarray(z["sample_tReal"], dtype=np.float64)
    S = treal.size
    if S < 2:
        return dict(n_samples=int(S), n_backward=0,
                     max_backward=0.0, median_dt=0.0)
    dt = np.diff(treal)
    n_backward = int((dt < 0).sum())
    max_backward = float(-dt.min()) if n_backward else 0.0
    median_dt = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0
    return dict(n_samples=int(S), n_backward=n_backward,
                 max_backward=max_backward,
                 max_backward_over_median_dt=(
                     float(max_backward / median_dt)
                     if median_dt > 0 else 0.0),
                 median_dt=median_dt)


def _temporal_monotonicity(f: np.ndarray, in_disk: np.ndarray) -> dict:
    """Rise stats along t at every in-disk cell.

    Returns:
      max_rise       largest positive du_z at any in-disk cell / t
      p99_rise       99th percentile of per-cell max rise; robust to
                     a single outlier and shows what the 'typical bad
                     cell' looks like
      median_rise    the median per-cell max rise (numerical-noise
                     baseline; anything WAY above this is signal)
      n_cells_with_rise / frac_cells_with_rise
                     count of cells whose max rise exceeds 1% of
                     peak descent (relative-signal cells, not noise)
      peak_descent   sim's most-negative u_z (abs value); the
                     natural scale for reading rise magnitudes
      max_rise_rel   max_rise / peak_descent  (0 = no rise;
                     0.05 = rise is 5% of descent -- FAIL)
    """
    diff = np.diff(f, axis=-1)                # (Nx, Ny, Nt-1)
    diff_in_disk = diff[in_disk]              # (M, Nt-1)
    peak_descent = float(np.abs(np.minimum(f[in_disk], 0.0)).max())
    if diff_in_disk.size == 0 or peak_descent == 0:
        return dict(max_rise=0.0, p99_rise=0.0, median_rise=0.0,
                     n_cells_with_rise=0, frac_cells_with_rise=0.0,
                     peak_descent=peak_descent, max_rise_rel=0.0)
    per_cell_max = diff_in_disk.max(axis=-1)
    max_rise = float(per_cell_max.max())
    p99_rise = float(np.percentile(per_cell_max, 99))
    median_rise = float(np.percentile(per_cell_max, 50))
    threshold = _RISE_COUNT_REL_THRESHOLD * peak_descent
    n_bad = int((per_cell_max > threshold).sum())
    return dict(max_rise=max_rise, p99_rise=p99_rise,
                 median_rise=median_rise,
                 n_cells_with_rise=n_bad,
                 frac_cells_with_rise=float(n_bad
                                             / diff_in_disk.shape[0]),
                 peak_descent=peak_descent,
                 max_rise_rel=float(max_rise / peak_descent))


def _radial_kink_stats(f: np.ndarray, x_canon: np.ndarray,
                        y_canon: np.ndarray,
                        angles: tuple = (0.0, 45.0, 90.0),
                        r_query: tuple = (0.85, 0.90, 0.95, 0.97,
                                            0.99, 0.995, 0.999)
                        ) -> dict:
    """At the final timestep, look for an upward kink at the disk
    edge -- the artifact the operator sees in the radial-anim viz.

    'edge kink' = the rise-back-from-deepest-descent along the ray.
    Concretely:
      1. Sample u_z along the ray at 512 r points in [0, 1].
      2. Find r_min = argmin(u_z) -- deepest descent along the ray.
      3. Report max_u_after_min = max(u_z[r >= r_min]) which is the
         edge cell's value if u_z stays flat, or something less
         descended if u_z curves up.
      4. rise_from_min = max_u_after_min - u_z_at_r_min.
      5. rel_kink = rise_from_min / peak_descent  (0 = no kink;
         0.5 = edge is 50% less descended than the deepest point).

    Physical wafer bonding curves down MONOTONICALLY from center to
    edge (or is flat once fully bonded), so any rise_from_min > 0
    means the loader is producing a non-physical bump right at the
    rim. r_of_kink pinpoints where; a small r_of_kink relative to 1
    means the artifact is right at the edge (the r > 0.99 region
    the operator identified).

    Also stored: raw u_z values at the requested r_query points so
    the dumper can show the full curve shape without re-computing.
    """
    # Only scan the UNMASKED range (r < _DISK_MASK_R_END). Cells
    # beyond that are intentionally zeroed by the loader; treating
    # them as a kink would confuse the metric with the mask itself.
    # Leave a small epsilon so the very edge of the mask is not
    # sampled (bilinear resampling can bleed the mask edge).
    r_upper = max(0.0, _DISK_MASK_R_END - 0.005)
    out = {}
    for th in angles:
        n_r = 512
        km = _sample_radial_kymograph(
            f.astype(np.float64), x_canon, y_canon, th, n_r=n_r)
        final_full = km[:, -1]                            # (n_r,)
        r_axis_full = np.linspace(0.0, 1.0, n_r)
        inside_mask = r_axis_full <= r_upper
        final = final_full[inside_mask]
        r_axis = r_axis_full[inside_mask]
        # r_query values in the masked shell (>= _DISK_MASK_R_END)
        # will render as 0 by definition of the mask. Show them
        # anyway so the dumper output reveals the mask working.
        vals = {r: float(np.interp(r, r_axis_full, final_full))
                 for r in r_query}
        if final.size == 0:
            # Degenerate: mask covers everything. Should not happen
            # unless _DISK_MASK_R_END was configured near 0.
            out[f"theta={th:g}"] = dict(
                values_at_r=vals, rel_kink=0.0,
                rise_from_min=0.0, r_of_min=0.0, r_of_kink=0.0,
                u_at_min=0.0, edge_less_descended=False)
            continue
        argmin = int(np.argmin(final))
        r_of_min = float(r_axis[argmin])
        u_at_min = float(final[argmin])
        after_min = final[argmin:]
        max_u_after = float(after_min.max())
        rise_from_min = float(max(0.0, max_u_after - u_at_min))
        peak_descent = float(abs(u_at_min))
        rel_kink = float(rise_from_min / max(peak_descent, 1e-12))
        argmax_after = int(np.argmax(after_min))
        r_of_kink = float(r_axis[argmin + argmax_after])
        edge_less_descended = rise_from_min > 0
        out[f"theta={th:g}"] = dict(
            values_at_r=vals,
            rel_kink=rel_kink,
            rise_from_min=rise_from_min,
            r_of_min=r_of_min,
            r_of_kink=r_of_kink,
            u_at_min=u_at_min,
            edge_less_descended=edge_less_descended)
    return out


def _dead_cell_stats(f: np.ndarray, in_disk: np.ndarray) -> dict:
    """Cells where f is IDENTICALLY zero across every timestep."""
    all_zero = (f == 0).all(axis=-1)          # (Nx, Ny) bool
    dead_in_disk = int((all_zero & in_disk).sum())
    total_in_disk = int(in_disk.sum())
    return dict(n_dead=dead_in_disk, n_in_disk=total_in_disk,
                 frac_dead=(float(dead_in_disk / total_in_disk)
                             if total_in_disk > 0 else 0.0))


def _verdict(stats: dict) -> str:
    """Combine per-sim stat dicts into PASS / WARN / FAIL.

    Rise is judged RELATIVE to the sim's own peak descent so a
    150 um-descent sim with 1 um rise (0.7% relative) is not the same
    as a 15 um-descent sim with 1 um rise (6.7% relative). See
    _TEMPORAL_RISE_REL_WARN / FAIL for the threshold definitions.
    """
    rr = stats["temporal"].get("max_rise_rel", 0.0)
    if rr > _TEMPORAL_RISE_REL_FAIL:
        return "FAIL"
    if stats["dead_cells"]["frac_dead"] > _DEAD_CELL_FAIL:
        return "FAIL"
    if any(v["rel_kink"] > _KINK_REL_FAIL and v["edge_less_descended"]
           for v in stats["radial_kink"].values()):
        return "FAIL"
    if rr > _TEMPORAL_RISE_REL_WARN:
        return "WARN"
    if stats["dead_cells"]["frac_dead"] > _DEAD_CELL_WARN:
        return "WARN"
    if any(v["rel_kink"] > _KINK_REL_WARN and v["edge_less_descended"]
           for v in stats["radial_kink"].values()):
        return "WARN"
    if stats["raw_treal"]["n_backward"] > _TREAL_BACKWARD_WARN:
        return "WARN"
    return "PASS"


def _pick_files(folder: Path, n_check: int, sim_arg: str | None,
                 seed: int) -> list[Path]:
    """Pick n_check NPZ files from folder (or --sim list)."""
    all_files = sorted(p for p in folder.glob("*.npz")
                        if not p.name.startswith("_"))
    if sim_arg:
        wanted = set(s.strip() for s in sim_arg.split(",") if s.strip())
        picked = [p for p in all_files if p.name in wanted]
        missing = wanted - {p.name for p in picked}
        if missing:
            print(f"WARN: {len(missing)} basename(s) not found in "
                  f"folder: {sorted(missing)[:5]}"
                  f"{'...' if len(missing) > 5 else ''}",
                  file=sys.stderr)
        if not picked:
            raise SystemExit("no requested sims found in folder")
        return picked
    if not all_files:
        raise SystemExit(f"no NPZ files in {folder}")
    if n_check >= len(all_files):
        return all_files
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_files), size=n_check, replace=False)
    return [all_files[i] for i in sorted(idx)]


def _load_via_tempdir(files: list[Path], nx: int, ny: int, nt: int,
                       drop_first_steps: int, workers: int | None):
    """Symlink picked files into a tempdir and run load_dataset on it.

    Uses cache=True so the temp folder can benefit from the loader's
    parallel-workers path even for a single-file call, but the cache
    dies with the tempdir so nothing persists on disk."""
    td = Path(tempfile.mkdtemp(prefix="inspect_gt_"))
    try:
        for src in files:
            (td / src.name).symlink_to(src.resolve())
        return td, load_dataset(
            td, nx=nx, ny=ny, nt=nt, cache=True,
            workers=workers,
            drop_first_steps=drop_first_steps)
    except Exception:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
        raise


def _render_report(all_stats: list, per_sim_final_curves: list,
                    out_png: Path, tag: str = ""):
    """6-panel PNG summarizing findings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_stats)
    max_rises = np.array([s["temporal"]["max_rise"] for s in all_stats])
    frac_deads = np.array([s["dead_cells"]["frac_dead"]
                            for s in all_stats])
    max_backwards = np.array([s["raw_treal"]["max_backward_over_median_dt"]
                               for s in all_stats])
    # Take worst kink across angles per sim.
    per_sim_kink = np.array([
        max(v["rel_kink"] for v in s["radial_kink"].values())
        for s in all_stats])
    per_sim_kink_45 = np.array([
        s["radial_kink"]["theta=45"]["rel_kink"] for s in all_stats])

    verdicts = [s["verdict"] for s in all_stats]
    n_pass = verdicts.count("PASS")
    n_warn = verdicts.count("WARN")
    n_fail = verdicts.count("FAIL")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5),
                              constrained_layout=True)

    # (0,0): raw tReal backward jump distribution
    ax = axes[0, 0]
    ax.hist(max_backwards, bins=20, color="0.4",
             edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("max raw tReal backward (multiples of median dt)")
    ax.set_ylabel("# sims (log)")
    ax.set_title(f"A. Raw tReal backward jumps  "
                 f"(sims with any: {int((max_backwards > 0).sum())}/{n})",
                 fontsize=10)
    ax.axvline(1.0, color="orange", ls="--", lw=1, label="1x median dt")
    ax.legend(fontsize=8)

    # (0,1): temporal rise (loaded) RELATIVE to peak descent
    ax = axes[0, 1]
    rise_rel = np.array([s["temporal"].get("max_rise_rel", 0.0)
                          for s in all_stats])
    ax.hist(rise_rel, bins=20, color=SENSOR_MARKER_COLOR,
             edgecolor="black", linewidth=0.5)
    ax.set_xlabel("max rise / peak descent  (dimensionless)")
    ax.set_ylabel("# sims")
    ax.set_title(f"B. Loaded temporal monotonicity  "
                 f"(FAIL >= {_TEMPORAL_RISE_REL_FAIL:.2f} = 5%)",
                 fontsize=10)
    ax.axvline(_TEMPORAL_RISE_REL_WARN, color="orange",
                ls="--", lw=1, label=f"WARN {_TEMPORAL_RISE_REL_WARN:.1%}")
    ax.axvline(_TEMPORAL_RISE_REL_FAIL, color="red",
                ls="--", lw=1, label=f"FAIL {_TEMPORAL_RISE_REL_FAIL:.1%}")
    ax.legend(fontsize=8)

    # (0,2): radial kink -- worst angle per sim
    ax = axes[0, 2]
    ax.hist(per_sim_kink, bins=20, color="0.4",
             edgecolor="black", linewidth=0.5,
             label="worst angle")
    ax.hist(per_sim_kink_45, bins=20, color="orange",
             alpha=0.5, edgecolor="black", linewidth=0.5,
             label="theta=45 only")
    ax.set_xlabel("relative kink at r=0.99 vs r=0.95")
    ax.set_ylabel("# sims")
    ax.set_title(f"C. Radial kink at r~1 (final frame)",
                 fontsize=10)
    ax.axvline(_KINK_REL_WARN, color="orange", ls="--", lw=1,
                label=f"WARN >= {_KINK_REL_WARN:g}")
    ax.axvline(_KINK_REL_FAIL, color="red", ls="--", lw=1,
                label=f"FAIL >= {_KINK_REL_FAIL:g}")
    ax.legend(fontsize=8)

    # (1, 0-2): 3 worst-kink sims' final-frame u_z(r) at theta=45
    # so the operator can see the artifact directly.
    kink_45 = np.array([
        (s["radial_kink"]["theta=45"]["rel_kink"], i, s["basename"])
        for i, s in enumerate(all_stats)],
        dtype=object)
    order = np.argsort([-x[0] for x in kink_45])[:3]
    for slot, orig_idx in enumerate(order):
        ax = axes[1, slot]
        sim_slot = int(kink_45[orig_idx][1])
        bname = str(kink_45[orig_idx][2])
        rel = float(kink_45[orig_idx][0])
        curves = per_sim_final_curves[sim_slot]
        r_axis = np.linspace(0, 1, curves["theta=0"].size)
        for th_key, color in (("theta=0", "0.2"),
                              ("theta=45", SENSOR_MARKER_COLOR),
                              ("theta=90", "steelblue")):
            ax.plot(r_axis, curves[th_key] * 1e6,
                     color=color, lw=1.5, label=th_key)
        ax.axvline(0.95, color="gray", ls=":", lw=0.7)
        ax.axvline(0.99, color="gray", ls=":", lw=0.7)
        ax.text(0.97, ax.get_ylim()[1], "r=0.95..0.99", fontsize=7,
                 ha="center", va="top", color="gray")
        ax.set_xlabel("r (normalized)")
        ax.set_ylabel("u_z * 1e6")
        ax.set_title(f"worst-kink #{slot+1}: "
                     f"{bname}\n"
                     f"theta=45 rel_kink={rel:.2f}",
                     fontsize=9)
        ax.grid(alpha=0.3)
        if slot == 0:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        f"GT quality inspection  |  n={n} sims  |  "
        f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}",
        fontsize=12)
    provenance_footer(fig, tag=tag, extras={"n_check": n})
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", required=True,
                    help="folder of converted 3D NPZ files")
    ap.add_argument("--n-check", type=int, default=20,
                    help="sample this many random sims from the folder "
                    "(default 20; ~2-3 min at default workers)")
    ap.add_argument("--sim", default=None,
                    help="comma list of specific basenames to inspect; "
                    "overrides --n-check + --seed")
    ap.add_argument("--seed", type=int, default=0,
                    help="rng seed for the random pick (default 0 so "
                    "re-runs are stable)")
    ap.add_argument("--out", required=True,
                    help="output directory for summary.json + "
                    "gt_quality.png")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None,
                    help="loader workers (default: auto, capped at 32)")
    ap.add_argument("--tag", default="",
                    help="optional label for the report suptitle")
    args = ap.parse_args()

    folder = Path(args.npz_dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"--npz-dir not a directory: {folder}", file=sys.stderr)
        return 2
    files = _pick_files(folder, args.n_check, args.sim, args.seed)
    print(f"selected {len(files)} sim(s) from {folder}", flush=True)

    # A: Raw tReal stats for ALL selected sims (fast, no canonicalize).
    print("A. reading raw tReal for each selected sim...", flush=True)
    raw_stats = {}
    for f in files:
        try:
            raw_stats[f.name] = _raw_treal_stats(f)
        except Exception as e:                     # noqa: BLE001
            print(f"  {f.name}: raw tReal read failed "
                  f"({type(e).__name__}: {e})",
                  file=sys.stderr)
            raw_stats[f.name] = dict(n_samples=0, n_backward=-1,
                                       max_backward=0.0,
                                       max_backward_over_median_dt=0.0,
                                       median_dt=0.0)

    # B-D: load via current loader (rebound-fix + nearest-fill).
    print(f"B-D. loading {len(files)} sim(s) via current loader "
          f"(tempdir + symlinks; will not persist)...", flush=True)
    t0 = time.time()
    td, (x_canon, y_canon, sims) = _load_via_tempdir(
        files, args.nx, args.ny, args.nt, args.drop_first_steps,
        args.workers)
    print(f"  loaded {len(sims)} sim(s) in "
          f"{time.time() - t0:.1f}s", flush=True)

    X, Y = np.meshgrid(x_canon, y_canon, indexing="ij")
    in_disk = (X * X + Y * Y) <= 1.0

    all_stats = []
    per_sim_final_curves = []
    for sim in sims:
        bname = sim.params.get("basename",
                                f"unknown_{len(all_stats)}")
        raw = raw_stats.get(bname, dict(n_samples=0, n_backward=0,
                                          max_backward=0.0,
                                          max_backward_over_median_dt=0.0,
                                          median_dt=0.0))
        f = np.asarray(sim.f, dtype=np.float32)
        temporal = _temporal_monotonicity(f, in_disk)
        radial_kink = _radial_kink_stats(f, x_canon, y_canon)
        dead = _dead_cell_stats(f, in_disk)
        stats = dict(basename=bname, raw_treal=raw,
                      temporal=temporal, radial_kink=radial_kink,
                      dead_cells=dead)
        stats["verdict"] = _verdict(stats)
        all_stats.append(stats)

        # Store final-frame radial curves for the report figure.
        curves = {}
        n_r = 512
        for th in (0.0, 45.0, 90.0):
            km = _sample_radial_kymograph(
                f.astype(np.float64), x_canon, y_canon, th, n_r=n_r)
            curves[f"theta={th:g}"] = km[:, -1]
        per_sim_final_curves.append(curves)

    # Cleanup the tempdir.
    import shutil
    shutil.rmtree(td, ignore_errors=True)

    # Aggregate + write outputs.
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        folder=str(folder), tag=args.tag,
        n_checked=len(all_stats),
        thresholds=dict(
            treal_backward_warn=_TREAL_BACKWARD_WARN,
            temporal_rise_rel_warn=_TEMPORAL_RISE_REL_WARN,
            temporal_rise_rel_fail=_TEMPORAL_RISE_REL_FAIL,
            rise_count_rel_threshold=_RISE_COUNT_REL_THRESHOLD,
            kink_rel_warn=_KINK_REL_WARN,
            kink_rel_fail=_KINK_REL_FAIL,
            dead_cell_warn=_DEAD_CELL_WARN,
            dead_cell_fail=_DEAD_CELL_FAIL),
        aggregate=dict(
            n_pass=sum(1 for s in all_stats if s["verdict"] == "PASS"),
            n_warn=sum(1 for s in all_stats if s["verdict"] == "WARN"),
            n_fail=sum(1 for s in all_stats if s["verdict"] == "FAIL"),
            max_temporal_rise_across_all=max(
                s["temporal"]["max_rise"] for s in all_stats),
            max_rise_rel_across_all=max(
                s["temporal"].get("max_rise_rel", 0.0)
                for s in all_stats),
            max_kink_45_across_all=max(
                s["radial_kink"]["theta=45"]["rel_kink"]
                for s in all_stats),
            sims_with_any_raw_backward=sum(
                1 for s in all_stats
                if s["raw_treal"]["n_backward"] > 0)),
        per_sim=all_stats)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_dir}/summary.json", flush=True)

    _render_report(all_stats, per_sim_final_curves,
                    out_dir / "gt_quality.png", tag=args.tag)
    print(f"wrote {out_dir}/gt_quality.png", flush=True)

    agg = summary["aggregate"]
    print(f"\nverdict summary ({len(all_stats)} sims):")
    print(f"  PASS: {agg['n_pass']}")
    print(f"  WARN: {agg['n_warn']}")
    print(f"  FAIL: {agg['n_fail']}")
    print(f"  raw tReal backward jumps in: "
          f"{agg['sims_with_any_raw_backward']} sim(s)")
    print(f"  max loaded temporal rise: "
          f"{agg['max_temporal_rise_across_all']:.3e} m "
          f"({100 * agg.get('max_rise_rel_across_all', 0):.2f}% of "
          f"peak descent)")
    print(f"  max relative kink at theta=45: "
          f"{agg['max_kink_45_across_all']:.3f}")
    if agg["n_fail"] > 0:
        print(f"\nnot yet safe to retrain: {agg['n_fail']} sim(s) FAIL",
              file=sys.stderr)
        return 1
    if agg["n_warn"] > 0:
        print(f"\nretraining OK but {agg['n_warn']} sim(s) WARN -- "
              f"review the report PNG before committing to a long run",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
