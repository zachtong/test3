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

from data.loader import load_dataset                            # noqa: E402
from scripts.viz_radial_kymograph import _sample_radial_kymograph  # noqa: E402
from scripts.fieldviz import (provenance_footer,                # noqa: E402
                               WAFER_CMAP, SENSOR_MARKER_COLOR)


# Thresholds. WARN = suspicious; FAIL = definitely a data problem
# (retraining on this sim would learn artifact).
_TREAL_BACKWARD_WARN = 1     # any raw backward is worth noting
_TEMPORAL_RISE_WARN = 1e-9   # metres; float32 noise floor
_TEMPORAL_RISE_FAIL = 1e-7   # metres; ~0.1 micrometre, well above noise
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
    """Largest positive diff along t at any in-disk cell."""
    diff = np.diff(f, axis=-1)                # (Nx, Ny, Nt-1)
    diff_in_disk = diff[in_disk]              # (M, Nt-1)
    if diff_in_disk.size == 0:
        return dict(max_rise=0.0, n_cells_with_rise=0,
                     frac_cells_with_rise=0.0)
    max_rise = float(diff_in_disk.max())
    per_cell_max = diff_in_disk.max(axis=-1)
    n_bad = int((per_cell_max > _TEMPORAL_RISE_WARN).sum())
    return dict(max_rise=max_rise, n_cells_with_rise=n_bad,
                 frac_cells_with_rise=float(n_bad
                                             / diff_in_disk.shape[0]))


def _radial_kink_stats(f: np.ndarray, x_canon: np.ndarray,
                        y_canon: np.ndarray,
                        angles: tuple = (0.0, 45.0, 90.0),
                        r_query: tuple = (0.85, 0.90, 0.95, 0.99)
                        ) -> dict:
    """At final t, sample u_z along each ray at r_query values.
    Report the (r=0.99 vs r=0.95) relative jump per angle."""
    out = {}
    for th in angles:
        # Reuse _sample_radial_kymograph with a small n_r; we only
        # want to know the final-frame values at specific r's.
        n_r = 512
        km = _sample_radial_kymograph(
            f.astype(np.float64), x_canon, y_canon, th, n_r=n_r)
        final = km[:, -1]
        r_axis = np.linspace(0.0, 1.0, n_r)
        vals = {r: float(np.interp(r, r_axis, final)) for r in r_query}
        # Relative kink between r=0.99 and r=0.95.
        v95 = vals[0.95]; v99 = vals[0.99]
        base = max(abs(v95), abs(v99), 1e-12)
        rel_kink = float(abs(v99 - v95) / base)
        # 'edge_less_descended' = True if r=0.99 is closer to 0 (less
        # descended) than r=0.95 -- matches the operator's 'sharp
        # upward kink' observation.
        edge_less_descended = (abs(v99) < abs(v95))
        out[f"theta={th:g}"] = dict(
            values_at_r=vals, rel_kink=rel_kink,
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
    """Combine per-sim stat dicts into PASS / WARN / FAIL."""
    if stats["temporal"]["max_rise"] > _TEMPORAL_RISE_FAIL:
        return "FAIL"
    if stats["dead_cells"]["frac_dead"] > _DEAD_CELL_FAIL:
        return "FAIL"
    if any(v["rel_kink"] > _KINK_REL_FAIL and v["edge_less_descended"]
           for v in stats["radial_kink"].values()):
        return "FAIL"
    if stats["temporal"]["max_rise"] > _TEMPORAL_RISE_WARN:
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

    # (0,1): temporal rise (loaded)
    ax = axes[0, 1]
    ax.hist(np.log10(np.maximum(max_rises, 1e-12)), bins=20,
             color=SENSOR_MARKER_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("log10 max positive du_z/dt (m)  in-disk")
    ax.set_ylabel("# sims")
    ax.set_title(f"B. Loaded temporal monotonicity  "
                 f"(FAIL >= log10({_TEMPORAL_RISE_FAIL:g}))",
                 fontsize=10)
    ax.axvline(np.log10(_TEMPORAL_RISE_WARN), color="orange",
                ls="--", lw=1, label="WARN")
    ax.axvline(np.log10(_TEMPORAL_RISE_FAIL), color="red",
                ls="--", lw=1, label="FAIL")
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
            temporal_rise_warn=_TEMPORAL_RISE_WARN,
            temporal_rise_fail=_TEMPORAL_RISE_FAIL,
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
          f"{agg['max_temporal_rise_across_all']:.3e} m")
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
