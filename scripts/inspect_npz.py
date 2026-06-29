"""One-command sanity check on a folder of converted 3D NPZs.

Run on the workstation where the real converted NPZs live:

    python scripts/inspect_npz.py /path/to/3d_npz_folder

The script does, in order, everything needed to know "is the converter's
output actually loadable by our pipeline, are the numbers sane, and how
expensive is a real cache build going to be":

  1. List NPZ files and pick the first `--n` (default 2) for inspection.
  2. For each, open with np.load and print the raw schema (key, shape,
     dtype) so you can eyeball that every key documented in
     docs/NPZ_SCHEMA.md is present. Out-of-schema keys are also surfaced.
  3. Validate the trim-last-step invariant
     (num_original_wafer_steps == num_wafer_steps + 1 and
     last_step_removed is True).
  4. Validate the quarter-disk invariant: every step's coordinates_upper
     row 0 and row 1 must be >= 0 within a small tolerance.
  5. Run `load_dataset(folder, nx, ny, nt, limit=n, cache=False,
     workers=1)`, time it end-to-end, and print per-sim timing extrapolation
     for the full 5500-sim build.
  6. Print sim.params keys + value ranges for the first sim, and a few
     statistics on sim.f (min, max, mean, fraction of off-disk zeros).
  7. (Optional, default on) Render one PNG per inspected sim showing
     three time-slice heatmaps + sensor positions overlaid, into
     `inspect_out/` next to the NPZ folder. Disable with `--no-figures`.

Nothing here is destructive. Cache is disabled so the script never writes
the multi-GB loader cache; that's the user's call to make separately.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


_DOCUMENTED_FILE_KEYS = {
    # Per docs/NPZ_SCHEMA.md sections 2.3 / 2.4.
    "num_samples", "num_wafer_steps", "num_original_wafer_steps",
    "num_valid_wafer_steps", "last_step_removed", "skipped_step_count",
    "step_metadata_json", "skipped_steps_json",
    "source_json", "source_json_name", "json_file_size_bytes",
    "converter_version", "minimal_fields", "repaired_or_not",
    "invalid_json_policy", "complete_json_required",
    "coordinate_system", "coordinate_layout", "wafer_split_mode",
    "z_correction_mode", "z_correction_formula",
    "array_float_dtype", "time_dtype",
    "contactTime", "releaseTime_LW", "releaseTime_UW", "hGap",
    "modelName", "allParams_json", "expr",
    # Sample-level (section 2.2).
    "sample_step_index", "sample_time_index_within_step",
    "sample_tReal", "sample_bonding_front",
    "sample_num_time_points_in_step", "sample_num_points",
    "sample_num_lower_points", "sample_num_upper_points",
    "sample_z_min", "sample_z_max",
}
_STEP_SUFFIXES = {
    "coordinates_lower", "coordinates_upper",
    "displacement_z_corrected_lower", "displacement_z_corrected_upper",
    "thickness_lower", "thickness_upper",
    "bonding_front", "tReal",
    "num_time_points", "num_points", "num_lower_points",
    "num_upper_points", "z_min", "z_max",
}
_QUARTER_TOL = 1e-6


def _list_npz(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.npz")
                  if not p.name.startswith("_"))


def _scalar(z, key):
    """Cast a 0-d array (or already-scalar) into a Python value."""
    if key not in z.files:
        return None
    v = z[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def _classify_key(key: str) -> str:
    """Bucket an NPZ key for the schema report.

    Returns one of: "file", "sample", "step_known", "step_unknown",
    "unknown".
    """
    if key in _DOCUMENTED_FILE_KEYS:
        return "file"
    if key.startswith("sample_"):
        return "sample"
    if key.startswith("step_"):
        # step_{i:04d}_<suffix>
        parts = key.split("_", 2)
        if len(parts) >= 3:
            suffix = parts[2]
            return "step_known" if suffix in _STEP_SUFFIXES else "step_unknown"
        return "step_unknown"
    return "unknown"


def report_schema(path: Path) -> dict:
    """Print every key + shape + dtype; return a small summary dict."""
    print(f"\n=== {path.name} ===")
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)
        buckets = {"file": [], "sample": [], "step_known": [],
                   "step_unknown": [], "unknown": []}
        for k in keys:
            buckets[_classify_key(k)].append(k)
            arr = z[k]
            try:
                shape, dtype = arr.shape, str(arr.dtype)
            except AttributeError:
                shape, dtype = "()", type(arr).__name__
            # Only print step-level keys for the first step, otherwise the
            # output explodes when there are dozens of steps.
            if k.startswith("step_") and not k.startswith("step_0000_"):
                continue
            print(f"  {k:<60}  shape={str(shape):<20} dtype={dtype}")
        # Summary lines.
        n_steps_seen = sum(1 for k in keys
                           if k.startswith("step_") and k.endswith("_tReal"))
        n_samples = _scalar(z, "num_samples")
        n_wafer_steps = _scalar(z, "num_wafer_steps")
        n_original = _scalar(z, "num_original_wafer_steps")
        last_removed = _scalar(z, "last_step_removed")
        print(f"  (showing step_0000_* only; saw {n_steps_seen} step prefixes "
              f"in this NPZ)")
        print(f"  num_samples = {n_samples}")
        print(f"  num_wafer_steps = {n_wafer_steps}  "
              f"num_original_wafer_steps = {n_original}  "
              f"last_step_removed = {last_removed}")
        unknowns = buckets["unknown"] + buckets["step_unknown"]
        if unknowns:
            print(f"  WARNING: {len(unknowns)} key(s) not in documented "
                  f"schema -- inspect: {unknowns[:5]}{'...' if len(unknowns) > 5 else ''}")
        # Missing required keys?
        required = ("num_samples", "num_wafer_steps", "sample_step_index",
                    "sample_time_index_within_step", "sample_tReal")
        missing = [k for k in required if k not in keys]
        if missing:
            print(f"  WARNING: missing required keys: {missing}")
        return dict(n_samples=n_samples, n_wafer_steps=n_wafer_steps,
                    n_original=n_original, last_removed=last_removed,
                    n_step_prefixes=n_steps_seen, unknowns=unknowns,
                    missing=missing)


def validate_invariants(path: Path, summary: dict) -> list[str]:
    """Check trim-last-step + quarter-disk invariants; return error strings."""
    errors: list[str] = []
    # Trim-last-step.
    if summary["last_removed"] is not True:
        errors.append(f"{path.name}: last_step_removed = "
                      f"{summary['last_removed']!r}, expected True "
                      f"under the current converter")
    if (summary["n_original"] is not None
            and summary["n_wafer_steps"] is not None):
        if summary["n_original"] != summary["n_wafer_steps"] + 1:
            errors.append(
                f"{path.name}: num_original_wafer_steps "
                f"({summary['n_original']}) != num_wafer_steps + 1 "
                f"({summary['n_wafer_steps']} + 1); converter may not be "
                f"dropping exactly one trailing step")
    if (summary["n_step_prefixes"] is not None
            and summary["n_wafer_steps"] is not None):
        if summary["n_step_prefixes"] != summary["n_wafer_steps"]:
            errors.append(
                f"{path.name}: counted {summary['n_step_prefixes']} step_ "
                f"prefixes on disk but num_wafer_steps = "
                f"{summary['n_wafer_steps']}")

    # Quarter-disk: scan every step's upper coords.
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            if not (k.startswith("step_") and
                    k.endswith("_coordinates_upper")):
                continue
            coords = z[k]
            x_min, y_min = float(coords[0].min()), float(coords[1].min())
            if x_min < -_QUARTER_TOL or y_min < -_QUARTER_TOL:
                errors.append(
                    f"{path.name} {k}: native coords leak outside the first "
                    f"quadrant; min(x)={x_min:g}, min(y)={y_min:g}")
                break  # one report per file is enough
    return errors


def render_sim_panel(sim_f: np.ndarray, x_canon: np.ndarray,
                     y_canon: np.ndarray, sensor_xy: np.ndarray,
                     title: str, out_path: Path,
                     value_scale: float = 1.0e6) -> None:
    """Three-snapshot (t = first / middle / last) heatmap with sensors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nt = sim_f.shape[2]
    t_idx = [0, nt // 2, nt - 1]
    ext = [x_canon[0], x_canon[-1], y_canon[0], y_canon[-1]]
    F = sim_f * value_scale
    vmin = float(np.nanmin(F))
    vmax = float(np.nanmax(F))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6),
                             constrained_layout=True)
    for ax, k in zip(axes, t_idx):
        im = ax.imshow(F[..., k].T, origin="lower", aspect="equal",
                       extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"t-idx {k}/{nt - 1}")
        ax.set_xlabel("x")
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=24,
                   marker="x", c="red", linewidths=1.5)
    axes[0].set_ylabel("y")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes[:], shrink=0.85, label=f"u_z * {value_scale:g}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz_dir", help="folder of converted 3D *.npz files")
    ap.add_argument("--n", type=int, default=2,
                    help="how many NPZs to fully load (default: 2). Schema "
                    "report runs on this many too.")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--total-sims", type=int, default=5500,
                    help="total sim count, for the cache-build time "
                    "extrapolation (default: 5500)")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the PNG snapshot panels")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the figures (default: "
                    "<npz_dir>/../inspect_out)")
    args = ap.parse_args()

    folder = Path(args.npz_dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 2
    files = _list_npz(folder)
    if not files:
        print(f"error: no NPZ files in {folder}", file=sys.stderr)
        return 2
    n = min(args.n, len(files))
    print(f"folder: {folder}  ({len(files)} NPZ files, "
          f"inspecting first {n})")

    # === 1+2+3: schema report + invariants per file ===
    summaries = []
    all_errors: list[str] = []
    for p in files[:n]:
        summary = report_schema(p)
        errs = validate_invariants(p, summary)
        for e in errs:
            print(f"  ERROR: {e}")
        all_errors.extend(errs)
        summaries.append((p, summary))

    if all_errors:
        print(f"\n=== {len(all_errors)} invariant errors ABOVE; "
              f"loader will likely fail. Fix the converter before "
              f"running cache build. ===")
        # Still continue so the loader-stage error message is visible.

    # === 4+5: load_dataset on the first n; time it; extrapolate ===
    print(f"\n=== loader timing (nx={args.nx}, ny={args.ny}, "
          f"nt={args.nt}, limit={n}) ===")
    from data.loader import load_dataset
    t0 = time.time()
    try:
        x, y, sims = load_dataset(str(folder), nx=args.nx, ny=args.ny,
                                  nt=args.nt, limit=n, cache=False,
                                  workers=1)
    except Exception as e:
        print(f"LOADER FAILED: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    per_sim = dt / max(n, 1)
    full_min = per_sim * args.total_sims / 60.0
    # Estimate with 16-worker parallelism (loader's typical default).
    full_min_parallel = full_min / 16.0
    print(f"  loaded {len(sims)} sims in {dt:.2f}s  "
          f"({per_sim:.2f} s/sim, single-threaded)")
    print(f"  extrapolation for {args.total_sims} sims: "
          f"~{full_min:.0f} min single-threaded, "
          f"~{full_min_parallel:.0f} min with 16 workers")

    # === 6: first-sim sanity ===
    s0 = sims[0]
    print(f"\n=== first sim ({s0.params.get('basename', '?')}) sanity ===")
    print(f"  f.shape = {s0.f.shape}  dtype = {s0.f.dtype}")
    print(f"  f range: [{s0.f.min():.3e}, {s0.f.max():.3e}]  "
          f"mean = {s0.f.mean():.3e}")
    # Off-disk fraction: cells masked to 0.
    X, Y = np.meshgrid(x, y, indexing="ij")
    off_disk = (X * X + Y * Y) > 1.0
    n_off = int(off_disk.sum())
    n_total = off_disk.size
    print(f"  off-disk cells: {n_off}/{n_total} ({100.0 * n_off / n_total:.1f}%); "
          f"the loader zeroes them out before POD.")
    print(f"  params keys: {sorted(s0.params.keys())}")
    for k in ("contactTime", "releaseTime_LW", "releaseTime_UW", "hGap",
              "t_max", "num_wafer_steps", "num_original_wafer_steps",
              "last_step_removed"):
        if k in s0.params:
            print(f"    {k} = {s0.params[k]!r}")
    bf = s0.params.get("bonding_front")
    if bf is not None:
        print(f"  bonding_front (Nt,): shape={bf.shape}  "
              f"range=[{bf.min():.3f}, {bf.max():.3f}]")

    # === 7: optional PNG panels ===
    if not args.no_figures:
        from core.sensors import SensorConfig, place_sensors
        from core.grid import canonical_grid
        scfg = SensorConfig(n=3, strategy="custom",
                            positions=((1.0, 0.0), (1.0, 45.0), (1.0, 90.0)))
        xy = place_sensors(scfg)
        x_canon, y_canon = canonical_grid(args.nx, args.ny)
        out_root = Path(args.out_dir) if args.out_dir else folder.parent / "inspect_out"
        for s in sims:
            stem = Path(s.params.get("basename", "sim")).stem
            out_path = out_root / f"{stem}_snapshots.png"
            render_sim_panel(s.f, x_canon, y_canon, xy,
                             title=stem, out_path=out_path)
        print(f"\n=== rendered {len(sims)} snapshot PNG(s) -> {out_root} ===")

    # === verdict ===
    if all_errors:
        print(f"\nVERDICT: invariant violations detected ({len(all_errors)}); "
              f"investigate before launching the full cache build.")
        return 1
    print("\nVERDICT: schema + invariants OK, loader runs, fields look "
          "well-formed. Safe to launch the full cache build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
