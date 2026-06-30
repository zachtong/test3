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
    # Per docs/NPZ_SCHEMA.md section 2.3 (file-level metadata).
    "num_samples", "num_wafer_steps", "num_original_wafer_steps",
    "num_valid_wafer_steps", "last_step_removed", "skipped_step_count",
    "step_metadata_json", "skipped_steps_json",
    "source_json", "source_json_name", "json_file_size_bytes",
    "converter_version", "minimal_fields", "repaired_or_not",
    "invalid_json_policy", "complete_json_required",
    "coordinate_system", "coordinate_layout", "wafer_split_mode",
    "z_correction_mode", "z_correction_formula",
    "array_float_dtype", "time_dtype",
    # Per docs/NPZ_SCHEMA.md section 2.4 (COMSOL physical metadata).
    "contactTime", "releaseTime_LW", "releaseTime_UW", "hGap",
    "modelName", "allParams_json", "expr",
}

# Per docs/NPZ_SCHEMA.md section 2.2. Documenting the sample-level keys in
# their OWN set means `_classify_key` can flag any unknown sample_* name
# (e.g. a typo like `sample_step_idx`) as `sample_unknown` instead of
# silently accepting it.
_DOCUMENTED_SAMPLE_KEYS = {
    "sample_step_index", "sample_time_index_within_step",
    "sample_tReal", "sample_bonding_front",
    "sample_num_time_points_in_step", "sample_num_points",
    "sample_num_lower_points", "sample_num_upper_points",
    "sample_z_min", "sample_z_max",
}

# Per docs/NPZ_SCHEMA.md section 2.1, split by whether the suffix names
# an array (variable shape, gets per-step shape validation against the
# scalar counts) or a scalar (0-d).
_STEP_ARRAY_SUFFIXES = {
    "coordinates_lower", "coordinates_upper",
    "displacement_z_corrected_lower", "displacement_z_corrected_upper",
    "thickness_lower", "thickness_upper",
    "bonding_front", "tReal",
}
_STEP_SCALAR_SUFFIXES = {
    "num_time_points", "num_points", "num_lower_points",
    "num_upper_points", "z_min", "z_max",
}
_STEP_SUFFIXES = _STEP_ARRAY_SUFFIXES | _STEP_SCALAR_SUFFIXES

# Documented string-typed scalars: the converter author should emit these
# as numpy <U... so the loader can use allow_pickle=False. Object dtype
# loads, but it ties future loads to allow_pickle=True. Inspector warns,
# does not error.
_DOCUMENTED_STRING_KEYS = {
    "source_json", "source_json_name", "converter_version", "modelName",
    "allParams_json", "step_metadata_json", "skipped_steps_json",
    "invalid_json_policy", "coordinate_system", "coordinate_layout",
    "wafer_split_mode", "z_correction_mode", "z_correction_formula",
    "array_float_dtype", "time_dtype",
    "expr",  # this one is shape (7,), not scalar, but same dtype concern
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

    Returns one of: "file", "sample", "sample_unknown", "step_known",
    "step_unknown", "unknown". The "sample" vs "sample_unknown" split
    is important: a malformed key like `sample_step_idx` (typo) must
    surface as unknown, not get a free pass on being prefixed
    `sample_`.
    """
    if key in _DOCUMENTED_FILE_KEYS:
        return "file"
    if key in _DOCUMENTED_SAMPLE_KEYS:
        return "sample"
    if key.startswith("sample_"):
        return "sample_unknown"
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
    # allow_pickle stays True because the current converter emits object
    # arrays for documented string fields; the inspector warns about that
    # separately rather than refusing to load.
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)
        buckets = {"file": [], "sample": [], "sample_unknown": [],
                   "step_known": [], "step_unknown": [], "unknown": []}
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
        n_valid = _scalar(z, "num_valid_wafer_steps")
        n_original = _scalar(z, "num_original_wafer_steps")
        n_skipped = _scalar(z, "skipped_step_count")
        last_removed = _scalar(z, "last_step_removed")
        print(f"  (showing step_0000_* only; saw {n_steps_seen} step prefixes "
              f"in this NPZ)")
        print(f"  num_samples = {n_samples}")
        print(f"  num_wafer_steps = {n_wafer_steps}  "
              f"num_valid_wafer_steps = {n_valid}  "
              f"num_original_wafer_steps = {n_original}  "
              f"last_step_removed = {last_removed}  "
              f"skipped_step_count = {n_skipped}")
        unknowns = (buckets["unknown"] + buckets["sample_unknown"]
                    + buckets["step_unknown"])
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
                    n_valid=n_valid, n_original=n_original,
                    n_skipped=n_skipped, last_removed=last_removed,
                    n_step_prefixes=n_steps_seen, unknowns=unknowns,
                    missing=missing)


def _validate_sample_shapes(z, S, errors: list[str], tag: str) -> None:
    """Every documented sample_* key must exist and have shape (S,).

    The loader treats global sample indexing as the source of truth; a
    mis-sized sample array makes every downstream step subtly wrong (e.g.
    sample_tReal[k] for k beyond the array bounds wraps or crashes).
    Catching this here is cheap and saves hours of debugging later.
    """
    for k in _DOCUMENTED_SAMPLE_KEYS:
        if k not in z.files:
            errors.append(f"{tag}: missing required sample-level key: {k}")
            continue
        arr = z[k]
        if arr.shape != (S,):
            errors.append(f"{tag}: {k}.shape = {arr.shape}, expected ({S},)")


def _validate_num_samples_consistency(z, S, n_wafer_steps,
                                      errors: list[str], tag: str) -> None:
    """num_samples must equal sum_i T_i over CONVERTED steps.

    Catches the mismatch between the step-wise arrays and the global
    sample-index arrays -- one of the strongest invariants in the schema.
    """
    total = 0
    for i in range(n_wafer_steps):
        k = f"step_{i:04d}_num_time_points"
        if k in z.files:
            total += int(z[k])
        else:
            errors.append(f"{tag}: missing {k}")
            return
    if total != S:
        errors.append(
            f"{tag}: num_samples ({S}) != sum_i step_{{i}}_num_time_points "
            f"({total}); step-wise and sample-wise indexing are out of sync")


def _validate_step_shapes(z, n_wafer_steps,
                          errors: list[str], tag: str) -> None:
    """Per-step: every array's shape must match the step's scalar counts."""
    for i in range(n_wafer_steps):
        prefix = f"step_{i:04d}"
        t_key = f"{prefix}_num_time_points"
        nu_key = f"{prefix}_num_upper_points"
        nl_key = f"{prefix}_num_lower_points"
        missing_scalars = [k for k in (t_key, nu_key, nl_key)
                           if k not in z.files]
        if missing_scalars:
            errors.append(f"{tag}: {prefix}: missing scalar(s): "
                          f"{missing_scalars}")
            continue
        Ti = int(z[t_key])
        N_up = int(z[nu_key])
        N_lo = int(z[nl_key])
        expected = {
            "coordinates_lower": (3, N_lo),
            "coordinates_upper": (3, N_up),
            "displacement_z_corrected_lower": (Ti, N_lo),
            "displacement_z_corrected_upper": (Ti, N_up),
            "thickness_lower": (Ti, N_lo),
            "thickness_upper": (Ti, N_up),
            "bonding_front": (Ti,),
            "tReal": (Ti,),
        }
        bad = []
        for suffix, exp_shape in expected.items():
            k = f"{prefix}_{suffix}"
            if k not in z.files:
                bad.append(f"missing {suffix}")
                continue
            if z[k].shape != exp_shape:
                bad.append(f"{suffix}.shape = {z[k].shape}, expected "
                           f"{exp_shape}")
        if bad:
            # Roll up per-step problems into one line per step so the
            # output stays readable when multiple steps are bad.
            errors.append(f"{tag}: {prefix}: " + "; ".join(bad))


def _validate_object_dtype(z, warnings: list[str], tag: str) -> None:
    """String-typed scalars should be numpy <U... not object.

    Object dtype loads only with allow_pickle=True; switching to <U... lets
    every downstream consumer (loader, scripts, eval) use the safer
    allow_pickle=False. This is a warning, not an error, because the
    inspector can still read object arrays.
    """
    offenders = []
    for k in _DOCUMENTED_STRING_KEYS:
        if k in z.files and z[k].dtype == object:
            offenders.append(k)
    if offenders:
        warnings.append(
            f"{tag}: {len(offenders)} string key(s) saved as dtype=object; "
            f"converter should switch to numpy <U... so loaders can use "
            f"allow_pickle=False. Offenders: "
            f"{offenders[:5]}{'...' if len(offenders) > 5 else ''}")


def validate_invariants(path: Path, summary: dict
                        ) -> tuple[list[str], list[str]]:
    """Run every project-level invariant; return (errors, warnings).

    Errors block the cache build; warnings are converter-hygiene
    suggestions that the loader can still tolerate.
    """
    tag = path.name
    errors: list[str] = []
    warnings: list[str] = []

    # --- trim-last-step + step-count consistency ---
    if summary["last_removed"] is not True:
        errors.append(f"{tag}: last_step_removed = "
                      f"{summary['last_removed']!r}, expected True "
                      f"under the current converter")
    if (summary["n_original"] is not None
            and summary["n_wafer_steps"] is not None):
        if summary["n_original"] != summary["n_wafer_steps"] + 1:
            errors.append(
                f"{tag}: num_original_wafer_steps "
                f"({summary['n_original']}) != num_wafer_steps + 1 "
                f"({summary['n_wafer_steps']} + 1); converter may not be "
                f"dropping exactly one trailing step")

    # No-skip policy: skipped_step_count must be 0 and
    # num_valid_wafer_steps must equal num_wafer_steps.
    if summary["n_skipped"] is not None and summary["n_skipped"] != 0:
        errors.append(
            f"{tag}: skipped_step_count = {summary['n_skipped']}, "
            f"expected 0 under the current no-skip policy (any internal "
            f"step that fails to convert is a data-quality issue)")
    if (summary["n_valid"] is not None
            and summary["n_wafer_steps"] is not None
            and summary["n_valid"] != summary["n_wafer_steps"]):
        errors.append(
            f"{tag}: num_valid_wafer_steps ({summary['n_valid']}) != "
            f"num_wafer_steps ({summary['n_wafer_steps']}); some internal "
            f"steps were skipped during conversion")

    # On-disk step prefix count vs the authoritative count.
    expected_prefix_count = summary["n_wafer_steps"]
    if (summary["n_step_prefixes"] is not None
            and expected_prefix_count is not None):
        if summary["n_step_prefixes"] != expected_prefix_count:
            errors.append(
                f"{tag}: counted {summary['n_step_prefixes']} step_*_tReal "
                f"prefixes on disk but num_wafer_steps = "
                f"{expected_prefix_count}")

    # --- shape + count consistency + dtype hygiene ---
    with np.load(path, allow_pickle=True) as z:
        _validate_object_dtype(z, warnings, tag)
        if summary["n_samples"] is not None:
            _validate_sample_shapes(z, summary["n_samples"], errors, tag)
            if summary["n_wafer_steps"] is not None:
                _validate_num_samples_consistency(
                    z, summary["n_samples"], summary["n_wafer_steps"],
                    errors, tag)
        if summary["n_wafer_steps"] is not None:
            _validate_step_shapes(z, summary["n_wafer_steps"], errors, tag)

        # --- quarter-disk: scan every step's upper coords ---
        for k in z.files:
            if not (k.startswith("step_") and
                    k.endswith("_coordinates_upper")):
                continue
            coords = z[k]
            x_min, y_min = float(coords[0].min()), float(coords[1].min())
            if x_min < -_QUARTER_TOL or y_min < -_QUARTER_TOL:
                errors.append(
                    f"{tag} {k}: native coords leak outside the first "
                    f"quadrant; min(x)={x_min:g}, min(y)={y_min:g}")
                break  # one report per file is enough

    return errors, warnings


def render_sim_panel(sim_f: np.ndarray, x_canon: np.ndarray,
                     y_canon: np.ndarray, sensor_xy: np.ndarray,
                     title: str, out_path: Path,
                     value_scale: float = 1.0e6) -> None:
    """Three-snapshot (t = first / middle / last) heatmap with sensors.

    Per-frame colour scale: each subplot gets its own vmin/vmax so a
    big late-time peak does not visually squash the early/mid frames
    into uniform black. The cost is that the three panels are no
    longer directly comparable in absolute amplitude -- the per-frame
    title prints the magnitude range so the reader still sees the
    relative scales. Each subplot also gets its own colourbar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nt = sim_f.shape[2]
    t_idx = [0, nt // 2, nt - 1]
    ext = [x_canon[0], x_canon[-1], y_canon[0], y_canon[-1]]
    F = sim_f * value_scale
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0),
                             constrained_layout=True)
    for ax, k in zip(axes, t_idx):
        slice_ = F[..., k].T
        finite = slice_[np.isfinite(slice_)]
        if finite.size == 0 or finite.max() == finite.min():
            vmin_k, vmax_k = -1.0, 1.0
        else:
            # 1st / 99th percentile clipping keeps a single off-disk
            # outlier or zero-mask cell from owning the colour scale.
            vmin_k = float(np.percentile(finite, 1))
            vmax_k = float(np.percentile(finite, 99))
            if vmin_k == vmax_k:
                vmin_k, vmax_k = finite.min(), finite.max()
        im = ax.imshow(slice_, origin="lower", aspect="equal",
                       extent=ext, vmin=vmin_k, vmax=vmax_k, cmap="viridis")
        amp = float(np.abs(finite).max()) if finite.size else 0.0
        ax.set_title(f"t-idx {k}/{nt - 1}  |peak|={amp:.2e}")
        ax.set_xlabel("x")
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=24,
                   marker="x", c="red", linewidths=1.5)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("y")
    fig.suptitle(f"{title}  (per-frame colour scale; "
                 f"u_z * {value_scale:g})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz_dir", help="folder of converted 3D *.npz files")
    ap.add_argument("--n", type=int, default=2,
                    help="how many GOOD NPZs (those that pass preflight) to "
                    "fully load and visualize (default: 2). The schema scan "
                    "covers EVERY NPZ in the folder regardless of this flag.")
    ap.add_argument("--schema-verbose", type=int, default=2,
                    help="print the full key/shape/dtype dump for the first "
                    "N files (default: 2). Beyond that, each file gets a "
                    "single OK/BAD line.")
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
    ap.add_argument("--drop-first-steps", type=int, default=0,
                    help="match the loader's drop_first_steps for Mode 2: "
                    "discard the first N waferData steps' samples before "
                    "canonicalize so the snapshot reflects what training "
                    "will actually see. Default 0 = no trim.")
    ap.add_argument("--scan-workers", type=int, default=16,
                    help="parallel worker count for Mode 1's compact-path "
                    "preflight scan. Default 16. Set to 1 to disable "
                    "parallelism, e.g. for debugging.")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="cap Mode 1's scan to the first N files in the "
                    "folder. Default 0 = scan all. Useful when the folder "
                    "is huge (5000+ files) and you just want a quick "
                    "preview before committing to the full scan.")
    args = ap.parse_args()

    folder = Path(args.npz_dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 2
    files = _list_npz(folder)
    if not files:
        print(f"error: no NPZ files in {folder}", file=sys.stderr)
        return 2
    print(f"folder: {folder}  ({len(files)} NPZ files)")

    # === MODE 1: schema scan ALL files ===
    # The schema scan visits every file in the folder so we can report
    # the full set of bad ones, then continues to the loader test. This
    # is the policy change from prior versions: bad files don't abort
    # the inspector. On large folders the scan is parallelised across
    # processes (preflight is pure I/O + metadata; pickles fine), and
    # OK lines are suppressed to keep terminal output tractable -- only
    # BAD lines + a progress tick every 5s reach the screen.
    scan_files = files if args.max_scan <= 0 else files[: args.max_scan]
    print(f"\n=== MODE 1: schema scan ({len(scan_files)} file(s) of "
          f"{len(files)} total{', --max-scan capped' if args.max_scan > 0 and len(files) > args.max_scan else ''}) ===")
    from data.loader import preflight_npz                       # noqa: E402
    bad_files: list[tuple[Path, list[str]]] = []
    warnings_count = 0
    good_set: list[Path] = []     # files that pass preflight, kept for MODE 2

    # Verbose pass: serial, first --schema-verbose files. The detailed
    # dump only makes sense in order and on a handful of files.
    verbose_n = min(args.schema_verbose, len(scan_files))
    for i, p in enumerate(scan_files[:verbose_n]):
        ok, reason = preflight_npz(p)
        summary = report_schema(p)
        errs, warns = validate_invariants(p, summary)
        for w in warns:
            print(f"  WARN:  {w}")
        for e in errs:
            print(f"  ERROR: {e}")
        warnings_count += len(warns)
        reasons = list(errs)
        if not ok and not any(reason and reason in e for e in reasons):
            reasons.append(f"preflight: {reason}")
            print(f"  ERROR: preflight: {reason}")
        if reasons:
            bad_files.append((p, reasons))
        if ok:
            good_set.append(p)

    # Compact pass: parallel preflight over the remaining files.
    remaining = scan_files[verbose_n:]
    if remaining:
        workers = max(1, args.scan_workers)
        workers = min(workers, len(remaining))
        print(f"  parallel scan: {len(remaining)} file(s), "
              f"{workers} worker(s); only BAD files are printed.",
              flush=True)
        t0_scan = time.time()
        last_tick = [0.0]

        def _tick(done: int, n_total: int, n_bad: int) -> None:
            now = time.time()
            if now - last_tick[0] >= 5.0 or done == n_total:
                rate = done / max(now - t0_scan, 1e-9)
                eta_min = (n_total - done) / max(rate, 1e-9) / 60.0
                print(f"  scan: {done}/{n_total} "
                      f"({100.0 * done / n_total:.1f}%)  "
                      f"{rate:.0f}/s  ETA {eta_min:.1f} min  "
                      f"(bad so far: {n_bad})", flush=True)
                last_tick[0] = now

        from concurrent.futures import ProcessPoolExecutor      # noqa: E402
        if workers <= 1:
            for i, p in enumerate(remaining, 1):
                ok, reason = preflight_npz(p)
                if ok:
                    good_set.append(p)
                else:
                    bad_files.append((p, [reason or "preflight failed"]))
                    print(f"  BAD  {p.name}: {reason}", flush=True)
                _tick(i, len(remaining), len(bad_files))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                # ex.map preserves input order so we can pair with `remaining`.
                results = ex.map(preflight_npz, remaining, chunksize=16)
                for i, (p, (ok, reason)) in enumerate(zip(remaining, results), 1):
                    if ok:
                        good_set.append(p)
                    else:
                        bad_files.append((p, [reason or "preflight failed"]))
                        print(f"  BAD  {p.name}: {reason}", flush=True)
                    _tick(i, len(remaining), len(bad_files))

    print(f"\n=== schema scan summary ===")
    print(f"  files scanned: {len(scan_files)} (of {len(files)} in folder)")
    print(f"  pass preflight: {len(good_set)}")
    print(f"  fail preflight: {len(bad_files)}")
    if bad_files:
        from collections import Counter
        c = Counter()
        for _, errs in bad_files:
            head = errs[0].split(";")[0].split(",")[0][:80]
            c[head] += 1
        print("  bad-file reason breakdown:")
        for head, count in c.most_common():
            print(f"    {count:5d} x  {head}")
    if warnings_count:
        print(f"  hygiene warnings (verbose scan only): {warnings_count}")

    # === MODE 2: loader test on first --n GOOD files ===
    if not good_set:
        print("\n=== MODE 2: SKIPPED (no NPZ passed preflight) ===")
        print("\nVERDICT: 0 valid NPZ in folder. The loader cannot proceed; "
              "investigate the converter run.")
        return 1

    pick = good_set[: args.n]
    print(f"\n=== MODE 2: loader test on {len(pick)} good NPZ(s) "
          f"(nx={args.nx}, ny={args.ny}, nt={args.nt}, "
          f"drop_first_steps={args.drop_first_steps}) ===")
    for p in pick:
        print(f"  selected: {p.name}")
    from data.loader import _build_one                           # noqa: E402
    from core.grid import canonical_grid                         # noqa: E402
    from core.simulation import Simulation                       # noqa: E402
    x_canon, y_canon = canonical_grid(args.nx, args.ny)
    t_canon = np.linspace(0.0, 1.0, args.nt)
    sims = []
    t0 = time.time()
    try:
        for p in pick:
            f, params = _build_one((str(p), x_canon, y_canon, t_canon,
                                    args.drop_first_steps))
            params["basename"] = p.name
            sims.append(Simulation(f=f, params=params))
    except Exception as e:
        print(f"LOADER FAILED on {p.name}: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    per_sim = dt / max(len(pick), 1)
    full_min = per_sim * args.total_sims / 60.0
    full_min_parallel = full_min / 16.0
    print(f"  loaded {len(sims)} sim(s) in {dt:.2f}s  "
          f"({per_sim:.2f} s/sim, single-threaded)")
    print(f"  extrapolation for {args.total_sims} sims: "
          f"~{full_min:.0f} min single-threaded, "
          f"~{full_min_parallel:.0f} min with 16 workers")

    # === first-sim sanity ===
    s0 = sims[0]
    print(f"\n=== first sim ({s0.params.get('basename', '?')}) sanity ===")
    print(f"  f.shape = {s0.f.shape}  dtype = {s0.f.dtype}")
    print(f"  f range: [{s0.f.min():.3e}, {s0.f.max():.3e}]  "
          f"mean = {s0.f.mean():.3e}")
    X, Y = np.meshgrid(x_canon, y_canon, indexing="ij")
    off_disk = (X * X + Y * Y) > 1.0
    n_off = int(off_disk.sum())
    n_total = off_disk.size
    print(f"  off-disk cells: {n_off}/{n_total} "
          f"({100.0 * n_off / n_total:.1f}%); "
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

    # === optional PNG panels ===
    if not args.no_figures:
        from core.sensors import SensorConfig, place_sensors    # noqa: E402
        scfg = SensorConfig(n=3, strategy="custom",
                            positions=((1.0, 0.0), (1.0, 45.0), (1.0, 90.0)))
        xy = place_sensors(scfg)
        out_root = (Path(args.out_dir) if args.out_dir
                    else folder.parent / "inspect_out")
        for s in sims:
            stem = Path(s.params.get("basename", "sim")).stem
            out_path = out_root / f"{stem}_snapshots.png"
            render_sim_panel(s.f, x_canon, y_canon, xy,
                             title=stem, out_path=out_path)
        print(f"\n=== rendered {len(sims)} snapshot PNG(s) -> {out_root} ===")

    # === verdict ===
    if bad_files:
        print(f"\nVERDICT: {len(good_set)}/{len(files)} files pass "
              f"preflight; {len(bad_files)} will be skipped by the "
              f"loader (see breakdown above). Cache build will proceed "
              f"on the {len(good_set)} good file(s).")
        return 0 if good_set else 1
    print(f"\nVERDICT: all {len(files)} NPZ pass preflight; safe to "
          f"launch the full cache build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
