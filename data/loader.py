"""Load converted 3D NPZ simulations onto the canonical Cartesian grid.

Per-NPZ layout: step-wise (each waferData step has its own static mesh and
time-dependent fields; the mesh adapts across steps). Per sample (= one time
point of one step) the loader takes the upper-wafer (x, y) point cloud and
the corresponding `displacement_z_corrected_upper` slice, interpolates onto
the canonical quarter-disk Cartesian grid, then time-resamples the stack
onto a uniform time axis.

See docs/NPZ_SCHEMA.md for the full schema and field policy.

Skip-tolerant policy (matters for production!): the converter is too
expensive to re-run, so the dataset folder is allowed to contain a few
"bad" NPZs -- files that are structurally valid zips but internally
broken (e.g. skipped_step_count > 0, on-disk step prefix count <
num_wafer_steps, step array shape != metadata, etc.). The loader runs
`preflight_npz` on each file and SKIPS bad ones with a logged reason
rather than aborting the whole dataset. Only when zero sims load
successfully does load_dataset raise.

Pipeline per sim:
  1. preflight_npz: walk file-level / sample-level / per-step invariants
     and quarter-disk. Fast (no large-array decompression).
  2. Open NPZ, read sample index arrays + tReal + bonding_front + metadata.
  3. Group sample indices by step_idx.
  4. For each unique step:
     a. Read coordinates_upper[:2, :].T -- the native (x, y) point cloud.
     b. Build one scipy.spatial.Delaunay over those points and precompute
        the barycentric weights of every canonical grid query point.
     c. For each sample in the step, gather displacement values at the
        three vertices, dot with the weights, mask off-hull (-> 0), and
        reshape to (Nx, Ny). The Delaunay + weights are paid once per
        step, not once per sample.
  5. Stack per-sample (Nx, Ny) into f_stack (S, Nx, Ny).
  6. Validate tReal monotonicity (allow exact duplicates at step boundaries
     but abort on a backward jump > one median dt; same rule as 2D).
  7. Normalize sample times to [0, 1] and linearly resample each pixel's
     time trace onto t_canon (Nt,).
  8. Return f (Nx, Ny, Nt) float32 + a params dict.

Cache layer: parallel build, atomic write, disk-full resilience, stale
.tmp sweep, CRC-free bulk read. Cache records BOTH the loaded set and
the skipped set (with reasons); cache-hit check compares
(loaded U skipped) against the current folder file set so the cache is
self-consistent across re-runs that see the same skip outcome.
"""

from __future__ import annotations
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from core.simulation import Simulation
from core.grid import canonical_grid, disk_mask

_CACHE_PREFIX = "_loader_cache_"
_PARALLEL_MIN = 32
_QUARTER_TOL = 1e-6   # native x/y allowed slightly negative due to float roundoff

# Maximum tolerated backward jump in sample_tReal, expressed as a multiple
# of the median forward dt. Step boundaries can overlap by several
# sub-step samples in practice (a single physical-time instant gets
# written twice by adjacent waferData segments) -- a strict "one
# timestep" tolerance was rejecting otherwise-good NPZs. Larger values
# than the factor below get logged as a preflight failure with the
# observed magnitude so the operator can decide whether the file is
# salvageable.
_TREAL_BACKWARD_FACTOR = 10.0

# Wafer physical radius. The 3D converter writes coordinates_upper /
# _lower in raw physical metres (range 0 .. R), but the canonical grid
# downstream lives in normalised [0, 1]. The loader divides every
# native (x, y) by this constant before feeding it to Delaunay so the
# point cloud and the canonical grid agree on units. Matches the 2D
# pipeline's WAFER_RADIUS_M = 0.15. If you ever switch to a different
# wafer size, change this constant -- preflight checks that
# max(x), max(y) fall within [0.5*R, 1.5*R] and will reject otherwise.
_WAFER_RADIUS_M = 0.15


# Per-sim metadata pulled from file-level / COMSOL-level NPZ keys; copied
# into Simulation.params for downstream provenance and conditional-feature
# experiments. Keys that the NPZ may or may not contain (older / minimal
# conversions) are loaded with a `_get_optional` guard below.
_OPTIONAL_PHYSICAL_KEYS = ("contactTime", "releaseTime_LW",
                           "releaseTime_UW", "hGap")
_OPTIONAL_STRING_KEYS = ("source_json", "source_json_name", "converter_version",
                         "modelName", "z_correction_formula",
                         "step_metadata_json", "skipped_steps_json")
# Trim-last-step bookkeeping: the converter always drops the final
# waferData step before writing. These keys make that recoverable.
_OPTIONAL_INT_KEYS = ("num_original_wafer_steps", "num_valid_wafer_steps",
                      "skipped_step_count")
_OPTIONAL_BOOL_KEYS = ("last_step_removed",)


def _is_compressed_npz(path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return any(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist())
    except zipfile.BadZipFile:
        return False


def _read_npz_member_fast(path, member: str) -> np.ndarray:
    """Bulk-read one UNCOMPRESSED npz member at raw disk speed."""
    import numpy.lib.format as fmt
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{member} is compressed; cannot fast-read")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = fmt.read_magic(f)
        shape, fortran, dtype = fmt._read_array_header(f, version)
        a = np.fromfile(f, dtype=dtype, count=int(np.prod(shape)))
    return a.reshape(shape, order="F" if fortran else "C")


def _list_npz(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.npz")
                  if not p.name.startswith("_"))


# Documented sample-level keys whose shape must equal (num_samples,). The
# loader's sample-indexing route depends on every one of these being the
# correct length, so they are checked together.
_REQUIRED_SAMPLE_KEYS = (
    "sample_step_index", "sample_time_index_within_step",
    "sample_tReal", "sample_bonding_front",
)
# Per-step array suffixes whose shape is checked against the step's
# scalar counts T_i / N_upper_i / N_lower_i.
_STEP_ARRAY_SHAPES = {
    "coordinates_lower":               ("3", "N_lo"),
    "coordinates_upper":               ("3", "N_up"),
    "displacement_z_corrected_lower":  ("T",  "N_lo"),
    "displacement_z_corrected_upper":  ("T",  "N_up"),
    "thickness_lower":                 ("T",  "N_lo"),
    "thickness_upper":                 ("T",  "N_up"),
    "bonding_front":                   ("T",),
    "tReal":                           ("T",),
}


def preflight_npz(path) -> tuple[bool, str | None]:
    """Validate one NPZ against every invariant the loader needs.

    Returns (True, None) if the file is safe to canonicalize, or
    (False, reason) if it must be skipped. Reasons are short, one-line
    strings suitable for a summary log.

    Cheap by design: only the central directory + scalar metadata + array
    shapes are touched (no large-array decompression). Safe to call on a
    folder of 5500 NPZs in seconds.

    Skip semantics: the converter is too expensive to re-run, so files
    that pass `np.load` but are internally inconsistent (e.g.
    skipped_step_count > 0; on-disk step prefix count <
    num_wafer_steps; some step array shape != metadata) are not errors
    against the schema -- they are simulations that did not convert
    cleanly. The loader treats them as bad and proceeds with the rest.

    Invariants checked, in order:
      1. file opens as an NPZ archive at all
      2. required scalar / sample / step keys present
      3. trim-last-step (last_step_removed True;
         num_original_wafer_steps == num_wafer_steps + 1 when present)
      4. no-skip (skipped_step_count == 0 if present;
         num_valid_wafer_steps == num_wafer_steps if present)
      5. on-disk step_*_tReal prefix count == num_wafer_steps
      6. every documented sample_* array has shape (num_samples,)
      7. sample_step_index.max() < num_wafer_steps
      8. per-step: every named array's shape matches its scalar counts
      9. num_samples == sum_i T_i over converted steps
      10. native (x, y) of every upper-coord set lie in the first quadrant
      11. sample_tReal has no backward jump larger than
          `_TREAL_BACKWARD_FACTOR * typ_dt` (typ_dt = median forward dt).
          Step boundaries with a few overlapping sub-step samples are
          fine; only outright concat-order errors should trip this.
    """
    p = Path(path)
    try:
        with np.load(p, allow_pickle=True) as z:
            # (1) -- np.load already validated the zip envelope; the
            # exception path below catches anything that slips past.

            # (2) required keys
            required_scalars = ("num_samples", "num_wafer_steps",
                                "last_step_removed")
            for k in required_scalars + _REQUIRED_SAMPLE_KEYS:
                if k not in z.files:
                    return False, f"missing required key {k!r}"
            S = int(z["num_samples"])
            nws = int(z["num_wafer_steps"])

            # (3) trim-last-step
            if not bool(z["last_step_removed"]):
                return False, "last_step_removed != True"
            if "num_original_wafer_steps" in z.files:
                n_orig = int(z["num_original_wafer_steps"])
                if n_orig != nws + 1:
                    return False, (f"num_original_wafer_steps ({n_orig}) "
                                   f"!= num_wafer_steps + 1 ({nws + 1})")

            # (4) no-skip
            if "skipped_step_count" in z.files:
                n_skip = int(z["skipped_step_count"])
                if n_skip != 0:
                    return False, (f"skipped_step_count = {n_skip} (any "
                                   f"internal step skip means partial "
                                   f"trajectory)")
            if "num_valid_wafer_steps" in z.files:
                n_valid = int(z["num_valid_wafer_steps"])
                if n_valid != nws:
                    return False, (f"num_valid_wafer_steps ({n_valid}) "
                                   f"!= num_wafer_steps ({nws})")

            # (5) on-disk prefix count
            n_prefixes = sum(1 for k in z.files
                             if k.startswith("step_") and
                             k.endswith("_tReal"))
            if n_prefixes != nws:
                return False, (f"on-disk step prefix count ({n_prefixes}) "
                               f"!= num_wafer_steps ({nws})")

            # (6) sample_* shapes
            for k in _REQUIRED_SAMPLE_KEYS:
                if z[k].shape != (S,):
                    return False, (f"{k}.shape = {z[k].shape}, "
                                   f"expected ({S},)")

            # (7) sample_step_index bounded by num_wafer_steps
            if S > 0:
                max_step = int(z["sample_step_index"].max())
                if max_step >= nws:
                    return False, (f"sample_step_index.max() = {max_step} "
                                   f">= num_wafer_steps ({nws})")

            # (8) per-step shapes vs scalar counts
            total_T = 0
            for i in range(nws):
                prefix = f"step_{i:04d}"
                t_key = f"{prefix}_num_time_points"
                nu_key = f"{prefix}_num_upper_points"
                nl_key = f"{prefix}_num_lower_points"
                missing = [k for k in (t_key, nu_key, nl_key)
                           if k not in z.files]
                if missing:
                    return False, f"{prefix}: missing scalar(s) {missing}"
                Ti = int(z[t_key])
                N_up = int(z[nu_key])
                N_lo = int(z[nl_key])
                total_T += Ti
                dims = {"T": Ti, "N_up": N_up, "N_lo": N_lo, "3": 3}
                for suffix, shape_template in _STEP_ARRAY_SHAPES.items():
                    k = f"{prefix}_{suffix}"
                    if k not in z.files:
                        return False, f"{prefix}: missing {suffix}"
                    expected = tuple(dims[s] if s in dims else int(s)
                                     for s in shape_template)
                    if z[k].shape != expected:
                        return False, (f"{k}.shape = {z[k].shape}, "
                                       f"expected {expected}")

            # (9) num_samples consistency
            if total_T != S:
                return False, (f"num_samples ({S}) != sum_i T_i "
                               f"({total_T})")

            # (10) quarter-disk on every upper coord set + native-units
            # sanity check. Native coords must (a) lie in the first
            # quadrant and (b) reach a max in the expected physical
            # range [0.5*R, 1.5*R] -- the loader normalises by
            # _WAFER_RADIUS_M, so coords already in [0, 1] or in a wildly
            # different unit (mm, um, ...) would silently produce a
            # wrong-scale grid.
            r_lo = 0.5 * _WAFER_RADIUS_M
            r_hi = 1.5 * _WAFER_RADIUS_M
            for i in range(nws):
                coords = z[f"step_{i:04d}_coordinates_upper"]
                x_min, y_min = float(coords[0].min()), float(coords[1].min())
                if x_min < -_QUARTER_TOL or y_min < -_QUARTER_TOL:
                    return False, (f"step_{i:04d}_coordinates_upper: native "
                                   f"coords outside first quadrant; "
                                   f"min(x)={x_min:g}, min(y)={y_min:g}")
                x_max, y_max = float(coords[0].max()), float(coords[1].max())
                axis_max = max(x_max, y_max)
                if not (r_lo <= axis_max <= r_hi):
                    return False, (
                        f"step_{i:04d}_coordinates_upper: max(|x|, |y|) = "
                        f"{axis_max:g} outside expected range "
                        f"[{r_lo:g}, {r_hi:g}] m (loader assumes native "
                        f"coords in physical metres with R = "
                        f"{_WAFER_RADIUS_M} m)")

            # (11) sample_tReal monotonicity; small overlaps at step
            # boundaries are tolerated (see _TREAL_BACKWARD_FACTOR comment).
            if S >= 2:
                treal = np.asarray(z["sample_tReal"], dtype=np.float64)
                dt = np.diff(treal)
                forward = dt[dt > 0]
                if forward.size:
                    typ_dt = float(np.median(forward))
                    max_back = float(-dt.min()) if dt.min() < 0 else 0.0
                    limit = _TREAL_BACKWARD_FACTOR * typ_dt
                    if max_back > limit:
                        return False, (
                            f"sample_tReal backward jump {max_back:.4g}s "
                            f">{_TREAL_BACKWARD_FACTOR:g}x typ_dt "
                            f"({typ_dt:.4g}s); concat order is suspect")

            return True, None
    except (zipfile.BadZipFile, OSError, EOFError, ValueError, KeyError) as e:
        return False, f"unreadable: {type(e).__name__}: {e}"


def _interp_rows(x_new: np.ndarray, x_old: np.ndarray,
                 Y: np.ndarray) -> np.ndarray:
    """np.interp on every ROW of Y (n_rows, len(x_old)) onto x_new at once.

    Edge clamp mirrors np.interp default: query points past x_old's range
    take the nearest edge value. x_old MUST be strictly increasing.
    """
    idx = np.searchsorted(x_old, x_new, side="left").clip(1, x_old.size - 1)
    x0, x1 = x_old[idx - 1], x_old[idx]
    w = np.clip((x_new - x0) / (x1 - x0), 0.0, 1.0)
    return Y[:, idx - 1] * (1.0 - w) + Y[:, idx] * w


def _precompute_bary(xy_native: np.ndarray,
                     grid_pts: np.ndarray,
                     in_disk_flat: np.ndarray | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """Delaunay-based barycentric weights + nearest-neighbour edge fill.

    Builds one Delaunay over `xy_native`, locates each query point in the
    grid in a simplex, and ALSO computes a nearest-native-point fallback
    for query points that are in-disk-but-off-hull. This is the loader's
    response to the y-axis kymograph artifact: native point clouds rarely
    sample the wafer edge exactly along x = 0 or y = 0, so the Delaunay
    hull leaves a thin strip of in-disk canonical cells unfilled. Without
    this fallback the loader zeroes them and downstream POD / kymograph /
    overlay all show false zeros along the axes near the bonding front.

    Returns:
        vertices: (Nq, 3) int   indices into the native points (bary tri)
        weights:  (Nq, 3) float barycentric coordinates inside the hull
        inside:   (Nq,)   bool  True if the query is inside the hull
        fill_target: (Nq,) bool True where caller should use the nearest-
                                neighbour fill (in-disk AND off-hull)
        fill_src: (Nq,) int     nearest-native point index for each
                                query (only fill_target rows are used)

    in_disk_flat: optional (Nq,) bool -- True where the canonical grid
        cell is inside the wafer disk. If None, every off-hull cell is
        considered a fill target (no disk masking).
    """
    tri = Delaunay(xy_native)
    simplex = tri.find_simplex(grid_pts)
    inside = simplex >= 0
    # For off-hull points, find_simplex returns -1; index 0 is a safe
    # stand-in so the gather below does not raise; the inside mask
    # zeroes them later (before the fill overwrites them).
    simplex_safe = np.where(inside, simplex, 0)

    # scipy stores per-simplex affine maps in tri.transform with shape
    # (n_simplex, ndim+1, ndim). The first ndim rows of each
    # (ndim+1, ndim) block, multiplied by (q - last_vertex), yield
    # (ndim) barycentric coordinates b1..b_ndim;
    # b_{ndim+1} = 1 - sum(b1..b_ndim).
    T = tri.transform[simplex_safe]                              # (Nq, 3, 2)
    delta = grid_pts - T[:, 2, :]                                # (Nq, 2)
    b_first = np.einsum("ijk,ik->ij", T[:, :2, :], delta)        # (Nq, 2)
    b_last = 1.0 - b_first.sum(axis=1, keepdims=True)
    weights = np.concatenate([b_first, b_last], axis=1)          # (Nq, 3)
    vertices = tri.simplices[simplex_safe]                       # (Nq, 3)

    # Nearest-neighbour fallback for in-disk-but-off-hull cells.
    if in_disk_flat is None:
        fill_target = ~inside
    else:
        fill_target = (~inside) & in_disk_flat
    fill_src = np.zeros(grid_pts.shape[0], dtype=np.int64)
    if fill_target.any():
        kd = cKDTree(xy_native)
        _, nn = kd.query(grid_pts[fill_target], k=1)
        fill_src[fill_target] = nn
    return vertices, weights, inside, fill_target, fill_src


def _interp_to_grid(values: np.ndarray, vertices: np.ndarray,
                    weights: np.ndarray, inside: np.ndarray,
                    fill_target: np.ndarray, fill_src: np.ndarray,
                    nx: int, ny: int) -> np.ndarray:
    """Apply precomputed barycentric weights + nearest-neighbour fill.

    Per cell:
      - inside the convex hull -> linear interp via bary weights
      - off-hull AND in fill_target (i.e. inside the wafer disk) ->
        nearest native point's value
      - off-hull AND not in fill_target -> 0

    values: (N_native,) float per-native displacement at this timestep.
    returns: (nx, ny) float32.
    """
    q = (values[vertices] * weights).sum(axis=1)
    q[~inside] = 0.0
    if fill_target.any():
        q[fill_target] = values[fill_src[fill_target]]
    return q.astype(np.float32).reshape(nx, ny)


def _get_optional(z, key, default=None):
    """Read a (possibly missing) NPZ key. 0-d numpy scalars unwrap to Python."""
    if key not in z.files:
        return default
    val = z[key]
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    return val


def _build_one(args):
    """Load one converted-NPZ sim onto the canonical (Nx, Ny, Nt) grid.

    args is a 5-tuple (path_str, x_canon, y_canon, t_canon,
    drop_first_steps). The last element is the count of leading
    waferData steps to discard at load time. The 3D converter is known
    to keep a pre-contact equilibration step (step_0000) whose samples
    span ~99% of tReal but carry almost no signal (~98% of native
    samples sit in the dense bonding window in the last 5% of time).
    Setting drop_first_steps=1 throws away step_0000's samples before
    canonicalize, recovers a sane native sample density in time, and
    keeps the loader cache key separate from the no-drop case.
    """
    # Accept the legacy 4-tuple (no drop_first_steps) for backward
    # compatibility with callers that haven't been updated yet -- the
    # CLI diagnostic scripts and any external user code may still pass
    # the old shape. Default drop_first_steps=0 keeps the no-trim
    # behaviour.
    if len(args) == 4:
        path_str, x_canon, y_canon, t_canon = args
        drop_first_steps = 0
    else:
        path_str, x_canon, y_canon, t_canon, drop_first_steps = args
    p = Path(path_str)
    nx, ny, nt = x_canon.size, y_canon.size, t_canon.size

    # Canonical-grid query points (Nx*Ny, 2). Built once per sim.
    X, Y = np.meshgrid(x_canon, y_canon, indexing="ij")
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])

    with np.load(p, allow_pickle=True) as z:
        if "num_samples" not in z.files:
            raise ValueError(
                f"{p.name}: missing `num_samples`; not a converted 3D NPZ?")
        S_raw = int(z["num_samples"])
        step_idx_arr = np.asarray(z["sample_step_index"], dtype=np.int64)
        time_idx_arr = np.asarray(z["sample_time_index_within_step"],
                                  dtype=np.int64)
        treal = np.asarray(z["sample_tReal"], dtype=np.float64)
        bf = np.asarray(z["sample_bonding_front"], dtype=np.float32)

        # --- drop-first-steps: drop every sample whose original step
        # index is < drop_first_steps. Leaves the on-disk step_* keys
        # intact (we just never read them); re-numbering the surviving
        # step indices is unnecessary because the per-step Delaunay
        # loop below only iterates over the indices that actually
        # appear in the filtered step_idx_arr.
        if drop_first_steps > 0:
            keep = step_idx_arr >= drop_first_steps
            n_dropped = int((~keep).sum())
            step_idx_arr = step_idx_arr[keep]
            time_idx_arr = time_idx_arr[keep]
            treal = treal[keep]
            bf = bf[keep]
            if step_idx_arr.size == 0:
                raise ValueError(
                    f"{p.name}: drop_first_steps={drop_first_steps} "
                    f"removed every sample (no step >= "
                    f"{drop_first_steps} exists).")
        else:
            n_dropped = 0
        S = step_idx_arr.size

        # Optional file-level metadata; copied into params for provenance.
        params: dict = {"basename": p.name, "num_samples": S,
                        "num_samples_raw": S_raw,
                        "n_samples_dropped_from_first_steps": n_dropped,
                        "drop_first_steps": drop_first_steps}
        for k in _OPTIONAL_PHYSICAL_KEYS:
            v = _get_optional(z, k)
            if v is not None:
                params[k] = float(v)
        for k in _OPTIONAL_STRING_KEYS:
            v = _get_optional(z, k)
            if v is not None:
                params[k] = str(v)
        for k in _OPTIONAL_INT_KEYS:
            v = _get_optional(z, k)
            if v is not None:
                params[k] = int(v)
        for k in _OPTIONAL_BOOL_KEYS:
            v = _get_optional(z, k)
            if v is not None:
                params[k] = bool(v)
        if "num_wafer_steps" in z.files:
            params["num_wafer_steps"] = int(z["num_wafer_steps"])

        # --- disk mask computed once, used both for nearest-fill
        # decision (in_disk_flat) and for the final off-disk zeroing. ---
        mask2d = disk_mask(nx, ny, x_canon[-1], y_canon[-1])
        in_disk_flat = mask2d.ravel()

        # --- spatial: per-step Delaunay, shared across all samples in step ---
        # Pre-allocate at the FILTERED count S, not raw S_raw.
        f_stack = np.zeros((S, nx, ny), dtype=np.float32)
        for step_idx in np.unique(step_idx_arr):
            mask = step_idx_arr == step_idx
            sample_ks = np.where(mask)[0]
            prefix = f"step_{int(step_idx):04d}"
            coords = np.asarray(z[f"{prefix}_coordinates_upper"])    # (3, N_up)
            # Normalise from physical metres into the canonical [0, 1] frame.
            # The converter writes raw COMSOL coordinates (range 0..R, with
            # R = _WAFER_RADIUS_M); the canonical grid is normalised, so
            # this division is what makes Delaunay actually cover the full
            # quarter-disk. Without it the entire wafer mesh gets crammed
            # into the (0..R) corner of the canonical grid and ~99% of the
            # canonical cells fall outside the convex hull -> masked to 0.
            xy_native = (coords[:2, :].T.astype(np.float64)
                         / _WAFER_RADIUS_M)                           # (N_up, 2)

            # Quarter-symmetry validation: every native point must lie in the
            # first quadrant (small float tolerance). preflight already
            # checked this on the RAW coords, but the small-negative
            # tolerance is per-coord so we re-snap here.
            xy_native = np.maximum(xy_native, 0.0)

            vertices, weights, inside, fill_tgt, fill_src = _precompute_bary(
                xy_native, grid_pts, in_disk_flat=in_disk_flat)

            disp_all = np.asarray(
                z[f"{prefix}_displacement_z_corrected_upper"]
            )                                                        # (T_i, N_up)
            for k in sample_ks:
                values = disp_all[int(time_idx_arr[k]), :].astype(np.float64)
                f_stack[k] = _interp_to_grid(values, vertices, weights,
                                             inside, fill_tgt, fill_src,
                                             nx, ny)

    # --- mask off-disk cells (corners of the bounding square) ---
    # Defensive: nearest-neighbour fill only targets in-disk cells, so
    # this is mostly a no-op, but keeps the invariant explicit.
    f_stack[:, ~mask2d] = 0.0

    # --- temporal: dedupe step boundaries + resample to Nt ---
    # preflight_npz already validated length, span > 0, and that the
    # largest backward jump is below _TREAL_BACKWARD_FACTOR * typ_dt.
    # Re-checking span here only guards against the rare standalone call
    # to _build_one bypassing preflight (e.g. inspect_npz's loader test
    # bypasses the safe wrapper).
    if treal.size != S:
        raise ValueError(
            f"{p.name}: sample_tReal length {treal.size} != num_samples {S}")
    span = float(treal.max() - treal.min())
    if span <= 0:
        raise ValueError(f"{p.name}: sample_tReal has zero span.")
    s = (treal - treal.min()) / span                                   # (S,)

    # Collapse exact / sub-step duplicates at step boundaries (same as 2D).
    s_u, keep_t = np.unique(s, return_index=True)
    f_flat = f_stack.reshape(S, nx * ny)[keep_t].T                      # (Nx*Ny, S_u)
    bf_u = bf[keep_t]
    f_canon_flat = _interp_rows(t_canon, s_u, f_flat)                   # (Nx*Ny, Nt)
    f = f_canon_flat.T.reshape(nt, nx, ny).transpose(1, 2, 0)           # (Nx, Ny, Nt)
    # Time-resample bonding_front onto the same canonical Nt for params.
    bf_canon = np.interp(t_canon, s_u, bf_u.astype(np.float64)).astype(np.float32)

    params["t_max"] = span
    params["bonding_front"] = bf_canon
    return f.astype(np.float32, copy=False), params


def _build_one_safe(args):
    """Skip-tolerant wrapper around `_build_one`.

    Runs `preflight_npz` first; if the NPZ does not pass every invariant,
    returns (None, {"basename": ..., "skip_reason": ...}) so the caller
    can log + skip rather than abort the whole dataset. If preflight
    passes but `_build_one` raises anyway (e.g. an unexpected runtime
    error during interpolation), the exception is also captured into a
    skip reason so a single bad file cannot kill a multi-thousand-sim
    cache build.

    Top-level so ProcessPoolExecutor workers can pickle it.
    """
    path_str = args[0]
    p = Path(path_str)
    ok, reason = preflight_npz(p)
    if not ok:
        return None, {"basename": p.name, "skip_reason": reason}
    try:
        f, params = _build_one(args)
    except Exception as e:                                  # noqa: BLE001
        return None, {"basename": p.name,
                      "skip_reason": f"build failed: "
                                     f"{type(e).__name__}: {e}"}
    return f, params


def _summarize_skips(skip_log: list[dict]) -> None:
    """Print a grouped breakdown of why files were skipped.

    Groups by a short prefix of the skip reason so a long list of
    "skipped_step_count = N" reasons collapses to one line per N.
    """
    if not skip_log:
        return
    from collections import Counter
    short = Counter()
    for entry in skip_log:
        reason = entry["skip_reason"]
        head = reason.split(";")[0].split(",")[0][:80]
        short[head] += 1
    print("  loader: skip reasons breakdown:", flush=True)
    for head, count in short.most_common():
        print(f"    {count:5d} x  {head}", flush=True)
    # First handful of bad-file names + their full reason for triage.
    cap = 10
    print(f"  loader: first {min(cap, len(skip_log))} skipped file(s):",
          flush=True)
    for entry in skip_log[:cap]:
        print(f"    {entry['basename']}: {entry['skip_reason']}", flush=True)
    if len(skip_log) > cap:
        print(f"    ... and {len(skip_log) - cap} more", flush=True)


_DEFAULT_WORKER_CAP = 32


def _resolve_workers(requested: int | None, n_files: int) -> int:
    """Pick a safe ProcessPool worker count.

    Policy:
      - Explicit int request is respected as-is (capped at n_files).
      - None means auto: min(cpu_count - 2, cap), where cap is 32 by
        default and can be overridden by the env var
        WAFER3D_LOADER_WORKERS_CAP (e.g. set to 64 on a fat node, or 8
        on a shared box). The 32 ceiling exists because canonicalization
        is I/O- + Delaunay-bound and saturates well before 32 processes;
        going higher just oversubscribes BLAS threads inside each
        worker (numpy default = all cores per process) and starves the
        machine. On a 256-core server that protected default would
        otherwise resolve to 254 processes * many BLAS threads each.
    """
    if requested is not None:
        return max(1, min(int(requested), max(1, n_files)))
    try:
        cap = int(os.environ.get("WAFER3D_LOADER_WORKERS_CAP",
                                  str(_DEFAULT_WORKER_CAP)))
    except ValueError:
        cap = _DEFAULT_WORKER_CAP
    cap = max(1, cap)
    auto = max(1, (os.cpu_count() or 2) - 2)
    return min(auto, cap, max(1, n_files))


def _build(files, x_canon, y_canon, t_canon, workers: int | None = None,
           drop_first_steps: int = 0
           ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Canonicalize files; skip bad NPZs.

    Returns (F, loaded_meta, skip_log) where:
      F            (n_loaded, Nx, Ny, Nt) float32 stack
      loaded_meta  list of length n_loaded; per-sim params dicts
      skip_log     list of {basename, skip_reason} for skipped files
    """
    n = len(files)
    workers = _resolve_workers(workers, n)
    args = [(str(p), x_canon, y_canon, t_canon, drop_first_steps)
            for p in files]

    t0 = time.time()
    last = [0.0]
    nx, ny, nt = x_canon.size, y_canon.size, t_canon.size

    loaded_F: list[np.ndarray] = []
    loaded_meta: list[dict] = []
    skip_log: list[dict] = []

    def _tick(k: int) -> None:
        now = time.time()
        if now - last[0] >= 5.0 or k == n:
            rate = k / max(now - t0, 1e-9)
            eta_min = (n - k) / max(rate, 1e-9) / 60.0
            print(f"  loader: {k}/{n} files ({100.0 * k / n:.1f}%)  "
                  f"{rate:.2f}/s  ETA {eta_min:.1f} min  "
                  f"({len(loaded_F)} ok, {len(skip_log)} skipped)",
                  flush=True)
            last[0] = now

    def _consume(res):
        f, params = res
        if f is None:
            skip_log.append(params)
        else:
            loaded_F.append(f)
            loaded_meta.append(params)

    cores = os.cpu_count() or 0
    cap_env = os.environ.get("WAFER3D_LOADER_WORKERS_CAP")
    print(f"  loader: workers={workers}  (host_cores={cores}, "
          f"cap={cap_env or _DEFAULT_WORKER_CAP}, n_files={n})",
          flush=True)
    if workers <= 1 or n < _PARALLEL_MIN:
        for k, a in enumerate(args, 1):
            _consume(_build_one_safe(a))
            _tick(k)
    else:
        print(f"  loader: building {n}-file cache with {workers} workers ...",
              flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, res in enumerate(
                    # chunksize=1: first sim returns within ~80 s instead
                    # of ~5 min (chunksize=4 forced the first worker to
                    # finish a 4-sim batch before any progress tick
                    # appeared). IPC overhead per sim is < 1 ms vs the
                    # 80 s of actual work, so total time is unaffected.
                    ex.map(_build_one_safe, args, chunksize=1), 1):
                _consume(res)
                _tick(k)
    print(f"  loader: built {len(loaded_F)} sim(s) out of {n} file(s) "
          f"in {(time.time() - t0) / 60.0:.1f} min "
          f"({len(skip_log)} skipped)", flush=True)
    _summarize_skips(skip_log)

    if loaded_F:
        F = np.empty((len(loaded_F), nx, ny, nt), dtype=np.float32)
        for i, fi in enumerate(loaded_F):
            F[i] = fi
    else:
        F = np.empty((0, nx, ny, nt), dtype=np.float32)
    return F, loaded_meta, skip_log


def _save_cache(cache_path, x_canon, y_canon, F, loaded_meta, skip_log,
                nx, ny, nt):
    """Atomic uncompressed cache write of (loaded sims, skip log).

    Cache layout (so a re-run with the same folder + same skip outcome is
    a hit):
      F                (n_loaded, Nx, Ny, Nt) float32 -- canonicalized
      basenames        (n_loaded,) object -- loaded-sim filenames
      skipped_files    (n_skipped,) object -- skipped-sim filenames
      skip_reasons     (n_skipped,) object -- one-line reasons
      bonding_front    (n_loaded, Nt) float32
      params           (n_loaded,) object -- per-sim dict minus the
                       fields surfaced as their own arrays
      t_max            (n_loaded,) float64
      nx / ny / nt     scalars
      x_canon / y_canon  1-D coord arrays

    Note np.savez cannot store nested dicts; the per-sim metadata that
    isn't a primitive is kept in `params` as an object array of dicts.
    """
    tmp = cache_path.with_suffix(".tmp.npz")
    if loaded_meta:
        bf_stack = np.stack([m["bonding_front"] for m in loaded_meta],
                            axis=0).astype(np.float32)
    else:
        bf_stack = np.empty((0, nt), dtype=np.float32)
    basenames = np.array([m.get("basename", "") for m in loaded_meta],
                         dtype=object)
    t_max = np.array([m.get("t_max", float("nan")) for m in loaded_meta],
                     dtype=np.float64)
    params_obj = np.empty(len(loaded_meta), dtype=object)
    for i, m in enumerate(loaded_meta):
        m_copy = {k: v for k, v in m.items()
                  if k not in ("bonding_front", "basename", "t_max")}
        params_obj[i] = m_copy
    skipped_files = np.array([e["basename"] for e in skip_log], dtype=object)
    skip_reasons = np.array([e["skip_reason"] for e in skip_log],
                            dtype=object)
    np.savez(
        tmp, x_canon=x_canon, y_canon=y_canon, F=F, nx=nx, ny=ny, nt=nt,
        basenames=basenames, t_max=t_max,
        bonding_front=bf_stack, params=params_obj,
        skipped_files=skipped_files, skip_reasons=skip_reasons)
    tmp.replace(cache_path)


def _sims_from_arrays(F, basenames, t_max, bonding_front, params_obj):
    sims = []
    for i in range(len(F)):
        params = dict(params_obj[i]) if params_obj[i] is not None else {}
        params["basename"] = str(basenames[i])
        params["t_max"] = float(t_max[i])
        params["bonding_front"] = np.asarray(bonding_front[i], dtype=np.float32)
        sims.append(Simulation(f=F[i], params=params))
    return sims


def _try_load_cache(cache_path, nx, ny, nt, current_names: set[str]):
    """Return (x, y, sims, skip_log) if cache is valid, else None.

    Cache-hit criterion: the union of loaded basenames and skipped
    basenames recorded in the cache must equal `current_names`. That
    means the folder file set is exactly what the cache was built from,
    so the recorded skip outcome still applies. If a new NPZ has shown
    up (or one has been removed), force a rebuild.

    Old caches written before the skip-tolerant rewrite carry `src_files`
    instead of `basenames` + `skipped_files`; detect and force a rebuild
    rather than try to migrate, since we'd have no way to recover the
    skip-reasons information anyway.
    """
    t0 = time.time()
    with np.load(cache_path, allow_pickle=True) as z:
        if int(z["nx"]) != nx or int(z["ny"]) != ny or int(z["nt"]) != nt:
            return None
        if "skipped_files" not in z.files or "basenames" not in z.files:
            # Pre-skip-policy cache; rebuild so the new bookkeeping kicks in.
            return None
        loaded = set(map(str, z["basenames"]))
        skipped = set(map(str, z["skipped_files"]))
        if loaded | skipped != current_names:
            return None
        d = {k: z[k] for k in z.files if k != "F"}
        try:
            sz = zipfile.ZipFile(cache_path).getinfo("F.npy").file_size
            if sz > 1e9:
                print(f"  loader: reading cache F ({sz / 1e9:.1f} GB) ...",
                      flush=True)
            d["F"] = _read_npz_member_fast(cache_path, "F.npy")
        except (ValueError, KeyError, OSError):
            d["F"] = z["F"]
    if d["F"].nbytes > 1e9:
        dt = max(time.time() - t0, 1e-9)
        print(f"  loader: cache loaded in {dt:.0f}s "
              f"({d['F'].nbytes / 1e9 / dt:.1f} GB/s)", flush=True)
    if _is_compressed_npz(cache_path):
        print("  loader: migrating compressed cache to uncompressed ...",
              flush=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(cache_path)
    sims = _sims_from_arrays(d["F"], d["basenames"], d["t_max"],
                             d["bonding_front"], d["params"])
    skip_log = [{"basename": str(n), "skip_reason": str(r)}
                for n, r in zip(d["skipped_files"], d["skip_reasons"])]
    if skip_log:
        print(f"  loader: cache records {len(skip_log)} previously-skipped "
              f"file(s)", flush=True)
        _summarize_skips(skip_log)
    return d["x_canon"], d["y_canon"], sims


def load_dataset(path, nx: int = 128, ny: int = 128, nt: int = 300,
                 x_end: float = 1.0, y_end: float = 1.0,
                 cache: bool = True, limit: int | None = None,
                 workers: int | None = None,
                 drop_first_steps: int = 0):
    """Load converted 3D NPZ sims onto a common quarter-disk grid.

    Returns (x_canon, y_canon, [Simulation]). The number of returned sims
    may be SMALLER than the number of NPZ files in the folder: any NPZ
    that fails `preflight_npz` is skipped with a logged reason, and the
    rest are loaded normally. The loader raises only if zero sims pass.

    Cache hits never spawn workers; cache is keyed on
    (nx, ny, nt, drop_first_steps) plus the (loaded U skipped) basenames
    recorded in the cache file, so adding or removing an NPZ in the
    folder OR changing the drop count forces a rebuild.

    drop_first_steps: count of leading waferData steps to discard at
    load time. The current 3D converter keeps a long pre-contact
    equilibration step (step_0000) whose samples span ~99% of tReal
    but carry almost no signal; setting drop_first_steps=1 cuts that
    dead zone before canonicalize so the canonical time grid lands
    entirely on the bonding event. Default 0 preserves the no-drop
    behaviour for any downstream that wants the full trajectory.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"NPZ folder not found: {folder}")
    files = _list_npz(folder)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No NPZ files in {folder}")

    x_canon, y_canon = canonical_grid(nx, ny, x_end, y_end)
    t_canon = np.linspace(0.0, 1.0, nt)
    # Cache key includes drop_first_steps so the no-drop and
    # drop-step-0 outputs never collide. Suffix is suppressed when
    # drop=0 so pre-existing caches keep their original filename.
    suffix = "" if drop_first_steps == 0 else f"_drop{drop_first_steps}"
    cache_path = folder / f"{_CACHE_PREFIX}{nx}x{ny}x{nt}{suffix}.npz"

    for stale in folder.glob(f"{_CACHE_PREFIX}*.tmp.npz"):
        try:
            sz_gb = stale.stat().st_size / 1e9
            stale.unlink()
            print(f"  loader: removed stale partial cache {stale.name} "
                  f"({sz_gb:.1f} GB) from a prior failed write", flush=True)
        except OSError:
            pass

    if cache and limit is None and cache_path.exists():
        try:
            cached = _try_load_cache(cache_path, nx, ny, nt,
                                     {p.name for p in files})
        except (zipfile.BadZipFile, OSError, EOFError, ValueError, KeyError) as e:
            print(f"  loader: cache {cache_path.name} unreadable ({e}); "
                  f"rebuilding", flush=True)
            cached = None
        if cached is not None:
            return cached

    F, meta, skip_log = _build(files, x_canon, y_canon, t_canon,
                               workers=workers,
                               drop_first_steps=drop_first_steps)

    if not meta:
        raise RuntimeError(
            f"load_dataset: 0 of {len(files)} NPZ file(s) passed preflight; "
            f"every file in {folder} is bad (see skip reasons above). "
            f"Investigate the converter run before retrying.")

    if cache and limit is None:
        gb = F.nbytes / 1e9
        print(f"  loader: writing cache ({gb:.1f} GB) ...", flush=True)
        t0 = time.time()
        try:
            _save_cache(cache_path, x_canon, y_canon, F, meta, skip_log,
                        nx, ny, nt)
            print(f"  loader: cache written in {time.time() - t0:.0f}s "
                  f"-> {cache_path.name}", flush=True)
        except OSError as e:
            tmp = cache_path.with_suffix(".tmp.npz")
            tmp_gb = tmp.stat().st_size / 1e9 if tmp.exists() else 0.0
            tmp.unlink(missing_ok=True)
            print(f"  loader: WARNING -- cache write failed "
                  f"({type(e).__name__}: {e}); "
                  f"removed partial .tmp ({tmp_gb:.1f} GB) and continuing "
                  f"without cache. Free disk space (need ~{gb:.0f} GB) "
                  f"before the next run to enable caching.", flush=True)
    # In-memory return path (cache disabled, limit set, or write failed).
    sims = []
    for i, m in enumerate(meta):
        params = dict(m)
        # `basename` was set by _build_one inside params; keep it.
        sims.append(Simulation(f=F[i], params=params))
    return x_canon, y_canon, sims
