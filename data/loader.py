"""Load converted 3D NPZ simulations onto the canonical Cartesian grid.

Per-NPZ layout: step-wise (each waferData step has its own static mesh and
time-dependent fields; the mesh adapts across steps). Per sample (= one time
point of one step) the loader takes the upper-wafer (x, y) point cloud and
the corresponding `displacement_z_corrected_upper` slice, interpolates onto
the canonical quarter-disk Cartesian grid, then time-resamples the stack
onto a uniform time axis.

See docs/NPZ_SCHEMA.md for the full schema and field policy.

Pipeline per sim:
  1. Open NPZ, read sample index arrays + tReal + bonding_front + metadata.
  2. Group sample indices by step_idx.
  3. For each unique step:
     a. Read coordinates_upper[:2, :].T -- the native (x, y) point cloud
        (validated to lie in the first quadrant).
     b. Build one scipy.spatial.Delaunay over those points and precompute
        the barycentric weights of every canonical grid query point. Two
        outputs per step: a (Nx*Ny, 3) vertex-index array and a
        (Nx*Ny, 3) weight array.
     c. For each sample in the step, gather displacement values at the
        three vertices, dot with the weights, mask off-hull (-> 0), and
        reshape to (Nx, Ny). The Delaunay + weights are paid once per
        step, not once per sample.
  4. Stack per-sample (Nx, Ny) into f_stack (S, Nx, Ny).
  5. Validate tReal monotonicity (allow exact duplicates at step boundaries
     but abort on a backward jump > one median dt; same rule as 2D).
  6. Normalize sample times to [0, 1] and linearly resample each pixel's
     time trace onto t_canon (Nt,).
  7. Return f (Nx, Ny, Nt) float32 + a params dict.

Cache layer: identical framework to 2D (parallel build, atomic write,
disk-full resilience, stale .tmp sweep, CRC-free bulk read).
"""

from __future__ import annotations
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from core.simulation import Simulation
from core.grid import canonical_grid, disk_mask

_CACHE_PREFIX = "_loader_cache_"
_PARALLEL_MIN = 32
_QUARTER_TOL = 1e-6   # native x/y allowed slightly negative due to float roundoff


# Per-sim metadata pulled from file-level / COMSOL-level NPZ keys; copied
# into Simulation.params for downstream provenance and conditional-feature
# experiments. Keys that the NPZ may or may not contain (older / minimal
# conversions) are loaded with a `_get_optional` guard below.
_OPTIONAL_PHYSICAL_KEYS = ("contactTime", "releaseTime_LW",
                           "releaseTime_UW", "hGap")
_OPTIONAL_STRING_KEYS = ("source_json", "source_json_name", "converter_version",
                         "modelName", "z_correction_formula")


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
                     grid_pts: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delaunay-based barycentric weights for many "function" interpolations.

    Builds one Delaunay over `xy_native`, locates each query point in the
    grid in a simplex, then returns (vertices, weights, inside) so a caller
    can interpolate any value vector v defined on the native points at all
    query points via `(v[vertices] * weights).sum(axis=1)` -- without
    re-running find_simplex on every value vector.

    Returns:
        vertices: (Nq, 3) int   indices into the native points
        weights:  (Nq, 3) float barycentric coordinates summing to 1 inside
        inside:   (Nq,)   bool  False where the query is outside the hull
    """
    tri = Delaunay(xy_native)
    simplex = tri.find_simplex(grid_pts)
    inside = simplex >= 0
    # For off-hull points, find_simplex returns -1; index 0 is a safe stand-in
    # so the gather below does not raise; the inside mask zeroes them later.
    simplex_safe = np.where(inside, simplex, 0)

    # scipy stores per-simplex affine maps in tri.transform with shape
    # (n_simplex, ndim+1, ndim). The first ndim rows of each (ndim+1, ndim)
    # block, multiplied by (q - last_vertex), yield (ndim) barycentric
    # coordinates b1..b_ndim; b_{ndim+1} = 1 - sum(b1..b_ndim).
    T = tri.transform[simplex_safe]              # (Nq, 3, 2)
    delta = grid_pts - T[:, 2, :]                # (Nq, 2)
    b_first = np.einsum("ijk,ik->ij", T[:, :2, :], delta)  # (Nq, 2)
    b_last = 1.0 - b_first.sum(axis=1, keepdims=True)
    weights = np.concatenate([b_first, b_last], axis=1)    # (Nq, 3)
    vertices = tri.simplices[simplex_safe]                 # (Nq, 3)
    return vertices, weights, inside


def _interp_to_grid(values: np.ndarray, vertices: np.ndarray,
                    weights: np.ndarray, inside: np.ndarray,
                    nx: int, ny: int) -> np.ndarray:
    """Apply precomputed barycentric weights to a scalar field.

    values: (N_native,) float -- displacement at every native point.
    returns: (nx, ny) float32 -- value 0 off-hull (and where mask later kills).
    """
    q = (values[vertices] * weights).sum(axis=1)
    q[~inside] = 0.0
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
    """Load one converted-NPZ sim onto the canonical (Nx, Ny, Nt) grid."""
    path_str, x_canon, y_canon, t_canon = args
    p = Path(path_str)
    nx, ny, nt = x_canon.size, y_canon.size, t_canon.size

    # Canonical-grid query points (Nx*Ny, 2). Built once per sim.
    X, Y = np.meshgrid(x_canon, y_canon, indexing="ij")
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])

    with np.load(p, allow_pickle=True) as z:
        if "num_samples" not in z.files:
            raise ValueError(
                f"{p.name}: missing `num_samples`; not a converted 3D NPZ?")
        S = int(z["num_samples"])
        step_idx_arr = np.asarray(z["sample_step_index"], dtype=np.int64)
        time_idx_arr = np.asarray(z["sample_time_index_within_step"],
                                  dtype=np.int64)
        treal = np.asarray(z["sample_tReal"], dtype=np.float64)
        bf = np.asarray(z["sample_bonding_front"], dtype=np.float32)

        # Optional file-level metadata; copied into params for provenance.
        params: dict = {"basename": p.name, "num_samples": S}
        for k in _OPTIONAL_PHYSICAL_KEYS:
            v = _get_optional(z, k)
            if v is not None:
                params[k] = float(v)
        for k in _OPTIONAL_STRING_KEYS:
            v = _get_optional(z, k)
            if v is not None:
                params[k] = str(v)
        if "num_wafer_steps" in z.files:
            params["num_wafer_steps"] = int(z["num_wafer_steps"])

        # --- spatial: per-step Delaunay, shared across all samples in step ---
        f_stack = np.zeros((S, nx, ny), dtype=np.float32)
        for step_idx in np.unique(step_idx_arr):
            mask = step_idx_arr == step_idx
            sample_ks = np.where(mask)[0]
            prefix = f"step_{int(step_idx):04d}"
            coords = np.asarray(z[f"{prefix}_coordinates_upper"])    # (3, N_up)
            xy_native = coords[:2, :].T.astype(np.float64)           # (N_up, 2)

            # Quarter-symmetry validation: every native point must lie in the
            # first quadrant (small float tolerance). A negative coordinate
            # means either the converter wrote a wrong quadrant or the user
            # is feeding 2D / full-disk data to the 3D loader.
            if ((xy_native[:, 0] < -_QUARTER_TOL).any() or
                    (xy_native[:, 1] < -_QUARTER_TOL).any()):
                raise ValueError(
                    f"{p.name} step {int(step_idx)}: native (x, y) coordinates"
                    f" outside the first quadrant; loader expects "
                    f"quarter-symmetry data.")
            xy_native = np.maximum(xy_native, 0.0)  # snap tiny negatives to 0

            vertices, weights, inside = _precompute_bary(xy_native, grid_pts)

            disp_all = np.asarray(
                z[f"{prefix}_displacement_z_corrected_upper"]
            )                                                        # (T_i, N_up)
            for k in sample_ks:
                values = disp_all[int(time_idx_arr[k]), :].astype(np.float64)
                f_stack[k] = _interp_to_grid(values, vertices, weights,
                                             inside, nx, ny)

    # --- mask off-disk cells (corners of the bounding square) ---
    mask2d = disk_mask(nx, ny, x_canon[-1], y_canon[-1])
    f_stack[:, ~mask2d] = 0.0

    # --- temporal: validate tReal, dedupe step boundaries, resample to Nt ---
    if treal.size != S:
        raise ValueError(
            f"{p.name}: sample_tReal length {treal.size} != num_samples {S}")
    dt = np.diff(treal)
    forward = dt[dt > 0]
    typ_dt = float(np.median(forward)) if forward.size else 0.0
    if np.any(dt < -typ_dt):
        raise ValueError(
            f"{p.name}: sample_tReal jumps backward by more than one typical"
            f" timestep (typ_dt={typ_dt:g}); concat order is suspect.")
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


def _build(files, x_canon, y_canon, t_canon, workers: int | None = None):
    """Canonicalize all files; parallelize for large sets with progress/ETA."""
    n = len(files)
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, n)
    args = [(str(p), x_canon, y_canon, t_canon) for p in files]

    t0 = time.time()
    last = [0.0]

    def _tick(k: int) -> None:
        now = time.time()
        if now - last[0] >= 5.0 or k == n:
            rate = k / max(now - t0, 1e-9)
            eta_min = (n - k) / max(rate, 1e-9) / 60.0
            print(f"  loader: {k}/{n} files ({100.0 * k / n:.1f}%)  "
                  f"{rate:.2f}/s  ETA {eta_min:.1f} min", flush=True)
            last[0] = now

    nx, ny, nt = x_canon.size, y_canon.size, t_canon.size
    F = np.empty((n, nx, ny, nt), dtype=np.float32)
    meta: list = [None] * n

    def _store(j: int, res) -> None:
        F[j], meta[j] = res

    if workers <= 1 or n < _PARALLEL_MIN:
        for k, a in enumerate(args, 1):
            _store(k - 1, _build_one(a))
            _tick(k)
    else:
        print(f"  loader: building {n}-file cache with {workers} workers ...",
              flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, res in enumerate(ex.map(_build_one, args, chunksize=4), 1):
                _store(k - 1, res)
                _tick(k)
    print(f"  loader: built {n} files in {(time.time() - t0) / 60.0:.1f} min",
          flush=True)
    return F, meta


def _save_cache(cache_path, x_canon, y_canon, F, meta, files, nx, ny, nt):
    """Atomic uncompressed cache write -- per-sim params kept as object dtype.

    np.savez does not understand nested dicts; flatten each Simulation.params
    into per-sim arrays sized (N_sim,) and let the loader rehydrate dicts.
    """
    tmp = cache_path.with_suffix(".tmp.npz")
    # Per-sim arrays (object dtype) for everything not array-typed.
    # bonding_front is a (Nt,) array per sim -> stack to (N_sim, Nt).
    bf_stack = np.stack([m["bonding_front"] for m in meta], axis=0).astype(np.float32)
    basenames = np.array([m.get("basename", "") for m in meta], dtype=object)
    t_max = np.array([m.get("t_max", float("nan")) for m in meta], dtype=np.float64)
    # Carry the rest as an object array of dicts for fidelity.
    params_obj = np.empty(len(meta), dtype=object)
    for i, m in enumerate(meta):
        m_copy = {k: v for k, v in m.items()
                  if k not in ("bonding_front", "basename", "t_max")}
        params_obj[i] = m_copy
    np.savez(
        tmp, x_canon=x_canon, y_canon=y_canon, F=F, nx=nx, ny=ny, nt=nt,
        src_files=np.array([p.name for p in files], dtype=object),
        basenames=basenames, t_max=t_max,
        bonding_front=bf_stack, params=params_obj)
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


def _try_load_cache(cache_path, nx, ny, nt, current_names):
    t0 = time.time()
    with np.load(cache_path, allow_pickle=True) as z:
        if int(z["nx"]) != nx or int(z["ny"]) != ny or int(z["nt"]) != nt:
            return None
        if set(map(str, z["src_files"])) != current_names:
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
    return d["x_canon"], d["y_canon"], sims


def load_dataset(path, nx: int = 128, ny: int = 128, nt: int = 300,
                 x_end: float = 1.0, y_end: float = 1.0,
                 cache: bool = True, limit: int | None = None,
                 workers: int | None = None):
    """Load converted 3D NPZ sims onto a common quarter-disk grid.

    Returns (x_canon, y_canon, [Simulation]). Cache hits never spawn workers.
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
    cache_path = folder / f"{_CACHE_PREFIX}{nx}x{ny}x{nt}.npz"

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

    F, meta = _build(files, x_canon, y_canon, t_canon, workers=workers)
    if cache and limit is None:
        gb = F.nbytes / 1e9
        print(f"  loader: writing cache ({gb:.1f} GB) ...", flush=True)
        t0 = time.time()
        try:
            _save_cache(cache_path, x_canon, y_canon, F, meta, files,
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
    for i, p in enumerate(files):
        params = dict(meta[i])
        params["basename"] = p.name
        sims.append(Simulation(f=F[i], params=params))
    return x_canon, y_canon, sims
