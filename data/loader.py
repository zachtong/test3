"""Load converted 3D NPZ simulations onto the canonical Cartesian grid.

STUB. The NPZ schema for the 3D export is not finalized yet, so the
per-file canonicalization (`_canonicalize`) and the field read (`_read_npz`)
both raise `NotImplementedError`. The pieces that ARE in place are the
generic ones, ported from the validated 2D loader:

  - Per-file parallel build (`ProcessPoolExecutor`)
  - Streaming preallocated result stack (no end-of-build np.stack stall)
  - Atomic uncompressed cache write (.tmp -> rename) with stale .tmp sweep
  - CRC-free bulk cache read via fromfile + npz local-header parse
  - Cache validity keyed on (nx, ny, nt) + the source-filename set
  - Disk-full-resilient cache write: warn, drop .tmp, continue in-memory

When the NPZ schema is fixed, fill `_read_npz` (member keys) and
`_canonicalize` (resample the native field onto (nx_canon, ny_canon,
nt_canon)) -- the rest of the pipeline already works.
"""

from __future__ import annotations
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from core.simulation import Simulation
from core.grid import canonical_grid

_CACHE_PREFIX = "_loader_cache_"
_PARALLEL_MIN = 32


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


def _read_npz(path):
    """Pull the raw arrays out of one converted 3D NPZ.

    TODO -- pending NPZ schema. Expected to return:
      f_native: (Nx_native, Ny_native, Nt_native) gap field
      x_native: (Nx_native,) native x-coord (normalized by R)
      y_native: (Ny_native,) native y-coord (normalized by R)
      t_native: (Nt_native,) time axis (raw or normalized)
      src: provenance string for params
      r_max: physical max in-disk radius (m) if recorded, else NaN
    """
    raise NotImplementedError(
        "data/loader.py::_read_npz: 3D NPZ schema not yet defined. "
        "Fill once `data/json_to_npz_converter.py` finalizes the field list.")


def _canonicalize(f_native, x_native, y_native, t_native,
                  x_canon, y_canon, t_canon):
    """Resample one sim's field onto the canonical (nx, ny, nt) grid.

    TODO -- pending NPZ schema. Likely path: bilinear interpolation on the
    spatial axes (scipy.interpolate.RegularGridInterpolator over (x, y) for
    each timestep) followed by 1D linear interpolation on the time axis (the
    2D code's `_interp_rows` shape).
    """
    raise NotImplementedError(
        "data/loader.py::_canonicalize: 3D resample not yet defined. "
        "Pending NPZ schema; mirror the 2D `_canonicalize` shape -- "
        "spatial first (RegularGridInterpolator over (x, y)), then a 1D "
        "linear pass on the time axis.")


def _build_one(args):
    path_str, x_canon, y_canon, t_canon = args
    p = Path(path_str)
    f_native, x_native, y_native, t_native, src, r_max = _read_npz(p)
    f = _canonicalize(f_native, x_native, y_native, t_native,
                      x_canon, y_canon, t_canon)
    meta = {"source": src,
            "t_max": float(t_native[-1] - t_native[0]),
            "r_max": r_max,
            "nx_native": f_native.shape[0],
            "ny_native": f_native.shape[1],
            "nt_native": f_native.shape[2]}
    return f, meta


def _build(files, x_canon, y_canon, t_canon, workers: int | None = None):
    """Canonicalize all files in parallel; preallocate the result stack."""
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
                  f"{rate:.1f}/s  ETA {eta_min:.1f} min", flush=True)
            last[0] = now

    nx, ny, nt = x_canon.size, y_canon.size, t_canon.size
    F = np.empty((n, nx, ny, nt), dtype=np.float64)
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
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(
        tmp, x_canon=x_canon, y_canon=y_canon, F=F, nx=nx, ny=ny, nt=nt,
        sources=np.array([m["source"] for m in meta], dtype=object),
        t_max=np.array([m["t_max"] for m in meta], dtype=np.float64),
        r_max=np.array([m["r_max"] for m in meta], dtype=np.float64),
        nx_native=np.array([m["nx_native"] for m in meta], dtype=np.int64),
        ny_native=np.array([m["ny_native"] for m in meta], dtype=np.int64),
        nt_native=np.array([m["nt_native"] for m in meta], dtype=np.int64),
        src_files=np.array([p.name for p in files], dtype=object))
    tmp.replace(cache_path)


def _sims_from_arrays(F, sources, t_max, r_max, nx_native, ny_native,
                      nt_native, basenames=None):
    if basenames is None:
        basenames = [""] * len(F)
    return [Simulation(f=F[i],
                       params={"source": str(sources[i]),
                               "t_max": float(t_max[i]),
                               "r_max": float(r_max[i]),
                               "nx_native": int(nx_native[i]),
                               "ny_native": int(ny_native[i]),
                               "nt_native": int(nt_native[i]),
                               "basename": str(basenames[i])})
            for i in range(len(F))]


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
    sims = _sims_from_arrays(d["F"], d["sources"], d["t_max"], d["r_max"],
                             d["nx_native"], d["ny_native"], d["nt_native"],
                             basenames=d.get("src_files"))
    return d["x_canon"], d["y_canon"], sims


def load_dataset(path, nx: int = 128, ny: int = 128, nt: int = 300,
                 x_end: float = 1.0, y_end: float = 1.0,
                 cache: bool = True, limit: int | None = None,
                 workers: int | None = None):
    """Load converted 3D NPZ sims onto a common grid.

    Returns (x_canon, y_canon, [Simulation]). Cache hits never spawn workers.
    Will raise NotImplementedError until the 3D NPZ schema is wired up.
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

    # Sweep stale .tmp.npz left by a prior failed cache write (disk-full
    # safety -- a multi-GB .tmp can otherwise sit there forever).
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
    return x_canon, y_canon, _sims_from_arrays(
        F, [m["source"] for m in meta], [m["t_max"] for m in meta],
        [m["r_max"] for m in meta], [m["nx_native"] for m in meta],
        [m["ny_native"] for m in meta], [m["nt_native"] for m in meta],
        basenames=[p.name for p in files])
