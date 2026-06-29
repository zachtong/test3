"""Probe the LOADED canonical field to localise the 'middle frame is zero'
symptom.

Context: after the coord-units fix the last snapshot frame was a clean
quarter-disk, but the early and middle snapshot frames were still
all-zero. diagnose_timeline.py showed that the native displacement
envelope crosses 5% of peak at sample ~370, well before the middle
sample of the trajectory -- so the loader is reading non-zero data at
mid-trajectory native indices. The question is then whether:

  (A) `_canonicalize` is producing a canonical (Nx, Ny, Nt) field that
      genuinely has non-zero values at canonical mid-time (so the
      snapshot's all-zero middle frame is a RENDERING bug -- shared
      vmin/vmax across all three frames squashes anything below the
      last frame's peak into the bottom of the colour scale and it
      looks black), or
  (B) `_canonicalize` is producing a canonical field that IS zero at
      mid-time (so the bug is upstream: time-resampling collapses
      bonding onto the last few canonical t-indices, or the per-step
      Delaunay interpolation drops signal).

This script answers it. For one NPZ:

  1. run `load_dataset` (using the fixed loader) on just that file
  2. take the resulting `sim.f` (Nx, Ny, Nt)
  3. sample 20 evenly spaced canonical t-indices and print
     min/max/mean + count of non-zero cells per frame
  4. classify: if mid-time frames have signal -> rendering bug;
     if mid-time frames are 0 -> canonicalize bug

    python scripts/diagnose_loaded.py /path/to/3d_npz_folder
    python scripts/diagnose_loaded.py /path/to/one_sim.npz \\
        --nx 64 --ny 64 --nt 300
"""

from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.loader import load_dataset, preflight_npz          # noqa: E402


def _pick_one(folder: Path) -> Path | None:
    files = sorted(p for p in folder.glob("*.npz")
                   if not p.name.startswith("_"))
    for p in files:
        ok, _ = preflight_npz(p)
        if ok:
            return p
    return None


def diagnose(path: Path, nx: int, ny: int, nt: int) -> int:
    print(f"\n=== loaded-field probe: {path.name} (nx={nx}, ny={ny}, "
          f"nt={nt}) ===")

    # Stage the single NPZ into a temp folder so load_dataset's
    # folder-based interface works without caching to the user's tree.
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / path.name
        shutil.copy(path, staged)
        x_canon, y_canon, sims = load_dataset(
            Path(td), nx=nx, ny=ny, nt=nt,
            cache=False, workers=1)
    if not sims:
        print("load_dataset returned 0 sims -- preflight rejected the file?")
        return 1
    sim = sims[0]
    f = sim.f                                    # (Nx, Ny, Nt) float32
    print(f"sim.f shape={tuple(f.shape)}  dtype={f.dtype}")

    # 20 evenly spaced t-indices, plus first / middle / last guaranteed.
    sample_k = sorted(set(
        list(np.linspace(0, nt - 1, 20).astype(int))
        + [0, nt // 2, nt - 1]
    ))
    print(f"\nper-frame stats at {len(sample_k)} canonical t-indices:")
    print(f"  {'t-idx':>6}  {'t':>8}  {'min':>12}  {'max':>12}  "
          f"{'mean':>12}  {'nnz/N':>14}")
    t_canon = np.linspace(0.0, 1.0, nt)
    n_total = nx * ny
    nnz_per_frame = []
    peak_per_frame = []
    for k in sample_k:
        frame = f[:, :, k]
        nnz = int((frame != 0).sum())
        nnz_per_frame.append(nnz)
        peak_per_frame.append(float(np.abs(frame).max()))
        print(f"  {k:>6}  {t_canon[k]:>8.4f}  "
              f"{frame.min():>12.4e}  {frame.max():>12.4e}  "
              f"{frame.mean():>12.4e}  "
              f"{nnz:>6d}/{n_total:<6d} "
              f"({100.0 * nnz / n_total:4.1f}%)")

    # Center-cell trace
    ix = nx // 2
    iy = ny // 2
    trace = f[ix, iy, :]
    print(f"\ncenter cell ({ix}, {iy}) trace:")
    print(f"  min={trace.min():.4e}  max={trace.max():.4e}  "
          f"abs-peak={np.abs(trace).max():.4e}  "
          f"argmax-t-idx={int(np.argmax(np.abs(trace)))}")

    # Sensor-position trace (X-edge sensor)
    edge_ix = nx - 1
    edge_iy = 0
    trace_edge = f[edge_ix, edge_iy, :]
    print(f"\n+X edge cell ({edge_ix}, {edge_iy}) trace:")
    print(f"  min={trace_edge.min():.4e}  max={trace_edge.max():.4e}  "
          f"abs-peak={np.abs(trace_edge).max():.4e}  "
          f"argmax-t-idx={int(np.argmax(np.abs(trace_edge)))}")

    # === verdict ===
    print("\n[verdict]")
    full_peak = float(np.abs(f).max())
    if full_peak <= 0:
        print("  -> canonical field is ZERO throughout. _canonicalize is")
        print("     dropping all signal. Investigate Delaunay / time interp.")
        return 1

    # Did the middle frame have non-trivial signal?
    mid_k = nt // 2
    mid_peak = peak_per_frame[sample_k.index(mid_k)]
    last_peak = peak_per_frame[sample_k.index(nt - 1)]
    first_peak = peak_per_frame[sample_k.index(0)]
    rel_mid = mid_peak / full_peak if full_peak else 0.0
    rel_last = last_peak / full_peak if full_peak else 0.0

    print(f"  full-tensor |peak|:    {full_peak:.4e}")
    print(f"  first-frame |peak|:    {first_peak:.4e}  "
          f"({100.0 * first_peak / full_peak:.2f}% of full)")
    print(f"  middle-frame |peak|:   {mid_peak:.4e}  "
          f"({100.0 * rel_mid:.2f}% of full)")
    print(f"  last-frame |peak|:     {last_peak:.4e}  "
          f"({100.0 * rel_last:.2f}% of full)")

    if rel_mid < 1e-4:
        print("\n  HYPOTHESIS B (canonicalize bug): middle canonical frame")
        print("  is effectively zero. Either Delaunay interp is dropping")
        print("  signal at mid-trajectory native t-indices, or the time")
        print("  resample is collapsing bonding onto only the last few")
        print("  canonical t-indices. Inspect _canonicalize.")
        return 2
    if rel_mid < 0.05:
        print("\n  Middle frame has only <5% of the peak signal. The")
        print("  snapshot's shared vmin/vmax will visually squash it to")
        print("  near-black even though the data is there. This is a")
        print("  RENDERING bug (cosmetic). _canonicalize is fine.")
        print("  Fix: render each frame with its own vmin/vmax, or")
        print("  use percentile clipping on the colour scale.")
        return 3
    print("\n  Middle frame has >5% of the peak; the snapshot should be")
    print("  visible. If the user reports it as all-black, suspect a")
    print("  rendering or screen issue, not the data.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="folder or single .npz")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if p.is_dir():
        picked = _pick_one(p)
        if picked is None:
            print(f"no NPZ in {p} passes preflight")
            return 1
        print(f"folder mode: picked {picked.name}")
        return diagnose(picked, args.nx, args.ny, args.nt)
    if p.is_file() and p.suffix == ".npz":
        return diagnose(p, args.nx, args.ny, args.nt)
    print(f"error: {p} is not a folder or .npz file", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
