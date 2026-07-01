"""Dump native coordinate ranges + displacement summary for one converted NPZ.

One-purpose diagnostic for the "field crammed into a small corner" symptom
seen in inspect_npz's snapshot panels. Two competing hypotheses motivate
this script:

  (1) the converter writes `coordinates_upper` in PHYSICAL METERS
      (range 0 .. R, with R = 0.15 m), but the loader treats them as
      already-normalized [0, 1] coords -> Delaunay covers only the
      (0 .. 0.15) corner of the canonical grid -> 99 percent of the
      canonical grid is off-hull and gets masked to 0. Pattern matches
      what the user observed.
  (2) the converter writes already-normalized coords (range 0 .. 1) and
      the cramming is some other artifact (e.g. the displacement field
      is truly zero almost everywhere because of a sim setup quirk).

Run on the folder, the script picks the first NPZ that passes preflight
and prints:

  - coordinate (x, y, z) min / max / shape for steps 0, 1, 2 of upper +
    lower wafer
  - displacement_z_corrected_upper min / max / mean for the first,
    middle, and last sample of step 0
  - bonding_front range across the sim
  - a verdict line classifying the result against the two hypotheses

    python scripts/diagnose_coords.py /path/to/3d_npz_folder
    python scripts/diagnose_coords.py /path/to/one_sim.npz
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.loader import preflight_npz                        # noqa: E402


_ASSUMED_R_METRES = 0.15   # the 2D codebase's WAFER_RADIUS_M; 3D should match


def _pick_one(folder: Path) -> Path | None:
    """Return the first NPZ in `folder` that passes preflight, or None."""
    files = sorted(p for p in folder.glob("*.npz")
                   if not p.name.startswith("_"))
    for p in files:
        ok, _ = preflight_npz(p)
        if ok:
            return p
    return None


def _dump_step(z, i: int, side: str) -> None:
    """Print coord (x/y/z) min/max for one step + wafer side."""
    key = f"step_{i:04d}_coordinates_{side}"
    if key not in z.files:
        print(f"  step {i:04d} {side}: (no key)")
        return
    coords = z[key]
    if coords.ndim != 2 or coords.shape[0] != 3:
        print(f"  step {i:04d} {side}: unexpected coords shape {coords.shape}")
        return
    x, y, zc = coords[0], coords[1], coords[2]
    print(f"  step {i:04d} {side}: shape={tuple(coords.shape)} "
          f"dtype={coords.dtype}")
    print(f"    x: min={x.min():.6g}  max={x.max():.6g}  "
          f"span={x.max() - x.min():.6g}")
    print(f"    y: min={y.min():.6g}  max={y.max():.6g}  "
          f"span={y.max() - y.min():.6g}")
    print(f"    z: min={zc.min():.6g}  max={zc.max():.6g}  "
          f"span={zc.max() - zc.min():.6g}")


def _dump_disp(z, i: int) -> None:
    """Print displacement_z_corrected_upper summary for step i."""
    dkey = f"step_{i:04d}_displacement_z_corrected_upper"
    tkey = f"step_{i:04d}_tReal"
    if dkey not in z.files or tkey not in z.files:
        print(f"  step {i:04d}: missing displacement or tReal")
        return
    disp = z[dkey]
    t = z[tkey]
    print(f"  step {i:04d}: disp_z_upper shape={tuple(disp.shape)} "
          f"dtype={disp.dtype}  Ti={len(t)}")
    if disp.shape[0] == 0:
        print("    (zero time points)")
        return
    rows = [0, disp.shape[0] // 2, disp.shape[0] - 1]
    for r in rows:
        v = disp[r]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            print(f"    t-idx {r:3d} (t={float(t[r]):.4g}): all non-finite")
            continue
        nz = int((v != 0).sum())
        print(f"    t-idx {r:3d} (t={float(t[r]):.4g}): "
              f"min={v.min():.4e}  max={v.max():.4e}  "
              f"mean={v.mean():.4e}  nonzero={nz}/{v.size}")


def diagnose(path: Path) -> int:
    print(f"\n=== diagnose: {path.name} ===")
    with np.load(path, allow_pickle=True) as z:
        n_steps = int(z["num_wafer_steps"]) if "num_wafer_steps" in z.files else 0
        n_samples = int(z["num_samples"]) if "num_samples" in z.files else 0
        print(f"num_wafer_steps={n_steps}  num_samples={n_samples}")
        for k in ("contactTime", "releaseTime_LW", "releaseTime_UW", "hGap"):
            if k in z.files:
                v = z[k]
                v = v.item() if (isinstance(v, np.ndarray)
                                 and v.shape == ()) else v
                print(f"  {k} = {v}")

        # --- coordinates: first up-to-3 steps, both sides ---
        print("\n[coordinates: first three steps, both wafer sides]")
        for i in range(min(3, n_steps)):
            _dump_step(z, i, "upper")
            _dump_step(z, i, "lower")

        # --- displacement summary for step 0 ---
        print("\n[displacement_z_corrected_upper: step 0]")
        _dump_disp(z, 0)

        # --- bonding_front + sample_tReal sanity ---
        if "sample_bonding_front" in z.files:
            bf = np.asarray(z["sample_bonding_front"])
            print(f"\nsample_bonding_front (S={bf.size}): "
                  f"min={bf.min():.4g}  max={bf.max():.4g}  "
                  f"mean={bf.mean():.4g}")
        if "sample_tReal" in z.files:
            t = np.asarray(z["sample_tReal"])
            print(f"sample_tReal (S={t.size}): "
                  f"min={t.min():.4g}  max={t.max():.4g}  "
                  f"span={t.max() - t.min():.4g}")

        # --- verdict ---
        coords_upper_keys = sorted(
            k for k in z.files
            if k.startswith("step_") and k.endswith("_coordinates_upper"))
        if not coords_upper_keys:
            print("\n[verdict] no coordinates_upper keys present; cannot classify")
            return 1

        # Use step 0 upper as the indicator -- it's representative.
        coords0 = z[coords_upper_keys[0]]
        x_max_observed = float(coords0[0].max())
        y_max_observed = float(coords0[1].max())
        biggest_axis = max(x_max_observed, y_max_observed)

    print("\n[verdict]")
    print(f"  observed max(|x|, |y|) on step 0 upper = {biggest_axis:.6g}")
    tol = 0.05
    if abs(biggest_axis - _ASSUMED_R_METRES) < _ASSUMED_R_METRES * tol:
        print(f"  -> matches PHYSICAL METERS (R = {_ASSUMED_R_METRES} m) "
              f"within {tol*100:.0f}%.")
        print(f"     This is the EXPECTED format. The loader normalizes "
              f"native coords by R = {_ASSUMED_R_METRES} m so the "
              f"canonical [0, 1] grid covers the full quarter-disk.")
        return 0
    if abs(biggest_axis - 1.0) < 0.05:
        print("  -> matches NORMALIZED coordinates (~1.0).")
        print("     WARNING: the loader expects raw meters and will divide "
              f"by R = {_ASSUMED_R_METRES} m again, leading to "
              f"double-normalization. The converter has changed its "
              f"contract -- either drop the converter-side normalization "
              f"or remove the loader-side divide.")
        return 2
    print(f"  -> does not match either ~{_ASSUMED_R_METRES} m or ~1.0; "
          f"observed {biggest_axis:.6g}.")
    print("     Possible: coords in millimeters, micrometers, or some "
          "other unit. Investigate the converter output before continuing.")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="folder of NPZs (will pick first that "
                                   "passes preflight) OR single .npz file")
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if p.is_dir():
        picked = _pick_one(p)
        if picked is None:
            print(f"no NPZ in {p} passes preflight; cannot diagnose")
            return 1
        print(f"folder mode: picked {picked.name} (first preflight-passing "
              f"file in folder)")
        return diagnose(picked)
    if p.is_file() and p.suffix == ".npz":
        return diagnose(p)
    print(f"error: {p} is not a folder or .npz file", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
