"""Plot the full-time-history envelope of one converted NPZ.

Answers the question: 'my snapshots show all-zero early/mid and signal
only in the last frame -- is the wafer just not deformed yet at those
times, or is there a loader bug?' Renders three subplots so the answer
falls out:

  1. min(disp_upper) and max(disp_upper) vs sample index (envelope of
     the upper-wafer mid-plane displacement across all native points)
  2. bonding_front vs sample index (the converter's bonding-progress
     scalar, range 0..1)
  3. tReal vs sample index (so step boundaries / time gaps are visible)

If the envelope is flat at 0 for the first half of the trajectory and
only takes off near the end, the snapshots are PHYSICALLY CORRECT --
the wafer hasn't deformed yet. The snapshot tool happens to sample
[first, middle, last] which can easily put 'middle' into the not-yet-
bonded regime. The right follow-up is to render snapshots at
non-uniformly spaced t-indices that include the deformation onset.

    python scripts/diagnose_timeline.py /path/to/3d_npz_folder
    python scripts/diagnose_timeline.py /path/to/one_sim.npz
    python scripts/diagnose_timeline.py /path/to/3d_npz_folder \\
        --out /tmp/timeline.png
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


def _pick_one(folder: Path) -> Path | None:
    files = sorted(p for p in folder.glob("*.npz")
                   if not p.name.startswith("_"))
    for p in files:
        ok, _ = preflight_npz(p)
        if ok:
            return p
    return None


def _collect_envelope(z) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """Walk step-wise displacement arrays and build sample-level envelopes.

    Returns (sample_idx, min_disp, max_disp, sample_tReal,
    sample_bonding_front), all length S.
    """
    S = int(z["num_samples"])
    n_steps = int(z["num_wafer_steps"])
    step_idx_arr = np.asarray(z["sample_step_index"], dtype=np.int64)
    time_idx_arr = np.asarray(z["sample_time_index_within_step"],
                              dtype=np.int64)
    treal = np.asarray(z["sample_tReal"], dtype=np.float64)
    bf = np.asarray(z["sample_bonding_front"], dtype=np.float64)

    min_disp = np.empty(S, dtype=np.float64)
    max_disp = np.empty(S, dtype=np.float64)

    # Cache the step displacement arrays so we don't re-load for every k.
    for i in range(n_steps):
        prefix = f"step_{i:04d}"
        dkey = f"{prefix}_displacement_z_corrected_upper"
        if dkey not in z.files:
            continue
        disp = np.asarray(z[dkey])
        mask = step_idx_arr == i
        ks = np.where(mask)[0]
        ts = time_idx_arr[ks]
        # vectorised: for each sample k in this step, look up disp[ts[j]]
        # and reduce along the spatial axis
        rows = disp[ts]                              # (n_in_step, N_up)
        min_disp[ks] = rows.min(axis=1)
        max_disp[ks] = rows.max(axis=1)
    sample_idx = np.arange(S)
    return sample_idx, min_disp, max_disp, treal, bf


def _find_onset(envelope: np.ndarray, threshold_frac: float = 0.05
                ) -> int | None:
    """First sample index where |envelope| crosses threshold_frac * peak.

    Returns None if the envelope never moves (everything zero).
    """
    abs_env = np.abs(envelope)
    peak = abs_env.max()
    if peak <= 0:
        return None
    thresh = threshold_frac * peak
    above = np.where(abs_env > thresh)[0]
    return int(above[0]) if above.size else None


def diagnose(path: Path, out_path: Path) -> int:
    print(f"\n=== timeline: {path.name} ===")
    with np.load(path, allow_pickle=True) as z:
        s_idx, dmin, dmax, treal, bf = _collect_envelope(z)
        contact = z["contactTime"].item() if "contactTime" in z.files else None
        rel_lw = z["releaseTime_LW"].item() if "releaseTime_LW" in z.files else None
        rel_uw = z["releaseTime_UW"].item() if "releaseTime_UW" in z.files else None

    S = s_idx.size
    onset = _find_onset(dmax) or _find_onset(dmin)
    peak_idx = int(np.argmax(np.abs(dmax) + np.abs(dmin)))

    print(f"samples: {S}")
    print(f"|disp|: peak  = {max(abs(dmax).max(), abs(dmin).max()):.4e}")
    print(f"        early (sample 0):     [{dmin[0]:.4e}, {dmax[0]:.4e}]")
    print(f"        middle (sample {S//2}): "
          f"[{dmin[S//2]:.4e}, {dmax[S//2]:.4e}]")
    print(f"        last (sample {S-1}):   "
          f"[{dmin[-1]:.4e}, {dmax[-1]:.4e}]")
    if onset is None:
        print("        deformation never starts (envelope is 0 throughout)")
    else:
        frac = onset / max(S - 1, 1)
        print(f"        onset (|disp| > 5% of peak): sample {onset} "
              f"({frac*100:.1f}% through the trajectory, "
              f"tReal={treal[onset]:.4g}s)")
    print(f"bonding_front: [{bf.min():.4g}, {bf.max():.4g}]")
    print(f"tReal: [{treal.min():.4g}, {treal.max():.4g}] s")
    if contact is not None:
        print(f"  contactTime = {contact}")
    if rel_lw is not None:
        print(f"  releaseTime_LW = {rel_lw}")
    if rel_uw is not None:
        print(f"  releaseTime_UW = {rel_uw}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             constrained_layout=True)

    axes[0].plot(s_idx, dmax, "C0-", lw=1.0, label="max(disp_upper)")
    axes[0].plot(s_idx, dmin, "C3-", lw=1.0, label="min(disp_upper)")
    axes[0].fill_between(s_idx, dmin, dmax, color="C0", alpha=0.15)
    axes[0].axhline(0, color="0.6", lw=0.7)
    if onset is not None:
        axes[0].axvline(onset, color="green", ls="--", lw=1,
                        label=f"onset (sample {onset})")
    axes[0].set_ylabel("disp_z_corrected_upper")
    axes[0].set_title(f"{path.name}: spatial envelope over time")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(s_idx, bf, "C2-", lw=1.0)
    axes[1].axhline(0, color="0.6", lw=0.7)
    axes[1].set_ylabel("bonding_front")
    axes[1].grid(alpha=0.3)

    axes[2].plot(s_idx, treal, "C4-", lw=1.0)
    axes[2].set_ylabel("tReal (s)")
    axes[2].set_xlabel("sample index k (= global ML sample)")
    axes[2].grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {out_path}")

    # Interpretation
    print("\n[verdict]")
    if onset is None:
        print("  -> displacement envelope is FLAT 0 throughout. Either the")
        print("     sim never bonded (check the source COMSOL setup) or")
        print("     the displacement field stored in NPZ is broken.")
        return 1
    frac = onset / max(S - 1, 1)
    if frac > 0.7:
        print(f"  -> deformation onset at sample {onset} ({frac*100:.0f}% "
              f"into the trajectory).")
        print("     EXPLAINS THE SNAPSHOTS: render_sim_panel samples")
        print("     [0, S/2, S-1], and S/2 happens to be before onset, so")
        print("     the middle snapshot is all-zero. This is the data, not")
        print("     a bug. Render snapshots near `onset` to see the early")
        print("     wavefront, or use --pick spread in step4-style viz.")
        return 0
    if frac < 0.1:
        print(f"  -> deformation starts almost immediately "
              f"(sample {onset}). The snapshot middle frame should")
        print("     already show signal; if it doesn't, suspect a bug.")
        return 0
    print(f"  -> deformation onset is mid-trajectory (sample {onset}).")
    print("     Should be visible in the snapshot middle frame.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="folder of NPZs (will pick first that "
                                   "passes preflight) OR single .npz file")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <path-parent>/"
                    "diagnose_out/<stem>_timeline.png)")
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if p.is_dir():
        picked = _pick_one(p)
        if picked is None:
            print(f"no NPZ in {p} passes preflight; cannot diagnose")
            return 1
        out = (Path(args.out) if args.out
               else p.parent / "diagnose_out"
               / f"{picked.stem}_timeline.png")
        print(f"folder mode: picked {picked.name}")
        return diagnose(picked, out)
    if p.is_file() and p.suffix == ".npz":
        out = (Path(args.out) if args.out
               else p.parent / "diagnose_out"
               / f"{p.stem}_timeline.png")
        return diagnose(p, out)
    print(f"error: {p} is not a folder or .npz file", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
