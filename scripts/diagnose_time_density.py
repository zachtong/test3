"""Inspect native sample_tReal density to verify the COMSOL-adaptive-step
hypothesis.

The diagnose_loaded run showed the canonical (Nx, Ny, Nt) field has 99%
of its signal in the last few Nt indices and a flat plateau through the
middle. The working hypothesis is that COMSOL uses adaptive time
stepping: dense sampling during the bonding event (last few percent of
real time) and sparse sampling elsewhere. Uniform canonical resampling
then leaves the middle canonical t-indices interpolating between sparse
native samples that happen to carry almost-identical field values,
producing the observed plateau.

This script confirms or refutes that. It walks one NPZ's `sample_tReal`,
plots:

  1. Histogram of native samples per 5% bin of normalised time s in [0, 1]
  2. Cumulative density of native samples vs s
  3. log10(forward dt) vs s, so adaptive-step regions stand out

and prints:

  - the fraction of native samples in each 10% s window
  - the max / min forward dt and their ratio
  - the canonical Nt that would have to be used to keep ~10 native
    samples per canonical t-index in the densest region

A verdict line classifies:

  - peak-to-trough density ratio < 5 -> roughly uniform, time
    resampling is not the issue
  - 5 .. 50 -> moderately uneven; bumping Nt 2-3x will help
  - > 50 -> extremely uneven; trim or non-uniform t_canon needed

    python scripts/diagnose_time_density.py /path/to/3d_npz_folder
    python scripts/diagnose_time_density.py /path/to/one_sim.npz \\
        --out /tmp/density.png
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


def diagnose(path: Path, out_path: Path) -> int:
    print(f"\n=== time-density: {path.name} ===")
    with np.load(path, allow_pickle=True) as z:
        treal = np.asarray(z["sample_tReal"], dtype=np.float64)
        S = int(z["num_samples"])

    span = float(treal.max() - treal.min())
    if span <= 0:
        print("zero tReal span; cannot diagnose")
        return 1
    s = (treal - treal.min()) / span             # normalised to [0, 1]

    # --- 10% bin counts ---
    edges10 = np.linspace(0.0, 1.0, 11)
    counts10, _ = np.histogram(s, bins=edges10)
    print(f"\nnative samples: S = {S},  tReal span = {span:.4g} s")
    print("\nsamples per 10% normalised-time bin:")
    print(f"  {'bin':>10}  {'count':>8}  {'fraction':>10}")
    for i in range(10):
        lo, hi = edges10[i], edges10[i + 1]
        frac = counts10[i] / S
        bar = "#" * int(round(50 * frac))
        print(f"  [{lo:0.2f}, {hi:0.2f})  {counts10[i]:>8d}  "
              f"{100.0 * frac:>9.2f}%  {bar}")
    print(f"  TOTAL                                  {S} (100.00%)")

    # --- forward dt stats ---
    dt = np.diff(treal)
    forward = dt[dt > 0]
    if forward.size == 0:
        print("\nno forward dt found; tReal is non-monotonic everywhere")
        return 1
    dt_min = float(forward.min())
    dt_max = float(forward.max())
    dt_med = float(np.median(forward))
    ratio = dt_max / max(dt_min, 1e-30)
    print(f"\nforward dt stats: "
          f"min={dt_min:.4g}  median={dt_med:.4g}  max={dt_max:.4g}")
    print(f"  max/min ratio = {ratio:.2g}")

    # --- where does the densest region sit in normalised time? ---
    # 5% windows, find argmax
    edges20 = np.linspace(0.0, 1.0, 21)
    counts20, _ = np.histogram(s, bins=edges20)
    densest_bin = int(np.argmax(counts20))
    densest_lo, densest_hi = edges20[densest_bin], edges20[densest_bin + 1]
    densest_count = int(counts20[densest_bin])
    print(f"densest 5% window: [{densest_lo:.2f}, {densest_hi:.2f}) "
          f"with {densest_count} native samples "
          f"({100.0 * densest_count / S:.1f}% of all)")

    # --- Nt sizing ---
    # In the densest region, current canonical Nt=300 gives nt * 0.05 = 15
    # canonical t-indices. If you want ~10 native samples per canonical
    # index in that region, you need:
    #     nt_target = densest_count / 10 / 0.05
    nt_target = int(np.ceil(densest_count / 10.0 / 0.05))
    print(f"\nfor ~10 native samples per canonical t-idx in the densest "
          f"5% window, set Nt >= {nt_target}")
    print("(current default Nt=300 gives "
          f"{300 * 0.05:.0f} canonical t-indices in that window -> "
          f"~{densest_count / max(300 * 0.05, 1):.1f} native samples per "
          f"canonical t-idx)")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    axes[0].bar(edges20[:-1], counts20, width=np.diff(edges20),
                align="edge", color="C0", edgecolor="black", lw=0.4)
    axes[0].set_xlabel("normalised tReal  s  in [0, 1]")
    axes[0].set_ylabel("native samples per 5% bin")
    axes[0].set_title(f"{path.name}: native time density "
                      f"(S={S}, span={span:.3g}s)")
    axes[0].axhline(S / 20, color="0.5", ls="--", lw=1,
                    label="uniform reference (S/20)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    cdf = np.arange(1, S + 1) / S
    axes[1].plot(np.sort(s), cdf, "C0-", lw=1.2)
    axes[1].plot([0, 1], [0, 1], "0.5", ls="--", lw=1,
                 label="uniform reference")
    axes[1].set_xlabel("normalised tReal s")
    axes[1].set_ylabel("cumulative fraction of native samples")
    axes[1].set_title("CDF of native time samples")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # log dt vs s: use midpoint of each native interval
    s_mid = 0.5 * (s[1:] + s[:-1])
    pos_mask = dt > 0
    axes[2].semilogy(s_mid[pos_mask], dt[pos_mask], "C0.", ms=2)
    axes[2].axhline(dt_med, color="0.5", ls="--", lw=1,
                    label=f"median dt = {dt_med:.2g}")
    axes[2].set_xlabel("normalised tReal s")
    axes[2].set_ylabel("forward dt (s)")
    axes[2].set_title("forward dt vs normalised time "
                      f"(adaptive-step regions stand out; ratio "
                      f"max/min = {ratio:.1g})")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {out_path}")

    # --- verdict ---
    print("\n[verdict]")
    if ratio < 5:
        print(f"  -> dt ratio {ratio:.1f} is small (< 5); native time is")
        print("     roughly uniform. Time resampling is NOT the suspect")
        print("     for the canonical-field plateau; look elsewhere.")
        return 0
    if ratio < 50:
        print(f"  -> dt ratio {ratio:.1f} is moderate (5..50). Native time")
        print("     is uneven; uniform canonical resample loses some")
        print("     temporal resolution in the dense region. Bumping Nt")
        print("     by 2-3x should help -- e.g. Nt = 600 to 1000.")
        return 0
    print(f"  -> dt ratio {ratio:.1f} is EXTREME (> 50). COMSOL is adaptive-")
    print("     stepping aggressively; uniform canonical resample at Nt=300")
    print("     will (and does) crush the dense region into a handful of")
    print("     canonical t-indices.")
    print(f"     Densest 5% window holds {100.0 * densest_count / S:.0f}% of "
          f"all native samples.")
    print("     Options:")
    print(f"       A. bump Nt to ~{nt_target}+ to give the dense window")
    print("          enough canonical resolution (cache size scales x"
          f"{nt_target / 300:.1f})")
    print("       B. trim the trajectory to start near the dense region's")
    print("          left edge to drop the pre-bonding dead zone")
    print("       C. use a non-uniform t_canon that matches native density")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="folder of NPZs or single .npz file")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: "
                    "<path-parent>/diagnose_out/<stem>_time_density.png)")
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if p.is_dir():
        picked = _pick_one(p)
        if picked is None:
            print(f"no NPZ in {p} passes preflight")
            return 1
        out = (Path(args.out) if args.out
               else p.parent / "diagnose_out"
               / f"{picked.stem}_time_density.png")
        print(f"folder mode: picked {picked.name}")
        return diagnose(picked, out)
    if p.is_file() and p.suffix == ".npz":
        out = (Path(args.out) if args.out
               else p.parent / "diagnose_out"
               / f"{p.stem}_time_density.png")
        return diagnose(p, out)
    print(f"error: {p} is not a folder or .npz file", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
