"""POD energy-spectrum diagnostic: sigma decay + cumulative energy.

Two-panel static PNG meant to be the headline POD diagnostic in any
talk or report. Left panel: sigma_k / sigma_1 on semilog-y vs mode
index k. Right panel: cumulative explained-variance fraction
(sum_{j<=k} sigma_j^2 / sum_all sigma_j^2) on linear-y, with horizontal
guides at 0.99, 0.999, and 0.9999.

The currently-selected K (default 8) is marked as a vertical dashed
line on both panels. Mode k=6 -- known to misbehave in the
firehorse2 first run (per-mode median rel-L2 = 0.351 vs ~0.16 for its
neighbours) -- is marked with a small magenta dot on the left panel
so the spectrum view aligns visually with the per-mode error plots.

Input: a basis_cache .npz file (the one written by
training/basis_cache.py). The file carries Phi, sigma, spatial_shape,
k_cache. We use only sigma.

    python scripts/viz_pod_spectrum.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --K 8 --out viz/pod_spectrum.png
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.fieldviz import (provenance_footer,             # noqa: E402
                               SENSOR_MARKER_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True,
                    help="path to basis cache .npz file (has 'sigma')")
    ap.add_argument("--K", type=int, default=8,
                    help="currently selected K (annotated with a "
                    "vertical dashed line); default 8")
    ap.add_argument("--anomaly-mode", type=int, default=6,
                    help="optional: mark this mode index (1-based) as "
                    "anomalous with a colored dot. Pass 0 to disable.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    basis_path = Path(args.basis).expanduser().resolve()
    if not basis_path.is_file():
        print(f"basis cache file not found: {basis_path}", file=sys.stderr)
        return 2

    with np.load(basis_path, allow_pickle=False) as z:
        if "sigma" not in z.files:
            print(f"no 'sigma' in {basis_path}", file=sys.stderr)
            return 1
        sigma = np.asarray(z["sigma"], dtype=np.float64)
    if sigma.size == 0:
        print("empty sigma array", file=sys.stderr)
        return 1
    sigma = sigma[sigma > 0]   # drop any pathological zeros
    k_axis = np.arange(1, sigma.size + 1)
    rel = sigma / sigma[0]
    energy = sigma ** 2
    cum_frac = np.cumsum(energy) / energy.sum()

    print(f"basis: {basis_path.name}  k_cache={sigma.size}")
    print(f"  sigma_1 = {sigma[0]:.4g}")
    for kk in (1, 2, 4, 8, min(16, sigma.size)):
        if kk <= sigma.size:
            print(f"  K={kk:2d}  cumulative energy = "
                  f"{cum_frac[kk - 1] * 100:.4f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                    constrained_layout=True)

    # LEFT: sigma_k / sigma_1, semilog-y
    axL.semilogy(k_axis, rel, "o-", color="C0", lw=1.2, ms=4)
    if 0 < args.anomaly_mode <= sigma.size:
        axL.scatter([args.anomaly_mode], [rel[args.anomaly_mode - 1]],
                    s=80, color=SENSOR_MARKER_COLOR, zorder=5,
                    label=f"anomaly mode k={args.anomaly_mode}")
    axL.axvline(args.K, color="0.4", ls="--", lw=1.2,
                label=f"current K = {args.K}")
    axL.set_xlabel("mode index k")
    axL.set_ylabel("sigma_k / sigma_1  (log)")
    axL.set_title("singular value decay")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend(loc="upper right", fontsize=9)

    # RIGHT: cumulative explained variance, linear-y
    axR.plot(k_axis, cum_frac, "o-", color="C2", lw=1.2, ms=4)
    for thresh, label in ((0.99, "99%"), (0.999, "99.9%"),
                           (0.9999, "99.99%")):
        axR.axhline(thresh, color="0.7", ls=":", lw=0.9)
        axR.text(k_axis[-1], thresh, f"  {label}", va="center",
                 fontsize=8, color="0.4")
    axR.axvline(args.K, color="0.4", ls="--", lw=1.2,
                label=f"current K = {args.K}")
    if 1 <= args.K <= sigma.size:
        cap = cum_frac[args.K - 1]
        axR.annotate(f"K={args.K}: {cap * 100:.4f}%",
                     xy=(args.K, cap),
                     xytext=(args.K + 0.5,
                             min(cap + 0.02, 1.0)),
                     fontsize=10,
                     arrowprops=dict(arrowstyle="->",
                                     color="0.4", lw=0.8))
    axR.set_xlabel("mode index k")
    axR.set_ylabel("cumulative energy fraction")
    axR.set_ylim(min(cum_frac.min() * 0.99, 0.9), 1.001)
    axR.set_title("cumulative explained variance")
    axR.grid(alpha=0.3)
    axR.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"POD spectrum  |  basis: {basis_path.name}  |  "
                 f"k_cache={sigma.size}", fontsize=11)
    provenance_footer(fig, tag=args.tag,
                      basis_cache_file=basis_path,
                      extras={"K": args.K})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
