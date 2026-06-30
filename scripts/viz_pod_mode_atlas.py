"""POD mode atlas: 2 x (K/2) full-disk heatmaps of phi_k(x, y).

Each panel shows one POD basis function reshaped to (Nx, Ny), D2-
mirrored to the full disk, with a circular off-disk mask. Per-panel
normalisation so weak modes (low sigma) are still readable; this
trades amplitude legibility for STRUCTURE legibility -- the talk
audience cares about "what does the spatial pattern look like" more
than "how big is mode 6 vs mode 1".

sigma_k / sigma_1 is printed in each panel's corner so the relative
weight is not lost.

This is the figure the user looks at to test the angular-blind-spot
hypothesis for mode 6: if phi_6 has a clear cos(6*theta) pattern (six
lobes around the wafer), the lab-rig sensor configuration at theta =
0/45/90 cannot disambiguate it from its mirror image, explaining the
high per-mode prediction error.

Input: a basis_cache .npz file (Phi (Nx*Ny, K), sigma (K,),
spatial_shape (Nx, Ny)).

    python scripts/viz_pod_mode_atlas.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --K 8 --out viz/mode_atlas.png
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.fieldviz import (mirror_d2, render_full_disk,    # noqa: E402
                               provenance_footer, WAFER_CMAP,
                               wafer_value_range)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True)
    ap.add_argument("--K", type=int, default=8,
                    help="number of modes to render (default 8). Must "
                    "be <= k_cache stored in the basis file.")
    ap.add_argument("--per-panel-norm", action="store_true", default=True,
                    help="default ON: each panel gets its own colour "
                    "range so weak modes are still readable")
    ap.add_argument("--shared-norm", action="store_true",
                    help="opposite of --per-panel-norm: shared cmap "
                    "across all panels (then weak modes look flat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--symmetric", action="store_true",
                    help="symmetric vmin/vmax centred on 0 per panel "
                    "(default: asymmetric percentile clip)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    basis_path = Path(args.basis).expanduser().resolve()
    if not basis_path.is_file():
        print(f"basis file not found: {basis_path}", file=sys.stderr)
        return 2
    with np.load(basis_path, allow_pickle=False) as z:
        Phi = np.asarray(z["Phi"])
        sigma = np.asarray(z["sigma"], dtype=np.float64)
        spatial_shape = tuple(int(d) for d in z["spatial_shape"])
    n_space, k_cache = Phi.shape
    nx, ny = spatial_shape
    if nx * ny != n_space:
        print(f"shape mismatch: Phi {Phi.shape} vs spatial_shape "
              f"{spatial_shape}", file=sys.stderr)
        return 1
    K = min(args.K, k_cache)
    if K != args.K:
        print(f"WARN: basis has k_cache={k_cache}; rendering K={K}")

    # layout: 2 rows x ceil(K/2) cols
    n_cols = (K + 1) // 2
    n_rows = 2 if K > 1 else 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 3.6 * n_rows),
                             constrained_layout=True,
                             squeeze=False)
    x_canon = np.linspace(0.0, 1.0, nx)
    y_canon = np.linspace(0.0, 1.0, ny)

    use_shared = bool(args.shared_norm)
    if use_shared:
        # symmetric global range across all rendered modes
        all_phi = Phi[:, :K]
        v = float(np.percentile(np.abs(all_phi), 99))
        if v == 0:
            v = 1.0
        global_vmin, global_vmax = -v, v

    for k in range(K):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r, c]
        phi = Phi[:, k].reshape(nx, ny)
        if use_shared:
            vmin, vmax = global_vmin, global_vmax
        else:
            if args.symmetric:
                v = float(np.percentile(np.abs(phi), 99))
                if v == 0:
                    v = 1.0
                vmin, vmax = -v, v
            else:
                vmin, vmax = wafer_value_range(
                    phi, clip_positive_to_zero=False)
        render_full_disk(ax, phi, x_canon, y_canon,
                         cmap=WAFER_CMAP, vmin=vmin, vmax=vmax,
                         mirror=True, mask_off_disk=True)
        ax.set_title(f"phi_{k + 1}", fontsize=11)
        ax.text(0.02, 0.95, f"sigma_{k + 1}/sigma_1 = "
                            f"{sigma[k] / sigma[0]:.3g}",
                transform=ax.transAxes, fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=2,
                          edgecolor="none"),
                verticalalignment="top")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        if c == 0:
            ax.set_ylabel("y")
        if r == n_rows - 1:
            ax.set_xlabel("x")

    # Hide unused subplots if K is odd
    for k in range(K, n_rows * n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r, c].set_visible(False)

    fig.suptitle(f"POD mode atlas  |  basis: {basis_path.name}  |  "
                 f"K={K}/{k_cache}  |  "
                 f"{'shared' if use_shared else 'per-panel'} norm",
                 fontsize=12)
    provenance_footer(fig, tag=args.tag, basis_cache_file=basis_path,
                      extras={"K": K})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
