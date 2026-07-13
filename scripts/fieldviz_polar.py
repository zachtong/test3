"""Polar-unrolled field visualizations: the full (theta, r) field as
a 2D image, animated over time, plus dimension-aggregated static
error maps.

The radial_anim / kymograph views only show 5 angular slices. These
views show the ENTIRE quarter disk as a polar-unrolled image
(X = theta in [0, 90] deg, Y = r in [0, 1]), which is far more
informative and, aggregated, tells you WHERE (radius / angle /
time) the reconstruction error concentrates -- directly useful for
deciding where to place sensors.

Two products:

  render_polar_compare_anim  -- Method 1: a 3-panel animated polar
      heatmap (GT | prediction | |error|) that plays over the
      bonding process (~10 s by default, not too fast).

  render_polar_error_aggregates -- Method 2: 3 static panels, each
      collapsing ONE dimension of the |error| cube by averaging:
        (a) time-averaged   -> (theta, r) map: where in the disk
        (b) angle-averaged  -> (r, t) map:     which radius, when
        (c) radius-averaged -> (theta, t) map: which angle, when

Both build on a polar-time cube P[theta, r, t] sampled from the
canonical Cartesian field via the mask-aware bilinear sampler
(shared with the radial kymograph, so the disk arc is handled
correctly).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.viz_radial_kymograph import (              # noqa: E402
    _sample_radial_kymograph)


def polar_time_cube(f: np.ndarray, x_canon: np.ndarray,
                    y_canon: np.ndarray, n_theta: int = 91,
                    n_r: int = 128, r_max: float = 1.0,
                    r_disk: float = 1.0) -> tuple:
    """Sample the canonical field f (Nx, Ny, Nt) onto a polar grid.

    Returns (cube, thetas, rs) where cube is (n_theta, n_r, Nt),
    thetas in degrees [0, 90], rs in [0, r_max]. Uses the mask-aware
    bilinear sampler per angle so the disk arc does not leak zeros
    into oblique rays.
    """
    thetas = np.linspace(0.0, 90.0, n_theta)
    slabs = [
        _sample_radial_kymograph(
            f.astype(np.float64), x_canon, y_canon, float(th),
            n_r=n_r, r_max=r_max, r_disk=r_disk)          # (n_r, Nt)
        for th in thetas
    ]
    cube = np.stack(slabs, axis=0)                        # (n_theta, n_r, Nt)
    rs = np.linspace(0.0, r_max, n_r)
    return cube, thetas, rs


def _frame_indices(nt: int, max_frames: int) -> np.ndarray:
    if nt <= max_frames:
        return np.arange(nt)
    return np.linspace(0, nt - 1, max_frames).astype(int)


def render_polar_compare_anim(out_path, *, w_true_m, w_pred_m,
                              x_canon, y_canon,
                              sim_id="", tag="", rel_l2=None,
                              value_scale=1.0e6,
                              n_theta=91, n_r=128,
                              duration_sec=10.0, max_frames=120,
                              field_cmap="coolwarm",
                              err_cmap="magma") -> Path:
    """Method 1: 3-panel animated polar heatmap (GT | Pred | |Err|).

    X = theta [0, 90] deg, Y = r [0, 1], color = value. Plays over
    the bonding process in ~duration_sec seconds (fps derived so it
    is not too fast).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    gt, thetas, rs = polar_time_cube(w_true_m, x_canon, y_canon,
                                      n_theta=n_theta, n_r=n_r)
    pr, _, _ = polar_time_cube(w_pred_m, x_canon, y_canon,
                               n_theta=n_theta, n_r=n_r)
    err = np.abs(pr - gt)
    gt_s = gt * value_scale
    pr_s = pr * value_scale
    err_s = err * value_scale
    nt = gt_s.shape[-1]
    frames = _frame_indices(nt, max_frames)
    fps = max(1, int(round(len(frames) / max(duration_sec, 1e-3))))

    # Shared signed range for GT + Pred; error uses its own [0, max].
    finite = np.concatenate([gt_s.ravel(), pr_s.ravel()])
    finite = finite[np.isfinite(finite)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    emax = float(np.percentile(err_s[np.isfinite(err_s)], 99)) \
        if np.isfinite(err_s).any() else 1.0
    if emax <= 0:
        emax = 1.0

    ext = [thetas[0], thetas[-1], rs[0], rs[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             constrained_layout=True)
    titles = ["ground truth", "prediction", "|error|"]
    cubes = [gt_s, pr_s, err_s]
    cmaps = [field_cmap, field_cmap, err_cmap]
    vlims = [(-vmax, vmax), (-vmax, vmax), (0.0, emax)]
    ims = []
    for ax, ttl, cube, cmap, (lo, hi) in zip(
            axes, titles, cubes, cmaps, vlims):
        # display P[theta, r] with theta on X, r on Y -> transpose
        im = ax.imshow(cube[:, :, int(frames[0])].T, origin="lower",
                       aspect="auto", extent=ext, cmap=cmap,
                       vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xlabel("theta (deg)")
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)
    axes[0].set_ylabel("r (normalized)")

    def _suptitle(ti):
        parts = [p for p in (tag, sim_id) if p]
        head = "  |  ".join(parts)
        rl = f"  |  rel-L2={rel_l2:.4f}" if rel_l2 is not None else ""
        return (f"{head}{rl}  |  polar field (u_z * {value_scale:g})"
                f"  |  t-idx {ti}/{nt - 1}")

    fig.suptitle(_suptitle(int(frames[0])), fontsize=11)

    def update(k):
        ti = int(frames[k])
        for im, cube in zip(ims, cubes):
            im.set_data(cube[:, :, ti].T)
        fig.suptitle(_suptitle(ti), fontsize=11)
        return ims

    print(f"rendering {len(frames)} polar-anim frames at {fps} fps "
          f"(~{len(frames) / fps:.1f}s) -> {out_path}", flush=True)
    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return Path(out_path)


def render_polar_error_aggregates(out_path, *, w_true_m, w_pred_m,
                                  x_canon, y_canon,
                                  sim_id="", tag="", rel_l2=None,
                                  value_scale=1.0e6,
                                  n_theta=91, n_r=128,
                                  cmap="magma") -> Path:
    """Method 2: 3 static panels aggregating the |error| cube by
    averaging one dimension each.

      (a) mean over time   -> (theta, r): where in the disk
      (b) mean over theta  -> (r, t):     which radius, when
      (c) mean over radius -> (theta, t): which angle, when

    Reveals, in the average sense, at what radius / angle / time the
    reconstruction error is largest -- a sensor-placement guide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, thetas, rs = polar_time_cube(w_true_m, x_canon, y_canon,
                                      n_theta=n_theta, n_r=n_r)
    pr, _, _ = polar_time_cube(w_pred_m, x_canon, y_canon,
                               n_theta=n_theta, n_r=n_r)
    err = np.abs(pr - gt) * value_scale                  # (theta, r, t)
    nt = err.shape[-1]
    t_axis = np.linspace(0.0, 1.0, nt)

    m_time = err.mean(axis=2)                             # (theta, r)
    m_angle = err.mean(axis=0)                            # (r, t)
    m_radius = err.mean(axis=1)                           # (theta, t)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0),
                             constrained_layout=True)

    # (a) time-averaged: theta on X, r on Y
    im0 = axes[0].imshow(m_time.T, origin="lower", aspect="auto",
                         extent=[thetas[0], thetas[-1],
                                 rs[0], rs[-1]],
                         cmap=cmap, interpolation="nearest")
    axes[0].set_xlabel("theta (deg)")
    axes[0].set_ylabel("r (normalized)")
    axes[0].set_title("mean |error| over time\n(where in the disk)",
                      fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # (b) angle-averaged: time on X, r on Y
    im1 = axes[1].imshow(m_angle, origin="lower", aspect="auto",
                         extent=[t_axis[0], t_axis[-1],
                                 rs[0], rs[-1]],
                         cmap=cmap, interpolation="nearest")
    axes[1].set_xlabel("t (normalized)")
    axes[1].set_ylabel("r (normalized)")
    axes[1].set_title("mean |error| over angle\n(which radius, when)",
                      fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # (c) radius-averaged: time on X, theta on Y
    im2 = axes[2].imshow(m_radius, origin="lower", aspect="auto",
                         extent=[t_axis[0], t_axis[-1],
                                 thetas[0], thetas[-1]],
                         cmap=cmap, interpolation="nearest")
    axes[2].set_xlabel("t (normalized)")
    axes[2].set_ylabel("theta (deg)")
    axes[2].set_title("mean |error| over radius\n(which angle, when)",
                      fontsize=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    parts = [p for p in (tag, sim_id) if p]
    head = "  |  ".join(parts)
    rl = f"  |  rel-L2={rel_l2:.4f}" if rel_l2 is not None else ""
    fig.suptitle(f"{head}{rl}  |  error aggregates "
                 f"(|u_z err| * {value_scale:g})", fontsize=12)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
