"""Visualize 3D gap fields: spatial heatmap snapshots + time animation.

The 2D codebase's `fieldviz.py` produces 1D profile animations (w(r) over t)
and 1D kymographs (r-t imshows). In 3D the natural primitives change: the
field at any t is a 2D map over (x, y), so we plot:

  - `snapshot_panel`: a 1xK panel of (x, y) heatmaps at K selected times.
  - `field_animation`: an animated GIF of the (x, y) heatmap evolving over t,
    with sensor positions overlaid as markers.
  - `triptych`: side-by-side (GT, prediction, abs error) at a chosen t.

STUB / scaffold: the function signatures are committed but the matplotlib
draw bodies are kept minimal. They will be expanded once a real prediction
exists to plot against.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np


def snapshot_panel(field: np.ndarray, t_idx: Iterable[int],
                   out_path: Path, x_canon: np.ndarray, y_canon: np.ndarray,
                   sensor_xy: np.ndarray | None = None,
                   value_scale: float = 1.0e6, title: str = "") -> None:
    """Save a 1xK heatmap panel: field at each selected time index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_idx = list(t_idx)
    extent = [x_canon[0], x_canon[-1], y_canon[0], y_canon[-1]]
    F = field * value_scale
    vmin, vmax = float(np.nanmin(F)), float(np.nanmax(F))
    fig, axes = plt.subplots(1, len(t_idx), figsize=(3.2 * len(t_idx), 3.4),
                             constrained_layout=True, squeeze=False)
    for ax, k in zip(axes[0], t_idx):
        im = ax.imshow(F[..., k].T, origin="lower", aspect="equal",
                       extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"t-idx {k}")
        ax.set_xlabel("x")
        if sensor_xy is not None:
            ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1],
                       s=18, marker="x", c="red")
    axes[0, 0].set_ylabel("y")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes[0, :], shrink=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def field_animation(field: np.ndarray, x_canon: np.ndarray,
                    y_canon: np.ndarray, out_path: Path,
                    sensor_xy: np.ndarray | None = None,
                    frame_stride: int = 1, fps: int = 8,
                    value_scale: float = 1.0e6, title: str = "") -> None:
    """STUB: render an (x, y) heatmap animation over t to a GIF.

    Intended implementation: matplotlib FuncAnimation + PillowWriter, with
    a fixed vmin/vmax derived from the full field range so frames are
    comparable. Sensor positions overlaid as red x markers.
    """
    raise NotImplementedError(
        "evaluation/fieldviz.py::field_animation: deferred until a real "
        "prediction exists to render. Use snapshot_panel meanwhile.")


def triptych(gt: np.ndarray, pred: np.ndarray, x_canon: np.ndarray,
             y_canon: np.ndarray, t_idx: int, out_path: Path,
             sensor_xy: np.ndarray | None = None,
             value_scale: float = 1.0e6, title: str = "") -> None:
    """Save a 1x3 GT / prediction / abs-error heatmap at one time index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [x_canon[0], x_canon[-1], y_canon[0], y_canon[-1]]
    G = gt[..., t_idx] * value_scale
    P = pred[..., t_idx] * value_scale
    E = np.abs(G - P)
    lo, hi = float(min(G.min(), P.min())), float(max(G.max(), P.max()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, M, name, kw in (
            (axes[0], G, "ground truth", dict(vmin=lo, vmax=hi)),
            (axes[1], P, "prediction",   dict(vmin=lo, vmax=hi)),
            (axes[2], E, "abs error",    {})):
        im = ax.imshow(M.T, origin="lower", aspect="equal", extent=extent,
                       cmap="viridis", **kw)
        ax.set_title(name)
        ax.set_xlabel("x")
        if sensor_xy is not None:
            ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=18,
                       marker="x", c="red")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("y")
    fig.suptitle(f"{title}  t-idx {t_idx}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
