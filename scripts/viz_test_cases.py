"""Visualize test-set predictions vs GT for selected sims.

Per selected sim: one PNG containing
  TOP ROW (3 full-disk heatmaps at t*):
    - GT field
    - Model-predicted field
    - |GT - prediction| (absolute error)
  BOTTOM ROW (K subplots):
    - a_k(t) predicted vs true line plot for each of the K POD modes,
      so the operator can see which mode mispredictions are driving
      the field error.

t* is the time index where |GT - prediction| peaks; this is the most
informative single snapshot to render. Files are named with the
per-sim rel-L2 + sim basename so they sort naturally:
  <slot>_relL2<rel_l2>_<sim_basename>.png

FOUR SELECTION MODES (mutually exclusive):

  --pick worst        top-N by descending field error (default)
  --pick best         top-N by ascending field error
  --pick median       N centered around the median field error
  --pick random       N random test sims (deterministic --seed)
  --sim BASENAME[,BASENAME,...]
                      specific basenames; overrides --pick and --topn

FIRST RUN is heavyweight: load dataset (93 GB), rebuild split, load
basis, load 3 seed checkpoints, run inference. Cost ~1-2 minutes
after loader cache warm, ~5-30 minutes cold.

RE-RUNS ARE FAST via the test-cases cache: a hit reads pre-computed
(w_pred, w_true, a_pred, a_true) tensors for the selected sims from
outputs/<tag>/_test_<pick_key>_<ckpt_hash>.npz (~100 MB) and renders
in seconds. Cache key includes pick mode + seed + basename set +
checkpoint fingerprint; different picks / seeds coexist as separate
cache files under the same tag.

    # backward-compatible default: top-5 worst
    python scripts/viz_test_cases.py --tag firehorse2_n3_full \\
        --topn 5 --out viz/worst/

    # top-5 best -- see how the model does when it does well
    python scripts/viz_test_cases.py --tag firehorse2_n3_full \\
        --pick best --topn 5 --out viz/best/

    # 3 typical sims around the median error
    python scripts/viz_test_cases.py --tag firehorse2_n3_full \\
        --pick median --topn 3 --out viz/median/

    # 5 random test sims (seed 0)
    python scripts/viz_test_cases.py --tag firehorse2_n3_full \\
        --pick random --topn 5 --seed 0 --out viz/random/

    # one specific sim by basename
    python scripts/viz_test_cases.py --tag firehorse2_n3_full \\
        --sim run_00473.npz --out viz/that_one/
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.fieldviz import (render_full_disk, provenance_footer,  # noqa: E402
                               WAFER_CMAP, SENSOR_MARKER_COLOR,
                               wafer_value_range, mirror_d2,
                               wafer_cmap_to_plotly)
from scripts.fieldviz.render3d import estimate_lower_z             # noqa: E402


_TEST_CACHE_PREFIX = "_test_"
_TEST_CACHE_VERSION = 2   # v2 = new schema with pick/seed/sim key
_VALID_PICKS = ("worst", "best", "median", "random")


def _checkpoint_fingerprint(output_dir: str, tag: str,
                             seeds: list) -> str:
    """Sha of the seed checkpoint files' (size, mtime). Cheap and
    triggers cache invalidation whenever any checkpoint changes."""
    from training.checkpoint import checkpoint_path
    parts = []
    for seed in sorted(seeds):
        cp = checkpoint_path(output_dir, tag, seed)
        if cp.exists():
            st = cp.stat()
            parts.append(f"{seed}:{st.st_size}:{int(st.st_mtime)}")
        else:
            parts.append(f"{seed}:missing")
    if not parts:
        return "no_checkpoints"
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _selection_key(pick: str | None, topn: int | None,
                    seed: int | None,
                    sim_basenames: list | None) -> str:
    """8-char hash of the selection specification. --sim takes
    precedence; otherwise pick/topn/seed determine the key."""
    if sim_basenames:
        payload = {"sims": sorted(sim_basenames)}
    else:
        payload = {"pick": pick, "topn": topn, "seed": seed}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _test_cache_path(output_dir: str, tag: str, ckpt_fp: str,
                     sel_key: str) -> Path:
    return (Path(output_dir) / tag /
            f"{_TEST_CACHE_PREFIX}{sel_key}_{ckpt_fp}.npz")


def _load_test_cache(path: Path):
    """Return a predict_run_fields-shaped dict on hit, None on miss."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d.get("version", -1)) != _TEST_CACHE_VERSION:
                return None
            return dict(
                x_canon=d["x_canon"], y_canon=d["y_canon"],
                t=d["t"], sensor_xy=d["sensor_xy"],
                K=int(d["K"]),
                idx=d["idx"].astype(int),
                w_pred=d["w_pred"].astype(np.float64),
                w_true=d["w_true"].astype(np.float64),
                a_pred=d["a_pred"].astype(np.float64),
                a_true=d["a_true"].astype(np.float64),
                basenames=json.loads(str(d["basenames_json"])))
    except (OSError, ValueError, KeyError) as e:
        print(f"  test-cases cache read failed "
              f"({type(e).__name__}: {e}) -- rebuilding",
              flush=True)
        return None


def _save_test_cache(path: Path, payload: dict) -> None:
    """Atomic write. Fields stored float32 to keep the file ~100 MB
    for topn=5 on 128x128x300; loader auto-promotes to float64."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        version=np.int32(_TEST_CACHE_VERSION),
        x_canon=payload["x_canon"].astype(np.float64),
        y_canon=payload["y_canon"].astype(np.float64),
        t=payload["t"].astype(np.float64),
        sensor_xy=payload["sensor_xy"].astype(np.float64),
        K=np.int32(payload["K"]),
        idx=np.asarray(payload["idx"], dtype=np.int64),
        w_pred=payload["w_pred"].astype(np.float32),
        w_true=payload["w_true"].astype(np.float32),
        a_pred=payload["a_pred"].astype(np.float32),
        a_true=payload["a_true"].astype(np.float32),
        basenames_json=np.array(json.dumps(list(payload["basenames"]))))
    tmp.replace(path)


def _pick_indices(field_errs: np.ndarray, pick: str, topn: int,
                   seed: int) -> np.ndarray:
    """Return an array of test_idx per the --pick strategy.

    'worst'  -- topn largest errors, descending
    'best'   -- topn smallest errors, ascending
    'median' -- topn centered around the median position, sorted
                by error ascending
    'random' -- topn drawn uniformly at random (rng seeded)
    """
    n = field_errs.size
    topn = int(min(topn, n))
    if pick == "worst":
        return np.argsort(field_errs)[-topn:][::-1]
    if pick == "best":
        return np.argsort(field_errs)[:topn]
    if pick == "median":
        order = np.argsort(field_errs)
        centre = n // 2
        half = topn // 2
        lo = max(0, centre - half)
        hi = min(n, lo + topn)
        if hi - lo < topn:
            lo = max(0, hi - topn)
        return order[lo:hi]
    if pick == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=topn, replace=False))
    raise ValueError(f"unknown --pick {pick!r}, valid: {_VALID_PICKS}")


def _render_snapshot(out_path, *, gt, pr, err, a_pred, a_true, t, K,
                     x_canon, y_canon, sensor_xy, sim_id, tag,
                     rel_l2, test_idx, sel_label, results_path):
    """Legacy 2-row layout: (GT / pred / |err|) at t* on the top row +
    K subplots of a_k(t) on the bottom. Kept identical to the pre-
    --layout behavior so 'snapshot' outputs are unchanged."""
    import matplotlib.pyplot as plt

    err_integrated = err.sum(axis=(0, 1))
    t_star = int(np.argmax(err_integrated))
    vmin, vmax = wafer_value_range(
        np.stack([gt[..., t_star], pr[..., t_star]]))
    err_max = float(np.percentile(err[..., t_star], 99))
    if err_max <= 0:
        err_max = max(float(err.max()), 1e-12)

    n_cols = max(K, 3)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(3.5 * n_cols, 7.2),
                             constrained_layout=True)
    for c, (panel_data, title, cmap, panel_vmin, panel_vmax) in enumerate([
            (gt[..., t_star], "GT", WAFER_CMAP, vmin, vmax),
            (pr[..., t_star], "predicted", WAFER_CMAP, vmin, vmax),
            (err[..., t_star], "abs error", "viridis", 0, err_max),
    ]):
        ax = axes[0, c]
        render_full_disk(ax, panel_data, x_canon, y_canon,
                         cmap=cmap, vmin=panel_vmin, vmax=panel_vmax,
                         mirror=True, mask_off_disk=True,
                         sensor_xy=sensor_xy)
        ax.set_title(f"{title}  t-idx {t_star}/{gt.shape[-1] - 1}",
                     fontsize=10)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    for c in range(3, n_cols):
        axes[0, c].set_visible(False)

    for k_idx in range(K):
        c = k_idx
        ax = axes[1, c]
        ax.plot(t, a_true[k_idx], color="0.3", lw=1.2, label="true")
        ax.plot(t, a_pred[k_idx], color=SENSOR_MARKER_COLOR,
                lw=1.2, ls="--", label="predicted")
        err_k = float(np.linalg.norm(a_pred[k_idx] - a_true[k_idx])
                      / max(np.linalg.norm(a_true[k_idx]), 1e-12))
        ax.set_title(f"a_{k_idx + 1}  rel-L2={err_k:.3f}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("normalized t")
        if c == 0:
            ax.legend(fontsize=7, loc="best")
    for c in range(K, n_cols):
        axes[1, c].set_visible(False)

    fig.suptitle(
        f"{tag}  |  {sel_label}  |  test_idx={int(test_idx)}  |  "
        f"rel-L2={rel_l2:.4f}  |  {sim_id}", fontsize=11)
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      results_file=results_path,
                      extras={"test_idx": int(test_idx),
                              "rel_l2": f"{rel_l2:.4f}",
                              "t_star": t_star,
                              "sel": (sel_label.split()[0]),
                              "layout": "snapshot"})
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _render_radial_anim(out_path, *, w_true_m, w_pred_m,
                        x_canon, y_canon, angles,
                        sim_id, tag, rel_l2, test_idx, sel_label,
                        results_path, value_scale=1.0e6,
                        fps=18, max_frames=60):
    """Animated 1D radial-slice comparison of GT vs predicted.

    Layout: one subplot per angle in `angles` (default 0 / 45 / 90
    deg), side by side. Each subplot: x = r in [0, 1], y = u_z
    (per-sim locked), one solid line for GT and one dashed line for
    the prediction. As time advances the two curves morph together
    -- ideal for showing at a glance whether the model tracks the
    descending upper wafer along each radial ray in real-time.

    Output: GIF (PillowWriter, no ffmpeg needed).
    """
    from scripts.viz_radial_kymograph import _sample_radial_kymograph
    from data.loader import _DISK_MASK_R_END
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_r = w_true_m.shape[0]
    nt = w_true_m.shape[-1]
    # Sample only inside the loader's rim mask so the curve does not
    # dip back to zero at r ~ 1.0 (which the loader zeroes).
    r_stop = _DISK_MASK_R_END
    gts = [_sample_radial_kymograph(
        w_true_m.astype(np.float64), x_canon, y_canon, th,
        n_r=n_r, r_max=r_stop)
        for th in angles]
    prs = [_sample_radial_kymograph(
        w_pred_m.astype(np.float64), x_canon, y_canon, th,
        n_r=n_r, r_max=r_stop)
        for th in angles]
    scaled_gts = [g * value_scale for g in gts]
    scaled_prs = [p * value_scale for p in prs]

    # Per-sim locked y-limits so descent is monotonic on screen.
    all_signed = np.concatenate(
        [g.ravel() for g in scaled_gts]
        + [p.ravel() for p in scaled_prs])
    ymin, ymax = wafer_value_range(all_signed)
    span = max(ymax - ymin, 1.0)
    ymin -= 0.05 * span
    ymax += 0.05 * span

    if nt > max_frames:
        frame_idx = np.linspace(0, nt - 1, max_frames).astype(int)
    else:
        frame_idx = np.arange(nt)
    r_axis = np.linspace(0.0, r_stop, n_r)

    fig, axes = plt.subplots(
        1, len(angles),
        figsize=(4.5 * len(angles), 5.2),
        sharex=True, sharey=True, constrained_layout=True)
    if len(angles) == 1:
        axes = [axes]

    lines_gt = []
    lines_pr = []
    for i, (ax, th) in enumerate(zip(axes, angles)):
        lg, = ax.plot(r_axis, scaled_gts[i][:, int(frame_idx[0])],
                       color="0.2", lw=2.2, label="GT")
        lp, = ax.plot(r_axis, scaled_prs[i][:, int(frame_idx[0])],
                       color=SENSOR_MARKER_COLOR, lw=2.0, ls="--",
                       label="predicted")
        lines_gt.append(lg)
        lines_pr.append(lp)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("r (normalized)")
        ax.set_title(f"theta = {th:g} deg", fontsize=10)
        ax.grid(alpha=0.3)
        # Zero line as a visual reference for the rest state.
        ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    axes[0].set_ylabel(f"u_z * {value_scale:g}")
    axes[0].legend(fontsize=9, loc="lower right")

    def _title_at(t_idx: int) -> str:
        return (f"{tag}  |  {sel_label}  |  test_idx="
                f"{int(test_idx)}  |  rel-L2={rel_l2:.4f}  |  "
                f"{sim_id}  |  t-idx {t_idx}/{nt - 1}")

    fig.suptitle(_title_at(int(frame_idx[0])), fontsize=11)

    def update(i):
        t_idx = int(frame_idx[i])
        for lg, lp, gt, pr in zip(
                lines_gt, lines_pr, scaled_gts, scaled_prs):
            lg.set_ydata(gt[:, t_idx])
            lp.set_ydata(pr[:, t_idx])
        fig.suptitle(_title_at(t_idx), fontsize=11)
        return lines_gt + lines_pr

    print(f"rendering {len(frame_idx)} radial-anim frames at "
          f"{fps} fps -> {out_path}", flush=True)
    anim = FuncAnimation(fig, update, frames=len(frame_idx),
                          interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      results_file=results_path,
                      extras={"idx": int(test_idx),
                              "sel": (sel_label.split()[0]),
                              "layout": "radial_anim"})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)


def _render_interactive_compare(out_path, *, w_true_m, w_pred_m,
                                  x_canon, y_canon, sensor_xy,
                                  sim_id, tag, rel_l2, test_idx,
                                  sel_label, results_path,
                                  value_scale=1.0e6, max_frames=60,
                                  show_lower=False):
    """3-subplot Plotly HTML: GT / Predicted / |GT - Pred| as 3D
    surfaces sharing a time slider. Full disk via D2 mirror. Camera
    preset buttons switch all 3 scenes at once (Iso / Top / Side).
    Sensor markers appear on GT and Pred; the error panel has no
    sensors (they would clutter the artifact view).

    Output: standalone HTML embedding plotly.js from CDN. Viewable in
    any browser with WebGL (the interactive layout's whole point). No
    WebGL fallback -- if the operator does not have WebGL they should
    use the kymo / radial_anim / snapshot layouts instead."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  interactive_compare: plotly not installed; skipping",
              flush=True)
        return
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        print(f"  interactive_compare: plotly.subplots missing; "
              f"skipping", flush=True)
        return

    Nt = w_true_m.shape[-1]
    if Nt > max_frames:
        frame_idx = np.linspace(0, Nt - 1, max_frames).astype(int)
    else:
        frame_idx = np.arange(Nt)

    # Full disk axes
    x_full = np.concatenate([-x_canon[:0:-1], x_canon])
    y_full = np.concatenate([-y_canon[:0:-1], y_canon])
    X, Y = np.meshgrid(x_full, y_full, indexing="ij")
    # Match loader rim mask (0.99) so the interactive_compare view
    # does not show a waterfall drop at r=1 where the loader zeroed.
    from data.loader import _DISK_MASK_R_END
    in_disk = (X * X + Y * Y) <= _DISK_MASK_R_END * _DISK_MASK_R_END

    def _mask_scale(w_3d):
        out = []
        for t in frame_idx:
            full = mirror_d2(w_3d[..., int(t)]) * value_scale
            full = full.astype(np.float64)
            full[~in_disk] = np.nan
            out.append(full)
        return out

    z_gt = _mask_scale(w_true_m)
    z_pr = _mask_scale(w_pred_m)
    z_err = _mask_scale(np.abs(w_true_m - w_pred_m))

    # Shared color range for GT+Pred, so the eye compares apples to
    # apples across those two panels. Error gets its own viridis
    # scale starting at 0.
    finite = np.concatenate(
        [z[np.isfinite(z)].ravel() for z in z_gt]
        + [z[np.isfinite(z)].ravel() for z in z_pr])
    vmin, vmax = wafer_value_range(finite)
    finite_err = np.concatenate(
        [z[np.isfinite(z)].ravel() for z in z_err])
    err_max = (float(np.percentile(finite_err, 99))
                if finite_err.size else 1.0)
    if err_max <= 0:
        err_max = 1e-12
    wafer_scale = wafer_cmap_to_plotly()

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"},
                 {"type": "surface"}]],
        subplot_titles=("GT", "Predicted", "|GT - Pred|"),
        horizontal_spacing=0.02)

    # Initial traces (frame 0). Each row=1, col=N assignment routes
    # the trace to sceneN under the hood.
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_gt[0],
        cmin=vmin, cmax=vmax, colorscale=wafer_scale,
        colorbar=dict(x=0.28, thickness=15,
                       title=f"u_z * {value_scale:g}"),
        hovertemplate=("GT<br>x=%{x:.2f} y=%{y:.2f} "
                       "u_z=%{z:.3g}<extra></extra>")),
        row=1, col=1)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_pr[0],
        cmin=vmin, cmax=vmax, colorscale=wafer_scale,
        showscale=False,
        hovertemplate=("Pred<br>x=%{x:.2f} y=%{y:.2f} "
                       "u_z=%{z:.3g}<extra></extra>")),
        row=1, col=2)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_err[0],
        cmin=0.0, cmax=err_max, colorscale="Viridis",
        colorbar=dict(x=1.0, thickness=15,
                       title=f"|err| * {value_scale:g}"),
        hovertemplate=("|err|<br>x=%{x:.2f} y=%{y:.2f} "
                       "val=%{z:.3g}<extra></extra>")),
        row=1, col=3)

    # Sensor markers on GT and Pred panels (skip error to reduce
    # clutter). Sensors are static, so keep them out of frames data
    # so they persist during playback.
    if sensor_xy is not None and len(sensor_xy):
        for col, z_arr, scene_key in [(1, z_gt[0], "scene"),
                                        (2, z_pr[0], "scene2")]:
            sz = []
            for sx, sy in sensor_xy:
                ix = int(np.argmin(np.abs(X[:, 0] - sx)))
                iy = int(np.argmin(np.abs(Y[0, :] - sy)))
                v = z_arr[ix, iy]
                sz.append(float(v) if np.isfinite(v) else 0.0)
            fig.add_trace(go.Scatter3d(
                x=sensor_xy[:, 0], y=sensor_xy[:, 1], z=sz,
                mode="markers",
                marker=dict(size=6, color="magenta", symbol="x"),
                name=f"sensors ({['GT', 'Pred'][col - 1]})",
                showlegend=(col == 1)),
                row=1, col=col)

    # Optional lower-wafer plane on GT and Pred panels (static).
    if show_lower:
        lower_z = estimate_lower_z(np.asarray(w_true_m, dtype=np.float64))
        Z_lower = np.full_like(X, lower_z * value_scale, dtype=np.float64)
        Z_lower[~in_disk] = np.nan
        for col in (1, 2):
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z_lower,
                surfacecolor=np.zeros_like(Z_lower),
                colorscale=[[0, "rgb(140,140,140)"],
                             [1, "rgb(140,140,140)"]],
                showscale=False, opacity=0.30,
                name="lower wafer", showlegend=(col == 1),
                hovertemplate="lower wafer<extra></extra>"),
                row=1, col=col)

    # Animation frames: each frame replaces the 3 surface traces
    # (indices 0, 1, 2). Sensors + lower-wafer are indices 3+ and
    # stay static.
    frames = []
    for i, t in enumerate(frame_idx):
        frames.append(go.Frame(
            data=[
                go.Surface(z=z_gt[i], x=X, y=Y,
                            cmin=vmin, cmax=vmax,
                            colorscale=wafer_scale, showscale=True),
                go.Surface(z=z_pr[i], x=X, y=Y,
                            cmin=vmin, cmax=vmax,
                            colorscale=wafer_scale, showscale=False),
                go.Surface(z=z_err[i], x=X, y=Y,
                            cmin=0.0, cmax=err_max,
                            colorscale="Viridis", showscale=True),
            ],
            traces=[0, 1, 2],
            name=str(int(t))))
    fig.frames = frames

    # Camera presets applied to all 3 scenes simultaneously.
    iso = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    top = dict(eye=dict(x=0.01, y=0.01, z=2.5))
    side = dict(eye=dict(x=2.5, y=0.01, z=0.01))
    z_range_wafer = [vmin * 1.1, max(vmax * 1.1, abs(vmin) * 0.1)]
    z_range_err = [0.0, err_max * 1.1]

    def _cam_args(cam):
        return [{"scene.camera": cam,
                  "scene2.camera": cam,
                  "scene3.camera": cam}]

    fig.update_layout(
        title=dict(text=(f"{tag}  |  {sel_label}  |  "
                          f"test_idx={int(test_idx)}  |  "
                          f"rel-L2={rel_l2:.4f}  |  {sim_id}"),
                    x=0.5, xanchor="center"),
        scene=dict(camera=iso,
                    aspectratio=dict(x=1, y=1, z=0.4),
                    xaxis=dict(range=[-1.05, 1.05]),
                    yaxis=dict(range=[-1.05, 1.05]),
                    zaxis=dict(range=z_range_wafer,
                                title=f"u_z*{value_scale:g}")),
        scene2=dict(camera=iso,
                     aspectratio=dict(x=1, y=1, z=0.4),
                     xaxis=dict(range=[-1.05, 1.05]),
                     yaxis=dict(range=[-1.05, 1.05]),
                     zaxis=dict(range=z_range_wafer,
                                 title=f"u_z*{value_scale:g}")),
        scene3=dict(camera=iso,
                     aspectratio=dict(x=1, y=1, z=0.4),
                     xaxis=dict(range=[-1.05, 1.05]),
                     yaxis=dict(range=[-1.05, 1.05]),
                     zaxis=dict(range=z_range_err,
                                 title=f"|err|*{value_scale:g}")),
        updatemenus=[
            dict(type="buttons", direction="left",
                  x=0.5, xanchor="center", y=1.10, yanchor="bottom",
                  showactive=False, pad=dict(t=0, b=6),
                  buttons=[
                      dict(label="Iso", method="relayout",
                            args=_cam_args(iso)),
                      dict(label="Top", method="relayout",
                            args=_cam_args(top)),
                      dict(label="Side", method="relayout",
                            args=_cam_args(side)),
                  ]),
            dict(type="buttons", direction="left",
                  x=0.02, xanchor="left", y=1.10, yanchor="bottom",
                  showactive=False,
                  buttons=[
                      dict(label="Play", method="animate",
                            args=[None,
                                  dict(frame=dict(duration=100,
                                                    redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=0))]),
                      dict(label="Pause", method="animate",
                            args=[[None],
                                  dict(frame=dict(duration=0,
                                                    redraw=False),
                                        mode="immediate")]),
                  ]),
        ],
        sliders=[dict(
            active=0, currentvalue=dict(prefix="t-idx: "),
            steps=[dict(method="animate", label=str(int(t)),
                          args=[[str(int(t))],
                                dict(frame=dict(duration=0,
                                                  redraw=True),
                                     mode="immediate")])
                     for t in frame_idx])],
        margin=dict(l=10, r=10, t=100, b=10),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn",
                    full_html=True, auto_play=False)


def _render_kymo_compare(out_path, *, w_true_m, w_pred_m, x_canon,
                          y_canon, angles, sim_id, tag, rel_l2,
                          test_idx, sel_label, results_path,
                          value_scale=1.0e6):
    """3-row x 3-col kymograph comparison.

    Rows: theta = angles (default 0 / 45 / 90 deg)
    Cols: GT / predicted / |GT - predicted|
    Axes per panel: x = normalized time, y = r in [0, 1]

    Shares one WAFER_CMAP vmin/vmax across GT+pred (per sim, all
    angles combined) so the eye compares apples to apples across the
    six left+middle panels. Error column gets its own viridis scale
    with vmin=0.

    Inputs are in physical units (metres); value_scale (default 1e6
    for micrometres) is applied inside for display + colorbar labels.
    """
    from scripts.viz_radial_kymograph import _sample_radial_kymograph
    from data.loader import _DISK_MASK_R_END
    import matplotlib.pyplot as plt

    n_r = w_true_m.shape[0]
    # Sample only inside the loader's rim mask so kymo does not show
    # a hard 0-line at r ~ 1.0.
    r_stop = _DISK_MASK_R_END
    kymos_gt = [_sample_radial_kymograph(
        w_true_m.astype(np.float64), x_canon, y_canon, th,
        n_r=n_r, r_max=r_stop)
        for th in angles]
    kymos_pr = [_sample_radial_kymograph(
        w_pred_m.astype(np.float64), x_canon, y_canon, th,
        n_r=n_r, r_max=r_stop)
        for th in angles]
    scaled_gt = [k * value_scale for k in kymos_gt]
    scaled_pr = [k * value_scale for k in kymos_pr]
    scaled_err = [np.abs(g - p) for g, p in zip(scaled_gt, scaled_pr)]

    all_signed = np.concatenate(
        [k.ravel() for k in scaled_gt]
        + [k.ravel() for k in scaled_pr])
    vmin, vmax = wafer_value_range(all_signed)
    all_err = np.concatenate([k.ravel() for k in scaled_err])
    err_max = float(np.percentile(all_err, 99))
    if err_max <= 0:
        err_max = max(float(all_err.max()), 1e-12)

    nt = w_true_m.shape[-1]
    t_axis = np.linspace(0.0, 1.0, nt)
    r_axis = np.linspace(0.0, r_stop, n_r)
    n_rows = len(angles)
    fig, axes = plt.subplots(n_rows, 3,
                              figsize=(12.0, 3.0 * n_rows + 0.6),
                              sharex=True, sharey=True,
                              constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]
    ims_col = [None, None, None]
    for i, th in enumerate(angles):
        panels = [(scaled_gt[i], WAFER_CMAP, vmin, vmax),
                  (scaled_pr[i], WAFER_CMAP, vmin, vmax),
                  (scaled_err[i], "viridis", 0.0, err_max)]
        for j, (data, cmap, lo, hi) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(data, origin="lower", aspect="auto",
                           extent=[t_axis[0], t_axis[-1],
                                    r_axis[0], r_axis[-1]],
                           vmin=lo, vmax=hi, cmap=cmap,
                           interpolation="nearest")
            ims_col[j] = im
            if i == 0:
                ax.set_title(("GT", "predicted", "|GT - pred|")[j],
                             fontsize=11)
            if j == 0:
                ax.set_ylabel(f"theta={th:g} deg\nr (normalized)",
                              fontsize=9)
        axes[i, 0].set_ylim(r_axis[0], r_axis[-1])
    for j in range(3):
        axes[-1, j].set_xlabel("normalized time")

    # Two colorbars total: one shared across GT + pred columns
    # (they use the same vmin/vmax so a single bar is honest and
    # unclutters the layout), and one for the error column.
    fig.colorbar(ims_col[0], ax=axes[:, :2].ravel().tolist(),
                  shrink=0.85, location="right", pad=0.02,
                  label=f"u_z * {value_scale:g}  (GT and predicted)")
    fig.colorbar(ims_col[2], ax=axes[:, 2].tolist(),
                  shrink=0.85, location="right", pad=0.02,
                  label=f"|GT - pred| * {value_scale:g}")

    fig.suptitle(
        f"{tag}  |  {sel_label}  |  test_idx={int(test_idx)}  |  "
        f"rel-L2={rel_l2:.4f}  |  {sim_id}  |  radial kymo GT vs pred",
        fontsize=11)
    # Keep extras minimal: layout + angles are already in the suptitle.
    provenance_footer(fig, sim_id=sim_id, tag=tag,
                      results_file=results_path,
                      extras={"idx": int(test_idx),
                              "sel": (sel_label.split()[0])})
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _resolve_sim_basenames(basenames_list: list, wanted: list
                            ) -> tuple[np.ndarray, list]:
    """Map a comma list of user-supplied basenames to test-split
    indices. Unknown basenames go to a warning and are dropped."""
    idx_by_name = {name: i for i, name in enumerate(basenames_list)}
    hits, misses = [], []
    for w in wanted:
        if w in idx_by_name:
            hits.append(idx_by_name[w])
        else:
            misses.append(w)
    if misses:
        print(f"WARN: {len(misses)} basename(s) not in test split, "
              f"skipping: {misses[:5]}{'...' if len(misses) > 5 else ''}",
              file=sys.stderr)
    if not hits:
        raise SystemExit(
            "--sim gave 0 basenames in the test split; nothing to render")
    return np.asarray(hits, dtype=int), [basenames_list[i] for i in hits]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True,
                    help="training run tag (i.e. outputs/<tag>/...)")
    ap.add_argument("--topn", type=int, default=5,
                    help="number of sims to render for --pick modes; "
                    "ignored when --sim is used")
    ap.add_argument("--pick", default="worst",
                    help="selection strategy: worst / best / median / "
                    "random. Accepts a COMMA LIST (e.g. "
                    "'worst,best,median,random'); each mode runs its "
                    "own selection but predict_run_fields is called "
                    "ONCE on the union of needed indices, saving a "
                    "load_dataset + inference per extra mode. Single "
                    "value writes files directly under --out; comma "
                    "list writes each mode into --out/<mode>/. "
                    "Default: worst (legacy behavior).")
    ap.add_argument("--sim", default=None,
                    help="comma list of specific sim basenames to "
                    "render (overrides --pick and --topn). Basenames "
                    "not in the test split emit a warning and are "
                    "skipped.")
    ap.add_argument("--seed", type=int, default=0,
                    help="rng seed for --pick random (default 0 so "
                    "re-runs are stable)")
    ap.add_argument("--out", required=True,
                    help="output directory (one PNG per sim)")
    ap.add_argument("--output-dir", default="outputs",
                    help="where to find outputs/<tag>/ (default: outputs)")
    ap.add_argument("--data-dir-override", default=None,
                    help="override --data.npz_dir (e.g. when running on "
                    "a different machine than training)")
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    ap.add_argument("--basis-file", default=None,
                    help="direct path to a pod3d_*.npz basis file, "
                    "BYPASSING the cache-key lookup. Use when you know "
                    "which basis was fit during training (see "
                    "outputs/basis_cache/) but the auto key derivation "
                    "produces a MISS. The file's k_cache must be >= "
                    "cfg.pod.k; a smaller stored K raises. Runs "
                    "predict_run_fields without touching load_or_fit_basis.")
    ap.add_argument("--layout", default="snapshot",
                    help="figure layout(s) per selected sim. Accepts a "
                    "COMMA LIST; every listed layout emits one file "
                    "per sim. Valid layouts:\n"
                    "  snapshot -- 3 full-disk heatmaps at t* (GT/"
                    "pred/err) + K a_k(t) subplots. Legacy triage "
                    "view. Filename: <slot>_relL2*.png\n"
                    "  kymo -- 3x3 grid of radial-slice kymographs "
                    "(rows=theta 0/45/90, cols=GT/pred/|err|). "
                    "Filename: <slot>_relL2*_kymo.png\n"
                    "  radial_anim -- 3 subplots (one per angle) "
                    "showing w(r) as a 1D curve, GT solid + pred "
                    "dashed, ANIMATED over time. Ideal for showing "
                    "'does the predicted wafer descend the same way "
                    "the real one does'. Filename: <slot>_relL2*"
                    "_radial.gif\n"
                    "'both' is a legacy alias for 'snapshot,kymo'.")
    ap.add_argument("--show-lower", action="store_true",
                    help="only used by --layout interactive_compare: "
                    "add a translucent gray lower-wafer reference "
                    "plane on the GT and Pred subplots so the "
                    "bonding gap is visually obvious")
    ap.add_argument("--kymo-angles", default="0,45,90",
                    help="comma list of theta values (deg) for the "
                    "kymo layout (default: 0,45,90 -- lab rig)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the test-cases prediction cache and "
                    "re-run predict_run_fields (93 GB load + inference). "
                    "Use if you suspect the cache is stale.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild AND overwrite the test-cases cache "
                    "even on a valid hit. Implies --no-cache.")
    args = ap.parse_args()

    # Locate per-sim errors via results.json.
    results_path = (Path(args.output_dir) / args.tag /
                    "results.json").expanduser().resolve()
    if not results_path.is_file():
        print(f"results.json not found at {results_path}", file=sys.stderr)
        return 2
    with open(results_path) as fp:
        r = json.load(fp)
    field_errs = np.asarray(r.get("per_sim_field_errs", []), dtype=float)
    basenames = r.get("per_sim_basenames", [])
    if field_errs.size == 0 or not basenames:
        print("ERROR: results.json missing per_sim_field_errs or "
              "per_sim_basenames. Re-run training with the new scorer.py.",
              file=sys.stderr)
        return 1

    # ---- Resolve every selection up front ---------------------------
    # Each selection is a dict: {name, label, indices, sel_key,
    #                             cache_path, cache: hit-or-None}.
    # Selections come from --pick (a comma list of modes) plus an
    # optional --sim override that adds one more explicit selection.
    from evaluation.run_predict import predict_run_fields, load_run_config
    overrides = {}
    if args.data_dir_override:
        overrides["data.npz_dir"] = args.data_dir_override
    cfg = load_run_config(args.tag, output_dir=args.output_dir,
                          overrides=overrides)
    ckpt_fp = _checkpoint_fingerprint(args.output_dir, args.tag,
                                        list(cfg.seeds))

    selections = []
    pick_list = [p.strip() for p in args.pick.split(",") if p.strip()]
    unknown = [p for p in pick_list if p not in _VALID_PICKS]
    if unknown:
        print(f"unknown --pick mode(s): {unknown}; valid: "
              f"{list(_VALID_PICKS)}", file=sys.stderr)
        return 2
    for p in pick_list:
        indices = _pick_indices(field_errs, p, args.topn, args.seed)
        names = [basenames[i] for i in indices]
        label = (f"--pick {p} --topn {args.topn}"
                 + (f" --seed {args.seed}" if p == "random" else ""))
        sel_key = _selection_key(
            p, args.topn,
            args.seed if p == "random" else None,
            None)
        cache_path = _test_cache_path(args.output_dir, args.tag,
                                        ckpt_fp, sel_key)
        selections.append(dict(name=p, label=label, indices=indices,
                                 names=names, sel_key=sel_key,
                                 cache_path=cache_path))
    if args.sim:
        sim_list = [s.strip() for s in args.sim.split(",") if s.strip()]
        picked_indices, picked_names = _resolve_sim_basenames(
            basenames, sim_list)
        label = (f"--sim ({len(sim_list)} names, "
                 f"{len(picked_indices)} hit)")
        sel_key = _selection_key(None, None, None, picked_names)
        cache_path = _test_cache_path(args.output_dir, args.tag,
                                        ckpt_fp, sel_key)
        selections.append(dict(name="sim", label=label,
                                 indices=picked_indices,
                                 names=picked_names, sel_key=sel_key,
                                 cache_path=cache_path))

    multi_sel = len(selections) > 1
    print(f"total selections: {len(selections)}", flush=True)
    for s in selections:
        print(f"  [{s['name']}] {s['label']}: "
              f"{len(s['indices'])} sim(s)", flush=True)
        for j in s["indices"]:
            print(f"    test_idx={int(j):4d}  rel_l2="
                  f"{field_errs[j]:.4f}  {basenames[j]}", flush=True)

    # ---- Cache check for each selection ----------------------------
    for s in selections:
        s["cache"] = (None if (args.no_cache or args.force)
                       else _load_test_cache(s["cache_path"]))
        if s["cache"] is not None:
            print(f"  [{s['name']}] cache HIT -> "
                  f"{s['cache_path'].name}", flush=True)

    # ---- If any selection missed, ONE batch call ---------------------
    miss = [s for s in selections if s["cache"] is None]
    if miss:
        union_idx = sorted({int(i) for s in miss for i in s["indices"]})
        print(f"\n{len(miss)}/{len(selections)} selection(s) missed "
              f"cache; batching predict_run_fields ONCE for "
              f"{len(union_idx)} union index(es)", flush=True)
        t0 = time.time()
        batch = predict_run_fields(
            args.tag, idx=union_idx,
            output_dir=args.output_dir,
            overrides=overrides, verbose=True,
            basis_override_path=args.basis_file)
        print(f"  batch predict_run_fields done in "
              f"{time.time() - t0:.1f}s", flush=True)
        idx_to_slot = {int(v): i for i, v in enumerate(batch["idx"])}
        # Split into per-selection subsets and cache each.
        for s in miss:
            slots = [idx_to_slot[int(i)] for i in s["indices"]]
            sub = dict(
                x_canon=batch["x_canon"], y_canon=batch["y_canon"],
                t=batch["t"], sensor_xy=batch["sensor_xy"],
                K=batch["K"],
                idx=np.asarray([int(batch["idx"][j])
                                 for j in slots], dtype=int),
                w_pred=batch["w_pred"][slots],
                w_true=batch["w_true"][slots],
                a_pred=batch["a_pred"][slots],
                a_true=batch["a_true"][slots],
                basenames=[batch["basenames"][j] for j in slots])
            try:
                _save_test_cache(s["cache_path"], sub)
                print(f"  [{s['name']}] wrote cache -> "
                      f"{s['cache_path'].name} "
                      f"(~{(s['cache_path'].stat().st_size >> 20)} MB)",
                      flush=True)
            except OSError as e:
                print(f"  [{s['name']}] WARN: could not write cache "
                      f"({type(e).__name__}: {e})", flush=True)
            s["cache"] = sub

    # ---- Render every selection --------------------------------------
    out_dir_root = Path(args.out).expanduser().resolve()
    out_dir_root.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 (used by helpers)

    kymo_angles = [float(a) for a in args.kymo_angles.split(",")
                    if a.strip()]
    # Parse --layout as comma list. 'both' is a legacy alias.
    layout_list = [l.strip() for l in args.layout.split(",")
                    if l.strip()]
    if "both" in layout_list:
        layout_list = [l for l in layout_list if l != "both"] + [
            "snapshot", "kymo"]
    valid_layouts = {"snapshot", "kymo", "radial_anim",
                      "interactive_compare"}
    unknown_layouts = [l for l in layout_list if l not in valid_layouts]
    if unknown_layouts:
        print(f"unknown --layout(s): {unknown_layouts}; valid: "
              f"{sorted(valid_layouts)} (or 'both' for legacy alias)",
              file=sys.stderr)
        return 2
    want_snapshot = "snapshot" in layout_list
    want_kymo = "kymo" in layout_list
    want_radial_anim = "radial_anim" in layout_list
    want_interactive_compare = "interactive_compare" in layout_list

    total_files = 0
    for s in selections:
        out_data = s["cache"]
        # Multi-selection: each into its own subdir. Single: legacy
        # behavior, files directly in --out (backward-compat).
        this_dir = (out_dir_root / s["name"]) if multi_sel else out_dir_root
        this_dir.mkdir(parents=True, exist_ok=True)
        for slot, test_idx in enumerate(out_data["idx"]):
            gt = out_data["w_true"][slot] * args.value_scale
            pr = out_data["w_pred"][slot] * args.value_scale
            err = np.abs(gt - pr)
            bname = out_data["basenames"][slot]
            rel_l2 = float(field_errs[int(test_idx)])
            stem = Path(bname).stem if bname else f"test{int(test_idx)}"
            base_name = f"{slot:04d}_relL2{rel_l2:.4f}_{stem}"

            if want_snapshot:
                snap_out = this_dir / f"{base_name}.png"
                _render_snapshot(
                    snap_out,
                    gt=gt, pr=pr, err=err,
                    a_pred=out_data["a_pred"][slot],
                    a_true=out_data["a_true"][slot],
                    t=out_data["t"], K=out_data["K"],
                    x_canon=out_data["x_canon"],
                    y_canon=out_data["y_canon"],
                    sensor_xy=out_data["sensor_xy"],
                    sim_id=bname, tag=args.tag,
                    rel_l2=rel_l2, test_idx=int(test_idx),
                    sel_label=s["label"],
                    results_path=results_path)
                print(f"  wrote {snap_out}", flush=True)
                total_files += 1
            if want_kymo:
                kymo_out = this_dir / f"{base_name}_kymo.png"
                _render_kymo_compare(
                    kymo_out,
                    w_true_m=out_data["w_true"][slot],
                    w_pred_m=out_data["w_pred"][slot],
                    x_canon=out_data["x_canon"],
                    y_canon=out_data["y_canon"],
                    angles=kymo_angles,
                    sim_id=bname, tag=args.tag,
                    rel_l2=rel_l2, test_idx=int(test_idx),
                    sel_label=s["label"],
                    results_path=results_path,
                    value_scale=args.value_scale)
                print(f"  wrote {kymo_out}", flush=True)
                total_files += 1
            if want_radial_anim:
                rad_out = this_dir / f"{base_name}_radial.gif"
                _render_radial_anim(
                    rad_out,
                    w_true_m=out_data["w_true"][slot],
                    w_pred_m=out_data["w_pred"][slot],
                    x_canon=out_data["x_canon"],
                    y_canon=out_data["y_canon"],
                    angles=kymo_angles,
                    sim_id=bname, tag=args.tag,
                    rel_l2=rel_l2, test_idx=int(test_idx),
                    sel_label=s["label"],
                    results_path=results_path,
                    value_scale=args.value_scale)
                print(f"  wrote {rad_out}", flush=True)
                total_files += 1
            if want_interactive_compare:
                ic_out = this_dir / f"{base_name}_compare.html"
                _render_interactive_compare(
                    ic_out,
                    w_true_m=out_data["w_true"][slot],
                    w_pred_m=out_data["w_pred"][slot],
                    x_canon=out_data["x_canon"],
                    y_canon=out_data["y_canon"],
                    sensor_xy=out_data["sensor_xy"],
                    sim_id=bname, tag=args.tag,
                    rel_l2=rel_l2, test_idx=int(test_idx),
                    sel_label=s["label"],
                    results_path=results_path,
                    value_scale=args.value_scale,
                    show_lower=args.show_lower)
                print(f"  wrote {ic_out}", flush=True)
                total_files += 1

    print(f"\nall {total_files} test-case figure(s) across "
          f"{len(selections)} selection(s) -> {out_dir_root}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
