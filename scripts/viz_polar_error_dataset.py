"""Dataset-level polar error aggregation: average the |error| field
over MANY test samples to expose a model's SYSTEMATIC weak spots
(not one sim's noise), and compare models side by side.

Single-sim polar_agg maps have per-sim randomness. Averaging the
polar error cube over N samples (default 200, taken regardless of
good/bad) leaves only the error structure that is consistent across
the test set -- i.e., a genuine defect of the model or its sensor
placement. The three dimension-collapsed views then say WHERE
(radius / angle / time) that systematic error lives.

Absolute error (|pred - gt|, micrometres) is aggregated: for this
dataset the true displacement amplitudes are similar across
samples, so absolute error is the meaningful physical quantity and
no per-sim normalization is applied.

Multi-model: pass several --tags; each becomes one ROW of three
panels, with a shared color scale per column so models are directly
comparable. The row whose panels are hotter (or hot in a different
place) has the worse / differently-located systematic error --
directly attributable to that model's sensor layout.

Memory: predictions are fetched in chunks (--chunk sims at a time)
and folded into a running mean, so the full N-sim field stack is
never held at once.

    python scripts/viz_polar_error_dataset.py \\
        --tags merged_sweep_k12_n6_ABCDEF qrdeim_n6_k12 \\
        --data.npz_dir /data/merged_dataset \\
        --n-samples 200 --out viz/polar_error_dataset.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.fieldviz_polar import polar_time_cube       # noqa: E402
from evaluation.run_predict import predict_run_fields    # noqa: E402


def _parse_overrides(unknown: list[str]) -> dict:
    ov, i = {}, 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            val = unknown[i + 1] if i + 1 < len(unknown) else ""
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
            ov[key] = val
            i += 2
        else:
            i += 1
    return ov


def _n_test(tag: str, output_dir: str) -> int:
    res = Path(output_dir) / tag / "results.json"
    if not res.is_file():
        return 0
    try:
        d = json.loads(res.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    return len(d.get("per_sim_basenames", []))


def _mean_std_error_cubes(tag, idx, output_dir, overrides, n_theta,
                          n_r, value_scale, chunk) -> tuple:
    """Running mean AND std of the polar |error| cube over the given
    test indices, fetched in chunks (sum + sum-of-squares, so the
    full field stack is never held). Returns (mean_cube, std_cube,
    thetas, rs, n_used), each cube (theta, r, t).

    std_cube[theta,r,t] = across-sample standard deviation of the
    absolute error at that cell -- how much the error at this
    location/time varies from sample to sample."""
    e_sum = e_sqsum = None
    thetas = rs = None
    n_used = 0
    for start in range(0, len(idx), chunk):
        batch = idx[start:start + chunk]
        out = predict_run_fields(
            tag, idx=batch, output_dir=output_dir,
            overrides=overrides, verbose=False)
        x = out["x_canon"]
        y = out["y_canon"]
        wp = out["w_pred"]                          # (nb, Nx, Ny, Nt)
        wt = out["w_true"]
        for j in range(wp.shape[0]):
            gt, thetas, rs = polar_time_cube(
                wt[j], x, y, n_theta=n_theta, n_r=n_r)
            pr, _, _ = polar_time_cube(
                wp[j], x, y, n_theta=n_theta, n_r=n_r)
            e = np.abs(pr - gt) * value_scale
            if e_sum is None:
                e_sum = e.copy()
                e_sqsum = e * e
            else:
                e_sum += e
                e_sqsum += e * e
            n_used += 1
        print(f"  [{tag}] {n_used}/{len(idx)} samples folded",
              flush=True)
    if e_sum is None:
        raise ValueError(f"no samples predicted for {tag}")
    mean_cube = e_sum / n_used
    var_cube = np.maximum(e_sqsum / n_used - mean_cube ** 2, 0.0)
    std_cube = np.sqrt(var_cube)
    return mean_cube, std_cube, thetas, rs, n_used


def _aggregates(cube):
    """Collapse a (theta, r, t) cube to the three 2D views by
    averaging one dimension each."""
    return dict(
        m_time=cube.mean(axis=2),      # (theta, r)
        m_angle=cube.mean(axis=0),     # (r, t)
        m_radius=cube.mean(axis=1))    # (theta, t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", required=True,
                    help="one or more trained model tags; each is a "
                    "row of 3 panels")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="test samples to aggregate (default 200)")
    ap.add_argument("--random", action="store_true",
                    help="random sample of test sims instead of the "
                    "first N (deterministic default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40,
                    help="sims per predict batch (memory control; "
                    "default 40)")
    ap.add_argument("--n-theta", type=int, default=91)
    ap.add_argument("--n-r", type=int, default=128)
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--out", default="viz/polar_error_dataset.png")
    args, unknown = ap.parse_known_args()
    overrides = _parse_overrides(unknown)

    rng = np.random.default_rng(args.seed)
    results = []          # list of dict per tag
    for tag in args.tags:
        n_test = _n_test(tag, args.output_dir)
        if n_test == 0:
            print(f"WARN: no results.json / test sims for {tag}; "
                  f"skipping", file=sys.stderr)
            continue
        n = min(args.n_samples, n_test)
        if args.random:
            idx = sorted(rng.choice(n_test, size=n,
                                    replace=False).tolist())
        else:
            idx = list(range(n))
        print(f"[{tag}] aggregating {n} of {n_test} test sims",
              flush=True)
        mean_cube, std_cube, thetas, rs, n_used = \
            _mean_std_error_cubes(
                tag, idx, args.output_dir, overrides, args.n_theta,
                args.n_r, args.value_scale, args.chunk)
        nt = mean_cube.shape[-1]
        t_axis = np.linspace(0.0, 1.0, nt)
        rec = dict(tag=tag, n_used=n_used, thetas=thetas, rs=rs,
                   t_axis=t_axis)
        rec["mean"] = _aggregates(mean_cube)
        rec["std"] = _aggregates(std_cube)
        results.append(rec)

    if not results:
        print("no models aggregated", file=sys.stderr)
        return 1

    out = Path(args.out)
    mean_path = out.with_name(out.stem + "_mean" + out.suffix)
    std_path = out.with_name(out.stem + "_std" + out.suffix)
    _render(results, mean_path, args.value_scale, stat="mean",
            cmap="magma")
    _render(results, std_path, args.value_scale, stat="std",
            cmap="viridis")
    print(f"\nwrote {mean_path}")
    print(f"wrote {std_path}")
    _print_peaks(results)
    return 0


def _render(results, out_path, value_scale, stat="mean",
            cmap="magma"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(results)
    # shared vmax per column across all models
    vmax_time = max(r[stat]["m_time"].max() for r in results)
    vmax_angle = max(r[stat]["m_angle"].max() for r in results)
    vmax_radius = max(r[stat]["m_radius"].max() for r in results)

    word = "mean" if stat == "mean" else "std"
    fig, axes = plt.subplots(
        nrows, 3, figsize=(16, 4.6 * nrows), squeeze=False,
        constrained_layout=True)
    col_titles = [f"{word} |err| over time\n(where in the disk)",
                  f"{word} |err| over angle\n(which radius, when)",
                  f"{word} |err| over radius\n(which angle, when)"]
    for row, r in enumerate(results):
        a = r[stat]
        th, rr, ta = r["thetas"], r["rs"], r["t_axis"]
        im0 = axes[row][0].imshow(
            a["m_time"].T, origin="lower", aspect="auto",
            extent=[th[0], th[-1], rr[0], rr[-1]], cmap=cmap,
            vmin=0, vmax=vmax_time, interpolation="nearest")
        axes[row][0].set_xlabel("theta (deg)")
        axes[row][0].set_ylabel(f"{r['tag']}\n\nr (normalized)",
                                fontsize=9)
        fig.colorbar(im0, ax=axes[row][0], fraction=0.046, pad=0.04)

        im1 = axes[row][1].imshow(
            a["m_angle"], origin="lower", aspect="auto",
            extent=[ta[0], ta[-1], rr[0], rr[-1]], cmap=cmap,
            vmin=0, vmax=vmax_angle, interpolation="nearest")
        axes[row][1].set_xlabel("t (normalized)")
        axes[row][1].set_ylabel("r (normalized)")
        fig.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)

        im2 = axes[row][2].imshow(
            a["m_radius"], origin="lower", aspect="auto",
            extent=[ta[0], ta[-1], th[0], th[-1]], cmap=cmap,
            vmin=0, vmax=vmax_radius, interpolation="nearest")
        axes[row][2].set_xlabel("t (normalized)")
        axes[row][2].set_ylabel("theta (deg)")
        fig.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)

        if row == 0:
            for c in range(3):
                axes[row][c].set_title(col_titles[c], fontsize=10)

    kind = ("across-sample MEAN" if stat == "mean"
            else "across-sample STD")
    fig.suptitle(f"Dataset {kind} of polar |error| "
                 f"(|u_z err| * {value_scale:g}), "
                 f"n={results[0]['n_used']} samples, shared scale "
                 f"per column", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _print_peaks(results):
    print("\nSystematic error hotspots (argmax of MEAN aggregate):")
    for r in results:
        a = r["mean"]
        th, rr, ta = r["thetas"], r["rs"], r["t_axis"]
        ti, ri = np.unravel_index(a["m_time"].argmax(),
                                  a["m_time"].shape)
        rj, tj = np.unravel_index(a["m_angle"].argmax(),
                                  a["m_angle"].shape)
        thk, tk = np.unravel_index(a["m_radius"].argmax(),
                                   a["m_radius"].shape)
        print(f"  {r['tag']}:")
        print(f"    disk hotspot   : theta={th[ti]:.0f}deg "
              f"r={rr[ri]:.2f}")
        print(f"    radius-time    : r={rr[rj]:.2f} "
              f"t={ta[tj]:.2f}")
        print(f"    angle-time     : theta={th[thk]:.0f}deg "
              f"t={ta[tk]:.2f}")


if __name__ == "__main__":
    sys.exit(main())
