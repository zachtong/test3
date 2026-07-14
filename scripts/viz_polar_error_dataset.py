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

Batch over models: pass several --tags; each model gets its OWN
pair of figures (mean + std), written independently with its own
color scale. This is a batch runner, not a side-by-side
comparison -- every model is processed and plotted on its own.

Memory: predictions are fetched in chunks (--chunk sims at a time)
and folded into a running mean, so the full N-sim field stack is
never held at once.

    python scripts/viz_polar_error_dataset.py \\
        --tags merged_sweep_k12_n6_ABCDEF qrdeim_n6_k12 \\
        --data.npz_dir /data/merged_dataset \\
        --n-samples 200 --out-dir viz/polar_error
    # -> viz/polar_error/<tag>_mean.png and <tag>_std.png per model
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


def _aggregates(cube, r_band):
    """Collapse a (theta, r, t) cube to the three 2D views by
    averaging one dimension each.

    r_band is a boolean mask over the r axis marking the mountable /
    interesting radial band. The two views that KEEP r as an axis
    (m_time, m_angle) average over all r (the full field is still
    shown, just color-scaled to the band in _render). The view that
    AGGREGATES r away (m_radius) averages over the BAND ONLY -- else
    the unmountable extreme-r error (r<0.1 near the singular axis,
    r>0.98 at the arc) contaminates the theta-time map."""
    return dict(
        m_time=cube.mean(axis=2),                 # (theta, r)
        m_angle=cube.mean(axis=0),                # (r, t)
        m_radius=cube[:, r_band, :].mean(axis=1)) # (theta, t) band only


def _process_one_tag(task: dict) -> dict:
    """Aggregate + render one model. Runs in the main process (serial
    mode) or a worker process (parallel mode). Returns a small,
    picklable result: tag, status, and (on success) the peak record
    needed for the summary. Heavy arrays are NOT returned."""
    tag = task["tag"]
    out_dir = Path(task["out_dir"])
    mean_path = out_dir / f"{tag}_mean.png"
    std_path = out_dir / f"{tag}_std.png"
    if task["skip_existing"] and mean_path.is_file():
        return dict(tag=tag, status="skipped")
    n_test = _n_test(tag, task["output_dir"])
    if n_test == 0:
        return dict(tag=tag, status="failed",
                    msg="no results.json / test sims")
    n = min(task["n_samples"], n_test)
    if task["random"]:
        rng = np.random.default_rng(task["seed"])
        idx = sorted(rng.choice(n_test, size=n,
                                replace=False).tolist())
    else:
        idx = list(range(n))
    print(f"[{tag}] aggregating {n} of {n_test} test sims",
          flush=True)
    try:
        mean_cube, std_cube, thetas, rs, n_used = \
            _mean_std_error_cubes(
                tag, idx, task["output_dir"], task["overrides"],
                task["n_theta"], task["n_r"], task["value_scale"],
                task["chunk"])
    except (ValueError, KeyError, OSError) as e:
        return dict(tag=tag, status="failed",
                    msg=f"{type(e).__name__}: {e}")
    nt = mean_cube.shape[-1]
    t_axis = np.linspace(0.0, 1.0, nt)
    r_band = (rs >= task["r_focus_lo"]) & (rs <= task["r_focus_hi"])
    if not r_band.any():
        r_band = np.ones_like(rs, dtype=bool)
    rec = dict(tag=tag, n_used=n_used, thetas=thetas, rs=rs,
               t_axis=t_axis, r_band=r_band,
               clip_pctl=task["clip_pctl"],
               mean=_aggregates(mean_cube, r_band),
               std=_aggregates(std_cube, r_band))
    _render(rec, mean_path, task["value_scale"], stat="mean",
            cmap="magma")
    _render(rec, std_path, task["value_scale"], stat="std",
            cmap="viridis")
    print(f"  [{tag}] wrote {mean_path.name} + {std_path.name}",
          flush=True)
    # keep only the small aggregate maps for the peak summary
    return dict(tag=tag, status="done", n_used=n_used,
                thetas=thetas, rs=rs, t_axis=t_axis,
                mean=rec["mean"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=None,
                    help="one or more trained model tags. Each model "
                    "is processed independently and gets its OWN "
                    "pair of figures (batch, not a comparison). "
                    "Required unless --diff-tags is used.")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="test samples to aggregate (default 200)")
    ap.add_argument("--random", action="store_true",
                    help="random sample of test sims instead of the "
                    "first N (deterministic default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40,
                    help="sims per predict batch (memory control; "
                    "default 40)")
    ap.add_argument("--workers", type=int, default=1,
                    help="how many MODELS to process in parallel "
                    "(default 1 = serial). Each worker independently "
                    "loads its model's test-split fields, so peak "
                    "RAM scales with --workers; set it to what your "
                    "memory allows. BLAS threads are capped to "
                    "cores/workers to avoid oversubscription.")
    ap.add_argument("--n-theta", type=int, default=91)
    ap.add_argument("--n-r", type=int, default=128)
    ap.add_argument("--r-focus-lo", type=float, default=0.1,
                    help="lower edge of the mountable / interesting "
                    "radial band. Error outside [lo, hi] (near the "
                    "singular center and the disk arc, where sensors "
                    "cannot go) over-exposes so it does not wash out "
                    "the mid-disk color scale, and is excluded from "
                    "the radius-aggregated (theta-time) map. "
                    "Default 0.1")
    ap.add_argument("--r-focus-hi", type=float, default=0.98,
                    help="upper edge of the mountable band "
                    "(default 0.98)")
    ap.add_argument("--clip-pctl", type=float, default=99.0,
                    help="percentile within the band that sets the "
                    "color vmax (default 99)")
    ap.add_argument("--value-scale", type=float, default=1.0e6)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--out-dir", default="viz/polar_error",
                    help="directory for the per-model figures; "
                    "each model writes <tag>_mean.png and "
                    "<tag>_std.png here")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a tag whose <tag>_mean.png already "
                    "exists in --out-dir")
    ap.add_argument("--diff-tags", nargs=2, metavar=("A", "B"),
                    default=None,
                    help="difference mode: render the error field "
                    "B - A (blue = B has less error = where the "
                    "change from A to B helped). A and B must share "
                    "the same dataset + split. This -- not a single "
                    "config's error map -- is the placement signal. "
                    "Ignores --tags.")
    args, unknown = ap.parse_known_args()
    overrides = _parse_overrides(unknown)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    common = dict(
        out_dir=args.out_dir, output_dir=args.output_dir,
        overrides=overrides, n_samples=args.n_samples,
        random=args.random, seed=args.seed, chunk=args.chunk,
        n_theta=args.n_theta, n_r=args.n_r,
        r_focus_lo=args.r_focus_lo, r_focus_hi=args.r_focus_hi,
        clip_pctl=args.clip_pctl, value_scale=args.value_scale,
        skip_existing=args.skip_existing)

    if args.diff_tags:
        return _diff_mode(args.diff_tags[0], args.diff_tags[1],
                          common)

    if not args.tags:
        print("provide --tags (batch) or --diff-tags A B (diff)",
              file=sys.stderr)
        return 2
    tasks = [dict(tag=tag, **common) for tag in args.tags]

    workers = max(1, args.workers)
    if workers == 1:
        results = [_process_one_tag(t) for t in tasks]
    else:
        # spawn context + capped BLAS threads: children import numpy
        # fresh and inherit the thread limit from the environment,
        # so N model-workers do not each spin up all cores.
        import multiprocessing as mp
        import os
        try:
            n_cpu = len(os.sched_getaffinity(0))
        except AttributeError:
            n_cpu = os.cpu_count() or 1
        per = max(1, n_cpu // workers)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(per)
        print(f"parallel: {workers} model-workers x {per} BLAS "
              f"threads each ({n_cpu} cpus)", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_process_one_tag, tasks)

    done = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]
    for r in failed:
        print(f"[{r['tag']}] FAILED: {r.get('msg', '')}",
              file=sys.stderr)

    if not done and not skipped:
        print("no models produced", file=sys.stderr)
        return 1
    print(f"\nbatch done: {done} rendered, {skipped} skipped, "
          f"{len(failed)} failed")
    _print_peaks([r for r in results if r["status"] == "done"])
    return 0 if not failed else 1


def _render(rec, out_path, value_scale, stat="mean", cmap="magma"):
    """One model, one statistic: a 1x3 figure with its own scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = rec[stat]
    th, rr, ta = rec["thetas"], rec["rs"], rec["t_axis"]
    band = rec["r_band"]
    p = rec.get("clip_pctl", 99.0)
    word = "mean" if stat == "mean" else "std"
    lo, hi = float(rr[band][0]), float(rr[band][-1])
    col_titles = [f"{word} |err| over time\n(where in the disk)",
                  f"{word} |err| over angle\n(which radius, when)",
                  f"{word} |err| over radius\n(band r in "
                  f"[{lo:.2g},{hi:.2g}])"]

    # Color range from the mountable band only, so the unmountable
    # extreme-r error (r<band or r>band) OVER-EXPOSES (saturates)
    # instead of washing out the mid-disk detail.
    def _vmax(arr):
        arr = arr[np.isfinite(arr)]
        return float(np.percentile(arr, p)) if arr.size else 1.0
    vmax_t = _vmax(a["m_time"][:, band])       # (theta, r): band cols
    vmax_a = _vmax(a["m_angle"][band, :])      # (r, t): band rows
    vmax_r = _vmax(a["m_radius"])              # already band-averaged

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             constrained_layout=True)
    # dashed guides marking the mountable band on the r-axis panels
    im0 = axes[0].imshow(
        a["m_time"].T, origin="lower", aspect="auto",
        extent=[th[0], th[-1], rr[0], rr[-1]], cmap=cmap,
        vmin=0, vmax=vmax_t, interpolation="nearest")
    for yb in (lo, hi):
        axes[0].axhline(yb, color="w", ls="--", lw=0.8, alpha=0.6)
    axes[0].set_xlabel("theta (deg)")
    axes[0].set_ylabel("r (normalized)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        a["m_angle"], origin="lower", aspect="auto",
        extent=[ta[0], ta[-1], rr[0], rr[-1]], cmap=cmap,
        vmin=0, vmax=vmax_a, interpolation="nearest")
    for yb in (lo, hi):
        axes[1].axhline(yb, color="w", ls="--", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("t (normalized)")
    axes[1].set_ylabel("r (normalized)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        a["m_radius"], origin="lower", aspect="auto",
        extent=[ta[0], ta[-1], th[0], th[-1]], cmap=cmap,
        vmin=0, vmax=vmax_r, interpolation="nearest")
    axes[2].set_xlabel("t (normalized)")
    axes[2].set_ylabel("theta (deg)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for c in range(3):
        axes[c].set_title(col_titles[c], fontsize=10)

    kind = ("across-sample MEAN" if stat == "mean"
            else "across-sample STD")
    fig.suptitle(f"{rec['tag']}  |  dataset {kind} of polar "
                 f"|error| (|u_z err| * {value_scale:g}), "
                 f"n={rec['n_used']} samples", fontsize=13,
                 fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _mean_cube_for_tag(tag, idx, task) -> tuple:
    """Just the across-sample MEAN |error| cube for one tag (drops
    the std). Used by the diff mode."""
    mean_cube, _std, thetas, rs, n_used = _mean_std_error_cubes(
        tag, idx, task["output_dir"], task["overrides"],
        task["n_theta"], task["n_r"], task["value_scale"],
        task["chunk"])
    return mean_cube, thetas, rs, n_used


def _diff_mode(tag_a, tag_b, task) -> int:
    """Render the difference field B - A of two configs' across-
    sample mean |error| cubes. Negative (blue) = B has LESS error
    than A at that location/time; that is where going from A to B
    (adding / moving sensors) helped. This -- not a single config's
    error -- is the sensor-placement signal: it shows what a change
    in the sensor set actually bought, where.

    A and B must share the same dataset + split so the SAME test
    sims are compared (the test split depends on seed/fracs, not
    sensors, so configs on the same data are directly comparable)."""
    na = _n_test(tag_a, task["output_dir"])
    nb = _n_test(tag_b, task["output_dir"])
    if na == 0 or nb == 0:
        print(f"missing results.json: {tag_a}({na}) {tag_b}({nb})",
              file=sys.stderr)
        return 1
    n = min(task["n_samples"], na, nb)
    if task["random"]:
        rng = np.random.default_rng(task["seed"])
        # SAME idx for both so identical sims are differenced
        idx = sorted(rng.choice(min(na, nb), size=n,
                                replace=False).tolist())
    else:
        idx = list(range(n))
    print(f"[diff] {tag_b} minus {tag_a} on {n} shared test sims",
          flush=True)
    cube_a, tha, rsa, ua = _mean_cube_for_tag(tag_a, idx, task)
    cube_b, thb, rsb, ub = _mean_cube_for_tag(tag_b, idx, task)
    if cube_a.shape != cube_b.shape:
        print(f"shape mismatch {cube_a.shape} vs {cube_b.shape}",
              file=sys.stderr)
        return 1
    diff = cube_b - cube_a                       # (theta, r, t)
    rs = rsa
    r_band = (rs >= task["r_focus_lo"]) & (rs <= task["r_focus_hi"])
    if not r_band.any():
        r_band = np.ones_like(rs, dtype=bool)
    nt = diff.shape[-1]
    rec = dict(tag=f"{tag_b}  minus  {tag_a}", n_used=n,
               thetas=tha, rs=rs, t_axis=np.linspace(0, 1, nt),
               r_band=r_band, clip_pctl=task["clip_pctl"],
               diff=_aggregates(diff, r_band))
    out = Path(task["out_dir"]) / f"diff_{tag_b}_minus_{tag_a}.png"
    _render_diff(rec, out, task["value_scale"])
    print(f"wrote {out}")
    # summary: biggest improvement (most negative) location
    a = rec["diff"]
    ti, ri = np.unravel_index(a["m_time"].argmin(), a["m_time"].shape)
    print(f"  biggest error DROP (B<A): theta={tha[ti]:.0f}deg "
          f"r={rs[ri]:.2f}, delta={a['m_time'][ti, ri]:+.3f}")
    return 0


def _render_diff(rec, out_path, value_scale, cmap="RdBu_r"):
    """3-panel diverging difference (B - A), symmetric about 0,
    band-focused color. Blue = B lower error (improvement)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = rec["diff"]
    th, rr, ta = rec["thetas"], rec["rs"], rec["t_axis"]
    band = rec["r_band"]
    p = rec.get("clip_pctl", 99.0)
    lo, hi = float(rr[band][0]), float(rr[band][-1])

    def _sym(arr):
        arr = arr[np.isfinite(arr)]
        v = float(np.percentile(np.abs(arr), p)) if arr.size else 1.0
        return v if v > 0 else 1.0
    vt = _sym(a["m_time"][:, band])
    va = _sym(a["m_angle"][band, :])
    vr = _sym(a["m_radius"])

    col_titles = ["diff over time\n(where in the disk)",
                  "diff over angle\n(which radius, when)",
                  f"diff over radius\n(band r in [{lo:.2g},{hi:.2g}])"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             constrained_layout=True)
    im0 = axes[0].imshow(
        a["m_time"].T, origin="lower", aspect="auto",
        extent=[th[0], th[-1], rr[0], rr[-1]], cmap=cmap,
        vmin=-vt, vmax=vt, interpolation="nearest")
    for yb in (lo, hi):
        axes[0].axhline(yb, color="0.3", ls="--", lw=0.8, alpha=0.6)
    axes[0].set_xlabel("theta (deg)")
    axes[0].set_ylabel("r (normalized)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        a["m_angle"], origin="lower", aspect="auto",
        extent=[ta[0], ta[-1], rr[0], rr[-1]], cmap=cmap,
        vmin=-va, vmax=va, interpolation="nearest")
    for yb in (lo, hi):
        axes[1].axhline(yb, color="0.3", ls="--", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("t (normalized)")
    axes[1].set_ylabel("r (normalized)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        a["m_radius"], origin="lower", aspect="auto",
        extent=[ta[0], ta[-1], th[0], th[-1]], cmap=cmap,
        vmin=-vr, vmax=vr, interpolation="nearest")
    axes[2].set_xlabel("t (normalized)")
    axes[2].set_ylabel("theta (deg)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for c in range(3):
        axes[c].set_title(col_titles[c], fontsize=10)
    fig.suptitle(f"{rec['tag']}  |  polar |error| DIFFERENCE "
                 f"(* {value_scale:g}), n={rec['n_used']} shared "
                 f"sims.  blue = second config has LESS error "
                 f"(improvement)", fontsize=12, fontweight="bold")
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
