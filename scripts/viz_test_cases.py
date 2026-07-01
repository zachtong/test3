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
                               wafer_value_range)


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
    ap.add_argument("--pick", choices=_VALID_PICKS, default="worst",
                    help="selection strategy: worst / best / median / "
                    "random (default: worst, matches the legacy "
                    "viz_worst_cases behavior)")
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

    # Resolve selection: --sim wins, else --pick.
    sim_list = None
    if args.sim:
        sim_list = [s.strip() for s in args.sim.split(",") if s.strip()]
        picked_indices, picked_names = _resolve_sim_basenames(
            basenames, sim_list)
        label = f"--sim ({len(sim_list)} names, {len(picked_indices)} hit)"
    else:
        picked_indices = _pick_indices(field_errs, args.pick,
                                        args.topn, args.seed)
        picked_names = [basenames[i] for i in picked_indices]
        label = (f"--pick {args.pick} --topn {args.topn}"
                 + (f" --seed {args.seed}" if args.pick == "random" else ""))
    print(f"selection: {label}  ({len(picked_indices)} sim(s) to render):")
    for j in picked_indices:
        print(f"  test_idx={int(j):4d}  rel_l2={field_errs[j]:.4f}  "
              f"{basenames[j]}")

    # Cache key encodes (pick, topn, seed) OR --sim basenames.
    from evaluation.run_predict import predict_run_fields, load_run_config
    overrides = {}
    if args.data_dir_override:
        overrides["data.npz_dir"] = args.data_dir_override
    cfg = load_run_config(args.tag, output_dir=args.output_dir,
                          overrides=overrides)
    ckpt_fp = _checkpoint_fingerprint(args.output_dir, args.tag,
                                        list(cfg.seeds))
    sel_key = _selection_key(
        args.pick if not sim_list else None,
        args.topn if not sim_list else None,
        args.seed if (not sim_list and args.pick == "random") else None,
        picked_names if sim_list else None)
    cache_path = _test_cache_path(args.output_dir, args.tag,
                                    ckpt_fp, sel_key)
    out = (None if (args.no_cache or args.force)
           else _load_test_cache(cache_path))
    if out is not None:
        print(f"test-cases cache HIT -> {cache_path.name}  "
              f"({len(out['idx'])} sim(s), K={out['K']})", flush=True)
    else:
        if cache_path.exists() and args.no_cache:
            print(f"  --no-cache: skipping {cache_path.name}",
                  flush=True)
        print(f"test-cases cache MISS -> running predict_run_fields "
              f"(93 GB load + inference)", flush=True)
        t0 = time.time()
        out = predict_run_fields(args.tag,
                                  idx=picked_indices.tolist(),
                                  output_dir=args.output_dir,
                                  overrides=overrides, verbose=True)
        print(f"  predict_run_fields done in "
              f"{time.time() - t0:.1f}s", flush=True)
        try:
            _save_test_cache(cache_path, out)
            print(f"  wrote test-cases cache -> {cache_path.name} "
                  f"(~{(cache_path.stat().st_size >> 20)} MB)",
                  flush=True)
        except OSError as e:
            print(f"  WARN: could not write test-cases cache "
                  f"({type(e).__name__}: {e}); continuing anyway",
                  flush=True)
    x_canon = out["x_canon"]; y_canon = out["y_canon"]
    K = out["K"]
    t = out["t"]
    sensor_xy = out["sensor_xy"]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for slot, test_idx in enumerate(out["idx"]):
        gt = out["w_true"][slot] * args.value_scale
        pr = out["w_pred"][slot] * args.value_scale
        err = np.abs(gt - pr)
        a_pred = out["a_pred"][slot]                # (K, Nt)
        a_true = out["a_true"][slot]
        bname = out["basenames"][slot]
        rel_l2 = float(field_errs[test_idx])

        # t* = time of max abs error (integrate over space).
        err_integrated = err.sum(axis=(0, 1))
        t_star = int(np.argmax(err_integrated))

        # Shared sequential color scale for GT + pred (so the visual
        # comparison is valid). Error panel gets its own viridis range.
        vmin, vmax = wafer_value_range(
            np.stack([gt[..., t_star], pr[..., t_star]]))
        err_max = float(np.percentile(err[..., t_star], 99))
        if err_max <= 0:
            err_max = max(float(err.max()), 1e-12)

        n_cols = max(K, 3)
        fig, axes = plt.subplots(2, n_cols,
                                 figsize=(3.5 * n_cols, 7.2),
                                 constrained_layout=True)

        # TOP ROW: 3 full-disk heatmaps at t*
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
        # hide top-row spares
        for c in range(3, n_cols):
            axes[0, c].set_visible(False)

        # BOTTOM ROW: K subplots of a_k(t) predicted vs true
        for k_idx in range(K):
            c = k_idx
            ax = axes[1, c]
            ax.plot(t, a_true[k_idx], color="0.3", lw=1.2, label="true")
            ax.plot(t, a_pred[k_idx], color=SENSOR_MARKER_COLOR,
                    lw=1.2, ls="--", label="predicted")
            err_k = float(np.linalg.norm(a_pred[k_idx] - a_true[k_idx])
                          / max(np.linalg.norm(a_true[k_idx]), 1e-12))
            ax.set_title(f"a_{k_idx + 1}  rel-L2={err_k:.3f}",
                         fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlabel("normalized t")
            if c == 0:
                ax.legend(fontsize=7, loc="best")
        for c in range(K, n_cols):
            axes[1, c].set_visible(False)

        fig.suptitle(
            f"{args.tag}  |  {label}  |  test_idx={int(test_idx)}  |  "
            f"rel-L2={rel_l2:.4f}  |  {bname}", fontsize=11)
        provenance_footer(fig, sim_id=bname, tag=args.tag,
                          results_file=results_path,
                          extras={"test_idx": int(test_idx),
                                  "rel_l2": f"{rel_l2:.4f}",
                                  "t_star": t_star,
                                  "sel": (label.split()[0])})

        # Filename: <slot>_relL2<val>_<basename>.png so ls sorts by
        # slot; the 4-digit slot prefix avoids ties when two sims
        # share a basename root.
        stem = Path(bname).stem if bname else f"test{int(test_idx)}"
        fname = f"{slot:04d}_relL2{rel_l2:.4f}_{stem}.png"
        outp = out_dir / fname
        fig.savefig(outp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {outp}", flush=True)

    print(f"\nall {len(out['idx'])} test-case figure(s) -> {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
