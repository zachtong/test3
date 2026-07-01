"""Diagnose why an existing pod3d_*.npz did not HIT during viz.

Given a training tag, reads outputs/<tag>/results.json's stored config
and recomputes the training-time basis cache key. Lists every
pod3d_*.npz on disk and reports:

  - the expected training filename (pod3d_<training_hash>.npz)
  - whether that file exists (should be True)
  - any OTHER pod3d files (usually one: the mystery one that
    viz_test_cases just re-fit)
  - for each mystery file, probes a set of plausible key-element
    variants (npz_dir trailing slash, resolved abs path, expanduser,
    n_fit +-1) to see if any produces that hash -- pinpointing which
    field drifted between training and viz

Runs offline (no data load, no torch, no matplotlib). Takes ~50 ms.

    python scripts/diagnose_basis_cache.py firehorse1_and_2

    # override where to look for outputs/ or basis_cache/
    python scripts/diagnose_basis_cache.py firehorse1_and_2 \\
        --output-dir /elsewhere/outputs \\
        --basis-cache-dir /elsewhere/basis_cache
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path


_KEY_FIELDS = ("data.npz_dir", "data.nx", "data.ny", "data.nt",
                "data.x_end", "data.y_end", "data.drop_first_steps",
                "data.seed", "data.train_frac", "train.val_frac")


def _lookup(cfg: dict, dotted: str):
    v = cfg
    for p in dotted.split("."):
        if not isinstance(v, dict):
            return "<missing>"
        v = v.get(p, "<missing>")
    return v


def _key(npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
         seed, train_frac, val_frac, n_fit) -> str:
    """Mirror training/basis_cache.py::_key exactly. Any drift here
    would silently break the diagnosis; keep this function 1:1 with
    the source of truth."""
    raw = (f"pod3d|{npz_dir}|{nx}|{ny}|{nt}|{x_end}|{y_end}|"
           f"{drop_first_steps}|{seed}|{train_frac}|{val_frac}|{n_fit}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _npz_dir_variants(npz_dir: str) -> list:
    """Enumerate plausible in-the-wild strings the same physical path
    could be spelled as. Order matters: put more-likely first so the
    first match is the most informative."""
    p = str(npz_dir).rstrip()
    variants = [p]
    # Trailing slash on/off
    if p.endswith("/"):
        variants.append(p.rstrip("/"))
    else:
        variants.append(p + "/")
    # expanduser
    try:
        exp = str(Path(p).expanduser())
        if exp != p:
            variants.append(exp)
            variants.append(exp + "/")
            variants.append(exp.rstrip("/"))
    except Exception:
        pass
    # resolve (absolute + symlink)
    try:
        res = str(Path(p).expanduser().resolve())
        if res not in variants:
            variants.append(res)
            variants.append(res + "/")
    except Exception:
        pass
    # dedupe preserving order
    seen = set(); out = []
    for v in variants:
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def _type_variants(v):
    """Enumerate plausible str() representations of a JSON-parsed
    numeric value. Training's _key does f-string interpolation which
    calls str() on the raw type -- so int 1 -> '1' and float 1.0 ->
    '1.0' produce different hashes. Emit both forms whenever the
    value is numerically 1 or similar so the probe catches it."""
    out = [v]
    try:
        f = float(v)
        i = int(f)
        # if the value is an integer, both int and float representations
        if f == i:
            if isinstance(v, int) and not isinstance(v, bool):
                out.append(float(v))                       # int -> float
            elif isinstance(v, float):
                out.append(int(f))                          # float -> int
        # else values like 0.8 vs 0.80 are the same via str(), skip
    except (TypeError, ValueError):
        pass
    # dedupe preserving order
    seen = set(); dd = []
    for x in out:
        k = (type(x).__name__, x)
        if k not in seen:
            seen.add(k); dd.append(x)
    return dd


def _probe_hash(target_hash: str, base_key_tuple, npz_dir_variants,
                n_fit_range=range(-50, 51)) -> tuple | None:
    """Brute-force to find a tuple that hashes to target_hash.

    Combines: every (npz_dir_variant) x (n_fit + delta) x
    (int-vs-float type variant per numeric field). This still
    finishes in <1 s at typical cache sizes (~ 5 * 100 * 2^6 = 32000
    hashes), and covers the vast majority of real-world drifts:
      - npz_dir string spelling
      - n_fit off by a few (some sims added/removed since training)
      - x_end / y_end / train_frac / val_frac stored as int vs float
    """
    (nx, ny, nt, x_end, y_end, drop_first_steps,
     seed, train_frac, val_frac, base_n_fit) = base_key_tuple

    # Type-vary the fields that could reasonably be int-vs-float
    nx_vs = _type_variants(nx)
    ny_vs = _type_variants(ny)
    nt_vs = _type_variants(nt)
    xe_vs = _type_variants(x_end)
    ye_vs = _type_variants(y_end)
    drop_vs = _type_variants(drop_first_steps)
    seed_vs = _type_variants(seed)
    tf_vs = _type_variants(train_frac)
    vf_vs = _type_variants(val_frac)

    for nd in npz_dir_variants:
        for dn in n_fit_range:
            n = base_n_fit + dn
            if n <= 0:
                continue
            for _nx in nx_vs:
             for _ny in ny_vs:
              for _nt in nt_vs:
               for _xe in xe_vs:
                for _ye in ye_vs:
                 for _drop in drop_vs:
                  for _seed in seed_vs:
                   for _tf in tf_vs:
                    for _vf in vf_vs:
                     k = _key(nd, _nx, _ny, _nt, _xe, _ye, _drop,
                              _seed, _tf, _vf, n)
                     if k == target_hash:
                         return dict(npz_dir=nd, n_fit=n,
                                       nx=_nx, ny=_ny, nt=_nt,
                                       x_end=_xe, y_end=_ye,
                                       drop_first_steps=_drop,
                                       seed=_seed, train_frac=_tf,
                                       val_frac=_vf)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag", help="training run tag "
                    "(under --output-dir/<tag>/results.json)")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--basis-cache-dir", default=None,
                    help="override; default is <output-dir>/basis_cache")
    args = ap.parse_args()

    rs = Path(args.output_dir) / args.tag / "results.json"
    if not rs.is_file():
        print(f"no results.json at {rs}", file=sys.stderr)
        return 2
    r = json.loads(rs.read_text())
    cfg = r.get("config", {})
    if not cfg:
        print(f"results.json has empty 'config' field",
              file=sys.stderr)
        return 2
    # Extract required key elements
    key_elements = {k: _lookup(cfg, k) for k in _KEY_FIELDS}
    missing = [k for k, v in key_elements.items() if v == "<missing>"]
    if missing:
        print(f"WARN: results.json missing config fields: {missing}",
              file=sys.stderr)
        return 2
    # n_fit is len(train_sims + val_sims); training's results.json
    # typically stores split sizes under 'split' or 'n_train'/'n_val'.
    # Fall back to counting per_sim_basenames + inferring.
    n_train = r.get("n_train_sims") or r.get("split", {}).get("n_train")
    n_val = r.get("n_val_sims") or r.get("split", {}).get("n_val")
    if n_train is None or n_val is None:
        # Fall back: results.json didn't record split sizes explicitly.
        # split_dataset() uses:
        #   n_train_total = int(N * train_frac)
        #   n_val = int(n_train_total * val_frac)   (val CARVED FROM train)
        #   n_test = N - n_train_total
        # So load_or_fit_basis's sims = train + val = n_train_total,
        # and test_frac = 1 - train_frac (NOT 1 - train_frac - val_frac).
        n_test = len(r.get("per_sim_basenames", []))
        tf = float(key_elements["data.train_frac"])
        test_frac = 1.0 - tf
        if test_frac <= 0:
            print("cannot recover n_fit; test_frac <= 0",
                  file=sys.stderr)
            return 2
        # int() truncation of int(N * train_frac) means several N values
        # give the same n_test. Estimate N; the probe widens n_fit anyway.
        n_total_est = int(round(n_test / test_frac))
        n_fit_est = int(n_total_est * tf)
        print(f"note: results.json did not store split sizes; "
              f"estimating n_fit ~= {n_fit_est} from n_test="
              f"{n_test} / (1 - train_frac)={test_frac:.3f}",
              flush=True)
        n_fit_from_json = n_fit_est
    else:
        # split_dataset: n_val is INSIDE n_train_total. Reading
        # n_train_sims / n_val_sims recorded separately, load_or_fit_basis
        # was called with sims = train + val = n_train_total = n_train + n_val.
        n_fit_from_json = int(n_train) + int(n_val)

    # Compute training's expected hash using the RAW json-parsed types
    # (do not eagerly convert; f"{int} " and f"{float}" produce different
    # strings and the probe needs to see both). npz_dir stays str.
    nx = key_elements["data.nx"]
    ny = key_elements["data.ny"]
    nt = key_elements["data.nt"]
    x_end = key_elements["data.x_end"]
    y_end = key_elements["data.y_end"]
    drop = key_elements["data.drop_first_steps"]
    seed = key_elements["data.seed"]
    tf = key_elements["data.train_frac"]
    vf = key_elements["train.val_frac"]
    npz_dir = str(key_elements["data.npz_dir"])
    training_hash = _key(npz_dir, nx, ny, nt, x_end, y_end, drop,
                          seed, tf, vf, n_fit_from_json)
    print(f"\ntraining tag: {args.tag}")
    print(f"  key elements from results.json:")
    for k, v in key_elements.items():
        print(f"    {k} = {v!r}")
    print(f"    n_fit (from split sizes or est.) = {n_fit_from_json}")
    print(f"  training's expected basis hash: {training_hash}")
    print(f"  training's expected filename:   "
          f"pod3d_{training_hash}.npz")

    # List actual files
    bcdir = Path(args.basis_cache_dir
                  or Path(args.output_dir) / "basis_cache")
    files = sorted(bcdir.glob("pod3d_*.npz"),
                    key=lambda p: p.stat().st_mtime)
    print(f"\nbasis_cache dir: {bcdir}")
    if not files:
        print(f"  (no pod3d_*.npz files found)")
        return 0
    print(f"  files (oldest first):")
    for f in files:
        h = f.stem.replace("pod3d_", "")
        marker = "  <-- training's" if h == training_hash else ""
        print(f"    {f.name}  (mtime "
              f"{f.stat().st_mtime:.0f}){marker}")

    # Probe mystery files
    mystery = [f for f in files
                if f.stem.replace("pod3d_", "") != training_hash]
    if not mystery:
        print(f"\nno mystery files -- all pod3d files match training. "
              f"MISS during viz might have been a transient "
              f"argument-parse issue; try re-running viz_test_cases "
              f"without --basis-file and it should HIT now.")
        return 0

    print(f"\nprobing {len(mystery)} mystery file(s) to find the "
          f"drifted field:")
    base_key = (nx, ny, nt, x_end, y_end, drop, seed, tf, vf,
                n_fit_from_json)
    variants = _npz_dir_variants(npz_dir)
    base_all = {"npz_dir": npz_dir, "n_fit": n_fit_from_json,
                 "nx": nx, "ny": ny, "nt": nt,
                 "x_end": x_end, "y_end": y_end,
                 "drop_first_steps": drop, "seed": seed,
                 "train_frac": tf, "val_frac": vf}
    for f in mystery:
        target = f.stem.replace("pod3d_", "")
        hit = _probe_hash(target, base_key, variants)
        if hit is None:
            print(f"  {f.name}: NO MATCH with tested variants. "
                  f"The differing field is likely OUTSIDE (npz_dir, "
                  f"n_fit +- 50, int/float type variants). Send the "
                  f"whole training config to a maintainer -- the "
                  f"drift is in a field the probe doesn't cover yet.",
                  flush=True)
            continue
        print(f"  {f.name}: MATCH", flush=True)
        for k, v in hit.items():
            if v != base_all.get(k):
                bv = base_all.get(k)
                print(f"    {k} differs:", flush=True)
                print(f"      training:  {bv!r}  "
                      f"(type={type(bv).__name__})")
                print(f"      viz-time:  {v!r}  "
                      f"(type={type(v).__name__})", flush=True)

    print(f"\nfix: pass the SAME npz_dir string to viz_all as was in "
          f"training (see 'training:' line above). Or use "
          f"viz_test_cases --basis-file pod3d_"
          f"{training_hash}.npz to bypass the key entirely.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
