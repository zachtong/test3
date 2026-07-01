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


def _probe_hash(target_hash: str, base_key_tuple, npz_dir_variants,
                n_fit_deltas=(0, -1, +1, -2, +2)) -> tuple | None:
    """Brute-force (npz_dir_variant, n_fit + delta) to find a tuple
    that hashes to target_hash. Returns (npz_dir_variant, n_fit) or
    None. Base tuple minus (npz_dir, n_fit)."""
    (nx, ny, nt, x_end, y_end, drop_first_steps,
     seed, train_frac, val_frac, base_n_fit) = base_key_tuple
    for nd in npz_dir_variants:
        for dn in n_fit_deltas:
            n = base_n_fit + dn
            if n <= 0:
                continue
            k = _key(nd, nx, ny, nt, x_end, y_end, drop_first_steps,
                     seed, train_frac, val_frac, n)
            if k == target_hash:
                return (nd, n)
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
        # Fall back: per_sim_basenames is n_test; total is unknown from
        # results.json alone. Ask the user.
        n_test = len(r.get("per_sim_basenames", []))
        tf = float(key_elements["data.train_frac"])
        vf = float(key_elements["train.val_frac"])
        # test_frac = 1 - tf - vf ; so total = n_test / test_frac
        test_frac = 1.0 - tf - vf
        if test_frac <= 0:
            print("cannot recover n_fit; test_frac <= 0", file=sys.stderr)
            return 2
        n_total_est = int(round(n_test / test_frac))
        n_fit_est = int(round(n_total_est * (tf + vf)))
        print(f"note: results.json did not store split sizes; "
              f"estimating n_fit ~= {n_fit_est} from n_test="
              f"{n_test} / test_frac={test_frac:.3f}",
              flush=True)
        n_fit_from_json = n_fit_est
    else:
        n_fit_from_json = int(n_train) + int(n_val)

    # Compute training's expected hash
    nx = int(key_elements["data.nx"])
    ny = int(key_elements["data.ny"])
    nt = int(key_elements["data.nt"])
    x_end = float(key_elements["data.x_end"])
    y_end = float(key_elements["data.y_end"])
    drop = int(key_elements["data.drop_first_steps"])
    seed = int(key_elements["data.seed"])
    tf = float(key_elements["data.train_frac"])
    vf = float(key_elements["train.val_frac"])
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
    for f in mystery:
        target = f.stem.replace("pod3d_", "")
        hit = _probe_hash(target, base_key, variants)
        if hit is None:
            print(f"  {f.name}: NO MATCH with tested variants. "
                  f"The differing field is likely OUTSIDE (npz_dir, "
                  f"n_fit +- 2). Manually check config for e.g. "
                  f"data.x_end / data.y_end / data.drop_first_steps "
                  f"if they were re-parsed as different types "
                  f"(float 1.0 vs int 1) between train and viz.",
                  flush=True)
            continue
        used_dir, used_n_fit = hit
        print(f"  {f.name}: MATCH", flush=True)
        if used_dir != npz_dir:
            print(f"    npz_dir differs:")
            print(f"      training:  {npz_dir!r}")
            print(f"      viz-time:  {used_dir!r}", flush=True)
        if used_n_fit != n_fit_from_json:
            print(f"    n_fit differs: training={n_fit_from_json}, "
                  f"viz-time={used_n_fit} (delta "
                  f"{used_n_fit - n_fit_from_json:+d})", flush=True)

    print(f"\nfix: pass the SAME npz_dir string to viz_all as was in "
          f"training (see 'training:' line above). Or use "
          f"viz_test_cases --basis-file pod3d_"
          f"{training_hash}.npz to bypass the key entirely.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
