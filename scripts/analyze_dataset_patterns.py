"""Discover the bonding patterns in a dataset and map them onto POD
modes, on a common (merged) basis.

Motivation: the old dataset is mostly one bonding pattern, the new
one has several; K=8 -> K=12 improved reconstruction. This asks
whether the patterns manifest in distinct POD modes and whether
modes 9-12 correspond to the new patterns (explaining the K gain).

The patterns are characterized along two physically-motivated,
orthogonal axes rather than blind clustering:

  TIME axis  -- direct release vs hold-then-release. Captured by the
    overall bonding-progress curve over time (mode-1 coefficient
    a_1(t), which tracks the dominant descent). Direct = monotone
    fast; hold = plateau then rise.

  SPACE axis -- symmetric vs asymmetric. Captured by the ASYMMETRY
    RATIO: energy in azimuthal (angle-varying) modes over total.
    A mode is "azimuthal" if its spatial shape varies more along
    theta than along r (var over theta / var over r, sampled on a
    polar grid). Symmetric sims ~ 0; asymmetric sims clearly > 0.

Then: per-sim modal energy on the common basis feeds the pattern x
mode occupancy heatmap, and the per-pattern truncation floor at
K=8 vs K=12 shows where the extra modes pay off.

Datasets are passed as paths (never hardcoded):

    python scripts/analyze_dataset_patterns.py \\
        --basis outputs/basis_cache/pod3d_<merged_key>.npz \\
        --old-npz-dir /path/old --new-npz-dir /path/new \\
        --k 12 --limit 600 --out-dir viz/pattern_analysis
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid                    # noqa: E402
from scripts.viz_radial_kymograph import (              # noqa: E402
    _sample_radial_kymograph)


# --- data loading ---------------------------------------------------

def _load_a_for_dir(Phi, npz_dir, nx, ny, nt, drop, limit, seed):
    """Modal coefficient trajectories a (n_sim, K, Nt) for a random
    subsample of a dataset dir, projected onto the common Phi."""
    import os
    import tempfile
    from data.loader import load_dataset
    src = Path(npz_dir)
    files = sorted(p for p in src.glob("*.npz")
                   if not p.name.startswith("_"))
    if not files:
        raise ValueError(f"no npz in {npz_dir}")
    rng = np.random.default_rng(seed)
    n = min(limit, len(files))
    pick = rng.choice(len(files), size=n, replace=False)
    tmp = tempfile.TemporaryDirectory(prefix="patanalysis_")
    try:
        for i in pick:
            f = files[int(i)].resolve()
            os.symlink(f, Path(tmp.name) / f.name)
        _x, _y, sims = load_dataset(tmp.name, nx=nx, ny=ny, nt=nt,
                                    limit=None, drop_first_steps=drop)
    finally:
        tmp.cleanup()
    nspace, K = Phi.shape
    a = np.empty((len(sims), K, nt), dtype=np.float64)
    for i, s in enumerate(sims):
        f = np.asarray(s.f, dtype=np.float64).reshape(nspace, -1)
        a[i] = Phi.T @ f
    return a


# --- mode characterization ------------------------------------------

def _azimuthal_score(Phi, nx, ny, n_theta=48, n_r=64):
    """Per-mode azimuthal-vs-radial variation ratio. High => the mode
    varies mostly with theta (asymmetric); low => radially symmetric.

    For each mode, sample it on a polar grid and compute
    var_over_theta(mean over r) vs var_over_r(mean over theta)."""
    x, y = canonical_grid(nx, ny)
    K = Phi.shape[1]
    scores = np.zeros(K)
    for k in range(K):
        fk = Phi[:, k].reshape(nx, ny)[:, :, None]        # (nx,ny,1)
        # sample mode k on polar grid (single "time" slice)
        slabs = [_sample_radial_kymograph(
            fk.astype(np.float64), x, y, float(th), n_r=n_r,
            r_max=1.0) for th in np.linspace(0, 90, n_theta)]
        P = np.stack(slabs, axis=0)[:, :, 0]              # (n_theta, n_r)
        # radial profile (avg over theta) and its variation over r
        rad_prof = P.mean(axis=0)                          # (n_r,)
        var_r = float(np.var(rad_prof))
        # azimuthal profile (avg over r) and its variation over theta
        az_prof = P.mean(axis=1)                            # (n_theta,)
        var_th = float(np.var(az_prof))
        scores[k] = var_th / (var_r + var_th + 1e-12)
    return scores


# --- feature extraction ---------------------------------------------

def _time_feature(a):
    """Bonding-progress curve per sim: mode-1 coefficient over time,
    sign-normalized to descend, then min-max scaled to [0,1]. Shape
    (n_sim, Nt). Direct release rises fast+monotone; hold plateaus
    first."""
    a1 = a[:, 0, :].copy()                                 # (n_sim, Nt)
    # orient so the process goes 0 -> 1 (progress), regardless of sign
    if a1[:, -1].mean() < a1[:, 0].mean():
        a1 = -a1
    lo = a1.min(axis=1, keepdims=True)
    hi = a1.max(axis=1, keepdims=True)
    return (a1 - lo) / (hi - lo + 1e-12)


def _modal_energy(a):
    """Per-sim per-mode energy ||a_k||^2 over time, row-normalized to
    fractions. Shape (n_sim, K)."""
    e = (a ** 2).sum(axis=2)                               # (n_sim, K)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _asymmetry_ratio(energy_frac, az_score, az_thresh):
    """Per-sim asymmetry: fraction of energy in azimuthal modes."""
    az_mask = az_score >= az_thresh
    return energy_frac[:, az_mask].sum(axis=1)             # (n_sim,)


# --- clustering (light, dependency-free) ----------------------------

def _kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        newc = np.stack([X[lab == j].mean(0) if (lab == j).any()
                         else c[j] for j in range(k)])
        if np.allclose(newc, c):
            break
        c = newc
    return lab, c


def _silhouette(X, lab):
    k = lab.max() + 1
    if k < 2:
        return 0.0
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    sil = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        same[i] = False
        a_i = D[i, same].mean() if same.any() else 0.0
        b_i = np.inf
        for j in range(k):
            if j == lab[i]:
                continue
            m = lab == j
            if m.any():
                b_i = min(b_i, D[i, m].mean())
        sil[i] = (b_i - a_i) / (max(a_i, b_i) + 1e-12)
    return float(sil.mean())


def _time_cluster(time_feat, k_hint):
    """Cluster the time-progress curves; try k=2..max(k_hint,4) and
    keep the best silhouette."""
    best = None
    for k in range(2, max(k_hint, 4) + 1):
        lab, _ = _kmeans(time_feat, k, seed=0)
        s = _silhouette(time_feat, lab)
        if best is None or s > best[0]:
            best = (s, k, lab)
    return best[2], best[1], best[0]


# --- rendering ------------------------------------------------------

def _render(res, out_dir, value_scale=1.0e6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = res["source"]           # (n_sim,) 0=old 1=new
    tlab = res["time_label"]      # (n_sim,)
    asym = res["asym"]            # (n_sim,)
    energy = res["energy"]        # (n_sim, K)
    az = res["az_score"]          # (K,)
    K = energy.shape[1]
    is_new = src == 1
    is_old = src == 0
    # symmetric vs asymmetric by the asym threshold (median gap)
    asym_thr = res["asym_split"]
    is_asym = asym >= asym_thr

    # ---- Fig 1: the two axes (time cluster + asymmetry) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             constrained_layout=True)
    # 1a: time-progress curves colored by time cluster
    tf = res["time_feat"]
    t_axis = np.linspace(0, 1, tf.shape[1])
    for c in np.unique(tlab):
        m = tlab == c
        axes[0].plot(t_axis, tf[m].mean(0), lw=2,
                     label=f"time-cluster {c} (n={int(m.sum())})")
        axes[0].fill_between(t_axis, tf[m].mean(0) - tf[m].std(0),
                             tf[m].mean(0) + tf[m].std(0), alpha=0.15)
    axes[0].set_xlabel("normalized time")
    axes[0].set_ylabel("bonding progress (mode-1, scaled)")
    axes[0].set_title("TIME axis: direct vs hold-then-release")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # 1b: asymmetry histogram, old vs new
    bins = np.linspace(0, max(asym.max(), 1e-3), 30)
    axes[1].hist(asym[is_old], bins=bins, alpha=0.6, label="old",
                 color="#3d5a80")
    axes[1].hist(asym[is_new], bins=bins, alpha=0.6, label="new",
                 color="#e63946")
    axes[1].axvline(asym_thr, color="0.3", ls="--",
                    label=f"split={asym_thr:.3f}")
    axes[1].set_xlabel("asymmetry ratio (azimuthal energy frac)")
    axes[1].set_ylabel("count")
    axes[1].set_title("SPACE axis: symmetric vs asymmetric")
    axes[1].legend(fontsize=8)
    # 1c: 2x2 pattern grid, old vs new counts
    quad = np.array([["direct/sym", "direct/asym"],
                     ["hold/sym", "hold/asym"]])
    # map time clusters to direct(0)/hold(1) by which rises faster
    tf_rise = np.array([tf[tlab == c].mean(0)[:tf.shape[1] // 3].mean()
                        for c in range(tlab.max() + 1)])
    direct_c = int(np.argmax(tf_rise))     # fastest early rise = direct
    is_hold = tlab != direct_c
    counts = np.zeros((2, 2, 2), dtype=int)   # [hold, asym, src]
    for h in (0, 1):
        for aq in (0, 1):
            sel = (is_hold.astype(int) == h) & (is_asym.astype(int) == aq)
            counts[h, aq, 0] = int((sel & is_old).sum())
            counts[h, aq, 1] = int((sel & is_new).sum())
    axes[2].axis("off")
    txt = ["2x2 pattern occupancy (old / new)\n"]
    for h, hn in enumerate(["direct", "hold"]):
        for aq, an in enumerate(["sym", "asym"]):
            txt.append(f"  {hn:6s}/{an:4s}:  "
                       f"old={counts[h, aq, 0]:4d}   "
                       f"new={counts[h, aq, 1]:4d}")
    axes[2].text(0.0, 0.95, "\n".join(txt), va="top", ha="left",
                 family="monospace", fontsize=11,
                 transform=axes[2].transAxes)
    fig.suptitle("Bonding patterns along two physical axes",
                 fontweight="bold")
    (out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "01_two_axes.png", dpi=140,
                bbox_inches="tight")
    fig.savefig(out_dir / "01_two_axes.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: pattern x mode occupancy heatmap ----
    # define 4 patterns = (direct/hold) x (sym/asym)
    pat_id = is_hold.astype(int) * 2 + is_asym.astype(int)
    pat_names = ["direct/sym", "direct/asym", "hold/sym", "hold/asym"]
    occ = np.zeros((4, K))
    for p in range(4):
        m = pat_id == p
        occ[p] = energy[m].mean(0) if m.any() else 0.0
    fig, ax = plt.subplots(figsize=(1.1 * K + 2, 4.2),
                           constrained_layout=True)
    im = ax.imshow(occ, aspect="auto", cmap="magma",
                   interpolation="nearest")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"m{k+1}" for k in range(K)])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{pat_names[p]}\n(n={int((pat_id==p).sum())})"
                        for p in range(4)])
    # mark azimuthal modes
    for k in range(K):
        if az[k] >= res["az_thresh"]:
            ax.text(k, -0.6, "az", ha="center", fontsize=8,
                    color="#e63946")
    ax.set_title("Pattern x mode occupancy (mean energy fraction; "
                 "'az' = azimuthal mode)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(out_dir / "02_pattern_mode_occupancy.png", dpi=140,
                bbox_inches="tight")
    fig.savefig(out_dir / "02_pattern_mode_occupancy.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: floor K8 vs K12 per pattern ----
    fk = res["floor_k"]           # dict k -> (n_sim,) floor
    ks = sorted(fk.keys())
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    xg = np.arange(4)
    w = 0.8 / len(ks)
    for i, kk in enumerate(ks):
        vals = [float(np.mean(fk[kk][pat_id == p])) * value_scale
                if (pat_id == p).any() else 0.0 for p in range(4)]
        ax.bar(xg + i * w, vals, w, label=f"K={kk}")
    ax.set_xticks(xg + w * (len(ks) - 1) / 2)
    ax.set_xticklabels(pat_names)
    ax.set_ylabel(f"truncation floor (|f_perp| * {value_scale:g})")
    ax.set_title("Per-pattern POD truncation floor vs K "
                 "(drop = extra modes capture that pattern)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "03_floor_by_pattern.png", dpi=140,
                bbox_inches="tight")
    fig.savefig(out_dir / "03_floor_by_pattern.pdf",
                bbox_inches="tight")
    plt.close(fig)


def _write_summary(res, out_dir):
    src, tlab, asym = res["source"], res["time_label"], res["asym"]
    energy, az = res["energy"], res["az_score"]
    is_new = src == 1
    is_asym = asym >= res["asym_split"]
    tf = res["time_feat"]
    tf_rise = np.array([tf[tlab == c].mean(0)[:tf.shape[1] // 3].mean()
                        for c in range(tlab.max() + 1)])
    is_hold = tlab != int(np.argmax(tf_rise))
    pat_id = is_hold.astype(int) * 2 + is_asym.astype(int)
    pat_names = ["direct/sym", "direct/asym", "hold/sym", "hold/asym"]
    K = energy.shape[1]
    az_modes = [k + 1 for k in range(K) if az[k] >= res["az_thresh"]]

    lines = ["# Dataset pattern analysis\n"]
    lines.append(f"Sims: old={int((src==0).sum())} "
                 f"new={int(is_new.sum())}\n")
    lines.append(f"Time clusters: {res['n_time_clusters']} "
                 f"(silhouette {res['time_sil']:.3f})\n")
    lines.append(f"Asymmetry split threshold: {res['asym_split']:.4f}\n")
    lines.append(f"Azimuthal (asymmetric) modes: {az_modes}\n\n")
    lines.append("## 2x2 pattern occupancy (old / new)\n")
    for p in range(4):
        m = pat_id == p
        lines.append(f"  {pat_names[p]:12s}: old="
                     f"{int((m & (src==0)).sum()):4d}  "
                     f"new={int((m & is_new).sum()):4d}\n")
    # which modes are new-vs-old differentiating
    e_old = energy[src == 0].mean(0)
    e_new = energy[src == 1].mean(0) if is_new.any() else e_old
    diff = e_new - e_old
    top = np.argsort(-np.abs(diff))[:5]
    lines.append("\n## Modes most differently excited (new - old)\n")
    for k in top:
        lines.append(f"  mode {k+1}: old={e_old[k]:.3f} "
                     f"new={e_new[k]:.3f}  delta={diff[k]:+.3f}"
                     f"{'  (azimuthal)' if az[k]>=res['az_thresh'] else ''}\n")
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("".join(lines))
    print("".join(lines))


def analyze(basis_path, old_dir, new_dir, K, nt, drop, limit,
            k_hint, az_thresh, seed) -> dict:
    with np.load(basis_path) as z:
        Phi = z["Phi"][:, :K]
        sigma = z["sigma"][:K]
        nx, ny = (int(d) for d in z["spatial_shape"])
    az_score = _azimuthal_score(Phi, nx, ny)

    a_old = _load_a_for_dir(Phi, old_dir, nx, ny, nt, drop, limit,
                            seed)
    a_new = _load_a_for_dir(Phi, new_dir, nx, ny, nt, drop, limit,
                            seed + 1)
    a = np.concatenate([a_old, a_new], axis=0)
    src = np.concatenate([np.zeros(len(a_old), int),
                          np.ones(len(a_new), int)])

    time_feat = _time_feature(a)
    energy = _modal_energy(a)
    asym = _asymmetry_ratio(energy, az_score, az_thresh)
    tlab, n_tc, sil = _time_cluster(time_feat, k_hint)

    # asymmetry split: midpoint between the two natural groups (Otsu-ish
    # via the gap in a sorted asym array)
    s = np.sort(asym)
    gaps = np.diff(s)
    split = float((s[gaps.argmax()] + s[gaps.argmax() + 1]) / 2) \
        if len(s) > 1 else 0.0

    # truncation floor per sim at K=8 and K (=12): fraction of energy
    # NOT captured by the first m modes, using the full-K energy as a
    # proxy for total (we only have K modes, so this is relative among
    # the retained modes -- the drop between 8 and 12 is what matters).
    e_raw = (a ** 2).sum(axis=2)                            # (n_sim, K)
    total = e_raw.sum(axis=1) + 1e-12
    floor_k = {}
    for m in (8, K):
        if m <= K:
            floor_k[m] = np.sqrt(
                np.maximum(total - e_raw[:, :m].sum(axis=1), 0.0)
                / total)
    return dict(source=src, energy=energy, az_score=az_score,
                az_thresh=az_thresh, time_feat=time_feat,
                time_label=tlab, n_time_clusters=n_tc, time_sil=sil,
                asym=asym, asym_split=split, floor_k=floor_k,
                sigma=sigma)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True,
                    help="merged POD basis pod3d_*.npz (common frame)")
    ap.add_argument("--old-npz-dir", required=True)
    ap.add_argument("--new-npz-dir", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=600,
                    help="random sims per dataset (default 600)")
    ap.add_argument("--k-hint", type=int, default=2,
                    help="rough number of TIME patterns you expect "
                    "(default 2: direct + hold)")
    ap.add_argument("--az-thresh", type=float, default=0.35,
                    help="azimuthal-score threshold above which a "
                    "mode counts as asymmetric (default 0.35)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="viz/pattern_analysis")
    args = ap.parse_args()

    if not Path(args.basis).is_file():
        print(f"basis not found: {args.basis}", file=sys.stderr)
        return 2
    res = analyze(args.basis, args.old_npz_dir, args.new_npz_dir,
                  args.k, args.nt, args.drop_first_steps, args.limit,
                  args.k_hint, args.az_thresh, args.seed)
    out = Path(args.out_dir)
    _render(res, out)
    _write_summary(res, out)
    print(f"\nwrote figures + summary.md to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
