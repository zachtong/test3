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
    """Modal coefficients a (n_sim, K, Nt) projected onto the common
    Phi, AND the TRUE total field energy ||f||^2 per sim (n_sim,).

    The true field energy is required for a meaningful truncation
    floor: using only the retained modes' energy as the total makes
    the floor at m=K identically zero (there is nothing left to
    truncate), which hides the very drop we want to measure."""
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
    fe = np.empty(len(sims), dtype=np.float64)
    for i, s in enumerate(sims):
        f = np.asarray(s.f, dtype=np.float64).reshape(nspace, -1)
        a[i] = Phi.T @ f
        fe[i] = float(np.einsum("ij,ij->", f, f))    # ||f||^2
    return a, fe


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


def _patterns(res):
    """Generalized pattern grouping: (n_time_clusters) x (sym/asym).

    Time clusters are RENAMED by how fast they rise early on, so the
    fastest is 'direct' and the rest are 'hold-1', 'hold-2', ... in
    increasing hold length. This keeps the grouping correct when the
    silhouette scan finds more than two time patterns (e.g. direct +
    two different hold durations), which a hard-coded 2x2 would
    wrongly merge.

    Returns (pat_id, pat_names, time_names, is_asym)."""
    tf = res["time_feat"]
    tlab = res["time_label"]
    n_tc = int(tlab.max()) + 1
    early = tf.shape[1] // 3
    rise = np.array([tf[tlab == c].mean(0)[:early].mean()
                     if (tlab == c).any() else -np.inf
                     for c in range(n_tc)])
    order = np.argsort(-rise)            # fastest riser first
    rank = {int(c): i for i, c in enumerate(order)}
    time_names = ["direct" if i == 0 else f"hold-{i}"
                  for i in range(n_tc)]
    t_idx = np.array([rank[int(c)] for c in tlab])
    is_asym = res["asym"] >= res["asym_split"]
    pat_id = t_idx * 2 + is_asym.astype(int)
    pat_names = [f"{time_names[i]}/{s}"
                 for i in range(n_tc) for s in ("sym", "asym")]
    return pat_id, pat_names, time_names, is_asym


# --- rendering ------------------------------------------------------

def _render(res, out_dir, value_scale=1.0e6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = res["source"]
    asym = res["asym"]
    energy = res["energy"]
    az = res["az_score"]
    K = energy.shape[1]
    is_old, is_new = src == 0, src == 1
    pat_id, pat_names, time_names, is_asym = _patterns(res)
    n_pat = len(pat_names)

    # ---- Fig 1: the two axes ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8),
                             constrained_layout=True)
    tf = res["time_feat"]
    tlab = res["time_label"]
    t_axis = np.linspace(0, 1, tf.shape[1])
    early = tf.shape[1] // 3
    rise = np.array([tf[tlab == c].mean(0)[:early].mean()
                     if (tlab == c).any() else -np.inf
                     for c in range(int(tlab.max()) + 1)])
    for i, c in enumerate(np.argsort(-rise)):
        m = tlab == c
        if not m.any():
            continue
        axes[0].plot(t_axis, tf[m].mean(0), lw=2,
                     label=f"{time_names[i]} (n={int(m.sum())})")
        axes[0].fill_between(t_axis, tf[m].mean(0) - tf[m].std(0),
                             tf[m].mean(0) + tf[m].std(0), alpha=0.15)
    axes[0].set_xlabel("normalized time")
    axes[0].set_ylabel("bonding progress (mode-1, scaled)")
    axes[0].set_title(f"TIME axis: {len(time_names)} clusters found")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    hi = float(np.percentile(asym, 99.5)) if asym.size else 1e-3
    bins = np.linspace(0, max(hi, 1e-3), 40)
    axes[1].hist(asym[is_old], bins=bins, alpha=0.6, label="old",
                 color="#3d5a80")
    axes[1].hist(asym[is_new], bins=bins, alpha=0.6, label="new",
                 color="#e63946")
    axes[1].axvline(res["asym_split"], color="0.2", ls="--", lw=1.8,
                    label=f"split={res['asym_split']:.4f}\n"
                          f"({res.get('split_source', 'fixed')})")
    axes[1].set_xlabel("asymmetry ratio (azimuthal energy fraction)")
    axes[1].set_ylabel("count")
    axes[1].set_title("SPACE axis: symmetric vs asymmetric")
    axes[1].legend(fontsize=8)

    axes[2].axis("off")
    txt = [f"Pattern occupancy  ({len(time_names)} time x 2 sym)\n",
           f"{'pattern':16s} {'old':>6s} {'new':>6s}"]
    for p in range(n_pat):
        m = pat_id == p
        txt.append(f"{pat_names[p]:16s} "
                   f"{int((m & is_old).sum()):6d} "
                   f"{int((m & is_new).sum()):6d}")
    axes[2].text(0.0, 0.98, "\n".join(txt), va="top", ha="left",
                 family="monospace", fontsize=10.5,
                 transform=axes[2].transAxes)
    fig.suptitle("Bonding patterns along two physical axes",
                 fontweight="bold")
    fig.savefig(out_dir / "01_two_axes.png", dpi=140,
                bbox_inches="tight")
    fig.savefig(out_dir / "01_two_axes.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: occupancy, log scale + per-mode contrast ----
    occ = np.zeros((n_pat, K))
    for p in range(n_pat):
        m = pat_id == p
        occ[p] = energy[m].mean(0) if m.any() else 0.0
    fig, axes = plt.subplots(2, 1, figsize=(1.15 * K + 3, 7.5),
                             constrained_layout=True)
    # (a) absolute energy fraction, LOG scale so modes far below
    # mode-1 remain visible instead of being crushed to black.
    vmin = max(float(occ[occ > 0].min()) if (occ > 0).any() else 1e-6,
               1e-6)
    im0 = axes[0].imshow(np.maximum(occ, vmin), aspect="auto",
                         cmap="magma", interpolation="nearest",
                         norm=LogNorm(vmin=vmin, vmax=occ.max()))
    axes[0].set_title("mean energy fraction per mode (LOG scale)")
    fig.colorbar(im0, ax=axes[0], fraction=0.03)
    # (b) per-mode contrast: each column divided by its mean across
    # patterns -- shows which modes DISTINGUISH patterns, which is
    # the actual question (mode-1 dominance cancels out here).
    col_mean = occ.mean(axis=0, keepdims=True) + 1e-12
    contrast = occ / col_mean
    im1 = axes[1].imshow(contrast, aspect="auto", cmap="RdBu_r",
                         interpolation="nearest", vmin=0, vmax=2)
    axes[1].set_title("per-mode contrast (energy / column mean); "
                      "red = this pattern over-excites this mode")
    fig.colorbar(im1, ax=axes[1], fraction=0.03)
    for ax in axes:
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"m{k+1}" for k in range(K)])
        ax.set_yticks(range(n_pat))
        ax.set_yticklabels([f"{pat_names[p]}\n"
                            f"(n={int((pat_id==p).sum())})"
                            for p in range(n_pat)], fontsize=8)
        for k in range(K):
            if az[k] >= res["az_thresh"]:
                ax.text(k, -0.62, "az", ha="center", fontsize=8,
                        color="#2a9d8f", fontweight="bold")
    fig.suptitle("Pattern x mode occupancy  ('az' = azimuthal mode)",
                 fontweight="bold")
    fig.savefig(out_dir / "02_pattern_mode_occupancy.png", dpi=140,
                bbox_inches="tight")
    fig.savefig(out_dir / "02_pattern_mode_occupancy.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: truncation floor per pattern vs K ----
    fk = res["floor_k"]
    ks = sorted(fk.keys())
    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * n_pat, 5),
                           constrained_layout=True)
    xg = np.arange(n_pat)
    w = 0.8 / max(len(ks), 1)
    for i, kk in enumerate(ks):
        vals = [float(np.mean(fk[kk][pat_id == p])) * 100.0
                if (pat_id == p).any() else 0.0
                for p in range(n_pat)]
        b = ax.bar(xg + i * w, vals, w, label=f"K={kk}")
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xg + w * (len(ks) - 1) / 2)
    ax.set_xticklabels(pat_names, fontsize=8)
    ax.set_ylabel("truncation floor (% of field energy)")
    ax.set_title("Per-pattern POD truncation floor vs K "
                 "(drop = the extra modes capture that pattern)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "03_floor_by_pattern.png", dpi=140,
                bbox_inches="tight")
    fig.savefig(out_dir / "03_floor_by_pattern.pdf",
                bbox_inches="tight")
    plt.close(fig)


def _write_summary(res, out_dir):
    src, energy, az = res["source"], res["energy"], res["az_score"]
    is_new = src == 1
    pat_id, pat_names, time_names, is_asym = _patterns(res)
    K = energy.shape[1]
    az_modes = [k + 1 for k in range(K) if az[k] >= res["az_thresh"]]

    L = ["# Dataset pattern analysis\n\n"]
    L.append(f"Sims: old={int((src==0).sum())} "
             f"new={int(is_new.sum())}\n")
    L.append(f"Time clusters found: {res['n_time_clusters']} "
             f"({', '.join(time_names)}), "
             f"silhouette {res['time_sil']:.3f}\n")
    L.append(f"Asymmetry split: {res['asym_split']:.4f} "
             f"({res.get('split_source', 'fixed')})\n")
    L.append(f"Azimuthal modes: {az_modes}\n\n")
    L.append("## Pattern occupancy (old / new)\n")
    for p in range(len(pat_names)):
        m = pat_id == p
        L.append(f"  {pat_names[p]:16s}: old="
                 f"{int((m & (src==0)).sum()):4d}  "
                 f"new={int((m & is_new).sum()):4d}\n")
    e_old = energy[src == 0].mean(0)
    e_new = energy[is_new].mean(0) if is_new.any() else e_old
    diff = e_new - e_old
    L.append("\n## Modes most differently excited (new - old)\n")
    for k in np.argsort(-np.abs(diff))[:6]:
        L.append(f"  mode {k+1:2d}: old={e_old[k]:.4f} "
                 f"new={e_new[k]:.4f}  delta={diff[k]:+.4f}"
                 f"{'  (azimuthal)' if az[k]>=res['az_thresh'] else ''}\n")
    fk = res["floor_k"]
    ks = sorted(fk.keys())
    if len(ks) >= 2:
        L.append(f"\n## Truncation floor drop K={ks[0]} -> K={ks[-1]}"
                 f" (% of field energy)\n")
        for p in range(len(pat_names)):
            m = pat_id == p
            if not m.any():
                continue
            lo = float(np.mean(fk[ks[0]][m])) * 100
            hi = float(np.mean(fk[ks[-1]][m])) * 100
            L.append(f"  {pat_names[p]:16s}: {lo:6.3f} -> {hi:6.3f}"
                     f"   (drop {lo-hi:+.3f})\n")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "summary.md").write_text("".join(L))
    print("".join(L))



def analyze(basis_path, old_dir, new_dir, K, nt, drop, limit,
            k_hint, az_thresh, seed, asym_pctl=None,
            asym_split_override=None,
            asym_split_default=0.002) -> dict:
    with np.load(basis_path) as z:
        Phi = z["Phi"][:, :K]
        sigma = z["sigma"][:K]
        nx, ny = (int(d) for d in z["spatial_shape"])
    az_score = _azimuthal_score(Phi, nx, ny)

    a_old, fe_old = _load_a_for_dir(Phi, old_dir, nx, ny, nt, drop,
                                    limit, seed)
    a_new, fe_new = _load_a_for_dir(Phi, new_dir, nx, ny, nt, drop,
                                    limit, seed + 1)
    a = np.concatenate([a_old, a_new], axis=0)
    field_energy = np.concatenate([fe_old, fe_new])
    src = np.concatenate([np.zeros(len(a_old), int),
                          np.ones(len(a_new), int)])

    time_feat = _time_feature(a)
    energy = _modal_energy(a)
    asym = _asymmetry_ratio(energy, az_score, az_thresh)
    tlab, n_tc, sil = _time_cluster(time_feat, k_hint)

    # Asymmetry split. "How much azimuthal energy still counts as
    # symmetric" is a PHYSICAL judgement, so the default is a small
    # fixed threshold (asym_split_default) meaning "essentially any
    # azimuthal content is asymmetric". --asym-split sets it
    # explicitly. --asym-pctl (old-dataset percentile) is offered as
    # a data-driven alternative but is NOT the default, because the
    # distribution is continuous and a percentile lands too high.
    if asym_split_override is not None:
        split = float(asym_split_override)
        split_source = "--asym-split"
    elif asym_pctl is not None:
        ref = asym[src == 0]
        split = float(np.percentile(ref, asym_pctl)) if ref.size \
            else float(np.median(asym))
        split_source = f"old p{asym_pctl:g}"
    else:
        split = float(asym_split_default)
        split_source = "default fixed"

    # Truncation floor per sim at m modes, against the TRUE total
    # field energy ||f||^2 (not the retained-mode sum -- that would
    # make the floor at m=K identically zero and hide the drop).
    e_raw = (a ** 2).sum(axis=2)                            # (n_sim, K)
    total = field_energy + 1e-30
    floor_k = {}
    for m in sorted({8, K}):
        if m <= K:
            floor_k[m] = np.sqrt(
                np.maximum(total - e_raw[:, :m].sum(axis=1), 0.0)
                / total)
    return dict(source=src, energy=energy, az_score=az_score,
                az_thresh=az_thresh, time_feat=time_feat,
                time_label=tlab, n_time_clusters=n_tc, time_sil=sil,
                asym=asym, asym_split=split,
                split_source=split_source,
                floor_k=floor_k, field_energy=field_energy,
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
                    "MODE counts as azimuthal (default 0.35)")
    ap.add_argument("--asym-split", type=float, default=None,
                    help="absolute asymmetry-ratio threshold: a SIM "
                    "with more azimuthal-energy fraction than this "
                    "is asymmetric. This is a PHYSICAL choice -- set "
                    "it to how much azimuthal content you consider "
                    "still symmetric (e.g. 0.002). Default: "
                    "0.002.")
    ap.add_argument("--asym-pctl", type=float, default=None,
                    help="ALTERNATIVE data-driven split: asymmetric "
                    "if above this percentile of the OLD (reference) "
                    "dataset. Off by default; --asym-split is "
                    "preferred since the distribution is continuous.")
    ap.add_argument("--asym-split-default", type=float, default=0.002,
                    help="fixed split used when neither --asym-split "
                    "nor --asym-pctl is given (default 0.002)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="viz/pattern_analysis")
    args = ap.parse_args()

    if not Path(args.basis).is_file():
        print(f"basis not found: {args.basis}", file=sys.stderr)
        return 2
    res = analyze(args.basis, args.old_npz_dir, args.new_npz_dir,
                  args.k, args.nt, args.drop_first_steps, args.limit,
                  args.k_hint, args.az_thresh, args.seed,
                  asym_pctl=args.asym_pctl,
                  asym_split_override=args.asym_split,
                  asym_split_default=args.asym_split_default)
    out = Path(args.out_dir)
    _render(res, out)
    _write_summary(res, out)
    print(f"\nwrote figures + summary.md to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
