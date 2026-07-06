"""Deep analysis of the sensor-subset sweep. Produces a self-
contained HTML report + presentation-ready PNG figures under
viz/sweep_summary/.

Reads outputs/sweep_*/results.json produced by run_sweep.py and
aggregates across the 57 configs (C(6, n) for n in {2..6}, single
seed=7). Every figure references the same six-position catalogue:

    A = (r=0.520, theta=0 deg)     inner-X
    B = (r=0.520, theta=45 deg)    inner-D
    C = (r=0.520, theta=90 deg)    inner-Y
    D = (r=0.847, theta=0 deg)     outer-X
    E = (r=0.847, theta=45 deg)    outer-D
    F = (r=0.847, theta=90 deg)    outer-Y

Run:
    python scripts/analyze_sweep.py
    # or specify a different outputs / out dir:
    python scripts/analyze_sweep.py --outputs outputs \\
        --out-dir viz/sweep_summary --prefix sweep_
"""
from __future__ import annotations
import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- Sensor catalogue ---
POSITIONS = [
    ("A", 0.520, 0.0),
    ("B", 0.520, 45.0),
    ("C", 0.520, 90.0),
    ("D", 0.847, 0.0),
    ("E", 0.847, 45.0),
    ("F", 0.847, 90.0),
]
LETTERS = [p[0] for p in POSITIONS]

# One consistent color per sensor, reused across every plot so the
# catalog diagram acts as the single legend for the whole report.
SENSOR_COLORS = {
    "A": "#1f77b4",   # blue
    "B": "#2ca02c",   # green
    "C": "#9467bd",   # purple
    "D": "#ff7f0e",   # orange
    "E": "#d62728",   # red
    "F": "#8c564b",   # brown
}

INNER = {"A", "B", "C"}
OUTER = {"D", "E", "F"}
ANG0 = {"A", "D"}      # theta = 0 deg
ANG45 = {"B", "E"}     # theta = 45 deg
ANG90 = {"C", "F"}     # theta = 90 deg


def _code_from_positions(positions) -> str:
    coords_to_letter = {(round(p[1], 3), round(p[2], 3)): p[0]
                          for p in POSITIONS}
    letters = []
    for pos in positions:
        r = round(float(pos[0]), 3)
        th = round(float(pos[1]), 3)
        letters.append(coords_to_letter.get((r, th), "?"))
    return "".join(sorted(letters))


def load_rows(outputs_dir: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for tag_dir in sorted(outputs_dir.glob(f"{prefix}*")):
        res = tag_dir / "results.json"
        if not res.is_file():
            continue
        try:
            d = json.loads(res.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {tag_dir.name}: {type(e).__name__}: {e}")
            continue
        cfg = d.get("config", {}) or {}
        sens_cfg = cfg.get("sensors", {}) or {}
        positions = sens_cfg.get("positions") or []
        gs = d.get("global_stats", {}) or {}
        rows.append({
            "tag": tag_dir.name,
            "n": len(positions),
            "code": _code_from_positions(positions),
            "positions": positions,
            "median": gs.get("median"),
            "p95": gs.get("p95"),
            "gap_to_floor": d.get("gap_to_floor"),
            "n_params": d.get("n_params"),
            "per_sim_field_errs": d.get("per_sim_field_errs") or [],
        })
    return rows


def _cart(r: float, th_deg: float) -> tuple[float, float]:
    t = np.deg2rad(th_deg)
    return float(r * np.cos(t)), float(r * np.sin(t))


# --- Figures ---

def plot_sensor_positions(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    thetas = np.linspace(0, 90, 200)
    # outer arc (physical rim)
    ax.plot(np.cos(np.deg2rad(thetas)), np.sin(np.deg2rad(thetas)),
             color="0.35", lw=2.0)
    ax.text(np.cos(np.deg2rad(45)) * 1.02,
             np.sin(np.deg2rad(45)) * 1.02,
             "arc r=1", fontsize=9, color="0.35", ha="left")
    # x, y axes
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    # sensor rings as dashed arcs
    for r_ring, label in [(0.520, "r=0.520 (inner)"),
                            (0.847, "r=0.847 (outer)")]:
        xs = r_ring * np.cos(np.deg2rad(thetas))
        ys = r_ring * np.sin(np.deg2rad(thetas))
        ax.plot(xs, ys, color="0.6", lw=1, ls="--")
        ax.text(xs[int(len(xs) * 0.62)], ys[int(len(ys) * 0.62)],
                 label, fontsize=8, color="0.4",
                 ha="left", va="bottom")
    # sensors
    for letter, r, th in POSITIONS:
        x, y = _cart(r, th)
        ax.scatter([x], [y], s=280,
                    color=SENSOR_COLORS[letter],
                    edgecolor="black", linewidth=1.4, zorder=5)
        ax.annotate(letter, (x, y), xytext=(10, 10),
                     textcoords="offset points",
                     fontsize=15, fontweight="bold",
                     color=SENSOR_COLORS[letter])
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R (normalized)")
    ax.set_ylabel("y / R (normalized)")
    ax.set_title("Sensor catalogue: 6 physical positions")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diminishing_returns(rows: list[dict], out_path: Path) -> None:
    by_n: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["median"] is not None:
            by_n[r["n"]].append(float(r["median"]))
    ns = sorted(by_n.keys())
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bp = ax.boxplot([by_n[n] for n in ns], positions=ns,
                     widths=0.45, patch_artist=True,
                     boxprops=dict(facecolor="#e7ecf7",
                                    edgecolor="#3d5a80"),
                     medianprops=dict(color="#3d5a80", linewidth=1.5),
                     whiskerprops=dict(color="#3d5a80"),
                     capprops=dict(color="#3d5a80"),
                     flierprops=dict(marker="o", markersize=3,
                                      alpha=0.4))
    for n in ns:
        vals = by_n[n]
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.full(len(vals), n) + jitter, vals,
                    alpha=0.55, s=28, color="#3d5a80",
                    edgecolors="none")
    best = [(n, min(by_n[n])) for n in ns]
    ax.plot([b[0] for b in best], [b[1] for b in best], "o-",
             color="#e63946", linewidth=2.2, markersize=10,
             label="best per n", zorder=6)
    for n, v in best:
        ax.annotate(f"{v:.4f}", (n, v), xytext=(10, -4),
                     textcoords="offset points",
                     color="#e63946", fontsize=9, fontweight="bold")
    # highlight sweet spot
    sweet_n = min(best, key=lambda x: x[1])[0]
    ax.axvline(sweet_n, color="#e63946", ls=":", lw=1,
                alpha=0.4)
    ax.set_xlabel("n (number of sensors)")
    ax.set_ylabel("median field error")
    ax.set_title("Diminishing returns: error vs sensor count "
                 f"(sweet spot n={sweet_n})")
    ax.set_xticks(ns)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sensor_impact(rows: list[dict]) -> dict[str, float]:
    """For each letter, average (mean_err_with - mean_err_without)
    across n. Negative = including this sensor tends to lower the
    error (helpful); positive = neutral or harmful. Stratifying by
    n controls for the confound that "n=6 configs always contain
    every letter but n=2 configs never do."""
    impact: dict[str, float] = {}
    ns = sorted(set(r["n"] for r in rows))
    for letter in LETTERS:
        deltas: list[float] = []
        for n in ns:
            n_rows = [r for r in rows
                      if r["n"] == n and r["median"] is not None]
            with_x = [float(r["median"]) for r in n_rows
                      if letter in r["code"]]
            without_x = [float(r["median"]) for r in n_rows
                         if letter not in r["code"]]
            if with_x and without_x:
                deltas.append(float(np.mean(with_x))
                              - float(np.mean(without_x)))
        impact[letter] = float(np.mean(deltas)) if deltas else 0.0
    return impact


def _top_k_frequency(rows: list[dict], k: int = 10) -> dict[str, int]:
    finite = [r for r in rows if r["median"] is not None]
    finite.sort(key=lambda r: float(r["median"]))
    top = finite[:k]
    freq = {L: 0 for L in LETTERS}
    for r in top:
        for L in r["code"]:
            if L in freq:
                freq[L] += 1
    return freq


def plot_sensor_importance(rows: list[dict], out_path: Path,
                             top_k: int = 10) -> None:
    impact = _sensor_impact(rows)
    freq = _top_k_frequency(rows, k=top_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              constrained_layout=True)
    # left: frequency in top-K
    ax = axes[0]
    letters_by_freq = sorted(LETTERS,
                             key=lambda L: -freq[L])
    values = [freq[L] for L in letters_by_freq]
    colors = [SENSOR_COLORS[L] for L in letters_by_freq]
    bars = ax.bar(letters_by_freq, values, color=colors,
                   edgecolor="black", linewidth=1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.15, f"{v}/{top_k}",
                 ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"count in top-{top_k}")
    ax.set_title(f"Sensor frequency in top-{top_k} configs")
    ax.set_ylim(0, top_k + 1)
    ax.grid(axis="y", alpha=0.3)

    # right: impact (delta median-err, controlled for n)
    ax = axes[1]
    letters_by_impact = sorted(LETTERS,
                                key=lambda L: impact[L])
    values = [impact[L] for L in letters_by_impact]
    colors = [SENSOR_COLORS[L] for L in letters_by_impact]
    bars = ax.bar(letters_by_impact, values, color=colors,
                   edgecolor="black", linewidth=1)
    for bar, v in zip(bars, values):
        offset = -0.0004 if v > 0 else 0.0004
        va = "top" if v > 0 else "bottom"
        ax.text(bar.get_x() + bar.get_width() / 2,
                 v + offset, f"{v:+.4f}",
                 ha="center", va=va, fontsize=9)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel("delta median err (with X) - (without X)")
    ax.set_title("Sensor impact (n-controlled, negative = helpful)")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _classify_composition(code: str) -> str:
    letters = set(code)
    inner = letters & INNER
    outer = letters & OUTER
    if inner and not outer:
        return "pure inner"
    if outer and not inner:
        return "pure outer"
    return "mixed"


def plot_ring_composition(rows: list[dict], out_path: Path) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["median"] is None:
            continue
        groups[_classify_composition(r["code"])].append(
            float(r["median"]))

    labels = ["pure inner", "mixed", "pure outer"]
    labels = [l for l in labels if groups.get(l)]
    data = [groups[l] for l in labels]
    counts = [len(g) for g in data]

    fig, ax = plt.subplots(figsize=(8, 5.2),
                             constrained_layout=True)
    palette = {"pure inner": "#8ecae6",
                "mixed": "#e9c46a",
                "pure outer": "#f4a261"}
    bp = ax.boxplot(data, positions=range(len(labels)),
                     widths=0.55, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.4))
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(palette.get(lab, "#cccccc"))
    rng = np.random.default_rng(43)
    for i, (lab, vals) in enumerate(zip(labels, data)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                    color="0.25", alpha=0.6, s=25,
                    edgecolors="none")
    for i, (lab, cnt) in enumerate(zip(labels, counts)):
        ax.text(i, ax.get_ylim()[1] * 0.98 if
                ax.get_ylim()[1] > 0 else max(max(data)) * 1.05,
                 f"n={cnt}", ha="center", fontsize=9,
                 color="0.35")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("median field error")
    ax.set_title("Ring composition: inner-only vs outer-only vs mixed")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_angular_coverage(rows: list[dict], out_path: Path) -> None:
    """Bucket by how many of the 3 angles (0, 45, 90) are represented
    in the config, ignoring how many sensors sit on each angle."""
    groups: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["median"] is None:
            continue
        letters = set(r["code"])
        covered = 0
        for ang_set in (ANG0, ANG45, ANG90):
            if letters & ang_set:
                covered += 1
        groups[covered].append(float(r["median"]))

    keys = sorted(groups.keys())
    data = [groups[k] for k in keys]
    counts = [len(g) for g in data]
    labels = [f"{k} of 3 angles" for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5.2),
                             constrained_layout=True)
    bp = ax.boxplot(data, positions=range(len(keys)),
                     widths=0.5, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.4))
    palette = ["#f4a261", "#e9c46a", "#8ecae6"]
    for patch, i in zip(bp["boxes"], range(len(keys))):
        patch.set_facecolor(palette[i % len(palette)])
    rng = np.random.default_rng(44)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                    color="0.25", alpha=0.6, s=25,
                    edgecolors="none")
    for i, cnt in enumerate(counts):
        y_top = max(vals for vals in data[i]) if data[i] else 0
        ax.text(i, y_top, f"  n={cnt}",
                 ha="left", va="bottom", fontsize=9,
                 color="0.35")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("median field error")
    ax.set_title("Angular coverage impact "
                 "(does the config span 0/45/90 deg?)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top10_bars(rows: list[dict], out_path: Path,
                     top_k: int = 10) -> None:
    finite = [r for r in rows if r["median"] is not None]
    finite.sort(key=lambda r: float(r["median"]))
    top = finite[:top_k]

    fig, (ax_bar, ax_mat) = plt.subplots(
        2, 1, figsize=(12, 7), constrained_layout=True,
        gridspec_kw=dict(height_ratios=[1.6, 1.0]))

    xs = np.arange(len(top))
    medians = [float(r["median"]) for r in top]
    p95s = [float(r["p95"]) if r["p95"] is not None else np.nan
             for r in top]
    ax_bar.bar(xs - 0.18, medians, width=0.35,
                color="#3d5a80", label="median")
    ax_bar.bar(xs + 0.18, p95s, width=0.35,
                color="#f4a261", label="p95")
    for i, (m, p) in enumerate(zip(medians, p95s)):
        ax_bar.text(i - 0.18, m, f"{m:.3f}", ha="center",
                     va="bottom", fontsize=8)
        if np.isfinite(p):
            ax_bar.text(i + 0.18, p, f"{p:.3f}", ha="center",
                         va="bottom", fontsize=8)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(
        [f"#{i + 1}\n{r['code']}\n(n={r['n']})"
         for i, r in enumerate(top)], fontsize=9)
    ax_bar.set_ylabel("field error")
    ax_bar.set_title(f"Top {top_k} configs by median field error")
    ax_bar.legend(loc="upper left")
    ax_bar.grid(axis="y", alpha=0.3)

    # sensor presence matrix (top-K rows, 6 columns)
    mat = np.zeros((len(top), len(LETTERS)))
    for i, r in enumerate(top):
        for L in r["code"]:
            if L in LETTERS:
                mat[i, LETTERS.index(L)] = 1.0
    ax_mat.imshow(mat.T, aspect="auto", cmap="Greys",
                    vmin=0, vmax=1)
    # color-tint the ON cells with the sensor color
    for i in range(len(top)):
        for j, L in enumerate(LETTERS):
            if mat[i, j] > 0:
                ax_mat.add_patch(plt.Rectangle(
                    (i - 0.5, j - 0.5), 1, 1,
                    color=SENSOR_COLORS[L], alpha=0.85, zorder=2))
                ax_mat.text(i, j, L, ha="center", va="center",
                             color="white", fontsize=10,
                             fontweight="bold", zorder=3)
    ax_mat.set_yticks(range(len(LETTERS)))
    ax_mat.set_yticklabels(LETTERS)
    ax_mat.set_xticks(range(len(top)))
    ax_mat.set_xticklabels([f"#{i + 1}" for i in range(len(top))])
    ax_mat.set_xlabel("rank")
    ax_mat.set_ylabel("sensor")
    ax_mat.set_title("Which sensors each top config uses")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top5_error_dist(rows: list[dict], out_path: Path,
                          top_k: int = 5) -> None:
    finite = [r for r in rows
              if r["median"] is not None and r["per_sim_field_errs"]]
    finite.sort(key=lambda r: float(r["median"]))
    top = finite[:top_k]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5),
                             constrained_layout=True)
    data = [np.asarray(r["per_sim_field_errs"], dtype=float)
             for r in top]
    labels = [f"#{i + 1}\n{r['code']}\nn={r['n']}"
              for i, r in enumerate(top)]
    parts = ax.violinplot(data, showmedians=True, widths=0.75)
    for pc in parts["bodies"]:
        pc.set_facecolor("#8ecae6")
        pc.set_edgecolor("#3d5a80")
        pc.set_alpha(0.75)
    ax.set_xticks(range(1, len(top) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("per-sim field error")
    ax.set_title(f"Per-sim error distribution across top-{top_k} configs")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Text report ---

def _sensor_impact_ranking(rows: list[dict]) -> list[tuple[str, float]]:
    impact = _sensor_impact(rows)
    return sorted(impact.items(), key=lambda kv: kv[1])


def build_report(rows: list[dict], out_dir: Path,
                  top_k: int = 10) -> None:
    finite = [r for r in rows if r["median"] is not None]
    finite.sort(key=lambda r: float(r["median"]))
    total = len(rows)
    n_ok = len(finite)

    by_n = defaultdict(list)
    for r in finite:
        by_n[r["n"]].append(float(r["median"]))
    best_per_n = {n: min(v) for n, v in by_n.items()}
    avg_per_n = {n: float(np.mean(v)) for n, v in by_n.items()}
    ordered_by_best = sorted(best_per_n.items(), key=lambda kv: kv[1])
    sweet_n = ordered_by_best[0][0] if ordered_by_best else None

    freq = _top_k_frequency(rows, k=top_k)
    impact_rank = _sensor_impact_ranking(rows)
    best = finite[0] if finite else None

    def _row_html(r: dict, rank: int) -> str:
        pos_str = "; ".join(
            f"({float(p[0]):.3f},{float(p[1]):.0f})" for p in r["positions"])
        p95 = r["p95"]
        p95_str = "" if p95 is None else f"{p95:.4f}"
        gtf = r["gap_to_floor"]
        gtf_str = "" if gtf is None else f"{gtf:.4f}"
        return (
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><code>{html.escape(r['tag'])}</code></td>"
            f"<td>{r['n']}</td>"
            f"<td>{html.escape(r['code'])}</td>"
            f"<td>{r['median']:.4f}</td>"
            f"<td>{p95_str}</td>"
            f"<td>{gtf_str}</td>"
            f"<td>{html.escape(pos_str)}</td>"
            "</tr>")

    parts: list[str] = []
    parts.append("<!doctype html>\n<html lang=\"en\">\n<head>\n"
                  "<meta charset=\"utf-8\">\n"
                  "<title>Sensor Sweep Analysis</title>\n"
                  "<style>\n"
                  "body{font-family:-apple-system,Segoe UI,Roboto,"
                  "Helvetica,Arial,sans-serif;max-width:1100px;"
                  "margin:2rem auto;padding:0 1rem;color:#222;"
                  "line-height:1.55;}\n"
                  "h1,h2,h3{color:#1a2b4a;}\n"
                  "h2{border-bottom:1px solid #dde;padding-bottom:.3em;"
                  "margin-top:2.5em;}\n"
                  "code{background:#f2f4f8;padding:.1em .3em;"
                  "border-radius:3px;font-size:0.92em;}\n"
                  "table{border-collapse:collapse;margin:1em 0;"
                  "font-size:.92em;}\n"
                  "th,td{border:1px solid #ccd;padding:.35em .7em;"
                  "text-align:left;}\n"
                  "th{background:#eef2f8;}\n"
                  "img{max-width:100%;height:auto;display:block;"
                  "margin:1em 0;border:1px solid #e0e4ec;"
                  "border-radius:4px;}\n"
                  ".chip{display:inline-block;padding:.1em .55em;"
                  "border-radius:12px;color:#fff;font-weight:600;"
                  "font-size:.85em;margin-right:.3em;}\n"
                  ".hilite{background:#fff5d6;padding:.15em .4em;"
                  "border-radius:3px;}\n"
                  ".subtle{color:#666;font-size:.9em;}\n"
                  "</style>\n</head>\n<body>\n")
    parts.append("<h1>Sensor Sweep Analysis</h1>\n")
    parts.append(
        f"<p>Total configs run: <b>{n_ok}</b> of {total} attempted "
        "(exhaustive C(6, n) for n in 2..6, single seed=7 each).</p>\n")

    # Executive summary
    parts.append("<h2>Executive summary</h2>\n<ul>\n")
    if best:
        parts.append(
            f"<li>Best config: <code>{html.escape(best['tag'])}</code> "
            f"(code <b>{best['code']}</b>, n={best['n']}) with "
            f"median field err <span class=\"hilite\">"
            f"{best['median']:.4f}</span></li>\n")
    if sweet_n is not None:
        parts.append(
            f"<li>Sweet-spot sensor count: <b>n={sweet_n}</b>. "
            f"Best-per-n ordering (best -> worst): "
            + " &gt; ".join(f"n={n} ({v:.4f})"
                              for n, v in ordered_by_best) + "</li>\n")
    top_freq_letters = sorted(LETTERS,
                                key=lambda L: -freq[L])
    parts.append(
        f"<li>Sensor frequency in top-{top_k} configs (helpful "
        "sensors show up often): "
        + ", ".join(f"{L}={freq[L]}" for L in top_freq_letters)
        + "</li>\n")
    parts.append(
        "<li>n-controlled impact ranking (most helpful first, "
        "delta = mean err with sensor - mean err without): "
        + ", ".join(f"{L} {v:+.4f}" for L, v in impact_rank)
        + "</li>\n")
    parts.append("</ul>\n")

    # Sensor catalog
    parts.append("<h2>Sensor catalogue</h2>\n")
    parts.append(
        "<p>Six physically-realizable positions on the wafer, "
        "two rings x three angles. Every downstream figure "
        "reuses this color/letter mapping.</p>\n")
    parts.append("<img src=\"sensor_positions.png\" "
                  "alt=\"sensor position diagram\">\n")
    parts.append("<table><thead><tr>"
                  "<th>ID</th><th>r</th><th>theta</th>"
                  "<th>Ring</th><th>Color</th></tr></thead><tbody>\n")
    for L, r, th in POSITIONS:
        ring = "inner" if L in INNER else "outer"
        parts.append(
            f"<tr><td><b>{L}</b></td><td>{r:.3f}</td>"
            f"<td>{th:g} deg</td><td>{ring}</td>"
            f"<td><span class=\"chip\" style=\"background:"
            f"{SENSOR_COLORS[L]}\">&nbsp;&nbsp;&nbsp;&nbsp;"
            "</span></td></tr>\n")
    parts.append("</tbody></table>\n")

    # Diminishing returns
    parts.append("<h2>Diminishing returns: error vs n</h2>\n")
    parts.append(
        "<p>Distribution of median field error at each sensor "
        "count. Red line traces the single best config per n. "
        "Watch where the red curve turns flat -- that is the "
        "sensor count beyond which extra sensors stop paying off.</p>\n")
    parts.append("<img src=\"diminishing_returns.png\" "
                  "alt=\"err vs n\">\n")
    parts.append("<table><thead><tr><th>n</th><th>best median</th>"
                  "<th>mean median (all C(6,n) configs)</th>"
                  "<th>count</th></tr></thead><tbody>\n")
    for n in sorted(by_n.keys()):
        parts.append(
            f"<tr><td>{n}</td><td>{best_per_n[n]:.4f}</td>"
            f"<td>{avg_per_n[n]:.4f}</td>"
            f"<td>{len(by_n[n])}</td></tr>\n")
    parts.append("</tbody></table>\n")

    # Sensor importance
    parts.append("<h2>Sensor importance</h2>\n")
    parts.append(
        "<p>Left: how often each sensor shows up in the top-"
        f"{top_k}. Right: mean error delta when the sensor is "
        "included, computed WITHIN each n and averaged across n so "
        "the confound that \"large-n configs contain every "
        "letter\" is controlled. Negative = including this "
        "sensor tends to reduce error.</p>\n")
    parts.append("<img src=\"sensor_importance.png\" "
                  "alt=\"sensor importance\">\n")

    # Ring composition
    parts.append("<h2>Ring composition</h2>\n")
    parts.append(
        "<p>Configs split by whether they use only inner "
        "sensors (ABC), only outer (DEF), or mix rings. Any "
        "config with n=1 stays in whichever ring it belongs to; "
        "n>=2 pure-inner or pure-outer configs are C(3, k)-limited.</p>\n")
    parts.append("<img src=\"ring_composition.png\" "
                  "alt=\"ring composition\">\n")

    # Angular coverage
    parts.append("<h2>Angular coverage</h2>\n")
    parts.append(
        "<p>Distinct from ring composition -- here we bucket by "
        "how many of the three angles (0, 45, 90 deg) the config "
        "sees at all, regardless of ring. Missing an angle is a "
        "candidate explanation for large residuals along that ray.</p>\n")
    parts.append("<img src=\"angular_coverage.png\" "
                  "alt=\"angular coverage\">\n")

    # Top-10 comparison
    parts.append(f"<h2>Top {top_k} configs</h2>\n")
    parts.append(
        "<p>Median + p95 field error for the top-ranked configs, "
        "plus a matrix showing which sensors each one uses.</p>\n")
    parts.append("<img src=\"top10_bars.png\" "
                  "alt=\"top-K comparison\">\n")
    parts.append("<table><thead><tr>"
                  "<th>#</th><th>Tag</th><th>n</th><th>Code</th>"
                  "<th>Median</th><th>P95</th>"
                  "<th>Gap to floor</th><th>Positions</th>"
                  "</tr></thead><tbody>\n")
    for i, r in enumerate(finite[:top_k]):
        parts.append(_row_html(r, i + 1))
    parts.append("</tbody></table>\n")

    # Per-sim distribution (top-5)
    parts.append("<h2>Per-sim error distribution (top 5)</h2>\n")
    parts.append(
        "<p>Even at similar medians, configs can differ in tail "
        "behaviour (long-tail failure modes vs uniformly OK). "
        "This violin plot shows the full per-test-sim error "
        "distribution for the top 5 configs.</p>\n")
    parts.append("<img src=\"top5_error_dist.png\" "
                  "alt=\"top-5 distributions\">\n")

    # Full ranking
    parts.append("<h2>Full ranking</h2>\n")
    parts.append(
        f"<p>All {n_ok} completed configs ranked by median field "
        "error. Sortable as CSV in "
        "<code>viz/sweep_summary.csv</code> if produced by "
        "<code>summarize_sweep.py</code>.</p>\n")
    parts.append("<table><thead><tr>"
                  "<th>#</th><th>Tag</th><th>n</th><th>Code</th>"
                  "<th>Median</th><th>P95</th>"
                  "<th>Gap to floor</th><th>Positions</th>"
                  "</tr></thead><tbody>\n")
    for i, r in enumerate(finite):
        parts.append(_row_html(r, i + 1))
    parts.append("</tbody></table>\n")

    parts.append("</body>\n</html>\n")

    (out_dir / "report.html").write_text("".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", default="outputs",
                    help="root dir containing sweep_*/results.json "
                    "(default: outputs)")
    ap.add_argument("--prefix", default="sweep_",
                    help="tag prefix (default: sweep_)")
    ap.add_argument("--out-dir", default="viz/sweep_summary",
                    help="where to write the report + figures "
                    "(default: viz/sweep_summary)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="how many configs to detail in top-K "
                    "sections (default: 10)")
    args = ap.parse_args()

    outputs_root = Path(args.outputs)
    if not outputs_root.is_dir():
        print(f"outputs dir not found: {outputs_root}",
               file=sys.stderr)
        return 2

    rows = load_rows(outputs_root, args.prefix)
    if not rows:
        print(f"no {args.prefix}* results found in {outputs_root}",
               file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loaded {len(rows)} configs")
    print(f"writing figures + report to {out_dir}")

    plot_sensor_positions(out_dir / "sensor_positions.png")
    plot_diminishing_returns(rows, out_dir / "diminishing_returns.png")
    plot_sensor_importance(rows, out_dir / "sensor_importance.png",
                             top_k=args.top_k)
    plot_ring_composition(rows, out_dir / "ring_composition.png")
    plot_angular_coverage(rows, out_dir / "angular_coverage.png")
    plot_top10_bars(rows, out_dir / "top10_bars.png",
                     top_k=args.top_k)
    plot_top5_error_dist(rows, out_dir / "top5_error_dist.png",
                          top_k=5)
    build_report(rows, out_dir, top_k=args.top_k)

    print(f"done. Open {out_dir / 'report.html'} in a browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
