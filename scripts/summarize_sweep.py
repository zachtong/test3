"""Aggregate sensor-sweep results into a ranked CSV + markdown report.

Walks every outputs/sweep_*/results.json, extracts the fields that
matter for comparing configs (median field err, p95, gap-to-floor,
sensor count + positions), sorts by median field err (best first),
and writes:
    viz/sweep_summary.csv    -- full table
    viz/sweep_summary.md     -- top-10 table + summary stats

Sensor "code" (e.g. ADE) refers to the fixed six-point catalogue:
    A=(0.520, 0),   B=(0.520, 45),  C=(0.520, 90)
    D=(0.847, 0),   E=(0.847, 45),  F=(0.847, 90)

Run:
    python scripts/summarize_sweep.py
    # or point at a different outputs dir / tag prefix:
    python scripts/summarize_sweep.py --outputs outputs --prefix sweep_
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


_POSITION_ID = {
    (0.52,  0.0):  "A",
    (0.52,  45.0): "B",
    (0.52,  90.0): "C",
    (0.847, 0.0):  "D",
    (0.847, 45.0): "E",
    (0.847, 90.0): "F",
}


def _code_from_positions(positions) -> str:
    """Recover the ABCDEF short-code from a list of (r, theta) pairs."""
    letters = []
    for p in positions:
        r = float(p[0])
        th = float(p[1])
        key = (round(r, 3), round(th, 3))
        # Accept exact hits on either raw or rounded values.
        for known_key, letter in _POSITION_ID.items():
            if (abs(r - known_key[0]) < 1e-3
                    and abs(th - known_key[1]) < 1e-3):
                letters.append(letter)
                break
        else:
            letters.append("?")
    return "".join(sorted(letters))


def _extract_row(tag_dir: Path) -> dict | None:
    res = tag_dir / "results.json"
    if not res.is_file():
        return None
    try:
        d = json.loads(res.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"skip {tag_dir.name}: {type(e).__name__}: {e}")
        return None
    cfg = d.get("config", {}) or {}
    sens_cfg = cfg.get("sensors", {}) or {}
    positions = sens_cfg.get("positions") or []
    gs = d.get("global_stats", {}) or {}
    return {
        "tag": tag_dir.name,
        "n_sensors": int(sens_cfg.get("n") or len(positions)),
        "code": _code_from_positions(positions),
        "positions": positions,
        "median_field_err": gs.get("median"),
        "p95_field_err": gs.get("p95"),
        "gap_to_floor": d.get("gap_to_floor"),
        "n_params": d.get("n_params"),
    }


def _sort_key(row: dict):
    v = row.get("median_field_err")
    return float("inf") if v is None else float(v)


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    field_names = ["rank", "tag", "n_sensors", "code",
                   "median_field_err", "p95_field_err",
                   "gap_to_floor", "n_params", "positions"]
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=field_names)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({
                "rank": i + 1,
                "tag": r["tag"],
                "n_sensors": r["n_sensors"],
                "code": r["code"],
                "median_field_err": r["median_field_err"],
                "p95_field_err": r["p95_field_err"],
                "gap_to_floor": r["gap_to_floor"],
                "n_params": r["n_params"],
                "positions": json.dumps(r["positions"]),
            })


def _fmt_num(v, fmt: str = "{:.4f}") -> str:
    if v is None:
        return "--"
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "--"


def _write_markdown(rows: list[dict], out_md: Path,
                     top_k: int = 10) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    lines = [
        "# Sensor sweep results\n",
        f"Total configs: {total}\n",
        "\n## Sensor catalogue\n\n",
        "| ID | r | theta |\n",
        "|---|---|---|\n",
        "| A | 0.520 | 0 deg |\n",
        "| B | 0.520 | 45 deg |\n",
        "| C | 0.520 | 90 deg |\n",
        "| D | 0.847 | 0 deg |\n",
        "| E | 0.847 | 45 deg |\n",
        "| F | 0.847 | 90 deg |\n",
        "\n",
        f"## Top {top_k} by median field error\n\n",
        "| Rank | Tag | n | Code | median | p95 | "
        "gap_to_floor |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for i, r in enumerate(rows[:top_k]):
        lines.append(
            f"| {i + 1} | {r['tag']} | {r['n_sensors']} | "
            f"{r['code']} | {_fmt_num(r['median_field_err'])} | "
            f"{_fmt_num(r['p95_field_err'])} | "
            f"{_fmt_num(r['gap_to_floor'])} |\n")

    # Best-per-n breakdown so you can see how much extra sensors
    # buy you in each subset size.
    by_n: dict[int, dict] = {}
    for r in rows:
        n = r["n_sensors"]
        if n not in by_n or _sort_key(r) < _sort_key(by_n[n]):
            by_n[n] = r
    lines.append("\n## Best-per-n\n\n")
    lines.append("| n | Best code | median | p95 | gap_to_floor |\n")
    lines.append("|---|---|---|---|---|\n")
    for n in sorted(by_n.keys()):
        r = by_n[n]
        lines.append(
            f"| {n} | {r['code']} | {_fmt_num(r['median_field_err'])} "
            f"| {_fmt_num(r['p95_field_err'])} | "
            f"{_fmt_num(r['gap_to_floor'])} |\n")
    out_md.write_text("".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", default="outputs",
                    help="root dir holding sweep_*/results.json "
                    "(default: outputs)")
    ap.add_argument("--prefix", default="sweep_",
                    help="tag prefix to match (default: sweep_)")
    ap.add_argument("--out-dir", default="viz",
                    help="where to write summary.csv / summary.md "
                    "(default: viz)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="how many top rows to show in the markdown "
                    "table (default: 10)")
    args = ap.parse_args()

    outputs_root = Path(args.outputs)
    if not outputs_root.is_dir():
        print(f"outputs dir not found: {outputs_root}", file=sys.stderr)
        return 2

    tag_dirs = sorted(outputs_root.glob(f"{args.prefix}*"))
    if not tag_dirs:
        print(f"no {args.prefix}* directories in {outputs_root}",
              file=sys.stderr)
        return 1

    rows: list[dict] = []
    for td in tag_dirs:
        row = _extract_row(td)
        if row is not None:
            rows.append(row)
    if not rows:
        print("no readable results.json found", file=sys.stderr)
        return 1
    rows.sort(key=_sort_key)

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "sweep_summary.csv"
    md_path = out_dir / "sweep_summary.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path, top_k=args.top_k)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {md_path}")
    if rows:
        best = rows[0]
        print(f"best config: {best['tag']}  "
              f"median={_fmt_num(best['median_field_err'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
