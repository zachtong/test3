"""Copy sweep result folders to a new location with fine-grained
filtering (file type, pick, sims-per-pick).

Mirrors each source directory into <dest>/<source-basename>/,
preserving internal structure. Filters, applied per file:

  --include        keep ONLY files matching these globs (e.g.
                   '*.gif' to export just the animations). Empty =
                   keep all types.
  --exclude        drop files matching these globs (default
                   '*.html' -- the heavy plotly dumps). Applied
                   after --include.
  --picks          keep only files under all_picks/<pick>/ for the
                   listed picks (e.g. 'worst,best'). Files not
                   under a pick dir always pass.
  --sims-per-pick  within each pick dir, keep only the first N sims
                   by slot order (filename starts with the 4-digit
                   slot; 0000 is the top-ranked sim for that pick).

Reports files + bytes copied vs skipped so the space saved is
visible. --dry-run previews without writing.

Sources named two ways:

  1. Explicit directories:
       python scripts/export_results.py \\
           --sources viz/old_sweep viz/smalltest_sweep_k12 \\
           --dest ~/wafer_export

  2. By prefix under a root (globs <root>/<prefix>*):
       python scripts/export_results.py \\
           --root viz --prefixes old_sweep smalltest_sweep_k12 \\
           --dest ~/wafer_export

Example -- export only the worst+best GIFs, 3 sims each:
    python scripts/export_results.py \\
        --root viz --prefixes smalltest_sweep_k12 \\
        --dest ~/wafer_export \\
        --include '*.gif' --picks worst,best --sims-per-pick 3
"""
from __future__ import annotations
import argparse
import fnmatch
import re
import shutil
import sys
from pathlib import Path

_SLOT_RE = re.compile(r"^(\d+)_")


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _pick_of(rel: Path) -> str | None:
    """The pick subdir name if rel is a file under
    .../all_picks/<pick>/..., else None."""
    parts = rel.parts
    for i, seg in enumerate(parts):
        # need a segment after <pick> too (the file itself), so the
        # pick index i+1 must be strictly before the last component.
        if seg == "all_picks" and i + 1 < len(parts) - 1:
            return parts[i + 1]
    return None


def _slot_of(name: str) -> int | None:
    m = _SLOT_RE.match(name)
    return int(m.group(1)) if m else None


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"


def _resolve_sources(args) -> list[Path]:
    sources: list[Path] = []
    if args.sources:
        for s in args.sources:
            p = Path(s).expanduser()
            if not p.is_dir():
                print(f"WARN: not a directory, skipping: {p}",
                      file=sys.stderr)
                continue
            sources.append(p)
    if args.prefixes:
        root = Path(args.root).expanduser()
        if not root.is_dir():
            raise SystemExit(f"--root not a directory: {root}")
        for prefix in args.prefixes:
            matched = sorted(d for d in root.glob(f"{prefix}*")
                             if d.is_dir())
            if not matched:
                print(f"WARN: no dirs under {root} matching "
                      f"'{prefix}*'", file=sys.stderr)
            sources.extend(matched)
    # De-dupe while preserving order (resolve to canonical path)
    seen = set()
    uniq = []
    for p in sources:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _keep_file(rel: Path, name: str, includes: list[str],
               excludes: list[str], picks: list[str] | None
               ) -> bool:
    """Name-level + pick-level keep decision (sims-per-pick is
    handled separately since it needs a per-pick pass)."""
    if includes and not _matches_any(name, includes):
        return False
    if excludes and _matches_any(name, excludes):
        return False
    if picks is not None:
        p = _pick_of(rel)
        # Files under a pick dir must be in the allowed set. Files
        # not under any pick dir (top-level summaries etc.) pass.
        if p is not None and p not in picks:
            return False
    return True


def _copy_tree(src: Path, dst: Path, includes: list[str],
               excludes: list[str], picks: list[str] | None,
               sims_per_pick: int | None, dry_run: bool) -> dict:
    """Two-phase: (1) decide which files survive name/pick/type
    filters, (2) within each pick apply the sims-per-pick cap by
    keeping the smallest-slot sims, (3) copy survivors. Everything
    filtered out is counted as skipped."""
    stats = dict(copied=0, skipped=0,
                 copied_bytes=0, skipped_bytes=0)

    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    survivors: list[tuple[Path, Path]] = []   # (item, rel)
    # (pick -> sorted set of slots seen among name/pick-passing files)
    pick_slots: dict[str, set[int]] = {}
    passing: list[tuple[Path, Path, str | None, int | None]] = []

    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        if not _keep_file(rel, item.name, includes, excludes, picks):
            stats["skipped"] += 1
            stats["skipped_bytes"] += _size(item)
            continue
        pick = _pick_of(rel)
        slot = _slot_of(item.name) if pick is not None else None
        passing.append((item, rel, pick, slot))
        if pick is not None and slot is not None:
            pick_slots.setdefault(pick, set()).add(slot)

    # Compute the allowed slot set per pick for the N-sims cap.
    allowed_slots: dict[str, set[int]] = {}
    if sims_per_pick is not None:
        for pick, slots in pick_slots.items():
            allowed_slots[pick] = set(
                sorted(slots)[:sims_per_pick])

    for item, rel, pick, slot in passing:
        if (sims_per_pick is not None and pick is not None
                and slot is not None):
            if slot not in allowed_slots.get(pick, set()):
                stats["skipped"] += 1
                stats["skipped_bytes"] += _size(item)
                continue
        survivors.append((item, rel))

    for item, rel in survivors:
        target = dst / rel
        size = _size(item)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(item, target)
            except OSError as e:
                print(f"  ERROR copying {rel}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                continue
        stats["copied"] += 1
        stats["copied_bytes"] += size
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", nargs="*",
                    help="explicit source directories to copy")
    ap.add_argument("--root", default="viz",
                    help="root dir for --prefixes globbing "
                    "(default: viz)")
    ap.add_argument("--prefixes", nargs="*",
                    help="directory-name prefixes under --root to "
                    "copy (globs <root>/<prefix>*)")
    ap.add_argument("--dest", required=True,
                    help="destination root; each source is mirrored "
                    "into <dest>/<source-basename>/")
    ap.add_argument("--exclude", default="*.html",
                    help="comma-separated glob patterns of files to "
                    "SKIP (default: *.html)")
    ap.add_argument("--include", default="",
                    help="comma-separated glob patterns; if set, "
                    "keep ONLY files matching one of them (e.g. "
                    "'*.gif'). Empty keeps all types.")
    ap.add_argument("--picks", default="",
                    help="comma list of pick names to keep (e.g. "
                    "'worst,best'). Files under all_picks/<pick>/ "
                    "for other picks are skipped. Empty keeps all "
                    "picks.")
    ap.add_argument("--sims-per-pick", type=int, default=None,
                    help="within each pick dir, keep only the first "
                    "N sims by slot order (0000 = top-ranked). "
                    "Default: keep all.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be copied without "
                    "writing anything")
    args = ap.parse_args()

    if not args.sources and not args.prefixes:
        print("provide --sources and/or --prefixes",
              file=sys.stderr)
        return 2

    excludes = [p.strip() for p in args.exclude.split(",")
                if p.strip()]
    includes = [p.strip() for p in args.include.split(",")
                if p.strip()]
    picks = [p.strip() for p in args.picks.split(",")
             if p.strip()] or None
    sources = _resolve_sources(args)
    if not sources:
        print("no source directories resolved", file=sys.stderr)
        return 1

    dest_root = Path(args.dest).expanduser()
    mode = "DRY-RUN (nothing written)" if args.dry_run else "COPY"
    print(f"[{mode}] {len(sources)} source dir(s) -> {dest_root}")
    if includes:
        print(f"  including only: {includes}")
    print(f"  excluding: {excludes}")
    if picks is not None:
        print(f"  picks: {picks}")
    if args.sims_per_pick is not None:
        print(f"  sims-per-pick: {args.sims_per_pick}")
    print()

    total = dict(copied=0, skipped=0,
                 copied_bytes=0, skipped_bytes=0)
    for src in sources:
        dst = dest_root / src.name
        s = _copy_tree(src, dst, includes, excludes, picks,
                       args.sims_per_pick, args.dry_run)
        for k in total:
            total[k] += s[k]
        print(f"  {src.name}:")
        print(f"    copied  {s['copied']:>6d} files  "
              f"{_human(s['copied_bytes']):>10}")
        print(f"    skipped {s['skipped']:>6d} files  "
              f"{_human(s['skipped_bytes']):>10}  (excluded)")

    print(f"\nTOTAL:")
    print(f"  copied  {total['copied']:>6d} files  "
          f"{_human(total['copied_bytes']):>10}")
    print(f"  skipped {total['skipped']:>6d} files  "
          f"{_human(total['skipped_bytes']):>10}  "
          f"(space saved by excluding)")
    if args.dry_run:
        print(f"\nRe-run without --dry-run to actually copy.")
    else:
        print(f"\nDone. Exported to {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
