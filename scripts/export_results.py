"""Copy sweep result folders to a new location, excluding heavy
files (interactive_compare plotly HTML by default).

Mirrors each source directory into <dest>/<source-basename>/,
preserving the internal structure, skipping any file whose name
matches an --exclude glob. Reports how many files and how many
bytes were copied vs skipped, so you can see the space saved.

Two ways to name the sources:

  1. Explicit directories:
       python scripts/export_results.py \\
           --sources viz/old_sweep viz/smalltest_sweep \\
                     viz/smalltest_sweep_k12 \\
           --dest ~/wafer_export

  2. By prefix under a root (globs <root>/<prefix>*):
       python scripts/export_results.py \\
           --root viz \\
           --prefixes old_sweep smalltest_sweep smalltest_sweep_k12 \\
           --dest ~/wafer_export

--dry-run reports what WOULD be copied/skipped without writing.

Default exclude is '*.html'. Add more with e.g.
    --exclude '*.html,*.mp4'
"""
from __future__ import annotations
import argparse
import fnmatch
import shutil
import sys
from pathlib import Path


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


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


def _copy_tree(src: Path, dst: Path, excludes: list[str],
               dry_run: bool) -> dict:
    """Copy src -> dst recursively, skipping excluded files.
    Returns counts + byte totals."""
    stats = dict(copied=0, skipped=0,
                 copied_bytes=0, skipped_bytes=0)
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        if _matches_any(item.name, excludes):
            try:
                stats["skipped_bytes"] += item.stat().st_size
            except OSError:
                pass
            stats["skipped"] += 1
            continue
        target = dst / rel
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
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
    sources = _resolve_sources(args)
    if not sources:
        print("no source directories resolved", file=sys.stderr)
        return 1

    dest_root = Path(args.dest).expanduser()
    mode = "DRY-RUN (nothing written)" if args.dry_run else "COPY"
    print(f"[{mode}] {len(sources)} source dir(s) -> {dest_root}")
    print(f"  excluding: {excludes}\n")

    total = dict(copied=0, skipped=0,
                 copied_bytes=0, skipped_bytes=0)
    for src in sources:
        dst = dest_root / src.name
        s = _copy_tree(src, dst, excludes, args.dry_run)
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
