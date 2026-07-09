"""Merge several NPZ datasets into one directory via prefixed symlinks.

The loader identifies each sim by its filename (Path.name) and globs
*.npz in a single directory, following symlinks. To sweep over the
UNION of two datasets you therefore just need one directory holding
links to every sim -- no multi-GB physical copy.

Each link is named '<prefix>__<original>.npz' where <prefix> is
derived from the source directory (or given explicitly). This:
  - guarantees uniqueness even if two datasets share a basename
    (the loader would otherwise collide on identity), and
  - records provenance in the name so you can later filter or
    trace a sim back to its source dataset.

Links use ABSOLUTE targets so they resolve regardless of cwd or
where the merged dir lives. Source cache files (names starting
with '_', e.g. _loader_cache_*.npz) are skipped. Re-running is
idempotent: existing correct links are left alone.

    python scripts/merge_datasets_symlink.py \\
        --sources /data/firehorse_1_and_2 /data/smalltest_set \\
        --dest /data/merged_firehorse_plus_small

    # explicit prefixes (else derived from dir basename):
    python scripts/merge_datasets_symlink.py \\
        --sources /data/A /data/B --prefixes fh small \\
        --dest /data/merged --dry-run
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def _clean_prefix(s: str) -> str:
    """Filesystem-safe short prefix; no '__' (our separator) inside."""
    keep = [c if (c.isalnum() or c in "-.") else "_" for c in s]
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "src"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", nargs="+", required=True,
                    help="source NPZ directories to merge")
    ap.add_argument("--dest", required=True,
                    help="destination directory for the symlinks "
                    "(created if missing)")
    ap.add_argument("--prefixes", nargs="*",
                    help="one prefix per source (same order). If "
                    "omitted, each prefix is derived from the "
                    "source directory's basename.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be linked without "
                    "creating anything")
    args = ap.parse_args()

    sources = [Path(s).expanduser().resolve() for s in args.sources]
    for s in sources:
        if not s.is_dir():
            print(f"ERROR: not a directory: {s}", file=sys.stderr)
            return 2
    if args.prefixes:
        if len(args.prefixes) != len(sources):
            print(f"ERROR: {len(args.prefixes)} prefixes for "
                  f"{len(sources)} sources", file=sys.stderr)
            return 2
        prefixes = [_clean_prefix(p) for p in args.prefixes]
    else:
        prefixes = [_clean_prefix(s.name) for s in sources]
    if len(set(prefixes)) != len(prefixes):
        print(f"ERROR: prefixes not unique: {prefixes}",
              file=sys.stderr)
        return 2

    dest = Path(args.dest).expanduser().resolve()
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    mode = "DRY-RUN" if args.dry_run else "LINK"
    print(f"[{mode}] merging {len(sources)} source(s) -> {dest}")
    link_names: dict[str, Path] = {}     # link name -> source file
    total_linked = total_skipped = total_existing = 0

    for src, prefix in zip(sources, prefixes):
        npz = sorted(p for p in src.glob("*.npz")
                     if not p.name.startswith("_"))
        made = existing = 0
        for p in npz:
            link_name = f"{prefix}__{p.name}"
            if link_name in link_names:
                print(f"  WARN: duplicate link name {link_name} "
                      f"(from {p} and {link_names[link_name]}); "
                      f"skipping the second", file=sys.stderr)
                continue
            link_names[link_name] = p
            link_path = dest / link_name
            target = p.resolve()
            if link_path.is_symlink() or link_path.exists():
                # Already present -- verify it points where we want.
                try:
                    cur = link_path.resolve()
                except OSError:
                    cur = None
                if cur == target:
                    existing += 1
                    continue
                print(f"  WARN: {link_name} exists but points to "
                      f"{cur}, not {target}; leaving as-is",
                      file=sys.stderr)
                continue
            if not args.dry_run:
                os.symlink(target, link_path)
            made += 1
        print(f"  {prefix}: {len(npz)} npz in source  ->  "
              f"{made} new link(s), {existing} already present")
        total_linked += made
        total_existing += existing

    print(f"\nTOTAL: {total_linked} new link(s), "
          f"{total_existing} already present, "
          f"{len(link_names)} sims in merged set")
    if args.dry_run:
        print("Re-run without --dry-run to create the links.")
    else:
        print(f"Done. Point the sweep at: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
