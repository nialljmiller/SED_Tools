#!/usr/bin/env python3
"""
Convert mirror filter curves written by SED_Tools 0.3.0 to the column-row layout

SED_Tools 0.3.0 wrote filter metadata as leading "# Key: value" lines, which
released MESA colors counts as samples. This moves those fields onto the
Wavelength,Transmission row, keeping every sample row byte for byte, and lists
curves released MESA would still misread (other comment or blank lines).
Offline and idempotent; regenerate index.json afterwards.

Usage: python3 migrate_filters.py [ROOT ...] [--dry-run]
       (default roots: data/filters data/atmospheres next to this script)
"""

import argparse
import sys
from pathlib import Path

from sed_tools.filter_metadata import has_column_row, iter_filter_files, migrate_filter_tree, released_mesa_reads


def migrate(roots, dry_run=False):
    """Return (converted, without a column row, still misread by released MESA)."""
    moved = [path for root in roots for path in migrate_filter_tree(root, dry_run=dry_run)]
    bare = [path for path in moved if not has_column_row(path)]
    misread = [path for root in roots for path in iter_filter_files(root)
               if not released_mesa_reads(path, after_migration=dry_run)]
    return moved, bare, misread


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("roots", nargs="*", type=Path,
                        default=[here / "data" / "filters", here / "data" / "atmospheres"])
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    args = parser.parse_args(argv)

    missing = [root for root in args.roots if not root.exists()]
    if missing:
        parser.error(f"not found: {', '.join(map(str, missing))}")

    moved, bare, misread = migrate(args.roots, dry_run=args.dry_run)
    verb = "would move" if args.dry_run else "moved"
    print(f"{len(moved)} curves: {verb} 0.3.0 '#' metadata to the column row")
    if bare:
        dropped = "would be removed" if args.dry_run else "were removed"
        print(f"{len(bare)} have no column-name row: their '#' lines {dropped} without keeping the metadata")
    if misread:
        print(f"{len(misread)} curves keep comment or blank lines that released MESA reads as junk samples; "
              "remove those lines by hand:")
        for path in misread:
            print(f"  {path}")
    return 1 if misread else 0


if __name__ == "__main__":
    sys.exit(main())
