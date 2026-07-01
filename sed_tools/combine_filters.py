#!/usr/bin/env python3
"""
Combine multiple photometric filter sets into one MESA-compatible set.

Unlike stellar atmosphere combination, filter combination does not need a flux
cube. A filter set is just a directory of ``*.dat`` transmission curves plus an
index file whose contents list those curve filenames.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .models import FILTER_DIR_DEFAULT

PathLike = Union[str, os.PathLike[str]]


def find_filter_sets(base_dir: Optional[PathLike] = None) -> List[Path]:
    """Find local filter-set directories containing ``*.dat`` files."""
    root = Path(base_dir) if base_dir is not None else FILTER_DIR_DEFAULT
    if not root.exists():
        return []

    filter_sets: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if any(child.is_file() and child.suffix.lower() == ".dat" for child in path.iterdir()):
            filter_sets.append(path)
    return filter_sets


def resolve_output_path(
    output: PathLike,
    base_dir: PathLike,
    *,
    facility: Optional[str] = None,
    instrument: Optional[str] = None,
) -> Path:
    """Resolve a friendly output name into a filter-set directory path."""
    output_path = Path(output)
    root = Path(base_dir)

    if output_path.is_absolute():
        return output_path
    if facility or instrument:
        return root / (facility or "Combined") / (instrument or output_path.name)
    if len(output_path.parts) == 1:
        return root / "Combined" / output_path.name
    return root / output_path


def resolve_filter_sources(source: PathLike, base_dir: PathLike) -> List[Path]:
    """Resolve one input into concrete ``*.dat`` files."""
    source_path = Path(source)
    root = Path(base_dir)
    candidates = [source_path] if source_path.is_absolute() else [source_path, root / source_path]

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".dat":
            return [candidate]
        if candidate.is_dir():
            files = sorted(
                child for child in candidate.iterdir()
                if child.is_file() and child.suffix.lower() == ".dat"
            )
            if files:
                return files

            child_dirs = sorted(child for child in candidate.iterdir() if child.is_dir())
            if len(child_dirs) == 1:
                nested = resolve_filter_sources(child_dirs[0], root)
                if nested:
                    return nested

    raise FileNotFoundError(f"Could not find .dat filters for input: {source}")


def unique_filter_target(output_dir: Path, source_file: Path) -> Path:
    """Choose a non-conflicting output name for a duplicate filter file."""
    parent_names = [parent.name for parent in source_file.parents[:2] if parent.name]
    prefix = "_".join(reversed(parent_names))
    stem = f"{prefix}_{source_file.stem}" if prefix else source_file.stem
    target = output_dir / f"{stem}{source_file.suffix}"

    counter = 2
    while target.exists():
        target = output_dir / f"{stem}_{counter}{source_file.suffix}"
        counter += 1
    return target


def write_filter_index(filter_dir: Path, instrument: str) -> Path:
    """Write the MESA index file for all ``*.dat`` files in ``filter_dir``."""
    dat_files = sorted(child.name for child in filter_dir.glob("*.dat") if child.is_file())
    index_path = filter_dir / instrument
    index_path.write_text("\n".join(dat_files) + "\n", encoding="utf-8")
    return index_path


def combine_filter_sets(
    output: PathLike,
    inputs: Sequence[PathLike],
    *,
    filter_root: Optional[PathLike] = None,
    facility: Optional[str] = None,
    instrument: Optional[str] = None,
    on_conflict: str = "rename",
) -> Path:
    """Combine filter files or filter-set directories into one filter set."""
    if not inputs:
        raise ValueError("Provide at least one filter file or filter-set directory to combine")

    mode = on_conflict.lower()
    if mode not in {"rename", "overwrite", "error"}:
        raise ValueError("on_conflict must be one of: 'rename', 'overwrite', 'error'")

    root = Path(filter_root) if filter_root is not None else FILTER_DIR_DEFAULT
    output_dir = resolve_output_path(output, root, facility=facility, instrument=instrument)
    output_instrument = instrument or output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for source in inputs:
        for source_file in resolve_filter_sources(source, root):
            target = output_dir / source_file.name
            if target.exists() and target.resolve() != source_file.resolve():
                if mode == "error":
                    raise FileExistsError(f"Filter name conflict while combining: {source_file.name}")
                if mode == "rename":
                    target = unique_filter_target(output_dir, source_file)

            if target.resolve() != source_file.resolve():
                shutil.copy2(source_file, target)
            copied.append(target.name)

    if not copied:
        raise ValueError("No .dat filter files were found in the supplied inputs")

    write_filter_index(output_dir, output_instrument)
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine filter sets into one MESA-compatible instrument"
    )
    parser.add_argument("output", help="Output name or Facility/Instrument path")
    parser.add_argument("inputs", nargs="+", help="Filter-set directories or .dat files")
    parser.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")
    parser.add_argument("--facility", default=None, help="Output facility label")
    parser.add_argument("--instrument", default=None, help="Output instrument/index-file name")
    parser.add_argument(
        "--on-conflict",
        choices=["rename", "overwrite", "error"],
        default="rename",
        help="How duplicate filter filenames are handled",
    )

    args = parser.parse_args(argv)
    output_dir = combine_filter_sets(
        args.output,
        args.inputs,
        filter_root=args.base,
        facility=args.facility,
        instrument=args.instrument,
        on_conflict=args.on_conflict,
    )
    print(f"[filters] Combined filter set written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
