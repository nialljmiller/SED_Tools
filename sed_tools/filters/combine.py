"""Combine filter sets into one MESA-compatible set: a directory of ``*.dat`` curves and the index file listing them."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .manifest import COMPONENTS_DIR, archived_bands, index_listing, is_auxiliary_path, write_filter_index
from .curve_metadata import _read_curve, copy_filter_file, migrate_filter_file
from ..cube.model import FILTER_DIR_DEFAULT

PathLike = Union[str, os.PathLike[str]]

logger = logging.getLogger(__name__)


def find_filter_sets(base_dir: Optional[PathLike] = None) -> List[Path]:
    """Find local filter-set directories containing ``*.dat`` files."""
    root = Path(base_dir) if base_dir is not None else FILTER_DIR_DEFAULT
    return [path for path in sorted(root.rglob("*"))
            if path.is_dir() and not is_auxiliary_path(path, root, dirs=(COMPONENTS_DIR,))
            and any(child.is_file() and child.suffix.lower() == ".dat" for child in path.iterdir())]


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
            files = _set_curves(candidate)
            if files:
                return files

            child_dirs = sorted(child for child in candidate.iterdir() if child.is_dir())
            if len(child_dirs) == 1:
                nested = resolve_filter_sources(child_dirs[0], root)
                if nested:
                    return nested

    raise FileNotFoundError(f"Could not find .dat filters for input: {source}")


def _set_curves(directory: Path) -> List[Path]:
    """The curves its MESA index file lists (every ``*.dat`` without one), less its ``archived_bands``.

    Other curve files beside the set, such as NJM's ``_svo`` and ``_old`` copies, are left out with a warning.
    """
    present = sorted(child for child in directory.iterdir()
                     if child.is_file() and child.suffix.lower() == ".dat")
    archived = archived_bands(directory)
    current = [path for path in present if path.stem not in archived]
    listed = index_listing(directory)
    if listed is None:
        return current
    missing = [name for name in listed if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory / directory.name} lists {', '.join(missing)}, not found in {directory}; "
            "restore the files or correct the index file")
    unlisted = [path.name for path in current if path.name not in listed]
    if unlisted:
        logger.warning("%s: leaving out %s, which its index file %s does not list",
                       directory, ", ".join(unlisted), directory.name)
    return [path for path in current if path.name in listed]


def unique_filter_target(output_dir: Path, source_file: Path, written: Sequence[str] = ()) -> Path:
    """The output name of a curve whose own name is taken: ``<facility>_<instrument>_<band>``, then ``..._2``, ...,
    the first no other curve holds (:func:`_taken`)."""
    parent_names = [parent.name for parent in source_file.parents[:2] if parent.name]
    prefix = "_".join(reversed(parent_names))
    stem = f"{prefix}_{source_file.stem}" if prefix else source_file.stem
    # MESA names a band after the text before the first dot of its file name
    # (Cameras/Sony.A200 would give "Cameras_Sony"), so the name made here has
    # none: "." becomes "p", as in component_sources._original.safe_band.
    stem = stem.replace(".", "p")
    target = output_dir / f"{stem}{source_file.suffix}"

    counter = 2
    while _taken(target, source_file, written):
        target = output_dir / f"{stem}_{counter}{source_file.suffix}"
        counter += 1
    return target


def _taken(target: Path, source_file: Path, written: Sequence[str]) -> bool:
    """Whether ``target`` holds another curve than ``source_file``: one this run wrote (``written``), or other
    samples.  A file an earlier run left with the same samples is ``source_file``'s own copy, used again."""
    if not target.exists() or target.resolve() == source_file.resolve():
        return False
    return target.name in written or not all(map(np.array_equal, _read_curve(target), _read_curve(source_file)))


def combine_filter_sets(
    output: PathLike,
    inputs: Sequence[PathLike],
    *,
    filter_root: Optional[PathLike] = None,
    facility: Optional[str] = None,
    instrument: Optional[str] = None,
    on_conflict: str = "rename",
) -> Path:
    """Combine filter files or filter-set directories into one filter set.

    A directory gives the curves its MESA index file lists (every ``*.dat``
    without one), less its manifest's ``archived_bands``.  A curve whose name
    another curve holds is renamed, overwrites it or stops (``on_conflict``);
    run again, each curve finds its own earlier copy and adds nothing.
    """
    mode = on_conflict.lower()
    root = Path(filter_root) if filter_root is not None else FILTER_DIR_DEFAULT
    output_dir = resolve_output_path(output, root, facility=facility, instrument=instrument)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for source in inputs:
        for source_file in resolve_filter_sources(source, root):
            target = output_dir / source_file.name
            if _taken(target, source_file, copied):
                if mode not in ("rename", "overwrite"):  # "error", or a mode that is neither
                    raise FileExistsError(f"Filter name conflict while combining: {source_file.name}")
                if mode == "rename":
                    target = unique_filter_target(output_dir, source_file, copied)

            if target.resolve() != source_file.resolve():
                copy_filter_file(source_file, target)
            else:
                migrate_filter_file(target)
            copied.append(target.name)

    if not copied:
        raise ValueError("No .dat filter files were found in the supplied inputs")

    write_filter_index(output_dir)  # named after the folder, where MESA opens it
    return output_dir

