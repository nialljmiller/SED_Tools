"""What a grid's flux cube or HDF5 bundle lacks of the ``.txt`` spectra beside it.

The files' contents decide, never their times, so a grid copied without its times is as complete as the original.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from ..grids.spectrum_io import bundled_spectra, list_text_spectra
from .build import spectra_not_in_cube


def _flux_cube_file(model_dir: Path) -> Optional[Path]:
    """The grid's flux cube: ``flux_cube.bin``, else ``variants_index.csv`` (one cube per variant), else None."""
    for name in ("flux_cube.bin", "variants_index.csv"):
        if (Path(model_dir) / name).is_file():
            return Path(model_dir) / name
    return None


def _flux_cube_shortfall(model_dir: Path) -> Optional[str]:
    """What the flux cube lacks of the grid's .txt spectra, judged by content; None if nothing or no cube.

    With ``variants_index.csv``, each variant's cube is compared with its own lookup table, and between
    them they must list every spectrum.
    """
    model_dir = Path(model_dir)
    cube = _flux_cube_file(model_dir)
    present = {path.name for path in list_text_spectra(model_dir)} if cube is not None else set()
    if not present:
        return None
    try:
        variants = [model_dir] if cube.name == "flux_cube.bin" else [model_dir / v for v in _variant_paths(cube)]
        listed, lacking = set(), set()
        for directory in variants:
            names, missing = spectra_not_in_cube(directory / "flux_cube.bin", directory / "lookup_table.csv")
            listed.update(names)
            lacking.update(missing)
    except (OSError, ValueError, KeyError, TypeError) as exc:  # unreadable or malformed files
        return f"cannot be compared with the spectra beside it ({exc})"
    lacking = (lacking | (present - listed)) & present
    if not lacking:
        return None
    return f"does not include {len(lacking)} of the {len(present)} spectra beside it"


def _variant_paths(variants_index: Path) -> List[str]:
    """The folder of each variant ``variants_index.csv`` lists, relative to the grid's folder."""
    with open(variants_index, newline="", encoding="utf-8") as fh:
        fields = [name.lstrip("#").strip() for name in fh.readline().strip().split(",")]
        return [row["path"] for row in csv.DictReader(fh, fieldnames=fields)]


def _bundle_shortfall(bundle: Path, model_dir: Path) -> Optional[str]:
    """How an HDF5 bundle's spectra differ from the .txt spectra in ``model_dir``, by name; None when the same."""
    present = {path.name for path in list_text_spectra(model_dir)}
    try:
        held = set(bundled_spectra(bundle))
    except (OSError, ValueError) as exc:
        return f"cannot be compared with the spectra beside it ({exc})"
    if held == present:
        return None
    extra = len(held - present)
    return (f"holds {len(held & present)} of the {len(present)} spectra beside it"
            + (f" and {extra} that are not there" if extra else ""))
