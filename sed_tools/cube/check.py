"""What a grid's flux cube or HDF5 bundle lacks of the ``.txt`` spectra beside it.

The files' contents decide, never their times, so a grid copied without its times is as complete as the original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..grids.spectrum_io import bundled_spectra, list_text_spectra
from .build import spectra_not_in_cube
from .variants import _variant_paths


def _flux_cube_file(model_dir: Path) -> Optional[Path]:
    """The grid's flux cube: ``variants_index.csv`` (one cube per variant), else ``flux_cube.bin``, else None.

    A ``flux_cube.bin`` beside ``variants_index.csv`` is a copy of one variant's cube (NJM serves its default
    variant's there, for MESA pointed at the grid's folder); it is not the grid's cube, so it is never compared
    with the grid's spectra, which every variant's rows list.
    """
    for name in ("variants_index.csv", "flux_cube.bin"):
        if (Path(model_dir) / name).is_file():
            return Path(model_dir) / name
    return None


def _flux_cube_shortfall(model_dir: Path) -> Optional[str]:
    """What the flux cube lacks of the grid's .txt spectra, judged by content; None if nothing or no cube.

    With ``variants_index.csv``, each variant's cube is compared with its own lookup table, and between
    them they must list every spectrum; a variant whose folder or cube is missing (a split grid downloaded
    without its variant folders) cannot be compared, which is reported too.
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
    return f"does not include {len(lacking)} of the {len(present)} spectra beside it" if lacking else None


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
