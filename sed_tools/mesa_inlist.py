"""The MESA colors inlist lines that point MESA at a local spectral grid or filter set.

mesa_inlist_lines gives them; _print_mesa_inlist is the one place that prints them, after a download, a rebuild,
a combine, mesa_prepare and ``sed-tools mesa-path``. MESA's ``&colors`` namelist reads a grid's folder as
``stellar_atm`` and a filter set's as ``instrument``. It reads a path that does not start with ``/`` relative to
MESA_DIR, and opens ``<instrument>/<its last name>``, so the paths are absolute and have no trailing slash.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

from .cube.variants import _variant_paths


def _line(name: str, path) -> str:
    """``name = '<path>'``, an apostrophe in the path doubled, as a Fortran namelist string reads it."""
    quoted = str(path).replace("'", "''")
    return f"{name} = '{quoted}'"


def mesa_inlist_lines(path: Union[str, os.PathLike]) -> List[str]:
    """The MESA colors inlist line(s) for the spectral grid or filter set in folder ``path``.

    - A grid split into variants (``variants_index.csv``) gives ``stellar_atm = '<variant folder>'`` for each
      variant: MESA reads one variant, and cannot read the grid's own folder.
    - Any other grid (``lookup_table.csv``) gives ``stellar_atm = '<folder>'``.
    - A filter set (the MESA index file named after its folder) gives ``instrument = '<folder>'``.

    Paths are absolute, as given (symlinks are not resolved), with no trailing slash; an apostrophe is doubled, as
    MESA's namelist reads it. ValueError for a folder that is none of these.
    """
    folder = Path(os.path.abspath(os.path.expanduser(path)))
    if (folder / "variants_index.csv").is_file():
        return [_line("stellar_atm", os.path.normpath(folder / variant))
                for variant in _variant_paths(folder / "variants_index.csv")]
    if (folder / "lookup_table.csv").is_file():
        return [_line("stellar_atm", folder)]
    if (folder / folder.name).is_file():
        return [_line("instrument", folder)]
    raise ValueError(f"{folder} is neither a spectral grid (it has no lookup_table.csv) nor a filter set (it has "
                     f"no index file {folder.name}), so MESA cannot read it")


def _print_mesa_inlist(path: Union[str, os.PathLike]) -> None:
    """Print ``The MESA colors inlist path for this is:`` and the lines; of four or more variants, the first two,
    ``…`` and the last. Nothing for a folder MESA cannot read (a fetch whose bounds no spectrum meets leaves none)."""
    try:
        lines = mesa_inlist_lines(path)
    except ValueError:
        return
    print("\nThe MESA colors inlist path for this is:")
    for line in lines if len(lines) < 4 else [*lines[:2], "…", lines[-1]]:
        print(f"    {line}")
    if len(lines) > 1:
        print(f"  (one line per variant: MESA reads one, so use one of these {len(lines)}; "
              "variants_index.csv lists them)")
