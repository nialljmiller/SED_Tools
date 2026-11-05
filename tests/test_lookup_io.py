"""Tests for ``sed_tools._lookup_io``: column detection and CSV writing.

All grabbers, ``precompute_flux_cube``, and ``spectra_cleaner`` are
documented to import from this module rather than reimplementing column
detection or CSV writing, so its correctness matters project-wide.
"""

import csv

import pytest

from sed_tools._lookup_io import (
    find_column,
    find_file_column,
    find_logg_column,
    find_metallicity_column,
    find_teff_column,
    write_lookup_csv,
)


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------

def _read_hash_prefixed_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").strip()
        cols = [c.strip() for c in header.split(",")]
        rows = list(csv.reader(fh))
    return cols, rows


