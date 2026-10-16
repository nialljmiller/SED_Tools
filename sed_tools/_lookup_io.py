"""
sed_tools/_lookup_io.py

Canonical utilities for reading and writing lookup_table.csv files.

All grabbers, precompute_flux_cube, spectra_cleaner, and api should
import from here rather than reimplementing column detection or CSV writing.
"""

from __future__ import annotations

import csv
import logging
from typing import Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

# Ordered candidate lists for each physical axis.
# Extends header_parser.ALIASES for DataFrame column matching.
_TEFF_CANDIDATES  = ["teff", "t_eff", "t eff", "temperature", "temp"]
_LOGG_CANDIDATES  = ["logg", "log_g", "log(g)", "log g", "surface_gravity", "gravity"]
_META_CANDIDATES  = ["metallicity", "meta", "feh", "[fe/h]", "[m/h]", "m/h", "m_h", "mh",
                     "z", "zh", "zmet", "z/z0", "z/zsun"]
_FILE_CANDIDATES  = ["file_name", "filename", "file"]


def find_column(keys: Iterable[str], candidates: List[str]) -> Optional[str]:
    """Return the first key from *keys* that matches any candidate (case-insensitive).

    Parameters
    ----------
    keys :
        Available column names (DataFrame columns, dict keys, …).
    candidates :
        Ordered list of names to try, from most to least preferred.

    Returns
    -------
    str or None
        The matched key as it appears in *keys*, or ``None`` if no match found.
    """
    lc_map = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lc_map:
            return lc_map[c.lower()]
    return None


def find_teff_column(keys: Iterable[str]) -> Optional[str]:
    return find_column(keys, _TEFF_CANDIDATES)

def find_logg_column(keys: Iterable[str]) -> Optional[str]:
    return find_column(keys, _LOGG_CANDIDATES)

def find_metallicity_column(keys: Iterable[str]) -> Optional[str]:
    return find_column(keys, _META_CANDIDATES)

def find_file_column(keys: Iterable[str]) -> Optional[str]:
    return find_column(keys, _FILE_CANDIDATES)


def write_lookup_csv(
    records: Union[Dict[str, List], List[Dict]],
    path: str,
    *,
    columns: Optional[List[str]] = None,
) -> int:
    """Write a lookup table to *path* with a '#'-prefixed header row.

    Parameters
    ----------
    records :
        Either a column-major dict ``{col_name: [val, ...]}`` or a list of
        row dicts ``[{col: val, ...}, ...]``.
    path :
        Destination file path.
    columns :
        Explicit column order.  If omitted, inferred from *records*.

    Returns
    -------
    int
        Number of data rows written.
    """
    if isinstance(records, dict):
        cols = columns or list(records.keys())
        n = len(records[cols[0]]) if cols else 0
        rows = [{c: records[c][i] for c in cols} for i in range(n)]
    else:
        rows = list(records)
        cols = columns or (list(rows[0].keys()) if rows else [])
        n = len(rows)

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("#" + ",".join(cols) + "\n")
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writerows(rows)

    logger.debug("Wrote lookup table: %s (%d rows)", path, n)
    return n
