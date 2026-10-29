"""lookup_table.csv: finding its columns, writing it, and rebuilding it from spectrum headers.

The old import paths sed_tools._lookup_io and sed_tools.svo_regen_spectra_lookup forward here.
"""

import csv
import logging
import math
import os
from typing import Dict, Iterable, List, Optional, Union

from .header_parser import parse_header

logger = logging.getLogger("sed_tools._lookup_io")


# The column names find_*_column try, most preferred first
_TEFF_CANDIDATES  = ["teff", "t_eff", "t eff", "temperature", "temp"]
_LOGG_CANDIDATES  = ["logg", "log_g", "log(g)", "log g", "surface_gravity", "gravity"]
_META_CANDIDATES  = ["metallicity", "meta", "feh", "[fe/h]", "[m/h]", "m/h", "m_h", "mh",
                     "z", "zh", "zmet", "z/z0", "z/zsun"]
_FILE_CANDIDATES  = ["file_name", "filename", "file"]


def find_column(keys: Iterable[str], candidates: List[str]) -> Optional[str]:
    """The key in *keys* (column names, dict keys, ...) that matches the earliest of *candidates*.

    Matching ignores case; the key is returned as *keys* spells it, or None if none matches.
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
    comment: bool = True,
    restval="",
) -> int:
    """Write a lookup table to *path* and return the number of data rows written.

    *records* is a column-major dict ``{column: [value, ...]}`` or a list of row dicts.
    The header row is '#'-prefixed, or a plain csv header with ``comment=False``.
    *columns* gives the column order (default: the records' own), and *restval*
    is what a row without a column's key gets in that column.
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
        writer = csv.DictWriter(f, fieldnames=cols, restval=restval, extrasaction="ignore")
        if comment:
            f.write("#" + ",".join(cols) + "\n")
        else:
            writer.writeheader()
        writer.writerows(rows)

    logger.debug("Wrote lookup table: %s (%d rows)", path, n)
    return n


def scan_existing_files(base_dir):
    """Map each folder name under base_dir to the paths of the .txt spectra in it."""
    existing_files = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                model_name = os.path.basename(root)
                if model_name not in existing_files:
                    existing_files[model_name] = []
                existing_files[model_name].append(os.path.join(root, file))
    return existing_files


def _discover_txt_files(directory: str) -> List[str]:
    """Return a sorted list of .txt spectra inside *directory*."""
    return sorted(
        [
            os.path.join(directory, fname)
            for fname in os.listdir(directory)
            if fname.lower().endswith(".txt")
        ]
    )


# parse_header gives these as NaN when a spectrum's header does not state them
_PARAMETERS = ("teff", "logg", "metallicity")


def _drop_unstated_parameter(rows, param, output_dir):
    """Remove *param* from every row if no spectrum states it; ValueError if only some do.

    A NaN column would make the flux-cube builder fail; a grid with no column for an axis is built with one node.
    """
    unstated = [row["file_name"] for row in rows if math.isnan(row[param])]
    if not unstated:
        return
    if len(unstated) < len(rows):
        raise ValueError(
            f"{len(unstated)} of {len(rows)} spectra in {output_dir} do not state {param} "
            f"in their headers: {', '.join(unstated)}. Fix or remove them, then rerun.")
    print(f"No spectrum in {output_dir} states {param}, so lookup_table.csv has no {param} column "
          f"(if the headers give it under another name, add that name to header_parser.ALIASES).")
    for row in rows:
        del row[param]


def regenerate_lookup_table(
    model: Optional[str] = None,
    files: Optional[Iterable[str]] = None,
    output_dir: Optional[str] = None,
):
    """Write <output_dir>/lookup_table.csv from the headers of the given spectra.

    *files* defaults to every .txt in output_dir. A parameter (teff, logg,
    metallicity) that no spectrum states gets no column, e.g. logg and
    metallicity for a blackbody grid. Raises ValueError if only some spectra
    state a parameter, RuntimeError if there are no spectra, and OSError if
    one cannot be read.
    """
    if output_dir is None and isinstance(model, str) and os.path.isdir(model):
        output_dir = model
        model_name = os.path.basename(os.path.normpath(model))
    else:
        model_name = model if isinstance(model, str) else "unknown"

    if output_dir is None:
        raise TypeError("output_dir must be supplied or derivable from 'model'.")

    if files is None:
        files = _discover_txt_files(output_dir)

    files = list(files)
    if not files:
        raise RuntimeError(
            f"No spectra files found for model '{model_name}' in {output_dir}."
        )

    lookup_table_path = os.path.join(output_dir, "lookup_table.csv")
    all_keys = set()
    metadata_rows = []

    for file_path in files:
        row = parse_header(file_path)
        row["file_name"] = os.path.basename(file_path)
        metadata_rows.append(row)

    for param in _PARAMETERS:
        _drop_unstated_parameter(metadata_rows, param, output_dir)
    for row in metadata_rows:
        all_keys.update(row.keys())

    with open(lookup_table_path, mode="w", newline="") as csv_file:
        header = ["file_name"] + sorted(all_keys - {"file_name"})
        csv_file.write("#" + ", ".join(header) + "\n")
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writerows(metadata_rows)
    print(f"Lookup table regenerated: {lookup_table_path}")


def main():
    """Rewrite lookup_table.csv in every grid folder under STELLAR_DIR_DEFAULT."""
    from ..cube.model import STELLAR_DIR_DEFAULT
    base_dir = str(STELLAR_DIR_DEFAULT)
    os.makedirs(base_dir, exist_ok=True)
    existing_files = scan_existing_files(base_dir)
    for model, files in existing_files.items():
        print(f"Processing spectral grid: {model}")
        output_dir = os.path.join(base_dir, model)
        regenerate_lookup_table(model, files, output_dir)


if __name__ == "__main__":
    main()
