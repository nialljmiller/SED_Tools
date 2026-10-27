import csv
import math
import os
from typing import Iterable, List, Optional
from .header_parser import parse_header



def scan_existing_files(base_dir):
    """
    Scan the directory structure to find existing models and spectra files.

    Returns:
        dict: A mapping of model names to their respective spectra files.
    """
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

    A column of NaN would make the flux-cube builder fail, while a grid with
    no column for an axis (e.g. a blackbody grid) is built with one node.
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
    from .models import STELLAR_DIR_DEFAULT
    base_dir = str(STELLAR_DIR_DEFAULT)
    os.makedirs(base_dir, exist_ok=True)

    # Scan for existing models and files
    existing_files = scan_existing_files(base_dir)

    # Regenerate lookup tables for each model
    for model, files in existing_files.items():
        print(f"Processing spectral grid: {model}")
        output_dir = os.path.join(base_dir, model)
        regenerate_lookup_table(model, files, output_dir)


if __name__ == "__main__":
    main()
