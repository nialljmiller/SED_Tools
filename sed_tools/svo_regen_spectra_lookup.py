import csv
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


def regenerate_lookup_table(
    model: Optional[str] = None,
    files: Optional[Iterable[str]] = None,
    output_dir: Optional[str] = None,
):
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
        all_keys.update(row.keys())

    if metadata_rows:
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
        print(f"Processing model: {model}")
        output_dir = os.path.join(base_dir, model)
        regenerate_lookup_table(model, files, output_dir)


if __name__ == "__main__":
    main()
