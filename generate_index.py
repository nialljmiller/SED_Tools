#!/usr/bin/env python3
"""
Generate index.json for SED_Tools data archive
Now includes both stellar models and filter profiles
"""

import json
import os
from datetime import datetime
from pathlib import Path


def scan_stellar_models(base_path: Path):
    """Scan stellar_models directory for model information."""
    stellar_dir = base_path / "stellar_models"

    if not stellar_dir.exists():
        return [], {}

    models = []
    models_detailed = {}

    for model_dir in sorted(stellar_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Skip hidden directories (.cache, .tmp, etc.)
        if model_name.startswith('.'):
            continue

        # Check for key files
        flux_cube = model_dir / "flux_cube.bin"
        lookup_table = model_dir / "lookup_table.csv"
        h5_file = model_dir / f"{model_name}.h5"

        # Count spectra files
        spectra_files = list(model_dir.glob("*.txt"))
        spectra_count = len(spectra_files)

        # Calculate size
        total_size = 0
        for file in model_dir.rglob("*"):
            if file.is_file():
                try:
                    total_size += file.stat().st_size
                except (OSError, PermissionError):
                    pass

        models_detailed[model_name] = {
            "name": model_name,
            "has_flux_cube": flux_cube.exists(),
            "has_lookup_table": lookup_table.exists(),
            "has_h5_bundle": h5_file.exists(),
            "spectra_count": spectra_count,
            "size_bytes": total_size,
            "size_human": format_size(total_size),
        }

        # Only advertise models that actually have spectra
        if spectra_count > 0:
            models.append(model_name)

    return models, models_detailed


def scan_filters(base_path: Path):
    """Scan filters directory for facility/instrument information."""
    filters_dir = base_path / "filters"

    if not filters_dir.exists():
        return [], {}

    facilities = []
    filters_detailed = {}
    total_filters = 0

    for facility_dir in sorted(filters_dir.iterdir()):
        if not facility_dir.is_dir():
            continue

        facility_name = facility_dir.name
        facilities.append(facility_name)

        instruments = {}

        for instrument_dir in sorted(facility_dir.iterdir()):
            if not instrument_dir.is_dir():
                continue

            instrument_name = instrument_dir.name

            # Count filter files (.dat files)
            filter_files = list(instrument_dir.glob("*.dat"))
            filter_count = len(filter_files)
            total_filters += filter_count

            # Calculate size
            total_size = 0
            for file in instrument_dir.rglob("*"):
                if file.is_file():
                    try:
                        total_size += file.stat().st_size
                    except (OSError, PermissionError):
                        pass

            instruments[instrument_name] = {
                "name": instrument_name,
                "filter_count": filter_count,
                "filters": sorted([f.stem for f in filter_files]),
                "size_bytes": total_size,
                "size_human": format_size(total_size),
            }

        filters_detailed[facility_name] = {
            "name": facility_name,
            "instruments": instruments,
            "instrument_count": len(instruments),
        }

    return facilities, filters_detailed, total_filters


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def generate_index(data_dir: str = None):
    """Generate index.json for the data archive."""

    if data_dir is None:
        # Try to find the data directory
        script_dir = Path(__file__).parent

        # Check common locations
        candidates = [
            script_dir / "data",
            script_dir / ".." / "data",
            Path("/media/data3/MESA/SED_Tools/data"),
        ]

        for candidate in candidates:
            if candidate.exists():
                data_dir = candidate
                break

        if data_dir is None:
            data_dir = Path("data")
    else:
        data_dir = Path(data_dir)

    print(f"Scanning {data_dir}...")

    # Scan stellar models
    print("Scanning stellar_models/...")
    models, models_detailed = scan_stellar_models(data_dir)

    # Calculate stellar model statistics
    total_spectra = sum(m["spectra_count"] for m in models_detailed.values())
    total_size = sum(m["size_bytes"] for m in models_detailed.values())
    complete_models = sum(1 for m in models_detailed.values()
                         if m["has_flux_cube"] and m["has_lookup_table"])

    # Scan filters
    print("Scanning filters/...")
    facilities, filters_detailed, total_filters = scan_filters(data_dir)

    # Build index structure
    index = {
        "version": "1.0",
        "generated": datetime.utcnow().isoformat(),
        "base_url": "http://nillmill.ddns.net/sed_tools",
        "statistics": {
            "total_models": len(models),
            "complete_models": complete_models,
            "total_spectra": total_spectra,
            "total_size_bytes": total_size,
            "total_size_human": format_size(total_size),
            "total_facilities": len(facilities),
            "total_filters": total_filters,
        },
        "models": models,
        "models_detailed": models_detailed,
        "filters": {
            "facilities": facilities,
            "facilities_detailed": filters_detailed,
        },
    }

    # Write index.json
    output_file = data_dir / "index.json"
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"Generated {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Total models: {len(models)}")
    print(f"  Complete models: {complete_models}")
    print(f"  Total spectra: {total_spectra:,}")
    if facilities:
        print(f"  Total facilities: {len(facilities)}")
        print(f"  Total filters: {total_filters:,}")
    print(f"  Total size: {format_size(total_size)}")

    return output_file


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    generate_index(data_dir)
