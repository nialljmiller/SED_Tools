#!/usr/bin/env python3
"""
Generate index.json for SED_Tools data archive
Now includes both stellar models and filter profiles
"""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from filter_info.compose import SCHEMA as INFO_SCHEMA, InfoSources, write_info
from filter_info.references import UNRESOLVED
from sed_tools import __version__ as SED_TOOLS_VERSION
from sed_tools.filter_colors import instrument_colors
from sed_tools.filter_properties import UNITS, curve_properties, extinction_law, load_sun, load_vega
from sed_tools.filter_metadata import read_filter_header

# Vega spectrum for the Vega-weighted filter properties: data/vega/vega_flam.csv, which is
# CALSPEC alpha_lyr_mod_004 exactly as MESA colors ships it (data/vega/README.md), so the
# zero points NJM serves match MESA colors. SED_VEGA_SPECTRUM overrides it.
REPO_VEGA = Path(__file__).resolve().parent / "data" / "vega" / "vega_flam.csv"
VEGA_CANDIDATES = [
    os.environ.get("SED_VEGA_SPECTRUM", ""),
    str(REPO_VEGA),
    os.path.join(os.environ.get("MESA_DIR", ""), "data/colors_data/stellar_models/vega_flam.csv"),
    "/media/data3/MESA/mesa/data/colors_data/stellar_models/vega_flam.csv",
]
# Solar spectrum for Fsun: data/sun/sun_flam.csv, CALSPEC sun_reference_stis_002 below 9600 A and
# sun_mod_001 from 9600 A, the composite SVO uses (data/sun/README.md). SED_SUN_SPECTRUM overrides it.
REPO_SUN = Path(__file__).resolve().parent / "data" / "sun" / "sun_flam.csv"
SUN_CANDIDATES = [os.environ.get("SED_SUN_SPECTRUM", ""), str(REPO_SUN)]
# Which SVO instruments each NJM instrument duplicates (filter_info/svo_instruments.py writes it).
SVO_INSTRUMENTS = Path(__file__).resolve().parent / "filter_info" / "svo_instruments.json"
# SVO's own Fsun and AfAv, which info.json copies under "svo"; the units are SVO's.
SVO_FIELDS = {
    "Fsun": {
        "unit": "erg/s/cm2/Angstrom",
        "meaning": "SVO's band-averaged f_lambda of the Sun (CALSPEC sun_reference_stis_002 composed with "
                   "sun_mod_001), the quantity NJM's properties.json gives as Fsun",
        "unit_source": ('SVO\'s VOTable: <PARAM name="Fsun" value="22.596337946016" unit="erg/cm2/s/A" '
                        'ucd="phot.flux.density" datatype="double" > (2MASS/2MASS.H, Internet Archive capture '
                        '20250323174315 of http://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php?ID=2MASS/2MASS.H); '
                        'SVO\'s browse page: "Unit :erg/cm2/s/A UCD :phot.flux.density"'),
        "precision": "rounded as SVO's browse page displays it: 2 decimals from 1 up, 3 significant figures below 1",
    },
    "AfAv": {
        "unit": "dimensionless",
        "meaning": ("SVO's A_f/A_V at its reference wavelength (the pivot): k(lambda)/k_V with k_V = 211.4, from SVO's "
                    "opacity table for 'the extinction law by Fitzpatrick (1999) improved by Indebetouw et al (2005) "
                    "in the infrared'. Not the law of NJM's AfAv (properties_definitions.extinction), so the two differ."),
        "unit_source": "SVO's browse page shows the unit as ()",
        "precision": "rounded as SVO's browse page displays it, to 3 significant figures",
    },
}


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


def read_curve_metadata(path: Path):
    """Read a curve's metadata without loading the transmission samples.

    SED_Tools writes it as a third field of the column row
    (``Wavelength,Transmission,Key=value;...``); curves written by 0.3.0 carry
    it as leading ``# Key: value`` lines instead.
    """
    metadata = {}
    with path.open(encoding="utf-8") as curve:
        for line in curve:
            if line.startswith("#"):
                key, separator, value = line[1:].partition(":")
                if separator:
                    metadata[key.strip()] = value.strip()
                continue
            columns, separator, field = line.strip().rpartition(",")
            items = [item.partition("=") for item in field.split(";")] if separator else []
            if items and all(sep and key.isalpha() for key, sep, _ in items):
                metadata.update((key, unquote(value)) for key, _, value in items)
            break
    return metadata


def separate_component_coverage(instrument_dir: Path, bands):
    """Count bands with a real, downloadable curve for each manifest role."""
    manifest_path = instrument_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = {}
    for role, files in manifest.get("components", {}).items():
        available = {
            band for band in bands
            if (path := files.get(band, files.get("*")))
            and (instrument_dir / path).is_file()
        }
        if available:
            coverage[role] = len(available)
    return coverage


def find_vega():
    """Return (spectrum, description) for the Vega reference, or (None, None)."""
    for candidate in VEGA_CANDIDATES:
        path = Path(candidate) if candidate else None
        if path and path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source = {"file": str(path), "sha256": digest}
            if path.resolve() == REPO_VEGA:
                source.update(file="data/vega/vega_flam.csv", url="vega/vega_flam.csv",
                              spectrum="CALSPEC alpha_lyr_mod_004 (vega/README.md)")
            return load_vega(path), source
    return None, None


def find_sun():
    """Return (spectrum, description) for the solar spectrum, or (None, None)."""
    for candidate in SUN_CANDIDATES:
        path = Path(candidate) if candidate else None
        if path and path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            spectrum = load_sun(path)
            source = {"file": str(path), "sha256": digest,
                      "range_angstrom": [float(spectrum[0][0]), float(spectrum[0][-1])]}
            if path.resolve() == REPO_SUN:
                source.update(
                    file="data/sun/sun_flam.csv", url="sun/sun_flam.csv",
                    spectrum=("CALSPEC sun_reference_stis_002 below 9600 Angstrom, sun_mod_001 from 9600 Angstrom "
                              "(sun/README.md)"),
                    units="WAVELENGTH Angstrom (vacuum), FLUX erg/s/cm2/Angstrom at 1 AU",
                    references=["Bohlin, Gordon & Tremblay 2014, PASP 126, 711 (2014PASP..126..711B, "
                                "doi:10.1086/677655): CALSPEC",
                                "Colina, Bohlin & Castelli 1996, AJ 112, 307 (doi:10.1086/118016): "
                                "the solar reference spectrum"])
            return spectrum, source
    return None, None


def _rounded(value):
    return float(f"{value:.7g}") if isinstance(value, float) else value


def write_properties(instrument_dir: Path, filter_files, vega, reference, sun=None):
    """Write the instrument's properties.json (published curves only); True when written."""
    bands = {}
    for curve in sorted(filter_files):
        try:
            props = {k: _rounded(v) for k, v in curve_properties(curve, vega, sun).items()}
        except ValueError as exc:  # an unusable curve is reported, not fatal
            props = {"error": str(exc)}
        header = read_filter_header(curve)
        props["DetectorType"] = header.get("DetectorType", "energy")
        props["DetectorTypeSource"] = header.get("DetectorTypeSource", "assumed")
        bands[curve.stem] = props
    if not bands:
        return False
    # Plotting colours depend on the instrument's other bands (false colours go in order).
    for band, colors in instrument_colors(sorted(filter_files)).items():
        bands[band].update(colors)
    document = {"definitions": reference, "bands": bands}
    path = instrument_dir / "properties.json"
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)
    return True


def picker_fields(curated_dir: Path, svo_instruments, facility: str, instrument: str):
    """The instrument's one-line "description" and the "svo_instruments" it duplicates.

    "description" is the curated short_description; "svo_instruments" comes from
    svo_instruments.json, where an empty list means assessed and none is the same
    product.  A key is left out while the instrument is unassessed.
    """
    fields = {}
    curated = curated_dir / f"{facility}__{instrument}.json"
    short = json.loads(curated.read_text()).get("short_description") if curated.is_file() else None
    if short:
        fields["description"] = short
    if f"{facility}/{instrument}" in svo_instruments:
        fields["svo_instruments"] = svo_instruments[f"{facility}/{instrument}"]
    return fields


def scan_filters(base_path: Path, vega=None, reference=None, info_sources=None, sun=None, svo_instruments=None):
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
            published_paths = Counter()
            detector_types = Counter()
            for curve in filter_files:
                metadata = read_curve_metadata(curve)
                published_paths[metadata.get("Components", "unspecified")] += 1
                detector_types[metadata.get("DetectorType", "unclassified")] += 1
            has_properties = reference is not None and write_properties(
                instrument_dir, filter_files, vega, reference, sun)
            band_properties = json.loads(
                (instrument_dir / "properties.json").read_text())["bands"] if has_properties else None
            has_info = info_sources is not None and write_info(
                info_sources, instrument_dir, filter_files, band_properties)

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
                "published_paths": dict(sorted(published_paths.items())),
                "detector_types": dict(sorted(detector_types.items())),
                "separate_components": separate_component_coverage(
                    instrument_dir, [file.stem for file in filter_files]
                ),
                "has_component_manifest": (instrument_dir / "manifest.json").is_file(),
                "properties": "properties.json" if has_properties else None,
                "info": "info.json" if has_info else None,
                "size_bytes": total_size,
                "size_human": format_size(total_size),
            }
            if info_sources is not None:
                instruments[instrument_name].update(picker_fields(
                    info_sources.curated_dir, svo_instruments or {}, facility_name, instrument_name))

        filters_detailed[facility_name] = {
            "name": facility_name,
            "instruments": instruments,
            "instrument_count": len(instruments),
        }

    return facilities, filters_detailed, total_filters


def warn_unassessed(filters_detailed):
    """Name the instruments that index.json gives no description or svo_instruments."""
    for key, how in (("description", "a short_description in filter_info/curated/instruments/"),
                     ("svo_instruments", "a run of python3 -m filter_info.svo_instruments")):
        missing = [f"{facility}/{name}" for facility, entry in filters_detailed.items()
                   for name, instrument in entry["instruments"].items() if key not in instrument]
        if missing:
            print(f"WARNING: {len(missing)} instruments have no '{key}' in index.json (they need {how}): "
                  + ", ".join(missing))


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
    vega, vega_source = find_vega()
    sun, sun_source = find_sun()
    reference = {
        "module": f"sed_tools.filter_properties (SED_Tools {SED_TOOLS_VERSION})",
        "units": UNITS,
        "vega_spectrum": vega_source,
        "sun_spectrum": sun_source,
        "extinction": extinction_law(),
        "Fsun": ("band-averaged f_lambda of the Sun at 1 AU, integral(W S) / integral(W) with S the sun_spectrum "
                 "and W the curve's detector-aware weight (T energy-counting, T*lambda photon-counting), "
                 "trapezoidal rule on the curve's samples: the weight of ZeroPointVega_flam"),
        "AfAv": ("A(lambda)/A(V) at lambda = WavelengthPivot from the extinction law (see extinction); "
                 "dimensionless"),
        "note": ("Vega quantities are null when the Vega spectrum does not cover the curve (where it transmits at "
                 "least 1 % of its peak), Fsun likewise with the sun_spectrum. AfAv is null where the pivot lies "
                 "outside the extinction law's valid range, or when dust_extinction is not installed "
                 "(extinction.dust_extinction_version is then null)."),
        "colors": ("DisplayColor is an sRGB hex for plotting (sed_tools.filter_colors). ColorMethod "
                   "'true': CIE D65 daylight seen through the curve, for bands with VisibleFraction "
                   "(share of the throughput between 380 and 750 nm) of at least 0.5; 'false': "
                   "ordered false colour for ultraviolet and infrared bands."),
    }
    info_sources = InfoSources(Path(__file__).resolve().parent)
    duplicates = json.loads(SVO_INSTRUMENTS.read_text())
    facilities, filters_detailed, total_filters = scan_filters(
        data_dir, vega, reference, info_sources, sun, duplicates["instruments"])
    warn_unassessed(filters_detailed)
    # Reference keys info.json uses but references.json lacks:
    # `python -m filter_info.references resolve` looks them up for the next run.
    UNRESOLVED.write_text("".join(f"{k}\n" for k in sorted(info_sources.unresolved)))
    info_definitions = {
        "schema": INFO_SCHEMA,
        "svo_records": info_sources.svo_records,
        "references": len(info_sources.references),
        "unresolved_references": len(info_sources.unresolved),
        "svo_fields": SVO_FIELDS,
        "svo_instruments": {k: duplicates[k] for k in ("generated", "svo_list", "thresholds", "method")},
    }

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
            "properties_definitions": reference,
            "info_definitions": info_definitions,
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
