"""Check that every manifest band is listed in its instrument's MESA index file.

SED_Tools 0.4.1 downloads the curves the index file ``<Facility>/<Instrument>/<Instrument>``
lists, then requires a curve for every band in ``manifest.json`` ``bands``. A band missing
from the index file makes it fall back to SVO, which overwrites NJM's curves with SVO's
copies under the same names and exits 1 (or fails outright when SVO is down).

Archive copies kept beside the current curves (``<band>_svo``, ``<band>_old``) stay out of
the index file, so MESA does not load them, and out of ``bands``: they go under
``archived_bands``, which the clients ignore.

Usage: python3 -m filter_info.check_manifests [DATA_DIR]   (default: data)
Prints every offending manifest and exits 1; prints a one-line summary and exits 0 otherwise.
"""
import json
import sys
from pathlib import Path


def index_entries(index):
    """The file names an index file lists (blank and '#' lines skipped, as the clients do)."""
    return {line.strip() for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def problems(filters_dir):
    """(manifest path, message) for each manifest whose bands are not all in the index file."""
    for manifest in sorted(Path(filters_dir).glob("*/*/manifest.json")):
        instrument_dir = manifest.parent
        try:
            bands = json.loads(manifest.read_text(encoding="utf-8")).get("bands")
        except (OSError, ValueError, AttributeError) as exc:
            yield manifest, f"cannot be read ({exc})"
            continue
        if not isinstance(bands, list) or not all(isinstance(b, str) for b in bands):
            yield manifest, "'bands' is not a list of band names"
            continue
        index = instrument_dir / instrument_dir.name
        if not index.is_file():
            yield manifest, f"has no index file {index.name} for its {len(bands)} band(s)"
            continue
        listed = index_entries(index)
        missing = [band for band in bands if f"{band}.dat" not in listed]
        if missing:
            yield manifest, (f"lists {len(missing)} band(s) that its index file {index.name} does not: "
                             f"{', '.join(missing)} (archive copies belong under 'archived_bands')")


def main(argv):
    data_dir = Path(argv[1] if len(argv) > 1 else "data")
    filters_dir = data_dir / "filters"
    if not filters_dir.is_dir():
        print(f"check_manifests: {filters_dir} is not a directory", file=sys.stderr)
        return 2
    found = list(problems(filters_dir))
    for manifest, message in found:
        print(f"ERROR {manifest}: {message}", file=sys.stderr)
    checked = sum(1 for _ in filters_dir.glob("*/*/manifest.json"))
    if found:
        print(f"check_manifests: {len(found)} of {checked} manifests fail; SED_Tools 0.4.1 would replace those "
              "instruments' curves with SVO's. Fix them before refreshing.", file=sys.stderr)
        return 1
    print(f"check_manifests: all {checked} manifests list only bands in their index file")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
