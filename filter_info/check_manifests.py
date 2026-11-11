"""Check every served filter set's records before a refresh: one rule, SED_Tools' own.

``sed_tools.filters.manifest.set_problems`` is the rule SED_Tools checks installed sets by: the MESA
index file ``<Facility>/<Instrument>/<Instrument>`` lists exactly the curves in the folder, less the
manifest's ``archived_bands``; ``manifest.json`` names its own folder and declares no band the index
does not list; and no curve's ``FilterID`` is of a source instrument another set claims in
``filter_info/svo_instruments.json`` while this one does not (one release filed in another's folder,
as GAIA/GAIA held GAIA/Gaia2m's Gbp_bright and Gbp_faint).

A manifest band missing from the index file matters most: SED_Tools 0.4.1 downloads the curves the
index lists, then requires a curve for every band in ``manifest.json``, so it falls back to SVO,
which overwrites NJM's curves with SVO's copies under the same names and exits 1. Archive copies kept
beside the current curves (such as ``<band>_old``) stay out of the index file, so MESA does not load
them, and out of ``bands``: they go under ``archived_bands``, which the clients ignore.

Usage: python3 -m filter_info.check_manifests [DATA_DIR]   (default: data)
Prints every offending set and exits 1; prints a one-line summary and exits 0 otherwise.
"""
import json
import sys
from pathlib import Path

from sed_tools.filters.manifest import set_problems

CLAIMS = Path(__file__).with_name("svo_instruments.json")


def problems(filters_dir, claims=None):
    """(set folder, message naming the file) for each way a served set's records disagree."""
    filters_dir = Path(filters_dir)
    for manifest in sorted(filters_dir.glob("*/*/manifest.json")):
        try:
            bands = json.loads(manifest.read_text(encoding="utf-8")).get("bands")
        except (OSError, ValueError, AttributeError) as exc:
            yield manifest.parent, f"{manifest} cannot be read ({exc})"
            continue
        if not isinstance(bands, list) or not all(isinstance(b, str) for b in bands):
            yield manifest.parent, f"{manifest}: 'bands' is not a list of band names"
        elif not any(manifest.parent.glob("*.dat")):
            yield manifest.parent, f"{manifest} declares {len(bands)} band(s) of a folder with no curves"
    for instrument_dir in sorted(p for p in filters_dir.glob("*/*") if p.is_dir()):
        for message in set_problems(instrument_dir, claims):
            yield instrument_dir, message


def main(argv):
    data_dir = Path(argv[1] if len(argv) > 1 else "data")
    filters_dir = data_dir / "filters"
    if not filters_dir.is_dir():
        print(f"check_manifests: {filters_dir} is not a directory", file=sys.stderr)
        return 2
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))["instruments"] if CLAIMS.is_file() else None
    found = list(problems(filters_dir, claims))
    for _, message in found:
        print(f"ERROR {message}", file=sys.stderr)
    checked = sum(1 for p in filters_dir.glob("*/*") if p.is_dir())
    if found:
        print(f"check_manifests: {len({where for where, _ in found})} of {checked} sets fail. A manifest band "
              "outside the index file makes SED_Tools 0.4.1 replace that set's curves with SVO's; a curve of "
              "another set's instrument mixes two releases in one set. Fix them before refreshing.",
              file=sys.stderr)
        return 1
    print(f"check_manifests: all {checked} sets agree with their index files, manifests and claims")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
