#!/usr/bin/env python3
"""Regenerate sed_tools/instrument_catalogue.json from NJM's index.json, at release time.

    python3 scripts/make_instrument_catalogue.py                  # NJM's current index.json
    python3 scripts/make_instrument_catalogue.py --index FILE     # a saved copy

The bundled catalogue keeps, for each NJM Facility/Instrument (keyed by NJM's
spelling), its one-line ``description`` and ``svo_instruments``, the SVO
instruments that are the same product.  The filter pickers use it to name,
describe and merge instruments whether or not NJM is reachable.  Review the
diff, then commit the file.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's sed_tools, installed or not

from sed_tools.filter_library import INSTRUMENT_CATALOGUE, instrument_catalogue  # noqa: E402

INDEX_URL = "https://nillmill.ddns.net/sed_tools/index.json"
USER_AGENT = "SED_Tools-builder/1 (https://github.com/nialljmiller/SED_Tools)"


def read_index(source):
    """NJM's index.json from a URL or a file."""
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, headers={"User-Agent": USER_AGENT}, timeout=(10, 120))
        response.raise_for_status()
        return response.json()
    return json.loads(Path(source).read_text(encoding="utf-8"))


def compact(catalogue):
    """The catalogue without its empty fields, as it is stored."""
    stored = {}
    for facility, record in catalogue.items():
        entry = {"instruments": {name: {key: value for key, value in meta.items() if value}
                                 for name, meta in record["instruments"].items()}}
        if record["description"]:
            entry["description"] = record["description"]
        stored[facility] = entry
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=INDEX_URL, help=f"index.json URL or file (default: {INDEX_URL})")
    parser.add_argument("--output", default=str(INSTRUMENT_CATALOGUE), help="where to write the catalogue")
    args = parser.parse_args()

    index = read_index(args.index)
    detailed = (index.get("filters") or {}).get("facilities_detailed")
    if not isinstance(detailed, dict):
        raise SystemExit(f"{args.index} has no filters.facilities_detailed; not an NJM index.json")
    catalogue = instrument_catalogue(detailed)
    data = {"schema": 1, "source": args.index if args.index.startswith("http") else INDEX_URL,
            "index_generated": index.get("generated"), "facilities": compact(catalogue)}
    Path(args.output).write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                                 encoding="utf-8")

    instruments = [meta for record in catalogue.values() for meta in record["instruments"].values()]
    print(f"{args.output}: {len(catalogue)} facilities, {len(instruments)} instruments; "
          f"{sum(bool(m['description']) for m in instruments)} with a description, "
          f"{sum(bool(m['svo_instruments']) for m in instruments)} with svo_instruments "
          f"(index generated {index.get('generated')})")


if __name__ == "__main__":
    main()
