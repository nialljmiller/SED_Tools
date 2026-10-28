#!/usr/bin/env python3
"""Regenerate sed_tools/instrument_catalogue.json from NJM's index.json, at release time.

    python3 scripts/make_instrument_catalogue.py --svo-evidence svo_instruments.json   # as released
    python3 scripts/make_instrument_catalogue.py --index FILE ...                     # a saved index.json

The bundled catalogue keeps, for each NJM Facility/Instrument (keyed by NJM's
spelling), its one-line ``description`` and ``svo_instruments``, the SVO
instruments that are the same product.  The filter pickers use it to name,
describe and merge instruments whether or not NJM is reachable.

Each instrument also gets ``bands``, the bands NJM serves (its MESA index file,
read from the mirror: archived copies such as ``H_svo`` are not in it), so that
they are known while NJM is down.

``--svo-evidence`` is NJM's ``filter_info/svo_instruments.json`` (in the server
checkout).  With it, each instrument that has ``svo_instruments`` also gets
``svo_bands``: for each of those SVO instruments, its bands whose curve is one of
NJM's served bands, written ``"Gbp=BP"`` where NJM names the band differently.
An instrument can then be filled from SVO under NJM's band names while NJM is
down.  An SVO band whose curve is only an archived copy, or more than one served
band, is left out.  Review the diff, then commit the file.

The file has one line per instrument, to keep it small and its diffs readable.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's sed_tools, installed or not

from sed_tools.filter_library import INSTRUMENT_CATALOGUE, instrument_catalogue  # noqa: E402

MIRROR_URL = "https://nillmill.ddns.net/sed_tools"
INDEX_URL = f"{MIRROR_URL}/index.json"
USER_AGENT = "SED_Tools-builder/1 (https://github.com/nialljmiller/SED_Tools)"
PAUSE = 0.2  # seconds between requests: NJM blocks bursts


def _get(url):
    import requests

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(10, 120))
    response.raise_for_status()
    return response


def read_index(source):
    """NJM's index.json from a URL or a file."""
    if source.startswith(("http://", "https://")):
        return _get(source).json()
    return json.loads(Path(source).read_text(encoding="utf-8"))


def served_bands(mirror, facility, instrument):
    """The bands NJM serves for an instrument: its MESA index file's curves, without ``.dat``."""
    text = _get(f"{mirror}/filters/{quote(facility)}/{quote(instrument)}/{quote(instrument)}").text
    names = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return [name[:-len(".dat")] for name in names if name.endswith(".dat")]


def svo_bands(evidence, facility, instrument, claims, bands):
    """``{SVO filter ID: NJM band}`` for each ID of the claimed SVO instruments held as one served band."""
    mapping, skipped = {}, []
    for claim in claims:
        for filter_id, record in sorted(((evidence.get(claim) or {}).get("bands") or {}).items()):
            same = [key.rsplit("/", 1)[1] for key, match in record.get("njm", {}).items()
                    if match.get("same") and key.rsplit("/", 1)[0] == f"{facility}/{instrument}"]
            served = [band for band in same if band in bands]
            if len(served) == 1 and filter_id.startswith(claim + "."):
                mapping[filter_id] = served[0]
            else:
                skipped.append(f"{filter_id} -> {', '.join(same) or 'nothing'}")
    return mapping, skipped


def add_bands(catalogue, evidence, mirror):
    """Give every instrument its served ``bands``, and with ``evidence`` its ``svo_bands``."""
    groups = None
    if evidence is not None:
        groups = evidence.get("svo_groups")
        if not isinstance(groups, dict):
            raise SystemExit("the SVO evidence file has no svo_groups; not NJM's svo_instruments.json")
    for facility, record in catalogue.items():
        for instrument, meta in record["instruments"].items():
            time.sleep(PAUSE)
            meta["bands"] = served_bands(mirror, facility, instrument)
            if groups is None or not meta["svo_instruments"]:
                continue
            meta["svo_bands"], skipped = svo_bands(groups, facility, instrument, meta["svo_instruments"],
                                                   meta["bands"])
            for line in skipped:
                print(f"  {facility}/{instrument}: not mapped (archived copy or ambiguous): {line}")


def stored_svo_bands(mapping, claims):
    """``{SVO filter ID: NJM band}`` as stored: ``{SVO instrument: ["G", "Gbp=BP", ...]}``."""
    stored = {}
    for filter_id, band in sorted(mapping.items()):
        claim = next(c for c in claims if filter_id.startswith(c + "."))
        svo_band = filter_id[len(claim) + 1:]
        stored.setdefault(claim, []).append(svo_band if svo_band == band else f"{svo_band}={band}")
    return stored


def compact(catalogue):
    """The catalogue without its empty fields, as it is stored."""
    stored = {}
    for facility, record in catalogue.items():
        instruments = {}
        for name, meta in record["instruments"].items():
            meta = dict(meta, svo_bands=stored_svo_bands(meta["svo_bands"], meta["svo_instruments"]))
            instruments[name] = {key: value for key, value in meta.items() if value}
        entry = {"instruments": instruments}
        if record["description"]:
            entry["description"] = record["description"]
        stored[facility] = entry
    return stored


def dumps(data):
    """The catalogue as JSON with one line per instrument."""
    line = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(", ", ": "))
    facilities = []
    for facility, entry in sorted(data["facilities"].items()):
        rows = [f"   {line(name)}: {line(meta)}" for name, meta in sorted(entry["instruments"].items())]
        head = f"  {line(facility)}: {{"
        if "description" in entry:
            head += f"\n   \"description\": {line(entry['description'])},"
        facilities.append(head + "\n   \"instruments\": {\n" + ",\n".join(" " + r for r in rows) + "\n   }\n  }")
    rest = [f" {line(key)}: {line(value)}" for key, value in sorted(data.items()) if key != "facilities"]
    return ("{\n \"facilities\": {\n" + ",\n".join(facilities) + "\n },\n" + ",\n".join(rest) + "\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=INDEX_URL, help=f"index.json URL or file (default: {INDEX_URL})")
    parser.add_argument("--svo-evidence", help="NJM's filter_info/svo_instruments.json, for bands and svo_bands")
    parser.add_argument("--mirror", default=MIRROR_URL, help=f"where the served index files are read "
                                                              f"(default: {MIRROR_URL})")
    parser.add_argument("--output", default=str(INSTRUMENT_CATALOGUE), help="where to write the catalogue")
    args = parser.parse_args()

    index = read_index(args.index)
    detailed = (index.get("filters") or {}).get("facilities_detailed")
    if not isinstance(detailed, dict):
        raise SystemExit(f"{args.index} has no filters.facilities_detailed; not an NJM index.json")
    catalogue = instrument_catalogue(detailed)
    evidence = json.loads(Path(args.svo_evidence).read_text(encoding="utf-8")) if args.svo_evidence else None
    if evidence is None:
        print("No --svo-evidence: the catalogue has no svo_bands, so no instrument can be filled from SVO "
              "under NJM's names while NJM is down")
    add_bands(catalogue, evidence, args.mirror)
    data = {"schema": 1, "source": args.index if args.index.startswith("http") else INDEX_URL,
            "index_generated": index.get("generated"), "facilities": compact(catalogue)}
    text = dumps(data)
    if instrument_catalogue(json.loads(text)["facilities"]) != catalogue:
        raise SystemExit("the written catalogue would not read back as generated")
    Path(args.output).write_text(text, encoding="utf-8")

    instruments = [meta for record in catalogue.values() for meta in record["instruments"].values()]
    print(f"{args.output}: {len(catalogue)} facilities, {len(instruments)} instruments; "
          f"{sum(bool(m['description']) for m in instruments)} with a description, "
          f"{sum(bool(m['svo_instruments']) for m in instruments)} with svo_instruments, "
          f"{sum(bool(m['svo_bands']) for m in instruments)} with svo_bands "
          f"(index generated {index.get('generated')})")


if __name__ == "__main__":
    main()
