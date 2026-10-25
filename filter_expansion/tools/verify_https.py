#!/usr/bin/env python3
"""After deployment: fetch every deployed file over HTTPS and compare with the staged copy.

Usage: verify_https.py Facility/Instrument [...]
Checks each published .dat, component, index file and manifest byte-for-byte, that
row 1 is the metadata row, and that index.json lists the instrument with all its bands.
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://nillmill.ddns.net/sed_tools"
STAGE = Path(__file__).resolve().parent.parent / "stage" / "filters"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "SED_Tools-filter-expansion",
                                               "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main(keys):
    index = json.loads(get(f"{BASE}/index.json"))["filters"]["facilities_detailed"]
    bad = 0
    for key in keys:
        fac, inst = key.split("/", 1)
        local = STAGE / fac / inst
        files = sorted(p for p in local.rglob("*") if p.is_file())
        mismatched = [p for p in files if get(f"{BASE}/filters/{fac}/{inst}/{p.relative_to(local).as_posix()}")
                      != p.read_bytes()]
        rows = [get(f"{BASE}/filters/{fac}/{inst}/{p.name}").split(b"\n", 1)[0] for p in local.glob("*.dat")]
        bad_rows = [r for r in rows if not r.startswith(b"Wavelength,Transmission,FilterID=")]
        entry = index.get(fac, {}).get("instruments", {}).get(inst)
        bands = sorted(p.stem for p in local.glob("*.dat"))
        idx_ok = bool(entry) and sorted(entry.get("filters", [])) == bands
        ok = not mismatched and not bad_rows and idx_ok
        bad += not ok
        print(f"{'ok' if ok else 'FAIL':4} {key}: {len(files)} files over HTTPS, "
              f"{len(mismatched)} mismatched, {len(bad_rows)} bad row-1, index.json "
              f"{'lists ' + str(len(bands)) + ' bands' if idx_ok else 'MISSING/INCOMPLETE'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
