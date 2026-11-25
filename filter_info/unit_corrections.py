"""Curves whose wavelengths are in another unit than the header declares (Nill, 24 Nov 2026: "Fix it").

filter_info/curated/unit_corrections.json lists each proven case: the curve, the unit its numbers are really in, and
the publisher's statement that proves it. The wavelength column is rescaled to the declared Angstrom by exact decimal
arithmetic (transmission rows unchanged), the evidence is added to the curve's Sources, and the change is recorded
in filter_expansion/provenance/<Facility>__<Instrument>.json (bands.<band>.cleaning), which info.json shows.

  python3 -m filter_info.unit_corrections DATA_DIR --dry-run
  python3 -m filter_info.unit_corrections DATA_DIR --apply BACKUP_DIR
"""
import argparse
import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURATED = Path(__file__).with_name("curated") / "unit_corrections.json"
PROVENANCE = ROOT / "filter_expansion" / "provenance"
TO_ANGSTROM = {"nm": Decimal(10), "um": Decimal(10000)}
NOTE = ("Wavelengths were in {unit} although the header (as SVO's record) declared Angstrom; multiplied by {scale} to "
        "Angstrom ({evidence}; Nill, 24 Nov 2026).")


def _scaled(line, scale):
    wl, rest = line.split(",", 1)
    value = Decimal(wl) * scale
    text = format(value.normalize(), "f")
    return f"{text if '.' in text else text + '.0'},{rest}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", type=Path, metavar="BACKUP_DIR")
    args = ap.parse_args(argv)
    from sed_tools.filters.curve_metadata import read_filter_header, write_filter_header
    from sed_tools.photometry import load_filter_curve, pivot_wavelength

    provenance = {}
    for entry in json.loads(CURATED.read_text()):
        path = args.data_dir / "filters" / entry["path"]
        head, _, body = path.read_text().partition("\n")
        fields = read_filter_header(path)
        if fields.get("WavelengthUnit", "").lower() != "angstrom":
            print(f"{entry['path']}: header declares {fields.get('WavelengthUnit')!r}, not Angstrom; skipped")
            continue
        curve = load_filter_curve(str(path))
        before = pivot_wavelength(curve.wavelength, curve.transmission, curve.detector_type)
        if not entry["pivot_range_A"][0] <= before * float(TO_ANGSTROM[entry["unit"]]) <= entry["pivot_range_A"][1]:
            sys.exit(f"{entry['path']}: rescaled pivot outside the publisher's range; stop")
        if entry["pivot_range_A"][0] <= before <= entry["pivot_range_A"][1]:
            print(f"{entry['path']}: already in Angstrom (pivot {before:.1f}); skipped")
            continue
        scale = TO_ANGSTROM[entry["unit"]]
        rows = [_scaled(l, scale) for l in body.split("\n") if l.strip()]
        note = NOTE.format(unit=entry["unit"], scale=scale, evidence=entry["evidence"])
        print(f"{entry['path']}: pivot {before:.2f} -> {before * float(scale):.1f} A; {len(rows)} rows rescaled")
        fac, inst, band = entry["path"][:-4].split("/")
        provenance.setdefault(f"{fac}__{inst}", {})[band] = {"cleaning": note, "sources": entry["source"]}
        if args.apply:
            target = args.apply / "data" / "filters" / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            path.write_text(head + "\n" + "\n".join(rows) + "\n")
            fields = read_filter_header(path)
            if entry["source"] not in fields.get("Sources", ""):
                fields["Sources"] = " | ".join(x for x in (fields.get("Sources"), entry["source"]) if x)
                write_filter_header(path, fields)
    for name, bands in sorted(provenance.items()):
        path = PROVENANCE / f"{name}.json"
        doc = json.loads(path.read_text()) if path.is_file() else {"facility": name.split("__")[0],
                                                                    "instrument": name.split("__")[1], "bands": {}}
        for band, fields in bands.items():
            entry = doc.setdefault("bands", {}).setdefault(band, {})
            for key, value in fields.items():
                old = entry.get(key)
                entry[key] = value if not old or value in old else f"{old} | {value}" if key == "sources" else f"{old} {value}"
        if args.apply:
            if path.is_file():
                target = args.apply / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
