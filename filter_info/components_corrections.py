"""Corrections of what served curves contain (Nill, 25 Nov 2026, Q33 "Investigate, then fix"; evidence in the SED_Tools
report round3-research.md, part B, and in filter_info/curated/components_corrections.json).

Each entry gives a band's corrected recipe, for its manifest's ``published_bands`` and, where the curve header was
wrong too, for its ``Components`` header (with ``ComponentsSource``). A filter component that points at the band's own
curve stays only while that curve is filter-only (a curve that holds more is no filter piece). The change goes to the
set's provenance record (filter_expansion/provenance/<Facility>__<Instrument>.json, bands.<band>.cleaning), which info.json shows.

``problems`` is the class guard filter_info.check_manifests runs:
- a manifest whose ``published`` source is ``assumed`` whose ``published_bands`` entry disagrees with a header that
  names the curve's components;
- a band whose components come from SVO (``ComponentsSource=svo``) where SVO's own Description contradicts them:
  "filter only" against more than one role, or "filter+ccd" (or "filter + ccd") against ``filter`` alone.

  python3 -m filter_info.components_corrections DATA_DIR --dry-run
  python3 -m filter_info.components_corrections DATA_DIR --apply BACKUP_DIR
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURATED = Path(__file__).with_name("curated") / "components_corrections.json"
PROVENANCE = ROOT / "filter_expansion" / "provenance"
SVO_PARSED = ROOT / "source_evidence" / "svo-archive" / "parsed"


def _roles(text):
    from sed_tools.filters.manifest import parse_components
    try:
        return frozenset(parse_components(text)) if text else None
    except Exception:
        return None


def _dumps_like(text, doc):
    old = json.loads(text)
    for ensure_ascii in (False, True):
        if json.dumps(old, indent=2, ensure_ascii=ensure_ascii) + "\n" == text:
            return json.dumps(doc, indent=2, ensure_ascii=ensure_ascii) + "\n"
    raise ValueError("manifest is not in the indent-2 JSON layout; not rewritten")


_svo_descriptions = {}


def _svo_description(filter_id):
    facility = filter_id.split("/", 1)[0]
    if facility not in _svo_descriptions:
        path = SVO_PARSED / f"{re.sub(r'[^A-Za-z0-9._-]', '_', facility)}.json"
        records = json.loads(path.read_text()) if path.is_file() else {}
        out = {}
        for fid, rec in records.items():
            for kind in ("votable", "browse"):
                for item in rec.get(kind) or []:
                    record = item.get("record") or {}
                    text = (record.get("params") or {}).get("Description") or (record.get("fields") or {}).get("Description")
                    if text:
                        out[fid] = text
        _svo_descriptions[facility] = out
    return _svo_descriptions[facility].get(filter_id)


def problems(set_dir, manifest):
    from sed_tools.filters.curve_metadata import read_filter_header
    set_dir = Path(set_dir)
    published = manifest.get("published_bands") or {}
    assumed = (manifest.get("published") or {}).get("source") == "assumed"
    for band in manifest.get("bands") or []:
        curve = set_dir / f"{band}.dat"
        if not curve.is_file():
            continue
        header = read_filter_header(curve)
        roles = _roles(header.get("Components"))
        if assumed and roles and published.get(band) and _roles(published[band]) != roles:
            yield (f"{set_dir}/manifest.json: {band} published_bands {published[band]!r} (source assumed) disagrees "
                   f"with its header Components {header.get('Components')!r}")
        if roles and header.get("ComponentsSource") == "svo" and header.get("FilterID"):
            text = (_svo_description(header["FilterID"]) or "").lower().replace(" ", "")
            if "filteronly" in text and len(roles) > 1:
                yield f"{curve}: Components {header['Components']!r} but SVO's Description says filter only"
            if "filter+ccd" in text and roles == frozenset({"filter"}):
                yield f"{curve}: Components 'filter' but SVO's Description says filter+ccd"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", type=Path, metavar="BACKUP_DIR")
    args = ap.parse_args(argv)
    from sed_tools.filters.curve_metadata import read_filter_header, write_filter_header

    filters = args.data_dir / "filters"
    by_set = {}
    for entry in json.loads(CURATED.read_text()):
        by_set.setdefault(entry["set"], []).append(entry)
    provenance = {}
    for set_name, entries in sorted(by_set.items()):
        path = filters / set_name / "manifest.json"
        text = path.read_text(encoding="utf-8")
        manifest = json.loads(text)
        published = manifest.setdefault("published_bands", {})
        comp_filters = (manifest.get("components") or {}).get("filter")
        headers = []
        for e in entries:
            band, recipe = e["band"], e["recipe"]
            old = published.get(band)
            published[band] = recipe
            dropped = None
            if comp_filters is not None and comp_filters.get(band) == f"{band}.dat" and recipe != "filter":
                dropped = comp_filters.pop(band)
            curve = filters / set_name / f"{band}.dat"
            fields = read_filter_header(curve)
            if e.get("header"):
                if fields.get("Components") != recipe or fields.get("ComponentsSource") != e["components_source"]:
                    headers.append((curve, {**fields, "Components": recipe, "ComponentsSource": e["components_source"]},
                                    fields.get("Components")))
            print(f"{set_name}/{band}: published_bands {old!r} -> {recipe!r}"
                  + (f"; header {fields.get('Components')!r} -> {recipe!r} ({e['components_source']})" if e.get("header") else "")
                  + (f"; own-curve filter component {dropped!r} removed" if dropped else ""))
            fac, inst = set_name.split("/")
            provenance.setdefault(f"{fac}__{inst}", {})[band] = (
                f"Components {recipe} (was {old or 'none'} in the manifest"
                + (f", {fields.get('Components') or 'none'} in the header" if e.get("header") else "")
                + f"): {e['evidence']} (Nill, 25 Nov 2026, Q33).")
        if args.apply:
            target = args.apply / path.relative_to(args.data_dir.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            path.write_text(_dumps_like(text, manifest), encoding="utf-8")
            for curve, fields, _ in headers:
                target = args.apply / curve.relative_to(args.data_dir.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(curve, target)
                write_filter_header(curve, fields)
    for name, bands in sorted(provenance.items()):
        path = PROVENANCE / f"{name}.json"
        doc = json.loads(path.read_text()) if path.is_file() else {"facility": name.split("__")[0],
                                                                    "instrument": name.split("__")[1], "bands": {}}
        for band, note in bands.items():
            entry = doc.setdefault("bands", {}).setdefault(band, {})
            old = entry.get("cleaning")
            entry["cleaning"] = note if not old or note in old else f"{old} {note}"
        if args.apply:
            if path.is_file():
                target = args.apply / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
