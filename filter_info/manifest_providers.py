"""Every band a served filter manifest lists has a provider (Nill, 24 Nov 2026, decision Q17).

SED_Tools describes a set from its curves' headers by one rule (``InstrumentManifest.from_headers``): a band's
``published_bands`` entry is its curve's ``Components`` header, and a filter-only curve is its own filter component
(``<band>.dat``). NJM's manifests listed bands that had neither, chiefly SVO's filter-only siblings (LSST
``g_filter``, VISTA ``H_filter``), so SED_Tools' compose could not provide them ("cannot provide 'filter' for band
'g_filter'"). This adds what that rule gives and the manifest lacks, and never changes an entry it already has:

- a band without a ``published_bands`` entry gets its curve's ``Components`` header;
- in a manifest that declares components, a filter-only band that has no filter piece gets ``<band>.dat``.

``problems`` is the class guard filter_info.check_manifests runs: a manifest band whose curve says what it contains
but which has no ``published_bands`` entry, or a filter-only band of a component manifest that no filter piece covers,
is an error. (A curve whose header names no components is a labelling question, label_check's, not this one's.)

  python3 -m filter_info.manifest_providers DATA_DIR --dry-run
  python3 -m filter_info.manifest_providers DATA_DIR --apply BACKUP_DIR
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from sed_tools.filters.curve_metadata import read_filter_header
from sed_tools.filters.manifest import FILTER, parse_components, recipe_name


def _header_recipe(curve):
    """The recipe the curve's Components header names, or None (no header, or no known roles)."""
    try:
        roles = parse_components(read_filter_header(curve).get("Components"))
    except Exception:  # an unreadable or unlabelled header gives no provider; the guard reports it
        return None
    return recipe_name(roles) if roles else None


def _covers_filter(manifest, band):
    filters = (manifest.get("components") or {}).get(FILTER) or {}
    return band in filters or "*" in filters


def additions(set_dir, manifest):
    """{"published_bands": {band: recipe}, "filter": {band: file}} the rule adds to this manifest."""
    published = manifest.get("published_bands") or {}
    out = {"published_bands": {}, "filter": {}}
    for band in manifest.get("bands") or []:
        recipe = published.get(band) or _header_recipe(set_dir / f"{band}.dat")
        if band not in published and recipe:
            out["published_bands"][band] = recipe
        if manifest.get("components") and recipe == FILTER and not _covers_filter(manifest, band):
            out["filter"][band] = f"{band}.dat"
    return out


def problems(set_dir, manifest):
    """The guard: messages for manifest bands without a provider."""
    published = manifest.get("published_bands") or {}
    for band in manifest.get("bands") or []:
        if not published.get(band):
            if _header_recipe(Path(set_dir) / f"{band}.dat"):
                yield f"{set_dir}/manifest.json: band {band} has no published_bands entry (no provider)"
        elif manifest.get("components") and published[band] == FILTER and not _covers_filter(manifest, band):
            yield f"{set_dir}/manifest.json: filter-only band {band} has no filter component (no provider)"


def _dumps_like(text, doc):
    """``doc`` written the way ``text`` was (indent 2, ASCII-escaped or not), so only the added entries differ."""
    old = json.loads(text)
    for ensure_ascii in (False, True):
        if json.dumps(old, indent=2, ensure_ascii=ensure_ascii) + "\n" == text:
            return json.dumps(doc, indent=2, ensure_ascii=ensure_ascii) + "\n"
    raise ValueError("manifest is not in the indent-2 JSON layout; not rewritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", type=Path, metavar="BACKUP_DIR")
    args = ap.parse_args(argv)
    filters = args.data_dir / "filters"
    changed = 0
    totals = {"published_bands": 0, "filter": 0}
    unresolved = []
    for path in sorted(filters.glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        add = additions(path.parent, manifest)
        if any(add.values()):
            changed += 1
            for k in totals:
                totals[k] += len(add[k])
            print(f"{path.parent.relative_to(filters)}: published_bands +{sorted(add['published_bands'])}"
                  + (f"; filter +{sorted(add['filter'])}" if add["filter"] else ""))
            manifest.setdefault("published_bands", {}).update(add["published_bands"])
            if add["filter"]:
                manifest["components"].setdefault(FILTER, {}).update(add["filter"])
            if args.apply:
                target = args.apply / path.resolve().relative_to(args.data_dir.resolve().parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                path.write_text(_dumps_like(path.read_text(encoding="utf-8"), manifest), encoding="utf-8")
        unresolved += list(problems(path.parent, manifest))
    print(f"{'changed' if args.apply else 'would change'} {changed} manifests: {totals['published_bands']} "
          f"published_bands entries and {totals['filter']} filter components added")
    for message in unresolved:
        print("STILL WITHOUT PROVIDER:", message)
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
