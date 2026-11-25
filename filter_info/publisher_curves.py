"""Curves replaced by, or cut to, what their publisher states (Nill, 25 Nov 2026; evidence in the SED_Tools report
round3-research.md and in source_evidence/, recorded in filter_info/curated/publisher_curves.json).

Two kinds of entry:
- a band entry names the publisher's file for one band ("replace with the real i": LCO/e2v i, whose served copy, and
  SVO's record, held the r curve), or, with ``whole_file``, a curated curve file served as written, header included
  (the IRTF/MORIS curves re-digitised from Gulbis et al. 2011, Fig. 4);
- a set entry states one rule for every band of a set whose header Sources names a publisher file matching
  ``source_pattern`` and whose curve runs past ``detector_max_A``: when the publisher also gives the band's file with ``prefer_suffix`` (IRTF's "plus blocker"
  scans, used with the PK50 blocker), that file is served; every band is cut at ``detector_max_A``, the publisher's
  stated detector range (one interpolated sample at the cut); ``incomplete`` bands get a note saying what the
  publisher's file does not cover.
The publisher's files are read from source_evidence (wavelengths scaled to Angstrom by ``unit_A``). The header row is
kept, with Sources naming the file served; the change and its evidence go to the set's provenance record
(bands.<band>.cleaning), which info.json shows.

  python3 -m filter_info.publisher_curves DATA_DIR --dry-run
  python3 -m filter_info.publisher_curves DATA_DIR --apply BACKUP_DIR
"""
import argparse
import json
import re
import shutil
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CURATED = Path(__file__).with_name("curated") / "publisher_curves.json"
PROVENANCE = ROOT / "filter_expansion" / "provenance"


def _publisher_rows(path, unit_a):
    """(wavelength string in A, transmission string) per data line of a publisher's two-column file."""
    rows = []
    for line in Path(path).read_text().split("\n"):
        parts = line.split()
        if len(parts) < 2 or line.lstrip().startswith("#"):
            continue
        try:
            wl, t = Decimal(parts[0]) * Decimal(unit_a), Decimal(parts[1])
        except Exception:
            continue
        text = format(wl.normalize(), "f")
        rows.append((text if "." in text else text + ".0", parts[1]))
    return rows


def _cut(rows, max_a):
    """Rows up to max_a, with one interpolated row at max_a when the curve runs beyond it; and how many were removed."""
    lam = np.array([float(w) for w, _ in rows]); t = np.array([float(x) for _, x in rows])
    keep = [r for r, x in zip(rows, lam) if x <= max_a]
    removed = len(rows) - len(keep)
    if removed and float(keep[-1][0]) < max_a:
        keep.append((f"{float(max_a)!r}", f"{float(np.interp(max_a, lam, t))!r}"))
    return keep, removed


def _max_wavelength(curve):
    vals = []
    for line in Path(curve).read_text().split("\n")[1:]:
        try:
            vals.append(float(line.split(",")[0]))
        except ValueError:
            continue
    return max(vals) if vals else 0.0


def plan(data_dir, only=()):
    from sed_tools.filters.curve_metadata import read_filter_header
    filters = Path(data_dir) / "filters"
    out = []
    for entry in json.loads(CURATED.read_text()):
        if only and entry["set"] not in only:
            continue
        set_dir = filters / entry["set"]
        if "band" in entry and entry.get("whole_file"):  # one band, a curated curve file served as written, header too
            out.append((set_dir / f"{entry['band']}.dat", ROOT / entry["file"], entry["url"], entry["note"]))
            continue
        if "band" in entry:  # one band, the publisher's file
            rows = _publisher_rows(ROOT / entry["file"], entry.get("unit_A", 1))
            out.append((set_dir / f"{entry['band']}.dat", rows, entry["url"], entry["note"]))
            continue
        pattern = re.compile(entry["source_pattern"])
        for curve in sorted(set_dir.glob("*.dat")):
            m = pattern.search(read_filter_header(curve).get("Sources", ""))
            if not m:
                continue
            name = m.group("name")
            if _max_wavelength(curve) <= entry["detector_max_A"]:
                continue  # the class: scans that run past the detector's stated range
            served = name
            preferred = ROOT / entry["evidence_dir"] / entry["file_template"].format(name=name + entry["prefer_suffix"])
            if (name + entry["prefer_suffix"]) in entry["publisher_lists"] and preferred.is_file():
                served = name + entry["prefer_suffix"]
            path = ROOT / entry["evidence_dir"] / entry["file_template"].format(name=served)
            rows, removed = _cut(_publisher_rows(path, entry["unit_A"]), entry["detector_max_A"])
            note = []
            if served != name:
                note.append(entry["prefer_note"].format(file=path.name))
            if removed:
                note.append(entry["cut_note"].format(n=removed, max_um=entry["detector_max_A"] / 1e4))
            if curve.stem in entry.get("incomplete", {}):
                note.append(entry["incomplete"][curve.stem])
            url = entry["url_template"].format(file=path.name)
            out.append((curve, rows, url, " ".join(note) + f" ({entry['evidence']}; Nill, 25 Nov 2026)"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", type=Path, metavar="BACKUP_DIR")
    ap.add_argument("--set", nargs="+", default=(), help="apply only the entries of these sets")
    args = ap.parse_args(argv)
    from sed_tools.filters.curve_metadata import read_filter_header, write_filter_header
    from sed_tools.photometry import load_filter_curve, pivot_wavelength

    provenance = {}
    for curve, rows, url, note in plan(args.data_dir, args.set):
        head, _, body = curve.read_text().partition("\n")
        old = [l for l in body.split("\n") if l.strip()]
        whole = isinstance(rows, Path)
        if whole:  # the curated file's own header and rows
            head, _, wbody = rows.read_text().partition("\n")
            new = [l for l in wbody.split("\n") if l.strip()]
            rows = [tuple(l.split(",")[:2]) for l in new]
        else:
            new = [f"{w},{t}" for w, t in rows]
        c0 = load_filter_curve(str(curve))
        p0 = pivot_wavelength(c0.wavelength, c0.transmission, c0.detector_type)
        lam = np.array([float(w) for w, _ in rows]); tt = np.array([float(t) for _, t in rows])
        p1 = pivot_wavelength(lam, np.clip(tt, 0, None), c0.detector_type)
        print(f"{curve.parent.parent.name}/{curve.parent.name}/{curve.stem}: {len(old)} -> {len(new)} rows, "
              f"{lam[0]:g}-{lam[-1]:g} A, pivot {p0:.1f} -> {p1:.1f} A; {note}")
        fac, inst = curve.parent.parent.name, curve.parent.name
        provenance.setdefault(f"{fac}__{inst}", {})[curve.stem] = (note, url)
        if args.apply:
            target = args.apply / curve.relative_to(args.data_dir.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(curve, target)
            curve.write_text(head + "\n" + "\n".join(new) + "\n")
            fields = read_filter_header(curve)
            if not whole and url not in fields.get("Sources", ""):
                fields["Sources"] = " | ".join(x for x in (url, fields.get("Sources")) if x)
                write_filter_header(curve, fields)
    for name, bands in sorted(provenance.items()):
        path = PROVENANCE / f"{name}.json"
        doc = json.loads(path.read_text()) if path.is_file() else {"facility": name.split("__")[0],
                                                                    "instrument": name.split("__")[1], "bands": {}}
        for band, (note, url) in bands.items():
            entry = doc.setdefault("bands", {}).setdefault(band, {})
            old = entry.get("cleaning")
            entry["cleaning"] = note if not old or note in old else f"{old} {note}"
            src = entry.get("sources")
            entry["sources"] = url if not src or url in src else f"{src} | {url}"
        if args.apply:
            if path.is_file():
                target = args.apply / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
