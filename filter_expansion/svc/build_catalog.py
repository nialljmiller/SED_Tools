#!/usr/bin/env python3
"""Build catalog.json for the NJM SVO-compatible Filter Profile Service.

Reads the published NJM curves (data/filters/<Facility>/<Instrument>/*.dat,
the same set generate_index.py publishes), their column-row metadata and each
instrument's properties.json, and writes one compact JSON catalog that
fps.php and getdata.php serve.  Standard library only; re-runnable: run it
again after new curves are added and generate_index.py has rewritten the
properties.json files.

    python3 filter_expansion/svc/build_catalog.py     # in the SED_Tools checkout:
        # reads data/filters, filter_expansion/svo_aliases.json; writes data/svo_fps/catalog.json
    python3 build_catalog.py --data /media/data3/MESA/SED_Tools/data --out DIR [--aliases FILE]

Run it again after new curves are deployed (and generate_index.py has
rewritten properties.json); the swap of catalog.json is atomic.

SVO-form IDs: ?ID= resolves, besides each curve's FilterID, (1) the NJM path
ID Facility/Instrument.band, (2) every info.json bands.<b>.svo.filterID and
bands.<b>.svo_filterID, and (3) the "aliases" of svo_aliases.json (SVO ID ->
NJM Facility/Instrument/band, built by filter_expansion/phase4/tools/
build_svo_aliases.py).  A FilterID always wins over an alias.

Copies: <band>_svo (SVO's copy of a curve NJM holds from the original
source) and <band>_old (a superseded curve) are served under their own NJM
path ID Facility/Instrument.<band>_svo|_old, never under the FilterID header
they share with the plain-named curve, so SVO-form IDs reach NJM's current
curve.  The same applies to any curve whose FilterID header is another
curve's NJM path ID (BOK/90prime.g carries BOK/BASS.g, the path ID of
BOK/BASS/g).  The displaced header ID stays an alias of last resort.

Nothing is computed from the curves here; every property comes from
properties.json (sed_tools.filter_properties).  Missing values stay null.

The instrument's info.json (schema 1, written by generate_index.py; see
filter_info/README.md in the SED_Tools server checkout) supplies, at
bands.<band> under SVO PARAM names:
  Description, Comments, PhotSystem, Band, MagSys, PhotCalID, ZeroPointType
  ProfileReference, CalibrationReference   lists of {ref, label}, rendered as
      one text value "label (url); ..." with the URL taken from
      bibliography.<ref> (url, else doi, else bibcode) or from the ref key
  specified{...}   owner-published values; they replace the properties.json
      value for the SVO PARAMs listed in SPECIFIED_PARAMS, and
      specified.ZeroPoint.<Vega|AB|ST>.Jy replaces that zero point.
Its top-level acknowledgements (text + references) credit SVO in Comments of
SVO-derived records and in the catalog's "acknowledgements" (fps.php prints
them as INFO elements).  When info.json lacks a field the defaults below are
kept; they are never presented as SVO values.
"""
import argparse
import datetime
import json
import math
import os
import sys
import tempfile
from urllib.parse import unquote

# Catalog columns, in SVO FPS order (2022 FIELD list), then NJM extras.
FIELDS = [
    "FilterProfileService", "filterID", "WavelengthUnit", "WavelengthUCD", "PhotSystem",
    "DetectorType", "Band", "Instrument", "Facility", "ProfileReference",
    "CalibrationReference", "Description", "Comments", "WavelengthRef", "WavelengthMean",
    "WavelengthEff", "WavelengthMin", "WavelengthMax", "WidthEff", "WavelengthCen",
    "WavelengthPivot", "WavelengthPeak", "WavelengthPhot", "FWHM", "Fsun",
    # Zero points kept per system so PhotCalID=<id>/Vega|AB|ST can be answered.
    "ZeroPointVega", "ZeroPointAB", "ZeroPointST",
    # info.json's own PhotCal (its MagSys, PhotCalID, ZeroPointType); fps.php
    # uses them for the matching magnitude system only.
    "MagSys", "PhotCalID", "ZeroPointType",
    # NJM extras (not in SVO).
    "Components", "ComponentsSource", "DetectorTypeSource", "NJMID", "SpecifiedParams",
]
PROPS = ["WavelengthMean", "WavelengthEff", "WavelengthMin", "WavelengthMax", "WidthEff",
         "WavelengthCen", "WavelengthPivot", "WavelengthPeak", "WavelengthPhot", "FWHM"]
RANGE_PARAMS = ["WavelengthRef", "WavelengthMean", "WavelengthEff", "WavelengthMin",
                "WavelengthMax", "WidthEff", "FWHM"]
# Fallback credit, used only when an SVO-derived record has no info.json
# acknowledgements (otherwise those are used verbatim).
SVO_CREDIT = ("SVO-derived; credit: SVO Filter Profile Service "
              "(Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024)")
FPS_IVORN = "ivo://nillmill.ddns.net/sed_tools/fps"
# Catalog columns that info.json may supply (SVO VOTable PARAM names).
INFO_FIELDS = ["PhotSystem", "Band", "Instrument", "Facility", "Description", "Comments",
               "MagSys", "PhotCalID", "ZeroPointType"]
REF_FIELDS = ["ProfileReference", "CalibrationReference"]
# SVO VOTable PARAMs (archived SVO FPS VOTables, 2024-2026) that
# bands.<band>.specified may set; all in Angstrom except Fsun (erg/cm2/s/A).
SPECIFIED_PARAMS = ["WavelengthRef", "WavelengthMean", "WavelengthEff", "WavelengthMin",
                    "WavelengthMax", "WidthEff", "WavelengthCen", "WavelengthPivot",
                    "WavelengthPeak", "WavelengthPhot", "FWHM", "Fsun"]
MAGSYS = {"vega": "Vega", "ab": "AB", "st": "ST"}
COPY_SUFFIXES = ("_svo", "_old")


def read_header(path):
    """Metadata dict from the column row (``Wavelength,Transmission,K=V;...``)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
    parts = first.split(",", 2)
    meta = {}
    if len(parts) == 3:
        for item in parts[2].split(";"):
            key, sep, value = item.partition("=")
            if sep and key.replace("_", "").isalnum():
                meta[key] = unquote(value).strip()
    return meta


def split_id(filter_id):
    """SVO ID ``Facility/Instrument.Band`` -> (facility, instrument, band)."""
    facility, sep, rest = filter_id.partition("/")
    if not sep:
        return "", "", filter_id
    instrument, dot, band = rest.partition(".")
    return (facility, instrument, band) if dot else (facility, "", rest)


def info_entry(doc, band, filter_id):
    """This band's info.json record: under "bands" by band name or "filters" by filterID."""
    for key, name in (("bands", band), ("filters", filter_id)):
        entry = (doc.get(key) or {}).get(name)
        if isinstance(entry, dict):
            return entry
    return {}


def info_value(entry, field):
    """Field value from an info.json record; a dict value carries it under "value"."""
    value = entry.get(field)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    return value.strip() if isinstance(value, str) else ""


def ref_url(ref, bib):
    """Resolvable URL for a reference key: bibliography url, doi, bibcode, else the key."""
    entry = bib.get(ref) if isinstance(bib.get(ref), dict) else {}
    if entry.get("url"):
        return str(entry["url"])
    if entry.get("doi"):
        return "https://doi.org/" + str(entry["doi"])
    if entry.get("bibcode"):
        return "https://ui.adsabs.harvard.edu/abs/" + str(entry["bibcode"])
    kind, _, value = ref.partition(":")
    return {"url": value, "doi": "https://doi.org/" + value,
            "bibcode": "https://ui.adsabs.harvard.edu/abs/" + value,
            "arxiv": "https://arxiv.org/abs/" + value}.get(kind, "") if value else ""


def ref_label(ref, bib, label=""):
    entry = bib.get(ref) if isinstance(bib.get(ref), dict) else {}
    return (label or entry.get("short") or ref.partition(":")[2] or ref).strip()


def ref_text(items, bib):
    """SVO-style text value for a [{ref, label}] list: "label (url); ..."."""
    if isinstance(items, (str, dict)):
        items = [items]
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            item = {"ref": item}
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        label = ref_label(ref, bib, str(item.get("label") or ""))
        url = ref_url(ref, bib) if ref else ""
        text = label if not url or url == label else (f"{label} ({url})" if label else url)
        if text and text not in out:
            out.append(text)
    return "; ".join(out)


def ack_list(doc, bib):
    """info.json acknowledgements -> [{for, applies_to, text, references[{ref,label,url}], citation}]."""
    out = []
    for ack in doc.get("acknowledgements") or []:
        if not isinstance(ack, dict) or not str(ack.get("text") or "").strip():
            continue
        refs = [{"ref": r, "label": ref_label(r, bib), "url": ref_url(r, bib)}
                for r in ack.get("references") or [] if isinstance(r, str)]
        text = str(ack["text"]).strip()
        cites = ref_text([{"ref": r["ref"]} for r in refs], bib)
        out.append({"for": str(ack.get("for") or ""), "applies_to": str(ack.get("applies_to") or ""),
                    "text": text, "references": refs,
                    "citation": f"{text}. References: {cites}" if cites else text})
    return out


def acks_for(acks, svo_curve, svo_meta):
    """Acknowledgements that apply to one record (by applies_to: curves and/or metadata)."""
    hits = [a for a in acks
            if (svo_curve and ("curve" in a["applies_to"] or not a["applies_to"]))
            or (svo_meta and ("metadata" in a["applies_to"] or not a["applies_to"]))]
    if svo_curve and not hits:  # SVO curve, but no ack scoped to curves: SVO's own ack still applies
        hits = [a for a in acks if "svo" in a["for"].lower()]
    return hits


def is_copy(band):
    """``<band>_svo`` (SVO's copy) and ``<band>_old`` (superseded curve)."""
    return band.endswith(COPY_SUFFIXES)


def path_ids(filters_dir):
    """NJM path IDs ``Facility/Instrument.band`` of every published curve."""
    ids = set()
    for facility in os.listdir(filters_dir):
        fdir = os.path.join(filters_dir, facility)
        for instrument in os.listdir(fdir) if os.path.isdir(fdir) else []:
            idir = os.path.join(fdir, instrument)
            for name in os.listdir(idir) if os.path.isdir(idir) else []:
                if name.endswith(".dat") and os.path.isfile(os.path.join(idir, name)):
                    ids.add(f"{facility}/{instrument}.{name[:-4]}")
    return ids


def same_curve(path_a, path_b):
    """True when two curve files hold the same table (column rows may differ)."""
    try:
        with open(path_a, "rb") as a, open(path_b, "rb") as b:
            a.readline(), b.readline()
            return a.read() == b.read()
    except OSError:
        return False


def copy_note(filters_dir, facility, instrument, band, header_id, njm_ids):
    """Comments text for a curve served under its path ID instead of its FilterID header."""
    plain = f"{facility}/{instrument}.{band[:-4]}"
    if band.endswith("_svo"):
        lead = (f"SVO FPS copy of NJM curve {plain}" if plain in njm_ids
                else "SVO FPS copy")
        return f"{lead}; SVO serves it as {header_id}"
    if band.endswith("_old"):
        lead = (f"Superseded curve; the current one is {plain}" if plain in njm_ids
                else "Superseded curve")
        return f"{lead}; FilterID header {header_id}"
    fac, ins, other = split_id(header_id)
    if same_curve(os.path.join(filters_dir, facility, instrument, band + ".dat"),
                  os.path.join(filters_dir, fac, ins, other + ".dat")):
        return f"Same curve as {header_id}"
    return f"FilterID header {header_id} is another NJM curve's ID"


def num(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def build(data_dir):
    filters_dir = os.path.join(data_dir, "filters")
    rows, files, warnings, svo_ids, header_ids = [], [], [], [], []
    njm_ids = path_ids(filters_dir)
    vega_info, module = None, None
    all_acks, ack_seen = [], {}
    for facility in sorted(os.listdir(filters_dir)):
        fdir = os.path.join(filters_dir, facility)
        if not os.path.isdir(fdir):
            continue
        for instrument in sorted(os.listdir(fdir)):
            idir = os.path.join(fdir, instrument)
            if not os.path.isdir(idir):
                continue
            curves = sorted(n for n in os.listdir(idir)
                            if n.endswith(".dat") and os.path.isfile(os.path.join(idir, n)))
            if not curves:
                continue
            props_path = os.path.join(idir, "properties.json")
            bands = {}
            if os.path.isfile(props_path):
                with open(props_path, encoding="utf-8") as handle:
                    doc = json.load(handle)
                bands = doc.get("bands", {})
                defs = doc.get("definitions") or {}
                vega_info = vega_info or defs.get("vega_spectrum")
                module = module or defs.get("module")
            else:
                warnings.append(f"no properties.json: {facility}/{instrument}")
            info_path = os.path.join(idir, "info.json")
            info = {}
            if os.path.isfile(info_path):
                try:
                    with open(info_path, encoding="utf-8") as handle:
                        info = json.load(handle)
                except (OSError, ValueError) as exc:
                    warnings.append(f"unreadable info.json: {facility}/{instrument}: {exc}")
                if not isinstance(info, dict):
                    info = {}
            bib = info.get("bibliography") if isinstance(info.get("bibliography"), dict) else {}
            acks = ack_list(info, bib)
            for ack in acks:
                key = (ack["for"], ack["text"])
                if key not in ack_seen:
                    ack_seen[key] = len(all_acks)
                    all_acks.append({k: ack[k] for k in ("for", "text", "references", "citation")})
            for name in curves:
                band = name[:-4]
                meta = read_header(os.path.join(idir, name))
                njm_id = f"{facility}/{instrument}.{band}"
                header_id = meta.get("FilterID") or njm_id
                fid, note = header_id, ""
                if header_id != njm_id and (is_copy(band) or header_id in njm_ids):
                    fid = njm_id
                    note = copy_note(filters_dir, facility, instrument, band, header_id, njm_ids)
                s_fac, s_ins, s_band = split_id(fid)
                p = bands.get(band)
                if p is None:
                    warnings.append(f"no properties for {njm_id}")
                    p = {}
                elif "error" in p:
                    warnings.append(f"properties error for {njm_id}: {p['error']}")
                det = {"photon": "1", "energy": "0"}.get(meta.get("DetectorType", "").lower(), "")
                dts = meta.get("DetectorTypeSource", "")
                cs = meta.get("ComponentsSource", "")
                svo_derived = dts in {"svo", "svo-archive"} or cs in {"svo", "svo-archive"}
                rec = {
                    "FilterProfileService": FPS_IVORN,
                    "filterID": fid,
                    # NJM curves are stored in Angstrom (format contract); the
                    # 58 files without a WavelengthUnit key are Angstrom too.
                    "WavelengthUnit": "Angstrom",
                    "WavelengthUCD": "em.wl",
                    "PhotSystem": "",
                    "DetectorType": det,
                    "Band": s_band,
                    "Instrument": s_ins,
                    "Facility": s_fac,
                    "ProfileReference": meta.get("Sources", ""),
                    "CalibrationReference": "",
                    "Description": f"{s_ins} {s_band}".strip(),
                    "Comments": "",  # set below: info.json Comments + SVO credit
                    "WavelengthRef": num(p.get("WavelengthPivot")),  # SVO: Ref == pivot
                    "Fsun": None,
                    "ZeroPointVega": num(p.get("ZeroPointVega_Jy")),
                    "ZeroPointAB": num(p.get("ZeroPointAB_Jy")),
                    "ZeroPointST": num(p.get("ZeroPointST_Jy")),
                    "Components": meta.get("Components", ""),
                    "ComponentsSource": cs,
                    "DetectorTypeSource": dts,
                    "NJMID": njm_id,
                    "MagSys": "",
                    "PhotCalID": "",
                    "ZeroPointType": "",
                    "SpecifiedParams": "",
                }
                for key in PROPS:
                    rec[key] = num(p.get(key))
                entry = info_entry(info, band, header_id)
                for key in INFO_FIELDS:
                    value = info_value(entry, key)
                    if value:
                        rec[key] = value
                for key in REF_FIELDS:
                    value = ref_text(entry.get(key), bib) if entry.get(key) else ""
                    if value:
                        rec[key] = value
                if rec["MagSys"]:
                    rec["MagSys"] = MAGSYS.get(rec["MagSys"].lower(), "")
                    if not rec["MagSys"]:
                        warnings.append(f"unknown MagSys in info.json for {njm_id}: "
                                        f"{info_value(entry, 'MagSys')}")
                        rec["PhotCalID"] = rec["ZeroPointType"] = ""
                if note:   # the shared PhotCalID belongs to the plain-named curve
                    rec["PhotCalID"] = ""
                spec = entry.get("specified") if isinstance(entry.get("specified"), dict) else {}
                used = []
                for key in SPECIFIED_PARAMS:
                    value = spec.get(key)
                    value = num(value.get("value") if isinstance(value, dict) else value)
                    if value is not None:
                        rec[key] = value
                        used.append(key)
                zps = spec.get("ZeroPoint") if isinstance(spec.get("ZeroPoint"), dict) else {}
                for system, value in zps.items():
                    system = MAGSYS.get(str(system).lower())
                    jy = num(value.get("Jy")) if isinstance(value, dict) else None
                    if system and jy is not None:   # SVO publishes ZeroPoint in Jy only
                        rec["ZeroPoint" + system] = jy
                        used.append("ZeroPoint." + system)
                rec["SpecifiedParams"] = ",".join(used)
                svo_meta = isinstance(entry.get("svo"), dict)
                credit = [a["citation"] for a in acks_for(acks, svo_derived, svo_meta)]
                if svo_derived and not credit:
                    credit = [SVO_CREDIT]
                rec["Comments"] = "; ".join(t for t in [note, info_value(entry, "Comments")] + credit if t)
                bad = [k for k in FIELDS if isinstance(rec[k], str) and "\ufffd" in rec[k]]
                if bad:   # lossy decode upstream (e.g. ISO-8859-1 page read as UTF-8): fix the source
                    warnings.append(f"U+FFFD replacement character in {njm_id}: {', '.join(bad)}")
                rows.append([rec[k] for k in FIELDS])
                files.append(f"{facility}/{instrument}/{name}")
                svo = entry.get("svo") if isinstance(entry.get("svo"), dict) else {}
                svo_ids.append([v for v in (svo.get("filterID"), entry.get("svo_filterID"))
                                if isinstance(v, str) and v.strip()])
                header_ids.append(header_id if header_id != fid else "")
    return rows, files, warnings, vega_info, module, all_acks, svo_ids, header_ids


def default_paths():
    """(data, out, aliases) for this file in filter_expansion/svc/ or in data/svo_fps/."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(os.path.dirname(here), "filters")):      # data/svo_fps/
        data = os.path.dirname(here)
        return data, here, os.path.join(os.path.dirname(data), "filter_expansion", "svo_aliases.json")
    repo = os.path.dirname(os.path.dirname(here))                          # filter_expansion/svc/
    data = os.path.join(repo, "data")
    return data, os.path.join(data, "svo_fps"), os.path.join(repo, "filter_expansion", "svo_aliases.json")


def main():
    d_data, d_out, d_alias = default_paths()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=d_data, help=f"NJM data dir (has filters/); default {d_data}")
    ap.add_argument("--out", default=d_out, help=f"output dir for catalog.json; default {d_out}")
    ap.add_argument("--aliases", default=d_alias, help=f"svo_aliases.json; default {d_alias} (skipped if absent)")
    args = ap.parse_args()
    rows, files, warnings, vega_info, module, acks, svo_ids, header_ids = build(args.data)
    ids = {}
    for i, row in enumerate(rows):
        fid = row[FIELDS.index("filterID")]
        if fid in ids:
            warnings.append(f"duplicate filterID {fid}: {files[ids[fid]]} and {files[i]}")
            continue
        ids[fid] = i
    aliases = {}
    for i, row in enumerate(rows):  # NJM path IDs that differ from the FilterID
        njm = row[FIELDS.index("NJMID")]
        if njm != row[FIELDS.index("filterID")] and njm not in ids:
            aliases[njm] = i
    n_path = len(aliases)
    for i, sids in enumerate(svo_ids):   # info.json SVO matches
        for sid in sids:
            if sid not in ids and sid not in aliases:
                aliases[sid] = i
    n_info = len(aliases) - n_path
    by_njm = {row[FIELDS.index("NJMID")]: i for i, row in enumerate(rows)}
    alias_source = None
    if args.aliases and os.path.isfile(args.aliases):
        with open(args.aliases, encoding="utf-8") as handle:
            amap = json.load(handle)
        alias_source = {"file": os.path.abspath(args.aliases), "generated": amap.get("generated")}
        for sid, rec in sorted((amap.get("aliases") or {}).items()):
            fac, _, rest = str(rec.get("njm", "")).partition("/")
            inst, _, band = rest.partition("/")
            i = by_njm.get(f"{fac}/{inst}.{band}")
            if i is None:
                warnings.append(f"svo_aliases.json: {sid} -> {rec.get('njm')} is not a published curve")
            elif sid not in ids and sid not in aliases:
                aliases[sid] = i
    elif args.aliases:
        warnings.append(f"alias map not found: {args.aliases}")
    n_map = len(aliases) - n_path - n_info
    for i, hid in enumerate(header_ids):   # FilterID headers displaced from copies
        if hid and hid not in ids and hid not in aliases:
            aliases[hid] = i
    n_header = len(aliases) - n_path - n_info - n_map
    col = {k: FIELDS.index(k) for k in FIELDS}
    photcal = {}  # info.json PhotCalIDs that differ from <filterID>/<MagSys>
    for i, row in enumerate(rows):
        pc, ms = row[col["PhotCalID"]], row[col["MagSys"]]
        if pc and ms and pc != f"{row[col['filterID']]}/{ms}":
            if pc.rpartition("/")[0] in ids:   # another curve's own <filterID>/<MagSys>
                continue
            if pc in photcal:
                warnings.append(f"duplicate PhotCalID {pc}: {files[photcal[pc]]} and {files[i]}")
                continue
            photcal[pc] = i
    ranges = {}
    for key in RANGE_PARAMS:
        vals = [r[col[key]] for r in rows if r[col[key]] is not None]
        ranges[key] = [min(vals), max(vals)] if vals else [None, None]
    catalog = {
        "service": "NJM SVO-compatible Filter Profile Service",
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "properties_module": module,
        "vega_spectrum": vega_info,
        "count": len(rows),
        "fields": FIELDS,
        "rows": rows,
        "files": files,
        "aliases": aliases,
        "alias_counts": {"njm_path": n_path, "info_json_svo": n_info, "svo_aliases_json": n_map,
                         "copy_header": n_header},
        "alias_source": alias_source,
        "photcal": photcal,
        "acknowledgements": acks,
        "ranges": ranges,
        "facilities": sorted({r[col["Facility"]] for r in rows if r[col["Facility"]]}),
        "instruments": sorted({r[col["Instrument"]] for r in rows if r[col["Instrument"]]}),
        "warnings": warnings,
    }
    os.makedirs(args.out, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".catalog.", suffix=".json", dir=args.out)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    os.chmod(tmp, 0o644)
    os.replace(tmp, os.path.join(args.out, "catalog.json"))  # atomic swap for live readers
    print(f"catalog.json: {len(rows)} filters, {len(aliases)} alias IDs ({n_path} NJM path, "
          f"{n_info} info.json SVO, {n_map} svo_aliases.json, {n_header} copy header), "
          f"{len(catalog['facilities'])} facilities, {len(warnings)} warnings -> {args.out}")
    for w in warnings[:20]:
        print("  warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
