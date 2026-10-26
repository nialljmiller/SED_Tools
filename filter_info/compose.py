"""Compose each instrument's info.json from committed inputs; no network access.

Inputs, lowest to highest precedence:

1. SVO's archived records (``filter_info/svo/<Facility>.json``), matched to each
   curve by its FilterID, its provenance ``svo_id``, or its band name when the
   SVO pivot wavelength agrees within 1 %;
2. NJM's own provenance: the curve's header row, the instrument's manifest.json
   and ``filter_expansion/provenance/<Facility>__<Instrument>.json``;
3. hand-checked files: ``filter_info/curated/instruments/<Facility>__<Instrument>.json``.

References are keys into ``filter_info/references.json`` (see references.py);
each info.json carries the resolved entries it uses as its ``bibliography``.
The curve header row stays authoritative for DetectorType: SVO's value is only
reported, under ``svo``.
"""

import csv
import hashlib
import json
import os
import re
from pathlib import Path

from sed_tools.filter_metadata import normalize_detector_type, read_filter_header

from .references import canonical, link, load_references, reference_keys
from .svo_archive import curve_digest

SCHEMA = 1
HERE = Path(__file__).resolve().parent
# SVO fields copied under their VOTable PARAM names.
SVO_FIELDS = ("Description", "Comments", "PhotSystem", "Band", "MagSys", "PhotCalID", "ZeroPointType")
BIBLIOGRAPHY_FIELDS = ("short", "type", "title", "authors", "year", "journal", "volume", "page",
                       "doi", "bibcode", "arxiv", "url", "publisher_url", "bibtex", "same_as")
PIVOT_TOLERANCE = 0.01


def _norm(name):
    """Band name for matching: lower case, no separators, ``2p00``/``2.00`` -> ``200``, prime -> p."""
    name = re.sub(r"(?<=\d)[p.](?=\d)", "", name.lower()).replace("prime", "p")
    return re.sub(r"[^a-z0-9]", "", name)


def _name_forms(*names):
    """A band name and its leading-token forms: ``H_CONICA`` -> {hconica, h}."""
    forms = set()
    for name in names:
        tokens = re.split(r"[_\-\s]+", name)
        forms.update(_norm("".join(tokens[:n])) for n in range(1, len(tokens) + 1))
    return forms - {""}


def _load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _samples(curve):
    rows = []
    for line in curve.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and not line.startswith("#"):
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return rows


class InfoSources:
    """Everything info.json is built from, loaded once per generate_index run."""

    def __init__(self, repo_root, svo_dir=None, curated_dir=None, references=None):
        root = Path(repo_root)
        # Parsed SVO records live on NJM beside their raw captures (svo_archive.py build).
        self.svo_dir = Path(svo_dir or os.environ.get("SED_SVO_RECORDS")
                            or root / "source_evidence" / "svo-archive" / "parsed")
        self.provenance_dir = root / "filter_expansion" / "provenance"
        self.curated_dir = Path(curated_dir or HERE / "curated" / "instruments")
        self.references = load_references() if references is None else references
        self.service = _load(self.svo_dir / "_service.json")
        self.renamed = {}
        rename_map = root / "filter_expansion" / "renamed_filters_20261025.csv"
        if rename_map.is_file():
            with rename_map.open() as f:
                for row in csv.DictReader(f):
                    self.renamed[row["new_path"][:-4]] = row["old_path"].rsplit("/", 1)[1][:-4]
        self._facilities = {}
        self.unresolved = set()
        files = sorted(self.svo_dir.glob("*.json")) if self.svo_dir.is_dir() else []
        self.svo_ids = sum(len(_load(p)) for p in files if not p.name.startswith("_"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
        self.svo_records = {"dir": str(self.svo_dir), "ids": self.svo_ids,
                            "sha256": digest.hexdigest() if files else None}

    def svo_facility(self, facility):
        if facility not in self._facilities:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", facility)
            self._facilities[facility] = _load(self.svo_dir / f"{safe}.json")
        return self._facilities[facility]

    def svo(self, filter_id):
        return self.svo_facility(filter_id.split("/", 1)[0]).get(filter_id) if filter_id else None

    def key(self, key):
        """Canonical form of a reference key; unknown keys are noted for resolve (URLs need none)."""
        if key not in self.references:
            if not key.startswith("url:"):
                self.unresolved.add(key)
            return key
        return canonical(key, self.references)


def _svo_pivot(record):
    for version in reversed(record.get("votable", [])):
        pivot = version["record"]["params"].get("WavelengthPivot")
        if pivot:
            return float(pivot)
    for version in reversed(record.get("browse", [])):
        pivot = version["record"]["calculated"].get("WavelengthPivot")
        if pivot:
            return float(pivot)
    return None


def match_svo(sources, facility, instrument, band, header, provenance_band, pivot):
    """(svo filterID, record, how) for a curve, or (None, None, None)."""
    for how, filter_id in (("FilterID", header.get("FilterID")),
                           ("provenance", (provenance_band or {}).get("svo_id"))):
        record = sources.svo(filter_id)
        if record:
            return filter_id, record, how
    if not pivot:
        return None, None, None
    names = _name_forms(band, sources.renamed.get(f"{facility}/{instrument}/{band}", band))
    groups = {f"{facility}/{instrument}"}
    if header.get("FilterID"):
        groups.add(header["FilterID"].rsplit(".", 1)[0])
    found = []
    for group in sorted(groups):
        for filter_id, record in sources.svo_facility(group.split("/", 1)[0]).items():
            prefix, _, svo_band = filter_id.rpartition(".")
            svo_pivot = _svo_pivot(record)
            if (prefix == group and _norm(svo_band) in names and svo_pivot
                    and abs(svo_pivot - pivot) <= PIVOT_TOLERANCE * pivot):
                found.append((filter_id, record))
    if len(found) == 1:  # ambiguous name matches are left unmatched
        return found[0][0], found[0][1], "band name and pivot"
    return None, None, None


def _svo_entry(sources, filter_id, record, how, digest, header):
    """SVO's fields for one band, from the version whose curve matches NJM's when there is one."""
    votables = record.get("votable", [])
    same = [v for v in votables if v["record"]["curve_sha256"] == digest]
    votable = (same or votables or [None])[-1]
    browse = (record.get("browse") or [None])[-1]
    params = votable["record"]["params"] if votable else {}
    page = browse["record"] if browse else {}
    fields = {k: params[k] for k in SVO_FIELDS if params.get(k)}
    for k, v in page.get("fields", {}).items():
        if k in SVO_FIELDS and v and k not in fields:
            fields[k] = v
    refs = {}
    for name in ("ProfileReference", "CalibrationReference"):
        labelled = page.get("references", {}).get(name, {})
        texts = [params.get(name, ""), labelled.get("url", "")]
        keys = [sources.key(k) for text in texts for k in reference_keys(text)]
        if keys:
            refs[name] = [{"ref": k, **({"label": labelled["label"]} if labelled.get("label") else {})}
                          for k in dict.fromkeys(keys)]
    specified = dict(page.get("specified", {}))
    zero_points = {system: zp["specified"] for system, zp in page.get("zero_points", {}).items()
                   if zp.get("specified")}
    if zero_points:
        specified["ZeroPoint"] = zero_points
    from_votable = bool(params.get("DetectorType"))
    svo_detector = normalize_detector_type(params.get("DetectorType") or page.get("fields", {}).get("DetectorType"))
    njm_detector = normalize_detector_type(header.get("DetectorType"))
    svo = {"filterID": filter_id, "matched_by": how,
           "curve": "identical" if same else ("differs" if votables else "not harvested")}
    for k in ("components", "added", "updated"):
        value = params.get(k) or page.get("fields", {}).get(k)
        if value:
            svo[k] = value
    if svo_detector:
        svo["DetectorType"] = svo_detector
        if njm_detector and svo_detector != njm_detector:
            stamp = (votable if from_votable else browse)["captures"][-1]["timestamp"]
            said = f"SVO's record captured {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} says {svo_detector}"
            svo["DetectorType_note"] = (
                f"{said}; NJM's {njm_detector} was also taken from SVO, at another time, so SVO changed its classification"
                if header.get("DetectorTypeSource") == "svo" else
                f"{said}; NJM's curve header says {njm_detector} ({header.get('DetectorTypeSource', 'unspecified')}) and is authoritative")
    for k in ("Fsun", "AfAv"):
        if page.get("calculated", {}).get(k) is not None:
            svo[k] = page["calculated"][k]
    svo["captures"] = {kind: v["captures"][-1] for kind, v in (("votable", votable), ("browse", browse)) if v}
    return fields, refs, specified, svo, page


def _curve_sources(sources, header, manifest_sources, provenance_band):
    keys = []
    for text in (header.get("Sources", ""), (provenance_band or {}).get("sources", ""), *manifest_sources):
        keys += [sources.key(k) for k in reference_keys(text)]
    return list(dict.fromkeys(keys))


def compose_info(sources, instrument_dir, curves, properties=None):
    """The info.json document for one instrument directory."""
    instrument_dir = Path(instrument_dir)
    facility, instrument = instrument_dir.parent.name, instrument_dir.name
    provenance = _load(sources.provenance_dir / f"{facility}__{instrument}.json")
    curated = _load(sources.curated_dir / f"{facility}__{instrument}.json")
    manifest = _load(instrument_dir / "manifest.json")
    manifest_sources = [v for v in (manifest.get("sources") or {}).values() if isinstance(v, str)]
    info = {"schema": SCHEMA, "facility": facility, "instrument": instrument}
    bands, unmatched, svo_groups, svo_curves = {}, [], {}, False
    for curve in sorted(curves):
        band = curve.stem
        header = read_filter_header(curve)
        prov_band = (provenance.get("bands") or {}).get(band)
        pivot = ((properties or {}).get(band) or {}).get("WavelengthPivot")
        filter_id, record, how = match_svo(sources, facility, instrument, band, header, prov_band, pivot)
        entry = {}
        if record:
            fields, refs, specified, svo, page = _svo_entry(
                sources, filter_id, record, how, curve_digest(_samples(curve)), header)
            entry.update(fields)
            entry.update(refs)
            if specified:
                entry["specified"] = specified
            entry["svo"] = svo
            group = svo_groups.setdefault(filter_id.rsplit(".", 1)[0], {})
            for key, field in (("instrument", "description"), ("facility", "facility_description")):
                if page.get(key, {}).get("description"):
                    group.setdefault(field, page[key]["description"])
                for item in page.get(key, {}).get("links", []):
                    if item["url"] not in [l["url"] for l in group.setdefault("links", [])]:
                        group["links"].append({**item, "source": "svo"})
            svo_curves |= svo["curve"] == "identical"
        else:
            unmatched.append(band)
        curve_info = {k: header[k] for k in ("FilterID", "DetectorType", "DetectorTypeSource",
                                              "Components", "ComponentsSource") if header.get(k)}
        curve_sources = _curve_sources(sources, header, manifest_sources, prov_band)
        if curve_sources:
            curve_info["sources"] = curve_sources
        if prov_band and prov_band.get("cleaning"):
            curve_info["cleaning"] = prov_band["cleaning"]
        entry["curve"] = curve_info
        svo_curves |= header.get("ComponentsSource") == "svo"
        entry.update((curated.get("bands") or {}).get(band, {}))
        # To cite: SVO's references (papers and documentation pages), the papers among the
        # curve's sources, and curated ones. Data-file URLs stay under curve.sources only.
        cited = [r["ref"] for name in ("CalibrationReference", "ProfileReference") for r in entry.get(name, [])]
        cited += [k for k in curve_sources if not k.startswith("url:")]
        cited += [sources.key(k) for k in entry.get("references", [])]
        entry["references"] = list(dict.fromkeys(cited))
        bands[band] = entry
    # Instrument level: SVO's descriptions, then provenance, then curated. SVO describes its own
    # groups; one stands for the instrument only when it is the instrument's own path, or every
    # band matched it (NJM's GAIA/GAIA mixes SVO's GAIA/GAIA3 and GAIA/Gaia2m, for example).
    own = f"{facility}/{instrument}"
    whole = own if own in svo_groups else (
        next(iter(svo_groups)) if len(svo_groups) == 1 and not unmatched else None)
    if whole:
        info.update(svo_groups[whole])
    if svo_groups:
        info["svo_groups"] = svo_groups
    notes = [n for n in (provenance.get("notes"), manifest.get("notes")) if n]
    if notes:
        info["notes"] = " ".join(dict.fromkeys(notes))
    for k in ("description", "facility_description", "notes"):
        if curated.get(k):
            info[k] = curated[k]
    if curated.get("links"):
        info["links"] = curated["links"] + [l for l in info.get("links", [])
                                            if l["url"] not in {c["url"] for c in curated["links"]}]
    instrument_refs = [sources.key(k) for k in curated.get("references", [])]
    for entry in bands.values():
        instrument_refs += entry["references"]
    if (svo_groups or svo_curves) and sources.service:
        # SVO asks to be acknowledged; its papers are citations only when the curves came from SVO.
        svo_refs = [sources.key(f"bibcode:{b}") for b in sources.service.get("references", [])]
        info["acknowledgements"] = [{"text": sources.service["acknowledgement"], "for": "SVO Filter Profile Service",
                                     "applies_to": "curves and metadata" if svo_curves else "metadata",
                                     "references": svo_refs}]
        if svo_curves:
            instrument_refs += svo_refs
    info["references"] = list(dict.fromkeys(instrument_refs))
    info["bands"] = bands
    if unmatched:
        info["svo_unmatched"] = unmatched
    labels = {r["ref"]: r["label"] for entry in bands.values()
              for name in ("CalibrationReference", "ProfileReference") for r in entry.get(name, []) if "label" in r}
    info["bibliography"] = {}
    acknowledged = [k for a in info.get("acknowledgements", []) for k in a["references"]]
    for key in dict.fromkeys(info["references"] + acknowledged):
        kind, _, value = key.partition(":")
        entry = sources.references.get(key) or (
            {"type": "web", "url": value} if kind == "url" else {"url": link(key), "resolved": False})
        entry = {k: v for k, v in entry.items() if k in BIBLIOGRAPHY_FIELDS or k == "resolved"}
        if "short" not in entry and key in labels:
            entry["short"] = labels[key]
        info["bibliography"][key] = entry
    return info


def write_info(sources, instrument_dir, curves, properties=None):
    """Write the instrument's info.json when its content changed; True when written or current."""
    if not curves:
        return False
    text = json.dumps(compose_info(sources, instrument_dir, curves, properties),
                      indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    path = Path(instrument_dir) / "info.json"
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)
    return True
