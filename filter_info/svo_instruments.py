#!/usr/bin/env python3
"""Which SVO instruments each NJM instrument duplicates: ``filter_info/svo_instruments.json``.

An SVO instrument (a ``Facility/Instrument`` group of SVO's current filter list) is
the same product as an NJM instrument when every one of its filter IDs is held in
that NJM instrument with the same curve.  NJM may hold more bands than SVO.

- **Candidate NJM band for an SVO ID:** the band's SVO match in info.json, the
  curve's ``FilterID`` header, ``filter_expansion/svo_aliases.json``, or the same
  path (``F/I.b`` -> ``F/I/b``).  An ID none of these links is searched for
  among the curves of the NJM instruments holding the group's other IDs, using
  only bands no other ID of the group holds.  The alias map's "related" and
  "rejected_candidates" pairs and its "not_instrument_aliases" never count.
- **Same curve:** SVO's newest archived VOTable and the NJM curve agree to the
  alias map's "identical" thresholds (peak-normalised overlap >= 0.995, pivot
  within 0.2 %), compared as published or after converting both to photon
  response with their own DetectorType (SVO multiplies photon-counting curves
  by lambda, e.g. GAIA/GAIA2).

No network.  Run on NJM from the server checkout:

    python3 -m filter_info.svo_instruments SVO_LIST_DIR OUT.json

SVO_LIST_DIR holds SVO's filter list queries (``raw/*fps_list*.xml``), e.g.
source_evidence/svo-archive/svo-live-2026-10-27-b00-lists; its wavelength-range
queries list every filter, including facilities it has no per-facility list for.
"""

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sed_tools.filter_metadata import normalize_detector_type, read_filter_header

from .compose import _samples
from .svo_archive import decode, parse_votable

OVERLAP_SAME, PIVOT_SAME = 0.995, 0.002  # svo_aliases.json "identical" thresholds


def read_svo_lists(list_dir):
    """{SVO group: {filterID, ...}} from SVO's list VOTables."""
    groups = defaultdict(set)
    for path in sorted(Path(list_dir).glob("raw/*fps_list*.xml")):
        text = decode(path.read_bytes())
        fields = re.findall(r'<FIELD name="([^"]+)"', text)
        for row in re.findall(r"<TR>(.*?)</TR>", text, re.S):
            cells = [html.unescape(c) for c in re.findall(r"<TD>(.*?)</TD>", row, re.S)]
            filter_id = dict(zip(fields, cells)).get("filterID")
            if filter_id:
                groups[filter_id.rsplit(".", 1)[0]].add(filter_id)
    return groups


def newest_votables(archive_dir):
    """{filterID: (capture timestamp, file)} for SVO's newest archived VOTable of each ID."""
    newest = {}
    for manifest in sorted(Path(archive_dir).glob("*/manifest.jsonl")):
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("kind") != "fps_votable" or entry.get("status") != "ok":
                continue
            stamp = entry["capture_timestamp"]
            if entry["id"] not in newest or stamp > newest[entry["id"]][0]:
                newest[entry["id"]] = (stamp, manifest.parent / entry["file"])
    return newest


def excluded_pairs(aliases):
    """Predicate: may this (SVO filterID, NJM band) pair count?  The alias map's verdicts say no."""
    pairs = {(k, v["njm"]) for part in ("related", "rejected_candidates") for k, v in aliases.get(part, {}).items()}
    not_same = {tuple(pair.split(" -> ")) for pair in aliases.get("not_instrument_aliases", {})}

    def allowed(filter_id, band):
        group, instrument = filter_id.rsplit(".", 1)[0], band.rsplit("/", 1)[0]
        return (filter_id, band) not in pairs and (group, instrument) not in not_same
    return allowed


def candidates(filters_dir, aliases, allowed):
    """{SVO filterID: {NJM band path F/I/b: how}} from every piece of evidence."""
    found = defaultdict(dict)
    for info_path in sorted(filters_dir.glob("*/*/info.json")):
        instrument = f"{info_path.parent.parent.name}/{info_path.parent.name}"
        for band, entry in json.loads(info_path.read_text())["bands"].items():
            if entry.get("svo"):
                found[entry["svo"]["filterID"]][f"{instrument}/{band}"] = "info.json"
    for curve in sorted(filters_dir.glob("*/*/*.dat")):
        band = f"{curve.parent.parent.name}/{curve.parent.name}/{curve.stem}"
        filter_id = read_filter_header(curve).get("FilterID")
        if filter_id:
            found[filter_id].setdefault(band, "FilterID")
        found[band.rsplit("/", 1)[0] + "." + curve.stem].setdefault(band, "path")
    for filter_id, alias in aliases.get("aliases", {}).items():
        found[filter_id].setdefault(alias["njm"], f"svo_aliases.json ({alias['kind']})")
    return {filter_id: {band: how for band, how in bands.items() if allowed(filter_id, band)}
            for filter_id, bands in found.items()}


def prepared(samples, detector, photon_response):
    """(wavelength, peak-normalised response, pivot) of a curve, or None if it is unusable."""
    if len(samples) < 2:
        return None
    curve = np.array(sorted(samples), dtype=float)
    wave, response = curve[:, 0], np.clip(curve[:, 1], 0, None)
    if not (np.all(np.isfinite(curve)) and wave.min() > 0 and response.max() > 0):
        return None
    if photon_response and detector == "energy":
        response = response / wave
    response = response / response.max()
    numerator, denominator = np.trapezoid(response * wave, wave), np.trapezoid(response / wave, wave)
    return wave, response, float(np.sqrt(numerator / denominator))


def compare(a, b):
    """(overlap, relative pivot difference) of two prepared curves, as build_svo_aliases.py computes them."""
    grid = np.union1d(a[0], b[0])
    x = np.interp(grid, a[0], a[1], left=0, right=0)
    y = np.interp(grid, b[0], b[1], left=0, right=0)
    total = np.trapezoid(x + y, grid)
    overlap = float(1 - np.trapezoid(np.abs(x - y), grid) / total) if total > 0 else 0.0
    return overlap, abs(b[2] / a[2] - 1)


def same_curve(njm_curve, svo_votable):
    """Evidence dict for one NJM curve against one SVO VOTable; "same" is True or False."""
    njm_samples = _samples(njm_curve)
    njm_detector = normalize_detector_type(read_filter_header(njm_curve).get("DetectorType")) or "energy"
    svo = parse_votable(decode(svo_votable.read_bytes()))
    if svo is None or not svo["samples"]:
        return {"same": False, "reason": "SVO VOTable has no curve"}
    svo_detector = normalize_detector_type(svo["params"].get("DetectorType")) or "energy"
    best = None
    for photon_response, label in ((False, "as published"), (True, "as photon response")):
        a = prepared(njm_samples, njm_detector, photon_response)
        b = prepared(svo["samples"], svo_detector, photon_response)
        if a is None or b is None:
            continue
        overlap, pivot = compare(a, b)
        trial = {"same": overlap >= OVERLAP_SAME and pivot <= PIVOT_SAME, "compared": label,
                 "overlap": round(overlap, 4), "pivot_diff_pct": round(100 * pivot, 3)}
        if best is None or (trial["same"], trial["overlap"]) > (best["same"], best["overlap"]):
            best = trial
    if best is None:
        return {"same": False, "reason": "a curve is unusable"}
    if njm_detector != svo_detector:
        best["detector"] = f"NJM {njm_detector}, SVO {svo_detector}"
    return best


def band_evidence(filters_dir, capture, bands):
    """{NJM band: evidence} for one SVO ID's newest curve against each candidate NJM band."""
    evidence = {}
    for band, how in sorted(bands.items()):
        curve = filters_dir / f"{band}.dat"
        if not curve.is_file():
            continue
        result = same_curve(curve, capture[1]) if capture else {"same": False, "reason": "SVO curve not archived"}
        evidence[band] = {"how": how, **result}
    return evidence


def held_in(evidence):
    """NJM instruments holding a band with the same curve."""
    return {band.rsplit("/", 1)[0] for band, e in evidence.items() if e["same"]}


def assess_group(filters_dir, members, votables, found, allowed):
    """{filterID: {capture, njm: evidence}} for one SVO group.

    An ID that no evidence links is then searched for by curve in the NJM
    instruments holding the group's other IDs (e.g. PLANCK/LFI.070GHz is NJM's
    Planck/LFI/70).  Only bands no other ID holds count, so twin detectors with
    near-identical curves (left and right cameras, 85 K and 95 K) stay distinct.
    """
    bands = {}
    for filter_id in sorted(members):
        capture = votables.get(filter_id)
        bands[filter_id] = {"capture": capture[0] if capture else None,
                            "njm": band_evidence(filters_dir, capture, found.get(filter_id, {}))}
    linked = set().union(*(held_in(entry["njm"]) for entry in bands.values()))
    taken = {band for entry in bands.values() for band, e in entry["njm"].items() if e["same"]}
    for filter_id, entry in bands.items():
        capture = votables.get(filter_id)
        if entry["njm"] or capture is None:
            continue
        search = {f"{instrument}/{curve.stem}": "curve search"
                  for instrument in sorted(linked) for curve in sorted((filters_dir / instrument).glob("*.dat"))}
        search = {band: how for band, how in search.items() if band not in taken and allowed(filter_id, band)}
        matches = {band: e for band, e in band_evidence(filters_dir, capture, search).items() if e["same"]}
        entry["njm"].update(matches)
        taken.update(matches)
    return bands


def group_status(bands, instruments):
    """(status, NJM instruments holding every ID) of one assessed SVO group."""
    holders = set.intersection(*(held_in(entry["njm"]) for entry in bands.values()))
    held = sorted(holders & set(instruments))
    held_ids = sum(1 for entry in bands.values() if held_in(entry["njm"]))
    status = "held" if held else "split" if held_ids == len(bands) else "partial" if held_ids else "not held"
    return status, held, held_ids


def assess(root, list_dir):
    root = Path(root)
    filters_dir = root / "data" / "filters"
    aliases = json.loads((root / "filter_expansion" / "svo_aliases.json").read_text())
    allowed = excluded_pairs(aliases)
    votables = newest_votables(root / "source_evidence" / "svo-archive")
    found = candidates(filters_dir, aliases, allowed)
    instruments = {f"{d.parent.name}/{d.name}": [] for d in sorted(filters_dir.glob("*/*")) if any(d.glob("*.dat"))}
    groups = {}
    for group, members in sorted(read_svo_lists(list_dir).items()):
        bands = assess_group(filters_dir, members, votables, found, allowed)
        status, held, held_ids = group_status(bands, instruments)
        groups[group] = {"status": status, "njm": held, "held_ids": held_ids, "ids": len(bands)}
        if any(entry["njm"] for entry in bands.values()):
            groups[group]["bands"] = bands
        for instrument in held:
            instruments[instrument].append(group)
    return {
        "schema": 1,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": __doc__.split("\n\n")[1].strip(),
        "svo_list": os.path.relpath(Path(list_dir).resolve(), root.resolve()),
        "counts": dict(Counter(group["status"] for group in groups.values())),
        "thresholds": {"overlap": OVERLAP_SAME, "pivot": PIVOT_SAME},
        "instruments": instruments,
        "svo_groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("svo_list_dir")
    parser.add_argument("out")
    parser.add_argument("--root", default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    document = assess(args.root, args.svo_list_dir)
    Path(args.out).write_text(json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"{len(document['instruments'])} NJM instruments; SVO groups: {document['counts']}")


if __name__ == "__main__":
    main()
