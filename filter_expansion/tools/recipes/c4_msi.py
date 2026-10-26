"""Sentinel-2A, -2B and -2C MSI: band-averaged spectral response functions from ESA SentiWiki.

HELD (source-builder recipe): SentiWiki's terms (https://sentiwiki.copernicus.eu/web/terms-and-conditions)
say "ESA does not grant the right to resell or redistribute any information, documents, images or
material from its website or to compile or create derivative works from material on its website", and
the workbook carries no licence of its own.  This recipe therefore builds into
phase4/builds/CB4/held_stage/ (not stage/) unless FX_MSI_PUBLISH=1 is set after the terms are cleared.

Source page: https://sentiwiki.copernicus.eu/web/s2-documents, attachment
"sentinel-2-spectral-response-functions-v5.xlsx" (modified Aug 06, 2026): COPE-GSEG-EOPG-TN-15-0007
"Sentinel-2 Spectral Response Functions" [S2-SRF] version 5.0.  Sheets "Spectral Responses (S2A|S2B|S2C)"
hold column SR_WL (nm, 300-2600 step 1) and S2x_SR_AV_B1..B12, B8A: "the average spectral response
determined through ground-based testing of the MSI" (Overview sheet), peak 1.  The workbook calls the
curve the "Instrument Spectral Response Function", so components are the whole band response
(filter+optics+qe).  Sheet "Equivalent Wavelengths" gives lambda_c = int(lambda S) / int(S) per band and
satellite; it is recomputed here as a parse check and passed as ref_pivot.

Naming: Facility "Sentinel" (as in SVO's Sentinel/MSI IDs); one instrument per satellite, MSI-2A, MSI-2B,
MSI-2C (owner's satellite names; no archived SVO record was available to map SVO's -A/-B suffixes);
band names B01..B12 and B8A as in the owner's "Equivalent Wavelengths" sheet.
The workbook is read with the standard library (no openpyxl in the venv).
"""
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from fxlib import DOWNLOADS, ROOT, STAGE, build_instrument, fetch, make_band  # noqa: F401

PAGE = "https://sentiwiki.copernicus.eu/web/s2-documents"
TERMS = "https://sentiwiki.copernicus.eu/web/terms-and-conditions"
URL = ("https://sentiwiki.copernicus.eu/__attachments/"
       "a_ad595cc9fa0018a4e9eb28e1e234124a751149b5933b0b8b1f6b80612eaab586/"
       "sentinel-2-spectral-response-functions-v5.xlsx?cb=0735bcc384fde7424ba8d58a19c4a9dd")
OUT = STAGE if os.environ.get("FX_MSI_PUBLISH") == "1" else ROOT / "phase4" / "builds" / "CB4" / "held_stage"
SATS = ["S2A", "S2B", "S2C"]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

DET_NOTE = (
    "DetectorType=energy is assumed: the workbook gives the 'average spectral response determined through "
    "ground-based testing of the MSI' (relative, peak 1) without stating photon or energy weighting; "
    "ESA's own 'equivalent wavelength' int(lambda S)/int(S) weights S directly, and Earth-observation SRFs "
    "are conventionally applied as the weighting of spectral radiance (energy convention), as for the "
    "weather-satellite sets in this expansion.")


def read_workbook(path, wanted):
    """Return {sheet name: [ {column letter: value} per row ]} for the sheets in ``wanted``."""
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS):
        if s.get("name") not in wanted:
            continue
        target = rels[s.get(f"{{{NS['r']}}}id")].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).iter(f"{{{NS['m']}}}row"):
            cells = {}
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is None:
                    continue
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                cells[col] = shared[int(v.text)] if c.get("t") == "s" else (
                    v.text if c.get("t") == "str" else float(v.text))
            rows.append(cells)
        out[s.get("name")] = rows
    return out


def run():
    path = fetch(URL, "Sentinel", "sentinel-2-spectral-response-functions-v5.xlsx")
    raw = [path] + [p for p in (DOWNLOADS / "Sentinel" / "s2-documents.html",
                                DOWNLOADS / "Sentinel" / "sentiwiki-terms-and-conditions.html",
                                DOWNLOADS / "Sentinel" / "ads-ec-sentinel-licence.html") if p.exists()]
    sheets = [f"Spectral Responses ({s})" for s in SATS] + ["Overview", "Equivalent Wavelengths"]
    book = read_workbook(path, sheets)
    ov = {r.get("A"): r.get("F") for r in book["Overview"] if r.get("A")}
    assert ov["Code:"].strip() == "COPE-GSEG-EOPG-TN-15-0007" and ov["Version:"] == "5.0", ov
    eq_rows = {r["B"]: r for r in book["Equivalent Wavelengths"] if r.get("B") in NAMES}
    assert len(eq_rows) == 13, sorted(eq_rows)
    for k, sat in enumerate(SATS):
        rows = book[f"Spectral Responses ({sat})"]
        cols = "BCDEFGHIJKLMN"
        assert rows[0].get("A") == "SR_WL", rows[0]
        assert [rows[0].get(c) for c in cols] == [f"{sat}_SR_AV_{b}" for b in BANDS], rows[0]
        data = [r for r in rows[1:] if isinstance(r.get("A"), float)]
        assert all(len(r) == 14 for r in data) and len(data) == 2301, (sat, len(data))
        nm = np.array([r["A"] for r in data])
        assert nm[0] == 300 and nm[-1] == 2600 and np.all(np.diff(nm) == 1), sat
        bands = []
        for col, b, name in zip(cols, BANDS, NAMES):
            resp = np.array([r[col] for r in data])
            assert 0.999 < resp.max() <= 1.0001 and resp.min() >= 0, (sat, b, resp.min(), resp.max())
            eq = eq_rows[name]["CDE"[k]]
            lam_c = np.trapezoid(nm * resp, nm) / np.trapezoid(resp, nm)
            assert abs(lam_c - eq) < 0.2, (sat, b, lam_c, eq)
            bands.append(make_band(
                name, nm, resp, unit="nm", detector_type="energy", detector_source="assumed",
                components="filter+optics+qe", components_source="original",
                sources=f"{URL.split('?')[0]} (sheet 'Spectral Responses ({sat})', column "
                        f"{sat}_SR_AV_{b}) | {PAGE}",
                ref_pivot=(eq * 10, f"{PAGE} (sentinel-2-spectral-response-functions-v5.xlsx, sheet "
                                    f"'Equivalent Wavelengths', {sat} {name}: int(lambda S)/int(S))")))
        build_instrument("Sentinel", f"MSI-{sat[1:]}", bands, raw_files=raw, ground_based=False, root=OUT,
                         notes=(
            f"MultiSpectral Instrument (MSI) on Sentinel-{sat[1:]}: band-averaged spectral response "
            f"functions from ESA SentiWiki ({PAGE}), workbook COPE-GSEG-EOPG-TN-15-0007 'Sentinel-2 "
            "Spectral Response Functions' v5.0 (sentinel-2-spectral-response-functions-v5.xlsx). Overview "
            "sheet (verbatim): 'The values in the SR_AV_Bx columns represent the average spectral response "
            "determined through ground-based testing of the MSI.' and 'Data are based on ground measurements "
            "performed in the frame of the Assembly, Integration and Test (AIT) campaign of the MSI.' "
            "Wavelength column SR_WL in nm (300-2600, 1 nm step) converted to Angstrom; response peak 1, not "
            "rescaled; the full 300-2600 nm grid is kept (zeros outside each band, as published). Components: band "
            "instrument spectral response (filter+optics+qe as a whole; not split by the owner). "
            + DET_NOTE + " Ref pivots are ESA's 'Equivalent Wavelengths' sheet values (int(lambda S)/int(S)), "
            "which this recipe reproduces from the curve to <0.2 nm. Licence: SentiWiki terms forbid "
            f"redistribution ({TERMS}); held from publication pending clearance."))


if __name__ == "__main__":
    run()
