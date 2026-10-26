"""Himawari-8 and -9 AHI: band spectral response functions from JMA MSC.

Source page: https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html
("AHI-8_SpectralResponsivity.zip (update September 2013; provided by ITT Exelis)",
"AHI-9_SpectralResponsivity.zip (update October 2013; provided by ITT Exelis)").  Each zip holds one
Excel workbook with a sheet per band ("Band 1".."Band 16"; header "Flight Model: AHI-08|AHI-09",
"CH0047".."CH1330") and three columns: Wavelength [um], Wavenumber [cm^-1], "Relative Responsivity" [-],
peak 1.  Each curve is the band (whole-instrument) relative spectral responsivity: filter+optics+qe.
The workbooks are read with the standard library (no openpyxl in the venv).

Naming follows SVO: Himawari/AHI8 = Facility "Himawari-8", AHI9 = "Himawari-9" (Facility PARAMs of the
archived SVO VOTables in phase3/ia/raw/); band names Band01..Band16 as in SVO.
Ref pivots: JMA's Table 1 central wavelengths per satellite on the same page (AHI-8 and AHI-9 columns).
"""
import io
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from fxlib import DOWNLOADS, build_instrument, fetch, make_band  # noqa: F401

PAGE = "https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html"
BASE = "https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/"
PERF = BASE + "fig/AHI8_performance_test_en.pdf"
# JMA Table 1 "Central wave length [um]" (spsg_ahi.html), bands 1-16.
CENTRAL_UM = {
    "AHI8": [0.47063, 0.51000, 0.63914, 0.85670, 1.6101, 2.2568, 3.8853, 6.2429, 6.9410, 7.3467,
             8.5926, 9.6372, 10.4073, 11.2395, 12.3806, 13.2807],
    "AHI9": [0.47059, 0.50993, 0.63972, 0.85668, 1.6065, 2.2570, 3.8289, 6.2479, 6.9555, 7.3437,
             8.5936, 9.6274, 10.4074, 11.2080, 12.3648, 13.3107],
}
# Nominal band wavelengths (CHxxxx code in 0.01 um) used only to check sheet identity.
NOMINAL_CODE = [47, 51, 64, 86, 161, 226, 390, 618, 695, 734, 850, 961, 1035, 1120, 1230, 1330]
SETS = [
    ("AHI8", "srf_201309/AHI-08_SpectralResponsivity.zip", "AHI-08", "Himawari-8", "September 2013"),
    ("AHI9", "srf_201310/AHI-09_SpectralResponsivity.zip", "AHI-09", "Himawari-9", "October 2013"),
]

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

DET_NOTE = (
    "DetectorType=energy is assumed: JMA's workbooks give a 'Relative Responsivity' [-] without stating "
    "photon or energy weighting; weather-satellite SRFs are conventionally applied as the weighting of "
    "spectral radiance (energy convention), as for the other weather-satellite sets in this expansion. "
    "Note SVO's archived Himawari/AHI8 and AHI9 records carry DetectorType 1 (photon counter).")


def read_workbook(zpath):
    """Return {sheet name: [ {column letter: value} per row ]} from the single xlsx in ``zpath``."""
    with zipfile.ZipFile(zpath) as outer:
        names = outer.namelist()
        assert len(names) == 1 and names[0].endswith(".xlsx"), names
        z = zipfile.ZipFile(io.BytesIO(outer.read(names[0])))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS):
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


def read_band(rows, fm, n):
    """Check a band sheet's header and return (wavelength um, wavenumber cm-1, response)."""
    assert rows[1].get("A") == f"Flight Model: {fm}", rows[1]
    assert rows[1].get("B") == f"CH{NOMINAL_CODE[n - 1]:04d}", (n, rows[1])
    assert [rows[2].get(k) for k in "ABC"] == ["Wavelength", "Wavenumber", "Relative Responsivity"], rows[2]
    assert [rows[3].get(k) for k in "ABC"] == ["[um]", "[cm^-1]", "[-]"], rows[3]
    d = np.array([[r["A"], r["B"], r["C"]] for r in rows[4:] if all(isinstance(r.get(k), float) for k in "ABC")])
    assert len(d) == len(rows) - 4, (n, len(d), len(rows))
    return d[:, 0], d[:, 1], d[:, 2]


def run():
    # Page and test PDF were saved alongside the zips (curl, 2026-10-26) as provenance evidence.
    raw =[DOWNLOADS / "Himawari" / "spsg_ahi.html", DOWNLOADS / "Himawari" / "AHI8_performance_test_en.pdf"]
    for inst, rel, fm, sat, released in SETS:
        zpath = fetch(BASE + rel, "Himawari")
        book = read_workbook(zpath)
        bands = []
        for n in range(1, 17):
            um, wn, resp = read_band(book[f"Band {n}"], fm, n)
            assert np.allclose(1e4 / um, wn, rtol=2e-4), (inst, n)
            assert 0.999 < resp.max() <= 1.0001, (inst, n, resp.max())
            central = CENTRAL_UM[inst][n - 1]
            assert um.min() < central < um.max(), (inst, n)
            band = f"Band{n:02d}"
            bands.append(make_band(
                band, um, resp, unit="micron", detector_type="energy", detector_source="assumed",
                components="filter+optics+qe", components_source="original",
                sources=f"{BASE}{rel} (sheet 'Band {n}', {fm} CH{NOMINAL_CODE[n - 1]:04d}) | {PAGE}",
                svo_id=f"Himawari/{inst}.{band}",
                ref_pivot=(central * 1e4, f"{PAGE} (Table 1, {fm} ({sat}) band {n} central wavelength)")))
        build_instrument("Himawari", inst, bands, raw_files=[zpath] + [p for p in raw if p.exists()],
                         ground_based=False, notes=(
            f"Advanced Himawari Imager (AHI) flight model {fm} on {sat}: band spectral response functions "
            f"from JMA Meteorological Satellite Center ({PAGE}), workbook released {released}, "
            "'provided by ITT Exelis' (page text). Columns used: wavelength [um] (checked against the "
            "wavenumber column) and 'Relative Responsivity' [-] (peak 1, not rescaled); sorted to increasing "
            "wavelength; small negative noise samples set to 0 (see per-band clean notes). Components: band "
            "system response (filter+optics+qe as a whole; not split by the owner). " + DET_NOTE +
            " Ref pivots are JMA's Table 1 central wavelengths for this satellite; JMA's AHI-8 test-result "
            f"PDF ({PERF}) defines its 'Center Wavelength' as the centre of the FWHM bandwidth. "
            "Source: Japan Meteorological Agency website (" + PAGE + "); curves edited only by unit "
            "conversion (um to Angstrom), sorting and clipping of negative noise."))
