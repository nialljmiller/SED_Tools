"""KPNO Mosaic (Mayall 4 m prime-focus CCD mosaic imager) broad-band filters.

Sources: NOIRLab KPNO filter pages (https://noirlab.edu/science/programs/kpno/filters/mosaic;
each https://noirlab.edu/science/filters/kpNNNN shows "Instrument: Mosaic" and links
/science/documents/kpNNNN, whose data file is under
https://noirlab.edu/science/sites/default/files/media/archives/documents/).  All are NOAO
Lambda 9 spectrophotometer lab scans ("== Filter Data - Lambda 9 Spectrophotometer =="),
Angstrom, transmission in percent, header with Filter id / cwl / fwhm / Tmax:
  U  k1044 "Us solid U"        k1044bp.txt        (header "MOSA USOLID 00D")
  B  k1002 "B Harris"          k1002bp_jul04.txt  (header "Filter: K1002")
  V  k1003 "V Harris"          k1003bp_jul04.txt
  R  k1004 "R Harris"          k1004bp_aug04.txt
  I  k1005 "I Nearly-Mould"    k1005bp_aug04.txt
  r_SDSS k1018, i_SDSS k1019, z_SDSS k1020 ("*** Retired!")  k10NN.txt (header "SDSS R/I/Z")
Filter only: no telescope/corrector, CCD QE or atmosphere.  Photon-counting CCDs.

HELD: g SDSS k1017 (the NOIRLab kp1017 document/data file is a 1993 scan of a different filter,
"KP1017 75DEG", CWL 9556 A, FWHM 139 A; no other original g curve found; SVO-only);
k1026 815_v1 (narrow band: the 0-deg scan k10260d.txt peaks at 8164 A, the "f/3.1
simulation" k1026sim04.txt is from a 12-deg scan with CWL 8087 A, while the page says the
f/3.1 shift is only ~12 A; ambiguous which curve represents Mosaic, and k1046 replaced it).
"""
import re

import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

DOCS = "https://noirlab.edu/science/sites/default/files/media/archives/documents/"
PAGE = "https://noirlab.edu/science/programs/kpno/filters/mosaic"
# band: (k number, file, regex the "Filter:" header line must match)
FILTERS = {
    "U": (1044, "k1044bp.txt", r"MOSA USOLID"),
    "B": (1002, "k1002bp_jul04.txt", r"Filter: K1002\b"),
    "V": (1003, "k1003bp_jul04.txt", r"Filter: K1003\b"),
    "R": (1004, "k1004bp_aug04.txt", r"Filter: K1004\b"),
    "I": (1005, "k1005bp_aug04.txt", r"Filter: K1005\b"),
    "r_SDSS": (1018, "k1018.txt", r"Filter: SDSS R\b"),
    "i_SDSS": (1019, "k1019.txt", r"Filter: SDSS I\b"),
    # z_SDSS k1020 held (manager): long-pass, scan ends at 11000 A with T=0.97; red edge needs CCD QE.
}
TITLES = {1044: "Us solid U", 1002: "B Harris", 1003: "V Harris", 1004: "R Harris",
          1005: "I Nearly-Mould", 1018: "r SDSS", 1019: "i SDSS", 1020: "z SDSS (retired)"}


def _parse(path):
    text = path.read_text(encoding="latin-1").replace("\r", "")
    if not text.startswith("== Filter Data - Lambda 9 Spectrophotometer =="):
        raise ValueError(f"{path}: not a KPNO Lambda 9 filter scan")
    rows = []
    for ln in text.splitlines():
        f = ln.split()
        if len(f) == 2 and all(re.fullmatch(r"-?[0-9.]+", x) for x in f):
            rows.append((float(f[0]), float(f[1])))
    m = re.search(r"cwl=\s*([0-9.]+)A\s+fwhm=\s*([0-9.]+)A\s+Tmax=\s*([0-9.]+)%", text)
    return text, np.array(rows), (tuple(float(x) for x in m.groups()) if m else None)


def run():
    bands, raw, hdr = [], [], {}
    for name, (k, fn, pat) in FILTERS.items():
        url = DOCS + fn
        p = fetch(url, "KPNO", f"Mosaic_{fn}")
        raw.append(p)
        text, a, (cwl, fwhm, tmax) = _parse(p)
        if not re.search(pat, text.splitlines()[3]):
            raise ValueError(f"{p}: filter line {text.splitlines()[3]!r} does not match {pat}")
        if abs(a[:, 1].max() - tmax) > 0.05:
            raise ValueError(f"{p}: data peak {a[:, 1].max()} != header Tmax {tmax} %")
        b = make_band(name, a[:, 0], a[:, 1] / 100.0, unit="Angstrom", detector_type="photon",
                      detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                      sources=f"{url} | https://noirlab.edu/science/filters/kp{k}",
                      svo_id=f"KPNO/Mosaic.{name.split('_')[0]}" if name != "U" else None)
        hdr[name] = {"k": k, "title": TITLES[k], "header_cwl_A": cwl, "header_fwhm_A": fwhm,
                     "header_tmax_pct": tmax}
        bands.append(b)
    note = ("KPNO Mayall 4 m Mosaic filters, NOAO Lambda 9 lab scans from the NOIRLab KPNO filter "
            "pages (" + "; ".join(f"{b}=k{v[0]} {TITLES[v[0]]} ({v[1]})" for b, v in FILTERS.items())
            + "). Angstrom, percent -> fraction (data peak checked against header Tmax). Filter only: "
            "no telescope/corrector, CCD QE (photon-counting CCDs) or atmosphere. Lab scans at normal "
            "incidence; NOIRLab notes a small blue shift in the f/3.1 beam (~12 A for narrow bands). "
            "Negative noise values set to 0. Header cwl/fwhm (FWHM centre, not pivot) are in provenance. HELD: "
            "g SDSS k1017 (NOIRLab's kp1017 file is a different 9556 A filter) and 815_v1 k1026 "
            "(conflicting f/3.1 curves), and z SDSS k1020 (long-pass, T=0.97 where the scan ends at "
            "11000 A; its red edge is the CCD QE, which is not published as data).")
    build_instrument("KPNO", "Mosaic", bands, notes=note, ground_based=True, raw_files=raw,
                     extra_provenance={"lab_headers": hdr, "filter_page": PAGE})
