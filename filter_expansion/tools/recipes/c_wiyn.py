"""WIYN 3.5 m One Degree Imager (ODI), Kitt Peak.

Source page: https://www.wiyn.org/ODI/Observe/wiynodifilters.html ("ODI Filter Throughput",
data column "THROUGHPUT DATA - FILTER ONLY"; NB422/NB659/NB746 marked "filter-only throughput,
averaged over the field of view").  Files (www.wiyn.org/ODI/Observe/):
  odi_u.dat, odi_r.txt, odi_NB695.txt   Angstrom, fraction, no header
  NB_422.txt                            nm, percent; header "Wavelength[nm] T% in a parallel
                                        beam at AOI 5 deg" (descending order)
Filter only: no telescope, corrector, detector QE or atmosphere.  Detector: STA orthogonal-
transfer-array CCDs (photon counting; https://www.wiyn.org/ODI/wiynODI.html).

HELD (not built): odi_g, odi_i, odi_z, NB659, NB746.  Their filter-only scans transmit strongly
beyond ~11000 A (g 0.55, i 0.61, z 0.36 at 12000 A; NB659 0.20 and NB746 0.52 at 1200 nm), past
the CCD cutoff; 49/25/7/37/68 % of the photon-weighted response lies above 10500 A, so the
filter-only pivots are wrong (g 6223 A vs 4786 A without the leak).  WIYN publishes no ODI QE
curve as data (only predicted tables inside the Java ETC), so these need a QE curve or an
explicit decision to cut at the Si limit.  r has a 1 % leak (T 0.024 at 12000 A, pivot shift
0.3 %) and is built as published.
"""
import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

BASE = "https://www.wiyn.org/ODI/Observe/"
PAGE = BASE + "wiynodifilters.html"
DET = "https://www.wiyn.org/ODI/wiynODI.html"
FILES = {"u": ("odi_u.dat", "Angstrom", 1.0), "r": ("odi_r.txt", "Angstrom", 1.0),
         "NB695": ("odi_NB695.txt", "Angstrom", 1.0), "NB422": ("NB_422.txt", "nm", 100.0)}
HELD = {"g": "odi_g.txt", "i": "odi_i.txt", "z": "odi_z.txt", "NB659": "NB659.txt",
        "NB746": "NB746.txt"}


def _table(path):
    rows = []
    for ln in path.read_text(encoding="latin-1").splitlines():
        f = ln.split()
        try:
            rows.append((float(f[0]), float(f[1])))
        except (ValueError, IndexError):
            pass
    return np.array(rows)


def _leak(wl, tr):
    w = np.clip(tr, 0, None) * wl
    m = wl > 10500
    return float(np.trapezoid(w[m], wl[m]) / np.trapezoid(w, wl))


def run():
    bands, raw, leak = [], [], {}
    for name, (fn, unit, scale) in FILES.items():
        p = fetch(BASE + fn, "WIYN", f"ODI_{fn}")
        raw.append(p)
        a = _table(p)
        peak = a[:, 1].max()
        if (scale == 100.0) != (peak > 1.5):
            raise ValueError(f"{p}: transmission scale unexpected (peak {peak})")
        if fn == "NB_422.txt" and not p.read_text().startswith("Wavelength[nm]\tT%"):
            raise ValueError(f"{p}: unexpected header")
        b = make_band(name, a[:, 0], a[:, 1] / scale, unit=unit, detector_type="photon",
                      detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                      sources=f"{BASE}{fn} | {PAGE}")
        leak[name] = round(_leak(b.wavelength, b.transmission), 4)
        if leak[name] > 0.02:
            raise ValueError(f"ODI {name}: {leak[name]:.0%} of the response is above 10500 A")
        bands.append(b)
    note = ("WIYN 3.5 m ODI filter-only throughputs from wiynodifilters.html: u=odi_u.dat, "
            "r=odi_r.txt, NB695=odi_NB695.txt (Angstrom, fraction) and NB422=NB_422.txt (nm, percent "
            "'in a parallel beam at AOI 5 deg', although the page calls it the field-averaged "
            "filter-only throughput). No telescope, corrector, CCD QE (STA OTA CCDs, photon counting) "
            "or atmosphere. r keeps a small published red leak (T 0.024 at 12000 A, ~1 % of the "
            "photon-weighted response, pivot +0.3 %). odi_NB695.txt repeats its first 752 rows "
            "(3000-6755 A) identically before the full scan; the duplicates are merged (no change). "
            "Negative noise values set to 0. g, i, z, NB659 "
            "and NB746 are HELD: their filter-only scans transmit 0.2-0.6 beyond 11000 A (past the "
            "CCD cutoff), which moves the filter-only pivot badly (g 6223 vs 4786 A), and no ODI QE "
            "curve is published as data. No published pivots.")
    build_instrument("WIYN", "ODI", bands, notes=note, ground_based=True, raw_files=raw,
                     extra_provenance={"detector_evidence": DET,
                                       "fraction_of_photon_response_above_10500A": leak,
                                       "held": {k: BASE + v for k, v in HELD.items()}})
