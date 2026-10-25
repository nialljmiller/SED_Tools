"""Subaru Suprime-Cam and FOCAS broad-band filters (gemini.csv, block C).

Suprime-Cam (prime-focus camera, 2000-2017).  Source page "Suprime-Cam general filters"
https://subarutelescope.org/Archive/Instruments/SCam/sensitivity.html, ASCII lists txt/<band>.txt
(Angstrom, transmission fraction): B, V, Rc, Ic (Johnson-Cousins), g', r', i', z' (SDSS) and Y.
The page plots them as "Transmission" and tells users to also apply the FDCCD QE, the Primary Focus
Unit transmittance and the primary-mirror reflectivity, so they are filter-only curves.  The QE of
the Hamamatsu fully-depleted CCDs (S10892-01, installed 2008) is published as txt/QE_Hamamatsu.txt
(ccd.html; nm, fraction, 360-1040 nm) and is attached as a shared qe component.  Band names: Rc/Ic
for the Cousins filters (R/I would collide with r/i in case-insensitive file systems), g/r/i/z for
the SDSS primes.  No published pivot/central wavelengths, so no ref_pivot.

FOCAS (Cassegrain imager/spectrograph).  Source page "FOCAS filters and sensitivities"
https://www.naoj.org/Instruments/FOCAS/camera/filters.html, ASCII lists camera/txt/<band>.txt
(nm, transmission in percent -> divided by 100) for U, B, V, R, I; the page's plot (jpg/broad.gif)
is filter "Transmission".  The page's Centre wavelengths (3600, 4400, 5500, 6600, 8050 Å) are
checked as ref_pivot.  Filter-only curves.

Detectors: CCDs (Hamamatsu fully-depleted) -> photon counting.
"""
import numpy as np
from fxlib import ORIGINAL, Component, build_instrument, fetch, make_band

SC = "https://subarutelescope.org/Archive/Instruments/SCam/"
SC_PAGE = SC + "sensitivity.html"
SC_CCD = SC + "ccd.html"
SC_BANDS = {"B": "B", "V": "V", "Rc": "R", "Ic": "I", "g": "g", "r": "r", "i": "i", "z": "z", "Y": "Y"}

FO = "https://www.naoj.org/Instruments/FOCAS/camera/"
FO_PAGE = FO + "filters.html"
FO_CCD = "https://www.naoj.org/Instruments/FOCAS/ccdinfo.html"
FO_CENTRE = {"U": 3600, "B": 4400, "V": 5500, "R": 6600, "I": 8050}


def _two_col(path):
    rows = [ln.split()[:2] for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]
    return np.array(rows, float)          # raises if any line is not numeric


def _check(d, lo, hi, peak_lo, peak_hi, what):
    if not (lo <= d[:, 0].min() and d[:, 0].max() <= hi):
        raise ValueError(f"{what}: wavelength range {d[:, 0].min()}-{d[:, 0].max()} outside {lo}-{hi}")
    if not peak_lo <= d[:, 1].max() <= peak_hi:
        raise ValueError(f"{what}: peak {d[:, 1].max()} outside {peak_lo}-{peak_hi}")


def _suprimecam():
    bands, raw = [], []
    for name, f in SC_BANDS.items():
        url = f"{SC}txt/{f}.txt"
        path = fetch(url, "Subaru", f"SCam_{f}.txt")
        raw.append(path)
        d = _two_col(path)
        _check(d, 3000, 12000, 0.8, 1.0, url)          # Angstrom, fraction
        bands.append(make_band(
            name, d[:, 0], d[:, 1], unit="Angstrom", detector_type="photon", detector_source=ORIGINAL,
            components="filter", components_source=ORIGINAL, sources=f"{url} | {SC_PAGE}",
            svo_id=f"Subaru/Suprime.{name}"))
    qe_url = f"{SC}txt/QE_Hamamatsu.txt"
    qe_path = fetch(qe_url, "Subaru", "SCam_QE_Hamamatsu.txt")
    raw.append(qe_path)
    qe = _two_col(qe_path)
    _check(qe, 300, 1100, 0.5, 1.0, qe_url)            # nm, fraction
    comps = [Component("qe", qe[:, 0] * 10.0, qe[:, 1], band="*",
                       source=f"{qe_url} (Hamamatsu S10892-01 FDCCD QE, {SC_CCD})")]
    note = ("Subaru Suprime-Cam general broad-band filters (B, V, Rc, Ic, SDSS g' r' i' z', Y): filter-only "
            f"transmission lists from {SC_PAGE}. The page advises applying the CCD QE, Primary Focus Unit "
            "transmittance and primary-mirror reflectivity separately; the Hamamatsu FDCCD QE (2008-2017 "
            "detectors) is attached as a qe component (360-1040 nm). Pre-2008 MIT/LL CCD QE not included. "
            f"Detector: CCD, photon counting ({SC_CCD}).")
    build_instrument("Subaru", "SuprimeCam", bands, notes=note, components=comps, raw_files=raw,
                     ground_based=True)


def _focas():
    bands, raw = [], []
    for name, centre in FO_CENTRE.items():
        url = f"{FO}txt/{name}.txt"
        path = fetch(url, "Subaru", f"FOCAS_{name}.txt")
        raw.append(path)
        d = _two_col(path)
        _check(d, 290, 1001, 50, 100, url)             # nm, percent
        bands.append(make_band(
            name, d[:, 0], d[:, 1] / 100.0, unit="nm", detector_type="photon", detector_source=ORIGINAL,
            components="filter", components_source=ORIGINAL, sources=f"{url} | {FO_PAGE}",
            svo_id=f"Subaru/FOCAS.{name}",
            ref_pivot=(float(centre), f"FOCAS filters page Centre wavelength ({FO_PAGE})")))
    note = ("Subaru FOCAS broad-band U, B, V, R, I filter-only transmission lists from "
            f"{FO_PAGE} (percent converted to fractions). No QE, optics or atmosphere included. "
            f"Detector: CCD (Hamamatsu fully-depleted), photon counting ({FO_CCD}).")
    build_instrument("Subaru", "FOCAS", bands, notes=note, raw_files=raw, ground_based=True)


def run():
    _suprimecam()
    _focas()
