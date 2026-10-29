"""Standard photometric systems (facility ``Generic``) from their defining papers' files.

Generic/Bessell  Bessell & Murphy (2012) UBVRI photonic responses, CDS J/PASP/124/140 table1.
Generic/MKO      MKO-NIR J H K L' M' (Tokunaga, Simons & Vacca 2002): IRTF NSFCam cold filter
                 scans nsfcam_<b>mk_trans.dat (percent -> fraction; filter only).
Generic/USNO     u'g'r'i'z' (Smith et al. 2002): J. Gunn's usno_[ugriz].res responses at 1.3
                 airmasses.  The Fermilab site answers HTTP 403 to scripts, so the files are read
                 from their Wayback Machine raw snapshots (2021-03-19), which the curves cite.

Held: Johnson, Cousins, Stromgren, Kron, Thuan-Gunn and Bessell (1990) have no numeric original
(only SVO or PDF tables).
"""

from __future__ import annotations

import numpy as np

from ..filters.curve_metadata import PHOTON
from ._original import Fetcher, Instrument, fixed_width_pairs, make_band, make_builder
FAC = "Generic"

B12_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/table1.dat"
B12_PAPER = "https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B"
# 0-based slices from the ReadMe byte description (bytes 1-4 / 6-10, 12-15 / 17-21, ...).
B12_COLS = {b: (11 * i, 11 * i + 4, 11 * i + 5, 11 * i + 10) for i, b in enumerate("UBVRI")}

MKO_URL = "https://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_{}_trans.dat"
MKO_PAGE = "https://irtfweb.ifa.hawaii.edu/~nsfcam2/Filter_Profiles.html"
MKO_PAPER = "https://ui.adsabs.harvard.edu/abs/2002PASP..114..180T"
MKO_DET = "https://irtfweb.ifa.hawaii.edu/~nsfcam2/Welcome.html"
MKO = {"J": "jmk", "H": "hmk", "K": "kmk", "Lp": "lpmk", "Mp": "mpmk"}

USNO_ORIG = "https://www-star.fnal.gov/ugriz/Filters/usno_{}.res"
USNO_WB = {"u": "20210319150010", "g": "20210319132508", "r": "20210319053959",
           "i": "20210319071508", "z": "20210319124657"}
USNO_README = "https://www-star.fnal.gov/ugriz/Filters/readme.txt"
USNO_README_WB = "https://web.archive.org/web/20210319075936id_/" + USNO_README
USNO_PAGE = "https://www-star.fnal.gov/ugriz/Filters/response.html"
USNO_PAPER = "https://ui.adsabs.harvard.edu/abs/2002AJ....123.2121S"


def bessell(fetch: Fetcher) -> Instrument:
    readme = fetch.text(B12_URL.replace("table1.dat", "ReadMe"))
    if "Normalized UBVRI photonic responses" not in readme:
        raise ValueError("unexpected Bessell & Murphy ReadMe")
    curves = fixed_width_pairs(fetch.text(B12_URL), B12_COLS)
    bands = [make_band(b, wl, tr, unit="Angstrom", detector_type=PHOTON, components="filter+optics+qe",
                       sources=f"{B12_URL} | {B12_PAPER}") for b, (wl, tr) in curves.items()]
    note = ("Bessell & Murphy (2012, PASP 124, 140) table 1 (CDS J/PASP/124/140): the 2012 revised "
            "UBVRI passbands, normalised PHOTONIC responses for photon-counting synthetic photometry. "
            "This is NOT the Bessell (1990, PASP 102, 1181) energy-response realization (held). "
            "Wavelength unit 0.1 nm = Angstrom. Abstract standard-system responses (whole system; "
            "not separable into filter/optics/QE); the ReadMe does not state whether atmospheric "
            "extinction is included.")
    return Instrument(FAC, "Bessell", bands, note, ground_based=True)


def mko_scan(text: str, url: str = "") -> np.ndarray:
    """(micron, percent) rows of an NSFCam MK scan, after checking its header."""
    lines = text.splitlines()
    if lines[0].split("\t") != ["WL (Micrometers)", "Transmission (%)"]:
        raise ValueError(f"{url}: unexpected header {lines[0]!r}")
    return np.array([ln.split() for ln in lines[1:] if ln.strip()], float)


def mko(fetch: Fetcher) -> Instrument:
    bands = []
    for name, stem in MKO.items():
        url = MKO_URL.format(stem)
        data = mko_scan(fetch.text(url), url)
        bands.append(make_band(name, data[:, 0], data[:, 1] / 100.0, unit="micron", detector_type=PHOTON,
                               components="filter", sources=f"{url} | {MKO_PAGE} | {MKO_PAPER}"))
    note = ("MKO-NIR J H K L' M' (Tokunaga, Simons & Vacca 2002, PASP 114, 180): IRTF NSFCam cold "
            f"filter scans ({MKO_PAGE}). FILTER ONLY: no detector QE, optics or atmosphere. Source "
            "header gives percent (divided by 100); microns converted to Angstrom; negative baseline "
            "noise set to 0; L' duplicate wavelengths averaged. Photon convention: NSFCam uses a "
            f"Hawaii-2RG IR array ({MKO_DET}). Numerically identical (atol 1e-7) to the mirror's "
            "IRTF/NSFCam JMK/HMK/KMK/LPMK/MPMK, which come from the same files; provided here under "
            "standard-system names.")
    return Instrument(FAC, "MKO", bands, note, ground_based=True)


def usno_table(text: str, band: str, url: str = "") -> np.ndarray:
    lines = text.splitlines()
    if (lines[0].strip() != f"\\ FILE: /surtel/filters/usno/usno_{band}.res"
            or lines[2].split() != ["\\", "lam", "vac", "air1.0", "air1.3", "air1.6", "air1.9"]):
        raise ValueError(f"{url}: unexpected header")
    data = np.array([ln.split() for ln in lines if ln.strip() and not ln.startswith("\\")], float)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"{url}: expected 6 columns")
    return data


def usno(fetch: Fetcher) -> Instrument:
    rtext = fetch.text(USNO_README_WB)
    if "USNO 40-inch/SITe 1024^2" not in rtext or "median airmass of about 1.3" not in rtext:
        raise ValueError("unexpected USNO readme")
    bands = []
    for f in "ugriz":
        wb = f"https://web.archive.org/web/{USNO_WB[f]}id_/{USNO_ORIG.format(f)}"
        data = usno_table(fetch.text(wb), f, wb)
        # Column 4 = 1.3 airmasses, the natural-system median airmass.
        bands.append(make_band(f + "p", data[:, 0], data[:, 3], unit="Angstrom", detector_type=PHOTON,
                               components="filter+optics+qe+atmosphere",
                               sources=f"{USNO_ORIG.format(f)} (via {wb}) | {USNO_PAGE} | {USNO_PAPER}"))
    note = ("USNO 40-inch u'g'r'i'z' standard-star system (Smith et al. 2002, AJ 123, 2121): "
            "J. Gunn's 2001-12-18 response functions usno_[ugriz].res for the USNO 40-in telescope "
            f"+ SITe 1024^2 UV-coated CCD ({USNO_PAGE}, {USNO_README}). Column 4 = 1.3 airmasses, "
            "the natural-system median airmass: response includes filter, CCD QE (corrected for IR "
            "scattering; point sources), telescope and atmosphere. Photon-counting CCD QE functions. "
            "z' response is quite uncertain (no USNO CCD measurement, per the readme). The site "
            "returns HTTP 403 to scripts; payload from Wayback Machine raw snapshots (2021-03-19). "
            "Not the SDSS 2.5 m ugriz responses; USNO->AB offsets are in the readme.")
    return Instrument(FAC, "USNO", bands, note, ground_based=True)


RECIPES = {
    "Bessell": (FAC, bessell),
    "MKO": (FAC, mko),
    "USNO": (FAC, usno),
}

build = make_builder(RECIPES)
