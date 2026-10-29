"""Swift/UVOT rebuilt from the HEASARC calibration database (CALDB).

Published curves are the current (CAL_QUAL=0) UVOT ancillary response files
(``cpf/arf``, effective area in cm^2).  The separately published factors are
the filter transmissions (``bcf/ftrans``), the primary-mirror effective area
(``bcf/effarea/swumirrorea``) and the instrumental effective area without
filter (``bcf/effarea/swuinstea``).  Effective areas are divided by the
659 cm^2 geometric collecting area of the primary mirror (Poole et al. 2008,
MNRAS 383, 627, sect. 4), so every curve is a dimensionless throughput on one
absolute scale.

The HEASARC host serving ``/FTP/caldb`` presented a certificate for another
host on 2026-10-26, so each file is also tried on the HEASARC mirror in the
AWS Open Data registry (bucket ``nasa-heasarc``), which holds the same CALDB
tree.
"""

from __future__ import annotations

import gzip
import io
import logging
from typing import Dict, Sequence, Tuple

import numpy as np
from requests import RequestException

from ._original import Component, Fetcher, Instrument, clean_curve, make_band, make_builder

logger = logging.getLogger(__name__)

PHOTON = "photon"

#: CALDB tree of the UVOT, on HEASARC and on its AWS Open Data mirror (tried in order).
CALDB_ROOTS = (
    "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/swift/uvota/",
    "https://nasa-heasarc.s3.amazonaws.com/caldb/data/swift/uvota/",
)
CALDB_PAGE = "https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/swift/"
POOLE = "https://doi.org/10.1111/j.1365-2966.2007.12563.x (Poole et al. 2008, arXiv:0708.2259)"
BREEVELD = "https://arxiv.org/abs/1102.4717 (Breeveld et al. 2011, AIP Conf. Proc. 1358, 373)"
#: Geometric collecting area of the UVOT primary mirror, cm^2 (Poole et al. 2008, sect. 4 (i)).
GEOMETRIC_AREA_CM2 = 659.0

#: band -> (ftrans extension filter name, current ARF in cpf/arf). The ARFs are the
#: CAL_QUAL=0 entries of caldb.indx (2024-02-01); the UV ones are the end-2010
#: recalibration (Breeveld et al. 2011).
BANDS: Dict[str, Tuple[str, str]] = {
    "V": ("V", "swuvv_20041120v104.arf"),
    "B": ("B", "swubb_20041120v104.arf"),
    "U": ("U", "swuuu_20041120v104.arf"),
    "UVW1": ("UVW1", "swuw1_20041120v106.arf"),
    "UVM2": ("UVM2", "swum2_20041120v105.arf"),
    "UVW2": ("UVW2", "swuw2_20041120v105.arf"),
    "white": ("WHITE", "swuwh_20041120v104.arf"),
}
FTRANS = "bcf/ftrans/swuftrans20041120v102.fits"
MIRROR = "bcf/effarea/swumirrorea20010101v015.fits"
INSTEA = "bcf/effarea/swuinstea20041120v001.fits"
SENSCORR = "bcf/senscorr/swusenscorr20041120v007.fits"


def _fits(raw: bytes):
    from astropy.io import fits

    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return fits.open(io.BytesIO(raw))


def caldb_get(fetch: Fetcher, path: str, roots: Sequence[str] = CALDB_ROOTS) -> Tuple[bytes, str]:
    """Download a CALDB file from the first of ``roots`` that serves it; return (data, URL).

    Each root that fails is named in a warning when a later one serves the
    file, and in the error when none does.
    """
    errors = []
    for root in roots:
        url = root + path
        try:
            data = fetch.get(url)
        except (RequestException, ValueError) as exc:  # TLS, host and HTTP errors; HTML instead of data
            errors.append(f"{url}: {exc}")
            continue
        if not data.startswith(b"SIMPLE  ="):
            errors.append(f"{url}: not a FITS file")
            continue
        if errors:
            logger.warning("UVOT CALDB: %s; using %s", "; ".join(errors), url)
        return data, url
    raise RuntimeError("UVOT CALDB file unavailable: " + "; ".join(errors))


def _column_unit(table, name: str) -> str:
    return (table.columns[name].unit or "").strip().lower()


def arf_curve(raw: bytes, filt: str) -> Tuple[np.ndarray, np.ndarray]:
    """Bin-centre wavelength (Angstrom) and total effective area (cm^2) of an ARF."""
    with _fits(raw) as h:
        t = h[1]
        if t.header.get("FILTER", "").strip() != filt or t.header.get("CONTENT", "").strip() != "SPECRESP":
            raise ValueError(f"UVOT ARF: FILTER={t.header.get('FILTER')!r}, expected {filt!r} SPECRESP")
        for col, unit in (("WAVE_MIN", "angstrom"), ("WAVE_MAX", "angstrom"), ("SPECRESP", "cm**2")):
            if _column_unit(t, col) != unit:
                raise ValueError(f"UVOT ARF {filt}: {col} unit {_column_unit(t, col)!r}, expected {unit!r}")
        d = t.data
        wl = 0.5 * (np.asarray(d["WAVE_MIN"], float) + np.asarray(d["WAVE_MAX"], float))
        return wl, np.asarray(d["SPECRESP"], float)


def ftrans_curves(raw: bytes) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    with _fits(raw) as h:
        for filt, _arf in BANDS.values():
            t = h["TRANSMISSION" + filt]
            if t.header.get("FILTER", "").strip() != filt or _column_unit(t, "WAVELENGTH") != "angstrom":
                raise ValueError(f"UVOT ftrans: unexpected {t.name} header")
            out[filt] = (np.asarray(t.data["WAVELENGTH"], float), np.asarray(t.data["TRANSMIS"], float))
    return out


def area_curve(raw: bytes, extname: str) -> Tuple[np.ndarray, np.ndarray]:
    with _fits(raw) as h:
        t = h[extname]
        if _column_unit(t, "WAVELENGTH") != "angstrom" or _column_unit(t, "EA") != "cm**2":
            raise ValueError(f"UVOT {extname}: unexpected units")
        return np.asarray(t.data["WAVELENGTH"], float), np.asarray(t.data["EA"], float)


def senscorr_summary(raw: bytes) -> str:
    """The latest row of each filter's sensitivity-loss table, for the notes."""
    from astropy.time import Time

    parts = []
    with _fits(raw) as h:
        for filt, _arf in BANDS.values():
            t = h["SENSCORR" + filt]
            ref = t.header["MJDREFI"] + t.header["MJDREFF"]
            row = t.data[-1]
            start = Time(ref + float(row["TIME"]) / 86400.0, format="mjd", scale="tt").iso[:10]
            parts.append(f"{filt} 1+OFFSET={1 + float(row['OFFSET']):.4f}, SLOPE={float(row['SLOPE']):g} from {start}")
    return "; ".join(parts)


def _photon_mag(wl, resp, temperature):
    lam = wl * 1e-8
    f = 1.0 / lam**5 / np.expm1(1.4388 / (lam * temperature))
    return -2.5 * np.log10(np.trapezoid(f * resp * wl, wl) / np.trapezoid(resp * wl, wl))


def _composition_check(wl, published, composed) -> str:
    """Integral ratio and black-body (3000 K, 30000 K) magnitude shifts, composed minus published."""
    ratio = np.trapezoid(composed, wl) / np.trapezoid(published, wl)
    dmag = [_photon_mag(wl, composed, t) - _photon_mag(wl, published, t) for t in (3000.0, 30000.0)]
    return f"integral ratio {ratio:.4f}, dmag {dmag[0]:+.4f} (3000 K) {dmag[1]:+.4f} (30000 K)"


def uvot(fetch: Fetcher) -> Instrument:
    roots = list(CALDB_ROOTS)

    def get(path):
        # The root that served the last file is asked first: one warning, not one per file, when a host is down.
        data, url = caldb_get(fetch, path, roots)
        served = url[:-len(path)]
        roots.remove(served)
        roots.insert(0, served)
        return data, url

    raw_ft, url_ft = get(FTRANS)
    raw_mi, url_mi = get(MIRROR)
    raw_in, url_in = get(INSTEA)
    raw_sc, url_sc = get(SENSCORR)
    filters = ftrans_curves(raw_ft)
    wl_mi, ea_mi = area_curve(raw_mi, "MIRROREFFAREA")
    wl_in, ea_in = area_curve(raw_in, "INSTEFFAREA")

    bands, comps, checks = [], [], []
    for band, (filt, arf) in BANDS.items():
        raw_arf, url_arf = get("cpf/arf/" + arf)
        wl, area = arf_curve(raw_arf, filt)
        b = make_band(band, wl, area / GEOMETRIC_AREA_CM2, unit="Angstrom", detector_type=PHOTON,
                      components="filter+optics+qe", sources=f"{url_arf} | {POOLE} | {BREEVELD}")
        bands.append(b)
        fw, ft, _ = clean_curve(*filters[filt])
        comps.append(Component("filter", fw, ft, band=band,
                               source=f"{url_ft} extension TRANSMISSION{filt} (TRANSMIS, fraction)"))
        composed = np.interp(b.wavelength, fw, ft) * np.interp(b.wavelength, wl_in, ea_in) / GEOMETRIC_AREA_CM2
        checks.append(f"{band}: {_composition_check(b.wavelength, b.transmission, composed)}")
    mw, mt, _ = clean_curve(wl_mi, ea_mi / GEOMETRIC_AREA_CM2)
    comps.append(Component("optics", mw, mt, name="mirror",
                           source=f"{url_mi} (EA cm^2 / {GEOMETRIC_AREA_CM2:g} cm^2; on-axis mirror effective area)"))
    iw, it, _ = clean_curve(wl_in, ea_in / GEOMETRIC_AREA_CM2)
    comps.append(Component("optics+qe", iw, it, name="instrument",
                           source=f"{url_in} (EA cm^2 / {GEOMETRIC_AREA_CM2:g} cm^2; instrumental effective "
                                  f"area without filter, in-orbit, Poole et al. 2008 sect. 4)"))

    note = (
        "Swift/UVOT from the HEASARC CALDB (uvota; files fetched from "
        f"{url_arf.rsplit('/cpf/', 1)[0]}/). Published curves: current (CAL_QUAL=0 in caldb.indx 2024-02-01) "
        "ancillary response files cpf/arf/swu{vv,bb,uu}_20041120v104, swuw1_20041120v106, swum2_20041120v105, "
        "swuw2_20041120v105, swuwh_20041120v104 (SPECRESP = total effective area in cm^2 on 10 A energy bins; "
        "wavelength = midpoint of WAVE_MIN and WAVE_MAX). The UV ARFs are the end-2010 recalibration "
        f"(Breeveld et al. 2011, {BREEVELD}); the RSP matrices of the same release agree with these ARFs (checked 2026-10-26). "
        f"Units: every effective area (published curves, mirror and instrument components) is divided by the "
        f"{GEOMETRIC_AREA_CM2:g} cm^2 geometric collecting area of the primary mirror (Poole et al. 2008 sect. 4 "
        "(i)), so the curves are dimensionless photon-to-count throughputs relative to the light falling on "
        "659 cm^2 (U peaks at 0.075); a constant does not change synthetic photometry. Components: filter = "
        "bcf/ftrans/swuftrans20041120v102 TRANSMIS (fraction; lab measurements, except that UVW2 below 1800 A "
        "is an extrapolation and the white filter is a model, Poole et al. 2008 sect. 4); optics (file "
        "mirror) = bcf/effarea/swumirrorea20010101v015 on-axis mirror effective area / 659 cm^2, i.e. mirror "
        "reflectivity, tabulated only for 1610-6000 A (swueffarea20010101v014 holds identical data); "
        "optics+qe (file instrument) = bcf/effarea/swuinstea20041120v001, the in-orbit instrumental effective "
        "area without filter (geometric area x mirror reflectivity x detector quantum efficiency; flight data, "
        "Poole et al. 2008 sect. 4 and Fig. 1) / 659 cm^2, 1400-8000 A. No standalone detector QE is published, so none is derived. outside_range=error: "
        "the mirror curve does not cover V or white redward of 6000 A, so filter+optics is refused there. "
        "Composed filter x instrument vs published ARF (photon-weighted): " + "; ".join(checks) + ". "
        "(dmag: band-averaged photon-weighted F_lambda of a black body, so a constant scale cancels.) "
        "U, B and V are reproduced exactly; white differs by a constant (shape identical); the UV filters "
        "differ, most for cool sources through the red-leak tails, because the end-2010 ARF recalibration (Breeveld et al. 2011) did not update ftrans or "
        "instea, so composed UVW1, UVM2 and UVW2 are the 2007 (Poole et al. 2008) calibration and must not "
        "replace the published curves. DetectorType=photon: the UVOT detector is a microchannel-plate "
        "intensified CCD operated in photon-counting mode (Poole et al. 2008 sect. 2: 'it is photon "
        "counting'; abstract: 'the UVOT is a photon counting instrument'), and an effective area multiplies "
        "the photon flux to give a count rate. Time dependence NOT applied: the CALDB sensitivity-loss "
        f"correction {url_sc} multiplies observed count rates by (1+OFFSET)(1+SLOPE)^DT, DT = years since "
        "the row's TIME, one achromatic factor per filter; its last rows give " + senscorr_summary(raw_sc)
        + ". These curves are the mission-start response; scale count-rate predictions for later epochs "
        "by that factor. Not included: the magnifier and the UV and V grisms (not photometric bands).")
    return Instrument("Swift", "UVOT", bands, note, ground_based=False, components=comps,
                      outside_range="error")


RECIPES = {"UVOT": ("Swift", uvot)}

build = make_builder(RECIPES)
