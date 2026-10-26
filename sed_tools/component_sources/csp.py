"""Carnegie Supernova Project (Las Campanas) passbands from the CSP team.

CSP-I (``LCO/CSP``): ``CSP_filter_package.tgz`` from the CSP filters page,
with the Swope + SITe CCD optical bands, Swope + RetroCam and du Pont + WIRC
near-infrared bands, the TAMU scans of the telescope and of the telescope
with the camera, and the raw (atmosphere-free) optical system scans.
CSP-II (``LCO/CSP2``): the Swope + e2v CCD and du Pont + RetroCam passbands
linked one by one from the same page (no component files are published).

The final passbands include the atmosphere at airmass 1.2 (package README);
the CSP atmosphere itself is registered separately as a site, not here.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import PurePosixPath
from typing import Dict, Tuple

import numpy as np

from ._original import Band, Component, Fetcher, Instrument, clean_curve, make_band, make_builder, safe_member

PHOTON = "photon"
PAGE = "https://csp.obs.carnegiescience.edu/data/filters"
DATA = "https://csp.obs.carnegiescience.edu/data/"
PACKAGE = DATA + "CSP_filter_package.tgz"
KRISCIUNAS = "https://ui.adsabs.harvard.edu/abs/2017AJ....154..211K"

#: CSP-I optical bands (Swope + SITe CCD): band -> file stem in the package.
CSP1_OPTICAL = {b: b for b in ("u", "g", "r", "i", "B", "V_LC3014", "V_LC3009", "V_LC9844")}
#: CSP-I near-infrared bands: band -> package file.  Y/Jrc1/Jrc2/H are Swope +
#: RetroCam, WIRC_* are du Pont + WIRC (README: *_SWO_* and *_DUP_* files).
CSP1_NIR = {
    "Y": "Y_SWO_TAM_scan_atm.dat", "Jrc1": "Jrc1_SWO_TAM_scan_atm.dat", "Jrc2": "Jrc2_SWO_TAM_scan_atm.dat",
    "H": "H_SWO_TAM_scan_atm.dat", "WIRC_Y": "Y_DUP_TAM_scan_atm.dat", "WIRC_J": "J_DUP_TAM_scan_atm.dat",
    "WIRC_H": "H_DUP_TAM_scan_atm.dat",
}
#: Date ranges as given in the CSP filters page table (M/D/Y).
CSP1_EPOCHS = {
    "u": "01/01/2003-10/2013", "g": "01/01/2003-10/2013", "r": "01/01/2003-10/2013", "i": "01/01/2003-10/2013",
    "B": "01/01/2003-10/2013", "V_LC3014": "01/01/2003-01/13/2006", "V_LC3009": "01/14/2006-01/24/2006",
    "V_LC9844": "01/25/2006-10/2013", "Y": "01/01/2003-07/2011", "Jrc1": "01/01/2003-01/15/2009",
    "Jrc2": "01/15/2009-07/2011", "H": "01/01/2003-07/2011", "WIRC_Y": "01/01/2003-07/2011",
    "WIRC_J": "01/01/2003-07/2011", "WIRC_H": "01/01/2003-07/2011",
}
#: CSP-II: band -> file on the CSP data page.
CSP2_FILES = {
    **{b: f"{b}_CSP2_tel_ccd_atm_ext_1.2.dat" for b in ("u", "g", "r", "i", "B", "V")},
    **{b: f"{b}_RetroCam_duPont_TAMU_atm.dat" for b in ("Y", "J", "H")},
}


def package_files(raw: bytes) -> Dict[str, bytes]:
    """Regular members of the filter package by base name (macOS ``._*`` resource forks skipped)."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            safe_member(m.name)
            name = PurePosixPath(m.name).name
            if not m.isfile() or name.startswith("._"):
                continue
            out[name] = tf.extractfile(m).read()
    return out


def two_columns(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(io.StringIO(data.decode("ascii")), comments="#", ndmin=2)
    if a.shape[1] != 2:
        raise ValueError(f"expected two columns, got {a.shape[1]}")
    return a[:, 0], a[:, 1]


def _support(wl, tr, frac=1e-3):
    above = np.nonzero(tr > frac * tr.max())[0]
    return wl[above[0]], wl[above[-1]]


def _atmosphere_check(files, band: str, raw_wl, raw_tr) -> str:
    """final / (raw scan x atm_ext_ctio^1.2 x atm_ext_lines) where the final exceeds 10 % of peak."""
    fw, ft = two_columns(files[f"{band}_tel_ccd_atm_ext_1.2.dat"])
    aw, at = two_columns(files["atm_ext_ctio.dat"])
    lw, lt = two_columns(files["atm_ext_lines.dat"])
    raw = np.interp(fw, raw_wl, raw_tr, left=np.nan, right=np.nan)
    m = (ft > 0.1 * ft.max()) & np.isfinite(raw) & (raw > 0)
    q = ft[m] / (raw[m] * np.interp(fw[m], aw, at) ** 1.2 * np.interp(fw[m], lw, lt))
    med = float(np.median(q))
    return f"{band} {med:.3f} (+-{float(np.std(q / med)):.3f})"


def csp1(fetch: Fetcher) -> Instrument:
    files = package_files(fetch.get(PACKAGE))
    src = f"{PACKAGE} | {PAGE}"
    bands, comps, checks, raw_notes = [], [], [], []
    tel_wl, tel_tr = two_columns(files["Telescope.txt"])          # header: "#Wavelength (nm)"
    tc_wl, tc_tr = two_columns(files["Telescope+Camera.txt"])     # no header; nm by range
    tel = clean_curve(tel_wl * 10.0, tel_tr)
    tc = clean_curve(tc_wl * 10.0, tc_tr)
    for band, stem in CSP1_OPTICAL.items():
        member = f"{stem}_tel_ccd_atm_ext_1.2.dat"
        wl, tr = two_columns(files[member])
        b = make_band(band, wl, tr, unit="Angstrom", detector_type=PHOTON,
                      components="filter+optics+qe+atmosphere", sources=f"{src} (member {member})")
        b.atmosphere = "airmass=1.2"
        bands.append(b)
        raw_member = f"{stem}_texas_WLcorr.txt"
        rw, rt = two_columns(files[raw_member])
        rw_c, rt_c, raw_note = clean_curve(rw, rt)
        if raw_note:
            raw_notes.append(f"{band}: {raw_note}")
        comps.append(Component("filter+optics+qe", rw_c, rt_c, band=band, name=band,
                               source=f"{PACKAGE} member {raw_member} (raw telescope+camera+filter scan, "
                                      "no atmosphere)"))
        checks.append(_atmosphere_check(files, stem, rw_c, rt_c))
        lo, hi = _support(b.wavelength, b.transmission)
        if tel[0][0] <= lo and tel[0][-1] >= hi:
            comps.append(Component("optics", tel[0], tel[1], band=band, name="telescope",
                                   source=f"{PACKAGE} member Telescope.txt (telescope throughput, relative scale)"))
        if tc[0][0] <= lo and tc[0][-1] >= hi:
            comps.append(Component("optics+qe", tc[0], tc[1], band=band, name="telescope_camera",
                                   source=f"{PACKAGE} member Telescope+Camera.txt (telescope and camera, "
                                          "relative scale)"))
    for band, member in CSP1_NIR.items():
        wl, tr = two_columns(files[member])
        b = make_band(band, wl, tr, unit="Angstrom", detector_type=PHOTON,
                      components="filter+optics+qe+atmosphere", sources=f"{src} (member {member})")
        b.atmosphere = "airmass=1.2"
        bands.append(b)
    with_tel = sorted({c.band for c in comps if c.name == "telescope"})
    clean = "; ".join(f"{b.name}: {b.clean_note}" for b in bands if b.clean_note)
    note = (
        "Carnegie Supernova Project I passbands from the CSP team's filter package "
        f"({PACKAGE}, README updated 03/16/2018; filters page {PAGE}; Krisciunas et al. 2017, {KRISCIUNAS}). "
        "TAMU scans (Rheault, Marshall, DePoy, Villanueva) with a 2.5 nm FWHM monochromator. Published curves "
        "are the README's final passbands, which 'have atmosphere corresponding to 1.2 airmass' (the README "
        "states this after listing the final optical *_tel_ccd_atm_ext_1.2.dat and NIR *_SWO_/*_DUP_"
        "TAM_scan_atm.dat files): Components filter+optics+qe+atmosphere, Atmosphere airmass=1.2. Optical "
        "bands u g r i B V_LC3014 V_LC3009 V_LC9844 are Swope + SITe CCD (README: 'Swope Site#3 CCD'; the "
        "page's table says 'Swope Site2 CCD'); Y Jrc1 Jrc2 H are Swope + RetroCam; WIRC_Y WIRC_J WIRC_H are "
        "du Pont + WIRC. Epochs (page table): "
        + ", ".join(f"{k} {v}" for k, v in CSP1_EPOCHS.items())
        + ". Three V filters: V_LC3014 (original, damaged), V_LC3009 (the README calls it 'V_LC_3099' and the "
        "page text 'LC-3099'; the file names and the page table say 3009), V_LC9844; two RetroCam J filters: "
        "Jrc1 (to 2009-01-15) and Jrc2 (from 2009-01-15). The page links the same files as V_tel_ccd_atm_"
        "ext_1.2.dat (= V_LC9844), J_old_retrocam_swope_atm.dat (= Jrc1) and J_SWO_TAM_atm.dat (= Jrc2), "
        "byte-identical to the package members; by those link names SVO's LCO/CSP keys V, J_old and J "
        "appear to be V_LC9844, Jrc1 and Jrc2 (not checked against SVO's curves). Wavelengths in Angstrom (the files have no header; the ranges match the page's effective "
        "wavelengths in Angstrom). Transmission kept on the published scale; negative scan noise set to 0"
        + (f" ({clean})" if clean else "") + ". Components (optical bands only, registered per band, never "
        "shared with the NIR bands): filter+optics+qe = the README's 'Raw optical Telescope+Camera+Filter "
        "scans' {band}_texas_WLcorr.txt (no atmosphere; sorted by wavelength"
        + (f"; {'; '.join(raw_notes)}" if raw_notes else "") + "); optics (file "
        "telescope) = Telescope.txt 'Telescope throughput' (header: nm" + (f"; {tel[2]}" if tel[2] else "")
        + "), for the bands it covers ("
        + " ".join(with_tel) + "; it ends at 800 nm, so not i); optics+qe (file telescope_camera) = "
        "Telescope+Camera.txt 'Telescope and Camera together' (no header; nm by range, 302-900 nm). Both are "
        "TAMU scans on an arbitrary relative scale (about 0.01), not absolute fractions; only their shape is "
        "meaningful. No filter-only curve is published, so none is derived and no recipe starting from "
        "'filter' is buildable. Check final / (raw x atm_ext_ctio.dat^1.2 x atm_ext_lines.dat), median "
        "(+-scatter) where the final exceeds 10 % of peak: " + ", ".join(checks)
        + "; i.e. the finals are the raw scans times this atmosphere at airmass 1.2 (atm_ext_ctio.dat read as "
        "the transmission at airmass 1, an inference), with V_LC3014 and V_LC3009 carrying an additional "
        "achromatic factor of about 0.93. The atmosphere files are not registered here. DetectorType=photon: "
        "the README defines zero points with F = (1/ch) Int S_X f_lambda lambda dlambda and the page calls F "
        "'the photon flux through filter S_X'. Not included: K_DUP_TAM_scan_atm.dat (the page says the K "
        "scan is 'still a work in progress' and the file stops at 23996 A at about 6 % of peak), kfilter (abandoned K, "
        "Contreras et al. 2010 guess), hfilter (undocumented) and the raw NIR *_texas_SWO/DUP.dat scans "
        "(the README calls them 'raw NIR filter scans' without saying whether they include the telescope "
        "and camera).")
    return Instrument("LCO", "CSP", bands, note, ground_based=True, components=comps, outside_range="error")


def csp2(fetch: Fetcher) -> Instrument:
    bands = []
    for band, name in CSP2_FILES.items():
        url = DATA + name
        wl, tr = two_columns(fetch.get(url))
        b = make_band(band, wl, tr, unit="Angstrom", detector_type=PHOTON,
                      components="filter+optics+qe+atmosphere", sources=f"{url} | {PAGE}")
        if "ext_1.2" in name:
            b.atmosphere = "airmass=1.2"
        bands.append(b)
    clean = "; ".join(f"{b.name}: {b.clean_note}" for b in bands if b.clean_note)
    note = (
        f"Carnegie Supernova Project II passbands, linked one by one from the CSP filters page {PAGE} "
        "(not part of CSP_filter_package.tgz). u g r i B V: Swope + e2v CCD, 10/2013-present, files "
        "{band}_CSP2_tel_ccd_atm_ext_1.2.dat; Y J H: du Pont + RetroCam, 07/2011-present, files "
        "{band}_RetroCam_duPont_TAMU_atm.dat (dates from the page table). TAMU scans of the telescope and "
        "camera with each filter (page text: 'also e2V later for CSPII', 'later Dupont + Retrocam'), "
        "including the atmosphere: the optical file names carry the package's 'atm_ext_1.2' (airmass 1.2) "
        "label; the NIR files say only 'atm', so their airmass is not stated (no Atmosphere header). "
        "Components filter+optics+qe+atmosphere; no separate components are published for CSP-II (the "
        "package's Telescope and Telescope+Camera scans are for the CSP-I SITe camera and are not reused). "
        "Wavelengths in Angstrom (no header; ranges match the page's effective wavelengths). Transmission "
        "kept on the published scale; negative scan noise set to 0" + (f" ({clean})" if clean else "")
        + ". DetectorType=photon: the page defines the zero points with F 'the photon flux through filter "
        "S_X'. Same band names as SVO's LCO/CSP2.")
    return Instrument("LCO", "CSP2", bands, note, ground_based=True, outside_range="error")


RECIPES = {"CSP": ("LCO", csp1), "CSP2": ("LCO", csp2)}

build = make_builder(RECIPES)
