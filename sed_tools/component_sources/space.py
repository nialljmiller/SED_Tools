"""Space missions rebuilt from the original archive files.

Gaia DR2 and DR2 revised passbands (ESA), Herschel PACS/SPIRE (HSA legacy
ADP), Hipparcos/Tycho (Bessell & Murphy 2012), IRAS (Explanatory Supplement),
MSX (IRSA RSR tables), Planck HFI/LFI (PR2 RIMO), Spitzer IRAC/MIPS/IRS peak-up
(IRSA/SSC) and XMM-Newton OM (HEASARC effective areas).
"""

from __future__ import annotations

import html
import io
import re
import zipfile

import numpy as np

from ._original import Component, Fetcher, Instrument, fixed_width_pairs, make_band, make_builder, open_fits

PHOTON, ENERGY = "photon", "energy"


# ---------------------------------------------------------------------- Gaia
GAIA_ZIPS = {
    "DR2": ("https://www.cosmos.esa.int/documents/29201/1645651/GaiaDR2_Passbands_ZeroPoints.zip/"
            "49cdce41-8eee-655d-7ed2-4e7a83598c1d", "GaiaDR2_Passbands_ZeroPoints.zip", "GaiaDR2_Passbands.dat"),
    "DR2r": ("https://www.cosmos.esa.int/documents/29201/1645651/GaiaDR2_Revised_Passbands_ZeroPoints.zip/"
             "54db454f-69cb-ea0c-15be-f1b1f597f191", "GaiaDR2_Revised_Passbands_ZeroPoints.zip",
             "GaiaDR2_RevisedPassbands.dat"),
}
GAIA_PAGE = "https://www.cosmos.esa.int/web/gaia/iow_20180316"


def gaia_passbands(raw: bytes, member: str) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return np.loadtxt(io.StringIO(zf.read(member).decode("utf-8")))


def _gaia(inst: str, fetch: Fetcher) -> Instrument:
    url, zname, dat = GAIA_ZIPS[inst]
    a = gaia_passbands(fetch.get(url), dat)
    # 99.99 = undefined (ReadMe note 1), dropped as sentinel.
    bands = [make_band(name, a[:, 0], a[:, col], unit="nm", detector_type=PHOTON,
                       components="filter+optics+qe", sources=f"{url} | {GAIA_PAGE}", sentinel=99.99)
             for name, col in (("G", 1), ("BP", 3), ("RP", 5))]
    which = ("passbands used to generate the released DR2 magnitudes and astrophysical parameters"
             if inst == "DR2" else "revised passbands based on the latest nominal instrument knowledge")
    note = (f"Gaia DR2 {which} (Evans et al. 2018, A&A 616, A4; ESA zip {zname}). Wavelength nm; "
            "99.99 = undefined (ReadMe note 1), dropped as sentinel. Whole-instrument (telescope, "
            "optics, CCD QE) photon-counting transmissivities for CCD photometry -> DetectorType=photon. "
            "Uncertainty columns not used. Distinct from GAIA/GAIA (EDR3/DR3, Gaia2m BP bright/faint).")
    return Instrument("GAIA", inst, bands, note, ground_based=False)


def gaia_dr2(fetch):
    return _gaia("DR2", fetch)


def gaia_dr2r(fetch):
    return _gaia("DR2r", fetch)


# ------------------------------------------------------------------ Herschel
PACS = "http://archives.esac.esa.int/hsa/legacy/ADP/PACS/PACS-P_filter_curves/PACS_phot_filterTrans_{}.fits.gz"
SPIRE = "http://archives.esac.esa.int/hsa/legacy/ADP/SPIRE/SPIRE-P_filter_curves/SPIRE_phot_RSRF_14_3.fits"
HERSCHEL_INFO = "https://www.herschel.caltech.edu/page/instruments"


def pacs_curve(raw: bytes, key: str):
    with open_fits(raw) as h:
        t = h[1]
        if t.header.get("BAND") != key:
            raise ValueError(f"PACS: FITS BAND {t.header.get('BAND')!r}, expected {key!r}")
        if t.columns["wavelength"].unit != "um":
            raise ValueError("PACS: wavelength unit is not um")
        return np.asarray(t.data["wavelength"], float), np.asarray(t.data["transmission"], float)


def pacs(fetch: Fetcher) -> Instrument:
    bands = []
    for name, col, key in (("70", "blue", "BS"), ("100", "green", "BL"), ("160", "red", "R")):
        url = PACS.format(col)
        wl, tr = pacs_curve(fetch.get(url), key)
        bands.append(make_band(name, wl, tr, unit="micron", detector_type=ENERGY,
                               components="filter", sources=f"{url} | {HERSCHEL_INFO}"))
    note = ("PACS photometer filter transmission curves (HSA legacy ADP, from PacsCal_FM_FilterTransmission; "
            "FITS BAND keyword BS/BL/R = 70/100/160 um). Filter only: telescope, optics and bolometer "
            "absorption not included. DetectorType=energy because PACS photometry uses bolometer arrays "
            "(power detectors). Native transmission scale kept.")
    return Instrument("Herschel", "PACS", bands, note, ground_based=False)


def spire_rsrf(raw: bytes):
    with open_fits(raw) as h:
        t = h["rsrf"]
        if t.columns["frequency"].unit != "GHz":
            raise ValueError("SPIRE: frequency unit is not GHz")
        return (np.asarray(t.data["frequency"], float),
                {k: np.asarray(t.data[k], float) for k in ("psw", "pmw", "plw")})


def spire(fetch: Fetcher) -> Instrument:
    nu, cols = spire_rsrf(fetch.get(SPIRE))
    bands = [make_band(name, nu, cols[k], unit="GHz", detector_type=ENERGY,
                       components="filter+optics+qe", sources=f"{SPIRE} | {HERSCHEL_INFO}", normalize=True)
             for name, k in (("250", "psw"), ("350", "pmw"), ("500", "plw"))]
    note = ("SPIRE photometer Relative Spectral Response Functions (SCalPhotRsrf v3, FM; HSA legacy ADP "
            "SPIRE_phot_RSRF_14_3.fits, ext 'rsrf'), PSW/PMW/PLW = 250/350/500 um. Frequency grid in GHz "
            "converted to Angstrom and sorted. Instrument RSRF (filters, feedhorn/optics, bolometer "
            "absorption); telescope and aperture efficiency not included. DetectorType=energy (bolometers). "
            "Relative response: tiny negative samples set to 0, then peak-normalised (normalize=True).")
    return Instrument("Herschel", "SPIRE", bands, note, ground_based=False)


# ----------------------------------------------------------------- Hipparcos
HIP_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/table2.dat"
HIP_PAPER = "https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B"
HIP_COLS = {"Hp": (0, 4, 5, 10), "BT": (11, 15, 16, 21), "VT": (22, 26, 27, 32)}
HIP_NOTE = ("Bessell & Murphy (2012) table 2 normalised photonic responses (CDS J/PASP/124/140), "
            "for photon-counting synthetic photometry. Wavelength column is 0.1 nm = Angstrom.")


def _hip_band(curves, name):
    wl, tr = curves[name]
    return make_band(name, wl, tr, unit="Angstrom", detector_type=PHOTON,
                     components="filter+optics+qe", sources=f"{HIP_URL} | {HIP_PAPER}")


def hipparcos(fetch: Fetcher) -> Instrument:
    curves = fixed_width_pairs(fetch.text(HIP_URL), HIP_COLS)
    return Instrument("Hipparcos", "Hipparcos", [_hip_band(curves, "Hp")], HIP_NOTE, ground_based=False)


def tycho(fetch: Fetcher) -> Instrument:
    curves = fixed_width_pairs(fetch.text(HIP_URL), HIP_COLS)
    return Instrument("Hipparcos", "Tycho", [_hip_band(curves, "BT"), _hip_band(curves, "VT")],
                      HIP_NOTE, ground_based=False)


# ---------------------------------------------------------------------- IRAS
IRAS_URL = "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/tabC5.html"
IRAS_EVID = ("https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/C4.html | "
             "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch6/C3.html")


def iras_tables(page: str) -> dict:
    """Strip tags; after each 'NN um Band' marker read rows of 4 numbers (wl, T, det, system)."""
    text = html.unescape(re.sub(r"<[^>]*>", " ", page))
    parts = re.split(r"(\d+)\s*\S{0,2}m\s+Band", text)
    out = {}
    for i in range(1, len(parts), 2):
        nums = [float(x) for x in re.findall(r"(?<![\w.])\d+\.\d+|(?<![\w.])\d+(?![\w.])", parts[i + 1])]
        if len(nums) % 4:
            raise ValueError(f"IRAS {parts[i]} um band: {len(nums)} numbers do not make rows of 4")
        out[parts[i]] = np.array(nums).reshape(-1, 4)
    return out


def iras(fetch: Fetcher) -> Instrument:
    tables = iras_tables(fetch.text(IRAS_URL, errors="replace", allow_html=True))
    if sorted(tables, key=int) != ["12", "25", "60", "100"]:
        raise ValueError(f"IRAS: unexpected bands {sorted(tables)}")
    bands, comps = [], []
    for name, t in tables.items():
        if not np.all(np.diff(t[:, 0]) > 0):
            raise ValueError(f"IRAS {name}: wavelengths not increasing")
        bands.append(make_band(name, t[:, 0], t[:, 3], unit="micron", detector_type=ENERGY,
                               components="filter+optics+qe", sources=f"{IRAS_URL} | {IRAS_EVID}",
                               normalize=True))
        comps.append(Component("filter+optics", t[:, 0] * 1e4, t[:, 1], band=name,
                               source=f"{IRAS_URL} (Transmission: filters and field lenses)"))
        comps.append(Component("qe", t[:, 0] * 1e4, t[:, 2] / t[:, 2].max(), band=name,
                               source=f"{IRAS_URL} (Relative Detector Response to constant energy input, peak-normalised)"))
    note = ("IRAS Explanatory Supplement Table II.C.5, 'Relative System Response' column as printed "
            "(including the 25 um 29.0 um row, whose detector value 0.19 is inconsistent with its "
            "system value 0.877; possible source typo, kept unchanged). DetectorType=energy: sect. II.C.4 "
            "Fig. II.C.9 caption gives the detector response 'to constant energy input', and sect. VI.C.3 "
            "eqs. VI.C.4-6 integrate f_nu*R over frequency with the nu*f_nu=const quoting convention. "
            "System response peak-normalised (normalize=True). Components: optical transmission (filters "
            "and field lenses, role filter+optics) and relative detector response (role qe, scaled to peak 1); "
            "pre-flight measurements of sample/spare parts.")
    return Instrument("IRAS", "IRAS", bands, note, ground_based=False, components=comps)


# ----------------------------------------------------------------------- MSX
MSX_URL = "https://irsa.ipac.caltech.edu/data/MSX/docs/rsr/msx_rsr_{}.tbl"
MSX_DESC = "https://irsa.ipac.caltech.edu/data/MSX/imageDescriptions.htm"


def msx_table(text: str) -> np.ndarray:
    head = [l for l in text.splitlines() if l.startswith("|")]
    if len(head) < 3 or "microns" not in head[2]:
        raise ValueError(f"MSX: unexpected IPAC header {head}")
    return np.loadtxt(io.StringIO(text), comments=("\\", "|"))


def msx(fetch: Fetcher) -> Instrument:
    bands = []
    for b in "ACDE":
        url = MSX_URL.format(b.lower())
        a = msx_table(fetch.text(url))
        bands.append(make_band(b, a[:, 0], a[:, 1], unit="micron", detector_type=ENERGY,
                               components="filter+optics+qe", sources=f"{url} | {MSX_DESC}", normalize=True))
    note = ("MSX SPIRIT III relative spectral response (RSR) curves, IRSA IPAC tables from the MSX Point "
            "Source Catalog V1.2 (Egan, Price et al. 1999); whole-system relative response. "
            "DetectorType=energy although the Si:As BIB detectors are photoconductors: the RSR is applied "
            "as an energy weighting. IRSA imageDescriptions gives images in in-band radiance (W m-2 sr-1) "
            "and radiance-to-Jy/sr factors for S_nu ~ nu^-1 (A 7.133e12, C 2.863e13, D 3.216e13, E 2.476e13). "
            "lambda_iso/(c*Int RSR/lambda dlambda) from these tables reproduces C/D/E to 4 digits and A to "
            "1.1%; the photon-weighted alternative is off by 0.3-4.6%. Peak-normalised (normalize=True; "
            "already 1). B1/B2 held: no RSR table in the IRSA rsr directory.")
    return Instrument("MSX", "MSX", bands, note, ground_based=False)


# -------------------------------------------------------------------- Planck
PLANCK_BASE = "https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/"
HFI = PLANCK_BASE + "HFI_RIMO_R2.00.fits"
LFI = PLANCK_BASE + "LFI_RIMO_R2.50.fits"
HFI_EV = "https://arxiv.org/abs/1101.2039"
LFI_EV = "https://arxiv.org/abs/1001.3321"


def planck_bandpass(hdul, ext: str, unit: str):
    t = hdul[ext]
    if t.columns[0].name != "WAVENUMBER" or t.columns[0].unit != unit:
        raise ValueError(f"Planck {ext}: first column {t.columns[0].name} [{t.columns[0].unit}]")
    return np.asarray(t.data.field(0), float), np.asarray(t.data.field(1), float)


def hfi(fetch: Fetcher) -> Instrument:
    bands = []
    with open_fits(fetch.get(HFI)) as h:
        for f in (100, 143, 217, 353, 545, 857):
            x, t = planck_bandpass(h, f"BANDPASS_F{f}", "1/cm")
            keep = x >= 1.0  # drop the far tail beyond 1 cm (< 1e-7 of the integral)
            bands.append(make_band(str(f), x[keep], t[keep], unit="1/cm", detector_type=ENERGY,
                                   components="filter+optics+qe", sources=f"{HFI} | {HFI_EV}", normalize=True))
    note = ("Planck HFI RIMO R2.00 (PR2) band-averaged spectral transmission, extensions BANDPASS_F100..F857 "
            "('weighted combination of the individual detector bandpasses'). Wavenumber in cm^-1 converted "
            "to Angstrom and sorted; the 60 samples below 1 cm^-1 (lambda > 1 cm, incl. 0 cm^-1; max transmission <= 4e-7, <= 5e-8 of the band integral) dropped; FLAG column (marks the "
            "in-band region) ignored, all samples kept. Ground-measured instrument response (feedhorns, "
            "filters, bolometers), telescope not included. DetectorType=energy (bolometers; band-averaged "
            "fluxes integrate tau(nu) I_nu dnu). Peak-normalised (already 1).")
    return Instrument("Planck", "HFI", bands, note, ground_based=False)


def lfi(fetch: Fetcher) -> Instrument:
    bands = []
    with open_fits(fetch.get(LFI)) as h:
        for f in (30, 44, 70):
            x, t = planck_bandpass(h, f"BANDPASS_{f:03d}", "GHz")
            bands.append(make_band(str(f), x, t, unit="GHz", detector_type=ENERGY,
                                   components="filter+optics+qe", sources=f"{LFI} | {LFI_EV}", normalize=True))
    # Notes as in the curated build (its wording predates including the 30 GHz band).
    note = ("Planck LFI RIMO R2.50 (PR2) band-averaged bandpasses, extensions BANDPASS_044/070 (030 held: "
            "its 7.5e7-1.4e8 A range exceeds the validator's 1e8 A plausibility limit) (the "
            "per-radiometer BANDPASS_0xx-NNS/M curves are not used). Column named WAVENUMBER is in GHz "
            "(TUNIT), converted to Angstrom and sorted. Modelled radiometer-chain response (feedhorn, OMT, "
            "HEMT amplifiers, filters, detector diode), telescope not included. DetectorType=energy (coherent "
            "radiometers measure power; band integrals of tau(nu) I_nu dnu). Peak-normalised (normalize=True).")
    return Instrument("Planck", "LFI", bands, note, ground_based=False)


# ------------------------------------------------------------------- Spitzer
IRAC = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/201125ch{}trans_full.txt"
MIPS = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/MIPSfiltsumm.txt"
MIPS_HB = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/6/"
IRS = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/{}PUtrans.txt"


def two_col(text: str, comment: str):
    wl, tr = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(comment):
            continue
        a, b = s.split()[:2]
        wl.append(float(a))
        tr.append(float(b))
    return np.array(wl), np.array(tr)


def mips_columns(text: str) -> dict:
    """Three (lambda, response) pairs side by side; shorter columns end early, so
    assign each token to a column by the character position where it ends."""
    num = re.compile(r"^-?\d+(\.\d*)?([eE][-+]?\d+)?$")
    data = [l for l in text.splitlines() if l.strip() and all(num.match(t) for t in l.split())]
    full = next(l for l in data if len(l.split()) == 6)
    ends, pos = [], 0
    for tok in full.split():
        pos = full.index(tok, pos) + len(tok)
        ends.append(pos)
    cols = [[] for _ in range(6)]
    for l in data:
        pos = 0
        for tok in l.split():
            pos = l.index(tok, pos) + len(tok)
            cols[int(np.argmin([abs(pos - e) for e in ends]))].append(float(tok))
    out = {}
    for name, k in (("24", 0), ("70", 2), ("160", 4)):
        if len(cols[k]) != len(cols[k + 1]):
            raise ValueError(f"MIPS {name}: column lengths differ")
        out[name] = (np.array(cols[k]), np.array(cols[k + 1]))
    return out


def irac(fetch: Fetcher) -> Instrument:
    # Header "microns  electrons / photon" -> photon; whole-system response.
    bands = []
    for ch in (1, 2, 3, 4):
        url = IRAC.format(ch)
        wl, tr = two_col(fetch.text(url, errors="replace"), "#")
        bands.append(make_band(f"I{ch}", wl, tr, unit="micron", detector_type=PHOTON,
                               components="filter+optics+qe", sources=url))
    note = ("IRAC full-array average spectral response curves (IRSA/SSC, 201125ch*trans_full). File header "
            "gives microns and electrons/photon, so DetectorType=photon; the response includes telescope, "
            "optics, filters and detector QE. Native e-/photon scale kept (not normalised).")
    return Instrument("Spitzer", "IRAC", bands, note, ground_based=False)


def mips(fetch: Fetcher) -> Instrument:
    # Handbook sect. 6: "R is the intrinsic spectral response of the detector, in units of e- per
    # unit energy"; the summary file header gives only 'lambda response' -> energy.
    cols = mips_columns(fetch.text(MIPS))
    bands = [make_band(n, wl, tr, unit="micron", detector_type=ENERGY, components="filter+qe",
                       sources=f"{MIPS} | {MIPS_HB}", normalize=True) for n, (wl, tr) in cols.items()]
    note = ("MIPS imaging filter bandpasses (MIPSfiltsumm.txt, 21 March 2000), combined detector plus filter "
            "spectral response. The MIPS Instrument Handbook (sect. 6) defines R in electrons per unit "
            "energy, so DetectorType=energy (the file header itself says only 'response'). Relative "
            "response: peak-normalised (normalize=True). 70 um is the NB-labelled column of the file. "
            "Telescope optics not stated as included.")
    return Instrument("Spitzer", "MIPS", bands, note, ground_based=False)


def irs(fetch: Fetcher) -> Instrument:
    # Header "RQExT ... multiply incident photons/second ... get electrons/second" -> photon.
    bands = []
    for name, col in (("PU_blue", "blue"), ("PU_red", "red")):
        url = IRS.format(col)
        wl, tr = two_col(fetch.text(url, errors="replace"), ";")
        bands.append(make_band(name, wl, tr, unit="micron", detector_type=PHOTON,
                               components="filter+qe", sources=url))
    note = ("IRS blue/red peak-up RQExT curves at 2.0 V (IRSA/SSC; same as SMART). Header: RQExT = detector "
            "responsive QE (QE x photoconductive gain) x filter transmission; multiplying photons/s gives "
            "electrons/s, so DetectorType=photon. Telescope optics not included. Native scale kept; "
            "negative noise samples set to 0.")
    return Instrument("Spitzer", "IRS", bands, note, ground_based=False)


# ----------------------------------------------------------------------- XMM
XMM_URL = "https://heasarc.gsfc.nasa.gov/FTP/xmm/data/responses/OM/OM_response_081008.txt"
XMM_COLS = {"UVW2": 1, "UVM2": 2, "UVW1": 3, "U": 4, "B": 5, "V": 6}


def xmm_om(fetch: Fetcher) -> Instrument:
    a = np.loadtxt(io.StringIO(fetch.text(XMM_URL)), comments="#")
    if a.ndim != 2 or a.shape[1] != 7:
        raise ValueError("XMM OM: expected 7 columns")
    bands = [make_band(n, a[:, 0], a[:, XMM_COLS[n]], unit="nm", detector_type=PHOTON,
                       components="filter+optics+qe", sources=XMM_URL, normalize=True)
             for n in ("V", "B", "U", "UVW1", "UVM2", "UVW2")]
    note = ("XMM-Newton OM effective area for all filters (HEASARC OM_response_081008.txt; header: nm, "
            "m^2 x 1000). Effective area includes telescope, filter and the photon-counting "
            "MCP-intensified detector, so DetectorType=photon. Peak-normalised (normalize=True), since "
            "the absolute area scale does not affect synthetic photometry. Zero-padded 100-700 nm grid "
            "kept as published (UV filters keep their red-leak tails).")
    return Instrument("XMM", "OM", bands, note, ground_based=False)


RECIPES = {
    "DR2": ("GAIA", gaia_dr2),
    "DR2r": ("GAIA", gaia_dr2r),
    "PACS": ("Herschel", pacs),
    "SPIRE": ("Herschel", spire),
    "Hipparcos": ("Hipparcos", hipparcos),
    "Tycho": ("Hipparcos", tycho),
    "IRAS": ("IRAS", iras),
    "MSX": ("MSX", msx),
    "HFI": ("Planck", hfi),
    "LFI": ("Planck", lfi),
    "IRAC": ("Spitzer", irac),
    "MIPS": ("Spitzer", mips),
    "IRS": ("Spitzer", irs),
    "OM": ("XMM", xmm_om),
}

build = make_builder(RECIPES)
