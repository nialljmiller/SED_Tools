"""Optical/IR observatory imagers rebuilt from the instrument teams' own files.

ESO Paranal (VLT FORS1, FORS2; VST OmegaCAM), Keck (NIRC2, LRIS, MOSFIRE,
DEIMOS), KPNO Mosaic, Magellan (FourStar, MegaCam), NOT ALFOSC, Subaru
(Suprime-Cam, FOCAS) and WIYN ODI.  Bands whose original curve is unusable
(long-pass filters whose red edge is the unpublished CCD QE, mislabelled or
conflicting scans) are held, as the instrument notes explain.

Keck NIRC2 and LRIS I ship as Excel ``.xls`` workbooks and need ``xlrd``.
"""

from __future__ import annotations

import io
import re

import numpy as np

from ._original import Component, Fetcher, Instrument, clean_curve, make_band, make_builder
from .surveys import tar_members

PHOTON = "photon"


def _rows(text: str, cols=(0, 1)) -> np.ndarray:
    """Lines whose tokens are all numbers (title / column-name lines skipped)."""
    rows = []
    for ln in text.splitlines():
        try:
            vals = [float(x) for x in ln.split()]
        except ValueError:
            continue
        if len(vals) > max(cols):
            rows.append([vals[c] for c in cols])
    return np.array(rows)


def _first_two(text: str) -> np.ndarray:
    """First two numbers of every line that starts with two numbers."""
    rows = []
    for ln in text.splitlines():
        f = ln.split()
        if len(f) >= 2:
            try:
                rows.append((float(f[0]), float(f[1])))
            except ValueError:
                pass
    return np.array(rows)


def _pairs(text: str, pattern: str = r"[-+0-9.eE]+") -> np.ndarray:
    """Lines of exactly two tokens matching ``pattern``."""
    rows = []
    for ln in text.splitlines():
        f = ln.split()
        if len(f) == 2 and all(re.fullmatch(pattern, x) for x in f):
            rows.append((float(f[0]), float(f[1])))
    return np.array(rows)


# ======================================================================= ESO
FORS = "https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/"
FORS_PAGE = "https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/curves.html"
FORS_MAN117 = "https://www.eso.org/sci/facilities/paranal/instruments/fors/doc/VLT-MAN-ESO-13100-1543_P117_3.pdf"
FORS_CUT_NM = 1100.0  # long end of the FORS design range (manual P117 sec. 2.1)

# band -> (file, unit, percent)
FORS1 = {"U_BESS": ("M_BESS_U.txt", "nm", False), "B_BESS": ("M_BESS_B.txt", "nm", False),
         "V_BESS": ("M_BESS_V.txt", "nm", False), "R_BESS": ("M_BESS_R.txt", "nm", False),
         "I_BESS": ("M_BESS_I.txt", "nm", False), "u_GUNN": ("M_GUNN_U.txt", "nm", False),
         "v_GUNN": ("M_GUNN_V.txt", "nm", False), "g_GUNN": ("M_GUNN_G.txt", "nm", False),
         "r_GUNN": ("M_GUNN_R.txt", "nm", False)}
FORS2 = {"u_HIGH": ("u_HIGH.txt", "nm", True), "b_HIGH": ("b_HIGH+113COLL_SR.ascii", "Angstrom", False),
         "v_HIGH": ("v_HIGH+114COLL_SR.ascii", "Angstrom", False), "g_HIGH": ("g_HIGH.txt", "nm", True),
         "R_SPECIAL": ("R_SPECIAL+76COLL_SR.ascii", "Angstrom", False),
         "I_BESS": ("I_BESS+77COLL_SR.ascii", "Angstrom", False)}
FORS_DETAIL = {
    "FORS1": ("original FORS1 Bessell and Gunn set, optical-bench lab scans (M_*.txt); "
              "B/V/I_BESS existed in both FORS1 (+34/+35/+37) and FORS2 (+74/+75/+77), and the page does "
              "not say which unit was scanned. u/b/v/g_HIGH (FORS1 2007-2009) are in Paranal/FORS2."),
    "FORS2": ("current FORS2 standard configuration (manual P117 Table 2.1); u/g_HIGH are lab "
              "scans in %T (converted to fractions), b/v_HIGH, R_SPECIAL, I_BESS are the *COLL_SR.ascii tables."),
}


def fors_curve(name: str, fname: str, text: str, unit: str, percent: bool):
    """Wavelength (nm) and fraction of one FORS scan, cut at 1100 nm."""
    if fname in ("u_HIGH.txt", "g_HIGH.txt"):
        head = text.lower()
        if f"{name[0]}-fors" not in head or "%t" not in head:
            raise ValueError(f"{fname}: unexpected header")
    d = _rows(text)
    wl_nm = d[:, 0] / (10.0 if unit == "Angstrom" else 1.0)
    if not (250 < wl_nm.min() and wl_nm.max() < 1600):
        raise ValueError(f"{fname}: wavelength range {wl_nm.min()}-{wl_nm.max()} nm unexpected")
    keep = wl_nm <= FORS_CUT_NM
    tr = d[keep, 1] / (100.0 if percent else 1.0)
    if not 0.4 < tr.max() <= 1.0:
        raise ValueError(f"{fname}: peak {tr.max()} not a fraction")
    return wl_nm[keep], tr


def _fors(instrument: str, fetch: Fetcher) -> Instrument:
    table = {"FORS1": FORS1, "FORS2": FORS2}[instrument]
    bands = []
    for name, (fname, unit, percent) in table.items():
        wl, tr = fors_curve(name, fname, fetch.text(FORS + fname, "latin-1"), unit, percent)
        bands.append(make_band(name, wl, tr, unit="nm", detector_type=PHOTON, components="filter",
                               sources=f"{FORS + fname} | {FORS_PAGE}"))
    note = (f"VLT {instrument} broad-band filter-only transmission curves from the ESO FORS filter page "
            f"({FORS_PAGE}); {FORS_DETAIL[instrument]} Curves cut at 1100 nm (end of the FORS 330-1100 nm range, "
            "manual P117 sec. 2.1) to drop the scans' spurious 1200 nm point and the >1.1 um leak of the "
            "u/g_HIGH lab scans. No detector QE, optics or atmosphere included (ESO publishes no QE table). "
            "z_GUNN and H_Alpha held (see recipe c_eso.py). "
            f"Detector: CCD, photon counting ({FORS_MAN117}).")
    return Instrument("Paranal", instrument, bands, note, ground_based=True)


def fors1(fetch):
    return _fors("FORS1", fetch)


def fors2(fetch):
    return _fors("FORS2", fetch)


OCAM_URL = "https://www.eso.org/sci/facilities/paranal/instruments/omegacam/tools/OmegaCAM_transmission.tar.gz"
OCAM_PAGE = "https://www.eso.org/sci/facilities/paranal/instruments/omegacam/inst.html"
OCAM = {"u_SDSS": ["u"], "g_SDSS": ["g"], "r_SDSS": ["r"], "i_SDSS": ["i"], "z_SDSS": ["z"],
        "B_JOHN": [f"B-part{p}" for p in "ABCD"], "V_JOHN": [f"V-part{p}" for p in "ABCD"]}


def ocam_centre(text: str):
    """Wavelength (nm) and the filter-centre scan (data column 2, at x=y=0)."""
    lines = text.splitlines()
    xpos = next(l for l in lines if l.startswith("# x_pos")).split()[2:]
    ypos = next(l for l in lines if l.startswith("# y_pos")).split()[2:]
    if float(xpos[0]) != 0.0 or float(ypos[0]) != 0.0:
        raise ValueError("first scan is not at the filter centre")
    d = np.array([l.split() for l in lines if l.strip() and not l.startswith("#")], float)
    return d[:, 0], d[:, 1]


def omegacam(fetch: Fetcher) -> Instrument:
    wanted = [f"{s}_modified_transmission_data{c}" for ss in OCAM.values() for s in ss
              for c in ("", "_with_ccd")]
    text = tar_members(fetch.get(OCAM_URL), set(wanted), flat=True)
    bands, comps = [], []
    for name, segs in OCAM.items():
        curves = {}
        for c in ("", "_with_ccd"):
            scans = [ocam_centre(text[f"{s}_modified_transmission_data{c}"]) for s in segs]
            wl = scans[0][0]
            if any(not np.array_equal(wl, w) for w, _ in scans):
                raise ValueError(f"{name}: segment wavelength grids differ")
            if not (295 < wl.min() < 305 and 1195 < wl.max() < 1205):
                raise ValueError(f"{name}: wavelength range not 300-1200 nm")
            curves[c] = (wl, np.mean([t for _, t in scans], axis=0))
        member = " + ".join(f"{s}_modified_transmission_data_with_ccd" for s in segs)
        how = "centre scan" if len(segs) == 1 else "mean of the 4 segment-centre scans"
        bands.append(make_band(name, *curves["_with_ccd"], unit="nm", detector_type=PHOTON,
                               components="filter+qe",
                               sources=f"{OCAM_URL}#{member} ({how}) | {OCAM_PAGE}"))
        fb = make_band(name, *curves[""], unit="nm", detector_type=PHOTON, components="filter",
                       sources=OCAM_URL)
        comps.append(Component("filter", fb.wavelength, fb.transmission, band=name,
                               source=f"{OCAM_URL}#{member.replace('_with_ccd', '')} ({how})"))
    note = ("VST OmegaCAM ugriz (SDSS, monolithic) and Johnson B, V (segmented) from ESO's "
            "OmegaCAM_transmission.tar.gz (2011): lab scans at the filter centre multiplied by the CCD QE "
            "(*_with_ccd members); B/V are the mean of their 4 segment-centre scans. Filter-only scans "
            "attached as filter components. No telescope mirrors, corrector or atmosphere included. "
            "Negative scan noise set to 0. Detector: 32 E2V 44-82 CCDs, photon counting (" + OCAM_PAGE + ").")
    return Instrument("Paranal", "OmegaCAM", bands, note, ground_based=True, components=comps)


# ====================================================================== Keck
KECK = "http://www2.keck.hawaii.edu/inst/"  # http only: the https certificate fails


def _xlrd():
    try:
        import xlrd
    except ImportError:  # a RuntimeError lets FilterLibrary fall back to other sources
        raise RuntimeError("Reading Keck .xls filter workbooks needs xlrd (pip install xlrd)") from None
    return xlrd


def _half_points(wl, tr):
    i = np.where(tr >= tr.max() / 2)[0]
    return wl[i[0]], wl[i[-1]]


def check_width(name, wl, tr, centre, fwhm, tol=0.03):
    """The 50 % points must reproduce the instrument page's centre and width."""
    lo, hi = _half_points(wl, tr)
    c, w = (lo + hi) / 2, hi - lo
    if abs(c / centre - 1) > tol or abs(w / fwhm - 1) > tol:
        raise ValueError(f"{name}: 50% points {lo:.1f}-{hi:.1f} A (centre {c:.1f}, FWHM {w:.1f}) "
                         f"vs page centre {centre} FWHM {fwhm}")


NIRC2_PAGE = KECK + "nirc2/filters.html"
NIRC2_XLS = KECK + "nirc2/filter_passbands.xls"
NIRC2 = {"J": (1.248, 0.163), "H": (1.633, 0.296), "K": (2.196, 0.336), "Ks": (2.146, 0.311),
         "Kp": (2.124, 0.351), "Lp": (3.776, 0.700), "Ms": (4.670, 0.241)}  # page centre, width (um)


def nirc2_sheet(book, name) -> np.ndarray:
    """(micron, percent) rows of one filter sheet of filter_passbands.xls."""
    sh = book.sheet_by_name(name)
    if [str(v) for v in sh.row_values(0)[:2]] != ["Wavelength", "Transmission"]:
        raise ValueError(f"NIRC2 sheet {name}: unexpected header")
    a = np.array([sh.row_values(r)[:2] for r in range(1, sh.nrows)], float)
    if not (0.9 < a[:, 0].min() and a[:, 0].max() < 6 and 50 < a[:, 1].max() <= 100):
        raise ValueError(f"NIRC2 {name}: not micron / percent")
    return a


def nirc2(fetch: Fetcher, book=None) -> Instrument:
    book = book or _xlrd().open_workbook(file_contents=fetch.get(NIRC2_XLS))
    bands = []
    for name, (c_um, w_um) in NIRC2.items():
        a = nirc2_sheet(book, name)
        b = make_band(name, a[:, 0], a[:, 1] / 100.0, unit="micron", detector_type=PHOTON,
                      components="filter", sources=f"{NIRC2_XLS}#{name} | {NIRC2_PAGE}")
        check_width(name, b.wavelength, b.transmission, c_um * 1e4, w_um * 1e4)
        bands.append(b)
    note = ("Keck NIRC2 broad-band filter transmission from the Keck spreadsheet filter_passbands.xls "
            "(micron, percent -> fraction). Filter only: no telescope or AO optics, no InSb QE, no "
            "atmosphere. The scan temperature is not stated; the curves reproduce the page's 50% "
            "points, which the page says are manufacturer room-temperature specs. Aladdin-3 InSb "
            "array (photon counting). z and Y have no numeric curve (PDF only); the narrow-band "
            "sheets disagree with the page's cold-measured passbands and are not included.")
    return Instrument("Keck", "NIRC2", bands, note, ground_based=True)


LRIS_PAGE = KECK + "lris/filters.html"
LRIS_DAT = {"g": "g_blue.dat", "B": "B.dat", "V": "V.dat", "R": "R.dat", "Rs": "Rs.dat"}
LRIS_I = KECK + "lris/data_files/I.xls"
LRIS = {"g": (4731, 1082), "B": (4377, 909), "V": (5473, 948), "R": (6417, 1185),
        "Rs": (6809, 1269), "I": (7599, 1225)}  # page lambda_eff, filter FWHM (A)


def lris_i_sheet(book) -> np.ndarray:
    """Vendor test report I.xls: 'center' column, (nm, percent) -> (A, fraction)."""
    sh = book.sheet_by_index(0)
    if not (str(sh.row_values(0)[0]).startswith("W.M. Keck Observatory")
            and sh.row_values(6)[1] == "center" and sh.row_values(10)[:2] == ["wl(nm)", "T%"]):
        raise ValueError("I.xls: unexpected layout")
    a = np.array([sh.row_values(r)[:2] for r in range(11, sh.nrows)], float)
    return np.column_stack([a[:, 0] * 10.0, a[:, 1] / 100.0])


def lris(fetch: Fetcher, book=None) -> Instrument:
    curves, srcs = {}, {}
    for name, fn in LRIS_DAT.items():
        url = KECK + "lris/data_files/" + fn
        a = np.loadtxt(io.StringIO(fetch.text(url)))
        curves[name], srcs[name] = (a[:, 0], a[:, 1]), url
    book = book or _xlrd().open_workbook(file_contents=fetch.get(LRIS_I))
    a = lris_i_sheet(book)
    curves["I"], srcs["I"] = (a[:, 0], a[:, 1]), LRIS_I + "#test data, center column"
    bands = []
    for name, (leff, fwhm) in LRIS.items():
        wl, tr = curves[name]
        if not (3000 < wl.min() and wl.max() < 11000 and 0.5 < tr.max() <= 1.0):
            raise ValueError(f"LRIS {name}: not Angstrom / fraction")
        b = make_band(name, wl, tr, unit="angstrom", detector_type=PHOTON, components="filter",
                      sources=f"{srcs[name]} | {LRIS_PAGE}")
        if name != "g":   # g: page FWHM 1082 A, manufacturer file 980 A (lambda_eff agrees)
            check_width(name, b.wavelength, b.transmission, leff, fwhm)
        bands.append(b)
    note = ("Keck LRIS imaging filters, filter-only transmission from the LRIS filter page. g is the blue-arm "
            "filter (g_blue.dat, manufacturer curve); B, V, R, Rs and I are red-arm filters. I is the "
            "vendor test report I.xls (centre position, nm, percent). No CCD QE, optics or "
            "atmosphere. Photon-counting CCDs (e2v blue, LBNL red). Page FWHM for g is 1082 A; the "
            "file gives 980 A. u' (no ASCII) and RG850 (truncated long-pass) are not included.")
    return Instrument("Keck", "LRIS", bands, note, ground_based=True)


MOS_PAGE = KECK + "mosfire/filters.html"
MOS_FILES = {"Y": "mosfire_Y.txt", "J": "mosfire_J.txt", "H": "mosfire_H.txt", "K": "mosfire_K.txt",
             "Ks": "mosfire_Ks.txt", "J2": "J2_center_corr.txt", "J3": "J3_center_corr.txt",
             "H1": "H1_center_corr.txt", "H2": "H2_center_corr.txt"}
MOS = {"Y": (1.048, 0.152), "J": (1.253, 0.200), "H": (1.637, 0.341), "K": (2.162, 0.483),
       "Ks": (2.147, 0.314), "J2": (1.181, 0.129), "J3": (1.288, 0.122), "H1": (1.556, 0.165),
       "H2": (1.709, 0.167)}  # page lambda_C, FWHM (um)


def mosfire(fetch: Fetcher) -> Instrument:
    bands = []
    for name, fn in MOS_FILES.items():
        url = KECK + "mosfire/data/filter/" + fn
        a = np.loadtxt(io.StringIO(fetch.text(url)), comments="#")
        if not (0.85 < a[:, 0].min() and a[:, 0].max() < 2.6 and 0.8 < a[:, 1].max() <= 1.0):
            raise ValueError(f"MOSFIRE {name}: not micron / fraction")
        c_um, w_um = MOS[name]
        b = make_band(name, a[:, 0], a[:, 1], unit="micron", detector_type=PHOTON, components="filter",
                      sources=f"{url} | {MOS_PAGE}")
        check_width(name, b.wavelength, b.transmission, c_um * 1e4, w_um * 1e4)
        bands.append(b)
    note = ("Keck MOSFIRE filter transmission curves from the MOSFIRE filter page (micron, fraction). "
            "Filter only: no telescope optics, H2RG QE or atmosphere. Y, J, H, K and Ks are the "
            "order-sorting filters; the page says K is not intended for imaging. J2, J3, H1 and H2 "
            "are the intermediate bands (*_center_corr files). mosfire_Y.txt's column header reads "
            "'xZ yZ', but its passband matches the page's Y. Teledyne Hawaii-2RG HgCdTe array "
            "(photon counting).")
    return Instrument("Keck", "MOSFIRE", bands, note, ground_based=True)


DEI_PAGE = KECK + "deimos/filter_list.html"
DEI_FILES = {"B": "Bfaber13norm.txt", "R": "RlrisDredonorm.txt", "I": "Ibess2norm.txt", "Z": "Zsdssredonorm.txt"}
DEI_PIVOT = {"B": 4416, "R": 6487, "I": 8390, "Z": 9085}   # page "Pivot Wavelength [A]"


def deimos_table(name: str, text: str):
    """Column 2 (THROUGHPUT) of a DEIMOS synphot table, after checking its header."""
    lines = text.splitlines()
    hdr = {ln.split()[0]: ln.split(None, 2)[2].strip() for ln in lines if ln[:1].isalpha()}
    if not (hdr["EXPR"].startswith("dmr,ccd_lot14,") and "angstroms" in hdr["WAVELENGTH"]):
        raise ValueError(f"DEIMOS {name}: unexpected synphot header")
    a = np.array([ln.split() for ln in lines if ln[:1] == " " and len(ln.split()) == 3], float)
    wl, thr, norm = a.T
    m = norm > 1e-3
    ratio = thr[m] / norm[m]
    if np.ptp(ratio) > 1e-4 * ratio.mean() or abs(thr.max() - float(hdr["TPEAK"])) > 1e-4:
        raise ValueError(f"DEIMOS {name}: column 3 is not column 2 / TPEAK")
    if abs(float(hdr["PIVWV"]) - DEI_PIVOT[name]) > 1:
        raise ValueError(f"DEIMOS {name}: header PIVWV {hdr['PIVWV']} != page {DEI_PIVOT[name]}")
    return wl, thr


def deimos(fetch: Fetcher) -> Instrument:
    bands = []
    for name, fn in DEI_FILES.items():
        url = KECK + "deimos/" + fn
        wl, thr = deimos_table(name, fetch.text(url))
        bands.append(make_band(name, wl, thr, unit="angstrom", detector_type=PHOTON,
                               components="filter+optics+qe", sources=f"{url}#col2 THROUGHPUT | {DEI_PAGE}"))
    note = ("Keck DEIMOS imaging filters: column 2 (THROUGHPUT) of the synphot tables linked from the "
            "DEIMOS filter list. The synphot EXPR is dmr * ccd_lot14 * filter glasses. These are "
            "theoretical glass filter curves x the MIT/LL CCD QE x 'dmr', presumably the DEIMOS imaging "
            "mirror (that name is not documented), so components are filter+optics+qe. No telescope and "
            "no atmosphere. Column 3 is not a filter-only curve: it is column 2 scaled to a peak of 1. "
            "The pivots equal the header PIVWV and the page values. V is not included, because its "
            "ASCII is the retired, scratched filter and the 2015 replacement has only a PDF.")
    return Instrument("Keck", "DEIMOS", bands, note, ground_based=True)


# ====================================================================== KPNO
KPNO_DOCS = "https://noirlab.edu/science/sites/default/files/media/archives/documents/"
KPNO_FILTERS = {  # band: (k number, file, regex the "Filter:" header line must match)
    "U": (1044, "k1044bp.txt", r"MOSA USOLID"),
    "B": (1002, "k1002bp_jul04.txt", r"Filter: K1002\b"),
    "V": (1003, "k1003bp_jul04.txt", r"Filter: K1003\b"),
    "R": (1004, "k1004bp_aug04.txt", r"Filter: K1004\b"),
    "I": (1005, "k1005bp_aug04.txt", r"Filter: K1005\b"),
    "r_SDSS": (1018, "k1018.txt", r"Filter: SDSS R\b"),
    "i_SDSS": (1019, "k1019.txt", r"Filter: SDSS I\b"),
}
KPNO_TITLES = {1044: "Us solid U", 1002: "B Harris", 1003: "V Harris", 1004: "R Harris",
               1005: "I Nearly-Mould", 1018: "r SDSS", 1019: "i SDSS"}


def kpno_scan(text: str, pattern: str) -> np.ndarray:
    """(A, percent) rows of a NOAO Lambda 9 scan, checking the filter line and Tmax."""
    text = text.replace("\r", "")
    if not text.startswith("== Filter Data - Lambda 9 Spectrophotometer =="):
        raise ValueError("not a KPNO Lambda 9 filter scan")
    if not re.search(pattern, text.splitlines()[3]):
        raise ValueError(f"filter line {text.splitlines()[3]!r} does not match {pattern}")
    a = _pairs(text, r"-?[0-9.]+")
    m = re.search(r"cwl=\s*([0-9.]+)A\s+fwhm=\s*([0-9.]+)A\s+Tmax=\s*([0-9.]+)%", text)
    if m is None:
        raise ValueError("no cwl/fwhm/Tmax header")
    if abs(a[:, 1].max() - float(m.group(3))) > 0.05:
        raise ValueError(f"data peak {a[:, 1].max()} != header Tmax {m.group(3)} %")
    return a


def mosaic(fetch: Fetcher) -> Instrument:
    bands = []
    for name, (k, fn, pat) in KPNO_FILTERS.items():
        url = KPNO_DOCS + fn
        try:
            a = kpno_scan(fetch.text(url, "latin-1"), pat)
        except ValueError as exc:
            raise ValueError(f"{url}: {exc}") from None
        bands.append(make_band(name, a[:, 0], a[:, 1] / 100.0, unit="Angstrom", detector_type=PHOTON,
                               components="filter",
                               sources=f"{url} | https://noirlab.edu/science/filters/kp{k}"))
    note = ("KPNO Mayall 4 m Mosaic filters, NOAO Lambda 9 lab scans from the NOIRLab KPNO filter "
            "pages (" + "; ".join(f"{b}=k{v[0]} {KPNO_TITLES[v[0]]} ({v[1]})" for b, v in KPNO_FILTERS.items())
            + "). Angstrom, percent -> fraction (data peak checked against header Tmax). Filter only: "
            "no telescope/corrector, CCD QE (photon-counting CCDs) or atmosphere. Lab scans at normal "
            "incidence; NOIRLab notes a small blue shift in the f/3.1 beam (~12 A for narrow bands). "
            "Negative noise values set to 0. Header cwl/fwhm (FWHM centre, not pivot) are in provenance. HELD: "
            "g SDSS k1017 (NOIRLab's kp1017 file is a different 9556 A filter) and 815_v1 k1026 "
            "(conflicting f/3.1 curves), and z SDSS k1020 (long-pass, T=0.97 where the scan ends at "
            "11000 A; its red edge is the CCD QE, which is not published as data).")
    return Instrument("KPNO", "Mosaic", bands, note, ground_based=True)


# ================================================================== Magellan
FS = "https://instrumentation.obs.carnegiescience.edu/FourStar/OPTICS/curves/"
FS_BANDS = {"J": "J", "H": "H", "Ks": "Ks", "J1": "J1", "J2": "J2", "J3": "J3",
            "Hshort": "Hshort", "Hlong": "Hlong", "NB118": "FS118", "NB209": "FS209"}


def _rolloff(wl_a):
    """Glass-absorption roll-off that plot.sm applies to Ks and NB209."""
    return 1.714 - 0.380 * wl_a / 1e4


def fourstar_rederive(stem: str, linked: np.ndarray):
    """Re-derive an HST_STD curve from the file linked on filters.html (as plot.sm does)."""
    if stem.startswith("FS"):
        wl, tr = linked[:, 0] * 10, linked[:, 1]
        if stem == "FS209":
            tr = tr / 100 * _rolloff(wl)
    else:
        wl, tr = linked[:, 0], linked[:, 1] * (_rolloff(linked[:, 0]) if stem == "Ks" else 1.0)
    return wl, tr


def fourstar(fetch: Fetcher) -> Instrument:
    bands = []
    for name, stem in FS_BANDS.items():
        a = _first_two(fetch.text(FS + f"HST_STD/{stem}.txt", "latin-1"))
        linked = f"{stem}_convolve.txt" if stem.startswith("FS") else f"{stem}_cam_optics.txt"
        wl, tr = fourstar_rederive(stem, _first_two(fetch.text(FS + linked, "latin-1")))
        dev = float(np.max(np.abs(np.interp(a[:, 0], wl, tr) - a[:, 1])))
        if dev > 1e-3:
            raise ValueError(f"FourStar {name}: HST_STD differs from re-derived curve by {dev}")
        bands.append(make_band(name, a[:, 0], a[:, 1], unit="Angstrom", detector_type=PHOTON,
                               components="filter+optics",
                               sources=f"{FS}HST_STD/{stem}.txt | {FS}{linked} | {FS}plot.sm"))
    note = ("Magellan Baade FourStar (4 x HAWAII-2RG, photon counting). Curves = Carnegie "
            "OPTICS/curves/HST_STD/<band>.txt, the team's synthetic-photometry inputs: filter "
            "transmission x 0.995^16 camera AR-surface losses (plot.sm: 'no detector, no atmos, no "
            "telescope'), with the glass-absorption roll-off (1.714-0.380 lambda_um) applied to Ks and "
            "NB209; NB118/NB209 (FS118/FS209) include their J2/Ks blocking filters. Each curve was "
            "re-derived from the *_cam_optics.txt / *_convolve.txt files linked on filters.html "
            "(max |diff| <= 1e-3, rounding). Components filter+optics (product only). No detector QE, "
            "telescope or atmosphere. Reference pivots from HST_STD/output.txt (team fwhm.sm; may "
            "include an ATRAN sky). SVO ids (LCO/FourStar.*) are unverified leads.")
    return Instrument("Magellan", "FourStar", bands, note, ground_based=True)


MC = "https://lweb.cfa.harvard.edu/~bmcleod/Megacam/Filters/"
MC_COLS = ["WAVE", "FLUX", "T", "QE", "NU", "STARToverNU", "EXTINCT", "THROUGHPUT", "R",
           "SOLGEL", "WINDOW", "TOTAL"]


def megacam_table(text: str) -> dict:
    """Columns of a tab-separated filterthroughput.mag<b> table (blank cells -> NaN)."""
    lines = text.splitlines()
    if lines[0].split("\t") != MC_COLS:
        raise ValueError(f"unexpected MegaCam columns {lines[0]!r}")
    rows = []
    for ln in lines[2:]:
        f = ln.split("\t") + [""] * len(MC_COLS)
        rows.append([float(x) if x.strip() else np.nan for x in f[:len(MC_COLS)]])
    a = np.array(rows)
    return {c: a[:, i] for i, c in enumerate(MC_COLS)}


def megacam(fetch: Fetcher) -> Instrument:
    bands, comps, ref = [], [], None
    for b in "ugriz":
        url = MC + f"filterthroughput.mag{b}"
        t = megacam_table(fetch.text(url, "latin-1"))
        prod = t["T"] * t["QE"] * t["EXTINCT"] * t["THROUGHPUT"]
        if np.nanmax(np.abs(prod - t["TOTAL"])) > 1e-5:
            raise ValueError(f"MegaCam {b}: TOTAL != T*QE*EXTINCT*THROUGHPUT")
        if b != "u":  # filter-only table published separately for g,r,i,z
            f = _first_two(fetch.text(MC + f"mag{b}.tab", "latin-1"))
            m = (t["WAVE"] >= f[0, 0]) & (t["WAVE"] <= f[-1, 0])
            if np.max(np.abs(np.interp(t["WAVE"][m], f[:, 0], f[:, 1]) - t["T"][m])) > 1e-6:
                raise ValueError(f"MegaCam {b}: T column differs from mag{b}.tab")
        if ref is None:
            ref = t
        else:
            for c in ("QE", "EXTINCT", "THROUGHPUT"):
                if not np.allclose(t[c], ref[c]):
                    raise ValueError(f"MegaCam {b}: {c} differs between tables")
        bands.append(make_band(b, t["WAVE"], t["TOTAL"], unit="Angstrom", detector_type=PHOTON,
                               components="filter+optics+qe+atmosphere", sources=f"{url} | {MC}"))
        fw, ft, _ = clean_curve(t["WAVE"], t["T"])
        comps.append(Component("filter", fw, ft, band=b, source=url + " column T"))
    url_g = MC + "filterthroughput.magg"
    comps.append(Component("optics", ref["WAVE"], ref["THROUGHPUT"], source=url_g + " column THROUGHPUT"))
    comps.append(Component("qe", ref["WAVE"], ref["QE"], source=url_g + " column QE"))
    note = ("Magellan Clay MegaCam (36 e2v CCD42-90, photon counting; McLeod et al. 2015 PASP 127 "
            "366). Published curve = TOTAL column of Filters/filterthroughput.mag{u,g,r,i,z} = filter T "
            "x CCD QE x telescope/corrector throughput (2 Al mirrors, sol-gel and window surfaces) x "
            "atmosphere at 1.2 airmasses (product verified to 5e-7). Components: filter (T column; "
            "equals the filter-only mag{g,r,i,z}.tab), optics (THROUGHPUT) and qe (QE), the last two "
            "identical in all tables. The 1.2-airmass atmosphere (EXTINCT column) is recorded in "
            "provenance only (SED_Tools instrument components are filter/optics/qe). Filter T peaks "
            "slightly above 1 in z (1.0014, measurement noise, kept); small negative T values (noise) set to 0 in the filter components. No published pivot. MMT-era "
            "MegaCam curves on the same page are not included.")
    return Instrument("Magellan", "MegaCam", bands, note, ground_based=True, components=comps)


# ======================================================================= NOT
NOT_BASE = "https://www.not.iac.es/instruments/filters/curves-ascii/"
NOT_FILTERS = {  # band: (NOT id, unit, percent?, header check)
    "U": (7, "nm", True, None), "B": (74, "nm", True, None), "V": (75, "nm", True, None),
    "R": (76, "nm", True, None), "i_int": (12, "nm", True, None),
    "up": (109, "Angstrom", False, "u' SDSS"), "gp": (120, "Angstrom", False, "g' SDSS"),
    "rp": (110, "Angstrom", False, "r' SDSS"), "ip": (111, "Angstrom", False, "i' SDSS"),
}
NOT_NAMES = {7: "U_Bes 362_60", 74: "B_Bes 440_100", 75: "V_Bes 530_80", 76: "R_Bes 650_130",
             12: "i_int 797_157", 109: "u'_SDSS 353_55", 120: "g'_SDSS 480_145",
             110: "r'_SDSS 618_148", 111: "i'_SDSS 771_171"}


def not_curve(fid: int, text: str, pct: bool, head) -> np.ndarray:
    if head:  # first line "filter #<id> [<band> SDSS]" (#111 omits the band label)
        first = text.splitlines()[0].strip()
        if not first.startswith(f"filter #{fid}") or first[len(f"filter #{fid}"):].strip() not in ("", head):
            raise ValueError(f"header {first!r} does not identify filter #{fid} {head}")
    a = _pairs(text)
    peak = a[:, 1].max()
    if pct != (peak > 1.5):
        raise ValueError(f"filter #{fid}: transmission scale unexpected (peak {peak})")
    return np.column_stack([a[:, 0], a[:, 1] / (100.0 if pct else 1.0)])


def alfosc(fetch: Fetcher) -> Instrument:
    bands = []
    for name, (fid, unit, pct, head) in NOT_FILTERS.items():
        url = f"{NOT_BASE}{fid}.txt"
        a = not_curve(fid, fetch.text(url, "latin-1"), pct, head)
        bands.append(make_band(name, a[:, 0], a[:, 1], unit=unit, detector_type=PHOTON, components="filter",
                               sources=f"{url} | https://www.not.iac.es/instruments/filters/filters.php"))
    note = ("NOT ALFOSC imaging filters, filter-only transmission from the NOT filter database "
            "(curves-ascii/<id>.txt): " + "; ".join(f"{b}=#{v[0]} {NOT_NAMES[v[0]]}" for b, v in NOT_FILTERS.items())
            + ". Bessell/interference-i: lab scans in nm and percent (converted to Angstrom, /100). "
            "SDSS u'g'r'i'z' (FASU standard set-up, ALFOSC/MOSCA): in-situ scans in ALFOSC with "
            "grisms (J. Naranen 2004-05), Angstrom and fraction. No telescope, CCD QE (e2v "
            "CCD231-42, photon counting) or atmosphere. Caveats: the u' scan starts at 3176 A with "
            "T~0.1-0.19 (blue wing truncated by the measurement range). HELD: z' #112, a long-pass whose "
            "scan ends at 10988 A with T=0.33 (red edge set by the CCD QE, not included). Negative noise values set to 0. No published pivot (the "
            "database gives centre/FWHM only).")
    return Instrument("NOT", "ALFOSC", bands, note, ground_based=True)


# ==================================================================== Subaru
SC = "https://subarutelescope.org/Archive/Instruments/SCam/"
SC_PAGE = SC + "sensitivity.html"
SC_CCD = SC + "ccd.html"
SC_BANDS = {"B": "B", "V": "V", "Rc": "R", "Ic": "I", "g": "g", "r": "r", "i": "i", "z": "z", "Y": "Y"}
FO = "https://www.naoj.org/Instruments/FOCAS/camera/"
FO_PAGE = FO + "filters.html"
FO_CCD = "https://www.naoj.org/Instruments/FOCAS/ccdinfo.html"
FO_BANDS = ("U", "B", "V", "R", "I")


def subaru_table(text: str, lo, hi, peak_lo, peak_hi, what) -> np.ndarray:
    """Two numeric columns (every non-blank line must parse) within the expected ranges."""
    d = np.array([ln.split()[:2] for ln in text.splitlines() if ln.strip()], float)
    if not (lo <= d[:, 0].min() and d[:, 0].max() <= hi):
        raise ValueError(f"{what}: wavelength range {d[:, 0].min()}-{d[:, 0].max()} outside {lo}-{hi}")
    if not peak_lo <= d[:, 1].max() <= peak_hi:
        raise ValueError(f"{what}: peak {d[:, 1].max()} outside {peak_lo}-{peak_hi}")
    return d


def suprimecam(fetch: Fetcher) -> Instrument:
    bands = []
    for name, f in SC_BANDS.items():
        url = f"{SC}txt/{f}.txt"
        d = subaru_table(fetch.text(url, "latin-1"), 3000, 12000, 0.8, 1.0, url)   # Angstrom, fraction
        bands.append(make_band(name, d[:, 0], d[:, 1], unit="Angstrom", detector_type=PHOTON,
                               components="filter", sources=f"{url} | {SC_PAGE}"))
    qe_url = f"{SC}txt/QE_Hamamatsu.txt"
    qe = subaru_table(fetch.text(qe_url, "latin-1"), 300, 1100, 0.5, 1.0, qe_url)  # nm, fraction
    comps = [Component("qe", qe[:, 0] * 10.0, qe[:, 1], band="*",
                       source=f"{qe_url} (Hamamatsu S10892-01 FDCCD QE, {SC_CCD})")]
    note = ("Subaru Suprime-Cam general broad-band filters (B, V, Rc, Ic, SDSS g' r' i' z', Y): filter-only "
            f"transmission lists from {SC_PAGE}. The page advises applying the CCD QE, Primary Focus Unit "
            "transmittance and primary-mirror reflectivity separately; the Hamamatsu FDCCD QE (2008-2017 "
            "detectors) is attached as a qe component (360-1040 nm). Pre-2008 MIT/LL CCD QE not included. "
            f"Detector: CCD, photon counting ({SC_CCD}).")
    return Instrument("Subaru", "SuprimeCam", bands, note, ground_based=True, components=comps)


def focas(fetch: Fetcher) -> Instrument:
    bands = []
    for name in FO_BANDS:
        url = f"{FO}txt/{name}.txt"
        d = subaru_table(fetch.text(url, "latin-1"), 290, 1001, 50, 100, url)     # nm, percent
        bands.append(make_band(name, d[:, 0], d[:, 1] / 100.0, unit="nm", detector_type=PHOTON,
                               components="filter", sources=f"{url} | {FO_PAGE}"))
    note = ("Subaru FOCAS broad-band U, B, V, R, I filter-only transmission lists from "
            f"{FO_PAGE} (percent converted to fractions). No QE, optics or atmosphere included. "
            f"Detector: CCD (Hamamatsu fully-depleted), photon counting ({FO_CCD}).")
    return Instrument("Subaru", "FOCAS", bands, note, ground_based=True)


# ====================================================================== WIYN
ODI = "https://www.wiyn.org/ODI/Observe/"
ODI_PAGE = ODI + "wiynodifilters.html"
ODI_FILES = {"u": ("odi_u.dat", "Angstrom", 1.0), "r": ("odi_r.txt", "Angstrom", 1.0),
             "NB695": ("odi_NB695.txt", "Angstrom", 1.0), "NB422": ("NB_422.txt", "nm", 100.0)}


def odi_leak(wl, tr) -> float:
    """Fraction of the photon-weighted response above 10500 A (past the CCD cutoff)."""
    w = np.clip(tr, 0, None) * wl
    m = wl > 10500
    return float(np.trapezoid(w[m], wl[m]) / np.trapezoid(w, wl))


def odi(fetch: Fetcher) -> Instrument:
    bands = []
    for name, (fn, unit, scale) in ODI_FILES.items():
        text = fetch.text(ODI + fn, "latin-1")
        a = _first_two(text)
        peak = a[:, 1].max()
        if (scale == 100.0) != (peak > 1.5):
            raise ValueError(f"{fn}: transmission scale unexpected (peak {peak})")
        if fn == "NB_422.txt" and not text.startswith("Wavelength[nm]\tT%"):
            raise ValueError(f"{fn}: unexpected header")
        b = make_band(name, a[:, 0], a[:, 1] / scale, unit=unit, detector_type=PHOTON, components="filter",
                      sources=f"{ODI}{fn} | {ODI_PAGE}")
        leak = odi_leak(b.wavelength, b.transmission)
        if round(leak, 4) > 0.02:
            raise ValueError(f"ODI {name}: {leak:.0%} of the response is above 10500 A")
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
    return Instrument("WIYN", "ODI", bands, note, ground_based=True)


RECIPES = {
    "FORS1": ("Paranal", fors1),
    "FORS2": ("Paranal", fors2),
    "OmegaCAM": ("Paranal", omegacam),
    "NIRC2": ("Keck", nirc2),
    "LRIS": ("Keck", lris),
    "MOSFIRE": ("Keck", mosfire),
    "DEIMOS": ("Keck", deimos),
    "Mosaic": ("KPNO", mosaic),
    "FourStar": ("Magellan", fourstar),
    "MegaCam": ("Magellan", megacam),
    "ALFOSC": ("NOT", alfosc),
    "SuprimeCam": ("Subaru", suprimecam),
    "FOCAS": ("Subaru", focas),
    "ODI": ("WIYN", odi),
}

build = make_builder(RECIPES)
