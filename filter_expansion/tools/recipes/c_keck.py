"""W. M. Keck Observatory: NIRC2, LRIS, MOSFIRE and DEIMOS imaging filters (gemini.csv, block C).

All curves come from the Keck instrument pages (http only; the https certificate fails here):

NIRC2  http://www2.keck.hawaii.edu/inst/nirc2/filters.html -> filter_passbands.xls
       One sheet per filter: Wavelength (micron), Transmission (PERCENT -> /100). Filter only.
       The scan temperature is not stated. The page says its broad-band specs are the
       manufacturer's room-temperature measurements, and the sheets reproduce those 50 %
       points (checked below). J H K Ks Kp Lp Ms built. The workbook has no z or Y sheets (PDF only).
       Its narrow-band sheets disagree with the page's cold NIRC2-measured passbands, so they are held.
       Examples: FeII centre +0.005 um; Br gamma 50 % band 2.1475-2.1676 vs 2.1523-2.1849 um.
       Detector: 1024x1024 Aladdin-3 InSb (http://www2.keck.hawaii.edu/inst/nirc2/) -> photon.
LRIS   http://www2.keck.hawaii.edu/inst/lris/filters.html
       g_blue.dat (blue arm; the manufacturer curve), B/V/R/Rs.dat (red arm): Angstrom, fraction.
       I.xls: vendor test report for the red I filter. Uses the 'center' column (wl nm, T %).
       All are filter only. The u' filter's ASCII is "TBD" on the page (SVO only), so it is held.
       Blue-arm B and V are also "ASCII-TBD", so the B and V built here are the red-arm filters.
       RG850 is held: it is a long-pass whose file ends at 9760 A with T = 0.83, and the band's
       red edge is set by the CCD QE, which is not published numerically.
       Detectors: e2v CCDs (blue arm), LBNL Mark IV CCD (red arm)
       (http://www2.keck.hawaii.edu/inst/lris/detectors.html) -> photon.
MOSFIRE http://www2.keck.hawaii.edu/inst/mosfire/filters.html -> data/filter/*.txt
       Micron, fraction, filter only. mosfire_Y.txt's column header reads "xZ yZ", but its
       range and 50 % points match the page's Y (1.048 / 0.152 um). J2/J3/H1/H2 are the
       "*_center_corr" files linked from the page. Detector: Teledyne Hawaii-2RG HgCdTe
       (http://www2.keck.hawaii.edu/inst/mosfire/detector.html) -> photon.
DEIMOS http://www2.keck.hawaii.edu/inst/deimos/filter_list.html -> *norm.txt
       The files are synphot tables with three columns: wavelength (A), THROUGHPUT, and a third
       column. Their header EXPR is "dmr,ccd_lot14,<filter glasses>", and PIVWV equals the page's
       pivot. The research row said column 3 is the "pure filter", but that is wrong. Column 3
       is exactly column 2 scaled to a peak of 1, so the ratio is constant (asserted below).
       Neither column is filter-only. The page's "two profiles" are the JPEG plots. Column 2 is
       built: the theoretical glass filter x CCD QE (ccd_lot14) x 'dmr', presumably the DEIMOS
       imaging mirror; that name is not documented. Components "filter+optics+qe".
       V is held: the page comments out the Vbess3redonorm row ("scratched, not recommended").
       The V filter in service since Dec 2015 has only a PDF.
       Detector: eight MIT/Lincoln Labs CCDs (http://www2.keck.hawaii.edu/inst/deimos/) -> photon.

No atmosphere or telescope in any curve. ref_pivot values are the page's centre (NIRC2, MOSFIRE:
midpoint of 50 % points), lambda_eff (LRIS: filter x QE, flat f_lambda) or pivot (DEIMOS).
For NIRC2, MOSFIRE and LRIS these are not strict pivots but agree to <1.5 %. The 50 % points
are also checked here against the page within 3 %; LRIS g is the exception (page FWHM 1082 A,
file 980 A, noted).
"""
import numpy as np
import xlrd
from fxlib import ORIGINAL, build_instrument, fetch, make_band

K = "http://www2.keck.hawaii.edu/inst/"
FAC = "Keck"


def _half_points(wl, tr):
    i = np.where(tr >= tr.max() / 2)[0]
    return wl[i[0]], wl[i[-1]]


def _check_width(name, wl, tr, centre, fwhm, tol=0.03):
    lo, hi = _half_points(wl, tr)
    c, w = (lo + hi) / 2, hi - lo
    if abs(c / centre - 1) > tol or abs(w / fwhm - 1) > tol:
        raise ValueError(f"{name}: 50% points {lo:.1f}-{hi:.1f} A (centre {c:.1f}, FWHM {w:.1f}) "
                         f"vs page centre {centre} FWHM {fwhm}")


# ---------------------------------------------------------------- NIRC2
NIRC2_PAGE = K + "nirc2/filters.html"
NIRC2_XLS = K + "nirc2/filter_passbands.xls"
# page: central wavelength, bandpass width (um)
NIRC2 = {"J": (1.248, 0.163), "H": (1.633, 0.296), "K": (2.196, 0.336), "Ks": (2.146, 0.311),
         "Kp": (2.124, 0.351), "Lp": (3.776, 0.700), "Ms": (4.670, 0.241)}


def _nirc2():
    path = fetch(NIRC2_XLS, FAC)
    book = xlrd.open_workbook(str(path))
    bands = []
    for name, (c_um, w_um) in NIRC2.items():
        sh = book.sheet_by_name(name)
        if [str(v) for v in sh.row_values(0)[:2]] != ["Wavelength", "Transmission"]:
            raise ValueError(f"NIRC2 sheet {name}: unexpected header")
        a = np.array([sh.row_values(r)[:2] for r in range(1, sh.nrows)], float)
        if not (0.9 < a[:, 0].min() and a[:, 0].max() < 6 and 50 < a[:, 1].max() <= 100):
            raise ValueError(f"NIRC2 {name}: not micron / percent")
        b = make_band(name, a[:, 0], a[:, 1] / 100.0, unit="micron", detector_type="photon",
                      detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                      sources=f"{NIRC2_XLS}#{name} | {NIRC2_PAGE}", svo_id=f"Keck/NIRC2.{name}",
                      ref_pivot=(c_um * 1e4, f"{NIRC2_PAGE} central wavelength (50% points midpoint)"))
        _check_width(name, b.wavelength, b.transmission, c_um * 1e4, w_um * 1e4)
        b.clean_note = "; ".join(x for x in ("percent /100", b.clean_note) if x)
        bands.append(b)
    note = ("Keck NIRC2 broad-band filter transmission from the Keck spreadsheet filter_passbands.xls "
            "(micron, percent -> fraction). Filter only: no telescope or AO optics, no InSb QE, no "
            "atmosphere. The scan temperature is not stated; the curves reproduce the page's 50% "
            "points, which the page says are manufacturer room-temperature specs. Aladdin-3 InSb "
            "array (photon counting). z and Y have no numeric curve (PDF only); the narrow-band "
            "sheets disagree with the page's cold-measured passbands and are not included.")
    build_instrument(FAC, "NIRC2", bands, notes=note, raw_files=[path], ground_based=True)


# ---------------------------------------------------------------- LRIS
LRIS_PAGE = K + "lris/filters.html"
LRIS_DAT = {"g": "g_blue.dat", "B": "B.dat", "V": "V.dat", "R": "R.dat", "Rs": "Rs.dat"}
LRIS_I = K + "lris/data_files/I.xls"
# page: lambda_eff (filter x CCD QE), FWHM of the filter (A)
LRIS = {"g": (4731, 1082), "B": (4377, 909), "V": (5473, 948), "R": (6417, 1185),
        "Rs": (6809, 1269), "I": (7599, 1225)}


def _lris():
    curves, raws, srcs = {}, [], {}
    for name, fn in LRIS_DAT.items():
        url = K + "lris/data_files/" + fn
        p = fetch(url, FAC)
        a = np.loadtxt(p)
        curves[name], srcs[name] = (a[:, 0], a[:, 1]), url
        raws.append(p)
    p = fetch(LRIS_I, FAC)
    sh = xlrd.open_workbook(str(p)).sheet_by_index(0)
    if not (str(sh.row_values(0)[0]).startswith("W.M. Keck Observatory")
            and sh.row_values(6)[1] == "center" and sh.row_values(10)[:2] == ["wl(nm)", "T%"]):
        raise ValueError("I.xls: unexpected layout")
    a = np.array([sh.row_values(r)[:2] for r in range(11, sh.nrows)], float)
    curves["I"], srcs["I"] = (a[:, 0] * 10.0, a[:, 1] / 100.0), LRIS_I + "#test data, center column"
    raws.append(p)
    bands = []
    for name, (leff, fwhm) in LRIS.items():
        wl, tr = curves[name]
        if not (3000 < wl.min() and wl.max() < 11000 and 0.5 < tr.max() <= 1.0):
            raise ValueError(f"LRIS {name}: not Angstrom / fraction")
        b = make_band(name, wl, tr, unit="angstrom", detector_type="photon",
                      detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                      sources=f"{srcs[name]} | {LRIS_PAGE}", svo_id=f"Keck/LRIS.{name}",
                      ref_pivot=(float(leff), f"{LRIS_PAGE} lambda_eff (filter x CCD QE, flat f_lambda)"))
        if name != "g":   # g: page FWHM 1082 A, manufacturer file 980 A (lambda_eff agrees)
            _check_width(name, b.wavelength, b.transmission, leff, fwhm)  # centre vs lambda_eff: loose
        if name == "I":
            b.clean_note = "; ".join(x for x in ("nm -> A, percent /100, vendor centre position", b.clean_note) if x)
        bands.append(b)
    note = ("Keck LRIS imaging filters, filter-only transmission from the LRIS filter page. g is the blue-arm "
            "filter (g_blue.dat, manufacturer curve); B, V, R, Rs and I are red-arm filters. I is the "
            "vendor test report I.xls (centre position, nm, percent). No CCD QE, optics or "
            "atmosphere. Photon-counting CCDs (e2v blue, LBNL red). Page FWHM for g is 1082 A; the "
            "file gives 980 A. u' (no ASCII) and RG850 (truncated long-pass) are not included.")
    build_instrument(FAC, "LRIS", bands, notes=note, raw_files=raws, ground_based=True)


# ---------------------------------------------------------------- MOSFIRE
MOS_PAGE = K + "mosfire/filters.html"
MOS_FILES = {"Y": "mosfire_Y.txt", "J": "mosfire_J.txt", "H": "mosfire_H.txt", "K": "mosfire_K.txt",
             "Ks": "mosfire_Ks.txt", "J2": "J2_center_corr.txt", "J3": "J3_center_corr.txt",
             "H1": "H1_center_corr.txt", "H2": "H2_center_corr.txt"}
# page: lambda_C, FWHM (um)
MOS = {"Y": (1.048, 0.152), "J": (1.253, 0.200), "H": (1.637, 0.341), "K": (2.162, 0.483),
       "Ks": (2.147, 0.314), "J2": (1.181, 0.129), "J3": (1.288, 0.122), "H1": (1.556, 0.165),
       "H2": (1.709, 0.167)}


def _mosfire():
    bands, raws = [], []
    for name, fn in MOS_FILES.items():
        url = K + "mosfire/data/filter/" + fn
        p = fetch(url, FAC)
        raws.append(p)
        a = np.loadtxt(p, comments="#")
        if not (0.85 < a[:, 0].min() and a[:, 0].max() < 2.6 and 0.8 < a[:, 1].max() <= 1.0):
            raise ValueError(f"MOSFIRE {name}: not micron / fraction")
        c_um, w_um = MOS[name]
        b = make_band(name, a[:, 0], a[:, 1], unit="micron", detector_type="photon",
                      detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                      sources=f"{url} | {MOS_PAGE}", svo_id=f"Keck/MOSFIRE.{name}",
                      ref_pivot=(c_um * 1e4, f"{MOS_PAGE} lambda_C (50% points midpoint)"))
        _check_width(name, b.wavelength, b.transmission, c_um * 1e4, w_um * 1e4)
        bands.append(b)
    note = ("Keck MOSFIRE filter transmission curves from the MOSFIRE filter page (micron, fraction). "
            "Filter only: no telescope optics, H2RG QE or atmosphere. Y, J, H, K and Ks are the "
            "order-sorting filters; the page says K is not intended for imaging. J2, J3, H1 and H2 "
            "are the intermediate bands (*_center_corr files). mosfire_Y.txt's column header reads "
            "'xZ yZ', but its passband matches the page's Y. Teledyne Hawaii-2RG HgCdTe array "
            "(photon counting).")
    build_instrument(FAC, "MOSFIRE", bands, notes=note, raw_files=raws, ground_based=True)


# ---------------------------------------------------------------- DEIMOS
DEI_PAGE = K + "deimos/filter_list.html"
DEI_FILES = {"B": "Bfaber13norm.txt", "R": "RlrisDredonorm.txt", "I": "Ibess2norm.txt", "Z": "Zsdssredonorm.txt"}
DEI_PIVOT = {"B": 4416, "R": 6487, "I": 8390, "Z": 9085}   # page "Pivot Wavelength [A]"


def _deimos():
    bands, raws = [], []
    for name, fn in DEI_FILES.items():
        url = K + "deimos/" + fn
        p = fetch(url, FAC)
        raws.append(p)
        lines = p.read_text().splitlines()
        hdr = {ln.split()[0]: ln.split(None, 2)[2].strip() for ln in lines if ln[:1].isalpha()}
        if not (hdr["EXPR"].startswith("dmr,ccd_lot14,") and "angstroms" in hdr["WAVELENGTH"]):
            raise ValueError(f"DEIMOS {name}: unexpected synphot header")
        a = np.array([ln.split() for ln in lines if ln[:1] == " " and len(ln.split()) == 3], float)
        wl, thr, norm = a.T
        m = norm > 1e-3
        ratio = thr[m] / norm[m]
        if np.ptp(ratio) > 1e-4 * ratio.mean() or abs(thr.max() - float(hdr["TPEAK"])) > 1e-4:
            raise ValueError(f"DEIMOS {name}: column 3 is not column 2 / TPEAK")
        pivwv = float(hdr["PIVWV"])
        if abs(pivwv - DEI_PIVOT[name]) > 1:
            raise ValueError(f"DEIMOS {name}: header PIVWV {pivwv} != page {DEI_PIVOT[name]}")
        bands.append(make_band(
            name, wl, thr, unit="angstrom", detector_type="photon", detector_source=ORIGINAL,
            components="filter+optics+qe", components_source=ORIGINAL,
            sources=f"{url}#col2 THROUGHPUT | {DEI_PAGE}", svo_id=f"Keck/DEIMOS.{name}",
            ref_pivot=(pivwv, f"{url} PIVWV (= {DEI_PAGE} pivot {DEI_PIVOT[name]} A)")))
    note = ("Keck DEIMOS imaging filters: column 2 (THROUGHPUT) of the synphot tables linked from the "
            "DEIMOS filter list. The synphot EXPR is dmr * ccd_lot14 * filter glasses. These are "
            "theoretical glass filter curves x the MIT/LL CCD QE x 'dmr', presumably the DEIMOS imaging "
            "mirror (that name is not documented), so components are filter+optics+qe. No telescope and "
            "no atmosphere. Column 3 is not a filter-only curve: it is column 2 scaled to a peak of 1. "
            "The pivots equal the header PIVWV and the page values. V is not included, because its "
            "ASCII is the retired, scratched filter and the 2015 replacement has only a PDF.")
    build_instrument(FAC, "DEIMOS", bands, notes=note, raw_files=raws, ground_based=True)


def run():
    _nirc2()
    _lris()
    _mosfire()
    _deimos()
