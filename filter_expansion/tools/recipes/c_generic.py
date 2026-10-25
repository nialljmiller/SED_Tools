"""Generic facility (standard systems), block C1 (codex-c.csv / codex-c.md).

Generic/Bessell -- Bessell & Murphy (2012, PASP 124, 140), CDS J/PASP/124/140 table1.dat:
    normalised UBVRI *photonic* responses (the 2012 revised realization, NOT Bessell 1990).
    Fixed-width pairs per band, wavelength unit 0.1 nm = Angstrom (no factor 10). Same CDS
    catalogue and parsing as recipes/b_hipparcos.py (table2 there).  Abstract standard-system
    responses for photon-counting synthetic photometry; not separable into filter/optics/QE.
Generic/MKO -- MKO-NIR J H K L' M' (Tokunaga, Simons & Vacca 2002, PASP 114, 180): the IRTF
    NSFCam cold filter scans linked from https://irtfweb.ifa.hawaii.edu/~nsfcam2/Filter_Profiles.html
    (nsfcam_<b>mk_trans.dat).  Header says "Transmission (%)" (the landing page's "0 to 1" is
    wrong): divided by 100; microns -> Angstrom; sorted; L' duplicate wavelengths averaged;
    negative baseline noise set to 0.  Filter only (no QE, optics or atmosphere).  Checked
    numerically against the live mirror IRTF/NSFCam JMK/HMK/KMK/LPMK/MPMK (provenance).
Generic/USNO -- u'g'r'i'z' standard system (Smith et al. 2002, AJ 123, 2121): J. Gunn's
    2001-12-18 USNO 40-in + SITe CCD response files usno_[ugriz].res from
    https://www-star.fnal.gov/ugriz/Filters/response.html.  The live site answers HTTP 403
    (Cloudflare challenge), so the byte payload is taken from the Wayback Machine raw (id_)
    snapshots of 2021-03-19.  Columns lam(A), vac, air1.0, air1.3, air1.6, air1.9; column 4
    (1.3 airmasses, the natural-system median airmass per the readme) is used: filter + CCD QE
    + telescope + atmosphere.

HELD (not built): Johnson UBVRIJHKLMN, Cousins RI, Stromgren uvby + H-beta, Kron RI,
Thuan-Gunn uvgr, Bessell 1990 UBVRI -- only SVO (unreachable) or no numeric original.
"""
import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

FAC = "Generic"

# --- Bessell & Murphy 2012 -------------------------------------------------
B12_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/table1.dat"
B12_PAPER = "https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B"
# 0-based slices from the ReadMe byte description (bytes 1-4 / 6-10, 12-15 / 17-21, ...).
B12_COLS = {b: (11 * i, 11 * i + 4, 11 * i + 5, 11 * i + 10) for i, b in enumerate("UBVRI")}

# --- MKO (IRTF NSFCam) -----------------------------------------------------
MKO_URL = "https://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_{}_trans.dat"
MKO_PAGE = "https://irtfweb.ifa.hawaii.edu/~nsfcam2/Filter_Profiles.html"
MKO_PAPER = "https://ui.adsabs.harvard.edu/abs/2002PASP..114..180T"
MKO_DET = "https://irtfweb.ifa.hawaii.edu/~nsfcam2/Welcome.html"
MKO_ISO = "https://irtfweb.ifa.hawaii.edu/IRrefdata/iwafdv.html"
MKO = {"J": ("jmk", "JMK", 12500), "H": ("hmk", "HMK", 16440), "K": ("kmk", "KMK", 21980),
       "Lp": ("lpmk", "LPMK", 37540), "Mp": ("mpmk", "MPMK", 47020)}
MIRROR = "https://nillmill.ddns.net/sed_tools/filters/IRTF/NSFCam/{}.dat"

# --- USNO u'g'r'i'z' -------------------------------------------------------
USNO_ORIG = "https://www-star.fnal.gov/ugriz/Filters/usno_{}.res"
USNO_WB = {"u": "20210319150010", "g": "20210319132508", "r": "20210319053959",
           "i": "20210319071508", "z": "20210319124657"}
USNO_README = "https://www-star.fnal.gov/ugriz/Filters/readme.txt"
USNO_README_WB = "https://web.archive.org/web/20210319075936id_/" + USNO_README
USNO_PAGE = "https://www-star.fnal.gov/ugriz/Filters/response.html"
USNO_PAPER = "https://ui.adsabs.harvard.edu/abs/2002AJ....123.2121S"

HELD = {
    "Generic/Johnson U B V R I J H K L M N": "SVO-only numeric curves (SVO unreachable); LI/LII choice open",
    "Generic/Cousins R I": "SVO-only numeric curves (SVO unreachable)",
    "Generic/Stromgren u v b y": "SVO-only numeric curves (SVO unreachable)",
    "Generic/Stromgren beta_narrow beta_wide": "no numeric original (Crawford & Mander 1966)",
    "Generic/Kron R I": "no numeric original response (Kron & Smith 1951)",
    "Generic/ThuanGunn u v g r": "no numeric original response (Thuan & Gunn 1976)",
    "Generic/Bessell1990 U B V R I": "only a PDF table (Bessell 1990 Table 2); not transcribed",
}


def _bessell():
    path = fetch(B12_URL, FAC)
    readme = fetch(B12_URL.replace("table1.dat", "ReadMe"), FAC)
    if "Normalized UBVRI photonic responses" not in readme.read_text():
        raise ValueError("unexpected Bessell & Murphy ReadMe")
    curves = {b: ([], []) for b in B12_COLS}
    for line in path.read_text().splitlines():
        for b, (a, e, c, d) in B12_COLS.items():
            w, t = line[a:e].strip(), line[c:d].strip()
            if w and t:
                curves[b][0].append(float(w))
                curves[b][1].append(float(t))
    bands = [make_band(b, np.array(wl), np.array(tr), unit="Angstrom", detector_type="photon",
                       detector_source=ORIGINAL, components="filter+optics+qe",
                       components_source=ORIGINAL, sources=f"{B12_URL} | {B12_PAPER}",
                       svo_id=f"Generic/Bessell.{b}")
             for b, (wl, tr) in curves.items()]
    note = ("Bessell & Murphy (2012, PASP 124, 140) table 1 (CDS J/PASP/124/140): the 2012 revised "
            "UBVRI passbands, normalised PHOTONIC responses for photon-counting synthetic photometry. "
            "This is NOT the Bessell (1990, PASP 102, 1181) energy-response realization (held). "
            "Wavelength unit 0.1 nm = Angstrom. Abstract standard-system responses (whole system; "
            "not separable into filter/optics/QE); the ReadMe does not state whether atmospheric "
            "extinction is included.")
    build_instrument(FAC, "Bessell", bands, notes=note, raw_files=[path, readme], ground_based=True)


def _read_mirror(band):
    p = fetch(MIRROR.format(band), FAC, f"mirror_IRTF_NSFCam_{band}.dat")
    rows = [ln.split(",")[:2] for ln in p.read_text().splitlines()[1:] if ln.strip()]
    return np.array(rows, float), p


def _mko():
    bands, raws, check = [], [], {}
    for name, (stem, mirror_band, iso) in MKO.items():
        url = MKO_URL.format(stem)
        path = fetch(url, FAC)
        lines = path.read_text().splitlines()
        if lines[0].split("\t") != ["WL (Micrometers)", "Transmission (%)"]:
            raise ValueError(f"{url}: unexpected header {lines[0]!r}")
        data = np.array([ln.split() for ln in lines[1:] if ln.strip()], float)
        b = make_band(name, data[:, 0], data[:, 1] / 100.0, unit="micron", detector_type="photon",
                      detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                      sources=f"{url} | {MKO_PAGE} | {MKO_PAPER}",
                      ref_pivot=(float(iso), f"Tokunaga & Vacca 2005 isophotal wavelength (Vega), {MKO_ISO}"))
        b.clean_note = "; ".join(x for x in ("percent / 100", b.clean_note) if x)
        bands.append(b)
        raws.append(path)
        m, mp = _read_mirror(mirror_band)
        same = (len(m) == len(b.wavelength) and np.allclose(m[:, 0], b.wavelength, rtol=0, atol=1e-6)
                and np.allclose(m[:, 1], b.transmission, rtol=0, atol=1e-7))
        check[name] = {"mirror": MIRROR.format(mirror_band), "raw_samples": len(data),
                       "out_samples": len(b.wavelength), "mirror_samples": len(m),
                       "max_abs_diff": float(np.abs(m[:, 1] - b.transmission).max()) if len(m) == len(b.wavelength) else None,
                       "identical_within_1e-7": bool(same)}
        if not same:
            raise ValueError(f"MKO {name} differs from mirror {mirror_band}: {check[name]}")
    note = ("MKO-NIR J H K L' M' (Tokunaga, Simons & Vacca 2002, PASP 114, 180): IRTF NSFCam cold "
            f"filter scans ({MKO_PAGE}). FILTER ONLY: no detector QE, optics or atmosphere. Source "
            "header gives percent (divided by 100); microns converted to Angstrom; negative baseline "
            "noise set to 0; L' duplicate wavelengths averaged. Photon convention: NSFCam uses a "
            f"Hawaii-2RG IR array ({MKO_DET}). Numerically identical (atol 1e-7) to the mirror's "
            "IRTF/NSFCam JMK/HMK/KMK/LPMK/MPMK, which come from the same files; provided here under "
            "standard-system names.")
    build_instrument(FAC, "MKO", bands, notes=note, raw_files=raws, ground_based=True,
                     extra_provenance={"mirror_check_IRTF_NSFCam": check})


def _usno():
    readme = fetch(USNO_README_WB, FAC, "usno_readme.txt")
    rtext = readme.read_text()
    if "USNO 40-inch/SITe 1024^2" not in rtext or "median airmass of about 1.3" not in rtext:
        raise ValueError("unexpected USNO readme")
    bands, raws = [], [readme]
    for f in "ugriz":
        wb = f"https://web.archive.org/web/{USNO_WB[f]}id_/{USNO_ORIG.format(f)}"
        path = fetch(wb, FAC, f"usno_{f}.res")
        lines = path.read_text().splitlines()
        if (lines[0].strip() != f"\\ FILE: /surtel/filters/usno/usno_{f}.res"
                or lines[2].split() != ["\\", "lam", "vac", "air1.0", "air1.3", "air1.6", "air1.9"]):
            raise ValueError(f"{wb}: unexpected header")
        data = np.array([ln.split() for ln in lines if ln.strip() and not ln.startswith("\\")], float)
        if data.shape[1] != 6:
            raise ValueError(f"{wb}: expected 6 columns")
        bands.append(make_band(
            f + "p", data[:, 0], data[:, 3], unit="Angstrom", detector_type="photon",
            detector_source=ORIGINAL, components="filter+optics+qe+atmosphere",
            components_source=ORIGINAL,
            sources=f"{USNO_ORIG.format(f)} (via {wb}) | {USNO_PAGE} | {USNO_PAPER}",
            svo_id=None))
        raws.append(path)
    note = ("USNO 40-inch u'g'r'i'z' standard-star system (Smith et al. 2002, AJ 123, 2121): "
            "J. Gunn's 2001-12-18 response functions usno_[ugriz].res for the USNO 40-in telescope "
            f"+ SITe 1024^2 UV-coated CCD ({USNO_PAGE}, {USNO_README}). Column 4 = 1.3 airmasses, "
            "the natural-system median airmass: response includes filter, CCD QE (corrected for IR "
            "scattering; point sources), telescope and atmosphere. Photon-counting CCD QE functions. "
            "z' response is quite uncertain (no USNO CCD measurement, per the readme). The site "
            "returns HTTP 403 to scripts; payload from Wayback Machine raw snapshots (2021-03-19). "
            "Not the SDSS 2.5 m ugriz responses; USNO->AB offsets are in the readme.")
    build_instrument(FAC, "USNO", bands, notes=note, raw_files=raws, ground_based=True,
                     extra_provenance={"column_used": "air1.3 (4th column)",
                                       "wayback_snapshots": USNO_WB})


def run():
    _bessell()
    _mko()
    _usno()
