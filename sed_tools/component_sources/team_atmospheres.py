"""Standard atmospheres published by instrument teams, as named sites.

A team that derives its passbands through a standard atmosphere often
publishes that atmosphere too.  Each is registered here as a
:class:`~._common.SiteAtmosphere` with ``kind="team"``: its observing
conditions (airmass, PWV) as the source states them, the model and its
version where stated, the citation and any usage terms.  Grids that
SED_Tools computes itself (ESO SkyCalc, :mod:`.skycalc`) are ``derived``.

Site names follow ``<location>_<team>``, like the existing ``ctio_des`` and
``maunakea_hsc``.  Which instruments may use each site is registered in
:data:`sed_tools.facility_sites._INSTRUMENT_SITES`; an atmosphere is never
offered to an instrument at another location.

Entries are stored as published (only wavelengths converted to Angstrom)
unless their ``derivation`` says otherwise; derived entries carry
``ComponentsSource=derived`` in their curve header.

Build (resumable per site; ``--archive`` keeps every raw download with its
URL, retrieval time and SHA-256)::

    sed-tools filters-build-components team_atmospheres --sites haleakala_ps1
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import io
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ..filter_components import TEAM
from ._common import DerivedCurve, SiteAtmosphere, write_site
from ._original import CONNECT_TIMEOUT

logger = logging.getLogger(__name__)

Curve = Tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
class CachingSession:
    """Fetch each URL once per build; optionally archive every raw download.

    The archive (``archive_dir``) holds the bytes exactly as received plus a
    ``manifest.json`` with URL, retrieval time (UTC) and SHA-256.
    """

    def __init__(self, session=None, archive_dir: Union[str, Path, None] = None):
        if session is None:
            import requests

            from ._original import USER_AGENT

            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
        self.session = session
        self.archive = Path(archive_dir) if archive_dir else None
        self._cache: Dict[str, object] = {}

    def get(self, url: str, **kwargs):
        if url not in self._cache:
            kwargs.setdefault("timeout", (CONNECT_TIMEOUT, 300))
            resp = self.session.get(url, **kwargs)
            resp.raise_for_status()
            if self.archive is not None:
                self._record(url, resp)
            self._cache[url] = resp
        return self._cache[url]

    def _record(self, url: str, resp) -> None:
        self.archive.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", re.sub(r"^https?://", "", url)).strip("_")
        (self.archive / name).write_bytes(resp.content)
        path = self.archive / "manifest.json"
        data = json.loads(path.read_text()) if path.exists() else {"files": []}
        data["files"] = [f for f in data["files"] if f["file"] != name]
        data["files"].append({
            "file": name, "url": url,
            "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": hashlib.sha256(resp.content).hexdigest(), "bytes": len(resp.content),
            "http_status": getattr(resp, "status_code", None),
            "last_modified": getattr(resp, "headers", {}).get("last-modified"),
        })
        path.write_text(json.dumps(data, indent=1) + "\n")


def _bytes(session, url: str) -> bytes:
    resp = session.get(url, timeout=(CONNECT_TIMEOUT, 300))
    resp.raise_for_status()
    raw = resp.content
    head = raw[:512].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        if b"<pre>" not in raw[:2000].lower():
            raise ValueError(f"{url} returned HTML, not data")
    return raw


def _columns(text: str, ncol: int, skip: int = 0) -> np.ndarray:
    """Numeric rows (``ncol`` leading values) of a whitespace table."""
    rows = []
    for line in text.splitlines()[skip:]:
        parts = line.split()
        if len(parts) < ncol:
            continue
        try:
            rows.append([float(p) for p in parts[:ncol]])
        except ValueError:
            continue
    if not rows:
        raise ValueError("no numeric rows")
    return np.array(rows)


def _checked(wl, tr, what: str) -> Curve:
    wl, tr = np.asarray(wl, dtype=float), np.asarray(tr, dtype=float)
    if np.any(np.diff(wl) <= 0):
        raise ValueError(f"{what}: wavelengths are not strictly increasing")
    if np.any(tr < 0) or np.any(tr > 1.0 + 1e-6) or not np.all(np.isfinite(tr)):
        raise ValueError(f"{what}: values are not a 0-1 transmission")
    return wl, tr


# --------------------------------------------------------------------------
# Pan-STARRS1 (Tonry et al. 2012)
# --------------------------------------------------------------------------
PS1_TABLE3 = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/750/99/table3.dat"
PS1_README = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/750/99/ReadMe"
PS1_PAPER = "https://ui.adsabs.harvard.edu/abs/2012ApJ...750...99T"


def ps1_atmosphere(text: str) -> Curve:
    """Aero x Ray x Mol of CDS J/ApJ/750/99 table3 (ReadMe bytes 48-52, 54-58, 60-64)."""
    wl, tr = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 11:
            raise ValueError(f"PS1 table3: expected 11 columns, got {len(parts)}")
        aero, ray, mol = float(line[47:52]), float(line[53:58]), float(line[59:64])
        if (aero, ray, mol) != tuple(float(p) for p in parts[8:11]):
            raise ValueError("PS1 table3: fixed-width and whitespace columns disagree")
        wl.append(float(line[0:4]) * 10.0)
        tr.append(aero * ray * mol)
    return _checked(wl, tr, "PS1 table3 atmosphere")


HALEAKALA_PS1 = SiteAtmosphere(
    site="haleakala_ps1",
    altitude_m=3000.0,
    entries=[(1.2, None, DerivedCurve(
        PS1_TABLE3, lambda s: ps1_atmosphere(_bytes(s, PS1_TABLE3).decode("ascii")),
        derivation="product of the Aero, Ray and Mol columns (aerosol scattering, Rayleigh scattering, "
                   "molecular absorption) of table3, each 'of the nominal 1.2 airmasses of the Pan-STARRS1 "
                   "standard atmosphere' (ReadMe note 2); 1 nm sampling, 380-1100 nm"))],
    source="Pan-STARRS1 standard atmosphere at 1.2 airmasses, Tonry et al. 2012 (CDS J/ApJ/750/99 table3)",
    notes=("Tonry et al. (2012, section 'The PS1 photometric system'): 'the MODTRAN atmosphere at 1.2 airmass from elevation 3 km with "
           "0.65 mm of PWV at sea level and 0.7 aerosol' (the same paper elsewhere gives PWV as 'typically "
           "0.65 cm at sea level'; so no PWV is recorded here). The PS1 passbands of table3 include this "
           "atmosphere. altitude_m is the model elevation the paper states (3 km)."),
    kind=TEAM, model="MODTRAN", model_version="not stated",
    citation=f"Tonry et al. 2012, ApJ 750, 99 (2012ApJ...750...99T); CDS J/ApJ/750/99 ({PS1_README})",
    location="Haleakala, Maui (Pan-STARRS1)",
    parameters={"airmass": 1.2, "elevation": "3 km", "pwv": "0.65 mm (or cm) at sea level, as stated",
                "aerosol exponent": 0.7},
    outside_range="error", coverage_tolerance=1e-3,
)


# --------------------------------------------------------------------------
# VISTA/VIRCAM (ESO ETC curves)
# --------------------------------------------------------------------------
VIRCAM_BUNDLE = "https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst/Filters_QE_Atm_curves.tar.gz"
VIRCAM_PAGE = "https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst.html"
VISTA_PAGE = "https://www.eso.org/public/teles-instr/paranal-observatory/surveytelescopes/vista/"


def vircam_atmosphere(raw: bytes, member: str, header: str) -> Curve:
    """A ``trans_*`` member: two title lines, then micron and transmission."""
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        info = tf.getmember(f"Filters_QE_Atm_curves/{member}")
        if not info.isfile():
            raise ValueError(f"{member} is not a regular file")
        text = gzip.decompress(tf.extractfile(info).read()).decode("ascii")
    if header not in text.splitlines()[0]:
        raise ValueError(f"{member}: unexpected title {text.splitlines()[0]!r}")
    data = _columns(text, 2, skip=2)  # f02.sm reads 'lines 3 999999'
    return _checked(data[:, 0] * 1e4, data[:, 1], member)


def _vircam(member: str, header: str) -> DerivedCurve:
    return DerivedCurve(f"{VIRCAM_BUNDLE}#Filters_QE_Atm_curves/{member}",
                        lambda s: vircam_atmosphere(_bytes(s, VIRCAM_BUNDLE), member, header),
                        derivation="", note=f"as published ('{header}'); micron converted to Angstrom")


PARANAL_VIRCAM = SiteAtmosphere(
    site="paranal_vircam",
    altitude_m=2518.0,
    entries=[(1.0, 1.0, _vircam("trans_10_10.dat.gz", "1.0mm H2O, airmass=1.0")),
             (2.0, 3.0, _vircam("trans_30_20.dat.gz", "3.0mm H2O, airmass=2.0"))],
    source="Model atmospheres distributed with ESO's VIRCAM filter, QE and atmosphere curves "
           "(Filters_QE_Atm_curves.tar.gz)",
    notes=("ESO's VIRCAM page calls these 'a theoretical model for the atmospheric transmission for various "
           "airmasses and water vapor content'; the bundle does not name the model. Wavelengths in micron and "
           "transmission as a fraction (the bundle's f02.sm plots them so, after skipping two title lines); "
           "0.9-5.6 um only: composition is refused when more than 0.1% of a band's throughput lies outside "
           "(Z), and the 0.9 um value is held over fainter leak wings (Y, J, H, Ks: <=1.6e-4 of the band). "
           "altitude_m is VISTA's (ESO); the SkyCalc 'paranal' grid is computed for 2640 m."),
    kind=TEAM, model="not named by the source", model_version="not stated",
    citation=f"ESO VIRCAM instrument page ({VIRCAM_PAGE})",
    location="VISTA, Cerro Paranal",
    outside_range="error", coverage_tolerance=1e-3,
)


# --------------------------------------------------------------------------
# 2MASS (Cohen et al. 2003; Explanatory Supplement III.1.b1)
# --------------------------------------------------------------------------
TWOMASS_DOC = "https://www.ipac.caltech.edu/2mass/releases/allsky/doc/"


def twomass_table(raw: bytes, title: str) -> Curve:
    """An Explanatory Supplement PRE table: n, xlam (micron), yval."""
    text = raw.decode("ascii", "replace")
    if title not in text:
        raise ValueError(f"2MASS table: title {title!r} not found")
    body = re.search(r"<pre>(.*?)</pre>", text, re.S | re.I)
    if body is None:
        raise ValueError("2MASS table: no PRE block")
    data = _columns(body.group(1), 3)
    if not np.array_equal(data[:, 0], np.arange(1, len(data) + 1)):
        raise ValueError("2MASS table: row numbers are not consecutive (truncated?)")
    return _checked(data[:, 1] * 1e4, data[:, 2], title)


def _twomass(table: str, title: str) -> DerivedCurve:
    url = TWOMASS_DOC + table
    return DerivedCurve(url, lambda s: twomass_table(_bytes(s, url), title), derivation="",
                        note=f"as published ('{title}'); xlam (micron) converted to Angstrom")


_PLEXUS = dict(
    kind=TEAM, model="PLEXUS (AFRL; MODTRAN 3.7, SAMM, FASCODE3P, HITRAN98)", model_version="not stated",
    citation=("Cohen, Wheaton & Megeath 2003, AJ 126, 1090 (2003AJ....126.1090C), Sect. 2; 2MASS All-Sky "
              f"Explanatory Supplement III.1.b1 ({TWOMASS_DOC}sec3_1b1.html)"),
    outside_range="error", coverage_tolerance=1e-3,
)
_PLEXUS_NOTE = ("PLEXUS model provided by M. Cohen; 'in addition to atomic and molecular absorption, this code "
                "accounts for Rayleigh and site-specific mean aerosol scattering' (Explanatory Supplement). "
                "Neither the table nor Cohen et al. (2003) states the airmass or water vapour, so it is "
                "registered with airmass unstated: select it with airmass=None. Cohen et al. used 'a single "
                "PLEXUS atmosphere representative of typical survey conditions' for both telescopes; which "
                "of the two is not stated. 1.0-2.5 micron.")

HOPKINS_2MASS = SiteAtmosphere(
    site="hopkins_2mass", altitude_m=None,
    entries=[(None, None, _twomass("sec3_1b1.tbl15.html", "ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE"))],
    source="2MASS north (Mt. Hopkins) PLEXUS atmospheric transmission, Explanatory Supplement III.1.b1 table 15",
    notes=_PLEXUS_NOTE, location="2MASS north telescope, Mt. Hopkins, Arizona", **_PLEXUS,
)
CTIO_2MASS = SiteAtmosphere(
    site="ctio_2mass", altitude_m=None,
    entries=[(None, None, _twomass("sec3_1b1.tbl16.html", "ATMOSPHERIC TRANSMISSION AT 2MASS SOUTH SITE"))],
    source="2MASS south (CTIO) PLEXUS atmospheric transmission, Explanatory Supplement III.1.b1 table 16",
    notes=_PLEXUS_NOTE, location="2MASS south telescope, Cerro Tololo", **_PLEXUS,
)


# --------------------------------------------------------------------------
# Carnegie Supernova Project (CSP_filter_package.tgz)
# --------------------------------------------------------------------------
CSP_PACKAGE = "https://csp.obs.carnegiescience.edu/data/CSP_filter_package.tgz"
CSP_PAGE = "https://csp.obs.carnegiescience.edu/data/filters"
CSP_AIRMASS = 1.2


def _csp_member(raw: bytes, name: str) -> np.ndarray:
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        info = tf.getmember(f"CSP_filter_package/{name}")
        if not info.isfile():
            raise ValueError(f"{name} is not a regular file")
        return _columns(tf.extractfile(info).read().decode("ascii"), 2)


def csp_atmosphere(raw: bytes, airmass: float = CSP_AIRMASS) -> Curve:
    """ext ** airmass x lines on the merged grid of both files' common range."""
    ext = _csp_member(raw, "atm_ext_ctio.dat")
    lines = _csp_member(raw, "atm_ext_lines.dat")
    _checked(ext[:, 0], ext[:, 1], "atm_ext_ctio.dat")
    # The lines file is a normalised spectrum: two samples (7150, 7574.2 A)
    # reach 1.0025; they are kept, and the product below must be a transmission.
    if np.any(np.diff(lines[:, 0]) <= 0) or np.any(lines[:, 1] < 0) or lines[:, 1].max() > 1.01:
        raise ValueError("atm_ext_lines.dat: unexpected values")
    lo, hi = max(ext[0, 0], lines[0, 0]), min(ext[-1, 0], lines[-1, 0])
    grid = np.unique(np.concatenate([ext[:, 0], lines[:, 0]]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    tr = np.interp(grid, ext[:, 0], ext[:, 1]) ** airmass * np.interp(grid, lines[:, 0], lines[:, 1])
    return _checked(grid, tr, "CSP atmosphere")


LCO_CSP = SiteAtmosphere(
    site="lco_csp",
    altitude_m=None,
    entries=[(CSP_AIRMASS, None, DerivedCurve(
        f"{CSP_PACKAGE}#CSP_filter_package/atm_ext_ctio.dat | {CSP_PACKAGE}#CSP_filter_package/atm_ext_lines.dat",
        lambda s: csp_atmosphere(_bytes(s, CSP_PACKAGE)),
        derivation=("atm_ext_ctio.dat ** 1.2 x atm_ext_lines.dat, linearly interpolated onto the union of "
                    "both grids over their common range (2959-54995 A)")))],
    source="Carnegie Supernova Project atmosphere at 1.2 airmass from CSP_filter_package.tgz",
    notes=("The package README lists atm_ext_ctio.dat ('Atmospheric extinction') and atm_ext_lines.dat "
           "('Atmospheric telluric lines') and says the final passbands 'have atmosphere corresponding to 1.2 "
           "airmass'. atm_ext_ctio.dat is already a transmission through one airmass: from 3200 to 10870 A it "
           "equals 10**(-0.4 k) for the CTIO extinction k of IRAF onedstds/ctioextinct.dat to all six "
           "printed digits; 2959-3151 A extends it and 11000-70000 A is 1.0. Continuum transmission at 1.2 "
           "airmasses is therefore its 1.2th power (Bouguer). The lines file states no airmass; CSP applied "
           "it unscaled: the final NIR scans (*_TAM_scan_atm.dat) equal the raw scans times the lines file "
           "to <=3e-4 rms (H, J; 1.3% in Y), and at line-free wavelengths the final i and V (LC-9844) "
           "scans equal the raw scans times ext**1.2 to <=0.3% (median 0.02%); the other optical bands "
           "differ from their raw scans by up to a few percent for reasons the package does not state. "
           "Two samples of the lines file exceed 1 (1.0025 at 7150 A, 1.0017 at 7574.2 A); they are kept "
           "as published (the product stays below 1). Swope at Las Campanas; the continuum extinction is CTIO's."),
    kind=TEAM, model=("CTIO extinction (IRAF onedstds/ctioextinct.dat: Stone & Baldwin 1983, MNRAS 204, 347; "
                      "Baldwin & Stone 1984, MNRAS 206, 241) with a telluric line spectrum of unstated origin"),
    model_version="CSP_filter_package README updated 2018-03-16; atm files dated 2011-09-01",
    citation=f"CSP filter package and filters page ({CSP_PAGE}), which cites Stritzinger et al. (2011) for the scans",
    location="Swope telescope, Las Campanas Observatory (CTIO extinction)",
    parameters={"airmass": CSP_AIRMASS, "continuum": "atm_ext_ctio.dat ** airmass",
                "lines": "atm_ext_lines.dat, unscaled"},
    outside_range="error", coverage_tolerance=1e-3,
)


# --------------------------------------------------------------------------
# SDSS (DR7 imager response tables)
# --------------------------------------------------------------------------
SDSS_DR7 = "https://classic.sdss.org/dr7/instruments/imager/"
SDSS_FILES = {b: f"{SDSS_DR7}filters/{b}.dat" for b in "ugriz"}


def sdss_transparency(session) -> Curve:
    """Column 5 ('assumed atmospheric transparency at one airmass') of u, g, r, i, z.dat, merged."""
    values: Dict[float, float] = {}
    for band, url in SDSS_FILES.items():
        text = _bytes(session, url).decode("ascii")
        if "extinction" not in text[:600] or "1.0 airmasses" not in text[:600]:
            raise ValueError(f"{url}: unexpected header")
        for wl, t in _columns(text, 5)[:, [0, 4]]:
            if wl in values and values[wl] != t:
                raise ValueError(f"SDSS {band}: transparency at {wl} A disagrees between band files")
            values[wl] = t
    wl = np.array(sorted(values))
    return _checked(wl, np.array([values[w] for w in wl]), "SDSS transparency")


APO_SDSS = SiteAtmosphere(
    site="apo_sdss",
    altitude_m=None,
    entries=[
        (1.0, None, DerivedCurve(" | ".join(SDSS_FILES.values()), sdss_transparency,
                                 derivation="", note="column 5 of the five band files, merged (identical where "
                                                     "they overlap); 2980-11230 A at 25 A")),
        (1.3, None, DerivedCurve(" | ".join(SDSS_FILES.values()),
                                 lambda s: (lambda c: (c[0], c[1] ** 1.3))(sdss_transparency(s)),
                                 derivation="column 5 (transparency through one airmass) to the power 1.3")),
    ],
    source="SDSS imager 'assumed atmospheric transparency at one airmass at Apache Point Observatory' (DR7)",
    notes=("Column 5 of the DR7 imager response tables. The survey's photometric system uses 1.3 airmasses; "
           "the 1.3 entry is column 5 to the power 1.3 (Bouguer). Check: in every band column 3 "
           "(1.3 airmasses) / column 4 (no atmosphere) = 1.0204 x column5**1.300 to <=0.13% (g, r, i) and "
           "0.6% (u, z) where column 4 exceeds 10% of its peak; the constant 1.0204 is not explained by the "
           "source. Point-source and extended columns differ only by infrared scattering, not atmosphere."),
    kind=TEAM, model="not stated", model_version="DR7",
    citation=f"SDSS DR7 imager page ({SDSS_DR7}), which cites Doi et al. 2010 (arXiv:1002.3701)",
    location="Apache Point Observatory (SDSS 2.5 m)",
    outside_range="error", coverage_tolerance=1e-3,
)


# --------------------------------------------------------------------------
# BASS (Legacy Surveys DR10 filter files)
# --------------------------------------------------------------------------
BASS_FILES = {b: f"https://www.legacysurvey.org/files/bass-{b}.txt" for b in "gr"}
BASS_PAGE = "https://www.legacysurvey.org/dr10/description/"
BASS_MIN_RESPONSE = 0.1  # filter x QE column; 3-decimal rounding limits the ratio below this


def bass_atmosphere(session) -> Curve:
    """res_qe_atm / respons_qe (0-based columns 3/2) where respons_qe >= 0.1, g and r merged."""
    values: Dict[float, float] = {}
    for band, url in BASS_FILES.items():
        text = _bytes(session, url).decode("ascii")
        if "atmospheric extinction at airmass of 1.0" not in text[:600]:
            raise ValueError(f"{url}: unexpected header")
        data = _columns(text, 4)
        ok = data[:, 2] >= BASS_MIN_RESPONSE
        for wl, ratio in zip(data[ok, 0], data[ok, 3] / data[ok, 2]):
            if wl in values:
                raise ValueError(f"BASS: {wl} A is in-band for two filters")
            values[wl] = ratio
    wl = np.array(sorted(values))
    return _checked(wl, np.array([values[w] for w in wl]), "BASS atmosphere")


KPNO_BASS = SiteAtmosphere(
    site="kpno_bass",
    altitude_m=None,
    entries=[(1.0, None, DerivedCurve(
        " | ".join(BASS_FILES.values()), bass_atmosphere,
        derivation=("column 4 / column 3 (filter x QE x atmosphere at airmass 1.0 / filter x QE) of "
                    "bass-g.txt and bass-r.txt where column 3 >= 0.1; g and r merged (they do not overlap)")))],
    source="BASS (90Prime) atmosphere at airmass 1.0 from the Legacy Surveys DR10 filter files",
    notes=("The files' header: 'forth columns is the response including both CCD QE and atmospheric "
           "extinction at airmass of 1.0'. The ratio exists only where a filter transmits, 3952-5523 A (g) "
           "and 5660-7170 A (r); values are rounded to 3 decimals, so rows with filter x QE < 0.1 are "
           "dropped (ratio error > 1%). Composition holds the edge values beyond this range (outside_range "
           "'edge'), where the BASS response is small. The ratio equals 10**(-0.4 k) of IRAF "
           "onedstds/kpnoextinct.dat to within the rounding (median 1.0000, range 0.996-1.002)."),
    kind=TEAM, model="KPNO standard extinction (matches IRAF kpnoextinct.dat)", model_version="not stated",
    citation=f"Legacy Surveys DR10 description ({BASS_PAGE})",
    location="Bok 2.3 m, Kitt Peak",
    outside_range="edge",
)


# --------------------------------------------------------------------------
# Gemini (ATRAN, Lord 1992)
# --------------------------------------------------------------------------
GEMINI_MODELS = "https://www.gemini.edu/sciops/ObsProcess/obsConstraints/atm-models/"
GEMINI_PAGE = "https://www.gemini.edu/observing/telescopes-and-sites/sites"
GEMINI_TERMS = ("Any use of these data should reference Lord, S. D., 1992, NASA Technical Memorandum 103957, "
                "and acknowledge Gemini Observatory.")
GEMINI_AIRMASSES = (1.0, 1.5, 2.0)


def gemini_atran(raw: bytes, url: str) -> Curve:
    data = _columns(raw.decode("ascii"), 2)
    if data[0, 0] > 0.9 + 1e-9 or data[-1, 0] < 5.599:
        raise ValueError(f"{url}: covers {data[0, 0]}-{data[-1, 0]} um, not 0.9-5.6 um (truncated?)")
    return _checked(data[:, 0] * 1e4, data[:, 1], url)


def _gemini_entries(prefix: str, pwvs: Sequence[float]):
    entries = []
    for airmass in GEMINI_AIRMASSES:
        for pwv in pwvs:
            url = f"{GEMINI_MODELS}{prefix}_zm_{round(pwv * 10)}_{round(airmass * 10)}.dat"
            entries.append((airmass, pwv, DerivedCurve(
                url, (lambda u: lambda s: gemini_atran(_bytes(s, u), u))(url), derivation="",
                note="as published; micron converted to Angstrom")))
    return entries


_ATRAN = dict(
    kind=TEAM, model="ATRAN (Lord 1992)", model_version="not stated",
    citation=f"Lord 1992, NASA Technical Memorandum 103957; Gemini Observatory ({GEMINI_PAGE})",
    terms=GEMINI_TERMS,
    parameters={"sampling_um": 2e-5, "resolution_um": 4e-5, "range_um": [0.9, 5.6]},
    outside_range="error", coverage_tolerance=1e-3,
)
_ATRAN_NOTE = ("Gemini's near-infrared sky transmission models used by its integration time calculators: "
               "'generated using the ATRAN modelling software (Lord, S.D. 1992, NASA Technical Memor. 103957)'; "
               "'ASCII two-column format: (a) wavelength with a sampling of 0.00002um and a resolution of "
               "0.00004um (0.04nm) and (b) transmission'. 0.9-5.6 micron only (the 7-26 micron files are "
               "not ingested). File names give 10x the water vapour in mm and 10x the airmass.")

MAUNAKEA_GEMINI = SiteAtmosphere(
    site="maunakea_gemini", altitude_m=4213.0,
    entries=_gemini_entries("mktrans", (1.0, 1.6, 3.0, 5.0)),
    source="Gemini Observatory ATRAN sky transmission for Mauna Kea, 0.9-5.6 micron (mktrans_zm_*)",
    notes=_ATRAN_NOTE + " altitude_m is Gemini North's (Gemini).", location="Gemini North, Maunakea", **_ATRAN,
)
CPACHON_GEMINI = SiteAtmosphere(
    site="cpachon_gemini", altitude_m=2722.0,
    entries=_gemini_entries("cptrans", (2.3, 4.3, 7.6, 10.0)),
    source="Gemini Observatory ATRAN sky transmission for Cerro Pachon, 0.9-5.6 micron (cptrans_zm_*)",
    notes=_ATRAN_NOTE + " altitude_m is Gemini South's (Gemini).", location="Gemini South, Cerro Pachon",
    **_ATRAN,
)


# --------------------------------------------------------------------------
# Subaru/HSC: the very site its builder writes (one definition, in hsc.py)
# --------------------------------------------------------------------------
def _maunakea_hsc() -> SiteAtmosphere:
    from .hsc import SPEC

    return SPEC.site


# --------------------------------------------------------------------------
# Registry and builder
# --------------------------------------------------------------------------
SITES: Dict[str, SiteAtmosphere] = {
    s.site: s for s in (HALEAKALA_PS1, PARANAL_VIRCAM, HOPKINS_2MASS, CTIO_2MASS, LCO_CSP, APO_SDSS,
                        KPNO_BASS, MAUNAKEA_GEMINI, CPACHON_GEMINI)
}
SITES["maunakea_hsc"] = _maunakea_hsc()


def build(
    filter_root: Union[str, Path],
    *,
    atmosphere_root: Union[str, Path, None] = None,
    sites: Optional[Iterable[str]] = None,
    session=None,
    archive_dir: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> Path:
    """Write the named team sites (all by default) under ``atmosphere_root``.

    Sites whose manifest already exists are kept unless ``overwrite``.
    """
    root = Path(atmosphere_root) if atmosphere_root else Path(filter_root).parent / "atmospheres"
    names = list(SITES) if sites is None else list(sites)
    unknown = [n for n in names if n not in SITES]
    if unknown:
        raise ValueError(f"Unknown team atmosphere site(s) {unknown}. Choose from {sorted(SITES)}")
    fetch = CachingSession(session, archive_dir)
    for name in names:
        if (root / name / "manifest.json").exists() and not overwrite:
            logger.info("%s exists; keeping it", root / name)
            continue
        logger.info("writing team atmosphere %s", name)
        write_site(root, SITES[name], fetch)
    return root
