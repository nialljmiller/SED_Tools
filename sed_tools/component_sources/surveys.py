"""Ground-based survey systems rebuilt from their teams' original files.

ATLAS (Tonry et al. 2018), Pan-STARRS1 (Tonry et al. 2012), OGLE-IV
(Skowron's OGLE-IV directory), UKIRT/WFCAM (Hewett et al. 2006) and the Las
Cumbres Observatory Sinistro and Spectral imagers (LCO filter-curve zip).
Each recipe documents what its curves contain; the notes are copied into the
instrument manifest.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import zipfile

import numpy as np

from ._original import (Component, Fetcher, Instrument, make_band, make_builder,
                        safe_member, tar_members)  # noqa: F401  safe_member stays importable here
from .team_atmospheres import HALEAKALA_PS1

PHOTON = "photon"

# --------------------------------------------------------------------- ATLAS
ATLAS_URL = "https://arxiv.org/e-print/1802.00879"
ATLAS_PAPER = "https://ui.adsabs.harvard.edu/abs/2018PASP..130f4505T"
ATLAS_TABLE3 = {"c": 533.0, "o": 679.0}  # Tonry 2018 Table 3 "pivot" [nm], c and o(1)


def atlas_tex(raw: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as t:
        m = t.getmember("atlas_defn.tex")
        if not m.isfile():
            raise ValueError("unsafe member")
        return t.extractfile(m).read().decode("utf-8", "replace")


def atlas_table2(tex: str) -> dict:
    """Table 2 (\\label{tab:coband}): (lambda nm, Tc) x3 then (lambda nm, To) x3 per row."""
    end = tex.index(r"\label{tab:coband}")
    start = tex.rindex(r"\begin{tabular}", 0, end)
    curves = {"c": [], "o": []}
    for line in tex[start:end].splitlines():
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        if len(cells) == 12 and all(re.fullmatch(r"\d+(\.\d+)?", c) for c in cells):
            v = [float(c) for c in cells]
            for k in range(3):
                curves["c"].append((v[2 * k], v[2 * k + 1]))
                curves["o"].append((v[6 + 2 * k], v[7 + 2 * k]))
    return {k: np.array(v) for k, v in curves.items()}


def atlas(fetch: Fetcher, expected_rows: int = 57) -> Instrument:
    curves = atlas_table2(atlas_tex(fetch.get(ATLAS_URL)))
    bands = []
    for name, a in curves.items():
        if len(a) != expected_rows:
            raise ValueError(f"ATLAS {name}: parsed {len(a)} rows, expected {expected_rows}")
        b = make_band(name, a[:, 0], a[:, 1], unit="nm", detector_type=PHOTON,
                      components="filter+optics+qe+atmosphere",
                      sources=f"{ATLAS_URL} (atlas_defn.tex Table 2) | {ATLAS_PAPER}")
        l, tr = b.wavelength, b.transmission
        stat = float(np.trapezoid(tr, l) / np.trapezoid(tr / l, l))
        if abs(stat / (ATLAS_TABLE3[name] * 10) - 1) > 0.005:
            raise ValueError(f"ATLAS {name}: Table-3 statistic {stat:.1f} vs Table 3 {ATLAS_TABLE3[name]} nm")
        bands.append(b)
    note = ("Tonry et al. (2018) ATLAS c and o system bandpasses, Table 2 of the arXiv 1802.00879 "
            "source (atlas_defn.tex), 5 nm sampling. Includes 1.2 airmasses of atmosphere, mirror "
            "reflection (0.92 overcoated Al), AR coatings, filter transmission for the f/2 beam "
            "(from manufacturer curves, effective index 2; no in-situ measurement) and measured CCD "
            "QE. Photon convention (count-rate definition in the paper); SVO's DetectorType=energy "
            "label for Misc/Atlas is not inherited. o is o(1) (Haleakala); o(2) at Mauna Loa and the "
            "later southern units are not tabulated. Components are only plotted (raster), so none "
            "are published separately. No true pivot published: Table 3's 'pivot' is "
            "int T dl / int T/l dl, reproduced to <0.5 % (provenance).")
    return Instrument("ATLAS", "ATLAS", bands, note, ground_based=True)


# ---------------------------------------------------------------- PAN-STARRS
PS1_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/750/99/"
PS1_URL = PS1_BASE + "table3.dat"
PS1_PAPER = "https://ui.adsabs.harvard.edu/abs/2012ApJ...750...99T"
PS1_COLS = {"g": (11, 16), "r": (17, 22), "i": (23, 28), "z": (29, 34), "y": (35, 40), "w": (41, 46)}


def ps1_table4(text: str) -> dict:
    """table4 "pivot" (nm -> Angstrom); numerically the log-mean wavelength."""
    out = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) > 2 and parts[0].endswith("_P1"):
            out[parts[0][0]] = float(parts[2]) * 10.0
    return out


def ps1_table3(text: str):
    rows = [ln for ln in text.splitlines() if ln.strip()]
    wl = np.array([float(ln[0:4]) for ln in rows])
    return wl, {name: np.array([float(ln[a:b]) for ln in rows]) for name, (a, b) in PS1_COLS.items()}


def ps1(fetch: Fetcher) -> Instrument:
    wl, cols = ps1_table3(fetch.text(PS1_URL))
    logmean = ps1_table4(fetch.text(PS1_BASE + "table4.dat"))
    bands = [make_band(name, wl, tr, unit="nm", detector_type=PHOTON,
                       components="filter+optics+qe+atmosphere", sources=f"{PS1_URL} | {PS1_PAPER}")
             for name, tr in cols.items()]
    t = np.trapezoid
    for b in bands:
        l, a = b.wavelength, b.transmission
        lm = float(np.exp(t(a * np.log(l) / l, l) / t(a / l, l)))
        if abs(lm / logmean[b.name] - 1) > 0.005:
            raise ValueError(f"PS1 {b.name}: log-mean {lm:.1f} vs table4 {logmean[b.name]}")
    note = ("Tonry et al. (2012) PS1 system capture cross-sections, CDS J/ApJ/750/99 table3, 1 nm "
            "sampling 380-1100 nm. Values are m^2 e-/photon (effective area including telescope, "
            "corrector, filter, detector QE and the nominal 1.2 airmasses of the PS1 standard "
            "atmosphere); kept unscaled (synthetic photometry is scale-free). Photon-counting CCDs. "
            "The table also gives the 1.2-airmass atmosphere (Aero x Ray x Mol) on the same grid; it is "
            "registered as the team atmosphere site 'haleakala_ps1' (no separate instrument-only "
            "component is published, so the curves above already contain it). No published pivot: "
            "table4's 'pivot' column is the log-mean wavelength, reproduced here to <0.1 % (provenance).")
    return Instrument("PAN-STARRS", "PS1", bands, note, ground_based=True,
                      sites=[HALEAKALA_PS1], default_site=HALEAKALA_PS1.site)


# ---------------------------------------------------------------------- OGLE
OGLE_BASE = "https://www.astrouw.edu.pl/~jskowron/pub/ogle/OGLE-IV/"
OGLE_FILT = {b: OGLE_BASE + f"OGLE-IV-{b}-filter-transmission.dat" for b in ("V", "I")}
OGLE_QE = OGLE_BASE + "OGLE-IV-QE-curve-from-specification.dat"


def ogle_table(text: str, ncol: int) -> np.ndarray:
    rows = [ln.split()[:ncol] for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    return np.array(rows, float)


def ogle(fetch: Fetcher) -> Instrument:
    texts = {b: fetch.text(u) for b, u in OGLE_FILT.items()}
    for b, text in texts.items():
        if f"OGLE-IV {b}-band filter" not in text[:200] or "wavelength[A]" not in text[:2000]:
            raise ValueError(f"{OGLE_FILT[b]}: unexpected header")
    qs = ogle_table(fetch.text(OGLE_QE), 2)
    bands, comps = [], [Component("qe", qs[:, 0], qs[:, 1], band="*", source=OGLE_QE)]
    for b, text in texts.items():
        f = ogle_table(text, 2)
        filt = make_band(b, f[:, 0], f[:, 1], unit="Angstrom", detector_type=PHOTON,
                         components="filter", sources=OGLE_FILT[b])
        wl = filt.wavelength
        prod = filt.transmission * np.interp(wl, qs[:, 0], qs[:, 1], left=0.0, right=0.0)
        bands.append(make_band(b, wl, prod, unit="Angstrom", detector_type=PHOTON,
                               components="filter+qe", sources=f"{OGLE_FILT[b]} | {OGLE_QE}"))
        comps.append(Component("filter", wl, filt.transmission, band=b, source=OGLE_FILT[b]))
    note = ("OGLE-IV (Warsaw 1.3 m, Las Campanas) V and I. Published curve = Asahi filter "
            "transmission (mean of 9 spots, OGLE-IV-{V,I}-filter-transmission.dat) x e2v CCD44-82 "
            "specification QE (OGLE-IV-QE-curve-from-specification.dat, set to 0 outside "
            "3032-10787 A); photon-counting CCDs. No telescope optics or atmosphere included. "
            "Components: filter (per band) and qe (shared). The measured QE min/max envelope "
            "(OGLE-IV-QE-measured-min-max-curves.dat, columns verified as min<=max) is not used; "
            "the pivot with its midpoint is in provenance. Caveat: the filter scans have a "
            "low-signal noise floor below ~3500 A (quantised spot values, up to 0.1 in V and 0.29 "
            "in I) kept as published apart from negatives set to 0; after the QE (0 below 3032 A) it carries ~0.2 % of the photon-weighted response and shifts the pivot by ~0.2-0.5 % vs zeroing below 3500 A. No published pivot.")
    return Instrument("OGLE", "OGLE-IV", bands, note, ground_based=True, components=comps)


# --------------------------------------------------------------------- UKIRT
UKIRT_URL = "http://www.ukidss.org/technical/photom/hewett-ukidss-tables.tar"
UKIRT_EPRINT = "https://arxiv.org/e-print/astro-ph/0601592"
UKIRT_PAPER = "https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..454H"
UKIRT_TABLES = {"Z": "Table02_online.dat", "Y": "Table03_online.dat", "J": "Table04_online.dat",
                "H": "Table05_online.dat", "K": "Table06_online.dat"}
# Hewett et al. 2006 Table 1 (array q.e.), transcribed from photpap_aph.tex in the e-print.
UKIRT_QE_UM = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.2]
UKIRT_QE = [0.53, 0.55, 0.54, 0.54, 0.54, 0.57, 0.62, 0.61]


def wfcam(fetch: Fetcher) -> Instrument:
    text = tar_members(fetch.get(UKIRT_URL), set(UKIRT_TABLES.values()))
    bands = []
    for name, member in UKIRT_TABLES.items():
        body = text[member]
        if "1.0mm water vapour, 1.3 airmass" not in body or f"WFCAM {name} filter" not in body:
            raise ValueError(f"{member}: unexpected header")
        data = np.array([ln.split() for ln in body.splitlines()
                         if ln.strip() and not ln.startswith("#")], float)
        bands.append(make_band(name, data[:, 0], data[:, 1], unit="micron", detector_type=PHOTON,
                               components="filter+optics+qe+atmosphere",
                               sources=f"{UKIRT_URL}#{member} | {UKIRT_PAPER}"))
    qe = Component("qe", np.array(UKIRT_QE_UM) * 1e4, np.array(UKIRT_QE), band="*",
                   source=f"Hewett et al. 2006 Table 1 (array q.e.), {UKIRT_EPRINT} photpap_aph.tex")
    note = ("Hewett et al. (2006) UKIDSS/WFCAM ZYJHK system throughput, tables 2-6 of "
            "hewett-ukidss-tables.tar, for the default 1.0 mm water vapour and 1.3 airmasses. "
            "Includes atmosphere, 3 Al mirror reflections, other optics (flat), filter scans "
            "corrected to 120 K, and Hawaii-II QE; normalised to the measured photon count rate "
            "(photon-counting HgCdTe arrays). These are the corrected total-system curves, not "
            "the room-temperature filter-only scans. Single filter per band (4 per band in "
            "WFCAM; <=0.01 mag differences per the paper). Component: array QE (paper Table 1, "
            "8 points, constant beyond 2.2 um per the paper). No published pivot wavelength "
            "(Hewett Table 7 lambda_eff is Vega-weighted), so none is checked.")
    return Instrument("UKIRT", "WFCAM", bands, note, ground_based=True, components=[qe])


# ---------------------------------------------------------------- Las Cumbres
LCO_URL = "https://lco.global/documents/2/lco_filter_curves.zip"
LCO_PAGE = "https://lco.global/observatory/instruments/filters/"
LCO_SHA256 = "4e9ecf215af1d1210c43342dd7604492d07cea588239b148836eff6f1bf727f0"
# Published band name (case-insensitively unique; LCO filter codes) -> zip member, codex band.
LCO_FILES = {"U": ("bssl-ux-022.txt", "U"), "B": ("bssl-bx-004.txt", "B"), "V": ("bssl-vx-022.txt", "V"),
             "R": ("bssl-rx-007.txt", "R"), "I": ("bssl-ix-023.txt", "I"),
             "up": ("SDSS.up.txt", "u"), "gp": ("SDSS.gp.txt", "g"), "rp": ("SDSS.rp.txt", "r"),
             "ip": ("SDSS.ip.txt", "i"), "zs": ("PSTR-ZS-avg.txt", "zs"), "Y": ("PSTR-YX-avg.txt", "Y")}
LCO_SETS = {"Sinistro": ["U", "B", "V", "R", "I", "up", "gp", "rp", "ip", "zs", "Y"],
            "Spectral": ["B", "V", "R", "I", "up", "gp", "rp", "ip", "zs", "Y"]}


def lco_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    rows = []
    for line in zf.read(member).decode("latin-1").splitlines():
        parts = line.split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (IndexError, ValueError):
            continue  # header such as "\lambda [nm]"
    return np.array(rows)


def _lco(instrument: str, fetch: Fetcher, sha256: str = LCO_SHA256) -> Instrument:
    raw = fetch.get(LCO_URL)
    if sha256 and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("lco_filter_curves.zip differs from the verified copy")
    names = LCO_SETS[instrument]
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        data = {n: lco_member(zf, LCO_FILES[n][0]) for n in names}
    mapping = "; ".join(f"{b}={m} (codex band {c})" for b, (m, c) in LCO_FILES.items())
    bands = [make_band(n, data[n][:, 0], data[n][:, 1], unit="nm", detector_type=PHOTON,
                       components="filter", sources=f"{LCO_URL}#{LCO_FILES[n][0]} | {LCO_PAGE}")
             for n in names]
    cleaned = "; ".join(f"{b.name}: {b.clean_note}" for b in bands if b.clean_note)
    note = (f"LCO {instrument} filters from LCO's lco_filter_curves.zip (wavelength nm -> Angstrom). "
            "Laboratory filter transmission scans only: no telescope, detector QE or "
            "atmosphere, so these are filter-only curves (DetectorType photon for the CCD "
            "imager; a total response needs the detector generation, which changed over time). "
            f"Band names follow LCO filter codes to avoid case collisions: {mapping}. "
            f"Cleaning (measurement noise below 0 set to 0): {cleaned or 'none'}. "
            "No published pivot wavelengths. The w filter on LCO's filter page is not in this "
            "zip and is not substituted."
            + (" Spectral is historical (replaced at FTN/FTS)." if instrument == "Spectral" else ""))
    return Instrument("LasCumbres", instrument, bands, note, ground_based=True)


def sinistro(fetch: Fetcher) -> Instrument:
    return _lco("Sinistro", fetch)


def spectral(fetch: Fetcher) -> Instrument:
    return _lco("Spectral", fetch)


RECIPES = {
    "ATLAS": ("ATLAS", atlas),
    "PS1": ("PAN-STARRS", ps1),
    "OGLE-IV": ("OGLE", ogle),
    "WFCAM": ("UKIRT", wfcam),
    "Sinistro": ("LasCumbres", sinistro),
    "Spectral": ("LasCumbres", spectral),
}

build = make_builder(RECIPES)
